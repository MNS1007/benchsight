"""Global camera motion between frames, so tracked boxes can follow when the laptop or
camera is bumped. Sparse Lucas-Kanade every frame (a few ms), ORB every few frames
(about 15 ms at 640 px) to re-anchor; both fit one rotation+scale+translation affine with
RANSAC. The same ORB matcher re-finds a lost object's template.

adapted from VISTA (UMD Ironsite Hackathon 2025, Team Cortex)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

MAX_SIDE = 640  # motion and tracking run at this size; ORB is O(w*h)
RATIO = 0.75  # Lowe ratio test
IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

_orb = cv2.ORB_create(nfeatures=500)
_bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


@dataclass
class Motion:
    """How scene points moved from the previous frame to this one: p_now = M @ [x, y, 1]."""
    M: np.ndarray = field(default_factory=IDENTITY.copy)
    confidence: float = 0.0  # inlier share of the matched points, 0-1
    magnitude: float = 0.0  # translation in pixels of the frame given
    method: str = "none"

    @property
    def rotation(self) -> float:
        return math.degrees(math.atan2(self.M[1, 0], self.M[0, 0]))

    @property
    def scale(self) -> float:
        return math.hypot(self.M[0, 0], self.M[1, 0])


def gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame


def downscale(frame: np.ndarray, max_side: int = MAX_SIDE) -> tuple[np.ndarray, float]:
    """(frame with longest side <= max_side, scale applied)."""
    h, w = frame.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return frame, 1.0
    return cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA), s


def affine_from_points(src: np.ndarray, dst: np.ndarray, min_points: int = 6) -> tuple[np.ndarray | None, float]:
    """Rotation+scale+translation fit with RANSAC. Returns (2x3 or None, inlier share)."""
    if len(src) < min_points:
        return None, 0.0
    M, inliers = cv2.estimateAffinePartial2D(
        np.float32(src).reshape(-1, 1, 2), np.float32(dst).reshape(-1, 1, 2),
        method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None:
        return None, 0.0
    n = int(inliers.sum()) if inliers is not None else 0
    if n < min_points:
        return None, 0.0
    return M, n / len(src)


def orb_features(img: np.ndarray, mask: np.ndarray | None = None):
    """(keypoints, descriptors) of a grayscale image."""
    return _orb.detectAndCompute(img, mask)


def orb_match(kp1, des1, kp2, des2) -> tuple[np.ndarray, np.ndarray]:
    """Matched point pairs after the ratio test, as two (N, 2) arrays."""
    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    good = [p[0] for p in _bf.knnMatch(des1, des2, k=2) if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)
    return src, dst


def orb_motion(prev: np.ndarray, curr: np.ndarray, mask: np.ndarray | None = None) -> Motion:
    """ORB (500) + ratio 0.75 + RANSAC affine between two grayscale frames."""
    kp1, des1 = orb_features(prev, mask)
    kp2, des2 = orb_features(curr, mask)
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return Motion(method="orb")
    M, conf = affine_from_points(*orb_match(kp1, des1, kp2, des2))
    if M is None:
        return Motion(method="orb")
    return Motion(M, conf, float(math.hypot(M[0, 2], M[1, 2])), "orb")


def lk_motion(prev: np.ndarray, curr: np.ndarray, pts: np.ndarray | None = None,
              mask: np.ndarray | None = None) -> tuple[Motion, np.ndarray | None]:
    """Sparse LK flow of corners from prev to curr. Returns (motion, tracked points in curr)."""
    if pts is None or len(pts) < 20:
        pts = cv2.goodFeaturesToTrack(prev, maxCorners=200, qualityLevel=0.01, minDistance=8, mask=mask)
    if pts is None or len(pts) < 6:
        return Motion(method="lk"), None
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, curr, pts, None, winSize=(21, 21), maxLevel=3)
    ok = st.reshape(-1) == 1
    src, dst = pts.reshape(-1, 2)[ok], nxt.reshape(-1, 2)[ok]
    M, conf = affine_from_points(src, dst)
    if M is None:
        return Motion(method="lk"), None
    conf *= ok.mean()  # points lost by the flow count against the fit
    return Motion(M, conf, float(math.hypot(M[0, 2], M[1, 2])), "lk"), dst.reshape(-1, 1, 2)


class CameraMotion:
    """Frame-to-frame global motion. Feed frames of one size (e.g. already downscaled);
    LK runs every frame, ORB replaces it every `orb_every` frames or when LK fails."""

    def __init__(self, orb_every: int = 4, max_side: int = MAX_SIDE):
        self.orb_every, self.max_side = orb_every, max_side
        self._prev: np.ndarray | None = None
        self._pts: np.ndarray | None = None
        self._n = 0

    def reset(self) -> None:
        self._prev, self._pts, self._n = None, None, 0

    def update(self, frame: np.ndarray, mask: np.ndarray | None = None) -> Motion:
        """Motion from the last frame to this one, in this frame's pixels. mask (uint8,
        nonzero = use) leaves out tracked objects, which move on their own."""
        small, s = downscale(gray(frame), self.max_side)
        if mask is not None and s != 1.0:
            mask = cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        prev, self._prev = self._prev, small
        self._n += 1
        if prev is None or prev.shape != small.shape:
            self._pts = None
            return Motion(confidence=1.0)
        m = Motion()
        if self._n % self.orb_every != 0:
            m, self._pts = lk_motion(prev, small, self._pts, mask)
        if m.confidence < 0.3:
            m = orb_motion(prev, small, mask)
            self._pts = None
        if s != 1.0:  # back to the caller's pixels
            m.M = m.M.copy()
            m.M[:, 2] /= s
            m.magnitude /= s
        return m


def apply(M: np.ndarray, box) -> list[float]:
    """Move an axis-aligned box [x1, y1, x2, y2] by an affine (its corners' bounding box)."""
    x1, y1, x2, y2 = box
    c = np.array([[x1, y1, 1], [x2, y1, 1], [x1, y2, 1], [x2, y2, 1]], dtype=np.float64) @ M.T
    return [float(c[:, 0].min()), float(c[:, 1].min()), float(c[:, 0].max()), float(c[:, 1].max())]
