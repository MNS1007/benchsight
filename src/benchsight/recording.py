"""Record the bench while the user works, so the agent can review it afterwards.

The agent cannot watch live video, but it can look at a recording once the user is done:
start a recording, let the user work, stop it, then review a contact sheet of evenly
spaced frames and zoom into the moments that matter. Each frame also gets an activity
score (how much of the work area changed), which is only a hint for where to look: frames
for review are always picked evenly in time, never by activity.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import camera

ACTIVITY_WIDTH = 160  # activity is scored on a grayscale this wide
DIFF = 0.6  # a pixel changed if it moved this many standard deviations (after normalizing)
MIN_BLOB = 0.002  # changed areas smaller than this share of the roi are speckle


def small_gray(frame: np.ndarray, roi: tuple[float, float, float, float] | None = None,
               width: int = ACTIVITY_WIDTH) -> np.ndarray:
    """The roi of a frame as a small float grayscale, normalized to zero mean and unit
    spread, so a global change in brightness or gain (auto-exposure, a flickering light)
    leaves it the same. A flat image (no spread) comes back all zeros."""
    img = camera.crop(frame, roi)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = g.shape[:2]
    if w > width:
        g = cv2.resize(g, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(g.astype(np.float32), (3, 3), 0)
    std = float(g.std())
    if std < 2.0:  # nothing to compare against: noise would look like change
        return np.zeros_like(g)
    return (g - float(g.mean())) / std


def activity(prev: np.ndarray | None, curr: np.ndarray) -> float:
    """Percent (0-100) of the roi that changed between two small_gray images, with tiny
    speckle ignored. 0 for the first frame or when the sizes differ."""
    if prev is None or prev.shape != curr.shape or (not curr.any() and not prev.any()):
        return 0.0
    changed = (np.abs(curr - prev) > DIFF).astype(np.uint8)
    changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(changed, connectivity=8)
    min_area = max(4, MIN_BLOB * changed.size)
    area = int(sum(stats[i, cv2.CC_STAT_AREA] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area))
    return round(100.0 * area / changed.size, 2)


class Activity:
    """Scores each frame against the one before it, inside an roi."""

    def __init__(self, roi: tuple[float, float, float, float] | None = None):
        self.roi = roi
        self._prev: np.ndarray | None = None

    def update(self, frame: np.ndarray) -> float:
        curr = small_gray(frame, self.roi)
        score = activity(self._prev, curr)
        self._prev = curr
        return score


def timeline(index: list[dict], threshold: float = 1.0, gap_s: float = 1.0, min_s: float = 0.3) -> list[list[float]]:
    """Stretches of activity as [start_s, end_s]: frames scoring at least threshold, from
    the frame before (the change happened between the two), merging gaps under gap_s and
    dropping stretches shorter than min_s."""
    segs: list[list[float]] = []
    for k, e in enumerate(index):
        if e.get("activity", 0.0) < threshold:
            continue
        start, end = float(index[k - 1]["t"] if k else e["t"]), float(e["t"])
        if segs and start - segs[-1][1] < gap_s:
            segs[-1][1] = end
        else:
            segs.append([start, end])
    return [[round(a, 2), round(b, 2)] for a, b in segs if b - a >= min_s]


def sample(index: list[dict], start_s: float | None = None, end_s: float | None = None,
           frames: int = 12) -> list[dict]:
    """Up to `frames` entries spread evenly in time over [start_s, end_s] (default: the
    whole recording), both ends included, each picked as the frame nearest its time.
    Uniform on purpose: activity never decides what is shown. [] if the window is empty."""
    if start_s is not None and end_s is not None and start_s > end_s:
        raise ValueError("start_s is after end_s")
    lo = index[0]["t"] if start_s is None and index else start_s
    hi = index[-1]["t"] if end_s is None and index else end_s
    inside = [e for e in index if lo <= e["t"] <= hi] if index else []
    if not inside or frames < 1:
        return []
    times = np.array([e["t"] for e in inside], dtype=np.float64)
    targets = [inside[0]["t"]] if frames == 1 else np.linspace(times[0], times[-1], frames)
    out, seen = [], set()
    for t in targets:
        k = int(np.abs(times - t).argmin())
        if k not in seen:
            seen.add(k)
            out.append(inside[k])
    return out
