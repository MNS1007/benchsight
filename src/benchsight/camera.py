"""Camera access and image helpers. Frames are taken only when a tool asks for one:
nothing records in the background."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import cleanup

STORE = Path(os.environ.get("BENCHSIGHT_DIR", ".benchsight"))  # where photos are kept
KEEP_SECONDS = 24 * 3600  # saved frames older than this are deleted on the next capture
# (a backstop: each session deletes its own files when it ends, see cleanup.py)


@dataclass
class Shot:
    path: Path
    jpeg: bytes
    width: int
    height: int


def list_cameras(max_index: int = 6) -> list[int]:
    """Indices of cameras that open and return a frame."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        if ok:
            found.append(i)
    return found


def grab(index: int = 0, warmup: int = 5) -> np.ndarray:
    """One fresh frame. A few frames are dropped first so auto-exposure settles."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"camera {index} did not open (check the index and the OS camera permission)")
    try:
        ok, frame = False, None
        for _ in range(warmup + 1):
            ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"camera {index} returned no frame")
        return frame
    finally:
        cap.release()


def fit(img: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale so the longer side is at most max_side (keeps tokens and latency low)."""
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    return img if s >= 1 else cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def crop(img: np.ndarray, box: tuple[float, float, float, float] | None) -> np.ndarray:
    """box = (x1, y1, x2, y2) as fractions 0-1 of the frame, for a close look at one region."""
    if not box:
        return img
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (min(1.0, max(0.0, v)) for v in box)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("crop box is empty")
    return img[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]


def contact_sheet(frames: list[np.ndarray], cols: int = 3, label: bool = True) -> np.ndarray:
    """Several frames in one image, numbered, so the agent sees a sequence in one look."""
    if not frames:
        raise ValueError("no frames")
    h, w = frames[0].shape[:2]
    cols = min(cols, len(frames))
    rows = (len(frames) + cols - 1) // cols
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, f in enumerate(frames):
        f = cv2.resize(f, (w, h)) if f.shape[:2] != (h, w) else f
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = f
        if label:
            cv2.putText(sheet, str(i + 1), (c * w + 8, r * h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return sheet


def mark(img: np.ndarray, x: float, y: float, text: str = "") -> np.ndarray:
    """Circle a point (fractions 0-1) and write a short label, to point the builder at a part."""
    out = img.copy()
    h, w = out.shape[:2]
    cx, cy = int(x * w), int(y * h)
    r = max(12, min(h, w) // 25)
    cv2.circle(out, (cx, cy), r, (0, 0, 255), 3)
    if text:
        cv2.putText(out, text, (cx + r + 6, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return out


def save(img: np.ndarray, name: str, quality: int = 85, store: Path = STORE) -> Shot:
    """Write a JPEG to the local store and return its bytes. It is deleted when this
    session ends (unless BENCHSIGHT_KEEP=1), or after 24 hours at the latest."""
    store.mkdir(parents=True, exist_ok=True)
    now = time.time()
    cleanup.delete(cleanup.store_files(store, KEEP_SECONDS, now))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("could not encode the image")
    path = store / f"{time.strftime('%Y%m%d-%H%M%S')}-{int(now * 1000) % 1000:03d}-{name}.jpg"
    path.write_bytes(buf.tobytes())
    cleanup.record(path, store)
    h, w = img.shape[:2]
    return Shot(path, buf.tobytes(), w, h)
