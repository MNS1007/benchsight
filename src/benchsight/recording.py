"""Record the bench while the user works, so the agent can review it afterwards.

The agent cannot watch live video, but it can look at a recording once the user is done:
start a recording, let the user work, stop it, then review a contact sheet of evenly
spaced frames and zoom into the moments that matter. Each frame also gets an activity
score (how much of the work area changed), which is only a hint for where to look: frames
for review are always picked evenly in time, never by activity.
"""
from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from . import camera, cleanup

ACTIVITY_WIDTH = 160  # activity is scored on a grayscale this wide
DIFF = 0.6  # a pixel changed if it moved this many standard deviations (after normalizing)
MIN_BLOB = 0.002  # changed areas smaller than this share of the roi are speckle
ACTIVE = 1.0  # timeline(): a frame is active when at least this percent of the roi changed
MAX_SECONDS = 900.0  # hard cap on one recording
MAX_FPS = 15.0
STALL_S = 3.0  # no new camera frame for this long: the camera froze or was unplugged
INDEX = cleanup.INDEX


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


# ── the recorder ──

_guard = threading.RLock()  # reentrant: session end may stop a recording from a signal handler
_current: "Recorder | None" = None


def current() -> "Recorder | None":
    """The recording in progress, if any (only one at a time)."""
    with _guard:
        return _current


class Recorder:
    """Saves the live feed's latest frame fps times a second as small JPEGs in
    <store>/rec-YYYYmmdd-HHMMSS/, with one index.jsonl line per frame. Every file is
    registered with cleanup, so it goes when the session ends. Stops by itself after
    max_seconds (at most MAX_SECONDS), or when the feed stops. A frame the feed already gave
    (same feed.seq) is never saved twice, and no new frame for STALL_S stops it as "stalled"."""

    def __init__(self, feed, store: Path | str, fps: float = 8, max_side: int = 640,
                 max_seconds: float = 300, quality: int = 70,
                 roi: tuple[float, float, float, float] | None = None, clock=time.time):
        if roi is not None:
            roi = tuple(float(v) for v in roi)
            if len(roi) != 4:
                raise ValueError("roi is [x1, y1, x2, y2] as fractions 0-1")
            camera.crop(np.zeros((10, 10), np.uint8), roi)  # raises if the box is empty
        self.feed, self.store = feed, Path(store)
        self.fps = min(MAX_FPS, max(0.5, float(fps)))
        self.max_seconds = min(MAX_SECONDS, max(1.0, float(max_seconds)))
        self.max_side = int(min(1280, max(160, max_side)))
        self.quality = int(min(95, max(30, quality)))
        self.roi, self.clock = roi, clock
        self.folder: Path | None = None
        self.index: list[dict] = []
        self.reason = ""  # why it stopped: "stopped", "cap", "feed", "stalled" or "error: ..."
        self._activity = Activity(roi)
        self._start = self._end = 0.0
        self._seq: int | None = None  # the feed.seq of the last frame saved
        self._fresh = 0.0  # when the feed last gave a new frame
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self.folder is not None and not self.reason

    def elapsed(self) -> float:
        if self.folder is None:
            return 0.0
        return round((self._end if self.reason else self.clock()) - self._start, 3)

    def start(self, background: bool = True) -> "Recorder":
        """Make the folder, take the first frame now and keep going in a thread.
        background=False leaves the ticking to the caller (capture()), for tests."""
        global _current
        with _guard:
            if self.folder is not None:
                raise RuntimeError("this recorder already ran; make a new one")
            if _current is not None and _current.running:
                raise RuntimeError(f"already recording into {_current.folder}; stop that first")
            if not getattr(self.feed, "running", True):
                raise RuntimeError("the camera is not running")
            self.folder = self._make_folder()
            _current = self
        self._start = self.clock()
        self._next = self._fresh = self._start
        try:
            if self.capture() and background:
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()
        except Exception as err:  # disk full, read-only store: do not stay "recording"
            self._finish(f"error: {err}", self.clock())
            raise
        return self

    def _make_folder(self) -> Path:
        self.store.mkdir(parents=True, exist_ok=True)
        base = f"rec-{time.strftime('%Y%m%d-%H%M%S')}"
        for k in range(1, 1000):
            folder = self.store / (base if k == 1 else f"{base}-{k}")
            try:
                folder.mkdir()
            except FileExistsError:
                continue
            cleanup.record(folder, self.store)
            return folder
        raise RuntimeError("could not make a recording folder")

    def capture(self) -> bool:
        """Save one frame now. False once the recording should stop (cap reached or
        feed gone); the reason is set and nothing more is written."""
        if self.folder is None:
            raise RuntimeError("recorder not started")
        if self.reason:
            return False
        now = self.clock()
        if now - self._start >= self.max_seconds:
            return self._finish("cap", self._start + self.max_seconds)
        if not getattr(self.feed, "running", True):
            return self._finish("feed", now)
        seq = getattr(self.feed, "seq", None)  # read first: a newer frame may come with it
        if seq is not None and seq == self._seq:  # nothing new since the last frame saved
            if now - self._fresh >= STALL_S:
                return self._finish("stalled", now)
            return True
        frame = self.feed.latest()
        if frame is None:
            return True
        small = camera.fit(frame, self.max_side)
        score = self._activity.update(small)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return True
        i = len(self.index)
        path = self.folder / f"{i:06d}.jpg"
        path.write_bytes(buf.tobytes())  # the frame first, so the index never names a missing file
        cleanup.record(path, self.store)
        entry = {"i": i, "t": round(now - self._start, 3), "file": path.name, "activity": score}
        index = self.folder / INDEX
        with index.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        if i == 0:
            cleanup.record(index, self.store)
        self.index.append(entry)
        self._seq, self._fresh = seq, now
        return True

    def _finish(self, reason: str, end: float) -> bool:
        global _current
        self.reason, self._end = reason, end
        with _guard:
            if _current is self:
                _current = None
        return False

    def _run(self) -> None:
        period = 1.0 / self.fps
        while not self._stop.is_set():
            self._next += period
            wait = self._next - self.clock()
            if wait < -period:  # fell behind: do not burst to catch up
                self._next, wait = self.clock(), 0.0
            if self._stop.wait(max(0.0, wait)):
                break
            try:
                if not self.capture():
                    break
            except Exception as err:  # disk full, folder deleted: stop and say why
                self._finish(f"error: {err}", self.clock())
                break

    def stop(self) -> dict:
        """Stop (if still going) and return the summary. Safe to call again."""
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)
        if not self.reason and self.folder is not None:
            self._finish("stopped", self.clock())
        return self.summary()

    def summary(self) -> dict:
        """What was recorded: folder, frames, duration, the activity hint."""
        acts = [e["activity"] for e in self.index]
        return {"folder": str(self.folder) if self.folder else None, "frames": len(self.index),
                "duration_s": self.elapsed(), "fps": self.fps, "max_seconds": self.max_seconds,
                "roi": list(self.roi) if self.roi else None, "stopped": self.reason or None,
                "activity": timeline(self.index), "peak_activity": max(acts, default=0.0)}

    def status(self) -> dict:
        """For the viewer's REC badge."""
        return {"recording": self.running, "elapsed_s": self.elapsed(), "max_seconds": self.max_seconds,
                "frames": len(self.index), "folder": self.folder.name if self.folder else None}


# ── reading recordings back ──

def load_index(folder: Path | str) -> list[dict]:
    """A recording's frames, in order. A half-written last line (still recording) is skipped."""
    out = []
    try:
        lines = (Path(folder) / INDEX).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if isinstance(e, dict) and {"t", "file"} <= e.keys():
            out.append(e)
    return out


def recordings(store: Path | str) -> list[Path]:
    """The store's recording folders, oldest first."""
    def key(p: Path):
        parts = p.name.split("-")
        return parts[1], parts[2], int(parts[3]) if len(parts) > 3 else 1
    try:
        found = [p for p in Path(store).iterdir() if cleanup.rec_folder(p)]
    except OSError:
        return []
    return sorted(found, key=key)


def find(store: Path | str, name: str | None = None) -> Path:
    """The named recording folder in the store (rec-...), else the newest one."""
    if name:
        given = Path(name.rstrip("/"))
        folder = Path(store) / given.name  # a bare name, or a path to one in this store
        path_given = str(given) != given.name
        if not cleanup.rec_folder(folder) or (path_given and given.resolve() != folder.resolve()):
            raise ValueError(f"no recording {name!r} in {store}")
        return folder
    found = recordings(store)
    if not found:
        raise ValueError("no recordings yet: start_recording first")
    return found[-1]


def started(folder: Path | str) -> str:
    """When a recording began, from its folder name: rec-20261106-090000 -> 2026-11-06 09:00:00."""
    parts = Path(folder).name.split("-")
    try:
        d, t = parts[1], parts[2]
        return f"{d[:4]}-{d[4:6]}-{d[6:]} {t[:2]}:{t[2:4]}:{t[4:]}"
    except IndexError:
        return "?"


def labeled_sheet(items: list[tuple[float, np.ndarray]], max_side: int = 1600) -> np.ndarray:
    """One contact sheet of (seconds, frame) pairs, each tile labeled with its number and
    time ("3  4.2s"), fitted inside max_side. Tiles are sized first so labels stay readable."""
    if not items:
        raise ValueError("no frames")
    n = len(items)
    cols = min(n, math.ceil(math.sqrt(n * 4 / 3)))
    rows = math.ceil(n / cols)
    h, w = items[0][1].shape[:2]
    tile_w = max(32, min(max_side // cols, int(max_side / rows * w / h), w))
    tile_h = max(24, round(tile_w * h / w))
    tiles = [cv2.resize(f, (tile_w, tile_h), interpolation=cv2.INTER_AREA) for _, f in items]
    labels = [f"{k}  {t:.1f}s" for k, (t, _) in enumerate(items, 1)]
    return camera.contact_sheet(tiles, cols=cols, labels=labels)


def review(folder: Path | str, start_s: float | None = None, end_s: float | None = None,
           frames: int = 12, max_side: int = 1600, store: Path | None = None) -> tuple[camera.Shot, list[dict]]:
    """A labeled contact sheet of evenly spaced frames from a recording (it may still be
    going), saved with camera.save so cleanup deletes it. Returns (shot, the entries shown)."""
    folder = Path(folder)
    index = load_index(folder)
    if not index:
        raise ValueError(f"{folder.name} has no frames yet")
    picked = sample(index, start_s, end_s, frames)
    if not picked:
        raise ValueError(f"no frames between {start_s}s and {end_s}s; the recording runs "
                         f"{index[0]['t']:.1f}s to {index[-1]['t']:.1f}s")
    items = []
    for e in picked:
        img = cv2.imread(str(folder / Path(e["file"]).name))
        if img is not None:
            items.append((float(e["t"]), img))
    if not items:
        raise ValueError(f"the frames of {folder.name} are gone")
    sheet = labeled_sheet(items, max_side)
    shot = camera.save(sheet, "review") if store is None else camera.save(sheet, "review", store=store)
    return shot, picked


def video_frames(path: Path | str, start_s: float | None = None, end_s: float | None = None,
                 frames: int = 12, max_side: int = 640) -> list[tuple[float, np.ndarray]]:
    """(seconds, frame) pairs spread evenly over [start_s, end_s] of a video file (e.g. a
    LeRobot episode mp4), found by seeking. Each time is that of the frame actually read.
    Nothing is copied: frames are decoded in memory only."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise ValueError(f"no video file at {path}")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"could not open {path} as a video")
        fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or count <= 0:
            raise ValueError(f"could not read the frame rate or length of {path}")
        last = (count - 1) / fps
        lo = 0.0 if start_s is None else min(max(0.0, float(start_s)), last)
        hi = last if end_s is None else min(max(0.0, float(end_s)), last)
        if lo > hi:
            raise ValueError(f"start_s is after end_s; the video runs 0.0s to {last:.1f}s")
        picks = sorted({min(count - 1, round(t * fps)) for t in np.linspace(lo, hi, max(1, frames))})
        out = []
        for k in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            ok, frame = cap.read()
            if ok and frame is not None:
                out.append((k / fps, camera.fit(frame, max_side)))
        if not out:
            raise ValueError(f"could not decode frames of {path}")
        return out
    finally:
        cap.release()


def review_video(path: Path | str, start_s: float | None = None, end_s: float | None = None,
                 frames: int = 12, max_side: int = 1600) -> camera.Shot:
    """A labeled contact sheet of evenly spaced frames from a video file, saved with
    camera.save so cleanup deletes it. The video itself is never copied."""
    sheet = labeled_sheet(video_frames(path, start_s, end_s, frames), max_side)
    return camera.save(sheet, "review")
