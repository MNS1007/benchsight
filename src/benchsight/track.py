"""Follow the boxes the agent chose, frame by frame, on the CPU, with no model calls.

The agent is the zero-shot part: it looks at a photo and gives each object a label and a
box. Here one small OpenCV tracker per object follows that box on the live feed, at about
640 px. VitTrack (one ONNX file under 1 MB, downloaded once into ~/.cache/benchsight/models
and checked against a pinned sha256) is preferred; TrackerMIL, which needs no file, is the
fallback when offline. Boxes are fractions 0-1 of the frame, as in every other tool.

Objects are remembered like in a scene graph: a new box whose label shares a word with a
nearby object's ("slider plate" next to "white plate") merges into that object's id, shown
boxes are smoothed, and lost objects keep their last box while their confidence decays.

Registry ideas adapted from VISTA (UMD Ironsite Hackathon 2025, Team Cortex)
"""
from __future__ import annotations

import hashlib
import os
import re
import tempfile
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import motion

MODEL_DIR = Path(os.environ.get("BENCHSIGHT_MODELS", Path.home() / ".cache" / "benchsight" / "models"))
MODELS = {
    "vittrack": {
        "file": "object_tracking_vittrack_2023sep.onnx",
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx",
        "sha256": "2990f0b7cd44d92afa48cd97db6de7be113fc1d9594fddb74e2725c10478e91d",
    },
}
MAX_OBJECTS = 8
LOST_AFTER = 3  # bad frames in a row before an object is marked lost
VIT_MIN_SCORE = 0.3
MIN_LIKENESS = 0.3  # MIL gives no score: compare the box to the agent's crop instead
ALPHA = 0.3  # EMA weight of the newest box, so drawn boxes don't jitter
DECAY = 0.015  # confidence lost per second while an object is not seen
STALE_BELOW = 0.5  # lost for ~33 s: shown as stale
PRUNE_BELOW = 0.2  # lost for ~53 s: forgotten
STOPWORDS = {"the", "a", "an", "of", "and", "or", "in", "on", "at", "to", "is", "with", "for"}


# ── model files ─────────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_path(name: str = "vittrack", allow_download: bool = True, timeout: float = 20.0,
               model_dir: Path | None = None) -> Path | None:
    """The checked model file, downloading it once if allowed. None when offline or the
    download does not match its pinned hash (the caller then falls back)."""
    spec = MODELS[name]
    folder = Path(model_dir or MODEL_DIR)
    path = folder / spec["file"]
    if path.exists():
        if _sha256(path) == spec["sha256"]:
            return path
        path.unlink(missing_ok=True)  # corrupt or replaced: fetch again
    if not allow_download:
        return None
    try:
        folder.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(spec["url"], timeout=timeout) as r:
            data = r.read()
        if hashlib.sha256(data).hexdigest() != spec["sha256"]:
            return None
        fd, tmp = tempfile.mkstemp(dir=folder, suffix=".part")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
        return path
    except OSError:  # URLError, timeouts, read-only disk: work without the model
        return None


# ── tracker backends ────────────────────────────────────────────────────────

class Backend:
    """Makes one OpenCV tracker per object and reads its score (None if it has none)."""

    def __init__(self, name: str, net: Path | None = None):
        self.name, self.net = name, net

    def create(self):
        if self.name == "vit":
            p = cv2.TrackerVit_Params()
            p.net = str(self.net)
            level = cv2.utils.logging.getLogLevel()
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)  # a harmless DNN target warning
            try:
                return cv2.TrackerVit_create(p)
            finally:
                cv2.utils.logging.setLogLevel(level)
        return cv2.TrackerMIL_create()

    def score(self, tracker) -> float | None:
        return float(tracker.getTrackingScore()) if self.name == "vit" else None


def make_backend(prefer: str = "auto", allow_download: bool = True, model_dir: Path | None = None) -> Backend:
    """VitTrack when OpenCV has it and the model is available, else MIL.
    (NanoTrack needs two model files with two hashes and is not better than VitTrack, so
    it is left out.)"""
    if prefer in ("auto", "vit") and hasattr(cv2, "TrackerVit_create"):
        net = model_path("vittrack", allow_download=allow_download, model_dir=model_dir)
        if net is not None:
            b = Backend("vit", net)
            try:
                b.create()
                return b
            except cv2.error:
                pass
    if prefer == "vit":
        raise RuntimeError("VitTrack is not available (no model file, or this OpenCV lacks TrackerVit)")
    return Backend("mil")


# ── tracked objects ─────────────────────────────────────────────────────────

def _to_px(box, w: int, h: int) -> list[float]:
    return [box[0] * w, box[1] * h, box[2] * w, box[3] * h]


def _to_frac(box, w: int, h: int) -> list[float]:
    return [min(1.0, max(0.0, v)) for v in (box[0] / w, box[1] / h, box[2] / w, box[3] / h)]


def _clip(box, w: int, h: int) -> list[int]:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1 = max(0, min(w - 2, x1)), max(0, min(h - 2, y1))
    return [x1, y1, max(x1 + 2, min(w, x2)), max(y1 + 2, min(h, y2))]


def check_box(box) -> list[float]:
    if not (isinstance(box, (list, tuple)) and len(box) == 4):
        raise ValueError("box must be [x1, y1, x2, y2] as fractions 0-1")
    x1, y1, x2, y2 = (float(v) for v in box)
    if not all(0 <= v <= 1 for v in (x1, y1, x2, y2)):
        raise ValueError(f"box {list(box)} is outside the frame; use fractions 0-1")
    x1, x2, y1, y2 = min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)
    if x2 - x1 < 0.01 or y2 - y1 < 0.01:
        raise ValueError(f"box {list(box)} is too small to track")
    return [x1, y1, x2, y2]


def likeness(g: np.ndarray, box, template: np.ndarray, pad: float = 0.25, side: int = 24) -> float:
    """How much the frame around box looks like the template, -1 to 1: the best normalized
    correlation of the template, scaled to the box and shrunk to ~side px, within the box
    grown by pad. Tolerant of a few pixels of tracker misalignment."""
    h, w = g.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw < 2 or bh < 2 or template.size == 0:
        return -1.0
    f = side / max(bw, bh)
    tw, th = max(4, round(bw * f)), max(4, round(bh * f))
    rx1, ry1, rx2, ry2 = _clip([x1 - pad * bw, y1 - pad * bh, x2 + pad * bw, y2 + pad * bh], w, h)
    region = cv2.resize(g[ry1:ry2, rx1:rx2], (max(tw, round((rx2 - rx1) * f)), max(th, round((ry2 - ry1) * f))),
                        interpolation=cv2.INTER_AREA)
    tpl = cv2.resize(template, (tw, th), interpolation=cv2.INTER_AREA)
    return float(cv2.matchTemplate(region, tpl, cv2.TM_CCOEFF_NORMED).max())


@dataclass
class TrackedObject:
    id: str
    label: str
    raw: list[float]  # tracker box in tracking-frame pixels
    box: list[float]  # the box shown and returned, same pixels
    template: np.ndarray  # grayscale crop from the frame the agent saw
    tracker: object = None
    status: str = "tracking"  # tracking, lost or stale
    score: float = 1.0
    confidence: float = 1.0
    first_seen: float = 0.0
    last_seen: float = 0.0
    bad: int = 0  # bad frames in a row

    def snapshot(self, w: int, h: int, backend: str) -> dict:
        return {"id": self.id, "label": self.label, "box": [round(v, 4) for v in _to_frac(self.box, w, h)],
                "status": self.status, "score": round(self.score, 2), "confidence": round(self.confidence, 2),
                "last_seen": round(self.last_seen, 2), "backend": backend}


def _label_words(label: str) -> set[str]:
    """Significant words of a label: 'M3 screw_2' -> {'screw'}."""
    words = re.split(r"[^a-z0-9]+", label.lower())
    return {w for w in words if len(w) > 2 and not w.isdigit() and w not in STOPWORDS}


def labels_match(a: str, b: str) -> bool:
    """Exact, substring, or a shared root word ('slider plate' ~ 'white plate')."""
    a, b = a.lower().strip(), b.lower().strip()
    return (a == b or (len(a) > 3 and a in b) or (len(b) > 3 and b in a)
            or bool(_label_words(a) & _label_words(b)))


def _near(a, b) -> float | None:
    """Center distance of two fractional boxes if they are close enough to be the same
    object (within 0.75 of the larger box's longer side, at least 0.1), else None."""
    d = float(np.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2))
    reach = max(0.1, 0.75 * max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1]))
    return d if d <= reach else None


def _ema(old, new, alpha: float = ALPHA) -> list[float]:
    return [o * (1 - alpha) + n * alpha for o, n in zip(old, new)]


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:40] or "object"


class Tracker:
    """Tracks objects on a LiveFeed in a background thread, or frame by frame via step().
    All methods are thread-safe; OpenCV trackers are only used under the lock."""

    def __init__(self, feed=None, backend: str | Backend = "auto", allow_download: bool = True,
                 max_objects: int = MAX_OBJECTS, max_side: int = motion.MAX_SIDE, fps: float = 15.0,
                 cv_threads: int | None = 1, clock=time.time):
        self.feed, self.max_objects, self.max_side, self.fps = feed, max_objects, max_side, fps
        self.cv_threads, self.clock = cv_threads, clock
        self.backend = backend if isinstance(backend, Backend) else make_backend(backend, allow_download)
        self._objs: dict[str, TrackedObject] = {}
        self._lock = threading.RLock()
        self._size: tuple[int, int] | None = None  # (w, h) of the tracking frame
        self._last: np.ndarray | None = None  # last tracking frame (small, BGR)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.steps = 0
        self.error = ""
        self._t: float | None = None  # time of the last step, for decay
        self.changes: deque[dict] = deque(maxlen=200)  # object_lost / object_pruned, newest last

    # ── frames ──

    def _small(self, frame: np.ndarray) -> np.ndarray:
        return motion.downscale(frame, self.max_side)[0]

    # ── public API ──

    def add(self, objects: list[dict], frame: np.ndarray | None = None, replace: bool = False) -> list[dict]:
        """Start tracking [{id?, label, box: [x1, y1, x2, y2]}] (fractions 0-1). frame is the
        image the agent chose the boxes on; by default the latest frame. Returns
        [{id, label}] in the same order."""
        if frame is None and self.feed is not None:
            frame = self.feed.latest()
        with self._lock:
            ref = self._small(frame) if frame is not None else self._last
            if ref is None:
                raise ValueError("no camera frame yet; open the viewer or take a look first")
            if self._last is not None and ref.shape != self._last.shape:  # keep one pixel space
                ref = cv2.resize(ref, (self._last.shape[1], self._last.shape[0]), interpolation=cv2.INTER_AREA)
            items = [(str(o.get("label") or o.get("id") or "object")[:60], check_box(o.get("box")), o.get("id"))
                     for o in objects]
            now = self.clock()
            h, w = ref.shape[:2]
            g = motion.gray(ref)
            out, keep = [], set()
            for label, box, oid in items:
                obj = self._match(oid, label, box, w, h)
                if obj is None:
                    self._make_room()
                    obj = TrackedObject(self._new_id(oid or label), label, [], [], g, first_seen=now)
                    self._objs[obj.id] = obj
                obj.label = label
                self._init(obj, ref, _to_px(box, w, h), now)
                keep.add(obj.id)
                out.append({"id": obj.id, "label": obj.label})
            if replace:
                for oid in [k for k in self._objs if k not in keep]:
                    del self._objs[oid]
            if self._size is None or self._size != (w, h):
                self._size = (w, h)
            return out

    def untrack(self, ids: list[str] | None = None) -> list[str]:
        """Stop tracking these ids (all when None). Returns the ids removed."""
        with self._lock:
            gone = list(self._objs) if ids is None else [i for i in ids if i in self._objs]
            for i in gone:
                del self._objs[i]
            return gone

    def objects(self) -> list[dict]:
        """Snapshot of every object: id, label, box (fractions), status, score, last_seen."""
        with self._lock:
            if self._size is None:
                return []
            w, h = self._size
            return [o.snapshot(w, h, self.backend.name) for o in self._objs.values()]

    # ── internals ──

    def _new_id(self, base: str) -> str:
        base = _slug(base)
        oid, n = base, 2
        while oid in self._objs:
            oid, n = f"{base}-{n}", n + 1
        return oid

    def _match(self, oid, label, box, w, h) -> TrackedObject | None:
        """The object a new box belongs to: the same id if given, else the nearest object
        with a matching label (as VISTA's scene graph merges repeat sightings)."""
        if oid:
            return self._objs.get(oid)
        best, best_d = None, None
        for obj in self._objs.values():
            if not labels_match(label, obj.label):
                continue
            d = _near(box, _to_frac(obj.box, w, h))
            if d is not None and (best_d is None or d < best_d):
                best, best_d = obj, d
        return best

    def _make_room(self) -> None:
        """Forget the least confident lost object if full; refuse if all are tracking."""
        if len(self._objs) < self.max_objects:
            return
        idle = [o for o in self._objs.values() if o.status != "tracking"]
        if not idle:
            raise ValueError(f"already tracking {self.max_objects} objects; untrack some first")
        del self._objs[min(idle, key=lambda o: o.confidence).id]

    def _note(self, event: str, obj: TrackedObject, now: float) -> None:
        self.changes.append({"event": event, "id": obj.id, "label": obj.label, "t": round(now, 2)})

    def _decay(self, now: float) -> None:
        dt = 0.0 if self._t is None else max(0.0, now - self._t)
        self._t = now
        for obj in list(self._objs.values()):
            if obj.status == "tracking":
                continue
            obj.confidence = max(0.0, obj.confidence - DECAY * dt)
            if obj.confidence < PRUNE_BELOW:
                del self._objs[obj.id]
                self._note("object_pruned", obj, now)
            elif obj.confidence < STALE_BELOW:
                obj.status = "stale"

    def _init(self, obj: TrackedObject, small: np.ndarray, box_px, now: float) -> None:
        h, w = small.shape[:2]
        x1, y1, x2, y2 = _clip(box_px, w, h)
        obj.template = motion.gray(small)[y1:y2, x1:x2].copy()
        obj.tracker = self.backend.create()
        obj.tracker.init(small, (x1, y1, x2 - x1, y2 - y1))
        obj.raw, obj.box = [x1, y1, x2, y2], [float(v) for v in (x1, y1, x2, y2)]
        obj.status, obj.bad, obj.score, obj.confidence, obj.last_seen = "tracking", 0, 1.0, 1.0, now

    def _plausible(self, old, new) -> bool:
        ow, oh, nw, nh = old[2] - old[0], old[3] - old[1], new[2] - new[0], new[3] - new[1]
        if nw < 2 or nh < 2:
            return False
        jump = np.hypot((new[0] + new[2] - old[0] - old[2]) / 2, (new[1] + new[3] - old[1] - old[3]) / 2)
        area = (nw * nh) / max(1.0, ow * oh)
        return jump <= 1.5 * max(ow, oh) and 0.4 <= area <= 2.5

    def _update(self, obj: TrackedObject, small: np.ndarray, g: np.ndarray, now: float) -> None:
        ok, rect = obj.tracker.update(small)
        x, y, bw, bh = rect
        new = [float(x), float(y), float(x + bw), float(y + bh)]
        score = self.backend.score(obj.tracker)
        good = bool(ok) and self._plausible(obj.raw, new)
        if good:
            like = likeness(g, new, obj.template)
            if score is None:
                score, good = like, like >= MIN_LIKENESS
            else:
                good = score >= VIT_MIN_SCORE
        if good:
            obj.raw, obj.score, obj.bad = new, float(score), 0
            obj.box = _ema(obj.box, new)
            obj.last_seen, obj.confidence = now, 1.0
            return
        obj.bad += 1
        if obj.bad >= LOST_AFTER:
            obj.status, obj.tracker = "lost", None
            self._note("object_lost", obj, now)

    def step(self, frame: np.ndarray, now: float | None = None) -> None:
        """Track every object on one frame (full size; it is downscaled here)."""
        now = self.clock() if now is None else now
        small = self._small(frame)
        g = motion.gray(small)
        with self._lock:
            self._last = small
            self._size = (small.shape[1], small.shape[0])
            for obj in list(self._objs.values()):
                if obj.status == "tracking" and obj.tracker is not None:
                    self._update(obj, small, g, now)
            self._decay(now)
            self.steps += 1

    # ── thread ──

    def start(self) -> None:
        if self.feed is None:
            raise ValueError("no feed to track")
        if self.running:
            return
        if self.cv_threads is not None:
            # Process-wide. Small-ROI trackers gain little from threads but cost a lot of
            # CPU (MIL: 0.40 cpu-s per update on 8 threads vs 0.05 on 1), so default to 1.
            cv2.setNumThreads(self.cv_threads)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="benchsight-tracker")
        self._thread.start()

    def _run(self) -> None:
        period, seen = 1.0 / self.fps, -1
        while not self._stop.is_set():
            t0 = time.time()
            seq = getattr(self.feed, "seq", None)
            if seq is None or seq != seen:
                frame = self.feed.latest()
                if frame is not None:
                    seen = seq if seq is not None else seen
                    try:
                        self.step(frame)
                    except Exception as err:  # keep tracking alive; report via .error
                        self.error = str(err)
            self._stop.wait(max(0.0, period - (time.time() - t0)))

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._thread = None
