"""Follow the boxes the agent chose, frame by frame, on the CPU, with no model calls.

The agent is the zero-shot part: it looks at a photo and gives each object a label and a
box. Here one small OpenCV tracker per object follows that box on the live feed, at about
640 px. VitTrack (one ONNX file under 1 MB, downloaded once into ~/.cache/benchsight/models
and checked against a pinned sha256) is preferred; TrackerMIL, which needs no file, is the
fallback when offline. Boxes are fractions 0-1 of the frame, as in every other tool.

A lost object is looked for again every few frames: its crop from the agent's frame is
matched with ORB (falling back to template matching for plain parts) near its last box,
then in the whole frame. When the whole frame shifts (the laptop is bumped), the camera's
motion is applied to every box first and the trackers refine from there.

The live tracker keeps to a CPU budget (BUDGET, a share of one core): when updating every
object every frame would cost more, the objects updated longest ago go first and the rest
wait (at most MAX_STRIDE - 1 frames), while camera motion still moves every box.

Objects are remembered like in a scene graph: a new box whose label shares a word with an
overlapping object's ("slider plate" over "white plate") merges into that object's id, shown
boxes are smoothed, and lost objects keep their last box while their confidence decays.

Registry ideas adapted from VISTA (UMD Ironsite Hackathon 2025, Team Cortex)
"""
from __future__ import annotations

import hashlib
import http.client
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
REACQUIRE_EVERY = 5  # frames between searches for each lost object
MIN_INLIERS = 10  # ORB matches agreeing on one placement to accept a re-found object
MIN_MATCH = 0.65  # template-matching score to accept one
MERGE_IOU = 0.3  # overlap needed to merge a box whose label only shares a word
BUDGET = 0.15  # share of one core the live tracker aims for; objects beyond it update on later frames
MAX_STRIDE = 4  # but every tracked object updates at least every 4th frame, whatever it costs
MOTION_SIDE = 320  # camera motion is estimated at this size: 3x cheaper than 640, and enough
MIL_WINDOW = 15.0  # MIL's search radius in px (OpenCV's default 25 costs 34 ms an update, this 19)
IDLE_PERIOD = 0.5  # with nothing tracked, camera motion and events run at 2 Hz only
COMPENSATE = 0.3  # re-init a tracker when camera motion moved its box this share of its size
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
    except (OSError, http.client.HTTPException, ValueError):
        # URLError, timeouts, truncated downloads, read-only disk: work without the model
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
        p = cv2.TrackerMIL_Params()
        p.samplerSearchWinSize = MIL_WINDOW
        return cv2.TrackerMIL_create(p)

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


def _find_in(region: np.ndarray, template: np.ndarray, kp, des) -> list[float] | None:
    th, tw = template.shape[:2]
    if des is not None and len(kp) >= MIN_INLIERS:  # ORB: survives rotation and scale
        kp2, des2 = motion.orb_features(region, orb=motion.ORB_SMALL)
        M, _ = motion.affine_from_points(*motion.orb_match(kp, des, kp2, des2), min_points=MIN_INLIERS)
        if M is not None and 0.5 <= float(np.hypot(M[0, 0], M[1, 0])) <= 2.0:
            return motion.apply(M, [0, 0, tw, th])
    f = 0.5 if min(tw, th) >= 24 else 1.0  # plain parts: multi-scale template matching
    reg = cv2.resize(region, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f != 1 else region
    best = (MIN_MATCH, None)
    for sc in (0.8, 1.0, 1.25):
        t = cv2.resize(template, (max(4, round(tw * sc * f)), max(4, round(th * sc * f))), interpolation=cv2.INTER_AREA)
        if t.shape[0] > reg.shape[0] or t.shape[1] > reg.shape[1]:
            continue
        _, val, _, (x, y) = cv2.minMaxLoc(cv2.matchTemplate(reg, t, cv2.TM_CCOEFF_NORMED))
        if val > best[0]:
            best = (val, [x / f, y / f, (x + t.shape[1]) / f, (y + t.shape[0]) / f])
    return best[1]


def find(g: np.ndarray, template: np.ndarray, kp=None, des=None, near=None, full: bool = True) -> list[float] | None:
    """Where the template is in grayscale frame g, as a pixel box, or None. Searches a
    window around `near` (a pixel box) first, then (if full) the whole frame."""
    h, w = g.shape[:2]
    regions = []
    if near is not None:
        bw, bh = near[2] - near[0], near[3] - near[1]
        regions.append(_clip([near[0] - 1.5 * bw, near[1] - 1.5 * bh, near[2] + 1.5 * bw, near[3] + 1.5 * bh], w, h))
    if full or near is None:
        regions.append([0, 0, w, h])
    for rx1, ry1, rx2, ry2 in regions:
        box = _find_in(g[ry1:ry2, rx1:rx2], template, kp, des)
        if box is not None:
            return [box[0] + rx1, box[1] + ry1, box[2] + rx1, box[3] + ry1]
    return None


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


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
    kp: tuple = ()  # ORB features of the template, for re-acquiring
    des: np.ndarray | None = None
    tries: int = 0  # steps since the object was lost
    updated: int = -1  # step of the last tracker update, for round-robin under the budget
    fresh: bool = True  # (re)started and not updated since: updates next step, budget or not

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
                 cv_threads: int | None = 1, clock=time.time, events=None, budget: float | None = None):
        """events: an events.Events to write wake-ups to (lost, found, scene changes).
        budget: share of one core each step aims to stay under (at `fps` steps a second).
        When updating every object would cost more, the objects seen longest ago update
        first and the rest wait a frame or more (at most MAX_STRIDE - 1), while camera
        motion still moves every box. None (the default here; the server uses BUDGET)
        updates every object every frame."""
        self.feed, self.max_objects, self.max_side, self.fps = feed, max_objects, max_side, fps
        self.budget = budget
        self._cost: float | None = None  # running mean of one object update, CPU seconds
        self._credit = 0.0  # CPU seconds the budget has left over, see step()
        self.events = events
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
        self.changes: deque[dict] = deque(maxlen=200)  # object_lost/found/pruned, newest last
        self.cam = motion.CameraMotion(max_side=min(MOTION_SIDE, max_side))
        self.motion = motion.Motion(confidence=1.0)  # the camera's motion on the last step
        self._idle_t: float | None = None  # last idle step that measured motion
        self._skipped = False  # an idle step skipped motion, so self.cam's last frame is old

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
            out, keep = [], set()  # keep: ids made or matched in this call
            for label, box, oid in items:
                obj = self._match(oid, label, box, w, h, skip=keep)
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

    def _match(self, oid, label, box, w, h, skip=()) -> TrackedObject | None:
        """The object a new box belongs to: the same id if given, else the nearest object
        with a matching label (as VISTA's scene graph merges repeat sightings). Objects in
        skip (made or matched earlier in the same call) are never merged into: one call
        lists distinct objects, like a left and a right screw. A label that only shares a
        word ('black wire' vs 'red wire') must also overlap the old box, so two parts side
        by side stay two; the same label merges anywhere nearby."""
        if oid:
            return self._objs.get(oid)
        best, best_d = None, None
        for obj in self._objs.values():
            if obj.id in skip or not labels_match(label, obj.label):
                continue
            old = _to_frac(obj.box, w, h)
            if label.lower().strip() != obj.label.lower().strip() and _iou(box, old) < MERGE_IOU:
                continue
            d = _near(box, old)
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
        if self.events is not None and event in ("object_lost", "object_found"):
            self.events.add(event, now, id=obj.id, label=obj.label)

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

    def _init(self, obj: TrackedObject, small: np.ndarray, box_px, now: float, template: bool = True) -> None:
        """(Re)start the object's tracker at box_px. template=False keeps the agent's crop
        (re-acquire and camera compensation must not replace it with what is there now)."""
        h, w = small.shape[:2]
        x1, y1, x2, y2 = _clip(box_px, w, h)
        if template:
            obj.template = motion.gray(small)[y1:y2, x1:x2].copy()
            obj.kp, obj.des = motion.orb_features(obj.template, orb=motion.ORB_SMALL)
        obj.tries, obj.updated, obj.fresh = 0, self.steps - 1, True  # its box is exact now; update it next step
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
        # Smooth as if every waited frame had been seen: one update after k frames moves
        # the shown box as far as k updates would have, so waiting objects don't lag.
        alpha = 1 - (1 - ALPHA) ** max(1, min(MAX_STRIDE, self.steps - obj.updated))
        ok, rect = obj.tracker.update(small)
        x, y, bw, bh = rect
        new = [float(x), float(y), float(x + bw), float(y + bh)]
        score = self.backend.score(obj.tracker)
        good = bool(ok) and self._plausible(obj.raw, new)
        if good and score is None:  # MIL: judge by likeness to the agent's crop
            score = likeness(g, new, obj.template)
            good = score >= MIN_LIKENESS
        elif good:
            good = score >= VIT_MIN_SCORE
        if good:
            obj.raw, obj.score, obj.bad = new, float(score), 0
            obj.box = _ema(obj.box, new, alpha)
            obj.last_seen, obj.confidence = now, 1.0
            return
        # A tracker that slipped off (MIL drifts) but the object is still close by: snap
        # back with a local search instead of counting a bad frame.
        near = find(g, obj.template, obj.kp, obj.des, near=obj.raw, full=False)
        if near is not None and self._plausible(obj.raw, near):
            shown = obj.box
            self._init(obj, small, near, now, template=False)
            obj.box = _ema(shown, near, alpha)
            return
        obj.bad += 1
        if obj.bad >= LOST_AFTER:
            obj.status, obj.tracker = "lost", None
            self._note("object_lost", obj, now)

    def _compensate(self, small: np.ndarray, g: np.ndarray, now: float) -> None:
        """Estimate how the whole frame moved, leaving tracked objects out, and move every
        box with it. OpenCV trackers can't be told to shift their search window, so a
        tracker is re-initialized at its moved box only when the shift is large for it;
        small shifts are left to the tracker to refine."""
        mask = np.full(g.shape, 255, np.uint8)
        for obj in self._objs.values():
            if obj.status == "tracking":
                x1, y1, x2, y2 = obj.raw
                px, py = 0.2 * (x2 - x1), 0.2 * (y2 - y1)
                mask[max(0, int(y1 - py)):max(0, int(y2 + py)), max(0, int(x1 - px)):max(0, int(x2 + px))] = 0
        m = self.motion = self.cam.update(g, mask)
        if m.confidence < 0.5 or m.magnitude < 1.0:
            return
        for obj in self._objs.values():
            moved = motion.apply(m.M, obj.raw)
            obj.box = motion.apply(m.M, obj.box)
            if obj.status != "tracking" or obj.tracker is None:
                obj.raw = moved
                continue
            shift = np.hypot(moved[0] - obj.raw[0], moved[1] - obj.raw[1])
            if shift > COMPENSATE * min(obj.raw[2] - obj.raw[0], obj.raw[3] - obj.raw[1]):
                shown = obj.box
                self._init(obj, small, moved, now, template=False)
                obj.box = shown

    def _reacquire(self, obj: TrackedObject, small: np.ndarray, g: np.ndarray, now: float) -> None:
        obj.tries += 1
        if obj.tries % REACQUIRE_EVERY:
            return
        box = find(g, obj.template, obj.kp, obj.des, near=obj.raw)
        if box is None or any(o is not obj and o.status == "tracking" and _iou(box, o.raw) > 0.5
                              for o in self._objs.values()):
            return
        self._init(obj, small, box, now, template=False)
        self._note("object_found", obj, now)

    def step(self, frame: np.ndarray, now: float | None = None) -> None:
        """Track every object on one frame (full size; it is downscaled here)."""
        c0 = time.thread_time()  # CPU time of this thread, for the budget
        now = self.clock() if now is None else now
        small = self._small(frame)
        g = motion.gray(small)
        with self._lock:
            if self._last is not None and self._last.shape != small.shape:
                self.cam.reset()
            self._last = small
            self._size = (small.shape[1], small.shape[0])
            if not self._objs:  # idle: only scene events need motion, and 2 Hz is enough
                if self._idle_t is not None and 0 <= now - self._idle_t < IDLE_PERIOD:
                    self._skipped = True
                    self.steps += 1
                    return
                self._idle_t = now
            elif self._skipped:  # tracking again: don't apply motion measured from an old frame
                self.cam.reset()
                self._skipped, self._idle_t = False, None
            self._compensate(small, g, now)
            live = [o for o in self._objs.values() if o.status == "tracking" and o.tracker is not None]
            if self.budget is not None:  # CPU credit: earned each step, spent by motion and updates
                allow = self.budget / self.fps
                spent = time.thread_time() - c0
                self._credit = min(allow * MAX_STRIDE, max(-allow, self._credit + allow - spent))
            for obj in sorted(live, key=lambda o: o.updated):  # longest waiting first
                if (self.budget is not None and not obj.fresh and self.steps - obj.updated < MAX_STRIDE
                        and self._credit < (self._cost or 0.0)):
                    continue  # over budget: this one waits; camera motion has moved its box
                s = time.thread_time()
                self._update(obj, small, g, now)
                cost = time.thread_time() - s
                self._cost = cost if self._cost is None else 0.8 * self._cost + 0.2 * cost
                if self.budget is not None:
                    self._credit = max(-allow, self._credit - cost)
                obj.updated, obj.fresh = self.steps, False
            updating = {o.id for o in live}
            for obj in list(self._objs.values()):
                if obj.id not in updating:
                    self._reacquire(obj, small, g, now)
            self._decay(now)
            if self.events is not None:
                self.events.observe(g, self.motion, now)
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
