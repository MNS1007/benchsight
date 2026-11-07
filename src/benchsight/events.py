"""Wake-up events for the agent, written while the tracker runs.

An agent can't watch video; it acts only on a message or a tool result. So the tracker
thread writes a short JSON line to <store>/events.jsonl when something worth a look
happens, and `benchsight events --follow` (run under Claude Code's background Monitor
tool) turns each line into a wake-up:

  object_lost    a tracked object left view or got covered
  object_found   a lost object was seen again
  scene_changed  the camera moved a lot, the view was suddenly blocked, or the picture
                 changed a lot (hands, a part swapped)
  still          the scene has been stable for about 2 s after motion: a good moment to look

Debounced: one scene_changed per stretch of motion (then "still" when it ends), and at
most one line about every 5 s. Everything that happens inside that window
is merged into the next line, newest state per object (a lost-then-found object inside
one window nets out to nothing). One line looks like:

  {"t": 1762340000.12, "time": "2026-11-05T10:13:20", "event": "object_lost",
   "events": [{"event": "object_lost", "id": "plate", "label": "slider plate"}],
   "summary": "slider plate (plate) lost"}

The debouncer is VISTA's KeyframeSampler, counted in seconds instead of frames.
adapted from VISTA (UMD Ironsite Hackathon 2025, Team Cortex)
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np

COOLDOWN = 5.0  # seconds between wake-ups at most
STILL_AFTER = 2.0  # seconds of calm after motion before "still"
SCENE_THRESHOLD = 40.0  # accumulated motion (camera px at 640 + % of picture changed) for scene_changed
MOVING = 1.0  # per-frame motion above this counts as "something is moving"
PX_FLOOR = 0.5  # camera shifts below this (px per frame) are estimation noise
CHANGED_LEVEL = 15  # a thumbnail pixel changed if it moved by more than this (0-255)
CHANGED_FLOOR = 0.5  # % of the picture changed below this is noise
EPISODE_MIN = 10.0  # motion an episode needs before its end is worth a "still"
SETTLE = 0.5  # seconds of calm that clear the accumulated motion
BLOCKED_FRAMES = 8  # confident motion, then this many low-confidence frames: view blocked
PRIORITY = ("object_lost", "object_found", "scene_changed", "still")


class KeyframeSampler:
    """Decides when to fire. Fire after a maximum interval (if set), or when the
    accumulated motion passes a threshold; never inside a minimum cooldown. Times are
    seconds (VISTA counted frames)."""

    def __init__(self, interval: float | None = None, motion_threshold: float = SCENE_THRESHOLD,
                 min_cooldown: float = COOLDOWN):
        self.interval = interval
        self.motion_threshold = motion_threshold
        self.min_cooldown = min_cooldown
        self._last: float | None = None
        self._accumulated = 0.0

    def should_sample(self, t: float, magnitude: float = 0.0) -> bool:
        if self._last is not None and t - self._last < self.min_cooldown:
            self._accumulated += magnitude
            return False
        if self.interval is not None and (self._last is None or t - self._last >= self.interval):
            self._fire(t)
            return True
        self._accumulated += magnitude
        if self._accumulated >= self.motion_threshold:
            self._fire(t)
            return True
        return False

    def _fire(self, t: float) -> None:
        self._last, self._accumulated = t, 0.0

    def settle(self) -> None:
        """The scene went calm: forget motion so far (benchsight addition; VISTA's
        accumulator only reset on a keyframe, which let small jitter add up)."""
        self._accumulated = 0.0


def changed(prev: np.ndarray | None, curr: np.ndarray) -> float:
    """% of a small grayscale thumbnail that changed between two frames."""
    if prev is None or prev.shape != curr.shape:
        return 0.0
    return float((cv2.absdiff(prev, curr) > CHANGED_LEVEL).mean() * 100)


def thumbnail(gray: np.ndarray) -> np.ndarray:
    return cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)


class Events:
    """Collects what the tracker sees and writes debounced JSON lines to path.
    on_write(path) is called on every write (the server records the file for cleanup)."""

    def __init__(self, path: Path | str, cooldown: float = COOLDOWN, still_after: float = STILL_AFTER,
                 on_write=None):
        self.path = Path(path)
        self.cooldown, self.still_after, self.on_write = cooldown, still_after, on_write
        self.scene = KeyframeSampler(motion_threshold=SCENE_THRESHOLD, min_cooldown=cooldown)
        self._pending: dict[str, dict] = {}  # key -> event, newest state per object
        self._last_write: float | None = None
        self._thumb: np.ndarray | None = None
        self._moving_since: float | None = None  # start of the current motion episode
        self._calm_since: float | None = None
        self._episode = 0.0  # motion in the current episode
        self._reported = False  # this motion episode already woke the agent with scene_changed
        self._confident = False  # camera motion was recently well estimated
        self._low = 0  # low-confidence frames in a row after that
        self._lock = threading.Lock()
        self.written: list[dict] = []  # lines written, for tests and the tool (last 50)

    # ── inputs ──

    def add(self, event: str, now: float, **data) -> None:
        """Queue an event. object_lost and object_found for one id replace each other;
        a lost-then-found pair inside one window cancels out."""
        with self._lock:
            if event in ("object_lost", "object_found"):
                key = f"obj:{data.get('id')}"
                prev = self._pending.get(key)
                if prev is not None and prev["event"] != event and prev.get("_was") == event:
                    del self._pending[key]  # back to how it was at the last wake-up
                    return
                was = prev.get("_was") if prev else ("object_found" if event == "object_lost" else "object_lost")
                self._pending[key] = {"event": event, **data, "_was": was}
            else:
                self._pending.setdefault(event, {"event": event, **data})

    def observe(self, gray: np.ndarray, cam, now: float) -> None:
        """One tracking frame: gray (small grayscale) and the camera Motion on it."""
        thumb = thumbnail(gray)
        diff = changed(self._thumb, thumb)
        diff = diff if diff >= CHANGED_FLOOR else 0.0
        self._thumb = thumb
        method = getattr(cam, "method", "none")
        conf = getattr(cam, "confidence", 0.0)
        mag = getattr(cam, "magnitude", 0.0)
        cam_px = mag if method != "none" and conf >= 0.5 and mag >= PX_FLOOR else 0.0
        amount = cam_px + diff
        reason = None
        if self.scene.should_sample(now, amount) and not self._reported:
            # One scene_changed per motion episode: someone working in view (an arm
            # moving, hands) would otherwise wake the agent every cooldown. The end of
            # the episode brings "still", the moment worth a look.
            reason = "camera moved" if cam_px > diff else "picture changed"
            self._reported = True
        # The view was blocked: motion was well estimated, then failed for several frames.
        # A plain desk never estimates well, so it never starts this.
        if method != "none":
            if conf >= 0.5:
                self._confident, self._low = True, 0
            elif self._confident:
                self._low += 1
                if self._low >= BLOCKED_FRAMES:
                    self._confident, self._low = False, 0
                    reason = reason or "view blocked or changed"
        if reason:
            self.add("scene_changed", now, reason=reason)
        # still: once per motion episode big enough to matter, after still_after s of calm
        if amount > MOVING:
            if self._moving_since is None:
                self._moving_since, self._episode = now, 0.0
            self._episode += amount
            self._calm_since = None
        else:
            if self._calm_since is None:
                self._calm_since = now
            if now - self._calm_since >= SETTLE:
                self.scene.settle()
            if self._moving_since is not None and now - self._calm_since >= self.still_after:
                if self._episode >= EPISODE_MIN:
                    self.add("still", now, calm_s=self.still_after)
                self._moving_since = self._calm_since = None
                self._reported = False
        self.flush(now)

    # ── output ──

    def flush(self, now: float, force: bool = False) -> dict | None:
        """Write the pending events as one line, unless inside the cooldown."""
        with self._lock:
            if not self._pending:
                return None
            if not force and self._last_write is not None and now - self._last_write < self.cooldown:
                return None
            items = [{k: v for k, v in e.items() if k != "_was"} for e in self._pending.values()]
            self._pending.clear()
            self._last_write = now
        items.sort(key=lambda e: PRIORITY.index(e["event"]) if e["event"] in PRIORITY else len(PRIORITY))
        line = {"t": round(now, 2), "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
                "event": items[0]["event"], "events": items, "summary": summary(items)}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:  # reopened each time: cleanup may delete it
                f.write(json.dumps(line) + "\n")
            if self.on_write is not None:
                self.on_write(self.path)
        except OSError:
            return None
        self.written = (self.written + [line])[-50:]
        return line


def summary(items: list[dict]) -> str:
    parts = []
    for e in items:
        name = f"{e.get('label')} ({e.get('id')})" if e.get("id") else ""
        if e["event"] == "object_lost":
            parts.append(f"{name} lost")
        elif e["event"] == "object_found":
            parts.append(f"{name} found again")
        elif e["event"] == "scene_changed":
            parts.append(f"scene changed: {e.get('reason', '')}".rstrip(": "))
        elif e["event"] == "still":
            parts.append("scene still: good moment to look")
    return "; ".join(parts)


# ── reading (the CLI) ──

def recent(path: Path | str, n: int = 20) -> list[str]:
    """The last n lines of the events file (none if it doesn't exist)."""
    try:
        with open(path, encoding="utf-8") as f:
            return [ln.rstrip("\n") for ln in f.readlines()[-n:] if ln.strip()] if n > 0 else []
    except OSError:
        return []


def follow(path: Path | str, out, stop: threading.Event | None = None, poll: float = 0.2,
           backlog: int = 0) -> None:
    """Print new lines as they are written, flushing each, like tail -F: waits for the file
    if it doesn't exist yet and reopens it if it is deleted, replaced or truncated.
    backlog: print this many existing lines first (0: only new ones, so a Monitor doesn't
    wake on old events). Runs until stop is set (forever by default)."""
    path = Path(path)
    stop = stop or threading.Event()
    for ln in recent(path, backlog):
        out.write(ln + "\n")
    out.flush()
    f, ino, pos = None, None, 0
    first = True
    while not stop.is_set():
        try:
            st = path.stat()
        except OSError:
            st = None
        if st is None or (ino is not None and st.st_ino != ino) or (f is not None and st.st_size < pos):
            if f is not None:
                f.close()
            f, ino, pos = None, None, 0
            first = False
        if f is None and st is not None:
            try:
                f = open(path, encoding="utf-8")
            except OSError:
                f = None
            else:
                ino = st.st_ino
                if first:  # existing file: start at its end
                    f.seek(0, 2)
                pos = f.tell()
            first = False
        if f is not None:
            while True:
                ln = f.readline()
                if not ln or not ln.endswith("\n"):
                    if ln:  # partial line: re-read it next time
                        f.seek(pos)
                    break
                pos = f.tell()
                if ln.strip():
                    out.write(ln)
                    out.flush()
        stop.wait(poll)
    if f is not None:
        f.close()
