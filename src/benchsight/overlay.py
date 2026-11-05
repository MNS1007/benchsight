"""Annotations the agent draws on the live view: arrows, circles, boxes, freehand
strokes and text, plus one caption line. Coordinates are fractions 0-1 of the frame,
so they stay put when the image is resized. The browser viewer draws them from JSON;
render() draws the same shapes onto a frame so the agent can check its own placement.
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np

COLORS = {  # name -> (BGR for OpenCV, CSS for the viewer)
    "red": ((40, 40, 230), "#e62828"),
    "green": ((60, 190, 60), "#3cbe3c"),
    "yellow": ((0, 215, 255), "#ffd700"),
    "blue": ((230, 140, 40), "#288ce6"),
    "white": ((255, 255, 255), "#ffffff"),
}
KINDS = ("arrow", "circle", "box", "path", "text")


def _pt(v, what):
    if not (isinstance(v, (list, tuple)) and len(v) == 2):
        raise ValueError(f"{what} must be [x, y] with x and y between 0 and 1")
    x, y = float(v[0]), float(v[1])
    if not (0 <= x <= 1 and 0 <= y <= 1):
        raise ValueError(f"{what} {v} is outside the frame; use fractions 0-1")
    return [x, y]


def clean(shape: dict) -> dict:
    """Check one shape and return it in a normal form. Raises ValueError with a plain reason."""
    kind = shape.get("type")
    if kind not in KINDS:
        raise ValueError(f"type must be one of {', '.join(KINDS)}")
    color = shape.get("color", "red" if kind != "text" else "yellow")
    if color not in COLORS:
        raise ValueError(f"color must be one of {', '.join(COLORS)}")
    out = {"type": kind, "color": color, "label": str(shape.get("label", ""))[:80]}
    if kind == "arrow":
        out["from"], out["to"] = _pt(shape.get("from"), "from"), _pt(shape.get("to"), "to")
    elif kind == "circle":
        out["at"] = _pt(shape.get("at"), "at")
        out["r"] = min(0.5, max(0.01, float(shape.get("r", 0.05))))
    elif kind == "box":
        b = shape.get("box")
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            raise ValueError("box must be [x1, y1, x2, y2]")
        (x1, y1), (x2, y2) = _pt(b[:2], "box corner"), _pt(b[2:], "box corner")
        out["box"] = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    elif kind == "path":
        pts = shape.get("points") or []
        if len(pts) < 2:
            raise ValueError("path needs at least 2 points")
        out["points"] = [_pt(p, "path point") for p in pts[:200]]
    elif kind == "text":
        out["at"] = _pt(shape.get("at"), "at")
        out["text"] = str(shape.get("text", ""))[:120]
        if not out["text"]:
            raise ValueError("text needs text")
    return out


class Overlay:
    """The current annotations, shared between the MCP tools and the viewer."""

    def __init__(self):
        self._lock = threading.Lock()
        self.shapes: list[dict] = []
        self.caption = ""
        self.version = 0
        self.changed = threading.Condition(self._lock)

    def set(self, shapes: list[dict], caption: str | None = None, replace: bool = True) -> dict:
        new = [clean(s) for s in shapes]
        with self._lock:
            self.shapes = new if replace else self.shapes + new
            if caption is not None:
                self.caption = str(caption)[:200]
            self.version += 1
            self.changed.notify_all()
            return self._state()

    def clear(self) -> dict:
        with self._lock:
            self.shapes, self.caption = [], ""
            self.version += 1
            self.changed.notify_all()
            return self._state()

    def state(self) -> dict:
        with self._lock:
            return self._state()

    def _state(self) -> dict:
        return {"version": self.version, "caption": self.caption, "shapes": list(self.shapes)}

    def wait(self, after: int, timeout: float = 15.0) -> dict:
        """Block until the version moves past `after` (for the viewer's event stream)."""
        with self._lock:
            self.changed.wait_for(lambda: self.version != after, timeout)
            return self._state()


def render(frame: np.ndarray, state: dict) -> np.ndarray:
    """Draw the annotations onto a copy of the frame."""
    out = frame.copy()
    h, w = out.shape[:2]
    t = max(2, min(h, w) // 200)
    px = lambda p: (int(p[0] * w), int(p[1] * h))  # noqa: E731

    def label(at, text, bgr):
        if text:
            cv2.putText(out, text, (at[0] + 8, at[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6 + t * 0.1, (0, 0, 0), t + 3)
            cv2.putText(out, text, (at[0] + 8, at[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6 + t * 0.1, bgr, t)

    for s in state.get("shapes", []):
        bgr = COLORS[s["color"]][0]
        if s["type"] == "arrow":
            a, b = px(s["from"]), px(s["to"])
            cv2.arrowedLine(out, a, b, bgr, t + 1, tipLength=0.08)
            label(a, s["label"], bgr)
        elif s["type"] == "circle":
            c = px(s["at"])
            cv2.circle(out, c, int(s["r"] * min(h, w)), bgr, t + 1)
            label((c[0] + int(s["r"] * min(h, w)), c[1]), s["label"], bgr)
        elif s["type"] == "box":
            a, b = px(s["box"][:2]), px(s["box"][2:])
            cv2.rectangle(out, a, b, bgr, t + 1)
            label(a, s["label"], bgr)
        elif s["type"] == "path":
            cv2.polylines(out, [np.array([px(p) for p in s["points"]], dtype=np.int32)], False, bgr, t + 1)
            label(px(s["points"][-1]), s["label"], bgr)
        elif s["type"] == "text":
            label(px(s["at"]), s["text"], bgr)
    if state.get("caption"):
        bar = max(30, h // 14)
        cv2.rectangle(out, (0, h - bar), (w, h), (0, 0, 0), -1)
        cv2.putText(out, state["caption"], (10, h - bar // 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6 + t * 0.1, (255, 255, 255), t)
    return out
