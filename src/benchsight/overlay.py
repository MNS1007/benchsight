"""Annotations the agent draws on the live view: arrows, circles, boxes, freehand
strokes and text, plus one caption line. Coordinates are fractions 0-1 of the frame,
so they stay put when the image is resized. The browser viewer draws them from JSON;
render() draws the same shapes onto a frame so the agent can check its own placement.

A shape can carry "anchor": "<tracked object id>". Its geometry is then stored relative
to that object's box at draw time, and resolve() moves it with the object's current box:
boxes and circles follow and scale, an arrow's head follows while its tail keeps its
offset, paths and text translate. A shape whose object is lost is marked lost, so it is
drawn dashed and faded instead of confidently in the wrong place.
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
    if shape.get("anchor"):
        out["anchor"] = str(shape["anchor"])[:60]
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

    def set(self, shapes: list[dict], caption: str | None = None, replace: bool = True,
            anchors: dict[str, list[float]] | None = None) -> dict:
        """anchors: {object id: box [x1, y1, x2, y2]} as the agent saw it, for shapes
        with an anchor. Raises ValueError for an anchor that isn't there."""
        new = [pin(clean(s), anchors or {}) for s in shapes]
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

    def drop(self, anchors) -> int:
        """Remove shapes anchored to these object ids. Returns how many went."""
        anchors = set(anchors)
        with self._lock:
            keep = [s for s in self.shapes if s.get("anchor") not in anchors]
            n = len(self.shapes) - len(keep)
            if n:
                self.shapes = keep
                self.version += 1
                self.changed.notify_all()
            return n

    @property
    def anchored(self) -> bool:
        with self._lock:
            return any("anchor" in s for s in self.shapes)

    def _state(self) -> dict:
        return {"version": self.version, "caption": self.caption, "shapes": list(self.shapes)}

    def wait(self, after: int, timeout: float = 15.0) -> dict:
        """Block until the version moves past `after` (for the viewer's event stream)."""
        with self._lock:
            self.changed.wait_for(lambda: self.version != after, timeout)
            return self._state()


# ── anchoring ──

def _size(b) -> tuple[float, float]:
    return max(1e-6, b[2] - b[0]), max(1e-6, b[3] - b[1])


def _to_rel(p, b) -> list[float]:
    w, h = _size(b)
    return [(p[0] - b[0]) / w, (p[1] - b[1]) / h]


def _from_rel(r, b) -> list[float]:
    w, h = _size(b)
    return [b[0] + r[0] * w, b[1] + r[1] * h]


def _center(b) -> list[float]:
    return [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]


def pin(shape: dict, anchors: dict) -> dict:
    """Store an anchored shape's geometry relative to its anchor's box (`rel`). The
    absolute coordinates stay too: they are where the shape was drawn, and where it is
    shown if the object is later forgotten."""
    oid = shape.get("anchor")
    if oid is None:
        return shape
    if oid not in anchors:
        known = ", ".join(sorted(anchors)) or "none; call track first"
        raise ValueError(f"anchor {oid!r} is not a tracked object (tracked: {known})")
    b = [float(v) for v in anchors[oid]]
    k, rel = shape["type"], {}
    if k == "box":
        rel["box"] = _to_rel(shape["box"][:2], b) + _to_rel(shape["box"][2:], b)
    elif k == "circle":
        rel["at"] = _to_rel(shape["at"], b)
        rel["r"] = shape["r"] / float(np.sqrt(np.prod(_size(b))))
    elif k == "arrow":
        rel["to"] = _to_rel(shape["to"], b)
        rel["tail"] = [shape["from"][0] - shape["to"][0], shape["from"][1] - shape["to"][1]]
    elif k == "path":
        c = _center(b)
        rel["points"] = [[p[0] - c[0], p[1] - c[1]] for p in shape["points"]]
    elif k == "text":
        c = _center(b)
        rel["at"] = [shape["at"][0] - c[0], shape["at"][1] - c[1]]
    return {**shape, "rel": rel}


def place(shape: dict, box) -> dict:
    """An anchored shape at its anchor's current box (fractions; may pass the frame edge)."""
    b, rel, out = [float(v) for v in box], shape["rel"], {**shape}
    k = shape["type"]
    if k == "box":
        out["box"] = _from_rel(rel["box"][:2], b) + _from_rel(rel["box"][2:], b)
    elif k == "circle":
        out["at"] = _from_rel(rel["at"], b)
        out["r"] = rel["r"] * float(np.sqrt(np.prod(_size(b))))
    elif k == "arrow":
        out["to"] = _from_rel(rel["to"], b)
        out["from"] = [out["to"][0] + rel["tail"][0], out["to"][1] + rel["tail"][1]]
    elif k == "path":
        c = _center(b)
        out["points"] = [[c[0] + p[0], c[1] + p[1]] for p in rel["points"]]
    elif k == "text":
        c = _center(b)
        out["at"] = [c[0] + rel["at"][0], c[1] + rel["at"][1]]
    return out


def _round(v, nd):
    if isinstance(v, float):
        return round(v, nd)
    if isinstance(v, list):
        return [_round(x, nd) for x in v]
    return v


def resolve(state: dict, objects) -> dict:
    """The state with every anchored shape moved to its object's current box.
    objects: the tracker's objects() list (or {id: object}). A shape whose object is
    lost, stale or gone gets "lost": True (a gone one stays where it was drawn).
    Unanchored shapes come back unchanged. Coordinates are rounded to 4 decimals."""
    if isinstance(objects, dict):
        objects = objects.values()
    by_id = {o["id"]: o for o in objects or ()}
    shapes = []
    for s in state.get("shapes", []):
        if "anchor" not in s:
            shapes.append(s)
            continue
        o = by_id.get(s["anchor"])
        out = place(s, o["box"]) if o is not None else {**s}
        out.pop("rel", None)
        out = {k: _round(v, 4) for k, v in out.items()}
        out["lost"] = o is None or o.get("status") != "tracking"
        shapes.append(out)
    return {**state, "shapes": shapes}


# ── drawing ──

def _dashed(img, pts, closed, bgr, t, dash: int = 14, gap: int = 10) -> None:
    pts = [np.array(p, float) for p in pts] + ([np.array(pts[0], float)] if closed else [])
    on, left = True, float(dash)
    for a, b in zip(pts, pts[1:]):
        seg = float(np.hypot(*(b - a)))
        pos = 0.0
        while pos < seg:
            step = min(left, seg - pos)
            if on:
                p, q = a + (b - a) * (pos / seg), a + (b - a) * ((pos + step) / seg)
                cv2.line(img, tuple(int(v) for v in p), tuple(int(v) for v in q), bgr, t, cv2.LINE_AA)
            pos += step
            left -= step
            if left <= 0:
                on, left = not on, float(dash if not on else gap)


def render(frame: np.ndarray, state: dict) -> np.ndarray:
    """Draw the annotations onto a copy of the frame. Pass a resolve()d state to draw
    anchored shapes where their objects are now; lost ones are drawn dashed and faded."""
    out = frame.copy()
    h, w = out.shape[:2]
    t = max(2, min(h, w) // 200)
    px = lambda p: (int(p[0] * w), int(p[1] * h))  # noqa: E731

    def label(at, text, bgr):
        if text:
            cv2.putText(out, text, (at[0] + 8, at[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6 + t * 0.1, (0, 0, 0), t + 3)
            cv2.putText(out, text, (at[0] + 8, at[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6 + t * 0.1, bgr, t)

    shapes = state.get("shapes", [])
    lost = [s for s in shapes if s.get("lost")]
    if lost:  # faded: drawn on a copy that is blended in at half strength
        ghost = out.copy()
        for s in lost:
            _draw_lost(ghost, s, px, min(h, w), t, label)
        out[:] = cv2.addWeighted(ghost, 0.5, out, 0.5, 0)
    for s in shapes:
        if s.get("lost"):
            continue
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


def _draw_lost(img, s, px, side, t, label) -> None:
    """A shape whose object is lost: dashed, with a "lost" tag."""
    bgr = COLORS[s["color"]][0]
    tag = (s.get("label") or s.get("text") or "").strip()
    tag = f"{tag} (lost)" if tag else "lost"
    if s["type"] == "arrow":
        a, b = px(s["from"]), px(s["to"])
        _dashed(img, [a, b], False, bgr, t + 1)
        near = (int(a[0] + 0.8 * (b[0] - a[0])), int(a[1] + 0.8 * (b[1] - a[1])))
        cv2.arrowedLine(img, near, b, bgr, t + 1, tipLength=0.4)  # just the head
        label(a, tag, bgr)
    elif s["type"] == "circle":
        c, r = px(s["at"]), max(1, int(s["r"] * side))
        _dashed(img, cv2.ellipse2Poly(c, (r, r), 0, 0, 360, 10).tolist(), True, bgr, t + 1)
        label((c[0] + r, c[1]), tag, bgr)
    elif s["type"] == "box":
        (x1, y1), (x2, y2) = px(s["box"][:2]), px(s["box"][2:])
        _dashed(img, [(x1, y1), (x2, y1), (x2, y2), (x1, y2)], True, bgr, t + 1)
        label((x1, y1), tag, bgr)
    elif s["type"] == "path":
        _dashed(img, [px(p) for p in s["points"]], False, bgr, t + 1)
        label(px(s["points"][-1]), tag, bgr)
    elif s["type"] == "text":
        label(px(s["at"]), tag, bgr)
