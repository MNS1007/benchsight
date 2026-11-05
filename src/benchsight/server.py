"""MCP server: bench camera tools for Claude Code (or any MCP client).

Photos are taken only when a tool asks. The live viewer (open_viewer) is a small local
window with the camera feed, where the agent draws arrows, circles, boxes, freehand
strokes and text to guide the builder. It runs only while open and serves 127.0.0.1 only.
Images are kept in ./.benchsight (or BENCHSIGHT_DIR) for 24 hours.
"""
from __future__ import annotations

import os
import time
import webbrowser

from mcp.server.mcpserver import Image, MCPServer

from . import camera
from .feed import LiveFeed
from .overlay import Overlay, render
from .viewer import Viewer

mcp = MCPServer("benchsight")
_last = None  # the last full frame taken, for point()
overlay = Overlay()
_live: dict = {"feed": None, "viewer": None}


def _frame(cam: int, warmup: int = 5):
    """The live feed's latest frame when the viewer is open on this camera, else a fresh grab."""
    feed = _live["feed"]
    if feed is not None and feed.running and feed.cam == cam:
        f = feed.latest()
        if f is not None:
            return f
    return camera.grab(cam, warmup=warmup)


def _jpeg(img, name):
    return Image(data=camera.save(img, name).jpeg, format="jpeg")


@mcp.tool()
def cameras() -> list[int]:
    """List camera indices that work on this machine."""
    return camera.list_cameras()


@mcp.tool()
def look(cam: int = 0, max_side: int = 1280, x1: float = 0, y1: float = 0, x2: float = 1, y2: float = 1,
         with_annotations: bool = False) -> Image:
    """Take one photo of the bench now. Ask the builder to point the camera first.
    Use x1..y2 (fractions 0-1) to zoom on a region, like a connector or a servo label.
    with_annotations=True draws your current annotations on it, to check their placement."""
    global _last
    frame = _frame(cam)
    _last = frame
    if with_annotations:
        frame = render(frame, overlay.state())
    return _jpeg(camera.fit(camera.crop(frame, (x1, y1, x2, y2)), max_side), "look")


@mcp.tool()
def watch(cam: int = 0, seconds: float = 6, frames: int = 6, max_side: int = 1600) -> Image:
    """Take several photos over a few seconds and return one numbered sheet, for
    something that moves: a servo sweep, a hand doing a step, a cable being seated."""
    frames = max(2, min(int(frames), 9))
    shots = []
    for i in range(frames):
        shots.append(_frame(cam, warmup=2 if i else 5))
        if i < frames - 1:
            time.sleep(max(0.0, seconds) / (frames - 1))
    sheet = camera.fit(camera.contact_sheet([camera.fit(s, 640) for s in shots]), max_side)
    return _jpeg(sheet, "watch")


@mcp.tool()
def point(x: float, y: float, label: str = "", max_side: int = 1280) -> Image:
    """Circle a spot on the last photo (fractions 0-1) with a short label. If the live
    viewer is open, the circle also appears there."""
    if _last is None:
        raise ValueError("take a photo with look first")
    if _live["viewer"] is not None:
        overlay.set([{"type": "circle", "at": [x, y], "r": 0.04, "label": label}], replace=False)
    return _jpeg(camera.fit(camera.mark(_last, x, y, label), max_side), "point")


@mcp.tool()
def open_viewer(cam: int = 0, open_browser: bool = True) -> dict:
    """Open the live viewer: a small local window with the camera feed, where your
    annotations appear on top. Use it when guiding the builder through a physical step.
    Returns the local URL. The camera stays on until close_viewer."""
    if _live["viewer"] is not None:
        return {"ok": True, "url": _live["viewer"].url, "note": "already open"}
    feed = LiveFeed(cam)
    feed.start()
    port = int(os.environ.get("BENCHSIGHT_PORT", "0"))
    viewer = Viewer(feed, overlay, port=port)
    url = viewer.start()
    _live.update(feed=feed, viewer=viewer)
    if open_browser:
        webbrowser.open(url)
    return {"ok": True, "url": url, "note": "Viewer open. Ask the builder to keep it beside the bench."}


@mcp.tool()
def annotate(shapes: list[dict], caption: str = "", replace: bool = True) -> dict:
    """Draw on the live viewer to guide the builder. Coordinates are fractions 0-1 of
    the frame, read from a recent look. Shapes (each a dict, color optional: red =
    wrong, green = right, yellow, blue, white):
      {"type": "arrow", "from": [x, y], "to": [x, y], "label": "move here"}
      {"type": "circle", "at": [x, y], "r": 0.05, "label": "this screw"}
      {"type": "box", "box": [x1, y1, x2, y2], "label": "wrong part"}
      {"type": "path", "points": [[x, y], ...], "label": "route the cable like this"}
      {"type": "text", "at": [x, y], "text": "flip it over"}
    caption: one short instruction shown under the video. replace=False adds to what is
    there. Then check placement with look(with_annotations=True)."""
    if _live["viewer"] is None:
        raise ValueError("open the viewer first with open_viewer")
    state = overlay.set(shapes, caption=caption, replace=replace)
    return {"ok": True, "shapes": len(state["shapes"]), "caption": state["caption"]}


@mcp.tool()
def clear_annotations() -> dict:
    """Remove all annotations and the caption from the live viewer."""
    overlay.clear()
    return {"ok": True}


@mcp.tool()
def close_viewer() -> dict:
    """Close the live viewer and turn the camera off."""
    viewer, feed = _live["viewer"], _live["feed"]
    if viewer is not None:
        viewer.stop()
    if feed is not None:
        feed.stop()
    _live.update(feed=None, viewer=None)
    overlay.clear()
    return {"ok": True, "note": "Viewer closed, camera off."}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
