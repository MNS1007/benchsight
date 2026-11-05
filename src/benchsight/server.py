"""MCP server: on-demand bench camera tools for Claude Code (or any MCP client).

Every tool call takes fresh frames; nothing runs in the background. Images go back to
the agent and are kept in ./.benchsight for 24 hours, so the builder can see what the
agent saw.
"""
from __future__ import annotations

import time

from mcp.server.mcpserver import Image, MCPServer

from . import camera

mcp = MCPServer("benchsight")
_last = None  # the last full frame taken, for point()


@mcp.tool()
def cameras() -> list[int]:
    """List camera indices that work on this machine."""
    return camera.list_cameras()


@mcp.tool()
def look(cam: int = 0, max_side: int = 1280, x1: float = 0, y1: float = 0, x2: float = 1, y2: float = 1) -> Image:
    """Take one photo of the bench now. Ask the builder to point the camera first.
    Use x1..y2 (fractions 0-1) to zoom on a region, like a connector or a servo label."""
    global _last
    frame = camera.grab(cam)
    _last = frame
    img = camera.fit(camera.crop(frame, (x1, y1, x2, y2)), max_side)
    return Image(data=camera.save(img, "look").jpeg, format="jpeg")


@mcp.tool()
def watch(cam: int = 0, seconds: float = 6, frames: int = 6, max_side: int = 1600) -> Image:
    """Take several photos over a few seconds and return one numbered sheet, for
    something that moves: a servo sweep, a hand doing a step, a cable being seated."""
    frames = max(2, min(int(frames), 9))
    shots = []
    for i in range(frames):
        shots.append(camera.grab(cam, warmup=2 if i else 5))
        if i < frames - 1:
            time.sleep(max(0.0, seconds) / (frames - 1))
    sheet = camera.fit(camera.contact_sheet([camera.fit(s, 640) for s in shots]), max_side)
    return Image(data=camera.save(sheet, "watch").jpeg, format="jpeg")


@mcp.tool()
def point(x: float, y: float, label: str = "", max_side: int = 1280) -> Image:
    """Circle a spot on the last full photo (fractions 0-1) with a short label, to show
    the builder exactly which part, pin or cable you mean."""
    if _last is None:
        raise ValueError("take a photo with look first")
    img = camera.fit(camera.mark(_last, x, y, label), max_side)
    return Image(data=camera.save(img, "point").jpeg, format="jpeg")


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
