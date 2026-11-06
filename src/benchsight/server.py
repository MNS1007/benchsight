"""MCP server: bench camera tools for Claude Code (or any MCP client).

Photos are taken only when a tool asks. The live viewer (open_viewer) is a small local
window with the camera feed, where the agent draws arrows, circles, boxes, freehand
strokes and text to guide the builder. It runs only while open and serves 127.0.0.1 only.
While it is open, objects the agent boxes with track() are followed frame by frame on the
CPU, and shapes anchored to them move with them. While it is open, start_recording keeps
small frames of the bench so the agent can review afterwards what the builder did.
Images are saved in ./.benchsight (or BENCHSIGHT_DIR) and deleted when the session ends
(BENCHSIGHT_KEEP=1 keeps them), or after 24 hours at the latest.
"""
from __future__ import annotations

import os
import time
import webbrowser

from mcp.server.mcpserver import Image, MCPServer

from . import camera, cleanup
from . import recording as rec_mod  # tools have a `recording` argument
from .events import Events
from .feed import LiveFeed
from .overlay import Overlay, render, resolve
from .track import BUDGET, Tracker, check_box
from .viewer import Viewer

mcp = MCPServer("benchsight")
_last = None  # the last full frame taken, for point() and track()
overlay = Overlay()
_live: dict = {"feed": None, "viewer": None, "tracker": None, "tracker_error": "", "recorder": None}
_look_boxes: dict[str, list[float]] = {}  # tracked boxes as they were in the last look


def _make_tracker(feed) -> Tracker:
    """The object tracker for the live feed. BENCHSIGHT_TRACKER picks the backend:
    auto (VitTrack, downloaded once, else MIL), vit or mil. Wake-up events go to
    <store>/events.jsonl, for `benchsight events --follow`. BENCHSIGHT_CPU is the share
    of one core tracking aims to stay under (default 0.15; 0 means no limit)."""
    cpu = float(os.environ.get("BENCHSIGHT_CPU", BUDGET))
    return Tracker(feed, backend=os.environ.get("BENCHSIGHT_TRACKER", "auto"), budget=cpu or None,
                   events=Events(camera.STORE / "events.jsonl", on_write=lambda p: cleanup.record(p, camera.STORE)))


def _tracker() -> Tracker:
    tracker = _live["tracker"]
    if tracker is None and _live["viewer"] is not None:
        raise ValueError(f"tracking is off: {_live['tracker_error']}")
    if tracker is None:
        raise ValueError("open the viewer first with open_viewer; tracking runs only while it is open")
    return tracker


def _drawn() -> dict:
    """The annotations with anchored shapes where their objects are now."""
    tracker = _live["tracker"]
    return resolve(overlay.state(), tracker.objects()) if tracker is not None else overlay.state()


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


def _rec_status() -> dict | None:
    """The recording's status for the viewer's REC badge, or None."""
    rec = rec_mod.current()
    return rec.status() if rec is not None else None


def _stop_recording() -> dict | None:
    """Stop the recording if there is one (it may have stopped by itself). Its summary, or None."""
    rec, _live["recorder"] = _live["recorder"], None
    return rec.stop() if rec is not None else None


@mcp.tool()
def cameras() -> list[int]:
    """List camera indices that work on this machine."""
    return camera.list_cameras()


@mcp.tool()
def look(cam: int = 0, max_side: int = 1280, x1: float = 0, y1: float = 0, x2: float = 1, y2: float = 1,
         with_annotations: bool = False) -> Image:
    """Take one photo of the bench now. Ask the builder to point the camera first.
    Use x1..y2 (fractions 0-1) to zoom on a region, like a connector or a servo label.
    with_annotations=True draws your current annotations on it, to check their placement;
    shapes anchored to tracked objects are drawn where the objects are now, and dashed
    with a "lost" tag if their object is lost."""
    global _last
    frame = _frame(cam)
    _last = frame
    if _live["tracker"] is not None:
        _look_boxes.clear()
        _look_boxes.update({o["id"]: o["box"] for o in _live["tracker"].objects()})
    if with_annotations:
        frame = render(frame, _drawn())
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
    Returns the local URL, and with tracking on a `watch` command: run it under a
    background monitor to be woken by events (object lost or found, scene changed, still).
    To see what the builder does over a while, record and review: start_recording (roi
    around the workspace), let them work, stop_recording, review(), then review(start_s,
    end_s) to zoom. The camera stays on until close_viewer."""
    if _live["viewer"] is not None:
        return {"ok": True, "url": _live["viewer"].url, "note": "already open"}
    feed = LiveFeed(cam)
    feed.start()
    note = "Viewer open. Ask the builder to keep it beside the bench."
    try:
        tracker = _make_tracker(feed)
        tracker.start()
        note += f" Tracking ready ({tracker.backend.name})."
    except Exception as err:  # the viewer still works without tracking
        tracker, note = None, note + f" Tracking is off: {err}"
        _live["tracker_error"] = str(err)
    port = int(os.environ.get("BENCHSIGHT_PORT", "0"))
    viewer = Viewer(feed, overlay, port=port, tracker=tracker, recording=_rec_status)
    url = viewer.start()
    _live.update(feed=feed, viewer=viewer, tracker=tracker)
    if open_browser:
        webbrowser.open(url)
    out = {"ok": True, "url": url, "note": note}
    if tracker is not None:
        store = camera.mark_active()
        out["events_file"] = str(store / "events.jsonl")
        out["watch"] = f"benchsight events --follow --dir '{store}'"
        out["note"] += " To wake on changes, run the watch command under a background monitor."
    return out


@mcp.tool()
def track(objects: list[dict], replace: bool = True) -> dict:
    """Follow objects on the live viewer, so annotations can stick to them as the part or
    the camera moves. First take a look (without zoom), decide what each object is, and
    give its box as fractions 0-1 of that full photo:
      [{"id": "plate", "label": "slider plate", "box": [x1, y1, x2, y2]}, ...]
    id is optional (made from the label). A box whose label matches a tracked object
    it overlaps (e.g. "slider plate" over "white plate") updates that object instead of
    adding a duplicate; objects listed in one call always stay separate. Up to 8 objects. replace=True (default) stops tracking objects
    not in this list and removes shapes anchored to them; replace=False adds.
    Returns the ids: put them in annotate shapes as "anchor". Check positions cheaply
    with objects()."""
    tracker = _tracker()
    if _last is None:
        raise ValueError("take a look first: boxes are read from the latest look")
    before = {o["id"] for o in tracker.objects()}
    got = tracker.add(objects, frame=_last, replace=replace)
    for o, given in zip(got, objects):  # the agent's boxes are where they were in the look
        _look_boxes[o["id"]] = check_box(given.get("box"))
    now = {o["id"] for o in tracker.objects()}
    gone = before - now
    for oid in gone:
        _look_boxes.pop(oid, None)
    removed = overlay.drop(gone)
    return {"ok": True, "tracking": got, "backend": tracker.backend.name, "untracked": sorted(gone),
            "shapes_removed": removed, "note": 'Anchor shapes with {"anchor": "<id>"} in annotate.'}


@mcp.tool()
def objects() -> list[dict]:
    """Where the tracked objects are now, with no image (cheap: use it instead of look
    to check positions). Each: id, label, box [x1, y1, x2, y2] (fractions 0-1), status
    (tracking; lost = not seen, kept at its last box; stale = lost for a while, soon
    forgotten) and last_seen_s_ago (seconds since it was last seen)."""
    now = time.time()
    return [{"id": o["id"], "label": o["label"], "box": o["box"], "status": o["status"],
             "last_seen_s_ago": round(max(0.0, now - o["last_seen"]), 1)} for o in _tracker().objects()]


@mcp.tool()
def untrack(ids: list[str] | None = None) -> dict:
    """Stop tracking these object ids (all when omitted) and remove the shapes anchored
    to them from the viewer."""
    gone = _tracker().untrack(ids)
    for oid in gone:
        _look_boxes.pop(oid, None)
    return {"ok": True, "untracked": gone, "shapes_removed": overlay.drop(gone)}


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
    Add "anchor": "<id>" (an id from track) to make a shape follow that object: draw it
    where it is in your latest look and it keeps that place on the object as it moves.
    Boxes and circles follow and scale with it, an arrow's head follows it while its
    tail keeps its offset, paths and text move with it. If the object is lost the shape
    is shown dashed and faded with a "lost" tag. Shapes without anchor stay put.
    caption: one short instruction shown under the video. replace=False adds to what is
    there. Then check placement with look(with_annotations=True)."""
    if _live["viewer"] is None:
        raise ValueError("open the viewer first with open_viewer")
    anchors = {}
    if _live["tracker"] is not None:
        anchors = {o["id"]: _look_boxes.get(o["id"], o["box"]) for o in _live["tracker"].objects()}
    state = overlay.set(shapes, caption=caption, replace=replace, anchors=anchors)
    return {"ok": True, "shapes": len(state["shapes"]), "caption": state["caption"]}


@mcp.tool()
def clear_annotations() -> dict:
    """Remove all annotations and the caption from the live viewer."""
    overlay.clear()
    return {"ok": True}


@mcp.tool()
def start_recording(roi: list[float] | None = None, fps: float = 8, max_seconds: float = 300) -> dict:
    """Record the bench while the builder works, to review it afterwards: you cannot
    watch live video, but you can look back once they are done. Use it when the builder
    says "watch this" or "start watching". Needs the viewer open (open_viewer); the
    builder sees a red REC badge there. roi: [x1, y1, x2, y2] (fractions 0-1 of a recent
    look) around the workspace, so the activity hint ignores the rest of the frame (all
    frames are still saved whole). fps up to 15; it stops by itself after max_seconds
    (at most 900). Then: stop_recording, review() for an overview, review(start_s, end_s)
    to zoom on a moment."""
    feed = _live["feed"]
    if _live["viewer"] is None or feed is None or not feed.running:
        raise ValueError("open the viewer first with open_viewer: the camera runs only while it is open")
    if _live["recorder"] is not None and _live["recorder"].running:
        raise ValueError(f"already recording into {_live['recorder'].folder.name}; call stop_recording first")
    try:
        rec = rec_mod.Recorder(feed, camera.STORE.resolve(), fps=fps, max_seconds=max_seconds, roi=roi).start()
    except RuntimeError as err:
        raise ValueError(str(err)) from err
    _live["recorder"] = rec
    return {"ok": True, "folder": str(rec.folder), "fps": rec.fps, "max_seconds": rec.max_seconds,
            "roi": list(rec.roi) if rec.roi else None,
            "note": f"Recording (stops by itself after {rec.max_seconds:g} s). Let the builder work, "
                    "then call stop_recording."}


@mcp.tool()
def stop_recording() -> dict:
    """Stop the recording and say what was recorded: duration_s, frames, folder, and
    activity: stretches [start_s, end_s] where the work area changed. Activity is a rough
    hint only (it misses slow or small motion): review the whole recording evenly first."""
    if _live["recorder"] is None:
        raise ValueError("nothing is recording: start_recording first")
    out = _stop_recording()
    out["note"] = ("Call review() for an even overview of the whole recording, then "
                   "review(start_s, end_s) to zoom on the moments that matter, and describe what happened. "
                   "The activity stretches are only a hint of where things moved.")
    return out


@mcp.tool()
def review(start_s: float | None = None, end_s: float | None = None, frames: int = 12,
           max_side: int = 1600, recording: str | None = None) -> Image:
    """One sheet of frames from the last recording (or the named rec-... folder), spread
    evenly over [start_s, end_s] (default: all of it), each labeled with its number and
    time ("3  4.2s"). Start with review() for the whole run, then zoom: review(4.0, 7.0)
    to see what happened between 4 s and 7 s. You pick the moments; frames are never
    chosen by activity. frames: 2-24. Works while still recording, on what is saved so far."""
    folder = rec_mod.find(camera.STORE, recording)
    shot, _ = rec_mod.review(folder, start_s, end_s, max(2, min(int(frames), 24)), max(320, min(int(max_side), 2400)))
    return Image(data=shot.jpeg, format="jpeg")


@mcp.tool()
def review_video(path: str, start_s: float | None = None, end_s: float | None = None, frames: int = 12,
                 max_side: int = 1600) -> Image:
    """Like review, but from any local video file (an absolute path), e.g. a LeRobot
    dataset episode mp4: frames spread evenly over [start_s, end_s] (default: all of it),
    each labeled with its number and time in the video. Zoom the same way. frames: 2-24.
    Needs no viewer; the video is read in place, never copied."""
    shot = rec_mod.review_video(path, start_s, end_s, max(2, min(int(frames), 24)), max(320, min(int(max_side), 2400)))
    return Image(data=shot.jpeg, format="jpeg")


@mcp.tool()
def close_viewer() -> dict:
    """Close the live viewer, stop tracking and any recording, and turn the camera off.
    The recording stays on disk for review until the session ends or cleanup."""
    viewer, feed, tracker = _live["viewer"], _live["feed"], _live["tracker"]
    summary = _stop_recording()  # first: it reads the feed
    if tracker is not None:  # before the feed: it reads it
        tracker.stop()
    if viewer is not None:
        viewer.stop()
    if feed is not None:
        feed.stop()
    _live.update(feed=None, viewer=None, tracker=None, tracker_error="", recorder=None)
    _look_boxes.clear()
    overlay.clear()
    out = {"ok": True, "note": "Viewer closed, camera off."}
    if summary is not None:
        out["recording"] = summary
        out["note"] += " The recording was stopped; review() still works on it."
    return out


@mcp.tool(name="cleanup")
def cleanup_files() -> dict:
    """Delete the photos, annotated images, contact sheets, recordings and events file
    this session has saved so far (they are deleted anyway when the session ends). A
    recording in progress is stopped first. Returns how many files and bytes were freed."""
    _stop_recording()
    n, freed = cleanup.session_end()
    return {"ok": True, "files": n, "bytes": freed, "freed": cleanup.human(freed)}


def main() -> None:
    cleanup.install()  # delete this session's files on exit and on SIGTERM/SIGINT/SIGHUP
    try:
        mcp.run()  # returns when the client closes stdio
    finally:
        cleanup.session_end()


if __name__ == "__main__":
    main()
