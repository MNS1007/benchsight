"""The live viewer and annotations, with a synthetic feed instead of a camera."""
import json
import threading
import time
import urllib.request

import numpy as np
import pytest

from benchsight.feed import LiveFeed
from benchsight.overlay import Overlay, clean, render
from benchsight.viewer import Viewer


def frame():
    return np.full((240, 320, 3), 40, dtype=np.uint8)


def test_shapes_are_checked_with_plain_reasons():
    assert clean({"type": "arrow", "from": [0.1, 0.1], "to": [0.5, 0.5]})["color"] == "red"
    assert clean({"type": "box", "box": [0.6, 0.6, 0.2, 0.2]})["box"] == [0.2, 0.2, 0.6, 0.6]
    for bad, why in [({"type": "blob"}, "type"), ({"type": "circle", "at": [1.5, 0.2]}, "outside"),
                     ({"type": "path", "points": [[0.1, 0.1]]}, "2 points"), ({"type": "text", "at": [0.1, 0.1]}, "text")]:
        with pytest.raises(ValueError, match=why):
            clean(bad)


def test_render_draws_every_kind_and_the_caption():
    ov = Overlay()
    state = ov.set([
        {"type": "arrow", "from": [0.1, 0.1], "to": [0.4, 0.4], "label": "here"},
        {"type": "circle", "at": [0.5, 0.5], "r": 0.1, "color": "green"},
        {"type": "box", "box": [0.6, 0.1, 0.9, 0.3]},
        {"type": "path", "points": [[0.1, 0.8], [0.3, 0.7], [0.5, 0.8]], "color": "yellow"},
        {"type": "text", "at": [0.6, 0.6], "text": "flip it"},
    ], caption="Wrong screw: use the short one")
    img = frame()
    out = render(img, state)
    assert (out != img).any() and (img == 40).all()
    assert (out[-5:, :] < 40).any(), "caption bar drawn at the bottom"


def test_replace_and_add():
    ov = Overlay()
    ov.set([{"type": "text", "at": [0.1, 0.1], "text": "a"}])
    ov.set([{"type": "text", "at": [0.2, 0.2], "text": "b"}], replace=False)
    assert len(ov.state()["shapes"]) == 2
    ov.set([{"type": "text", "at": [0.3, 0.3], "text": "c"}])
    assert [s["text"] for s in ov.state()["shapes"]] == ["c"]
    assert ov.clear()["shapes"] == []


def test_viewer_serves_page_overlay_stream_and_events():
    feed = LiveFeed(source=frame, fps=20)
    feed.start()
    ov = Overlay()
    v = Viewer(feed, ov)
    url = v.start()
    try:
        page = urllib.request.urlopen(url, timeout=5).read().decode()
        assert "/stream" in page and "EventSource" in page and "__COLORS__" not in page
        assert json.loads(urllib.request.urlopen(url + "overlay", timeout=5).read())["shapes"] == []

        with urllib.request.urlopen(url + "stream", timeout=5) as r:
            assert r.headers["Content-Type"].startswith("multipart/x-mixed-replace")
            chunk = r.read(4000)
        assert b"--frame" in chunk and b"\xff\xd8" in chunk

        got = []

        def listen():
            with urllib.request.urlopen(url + "events", timeout=10) as r:
                for line in r:
                    if line.startswith(b"data: "):
                        got.append(json.loads(line[6:]))
                        if any(s["caption"] == "seat the cable" for s in got):
                            return

        t = threading.Thread(target=listen, daemon=True)
        t.start()
        time.sleep(0.3)
        ov.set([{"type": "arrow", "from": [0.2, 0.2], "to": [0.6, 0.5], "label": "push in"}], caption="seat the cable")
        t.join(timeout=5)
        assert got and got[-1]["caption"] == "seat the cable" and got[-1]["shapes"][0]["type"] == "arrow"
    finally:
        v.stop()
        feed.stop()


def test_live_feed_gives_the_latest_frame_and_stops():
    n = {"i": 0}

    def src():
        n["i"] += 1
        return np.full((10, 10, 3), n["i"] % 255, dtype=np.uint8)

    feed = LiveFeed(source=src, fps=50)
    feed.start()
    time.sleep(0.2)
    assert feed.running and feed.latest() is not None
    feed.stop()
    assert not feed.running and feed.latest() is None


class StubTracker:
    """Stands in for track.Tracker: the test moves the boxes by hand."""

    def __init__(self, box):
        self.objs = [{"id": "plate", "label": "plate", "box": box, "status": "tracking"}]

    def objects(self):
        return [dict(o) for o in self.objs]


def sse(url, got, stop):
    """Collect (event, data) pairs from /events until stop is set."""
    with urllib.request.urlopen(url + "events", timeout=10) as r:
        event = "message"
        for line in r:
            if line.startswith(b"event: "):
                event = line[7:].strip().decode()
            elif line.startswith(b"data: "):
                got.append((event, json.loads(line[6:])))
                event = "message"
            if stop.is_set():
                return


def test_viewer_pushes_anchored_positions_only_when_they_change():
    feed = LiveFeed(source=frame, fps=20)
    feed.start()
    ov, tr = Overlay(), StubTracker([0.2, 0.2, 0.4, 0.4])
    ov.set([{"type": "box", "box": [0.2, 0.2, 0.4, 0.4], "anchor": "plate"},
            {"type": "circle", "at": [0.8, 0.8], "r": 0.05}], anchors={"plate": [0.2, 0.2, 0.4, 0.4]})
    v = Viewer(feed, ov, tracker=tr)
    url = v.start()
    got, stop = [], threading.Event()
    t = threading.Thread(target=sse, args=(url, got, stop), daemon=True)
    try:
        assert json.loads(urllib.request.urlopen(url + "overlay", timeout=5).read())["shapes"][0]["lost"] is False
        t.start()
        time.sleep(0.4)
        assert [e for e, _ in got] == ["message"], "one full state, then nothing while nothing moves"
        tr.objs[0]["box"] = [0.5, 0.5, 0.7, 0.7]
        time.sleep(0.4)
        tr.objs[0]["status"] = "lost"
        time.sleep(0.4)
        tr.objs[0]["box"] = [0.50001, 0.5, 0.7, 0.7]  # sub-pixel jitter is not sent
        time.sleep(0.4)
        pos = [d for e, d in got if e == "pos"]
        assert len(pos) == 2, got
        assert list(pos[0]["shapes"]) == ["0"], "only the shape that moved"
        assert pos[0]["shapes"]["0"]["box"] == [0.5, 0.5, 0.7, 0.7] and pos[0]["shapes"]["0"]["lost"] is False
        assert pos[1]["shapes"]["0"]["lost"] is True
        ov.set([{"type": "text", "at": [0.1, 0.1], "text": "next"}], replace=False)
        time.sleep(0.3)
        assert got[-1][0] == "message" and len(got[-1][1]["shapes"]) == 3
    finally:
        stop.set()
        v.stop()
        feed.stop()
