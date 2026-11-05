"""The tracking tools, called directly with a synthetic feed: no camera, no network."""
import time

import cv2
import numpy as np
import pytest

from benchsight import camera, server, track
from benchsight.feed import LiveFeed

W, H, S = 640, 480, 60


class Bench:
    """A textured square on a cluttered background; the test moves it."""

    def __init__(self):
        rng = np.random.default_rng(1)
        self.bg = cv2.GaussianBlur(rng.integers(0, 255, (H, W, 3), dtype=np.uint8), (0, 0), 3)
        for _ in range(30):
            x, y = int(rng.integers(0, W - 30)), int(rng.integers(0, H - 30))
            cv2.rectangle(self.bg, (x, y), (x + 20, y + 15), [int(v) for v in rng.integers(0, 255, 3)], -1)
        self.tex = cv2.resize(rng.integers(0, 255, (10, 10, 3), dtype=np.uint8), (S, S), interpolation=cv2.INTER_NEAREST)
        self.x, self.y = 200, 200

    def frame(self):
        f = self.bg.copy()
        f[self.y:self.y + S, self.x:self.x + S] = self.tex
        return f

    def box(self):
        return [self.x / W, self.y / H, (self.x + S) / W, (self.y + S) / H]


@pytest.fixture
def live(monkeypatch, tmp_path):
    bench = Bench()
    monkeypatch.setattr(server, "LiveFeed", lambda cam=0: LiveFeed(cam, fps=30, source=bench.frame))
    monkeypatch.setattr(server, "_make_tracker", lambda feed: track.Tracker(feed, backend="mil", fps=30))
    save = camera.save
    monkeypatch.setattr(server.camera, "save", lambda img, name: save(img, name, store=tmp_path))
    monkeypatch.setattr(server, "_last", None)
    n = cv2.getNumThreads()
    yield bench
    server.close_viewer()
    cv2.setNumThreads(n)


def wait_until(cond, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_tracking_tools_need_the_viewer_and_a_look(live):
    with pytest.raises(ValueError, match="open the viewer"):
        server.track([{"label": "plate", "box": live.box()}])
    with pytest.raises(ValueError, match="open the viewer"):
        server.objects()
    server.open_viewer(open_browser=False)
    with pytest.raises(ValueError, match="take a look first"):
        server.track([{"label": "plate", "box": live.box()}])


def test_track_annotate_follow_and_untrack(live):
    out = server.open_viewer(open_browser=False)
    assert "Tracking ready (mil)" in out["note"]
    server.look()
    got = server.track([{"label": "white plate", "box": live.box()}])
    assert got["tracking"] == [{"id": "white-plate", "label": "white plate"}] and got["backend"] == "mil"

    with pytest.raises(ValueError, match="tracked: white-plate"):
        server.annotate([{"type": "circle", "at": [0.3, 0.3], "anchor": "screw"}])
    b = live.box()
    server.annotate([{"type": "box", "box": b, "anchor": "white-plate", "label": "this plate"},
                     {"type": "circle", "at": [0.9, 0.9], "r": 0.03, "label": "static"}], caption="lift it")

    live.x += 40  # the plate moves; the anchored box follows it, the static circle doesn't
    assert wait_until(lambda: server.objects()[0]["box"][0] > b[0] + 0.03)
    objs = server.objects()
    assert set(objs[0]) == {"id", "label", "box", "status", "last_seen_s_ago"} and objs[0]["status"] == "tracking"
    drawn = server._drawn()["shapes"]
    assert np.allclose(drawn[0]["box"], server.objects()[0]["box"], atol=0.02), "anchored box sits on the object"
    assert drawn[0]["lost"] is False
    assert drawn[1] == {"type": "circle", "color": "red", "label": "static", "at": [0.9, 0.9], "r": 0.03}
    img = server.look(with_annotations=True)
    assert img.data[:2] == b"\xff\xd8"

    # A repeat box with a matching label merges into the same id.
    again = server.track([{"label": "slider plate", "box": live.box()}])
    assert again["tracking"][0]["id"] == "white-plate" and again["shapes_removed"] == 0

    gone = server.untrack(["white-plate"])
    assert gone == {"ok": True, "untracked": ["white-plate"], "shapes_removed": 1}
    assert server.objects() == [] and [s["label"] for s in server.overlay.state()["shapes"]] == ["static"]


def test_replace_drops_old_objects_and_their_shapes(live):
    server.open_viewer(open_browser=False)
    server.look()
    server.track([{"id": "plate", "label": "plate", "box": live.box()}])
    server.annotate([{"type": "arrow", "from": [0.1, 0.9], "to": [0.35, 0.45], "anchor": "plate"}])
    got = server.track([{"id": "bolt", "label": "bolt", "box": [0.7, 0.1, 0.8, 0.2]}])
    assert got["untracked"] == ["plate"] and got["shapes_removed"] == 1
    assert [o["id"] for o in server.objects()] == ["bolt"]


def test_anchored_shapes_show_lost_and_close_viewer_stops_tracking(live):
    server.open_viewer(open_browser=False)
    server.look()
    server.track([{"id": "plate", "label": "plate", "box": live.box()}])
    server.annotate([{"type": "box", "box": live.box(), "anchor": "plate"}])
    tracker = server._live["tracker"]
    assert tracker.running and server._drawn()["shapes"][0]["lost"] is False
    tracker.stop()  # hold it still, then mark it lost by hand: what matters is how lost is shown
    tracker._objs["plate"].status = "lost"
    assert server._drawn()["shapes"][0]["lost"] is True
    tracker.start()
    server.close_viewer()
    assert not tracker.running and server._live["tracker"] is None and server.overlay.state()["shapes"] == []


def test_viewer_opens_without_tracking_if_the_tracker_fails(live, monkeypatch):
    def broken(feed):
        raise RuntimeError("no tracker")
    monkeypatch.setattr(server, "_make_tracker", broken)
    out = server.open_viewer(open_browser=False)
    assert "Tracking is off: no tracker" in out["note"]
    server.annotate([{"type": "text", "at": [0.5, 0.5], "text": "still works"}])
    with pytest.raises(ValueError, match="call track first"):
        server.annotate([{"type": "text", "at": [0.5, 0.5], "text": "x", "anchor": "plate"}])
    with pytest.raises(ValueError, match="tracking is off: no tracker"):
        server.track([{"label": "plate", "box": live.box()}])
