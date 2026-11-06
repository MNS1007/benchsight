"""The recording tools, called directly with a synthetic feed: no camera, no network."""
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from benchsight import camera, cleanup, recording, server
from benchsight.feed import LiveFeed


class Bench:
    """A textured square that moves one step per frame on a textured background."""

    def __init__(self):
        rng = np.random.default_rng(2)
        self.bg = rng.integers(40, 200, (240, 320, 3), dtype=np.uint8)
        self.tex = rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)
        self.x = 0

    def frame(self):
        self.x = (self.x + 6) % 260
        f = self.bg.copy()
        f[100:140, self.x:self.x + 40] = self.tex
        return f


@pytest.fixture
def live(monkeypatch, tmp_path):
    bench = Bench()
    monkeypatch.delenv("BENCHSIGHT_KEEP", raising=False)
    monkeypatch.setattr(server, "LiveFeed", lambda cam=0: LiveFeed(cam, fps=30, source=bench.frame))

    def no_tracker(feed):
        raise RuntimeError("off for this test")
    monkeypatch.setattr(server, "_make_tracker", no_tracker)
    save = camera.save
    monkeypatch.setattr(server.camera, "save", lambda img, name: save(img, name, store=tmp_path / "store"))
    monkeypatch.setattr(camera, "STORE", tmp_path / "store")
    monkeypatch.setattr(camera, "ACTIVE", tmp_path / "active-store")
    monkeypatch.setattr(server, "_last", None)
    monkeypatch.setattr(recording, "_current", None)
    monkeypatch.setattr(cleanup, "_files", {})
    yield tmp_path / "store"
    server.close_viewer()


def wait_until(cond, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def frames_saved(n):
    return lambda: len(server._live["recorder"].index) >= n


def test_recording_needs_the_viewer(live):
    with pytest.raises(ValueError, match="open_viewer"):
        server.start_recording()
    with pytest.raises(ValueError, match="start_recording first"):
        server.stop_recording()
    with pytest.raises(ValueError, match="no recordings yet"):
        server.review()


def test_start_stop_and_review(live, monkeypatch):
    server.open_viewer(open_browser=False)
    got = server.start_recording(roi=[0.0, 0.2, 1.0, 0.8], fps=100, max_seconds=5000)
    assert got["fps"] == 15 and got["max_seconds"] == 900 and got["roi"] == [0.0, 0.2, 1.0, 0.8]
    folder = live.resolve() / Path(got["folder"]).name
    assert got["folder"] == str(folder) and folder.is_dir()
    with pytest.raises(ValueError, match="already recording"):
        server.start_recording()
    assert wait_until(frames_saved(4))
    assert server.review().data[:2] == b"\xff\xd8", "review works while recording"

    out = server.stop_recording()
    assert out["stopped"] == "stopped" and out["frames"] >= 4 and out["folder"] == str(folder)
    assert out["duration_s"] > 0 and isinstance(out["activity"], list) and "review()" in out["note"]
    assert out["peak_activity"] > 1.0, "the square moves inside the roi"
    assert recording.current() is None and server._live["recorder"] is None
    with pytest.raises(ValueError, match="start_recording first"):
        server.stop_recording()

    asked = []
    real = recording.review
    monkeypatch.setattr(recording, "review", lambda *a, **k: asked.append(a) or real(*a, **k))
    server.review(frames=100)
    server.review(frames=0, recording=folder.name)
    end = recording.load_index(folder)[-1]["t"]
    server.review(0.0, end / 2, frames=3, recording=str(folder))
    assert [a[3] for a in asked] == [24, 2, 3], "frames clamped to 2..24"
    assert asked[2][1:3] == (0.0, end / 2)
    with pytest.raises(ValueError, match="no recording"):
        server.review(recording="rec-20000101-000000")


def test_a_bad_roi_is_refused(live):
    server.open_viewer(open_browser=False)
    with pytest.raises(ValueError):
        server.start_recording(roi=[0.5, 0.5, 0.2, 0.2])
    assert server._live["recorder"] is None and recording.current() is None


def test_a_failing_first_frame_is_refused_and_recording_still_works(live, monkeypatch):
    server.open_viewer(open_browser=False)
    feed = server._live["feed"]
    real = feed.latest
    monkeypatch.setattr(feed, "latest", lambda: (_ for _ in ()).throw(OSError("Read-only file system")))
    with pytest.raises(ValueError, match="could not start recording: Read-only"):
        server.start_recording()
    assert server._live["recorder"] is None and recording.current() is None
    assert server._live["viewer"].rec() == {"recording": False}, "no REC badge left on"
    monkeypatch.setattr(feed, "latest", real)
    server.start_recording(fps=15)
    assert wait_until(frames_saved(1))


def test_stop_after_the_cap_still_gives_the_summary(live, monkeypatch):
    server.open_viewer(open_browser=False)
    monkeypatch.setattr(recording, "MAX_SECONDS", 0.3)
    server.start_recording(fps=15, max_seconds=60)
    assert wait_until(lambda: not server._live["recorder"].running)
    out = server.stop_recording()
    assert out["stopped"] == "cap" and out["max_seconds"] == 0.3 and out["frames"] >= 2


def test_close_viewer_stops_the_recording(live):
    server.open_viewer(open_browser=False)
    server.start_recording(fps=15)
    rec = server._live["recorder"]
    assert wait_until(frames_saved(2))
    out = server.close_viewer()
    assert not rec.running and rec.reason == "stopped" and recording.current() is None
    assert out["recording"]["frames"] == len(rec.index) and server._live["recorder"] is None
    assert server.review().data[:2] == b"\xff\xd8", "the recording can still be reviewed"


def test_cleanup_stops_and_deletes_the_recording(live):
    server.open_viewer(open_browser=False)
    folder = server.start_recording(fps=15)["folder"]
    rec = server._live["recorder"]
    assert wait_until(frames_saved(3))
    out = server.cleanup_files()
    assert not rec.running and server._live["recorder"] is None
    assert out["files"] >= 4, "frames and the index"
    assert not (live / Path(folder).name).exists()
    server.start_recording(fps=15)  # a new one can start after cleanup
    assert wait_until(frames_saved(1))



def test_session_end_stops_the_recording_before_deleting(live):
    """The exit and signal paths call session_end directly, not the cleanup tool."""
    server.open_viewer(open_browser=False)
    folder = Path(server.start_recording(fps=15)["folder"])
    rec = server._live["recorder"]
    assert wait_until(frames_saved(3))
    cleanup.session_end()
    assert not rec.running and recording.current() is None and server._live["recorder"] is None
    time.sleep(0.2)
    assert not folder.exists()


FPS, N = 10, 20


@pytest.fixture
def video(tmp_path):
    """A 2 s mp4 whose frame k is gray level 10 * k, so a tile shows which frame it is."""
    path = tmp_path / "episode.mp4"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (160, 120))
    if not out.isOpened():
        pytest.skip("no mp4 encoder here")
    for k in range(N):
        out.write(np.full((120, 160, 3), 10 * k, np.uint8))
    out.release()
    cap = cv2.VideoCapture(str(path))
    ok = cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == N
    cap.release()
    if not ok or path.stat().st_size == 0:
        pytest.skip("no mp4 decoder here")
    return path


def test_review_video_samples_evenly_and_labels_times(video, monkeypatch, tmp_path):
    save = camera.save
    monkeypatch.setattr(server.camera, "save", lambda img, name: save(img, name, store=tmp_path / "store"))
    shown = []
    real = recording.labeled_sheet
    monkeypatch.setattr(recording, "labeled_sheet", lambda items, m: shown.append(items) or real(items, m))
    before = {p.name for p in tmp_path.iterdir()}

    assert server.review_video(str(video), frames=5).data[:2] == b"\xff\xd8"
    times = [t for t, _ in shown[0]]
    assert times == [0.0, 0.5, 1.0, 1.4, 1.9]
    for t, f in shown[0]:
        assert abs(float(f.mean()) - 10 * round(t * FPS)) < 4, "the tile is the frame its label names"

    server.review_video(str(video), start_s=0.5, end_s=1.0, frames=100)
    times = [t for t, _ in shown[1]]
    assert times == [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "inside the window, deduped"
    server.review_video(str(video), start_s=1.5, end_s=99, frames=1)
    assert [t for t, _ in shown[2]] == [1.5, 1.9], "frames at least 2, end clamped to the last frame"

    assert {p.name for p in tmp_path.iterdir()} - before == {"store"}, "no copy of the video"
    assert all(p.name.endswith("-review.jpg") for p in (tmp_path / "store").iterdir())


def test_review_video_rejects_bad_paths(video, tmp_path):
    with pytest.raises(ValueError, match="no video file"):
        server.review_video(str(tmp_path / "missing.mp4"))
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="could not"):
        server.review_video(str(junk))
    with pytest.raises(ValueError, match="start_s is after end_s"):
        server.review_video(str(video), start_s=1.5, end_s=0.5)


def test_the_viewer_badge_follows_the_recording(live):
    server.open_viewer(open_browser=False)
    viewer = server._live["viewer"]
    assert viewer.rec() == {"recording": False}
    server.start_recording(fps=15)
    rec = viewer.rec()
    assert rec["recording"] is True and rec["folder"] == server._live["recorder"].folder.name
    server.stop_recording()
    assert viewer.rec() == {"recording": False}
