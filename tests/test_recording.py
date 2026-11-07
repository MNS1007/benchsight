"""Recordings: the activity hint, the timeline, uniform sampling and the recorder.
Synthetic frames only: no camera."""
import cv2
import numpy as np
import pytest

from benchsight import recording

W, H, S = 320, 240, 40


def bench():
    """A textured background kept in mid-range (40-200) so gain changes never clip."""
    rng = np.random.default_rng(3)
    bg = rng.integers(40, 200, (H // 8, W // 8, 3), dtype=np.uint8)
    bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_NEAREST)
    return cv2.GaussianBlur(bg, (0, 0), 1.5)


TEX = cv2.resize(np.random.default_rng(4).integers(40, 200, (5, 5, 3), dtype=np.uint8), (S, S),
                 interpolation=cv2.INTER_NEAREST)


def with_square(bg, x, y):
    f = bg.copy()
    f[y:y + S, x:x + S] = TEX
    return f


def gain(frame, g, offset=0.0):
    return np.clip(frame.astype(np.float32) * g + offset, 0, 255).astype(np.uint8)


def scores(frames, roi=None):
    a = recording.Activity(roi)
    return [a.update(f) for f in frames]


def test_first_frame_and_a_still_scene_score_zero():
    bg = bench()
    assert scores([bg] * 5) == [0.0] * 5


def test_a_brightness_ramp_scores_about_zero():
    bg = bench()
    ramp = [gain(bg, g, o) for g, o in zip(np.linspace(0.8, 1.25, 20), np.linspace(-10, 10, 20))]
    assert max(scores(ramp)) < 0.5


def test_a_flickering_light_scores_about_zero():
    bg = bench()
    flicker = [gain(bg, 1.25 if k % 2 else 0.8) for k in range(20)]
    assert max(scores(flicker)) < 0.5


def test_sensor_noise_scores_zero():
    bg, rng = bench(), np.random.default_rng(0)
    noisy = [np.clip(bg + rng.normal(0, 4, bg.shape), 0, 255).astype(np.uint8) for _ in range(10)]
    assert max(scores(noisy)) < 0.5


def test_a_flat_frame_never_turns_noise_into_activity():
    rng = np.random.default_rng(0)
    flat = [np.clip(120 + rng.normal(0, 0.5, (H, W, 3)), 0, 255).astype(np.uint8) for _ in range(5)]
    assert max(scores(flat)) == 0.0


def test_a_square_moving_inside_the_roi_scores_clearly():
    bg = bench()
    frames = [with_square(bg, 40 + 8 * k, 60) for k in range(10)]
    got = scores(frames, roi=(0.0, 0.0, 0.6, 0.6))
    assert got[0] == 0.0 and min(got[1:]) > 1.5


def test_motion_outside_the_roi_scores_about_zero():
    bg = bench()
    frames = [with_square(bg, 200 + 8 * k, 170) for k in range(10)]  # bottom right only
    assert max(scores(frames, roi=(0.0, 0.0, 0.5, 0.5))) < 0.5
    assert max(scores(frames)) > 1.0, "the same motion counts with no roi"


def test_motion_under_a_flickering_light_still_counts():
    bg = bench()
    frames = [gain(with_square(bg, 40 + 8 * k, 60), 1.2 if k % 2 else 0.85) for k in range(10)]
    assert min(scores(frames, roi=(0.0, 0.0, 0.6, 0.6))[1:]) > 1.5


# ── timeline and sample ──

def index(acts, fps=8):
    return [{"i": k, "t": round(k / fps, 3), "file": f"{k:06d}.jpg", "activity": a} for k, a in enumerate(acts)]


def test_timeline_merges_short_gaps_and_drops_blips():
    acts = [0] * 8 + [5] * 8 + [0] * 4 + [5] * 4 + [0] * 16 + [5] + [0] * 16 + [5] * 8
    #       still       active     gap 0.5 s   active             a blip at 5 s      active from 7.125 s
    segs = recording.timeline(index(acts), threshold=1.0)
    assert segs == [[0.88, 2.88], [7.0, 8.0]]


def test_timeline_threshold():
    assert recording.timeline(index([0, 2, 2, 2, 2, 0]), threshold=3.0) == []
    assert recording.timeline([], 1.0) == []


def test_sample_is_uniform_and_includes_both_ends():
    idx = index([0] * 81)  # 0 .. 10 s
    got = recording.sample(idx, frames=5)
    assert [e["t"] for e in got] == [0.0, 2.5, 5.0, 7.5, 10.0]
    got = recording.sample(idx, 2.0, 4.0, frames=3)
    assert [e["t"] for e in got] == [2.0, 3.0, 4.0]


def test_sample_ignores_activity():
    acts = [0] * 40 + [90] * 41
    assert [e["t"] for e in recording.sample(index(acts), frames=3)] == [0.0, 5.0, 10.0]


def test_sample_window_bounds_and_dedupe():
    idx = index([0] * 81)
    got = recording.sample(idx, 1.0, 1.3, frames=12)  # only 3 frames in the window
    assert [e["t"] for e in got] == [1.0, 1.125, 1.25]
    assert all(1.0 <= e["t"] <= 1.3 for e in got)
    assert recording.sample(idx, 20.0, 30.0) == []
    assert recording.sample([], frames=4) == []
    assert recording.sample(idx, frames=1) == [idx[0]]
    with pytest.raises(ValueError):
        recording.sample(idx, 5.0, 4.0)


# ── the recorder ──

import json  # noqa: E402
import time  # noqa: E402

from benchsight import camera, cleanup  # noqa: E402
from benchsight.feed import LiveFeed  # noqa: E402


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.delenv("BENCHSIGHT_KEEP", raising=False)
    monkeypatch.setattr(cleanup, "_files", {})
    yield
    rec = recording.current()
    if rec is not None:
        rec.stop()


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class Feed:
    """A stand-in for LiveFeed: a square that moves while `moving` is set."""

    def __init__(self):
        self.bg, self.x, self.moving, self.running = bench(), 40, False, True

    def latest(self):
        if self.moving:
            self.x = 40 + (self.x - 32) % 160
        return with_square(self.bg, self.x, 60)


def test_recorder_writes_frames_and_an_index(tmp_path):
    clock, feed = Clock(), Feed()
    rec = recording.Recorder(feed, tmp_path, fps=8, clock=clock).start(background=False)
    assert rec.folder.parent == tmp_path and cleanup.REC.match(rec.folder.name)
    for k in range(1, 16):
        clock.t += 0.125
        feed.moving = 4 <= k < 12
        assert rec.capture()
    got = rec.stop()
    idx = recording.load_index(rec.folder)
    assert len(idx) == 16 == got["frames"] and idx == rec.index
    assert set(idx[0]) == {"i", "t", "file", "activity"}
    assert [e["i"] for e in idx] == list(range(16)) and idx[3]["t"] == 0.375 and idx[0]["file"] == "000000.jpg"
    assert all((rec.folder / e["file"]).exists() for e in idx)
    img = cv2.imread(str(rec.folder / "000000.jpg"))
    assert img.shape == (240, 320, 3)
    assert got["duration_s"] == 1.875 and got["stopped"] == "stopped" and got["folder"] == str(rec.folder)
    assert got["activity"] == [[0.38, 1.38]] and got["peak_activity"] > 1.0


def test_frames_are_downsized_and_registered_for_cleanup(tmp_path):
    class Big(Feed):
        def latest(self):
            return cv2.resize(super().latest(), (1280, 960))
    rec = recording.Recorder(Big(), tmp_path, max_side=320, clock=Clock()).start(background=False)
    rec.capture()
    rec.stop()
    assert cv2.imread(str(rec.folder / "000000.jpg")).shape[:2] == (240, 320)
    files = set(cleanup.session_files())
    assert {rec.folder.resolve(), (rec.folder / "index.jsonl").resolve(), (rec.folder / "000001.jpg").resolve()} <= files
    assert cleanup.session_end()[0] == 3 and not rec.folder.exists()


def test_recorder_stops_itself_at_the_cap(tmp_path):
    clock = Clock()
    rec = recording.Recorder(Feed(), tmp_path, fps=4, max_seconds=2, clock=clock).start(background=False)
    clock.t += 0.25
    while rec.capture():
        clock.t += 0.25
    assert len(rec.index) == 8 and rec.index[-1]["t"] == 1.75
    assert not rec.running and recording.current() is None
    clock.t += 60
    got = rec.stop()
    assert got["stopped"] == "cap" and got["duration_s"] == 2.0 and got["frames"] == 8


def test_settings_are_clamped():
    rec = recording.Recorder(Feed(), "x", fps=100, max_seconds=10_000, max_side=5)
    assert rec.fps == recording.MAX_FPS and rec.max_seconds == recording.MAX_SECONDS == 900 and rec.max_side == 160
    with pytest.raises(ValueError):
        recording.Recorder(Feed(), "x", roi=[0.5, 0.5, 0.2, 0.9])


def test_only_one_recording_at_a_time(tmp_path, monkeypatch):
    monkeypatch.setattr(recording.time, "strftime", lambda fmt: "20260101-000000")  # same second
    a = recording.Recorder(Feed(), tmp_path, clock=Clock()).start(background=False)
    assert recording.current() is a
    with pytest.raises(RuntimeError, match="already recording"):
        recording.Recorder(Feed(), tmp_path, clock=Clock()).start(background=False)
    a.stop()
    assert recording.current() is None
    b = recording.Recorder(Feed(), tmp_path, clock=Clock()).start(background=False)
    assert a.folder.name == "rec-20260101-000000" and b.folder.name == "rec-20260101-000000-2"
    with pytest.raises(RuntimeError, match="already ran"):
        a.start()
    b.stop()
    assert a.stop() == a.summary(), "stop is safe to call again"


def test_capture_before_start_says_so(tmp_path):
    with pytest.raises(RuntimeError, match="not started"):
        recording.Recorder(Feed(), tmp_path).capture()


def test_recorder_stops_when_the_feed_does(tmp_path):
    clock, feed = Clock(), Feed()
    rec = recording.Recorder(feed, tmp_path, clock=clock).start(background=False)
    feed.running = False
    clock.t += 0.5
    assert not rec.capture() and rec.stop()["stopped"] == "feed"
    with pytest.raises(RuntimeError, match="not running"):
        recording.Recorder(feed, tmp_path).start()



def test_a_frozen_camera_is_not_recorded_as_a_still_scene(tmp_path):
    """LiveFeed keeps its last frame when the camera stops giving new ones (seq stops)."""
    bg, frozen = bench(), []
    feed = LiveFeed(fps=60, source=lambda: None if frozen else bg)
    feed.start()
    try:
        clock = Clock()
        rec = recording.Recorder(feed, tmp_path, clock=clock).start(background=False)
        frozen.append(True)
        time.sleep(0.1)  # let a frame in flight land
        clock.t += 0.125
        rec.capture()
        n = len(rec.index)
        for _ in range(3):
            clock.t += 0.125
            assert rec.capture() and len(rec.index) == n, "the same frame is not saved again"
        clock.t += recording.STALL_S
        assert not rec.capture() and rec.stop()["stopped"] == "stalled"
        assert recording.current() is None
    finally:
        feed.stop()


def test_the_thread_records_at_about_fps(tmp_path):
    bg = bench()
    feed = LiveFeed(fps=30, source=lambda: bg)
    feed.start()
    try:
        rec = recording.Recorder(feed, tmp_path, fps=10).start()
        time.sleep(0.6)
        assert rec.status()["recording"] and rec.status()["frames"] >= 1
        got = rec.stop()
    finally:
        feed.stop()
    assert 4 <= got["frames"] <= 9
    steps = np.diff([e["t"] for e in rec.index])
    assert abs(float(np.median(steps)) - 0.1) < 0.03
    assert not rec.status()["recording"]


def test_load_index_skips_a_half_written_line(tmp_path):
    folder = tmp_path / "rec-20260101-000000"
    folder.mkdir()
    (folder / "index.jsonl").write_text(json.dumps({"i": 0, "t": 0.0, "file": "000000.jpg", "activity": 0}) + '\n{"i": 1, "t"')
    assert [e["i"] for e in recording.load_index(folder)] == [0]
    assert recording.load_index(tmp_path / "missing") == []


def test_find_the_newest_or_a_named_recording(tmp_path):
    with pytest.raises(ValueError, match="no recordings"):
        recording.find(tmp_path)
    for name in ("rec-20260101-000000", "rec-20260101-000000-2", "rec-20260101-000000-10", "rec-20250101-000000"):
        (tmp_path / name).mkdir()
    (tmp_path / "rec-notes").mkdir()
    assert recording.find(tmp_path).name == "rec-20260101-000000-10"
    assert recording.find(tmp_path, "rec-20250101-000000").name == "rec-20250101-000000"
    assert recording.find(tmp_path, str(tmp_path / "rec-20250101-000000")).name == "rec-20250101-000000"
    for bad in ("rec-notes", "rec-20990101-000000", "../rec-20250101-000000", str(tmp_path / "x" / "rec-20250101-000000")):
        with pytest.raises(ValueError, match="no recording"):
            recording.find(tmp_path, bad)


# ── the review sheet ──

def test_labeled_sheet_fits_and_labels_every_tile():
    items = [(k * 0.5, np.full((480, 640, 3), 90, np.uint8)) for k in range(12)]
    sheet = recording.labeled_sheet(items, max_side=1600)
    assert max(sheet.shape[:2]) <= 1600 and sheet.shape[1] == 4 * 400 and sheet.shape[0] == 3 * 300
    for r in range(3):  # each tile's label band is dark, with yellow text on it
        for c in range(4):
            tag = sheet[r * 300:r * 300 + 30, c * 400:c * 400 + 90]
            assert (tag == 0).all(axis=2).any() and (tag[:, :, 2] > 150).any() and (tag[:, :, 0] < 60).all()
    assert recording.labeled_sheet(items[:2], 800).shape[1] <= 800
    with pytest.raises(ValueError):
        recording.labeled_sheet([])


def test_tile_labels_are_number_and_time(monkeypatch):
    seen = []
    real = camera.contact_sheet
    monkeypatch.setattr(camera, "contact_sheet", lambda tiles, **kw: seen.append(kw["labels"]) or real(tiles, **kw))
    recording.labeled_sheet([(0.0, bench()), (4.2, bench()), (12.25, bench())])
    assert seen == [["1  0.0s", "2  4.2s", "3  12.2s"]]


def test_review_samples_a_recording_into_one_saved_sheet(tmp_path):
    clock, feed = Clock(), Feed()
    store = tmp_path / "store"
    rec = recording.Recorder(feed, store, fps=8, clock=clock).start(background=False)
    for _ in range(39):  # 0 .. 4.875 s
        clock.t += 0.125
        rec.capture()
    shot, shown = recording.review(rec.folder, store=store)  # while still recording
    assert len(shown) == 12 and shown[0]["t"] == 0.0 and shown[-1]["t"] == 4.875
    assert shot.path.parent == store and shot.path.exists() and shot.path.resolve() in cleanup.session_files()
    rec.stop()
    _, shown = recording.review(rec.folder, 1.0, 2.0, frames=5, store=store)
    assert [e["t"] for e in shown] == [1.0, 1.25, 1.5, 1.75, 2.0]
    with pytest.raises(ValueError, match="runs 0.0s to 4.9s"):
        recording.review(rec.folder, 10.0, 20.0, store=store)
    cleanup.session_end()
    assert not rec.folder.exists() and not shot.path.exists()


def test_a_failing_first_frame_does_not_leave_a_recording_on(tmp_path):
    class Broken(Feed):
        def latest(self):
            raise OSError("No space left on device")
    rec = recording.Recorder(Broken(), tmp_path, clock=Clock())
    with pytest.raises(OSError, match="No space"):
        rec.start(background=False)
    assert recording.current() is None and not rec.running and rec.reason.startswith("error: ")
    assert rec.stop()["stopped"].startswith("error: ")
    b = recording.Recorder(Feed(), tmp_path, clock=Clock()).start(background=False)
    assert recording.current() is b
    b.stop()


def test_flicker_on_a_low_contrast_scene_is_not_activity():
    """A smooth, dim scene whose spread crosses the flat cutoff as the light flickers:
    one frame reads as flat, the next does not. That is not motion."""
    y, x = np.mgrid[0:360, 0:640]
    base = 5.0 * np.sin(x / 40.0) * np.cos(y / 30.0)
    dim, bright = [cv2.cvtColor(np.clip((100 + base) * g, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                   for g in (0.65, 1.35)]
    a, b = recording.small_gray(dim), recording.small_gray(bright)
    assert not a.any() and b.any(), "the scene must straddle the flat cutoff"
    act = recording.Activity()
    scores = [act.update(f) for f in [dim, bright] * 8]
    assert max(scores) == 0.0, scores
