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
