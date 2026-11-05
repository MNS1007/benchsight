"""Tracking on a synthetic video: a textured square moving over a noisy background.
No camera and no network. MIL is always tested; VitTrack only if its model is cached."""
import time

import cv2
import numpy as np
import pytest

from benchsight import track
from benchsight.feed import LiveFeed

W, H, S = 640, 480, 60  # frame size and square side
VIT = track.model_path("vittrack", allow_download=False)
BACKENDS = ["mil"] + (["vit"] if VIT else [])


@pytest.fixture(autouse=True)
def one_cv_thread():
    n = cv2.getNumThreads()
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)  # MIL samples at random; keep runs repeatable
    yield
    cv2.setNumThreads(n)


class Scene:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.bg = cv2.GaussianBlur(rng.integers(0, 255, (H, W, 3), dtype=np.uint8), (0, 0), 3)
        tex = rng.integers(0, 255, (S // 6, S // 6, 3), dtype=np.uint8)
        self.tex = cv2.resize(tex, (S, S), interpolation=cv2.INTER_NEAREST)  # blocky, high contrast
        cv2.rectangle(self.tex, (0, 0), (S - 1, S - 1), (255, 255, 255), 3)

    def frame(self, x=None, y=200, shift=(0, 0)):
        f = self.bg.copy()
        if x is not None:
            f[y:y + S, x:x + S] = self.tex
        if shift != (0, 0):
            f = cv2.warpAffine(f, np.float32([[1, 0, shift[0]], [0, 1, shift[1]]]), (W, H), borderMode=cv2.BORDER_REFLECT)
        noise = self.rng.normal(0, 4, f.shape)
        return np.clip(f + noise, 0, 255).astype(np.uint8)

    @staticmethod
    def box(x, y=200, shift=(0, 0)):
        return [(x + shift[0]) / W, (y + shift[1]) / H, (x + S + shift[0]) / W, (y + S + shift[1]) / H]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def tracker(backend, **kw):
    b = track.Backend("vit", VIT) if backend == "vit" else track.Backend("mil")
    return track.Tracker(backend=b, **kw)


def get(t, oid):
    return next(o for o in t.objects() if o["id"] == oid)


# ── model cache and backend choice ──

def test_model_cache_rejects_a_bad_file_and_never_downloads_when_told_not_to(tmp_path):
    bad = tmp_path / track.MODELS["vittrack"]["file"]
    bad.write_bytes(b"not a model")
    assert track.model_path("vittrack", allow_download=False, model_dir=tmp_path) is None
    assert not bad.exists(), "a file with the wrong hash is removed"


def test_offline_falls_back_to_mil(tmp_path, monkeypatch):
    def offline(*a, **k):
        raise OSError("no network")
    monkeypatch.setattr(track.urllib.request, "urlopen", offline)
    assert track.model_path("vittrack", model_dir=tmp_path) is None
    assert track.make_backend("auto", model_dir=tmp_path).name == "mil"
    with pytest.raises(RuntimeError, match="VitTrack"):
        track.make_backend("vit", model_dir=tmp_path)


@pytest.mark.skipif(not VIT, reason="VitTrack model not cached")
def test_vit_is_preferred_when_cached():
    assert track.make_backend("auto", allow_download=False, model_dir=VIT.parent).name == "vit"


# ── following and losing ──

@pytest.mark.parametrize("backend", BACKENDS)
def test_box_follows_the_square(backend):
    sc, t = Scene(), tracker(backend)
    t.add([{"label": "servo", "box": sc.box(100)}], frame=sc.frame(100))
    worst = 1.0
    for x in range(100, 400, 5):
        t.step(sc.frame(x))
        o = get(t, "servo")
        assert o["status"] == "tracking", (x, o)
        worst = min(worst, iou(o["box"], sc.box(x)))
    assert worst > 0.5


@pytest.mark.parametrize("backend", BACKENDS)
def test_lost_when_hidden(backend):
    sc, t = Scene(), tracker(backend)
    t.add([{"label": "servo", "box": sc.box(200)}], frame=sc.frame(200))
    for _ in range(3):
        t.step(sc.frame(200))
    seen = get(t, "servo")["last_seen"]
    for _ in range(track.LOST_AFTER + 2):
        t.step(sc.frame(None))
    o = get(t, "servo")
    assert o["status"] == "lost" and o["last_seen"] == seen
    assert iou(o["box"], sc.box(200)) > 0.5, "lost objects keep their last good box"


@pytest.mark.parametrize("backend", BACKENDS)
def test_initialized_on_the_frame_the_agent_saw(backend):
    """The square moved 12 px between the agent's look and the live frames."""
    sc, t = Scene(), tracker(backend)
    t.add([{"label": "servo", "box": sc.box(100)}], frame=sc.frame(100))
    for _ in range(3):
        t.step(sc.frame(112))
    assert abs(get(t, "servo")["box"][0] * W - 112) < 8


def test_boxes_are_checked_and_ids_are_unique():
    sc, t = Scene(), tracker("mil")
    with pytest.raises(ValueError, match="no camera frame"):
        t.add([{"label": "a", "box": [0.1, 0.1, 0.2, 0.2]}])
    for bad, why in [([0.1, 0.1, 1.5, 0.2], "outside"), ([0.1, 0.1, 0.1, 0.1], "too small"), ([0.1], "x1")]:
        with pytest.raises(ValueError, match=why):
            t.add([{"label": "a", "box": bad}], frame=sc.frame())
    ids = t.add([{"label": "M3 screw", "box": [0.1, 0.1, 0.2, 0.2]}, {"id": "s2", "label": "x", "box": [0.5, 0.5, 0.6, 0.6]}],
                frame=sc.frame())
    assert [i["id"] for i in ids] == ["m3-screw", "s2"]
    assert t.untrack(["s2", "nope"]) == ["s2"] and [o["id"] for o in t.objects()] == ["m3-screw"]
    t.add([{"label": "b", "box": [0.7, 0.7, 0.8, 0.8]}], frame=sc.frame(), replace=True)
    assert [o["id"] for o in t.objects()] == ["b"]


def test_thread_tracks_a_live_feed():
    """Plumbing only (following is tested frame by frame above): the thread reads the feed,
    skips frames it has already seen, and stops."""
    sc = Scene()
    feed = LiveFeed(source=lambda: sc.frame(300), fps=30)
    feed.start()
    t = track.Tracker(feed, backend=track.Backend("mil"), fps=60)
    try:
        t.add([{"label": "servo", "box": sc.box(300)}])
        t.start()
        deadline = time.time() + 5
        while t.steps < 5 and time.time() < deadline:
            time.sleep(0.05)
        assert t.running and t.steps >= 5 and not t.error
        assert t.steps <= feed.seq, "never tracks the same frame twice"
        o = get(t, "servo")
        assert o["status"] == "tracking" and iou(o["box"], sc.box(300)) > 0.7, o
    finally:
        t.stop()
        feed.stop()
    assert not t.running
