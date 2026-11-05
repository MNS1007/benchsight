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
    cv2.setRNGSeed(0)  # pins OpenCV RNG users; MIL stays nondeterministic, so tests allow for it
    yield
    cv2.setNumThreads(n)


class Scene:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.bg = cv2.GaussianBlur(rng.integers(0, 255, (H, W, 3), dtype=np.uint8), (0, 0), 3)
        for _ in range(30):  # bench clutter, so the camera's own motion can be measured
            x, y, c = int(rng.integers(0, W - 30)), int(rng.integers(0, H - 30)), [int(v) for v in rng.integers(0, 255, 3)]
            cv2.rectangle(self.bg, (x, y), (x + int(rng.integers(8, 30)), y + int(rng.integers(8, 30))), c, -1)
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


# ── object registry: merging, smoothing, decay ──

def test_labels_match_by_word_not_by_accident():
    assert track.labels_match("slider plate", "white plate")
    assert track.labels_match("servo", "servo_2") and track.labels_match("M3 screw", "screw")
    assert not track.labels_match("servo", "screw") and not track.labels_match("the arm", "the base")


def test_duplicate_labels_near_each_other_merge():
    sc, t = Scene(), tracker("mil")
    f = sc.frame(200)
    first = t.add([{"label": "white plate", "box": sc.box(200)}], frame=f)
    near = [(200 + 6) / W, 205 / H, (200 + S + 6) / W, (205 + S) / H]
    again = t.add([{"label": "slider plate", "box": near}], frame=f)
    assert again[0]["id"] == first[0]["id"] == "white-plate"
    assert [o["label"] for o in t.objects()] == ["slider plate"], "one object, with the newest label"
    t.add([{"label": "slider plate", "box": [0.8, 0.05, 0.9, 0.15]}], frame=f)  # same word, far away
    t.add([{"label": "servo", "box": near}], frame=f)  # near, different word
    assert sorted(o["id"] for o in t.objects()) == ["servo", "slider-plate", "white-plate"]


class Scripted(track.Backend):
    """A stand-in tracker that reports whatever box the test sets, to test the registry
    without a real tracker's noise."""

    def __init__(self):
        super().__init__("scripted")
        self.rect = None

    def create(self):
        backend = self

        class T:
            def init(self, frame, rect):
                backend.rect = rect

            def update(self, frame):
                return True, backend.rect

        return T()

    def score(self, tracker):
        return 1.0


def test_drawn_boxes_are_smoothed():
    sc, b = Scene(), Scripted()
    t = track.Tracker(backend=b)
    t.add([{"label": "servo", "box": sc.box(200)}], frame=sc.frame(200))
    b.rect = (210, 200, S, S)
    t.step(sc.frame(210))
    assert abs(get(t, "servo")["box"][0] * W - 203) < 0.5, "the shown box moves 30% of the way to the new spot"
    for _ in range(15):
        t.step(sc.frame(210))
    assert abs(get(t, "servo")["box"][0] * W - 210) < 0.5


def test_lost_objects_decay_go_stale_and_are_pruned():
    sc, clock = Scene(), {"t": 1000.0}
    t = tracker("mil", clock=lambda: clock["t"])
    t.add([{"label": "servo", "box": sc.box(200)}], frame=sc.frame(200))
    for _ in range(track.LOST_AFTER):
        t.step(sc.frame(None), now=clock["t"])
    assert get(t, "servo")["status"] == "lost"
    t.step(sc.frame(None), now=1020.0)  # 20 s: confidence 1 - 0.3
    o = get(t, "servo")
    assert o["status"] == "lost" and abs(o["confidence"] - 0.7) < 0.01
    t.step(sc.frame(None), now=1040.0)
    assert get(t, "servo")["status"] == "stale"
    t.step(sc.frame(None), now=1060.0)
    assert t.objects() == []
    assert [c["event"] for c in t.changes] == ["object_lost", "object_pruned"]


def test_at_most_eight_objects_and_lost_ones_make_room():
    sc, t = Scene(), tracker("mil", max_objects=3)
    f = sc.frame(None)
    t.add([{"id": f"o{i}", "label": f"part{i}", "box": [0.1 + 0.25 * i, 0.1, 0.2 + 0.25 * i, 0.2]} for i in range(3)], frame=f)
    with pytest.raises(ValueError, match="already tracking 3"):
        t.add([{"label": "extra", "box": [0.5, 0.6, 0.6, 0.7]}], frame=f)
    t._objs["o1"].status, t._objs["o1"].confidence = "lost", 0.6
    t.add([{"label": "extra", "box": [0.5, 0.6, 0.6, 0.7]}], frame=f)
    assert sorted(o["id"] for o in t.objects()) == ["extra", "o0", "o2"]
    assert track.Tracker(backend=track.Backend("mil")).max_objects == 8


# ── re-finding lost objects and following the camera ──

def test_find_locates_the_template_near_then_anywhere():
    sc = Scene()
    g0 = cv2.cvtColor(sc.frame(100), cv2.COLOR_BGR2GRAY)
    tpl = g0[200:200 + S, 100:100 + S].copy()
    kp, des = track.motion.orb_features(tpl, orb=track.motion.ORB_SMALL)
    g = cv2.cvtColor(sc.frame(450, y=80), cv2.COLOR_BGR2GRAY)
    for k, d in [(kp, des), ((), None)]:  # ORB, then template matching alone
        box = track.find(g, tpl, k, d, near=[100, 200, 160, 260])
        assert box is not None and iou(box, [450, 80, 510, 140]) > 0.7, box
    assert track.find(cv2.cvtColor(sc.frame(None), cv2.COLOR_BGR2GRAY), tpl, kp, des) is None


@pytest.mark.parametrize("backend", BACKENDS)
def test_refound_when_it_reappears(backend):
    sc, t = Scene(), tracker(backend)
    t.add([{"label": "servo", "box": sc.box(150)}], frame=sc.frame(150))
    for _ in range(track.LOST_AFTER + 1):
        t.step(sc.frame(None))
    assert get(t, "servo")["status"] == "lost"
    for _ in range(2 * track.REACQUIRE_EVERY):  # back, somewhere else
        t.step(sc.frame(420, y=100))
        if get(t, "servo")["status"] == "tracking":
            break
    o = get(t, "servo")
    assert o["status"] == "tracking" and iou(o["box"], sc.box(420, y=100)) > 0.5, o
    assert [c["event"] for c in t.changes] == ["object_lost", "object_found"]
    for x in range(420, 460, 5):  # and keeps following
        t.step(sc.frame(x, y=100))
    assert iou(get(t, "servo")["box"], sc.box(455, y=100)) > 0.5


@pytest.mark.parametrize("backend", BACKENDS)
def test_boxes_follow_when_the_camera_is_bumped(backend):
    """A 40 px jump of the whole frame is beyond MIL's search radius; the camera-motion
    estimate moves the box first."""
    sc, t = Scene(), tracker(backend)
    t.add([{"label": "servo", "box": sc.box(250)}], frame=sc.frame(250))
    for _ in range(3):
        t.step(sc.frame(250))
    t.step(sc.frame(250, shift=(40, 25)))
    assert t.motion.confidence > 0.5 and abs(t.motion.M[0, 2] - 40) < 2 and abs(t.motion.M[1, 2] - 25) < 2
    for _ in range(3):
        t.step(sc.frame(250, shift=(40, 25)))
    o = get(t, "servo")
    assert o["status"] == "tracking" and iou(o["box"], sc.box(250, shift=(40, 25))) > 0.5, o


def test_lost_boxes_move_with_the_camera():
    sc, t = Scene(), tracker("mil")
    t.add([{"label": "servo", "box": sc.box(250)}], frame=sc.frame(250))
    for _ in range(track.LOST_AFTER + 1):
        t.step(sc.frame(None))
    before = get(t, "servo")["box"]
    t.step(sc.frame(None, shift=(30, 0)))
    after = get(t, "servo")["box"]
    assert get(t, "servo")["status"] == "lost" and abs((after[0] - before[0]) * W - 30) < 3
