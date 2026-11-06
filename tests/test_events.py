"""Wake-up events: debounced, written by the tracker on synthetic frames, read by the CLI.
No camera, no network, no sleeps (a fake clock drives the debounce)."""
import io
import json
import threading
import time

import cv2
import numpy as np
import pytest

from benchsight import cli, events, motion, track
from test_track import Scene

FPS = 15


@pytest.fixture(autouse=True)
def one_cv_thread():
    n = cv2.getNumThreads()
    cv2.setNumThreads(1)
    yield
    cv2.setNumThreads(n)


def lines(path):
    return [json.loads(ln) for ln in path.read_text().splitlines()] if path.exists() else []


def run(ev, frames, t0, cam=None):
    """Feed (gray frame, camera motion) pairs at 15 fps; returns the time after the last."""
    cm = cam or motion.CameraMotion()
    t = t0
    for f in frames:
        g = motion.gray(f)
        ev.observe(g, cm.update(g), t)
        t += 1 / FPS
    return t


# ── the debouncer (VISTA's KeyframeSampler, in seconds) ──

def test_sampler_fires_on_accumulated_motion_but_never_inside_the_cooldown():
    s = events.KeyframeSampler(motion_threshold=10, min_cooldown=5)
    assert not s.should_sample(0.0, 4) and not s.should_sample(0.1, 4)
    assert s.should_sample(0.2, 4)  # 12 >= 10
    assert not s.should_sample(1.0, 100), "inside the cooldown"
    assert s.should_sample(5.3, 0), "motion seen during the cooldown still counts after it"


def test_sampler_interval():
    s = events.KeyframeSampler(interval=3, motion_threshold=1e9, min_cooldown=1)
    assert s.should_sample(0.0) and not s.should_sample(2.0) and s.should_sample(3.0)


# ── the events file ──

def test_events_are_merged_into_one_line_at_most_every_5_s(tmp_path):
    ev = events.Events(tmp_path / "events.jsonl")
    ev.add("object_lost", 0.0, id="plate", label="slider plate")
    ev.flush(0.0)
    ev.add("object_lost", 1.0, id="servo", label="servo")
    ev.add("scene_changed", 2.0, reason="camera moved")
    assert ev.flush(3.0) is None
    ev.flush(5.0)
    got = lines(tmp_path / "events.jsonl")
    assert [g["event"] for g in got] == ["object_lost", "object_lost"]
    assert [e["event"] for e in got[1]["events"]] == ["object_lost", "scene_changed"]
    assert got[0]["summary"] == "slider plate (plate) lost"
    assert "servo (servo) lost; scene changed: camera moved" == got[1]["summary"]


def test_lost_then_found_inside_one_window_cancels_out(tmp_path):
    ev = events.Events(tmp_path / "events.jsonl")
    ev.add("scene_changed", 0.0)
    ev.flush(0.0)
    ev.add("object_lost", 1.0, id="plate", label="plate")
    ev.add("object_found", 2.0, id="plate", label="plate")
    assert ev.flush(6.0) is None and len(lines(tmp_path / "events.jsonl")) == 1
    ev.add("object_lost", 7.0, id="plate", label="plate")
    ev.add("object_found", 8.0, id="plate", label="plate")
    ev.add("object_lost", 9.0, id="plate", label="plate")
    assert ev.flush(12.0)["event"] == "object_lost"


def test_the_file_is_recreated_if_deleted(tmp_path):
    path, seen = tmp_path / "sub" / "events.jsonl", []
    ev = events.Events(path, on_write=seen.append)
    ev.add("still", 0.0)
    ev.flush(0.0)
    path.unlink()
    ev.add("still", 10.0)
    ev.flush(10.0)
    assert len(lines(path)) == 1 and seen == [path, path]


# ── from frames ──

def test_a_static_noisy_scene_writes_nothing(tmp_path):
    sc, ev = Scene(), events.Events(tmp_path / "e.jsonl")
    run(ev, [sc.frame(200) for _ in range(20 * FPS)], 0.0)
    assert lines(tmp_path / "e.jsonl") == []


def test_a_plain_desk_writes_nothing(tmp_path):
    """No texture: camera motion is never confidently estimated, so no 'view blocked'."""
    rng, ev = np.random.default_rng(0), events.Events(tmp_path / "e.jsonl")
    frames = [np.clip(200 + rng.normal(0, 3, (480, 640, 3)), 0, 255).astype(np.uint8) for _ in range(10 * FPS)]
    run(ev, frames, 0.0)
    assert lines(tmp_path / "e.jsonl") == []


def test_camera_bump_then_still(tmp_path):
    sc, ev = Scene(), events.Events(tmp_path / "e.jsonl")
    cm = motion.CameraMotion()
    frames = [sc.frame(200) for _ in range(FPS)]
    frames += [sc.frame(200, shift=(4 * i, 2 * i)) for i in range(1, 16)]  # the camera slides
    frames += [sc.frame(200, shift=(60, 30)) for _ in range(10 * FPS)]  # and stays
    run(ev, frames, 0.0, cm)
    got = lines(tmp_path / "e.jsonl")
    assert [g["event"] for g in got] == ["scene_changed", "still"], got
    assert got[0]["events"][0]["reason"] == "camera moved"
    assert got[1]["t"] - got[0]["t"] >= events.COOLDOWN - 1e-6, "debounced"


def test_hands_in_view_wake_at_most_every_5_s(tmp_path):
    """Something big moving for 20 s: about 4 wake-ups, not 300."""
    sc, ev = Scene(), events.Events(tmp_path / "e.jsonl")
    frames = []
    for i in range(20 * FPS):
        f = sc.frame(200)
        x = 100 + int(300 * abs(np.sin(i / 6)))
        f[150:350, x:x + 120] = (170, 200, 240)  # a bright hand-sized block sweeping about
        frames.append(f)
    run(ev, frames, 0.0)
    got = lines(tmp_path / "e.jsonl")
    assert 2 <= len(got) <= 5, got
    assert all(b["t"] - a["t"] >= events.COOLDOWN - 1e-6 for a, b in zip(got, got[1:]))
    assert {g["event"] for g in got} == {"scene_changed"}


def test_the_tracker_writes_lost_and_found(tmp_path):
    sc = Scene()
    ev = events.Events(tmp_path / "e.jsonl")
    t = track.Tracker(backend=track.Backend("mil"), events=ev)
    t.add([{"label": "servo", "box": sc.box(200)}], frame=sc.frame(200))
    now = 100.0
    for f in [sc.frame(200)] * 3 + [sc.frame(None)] * (track.LOST_AFTER + 2):
        t.step(f, now=now)
        now += 1 / FPS
    for _ in range(3 * track.REACQUIRE_EVERY):
        t.step(sc.frame(200), now=now)
        now += 1 / FPS
    assert t.objects()[0]["status"] == "tracking"
    for _ in range(6 * FPS):  # quiet, so the pending events are written
        t.step(sc.frame(200), now=now)
        now += 1 / FPS
    evs = [e["event"] for g in lines(tmp_path / "e.jsonl") for e in g["events"]]
    assert evs[0] == "object_lost" and "object_found" in evs, evs


# ── the CLI ──

def test_cli_prints_recent_events(tmp_path, capsys):
    path = tmp_path / "events.jsonl"
    path.write_text("".join(json.dumps({"event": "still", "n": i}) + "\n" for i in range(30)))
    cli.main(["events", "--dir", str(tmp_path)])
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 20 and json.loads(out[-1])["n"] == 29
    cli.main(["events", "--dir", str(tmp_path / "none")])
    assert capsys.readouterr().out == ""


class Out(io.StringIO):
    def __init__(self):
        super().__init__()
        self.flushes = 0

    def flush(self):
        self.flushes += 1


def wait_for(cond, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end and not cond():
        time.sleep(0.02)
    return cond()


def test_follow_prints_only_new_lines_flushed_and_survives_deletion(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"old": 1}\n')
    out, stop = Out(), threading.Event()
    th = threading.Thread(target=events.follow, args=(path, out, stop, 0.02), daemon=True)
    th.start()
    try:
        time.sleep(0.1)
        with open(path, "a") as f:
            f.write('{"new": 1}\n{"par')
        assert wait_for(lambda: '{"new": 1}' in out.getvalue())
        with open(path, "a") as f:
            f.write('tial": 2}\n')
        assert wait_for(lambda: '{"partial": 2}\n' in out.getvalue())
        assert '"old"' not in out.getvalue() and out.flushes >= 2
        path.unlink()  # cleanup at a session end, then a new session writes again
        time.sleep(0.1)
        path.write_text('{"after": 3}\n')
        assert wait_for(lambda: '{"after": 3}' in out.getvalue())
    finally:
        stop.set()
        th.join(2)
