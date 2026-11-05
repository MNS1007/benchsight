"""Benchmark the tracker on a synthetic 640x480, 15 fps feed: fps reached and CPU use for
0, 1, 4 and 8 objects, with MIL and (if its model is cached) VitTrack. No camera needed.

  .venv/bin/python scripts/bench_tracking.py [--seconds 8] [--backends mil,vit]

CPU % is of one core: process CPU time (time.process_time, all threads) over wall time.
"net" subtracts a run with the feed alone, which counts drawing the synthetic frames.
"""
from __future__ import annotations

import argparse
import math
import platform
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from benchsight import track
from benchsight.events import Events
from benchsight.feed import LiveFeed

W, H, S, FPS = 640, 480, 48, 15


class Bench:
    """Up to 8 textured squares drifting slowly over a cluttered background."""

    def __init__(self, n: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.bg = cv2.GaussianBlur(rng.integers(0, 255, (H, W, 3), dtype=np.uint8), (0, 0), 3)
        for _ in range(40):
            x, y = int(rng.integers(0, W - 30)), int(rng.integers(0, H - 30))
            cv2.rectangle(self.bg, (x, y), (x + 20, y + 15), [int(v) for v in rng.integers(0, 255, 3)], -1)
        self.tex = [cv2.resize(rng.integers(0, 255, (8, 8, 3), dtype=np.uint8), (S, S), interpolation=cv2.INTER_NEAREST)
                    for _ in range(n)]
        self.home = [(60 + 140 * (i % 4), 80 + 200 * (i // 4)) for i in range(n)]
        self.t0 = time.time()

    def pos(self, i: int, t: float) -> tuple[int, int]:
        x, y = self.home[i]
        return int(x + 30 * math.sin(t * 0.8 + i)), int(y + 20 * math.cos(t * 0.6 + i))

    def frame(self) -> np.ndarray:
        t = time.time() - self.t0
        f = self.bg.copy()
        for i, tex in enumerate(self.tex):
            x, y = self.pos(i, t)
            f[y:y + S, x:x + S] = tex
        return f

    def boxes(self) -> list[dict]:
        t = time.time() - self.t0
        out = []
        for i in range(len(self.tex)):
            x, y = self.pos(i, t)
            out.append({"id": f"part{i}", "label": f"part {i}", "box": [x / W, y / H, (x + S) / W, (y + S) / H]})
        return out


def measure(seconds: float, warmup: float, tick) -> tuple[float, float]:
    """(cpu %, fps) over `seconds` after `warmup`; tick() returns a step counter."""
    time.sleep(warmup)
    c0, w0, s0 = time.process_time(), time.perf_counter(), tick()
    time.sleep(seconds)
    c1, w1, s1 = time.process_time(), time.perf_counter(), tick()
    return 100 * (c1 - c0) / (w1 - w0), (s1 - s0) / (w1 - w0)


def run(backend: str | None, n: int, seconds: float, store: Path) -> dict:
    bench = Bench(n)
    feed = LiveFeed(fps=FPS, source=bench.frame)
    feed.start()
    try:
        if backend is None:  # the feed alone: the baseline
            cpu, fps = measure(seconds, 1.0, lambda: feed.seq)
            return {"cpu": cpu, "fps": fps, "tracking": 0}
        b = track.Backend("vit", track.model_path("vittrack", allow_download=False)) if backend == "vit" \
            else track.Backend("mil")
        tracker = track.Tracker(feed, backend=b, fps=FPS, events=Events(store / "events.jsonl"))
        if n:
            tracker.add(bench.boxes(), frame=feed.latest())
        tracker.start()
        try:
            cpu, fps = measure(seconds, 1.5, lambda: tracker.steps)
            tracking = sum(o["status"] == "tracking" for o in tracker.objects())
        finally:
            tracker.stop()
        return {"cpu": cpu, "fps": fps, "tracking": tracking, "error": tracker.error}
    finally:
        feed.stop()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=8.0, help="measured seconds per run")
    p.add_argument("--backends", default="mil,vit")
    p.add_argument("--objects", default="0,1,4,8")
    a = p.parse_args()
    backends = [b for b in a.backends.split(",") if b]
    if "vit" in backends and not track.model_path("vittrack", allow_download=False):
        print("VitTrack model not cached; skipping vit (run the server once online to fetch it)")
        backends.remove("vit")
    print(f"{platform.platform()} | {platform.processor() or platform.machine()} | OpenCV {cv2.__version__}")
    print(f"synthetic {W}x{H} feed at {FPS} fps, {a.seconds:.0f} s per run, tracker on 1 OpenCV thread\n")
    with tempfile.TemporaryDirectory() as tmp:
        base = run(None, 0, a.seconds, Path(tmp))
        print(f"feed alone: {base['cpu']:5.1f}% of one core, {base['fps']:4.1f} fps\n")
        print(f"{'backend':8} {'objects':>7} {'fps':>6} {'cpu %':>7} {'net %':>7} {'tracking':>9}")
        for backend in backends:
            for n in [int(x) for x in a.objects.split(",")]:
                r = run(backend, n, a.seconds, Path(tmp))
                print(f"{backend:8} {n:7d} {r['fps']:6.1f} {r['cpu']:7.1f} {r['cpu'] - base['cpu']:7.1f} "
                      f"{r['tracking']:>4}/{n:<4}" + (f"  error: {r['error']}" if r.get("error") else ""))


if __name__ == "__main__":
    main()
