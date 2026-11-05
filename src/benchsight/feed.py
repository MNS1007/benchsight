"""A live camera feed for the viewer. One background thread owns the camera while the
viewer is open, so tools read its latest frame instead of opening the camera again."""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np


class LiveFeed:
    def __init__(self, cam: int = 0, fps: float = 15.0, source=None):
        """source: a function returning a frame, for tests; by default the camera."""
        self.cam, self.fps = cam, fps
        self._source = source
        self._cap = None
        self._frame: np.ndarray | None = None
        self.seq = 0  # counts frames, so readers can skip one they already handled
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error = ""

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        if self._source is None:
            self._cap = cv2.VideoCapture(self.cam)
            if not self._cap.isOpened():
                raise RuntimeError(f"camera {self.cam} did not open (another app may be using it, or camera permission is off)")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        deadline = time.time() + 5
        while self.latest() is None and time.time() < deadline and not self.error:
            time.sleep(0.05)
        if self.latest() is None:
            self.stop()
            raise RuntimeError(self.error or f"camera {self.cam} gave no frames")

    def _run(self) -> None:
        period = 1.0 / self.fps
        while not self._stop.is_set():
            t0 = time.time()
            try:
                if self._source is not None:
                    frame = self._source()
                else:
                    ok, frame = self._cap.read()
                    if not ok:
                        frame = None
                if frame is not None:
                    with self._lock:
                        self._frame = frame
                        self.seq += 1
            except Exception as err:  # keep the viewer alive; report on the next tool call
                self.error = str(err)
            time.sleep(max(0.0, period - (time.time() - t0)))

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        with self._lock:
            self._frame = None
