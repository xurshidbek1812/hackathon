"""Video reading: frame skipping, downscaling, and decode-ahead in a thread."""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMeta:
    path: str
    fps: float
    width: int
    height: int
    n_frames: int

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0


def read_meta(path: str) -> VideoMeta:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    meta = VideoMeta(str(path), float(fps), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return meta


def resize_to_width(frame: np.ndarray, width: int | None) -> np.ndarray:
    if not width or frame.shape[1] <= width:
        return frame
    h = int(round(frame.shape[0] * width / frame.shape[1]))
    return cv2.resize(frame, (width, h), interpolation=cv2.INTER_AREA)


def iter_frames(path: str, stride_fn, width: int | None = None,
                prefetch: int = 32) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, t_sec, bgr_frame) for every `stride_fn()`-th frame.

    Decoding (the bottleneck for 4K H.264) and downscaling to `width` run in a
    background thread, so they overlap with inference in the caller.
    `stride_fn` is re-evaluated after every frame, which lets the caller raise
    the stride when it is running behind its time budget. Skipped frames are
    only grabbed, never converted.
    """
    q: queue.Queue = queue.Queue(maxsize=prefetch)
    stop = threading.Event()
    done = object()

    def reader():
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        idx = 0
        try:
            while not stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                q.put((idx, idx / fps, resize_to_width(frame, width)))
                step = max(1, int(stride_fn()))
                if not all(cap.grab() for _ in range(step - 1)):
                    break
                idx += step
        finally:
            cap.release()
            q.put(done)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    try:
        while (item := q.get()) is not done:
            yield item
    finally:
        stop.set()
        while thread.is_alive():          # unblock a reader waiting on a full queue
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            thread.join(timeout=0.05)
