"""Sequential video reading with frame skipping."""
from __future__ import annotations

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


def iter_frames(path: str, stride_fn) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, t_sec, bgr_frame) for every `stride_fn()`-th frame.

    `stride_fn` is re-evaluated after every yielded frame, which lets the caller
    raise the stride on the fly when it is running behind its time budget.
    Skipped frames are only grabbed (demuxed/decoded), never converted.
    """
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, idx / fps, frame
            step = max(1, int(stride_fn()))
            for _ in range(step - 1):
                if not cap.grab():
                    return
            idx += step
    finally:
        cap.release()
