"""Team Infinity - WIUT Hackathon 2026, CV track.

Implements the competition interface. All logic lives in src/traffic_events.
"""
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from traffic_events import CLASSES  # noqa: E402
from traffic_events.pipeline import analyze  # noqa: E402
from traffic_events.risk import RiskModel  # noqa: E402

__all__ = ["CLASSES", "detect_events", "RiskEstimator"]

random.seed(0)
np.random.seed(0)


def detect_events(video_path: str) -> list[list]:
    """Part A. Return [[start_sec, end_sec, label], ...] for one .mp4.
    start_sec, end_sec: float seconds from the first frame; label: one of CLASSES.
    """
    return analyze(video_path).event_lists()


class RiskEstimator:
    """Part B. Causal: step() sees frames in order and nothing else."""

    def __init__(self):
        self._model = RiskModel()

    def reset(self, meta: dict) -> None:
        # meta = {"video_id", "fps", "width", "height", "n_frames"}
        self._model.reset(meta)

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        # frame: BGR uint8 (H, W, 3). Return P(accident starts within 5 s) in [0, 1].
        return self._model.step(frame, t_sec)
