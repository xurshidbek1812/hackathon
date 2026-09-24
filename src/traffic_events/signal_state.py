"""Traffic-signal state from fixed image regions of the signal head.

The config gives, per signal, either separate lamp boxes {"red": box, "green": box}
or one {"head": box}; boxes are normalised [x1, y1, x2, y2].
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import median_filter

RED, GREEN, UNKNOWN = 1, 0, -1


def _crop(frame: np.ndarray, box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(int)
    return frame[max(y1, 0):max(y2, y1 + 1), max(x1, 0):max(x2, x1 + 1)]


def _lit_fraction(hsv: np.ndarray, hue_ranges) -> float:
    if hsv.size == 0:
        return 0.0
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hue_ok = np.zeros(h.shape, bool)
    for lo, hi in hue_ranges:
        hue_ok |= (h >= lo) & (h <= hi)
    return float(((v > 150) & (s > 70) & hue_ok).mean())


RED_HUES = [(0, 10), (165, 180)]
GREEN_HUES = [(45, 95)]


def classify_signal(frame: np.ndarray, lamps: dict[str, np.ndarray], min_lit: float = 0.04) -> int:
    if "head" in lamps:
        hsv = cv2.cvtColor(_crop(frame, lamps["head"]), cv2.COLOR_BGR2HSV)
        red, green = _lit_fraction(hsv, RED_HUES), _lit_fraction(hsv, GREEN_HUES)
    else:
        red = _lit_fraction(cv2.cvtColor(_crop(frame, lamps["red"]), cv2.COLOR_BGR2HSV), RED_HUES) \
            if "red" in lamps else 0.0
        green = _lit_fraction(cv2.cvtColor(_crop(frame, lamps["green"]), cv2.COLOR_BGR2HSV), GREEN_HUES) \
            if "green" in lamps else 0.0
    if max(red, green) < min_lit:
        return UNKNOWN
    return RED if red > green else GREEN


@dataclass
class SignalTimeline:
    t: np.ndarray
    state: np.ndarray

    def state_at(self, t: float) -> int:
        if len(self.t) == 0:
            return UNKNOWN
        i = int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))
        return int(self.state[i])

    def next_change_to(self, t: float, target: int) -> float | None:
        idx = np.flatnonzero((self.t > t) & (self.state == target))
        return float(self.t[idx[0]]) if len(idx) else None


class SignalMonitor:
    def __init__(self, signals: dict[str, dict[str, np.ndarray]], scale: float = 1.0):
        """`signals` boxes are in video pixels; frames arrive downscaled by `scale`."""
        self.signals = {name: {lamp: box / scale for lamp, box in lamps.items()}
                        for name, lamps in signals.items()}
        self._t: list[float] = []
        self._raw: dict[str, list[int]] = {k: [] for k in signals}

    def update(self, t: float, frame: np.ndarray) -> None:
        if not self.signals:
            return
        self._t.append(t)
        for name, lamps in self.signals.items():
            self._raw[name].append(classify_signal(frame, lamps))

    def timelines(self, fps_hint: float = 12.5) -> dict[str, SignalTimeline]:
        t = np.asarray(self._t)
        k = max(3, int(fps_hint) | 1)              # ~1 s median filter kills blinks
        return {name: SignalTimeline(t, median_filter(np.asarray(v), size=k, mode="nearest"))
                for name, v in self._raw.items()}
