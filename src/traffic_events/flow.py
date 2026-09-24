"""Learned traffic-flow field: the dominant travel direction per grid cell.

Built from moving vehicle trajectories. A prior built from all sample videos
(scripts/build_scene_prior.py -> configs/flow_prior.npz) is combined with the
current video, so the field is dense even for a quiet test clip. Cells where
traffic goes both ways (intersections, lane boundaries) have low consistency
and are reported as "unknown" instead of guessing a direction.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .detector import BIKE, VEHICLE
from .params import CONFIG_DIR

GRID_W, GRID_H = 48, 27
PRIOR_PATH = CONFIG_DIR / "flow_prior.npz"


class FlowField:
    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.sum = np.zeros((GRID_H, GRID_W, 2), np.float64)   # sum of unit vectors
        self.count = np.zeros((GRID_H, GRID_W), np.float64)    # moving samples
        self.occupancy = np.zeros((GRID_H, GRID_W), np.float64)  # any vehicle sample
        self.stops = np.zeros((GRID_H, GRID_W), np.float64)      # distinct stops >= 3 s

    def _cells(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gx = np.clip((pts[:, 0] / self.width * GRID_W).astype(int), 0, GRID_W - 1)
        gy = np.clip((pts[:, 1] / self.height * GRID_H).astype(int), 0, GRID_H - 1)
        return gy, gx

    def add_tracks(self, tracks, moving_speed: float = 0.5, stationary_speed: float = 0.15) -> None:
        for tr in tracks:
            if tr.category not in (VEHICLE, BIKE):
                continue
            gy, gx = self._cells(tr.anchor)
            np.add.at(self.occupancy, (gy, gx), 1.0)
            self._add_stops(tr, stationary_speed)
            m = tr.speed > moving_speed
            if m.sum() < 3:
                continue
            u = tr.unit_dir()[m]
            np.add.at(self.sum, (gy[m], gx[m]), u)
            np.add.at(self.count, (gy[m], gx[m]), 1.0)

    def _add_stops(self, tr, stationary_speed: float, min_sec: float = 3.0) -> None:
        still = np.r_[False, tr.speed < stationary_speed, False].astype(int)
        edges = np.flatnonzero(np.diff(still))
        for a, b in zip(edges[::2], edges[1::2] - 1):
            if tr.t[b] - tr.t[a] >= min_sec:
                gy, gx = self._cells(np.median(tr.anchor[a:b + 1], axis=0, keepdims=True))
                self.stops[gy, gx] += 1.0

    def merge(self, other: "FlowField", weight: float = 1.0) -> None:
        self.sum += weight * other.sum
        self.count += weight * other.count
        self.occupancy += weight * other.occupancy
        self.stops += weight * other.stops

    # ------------------------------------------------------------------ queries
    def direction_grid(self, min_count: float = 8, min_consistency: float = 0.8):
        """(GRID_H, GRID_W, 2) unit directions, NaN where unknown."""
        s = cv2.GaussianBlur(self.sum, (3, 3), 0.7)
        c = cv2.GaussianBlur(self.count, (3, 3), 0.7)
        norm = np.linalg.norm(s, axis=2)
        consistency = norm / np.maximum(c, 1e-6)
        ok = (c >= min_count) & (consistency >= min_consistency)
        d = np.full_like(s, np.nan)
        d[ok] = s[ok] / norm[ok, None]
        return d

    def expected_direction(self, pts: np.ndarray) -> np.ndarray:
        """Dominant direction at each point, NaN rows where unknown."""
        grid = self.direction_grid()
        gy, gx = self._cells(np.atleast_2d(pts))
        return grid[gy, gx]

    def road_mask(self, min_frac: float = 0.002) -> np.ndarray:
        """Cells vehicles actually drive through, closed and slightly dilated."""
        occ = self.occupancy / max(self.occupancy.sum(), 1.0)
        m = (occ > min_frac / 10).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        return cv2.dilate(m, np.ones((3, 3), np.uint8))

    def is_queue_zone(self, pts: np.ndarray, min_stops: float = 3.0) -> np.ndarray:
        """Where many different vehicles routinely stop (signal queues)."""
        grid = cv2.dilate(cv2.GaussianBlur(self.stops, (3, 3), 0.7), np.ones((3, 3), np.uint8))
        gy, gx = self._cells(np.atleast_2d(pts))
        return grid[gy, gx] >= min_stops

    @property
    def total(self) -> float:
        return float(self.count.sum())

    # ---------------------------------------------------------------- persistence
    def save(self, path: Path = PRIOR_PATH) -> None:
        np.savez_compressed(path, sum=self.sum, count=self.count, occupancy=self.occupancy,
                            stops=self.stops, size=np.array([self.width, self.height]))

    @classmethod
    def load_prior(cls, width: int, height: int, path: Path = PRIOR_PATH) -> "FlowField | None":
        if not Path(path).exists():
            return None
        z = np.load(path)
        f = cls(width, height)          # grid is resolution independent
        f.sum, f.count, f.occupancy, f.stops = z["sum"], z["count"], z["occupancy"], z["stops"]
        return f


def build_flow(tracks, width: int, height: int, prior_weight: float = 0.5) -> FlowField:
    field = FlowField(width, height)
    field.add_tracks(tracks)
    prior = FlowField.load_prior(width, height)
    if prior is not None:
        # keep the current video from being drowned out by hours of prior data
        w = prior_weight * max(field.total, 1000.0) / max(prior.total, 1.0)
        field.merge(prior, weight=min(w, 1.0))
    return field
