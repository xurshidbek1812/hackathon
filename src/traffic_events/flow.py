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
    """Grid of travel directions, kept in the coordinates of the reference view
    (see registration.py) so videos recorded with a slightly different framing
    share one prior. `view` maps this video's frame to the reference view."""

    def __init__(self, width: int, height: int, view=None):
        self.width, self.height = width, height
        self.view = view
        self.sum = np.zeros((GRID_H, GRID_W, 2), np.float64)   # sum of unit vectors
        self.count = np.zeros((GRID_H, GRID_W), np.float64)    # moving samples
        self.occupancy = np.zeros((GRID_H, GRID_W), np.float64)  # any vehicle sample

    # ---------------------------------------------------------------- coordinates
    def _to_ref(self, pts: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(pts, np.float64))
        if self.view is None:
            return pts
        size = np.array([self.width, self.height], float)
        return self.view.to_reference(pts / size) * size

    def _dirs_to_ref(self, pts: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        if self.view is None:
            return dirs
        d = self._to_ref(pts + 5.0 * dirs) - self._to_ref(pts)
        return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)

    def _dirs_from_ref(self, ref_pts: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        if self.view is None:
            return dirs
        size = np.array([self.width, self.height], float)
        a = self.view.to_view(ref_pts / size) * size
        b = self.view.to_view((ref_pts + 5.0 * np.nan_to_num(dirs)) / size) * size
        d = (b - a) / np.maximum(np.linalg.norm(b - a, axis=1, keepdims=True), 1e-9)
        return np.where(np.isnan(dirs), np.nan, d)

    def _cells(self, ref_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gx = np.clip((ref_pts[:, 0] / self.width * GRID_W).astype(int), 0, GRID_W - 1)
        gy = np.clip((ref_pts[:, 1] / self.height * GRID_H).astype(int), 0, GRID_H - 1)
        return gy, gx

    # ---------------------------------------------------------------- learning
    def add_tracks(self, tracks, moving_speed: float = 0.5) -> None:
        for tr in tracks:
            if tr.category not in (VEHICLE, BIKE):
                continue
            gy, gx = self._cells(self._to_ref(tr.anchor))
            np.add.at(self.occupancy, (gy, gx), 1.0)
            m = tr.speed > moving_speed
            if m.sum() < 3:
                continue
            u = self._dirs_to_ref(tr.anchor[m], tr.unit_dir()[m])
            np.add.at(self.sum, (gy[m], gx[m]), u)
            np.add.at(self.count, (gy[m], gx[m]), 1.0)

    def merge(self, other: "FlowField", weight: float = 1.0) -> None:
        self.sum += weight * other.sum
        self.count += weight * other.count
        self.occupancy += weight * other.occupancy

    # ------------------------------------------------------------------ queries
    def direction_grid(self, min_count: float = 8, min_consistency: float = 0.8):
        """(GRID_H, GRID_W, 2) unit directions in the reference view, NaN where unknown."""
        s = cv2.GaussianBlur(self.sum, (3, 3), 0.7)
        c = cv2.GaussianBlur(self.count, (3, 3), 0.7)
        norm = np.linalg.norm(s, axis=2)
        consistency = norm / np.maximum(c, 1e-6)
        ok = (c >= min_count) & (consistency >= min_consistency)
        d = np.full_like(s, np.nan)
        d[ok] = s[ok] / norm[ok, None]
        return d

    def expected_direction(self, pts: np.ndarray) -> np.ndarray:
        """Dominant direction at each point of this video's frame (NaN rows where unknown)."""
        ref = self._to_ref(pts)
        gy, gx = self._cells(ref)
        return self._dirs_from_ref(ref, self.direction_grid()[gy, gx])

    def road_mask(self, min_frac: float = 0.002) -> np.ndarray:
        """Cells vehicles actually drive through (reference view), closed and slightly
        dilated. Only a fallback for scenes whose config has no road polygon."""
        occ = self.occupancy / max(self.occupancy.sum(), 1.0)
        m = (occ > min_frac / 10).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        return cv2.dilate(m, np.ones((3, 3), np.uint8))

    @property
    def total(self) -> float:
        return float(self.count.sum())

    # ---------------------------------------------------------------- persistence
    def save(self, path: Path = PRIOR_PATH) -> None:
        np.savez_compressed(path, sum=self.sum, count=self.count, occupancy=self.occupancy,
                            size=np.array([self.width, self.height]))

    @classmethod
    def load_prior(cls, width: int, height: int, path: Path = PRIOR_PATH, view=None) -> "FlowField | None":
        if not Path(path).exists():
            return None
        z = np.load(path)
        f = cls(width, height, view)        # the grid is resolution independent
        f.sum, f.count, f.occupancy = z["sum"], z["count"], z["occupancy"]
        return f


def build_flow(tracks, width: int, height: int, prior_weight: float = 0.5, view=None) -> FlowField:
    field = FlowField(width, height, view)
    field.add_tracks(tracks)
    prior = FlowField.load_prior(width, height, view=view)
    if prior is not None:
        # keep the current video from being drowned out by hours of prior data
        w = prior_weight * max(field.total, 1000.0) / max(prior.total, 1.0)
        field.merge(prior, weight=min(w, 1.0))
    return field
