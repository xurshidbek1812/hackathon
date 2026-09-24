"""Shared context and geometry helpers for the rule modules."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..detector import BIKE, OBSTACLE, PERSON, VEHICLE
from ..flow import FlowField
from ..frame_features import FrameFeatures
from ..params import MotionParams, RuleParams
from ..scene import Scene
from ..signal_state import SignalTimeline
from ..tracks import Track
from ..video import VideoMeta


@dataclass
class Context:
    meta: VideoMeta
    scene: Scene
    tracks: list[Track]
    flow: FlowField
    signals: dict[str, SignalTimeline]
    features: FrameFeatures | None
    rp: RuleParams
    mp: MotionParams
    cache: dict = field(default_factory=dict)     # shared intermediate results

    @property
    def duration(self) -> float:
        return self.meta.duration

    def of(self, *categories: int) -> list[Track]:
        return [t for t in self.tracks if t.category in categories]

    @property
    def vehicles(self) -> list[Track]:
        return self.of(VEHICLE, BIKE)

    @property
    def cars(self) -> list[Track]:
        return self.of(VEHICLE)

    @property
    def persons(self) -> list[Track]:
        return self.of(PERSON)

    @property
    def obstacles(self) -> list[Track]:
        return self.of(OBSTACLE)

    def track_end(self, tr: Track) -> float:
        """End of a track; a track still alive in the last second runs to the video end."""
        return self.duration if tr.end >= self.duration - 1.0 else tr.end


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive index ranges of True runs."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        return []
    idx = np.flatnonzero(mask)
    br = np.flatnonzero(np.diff(idx) > 1)
    return list(zip(np.r_[idx[0], idx[br + 1]], np.r_[idx[br], idx[-1]]))


def fill_short_gaps(t: np.ndarray, mask: np.ndarray, max_gap: float) -> np.ndarray:
    """Set False runs shorter than `max_gap` seconds (between Trues) to True."""
    out = np.asarray(mask, bool).copy()
    for a, b in runs(~out):
        if a > 0 and b < len(out) - 1 and t[b + 1] - t[a - 1] <= max_gap:
            out[a:b + 1] = True
    return out


def signed_side(polyline: np.ndarray, pts: np.ndarray, extent_margin: float = 0.05) -> np.ndarray:
    """Side of each point w.r.t. a polyline: +1 / -1, or 0 where the point does
    not project onto any segment (beyond the line's ends)."""
    pts = np.atleast_2d(pts).astype(np.float64)
    best_d = np.full(len(pts), np.inf)
    side = np.zeros(len(pts))
    for a, b in zip(polyline[:-1], polyline[1:]):
        ab = b - a
        L2 = float(ab @ ab)
        if L2 == 0:
            continue
        u = ((pts - a) @ ab) / L2
        ok = (u >= -extent_margin) & (u <= 1 + extent_margin)
        proj = a + np.clip(u, 0, 1)[:, None] * ab
        d = np.linalg.norm(pts - proj, axis=1)
        cross = ab[0] * (pts[:, 1] - a[1]) - ab[1] * (pts[:, 0] - a[0])
        better = ok & (d < best_d)
        best_d[better] = d[better]
        side[better] = np.sign(cross[better])
    return side


def front_point(box: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Approximate ground point of the vehicle's front for travel `direction`."""
    box = np.atleast_2d(box)
    w, h = box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]
    cx = (box[:, 0] + box[:, 2]) / 2 + direction[0] * w / 2
    cy = box[:, 3] - np.clip(-direction[1], 0, None) * h * 0.6
    return np.column_stack([cx, cy])


def heading_change_window(tr: Track, i0: int, i1: int, tol_deg: float = 15.0) -> tuple[float, float]:
    """Refine a turn to [start, end]: first deviation from the entry heading to
    the first sample that settles on the exit heading."""
    moving = tr.speed > 0.3
    idx = np.flatnonzero(moving[i0:i1 + 1]) + i0
    if len(idx) < 3:
        return float(tr.t[i0]), float(tr.t[i1])
    h = np.unwrap(tr.heading[idx])
    tol = np.deg2rad(tol_deg)
    dev_start = np.flatnonzero(np.abs(h - h[0]) > tol)
    s = idx[dev_start[0] - 1] if len(dev_start) and dev_start[0] > 0 else idx[0]
    settle = np.flatnonzero(np.abs(h - h[-1]) <= tol)
    e = idx[settle[0]] if len(settle) else idx[-1]
    return float(tr.t[s]), float(max(tr.t[e], tr.t[s] + 0.5))


def body_distance(a: Track, ia: np.ndarray, b: Track, ib: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(a.anchor[ia] - b.anchor[ib], axis=1)
    return d / (0.5 * (a.scale[ia] + b.scale[ib]))


def common_times(a: Track, b: Track) -> tuple[np.ndarray, np.ndarray]:
    """Indices into a and b of samples taken at the same frame."""
    ta, tb = np.round(a.t * 1000).astype(np.int64), np.round(b.t * 1000).astype(np.int64)
    _, ia, ib = np.intersect1d(ta, tb, assume_unique=True, return_indices=True)
    return ia, ib
