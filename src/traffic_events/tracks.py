"""Trajectories: raw tracker output -> smoothed kinematics per track."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .detector import CATEGORY_NAMES, COCO_NAMES
from .params import MotionParams, TrackerParams
from .tracker import iou_matrix


def moving_average(x: np.ndarray, t: np.ndarray, window_sec: float) -> np.ndarray:
    """Centred moving average over time (handles irregular sampling)."""
    if len(x) < 3 or window_sec <= 0:
        return x.copy()
    half = window_sec / 2
    lo = np.searchsorted(t, t - half, side="left")
    hi = np.searchsorted(t, t + half, side="right")
    cs = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)])
    return (cs[hi] - cs[lo]) / (hi - lo).reshape((-1,) + (1,) * (x.ndim - 1))


def time_gradient(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    if len(t) < 2:
        return np.zeros_like(x)
    return np.gradient(x, t, axis=0)


@dataclass
class Track:
    id: int
    category: int
    coco_cls: int
    t: np.ndarray                 # (n,) seconds
    box: np.ndarray               # (n, 4) smoothed x1 y1 x2 y2
    raw_box: np.ndarray           # (n, 4) as detected
    anchor: np.ndarray = field(init=False)      # ground contact point (bottom-centre)
    scale: np.ndarray = field(init=False)       # sqrt(w*h), smoothed
    vel: np.ndarray = field(init=False)         # px / s of the anchor
    speed: np.ndarray = field(init=False)       # body units / s
    heading: np.ndarray = field(init=False)     # radians, image coords

    def __post_init__(self):
        b = self.box
        self.anchor = np.column_stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]])
        wh = np.clip(b[:, 2:] - b[:, :2], 1, None)
        self.scale = np.sqrt(wh[:, 0] * wh[:, 1])
        self.vel = time_gradient(self.anchor, self.t)
        self.speed = np.linalg.norm(self.vel, axis=1) / self.scale
        self.heading = np.arctan2(self.vel[:, 1], self.vel[:, 0])

    @property
    def name(self) -> str:
        return COCO_NAMES.get(self.coco_cls, CATEGORY_NAMES[self.category])

    @property
    def start(self) -> float:
        return float(self.t[0])

    @property
    def end(self) -> float:
        return float(self.t[-1])

    def at(self, t: float) -> int | None:
        """Index of the sample nearest to time t, or None outside the track."""
        if t < self.t[0] - 1e-6 or t > self.t[-1] + 1e-6:
            return None
        return int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))

    def unit_dir(self) -> np.ndarray:
        n = np.linalg.norm(self.vel, axis=1, keepdims=True)
        return np.divide(self.vel, n, out=np.zeros_like(self.vel), where=n > 1e-6)


class TrackStore:
    """Accumulates tracker output during the pass; builds Track objects at the end."""

    def __init__(self, motion: MotionParams = MotionParams(), tracker: TrackerParams = TrackerParams()):
        self.motion = motion
        self.tracker = tracker
        self._obs: dict[int, list] = defaultdict(list)
        self._meta: dict[int, tuple[int, int, int]] = {}   # id -> (category, coco_cls, max hits)

    def add(self, t: float, tracked) -> None:
        for tid, box, cat, coco, score, hits in tracked:
            self._obs[tid].append((t, *box))
            _, _, h = self._meta.get(tid, (cat, coco, 0))
            self._meta[tid] = (cat, coco, max(h, hits))

    def build(self) -> list[Track]:
        tracks = []
        for tid, obs in self._obs.items():
            cat, coco, hits = self._meta[tid]
            if hits < self.tracker.min_hits or len(obs) < self.motion.min_track_samples:
                continue
            arr = np.asarray(obs, np.float64)
            t, raw = arr[:, 0], arr[:, 1:5]
            tracks.append(Track(tid, cat, coco, t, moving_average(raw, t, self.motion.smooth_sec), raw))
        return link_stationary_fragments(tracks, self.tracker, self.motion)


def link_stationary_fragments(tracks: list[Track], tp: TrackerParams, mp: MotionParams) -> list[Track]:
    """Join tracks of a parked vehicle that the tracker split during occlusion.

    If track A ends stationary and track B starts stationary later (within
    `stationary_link_gap_sec`) at the same place, B is appended to A.
    """
    tracks = sorted(tracks, key=lambda tr: tr.start)
    merged: dict[int, Track] = {}
    absorbed: set[int] = set()
    for a in tracks:
        if a.id in absorbed:
            continue
        cur = a
        while True:
            if cur.speed[-min(3, len(cur.speed)):].mean() >= mp.stationary_speed:
                break
            best, best_iou = None, tp.stationary_link_iou
            for b in tracks:
                if b.id in absorbed or b.id == cur.id or b.category != cur.category:
                    continue
                gap = b.start - cur.end
                if not (0 < gap <= tp.stationary_link_gap_sec):
                    continue
                if b.speed[:min(3, len(b.speed))].mean() >= mp.stationary_speed:
                    continue
                iou = iou_matrix(cur.raw_box[-1:], b.raw_box[:1])[0, 0]
                if iou > best_iou:
                    best, best_iou = b, iou
            if best is None:
                break
            absorbed.add(best.id)
            t = np.concatenate([cur.t, best.t])
            raw = np.concatenate([cur.raw_box, best.raw_box])
            cur = Track(cur.id, cur.category, cur.coco_cls, t,
                        np.concatenate([cur.box, best.box]), raw)
        merged[cur.id] = cur
    return [merged[k] for k in sorted(merged)]
