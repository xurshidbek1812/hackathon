"""ByteTrack-style multi-object tracker (two-stage IoU association).

Written here instead of using ultralytics' tracker so Part A and Part B can
each own an independent, deterministic tracker while sharing one detector,
and so there is no runtime dependency on the `lap` package.

Algorithm (Zhang et al., ByteTrack, ECCV 2022, simplified):
  1. predict every track with a constant-velocity box model;
  2. match high-confidence detections to all tracks (Hungarian on 1 - IoU);
  3. match low-confidence detections to the tracks still unmatched;
  4. unmatched high detections start tentative tracks, confirmed after
     `min_hits` updates; tracks unseen for `max_lost_sec` are dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .params import TrackerParams


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-6)


@dataclass
class _Track:
    id: int
    box: np.ndarray
    category: int
    coco_cls: int
    score: float
    last_t: float = 0.0
    vel: np.ndarray = field(default_factory=lambda: np.zeros(4, np.float32))   # px / s per box coord
    hits: int = 1
    lost: int = 0
    cls_votes: dict = field(default_factory=dict)

    def predict(self, t: float) -> np.ndarray:
        return self.box + self.vel * (t - self.last_t)

    def update(self, det: np.ndarray, t: float) -> None:
        new_box = det[:4].astype(np.float32)
        dt = max(t - self.last_t, 1e-3)
        inst_vel = (new_box - self.box) / dt
        self.vel = 0.6 * self.vel + 0.4 * inst_vel if self.hits > 1 else inst_vel
        self.box = new_box
        self.last_t = t
        self.score = float(det[4])
        self.hits += 1
        self.lost = 0
        key = int(det[6])
        self.cls_votes[key] = self.cls_votes.get(key, 0.0) + float(det[4])
        self.coco_cls = max(self.cls_votes, key=self.cls_votes.get)


class ByteTracker:
    def __init__(self, params: TrackerParams = TrackerParams()):
        self.p = params
        self.tracks: list[_Track] = []
        self._next_id = 1

    def _associate(self, tracks: list[_Track], dets: np.ndarray, t: float):
        if not tracks or len(dets) == 0:
            return [], list(range(len(tracks))), list(range(len(dets)))
        pred = np.stack([tr.predict(t) for tr in tracks])
        iou = iou_matrix(pred, dets[:, :4])
        # pedestrians never turn into cars: forbid cross-category matches
        # (cars and motorbikes may swap labels: both are category <= BIKE)
        tc = np.array([tr.category for tr in tracks])[:, None]
        dc = dets[None, :, 5].astype(int)
        same = (tc == dc) | ((tc <= 1) & (dc <= 1))
        cost = np.where(same, 1.0 - iou, 2.0)
        rows, cols = linear_sum_assignment(cost)
        matches, um_t, um_d = [], set(range(len(tracks))), set(range(len(dets)))
        for r, c in zip(rows, cols):
            if cost[r, c] <= 1.0 - self.p.iou_gate:
                matches.append((r, c))
                um_t.discard(r)
                um_d.discard(c)
        return matches, sorted(um_t), sorted(um_d)

    def update(self, dets: np.ndarray, t: float) -> list[tuple[int, np.ndarray, int, int, float, int]]:
        """dets: (N, 7) rows from Detector for the frame at time t. Returns every track updated this step as
        (track_id, box, category, coco_cls, score, hits). Tentative tracks are
        included so a track's history starts at its first detection; callers
        drop tracks that never reach `min_hits`."""
        dets = np.asarray(dets, np.float32).reshape(-1, 7)
        high = dets[dets[:, 4] >= self.p.high_conf]
        low = dets[(dets[:, 4] >= self.p.low_conf) & (dets[:, 4] < self.p.high_conf)]

        matches, um_t, um_d = self._associate(self.tracks, high, t)
        for ti, di in matches:
            self.tracks[ti].update(high[di], t)
        remaining = [self.tracks[i] for i in um_t]
        matches2, um_t2, _ = self._associate(remaining, low, t)
        for ti, di in matches2:
            remaining[ti].update(low[di], t)
        for i in um_t2:
            remaining[i].lost += 1

        for di in um_d:
            d = high[di]
            self.tracks.append(_Track(self._next_id, d[:4].copy(), int(d[5]), int(d[6]), float(d[4]),
                                      last_t=t, cls_votes={int(d[6]): float(d[4])}))
            self._next_id += 1
        self.tracks = [tr for tr in self.tracks
                       if t - tr.last_t <= self.p.max_lost_sec and not (tr.hits < self.p.min_hits and tr.lost > 0)]
        return [(tr.id, tr.box.copy(), tr.category, tr.coco_cls, tr.score, tr.hits)
                for tr in self.tracks if tr.lost == 0]
