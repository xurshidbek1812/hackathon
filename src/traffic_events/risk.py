"""Part B: causal accident-risk score from the frames seen so far.

Every `stride`-th frame is detected and tracked; the other frames return the
last score. Detection of a sampled frame runs in a worker thread while the
harness decodes the next frames; its result is folded in at the next sampled
frame. The score therefore lags by exactly one sample (0.1 s) - always the
same lag, so the output is deterministic and uses past frames only. The score is a logistic combination of interpretable cues:

  ttc      closest predicted approach between two road users (time-to-collision
           and miss distance, in body units)
  brake    hardest deceleration in the last second
  swerve   sharpest heading change of a moving vehicle in the last second
  wrong    a vehicle moving against the learned lane direction
  ped      a pedestrian on the carriageway next to moving traffic

Only scene knowledge (configs/*.json, the flow prior built from the sample
videos) and past frames are used, so the estimator is causal.
"""
from __future__ import annotations

import math
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .detector import BIKE, PERSON, VEHICLE, get_detector
from .flow import GRID_H, GRID_W, FlowField
from .params import RiskParams
from .scene import load_scene
from .tracker import ByteTracker
from .video import resize_to_width


class _History:
    __slots__ = ("t", "anchor", "scale", "category", "hits")

    def __init__(self, maxlen: int):
        self.t, self.anchor, self.scale = deque(maxlen=maxlen), deque(maxlen=maxlen), deque(maxlen=maxlen)
        self.category = VEHICLE
        self.hits = 0

    def add(self, t: float, box: np.ndarray, category: int, hits: int) -> None:
        self.t.append(t)
        self.anchor.append(((box[0] + box[2]) / 2, box[3]))
        self.scale.append(math.sqrt(max(box[2] - box[0], 1) * max(box[3] - box[1], 1)))
        self.category, self.hits = category, hits

    def velocity(self, t0: float, t1: float) -> np.ndarray | None:
        """Least-squares velocity (px/s) of the anchor over [t0, t1]."""
        t = np.asarray(self.t)
        m = (t >= t0) & (t <= t1)
        if m.sum() < 3:
            return None
        a = np.asarray(self.anchor)[m]
        tt = t[m] - t[m].mean()
        denom = float(tt @ tt)
        if denom <= 0:
            return None
        return (tt @ (a - a.mean(axis=0))) / denom


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


class RiskModel:
    def __init__(self, params: RiskParams = RiskParams(), scene_path=None):
        self.p = params
        self.scene_path = scene_path

    def reset(self, meta: dict) -> None:
        self.fps = float(meta.get("fps") or 25.0)
        self.width, self.height = int(meta["width"]), int(meta["height"])
        self.duration = float(meta.get("n_frames") or 0) / self.fps
        self.scene = load_scene(self.width, self.height, self.scene_path)
        self.flow = FlowField.load_prior(self.width, self.height)
        self.flow_dirs = self.flow.direction_grid() if self.flow is not None else None
        if not self.scene.has_road and self.flow is not None:
            self.scene.set_learned_road_mask(self.flow.road_mask())
        self.detector = get_detector()
        self.work_w = min(self.width, self.p.imgsz)
        self.scale = self.width / self.work_w
        self.worker = getattr(self, "worker", None) or ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.tracker = ByteTracker()
        self.stride = max(1, round(self.fps / self.p.sample_hz))
        self.hist: dict[int, _History] = {}
        self.frame_i = 0
        self.score = 0.0
        self.last_t = 0.0
        self.features: dict[str, float] = {}
        self._t0 = time.perf_counter()

    # ------------------------------------------------------------------ stepping
    def step(self, frame: np.ndarray, t_sec: float) -> float:
        i = self.frame_i
        self.frame_i += 1
        if i % self.stride:
            return self.score
        self._check_budget(t_sec)
        if self.pending is not None:
            t_prev, job = self.pending
            self._update(job.result(), t_prev)
        self.pending = (t_sec, self.worker.submit(self._detect, frame))
        return self.score

    def _detect(self, frame: np.ndarray) -> np.ndarray:
        dets = self.detector([resize_to_width(frame, self.work_w)], imgsz=self.p.imgsz)[0]
        dets[:, :4] *= self.scale
        return dets

    def _update(self, dets: np.ndarray, t_sec: float) -> None:
        maxlen = int(self.p.history_sec * self.fps / self.stride) + 2
        for tid, box, cat, _, _, hits in self.tracker.update(dets, t_sec):
            self.hist.setdefault(tid, _History(maxlen)).add(t_sec, box, cat, hits)
        self.hist = {k: h for k, h in self.hist.items() if t_sec - h.t[-1] <= 1.0}
        self.features = self._features(t_sec)
        p = self.p
        z = (p.bias + p.w_ttc * self.features["ttc"] + p.w_brake * self.features["brake"]
             + p.w_swerve * self.features["swerve"] + p.w_wrong_way * self.features["wrong"]
             + p.w_ped_road * self.features["ped"])
        target = _sigmoid(z)
        dt = max(t_sec - self.last_t, 1e-3)
        decayed = self.score * math.exp(-p.decay_per_sec * dt)
        self.score = float(np.clip(decayed + p.ema * (target - decayed) if target > decayed else decayed, 0, 1))
        self.last_t = t_sec

    def _check_budget(self, t_sec: float) -> None:
        """Emergency frame skipping on machines far slower than the target GPU."""
        if t_sec < 10 or self.stride >= self.p.max_stride:
            return
        elapsed = time.perf_counter() - self._t0
        if elapsed / t_sec > 2 * self.p.budget_ratio:
            self.stride += 1

    # ------------------------------------------------------------------ features
    def _features(self, t: float) -> dict[str, float]:
        min_scale = self.p.min_scale * self.height
        users = []
        for h in self.hist.values():
            if h.hits < 3 or t - h.t[-1] > 1e-6 or h.scale[-1] < min_scale:
                continue            # far-away road users: boxes overlap by perspective, motion is noise
            v = h.velocity(t - 0.6, t)
            if v is None:
                continue
            users.append((h, np.asarray(h.anchor[-1]), v, h.scale[-1]))
        return {
            "ttc": self._ttc(users),
            "brake": self._brake(users, t),
            "swerve": self._swerve(users, t),
            "wrong": self._wrong_way(users),
            "ped": self._ped_on_road(users),
        }

    @staticmethod
    def _ttc(users) -> float:
        """Closest predicted approach between two road users on crossing paths."""
        best = 0.0
        for i in range(len(users)):
            hi, pi, vi, si = users[i]
            for j in range(i + 1, len(users)):
                hj, pj, vj, sj = users[j]
                if hi.category == PERSON and hj.category == PERSON:
                    continue
                ni, nj = np.linalg.norm(vi) / si, np.linalg.norm(vj) / sj
                if ni < 1.0 or nj < 1.0:
                    continue        # rolling up to a standing queue closes the gap too
                cos = float(vi @ vj) / (np.linalg.norm(vi) * np.linalg.norm(vj))
                min_angle = 30 if PERSON in (hi.category, hj.category) else 45
                if abs(cos) > math.cos(math.radians(min_angle)):
                    continue        # following, merging, or passing on the opposite carriageway
                scale = 0.5 * (si + sj)
                rel_p, rel_v = pj - pi, vj - vi
                if np.linalg.norm(rel_p) / scale > 8:
                    continue
                v2 = float(rel_v @ rel_v)
                ttc = -float(rel_p @ rel_v) / v2
                if not 0 < ttc < 5:
                    continue
                miss = float(np.linalg.norm(rel_p + rel_v * ttc)) / scale
                best = max(best, math.exp(-ttc / 1.5) * max(0.0, 1.0 - miss))
        return best

    @staticmethod
    def _brake(users, t: float) -> float:
        """Hardest deceleration from real speed (body units/s lost in ~1 s, scaled to [0, 1])."""
        best = 0.0
        for h, _, v_now, s in users:
            if h.category not in (VEHICLE, BIKE):
                continue
            v_before = h.velocity(t - 1.6, t - 0.8)
            if v_before is None:
                continue
            before, now = np.linalg.norm(v_before) / s, np.linalg.norm(v_now) / s
            if before > 2.0:
                best = max(best, float(np.clip((before - now - 1.0) / 2.0, 0.0, 1.0)))
        return best

    @staticmethod
    def _swerve(users, t: float) -> float:
        best = 0.0
        for h, _, v_now, s in users:
            if h.category not in (VEHICLE, BIKE) or np.linalg.norm(v_now) / s < 1.5:
                continue
            v_before = h.velocity(t - 1.2, t - 0.6)
            if v_before is None or np.linalg.norm(v_before) / s < 1.5:
                continue
            cos = float(v_now @ v_before) / (np.linalg.norm(v_now) * np.linalg.norm(v_before))
            deg = math.degrees(math.acos(np.clip(cos, -1, 1)))
            best = max(best, float(np.clip((deg - 15.0) / 30.0, 0.0, 1.0)))
        return best

    def _wrong_way(self, users) -> float:
        if self.flow is None:
            return 0.0
        for h, p, v, s in users:
            if h.category not in (VEHICLE, BIKE) or np.linalg.norm(v) / s < 1.0:
                continue
            if self.scene.intersection is not None and self.scene.inside(self.scene.intersection, p[None])[0]:
                continue
            gx = min(int(p[0] / self.width * GRID_W), GRID_W - 1)
            gy = min(int(p[1] / self.height * GRID_H), GRID_H - 1)
            d = self.flow_dirs[gy, gx]
            if not np.isnan(d[0]) and float(v @ d) / np.linalg.norm(v) < -0.5:
                return 1.0
        return 0.0

    def _ped_on_road(self, users) -> float:
        """A pedestrian on the carriageway outside any crossing, next to moving traffic."""
        if self.scene.crossings is None or not self.scene.has_road:
            return 0.0
        movers = [(p, s) for h, p, v, s in users
                  if h.category in (VEHICLE, BIKE) and np.linalg.norm(v) / s > 1.0]
        for h, p, _, s in users:
            if h.category != PERSON:
                continue
            height = s * 1.6                      # a person box is ~2.5x taller than wide: h ~ 1.6 sqrt(w*h)
            if not self.scene.on_road(p[None], margin=0.5 * height)[0] or                     self.scene.in_crossing(p[None], 0.6 * height)[0]:
                continue
            for q, sq in movers:
                if np.linalg.norm(p - q) / sq < 3:
                    return 1.0
        return 0.0
