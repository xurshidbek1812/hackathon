"""Part A: one pass over the video, then rules on the collected trajectories."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .detector import CATEGORY_NAMES, get_detector
from .flow import FlowField, build_flow
from .frame_features import FrameFeatureCollector, FrameFeatures
from .params import PipelineParams
from .rules import Context, run_rules
from .registration import ViewTransform, estimate_view_from_video
from .scene import load_scene
from .segments import Event, finalize
from .signal_state import SignalMonitor, SignalTimeline
from .tracker import ByteTracker
from .tracks import Track, TrackStore
from .video import VideoMeta, iter_frames, read_meta

Progress = Callable[[float, str], None]


@dataclass
class Analysis:
    meta: VideoMeta
    events: list[Event]
    tracks: list[Track]
    flow: FlowField
    signals: dict[str, SignalTimeline]
    features: FrameFeatures | None
    stride: int
    view: ViewTransform                  # this video's framing vs. the reference view
    seconds: float                       # wall-clock time of the analysis

    def event_lists(self) -> list[list]:
        return [e.as_list() for e in self.events]

    def counts_per_second(self) -> dict[str, list[int]]:
        """Distinct tracked objects per second, by category (for EDA / dashboards)."""
        n = max(1, int(np.ceil(self.meta.duration)))
        out = {name: np.zeros(n, int) for name in CATEGORY_NAMES.values()}
        for tr in self.tracks:
            secs = np.unique(np.clip(tr.t.astype(int), 0, n - 1))
            out[CATEGORY_NAMES[tr.category]][secs] += 1
        return {k: v.tolist() for k, v in out.items()}


class _Budget:
    """Emergency frame-skipping for machines far slower than the target GPU.

    The stride is only raised when the projected wall time is more than twice
    the budget, so on the evaluation machine the run stays deterministic.
    """

    def __init__(self, duration: float, ratio: float, stride: int, max_stride: int):
        self.limit = duration * ratio
        self.stride = stride
        self.max_stride = max_stride
        self.t0 = time.perf_counter()

    def check(self, video_t: float, duration: float) -> None:
        if video_t < 10 or duration <= 0:
            return
        elapsed = time.perf_counter() - self.t0
        projected = elapsed / video_t * duration
        if projected > 2 * self.limit and self.stride < self.max_stride:
            self.stride += 1


def analyze(video_path: str, params: PipelineParams = PipelineParams(), scene_path=None,
            progress: Progress | None = None) -> Analysis:
    t_start = time.perf_counter()
    meta = read_meta(video_path)
    view = estimate_view_from_video(video_path)      # the camera is nudged between recordings
    scene = load_scene(meta.width, meta.height, scene_path, view)
    detector = get_detector(params.detector)
    base_stride = max(1, round(meta.fps / params.sample_hz))
    budget = _Budget(meta.duration, params.budget_ratio_part_a, base_stride, params.max_stride)

    # Frames are decoded and shrunk to the detector's input width in a reader
    # thread; boxes are mapped back to full-resolution pixels right away, so
    # everything downstream (scene, rules, exports) works in video coordinates.
    work_w = min(meta.width, params.detector.imgsz)
    scale = meta.width / work_w
    work_h = int(round(meta.height / scale))
    tracker = ByteTracker(params.tracker)
    store = TrackStore(params.motion, params.tracker, (meta.width, meta.height))
    signals = SignalMonitor(scene.signals, scale)
    features = FrameFeatureCollector(work_w, work_h, params.feature_every_sec, params.feature_width)

    def consume(batch):
        dets = detector([f for _, _, f in batch])
        for (_, t, frame), d in zip(batch, dets):
            d[:, :4] *= scale
            tracked = tracker.update(d, t)
            store.add(t, tracked)
            signals.update(t, frame)
            boxes = np.array([b for _, b, *_ in tracked]).reshape(-1, 4) / scale
            features.update(t, frame, boxes)

    batch = []
    for idx, t, frame in iter_frames(video_path, lambda: budget.stride, width=work_w):
        batch.append((idx, t, frame))
        if len(batch) == params.detector.batch:
            consume(batch)
            batch = []
            budget.check(t, meta.duration)
            if progress and meta.duration:
                progress(0.9 * min(t / meta.duration, 1.0), "detecting and tracking")
    if batch:
        consume(batch)

    if progress:
        progress(0.92, "applying event rules")
    tracks = store.build()
    flow = build_flow(tracks, meta.width, meta.height, view=view)
    if not scene.has_road and flow.total > 0:
        scene.set_learned_road_mask(flow.road_mask())
    timelines = signals.timelines(meta.fps / budget.stride)
    ctx = Context(meta, scene, tracks, flow, timelines, features.result(), params.rules, params.motion)
    events = finalize(run_rules(ctx), meta.duration, params.rules.min_event_sec, params.rules.merge_gap_sec,
                      dict(params.rules.class_merge_gap_sec))
    if progress:
        progress(1.0, "done")
    return Analysis(meta, events, tracks, flow, timelines, features.result(), budget.stride, view,
                    time.perf_counter() - t_start)
