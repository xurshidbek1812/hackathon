"""Serialise an analysis for the website, the demo and the renderer."""
from __future__ import annotations

from typing import Callable

import numpy as np

from .detector import CATEGORY_NAMES
from .pipeline import Analysis
from .risk import RiskModel
from .video import iter_frames, read_meta


def risk_curve(video_path: str, progress: Callable[[float, str], None] | None = None,
               every_sec: float = 0.2) -> list[list[float]]:
    """Stream the video through the causal RiskModel exactly as the harness
    does; keep one [t, score] point per `every_sec` for plotting."""
    meta = read_meta(video_path)
    model = RiskModel()
    model.reset({"video_id": meta.path, "fps": meta.fps, "width": meta.width,
                 "height": meta.height, "n_frames": meta.n_frames})
    out, next_t = [], 0.0
    for _, t, frame in iter_frames(video_path, lambda: 1):
        score = model.step(frame, t)
        if t >= next_t:
            out.append([round(t, 2), round(score, 4)])
            next_t = t + every_sec
            if progress and meta.duration:
                progress(min(t / meta.duration, 1.0), "estimating accident risk")
    return out


def tracks_json(analysis: Analysis, every: int = 2) -> list[dict]:
    out = []
    for tr in analysis.tracks:
        sl = slice(None, None, every)
        out.append({
            "id": tr.id,
            "cat": CATEGORY_NAMES[tr.category],
            "name": tr.name,
            "t": np.round(tr.t[sl], 2).tolist(),
            "box": np.round(tr.raw_box[sl]).astype(int).tolist(),
        })
    return out


def analysis_json(analysis: Analysis, risk: list[list[float]] | None = None, name: str | None = None) -> dict:
    m = analysis.meta
    return {
        "video": name or m.path,
        "meta": {"fps": m.fps, "width": m.width, "height": m.height,
                 "n_frames": m.n_frames, "duration": round(m.duration, 2)},
        "events": [[round(e.start, 2), round(e.end, 2), e.label, round(e.score, 3), e.track_ids]
                   for e in analysis.events],
        "risk": risk or [],
        "counts": analysis.counts_per_second(),
        "signals": {k: {"t": np.round(v.t[::5], 2).tolist(), "state": v.state[::5].tolist()}
                    for k, v in analysis.signals.items()},
        "tracks": tracks_json(analysis),
        "stats": {"stride": analysis.stride, "seconds": round(analysis.seconds, 1),
                  "n_tracks": len(analysis.tracks)},
    }
