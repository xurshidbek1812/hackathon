"""Event segments and their post-processing."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Event:
    start: float
    end: float
    label: str
    score: float = 1.0
    track_ids: list[int] = field(default_factory=list)

    def as_list(self) -> list:
        return [round(float(self.start), 2), round(float(self.end), 2), self.label]


def mask_to_segments(t: np.ndarray, mask: np.ndarray, min_dur: float = 0.0,
                     max_gap: float = 0.0) -> list[tuple[float, float]]:
    """Contiguous True runs of `mask` sampled at times `t` -> [(start, end)].

    Runs separated by less than `max_gap` seconds are joined; runs shorter than
    `min_dur` are dropped. A run ends at the last True sample.
    """
    t = np.asarray(t, float)
    mask = np.asarray(mask, bool)
    if len(t) == 0 or not mask.any():
        return []
    idx = np.flatnonzero(mask)
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]])
    segs = [(float(t[s]), float(t[e])) for s, e in zip(starts, ends)]
    segs = merge_intervals(segs, max_gap)
    return [(s, e) for s, e in segs if e - s >= min_dur]


def merge_intervals(segs, max_gap: float = 0.0) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for s, e in sorted(segs):
        if out and s - out[-1][1] <= max_gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def finalize(events: list[Event], duration: float, min_dur: float, merge_gap: float,
             class_gaps: dict[str, float] | None = None) -> list[Event]:
    """Clip to the video, drop blips, and union same-class events that touch.

    The task defines simultaneous same-class events as one segment covering
    both, and the harness drops same-class overlaps, so overlapping or nearly
    touching events of one class are merged here; `class_gaps` overrides the
    joining gap per class.
    """
    by_label: dict[str, list[Event]] = {}
    for ev in events:
        s, e = max(0.0, ev.start), min(duration, ev.end) if duration > 0 else ev.end
        if e - s < min_dur:
            continue
        by_label.setdefault(ev.label, []).append(Event(s, e, ev.label, ev.score, list(ev.track_ids)))

    out: list[Event] = []
    for label, evs in by_label.items():
        gap = (class_gaps or {}).get(label, merge_gap)
        evs.sort(key=lambda x: x.start)
        cur = evs[0]
        for ev in evs[1:]:
            if ev.start - cur.end <= gap:
                cur = Event(cur.start, max(cur.end, ev.end), label, max(cur.score, ev.score),
                            cur.track_ids + ev.track_ids)
            else:
                out.append(cur)
                cur = ev
        out.append(cur)
    out.sort(key=lambda x: (x.start, x.label))
    # rounding to 2 decimals must not create start >= end or same-class overlap
    final = []
    last_end: dict[str, float] = {}
    for ev in out:
        s, e = round(ev.start, 2), round(ev.end, 2)
        s = max(s, last_end.get(ev.label, -1.0) + 0.01)
        if e - s < 0.01:
            continue
        last_end[ev.label] = e
        final.append(Event(s, e, ev.label, ev.score, ev.track_ids))
    return final
