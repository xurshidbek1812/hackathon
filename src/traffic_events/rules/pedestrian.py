"""Pedestrian rules: jaywalking and failure_to_yield."""
from __future__ import annotations

import numpy as np

from ..segments import Event, mask_to_segments
from .common import Context


def jaywalking(ctx: Context) -> list[Event]:
    sc, rp = ctx.scene, ctx.rp
    if not sc.has_road or sc.crossings is None:
        return []           # without knowing where crossings are, every crossing pedestrian would fire
    events = []
    for tr in ctx.persons:
        height = float(np.median(tr.box[:, 3] - tr.box[:, 1]))
        on = sc.on_road(tr.anchor, margin=rp.road_margin * height) & ~sc.in_crossing(tr.anchor, rp.crossing_margin_px)
        for s, e in mask_to_segments(tr.t, on, min_dur=rp.jaywalk_min_sec, max_gap=1.0):
            events.append(Event(s, e, "jaywalking", min(1.0, 0.5 + (e - s) / 10), [tr.id]))
    return events


def failure_to_yield(ctx: Context) -> list[Event]:
    sc, rp = ctx.scene, ctx.rp
    if not sc.crossings:
        return []
    events = []
    for crossing in sc.crossings:
        ped_times = []
        for p in ctx.persons:
            on = sc.inside(crossing, p.anchor, -rp.crossing_margin_px)
            ped_times.extend(p.t[on].tolist())
        if not ped_times:
            continue
        ped_times = np.sort(np.asarray(ped_times))
        for v in ctx.vehicles:
            inside = sc.inside(crossing, v.anchor)
            for s, e in mask_to_segments(v.t, inside, min_dur=0.2, max_gap=0.5):
                m = (v.t >= s) & (v.t <= e)
                if v.speed[m].mean() < 0.3:          # waiting vehicle, not driving through
                    continue
                lo, hi = np.searchsorted(ped_times, [s, e])
                if hi > lo:
                    events.append(Event(s, e, "failure_to_yield", 0.7, [v.id]))
    return events
