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
        on = sc.on_road(tr.anchor, margin=rp.road_margin * height) &             ~sc.in_crossing(tr.anchor, rp.crossing_margin * height)
        for s, e in mask_to_segments(tr.t, on, min_dur=rp.jaywalk_min_sec, max_gap=1.0):
            events.append(Event(s, e, "jaywalking", min(1.0, 0.5 + (e - s) / 10), [tr.id]))
    return events


def failure_to_yield(ctx: Context, near_bodies: float = 3.0) -> list[Event]:
    """A vehicle drives through a crossing while a pedestrian is on it close to
    the vehicle's path. Pedestrians far along a long crossing, or waiting on the
    kerb, are not in conflict and do not count."""
    sc, rp = ctx.scene, ctx.rp
    if not sc.crossings:
        return []
    events = []
    for crossing in sc.crossings:
        # (t, x, y) of pedestrians on this crossing and off the kerb (people waiting
        # at the crossing's ends are not in the vehicle's way)
        on = []
        for p in ctx.persons:
            height = float(np.median(p.box[:, 3] - p.box[:, 1]))
            m = sc.inside(crossing, p.anchor, -0.15 * height) & sc.on_road(p.anchor)
            on.extend(np.column_stack([p.t[m], p.anchor[m]]).tolist())
        if not on:
            continue
        on = np.asarray(on)
        on = on[np.argsort(on[:, 0])]
        for v in ctx.vehicles:
            inside = sc.inside(crossing, v.anchor)
            for s, e in mask_to_segments(v.t, inside, min_dur=0.2, max_gap=0.5):
                m = (v.t >= s) & (v.t <= e)
                if v.speed[m].mean() < 0.3:          # waiting vehicle, not driving through
                    continue
                lo, hi = np.searchsorted(on[:, 0], [s, e + 1e-6])
                if hi <= lo:
                    continue
                peds = on[lo:hi]
                idx = np.clip(np.searchsorted(v.t, peds[:, 0]), 0, len(v.t) - 1)
                gap = np.linalg.norm(peds[:, 1:] - v.anchor[idx], axis=1) / v.scale[idx]
                if gap.min() <= near_bodies:
                    events.append(Event(s, e, "failure_to_yield", 0.7, [v.id]))
    return events
