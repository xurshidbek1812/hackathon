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
        on = sc.on_road(tr.anchor, margin=rp.road_margin * height) & \
            ~sc.in_crossing(tr.anchor, rp.crossing_margin * height)
        # pedestrian islands (the excluded areas) are footway: stepping off one is not jaywalking
        for island in sc.exclude:
            on &= ~sc.inside(island, tr.anchor, -rp.crossing_margin * height)
        for s, e in mask_to_segments(tr.t, on, min_dur=rp.jaywalk_min_sec, max_gap=1.0):
            events.append(Event(s, e, "jaywalking", min(1.0, 0.5 + (e - s) / 10), [tr.id]))
    return events


def failure_to_yield(ctx: Context, lateral_bodies: float = 1.5) -> list[Event]:
    """A vehicle drives through a crossing while a pedestrian on it is in the
    vehicle's way: close to its path, ahead of or beside it (not already passed),
    and standing in the path or walking into it (not walking away). Pedestrians
    waiting on the kerb or far along a long crossing are not in conflict."""
    sc = ctx.scene
    if not sc.crossings:
        return []
    events = []
    for crossing in sc.crossings:
        # pedestrian samples on this crossing and off the kerb: t, x, y, vx, vy
        rows = []
        for p in ctx.persons:
            height = float(np.median(p.box[:, 3] - p.box[:, 1]))
            m = sc.inside(crossing, p.anchor, -0.15 * height) & sc.on_road(p.anchor)
            rows.append(np.column_stack([p.t[m], p.anchor[m], p.vel[m]]))
        peds = np.concatenate(rows) if rows else np.zeros((0, 5))
        if not len(peds):
            continue
        peds = peds[np.argsort(peds[:, 0])]
        for v in ctx.vehicles:
            inside = sc.inside(crossing, v.anchor)
            for s, e in mask_to_segments(v.t, inside, min_dur=0.2, max_gap=0.5):
                m = (v.t >= s) & (v.t <= e)
                if v.speed[m].mean() < 0.3:          # waiting vehicle, not driving through
                    continue
                lo, hi = np.searchsorted(peds[:, 0], [s, e + 1e-6])
                if hi > lo and _in_the_way(v, peds[lo:hi], lateral_bodies):
                    events.append(Event(s, e, "failure_to_yield", 0.7, [v.id]))
    return events


def _in_the_way(v, peds: np.ndarray, lateral_bodies: float) -> bool:
    idx = np.clip(np.searchsorted(v.t, peds[:, 0]), 0, len(v.t) - 1)
    heading = v.unit_dir()[idx]
    size = v.scale[idx]
    rel = peds[:, 1:3] - v.anchor[idx]
    along = np.einsum("ij,ij->i", rel, heading) / size            # + ahead, - behind
    side = heading[:, 0] * rel[:, 1] - heading[:, 1] * rel[:, 0]    # signed lateral offset (px)
    lateral = np.abs(side) / size
    # velocity component toward the path (+ = walking into the vehicle's way)
    ped_v = peds[:, 3:5]
    toward = -np.sign(side) * (heading[:, 0] * ped_v[:, 1] - heading[:, 1] * ped_v[:, 0]) / size
    in_path = lateral <= 0.7
    walking_in = (lateral <= lateral_bodies) & (toward > 0.2)
    return bool(((along > -0.5) & (along < 4.0) & (in_path | walking_in)).any())
