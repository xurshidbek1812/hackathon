"""Road hazards: obstacles (detected animals/objects + static foreground blobs)
and fire/smoke (pixel cues)."""
from __future__ import annotations

import numpy as np

from ..frame_features import fire_events, static_foreground_events
from ..segments import Event, mask_to_segments
from .common import Context

ANIMALS = {15, 16, 17, 18, 19}


def _foreground(ctx: Context):
    if ctx.features is None:
        return [], []
    if "foreground" not in ctx.cache:
        mask = ctx.scene.road_mask(ctx.features.thumb_w, ctx.features.thumb_h)
        ctx.cache["foreground"] = static_foreground_events(ctx.features, mask, ctx.rp)
    return ctx.cache["foreground"]


def road_obstacle(ctx: Context) -> list[Event]:
    rp = ctx.rp
    events = []
    for tr in ctx.obstacles:
        on = ctx.scene.on_road(tr.anchor)
        if tr.coco_cls not in ANIMALS:
            # a bag carried by, or standing next to, a person is not debris
            on &= (tr.speed < ctx.mp.stationary_speed * 2) & ~_next_to_person(ctx, tr)
        min_dur = rp.obstacle_min_sec if tr.coco_cls in ANIMALS else 2 * rp.obstacle_min_sec
        for s, e in mask_to_segments(tr.t, on, min_dur=min_dur, max_gap=1.0):
            end = ctx.track_end(tr) if e >= tr.end - 1e-6 else e
            events.append(Event(s, end, "road_obstacle", 0.7, [tr.id]))
    obstacles, _ = _foreground(ctx)
    events += [Event(s, e, "road_obstacle", score) for s, e, score in obstacles]
    return events


def _next_to_person(ctx: Context, obj, reach: float = 0.5) -> np.ndarray:
    """Per sample of `obj`: is its centre inside some person's box grown by `reach`?"""
    centre = np.column_stack([(obj.box[:, 0] + obj.box[:, 2]) / 2, (obj.box[:, 1] + obj.box[:, 3]) / 2])
    near = np.zeros(len(obj.t), bool)
    for person in ctx.persons:
        if person.end < obj.start or person.start > obj.end:
            continue
        idx = np.clip(np.searchsorted(person.t, obj.t), 0, len(person.t) - 1)
        close_in_time = np.abs(person.t[idx] - obj.t) < 0.5
        b = person.box[idx]
        gw, gh = (b[:, 2] - b[:, 0]) * reach, (b[:, 3] - b[:, 1]) * reach
        inside = (centre[:, 0] > b[:, 0] - gw) & (centre[:, 0] < b[:, 2] + gw) &                  (centre[:, 1] > b[:, 1] - gh) & (centre[:, 1] < b[:, 3] + gh)
        near |= close_in_time & inside
    return near


def fire_smoke(ctx: Context) -> list[Event]:
    if ctx.features is None:
        return []
    _, smoke = _foreground(ctx)
    found = fire_events(ctx.features, ctx.rp) + smoke
    return [Event(s, min(e, ctx.duration), "fire_smoke", score) for s, e, score in found]
