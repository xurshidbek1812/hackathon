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
            on &= tr.speed < ctx.mp.stationary_speed * 2   # a carried bag is not debris
        min_dur = rp.obstacle_min_sec if tr.coco_cls in ANIMALS else 2 * rp.obstacle_min_sec
        for s, e in mask_to_segments(tr.t, on, min_dur=min_dur, max_gap=1.0):
            end = ctx.track_end(tr) if e >= tr.end - 1e-6 else e
            events.append(Event(s, end, "road_obstacle", 0.7, [tr.id]))
    obstacles, _ = _foreground(ctx)
    events += [Event(s, e, "road_obstacle", score) for s, e, score in obstacles]
    return events


def fire_smoke(ctx: Context) -> list[Event]:
    if ctx.features is None:
        return []
    _, smoke = _foreground(ctx)
    found = fire_events(ctx.features, ctx.rp) + smoke
    return [Event(s, min(e, ctx.duration), "fire_smoke", score) for s, e, score in found]
