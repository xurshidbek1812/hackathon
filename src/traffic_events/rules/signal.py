"""Rules that need the traffic-signal state: red_light and stop_line."""
from __future__ import annotations

import numpy as np

from ..scene import StopLine
from ..segments import Event
from ..signal_state import GREEN, RED
from .common import Context, fill_short_gaps, front_point, runs, signed_side


def _after_side(sl: StopLine) -> float:
    """Sign of `signed_side` for points past the line in the approach direction."""
    a, b = sl.line
    ab = b - a
    return float(np.sign(ab[0] * sl.approach[1] - ab[1] * sl.approach[0]))


def _signal_lines(ctx: Context):
    for sl in ctx.scene.stop_lines:
        if sl.signal and sl.signal in ctx.signals:
            yield sl, ctx.signals[sl.signal]


def red_light(ctx: Context) -> list[Event]:
    events = []
    for sl, sig in _signal_lines(ctx):
        after = _after_side(sl)
        for tr in ctx.vehicles:
            fp = front_point(tr.box, sl.approach)
            side = signed_side(sl.line, fp, extent_margin=0.1)
            heading_ok = tr.unit_dir() @ sl.approach > 0.5
            for k in np.flatnonzero((side[:-1] == -after) & (side[1:] == after)) + 1:
                if not heading_ok[max(0, k - 2):k + 1].any():
                    continue
                tc = float(tr.t[k])
                if sig.state_at(tc) != RED or not _drives_on(ctx, tr, k, sl):
                    continue
                end = _leave_time(ctx, tr, k)
                events.append(Event(tc, end, "red_light", 0.8, [tr.id]))
                break
    return events


def _drives_on(ctx: Context, tr, k: int, sl: StopLine, depth: float = 2.5, horizon: float = 5.0) -> bool:
    """After crossing, does the vehicle continue into the junction (red_light)
    rather than stopping just past the line (stop_line)?"""
    m = (tr.t >= tr.t[k]) & (tr.t <= tr.t[k] + horizon)
    if ctx.scene.intersection is not None and ctx.scene.inside(ctx.scene.intersection, tr.anchor[m]).any():
        return True
    mid = sl.line.mean(axis=0)
    fp = front_point(tr.box[m], sl.approach)
    return bool((((fp - mid) @ sl.approach) / tr.scale[m] > depth).any())


def _leave_time(ctx: Context, tr, k: int) -> float:
    inter = ctx.scene.intersection
    if inter is not None:
        inside = ctx.scene.inside(inter, tr.anchor[k:])
        entered = np.flatnonzero(inside)
        if len(entered):
            out = np.flatnonzero(~inside[entered[0]:])
            if len(out):
                return float(tr.t[k + entered[0] + out[0]])
    return min(ctx.track_end(tr), float(tr.t[k]) + 15.0)


def stop_line(ctx: Context) -> list[Event]:
    rp, mp = ctx.rp, ctx.mp
    events = []
    for sl, sig in _signal_lines(ctx):
        after = _after_side(sl)
        a, b = sl.line
        normal = sl.approach
        for tr in ctx.cars:
            still = fill_short_gaps(tr.t, tr.speed < mp.stationary_speed, 1.0)
            for i0, i1 in runs(still):
                if tr.t[i1] - tr.t[i0] < 2.0:
                    continue
                fp = front_point(tr.box[i0:i0 + 1], sl.approach)
                if signed_side(sl.line, fp, extent_margin=0.1)[0] != after:
                    continue
                # only just past the line, not inside the intersection proper
                depth = float((fp[0] - (a + b) / 2) @ normal) / tr.scale[i0]
                if depth > 2.5:
                    continue
                if ctx.scene.intersection is not None and ctx.scene.inside(ctx.scene.intersection, tr.anchor[i0:i0 + 1])[0]:
                    continue
                t0 = float(tr.t[i0])
                if sig.state_at(t0) != RED:
                    continue
                green = sig.next_change_to(t0, GREEN)
                end = green if green is not None else ctx.track_end(tr)
                if end - t0 >= rp.min_event_sec:
                    events.append(Event(t0, end, "stop_line", 0.7, [tr.id]))
    return events
