"""Rules that need the signal phase of an approach: red_light and stop_line.

The phase of a stop line comes from one of two sources:

* LampPhase  - the signal head is visible and configured (`signal` names a
               region in scene.json): red/green read from its pixels.
* QueuePhase - the head faces away from the camera. Red is inferred from the
               traffic itself: other vehicles standing still right at the
               stop line (two for 5 s, or one for 10 s) while nothing else
               crosses it only do that on red. Green is when
               that waiting queue moves off.
"""
from __future__ import annotations

import numpy as np

from ..scene import StopLine
from ..segments import Event
from ..signal_state import GREEN, RED, SignalTimeline
from .common import Context, fill_short_gaps, front_point, runs, signed_side


def line_depth(sl: StopLine, box: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Signed distance (body units) of the vehicle front past the stop line;
    negative = before the line, NaN = beyond the ends of the line."""
    fp = front_point(box, sl.approach)
    side = signed_side(sl.line, fp, extent_margin=0.1)
    a, b = sl.line
    ab = (b - a) / max(np.linalg.norm(b - a), 1e-6)
    rel = fp - a
    dist = np.abs(rel[:, 0] * ab[1] - rel[:, 1] * ab[0])
    after = np.sign(ab[0] * sl.approach[1] - ab[1] * sl.approach[0])
    depth = np.where(side == after, 1.0, -1.0) * dist / scale
    return np.where(side == 0, np.nan, depth)


class LampPhase:
    def __init__(self, timeline: SignalTimeline):
        self.tl = timeline

    def is_red(self, t: float, exclude: int | None = None) -> bool:
        return self.tl.state_at(t) == RED

    def next_green(self, t: float, exclude: int | None = None) -> float | None:
        return self.tl.next_change_to(t, GREEN)


class QueuePhase:
    WAIT_SEC = 5.0            # standing at the line at least this long ...
    MIN_WAITING = 2           # ... by this many vehicles,
    LONE_WAIT_SEC = 10.0      # or a single vehicle waiting at least this long
    ZONE = (-2.0, 0.3)        # front within 2 bodies before the line (or touching it)
    FLOW_WINDOW = (4.0, 2.0)  # no other crossing this long before / after (s): traffic is flowing = green

    def __init__(self, ctx: Context, sl: StopLine):
        self.stops = []       # (track_id, t_start, t_end) of standing still at the line
        self.crossings = []   # (track_id, t) of vehicles crossing the line in its approach direction
        for tr in ctx.vehicles:
            depth = line_depth(sl, tr.box, tr.scale)
            for k in np.flatnonzero((depth[:-1] < 0) & (depth[1:] >= 0)) + 1:
                self.crossings.append((tr.id, float(tr.t[k])))
            if tr.category != 0:
                continue
            still = fill_short_gaps(tr.t, tr.speed < ctx.mp.stationary_speed, 1.0)
            at_line = still & (depth >= self.ZONE[0]) & (depth <= self.ZONE[1])
            for i0, i1 in runs(at_line):
                if tr.t[i1] - tr.t[i0] >= 1.0:
                    self.stops.append((tr.id, float(tr.t[i0]), float(tr.t[i1])))

    def _waiting(self, t: float, exclude: int | None):
        return [(tid, s, e) for tid, s, e in self.stops
                if tid != exclude and s + self.WAIT_SEC <= t <= e]

    def _flowing(self, t: float, exclude: int | None) -> bool:
        before, after = self.FLOW_WINDOW
        return any(tid != exclude and t - before <= tc <= t + after for tid, tc in self.crossings)

    def is_red(self, t: float, exclude: int | None = None) -> bool:
        waiting = self._waiting(t, exclude)
        enough = len(waiting) >= self.MIN_WAITING or any(t - s >= self.LONE_WAIT_SEC for _, s, _ in waiting)
        return enough and not self._flowing(t, exclude)

    def next_green(self, t: float, exclude: int | None = None) -> float | None:
        ends = [e for _, _, e in self._waiting(t, exclude)]
        return min(ends) if ends else None


def _phases(ctx: Context):
    for sl in ctx.scene.stop_lines:
        if sl.signal and sl.signal in ctx.signals:
            yield sl, LampPhase(ctx.signals[sl.signal])
        else:
            yield sl, QueuePhase(ctx, sl)


def red_light(ctx: Context) -> list[Event]:
    events = []
    for sl, phase in _phases(ctx):
        for tr in ctx.vehicles:
            depth = line_depth(sl, tr.box, tr.scale)
            heading_ok = tr.unit_dir() @ sl.approach > 0.5
            crossed = np.flatnonzero((depth[:-1] < 0) & (depth[1:] >= 0)) + 1
            for k in crossed:
                if not heading_ok[max(0, k - 2):k + 1].any():
                    continue
                tc = float(tr.t[k])
                if not phase.is_red(tc, exclude=tr.id):
                    continue
                # it may pause after crossing; what counts is entering the junction before green
                entry = _entry_time(ctx, tr, k, depth)
                green = phase.next_green(tc, exclude=tr.id)
                if entry is None or (green is not None and entry > green + 0.5):
                    continue
                events.append(Event(tc, _leave_time(ctx, tr, k), "red_light", 0.8, [tr.id]))
                break
    return events


def _entry_time(ctx: Context, tr, k: int, depth: np.ndarray, min_depth: float = 2.5) -> float | None:
    """When, after crossing the stop line at sample k, the vehicle enters the
    junction (inside the junction box, or well past the line); None if it never does."""
    after = np.arange(k, len(tr.t))
    entered = np.nan_to_num(depth[after], nan=min_depth + 1) > min_depth   # NaN: beyond the line's ends
    if ctx.scene.intersection is not None:
        entered |= ctx.scene.inside(ctx.scene.intersection, tr.anchor[after])
    hit = np.flatnonzero(entered)
    return float(tr.t[after[hit[0]]]) if len(hit) else None


def _leave_time(ctx: Context, tr, k: int) -> float:
    """End of a red-light event: the vehicle leaves the junction box, or the frame."""
    inter = ctx.scene.intersection
    if inter is None:
        return min(ctx.track_end(tr), float(tr.t[k]) + 15.0)
    inside = ctx.scene.inside(inter, tr.anchor[k:])
    entered = np.flatnonzero(inside)
    if len(entered):
        out = np.flatnonzero(~inside[entered[0]:])
        if len(out):
            return float(tr.t[k + entered[0] + out[0]])
    return ctx.track_end(tr)


def stop_line(ctx: Context) -> list[Event]:
    rp, mp = ctx.rp, ctx.mp
    events = []
    for sl, phase in _phases(ctx):
        for tr in ctx.cars:
            depth = line_depth(sl, tr.box, tr.scale)
            still = fill_short_gaps(tr.t, tr.speed < mp.stationary_speed, 1.0)
            for i0, i1 in runs(still):
                if tr.t[i1] - tr.t[i0] < 2.0:
                    continue
                # stopped with its front clearly past the line (not just touching it) but not in the junction
                if not (0.5 <= depth[i0] <= 2.5):
                    continue
                if ctx.scene.intersection is not None and \
                        ctx.scene.inside(ctx.scene.intersection, tr.anchor[i0:i0 + 1])[0]:
                    continue
                t0 = float(tr.t[i0])
                # judge the phase a few seconds into the stop, once a queue has formed
                tq = min(float(tr.t[i1]), t0 + QueuePhase.WAIT_SEC + 1.0)
                if not phase.is_red(tq, exclude=tr.id):
                    continue
                green = phase.next_green(tq, exclude=tr.id)
                # a vehicle that goes on into the junction while still red is red_light instead
                entry = _entry_time(ctx, tr, i0, depth)
                if entry is not None and (green is None or entry < green - 0.5):
                    continue
                end = green if green is not None else ctx.track_end(tr)
                if end - t0 >= rp.min_event_sec:
                    events.append(Event(t0, end, "stop_line", 0.7, [tr.id]))
    return events
