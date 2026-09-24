"""Rules on single-vehicle trajectories plus the scene layout."""
from __future__ import annotations

import numpy as np

from ..segments import Event, mask_to_segments
from .common import Context, fill_short_gaps, heading_change_window, runs, signed_side


def stopped_vehicle(ctx: Context) -> list[Event]:
    rp, mp = ctx.rp, ctx.mp
    events = []
    for tr in ctx.cars:
        still = fill_short_gaps(tr.t, tr.speed < mp.stationary_speed, 1.0)
        for a, b in runs(still):
            t0, t1 = float(tr.t[a]), float(tr.t[b])
            if t1 - t0 < rp.stopped_min_sec:
                continue
            p = np.median(tr.anchor[a:b + 1], axis=0)
            if not ctx.scene.on_road(p[None])[0]:
                continue
            if _is_queued(ctx, tr, a, b, p):
                continue
            end = ctx.track_end(tr) if b == len(tr.t) - 1 else t1
            score = min(1.0, 0.5 + (t1 - t0) / 60)
            events.append(Event(t0, end, "stopped_vehicle", score, [tr.id]))
    return events


def _is_queued(ctx: Context, tr, a: int, b: int, p: np.ndarray) -> bool:
    """A stop that is part of normal signal queueing, not a breakdown/illegal stop."""
    rp = ctx.rp
    t0, t1 = tr.t[a], tr.t[b]
    dur = t1 - t0
    released_before_end = t1 < ctx.duration - 1.0
    # 1) stopped where vehicles routinely queue, and left again within a signal cycle
    if released_before_end and dur < 150 and ctx.flow.is_queue_zone(p[None])[0]:
        return True
    # 2) released together with neighbours: the whole queue moved off at once
    together = 0
    for other in ctx.cars:
        if other.id == tr.id or other.end < t0 or other.start > t1:
            continue
        i = other.at(t1)
        if i is None:
            continue
        near = np.linalg.norm(other.anchor[i] - p) / tr.scale[b] < rp.queue_neighbor_dist
        if not near:
            continue
        o_still = other.speed < ctx.mp.stationary_speed
        # the neighbour starts moving within the release window of our own start
        win = (other.t > t1 - rp.queue_release_window_sec) & (other.t < t1 + rp.queue_release_window_sec)
        if released_before_end and o_still[win].any() and (~o_still[win]).any():
            together += 1
    return together >= 1


def congestion(ctx: Context) -> list[Event]:
    rp = ctx.rp
    if ctx.duration <= 0:
        return []
    bins = np.arange(0.0, ctx.duration, 1.0)
    groups = ctx.scene.carriageways or [None]
    congested = np.zeros(len(bins), bool)
    for poly in groups:
        n = np.zeros(len(bins))
        slow = np.zeros(len(bins))
        for tr in ctx.cars:
            inside = ctx.scene.on_road(tr.anchor) if poly is None else ctx.scene.inside(poly, tr.anchor)
            if not inside.any():
                continue
            b = np.floor(tr.t[inside]).astype(int)
            b = b[b < len(bins)]
            if not len(b):
                continue
            sp = tr.speed[inside][: len(b)]
            present = np.bincount(b, minlength=len(bins)) > 0
            mean_sp = np.bincount(b, weights=sp, minlength=len(bins)) / np.maximum(
                np.bincount(b, minlength=len(bins)), 1)
            n += present
            slow += present & (mean_sp < rp.congestion_slow_speed)
        cond = (n >= rp.congestion_min_vehicles) & (slow / np.maximum(n, 1) >= rp.congestion_slow_frac)
        congested |= cond
    return [Event(s, min(e + 1.0, ctx.duration), "congestion", min(1.0, (e - s) / 60 + 0.5))
            for s, e in mask_to_segments(bins, congested, min_dur=rp.congestion_min_sec,
                                         max_gap=rp.congestion_merge_gap_sec)]


def wrong_way(ctx: Context) -> list[Event]:
    rp, mp = ctx.rp, ctx.mp
    events = []
    for tr in ctx.vehicles:
        moving = tr.speed > mp.moving_speed
        if moving.sum() < 3:
            continue
        if ctx.scene.lanes:
            exp = np.array([d if (d := ctx.scene.lane_direction(p[None])) is not None else [np.nan, np.nan]
                            for p in tr.anchor])
        else:
            exp = ctx.flow.expected_direction(tr.anchor)
        known = moving & ~np.isnan(exp[:, 0])
        cos = np.einsum("ij,ij->i", tr.unit_dir(), np.nan_to_num(exp))
        against = known & (cos < rp.wrong_way_cos)
        for s, e in mask_to_segments(tr.t, against, min_dur=rp.wrong_way_min_sec, max_gap=1.0):
            m = (tr.t >= s) & (tr.t <= e)
            frac = against[m].sum() / max(known[m].sum(), 1)
            step = np.linalg.norm(np.diff(tr.anchor[m], axis=0), axis=1) / tr.scale[m][1:]
            if frac < 0.7 or step.sum() < rp.wrong_way_min_dist:
                continue
            end = ctx.track_end(tr) if e >= tr.end - 1e-6 else e
            events.append(Event(s, end, "wrong_way", min(1.0, 0.4 + 0.1 * step.sum()), [tr.id]))
    return events


def illegal_u_turn(ctx: Context) -> list[Event]:
    rp = ctx.rp
    events = []
    min_rad = np.deg2rad(rp.u_turn_min_deg)
    for tr in ctx.vehicles:
        idx = np.flatnonzero(tr.speed > 0.3)
        if len(idx) < 5:
            continue
        h = np.unwrap(tr.heading[idx])
        t = tr.t[idx]
        u = tr.unit_dir()[idx]
        found = None
        for j in range(len(idx)):
            lo = np.searchsorted(t, t[j] - rp.u_turn_max_sec)
            dh = np.abs(h[j] - h[lo:j + 1])
            ok = np.flatnonzero((dh >= min_rad) & (u[lo:j + 1] @ u[j] < -0.7))
            if len(ok):
                found = (idx[lo + ok[-1]], idx[j])
                break
        if found is None:
            continue
        # the turn may keep going past the 150 deg mark: look 2 s further for it to settle
        i_end = min(len(tr.t) - 1, int(np.searchsorted(tr.t, tr.t[found[1]] + 2.0)))
        start, end = heading_change_window(tr, found[0], i_end)
        mid = tr.anchor[(found[0] + found[1]) // 2][None]
        if any(ctx.scene.inside(p, mid)[0] for p in ctx.scene.u_turn_allowed):
            continue
        events.append(Event(start, end, "illegal_u_turn", 0.7, [tr.id]))
    return events


def illegal_turn(ctx: Context) -> list[Event]:
    sc = ctx.scene
    if not sc.zones or not sc.prohibited_turns:
        return []
    banned = set(sc.prohibited_turns)
    events = []
    for tr in ctx.vehicles:
        zones = sc.zone_of(tr.anchor)
        visited = [(i, z) for i, z in enumerate(zones) if z is not None]
        if len(visited) < 2:
            continue
        entry_zone, exit_zone = visited[0][1], visited[-1][1]
        if (entry_zone, exit_zone) not in banned:
            continue
        i_leave = max(i for i, z in visited if z == entry_zone)
        later = [i for i, z in visited if z == exit_zone and i > i_leave]
        i_enter = later[0] if later else visited[-1][0]
        # the turn may begin a little before the vehicle leaves its entry zone
        a = int(np.searchsorted(tr.t, tr.t[i_leave] - 3.0))
        start, end = heading_change_window(tr, a, max(i_enter, a + 1))
        events.append(Event(start, end, "illegal_turn", 0.8, [tr.id]))
    return events


def solid_line_crossing(ctx: Context) -> list[Event]:
    rp = ctx.rp
    events = []
    for line in ctx.scene.solid_lines:
        for tr in ctx.vehicles:
            side = signed_side(line, tr.anchor)
            if not (side != 0).any():
                continue
            side = _hold_filter(tr.t, side, rp.line_cross_hold_sec)
            flips = np.flatnonzero((side[1:] * side[:-1]) < 0) + 1
            if not len(flips):
                continue
            left = signed_side(line, np.column_stack([tr.box[:, 0], tr.box[:, 3]]))
            right = signed_side(line, np.column_stack([tr.box[:, 2], tr.box[:, 3]]))
            for k in flips:
                new = side[k]
                # wheel crosses: first corner on the new side, looking back from the flip
                j = k
                while j > 0 and (left[j - 1] == new or right[j - 1] == new) and tr.t[k] - tr.t[j - 1] < 3:
                    j -= 1
                # fully in the new lane: both corners on the new side
                m = k
                while m < len(side) - 1 and not (left[m] == new and right[m] == new) and tr.t[m] - tr.t[k] < 3:
                    m += 1
                events.append(Event(float(tr.t[j]), float(max(tr.t[m], tr.t[j] + 0.5)),
                                    "solid_line_crossing", 0.7, [tr.id]))
    return events


def _hold_filter(t: np.ndarray, side: np.ndarray, hold: float) -> np.ndarray:
    """Keep only side changes that persist for `hold` seconds (removes jitter
    of vehicles driving along a line)."""
    stable = np.zeros_like(side)
    cur = 0.0
    i, n = 0, len(side)
    while i < n:
        j = i
        while j + 1 < n and side[j + 1] == side[i]:
            j += 1
        if side[i] != 0 and t[j] - t[i] >= hold:
            cur = side[i]
        stable[i:j + 1] = cur
        i = j + 1
    return stable
