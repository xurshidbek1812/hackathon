"""Accident and near-miss rules on pairs of trajectories.

Contact  = boxes overlap AND their ground points are at a similar depth
           (otherwise it is just perspective occlusion).
Accident = contact that was preceded by fast closing and followed by an
           abrupt change: hard deceleration, a swerve, a pedestrian falling,
           or the road users staying stopped together.
Near miss = predicted time-to-collision below a threshold with an evasive
           action (hard braking or swerve) and no contact.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from ..detector import OBSTACLE, PERSON
from ..segments import Event
from ..tracks import Track
from .common import Context, body_distance, common_times, runs


def _extent(tr: Track, pad_bodies: float) -> np.ndarray:
    pad = pad_bodies * float(tr.scale.max())
    return np.array([tr.box[:, 0].min() - pad, tr.box[:, 1].min() - pad,
                     tr.box[:, 2].max() + pad, tr.box[:, 3].max() + pad])


def _candidate_pairs(tracks: list[Track], pad_bodies: float = 0.0):
    """Pairs that coexist in time and whose (padded) image footprints meet."""
    ext = {tr.id: _extent(tr, pad_bodies) for tr in tracks}
    for a, b in combinations(tracks, 2):
        if a.category == PERSON and b.category == PERSON:
            continue
        if a.end < b.start or b.end < a.start:
            continue
        ea, eb = ext[a.id], ext[b.id]
        if ea[2] < eb[0] or eb[2] < ea[0] or ea[3] < eb[1] or eb[3] < ea[1]:
            continue
        yield a, b


def _contact(a: Track, ia, b: Track, ib, depth_tol: float) -> np.ndarray:
    ba, bb = a.box[ia], b.box[ib]
    ix = np.minimum(ba[:, 2], bb[:, 2]) - np.maximum(ba[:, 0], bb[:, 0])
    iy = np.minimum(ba[:, 3], bb[:, 3]) - np.maximum(ba[:, 1], bb[:, 1])
    hmin = np.minimum(ba[:, 3] - ba[:, 1], bb[:, 3] - bb[:, 1])
    same_depth = np.abs(ba[:, 3] - bb[:, 3]) < depth_tol * hmin
    return (ix > 0) & (iy > 0) & same_depth


def _window(tr: Track, t0: float, t1: float) -> np.ndarray:
    return (tr.t >= t0) & (tr.t <= t1)


def speed_drop(tr: Track, t: float, before: float = 0.7, after: float = 1.2) -> float:
    """Body units/s lost around time t (max speed before minus min speed after)."""
    pre, post = _window(tr, t - before, t), _window(tr, t, t + after)
    if not pre.any() or not post.any():
        return 0.0
    return float(tr.speed[pre].max() - tr.speed[post].min())


def swerve(tr: Track, t: float, span: float = 1.0) -> float:
    """Largest heading change (degrees) of a moving track within ±span of t."""
    m = _window(tr, t - span, t + span) & (tr.speed > 0.5)
    if m.sum() < 3:
        return 0.0
    h = np.unwrap(tr.heading[m])
    return float(np.rad2deg(h.max() - h.min()))


def fell(tr: Track, t: float) -> bool:
    """Pedestrian box turns from upright to lying within 2 s after t."""
    if tr.category != PERSON:
        return False
    pre, post = _window(tr, t - 1.0, t), _window(tr, t, t + 2.0)
    if not pre.any() or not post.any():
        return False
    ar = (tr.box[:, 3] - tr.box[:, 1]) / np.maximum(tr.box[:, 2] - tr.box[:, 0], 1)
    return bool(np.median(ar[pre]) > 1.6 and ar[post].min() < 0.6 * np.median(ar[pre]))


def closing_speed(a: Track, ia, b: Track, ib) -> np.ndarray:
    """Rate at which the gap shrinks, body units / s (positive = approaching)."""
    rel_p = b.anchor[ib] - a.anchor[ia]
    rel_v = b.vel[ib] - a.vel[ia]
    dist = np.maximum(np.linalg.norm(rel_p, axis=1), 1e-6)
    scale = 0.5 * (a.scale[ia] + b.scale[ib])
    return -np.einsum("ij,ij->i", rel_p, rel_v) / dist / scale


def time_to_collision(a: Track, ia, b: Track, ib) -> tuple[np.ndarray, np.ndarray]:
    """Time to closest approach and the predicted miss distance (body units)."""
    rel_p = b.anchor[ib] - a.anchor[ia]
    rel_v = b.vel[ib] - a.vel[ia]
    v2 = np.maximum(np.einsum("ij,ij->i", rel_v, rel_v), 1e-6)
    ttc = -np.einsum("ij,ij->i", rel_p, rel_v) / v2
    miss = np.linalg.norm(rel_p + rel_v * np.clip(ttc, 0, None)[:, None], axis=1)
    return ttc, miss / (0.5 * (a.scale[ia] + b.scale[ib]))


def _settle_time(ctx: Context, tr: Track, t0: float) -> float:
    """When a road user comes to rest (or leaves) after t0."""
    m = tr.t >= t0
    still = tr.speed[m] < ctx.mp.stationary_speed
    idx = np.flatnonzero(still)
    if len(idx):
        return float(tr.t[m][idx[0]])
    return ctx.track_end(tr)


def _judgeable_users(ctx: Context) -> list[Track]:
    """Road users large enough in the image to judge contact: far-away boxes
    overlap from perspective alone and their motion is mostly noise."""
    min_scale = ctx.rp.collision_min_scale * ctx.meta.height
    return [t for t in ctx.tracks if t.category != OBSTACLE and np.median(t.scale) >= min_scale]


def accident(ctx: Context) -> list[Event]:
    rp = ctx.rp
    users = _judgeable_users(ctx)
    events = []
    for a, b in _candidate_pairs(users):
        ia, ib = common_times(a, b)
        if len(ia) < 3:
            continue
        contact = _contact(a, ia, b, ib, rp.contact_depth_tol)
        t = a.t[ia]
        for i0, i1 in runs(contact):
            tc = float(t[i0])
            if t[i1] - t[i0] < 0.3:
                continue
            pre = (t >= tc - 1.0) & (t < tc)
            closing = closing_speed(a, ia[pre], b, ib[pre]).max() if pre.any() else 0.0
            drop = max(speed_drop(a, tc), speed_drop(b, tc))
            turn = max(swerve(a, tc), swerve(b, tc))
            fall = fell(a, tc) or fell(b, tc)
            stay = t[i1] - t[i0] >= rp.after_contact_slow_sec and (
                (a.speed[ia[i0:i1 + 1]] < 0.3).mean() > 0.6 or (b.speed[ib[i0:i1 + 1]] < 0.3).mean() > 0.6)
            ped = PERSON in (a.category, b.category)
            if ped:
                vehicle = b if a.category == PERSON else a
                # a car hitting a person moves and brakes hard; a box that only looks like a
                # fall is usually the person being hidden by a passing car
                if vehicle.speed[vehicle.at(tc)] < 1.0 or not fall or                         speed_drop(vehicle, tc) < 0.75 * rp.hard_decel:
                    continue
            else:
                abrupt = drop >= rp.hard_decel or turn >= rp.swerve_deg
                if closing < 1.0 or not abrupt or not stay:
                    continue                  # queues overlap but never close fast; crashes end at rest
                if not _pushed_if_standing(a, b, tc):
                    continue                  # rolling up behind a standing car is not an impact
            evidence = (0.35 * min(closing / 1.5, 1.0) + 0.3 * min(drop / rp.hard_decel, 1.0)
                        + 0.15 * min(turn / rp.swerve_deg, 1.0) + 0.25 * stay + 0.4 * fall)
            end = min(max(_settle_time(ctx, a, tc), _settle_time(ctx, b, tc)), tc + 60.0)
            events.append(Event(tc, max(end, tc + 1.0), "accident", min(1.0, evidence), [a.id, b.id]))
            break
    return events


def _pushed_if_standing(a: Track, b: Track, tc: float, push_bodies: float = 0.3) -> bool:
    """If one vehicle stood still before the contact, it must be shoved by it."""
    for tr in (a, b):
        before = _window(tr, tc - 1.0, tc)
        after = _window(tr, tc, tc + 1.5)
        if before.any() and after.any() and tr.speed[before].mean() < 0.3:
            i0, i1 = np.flatnonzero(after)[[0, -1]]
            moved = np.linalg.norm(tr.anchor[i1] - tr.anchor[i0]) / tr.scale[i0]
            if moved < push_bodies:
                return False
    return True


def _evasion_onset(tr: Track, tk: float, rp) -> float | None:
    """Start of hard braking (last moment at ~full speed) or of a swerve near tk."""
    if speed_drop(tr, tk + 0.5, before=1.0, after=1.5) >= rp.hard_decel:
        w = np.flatnonzero(_window(tr, tk - 1.5, tk + 1.0))
        sp = tr.speed[w]
        return float(tr.t[w[np.flatnonzero(sp >= 0.9 * sp.max())[-1]]])
    if swerve(tr, tk + 0.5, span=1.0) >= rp.swerve_deg:
        return max(tr.start, tk - 0.5)
    return None


def near_miss(ctx: Context, accidents: list[Event] | None = None) -> list[Event]:
    rp = ctx.rp
    users = _judgeable_users(ctx)
    crash_pairs = {tuple(sorted(e.track_ids)) for e in accidents or []}
    events = []
    for a, b in _candidate_pairs(users, pad_bodies=2.0):
        if tuple(sorted((a.id, b.id))) in crash_pairs:
            continue
        ia, ib = common_times(a, b)
        if len(ia) < 5:
            continue
        if _contact(a, ia, b, ib, rp.contact_depth_tol).sum() >= 2:
            continue
        ttc, miss = time_to_collision(a, ia, b, ib)
        dist = body_distance(a, ia, b, ib)
        # crossing paths only (both moving, headings well apart): following a car or
        # rolling up to a queue closes the gap too, and opposite carriageways look
        # like a head-on course once perspective squeezes the median
        moving = (a.speed[ia] > 1.0) & (b.speed[ib] > 1.0)
        cos = np.einsum("ij,ij->i", a.unit_dir()[ia], b.unit_dir()[ib])
        # vehicle-vehicle conflicts cross at a wide angle (narrower ones are lane changes/merges)
        min_angle = 30 if PERSON in (a.category, b.category) else 45
        crossing = moving & (np.abs(cos) < np.cos(np.deg2rad(min_angle)))
        big = np.minimum(a.scale[ia], b.scale[ib]) >= rp.collision_min_scale * ctx.meta.height
        danger = crossing & big & (ttc > 0) & (ttc < rp.near_miss_ttc) & (miss < 0.6) & (dist < 8.0)
        # a real conflict lasts; single samples are heading noise in dense, slow traffic
        sustained = [(i0, i1) for i0, i1 in runs(danger) if i1 - i0 + 1 >= 3]
        if not sustained:
            continue
        k = int(sustained[0][0])
        tk = float(a.t[ia[k]])
        # evasive action by either road user around the dangerous moment
        onsets = [_evasion_onset(tr, tk, rp) for tr in (a, b)]
        onsets = [o for o in onsets if o is not None]
        if not onsets:
            continue
        onset = min(onsets)
        # clear = separated again after the closest approach
        t_common = a.t[ia]
        horizon = np.flatnonzero((t_common >= tk) & (t_common <= tk + 6.0))
        k_min = horizon[int(np.argmin(dist[horizon]))]
        clear = np.flatnonzero((np.arange(len(dist)) > k_min) & (dist > 2.0))
        end = float(t_common[clear[0]]) if len(clear) else min(a.end, b.end)
        end = min(end, tk + 6.0)
        if end - onset >= 0.5:
            events.append(Event(onset, end, "near_miss", 0.6, [a.id, b.id]))
    return events
