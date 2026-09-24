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

from ..detector import PERSON
from ..segments import Event
from ..tracks import Track
from .common import Context, body_distance, common_times, runs


def _extent(tr: Track) -> np.ndarray:
    return np.array([tr.box[:, 0].min(), tr.box[:, 1].min(), tr.box[:, 2].max(), tr.box[:, 3].max()])


def _candidate_pairs(tracks: list[Track]):
    ext = {tr.id: _extent(tr) for tr in tracks}
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


def accident(ctx: Context) -> list[Event]:
    rp = ctx.rp
    users = [t for t in ctx.tracks if t.category != 3]
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
            evidence = (0.35 * min(closing / 1.5, 1.0) + 0.3 * min(drop / rp.hard_decel, 1.0)
                        + 0.15 * min(turn / rp.swerve_deg, 1.0) + 0.25 * stay + 0.4 * fall)
            if closing < 0.8 and not fall:
                continue                      # bumper-to-bumper queues overlap but never close fast
            if evidence < 0.6:
                continue
            end = min(max(_settle_time(ctx, a, tc), _settle_time(ctx, b, tc)), tc + 60.0)
            events.append(Event(tc, max(end, tc + 1.0), "accident", min(1.0, evidence), [a.id, b.id]))
            break
    return events


def _evasion_onset(tr: Track, tk: float, rp) -> float | None:
    """Start of hard braking (last moment at ~full speed) or of a swerve near tk."""
    if speed_drop(tr, tk + 0.5, before=1.0, after=1.5) >= rp.hard_decel * 0.75:
        w = np.flatnonzero(_window(tr, tk - 1.5, tk + 1.0))
        sp = tr.speed[w]
        return float(tr.t[w[np.flatnonzero(sp >= 0.9 * sp.max())[-1]]])
    if swerve(tr, tk + 0.5, span=1.0) >= rp.swerve_deg:
        return max(tr.start, tk - 0.5)
    return None


def near_miss(ctx: Context, accidents: list[Event] | None = None) -> list[Event]:
    rp = ctx.rp
    users = [t for t in ctx.tracks if t.category != 3]
    crash_pairs = {tuple(sorted(e.track_ids)) for e in accidents or []}
    events = []
    for a, b in _candidate_pairs(users):
        if tuple(sorted((a.id, b.id))) in crash_pairs:
            continue
        ia, ib = common_times(a, b)
        if len(ia) < 5:
            continue
        if _contact(a, ia, b, ib, rp.contact_depth_tol).sum() >= 2:
            continue
        ttc, miss = time_to_collision(a, ia, b, ib)
        dist = body_distance(a, ia, b, ib)
        danger = (ttc > 0) & (ttc < rp.near_miss_ttc) & (miss < 1.0) & (dist < 4.0)
        if not danger.any():
            continue
        k = int(np.flatnonzero(danger)[0])
        tk = float(a.t[ia[k]])
        # evasive action by either road user around the dangerous moment
        onsets = [_evasion_onset(tr, tk, rp) for tr in (a, b)]
        onsets = [o for o in onsets if o is not None]
        if not onsets:
            continue
        onset = min(onsets)
        clear = np.flatnonzero((a.t[ia] > tk) & (dist > 2.0))
        end = float(a.t[ia[clear[0]]]) if len(clear) else min(a.end, b.end)
        end = min(end, tk + 6.0)
        if end - onset >= 0.5:
            events.append(Event(onset, end, "near_miss", 0.6, [a.id, b.id]))
    return events
