"""Tracker, segment post-processing and the local metric."""
import numpy as np

import evaluate
from traffic_events.segments import Event, finalize, mask_to_segments
from traffic_events.tracker import ByteTracker


def det(x, y, conf=0.9, cat=0, coco=2, w=60, h=40):
    return [x, y, x + w, y + h, conf, cat, coco]


def test_tracker_keeps_ids_for_two_moving_objects():
    tr = ByteTracker()
    ids = []
    for k in range(20):
        out = tr.update(np.array([det(100 + 10 * k, 100), det(900 - 10 * k, 400)]), t=k * 0.08)
        ids.append(sorted(o[0] for o in out))
    assert all(i == ids[0] for i in ids)


def test_tracker_survives_short_occlusion():
    tr = ByteTracker()
    for k in range(10):
        tr.update(np.array([det(100 + 10 * k, 100)]), t=k * 0.08)
    for k in range(10, 14):
        tr.update(np.zeros((0, 7)), t=k * 0.08)
    out = tr.update(np.array([det(100 + 10 * 14, 100)]), t=14 * 0.08)
    assert [o[0] for o in out] == [1]


def test_tracker_never_turns_a_person_into_a_car():
    tr = ByteTracker()
    for k in range(5):
        tr.update(np.array([det(100, 100, cat=2, coco=0)]), t=k * 0.08)
    out = tr.update(np.array([det(100, 100, cat=0, coco=2)]), t=0.4)
    assert out[0][0] != 1


def test_mask_to_segments_merges_gaps_and_drops_blips():
    t = np.arange(0, 10, 0.5)
    m = np.zeros(len(t), bool)
    m[2:6] = m[7:10] = True        # 1.0-2.5 and 3.5-4.5 (gap 1.0)
    m[15] = True                   # blip
    assert mask_to_segments(t, m, min_dur=0.5, max_gap=1.0) == [(1.0, 4.5)]


def test_finalize_unions_same_class_and_clips():
    evs = [Event(1, 5, "jaywalking"), Event(4, 8, "jaywalking"), Event(3, 6, "accident"),
           Event(9, 9.2, "accident"), Event(58, 70, "congestion")]
    out = [e.as_list() for e in finalize(evs, duration=60, min_dur=0.5, merge_gap=0.5)]
    assert out == [[1.0, 8.0, "jaywalking"], [3.0, 6.0, "accident"], [58.0, 60.0, "congestion"]]


def test_metric_perfect_prediction_scores_one():
    gt = {"v.mp4": {"duration": 60, "fps": 25, "events": [[10, 15, "accident"], [30, 40, "red_light"]]}}
    risk = [[t, 1.0 if 7 <= t < 10 else 0.0] for t in np.arange(0, 60, 0.04)]
    pred = {"videos": {"v.mp4": {"events": gt["v.mp4"]["events"], "risk": risk}}}
    a, _ = evaluate.score_a(pred, gt)
    b, parts = evaluate.score_b(pred, gt)
    assert a == 1.0
    assert parts["AP"] > 0.99 and parts["F1_alarm"] == 1.0 and abs(parts["mTTA"] - 3.0) < 0.05


def test_metric_false_class_is_penalised():
    gt = {"v.mp4": {"events": [[10, 15, "accident"]]}}
    pred = {"videos": {"v.mp4": {"events": [[10, 15, "accident"], [20, 25, "fire_smoke"]], "risk": []}}}
    a, table = evaluate.score_a(pred, gt)
    assert a == 0.5 and table["fire_smoke"] == [0.0, 0.0, 0.0]
