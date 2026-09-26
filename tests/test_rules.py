"""Each rule on hand-built trajectories with a known answer."""
import numpy as np

from conftest import FPS, H, W, line_path, make_context, make_track, person_track
from traffic_events.rules.collision import accident, near_miss
from traffic_events.rules.motion import congestion, illegal_u_turn, solid_line_crossing, stopped_vehicle, wrong_way
from traffic_events.rules.pedestrian import failure_to_yield, jaywalking
from traffic_events.rules.signal import red_light, stop_line
from traffic_events.scene import Scene, StopLine
from traffic_events.signal_state import GREEN, RED, SignalTimeline


def n(sec):
    return int(round(sec * FPS))


def rightward_traffic(k=25, y=400.0):
    """k cars driving left -> right in one lane, staggered in time."""
    return [make_track(100 + i, line_path((50, y), (1200, y), n(8)), t0=i * 3.0) for i in range(k)]


def test_stopped_vehicle_found_with_correct_boundaries():
    path = np.concatenate([line_path((100, 500), (600, 500), n(5)),
                           np.repeat([[600, 500]], n(20), axis=0),
                           line_path((600, 500), (1100, 500), n(5))])
    tr = make_track(1, path)
    evs = stopped_vehicle(make_context([tr]))
    assert len(evs) == 1
    assert abs(evs[0].start - 5.0) < 0.6 and abs(evs[0].end - 25.0) < 0.6


def test_short_stop_is_not_a_stopped_vehicle():
    path = np.concatenate([line_path((100, 500), (600, 500), n(5)),
                           np.repeat([[600, 500]], n(6), axis=0),
                           line_path((600, 500), (1100, 500), n(5))])
    assert stopped_vehicle(make_context([make_track(1, path)])) == []


def test_wrong_way_against_learned_flow():
    flow = rightward_traffic()
    bad = make_track(1, line_path((1200, 400), (50, 400), n(8)), t0=30.0)
    evs = wrong_way(make_context(flow + [bad], flow_tracks=flow))
    assert [e.track_ids for e in evs] == [[1]]
    assert evs[0].start < 31.5


def test_no_wrong_way_for_normal_traffic():
    flow = rightward_traffic()
    assert wrong_way(make_context(flow, flow_tracks=flow)) == []


def test_u_turn():
    ang = np.linspace(-np.pi / 2, np.pi / 2, n(3))
    arc = np.column_stack([800 + 80 * np.cos(ang), 400 + 80 * np.sin(ang)])
    path = np.concatenate([line_path((200, 320), (800, 320), n(5)), arc, line_path((800, 480), (200, 480), n(5))])
    evs = illegal_u_turn(make_context([make_track(1, path)]))
    assert len(evs) == 1
    assert 4.0 < evs[0].start < 6.0 and 7.0 < evs[0].end < 9.5


def test_straight_driving_is_not_a_u_turn():
    assert illegal_u_turn(make_context(rightward_traffic(3))) == []


def test_congestion():
    tracks = [make_track(i, line_path((100 + 120 * i, 400), (130 + 120 * i, 400), n(60))) for i in range(6)]
    evs = congestion(make_context(tracks, duration=60))
    assert len(evs) == 1 and evs[0].end - evs[0].start > 50


def _road_scene():
    s = Scene(W, H)
    s.road = np.array([[0, 300], [W, 300], [W, 600], [0, 600]], np.float32)
    s.crossings = [np.array([[600, 300], [700, 300], [700, 600], [600, 600]], np.float32)]
    return s


def test_jaywalking_outside_crossing_only():
    scene = _road_scene()
    jay = person_track(1, line_path((300, 250), (300, 650), n(8)))
    legal = person_track(2, line_path((650, 250), (650, 650), n(8)))
    evs = jaywalking(make_context([jay, legal], scene=scene))
    assert [e.track_ids for e in evs] == [[1]]


def test_failure_to_yield():
    scene = _road_scene()
    ped = person_track(1, line_path((650, 280), (650, 620), n(10)))
    car = make_track(2, line_path((300, 450), (1000, 450), n(4)), t0=3.0)
    evs = failure_to_yield(make_context([ped, car], scene=scene))
    assert len(evs) == 1 and evs[0].track_ids == [2]


def test_solid_line_crossing():
    scene = Scene(W, H)
    scene.solid_lines = [np.array([[0, 400], [W, 400]], np.float32)]
    path = np.concatenate([line_path((100, 370), (500, 370), n(3)), line_path((500, 370), (800, 440), n(2)),
                           line_path((800, 440), (1200, 440), n(3))])
    evs = solid_line_crossing(make_context([make_track(1, path)], scene=scene))
    assert len(evs) == 1 and 2.0 < evs[0].start < 5.5


def _signal_scene():
    scene = Scene(W, H)
    scene.stop_lines = [StopLine("s", np.array([[600, 300], [600, 600]], np.float32),
                                 np.array([1.0, 0.0], np.float32), "main")]
    t = np.arange(0, 60, 0.08)
    sig = SignalTimeline(t, np.where(t < 30, RED, GREEN))
    return scene, {"main": sig}


def test_red_light_runner_detected_only_on_red():
    scene, signals = _signal_scene()
    runner = make_track(1, line_path((200, 450), (1100, 450), n(6)), t0=5.0)
    legal = make_track(2, line_path((200, 450), (1100, 450), n(6)), t0=35.0)
    evs = red_light(make_context([runner, legal], scene=scene, signals=signals, duration=60))
    assert [e.track_ids for e in evs] == [[1]]


def test_stop_line_violation_ends_at_green():
    scene, signals = _signal_scene()
    path = np.concatenate([line_path((200, 450), (610, 450), n(4)), np.repeat([[610, 450]], n(30), axis=0)])
    ctx = make_context([make_track(1, path, t0=2.0)], scene=scene, signals=signals, duration=60)
    evs = stop_line(ctx)
    assert len(evs) == 1 and abs(evs[0].end - 30.0) < 0.2
    assert red_light(ctx) == []          # it stopped at the line, it did not run the light


def test_accident_head_on():
    a_path = np.concatenate([line_path((200, 450), (590, 450), n(3)), np.repeat([[595, 450]], n(10), axis=0)])
    b_path = np.concatenate([line_path((1000, 450), (640, 450), n(3)), np.repeat([[635, 450]], n(10), axis=0)])
    evs = accident(make_context([make_track(1, a_path), make_track(2, b_path)]))
    assert len(evs) == 1 and 2.0 < evs[0].start < 3.2


def test_queue_contact_is_not_an_accident():
    a = make_track(1, np.repeat([[500, 450]], n(20), axis=0))
    b = make_track(2, np.repeat([[545, 452]], n(20), axis=0))
    assert accident(make_context([a, b])) == []


def test_near_miss_with_hard_braking():
    brake = np.concatenate([line_path((200, 450), (700, 450), n(2)), line_path((700, 450), (760, 450), n(1)),
                            np.repeat([[760, 450]], n(3), axis=0)])
    crosser = person_track(2, line_path((830, 250), (830, 700), n(4)), t0=0.48)
    evs = near_miss(make_context([make_track(1, brake), crosser]))
    assert len(evs) == 1


def _unsignalled_scene():
    scene = Scene(W, H)
    scene.stop_lines = [StopLine("s", np.array([[600, 300], [600, 600]], np.float32),
                                 np.array([1.0, 0.0], np.float32), None)]
    return scene


def _waiting_car(tid, y, until_sec):
    return make_track(tid, np.concatenate([line_path((300, y), (565, y), n(3)),
                                           np.repeat([[565, y]], n(until_sec - 3), axis=0)]))


def test_red_light_inferred_from_waiting_queue():
    scene = _unsignalled_scene()
    waiting = [_waiting_car(1, 350, 25), _waiting_car(4, 420, 25)]
    runner = make_track(2, line_path((200, 520), (1100, 520), n(6)), t0=10.0)
    later = make_track(3, line_path((200, 520), (1100, 520), n(6)), t0=40.0)   # nobody waiting: green
    evs = red_light(make_context(waiting + [runner, later], scene=scene, duration=60))
    assert [e.track_ids for e in evs] == [[2]]


def test_platoon_through_a_waiting_turn_lane_is_not_red_light():
    scene = _unsignalled_scene()
    waiting = [_waiting_car(1, 350, 40), _waiting_car(4, 420, 40)]           # e.g. a turn lane on red
    platoon = [make_track(10 + i, line_path((200, 520), (1100, 520), n(6)), t0=10.0 + 1.5 * i) for i in range(5)]
    assert red_light(make_context(waiting + platoon, scene=scene, duration=60)) == []


def test_stop_line_with_inferred_phase_ends_when_queue_moves():
    scene = _unsignalled_scene()
    waiting = [make_track(tid, np.concatenate([line_path((300, y), (565, y), n(3)),
                                               np.repeat([[565, y]], n(20), axis=0),
                                               line_path((565, y), (1100, y), n(3))]))
               for tid, y in ((1, 350), (3, 420))]
    over = make_track(2, np.concatenate([line_path((300, 520), (610, 520), n(4)),
                                         np.repeat([[610, 520]], n(19), axis=0),
                                         line_path((610, 520), (1100, 520), n(3))]))
    evs = stop_line(make_context(waiting + [over], scene=scene, duration=60))
    assert len(evs) == 1 and evs[0].track_ids == [2] and abs(evs[0].end - 23.0) < 0.5


def test_red_light_runner_that_pauses_before_entering():
    """Crosses on red, stops just past the line, then drives into the junction while
    the others are still waiting: red_light (not stop_line)."""
    scene = _unsignalled_scene()
    waiting = [_waiting_car(1, 350, 30), _waiting_car(4, 420, 30)]
    path = np.concatenate([line_path((200, 520), (630, 520), n(4)),       # crosses the line at ~t=10.6
                           np.repeat([[630, 520]], n(5), axis=0),          # pauses 5 s past the line
                           line_path((630, 520), (1100, 520), n(3))])      # enters the junction on red
    runner = make_track(2, path, t0=7.0)
    ctx = make_context(waiting + [runner], scene=scene, duration=60)
    assert [e.track_ids for e in red_light(ctx)] == [[2]]
    assert stop_line(ctx) == []
