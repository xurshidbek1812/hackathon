import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from traffic_events.detector import PERSON, VEHICLE  # noqa: E402
from traffic_events.flow import FlowField  # noqa: E402
from traffic_events.params import MotionParams, RuleParams  # noqa: E402
from traffic_events.rules import Context  # noqa: E402
from traffic_events.scene import Scene  # noqa: E402
from traffic_events.tracks import Track  # noqa: E402
from traffic_events.video import VideoMeta  # noqa: E402

W, H, FPS = 1280, 720, 12.5


def make_track(tid, anchors, t0=0.0, size=(60, 40), category=VEHICLE, coco=2, dt=1 / FPS):
    """Track whose bottom-centre follows `anchors` (n, 2) with a fixed box size."""
    anchors = np.asarray(anchors, float)
    w, h = size
    t = t0 + np.arange(len(anchors)) * dt
    box = np.column_stack([anchors[:, 0] - w / 2, anchors[:, 1] - h, anchors[:, 0] + w / 2, anchors[:, 1]])
    return Track(tid, category, coco, t, box, box.copy())


def line_path(p0, p1, n):
    return np.linspace(p0, p1, n)


def person_track(tid, anchors, t0=0.0):
    return make_track(tid, anchors, t0, size=(20, 50), category=PERSON, coco=0)


def make_context(tracks, scene=None, duration=120.0, signals=None, flow_tracks=None):
    scene = scene or Scene(W, H)
    flow = FlowField(W, H)
    flow.add_tracks(flow_tracks if flow_tracks is not None else tracks)
    meta = VideoMeta("synthetic.mp4", 25.0, W, H, int(duration * 25))
    return Context(meta, scene, tracks, flow, signals or {}, None, RuleParams(), MotionParams())


@pytest.fixture
def ctx_factory():
    return make_context
