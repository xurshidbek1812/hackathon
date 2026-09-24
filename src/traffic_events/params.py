"""All tunable numbers in one place.

Speeds are in "body units per second": pixel displacement of the ground point
divided by the object's apparent size sqrt(w*h). This keeps thresholds roughly
independent of perspective (far cars are small and move few pixels).
"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = ROOT / "weights"
CONFIG_DIR = ROOT / "configs"


@dataclass(frozen=True)
class DetectorParams:
    weights: str = "yolo11m.pt"
    imgsz: int = 960
    conf: float = 0.15            # low: the tracker's second stage uses weak boxes
    iou: float = 0.6
    batch: int = 8


@dataclass(frozen=True)
class TrackerParams:
    high_conf: float = 0.45
    low_conf: float = 0.15
    iou_gate: float = 0.15
    max_lost_sec: float = 1.5
    min_hits: int = 3
    stationary_link_gap_sec: float = 30.0   # re-join fragments of a parked vehicle
    stationary_link_iou: float = 0.5


@dataclass(frozen=True)
class MotionParams:
    smooth_sec: float = 0.6       # centred moving-average window (Part A only)
    stationary_speed: float = 0.15
    moving_speed: float = 0.5
    min_track_samples: int = 4


@dataclass(frozen=True)
class RuleParams:
    # stopped_vehicle
    stopped_min_sec: float = 10.0
    queue_release_window_sec: float = 4.0
    queue_neighbor_dist: float = 4.0        # body units
    # congestion
    congestion_min_vehicles: int = 4
    congestion_slow_speed: float = 0.35
    congestion_slow_frac: float = 0.75
    congestion_min_sec: float = 30.0
    congestion_merge_gap_sec: float = 10.0
    # wrong_way
    wrong_way_cos: float = -0.5
    wrong_way_min_sec: float = 1.5
    wrong_way_min_dist: float = 2.0         # body units travelled against the flow
    # u-turn / turns
    u_turn_min_deg: float = 150.0
    u_turn_max_sec: float = 20.0
    turn_min_deg: float = 45.0
    # line crossings
    line_cross_hold_sec: float = 0.8
    # pedestrians
    jaywalk_min_sec: float = 1.0
    road_margin: float = 0.15               # fraction of person height inside the road edge
    crossing_margin_px: float = 12.0
    # collisions
    contact_depth_tol: float = 0.35         # |y2a - y2b| / min(h)
    hard_decel: float = 2.0                 # body units / s lost within ~1 s
    swerve_deg: float = 30.0
    near_miss_ttc: float = 1.2
    after_contact_slow_sec: float = 2.0
    # obstacles / fire
    obstacle_min_sec: float = 2.0
    static_blob_min_sec: float = 8.0
    static_blob_min_area: float = 0.0006    # fraction of frame
    static_blob_max_area: float = 0.02
    fire_min_sec: float = 2.0
    fire_min_area: float = 0.0008
    smoke_min_sec: float = 4.0
    smoke_min_area: float = 0.004
    # post-processing
    min_event_sec: float = 0.8
    merge_gap_sec: float = 1.0


@dataclass(frozen=True)
class PipelineParams:
    stride: int = 2                         # process every Nth frame in Part A
    max_stride: int = 5
    budget_ratio_part_a: float = 1.2        # target wall time / video time for Part A
    feature_every_sec: float = 0.5          # background / fire sampling period
    feature_width: int = 320
    detector: DetectorParams = field(default_factory=DetectorParams)
    tracker: TrackerParams = field(default_factory=TrackerParams)
    motion: MotionParams = field(default_factory=MotionParams)
    rules: RuleParams = field(default_factory=RuleParams)


@dataclass(frozen=True)
class RiskParams:
    stride: int = 2
    max_stride: int = 6
    imgsz: int = 640
    budget_ratio: float = 1.0               # target wall time / video time for Part B
    history_sec: float = 2.0
    ema: float = 0.35
    decay_per_sec: float = 0.6              # how fast a peak fades when evidence disappears
    bias: float = -4.0
    w_ttc: float = 4.5
    w_brake: float = 1.8
    w_swerve: float = 1.0
    w_wrong_way: float = 1.5
    w_ped_road: float = 1.0
