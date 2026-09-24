"""Event rules. Each rule maps a Context to a list of scored Events and
switches itself off when the scene config lacks what it needs."""
from __future__ import annotations

from ..segments import Event
from .collision import accident, near_miss
from .common import Context
from .hazards import fire_smoke, road_obstacle
from .motion import congestion, illegal_turn, illegal_u_turn, solid_line_crossing, stopped_vehicle, wrong_way
from .pedestrian import failure_to_yield, jaywalking
from .signal import red_light, stop_line

RULES = {
    "accident": accident,
    "red_light": red_light,
    "wrong_way": wrong_way,
    "illegal_u_turn": illegal_u_turn,
    "stopped_vehicle": stopped_vehicle,
    "jaywalking": jaywalking,
    "failure_to_yield": failure_to_yield,
    "illegal_turn": illegal_turn,
    "solid_line_crossing": solid_line_crossing,
    "stop_line": stop_line,
    "congestion": congestion,
    "road_obstacle": road_obstacle,
    "fire_smoke": fire_smoke,
}

# A predicted class that never occurs in the test set costs a whole class of
# macro-F1, so each rule's events must clear this confidence to be reported.
MIN_SCORE = 0.5


def run_rules(ctx: Context) -> list[Event]:
    events: list[Event] = []
    for label, rule in RULES.items():
        if ctx.scene.class_enabled(label):
            events += rule(ctx)
    if ctx.scene.class_enabled("near_miss"):
        events += near_miss(ctx, [e for e in events if e.label == "accident"])
    return [e for e in events if e.score >= MIN_SCORE]


__all__ = ["Context", "RULES", "run_rules", "MIN_SCORE"]
