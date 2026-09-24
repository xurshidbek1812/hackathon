"""Traffic event detection and accident anticipation for a fixed CCTV camera.

Pipeline: YOLO detection -> ByteTrack-style tracking -> smoothed trajectories
-> scene-aware rules (one module per event family) -> segment post-processing.
"""

CLASSES = [
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
    "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "road_obstacle",
    "fire_smoke",
]
