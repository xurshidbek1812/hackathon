"""YOLO (COCO) detector wrapper returning plain arrays.

Output rows: x1, y1, x2, y2, conf, category, coco_cls
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

from .params import WEIGHTS_DIR, DetectorParams

os.environ.setdefault("YOLO_OFFLINE", "1")          # never try to reach the internet
os.environ.setdefault("YOLO_VERBOSE", "False")

# Our coarse categories.
VEHICLE, BIKE, PERSON, OBSTACLE = 0, 1, 2, 3
CATEGORY_NAMES = {VEHICLE: "vehicle", BIKE: "bike", PERSON: "person", OBSTACLE: "obstacle"}

COCO_TO_CATEGORY = {
    2: VEHICLE, 5: VEHICLE, 7: VEHICLE,                 # car, bus, truck
    1: BIKE, 3: BIKE,                                    # bicycle, motorcycle
    0: PERSON,
    15: OBSTACLE, 16: OBSTACLE, 17: OBSTACLE, 18: OBSTACLE, 19: OBSTACLE,  # cat dog horse sheep cow
    24: OBSTACLE, 26: OBSTACLE, 28: OBSTACLE, 56: OBSTACLE,               # backpack handbag suitcase chair
}
COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
              15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow", 24: "backpack",
              26: "handbag", 28: "suitcase", 56: "chair"}


def _containment(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fraction of each box in `a` covered by each box in `b` -> (len(a), len(b))."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    return inter / np.maximum(area_a[:, None], 1e-6)


def suppress_riders_and_occupants(dets: np.ndarray) -> np.ndarray:
    """Drop person boxes that are riders of a bike or visible inside a vehicle.

    Without this every motorcyclist would look like a pedestrian on the road.
    """
    if len(dets) == 0:
        return dets
    persons = dets[:, 5] == PERSON
    if not persons.any():
        return dets
    p = dets[persons, :4]
    keep_p = np.ones(len(p), bool)
    bikes = dets[dets[:, 5] == BIKE, :4]
    if len(bikes):
        # rider: the bike sits in the lower part of the person box, horizontally overlapping
        cov = _containment(bikes, p)                    # bike covered by person box
        keep_p &= ~(cov.max(axis=0) > 0.3)
    vehicles = dets[dets[:, 5] == VEHICLE, :4]
    if len(vehicles):
        keep_p &= ~(_containment(p, vehicles).max(axis=1) > 0.7)
    keep = np.ones(len(dets), bool)
    keep[np.flatnonzero(persons)[~keep_p]] = False
    return dets[keep]


class Detector:
    def __init__(self, params: DetectorParams = DetectorParams()):
        import torch
        from ultralytics import YOLO

        torch.backends.cudnn.benchmark = False
        torch.manual_seed(0)
        self.params = params
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        path = WEIGHTS_DIR / params.weights
        self.model = YOLO(str(path if path.exists() else params.weights))
        self.classes = sorted(COCO_TO_CATEGORY)

    def __call__(self, frames: list[np.ndarray], imgsz: int | None = None) -> list[np.ndarray]:
        if not frames:
            return []
        results = self.model.predict(
            frames, imgsz=imgsz or self.params.imgsz, conf=self.params.conf, iou=self.params.iou,
            classes=self.classes, device=self.device, half=self.device != "cpu", verbose=False)
        out = []
        for r in results:
            b = r.boxes
            if b is None or len(b) == 0:
                out.append(np.zeros((0, 7), np.float32))
                continue
            xyxy = b.xyxy.cpu().numpy()
            conf = b.conf.cpu().numpy()
            cls = b.cls.cpu().numpy().astype(int)
            cat = np.array([COCO_TO_CATEGORY[c] for c in cls])
            dets = np.column_stack([xyxy, conf, cat, cls]).astype(np.float32)
            out.append(suppress_riders_and_occupants(dets))
        return out


@lru_cache(maxsize=1)
def get_detector() -> Detector:
    """One model instance shared by Part A and Part B (weights loaded once)."""
    return Detector()
