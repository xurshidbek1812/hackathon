"""Scene layout from configs/scene.json, converted to pixel coordinates.

All coordinates in the JSON are normalised to [0, 1] so the config survives a
resolution change. A key that is missing or null means "unknown": rules that
depend on it switch themselves off instead of guessing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .params import CONFIG_DIR


def _poly(points, w: int, h: int) -> np.ndarray:
    return (np.asarray(points, dtype=np.float32) * [w, h]).astype(np.float32)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


@dataclass
class StopLine:
    name: str
    line: np.ndarray            # (2, 2) pixel endpoints
    approach: np.ndarray        # unit travel direction of vehicles that must stop
    signal: str | None


@dataclass
class Lane:
    name: str
    polygon: np.ndarray
    direction: np.ndarray       # unit vector, image coordinates


@dataclass
class Scene:
    width: int
    height: int
    road: np.ndarray | None = None                   # carriageway polygon
    exclude: list[np.ndarray] = field(default_factory=list)
    crossings: list[np.ndarray] | None = None        # None = unknown
    lanes: list[Lane] = field(default_factory=list)
    carriageways: list[np.ndarray] = field(default_factory=list)  # one per travel direction
    solid_lines: list[np.ndarray] = field(default_factory=list)
    stop_lines: list[StopLine] = field(default_factory=list)
    intersection: np.ndarray | None = None
    signals: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    zones: dict[str, np.ndarray] = field(default_factory=dict)
    prohibited_turns: list[tuple[str, str]] = field(default_factory=list)
    u_turn_allowed: list[np.ndarray] = field(default_factory=list)
    u_turn_zone: np.ndarray | None = None          # where a U-turn is physically possible
    enabled_classes: list[str] | None = None
    _road_mask: np.ndarray | None = None

    # ------------------------------------------------------------------ geometry
    @staticmethod
    def inside(poly: np.ndarray, pts: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Boolean per point: inside `poly` by at least `margin` pixels."""
        pts = np.atleast_2d(np.asarray(pts, dtype=np.float32))
        c = poly.reshape(-1, 1, 2)
        d = np.array([cv2.pointPolygonTest(c, (float(x), float(y)), True) for x, y in pts])
        return d >= margin

    def on_road(self, pts: np.ndarray, margin=0.0) -> np.ndarray:
        pts = np.atleast_2d(pts)
        if self.road is not None:
            ok = self.inside(self.road, pts, margin)
        elif self._road_mask is not None:
            ok = self._mask_lookup(self._road_mask, pts)
        else:
            ok = np.ones(len(pts), bool)
        for ex in self.exclude:
            ok &= ~self.inside(ex, pts)
        return ok

    def in_crossing(self, pts: np.ndarray, margin=0.0) -> np.ndarray:
        pts = np.atleast_2d(pts)
        out = np.zeros(len(pts), bool)
        for c in self.crossings or []:
            out |= self.inside(c, pts, -margin)
        return out

    def _mask_lookup(self, mask: np.ndarray, pts: np.ndarray) -> np.ndarray:
        mh, mw = mask.shape
        xs = np.clip((pts[:, 0] / self.width * mw).astype(int), 0, mw - 1)
        ys = np.clip((pts[:, 1] / self.height * mh).astype(int), 0, mh - 1)
        return mask[ys, xs] > 0

    def set_learned_road_mask(self, mask: np.ndarray) -> None:
        """Fallback carriageway from where vehicles actually drive (flow field)."""
        self._road_mask = mask

    @property
    def has_road(self) -> bool:
        return self.road is not None or self._road_mask is not None

    def road_mask(self, w: int, h: int) -> np.ndarray:
        """uint8 mask of the carriageway at an arbitrary resolution."""
        if self.road is not None:
            m = np.zeros((h, w), np.uint8)
            cv2.fillPoly(m, [(self.road * [w / self.width, h / self.height]).astype(np.int32)], 1)
        elif self._road_mask is not None:
            m = cv2.resize(self._road_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            m = np.ones((h, w), np.uint8)
        for ex in self.exclude:
            cv2.fillPoly(m, [(ex * [w / self.width, h / self.height]).astype(np.int32)], 0)
        return m

    def lane_direction(self, pt) -> np.ndarray | None:
        for lane in self.lanes:
            if self.inside(lane.polygon, pt)[0]:
                return lane.direction
        return None

    def zone_of(self, pts: np.ndarray) -> list[str | None]:
        out: list[str | None] = [None] * len(pts)
        for name, poly in self.zones.items():
            hit = self.inside(poly, pts)
            for i in np.flatnonzero(hit):
                if out[i] is None:
                    out[i] = name
        return out

    def class_enabled(self, label: str) -> bool:
        return self.enabled_classes is None or label in self.enabled_classes


def load_scene(width: int, height: int, path: str | Path | None = None, view=None) -> Scene:
    """Scene in this video's pixels. `view` (registration.ViewTransform) carries
    the reference-view coordinates of the config into this video's view."""
    path = Path(path) if path else CONFIG_DIR / "scene.json"
    cfg = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    w, h = width, height
    warp = (lambda pts: view.to_view(pts)) if view is not None else (lambda pts: np.asarray(pts, float))
    P = lambda pts: _poly(warp(pts), w, h)  # noqa: E731

    def D(at, d):
        """A direction attached to reference point `at`, as a unit vector in video pixels."""
        d = view.direction_to_view(at, d) if view is not None else np.asarray(d, float)
        return _unit(np.asarray(d) * [w, h])

    scene = Scene(width=w, height=h)
    if cfg.get("road"):
        scene.road = P(cfg["road"])
    scene.exclude = [P(p) for p in cfg.get("exclude") or []]
    if cfg.get("crossings") is not None:
        scene.crossings = [P(p) for p in cfg["crossings"]]
    scene.lanes = [Lane(l.get("name", f"lane{i}"), P(l["polygon"]),
                        D(np.mean(l["polygon"], axis=0), l["direction"]))
                   for i, l in enumerate(cfg.get("lanes") or [])]
    scene.carriageways = [P(p) for p in cfg.get("carriageways") or []]
    scene.solid_lines = [P(l) for l in cfg.get("solid_lines") or []]
    scene.stop_lines = [StopLine(s.get("name", f"stop{i}"), P(s["line"]),
                                 D(np.mean(s["line"], axis=0), s["approach"]), s.get("signal"))
                        for i, s in enumerate(cfg.get("stop_lines") or [])]
    if cfg.get("intersection"):
        scene.intersection = P(cfg["intersection"])
    scene.signals = {}
    for name, lamps in (cfg.get("signals") or {}).items():
        boxes = {}
        for lamp, (x1, y1, x2, y2) in lamps.items():
            corners = warp([[x1, y1], [x2, y2]]) * [w, h]
            boxes[lamp] = np.float32([*corners.min(axis=0), *corners.max(axis=0)])
        scene.signals[name] = boxes
    scene.zones = {k: P(v) for k, v in (cfg.get("zones") or {}).items()}
    scene.prohibited_turns = [tuple(p) for p in cfg.get("prohibited_turns") or []]
    scene.u_turn_allowed = [P(p) for p in cfg.get("u_turn_allowed") or []]
    if cfg.get("u_turn_zone"):
        scene.u_turn_zone = P(cfg["u_turn_zone"])
    scene.enabled_classes = cfg.get("enabled_classes")
    return scene
