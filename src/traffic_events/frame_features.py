"""Pixel-level cues for classes a COCO detector cannot see: static obstacles,
fire and smoke.

During the detection pass we keep a small thumbnail every
`feature_every_sec` together with the boxes of tracked objects. Afterwards:

* background  = per-pixel median of all thumbnails (moving traffic vanishes);
* short-term  = median of the next few seconds of thumbnails;
* foreground  = |short-term - background|, restricted to the carriageway and
  with every detected road user masked out.

A foreground blob that persists is a static object nobody detected (debris,
a fallen load) -> road_obstacle. A grey, low-texture, growing blob is smoke.
Fire is found separately: saturated orange-yellow pixels that flicker, minus
pixels that are orange for most of the video (street lamps).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .params import RuleParams


@dataclass
class Blob:
    t: float
    box: np.ndarray          # thumbnail pixels x1 y1 x2 y2
    area: float              # fraction of frame
    gray: bool = False
    texture_ratio: float = 1.0


@dataclass
class FrameFeatures:
    thumb_w: int
    thumb_h: int
    t: list[float] = field(default_factory=list)
    thumbs: list[np.ndarray] = field(default_factory=list)
    boxes: list[np.ndarray] = field(default_factory=list)       # object boxes in thumbnail px
    fire_t: list[float] = field(default_factory=list)
    fire_masks: list[np.ndarray] = field(default_factory=list)  # packed bool masks


class FrameFeatureCollector:
    def __init__(self, width: int, height: int, every_sec: float = 0.5, thumb_w: int = 320):
        self.sx = thumb_w / width
        th = int(round(height * self.sx))
        self.f = FrameFeatures(thumb_w, th)
        self.every = every_sec
        self._next_t = 0.0

    def update(self, t: float, frame: np.ndarray, boxes: np.ndarray) -> None:
        small = None
        if t + 1e-6 >= self._next_t:
            small = cv2.resize(frame, (self.f.thumb_w, self.f.thumb_h), interpolation=cv2.INTER_AREA)
            self.f.t.append(t)
            self.f.thumbs.append(small)
            self.f.boxes.append(np.asarray(boxes, np.float32).reshape(-1, 4) * self.sx)
            self._next_t = t + self.every
        # fire needs a higher rate for flicker: every processed frame
        if small is None:
            small = cv2.resize(frame, (self.f.thumb_w, self.f.thumb_h), interpolation=cv2.INTER_AREA)
        self.f.fire_t.append(t)
        self.f.fire_masks.append(np.packbits(fire_color_mask(small)))

    def result(self) -> FrameFeatures:
        return self.f


def fire_color_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    b, g, r = bgr[..., 0].astype(int), bgr[..., 1].astype(int), bgr[..., 2].astype(int)
    return (h <= 22) & (s > 90) & (v > 190) & (r > g) & (g > b) & (r - b > 90)


def _box_mask(shape, boxes: np.ndarray, pad: float = 0.1) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    for x1, y1, x2, y2 in boxes:
        pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
        cv2.rectangle(m, (int(x1 - pw), int(y1 - ph)), (int(x2 + pw), int(y2 + ph)), 1, -1)
    return m


def _link_blobs(blobs_per_t: list[list[Blob]], max_gap: int = 2, min_iou: float = 0.3) -> list[list[Blob]]:
    """Greedy temporal linking of blobs by box IoU."""
    from .tracker import iou_matrix
    chains: list[list[Blob]] = []
    open_chains: list[tuple[int, list[Blob]]] = []    # (last step index, chain)
    for step, blobs in enumerate(blobs_per_t):
        used = set()
        still_open = []
        for last, chain in open_chains:
            if step - last > max_gap:
                chains.append(chain)
                continue
            if blobs:
                ious = iou_matrix(chain[-1].box[None], np.stack([b.box for b in blobs]))[0]
                for j in np.argsort(-ious):
                    if ious[j] < min_iou:
                        break
                    if j not in used:
                        used.add(j)
                        chain.append(blobs[j])
                        last = step
                        break
            still_open.append((last, chain))
        for j, b in enumerate(blobs):
            if j not in used:
                still_open.append((step, [b]))
        open_chains = still_open
    chains.extend(c for _, c in open_chains)
    return chains


def static_foreground_events(ff: FrameFeatures, road_mask: np.ndarray, rp: RuleParams,
                             window_sec: float = 4.0) -> tuple[list[tuple[float, float, float]],
                                                               list[tuple[float, float, float]]]:
    """Returns (obstacles, smoke) as lists of (start, end, score)."""
    n = len(ff.thumbs)
    if n < 10:
        return [], []
    stack = np.stack(ff.thumbs)
    background = np.median(stack[:: max(1, n // 400)], axis=0).astype(np.float32)
    bg_gray = cv2.cvtColor(background.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    frame_area = float(ff.thumb_w * ff.thumb_h)
    road = cv2.resize(road_mask.astype(np.uint8), (ff.thumb_w, ff.thumb_h), interpolation=cv2.INTER_NEAREST)
    t = np.asarray(ff.t)
    dt = float(np.median(np.diff(t))) if n > 1 else 1.0
    win = max(3, int(round(window_sec / dt)))
    kernel = np.ones((3, 3), np.uint8)

    blobs_per_t: list[list[Blob]] = []
    for i in range(0, n - win + 1):
        short = np.median(stack[i:i + win], axis=0).astype(np.float32)
        diff = np.linalg.norm(short - background, axis=2)
        diff -= np.median(diff[road > 0]) if road.any() else np.median(diff)   # global lighting drift
        fg = (diff > 40).astype(np.uint8) & road
        occupied = np.zeros_like(fg)
        for b in ff.boxes[i:i + win]:
            occupied |= _box_mask(fg.shape, b)
        fg &= 1 - occupied
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
        blobs = []
        short_u8 = short.astype(np.uint8)
        for k in range(1, num):
            x, y, w, h, a = stats[k]
            frac = a / frame_area
            if frac < rp.static_blob_min_area:
                continue
            region = labels == k
            hsv = cv2.cvtColor(short_u8[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)
            sat = float(hsv[..., 1][region[y:y + h, x:x + w]].mean())
            gray_now = cv2.cvtColor(short_u8[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
            tex_now = cv2.Laplacian(gray_now, cv2.CV_32F).var()
            tex_bg = cv2.Laplacian(bg_gray[y:y + h, x:x + w], cv2.CV_32F).var()
            blobs.append(Blob(float(t[i] + window_sec / 2), np.array([x, y, x + w, y + h], np.float32),
                              float(frac), gray=sat < 40, texture_ratio=float(tex_now / max(tex_bg, 1.0))))
        blobs_per_t.append(blobs)

    obstacles, smoke = [], []
    for chain in _link_blobs(blobs_per_t):
        start, end = chain[0].t - window_sec / 2, chain[-1].t + window_sec / 2
        dur = end - start
        areas = np.array([b.area for b in chain])
        gray_frac = np.mean([b.gray for b in chain])
        smooth_frac = np.mean([b.texture_ratio < 0.6 for b in chain])
        growing = areas[-max(1, len(areas) // 3):].mean() > 1.3 * areas[:max(1, len(areas) // 3)].mean()
        if (gray_frac > 0.7 and smooth_frac > 0.6 and growing and dur >= rp.smoke_min_sec
                and areas.max() >= rp.smoke_min_area):
            smoke.append((start, end, float(min(1.0, dur / 20))))
        elif dur >= rp.static_blob_min_sec and np.median(areas) <= rp.static_blob_max_area:
            obstacles.append((start, end, float(min(1.0, dur / 60) * 0.6)))
    return obstacles, smoke


def fire_events(ff: FrameFeatures, rp: RuleParams) -> list[tuple[float, float, float]]:
    if len(ff.fire_masks) < 10:
        return []
    npix = ff.thumb_h * ff.thumb_w

    def chunks(size=256):
        for i in range(0, len(ff.fire_masks), size):
            yield np.stack([np.unpackbits(m)[:npix] for m in ff.fire_masks[i:i + size]]).astype(bool)

    counts = sum(c.sum(axis=0, dtype=np.int64) for c in chunks())
    static = counts / len(ff.fire_masks) > 0.3    # lamps, painted signs, sunset glare
    area = np.concatenate([(c & ~static).mean(axis=1) for c in chunks()])
    t = np.asarray(ff.fire_t)
    dt = float(np.median(np.diff(t)))
    k = max(3, int(round(1.0 / dt)))
    # flicker: relative std of the fire area within a 1 s window
    pad = np.pad(area, (k // 2, k // 2), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(pad, k)[: len(area)]
    flicker = windows.std(axis=1) / np.maximum(windows.mean(axis=1), 1e-6)
    active = (area >= rp.fire_min_area) & (flicker > 0.08)
    from .segments import mask_to_segments
    return [(s, e, float(min(1.0, (e - s) / 10))) for s, e in
            mask_to_segments(t, active, min_dur=rp.fire_min_sec, max_gap=1.0)]
