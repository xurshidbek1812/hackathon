"""Align each video's view to the reference view the scene config was drawn on.

The camera is "fixed", but between recordings it was nudged and zoomed a
little (up to ~4 % of the frame width across the sample videos). Every
polygon and line in configs/scene.json is in the coordinates of the reference
view (configs/reference_view.png: the middle frame of sample C3896, the view
the scene was drawn and checked on; C3897 has the identical framing), so a
video's own view is registered to it: SIFT keypoints on the static background,
ratio-test matching and a RANSAC homography. The result maps normalised [0, 1]
reference coordinates to normalised coordinates of the video.

Within one recording the view is stable (< 0.5 %), so one registration per video
is enough: Part B uses the first frame it receives (causal), Part A a frame
from the middle of the file.
"""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from .params import CONFIG_DIR

REFERENCE = CONFIG_DIR / "reference_view.png"
MAX_SHIFT = 0.15            # a larger apparent move means a bad match, not a nudged camera


class ViewTransform:
    """Homography between normalised reference coordinates and this video's."""

    def __init__(self, H: np.ndarray | None = None, inliers: int = 0):
        self.H = np.eye(3) if H is None else np.asarray(H, np.float64)
        self.H_inv = np.linalg.inv(self.H)
        self.inliers = inliers

    @property
    def is_identity(self) -> bool:
        return bool(np.allclose(self.H, np.eye(3)))

    @staticmethod
    def _map(H: np.ndarray, pts) -> np.ndarray:
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)

    def to_view(self, pts) -> np.ndarray:
        """Reference -> video (normalised)."""
        return self._map(self.H, pts)

    def to_reference(self, pts) -> np.ndarray:
        """Video -> reference (normalised)."""
        return self._map(self.H_inv, pts)

    def direction_to_view(self, at, d, eps: float = 0.01) -> np.ndarray:
        """Carry a direction attached to reference point `at` into the video view."""
        a, b = self.to_view([at, np.asarray(at) + eps * np.asarray(d)])
        return b - a

    def max_shift(self) -> float:
        grid = np.array([[x, y] for x in (0, .5, 1) for y in (0, .5, 1)], float)
        return float(np.linalg.norm(self.to_view(grid) - grid, axis=1).max())


@lru_cache(maxsize=1)
def _reference_features():
    ref = cv2.imread(str(REFERENCE))
    if ref is None:
        return None
    sift = cv2.SIFT_create(nfeatures=5000)
    kp, desc = sift.detectAndCompute(_prepare(ref), None)
    return ref.shape[1], ref.shape[0], np.float32([k.pt for k in kp]), desc


def _prepare(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.0).apply(gray)       # evening footage is darker


def estimate_view(frame: np.ndarray, min_inliers: int = 40) -> ViewTransform:
    """Register a frame of the video to the reference view; identity if unsure."""
    ref = _reference_features()
    if ref is None or frame is None:
        return ViewTransform()
    rw, rh, ref_pts, ref_desc = ref
    scale = rw / frame.shape[1]
    img = cv2.resize(frame, (rw, int(round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    kp, desc = cv2.SIFT_create(nfeatures=5000).detectAndCompute(_prepare(img), None)
    if desc is None or len(kp) < min_inliers:
        return ViewTransform()
    pairs = cv2.BFMatcher().knnMatch(ref_desc, desc, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < min_inliers:
        return ViewTransform()
    src = ref_pts[[m.queryIdx for m in good]]
    dst = np.float32([kp[m.trainIdx].pt for m in good])
    H_px, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    inliers = int(mask.sum()) if mask is not None else 0
    if H_px is None or inliers < min_inliers:
        return ViewTransform()
    ih, iw = img.shape[:2]
    H = np.diag([1 / iw, 1 / ih, 1.0]) @ H_px @ np.diag([rw, rh, 1.0])     # normalised -> normalised
    view = ViewTransform(H / H[2, 2], inliers)
    return view if view.max_shift() <= MAX_SHIFT else ViewTransform()


def estimate_view_from_video(path: str, at: float = 0.5) -> ViewTransform:
    """Part A may look anywhere in the file: use a frame from the middle, which
    skips a nudge of the camera right after recording starts (seen in C3896)."""
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(at * cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    ok, frame = cap.read()
    cap.release()
    return estimate_view(frame) if ok else ViewTransform()
