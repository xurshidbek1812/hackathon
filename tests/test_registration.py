"""View registration: a known shift of the reference view is recovered."""
import cv2
import numpy as np

from traffic_events.registration import REFERENCE, estimate_view


def test_known_shift_and_zoom_is_recovered():
    ref = cv2.imread(str(REFERENCE))
    h, w = ref.shape[:2]
    # the camera zoomed out 3 % and panned right by 2 % of the width
    M = np.float32([[0.97, 0, 0.02 * w + 0.015 * w], [0, 0.97, 0.015 * h]])
    moved = cv2.warpAffine(ref, M, (w, h))
    view = estimate_view(cv2.resize(moved, (3840, 2160)))        # works at the video's resolution
    probe = np.array([[0.2, 0.3], [0.5, 0.5], [0.8, 0.7]])
    expected = probe * 0.97 + [0.035, 0.015]
    assert view.inliers > 100
    assert np.abs(view.to_view(probe) - expected).max() < 0.004    # within 0.4 % of the frame
    assert np.allclose(view.to_reference(view.to_view(probe)), probe, atol=1e-6)


def test_unrelated_picture_falls_back_to_identity():
    noise = np.random.default_rng(0).integers(0, 255, (540, 960, 3), dtype=np.uint8)
    assert estimate_view(noise).is_identity
