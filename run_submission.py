"""PLACEHOLDER - replace with the official run_submission.py from the starter kit.

Re-implemented from the task PDF so the pipeline can be exercised end to end
before the starter kit is available. Behaviour follows the PDF:
walks a folder of videos, calls detect_events, streams every frame through
RiskEstimator, enforces 3x-duration time budget, drops malformed events and
writes predictions.json.
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import cv2

import solution


def clean_events(events, duration, log):
    kept, last_end = [], {}
    for ev in sorted(events, key=lambda e: e[0]):
        try:
            s, e, label = float(ev[0]), float(ev[1]), str(ev[2])
        except Exception:
            log(f"  drop malformed {ev!r}")
            continue
        if label not in solution.CLASSES or s >= e or s < 0 or (duration and e > duration + 1e-3):
            log(f"  drop invalid {ev!r}")
            continue
        if label in last_end and s < last_end[label]:
            log(f"  drop same-class overlap {ev!r}")
            continue
        last_end[label] = e
        kept.append([s, e, label])
    return kept


def run_video(path: Path, log):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    meta = {"video_id": path.name, "fps": fps, "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), "n_frames": n}
    cap.release()
    duration = n / fps
    t0 = time.perf_counter()
    try:
        events = clean_events(solution.detect_events(str(path)), duration, log)
    except Exception:
        log(traceback.format_exc())
        events = []
    risk = []
    try:
        est = solution.RiskEstimator()
        est.reset(meta)
        cap = cv2.VideoCapture(str(path))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = i / fps
            risk.append([round(t, 3), float(min(1.0, max(0.0, est.step(frame, t))))])
            i += 1
        cap.release()
    except Exception:
        log(traceback.format_exc())
        risk = []
    elapsed = time.perf_counter() - t0
    if elapsed > 3 * duration:
        log(f"  over time budget ({elapsed:.1f}s > {3 * duration:.1f}s): scored as empty")
        return {"events": [], "risk": []}, elapsed
    return {"events": events, "risk": risk}, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", default="predictions.json")
    ap.add_argument("--team", default="infinity")
    args = ap.parse_args()
    out = {"team": args.team, "videos": {}}
    for path in sorted(Path(args.videos).glob("*.mp4")):
        print(f"[run] {path.name}")
        result, elapsed = run_video(path, print)
        print(f"  {len(result['events'])} events, {elapsed:.1f}s")
        out["videos"][path.name] = result
    Path(args.out).write_text(json.dumps(out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
