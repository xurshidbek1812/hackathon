"""Run the full pipeline on a folder of videos and export everything the
website needs; optionally (re)build the scene flow prior.

    python scripts/analyze_videos.py --videos samples --build-prior

Writes
    website/data/results/<video>.json   events, risk curve, tracks, counts
    website/data/frames/<video>.jpg     a reference frame (EDA, scene editor)
    website/data/index.json             list of analysed videos
    configs/flow_prior.npz              with --build-prior
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic_events.export import analysis_json, risk_curve  # noqa: E402
from traffic_events.flow import FlowField  # noqa: E402
from traffic_events.pipeline import analyze  # noqa: E402
from traffic_events.video import read_meta  # noqa: E402

DATA = ROOT / "website" / "data"


def save_reference_frame(path: Path, out: Path, width: int = 1280) -> None:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ok, frame = cap.read()
    cap.release()
    if ok:
        h = int(frame.shape[0] * width / frame.shape[1])
        cv2.imwrite(str(out), cv2.resize(frame, (width, h), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=str(ROOT / "samples"))
    ap.add_argument("--build-prior", action="store_true", help="save configs/flow_prior.npz from these videos")
    ap.add_argument("--no-risk", action="store_true", help="skip the (slow) Part B curve")
    args = ap.parse_args()

    (DATA / "results").mkdir(parents=True, exist_ok=True)
    (DATA / "frames").mkdir(parents=True, exist_ok=True)
    videos = sorted(Path(args.videos).glob("*.mp4"))
    prior = None
    index = []
    for path in videos:
        print(f"[analyze] {path.name}", flush=True)
        analysis = analyze(str(path))
        if args.build_prior:
            meta = read_meta(str(path))
            prior = prior or FlowField(meta.width, meta.height)
            prior.add_tracks(analysis.tracks)
        risk = [] if args.no_risk else risk_curve(str(path))
        result = analysis_json(analysis, risk, name=path.name)
        (DATA / "results" / f"{path.stem}.json").write_text(json.dumps(result, separators=(",", ":")))
        save_reference_frame(path, DATA / "frames" / f"{path.stem}.jpg")
        index.append({"video": path.name, "stem": path.stem, "duration": result["meta"]["duration"],
                      "n_events": len(result["events"]), "seconds": result["stats"]["seconds"]})
        print(f"  {len(result['events'])} events, {result['stats']['n_tracks']} tracks, "
              f"{result['stats']['seconds']}s", flush=True)
    (DATA / "index.json").write_text(json.dumps(index, indent=1))
    if prior is not None:
        prior.save()
        print("saved configs/flow_prior.npz")


if __name__ == "__main__":
    main()
