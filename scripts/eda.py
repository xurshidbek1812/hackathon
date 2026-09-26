"""Exploratory data analysis of the sample videos for the website.

Run after scripts/analyze_videos.py (it reuses the exported tracks):

    python scripts/eda.py --videos samples

Writes website/data/eda/eda.json plus PNG overlays (heatmaps, trajectories,
flow field) per video.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic_events.flow import FlowField  # noqa: E402

DATA = ROOT / "website" / "data"
OUT = DATA / "eda"


def video_properties(path: Path, sample_every_sec: float = 5.0) -> dict:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    brightness, contrast, times = [], [], []
    step = max(1, int(fps * sample_every_sec))
    for i in range(0, n, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
        times.append(round(i / fps, 1))
        brightness.append(round(float(g.mean()), 1))
        contrast.append(round(float(g.std()), 1))
    cap.release()
    size_mb = path.stat().st_size / 2**20
    duration = n / fps
    mean_b = float(np.mean(brightness)) if brightness else 0.0
    return {
        "video": path.name, "width": w, "height": h, "fps": round(fps, 2), "n_frames": n,
        "duration": round(duration, 1), "size_mb": round(size_mb, 1),
        "bitrate_mbps": round(size_mb * 8 / max(duration, 1e-6), 2),
        "codec": "".join(chr((fourcc >> 8 * k) & 0xFF) for k in range(4)).strip(),
        "lighting": "night" if mean_b < 45 else "dusk/dawn" if mean_b < 70 else "day",   # tuned on the 4K sample
        "brightness": {"t": times, "mean": brightness, "std": contrast},
    }


def heatmap_overlay(frame: np.ndarray, pts: np.ndarray, weights=None, sigma: float = 12) -> np.ndarray:
    h, w = frame.shape[:2]
    acc = np.zeros((h, w), np.float32)
    if len(pts):
        xs = np.clip(pts[:, 0].astype(int), 0, w - 1)
        ys = np.clip(pts[:, 1].astype(int), 0, h - 1)
        np.add.at(acc, (ys, xs), 1.0 if weights is None else weights)
    acc = cv2.GaussianBlur(acc, (0, 0), sigma)
    if acc.max() > 0:
        acc = np.sqrt(acc / acc.max())
    color = cv2.applyColorMap((acc * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    alpha = np.clip(acc * 1.4, 0, 0.85)[..., None]
    base = (frame * 0.55).astype(np.float32)
    return (base * (1 - alpha) + color * alpha).astype(np.uint8)


def trajectories_overlay(frame: np.ndarray, tracks: list[dict], sx: float) -> np.ndarray:
    img = (frame * 0.5).astype(np.uint8)
    for tr in tracks:
        if tr["cat"] not in ("vehicle", "bike") or len(tr["t"]) < 4:
            continue
        b = np.asarray(tr["box"], float) * sx
        pts = np.column_stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]])
        d = pts[-1] - pts[0]
        if np.linalg.norm(d) < 20:
            continue
        hue = int((np.degrees(np.arctan2(d[1], d[0])) % 360) / 2)
        col = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        cv2.polylines(img, [pts.astype(np.int32)], False, col, 1, cv2.LINE_AA)
    return img


def flow_overlay(frame: np.ndarray, field: FlowField) -> np.ndarray:
    img = (frame * 0.55).astype(np.uint8)
    grid = field.direction_grid()
    gh, gw = grid.shape[:2]
    h, w = img.shape[:2]
    cw, ch = w / gw, h / gh
    for gy in range(gh):
        for gx in range(gw):
            d = grid[gy, gx]
            if np.isnan(d[0]):
                continue
            c = np.array([(gx + 0.5) * cw, (gy + 0.5) * ch])
            hue = int((np.degrees(np.arctan2(d[1], d[0])) % 360) / 2)
            col = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
            cv2.arrowedLine(img, tuple((c - d * cw * 0.4).astype(int)), tuple((c + d * cw * 0.4).astype(int)),
                            col, 2, cv2.LINE_AA, tipLength=0.4)
    return img


def per_minute(counts: dict[str, list[int]]) -> dict[str, list[float]]:
    out = {}
    for k, v in counts.items():
        v = np.asarray(v, float)
        m = int(np.ceil(len(v) / 60))
        out[k] = [round(float(v[i * 60:(i + 1) * 60].mean()), 2) for i in range(m)]
    return out


def speed_stats(tracks: list[dict], sx: float) -> list[float]:
    speeds = []
    for tr in tracks:
        if tr["cat"] != "vehicle" or len(tr["t"]) < 4:
            continue
        b = np.asarray(tr["box"], float)
        t = np.asarray(tr["t"])
        pts = np.column_stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]])
        scale = np.sqrt((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).mean()
        dist = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
        if t[-1] > t[0]:
            speeds.append(round(float(dist / scale / (t[-1] - t[0])), 2))
    return speeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=str(ROOT / "samples"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"videos": []}
    for path in sorted(p for p in Path(args.videos).iterdir() if p.suffix.lower() == ".mp4"):
        print(f"[eda] {path.name}", flush=True)
        props = video_properties(path)
        res_path = DATA / "results" / f"{path.stem}.json"
        frame_path = DATA / "frames" / f"{path.stem}.jpg"
        entry = {"props": props}
        if res_path.exists() and frame_path.exists():
            res = json.loads(res_path.read_text())
            frame = cv2.imread(str(frame_path))
            sx = frame.shape[1] / res["meta"]["width"]
            tracks = res["tracks"]

            def anchors(cats, moving_only=False):
                pts = []
                for tr in tracks:
                    if tr["cat"] in cats:
                        b = np.asarray(tr["box"], float) * sx
                        pts.append(np.column_stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]]))
                return np.concatenate(pts) if pts else np.zeros((0, 2))

            for name, img in {
                "vehicles_heatmap": heatmap_overlay(frame, anchors(("vehicle", "bike"))),
                "pedestrian_heatmap": heatmap_overlay(frame, anchors(("person",)), sigma=8),
                "trajectories": trajectories_overlay(frame, tracks, sx),
            }.items():
                cv2.imwrite(str(OUT / f"{path.stem}_{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            entry["counts_per_minute"] = per_minute(res["counts"])
            entry["counts_per_second"] = res["counts"]
            entry["speeds"] = speed_stats(tracks, sx)
            entry["n_tracks"] = {c: sum(t["cat"] == c for t in tracks) for c in ("vehicle", "bike", "person", "obstacle")}
            entry["events_by_class"] = {}
            for ev in res["events"]:
                entry["events_by_class"][ev[2]] = entry["events_by_class"].get(ev[2], 0) + 1
            if FlowField.load_prior(res["meta"]["width"], res["meta"]["height"]) is not None:
                prior = FlowField.load_prior(frame.shape[1], frame.shape[0])
                cv2.imwrite(str(OUT / "flow_field.jpg"), flow_overlay(frame, prior), [cv2.IMWRITE_JPEG_QUALITY, 85])
                report["flow_field"] = "data/eda/flow_field.jpg"
        report["videos"].append(entry)
    (OUT / "eda.json").write_text(json.dumps(report, separators=(",", ":")))
    print(f"wrote {OUT / 'eda.json'}")


if __name__ == "__main__":
    main()
