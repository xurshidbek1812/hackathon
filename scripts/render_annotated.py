"""Render an annotated copy of a video from its exported analysis JSON:
boxes with track ids, short trails, active events, the risk bar, and the
event timeline. Encodes browser-playable H.264 when imageio-ffmpeg is present.

    python scripts/render_annotated.py samples/clip.mp4 --out website/data/videos/clip.mp4
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "website" / "data"

COLORS = {"vehicle": (80, 200, 255), "bike": (255, 200, 80), "person": (120, 255, 120), "obstacle": (60, 60, 255)}
EVENT_COLOR = (40, 40, 230)


class TrackIndex:
    """Box of every track at an arbitrary time (nearest sample within 0.25 s)."""

    def __init__(self, tracks):
        self.tracks = [(tr, np.asarray(tr["t"]), np.asarray(tr["box"], float)) for tr in tracks]

    def at(self, t: float):
        for tr, ts, boxes in self.tracks:
            if ts[0] - 0.25 <= t <= ts[-1] + 0.25:
                i = int(np.clip(np.searchsorted(ts, t), 0, len(ts) - 1))
                if abs(ts[i] - t) <= 0.25:
                    trail = boxes[max(0, i - 12):i + 1]
                    yield tr, boxes[i], trail


def draw_frame(frame, t, index, events, risk_t, risk_v, scale, duration):
    for tr, box, trail in index.at(t):
        col = COLORS.get(tr["cat"], (255, 255, 255))
        x1, y1, x2, y2 = (box * scale).astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        cv2.putText(frame, f"{tr['name']} {tr['id']}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1,
                    cv2.LINE_AA)
        pts = np.column_stack([(trail[:, 0] + trail[:, 2]) / 2, trail[:, 3]]) * scale
        cv2.polylines(frame, [pts.astype(np.int32)], False, col, 1, cv2.LINE_AA)
    h, w = frame.shape[:2]
    active = [e for e in events if e[0] <= t <= e[1]]
    for k, ev in enumerate(active):
        label = f"{ev[2].upper().replace('_', ' ')}  {ev[0]:.1f}-{ev[1]:.1f}s"
        cv2.rectangle(frame, (10, 10 + 28 * k), (20 + 11 * len(label), 34 + 28 * k), EVENT_COLOR, -1)
        cv2.putText(frame, label, (16, 28 + 28 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        for tid in ev[4] if len(ev) > 4 else []:
            for tr, box, _ in index.at(t):
                if tr["id"] == tid:
                    x1, y1, x2, y2 = (box * scale).astype(int)
                    cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), EVENT_COLOR, 3)
    # risk bar
    if len(risk_t):
        r = float(np.interp(t, risk_t, risk_v))
        bw = int(w * 0.25)
        cv2.rectangle(frame, (w - bw - 12, 12), (w - 12, 30), (40, 40, 40), -1)
        col = (60, 200, 60) if r < 0.3 else (0, 200, 255) if r < 0.5 else (40, 40, 230)
        cv2.rectangle(frame, (w - bw - 12, 12), (w - bw - 12 + int(bw * r), 30), col, -1)
        cv2.putText(frame, f"risk {r:.2f}", (w - bw - 8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)
    # timeline strip
    y0 = h - 14
    cv2.rectangle(frame, (0, y0), (w, h), (30, 30, 30), -1)
    for ev in events:
        a, b = int(ev[0] / duration * w), max(int(ev[1] / duration * w), int(ev[0] / duration * w) + 2)
        cv2.rectangle(frame, (a, y0 + 2), (b, h - 2), EVENT_COLOR, -1)
    cx = int(t / duration * w)
    cv2.line(frame, (cx, y0), (cx, h), (255, 255, 255), 2)
    return frame


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def render(video: Path, result: dict, out: Path, width: int = 960) -> None:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = width / src_w
    size = (width, int(round(src_h * scale / 2)) * 2)
    index = TrackIndex(result["tracks"])
    risk = np.asarray(result["risk"], float).reshape(-1, 2)
    events = result["events"]
    duration = result["meta"]["duration"]
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = ffmpeg_exe()
    if ff:
        proc = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                 "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                                 "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p",
                                 "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
        write = proc.stdin.write
    else:
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        write = writer.write
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        frame = draw_frame(frame, i / fps, index, events, risk[:, 0], risk[:, 1], scale, duration)
        write(frame.tobytes() if ff else frame)
        i += 1
    cap.release()
    if ff:
        proc.stdin.close()
        proc.wait()
    else:
        writer.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="+")
    ap.add_argument("--out-dir", default=str(DATA / "videos"))
    ap.add_argument("--width", type=int, default=960)
    args = ap.parse_args()
    for v in map(Path, args.video):
        res = DATA / "results" / f"{v.stem}.json"
        if not res.exists():
            print(f"no analysis for {v.name}; run scripts/analyze_videos.py first", file=sys.stderr)
            continue
        out = Path(args.out_dir) / f"{v.stem}.mp4"
        print(f"[render] {v.name} -> {out}", flush=True)
        render(v, json.loads(res.read_text()), out, args.width)


if __name__ == "__main__":
    main()
