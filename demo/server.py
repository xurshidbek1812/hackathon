"""Live demo backend: serves the website and analyses uploaded videos.

    uvicorn demo.server:app --host 0.0.0.0 --port 7860

API
    POST /api/jobs              multipart "file" (.mp4, <= 200 MB, <= 150 s) -> {"id"}
    GET  /api/jobs/{id}         status, progress, message, and the result when done
    GET  /api/jobs/{id}/video   the uploaded clip, for playback under the overlay
    GET  /api/health

Jobs run one at a time in a background thread (CPU inference is fine).
"""
from __future__ import annotations

import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic_events.export import analysis_json, risk_curve  # noqa: E402
from traffic_events.pipeline import analyze  # noqa: E402
from traffic_events.video import read_meta  # noqa: E402

MAX_BYTES = 200 * 2**20
MAX_SECONDS = 150
MAX_QUEUE = 5
JOB_TTL_SEC = 3600
JOBS_DIR = ROOT / ".demo_jobs"

app = FastAPI(title="Team Infinity - traffic event detection demo")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

jobs: dict[str, dict] = {}
work: queue.Queue[str] = queue.Queue()
lock = threading.Lock()


def _set(job_id: str, **kw) -> None:
    with lock:
        jobs[job_id].update(kw)


def _browser_copy(src: str, dst: Path) -> str:
    """Playback copy the browser can decode (8-bit 4:2:0 H.264, no audio, same timing).

    Professional camera files (e.g. Sony XAVC 10-bit 4:2:2 with PCM audio) are
    analysed as they are, but browsers refuse to play them."""
    import imageio_ffmpeg
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", src,
           "-map", "0:v:0", "-vf", "scale='min(1280,iw)':-2,format=yuv420p", "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "24", "-fps_mode", "passthrough", "-an",
           "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True, timeout=600)
    return str(dst)


def _worker() -> None:
    while True:
        job_id = work.get()
        job = jobs[job_id]
        path = job["path"]
        try:
            _set(job_id, status="running", progress=0.0, message="starting")
            t0 = time.perf_counter()
            _set(job_id, message="preparing playback copy")
            playback = _browser_copy(path, Path(path).with_name("playback.mp4"))
            _set(job_id, playback=playback, progress=0.05)
            analysis = analyze(path, progress=lambda f, m: _set(job_id, progress=0.05 + 0.55 * f, message=m))
            risk = risk_curve(path, progress=lambda f, m: _set(job_id, progress=0.6 + 0.4 * f, message=m))
            result = analysis_json(analysis, risk, name=job["name"])
            result["stats"]["total_seconds"] = round(time.perf_counter() - t0, 1)
            _set(job_id, status="done", progress=1.0, message="done", result=result)
        except Exception as exc:        # report any failure to the page instead of dying
            traceback.print_exc()
            _set(job_id, status="error", message=f"analysis failed: {exc}")
        finally:
            work.task_done()


def _cleanup() -> None:
    now = time.time()
    with lock:
        for jid in [j for j, v in jobs.items() if now - v["created"] > JOB_TTL_SEC and v["status"] in ("done", "error")]:
            shutil.rmtree(JOBS_DIR / jid, ignore_errors=True)
            del jobs[jid]


threading.Thread(target=_worker, daemon=True).start()


@app.get("/api/health")
def health():
    return {"ok": True, "queued": work.qsize()}


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...)):
    _cleanup()
    if not (file.filename or "").lower().endswith(".mp4"):
        raise HTTPException(400, "please upload an .mp4 file")
    if work.qsize() >= MAX_QUEUE:
        raise HTTPException(503, "the demo is busy, please try again in a minute")
    job_id = uuid.uuid4().hex[:12]
    folder = JOBS_DIR / job_id
    folder.mkdir(parents=True)
    path = folder / "input.mp4"
    size = 0
    with path.open("wb") as f:
        while chunk := await file.read(2**20):
            size += len(chunk)
            if size > MAX_BYTES:
                shutil.rmtree(folder, ignore_errors=True)
                raise HTTPException(413, "file larger than 200 MB")
            f.write(chunk)
    try:
        meta = read_meta(str(path))
    except IOError:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(400, "could not decode the video")
    if meta.duration > MAX_SECONDS:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(400, f"video is {meta.duration:.0f} s long; the demo accepts up to {MAX_SECONDS} s")
    with lock:
        jobs[job_id] = {"status": "queued", "progress": 0.0, "message": "queued", "created": time.time(),
                        "path": str(path), "name": file.filename}
    work.put(job_id)
    return {"id": job_id, "duration": round(meta.duration, 1)}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return {k: v for k, v in job.items() if k not in ("path", "playback", "created")}


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return FileResponse(job.get("playback") or job["path"], media_type="video/mp4")


app.mount("/", StaticFiles(directory=ROOT / "website", html=True), name="site")
