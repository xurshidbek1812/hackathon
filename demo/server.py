"""Live demo backend: serves the website and analyses uploaded videos.

    uvicorn demo.server:app --host 0.0.0.0 --port 7860

API
    POST /api/jobs              multipart "file" (.mp4, size/length limits from DEMO_MAX_MB / DEMO_MAX_SECONDS) -> {"id"}
    GET  /api/jobs/{id}         status, progress, message, and the result when done
    GET  /api/jobs/{id}/video   the uploaded clip, for playback under the overlay
    GET  /api/health

Jobs run one at a time in a background thread (CPU inference is fine).
"""
from __future__ import annotations

import os
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

# Limits are settings: generous by default (a full camera file is ~6 GB for 5-6 min of 4K);
# a small shared CPU host should lower them, e.g. DEMO_MAX_MB=500 DEMO_MAX_SECONDS=120.
MAX_BYTES = int(os.environ.get("DEMO_MAX_MB", 8192)) * 2**20
MAX_SECONDS = int(os.environ.get("DEMO_MAX_SECONDS", 600))
MAX_QUEUE = 5
JOB_TTL_SEC = 3600
JOBS_DIR = ROOT / ".demo_jobs"

app = FastAPI(title="Team Infinity - traffic event detection demo")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def revalidate_site_files(request, call_next):
    """Make browsers re-check pages, scripts and data on every load, so an update
    (e.g. new upload limits) is never hidden behind a stale cached copy."""
    response = await call_next(request)
    if not request.url.path.startswith("/api/") and not request.url.path.endswith(".mp4"):
        response.headers["Cache-Control"] = "no-cache"
    return response

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
           "-map", "0:v:0", "-vf", "scale='min(960,iw)':-2,format=yuv420p", "-c:v", "libx264",
           "-preset", "ultrafast", "-crf", "26", "-fps_mode", "passthrough", "-an",
           "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True, timeout=3600)
    return str(dst)


def _worker() -> None:
    while True:
        job_id = work.get()
        job = jobs[job_id]
        path = job["path"]
        try:
            _set(job_id, status="running", progress=0.0, message="starting")
            t0 = time.perf_counter()
            # the playback copy (CPU) is made while the analysis runs (mostly GPU)
            copy_error: list[Exception] = []

            def make_copy():
                try:
                    _set(job_id, playback=_browser_copy(path, Path(path).with_name("playback.mp4")))
                except Exception as exc:          # analysis still succeeds; playback falls back to the upload
                    copy_error.append(exc)

            copier = threading.Thread(target=make_copy, daemon=True)
            copier.start()
            analysis = analyze(path, progress=lambda f, m: _set(job_id, progress=0.6 * f, message=m))
            risk = risk_curve(path, progress=lambda f, m: _set(job_id, progress=0.6 + 0.4 * f, message=m))
            if copier.is_alive():
                _set(job_id, message="finishing playback copy")
                copier.join()
            if copy_error:
                traceback.print_exception(copy_error[0])
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
    return {"ok": True, "queued": work.qsize(), "max_mb": MAX_BYTES // 2**20, "max_seconds": MAX_SECONDS}


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
                raise HTTPException(413, f"file larger than {MAX_BYTES // 2**20} MB")
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
