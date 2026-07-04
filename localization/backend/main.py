"""
FastAPI backend for video game perceptual bug detection.
"""

import json
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from inference import JOBS_DIR, run_inference

app = FastAPI(title="Bug Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_DURATION_SECONDS = 20 * 60  # 20 minutes


# ---------------------------------------------------------------------------
# POST /api/upload
# ---------------------------------------------------------------------------
@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)) -> JSONResponse:
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video.")

    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    video_path = job_dir / f"video{suffix}"

    # Stream upload to disk
    with open(video_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):  # 1 MB chunks
            f.write(chunk)

    # Kick off background inference
    thread = threading.Thread(
        target=run_inference,
        args=(job_id, str(video_path)),
        daemon=True,
    )
    thread.start()

    return JSONResponse({"job_id": job_id})


# ---------------------------------------------------------------------------
# GET /api/status/{job_id}
# ---------------------------------------------------------------------------
@app.get("/api/status/{job_id}")
def get_status(job_id: str) -> JSONResponse:
    status_file = JOBS_DIR / job_id / "status.json"
    if not status_file.exists():
        raise HTTPException(status_code=404, detail="Job not found.")
    return JSONResponse(json.loads(status_file.read_text()))


# ---------------------------------------------------------------------------
# GET /api/results/{job_id}
# ---------------------------------------------------------------------------
@app.get("/api/results/{job_id}")
def get_results(job_id: str) -> JSONResponse:
    result_file = JOBS_DIR / job_id / "result.json"
    if not result_file.exists():
        status_file = JOBS_DIR / job_id / "status.json"
        if not status_file.exists():
            raise HTTPException(status_code=404, detail="Job not found.")
        status = json.loads(status_file.read_text())
        raise HTTPException(
            status_code=202,
            detail=f"Job not complete yet. Status: {status.get('status')}",
        )
    return JSONResponse(json.loads(result_file.read_text()))


# ---------------------------------------------------------------------------
# GET /api/video/{job_id}
# ---------------------------------------------------------------------------
@app.get("/api/video/{job_id}")
def stream_video(job_id: str) -> FileResponse:
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found.")
    # Find the uploaded video file
    videos = list(job_dir.glob("video.*"))
    if not videos:
        raise HTTPException(status_code=404, detail="Video file not found.")
    video_path = videos[0]
    return FileResponse(
        str(video_path),
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"},
    )
