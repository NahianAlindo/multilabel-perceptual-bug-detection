"""
Dummy Inference Server — local demo/dev tool, NOT real inference.

Lets you click through the whole frontend (upload → processing → results →
review/edit/export) on your own machine with no GPU, no checkpoint, no
SLURM/Kaggle job, and no ngrok tunnel. It never calls train.py or touches
the model in any way — instead it waits a few seconds and fabricates a
handful of plausible-looking detections spread across your video's real
duration, in the exact same JSON shape run_infer() produces, so every piece
of frontend UI (scores, warnings banner, timeline, review tools) has
something real to render.

Usage:
    python dummy_inference_server.py
    # then open frontend/index.html (or `python -m http.server 3000` in
    # frontend/) — leave the settings modal's ngrok URL blank, it talks to
    # http://localhost:8000 by default, which is exactly where this listens.

Same /api/status, /api/results, /api/video, /api/review, and chunked-upload
(/api/upload/init|chunk|complete) contract as nibi_inference_server.py —
this is a drop-in stand-in for it, nothing more.
"""

import argparse
import json
import os
import random
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, FileResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("[ERROR] fastapi/uvicorn/pydantic not installed. "
          "Run: pip install fastapi uvicorn python-multipart pydantic")
    sys.exit(1)

try:
    from decord import VideoReader, cpu as decord_cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

app = FastAPI(title="Bug Detection Inference Server (DUMMY — no model)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS_DIR = Path("/tmp/bugdetection_jobs")
JOBS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR = Path("/tmp/bugdetection_uploads")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

MAX_DURATION_SECONDS = 20 * 60
FALLBACK_DURATION = 60.0  # used only if decord/cv2 can't read the real duration

BUG_TYPES = ["z_clipping", "corrupted_texture", "geometry_corruption", "z_fighting", "boundary_hole"]

# Set by main()
INJECT_WARNING = False

UPLOADS: Dict[str, dict] = {}


# ── probe (same logic as the real servers, kept local so this file has no
# dependency on them) ───────────────────────────────────────────────────────
def _probe_video(path: str) -> dict:
    if DECORD_AVAILABLE:
        try:
            vr = VideoReader(path, ctx=decord_cpu(0))
            n_frames = len(vr)
            fps = vr.get_avg_fps() or 0.0
            height, width = vr[0].shape[:2]
            del vr
            if width > 0 and height > 0:
                duration = (n_frames / fps) if fps > 0 else 0.0
                return {"ok": True, "width": int(width), "height": int(height),
                        "fps": fps, "duration": duration}
        except Exception as e:
            print(f"[WARN] decord probe failed for {path}: {e} — trying cv2")

    if CV2_AVAILABLE:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            return {"ok": False, "error": "This file could not be opened by the "
                     "server's video decoder (unsupported codec/container)."}
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        if width <= 0 or height <= 0:
            return {"ok": False, "error": "The file opened but reported no frame "
                     "dimensions — it may be empty or corrupt."}
        duration = (frame_count / fps) if fps > 0 and frame_count > 0 else 0.0
        return {"ok": True, "width": width, "height": height, "fps": fps, "duration": duration}

    print("[WARN] neither decord nor opencv importable — duration guessed as "
          f"{FALLBACK_DURATION}s instead of read from the file")
    return {"ok": True, "width": 0, "height": 0, "duration": 0.0}


def _write_status(job_dir: Path, status: str, progress: int, message: str = "", **extra):
    payload = {"status": status, "progress": progress, "message": message}
    payload.update(extra)
    (job_dir / "status.json").write_text(json.dumps(payload))


def _fake_annotations(duration: float) -> List[dict]:
    """Fabricates a handful of detections spread across the real video
    duration, in run_infer()'s exact output shape (bug_type/start/end/
    duration/segment_id/score) — nothing here is a real prediction."""
    duration = duration if duration > 0 else FALLBACK_DURATION
    n = random.randint(2, 6)
    anns = []
    for i in range(n):
        seg_len = random.uniform(1.0, min(5.0, duration / 2))
        start = random.uniform(0, max(0.1, duration - seg_len))
        end = start + seg_len
        n_labels = random.choices([1, 2], weights=[0.8, 0.2])[0]
        types = random.sample(BUG_TYPES, k=n_labels)
        for bug_type in types:
            anns.append({
                "bug_type": bug_type,
                "start": round(start, 2),
                "end": round(end, 2),
                "duration": round(end - start, 2),
                "segment_id": len(anns),
                "score": round(random.uniform(0.28, 0.93), 4),
            })
    anns.sort(key=lambda a: a["start"])
    return anns


def _run_dummy_worker(job_id: str, video_path: str, probed_duration: float):
    job_dir = JOBS_DIR / job_id
    result_path = job_dir / "result.json"

    _write_status(job_dir, "processing", 10, "Loading model… (dummy — nothing real is loading)")
    for pct, msg in [(30, "Decoding frames (simulated)…"),
                      (55, "Running temporal model (simulated)…"),
                      (80, "Post-processing detections (simulated)…")]:
        time.sleep(1.0)
        _write_status(job_dir, "processing", pct, msg)

    duration = probed_duration if probed_duration > 0 else FALLBACK_DURATION
    annotations = _fake_annotations(duration)
    result = {
        "video_id": job_id,
        "video_path": str(video_path),
        "duration": duration,
        "annotations": annotations,
    }
    result_path.write_text(json.dumps(result, indent=2))

    warnings = []
    if INJECT_WARNING:
        warnings.append(
            "This is a simulated warning (server started with --inject-warning) — "
            "on a real server this slot shows genuine pipeline problems like "
            "decode failures."
        )
    time.sleep(0.5)
    _write_status(job_dir, "done", 100, "Complete (dummy)", warnings=warnings)


# ── chunked upload — identical contract to nibi_inference_server.py ───────────
class UploadInitRequest(BaseModel):
    filename: str
    total_size: int
    total_chunks: int


@app.post("/api/upload/init")
def upload_init(req: UploadInitRequest):
    upload_id = str(uuid.uuid4())
    suffix = Path(req.filename).suffix or ".mp4"
    upload_path = UPLOADS_DIR / f"{upload_id}{suffix}"
    UPLOADS[upload_id] = {
        "path": str(upload_path),
        "total_size": req.total_size,
        "total_chunks": req.total_chunks,
        "received_chunks": 0,
    }
    open(upload_path, "wb").close()
    return JSONResponse({"upload_id": upload_id})


@app.post("/api/upload/chunk/{upload_id}/{index}")
async def upload_chunk(upload_id: str, index: int, request: Request):
    session = UPLOADS.get(upload_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown upload_id (server may have restarted)")
    chunk_bytes = await request.body()
    with open(session["path"], "ab") as f:
        f.write(chunk_bytes)
    session["received_chunks"] += 1
    return JSONResponse({"received_chunks": session["received_chunks"], "total_chunks": session["total_chunks"]})


@app.post("/api/upload/complete/{upload_id}")
def upload_complete(upload_id: str):
    session = UPLOADS.get(upload_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown upload_id (server may have restarted)")

    upload_path = Path(session["path"])
    actual_size = upload_path.stat().st_size if upload_path.exists() else 0
    if session["received_chunks"] != session["total_chunks"]:
        raise HTTPException(status_code=400, detail=f"Incomplete upload: "
                             f"{session['received_chunks']}/{session['total_chunks']} chunks received")
    if actual_size != session["total_size"]:
        raise HTTPException(status_code=400, detail=f"Size mismatch: received {actual_size} "
                             f"bytes, expected {session['total_size']}")

    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    video_path = job_dir / upload_path.name
    upload_path.rename(video_path)
    del UPLOADS[upload_id]

    probe = _probe_video(str(video_path))
    if not probe["ok"]:
        _write_status(job_dir, "error", 0, probe["error"])
        return JSONResponse({"job_id": job_id})

    probed_duration = probe.get("duration", 0.0)
    if probed_duration > MAX_DURATION_SECONDS:
        _write_status(job_dir, "error", 0,
                       f"Video is {probed_duration:.0f}s long — maximum allowed is "
                       f"{MAX_DURATION_SECONDS}s (20:00).")
        return JSONResponse({"job_id": job_id})

    _write_status(job_dir, "queued", 0, "Queued for (simulated) inference…")
    t = threading.Thread(target=_run_dummy_worker, args=(job_id, str(video_path), probed_duration), daemon=True)
    t.start()
    return JSONResponse({"job_id": job_id})


# ── API routes — same contract as the real servers ─────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "checkpoint": "DUMMY — no real model loaded"}


@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    status_file = JOBS_DIR / job_id / "status.json"
    if not status_file.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(json.loads(status_file.read_text()))


@app.get("/api/results/{job_id}")
def get_results(job_id: str):
    result_file = JOBS_DIR / job_id / "result.json"
    if not result_file.exists():
        raise HTTPException(status_code=404, detail="Results not ready")
    return JSONResponse(json.loads(result_file.read_text()))


@app.get("/api/video/{job_id}")
def get_video(job_id: str):
    job_dir = JOBS_DIR / job_id
    videos = list(job_dir.glob("*.mp4")) + list(job_dir.glob("*.avi")) + \
             list(job_dir.glob("*.mov")) + list(job_dir.glob("*.mkv")) + list(job_dir.glob("*.webm"))
    if not videos:
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(str(videos[0]), media_type="video/mp4")


@app.get("/api/review/{job_id}")
def get_review(job_id: str):
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    review_file = job_dir / "review.json"
    if not review_file.exists():
        return JSONResponse({})
    return JSONResponse(json.loads(review_file.read_text()))


@app.post("/api/review/{job_id}")
async def save_review(job_id: str, request: Request):
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    try:
        review_data = json.loads(await request.body())
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    if not isinstance(review_data, dict):
        raise HTTPException(status_code=400, detail="Review body must be a JSON object")
    (job_dir / "review.json").write_text(json.dumps(review_data))
    return JSONResponse({"saved": True})


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    global INJECT_WARNING

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--inject-warning", action="store_true",
                   help="Always attach a fake warning to completed jobs, to preview that UI.")
    args = p.parse_args()

    INJECT_WARNING = args.inject_warning

    print("=" * 60)
    print("  DUMMY inference server — no model, no checkpoint, no GPU")
    print(f"  Listening on http://localhost:{args.port}")
    print("  Point the frontend at it by leaving the settings modal's")
    print("  ngrok URL blank (localhost:8000 is the default).")
    if DECORD_AVAILABLE or CV2_AVAILABLE:
        print(f"  Video probing: {'decord' if DECORD_AVAILABLE else 'cv2'} available")
    else:
        print("  Video probing: unavailable (neither decord nor cv2 installed) — "
              f"durations will be guessed as {FALLBACK_DURATION}s")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
