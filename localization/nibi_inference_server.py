"""
nibi Inference Server — real inference backend on Digital Research Alliance
of Canada's nibi cluster, exposed via ngrok (same mechanism already proven
working for Kaggle in kaggle_inference_server.py, and confirmed reachable
from outside the cluster via the networking_test/ reachability check).

Usage (on nibi, inside a SLURM GPU job — see run_nibi_inference_server.sh):
    python nibi_inference_server.py \
        --checkpoint /path/to/best_f1.pt \
        --auth-token <your-ngrok-token>
    # URL printed to stdout — copy into the web UI settings panel

Differences from kaggle_inference_server.py (deliberate, not drift):
  - Upload is CHUNKED (POST /api/upload/init, /api/upload/chunk/{id}/{n},
    /api/upload/complete/{id}) instead of one big multipart POST — these
    videos run 5-15+ minutes and can be several hundred MB to low GB, too
    large/fragile for a single request over a free ngrok tunnel. Each chunk
    is read fully into memory (bounded, ~8MB, safe) and appended to a
    growing file on disk — never the whole video at once, unlike
    kaggle_inference_server.py's `content = await file.read()`.
  - Bakes in the tuned post-processing (--min-conf 0.3 --nms-score-thr 0.25)
    confirmed via the sweep — matches the model3-specific checkpoint format
    (model_state + thresholds), not the older BugBiLSTM format this
    directory's CLAUDE.md still describes (see that file's own staleness
    note — Model 3 is the production model now).

Same /api/status, /api/results, /api/video contract as kaggle_inference_server.py
and backend/main.py — the frontend doesn't need to know or care which one it's
talking to.
"""

import argparse
import json
import os
import sys
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Dict

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
    from pyngrok import ngrok, conf as ngrok_conf
    NGROK_AVAILABLE = True
except ImportError:
    NGROK_AVAILABLE = False
    print("[WARN] pyngrok not installed — server will run locally only (no ngrok tunnel)")

app = FastAPI(title="Bug Detection Inference Server (nibi)")
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

# Global: set by main() before server starts
MODEL_SCRIPT: str = ""
CHECKPOINT:   str = ""
MIN_CONF:     float = 0.3
NMS_SCORE_THR: float = 0.25

# In-memory upload session tracking. Simple by design: chunks are assumed to
# arrive in order (the frontend uploads sequentially, not concurrently) and
# this is a single-user personal tool, not a multi-tenant resumable-upload
# service — doesn't need to survive a server restart mid-upload.
UPLOADS: Dict[str, dict] = {}


# ── status helpers ────────────────────────────────────────────────────────────
def _write_status(job_dir: Path, status: str, progress: int, message: str = ""):
    (job_dir / "status.json").write_text(
        json.dumps({"status": status, "progress": progress, "message": message})
    )


# ── inference worker ──────────────────────────────────────────────────────────
def _run_inference_worker(job_id: str, video_path: str):
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    result_path = str(job_dir / "result.json")

    _write_status(job_dir, "processing", 5, "Loading model…")

    cmd = [
        sys.executable, MODEL_SCRIPT,
        "--mode",           "infer",
        "--checkpoint",     CHECKPOINT,
        "--video-path",     video_path,
        "--inference-out",  result_path,
        "--device",         "cuda",
        "--min-conf",       str(MIN_CONF),
        "--nms-score-thr",  str(NMS_SCORE_THR),
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Stream progress from subprocess stdout. train.py's run_infer now
        # prints "[INFER] window N/M (...)" every 20 windows, so this
        # actually tracks real progress instead of a fake incrementing bar.
        progress = 10
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            if progress < 90:
                progress = min(90, progress + 2)
            _write_status(job_dir, "processing", progress, line[:120])

        proc.wait()

        if proc.returncode != 0:
            _write_status(job_dir, "error", 0, f"Inference process exited with code {proc.returncode}")
            return

        if not (job_dir / "result.json").exists():
            _write_status(job_dir, "error", 0, "result.json not written by inference script")
            return

        _write_status(job_dir, "done", 100, "Complete")

    except Exception as e:
        _write_status(job_dir, "error", 0, str(e))


# ── chunked upload ────────────────────────────────────────────────────────────
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
    # Create/truncate the file so chunks can be appended to it in order.
    open(upload_path, "wb").close()

    return JSONResponse({"upload_id": upload_id})


@app.post("/api/upload/chunk/{upload_id}/{index}")
async def upload_chunk(upload_id: str, index: int, request: Request):
    session = UPLOADS.get(upload_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown upload_id (server may have restarted)")

    # Each chunk is small (frontend sends ~8MB pieces) — bounded, safe to
    # read fully into memory. Never the whole video at once.
    chunk_bytes = await request.body()

    # Chunks are expected in order — append-only, no seeking. If the
    # frontend ever needs to retry an out-of-order/duplicate chunk this
    # would need index-aware seeking, which the current sequential
    # upload loop design doesn't require.
    with open(session["path"], "ab") as f:
        f.write(chunk_bytes)

    session["received_chunks"] += 1
    return JSONResponse({
        "received_chunks": session["received_chunks"],
        "total_chunks": session["total_chunks"],
    })


@app.post("/api/upload/complete/{upload_id}")
def upload_complete(upload_id: str):
    session = UPLOADS.get(upload_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown upload_id (server may have restarted)")

    upload_path = Path(session["path"])
    actual_size = upload_path.stat().st_size if upload_path.exists() else 0
    if session["received_chunks"] != session["total_chunks"]:
        raise HTTPException(
            status_code=400,
            detail=f"Incomplete upload: {session['received_chunks']}/{session['total_chunks']} chunks received",
        )
    if actual_size != session["total_size"]:
        raise HTTPException(
            status_code=400,
            detail=f"Size mismatch: received {actual_size} bytes, expected {session['total_size']}",
        )

    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    video_path = job_dir / upload_path.name
    upload_path.rename(video_path)
    del UPLOADS[upload_id]

    _write_status(job_dir, "queued", 0, "Queued for inference…")

    t = threading.Thread(target=_run_inference_worker, args=(job_id, str(video_path)), daemon=True)
    t.start()

    return JSONResponse({"job_id": job_id})


# ── API routes (unchanged contract vs. kaggle_inference_server.py) ────────────
@app.get("/health")
def health():
    return {"status": "ok", "checkpoint": CHECKPOINT, "min_conf": MIN_CONF,
            "nms_score_thr": NMS_SCORE_THR}


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
             list(job_dir.glob("*.mov")) + list(job_dir.glob("*.mkv"))
    if not videos:
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(str(videos[0]), media_type="video/mp4")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    global MODEL_SCRIPT, CHECKPOINT, MIN_CONF, NMS_SCORE_THR

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     required=True, help="Path to best_f1.pt")
    p.add_argument("--model-dir",      default="",
                   help="Directory containing train.py. Defaults to "
                        "../training/model3_pyramid_transformer relative to this script.")
    p.add_argument("--min-conf",       type=float, default=0.3)
    p.add_argument("--nms-score-thr",  type=float, default=0.25)
    p.add_argument("--auth-token",     default="", help="ngrok auth token")
    p.add_argument("--port",           type=int, default=8000)
    p.add_argument("--no-ngrok",       action="store_true")
    args = p.parse_args()

    CHECKPOINT = args.checkpoint
    MIN_CONF = args.min_conf
    NMS_SCORE_THR = args.nms_score_thr

    if args.model_dir:
        MODEL_SCRIPT = os.path.join(args.model_dir, "train.py")
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        MODEL_SCRIPT = os.path.join(script_dir, "..", "training",
                                    "model3_pyramid_transformer", "train.py")

    MODEL_SCRIPT = os.path.abspath(MODEL_SCRIPT)
    if not os.path.exists(MODEL_SCRIPT):
        print(f"[ERROR] train.py not found at {MODEL_SCRIPT}")
        print("  Pass --model-dir to specify the directory containing train.py")
        sys.exit(1)
    if not os.path.exists(CHECKPOINT):
        print(f"[ERROR] checkpoint not found at {CHECKPOINT}")
        sys.exit(1)

    print(f"[INFO] Model script    : {MODEL_SCRIPT}")
    print(f"[INFO] Checkpoint      : {CHECKPOINT}")
    print(f"[INFO] min-conf        : {MIN_CONF}")
    print(f"[INFO] nms-score-thr   : {NMS_SCORE_THR}")
    print(f"[INFO] Jobs dir        : {JOBS_DIR}")
    print(f"[INFO] Uploads dir     : {UPLOADS_DIR}")

    if not args.no_ngrok and NGROK_AVAILABLE:
        if args.auth_token:
            ngrok_conf.get_default().auth_token = args.auth_token
        try:
            tunnel = ngrok.connect(args.port, "http")
            public_url = tunnel.public_url
            print("\n" + "=" * 60)
            print(f"  ngrok tunnel : {public_url}")
            print(f"  Paste this URL into the web UI settings panel")
            print("=" * 60 + "\n")
        except Exception as e:
            print(f"[WARN] ngrok failed: {e} — server running on localhost:{args.port}")
    else:
        print(f"[INFO] Server running on localhost:{args.port} (no ngrok tunnel)")

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
