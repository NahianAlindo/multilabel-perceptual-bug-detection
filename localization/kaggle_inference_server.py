"""
Kaggle Inference Server — Phase 3
Run this notebook cell on Kaggle to expose a FastAPI server via pyngrok.

Usage (Kaggle notebook cell):
    !pip install -q pyngrok fastapi uvicorn python-multipart
    !pip install -q timm decord  # or torchvision as fallback

    import subprocess, threading
    proc = subprocess.Popen(["python", "kaggle_inference_server.py",
                             "--checkpoint", "/kaggle/input/bugdetection/best_composite.pt",
                             "--model", "m3",     # m1 | m2 | m3
                             "--auth-token", "<your-ngrok-token>"])
    # URL printed to stdout — copy into the web UI settings panel

Then in the web UI, click the ⚙ icon and paste the ngrok URL.
"""

import argparse
import json
import os
import sys
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

# ── FastAPI ───────────────────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, UploadFile, File, HTTPException
    from fastapi.responses import JSONResponse, FileResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError:
    print("[ERROR] fastapi/uvicorn not installed. Run: pip install fastapi uvicorn python-multipart")
    sys.exit(1)

# ── pyngrok ───────────────────────────────────────────────────────────────────
try:
    from pyngrok import ngrok, conf as ngrok_conf
    NGROK_AVAILABLE = True
except ImportError:
    NGROK_AVAILABLE = False
    print("[WARN] pyngrok not installed — server will run locally only (no ngrok tunnel)")

app = FastAPI(title="Bug Detection Inference Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS_DIR = Path("/tmp/bugdetection_jobs")
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# Global: set by main() before server starts
MODEL_SCRIPT: str = ""
CHECKPOINT:   str = ""

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
        "--mode",          "infer",
        "--checkpoint",    CHECKPOINT,
        "--video-path",    video_path,
        "--inference-out", result_path,
        "--device",        "cuda",
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Stream progress from subprocess stdout
        progress = 10
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            # Advance a fake progress bar while waiting
            if progress < 90:
                progress = min(90, progress + 3)
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

# ── API routes ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "checkpoint": CHECKPOINT}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded video
    video_path = str(job_dir / file.filename)
    content = await file.read()
    with open(video_path, "wb") as f:
        f.write(content)

    _write_status(job_dir, "queued", 0, "Queued for inference…")

    # Start background inference thread
    t = threading.Thread(target=_run_inference_worker, args=(job_id, video_path), daemon=True)
    t.start()

    return JSONResponse({"job_id": job_id})


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
    videos  = list(job_dir.glob("*.mp4")) + list(job_dir.glob("*.avi")) + \
              list(job_dir.glob("*.mov")) + list(job_dir.glob("*.mkv"))
    if not videos:
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(str(videos[0]), media_type="video/mp4")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    global MODEL_SCRIPT, CHECKPOINT

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True, help="Path to best_composite.pt")
    p.add_argument("--model",       choices=["m1", "m2", "m3"], default="m3",
                   help="Which model's train.py to call for inference")
    p.add_argument("--model-dir",   default="",
                   help="Directory containing train.py. Defaults to same folder as this script.")
    p.add_argument("--auth-token",  default="", help="ngrok auth token")
    p.add_argument("--port",        type=int, default=8000)
    p.add_argument("--no-ngrok",    action="store_true")
    args = p.parse_args()

    CHECKPOINT = args.checkpoint

    # Resolve train.py path
    if args.model_dir:
        MODEL_SCRIPT = os.path.join(args.model_dir, "train.py")
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_folder = {
            "m1": "model1_bilstm_localization",
            "m2": "model2_mstcn_localization",
            "m3": "model3_pyramid_transformer",
        }[args.model]
        MODEL_SCRIPT = os.path.join(script_dir, "..", "training", model_folder, "train.py")

    MODEL_SCRIPT = os.path.abspath(MODEL_SCRIPT)
    if not os.path.exists(MODEL_SCRIPT):
        print(f"[ERROR] train.py not found at {MODEL_SCRIPT}")
        print("  Pass --model-dir to specify the directory containing train.py")
        sys.exit(1)

    print(f"[INFO] Model script : {MODEL_SCRIPT}")
    print(f"[INFO] Checkpoint   : {CHECKPOINT}")
    print(f"[INFO] Jobs dir     : {JOBS_DIR}")

    # ngrok tunnel
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
