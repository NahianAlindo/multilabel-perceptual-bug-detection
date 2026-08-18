"""
Inference module for the Bug Detection backend.

INFERENCE_MODE env var controls the execution path:
  local   (default) — runs model inference in-process via run_localization.py pipeline
  kaggle  — forwards all job requests to a Kaggle kernel exposed via ngrok

Set KAGGLE_NGROK_URL=https://<id>.ngrok.io when INFERENCE_MODE=kaggle.

The result.json schema is identical in both modes:
  {"video_id", "video_path", "duration", "annotations": [...]}
"""

import json
import os
import subprocess
import sys
import time
import threading
from pathlib import Path

import httpx

# ── config ────────────────────────────────────────────────────────────────────
INFERENCE_MODE  = os.environ.get("INFERENCE_MODE", "local").lower()   # "local" | "kaggle"
KAGGLE_NGROK_URL = os.environ.get("KAGGLE_NGROK_URL", "").rstrip("/")

# Local-mode settings (ignored in kaggle mode)
CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH",
    str(Path(__file__).parent.parent.parent /
        "training" / "model3_pyramid_transformer" /
        "checkpoints" / "best_composite.pt")
)
RUN_LOCALIZATION_SCRIPT = str(
    Path(__file__).parent.parent / "run_localization.py"
)

JOBS_DIR = Path(__file__).parent / "jobs"

BUG_TYPES = [
    "z_clipping",
    "corrupted_texture",
    "geometry_corruption",
    "z_fighting",
    "boundary_hole",
]

# ── helpers ───────────────────────────────────────────────────────────────────
def _write_status(job_dir: Path, status: str, progress: int, message: str = ""):
    (job_dir / "status.json").write_text(
        json.dumps({"status": status, "progress": progress, "message": message})
    )


# ── local inference ───────────────────────────────────────────────────────────
def _run_local_inference(job_id: str, video_path: str):
    """
    Runs inference by calling the best available model's train.py in infer mode,
    falling back to run_localization.py (Phase 1 pipeline) if no retrained checkpoint.
    """
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    result_path = str(job_dir / "result.json")

    _write_status(job_dir, "processing", 5, "Loading model…")

    # Prefer retrained Model 3 checkpoint; fall back to legacy pipeline
    if os.path.exists(CHECKPOINT_PATH):
        model3_train = str(
            Path(__file__).parent.parent.parent /
            "training" / "model3_pyramid_transformer" / "train.py"
        )
        if os.path.exists(model3_train):
            cmd = [
                sys.executable, model3_train,
                "--mode",          "infer",
                "--checkpoint",    CHECKPOINT_PATH,
                "--video-path",    video_path,
                "--inference-out", result_path,
                "--device",        "cuda",
            ]
        else:
            # Legacy: run_localization.py sliding-window pipeline
            cmd = [
                sys.executable, RUN_LOCALIZATION_SCRIPT,
                "--mode",        "infer",
                "--checkpoint",  CHECKPOINT_PATH,
                "--video-path",  video_path,
                "--inference-out", result_path,
            ]
    else:
        _write_status(job_dir, "error", 0,
                      f"No checkpoint found at {CHECKPOINT_PATH}. "
                      "Set CHECKPOINT_PATH env var or upload a checkpoint.")
        return

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )
        progress = 10
        tail_lines = []
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            tail_lines.append(line)
            tail_lines = tail_lines[-15:]
            if progress < 90:
                progress = min(90, progress + 2)
            _write_status(job_dir, "processing", progress, line[:120])

        proc.wait()

        if proc.returncode != 0:
            # Keep the subprocess's own last lines instead of a bare exit
            # code — the traceback that actually explains a failure almost
            # always shows up there.
            detail = " | ".join(tail_lines[-5:]) or "(no output captured)"
            _write_status(job_dir, "error", 0,
                          f"Inference exited with code {proc.returncode}: {detail}"[:500])
            return

        if not (job_dir / "result.json").exists():
            _write_status(job_dir, "error", 0, "result.json not written")
            return

        _write_status(job_dir, "done", 100, "Complete")

    except Exception as e:
        _write_status(job_dir, "error", 0, str(e))


# ── kaggle forwarding ─────────────────────────────────────────────────────────
def _run_kaggle_inference(job_id: str, video_path: str):
    """
    Uploads the video to the Kaggle ngrok server, then polls its status and
    mirrors the result into the local jobs directory so the existing
    /api/status and /api/results endpoints work without modification.
    """
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    _write_status(job_dir, "processing", 3, "Connecting to Kaggle inference server…")

    if not KAGGLE_NGROK_URL:
        _write_status(job_dir, "error", 0,
                      "KAGGLE_NGROK_URL env var not set. "
                      "Start the Kaggle notebook and paste the ngrok URL.")
        return

    try:
        with httpx.Client(timeout=60.0) as client:
            # Upload video to Kaggle server
            _write_status(job_dir, "processing", 8, "Uploading video to Kaggle…")
            with open(video_path, "rb") as vf:
                upload_resp = client.post(
                    f"{KAGGLE_NGROK_URL}/api/upload",
                    files={"file": (os.path.basename(video_path), vf, "video/mp4")},
                    timeout=120.0,
                )
            upload_resp.raise_for_status()
            remote_job_id = upload_resp.json()["job_id"]

        # Poll Kaggle server status, mirror locally
        with httpx.Client(timeout=30.0) as client:
            while True:
                try:
                    status_resp = client.get(
                        f"{KAGGLE_NGROK_URL}/api/status/{remote_job_id}"
                    )
                    status_resp.raise_for_status()
                    remote_status = status_resp.json()
                except Exception as e:
                    _write_status(job_dir, "processing", -1, f"Polling… ({e})")
                    time.sleep(5)
                    continue

                _write_status(
                    job_dir,
                    remote_status.get("status", "processing"),
                    remote_status.get("progress", 0),
                    remote_status.get("message", ""),
                )

                if remote_status["status"] == "done":
                    result_resp = client.get(
                        f"{KAGGLE_NGROK_URL}/api/results/{remote_job_id}"
                    )
                    result_resp.raise_for_status()
                    (job_dir / "result.json").write_text(
                        json.dumps(result_resp.json(), indent=2)
                    )
                    _write_status(job_dir, "done", 100, "Complete")
                    break

                elif remote_status["status"] == "error":
                    _write_status(job_dir, "error", 0,
                                  remote_status.get("message", "Kaggle inference failed"))
                    break

                time.sleep(3)

    except Exception as e:
        _write_status(job_dir, "error", 0, f"Kaggle forwarding error: {e}")


# ── public entry point ────────────────────────────────────────────────────────
def run_inference(job_id: str, video_path: str) -> None:
    """
    Entry point called in a background thread by main.py.
    Routes to local or Kaggle inference depending on INFERENCE_MODE.
    """
    if INFERENCE_MODE == "kaggle":
        _run_kaggle_inference(job_id, video_path)
    else:
        _run_local_inference(job_id, video_path)
