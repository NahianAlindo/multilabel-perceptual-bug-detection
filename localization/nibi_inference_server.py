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
import re
import sys
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Dict, List, Optional

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

if not DECORD_AVAILABLE and not CV2_AVAILABLE:
    print("[WARN] neither decord nor opencv importable — pre-flight video "
          "validation disabled, uploads go straight to the inference "
          "subprocess as before")

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

# Matches the frontend's own MAX_DURATION (index.html) — kept in sync manually
# since there's no shared config file between the two.
MAX_DURATION_SECONDS = 20 * 60

# Matches train.py's run_infer() default duration fallback (600.0) — used
# only to *detect* that fallback firing (a real bug in train.py we're not
# allowed to touch here), not to change any inference behavior.
_TRAIN_PY_DURATION_FALLBACK = 600.0

_WINDOW_PROGRESS_RE = re.compile(r"window\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
_DECODE_WARN_RE = re.compile(r"\[WARN\]\s+(decord|cv2)\s+failed", re.IGNORECASE)


_UNREADABLE_MSG = (
    "This file could not be opened by the server's video decoder (likely an "
    "unsupported codec or container). Try re-encoding as H.264 video in an "
    ".mp4 container."
)
_NO_DIMENSIONS_MSG = (
    "The file opened but reported no frame dimensions — it may be empty, "
    "corrupt, or an unsupported stream layout."
)


def _probe_video(path: str) -> dict:
    """Read-only pre-flight check, done and released before the inference
    subprocess ever sees the file. Never resizes, transcodes, or otherwise
    touches the bytes the model will see — only answers "can this be opened,
    and what does its container report" so obviously-broken uploads fail in
    ~1s with a real reason instead of a silent ~15s dead end.

    Tries decord first, falling back to cv2 — the same backend priority
    train.py itself uses for real decoding (extract_frames_at_fps tries
    decord, then opencv). Either one missing/broken on a given server no
    longer means validation silently disappears, it just falls through to
    whichever backend is actually available.
    """
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
            return {"ok": False, "error": _UNREADABLE_MSG}

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()

        if width <= 0 or height <= 0:
            return {"ok": False, "error": _NO_DIMENSIONS_MSG}

        duration = (frame_count / fps) if fps > 0 and frame_count > 0 else 0.0
        return {"ok": True, "width": width, "height": height, "fps": fps, "duration": duration}

    # Neither backend importable on this server — can't validate up front;
    # let the inference subprocess try (matches pre-existing behavior).
    return {"ok": True, "width": 0, "height": 0, "duration": 0.0}

# In-memory upload session tracking. Simple by design: chunks are assumed to
# arrive in order (the frontend uploads sequentially, not concurrently) and
# this is a single-user personal tool, not a multi-tenant resumable-upload
# service — doesn't need to survive a server restart mid-upload.
UPLOADS: Dict[str, dict] = {}


# ── status helpers ────────────────────────────────────────────────────────────
def _write_status(job_dir: Path, status: str, progress: int, message: str = "", **extra):
    payload = {"status": status, "progress": progress, "message": message}
    payload.update(extra)
    (job_dir / "status.json").write_text(json.dumps(payload))


# ── inference worker ──────────────────────────────────────────────────────────
def _run_inference_worker(job_id: str, video_path: str, probed_duration: float = 0.0):
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
        # Also re-print every line to OUR OWN stdout (lands in the SLURM
        # .out log, given PYTHONUNBUFFERED=1) and keep the last few lines
        # around — if the subprocess fails, its actual traceback is almost
        # always in those last lines, not in a generic exit-code message.
        progress = 10
        tail_lines: List[str] = []
        total_windows: Optional[int] = None
        decode_warn_count = 0
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            print(f"[infer:{job_id[:8]}] {line}")
            tail_lines.append(line)
            tail_lines = tail_lines[-15:]
            m = _WINDOW_PROGRESS_RE.search(line)
            if m:
                total_windows = int(m.group(2))
            if _DECODE_WARN_RE.search(line):
                decode_warn_count += 1
            if progress < 90:
                progress = min(90, progress + 2)
            _write_status(job_dir, "processing", progress, line[:120])

        proc.wait()

        if proc.returncode != 0:
            detail = " | ".join(tail_lines[-5:]) or "(no output captured)"
            _write_status(job_dir, "error", 0,
                         f"Inference process exited with code {proc.returncode}: {detail}"[:500])
            return

        if not (job_dir / "result.json").exists():
            _write_status(job_dir, "error", 0, "result.json not written by inference script")
            return

        # These don't change the result that was already written — they only
        # read the subprocess's own stdout and the result file it produced,
        # to distinguish "the model looked and found nothing" from "part of
        # the video never actually reached the model."
        warnings: List[str] = []
        if decode_warn_count > 0:
            extent = f" out of {total_windows} total windows" if total_windows else ""
            warnings.append(
                f"{decode_warn_count} frame-decode warning(s) logged during inference"
                f"{extent}. Those windows were likely processed as blank frames — gaps "
                f"in the results near them may reflect a decode problem, not an absence "
                f"of bugs."
            )
        try:
            result_data = json.loads(Path(result_path).read_text())
            result_duration = float(result_data.get("duration", 0) or 0)
            if (abs(result_duration - _TRAIN_PY_DURATION_FALLBACK) < 0.01
                    and probed_duration > 0
                    and abs(probed_duration - _TRAIN_PY_DURATION_FALLBACK) > 5.0):
                warnings.append(
                    f"The reported video duration ({result_duration:.0f}s) looks like an "
                    f"internal fallback value, not this video's real duration "
                    f"(~{probed_duration:.0f}s). Segment timestamps may not line up with "
                    f"the actual video."
                )
        except Exception:
            pass

        _write_status(job_dir, "done", 100, "Complete", warnings=warnings)

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

    probe = _probe_video(str(video_path))
    if not probe["ok"]:
        _write_status(job_dir, "error", 0, probe["error"])
        return JSONResponse({"job_id": job_id})

    probed_duration = probe.get("duration", 0.0)
    if probed_duration > MAX_DURATION_SECONDS:
        _write_status(
            job_dir, "error", 0,
            f"Video is {probed_duration:.0f}s long — maximum allowed is "
            f"{MAX_DURATION_SECONDS}s (20:00)."
        )
        return JSONResponse({"job_id": job_id})

    _write_status(
        job_dir, "queued", 0, "Queued for inference…",
        probed_width=probe.get("width", 0),
        probed_height=probe.get("height", 0),
        probed_duration=round(probed_duration, 1) if probed_duration else None,
    )

    t = threading.Thread(
        target=_run_inference_worker,
        args=(job_id, str(video_path), probed_duration),
        daemon=True,
    )
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


# ── review persistence ─────────────────────────────────────────────────────────
# Stores the user's mark-correct/incorrect, edited timestamps/labels, and
# removed detections next to the job's own result.json. Read-only for the
# model side of things — this never feeds back into inference, it's just a
# JSON blob the frontend reads/writes wholesale. Same durability envelope as
# result.json itself: lives in JOBS_DIR, so it doesn't outlive this job's
# /tmp any better or worse than the results it's reviewing.
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
