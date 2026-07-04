# Localization Project — CLAUDE.md

## What this project does

Multi-label visual bug **detection and temporal localization** in full-length gameplay videos (up to 20 minutes). Given a video, the system outputs timestamped segments indicating when and which bug types occur.

## Key decision: no retraining

The `BugBiLSTM` model (ResNet18 + BiLSTM + TemporalAttention) was already trained on 2-second clips at 224×224 @ 8fps extracted from the same 263 source videos. The existing checkpoint is used as-is. Temporal localization is achieved purely at **inference time** via a sliding-window approach — no fine-tuning needed.

## Directory structure

```
localization/
├── run_localization.py        # Main script: eval + infer modes
├── run_localization.sh        # SLURM job script for Narval HPC (eval mode)
├── temporal_bug_dataset.json  # 263 full-length videos with temporal annotations
├── backend/
│   ├── main.py                # FastAPI server (unchanged)
│   ├── inference.py           # Calls run_localization.py pipeline (real model)
│   └── requirements.txt
└── frontend/
    └── index.html             # SPA — no changes needed; handles multi-label natively
```

## Source of truth for model architecture

`../train_FINAL_ALL_METRICS.py` — `BugBiLSTM`, `TemporalAttention`, loss functions, threshold tuning logic. The class definitions in `run_localization.py` are copied verbatim and must stay in sync.

## Bug types (5 classes)

| Output name (dataset/backend) | Canonical (model) |
|---|---|
| `z_clipping` | `z-clipping` |
| `corrupted_texture` | `corrupted_texture` |
| `geometry_corruption` | `geometry_corruption` |
| `z_fighting` | `z-fighting` |
| `boundary_hole` | `boundary_hole` |

`norm_bug_type()` handles the underscore ↔ hyphen mapping transparently.

## Sliding window parameters (must match training)

| Parameter | Value |
|---|---|
| Window size | 2.0 seconds |
| Stride | 1.0 second |
| Sampling rate | 8 fps |
| Frame size | 224 × 224 |
| Frames per window | 16 |

Do not change these without retraining the model.

## Video backend priority

`extract_frames_at_fps()` tries backends in this order:
1. **decord** — preferred on HPC; chunked random access, fastest
2. **opencv** — sequential grab/read fallback

`torchvision.io.read_video` is intentionally excluded — it loads the entire file into RAM, impractical for 20-minute videos.

## Checkpoint format

Checkpoints saved by `train_FINAL_ALL_METRICS.py` contain:
- `model_state` — `BugBiLSTM` state dict (EMA weights if EMA was active)
- `thr_any` — tuned presence threshold (float)
- `thr_map` — per-class type thresholds (dict: canonical name → float)
- `config` — training args

`load_checkpoint()` reads `thr_any` and `thr_map` automatically. Use `--thr-any` to override only if needed.

## Dataset format (`temporal_bug_dataset.json`)

```json
{
  "videos": [
    {
      "video_id": "fnafautobug2_fnaf_003",
      "video_path": "fnafautobug2/fnaf_003.mp4",   // relative to --video-root
      "duration": 494.07,
      "annotations": [
        {"bug_type": "z_clipping", "start": 25.6, "end": 26.6, "duration": 1.0, "segment_id": 0}
      ]
    }
  ],
  "splits": {"train": [...], "val": [...], "test": [...]}
}
```

- Each annotation has one `bug_type` string (not an array). Overlapping annotations from different classes are separate entries.
- Evaluation is done **per class**: predicted segments for class X are compared against GT annotations for class X.
- No new JSON file needed — `temporal_bug_dataset.json` is used as-is.

## Backend result.json format

```json
{
  "video_id": "job_uuid",
  "video_path": "/path/to/video.mp4",
  "duration": 480.0,
  "annotations": [
    {"bug_type": "z_clipping", "start": 12.1, "end": 18.6, "duration": 6.5, "segment_id": 0},
    {"bug_type": "z_fighting", "start": 12.1, "end": 18.6, "duration": 6.5, "segment_id": 1}
  ]
}
```

Multi-label segments use the same `start`/`end` for each `bug_type` entry. The frontend groups entries sharing `(start, end)` into a single visual segment automatically.

## Metrics computed in eval mode

**Clip-level** (averaged over test videos):
- `f1_types` — micro F1 across all 5 bug type classes per window
- `f1_any` — binary F1 for presence detection per window

**Temporal localization** (global across all test videos):
- `mAP_0.3 / 0.5 / 0.7` — mean Average Precision at tIoU thresholds
- `mean_tIoU` — mean temporal IoU over all matched (pred, GT) pairs
- `per_class / <name> / AP_0.3|0.5|0.7` — per-class AP
- `segment_recall / tIoU_0.3|0.5|0.7` — fraction of GT segments matched

Results saved to `<out-dir>/localization_test_metrics.json` and logged to W&B (`--wandb-project localization-bugs`).

## Running on Narval HPC

```bash
# 1. Set the three path variables at the top of the script
# 2. Submit:
sbatch run_localization.sh
```

Required modules: `gcc/12.3.1 opencv/4.8.1`
Virtual env: `~/env_bugdetection`
Resources: 1 GPU, 4 CPUs, 32 GB RAM, 6-hour time limit

## Backend checkpoint path

Set `CHECKPOINT_PATH` in `backend/inference.py` (line ~28) before starting the server. The model is loaded once on first request and reused for all subsequent jobs.

## Common commands

```bash
# Evaluate full test set locally
python run_localization.py \
  --mode eval \
  --checkpoint /path/to/best_checkpoint.pt \
  --temporal-dataset ./temporal_bug_dataset.json \
  --video-root /path/to/videos \
  --out-dir ./localization_outputs

# Single-video inference
python run_localization.py \
  --mode infer \
  --checkpoint /path/to/best_checkpoint.pt \
  --video-path /path/to/video.mp4 \
  --inference-out result.json

# Start backend
cd backend && pip install -r requirements.txt && uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Start frontend
cd frontend && python -m http.server 3000
```
