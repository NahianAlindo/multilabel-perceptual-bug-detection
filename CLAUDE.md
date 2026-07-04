# Multi-Label Perceptual Bug Detection — CLAUDE.md

## Project Goal

Detect and **temporally localize** visual rendering bugs in full-length gameplay videos (up to 20 min). The system outputs timestamped segments indicating *when* and *which* bug types occur. 5 bug classes: `z-clipping`, `corrupted_texture`, `geometry_corruption`, `z-fighting`, `boundary_hole`.

## Current Baseline Results (test set, 57 videos)

| Metric | Value | Status |
|---|---|---|
| Clip micro F1 | 0.61 | acceptable |
| Clip presence F1 | 0.71 | acceptable |
| mAP@0.3 | 0.17 | needs improvement |
| mAP@0.5 | 0.02 | critical |
| mAP@0.7 | 0.00 | critical |
| Mean tIoU | 0.27 | critical |
| Segment over-prediction | ~1.5–2.5× | critical |

Checkpoint: `anygate_wgatedreg/checkpoint_epoch_060.pt`

## Active Branch

`localization-improvements` — implements the 3-phase improvement plan below.

## HPC Servers

| Server | GPU | W&B | Priority |
|---|---|---|---|
| fir (SHARCNET) | H100 | ✓ | ★ primary |
| nibi (SHARCNET) | H100 | ✓ | ★ primary |
| rorqual | H100 | ✗ | secondary |
| narval | A100 | ✗ | supporting only |

Account: `def-loutfouz` | User: `nahian26` | Email: `nahian.rifaat@ontariotechu.net`
Virtual env: `~/env_bugdetection` | Modules: `python/3.10 gcc/12.3.1 opencv/4.8.1`

## Directory Structure

```
multilabel-perceptual-bug-detection/
├── localization/
│   ├── run_localization.py        # sliding-window inference + eval (Phase 1 target)
│   ├── run_localization.sh        # SLURM eval job (narval)
│   ├── temporal_bug_dataset.json  # 263 videos + temporal annotations + splits
│   ├── localization_test_metrics.json  # baseline results
│   ├── kaggle_inference_server.py # NEW (Phase 3): async FastAPI + pyngrok
│   ├── backend/
│   │   ├── main.py                # FastAPI server
│   │   └── inference.py           # calls run_localization.py pipeline
│   └── frontend/
│       └── index.html             # SPA visualization
├── training/
│   ├── train_FINAL_ALL_METRICS.py # source of truth: BugBiLSTM architecture
│   ├── model1_bilstm_localization/ # NEW (Phase 2): BugBiLSTM + regression head
│   ├── model2_mstcn_localization/  # NEW (Phase 2): MS-TCN dense segmentation
│   └── model3_pyramid_transformer/ # NEW (Phase 2): Swin + Transformer + FPN
└── CLAUDE.md                      # this file
```

---

## Phase 1: Inference-Time Fixes (localization/run_localization.py)

`merge_window_predictions()` now accepts:
- `--min-conf 0.35` — discard merged segments below mean window confidence
- `--min-duration 1.0` — discard segments shorter than 1 second
- `--max-gap 1.0` — maximum gap (sec) between windows to merge
- `--stride-sec 0.5` — finer boundary resolution (vs default 1.0)

---

## Phase 2: Retraining — Research Workflow

**Architecture selection first, then HPO:**
1. Pilot each model (5 epochs, `--pilot`) → compare val mAP@0.5
2. HPO on selected architecture (`--hpo-trials 15`, Optuna TPE Bayesian)
3. Full training with best hyperparams → auto test at end

**Multi-task early stopping:**
composite = `0.3 × val_clip_micro_F1 + 0.7 × val_mAP@0.5`
Patience: 15 epochs. LR floor (1e-6) as hard stop. Three checkpoints: `best_composite.pt`, `best_f1.pt`, `best_mAP.pt`.

**Each model folder has:**
- `train.py` — `--mode train|infer`, pilot/HPO/full training + auto test
- `run_fir.sh` — fir/nibi SLURM script (W&B + TensorBoard)
- `run_narval.sh` — narval/rorqual SLURM script (TensorBoard only)
- `checkpoints/` — best_*.pt + rolling last 3 epoch checkpoints
- `logs/` — optuna_study.db, hpo_progress.log, test_metrics.json, tensorboard/

### Model 1: BugBiLSTM + Temporal Regression (training/model1_bilstm_localization/)
ResNet18 + BiLSTM (init from existing checkpoint) + temporal offset regression head.
Loss: ASL (Ridnik ICCV'21) + DIoU (Zheng AAAI'20) + Focal (Lin ICCV'17)

### Model 2: MS-TCN Dense Segmentation (training/model2_mstcn_localization/)
ResNet18 + 4-stage 1D dilated MS-TCN. Per-frame multi-label predictions.
Loss: ASL + Truncated MSE Smoothing (Abu Farha CVPR'19)

### Model 3: Feature Pyramid + Transformer / PRIMARY (training/model3_pyramid_transformer/)
Swin-S (timm) + TransformerEncoder + 1D multi-scale FPN + anchor-free head.
Loss: ASL + DIoU + Centerness BCE (ActionFormer ECCV'22 concept, FCOS ICCV'19)

---

## Phase 3: Kaggle + ngrok Async Backend

Inference runs on Kaggle kernel (free GPU), exposed via ngrok. Async pattern — no timeout:
1. Frontend POST → returns job_id immediately
2. Kaggle runs inference in background thread
3. Frontend polls GET /status/{job_id} every 3s
4. When done, fetches GET /result/{job_id} → renders detection timeline

Set `INFERENCE_MODE=kaggle` + `KAGGLE_NGROK_URL=<url>` in backend.

---

## Sliding Window Parameters (must match training)

| Param | Value |
|---|---|
| Window size | 2.0 sec |
| Default stride | 1.0 sec (use 0.5 for finer eval) |
| FPS | 8 |
| Frame size | 224×224 |
| Frames/window | 16 |

## Bug Type Normalization

`z_clipping` ↔ `z-clipping`, `z_fighting` ↔ `z-fighting` (handled by `norm_bug_type()`).
Output uses underscores; model uses hyphens.

## TensorBoard on narval/rorqual (SSH port forwarding)

```bash
ssh -L 6006:localhost:6006 nahian26@narval.computecanada.ca
tensorboard --logdir training/model3_pyramid_transformer/logs/tensorboard/ --port 6006
# Open http://localhost:6006
```
