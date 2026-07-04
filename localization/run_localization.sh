#!/bin/bash
# ==============================================================================
# run_localization.sh — SLURM job for temporal bug localization evaluation
#
# Runs sliding-window inference over all test-split videos in
# temporal_bug_dataset.json using the pretrained BugBiLSTM checkpoint.
# No training — the existing checkpoint is used as-is.
#
# Usage (from the localization/ directory on Narval):
#   sbatch run_localization.sh
#
# Adjust the three path variables below before submitting.
# ==============================================================================

#SBATCH --job-name=loc_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/loc_eval_%j.out
#SBATCH --error=logs/loc_eval_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# ── Paths — edit these before submitting ──────────────────────────────────────

# Path to best_*.pt checkpoint from train_FINAL_ALL_METRICS.py
CHECKPOINT="$SCRATCH/checkpoints/best_checkpoint.pt"

# Root folder containing game subfolders with full-resolution videos
# Structure: $VIDEO_ROOT/fnafautobug2/fnaf_003.mp4
#                        mkartautobug2/mkart_001.mp4  etc.
VIDEO_ROOT="$SCRATCH/videos"

# Dataset JSON (keep in the localization folder or copy to scratch)
DATASET="$SCRATCH/localization/temporal_bug_dataset.json"

# Output directory for localization_test_metrics.json
OUT_DIR="$SCRATCH/localization/eval_outputs"

# ── Environment ───────────────────────────────────────────────────────────────
module load gcc/12.3.1 opencv/4.8.1
source ~/env_bugdetection/bin/activate

mkdir -p "$OUT_DIR" logs

echo "============================================================"
echo " Job ID        : $SLURM_JOB_ID"
echo " Node          : $SLURMD_NODENAME"
echo " Checkpoint    : $CHECKPOINT"
echo " Video root    : $VIDEO_ROOT"
echo " Dataset       : $DATASET"
echo " Output dir    : $OUT_DIR"
echo " Start time    : $(date)"
echo "============================================================"

# ── Evaluation run ────────────────────────────────────────────────────────────
python run_localization.py \
    --mode          eval \
    --checkpoint    "$CHECKPOINT" \
    --temporal-dataset "$DATASET" \
    --video-root    "$VIDEO_ROOT" \
    --out-dir       "$OUT_DIR" \
    --window-sec    2.0 \
    --stride-sec    1.0 \
    --fps           8.0 \
    --img-size      224 \
    --batch-size    16 \
    --num-workers   4 \
    --device        cuda \
    --wandb-project "localization-bugs"
    # Add --no-wandb above to disable W&B logging

EXIT_CODE=$?
echo "============================================================"
echo " End time   : $(date)"
echo " Exit code  : $EXIT_CODE"
echo "============================================================"
exit $EXIT_CODE
