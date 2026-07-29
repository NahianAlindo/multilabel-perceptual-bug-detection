#!/bin/bash
#SBATCH --account=def-loutfouz_gpu
#SBATCH --partition=gpubase_bygpu_b4
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --job-name=m3_pilot_clean
#SBATCH --output=/home/nahian26/scratch/logs/model3_pilot_%j.out
#SBATCH --error=/home/nahian26/scratch/logs/model3_pilot_%j.err
#SBATCH --mail-user=nahian.rifaat@ontariotechu.net
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

# ==============================================================================
# Model 3: 5-epoch pilot on the corrected frame cache
# Server: fir / nibi
#
# Cheap confirmation step before committing a 2-3 day allocation to a full
# clean retrain. The frame-cache indexing bug (cache's actual fps silently
# differing from the nominal 8fps for most native capture rates, e.g. 30fps
# sources landing at 7.5fps) has been fixed both in extraction (new caches
# are genuinely 8fps) and in indexing (uses each video's real cache fps).
#
# PREREQUISITE: rename/move the OLD frame cache directory out of the way
# first (you're doing this: old -> frame_cache_fps8_224-bak2) so this run's
# ensure_frame_cache() sees an empty dir at FRAME_CACHE_DIR and regenerates
# every video with the corrected extraction (~30-60 min one-time, per
# CLAUDE.md) instead of silently reusing the old drifted cache.
#
# Uses fresh --save-dir/--logdir (suffixed _clean) so the original
# (corrupted-data) run's checkpoints/logs are left untouched for comparison
# — no renaming needed on your end for those.
#
# What to look for in the output: per-class classification signal should
# stop being near-random. Pilot mode doesn't compute per-class AUC-ROC
# directly, but watch F1 and mAP@0.5 move meaningfully above the corrupted
# run's early-epoch numbers — if the fix worked, this pilot's 5-epoch
# trajectory should look like normal learning, not noise.
# ==============================================================================

echo "=============================================="
echo "Model 3: Pilot on corrected frame cache (5 epochs, no HPO)"
echo "=============================================="
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Start      : $(date)"
echo "=============================================="

REPO_DIR="/home/nahian26/scratch/multilabel-perceptual-bug-detection"
SCRIPT_DIR="$REPO_DIR/training/model3_pyramid_transformer"
VIDEO_ROOT="/home/nahian26/scratch"
DATASET="/home/nahian26/scratch/temporal_bug_dataset.json"
CHECKPOINT_DIR="/home/nahian26/scratch/checkpoints/model3_pyramid_transformer_clean"
LOGDIR="/home/nahian26/scratch/runs/model3_pyramid_transformer_clean_pilot"
FRAME_CACHE_DIR="/home/nahian26/scratch/frame_cache_fps8_224"

echo ""
echo "Paths:"
echo "  Script dir  : $SCRIPT_DIR"
echo "  Checkpoints : $CHECKPOINT_DIR (fresh, does not touch the original run)"
echo "  Logs        : $LOGDIR"
echo "  Frame cache : $FRAME_CACHE_DIR (must be the RENAMED-EMPTY path — regenerates here)"
echo "=============================================="

echo ""
echo "Loading modules..."
module load python/3.10
module load gcc/12.3 opencv/4.8.1 2>/dev/null || module load gcc/12.3.1 opencv/4.8.1 2>/dev/null || \
    module load gcc opencv/4.8.1 2>/dev/null || echo "  ⚠️  OpenCV module optional"

echo ""
echo "Activating virtual environment..."
source ~/env_bugdetection/bin/activate
if [ -n "$VIRTUAL_ENV" ]; then
    echo "  ✓ Activated: $(which python) | $(python --version)"
else
    echo "  ❌ ERROR: Virtual environment not activated!"
    exit 1
fi

export OMP_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=============================================="

mkdir -p "$CHECKPOINT_DIR" "$LOGDIR/tensorboard" /home/nahian26/scratch/logs

# Guard: refuse to run if the frame cache dir still has the OLD (drifted)
# content sitting in it — this pilot is only meaningful against a cache
# that gets freshly (re)built with the corrected extraction.
EXISTING_NPY_COUNT=$(ls "$FRAME_CACHE_DIR"/*.npy 2>/dev/null | wc -l)
if [ "$EXISTING_NPY_COUNT" -gt 0 ]; then
    echo ""
    echo "⚠️  WARNING: $EXISTING_NPY_COUNT .npy files already present in $FRAME_CACHE_DIR"
    echo "   If these are from BEFORE the extraction fix, this pilot will silently"
    echo "   reuse the old drifted cache instead of regenerating it (ensure_frame_cache"
    echo "   only extracts videos that are missing). Rename/clear this dir first if you"
    echo "   haven't already moved the old cache to frame_cache_fps8_224-bak2."
fi

[ ! -f "$SCRIPT_DIR/train.py" ] && echo "❌ train.py not found: $SCRIPT_DIR/train.py" && exit 1
[ ! -f "$DATASET" ]             && echo "❌ Dataset not found: $DATASET" && exit 1
echo "  ✓ train.py + dataset found"
echo "=============================================="
echo ""

python "$SCRIPT_DIR/train.py" \
    --mode           train \
    --pilot \
    --video-root     "$VIDEO_ROOT" \
    --temporal-dataset "$DATASET" \
    --frame-cache-dir "$FRAME_CACHE_DIR" \
    --save-dir       "$CHECKPOINT_DIR" \
    --logdir         "$LOGDIR" \
    --no-wandb \
    --batch-size     4 \
    --num-workers    8 \
    --device         cuda \
    --backbone       swin_small_patch4_window7_224 \
    --window-sec     4.0 \
    --stride-sec     1.0 \
    --fps            8.0 \
    --img-size       224

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Job finished: $(date)"
echo "Exit code   : $EXIT_CODE"
echo "Duration    : $((SECONDS/60)) min"
echo "=============================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Pilot complete — review the per-epoch F1/mAP@0.5 trend above."
    echo "   If it looks like real learning (not flat/noisy), proceed to the full"
    echo "   clean retrain: sbatch run_fir.sh (after repointing its --save-dir/--logdir"
    echo "   to the _clean paths so it doesn't collide with the original corrupted run)."
else
    echo "❌ Pilot failed (exit $EXIT_CODE) — check logs before running the full retrain."
    echo "   /home/nahian26/scratch/logs/model3_pilot_${SLURM_JOB_ID}.out"
    echo "   /home/nahian26/scratch/logs/model3_pilot_${SLURM_JOB_ID}.err"
fi
echo "=============================================="
