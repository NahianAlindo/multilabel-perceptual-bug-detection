#!/bin/bash
#SBATCH --account=def-loutfouz_gpu
#SBATCH --partition=gpubase_bygpu_b4
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=m3_clean_tr
#SBATCH --output=/home/nahian26/scratch/logs/model3_clean_%j.out
#SBATCH --error=/home/nahian26/scratch/logs/model3_clean_%j.err
#SBATCH --mail-user=nahian.rifaat@ontariotechu.net
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

# ==============================================================================
# Model 3: Feature Pyramid + Transformer  [PRIMARY MODEL] — CLEAN RETRAIN
# Server: fir / nibi  (H100, W&B + TensorBoard)
# Runs: HPO (15 trials) -> full training -> test (one job)
#
# This is run_fir.sh's original full pipeline, repointed at fresh
# --save-dir/--logdir (suffixed _clean) so it never collides with or
# resumes from the ORIGINAL run's checkpoints/best_hparams.json — those
# were produced on data affected by the frame-cache fps-drift bug (native
# fps not evenly divisible by 8 -> cache silently sampled at ~7.5fps while
# indexing assumed 8fps, drifting frames away from their labels the longer
# each video ran). Both the extraction and the indexing are now fixed in
# train.py; this run's own --frame-cache-dir must point at an EMPTY dir
# (rename the old cache out of the way first, e.g. to
# frame_cache_fps8_224-bak2) so ensure_frame_cache() regenerates every
# video with the corrected extraction instead of silently reusing the old
# drifted one — it only re-extracts files that are missing.
#
# Only submit this after run_fir_pilot.sh's 5-epoch pilot confirms the fix
# is working (F1/mAP@0.5 trending like real learning, not flat/noisy).
#
# Architecture: Swin-S (timm) + TransformerEncoder + 1D multi-scale FPN
#               + anchor-free head (ActionFormer concept, ECCV 2022)
# Target: mAP@0.5 > 0.15, mAP@0.3 > 0.35, over-pred ratio < 1.2
# ==============================================================================

echo "=============================================="
echo "Model 3: Feature Pyramid + Transformer [PRIMARY] — CLEAN RETRAIN"
echo "Server: fir/nibi | W&B + TensorBoard"
echo "=============================================="
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Start      : $(date)"
echo "User       : nahian26"
echo "Account    : def-loutfouz"
echo "=============================================="

# ==============================================================================
# Paths — edit before submitting
# ==============================================================================

REPO_DIR="/home/nahian26/scratch/multilabel-perceptual-bug-detection"
SCRIPT_DIR="$REPO_DIR/training/model3_pyramid_transformer"
VIDEO_ROOT="/home/nahian26/scratch"
DATASET="/home/nahian26/scratch/temporal_bug_dataset.json"
# Fresh, separate from the original (corrupted-data) run — nothing there is touched
CHECKPOINT_DIR="/home/nahian26/scratch/checkpoints/model3_pyramid_transformer_clean"
LOGDIR="/home/nahian26/scratch/runs/model3_pyramid_transformer_clean"
# Must be an EMPTY dir at submit time (old cache renamed out of the way) so every
# video gets regenerated with the corrected extraction, not silently skipped.
FRAME_CACHE_DIR="/home/nahian26/scratch/frame_cache_fps8_224"

echo ""
echo "Paths:"
echo "  Script dir  : $SCRIPT_DIR"
echo "  Video root  : $VIDEO_ROOT"
echo "  Dataset     : $DATASET"
echo "  Checkpoints : $CHECKPOINT_DIR"
echo "  Logs        : $LOGDIR"
echo "  Frame cache : $FRAME_CACHE_DIR"
echo "=============================================="

# ==============================================================================
# Environment
# ==============================================================================

echo ""
echo "Loading modules..."
module load python/3.10
echo "  ✓ Python 3.10"
module load gcc/12.3 opencv/4.8.1 2>/dev/null || module load gcc/12.3.1 opencv/4.8.1 2>/dev/null || \
    module load gcc opencv/4.8.1 2>/dev/null || echo "  ⚠️  OpenCV module optional"
echo "  ✓ GCC + OpenCV (if available)"

echo ""
echo "Activating virtual environment..."
source ~/env_bugdetection/bin/activate
if [ -n "$VIRTUAL_ENV" ]; then
    echo "  ✓ Activated: $(which python) | $(python --version)"
else
    echo "  ❌ ERROR: Virtual environment not activated!"
    exit 1
fi

echo ""
echo "Checking Python packages (installing any missing ones)..."
ensure_pkg () {
    if python -c "import $1" 2>/dev/null; then
        echo "  ✓ $2"
    else
        echo "  ⚠ $2 missing — installing..."
        pip install --no-index "$2" 2>/dev/null || pip install "$2" || \
            echo "  ❌ Could not install $2 (continuing — a fallback may exist)"
    fi
}
ensure_pkg torch torch
ensure_pkg torchvision torchvision
ensure_pkg numpy numpy
ensure_pkg tqdm tqdm
ensure_pkg sklearn scikit-learn
ensure_pkg matplotlib matplotlib
ensure_pkg tensorboard tensorboard
ensure_pkg optuna optuna
ensure_pkg torchinfo torchinfo
ensure_pkg wandb wandb
ensure_pkg decord decord
ensure_pkg timm timm
python -c "import cv2; print(f'  ✓ OpenCV {cv2.__version__}')" || echo "  ⚠ OpenCV not found (will fallback to decord)"
python -c "import timm" 2>/dev/null || { echo "  ❌ timm still missing (required for Swin backbone)"; exit 1; }

export OMP_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# W&B configuration
export WANDB_API_KEY="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"
export WANDB_MODE="online"
export WANDB_PROJECT="basic-intro"
echo ""
echo "  ✓ W&B project: $WANDB_PROJECT"
echo "=============================================="

echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=============================================="

mkdir -p "$CHECKPOINT_DIR" "$LOGDIR/tensorboard" /home/nahian26/scratch/logs

# ==============================================================================
# Guard: refuse a silent no-op re-extraction if the "empty" cache dir isn't
# actually empty (e.g. the rename to -bak2 didn't happen / happened to the
# wrong path) — better to fail loudly here than train on stale data again.
# ==============================================================================

EXISTING_NPY_COUNT=$(ls "$FRAME_CACHE_DIR"/*.npy 2>/dev/null | wc -l)
TOTAL_VIDEO_COUNT=$(python -c "import json; d=json.load(open('$DATASET')); print(len(d.get('videos', [])))" 2>/dev/null || echo 0)
if [ "$EXISTING_NPY_COUNT" -gt 0 ] && [ "$EXISTING_NPY_COUNT" -ge "$TOTAL_VIDEO_COUNT" ]; then
    echo ""
    echo "⚠️  WARNING: $EXISTING_NPY_COUNT .npy files already present in $FRAME_CACHE_DIR"
    echo "   (dataset has $TOTAL_VIDEO_COUNT videos) — ensure_frame_cache() will treat this"
    echo "   as already-complete and SKIP extraction, silently reusing whatever is there."
    echo "   If this is the OLD drifted cache (not yet renamed to -bak2), this run will"
    echo "   repeat the original bug. Ctrl-C / scancel this job and verify first if unsure."
fi

# ==============================================================================
# Auto-resume: detect latest epoch checkpoint (within THIS clean run's own dir)
# ==============================================================================

RESUME_FLAG=""
LATEST_CHECKPOINT=$(ls -t "$CHECKPOINT_DIR"/checkpoint_epoch_*.pt 2>/dev/null | head -1)
if [ -n "$LATEST_CHECKPOINT" ]; then
    echo ""
    echo "🔄 Found existing checkpoint: $LATEST_CHECKPOINT"
    EPOCH_NUM=$(echo "$LATEST_CHECKPOINT" | grep -oP 'epoch_\K\d+')
    echo "   Completed epoch: $EPOCH_NUM — will resume"
    RESUME_FLAG="--checkpoint $LATEST_CHECKPOINT"
else
    echo ""
    echo "Starting fresh training (no checkpoint found in $CHECKPOINT_DIR)"
fi

echo ""
echo "Verifying files..."
[ ! -f "$SCRIPT_DIR/train.py" ]    && echo "❌ train.py not found: $SCRIPT_DIR/train.py" && exit 1
echo "  ✓ train.py"
[ ! -f "$DATASET" ]                && echo "❌ Dataset not found: $DATASET" && exit 1
echo "  ✓ Dataset"
[ ! -d "$VIDEO_ROOT" ]             && echo "❌ Video root not found: $VIDEO_ROOT" && exit 1
echo "  ✓ Video root"
echo "=============================================="

echo ""
echo "Configuration:"
echo "  Mode      : train (HPO trials=15, full training → test) — CLEAN frame cache"
echo "  Backbone  : swin_small_patch4_window7_224 (timm pretrained)"
echo "  Stopping  : composite(0.3*F1 + 0.7*mAP@0.5), patience=15, lr_floor=1e-6"
echo "  Max epochs: 100"
echo "  Window    : 4.0s | Stride: 1.0s | FPS: 8.0"
echo "=============================================="
echo ""

python "$SCRIPT_DIR/train.py" \
    --mode           train \
    --video-root     "$VIDEO_ROOT" \
    --temporal-dataset "$DATASET" \
    --frame-cache-dir "$FRAME_CACHE_DIR" \
    --save-dir       "$CHECKPOINT_DIR" \
    --logdir         "$LOGDIR" \
    --hpo-trials     15 \
    --epochs         100 \
    --batch-size     4 \
    --num-workers    8 \
    --device         cuda \
    --wandb-project  "$WANDB_PROJECT" \
    --backbone       swin_small_patch4_window7_224 \
    --window-sec     4.0 \
    --stride-sec     1.0 \
    --fps            8.0 \
    --img-size       224 \
    $RESUME_FLAG

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Job finished: $(date)"
echo "Exit code   : $EXIT_CODE"
echo "Duration    : $((SECONDS/60)) min (~$((SECONDS/3600))h)"
echo "=============================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Clean training + test complete!"
    BEST_CKPT=$(ls -t "$CHECKPOINT_DIR"/best_composite.pt 2>/dev/null | head -1)
    [ -n "$BEST_CKPT" ] && echo "🏆 Best checkpoint: $BEST_CKPT ($(du -h $BEST_CKPT | cut -f1))"
    echo ""
    echo "📊 Test metrics : $LOGDIR/test_metrics.json"
    echo "📈 W&B          : https://wandb.ai/nahian26/$WANDB_PROJECT"
    echo "📥 Download     : scp nahian26@fir:$CHECKPOINT_DIR/best_composite.pt ./"
elif [ $EXIT_CODE -eq 124 ] || [ $EXIT_CODE -eq 140 ]; then
    echo ""
    echo "⏱️  Job reached time limit"
    LATEST=$(ls -t "$CHECKPOINT_DIR"/checkpoint_epoch_*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        EPOCH_NUM=$(echo "$LATEST" | grep -oP 'epoch_\K\d+')
        echo "📍 Last epoch: $EPOCH_NUM"
    fi
    echo ""
    echo "📝 To resume: sbatch $SCRIPT_DIR/run_fir_clean.sh"
    echo "   (checkpoint auto-detected in $CHECKPOINT_DIR)"
else
    echo ""
    echo "❌ Training failed (exit $EXIT_CODE)"
    echo "🔍 Logs:"
    echo "   /home/nahian26/scratch/logs/model3_clean_${SLURM_JOB_ID}.out"
    echo "   /home/nahian26/scratch/logs/model3_clean_${SLURM_JOB_ID}.err"
fi

echo ""
echo "=============================================="
echo "📊 Job statistics: run 'seff $SLURM_JOB_ID'"
echo "=============================================="
