#!/bin/bash
#SBATCH --account=def-loutfouz
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=m2_mstcn_loc
#SBATCH --output=/home/nahian26/scratch/logs/model2_mstcn_%j.out
#SBATCH --error=/home/nahian26/scratch/logs/model2_mstcn_%j.err
#SBATCH --mail-user=nahian.rifaat@ontariotechu.net
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

# ==============================================================================
# Model 2: MS-TCN Dense Temporal Segmentation
# Server: fir / nibi  (H100, W&B + TensorBoard)
# Runs: HPO (15 trials) → full training → test (one job)
# ==============================================================================

echo "=============================================="
echo "Model 2: MS-TCN Dense Temporal Segmentation"
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

SCRIPT_DIR="/home/nahian26/projects/def-loutfouz/nahian26/bugdatasetbig/model2_mstcn_localization"
VIDEO_ROOT="/home/nahian26/scratch/videos"
DATASET="/home/nahian26/scratch/localization/temporal_bug_dataset.json"
CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints"
LOGDIR="$SCRIPT_DIR/logs"

echo ""
echo "Paths:"
echo "  Script dir  : $SCRIPT_DIR"
echo "  Video root  : $VIDEO_ROOT"
echo "  Dataset     : $DATASET"
echo "  Checkpoints : $CHECKPOINT_DIR"
echo "  Logs        : $LOGDIR"
echo "=============================================="

# ==============================================================================
# Environment
# ==============================================================================

echo ""
echo "Loading modules..."
module load python/3.10
echo "  ✓ Python 3.10"
module load gcc/12.3.1 opencv/4.8.1
echo "  ✓ GCC 12.3.1 + OpenCV 4.8.1"

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
echo "Testing OpenCV..."
python -c "import cv2; print(f'  ✓ OpenCV {cv2.__version__}')" || echo "  ⚠ OpenCV not found (will fallback to decord)"

# W&B
export WANDB_API_KEY="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"
export WANDB_PROJECT="localization-bugs"
echo ""
echo "  ✓ W&B project: $WANDB_PROJECT"
echo "=============================================="

# ==============================================================================
# GPU Information
# ==============================================================================

echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=============================================="

# ==============================================================================
# Create output directories
# ==============================================================================

mkdir -p "$CHECKPOINT_DIR" "$LOGDIR/tensorboard" "$LOGDIR/../logs"

# ==============================================================================
# Auto-resume: detect latest epoch checkpoint
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
    echo "Starting fresh training (no checkpoint found)"
fi

# ==============================================================================
# Verify required files
# ==============================================================================

echo ""
echo "Verifying files..."
[ ! -f "$SCRIPT_DIR/train.py" ]    && echo "❌ train.py not found: $SCRIPT_DIR/train.py" && exit 1
echo "  ✓ train.py"
[ ! -f "$DATASET" ]                && echo "❌ Dataset not found: $DATASET" && exit 1
echo "  ✓ Dataset"
[ ! -d "$VIDEO_ROOT" ]             && echo "❌ Video root not found: $VIDEO_ROOT" && exit 1
echo "  ✓ Video root"
echo "=============================================="

# ==============================================================================
# Configuration summary
# ==============================================================================

echo ""
echo "Configuration:"
echo "  Mode      : train (HPO trials=15, full training → test)"
echo "  Backbone  : ResNet18 (pretrained ImageNet)"
echo "  Temporal  : MS-TCN (4 stages, dilated 1D conv, dil=1,2,4,8,16)"
echo "  Head      : per-frame multi-label classification"
echo "  Loss      : ASL + Truncated MSE smoothing (summed across 4 stages)"
echo "  Stopping  : composite(0.3*F1 + 0.7*mAP), patience=15, lr_floor=1e-6"
echo "  Max epochs: 100"
echo "=============================================="
echo ""

# ==============================================================================
# Run training
# ==============================================================================

python "$SCRIPT_DIR/train.py" \
    --mode           train \
    --video-root     "$VIDEO_ROOT" \
    --temporal-dataset "$DATASET" \
    --save-dir       "$CHECKPOINT_DIR" \
    --logdir         "$LOGDIR" \
    --hpo-trials     15 \
    --epochs         100 \
    --batch-size     4 \
    --num-workers    8 \
    --device         cuda \
    --wandb-project  "$WANDB_PROJECT" \
    --fps            8.0 \
    --img-size       224 \
    $RESUME_FLAG

EXIT_CODE=$?

# ==============================================================================
# Post-training summary
# ==============================================================================

echo ""
echo "=============================================="
echo "Job finished: $(date)"
echo "Exit code   : $EXIT_CODE"
echo "Duration    : $((SECONDS/60)) min (~$((SECONDS/3600))h)"
echo "=============================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Training + test complete!"
    echo ""
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
        echo "   HPO progress: $(tail -1 $LOGDIR/hpo_progress.log 2>/dev/null || echo 'see log')"
    fi
    echo ""
    echo "📝 To resume: sbatch $SCRIPT_DIR/run_fir.sh"
    echo "   (Optuna study + checkpoint auto-detected)"

else
    echo ""
    echo "❌ Training failed (exit $EXIT_CODE)"
    echo "🔍 Logs:"
    echo "   $LOGDIR/../logs/model2_mstcn_${SLURM_JOB_ID}.out"
    echo "   $LOGDIR/../logs/model2_mstcn_${SLURM_JOB_ID}.err"
fi

echo ""
echo "=============================================="
echo "📊 Job statistics: run 'seff $SLURM_JOB_ID'"
echo "=============================================="
echo "📧 Email: nahian.rifaat@ontariotechu.net"
echo "=============================================="
echo "🎉 Script completed!"
echo "=============================================="
