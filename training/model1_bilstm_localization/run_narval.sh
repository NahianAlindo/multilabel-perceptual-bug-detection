#!/bin/bash
#SBATCH --account=def-loutfouz
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=m1_bilstm_loc
#SBATCH --output=/home/nahian26/scratch/logs/model1_bilstm_%j.out
#SBATCH --error=/home/nahian26/scratch/logs/model1_bilstm_%j.err
#SBATCH --mail-user=nahian.rifaat@ontariotechu.net
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

# ==============================================================================
# Model 1: BugBiLSTM + Temporal Regression Head
# Server: narval / rorqual  (TensorBoard only — no W&B)
# Use for: supporting runs, small ablations, re-runs of timed-out fir/nibi jobs
#
# TensorBoard (SSH port forward to view locally):
#   ssh -L 6006:localhost:6006 nahian26@narval.computecanada.ca
#   tensorboard --logdir <LOGDIR>/tensorboard --port 6006
#   Open http://localhost:6006
# ==============================================================================

echo "=============================================="
echo "Model 1: BugBiLSTM + Temporal Regression"
echo "Server: narval/rorqual | TensorBoard only"
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

SCRIPT_DIR="/home/nahian26/projects/def-loutfouz/nahian26/bugdatasetbig/model1_bilstm_localization"
VIDEO_ROOT="/home/nahian26/scratch/videos"
DATASET="/home/nahian26/scratch/localization/temporal_bug_dataset.json"
PRETRAIN_CKPT="/home/nahian26/scratch/checkpoints/anygate_wgatedreg/checkpoint_epoch_060.pt"
CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints"
LOGDIR="$SCRIPT_DIR/logs"
# Pre-extracted frame cache (shared by all 3 models; ~66 GB for 263 videos).
# train.py builds any missing entries automatically on first run, then skips.
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
echo "Checking Python packages (installing any missing ones)..."
ensure_pkg () {
    # $1 = import name, $2 = pip package name
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
ensure_pkg decord decord
python -c "import cv2; print(f'  ✓ OpenCV {cv2.__version__}')" || echo "  ⚠ OpenCV not found (will fallback to decord)"
echo "  ℹ W&B disabled on this server (narval/rorqual)"
echo "  ℹ TensorBoard: ssh -L 6006:localhost:6006 nahian26@narval.computecanada.ca"
echo "  ℹ             tensorboard --logdir $LOGDIR/tensorboard --port 6006"
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
# Auto-resume
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
[ ! -f "$SCRIPT_DIR/train.py" ]    && echo "❌ train.py not found" && exit 1
echo "  ✓ train.py"
[ ! -f "$DATASET" ]                && echo "❌ Dataset not found: $DATASET" && exit 1
echo "  ✓ Dataset"
[ ! -d "$VIDEO_ROOT" ]             && echo "❌ Video root not found: $VIDEO_ROOT" && exit 1
echo "  ✓ Video root"
[ -f "$PRETRAIN_CKPT" ] && echo "  ✓ Pretrain checkpoint" || echo "  ⚠ Pretrain checkpoint not found (training from scratch)"
echo "=============================================="

# ==============================================================================
# Run training (no W&B)
# ==============================================================================

PRETRAIN_FLAG=""
[ -f "$PRETRAIN_CKPT" ] && PRETRAIN_FLAG="--pretrain-checkpoint $PRETRAIN_CKPT"

python "$SCRIPT_DIR/train.py" \
    --mode           train \
    --video-root     "$VIDEO_ROOT" \
    --temporal-dataset "$DATASET" \
    --frame-cache-dir "$FRAME_CACHE_DIR" \
    --save-dir       "$CHECKPOINT_DIR" \
    --logdir         "$LOGDIR" \
    --hpo-trials     15 \
    --epochs         100 \
    --batch-size     32 \
    --num-workers    8 \
    --device         cuda \
    --no-wandb \
    --window-sec     2.0 \
    --stride-sec     1.0 \
    --fps            8.0 \
    --img-size       224 \
    $PRETRAIN_FLAG \
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
    BEST_CKPT=$(ls -t "$CHECKPOINT_DIR"/best_composite.pt 2>/dev/null | head -1)
    [ -n "$BEST_CKPT" ] && echo "🏆 Best checkpoint: $BEST_CKPT"
    echo "📊 Test metrics : $LOGDIR/test_metrics.json"
    echo "📈 TensorBoard  : ssh -L 6006:localhost:6006 nahian26@narval.computecanada.ca"
    echo "📥 Download     : scp nahian26@narval:$CHECKPOINT_DIR/best_composite.pt ./"

elif [ $EXIT_CODE -eq 124 ] || [ $EXIT_CODE -eq 140 ]; then
    echo ""
    echo "⏱️  Job reached time limit"
    LATEST=$(ls -t "$CHECKPOINT_DIR"/checkpoint_epoch_*.pt 2>/dev/null | head -1)
    [ -n "$LATEST" ] && echo "📍 Last epoch: $(echo $LATEST | grep -oP 'epoch_\K\d+')"
    echo "   HPO progress: $(tail -1 $LOGDIR/hpo_progress.log 2>/dev/null || echo 'see log')"
    echo "📝 To resume: sbatch $SCRIPT_DIR/run_narval.sh"

else
    echo "❌ Training failed (exit $EXIT_CODE)"
    echo "🔍 Logs: $LOGDIR/../logs/model1_bilstm_${SLURM_JOB_ID}.out"
fi

echo ""
echo "=============================================="
echo "📊 seff $SLURM_JOB_ID"
echo "📧 nahian.rifaat@ontariotechu.net"
echo "🎉 Script completed!"
echo "=============================================="
