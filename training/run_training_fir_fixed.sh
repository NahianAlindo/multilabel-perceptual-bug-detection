#!/bin/bash
#SBATCH --account=def-loutfouz
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=bugdet_original
#SBATCH --output=/home/nahian26/projects/def-loutfouz/nahian26/bugdatasetbig/logs/training_%j.out
#SBATCH --error=/home/nahian26/projects/def-loutfouz/nahian26/bugdatasetbig/logs/training_%j.err
#SBATCH --mail-user=nahian.rifaat@ontariotechu.net
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

# ==============================================================================
# Bug Detection Training - Original Config (60 Epochs)
# For: nahian26@narval
# Data & Scripts: /home/nahian26/projects/def-loutfouz/nahian26/bugdatasetbig/
# ==============================================================================

echo "=============================================="
echo "Bug Detection - Original Config (60 epochs)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "User: nahian26"
echo "Account: def-loutfouz"
echo "=============================================="

# ==============================================================================
# Paths
# ==============================================================================

BASE_DIR="/home/nahian26/projects/def-loutfouz/nahian26/bugdatasetbig"
DATA_ROOT="$BASE_DIR"
MANIFEST="$BASE_DIR/my_new_splits.jsonl"
SCRIPT="$BASE_DIR/scripts/train_bugdetector_original_config.py"
CHECKPOINT_DIR="/home/nahian26/scratch/checkpoints/anygate_wgatedreg"
TENSORBOARD_DIR="/home/nahian26/scratch/runs/anygate_wgatedreg"

echo ""
echo "Paths:"
echo "  Base: $BASE_DIR"
echo "  Data: $DATA_ROOT"
echo "  Manifest: $MANIFEST"
echo "  Script: $SCRIPT"
echo "  Checkpoints: $CHECKPOINT_DIR"
echo "  TensorBoard: $TENSORBOARD_DIR"
echo "=============================================="

# ==============================================================================
# Environment Setup - INCLUDES OPENCV MODULE
# ==============================================================================

echo ""
echo "Loading modules..."

# Load Python module
module load python/3.10
echo "  ✓ Python 3.10"

# Load OpenCV module (required by Alliance Canada)
module load gcc opencv/4.8.1
echo "  ✓ OpenCV 4.8.1"

# Activate virtual environment
echo ""
echo "Activating virtual environment: ~/env_bugdetection"
source ~/env_bugdetection/bin/activate

# Verify activation
if [ -n "$VIRTUAL_ENV" ]; then
    echo "  ✓ Virtual environment activated"
    echo "    Python: $(which python)"
    echo "    Version: $(python --version)"
else
    echo "  ❌ ERROR: Virtual environment not activated!"
    exit 1
fi

# Test OpenCV
echo ""
echo "Testing OpenCV..."
python -c "import cv2; print(f'  ✓ OpenCV {cv2.__version__} available')" || echo "  ❌ OpenCV not found!"

echo "=============================================="

# W&B configuration
export WANDB_API_KEY="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"
export WANDB_PROJECT="bug-detection-production"

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

mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$TENSORBOARD_DIR"
mkdir -p "$BASE_DIR/logs"

# ==============================================================================
# Check for resume checkpoint
# ==============================================================================

RESUME_FLAG=""
LATEST_CHECKPOINT=$(ls -t "$CHECKPOINT_DIR"/checkpoint_epoch_*.pt 2>/dev/null | head -1)

if [ -n "$LATEST_CHECKPOINT" ]; then
    echo ""
    echo "🔄 Found existing checkpoint:"
    echo "   $LATEST_CHECKPOINT"
    EPOCH_NUM=$(echo "$LATEST_CHECKPOINT" | grep -oP 'epoch_\K\d+')
    echo "   Completed epoch: $EPOCH_NUM"
    echo "   Will auto-resume"
    RESUME_FLAG="--resume $LATEST_CHECKPOINT"
else
    echo ""
    echo "Starting fresh training (no checkpoint found)"
fi
echo "=============================================="

# ==============================================================================
# Verify files exist
# ==============================================================================

echo ""
echo "Verifying files..."

if [ ! -f "$SCRIPT" ]; then
    echo "❌ ERROR: Python script not found: $SCRIPT"
    exit 1
fi
echo "✓ Python script: $SCRIPT"

if [ ! -f "$MANIFEST" ]; then
    echo "❌ ERROR: Manifest not found: $MANIFEST"
    exit 1
fi
echo "✓ Manifest: $MANIFEST"

if [ ! -d "$DATA_ROOT/clips" ]; then
    echo "❌ ERROR: Clips directory not found: $DATA_ROOT/clips"
    exit 1
fi
CLIP_COUNT=$(ls "$DATA_ROOT/clips" 2>/dev/null | wc -l)
echo "✓ Clips directory: $DATA_ROOT/clips ($CLIP_COUNT files)"

echo "=============================================="

# ==============================================================================
# Start Training
# ==============================================================================

echo ""
echo "🚀 Starting training..."
echo "📊 Monitor at: https://wandb.ai"
echo "   (Check output below for exact URL)"
echo ""
echo "Configuration:"
echo "  Epochs: 60 (no early stopping)"
echo "  Batch size: 32"
echo "  Learning rate: 7e-4"
echo "  Loss: ASL + Focal"
echo "  EMA: Enabled"
echo "=============================================="
echo ""

# Run training
python "$SCRIPT" \
  --data-root "$DATA_ROOT" \
  --manifests "$MANIFEST" \
  --use-cosine \
  --warmup-epochs 1 \
  --types-loss asl \
  --asl-gamma-pos 0.0 \
  --asl-gamma-neg 2.0 \
  --asl-clip 0.05 \
  --count-focal \
  --count-gamma 1.5 \
  --count-label-smooth 0.10 \
  --presence-recall 0.90 \
  --use-ema \
  --ema-decay 0.999 \
  --sampler-weighted \
  --batch-size 32 \
  --num-workers 8 \
  --lr 7e-4 \
  --weight-decay 3e-4 \
  --any-lambda 1.0 \
  --cons-lambda 0.1 \
  --neg-types-weight 0.25 \
  --neg-count-weight 0.25 \
  --logdir "$TENSORBOARD_DIR" \
  --save-dir "$CHECKPOINT_DIR" \
  --epochs 60 \
  --wandb-project "$WANDB_PROJECT" \
  $RESUME_FLAG

EXIT_CODE=$?

# ==============================================================================
# Post-Training Summary
# ==============================================================================

echo ""
echo "=============================================="
echo "Training completed: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Duration: $((SECONDS/60)) minutes (~$((SECONDS/3600)) hours)"
echo "=============================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully!"
    echo ""
    echo "📊 Results:"
    echo "   Checkpoints: $CHECKPOINT_DIR"
    echo "   TensorBoard: $TENSORBOARD_DIR"
    echo "   Logs: $BASE_DIR/logs/training_${SLURM_JOB_ID}.out"
    echo ""
    
    BEST_CKPT=$(ls -t "$CHECKPOINT_DIR"/*best*.pt 2>/dev/null | head -1)
    if [ -n "$BEST_CKPT" ]; then
        echo "🏆 Best Model: $BEST_CKPT"
        CKPT_SIZE=$(du -h "$BEST_CKPT" | cut -f1)
        echo "   Size: $CKPT_SIZE"
    fi
    echo ""
    
    echo "📈 View Results:"
    echo "   W&B: https://wandb.ai/$USER/$WANDB_PROJECT"
    echo ""
    
    echo "📥 Download:"
    echo "   scp nahian26@narval:$CHECKPOINT_DIR/*best*.pt ./"
    echo ""
    
elif [ $EXIT_CODE -eq 124 ] || [ $EXIT_CODE -eq 140 ]; then
    echo ""
    echo "⏱️  Job reached time limit"
    echo ""
    
    LATEST=$(ls -t "$CHECKPOINT_DIR"/checkpoint_epoch_*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        EPOCH_NUM=$(echo "$LATEST" | grep -oP 'epoch_\K\d+')
        REMAINING=$((60 - EPOCH_NUM))
        
        echo "📍 Progress: Epoch $EPOCH_NUM/60"
        echo "   Remaining: ~$REMAINING epochs"
        echo ""
    fi
    
    echo "📝 To continue:"
    echo "   cd $BASE_DIR"
    echo "   sbatch run_training_original_config.sh"
    echo ""
    
else
    echo ""
    echo "❌ Training failed with exit code: $EXIT_CODE"
    echo ""
    echo "🔍 Check logs:"
    echo "   $BASE_DIR/logs/training_${SLURM_JOB_ID}.out"
    echo "   $BASE_DIR/logs/training_${SLURM_JOB_ID}.err"
    echo ""
fi

echo "=============================================="
echo "📊 Job Statistics:"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo ""
echo "Run 'seff $SLURM_JOB_ID' for efficiency report"
echo "=============================================="

echo ""
echo "📧 Email sent to: nahian.rifaat@ontariotechu.net"
echo ""
echo "=============================================="
echo "🎉 Script completed!"
echo "=============================================="
