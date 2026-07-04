#!/bin/bash
#SBATCH --account=def-loutfouz
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=bugdet_r3d18
#SBATCH --output=/home/nahian26/projects/def-loutfouz/nahian26/bugdatasetbig/logs_r3d/training_%j.out
#SBATCH --error=/home/nahian26/projects/def-loutfouz/nahian26/bugdatasetbig/logs_r3d/training_%j.err
#SBATCH --mail-user=nahian.rifaat@ontariotechu.net
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

# ==============================================================================
# R3D-18 MODEL - Bug Detection Training (Train + Val + Test)
# For: nahian26@nibi.alliancecan.ca
# Data: /home/nahian26/projects/def-loutfouz/nahian26/bugdatasetbig/
#
# COMPLETE PIPELINE:
# 1. Train on "train" split (60 epochs)
# 2. Validate on "val" split (every epoch)
# 3. Test on "test" split (automatic after training)
# 4. Generate 4 visualization charts with 4 decimal precision
# 5. Save test_results.json with comprehensive metrics
# ==============================================================================

echo "=============================================="
echo "R3D-18 MODEL - Bug Detection Training"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "User: nahian26"
echo "Account: def-loutfouz"
echo "Server: Nibi"
echo "=============================================="

# ==============================================================================
# Paths (R3D-specific directories)
# ==============================================================================

BASE_DIR="/home/nahian26/projects/def-loutfouz/nahian26/bugdatasetbig"
DATA_ROOT="$BASE_DIR"
MANIFEST="$BASE_DIR/my_new_splits.jsonl"
SCRIPT="$BASE_DIR/scripts_r3d/train_r3d18_complete.py"
CHECKPOINT_DIR="/home/nahian26/scratch/checkpoints/r3d18_model"
TENSORBOARD_DIR="/home/nahian26/scratch/runs/r3d18_model"
LOGS_DIR="$BASE_DIR/logs_r3d"

echo ""
echo "Paths:"
echo "  Base: $BASE_DIR"
echo "  Data: $DATA_ROOT"
echo "  Manifest: $MANIFEST"
echo "  Script: $SCRIPT"
echo "  Checkpoints: $CHECKPOINT_DIR"
echo "  TensorBoard: $TENSORBOARD_DIR"
echo "  Logs: $LOGS_DIR"
echo "=============================================="

# ==============================================================================
# Environment Setup
# ==============================================================================

echo ""
echo "Loading modules..."

# Load Python module
module load python/3.10
echo "  ✓ Python 3.10"

# Load GCC and OpenCV (with fallback)
module load gcc/12.3 opencv/4.8.1 2>/dev/null || module load gcc opencv/4.8.1 2>/dev/null || echo "  ⚠️  OpenCV module optional"
echo "  ✓ GCC + OpenCV (if available)"

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

# Test OpenCV (optional - script has fallbacks)
echo ""
echo "Testing OpenCV..."
python -c "import cv2; print(f'  ✓ OpenCV {cv2.__version__} available')" 2>/dev/null || echo "  ⚠️  OpenCV not found (will use decord/torchvision fallback)"

echo "=============================================="

# W&B configuration (with fallback)
export WANDB_API_KEY="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"
export WANDB_PROJECT="r3d18-bug-detection"
export WANDB_MODE="online"  # or "disabled" to skip W&B

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
mkdir -p "$LOGS_DIR"
mkdir -p "$CHECKPOINT_DIR/visualizations"

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
    echo "   Will auto-resume from epoch $((EPOCH_NUM + 1))"
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
    echo "   Please upload train_r3d18_complete.py to scripts_r3d/ directory"
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
# Package verification
# ==============================================================================

echo ""
echo "Verifying required packages..."

# Check critical packages
python -c "import torch; print(f'  ✓ PyTorch {torch.__version__}')" || { echo "❌ PyTorch missing"; exit 1; }
python -c "import torchvision; print(f'  ✓ torchvision {torchvision.__version__}')" || { echo "❌ torchvision missing"; exit 1; }
python -c "import numpy; print('  ✓ numpy')" || { echo "❌ numpy missing"; exit 1; }
python -c "import matplotlib; print('  ✓ matplotlib')" || echo "  ⚠️  matplotlib missing (visualizations will be skipped)"
python -c "import seaborn; print('  ✓ seaborn')" || echo "  ⚠️  seaborn missing (visualizations will be skipped)"
python -c "import sklearn; print('  ✓ sklearn')" || echo "  ⚠️  sklearn missing (some metrics unavailable)"

# Check video backends
echo ""
echo "Available video backends:"
python -c "import decord; print('  ✓ decord')" 2>/dev/null || echo "  ⚠️  decord unavailable"
python -c "from torchvision.io import read_video; print('  ✓ torchvision.io')" 2>/dev/null || echo "  ⚠️  torchvision.io unavailable"
python -c "import cv2; print('  ✓ opencv')" 2>/dev/null || echo "  ⚠️  opencv unavailable"

echo "=============================================="

# ==============================================================================
# Start Training
# ==============================================================================

echo ""
echo "🚀 Starting R3D-18 training..."
echo "📊 Monitor at: https://wandb.ai (if enabled)"
echo "   TensorBoard: tensorboard --logdir=$TENSORBOARD_DIR"
echo ""
echo "Configuration:"
echo "  Model: R3D-18 (3D ResNet-18)"
echo "  Epochs: 60 (no early stopping)"
echo "  Batch size: 8 (3D models are memory-intensive)"
echo "  Learning rate: 1e-4"
echo "  Loss: ASL + Focal + BCE"
echo "  EMA: Enabled"
echo "  Scheduler: Cosine Annealing"
echo ""
echo "Expected Results:"
echo "  Test F1:     ~84% (3D CNN baseline)"
echo "  Comparison:  vs BiLSTM ~87%, vs I3D ~85%, vs GRU ~86%"
echo "  Purpose:     Compare R3D vs I3D (both 3D CNNs)"
echo ""
echo "Visualizations (4 charts with 4 decimal precision):"
echo "  1. Per-class bug type metrics"
echo "  2. Per-count class metrics"
echo "  3. Confusion matrix (count)"
echo "  4. Top 20 bug combinations"
echo "=============================================="
echo ""

# Run training
python "$SCRIPT" \
  --data-root "$DATA_ROOT" \
  --manifests "$MANIFEST" \
  --use-cosine \
  --warmup-epochs 1 \
  --use-ema \
  --ema-decay 0.999 \
  --batch-size 8 \
  --num-workers 8 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
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
    echo "   Logs: $LOGS_DIR/training_${SLURM_JOB_ID}.out"
    echo ""
    
    # Find best model
    BEST_CKPT=$(ls -t "$CHECKPOINT_DIR"/best_epoch_*.pt 2>/dev/null | head -1)
    if [ -n "$BEST_CKPT" ]; then
        echo "🏆 Best Model:"
        echo "   Path: $BEST_CKPT"
        CKPT_SIZE=$(du -h "$BEST_CKPT" | cut -f1)
        echo "   Size: $CKPT_SIZE"
        
        # Extract epoch and F1 from filename
        if [[ "$BEST_CKPT" =~ epoch_([0-9]+)_f1_([0-9.]+) ]]; then
            BEST_EPOCH="${BASH_REMATCH[1]}"
            BEST_F1="${BASH_REMATCH[2]}"
            echo "   Epoch: $BEST_EPOCH"
            echo "   Val F1: $BEST_F1"
        fi
    fi
    echo ""
    
    # Check for test results
    TEST_RESULTS="$CHECKPOINT_DIR/test_results.json"
    if [ -f "$TEST_RESULTS" ]; then
        echo "📈 Test Results:"
        echo "   File: $TEST_RESULTS"
        
        # Extract test F1 if possible
        if command -v jq &> /dev/null; then
            TEST_F1=$(jq -r '.test_f1_types' "$TEST_RESULTS" 2>/dev/null)
            if [ -n "$TEST_F1" ] && [ "$TEST_F1" != "null" ]; then
                echo "   Test F1: $TEST_F1"
            fi
        fi
    fi
    echo ""
    
    # Check for visualizations
    VIZ_DIR="$CHECKPOINT_DIR/visualizations"
    if [ -d "$VIZ_DIR" ]; then
        VIZ_COUNT=$(ls "$VIZ_DIR"/*.png 2>/dev/null | wc -l)
        if [ "$VIZ_COUNT" -gt 0 ]; then
            echo "📊 Visualizations:"
            echo "   Location: $VIZ_DIR"
            echo "   Charts: $VIZ_COUNT PNG files"
            echo "   Files:"
            ls -1 "$VIZ_DIR"/*.png 2>/dev/null | while read -r file; do
                echo "     - $(basename "$file")"
            done
        fi
    fi
    echo ""
    
    echo "📥 Download Commands:"
    echo "   # Best model"
    echo "   scp nahian26@nibi:$BEST_CKPT ./"
    echo ""
    echo "   # Test results"
    echo "   scp nahian26@nibi:$TEST_RESULTS ./"
    echo ""
    echo "   # All visualizations"
    echo "   scp -r nahian26@nibi:$VIZ_DIR ./"
    echo ""
    
    echo "📈 View Results:"
    echo "   W&B: https://wandb.ai/$USER/$WANDB_PROJECT"
    echo "   TensorBoard: tensorboard --logdir=$TENSORBOARD_DIR"
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
        echo "   Checkpoint: $LATEST"
        echo ""
    fi
    
    echo "🔄 To continue:"
    echo "   cd $BASE_DIR"
    echo "   sbatch run_r3d18_training.sh"
    echo ""
    echo "   (Script will auto-resume from latest checkpoint)"
    echo ""
    
else
    echo ""
    echo "❌ Training failed with exit code: $EXIT_CODE"
    echo ""
    echo "🔍 Check logs:"
    echo "   Output: $LOGS_DIR/training_${SLURM_JOB_ID}.out"
    echo "   Errors: $LOGS_DIR/training_${SLURM_JOB_ID}.err"
    echo ""
    echo "💡 Common issues:"
    echo "   - Out of memory → Reduce batch size (currently 8)"
    echo "   - Video loading fails → Check decord/opencv/torchvision"
    echo "   - Package missing → Run setup commands in guide"
    echo ""
fi

echo "=============================================="
echo "📊 Job Statistics:"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""
echo "Run 'seff $SLURM_JOB_ID' for detailed efficiency report"
echo "=============================================="

echo ""
echo "📧 Email sent to: nahian.rifaat@ontariotechu.net"
echo ""
echo "=============================================="
echo "🎉 Script completed!"
echo "=============================================="
