#!/bin/bash
#SBATCH --account=def-loutfouz_gpu
#SBATCH --partition=gpubase_bygpu_b4
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=resnet18_ablation
#SBATCH --output=/home/nahian26/projects/def-loutfouz/nahian26/bugdatasetbig/logs_ablation/ablation_%j.out
#SBATCH --error=/home/nahian26/projects/def-loutfouz/nahian26/bugdatasetbig/logs_ablation/ablation_%j.err
#SBATCH --mail-user=nahian.rifaat@ontariotechu.net
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

# ==============================================================================
# RESNET18-ONLY ABLATION TRAINING FOR NARVAL (A100)
# ==============================================================================
# Purpose: Prove temporal modeling is essential
# Model: ResNet18 → Average Pooling → Heads (NO BiLSTM)
# Expected F1: ~80% (vs ~87% with BiLSTM)
# ==============================================================================

echo "================================================================================"
echo "RESNET18-ONLY ABLATION TRAINING"
echo "================================================================================"
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "Start Time:  $(date)"
echo "User:        nahian26"
echo "Account:     def-loutfouz_gpu"
echo "GPU:         A100"
echo "================================================================================"

# ==============================================================================
# PATHS
# ==============================================================================

BASE_DIR="/home/nahian26/projects/def-loutfouz/nahian26/bugdatasetbig"
DATA_ROOT="$BASE_DIR"
MANIFEST="$BASE_DIR/my_new_splits.jsonl"
SCRIPT="$BASE_DIR/scripts_ablation/train_resnet18_ablation_complete.py"
CHECKPOINT_DIR="/home/nahian26/scratch/checkpoints/resnet18_ablation"
TENSORBOARD_DIR="/home/nahian26/scratch/runs/resnet18_ablation"

echo ""
echo "Paths:"
echo "  Base:        $BASE_DIR"
echo "  Data:        $DATA_ROOT"
echo "  Manifest:    $MANIFEST"
echo "  Script:      $SCRIPT"
echo "  Checkpoints: $CHECKPOINT_DIR"
echo "  TensorBoard: $TENSORBOARD_DIR"
echo "================================================================================"

# ==============================================================================
# ENVIRONMENT SETUP - CRITICAL SECTION
# ==============================================================================

echo ""
echo "================================================================================"
echo "ENVIRONMENT SETUP"
echo "================================================================================"

# Step 1: Load Python module
echo ""
echo "[1/5] Loading Python 3.10..."
module load python/3.10
if [ $? -eq 0 ]; then
    echo "  ✓ Python 3.10 loaded"
    echo "    Path: $(which python)"
    echo "    Version: $(python --version)"
else
    echo "  ✗ ERROR: Failed to load Python 3.10"
    exit 1
fi

# Step 2: Load GCC and OpenCV modules
echo ""
echo "[2/5] Loading GCC 12.3 and OpenCV 4.8.1..."
module load gcc opencv/4.8.1
if [ $? -eq 0 ]; then
    echo "  ✓ GCC and OpenCV modules loaded"
else
    echo "  ✗ ERROR: Failed to load GCC/OpenCV"
    exit 1
fi

# Step 3: Activate virtual environment
echo ""
echo "[3/5] Activating virtual environment..."
ENV_PATH="$HOME/env_bugdetection_ablation"

if [ ! -d "$ENV_PATH" ]; then
    echo "  ✗ ERROR: Virtual environment not found: $ENV_PATH"
    echo "  → Run setup commands from ABLATION_COMPLETE_GUIDE.md first!"
    exit 1
fi

source "$ENV_PATH/bin/activate"

if [ -n "$VIRTUAL_ENV" ]; then
    echo "  ✓ Virtual environment activated"
    echo "    Environment: $VIRTUAL_ENV"
    echo "    Python: $(which python)"
    echo "    Pip: $(which pip)"
else
    echo "  ✗ ERROR: Virtual environment activation failed"
    exit 1
fi

# Step 4: Add OpenCV to Python path (CRITICAL!)
echo ""
echo "[4/5] Configuring OpenCV Python path..."
if [ -n "$EBROOTOPENCV" ]; then
    export PYTHONPATH="$EBROOTOPENCV/lib/python3.10/site-packages:$PYTHONPATH"
    echo "  ✓ OpenCV path added: $EBROOTOPENCV/lib/python3.10/site-packages"
else
    echo "  ⚠ WARNING: EBROOTOPENCV not set, OpenCV may not work"
fi

# Step 5: Verify ALL required packages
echo ""
echo "[5/5] Verifying required packages..."

PACKAGES_OK=true

# Check OpenCV
echo -n "  Testing OpenCV... "
python -c "import cv2; print(f'v{cv2.__version__}')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓"
else
    echo "✗ FAILED"
    echo "    Installing OpenCV Python bindings..."
    pip install opencv-python-headless --break-system-packages
    PACKAGES_OK=false
fi

# Check PyTorch
echo -n "  Testing PyTorch... "
python -c "import torch; print(f'v{torch.__version__}')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓"
else
    echo "✗ FAILED - PyTorch missing!"
    PACKAGES_OK=false
fi

# Check W&B
echo -n "  Testing wandb... "
python -c "import wandb; print(f'v{wandb.__version__}')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓"
else
    echo "✗ FAILED"
    echo "    Installing wandb..."
    pip install wandb --break-system-packages
fi

# Check Matplotlib
echo -n "  Testing matplotlib... "
python -c "import matplotlib; print(f'v{matplotlib.__version__}')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓"
else
    echo "✗ FAILED"
    echo "    Installing matplotlib..."
    pip install matplotlib --break-system-packages
    PACKAGES_OK=false
fi

# Check Seaborn
echo -n "  Testing seaborn... "
python -c "import seaborn; print(f'v{seaborn.__version__}')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓"
else
    echo "✗ FAILED"
    echo "    Installing seaborn..."
    pip install seaborn --break-system-packages
    PACKAGES_OK=false
fi

# Check Scikit-learn
echo -n "  Testing scikit-learn... "
python -c "import sklearn; print(f'v{sklearn.__version__}')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓"
else
    echo "✗ FAILED"
    echo "    Installing scikit-learn..."
    pip install scikit-learn --break-system-packages
    PACKAGES_OK=false
fi

# Check TensorBoard
echo -n "  Testing tensorboard... "
python -c "import tensorboard; print('OK')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓"
else
    echo "✗ FAILED"
    echo "    Installing tensorboard..."
    pip install tensorboard --break-system-packages
fi

if [ "$PACKAGES_OK" = false ]; then
    echo ""
    echo "⚠️  Some packages were missing and have been installed."
    echo "   Re-testing all packages..."
    echo ""
    
    # Re-test critical packages
    python -c "import cv2, torch, wandb, matplotlib, seaborn, sklearn, tensorboard" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "✗ ERROR: Package installation failed!"
        echo "→ Please run setup commands manually from ABLATION_COMPLETE_GUIDE.md"
        exit 1
    fi
    echo "✓ All packages now working!"
fi

echo ""
echo "================================================================================"
echo "✓ ENVIRONMENT READY"
echo "================================================================================"

# ==============================================================================
# W&B CONFIGURATION
# ==============================================================================

echo ""
echo "Configuring W&B..."
export WANDB_API_KEY="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"
export WANDB_PROJECT="resnet18-ablation"
echo "  ✓ W&B configured"
echo "    Project: $WANDB_PROJECT"

# ==============================================================================
# GPU INFORMATION
# ==============================================================================

echo ""
echo "================================================================================"
echo "GPU INFORMATION"
echo "================================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "================================================================================"

# ==============================================================================
# CREATE DIRECTORIES
# ==============================================================================

echo ""
echo "Creating output directories..."
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$TENSORBOARD_DIR"
mkdir -p "$BASE_DIR/logs_ablation"
mkdir -p "$CHECKPOINT_DIR/visualizations"
echo "  ✓ Directories created"

# ==============================================================================
# CHECK FOR RESUME
# ==============================================================================

RESUME_FLAG=""
LATEST_CHECKPOINT=$(ls -t "$CHECKPOINT_DIR"/checkpoint_epoch_*.pt 2>/dev/null | head -1)

if [ -n "$LATEST_CHECKPOINT" ]; then
    echo ""
    echo "================================================================================"
    echo "RESUME DETECTED"
    echo "================================================================================"
    echo "  Checkpoint: $LATEST_CHECKPOINT"
    EPOCH_NUM=$(echo "$LATEST_CHECKPOINT" | grep -oP 'epoch_\K\d+')
    echo "  Completed epoch: $EPOCH_NUM"
    echo "  Will auto-resume training"
    RESUME_FLAG="--resume $LATEST_CHECKPOINT"
    echo "================================================================================"
else
    echo ""
    echo "Starting fresh training (no checkpoint found)"
fi

# ==============================================================================
# VERIFY FILES
# ==============================================================================

echo ""
echo "================================================================================"
echo "VERIFYING FILES"
echo "================================================================================"

ERROR=0

if [ ! -f "$SCRIPT" ]; then
    echo "✗ ERROR: Python script not found: $SCRIPT"
    ERROR=1
else
    echo "✓ Python script: $SCRIPT"
    SCRIPT_SIZE=$(du -h "$SCRIPT" | cut -f1)
    echo "  Size: $SCRIPT_SIZE"
fi

if [ ! -f "$MANIFEST" ]; then
    echo "✗ ERROR: Manifest not found: $MANIFEST"
    ERROR=1
else
    echo "✓ Manifest: $MANIFEST"
    MANIFEST_LINES=$(wc -l < "$MANIFEST")
    echo "  Lines: $MANIFEST_LINES"
fi

if [ ! -d "$DATA_ROOT/clips" ]; then
    echo "✗ ERROR: Clips directory not found: $DATA_ROOT/clips"
    ERROR=1
else
    CLIP_COUNT=$(ls "$DATA_ROOT/clips" 2>/dev/null | wc -l)
    echo "✓ Clips directory: $DATA_ROOT/clips"
    echo "  Files: $CLIP_COUNT"
fi

if [ $ERROR -eq 1 ]; then
    echo ""
    echo "✗ ERROR: Required files missing!"
    echo "→ Upload files from ABLATION_COMPLETE_GUIDE.md"
    exit 1
fi

echo "================================================================================"
echo "✓ ALL FILES VERIFIED"
echo "================================================================================"

# ==============================================================================
# START TRAINING
# ==============================================================================

echo ""
echo "================================================================================"
echo "STARTING TRAINING"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Model:       ResNet18-only (NO BiLSTM)"
echo "  Epochs:      60"
echo "  Batch size:  16 (lighter model)"
echo "  Learning rate: 7e-4"
echo "  Optimizer:   AdamW"
echo "  Scheduler:   Cosine"
echo "  Loss:        ASL + Focal"
echo ""
echo "Expected Results:"
echo "  Test F1:     ~80% (vs ~87% with BiLSTM)"
echo "  Proves:      Temporal modeling is essential"
echo ""
echo "Logging:"
echo "  W&B:         https://wandb.ai"
echo "  TensorBoard: $TENSORBOARD_DIR"
echo ""
echo "================================================================================"
echo ""

# Run training
python "$SCRIPT" \
  --data-root "$DATA_ROOT" \
  --manifests "$MANIFEST" \
  --batch-size 32 \
  --num-workers 8 \
  --lr 7e-4 \
  --weight-decay 3e-4 \
  --epochs 60 \
  --save-dir "$CHECKPOINT_DIR" \
  --logdir "$TENSORBOARD_DIR" \
  --wandb-project "$WANDB_PROJECT" \
  --use-cosine \
  $RESUME_FLAG

EXIT_CODE=$?

# ==============================================================================
# POST-TRAINING SUMMARY
# ==============================================================================

echo ""
echo "================================================================================"
echo "TRAINING COMPLETED"
echo "================================================================================"
echo "Exit code:   $EXIT_CODE"
echo "End time:    $(date)"
echo "Duration:    $((SECONDS/60)) minutes (~$((SECONDS/3600)) hours)"
echo "================================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ TRAINING SUCCESSFUL!"
    echo ""
    echo "📊 Results Location:"
    echo "  Checkpoints: $CHECKPOINT_DIR"
    echo "  TensorBoard: $TENSORBOARD_DIR"
    echo "  Logs:        $BASE_DIR/logs_ablation/ablation_${SLURM_JOB_ID}.out"
    echo ""
    
    BEST_CKPT=$(ls -t "$CHECKPOINT_DIR"/*best*.pt 2>/dev/null | head -1)
    if [ -n "$BEST_CKPT" ]; then
        echo "🏆 Best Model:"
        echo "  Path: $BEST_CKPT"
        CKPT_SIZE=$(du -h "$BEST_CKPT" | cut -f1)
        echo "  Size: $CKPT_SIZE"
        echo ""
    fi
    
    RESULTS_JSON="$CHECKPOINT_DIR/comprehensive_test_results.json"
    if [ -f "$RESULTS_JSON" ]; then
        echo "📈 Test Results:"
        echo "  JSON: $RESULTS_JSON"
        echo ""
        echo "  Quick view:"
        python -c "import json; d=json.load(open('$RESULTS_JSON')); print(f\"    Test F1: {d['overall_metrics']['f1']:.4f}\")" 2>/dev/null || echo "    (view file for metrics)"
        echo ""
    fi
    
    VIZ_DIR="$CHECKPOINT_DIR/visualizations"
    if [ -d "$VIZ_DIR" ]; then
        VIZ_COUNT=$(ls "$VIZ_DIR"/*.png 2>/dev/null | wc -l)
        if [ $VIZ_COUNT -gt 0 ]; then
            echo "📊 Visualizations ($VIZ_COUNT charts):"
            echo "  Directory: $VIZ_DIR"
            ls "$VIZ_DIR"/*.png 2>/dev/null | while read file; do
                echo "    ✓ $(basename $file)"
            done
            echo ""
        fi
    fi
    
    echo "📥 Download Commands:"
    echo "  # All results"
    echo "  scp -r nahian26@narval:$CHECKPOINT_DIR ./resnet18_ablation_results"
    echo ""
    echo "  # Just JSON"
    echo "  scp nahian26@narval:$RESULTS_JSON ./"
    echo ""
    echo "  # Just visualizations"
    echo "  scp nahian26@narval:$VIZ_DIR/*.png ./"
    echo ""
    
    echo "🌐 View in W&B:"
    echo "  https://wandb.ai/$USER/$WANDB_PROJECT"
    echo ""
    
elif [ $EXIT_CODE -eq 124 ] || [ $EXIT_CODE -eq 140 ]; then
    echo ""
    echo "⏱️  JOB REACHED TIME LIMIT"
    echo ""
    
    LATEST=$(ls -t "$CHECKPOINT_DIR"/checkpoint_epoch_*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        EPOCH_NUM=$(echo "$LATEST" | grep -oP 'epoch_\K\d+')
        REMAINING=$((60 - EPOCH_NUM))
        
        echo "📊 Progress:"
        echo "  Completed: Epoch $EPOCH_NUM/60"
        echo "  Remaining: ~$REMAINING epochs"
        echo ""
    fi
    
    echo "🔄 To continue:"
    echo "  cd $BASE_DIR"
    echo "  sbatch run_ablation_narval.sh"
    echo ""
    echo "  (Script will auto-resume from epoch $EPOCH_NUM)"
    echo ""
    
else
    echo ""
    echo "✗ TRAINING FAILED"
    echo ""
    echo "🔍 Check logs:"
    echo "  Output: $BASE_DIR/logs_ablation/ablation_${SLURM_JOB_ID}.out"
    echo "  Error:  $BASE_DIR/logs_ablation/ablation_${SLURM_JOB_ID}.err"
    echo ""
    echo "  tail -100 $BASE_DIR/logs_ablation/ablation_${SLURM_JOB_ID}.err"
    echo ""
fi

echo "================================================================================"
echo "📊 JOB STATISTICS"
echo "================================================================================"
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_JOB_NODELIST"
echo "Exit code:   $EXIT_CODE"
echo ""
echo "For efficiency report, run:"
echo "  seff $SLURM_JOB_ID"
echo "================================================================================"

echo ""
echo "📧 Email sent to: nahian.rifaat@ontariotechu.net"
echo ""
echo "================================================================================"
echo "✓ SCRIPT COMPLETED"
echo "================================================================================"
