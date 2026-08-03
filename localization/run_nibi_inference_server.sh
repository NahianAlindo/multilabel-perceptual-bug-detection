#!/bin/bash
#SBATCH --account=def-loutfouz_gpu
#SBATCH --partition=gpubase_bygpu_b4
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --job-name=nibi_infer_server
#SBATCH --output=/home/nahian26/scratch/logs/nibi_infer_server_%j.out
#SBATCH --error=/home/nahian26/scratch/logs/nibi_infer_server_%j.err
#SBATCH --mail-user=nahian.rifaat@ontariotechu.net
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

# ==============================================================================
# Real nibi inference server — not the networking_test/ toy, the actual
# model-serving backend. Runs localization/nibi_inference_server.py in the
# foreground for the job's whole time limit, tunneled via ngrok (proven
# reachable from outside the cluster by networking_test/ already).
#
# 3h walltime and 48G/8cpu match the training scripts' resourcing (this job
# runs the same model, just inference instead of training) — enough for a
# real test session covering multiple 5-15 minute videos, not just one
# request. Decoding a 15-minute video at 8fps/224x224 is a few GB in memory
# at once (not the whole 66GB frame-cache scale — that's the full 263-video
# training cache; this is one video at a time), 48G is generous headroom.
# ==============================================================================

echo "=============================================="
echo "nibi Inference Server (real model, not the reachability test)"
echo "=============================================="
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Date/time  : $(date)"
echo "=============================================="

REPO_DIR="/home/nahian26/scratch/multilabel-perceptual-bug-detection"
SCRIPT_DIR="$REPO_DIR/localization"
CHECKPOINT="/home/nahian26/scratch/checkpoints/model3_pyramid_transformer_clean/best_f1.pt"

# ==============================================================================
# ⚠️  REAL SECRET — DO NOT COMMIT THIS LINE ONCE FILLED IN.
# Same pattern as networking_test/run_flask_test.sh — fill in directly on
# nibi, never git add/commit/push this line. See that script's own comment
# block for the full rationale; not repeated here.
# ==============================================================================
export NGROK_AUTH_TOKEN="${NGROK_AUTH_TOKEN:-PASTE_YOUR_NGROK_TOKEN_HERE}"

echo ""
echo "Paths:"
echo "  Script dir  : $SCRIPT_DIR"
echo "  Checkpoint  : $CHECKPOINT"
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
# Same full model3 dependency set as the training scripts (train.py needs
# all of these importable even for --mode infer) plus the server-specific
# ones the training scripts don't need.
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
ensure_pkg fastapi fastapi
ensure_pkg uvicorn "uvicorn[standard]"
ensure_pkg multipart python-multipart
ensure_pkg pyngrok pyngrok
python -c "import cv2; print(f'  ✓ OpenCV {cv2.__version__}')" || echo "  ⚠ OpenCV not found (will fallback to decord)"
python -c "import timm" 2>/dev/null || { echo "  ❌ timm still missing (required for Swin backbone)"; exit 1; }

if [ "$NGROK_AUTH_TOKEN" = "PASTE_YOUR_NGROK_TOKEN_HERE" ]; then
    echo ""
    echo "  ❌ NGROK_AUTH_TOKEN still has its placeholder value — edit this script"
    echo "     on nibi and paste your real token in before submitting, or pass one"
    echo "     at submit time: sbatch --export=ALL,NGROK_AUTH_TOKEN=<token> $0"
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

echo ""
echo "Verifying files..."
[ ! -f "$SCRIPT_DIR/nibi_inference_server.py" ] && \
    echo "❌ nibi_inference_server.py not found: $SCRIPT_DIR/nibi_inference_server.py" && exit 1
echo "  ✓ nibi_inference_server.py found"
[ ! -f "$CHECKPOINT" ] && echo "❌ Checkpoint not found: $CHECKPOINT" && exit 1
echo "  ✓ Checkpoint found"
[ ! -f "$REPO_DIR/training/model3_pyramid_transformer/train.py" ] && \
    echo "❌ train.py not found" && exit 1
echo "  ✓ train.py found"
echo "=============================================="

echo ""
echo "=============================================="
echo "READY — nibi inference server about to start."
echo "  Compute node hostname : $SLURMD_NODENAME"
echo "  Checkpoint             : best_f1.pt (min-conf=0.3, nms-score-thr=0.25)"
echo ""
echo "  Public ngrok URL prints below once the server opens the tunnel"
echo "  (look for a line starting with 'ngrok tunnel :')."
echo "  Paste that URL into the local frontend's settings modal."
echo "=============================================="
echo ""

# Runs in the foreground on purpose — this job IS the server for its whole
# time limit. Cancel with `scancel $SLURM_JOB_ID` from the login node when
# you're done testing.
python "$SCRIPT_DIR/nibi_inference_server.py" \
    --checkpoint    "$CHECKPOINT" \
    --auth-token    "$NGROK_AUTH_TOKEN" \
    --min-conf      0.3 \
    --nms-score-thr 0.25

echo ""
echo "=============================================="
echo "Server exited (or job hit its time limit)."
echo "Job finished: $(date)"
echo "=============================================="
