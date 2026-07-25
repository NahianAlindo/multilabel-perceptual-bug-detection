#!/bin/bash
#SBATCH --account=def-loutfouz_gpu
#SBATCH --partition=gpubase_bygpu_b4
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=m3_finetune_prec
#SBATCH --output=/home/nahian26/scratch/logs/model3_finetune_%j.out
#SBATCH --error=/home/nahian26/scratch/logs/model3_finetune_%j.err
#SBATCH --mail-user=nahian.rifaat@ontariotechu.net
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

# ==============================================================================
# Model 3: precision-focused fine-tune (resumes from best_composite.pt)
# Server: run on the SAME cluster that produced best_composite.pt + the frame
# cache (SHARCNET/Alliance /scratch is per-cluster, not shared across fir/
# nibi — submitting this on a cluster that never ran the base training will
# fail the checkpoint-existence check below).
#
# IMPORTANT — run sweep_postprocess.sh FIRST. This fine-tune's own per-epoch
# val eval and best-checkpoint selection use --min-conf/--nms-score-thr too
# (not just the loss-weight change) — otherwise checkpoint selection still
# uses the old permissive post-processing that rewards over-prediction, and
# any real improvement from the loss reweighting gets masked. Once the sweep
# finishes, pass its winning combo in:
#     FINETUNE_MIN_CONF=<winner> FINETUNE_NMS_THR=<winner> sbatch run_fir_finetune.sh
# Falls back to mid-grid defaults (0.3 / 0.15) if not set.
#
# Loss weights: lambda_cls/lambda_reg are read relative to whatever the base
# run's Optuna HPO actually picked (best_hparams.json), not assumed to be the
# 1.0/1.0 hardcoded default — HPO may have already moved them. This fine-tune
# multiplies lambda_cls x1.5 and lambda_reg x0.7 from that base.
#
# LR: best_composite.pt has no optimizer state, so AdamW restarts from
# scratch on resume. Using the base run's full LR would risk jolting an
# already-converged model, so this fine-tune uses 1/10th of the base LR.
#
# --hpo-trials 0: weights/LR are set explicitly here, HPO is skipped.
# Separate --save-dir/--logdir: never overwrites the original run's outputs.
# --epochs 200 (well above any plausible resume point): train.py now exits
# loudly if the resumed epoch is >= --epochs instead of silently training
# zero epochs — this is just headroom, early stopping (patience=15) still
# governs actual runtime.
# ==============================================================================

echo "=============================================="
echo "Model 3: Precision-focused fine-tune"
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
# Original completed run — read-only source of the resume checkpoint + HPO hparams
BASE_CHECKPOINT_DIR="/home/nahian26/scratch/checkpoints/model3_pyramid_transformer_clean"
BASE_CKPT="$BASE_CHECKPOINT_DIR/best_composite.pt"
BASE_LOGDIR="/home/nahian26/scratch/runs/model3_pyramid_transformer_clean"
BASE_HPARAMS="$BASE_LOGDIR/best_hparams.json"
# This fine-tune's own, separate output dirs
CHECKPOINT_DIR="/home/nahian26/scratch/checkpoints/model3_finetune_precision"
LOGDIR="/home/nahian26/scratch/runs/model3_finetune_precision"
FRAME_CACHE_DIR="/home/nahian26/scratch/frame_cache_fps8_224"

# Post-processing for this fine-tune's own eval/checkpoint-selection — pass
# the sweep's winning combo via env vars once sweep_postprocess.sh has run.
FINETUNE_MIN_CONF="${FINETUNE_MIN_CONF:-0.3}"
FINETUNE_NMS_THR="${FINETUNE_NMS_THR:-0.15}"

echo ""
echo "Paths:"
echo "  Script dir       : $SCRIPT_DIR"
echo "  Base checkpoint  : $BASE_CKPT"
echo "  Base hparams     : $BASE_HPARAMS"
echo "  New checkpoints  : $CHECKPOINT_DIR"
echo "  New logs         : $LOGDIR"
echo "  Post-proc (in)   : min-conf=$FINETUNE_MIN_CONF nms-score-thr=$FINETUNE_NMS_THR"
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
# Derive fine-tune hyperparameters relative to what the base run's HPO
# actually picked (not assumed to be 1.0/1.0/1e-4) — falls back to those
# hardcoded defaults if best_hparams.json is missing/unreadable.
# ==============================================================================

DERIVED=$(python3 - "$BASE_HPARAMS" <<'EOF'
import json, sys

path = sys.argv[1]
base_lr, base_cls, base_reg = 1e-4, 1.0, 1.0
try:
    with open(path) as f:
        h = json.load(f)
    base_lr  = float(h.get("lr", base_lr))
    base_cls = float(h.get("lambda_cls", base_cls))
    base_reg = float(h.get("lambda_reg", base_reg))
except Exception as e:
    print(f"[WARN] could not read {path}: {e}", file=sys.stderr)

new_lr  = max(base_lr * 0.1, 1e-6)
new_cls = min(base_cls * 1.5, 2.5)
new_reg = max(base_reg * 0.7, 0.3)
print(f"{new_lr} {new_cls} {new_reg} {base_lr} {base_cls} {base_reg}")
EOF
)
read -r NEW_LR NEW_LAMBDA_CLS NEW_LAMBDA_REG BASE_LR BASE_LAMBDA_CLS BASE_LAMBDA_REG <<< "$(echo "$DERIVED" | tail -1)"

if [ -z "$NEW_LR" ]; then
    echo "❌ Failed to derive fine-tune hyperparameters — aborting."
    exit 1
fi

echo ""
echo "Base HPO hyperparameters (from $BASE_HPARAMS):"
echo "  lr=$BASE_LR  lambda_cls=$BASE_LAMBDA_CLS  lambda_reg=$BASE_LAMBDA_REG"
echo "Derived fine-tune hyperparameters:"
echo "  lr=$NEW_LR  lambda_cls=$NEW_LAMBDA_CLS  lambda_reg=$NEW_LAMBDA_REG"
echo "=============================================="

# ==============================================================================
# Auto-resume within this fine-tune's own checkpoint dir (separate from the
# base run). First submission resumes from BASE_CKPT; subsequent resubmits
# (e.g. after a time-limit) resume from this fine-tune's own latest epoch.
# ==============================================================================

RESUME_CKPT="$BASE_CKPT"
LATEST_OWN_CHECKPOINT=$(ls -t "$CHECKPOINT_DIR"/checkpoint_epoch_*.pt 2>/dev/null | head -1)
if [ -n "$LATEST_OWN_CHECKPOINT" ]; then
    echo ""
    echo "🔄 Found existing fine-tune checkpoint: $LATEST_OWN_CHECKPOINT"
    RESUME_CKPT="$LATEST_OWN_CHECKPOINT"
else
    echo ""
    echo "Starting fine-tune fresh from base checkpoint: $BASE_CKPT"
fi

echo ""
echo "Verifying files..."
[ ! -f "$SCRIPT_DIR/train.py" ] && echo "❌ train.py not found: $SCRIPT_DIR/train.py" && exit 1
echo "  ✓ train.py"
[ ! -f "$RESUME_CKPT" ]         && echo "❌ Resume checkpoint not found: $RESUME_CKPT" && \
    echo "   (if this is a fresh cluster, best_composite.pt from the base run isn't here — " && \
    echo "    copy it + the frame cache over first, don't submit blind)" && exit 1
echo "  ✓ Resume checkpoint"
[ ! -f "$DATASET" ]             && echo "❌ Dataset not found: $DATASET" && exit 1
echo "  ✓ Dataset"
echo "=============================================="

echo ""
echo "Configuration:"
echo "  Mode        : train (resumed, HPO skipped, manually-derived weights)"
echo "  Base ckpt   : $BASE_CKPT"
echo "  lr          : $NEW_LR (base was $BASE_LR)"
echo "  lambda_cls  : $NEW_LAMBDA_CLS (base was $BASE_LAMBDA_CLS)"
echo "  lambda_reg  : $NEW_LAMBDA_REG (base was $BASE_LAMBDA_REG)"
echo "  Post-proc   : min-conf=$FINETUNE_MIN_CONF nms-score-thr=$FINETUNE_NMS_THR"
echo "  Max epochs  : 200 (headroom; continues from resumed epoch, early stop patience=15)"
echo "=============================================="
echo ""

python "$SCRIPT_DIR/train.py" \
    --mode           train \
    --video-root     "$VIDEO_ROOT" \
    --temporal-dataset "$DATASET" \
    --frame-cache-dir "$FRAME_CACHE_DIR" \
    --save-dir       "$CHECKPOINT_DIR" \
    --logdir         "$LOGDIR" \
    --checkpoint     "$RESUME_CKPT" \
    --hpo-trials     0 \
    --lr             "$NEW_LR" \
    --lambda-cls     "$NEW_LAMBDA_CLS" \
    --lambda-reg     "$NEW_LAMBDA_REG" \
    --min-conf       "$FINETUNE_MIN_CONF" \
    --nms-score-thr  "$FINETUNE_NMS_THR" \
    --epochs         200 \
    --batch-size     4 \
    --num-workers    8 \
    --device         cuda \
    --wandb-project  "$WANDB_PROJECT" \
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
echo "Duration    : $((SECONDS/60)) min (~$((SECONDS/3600))h)"
echo "=============================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Fine-tune + test complete!"
    BEST_CKPT=$(ls -t "$CHECKPOINT_DIR"/best_composite.pt 2>/dev/null | head -1)
    [ -n "$BEST_CKPT" ] && echo "🏆 Best checkpoint: $BEST_CKPT ($(du -h $BEST_CKPT | cut -f1))"
    echo ""
    echo "📊 Test metrics : $LOGDIR/test_metrics.json"
    echo "📈 W&B          : https://wandb.ai/nahian26/$WANDB_PROJECT"
elif [ $EXIT_CODE -eq 124 ] || [ $EXIT_CODE -eq 140 ]; then
    echo ""
    echo "⏱️  Job reached time limit"
    echo "📝 To resume: sbatch $SCRIPT_DIR/run_fir_finetune.sh"
    echo "   (own checkpoint dir auto-detected — will not re-resume from base ckpt)"
else
    echo ""
    echo "❌ Fine-tune failed (exit $EXIT_CODE)"
    echo "🔍 Logs:"
    echo "   /home/nahian26/scratch/logs/model3_finetune_${SLURM_JOB_ID}.out"
    echo "   /home/nahian26/scratch/logs/model3_finetune_${SLURM_JOB_ID}.err"
fi

echo ""
echo "=============================================="
echo "📊 Job statistics: run 'seff $SLURM_JOB_ID'"
echo "=============================================="
