#!/bin/bash
#SBATCH --account=def-loutfouz_gpu
#SBATCH --partition=gpubase_bygpu_b4
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --job-name=m3_sweep_pp
#SBATCH --output=/home/nahian26/scratch/logs/model3_sweep_%j.out
#SBATCH --error=/home/nahian26/scratch/logs/model3_sweep_%j.err
#SBATCH --mail-user=nahian.rifaat@ontariotechu.net
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

# ==============================================================================
# Model 3: post-processing sweep (NO retraining)
# Server: fir / nibi
#
# Run this AFTER run_fir_clean.sh (the frame-alignment-fixed retrain) has
# produced checkpoints/model3_pyramid_transformer_clean/best_composite.pt —
# not against the original run, which was trained on frame-cache-drifted
# data (see run_fir_clean.sh's header for details).
#
# Re-scores the existing best_composite.pt checkpoint against the val split
# with different min-conf / NMS score-threshold settings, to find the
# combination that cuts over-prediction down toward ~1-3x without
# collapsing recall too far. Ends with
# one official test-set eval using the winning combo.
# ==============================================================================

echo "=============================================="
echo "Model 3: Post-processing sweep (eval-only)"
echo "=============================================="
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Start      : $(date)"
echo "=============================================="

# ==============================================================================
# Paths — must match run_fir.sh (same completed training run)
# ==============================================================================

REPO_DIR="/home/nahian26/scratch/multilabel-perceptual-bug-detection"
SCRIPT_DIR="$REPO_DIR/training/model3_pyramid_transformer"
VIDEO_ROOT="/home/nahian26/scratch"
DATASET="/home/nahian26/scratch/temporal_bug_dataset.json"
CHECKPOINT_DIR="/home/nahian26/scratch/checkpoints/model3_pyramid_transformer_clean"
FRAME_CACHE_DIR="/home/nahian26/scratch/frame_cache_fps8_224"
SWEEP_LOGDIR="/home/nahian26/scratch/runs/model3_pyramid_transformer_clean/sweep_postprocess"
CKPT="$CHECKPOINT_DIR/best_composite.pt"

echo ""
echo "Paths:"
echo "  Script dir  : $SCRIPT_DIR"
echo "  Checkpoint  : $CKPT"
echo "  Sweep logs  : $SWEEP_LOGDIR"
echo "=============================================="

# ==============================================================================
# Environment
# ==============================================================================

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

mkdir -p "$SWEEP_LOGDIR"

if [ ! -f "$CKPT" ]; then
    echo "❌ Checkpoint not found: $CKPT"
    echo "   (run_fir.sh must have completed a full training run first)"
    exit 1
fi
echo "  ✓ Checkpoint found: $CKPT"
echo "=============================================="

# ==============================================================================
# Sweep grid — min-conf raised above the current hardcoded 0.1 (over-firing),
# nms-score-thr raised above the current hardcoded 0.05 (weak suppression of
# near-duplicate detections). min-duration/nms-sigma left at defaults for
# this first pass.
# ==============================================================================

MIN_CONF_VALUES="0.2 0.3 0.4"
NMS_THR_VALUES="0.15 0.25"

echo ""
echo "Sweeping min-conf x nms-score-thr on val split..."
for MC in $MIN_CONF_VALUES; do
    for THR in $NMS_THR_VALUES; do
        TAG="mc${MC}_thr${THR}"
        echo ""
        echo "── min-conf=$MC  nms-score-thr=$THR ──"
        python "$SCRIPT_DIR/train.py" \
            --mode           eval \
            --eval-split     val \
            --checkpoint     "$CKPT" \
            --video-root     "$VIDEO_ROOT" \
            --temporal-dataset "$DATASET" \
            --frame-cache-dir "$FRAME_CACHE_DIR" \
            --logdir         "$SWEEP_LOGDIR" \
            --eval-tag       "$TAG" \
            --min-conf       "$MC" \
            --nms-score-thr  "$THR" \
            --batch-size     4 \
            --num-workers    8 \
            --device         cuda \
            --backbone       swin_small_patch4_window7_224 \
            --window-sec     4.0 \
            --stride-sec     1.0 \
            --fps            8.0 \
            --img-size       224
    done
done

# ==============================================================================
# Pick the winning combo (highest f1@0.5) and print a summary table
# ==============================================================================

echo ""
echo "=============================================="
echo "Sweep results (sorted by f1@0.5):"
echo "=============================================="

BEST_COMBO=$(python3 - "$SWEEP_LOGDIR" <<'EOF'
import json, sys, glob, os

sweep_dir = sys.argv[1]
rows = []
for path in glob.glob(os.path.join(sweep_dir, "eval_val_mc*_thr*.json")):
    with open(path) as f:
        d = json.load(f)
    p = d.get("_params", {})
    rows.append({
        "min_conf": p.get("min_conf"),
        "nms_score_thr": p.get("nms_score_thr"),
        "f1@0.5": d.get("f1@0.5", 0.0),
        "mAP@0.5": d.get("mAP@0.5", 0.0),
        "over_pred_ratio": d.get("segment_over_pred_ratio", 0.0),
    })

rows.sort(key=lambda r: r["f1@0.5"], reverse=True)
print(f"{'min_conf':>10} {'nms_thr':>10} {'f1@0.5':>10} {'mAP@0.5':>10} {'over_pred':>10}",
      file=sys.stderr)
for r in rows:
    print(f"{r['min_conf']:>10} {r['nms_score_thr']:>10} {r['f1@0.5']:>10.4f} "
          f"{r['mAP@0.5']:>10.4f} {r['over_pred_ratio']:>10.2f}", file=sys.stderr)

if rows:
    best = rows[0]
    print(f"{best['min_conf']} {best['nms_score_thr']}")
EOF
)

echo "$BEST_COMBO" | tail -n +1
read -r BEST_MC BEST_THR <<< "$(echo "$BEST_COMBO" | tail -1)"

if [ -z "$BEST_MC" ]; then
    echo "❌ No sweep results found — cannot pick a winning combo."
    exit 1
fi

echo ""
echo "🏆 Winning combo: min-conf=$BEST_MC  nms-score-thr=$BEST_THR"
echo "=============================================="

# ==============================================================================
# Final official test-set eval with the winning combo
# ==============================================================================

echo ""
echo "Running final test-set eval with winning combo..."
python "$SCRIPT_DIR/train.py" \
    --mode           eval \
    --eval-split     test \
    --checkpoint     "$CKPT" \
    --video-root     "$VIDEO_ROOT" \
    --temporal-dataset "$DATASET" \
    --frame-cache-dir "$FRAME_CACHE_DIR" \
    --logdir         "$SWEEP_LOGDIR" \
    --eval-tag       "winner" \
    --min-conf       "$BEST_MC" \
    --nms-score-thr  "$BEST_THR" \
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
    echo "✅ Sweep complete."
    echo "📊 Per-combo results : $SWEEP_LOGDIR/eval_val_mc*_thr*.json"
    echo "📊 Winning test eval : $SWEEP_LOGDIR/eval_test_winner.json"
else
    echo "❌ Final test eval failed (exit $EXIT_CODE) — per-combo val results above are still valid."
fi
echo "=============================================="
