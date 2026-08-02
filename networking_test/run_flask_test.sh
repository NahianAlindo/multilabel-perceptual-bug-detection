#!/bin/bash
#SBATCH --account=def-loutfouz_gpu
#SBATCH --partition=gpubase_bygpu_b4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=flask_reach_test
#SBATCH --output=/home/nahian26/scratch/logs/flask_reach_test_%j.out
#SBATCH --error=/home/nahian26/scratch/logs/flask_reach_test_%j.err

# ==============================================================================
# Networking feasibility test — NOT a real inference server.
# Server: nibi (or fir)
#
# Answers one question: can a process inside a SLURM GPU allocation be
# reached over HTTP from outside the cluster? Runs a minimal Flask app
# (networking_test/app.py) in the foreground for the job's whole time limit
# so there's a window to actually test connectivity against it. GPU is
# requested and nvidia-smi is printed just to confirm the allocation itself
# is healthy — this test does not otherwise use the GPU.
#
# Short 15-minute walltime is deliberate: this is a quick reachability check,
# not a workload, and backfills into the queue faster than the multi-hour/
# multi-day jobs elsewhere in this repo.
# ==============================================================================

echo "=============================================="
echo "Flask reachability test"
echo "=============================================="
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Date/time  : $(date)"
echo "=============================================="

REPO_DIR="/home/nahian26/scratch/multilabel-perceptual-bug-detection"
SCRIPT_DIR="$REPO_DIR/networking_test"

echo ""
echo "Loading modules..."
module load python/3.10

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
echo "Checking Flask (installing if missing)..."
python -c "import flask; print(f'  ✓ Flask {flask.__version__}')" 2>/dev/null || {
    echo "  ⚠ Flask missing — installing..."
    pip install --no-index flask 2>/dev/null || pip install flask
}

export OMP_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo ""
echo "GPU Information (allocation sanity check only — this test doesn't use the GPU):"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=============================================="

echo ""
echo "Verifying app.py..."
[ ! -f "$SCRIPT_DIR/app.py" ] && echo "❌ app.py not found: $SCRIPT_DIR/app.py" && exit 1
echo "  ✓ app.py found"
echo "=============================================="

echo ""
echo "=============================================="
echo "READY — Flask server about to start."
echo "  Compute node hostname : $SLURMD_NODENAME"
echo "  Listening on          : 0.0.0.0:5000"
echo ""
echo "  From the nibi LOGIN node, once this is running:"
echo "    curl http://$SLURMD_NODENAME:5000/health"
echo ""
echo "  From your own machine (see the chat response for the full explanation):"
echo "    ssh -L 5000:$SLURMD_NODENAME:5000 nahian26@nibi.alliancecan.ca"
echo "    curl http://localhost:5000/health"
echo "=============================================="
echo ""

# Runs in the foreground on purpose — this job IS the server for its whole
# time limit (15 min). Ctrl-C locally won't stop it; cancel with
# `scancel $SLURM_JOB_ID` from the login node when you're done testing.
python "$SCRIPT_DIR/app.py"

echo ""
echo "=============================================="
echo "Flask server exited (or job hit its time limit)."
echo "Job finished: $(date)"
echo "=============================================="
