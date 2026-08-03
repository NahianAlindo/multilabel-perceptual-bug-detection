#!/bin/bash
#SBATCH --account=def-loutfouz_gpu
#SBATCH --partition=gpubase_bygpu_b4
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=flask_reach_test
#SBATCH --output=/home/nahian26/scratch/logs/flask_reach_test_%j.out
#SBATCH --error=/home/nahian26/scratch/logs/flask_reach_test_%j.err
#SBATCH --mail-user=nahian.rifaat@ontariotechu.net
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
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

# ==============================================================================
# ⚠️  REAL SECRET — DO NOT COMMIT THIS LINE ONCE FILLED IN.
# This is a throwaway networking test, so the token is hardcoded here for
# convenience rather than passed via `sbatch --export`. Fill in the value
# below on nibi directly (e.g. `nano run_flask_test.sh`), but do NOT
# `git add`/commit/push that change — if it ever lands in git history it's
# recoverable forever, even after deleting it in a later commit. If you do
# accidentally commit it, rotate the token at dashboard.ngrok.com/authtokens
# immediately (delete the old one, generate a new one) rather than relying
# on removing it from history.
# `${NGROK_AUTH_TOKEN:-...}` means an explicit `--export=ALL,NGROK_AUTH_TOKEN`
# at submit time still overrides this default if you ever want to use a
# different token for one run without editing the file.
# ==============================================================================
export NGROK_AUTH_TOKEN="${NGROK_AUTH_TOKEN:-28PZDZRjDwUMiT2IElZaUYHD4gX_7Mt4UioEBv2LHQtqqU1t8}"

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

if [ "$NGROK_AUTH_TOKEN" = "PASTE_YOUR_NGROK_TOKEN_HERE" ]; then
    echo ""
    echo "  ⚠ NGROK_AUTH_TOKEN still has its placeholder value — edit this script"
    echo "    on nibi and paste your real token in, or pass one at submit time:"
    echo "    sbatch --export=ALL,NGROK_AUTH_TOKEN=<token> run_flask_test.sh"
    echo "    Continuing with direct-reachability test only, no tunnel."
    NGROK_AUTH_TOKEN=""
fi

if [ -n "$NGROK_AUTH_TOKEN" ]; then
    echo ""
    echo "Checking pyngrok (installing if missing)..."
    python -c "import pyngrok; print('  ✓ pyngrok present')" 2>/dev/null || {
        echo "  ⚠ pyngrok missing — installing..."
        pip install --no-index pyngrok 2>/dev/null || pip install pyngrok
    }
else
    echo ""
    echo "  ℹ NGROK_AUTH_TOKEN not set — direct-reachability test only, no tunnel."
    echo "    (resubmit with: sbatch --export=ALL,NGROK_AUTH_TOKEN=<token> run_flask_test.sh)"
fi

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
if [ -n "$NGROK_AUTH_TOKEN" ]; then
    echo ""
    echo "  ngrok tunnel requested — its public URL prints below once the"
    echo "  Flask process below opens it (look for a line starting [NGROK])."
fi
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
