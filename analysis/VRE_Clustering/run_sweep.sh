#!/bin/bash
# ============================================================================
# RMSE sweep for PowerGenome renewable site clustering
# ----------------------------------------------------------------------------
# This is a plain bash runner -- run inside a tmux session on an HPC node,
# no batch scheduler required.
#
# Typical workflow (if not already in tmux):
#
#   ssh hpc.example.edu
#   tmux new -s rmse_sweep
#   cd /path/to/Switch-USA-PG-ReEDS/analysis/VRE_Clustering
#   bash run_sweep.sh
#   # Ctrl-b d to detach from tmux
#   ...come back later...
#   tmux attach -t rmse_sweep
#
# If the run dies for any reason, just bash run_sweep.sh again -- the
# --resume flag skips already-completed tasks and picks up where it stopped.
#
# Environment variables (override defaults):
#   CONDA_ENV        -- conda env name (default: switch-pg-reeds)
#   N_JOBS           -- parallel worker count (default: nproc - 2)
#   PG_OUT_DIR       -- output dir; for resuming, set to an existing run dir
#   RUN_TAG          -- timestamp for output dir (default: now)
#   GRID             -- override the sweep grid (default: 500 1000 ... 3000)
#   SPLIT            -- tech split method (default: sqrt_capacity)
#   FEATURE          -- profile or cf (default: profile)
#   METHOD           -- agglomerative or kmeans (default: agglomerative)
# ============================================================================

# -e: exit on error.  No -u (conda's deactivate scripts trip on unset vars).
# pipefail: propagate failures through pipes.
set -eo pipefail

# ----------------------------------------------------------------------------
# Locate ourselves: where does this script actually live?
# ----------------------------------------------------------------------------
# This handles the case where bash is invoked from any directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ----------------------------------------------------------------------------
# Locate and activate conda
# ----------------------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-switch-pg-reeds}"

# Try a few standard places in order. First match wins.
CONDA_SH=""
for candidate in \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "/opt/miniconda3/etc/profile.d/conda.sh" \
    "/opt/anaconda3/etc/profile.d/conda.sh" \
    "${CONDA_PREFIX:-}/etc/profile.d/conda.sh" ; do
    if [ -f "$candidate" ]; then
        CONDA_SH="$candidate"
        break
    fi
done

# Fall back to whatever conda is on PATH if no .sh file found
if [ -z "$CONDA_SH" ]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base 2>/dev/null)"
        if [ -n "$CONDA_BASE" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
            CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
        fi
    fi
fi

if [ -z "$CONDA_SH" ]; then
    echo "ERROR: could not find conda.sh in any standard location."
    echo "Set CONDA_PREFIX to your conda install dir, or install miniconda."
    exit 1
fi

echo "Using conda from: ${CONDA_SH}"
# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

# ----------------------------------------------------------------------------
# Threading: single-threaded numerical libs so joblib does the parallelism
# ----------------------------------------------------------------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ----------------------------------------------------------------------------
# Output dir
# ----------------------------------------------------------------------------
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
export PG_OUT_DIR="${PG_OUT_DIR:-${REPO_ROOT}/analysis/VRE_Clustering/clustering_sweep/run_${RUN_TAG}}"
mkdir -p "${PG_OUT_DIR}"

# ----------------------------------------------------------------------------
# Parallelism
# ----------------------------------------------------------------------------
N_JOBS="${N_JOBS:-$(($(nproc) - 2))}"
[ "$N_JOBS" -lt 1 ] && N_JOBS=1

# ----------------------------------------------------------------------------
# Sweep parameters (env-overrideable)
# ----------------------------------------------------------------------------
GRID="${GRID:-500 1000 1500 2000 2500 3000}"
SPLIT="${SPLIT:-sqrt_capacity}"
FEATURE="${FEATURE:-profile}"
METHOD="${METHOD:-agglomerative}"

# ----------------------------------------------------------------------------
# Banner
# ----------------------------------------------------------------------------
echo "=========================================="
echo "Node:         $(hostname)"
echo "Script dir:   ${SCRIPT_DIR}"
echo "Repo root:    ${REPO_ROOT}"
echo "Output:       ${PG_OUT_DIR}"
echo "Conda env:    ${CONDA_ENV}"
echo "Python:       $(which python)"
echo "Cores avail:  $(nproc)"
echo "n_jobs:       ${N_JOBS}"
echo "Grid:         ${GRID}"
echo "Split:        ${SPLIT}"
echo "Feature:      ${FEATURE}"
echo "Method:       ${METHOD}"
echo "Time started: $(date)"
echo "=========================================="

# ----------------------------------------------------------------------------
# Always run from the script's directory so `python clustering_rmse_sweep.py`
# resolves correctly regardless of where bash was invoked from.
# ----------------------------------------------------------------------------
cd "${SCRIPT_DIR}"

# ----------------------------------------------------------------------------
# Clear stale __pycache__ so python doesn't load old bytecode if the .py
# was updated mid-conversation. Safe; pycache regenerates on next import.
# ----------------------------------------------------------------------------
rm -rf __pycache__

# ----------------------------------------------------------------------------
# Verify the script we're about to run is the one we expect
# (helps catch the "stale file at unexpected path" gotcha)
# ----------------------------------------------------------------------------
SCRIPT_PATH="${SCRIPT_DIR}/clustering_rmse_sweep.py"
if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "ERROR: clustering_rmse_sweep.py not found in ${SCRIPT_DIR}"
    exit 1
fi
echo "Will execute: $(python -c "import os; print(os.path.realpath('${SCRIPT_PATH}'))")"
echo ""

# ----------------------------------------------------------------------------
# Tee output to a log file. Writes are tee'd as well as written to terminal,
# so even if tmux dies the full output is captured.
# ----------------------------------------------------------------------------
LOG="${PG_OUT_DIR}/run.log"
exec > >(tee -a "${LOG}") 2>&1

# ----------------------------------------------------------------------------
# Step 1: pre-flight
# ----------------------------------------------------------------------------
echo "--- Pre-flight ---"
python "${SCRIPT_PATH}" preflight || {
    echo "PRE-FLIGHT FAILED -- aborting"
    exit 1
}

# ----------------------------------------------------------------------------
# Step 2: sweep
# ----------------------------------------------------------------------------
# Note: validate is skipped because saved extra_outputs may not be on HPC.
# Reproduction was already validated on Melek's laptop with ARI=1.0; PG's
# clustering is deterministic so HPC reproduction is guaranteed identical
# given the same package versions. To re-validate, scp one extra_outputs CSV
# up and run:
#   python clustering_rmse_sweep.py validate \
#       --tech onshorewind --region AL_and_e_MS_and_FL_pnh \
#       --feature cf --method agglomerative
echo ""
echo "--- Sweep ---"
# shellcheck disable=SC2086  # GRID is intentionally unquoted to expand
python "${SCRIPT_PATH}" sweep \
    --grid ${GRID} \
    --split "${SPLIT}" \
    --feature "${FEATURE}" \
    --method "${METHOD}" \
    --n-jobs "${N_JOBS}" \
    --resume

echo ""
echo "Time finished: $(date)"
echo "Output is in: ${PG_OUT_DIR}"
echo ""
echo "To send results back to Melek:"
echo "  cd $(dirname "${PG_OUT_DIR}")"
echo "  tar czf clustering_sweep_results_${RUN_TAG}.tar.gz $(basename "${PG_OUT_DIR}")/"
