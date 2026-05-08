#!/bin/bash
# ============================================================================
# RMSE sweep for PowerGenome renewable site clustering
# ----------------------------------------------------------------------------
# This is a plain bash runner -- intended to be run inside a tmux session on
# an HPC node, no batch scheduler required.
#
# Typical workflow:
#
#   ssh hpc.example.edu
#   tmux new -s rmse_sweep             # start a new tmux session
#   cd ~/Switch-USA-PG-ReEDS/analysis/VRE_Clustering
#   bash run_sweep.sh                  # launch the sweep
#                                      # (Ctrl-b then d to detach from tmux)
#   ...come back hours later...
#   ssh hpc.example.edu
#   tmux attach -t rmse_sweep          # reattach to the running session
#
# If the run dies for any reason (node reboot, killed by the scheduler, etc.),
# just bash run_sweep.sh again -- the --resume flag will skip already-completed
# tasks and pick up where it stopped.
# ============================================================================

set -euo pipefail

# ----- environment -----
# Activate the conda env. Edit this if her HPC uses a different conda layout.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate switch-pg-reeds

# ----- single-threaded numerical libs (joblib does the parallelism) -----
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ----- paths -----
# Edit PG_REPO_ROOT if the repo lives somewhere other than $HOME on the HPC
export PG_REPO_ROOT="${PG_REPO_ROOT:-${HOME}/Switch-USA-PG-ReEDS}"

# Output dir: timestamped so reruns don't overwrite each other.
# To resume an existing run, set PG_OUT_DIR to its directory before running.
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
export PG_OUT_DIR="${PG_OUT_DIR:-${PG_REPO_ROOT}/analysis/VRE_Clustering/clustering_sweep/run_${RUN_TAG}}"
mkdir -p "${PG_OUT_DIR}"

# ----- parallelism -----
# How many parallel workers? Set to (cores on this node - 2) for safety, or
# pick a number based on what you know about the node. Pass N_JOBS=8 to
# override.
N_JOBS="${N_JOBS:-$(($(nproc) - 2))}"
[ "$N_JOBS" -lt 1 ] && N_JOBS=1

echo "=========================================="
echo "Node:         $(hostname)"
echo "Repo:         ${PG_REPO_ROOT}"
echo "Output:       ${PG_OUT_DIR}"
echo "Cores avail:  $(nproc)"
echo "n_jobs:       ${N_JOBS}"
echo "Time started: $(date)"
echo "=========================================="

cd "${PG_REPO_ROOT}/analysis/VRE_Clustering"

# Tee stdout to a log file so the run is captured even if tmux dies
LOG="${PG_OUT_DIR}/run.log"
exec > >(tee -a "${LOG}") 2>&1

# ----- step 1: pre-flight -----
echo "--- Pre-flight ---"
python clustering_rmse_sweep.py preflight \
    || { echo "PRE-FLIGHT FAILED -- aborting"; exit 1; }

# ----- step 2: sweep -----
# Note: the validate step is skipped here since the saved extra_outputs may
# not exist on the HPC. Reproduction was already validated locally on Melek's
# laptop with ARI=1.0; PG's clustering is deterministic, so HPC reproduction
# is guaranteed identical given the same package versions. To re-validate on
# HPC, scp one extra_outputs CSV up and run:
#   python clustering_rmse_sweep.py validate \
#       --tech onshorewind --region AL_and_e_MS_and_FL_pnh \
#       --feature cf --method agglomerative
echo ""
echo "--- Sweep ---"
python clustering_rmse_sweep.py sweep \
    --grid 500 1000 1500 2000 2500 3000 \
    --split sqrt_capacity \
    --feature profile --method agglomerative \
    --n-jobs "${N_JOBS}" \
    --resume

echo ""
echo "Time finished: $(date)"
echo "Output is in: ${PG_OUT_DIR}"
echo ""
echo "To send results back to Melek, run:"
echo "  cd \$(dirname ${PG_OUT_DIR})"
echo "  tar czf clustering_sweep_results_${RUN_TAG}.tar.gz \$(basename ${PG_OUT_DIR})/"
