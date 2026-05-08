#!/bin/bash
#SBATCH --job-name=cluster-rmse-sweep
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --partition=standard      # adjust: long, large-mem, etc.
# #SBATCH --account=your_account  # uncomment + set if your HPC requires it

# ============================================================================
# RMSE sweep for PowerGenome renewable site clustering
# ----------------------------------------------------------------------------
# Resource ask:
#   - 64 cores: enough parallelism across regions x grid points
#   - 512 GB:   2x what's strictly needed but cheap insurance.
#               If your node has 1 TB available and supports more cores,
#               bump --cpus-per-task and --mem proportionally.
#   - 48 hours: well above the expected ~4-12 hr runtime; safe headroom.
#
# Memory math: agglomerative ward needs O(n^2) memory.
#   - Largest solar regions: ~20k CPAs -> ~3 GB distance matrix +
#     ~5 GB profile data = ~10 GB peak per worker.
#   - Typical regions: ~1-3 GB per worker.
# With 64 workers, peak instantaneous use is ~50-150 GB. 512 GB is comfortable.
# ============================================================================

set -euo pipefail
mkdir -p logs

# ----- environment -----
# Replace with the appropriate module loads / conda activation for your HPC
# module load anaconda3
source ~/miniconda3/etc/profile.d/conda.sh
conda activate switch-pg-reeds   # adjust env name if different

# ----- single-threaded numerical libs (joblib does the parallelism) -----
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ----- paths (override defaults baked into clustering_rmse_sweep.py) -----
# IMPORTANT: edit PG_REPO_ROOT to wherever the repo lives on the HPC
export PG_REPO_ROOT="${PG_REPO_ROOT:-${HOME}/Switch-USA-PG-ReEDS}"

# Output goes to a job-specific subdir. To resume an existing run, set
# PG_OUT_DIR explicitly to its directory before sbatch.
export PG_OUT_DIR="${PG_OUT_DIR:-${PG_REPO_ROOT}/analysis/VRE_Clustering/clustering_sweep/job_${SLURM_JOB_ID:-local}}"
mkdir -p "${PG_OUT_DIR}"

echo "=========================================="
echo "Job ID:       ${SLURM_JOB_ID:-local}"
echo "Node:         $(hostname)"
echo "Repo:         ${PG_REPO_ROOT}"
echo "Output:       ${PG_OUT_DIR}"
echo "CPUs:         ${SLURM_CPUS_PER_TASK:-1}"
echo "Memory:       ${SLURM_MEM_PER_NODE:-?}"
echo "Time started: $(date)"
echo "=========================================="

cd "${PG_REPO_ROOT}/analysis/VRE_Clustering"

# ----- step 1: pre-flight -----
# Verifies all required files, packages, and write permissions BEFORE
# burning HPC time on the sweep. Aborts on any failure.
echo "--- Pre-flight ---"
python clustering_rmse_sweep.py preflight \
    || { echo "PRE-FLIGHT FAILED -- aborting"; exit 1; }

# ----- step 2: validate -----
# Confirm bit-exact reproduction of PG's clustering on a known region.
# NOTE: Set --feature and --method to match what PG used to generate the
# saved files in extra_outputs. If those files are from CF clustering, use
# `--feature cf`. If from profile clustering, use `--feature profile`.
# The sweep itself uses whatever you set in step 3 -- they need not match.
echo ""
echo "--- Validate ---"
python clustering_rmse_sweep.py validate \
    --tech onshorewind --region AL_and_e_MS_and_FL_pnh \
    --feature cf --method agglomerative \
    || { echo "VALIDATION FAILED -- aborting sweep"; exit 1; }

# ----- step 3: sweep -----
echo ""
echo "--- Sweep ---"
python clustering_rmse_sweep.py sweep \
    --grid 500 1000 1500 2000 2500 3000 \
    --split sqrt_capacity \
    --feature profile --method agglomerative \
    --n-jobs "${SLURM_CPUS_PER_TASK:-4}" \
    --resume

echo ""
echo "Time finished: $(date)"
echo "Output is in: ${PG_OUT_DIR}"
echo ""
echo "To send results back to Melek, run:"
echo "  cd \$(dirname ${PG_OUT_DIR})"
echo "  tar czf clustering_sweep_results_${SLURM_JOB_ID:-local}.tar.gz \$(basename ${PG_OUT_DIR})/"
