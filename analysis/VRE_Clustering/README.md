# Clustering RMSE Sweep — HPC handoff

## What this is

A standalone analysis that benchmarks PowerGenome's renewable site clustering at different total cluster budgets (N_pv + N_wind ∈ {500, 1000, 1500, 2000, 2500, 3000}). The output is a per-tech curve of within-cluster reconstruction RMSE vs. N, used to identify diminishing returns for capacity expansion modeling.

The clustering itself is **not reimplemented** — we call PG's own `assign_site_cluster` and `calc_cluster_values` functions directly, so what we benchmark is bit-exactly what PG produces in the production pipeline. Reproduction has been validated locally against a saved PG output (Adjusted Rand Index = 1.0).

## Files in this package

- `clustering_rmse_sweep.py` — main script (validate, sweep, aggregate subcommands)
- `submit_sweep.slurm` — SLURM submission template
- `README.md` — this file

## Setup

1. Make sure the Switch-USA-PG-ReEDS repo is on the HPC, with the conda env (`switch-pg-reeds` or equivalent) built and PowerGenome installed in editable mode (`pip install -e PowerGenome/`).

2. Drop the three files in this package into `${REPO}/analysis/VRE_Clustering/`.

3. Confirm the data files exist (sizes are ~5GB each for the rev_profiles parquets):
   - `pg_data/profiles/{solar,onshorewind}_rev_profiles_20240801.parquet`
   - `pg_data/profiles/{solar,onshorewind}_site_mapping_20240801.parquet`
   - `pg/resource_groups_10weeks_7days_PROFILE_CLUSTERS/ReEDS-cpas-patched/{solar,onshorewind}_lcoe_resource_groups.parquet`
   - `pg/extra_outputs/<region>_*_site_cluster_assignments.csv` (only needed for `validate`)

4. Edit `submit_sweep.slurm`:
   - Set the partition / account flags for your HPC
   - Adjust the conda activation lines if needed (`module load`, `source ~/miniconda3/...`)
   - Verify `PG_REPO_ROOT` points to where the repo lives on the HPC

## Running

**Step 0 — pre-flight check (recommended).** Before submitting anything to the queue, run this on a login node to verify the environment is correctly set up:

```bash
python clustering_rmse_sweep.py preflight
```

This checks all required packages, paths, data files, write permissions, and prints a summary of what's there vs. missing. Fast (a few seconds). If anything is wrong, fix it before submitting the slurm job — otherwise the job will burn queue time only to fail.

The sweep subcommand also runs preflight automatically as its first step (with `strict=True`, so it'll abort on any failure). The standalone `preflight` subcommand is non-strict and prints a full report regardless.

**Step 1 — submit the sweep.** Once preflight is clean:

```bash
sbatch submit_sweep.slurm
```

The slurm script does three things:
1. Runs `preflight` (aborts if anything missing)
2. Runs `validate` on a small region (aborts if reproduction is broken)
3. Runs the full `sweep` with checkpoint resumability (`--resume`)

Outputs are written to a job-specific directory (`${REPO}/analysis/VRE_Clustering/clustering_sweep/job_<jobid>/`) so multiple runs don't overwrite each other.

## Output structure

```
${REPO}/analysis/VRE_Clustering/clustering_sweep/job_<jobid>/
├── results/
│   ├── total_n=500/
│   │   ├── solar/all/
│   │   │   ├── TX.parquet           # per-cluster detail
│   │   │   ├── TX.log               # timing + status
│   │   │   ├── CA.parquet
│   │   │   ├── CA.log
│   │   │   └── ...
│   │   └── onshorewind/all/
│   │       └── ...
│   ├── total_n=1000/
│   │   └── ...
│   └── ...
├── rmse_sweep.csv                   # summary (one row per task)
└── rmse_sweep_aggregated.csv        # capacity-weighted per (total_n, tech)
```

Each `.parquet` file holds per-cluster detail (cluster id, n_members, capacity, SSE × capacity, cluster-level RMSE) and embeds the task summary in pandas attrs.

## Resumability

The sweep is designed to handle SLURM time limits and worker crashes gracefully:
- Each task writes its output atomically to a per-task parquet file
- `--resume` (set in the slurm script) skips any task whose parquet already exists
- If the job hits the wall-time limit, just resubmit — it'll pick up exactly where it stopped
- To resume into the same output dir explicitly, set `PG_OUT_DIR` before sbatch:
  ```bash
  PG_OUT_DIR=/path/to/job_12345 sbatch submit_sweep.slurm
  ```

## Sending results back

After the sweep completes:

```bash
cd ${REPO}/analysis/VRE_Clustering/clustering_sweep
tar czf clustering_sweep_results_<jobid>.tar.gz job_<jobid>/
```

The tarball will be ~100MB-1GB depending on how many CPAs there are total. Send via whatever transfer mechanism is convenient (Globus, scp, etc.).

The full per-cluster detail is preserved, so all downstream aggregation and plotting can happen locally without re-running anything on the HPC.

## Resource notes

- **Memory**: agglomerative ward is O(n²) memory in CPAs. The biggest solar regions have ~20k+ CPAs, requiring ~3GB just for the distance matrix per worker, plus ~5GB for profile data — call it ~10GB peak per worker for the largest regions, more like 1-3GB for typical regions. With 64 workers and 512GB memory, peak instantaneous use is roughly 50-150GB — comfortable. If your HPC has 1TB nodes available, you can push `--cpus-per-task` and `--mem` higher in the slurm script.
- **Wall time**: locally observed ~30 min per task at the largest regions on a 2-core MacBook. With 64 workers on HPC and ~276 total tasks (46 regions × 2 techs × 6 grid points minus solar slicing), expected wall time is 2-8 hours. SLURM script books 48h for safety — a single hung task can't take the whole job down.
- **Threading**: `OMP_NUM_THREADS=1` etc. is set explicitly because sklearn's agglomerative is single-threaded internally and oversubscription hurts.
- **Disk**: per-task parquets are tiny (KB each). Total output footprint is well under 1GB, easily transferable.

## Troubleshooting

**Pre-flight fails** — the report tells you exactly what's missing or unreadable. Most common causes: paths in env vars don't match the HPC's actual layout (fix: edit `PG_REPO_ROOT` etc.), or a package missing from the conda env (fix: `pip install` it).

**Validate fails with `VALIDATION FAILED: ARI < 1.0`** — saved `extra_outputs` may be from a different clustering algorithm than the default. Check `pg/extra_outputs/<region>_*_site_cluster_assignments.csv` mtime vs the renewables_clusters block in the active settings. If the saved files were generated with `feature: cf`, run validate with `--feature cf`. The sweep itself uses `--feature profile` regardless.

**Validate fails with `no CPAs in metadata for region=...`** — region name mismatch. PG output filenames sometimes have qualifier suffixes (e.g. `_pnh`) that are part of the region name. Check the available regions list the script prints.

**Workers dying mid-run with memory warnings** — drop `--cpus-per-task` in the slurm script. n_jobs=8 with 256GB is very safe.

**A specific region/task fails with an exception** — its `.log` file in `results/.../` will contain the full traceback. The sweep continues past it; just diagnose afterward.

## Methodology summary (for the writeup)

PowerGenome's renewable clustering pipeline (`powergenome.cluster.renewables.assign_site_cluster`) was called directly via its public API at each (region, technology, N) combination in the sweep. Reproduction was validated bit-exactly against PG's own saved cluster assignments (Adjusted Rand Index = 1.0). Within-cluster reconstruction RMSE was computed as the capacity-weighted root-mean-square deviation between each candidate project area's individual hourly capacity-factor profile and the capacity-weighted centroid of its assigned cluster, then aggregated across regions using capacity weights. Per-region cluster budgets were allocated using sqrt(capacity) weighting (`allocate_clusters` function with min_per_zone=1).
