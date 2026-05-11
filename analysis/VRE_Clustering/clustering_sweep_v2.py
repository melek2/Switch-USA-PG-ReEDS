"""
clustering_sweep_v2.py
======================
Compute within-cluster reconstruction RMSE for renewable site clustering at
multiple per-tech N values. Per-zone breakdown so the full distribution of
cluster RMSEs is visible, not just an aggregate.

INDEPENDENT PER-TECH BUDGETS (vs prior single shared budget):
Each tech gets its own N values, paired by position into "runs". Run i uses
N_SOLAR[i] solar clusters and N_WIND[i] wind clusters. No sqrt(capacity)
split between techs — they're allocated independently. This matters because
wind profiles are more diverse than solar per unit capacity, so a capacity-
weighted split systematically under-allocates wind.

ARCHITECTURE:
For each (region, tech) pair, build the agglomerative ward linkage tree ONCE
using scipy.cluster.hierarchy.linkage. Then cut at each run's N using
fcluster -- which is essentially free.
  - Linkage cost: O(n^2 log n) per region, paid ONCE
  - Cut cost: O(n) per N value, paid once per run

FEATURE MODES:
  - feature='profile' -> linkage on full hourly CF vectors (61320 dims)
  - feature='cf'      -> linkage on annual mean CF (1 dim, for fast testing)

USAGE
  # Local CF test (~10 min):
  python clustering_sweep_v2.py --feature cf --n-jobs 4

  # Default: 4 runs at 1000, 1500, 2000, 2500 for both techs
  python clustering_sweep_v2.py --feature profile --n-jobs 60 --strict-jobs 16

  # Solar fixed, wind sweep:
  python clustering_sweep_v2.py --feature profile --n-jobs 60 --strict-jobs 16 \
      --n-solar 1500 --n-wind 1000 1500 2000 2500

OUTPUT
  Each row in per_cluster_rmse_<feature>.csv now has:
    - tech, region, cluster (identity)
    - run_idx, n_solar, n_wind (which run this row came from)
    - total_n: per-tech N for that run (= n_solar for solar rows, n_wind for wind)
    - N_total_for_region: actual cluster count this region got
    - RMSE stats, interconnect cost stats

ENV VARS (override defaults; defaults are repo-relative):
  PG_REPO_ROOT, PG_PROFILES_DIR, PG_RG_DIR, PG_OUT_DIR
"""
from __future__ import annotations

import os
import sys
import math
import time
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Force single-threaded numerical libs so joblib parallelism doesn't oversubscribe
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# ----------------------------------------------------------------------------
# Paths -- relative to this script's location
# ----------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
_REPO_FROM_SCRIPT = _THIS.parent.parent.parent  # script -> VRE_Clustering -> analysis -> repo

REPO_ROOT     = Path(os.environ.get("PG_REPO_ROOT", _REPO_FROM_SCRIPT))
PROFILES_DIR  = Path(os.environ.get("PG_PROFILES_DIR", REPO_ROOT / "pg_data" / "profiles"))
RG_DIR        = Path(os.environ.get("PG_RG_DIR",
                     REPO_ROOT / "pg" / "resource_groups_10weeks_7days_PROFILE_CLUSTERS" / "ReEDS-cpas-patched"))
OUT_DIR       = Path(os.environ.get("PG_OUT_DIR",
                     REPO_ROOT / "analysis" / "VRE_Clustering" / "sweep_v2a"))


# ----------------------------------------------------------------------------
# Tech registry
# ----------------------------------------------------------------------------
TECHS: Dict[str, Dict] = {
    "solar": {
        "metadata":  "solar_lcoe_resource_groups.parquet",
        "profiles":  "solar_rev_profiles_20240801.parquet",
        "site_map":  "solar_site_mapping_20240801.parquet",
    },
    "onshorewind": {
        "metadata":  "onshorewind_lcoe_resource_groups.parquet",
        "profiles":  "onshorewind_rev_profiles_20240801.parquet",
        "site_map":  "onshorewind_site_mapping_20240801.parquet",
    },
}

# Default per-tech N values, paired by position into "runs".
# Run i uses N_SOLAR[i] solar clusters and N_WIND[i] wind clusters.
DEFAULT_N_SOLAR = (1000, 1500, 2000, 2500)
DEFAULT_N_WIND  = (1000, 1500, 2000, 2500)
TECH_BY_KEY = {"solar": "solar", "wind": "onshorewind"}  # CLI key -> internal name


# ----------------------------------------------------------------------------
# Allocator (your function -- min_per_zone=1 for headroom at low N)
# ----------------------------------------------------------------------------
def allocate_clusters(cap, n_total, min_per_zone=1, max_sites=None):
    eligible = cap[cap > 0]
    ceil = (max_sites.reindex(eligible.index) if max_sites is not None
            else pd.Series(float("inf"), index=eligible.index))
    alloc = pd.Series(min_per_zone, index=eligible.index).clip(upper=ceil)
    if alloc.sum() > n_total:
        raise ValueError(f"Floor of {int(alloc.sum())} exceeds budget {n_total}.")
    w = eligible.pow(0.5)
    raw = (n_total - alloc.sum()) * w / w.sum()
    alloc = (alloc + raw.apply(math.floor)).clip(upper=ceil)
    leftover = int(n_total - alloc.sum())
    for z in (raw - raw.apply(math.floor)).sort_values(ascending=False).index:
        if leftover <= 0:
            break
        if alloc[z] < ceil[z]:
            alloc[z] += 1
            leftover -= 1
    return alloc.astype(int)


# ----------------------------------------------------------------------------
# Per-region work: build linkage once, cut at multiple Ns, compute RMSE
# ----------------------------------------------------------------------------
def process_region(tech: str, region: str,
                   metadata: pd.DataFrame, profiles_path: Path,
                   site_map: pd.Series,
                   n_clusters_list: List[int],
                   total_n_labels: List[int],
                   run_meta: List[Dict[str, int]],
                   feature: str = "profile") -> List[Dict]:
    """
    For one (tech, region):
      1. Pull all CPAs in this region
      2. Load each CPA's profile (or compute its annual mean CF)
      3. Build agglomerative ward linkage (ONCE)
      4. For each (N, total_n_label, run_meta) triple:
         - Cut the linkage at N
         - For each resulting cluster, compute the RMSE of every member CPA
           against the cluster's capacity-weighted centroid profile
         - Save per-cluster summary tagged with run metadata (n_solar, n_wind, run_idx)
    Returns a list of dicts -- one per (region, N, cluster).

    n_clusters_list, total_n_labels, run_meta must all have the same length.
    Each run_meta dict has keys 'run_idx', 'n_solar', 'n_wind'.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    assert len(n_clusters_list) == len(total_n_labels) == len(run_meta), \
        "n_clusters_list, total_n_labels, run_meta must align"

    renew_data = metadata[metadata["region"] == region].copy()
    renew_data = renew_data.rename(columns={"capacity_mw": "mw"}) if "mw" not in renew_data.columns else renew_data
    n_cpas = len(renew_data)
    if n_cpas == 0:
        return []

    # Map cpa_id -> site name
    cpa_to_site = site_map.reindex(renew_data["cpa_id"]).dropna()
    valid_cpas = cpa_to_site.index.tolist()
    sites = list(set(cpa_to_site.tolist()))
    if not sites:
        return []

    # Load profiles for these sites
    profiles_df = pq.read_table(profiles_path, columns=sites).to_pandas()
    T = len(profiles_df)

    # Build CPA -> profile array, in the order of valid_cpas
    profile_matrix = np.zeros((len(valid_cpas), T), dtype=np.float32)
    cpa_caps = np.zeros(len(valid_cpas), dtype=np.float32)
    cpa_iccap = np.full(len(valid_cpas), np.nan, dtype=np.float32)
    has_iccap = "interconnect_capex_mw" in renew_data.columns
    # Fast lookup by cpa_id -> row in renew_data
    rd_indexed = renew_data.set_index("cpa_id")
    for i, cpa in enumerate(valid_cpas):
        s = cpa_to_site[cpa]
        if s in profiles_df.columns:
            profile_matrix[i] = profiles_df[s].to_numpy(dtype=np.float32)
            cpa_caps[i] = rd_indexed.at[cpa, "mw"]
            if has_iccap:
                cpa_iccap[i] = rd_indexed.at[cpa, "interconnect_capex_mw"]

    # Build feature matrix for clustering
    if feature == "profile":
        feat = profile_matrix
    elif feature == "cf":
        feat = profile_matrix.mean(axis=1).reshape(-1, 1)
    else:
        raise ValueError(f"Unknown feature: {feature}")

    # Linkage (once)
    Z = linkage(feat, method="ward")

    rows = []
    for N, total_n_label, run in zip(n_clusters_list, total_n_labels, run_meta):
        run_cols = {
            "total_n": total_n_label,
            "run_idx": int(run["run_idx"]),
            "n_solar": int(run["n_solar"]),
            "n_wind": int(run["n_wind"]),
        }
        if N >= len(valid_cpas):
            # Each CPA its own cluster -> RMSE = 0
            for i, cpa in enumerate(valid_cpas):
                ic_val = cpa_iccap[i]
                ic = float(ic_val) if not np.isnan(ic_val) else None
                rows.append({
                    "tech": tech, "region": region,
                    **run_cols,
                    "N_total_for_region": N,
                    "cluster": i, "n_members": 1,
                    "cluster_cap_mw": float(cpa_caps[i]),
                    "centroid_mean_cf": float(profile_matrix[i].mean()),
                    "mean_member_rmse": 0.0,
                    "std_member_rmse": 0.0,
                    "max_member_rmse": 0.0,
                    "wt_mean_iccap_per_mw": ic,
                    "std_iccap_per_mw": 0.0 if ic is not None else None,
                    "min_iccap_per_mw": ic,
                    "max_iccap_per_mw": ic,
                })
            continue

        labels = fcluster(Z, t=N, criterion="maxclust")
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            members = profile_matrix[mask]
            weights = cpa_caps[mask]
            total_w = weights.sum()
            if total_w == 0:
                continue
            # Capacity-weighted centroid profile
            centroid = (members * weights[:, None]).sum(axis=0) / total_w
            # Per-member RMSE vs centroid
            diff = members - centroid[None, :]
            per_member_rmse = np.sqrt((diff * diff).mean(axis=1))

            # Interconnect cost stats: match PG's MERGE semantics.
            # PG produces None when ANY member lacks the value (the "uniques"
            # behavior); for "means" with all values present, it's a strict
            # MW-weighted mean: sum(mw_i * x_i) / sum(mw_i).
            ic_members = cpa_iccap[mask]
            if has_iccap and not np.isnan(ic_members).any() and total_w > 0:
                wt_mean_ic = float((ic_members * weights).sum() / total_w)
                # Capacity-weighted variance/std (population formula)
                wt_var = float(((ic_members - wt_mean_ic) ** 2 * weights).sum() / total_w)
                std_ic = math.sqrt(wt_var)
                min_ic = float(ic_members.min())
                max_ic = float(ic_members.max())
            else:
                wt_mean_ic = std_ic = min_ic = max_ic = None

            rows.append({
                "tech": tech, "region": region,
                **run_cols,
                "N_total_for_region": N,
                "cluster": int(cluster_id) - 1,
                "n_members": int(mask.sum()),
                "cluster_cap_mw": float(total_w),
                "centroid_mean_cf": float(centroid.mean()),
                "mean_member_rmse": float(per_member_rmse.mean()),
                "std_member_rmse": float(per_member_rmse.std()),
                "max_member_rmse": float(per_member_rmse.max()),
                "wt_mean_iccap_per_mw": wt_mean_ic,
                "std_iccap_per_mw": std_ic,
                "min_iccap_per_mw": min_ic,
                "max_iccap_per_mw": max_ic,
            })
    return rows


# ----------------------------------------------------------------------------
# Sweep driver
# ----------------------------------------------------------------------------
def run_sweep(n_solar: Tuple[int, ...], n_wind: Tuple[int, ...],
              n_jobs: int, feature: str,
              strict_jobs: int = 1, resume: bool = False):
    from joblib import Parallel, delayed

    # Pair n_solar and n_wind by position into runs. Broadcast length-1.
    if len(n_solar) == 1 and len(n_wind) > 1:
        n_solar = n_solar * len(n_wind)
    elif len(n_wind) == 1 and len(n_solar) > 1:
        n_wind = n_wind * len(n_solar)
    if len(n_solar) != len(n_wind):
        raise ValueError(
            f"--n-solar ({len(n_solar)} values) and --n-wind ({len(n_wind)} values) "
            f"must have the same length, or one must be length-1 (to broadcast)."
        )

    runs = [{"run_idx": i, "n_solar": int(s), "n_wind": int(w)}
            for i, (s, w) in enumerate(zip(n_solar, n_wind))]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # grid.txt now records the per-tech runs explicitly
    grid_lines = ["run_idx,n_solar,n_wind"]
    for r in runs:
        grid_lines.append(f"{r['run_idx']},{r['n_solar']},{r['n_wind']}")
    (OUT_DIR / "grid.txt").write_text("\n".join(grid_lines))

    out_csv = OUT_DIR / f"per_cluster_rmse_{feature}.csv"
    if not resume and out_csv.exists():
        print(f"Removing existing {out_csv} (resume=False; otherwise schemas could mix)")
        out_csv.unlink()

    print("=" * 64)
    print(f"Sweep config")
    print(f"  feature   : {feature}")
    print(f"  runs      :")
    for r in runs:
        print(f"    run {r['run_idx']}: n_solar={r['n_solar']}, n_wind={r['n_wind']}")
    print(f"  n_jobs    : {n_jobs}  (strict floor: {strict_jobs})")
    print(f"  out_csv   : {out_csv}")
    print(f"  resume    : {resume}")
    print("=" * 64)

    if n_jobs < strict_jobs:
        print(f"\nERROR: n_jobs={n_jobs} is below strict floor of {strict_jobs}.")
        print(f"This usually means nproc is wrong on this node, or the env var didn't propagate.")
        print(f"Pass --n-jobs explicitly. Aborting.")
        sys.exit(2)

    print("Loading tech metadata...")
    metadata, site_maps = {}, {}
    for tech, cfg in TECHS.items():
        md = pd.read_parquet(RG_DIR / cfg["metadata"])
        if "mw" not in md.columns:
            md = md.rename(columns={"capacity_mw": "mw"})
        sm = pd.read_parquet(PROFILES_DIR / cfg["site_map"]).set_index("CPA_ID")["Site"]
        metadata[tech] = md
        site_maps[tech] = sm
        print(f"  {tech}: {len(md):,} CPAs, {md['region'].nunique()} regions, "
              f"{md['mw'].sum()/1e3:,.1f} GW")

    # For each (tech, region), collect a list of (N, total_n_label, run_meta) triples.
    # Per-tech: each run gets its OWN N (no more sqrt-splitting of a single total).
    region_pairs: Dict[Tuple[str, str], List[Tuple[int, int, Dict]]] = {}
    tech_key_map = {"solar": "n_solar", "onshorewind": "n_wind"}
    for run in runs:
        for tech in TECHS:
            n_tech = run[tech_key_map[tech]]
            cap_by_region = metadata[tech].groupby("region")["mw"].sum()
            try:
                alloc = allocate_clusters(cap_by_region, n_tech)
            except ValueError as e:
                print(f"  WARN: alloc failed for run {run['run_idx']}/{tech} (N={n_tech}): {e}")
                continue
            for region, n in alloc.items():
                region_pairs.setdefault((tech, region), []).append(
                    (int(n), int(n_tech), dict(run))
                )
            print(f"  run {run['run_idx']} ({tech}): {n_tech} clusters, "
                  f"{len(alloc)} regions, range [{alloc.min()}, {alloc.max()}]")

    # Resume support
    if resume and out_csv.exists():
        prev = pd.read_csv(out_csv)
        done_keys = set((r["tech"], r["region"]) for _, r in
                        prev[["tech", "region"]].drop_duplicates().iterrows())
        before = len(region_pairs)
        region_pairs = {k: v for k, v in region_pairs.items() if k not in done_keys}
        print(f"Resume: {before - len(region_pairs)} (tech, region) tasks already done")

    # Build task list: (tech, region, [Ns], [total_n labels], [run_meta dicts])
    tasks = []
    for (tech, region), triples in region_pairs.items():
        triples_sorted = sorted(triples, key=lambda t: t[1])  # sort by total_n_label
        ns     = [t[0] for t in triples_sorted]
        labels = [t[1] for t in triples_sorted]
        rmeta  = [t[2] for t in triples_sorted]
        tasks.append((tech, region, ns, labels, rmeta))

    region_sizes = {(t, r): (metadata[t]["region"] == r).sum() for t, r in region_pairs}
    tasks.sort(key=lambda x: -region_sizes.get((x[0], x[1]), 0))

    print(f"\nDispatching {len(tasks)} (tech, region) jobs to {n_jobs} workers")
    print(f"Largest 5 regions:")
    for t, r, ns, labels, rmeta in tasks[:5]:
        pretty = ", ".join(f"N={n} (run {rm['run_idx']})" for n, rm in zip(ns, rmeta))
        print(f"  {t:14s} {r:35s} {region_sizes[(t,r)]:>6} CPAs, [{pretty}]")

    def _wrap(tech, region, ns, labels, rmeta):
        try:
            t0 = time.time()
            result = process_region(
                tech, region, metadata[tech],
                PROFILES_DIR / TECHS[tech]["profiles"],
                site_maps[tech], ns, labels, rmeta, feature=feature,
            )
            elapsed = time.time() - t0
            return {"ok": True, "tech": tech, "region": region,
                    "n_cpas": region_sizes[(tech, region)],
                    "elapsed_sec": elapsed, "rows": result}
        except Exception:
            return {"ok": False, "tech": tech, "region": region,
                    "tb": traceback.format_exc()}

    write_header = not (resume and out_csv.exists())
    n_done, n_err = 0, 0
    with Parallel(n_jobs=n_jobs, return_as="generator", verbose=10) as parallel:
        for res in parallel(delayed(_wrap)(t[0], t[1], t[2], t[3], t[4]) for t in tasks):
            if res["ok"]:
                df = pd.DataFrame(res["rows"])
                df.to_csv(out_csv, mode="a", header=write_header, index=False)
                write_header = False
                n_done += 1
                print(f"  [{n_done}/{len(tasks)}] {res['tech']:14s} {res['region']:35s} "
                      f"{res['n_cpas']:>6} CPAs in {res['elapsed_sec']:>6.1f}s "
                      f"-> {len(res['rows']):>4} cluster rows")
            else:
                n_err += 1
                print(f"  ERROR in {res['tech']}/{res['region']}:\n{res['tb']}")

    print(f"\nDone. {n_done} succeeded, {n_err} errored.")
    print(f"Output: {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", default="profile",
                        choices=["profile", "cf"],
                        help="profile = full hourly CF vectors, cf = annual mean CF (fast)")
    parser.add_argument("--n-jobs", type=int, required=True,
                        help="Parallel workers. Pass explicitly to avoid nproc surprises.")
    parser.add_argument("--strict-jobs", type=int, default=1,
                        help="Abort if n_jobs is below this. Use 16+ on HPC.")
    parser.add_argument("--n-solar", type=int, nargs="+",
                        default=list(DEFAULT_N_SOLAR),
                        help="N for solar in each run. Default: 1000 1500 2000 2500.")
    parser.add_argument("--n-wind", type=int, nargs="+",
                        default=list(DEFAULT_N_WIND),
                        help="N for wind in each run. Default: 1000 1500 2000 2500. "
                             "Must match --n-solar length, or be length-1 (broadcast).")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_sweep(tuple(args.n_solar), tuple(args.n_wind), args.n_jobs, args.feature,
              strict_jobs=args.strict_jobs, resume=args.resume)


if __name__ == "__main__":
    main()