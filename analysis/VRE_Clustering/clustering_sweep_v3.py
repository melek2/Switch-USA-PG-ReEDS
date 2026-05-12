"""
clustering_sweep_v3.py
======================
Replicate PowerGenome's renewables_clusters logic locally for a methodology
test, WITHOUT running the full PG pipeline.

INPUT: a resources.yml file (output of generate_clustering_methodology_yamls.py)
       specifying per-region per-tech bin+cluster blocks.

OUTPUT: per_cluster_rmse_<scenario>.csv with one row per resulting cluster,
        same schema as clustering_sweep_v2.py outputs so existing plot scripts
        work unchanged. Adds: scenario, bin_label (for diagnostics).

The script faithfully mirrors PowerGenome's value_bin + cluster_sites_binned
flow for landbasedwind and utilitypv entries. Skips offshorewind and imports.

ARCHITECTURE
- Load resources.yml, extract renewables_clusters list
- For each (region, tech) entry:
    1. Pull CPA metadata for that (region, tech)
    2. Optionally merge centroid columns (x_m, y_m) from a centroid parquet
       if the YAML references them
    3. Apply each bin in sequence (capacity-weighted quantile bins via
       statsmodels DescrStatsW, matching PG's value_bin with weights)
    4. Group by all bin columns
    5. Within each group, run scipy ward linkage on profile vectors, cut at
       n_clusters (with PG's min(n, len) clamp)
    6. For each resulting cluster, compute profile RMSE + iccap std

USAGE
  # Single scenario:
  python clustering_sweep_v3.py --yaml pg/settings_methodology_iccap_q4/resources.yml \
                                 --scenario iccap_q4 --n-jobs 60 --strict-jobs 16
  # All five:
  for s in profile lat_lon_q4 iccap_q2 iccap_q3 iccap_q4; do
    python clustering_sweep_v3.py --yaml pg/settings_methodology_$s/resources.yml \
                                   --scenario $s --n-jobs 60 --strict-jobs 16
  done

ENV VARS (override defaults; defaults are repo-relative):
  PG_REPO_ROOT, PG_PROFILES_DIR, PG_RG_DIR, PG_OUT_DIR, PG_CENTROIDS_PARQUET
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

# Block BLAS threading to avoid oversubscribing parallel workers
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

_THIS = Path(__file__).resolve()
_REPO_FROM_SCRIPT = _THIS.parent.parent.parent
REPO_ROOT = Path(os.environ.get("PG_REPO_ROOT", _REPO_FROM_SCRIPT))
PROFILES_DIR = Path(os.environ.get(
    "PG_PROFILES_DIR", REPO_ROOT / "pg_data" / "profiles"
))
RG_DIR = Path(os.environ.get(
    "PG_RG_DIR",
    REPO_ROOT / "pg" / "resource_groups_10weeks_7days_PROFILE_CLUSTERS" / "ReEDS-cpas-patched",
))
OUT_BASE = Path(os.environ.get(
    "PG_OUT_DIR", REPO_ROOT / "analysis" / "VRE_Clustering" / "sweep_v3"
))
DEFAULT_CENTROIDS = Path(os.environ.get(
    "PG_CENTROIDS_PARQUET", REPO_ROOT / "pg" / "extra_inputs" / "cpa_centroids.parquet"
))

TECHS = {
    "utilitypv": {
        "metadata":  "solar_lcoe_resource_groups.parquet",
        "profiles":  "solar_rev_profiles_20240801.parquet",
        "site_map":  "solar_site_mapping_20240801.parquet",
        "shp_tech":  "utilitypv",
    },
    "landbasedwind": {
        "metadata":  "onshorewind_lcoe_resource_groups.parquet",
        "profiles":  "onshorewind_rev_profiles_20240801.parquet",
        "site_map":  "onshorewind_site_mapping_20240801.parquet",
        "shp_tech":  "landbasedwind",
    },
}

# Features that require merging x_m/y_m centroid columns into metadata
_GEO_FEATURES = {"x_m", "y_m", "lat", "lon"}


# ============================================================================
# YAML loading + entry filtering
# ============================================================================
def load_scenario(yml_path: Path) -> List[dict]:
    """Load renewables_clusters list from a resources.yml file."""
    with open(yml_path) as f:
        cfg = yaml.safe_load(f)
    rc = cfg.get("renewables_clusters", []) or []
    keep = []
    for e in rc:
        tech = e.get("technology")
        if tech not in TECHS:
            continue  # skip offshorewind, imports
        keep.append(e)
    return keep


def referenced_features(entries: List[dict]) -> set:
    """Collect every numeric column the YAML references via bin/cluster blocks."""
    feats = set()
    for e in entries:
        for b in e.get("bin") or []:
            f = b.get("feature")
            if f and f != "profile":
                feats.add(f)
        for c in e.get("cluster") or []:
            f = c.get("feature")
            if f and f != "profile":
                feats.add(f)
    return feats


# ============================================================================
# Capacity-weighted quantile binning (matches PG's value_bin with weights)
# ============================================================================
def weighted_qcut(values: np.ndarray, weights: np.ndarray, q: int) -> np.ndarray:
    """Capacity-weighted quantile bin assignment.

    Matches PG's value_bin() lines 75-89: when 'weights' is provided alongside
    'q', PG uses statsmodels.stats.weightstats.DescrStatsW to get the weighted
    quantile edges, then pd.cut to assign labels. We replicate that here.
    Returns integer bin labels in [0, q-1].
    """
    if q <= 1 or len(values) < 2:
        return np.zeros(len(values), dtype=int)
    try:
        from statsmodels.stats.weightstats import DescrStatsW
        probs = np.linspace(0, 1, q + 1)
        wq = DescrStatsW(data=values, weights=weights)
        edges = wq.quantile(probs=probs, return_pandas=False)
    except ImportError:
        # Fallback: unweighted qcut
        edges = np.quantile(values, np.linspace(0, 1, q + 1))
    # Unique edges so pd.cut doesn't choke on ties
    edges = np.unique(edges)
    if len(edges) < 2:
        return np.zeros(len(values), dtype=int)
    labels = pd.cut(values, bins=edges, include_lowest=True,
                    labels=False, duplicates="drop")
    return np.asarray(pd.Series(labels).fillna(0).astype(int))


# ============================================================================
# Process one (tech, region) entry: bin → cluster → score
# ============================================================================
def process_entry(entry: dict, tech: str, region: str,
                  metadata: pd.DataFrame, profiles_path: Path,
                  site_map: pd.Series,
                  subsample_cap: int | None = None) -> List[Dict]:
    """Replicate PG's bin+cluster pipeline for one (region, tech) entry.

    Returns list of per-cluster summary dicts.
    """
    from scipy.cluster.hierarchy import linkage, fcluster

    region_md = metadata[metadata["region"] == region].copy()
    if region_md.empty:
        return []

    # Map cpa_id → site name, drop CPAs without a site
    region_md["cpa_id"] = pd.to_numeric(region_md["cpa_id"], errors="coerce").astype("Int64")
    region_md = region_md.dropna(subset=["cpa_id"])
    cpa_to_site = site_map.reindex(region_md["cpa_id"]).dropna()
    region_md = region_md[region_md["cpa_id"].isin(cpa_to_site.index)].reset_index(drop=True)
    if region_md.empty:
        return []

    # Validate features the YAML requires
    bin_specs = entry.get("bin") or []
    for b in bin_specs:
        f = b["feature"]
        if f not in region_md.columns:
            raise ValueError(f"Bin feature '{f}' not in metadata for "
                             f"{tech}/{region}. Available: {list(region_md.columns)[:20]}")

    # Optional MW-weighted subsampling (for laptop runs on big regions)
    if subsample_cap and len(region_md) > subsample_cap:
        rng = np.random.default_rng(0)
        w = region_md["mw"].values.astype(np.float64)
        p = w / w.sum()
        idx = rng.choice(len(region_md), size=subsample_cap, replace=False, p=p)
        region_md = region_md.iloc[idx].reset_index(drop=True)

    # Load profile matrix for these CPAs
    sites_for_cpas = site_map.reindex(region_md["cpa_id"]).dropna()
    unique_sites = list(set(sites_for_cpas.tolist()))
    profiles_df = pq.read_table(profiles_path,
                                 columns=unique_sites).to_pandas()
    T = len(profiles_df)
    n = len(region_md)
    profile_matrix = np.zeros((n, T), dtype=np.float32)
    cpa_caps  = np.zeros(n, dtype=np.float32)
    cpa_iccap = np.full(n, np.nan, dtype=np.float32)
    site_lookup = sites_for_cpas.to_dict()
    has_iccap = "interconnect_capex_mw" in region_md.columns
    for i in range(n):
        cpa_id = region_md.iloc[i]["cpa_id"]
        s = site_lookup.get(cpa_id)
        if s and s in profiles_df.columns:
            profile_matrix[i] = profiles_df[s].to_numpy(dtype=np.float32)
            cpa_caps[i] = region_md.iloc[i]["mw"]
            if has_iccap:
                cpa_iccap[i] = region_md.iloc[i]["interconnect_capex_mw"]

    # Apply binning iteratively. Each bin block adds a bin label column.
    # PG's pattern: bins are applied in sequence; the GROUPED clustering
    # operates on the cartesian product of all bin labels.
    bin_label_arrs: List[np.ndarray] = []
    for b in bin_specs:
        f = b["feature"]
        q = int(b.get("q", 1))
        # Weighting: 'weights: capacity_mw' or 'mw'. We always weight by cpa_caps.
        vals = region_md[f].to_numpy(dtype=np.float64)
        # Mask NaN values: bin them as 0
        ok = ~np.isnan(vals)
        labels = np.zeros(len(vals), dtype=int)
        if ok.sum() >= 2:
            labels[ok] = weighted_qcut(vals[ok], cpa_caps[ok], q)
        bin_label_arrs.append(labels)

    # Build composite bin key (tuple of all bin labels per CPA)
    if bin_label_arrs:
        bin_keys = np.array(list(zip(*bin_label_arrs)))
        bin_strs = ["_".join(map(str, k)) for k in bin_keys]
    else:
        bin_strs = ["all"] * n

    # Cluster spec. Default is profile (full hourly CF), but also support
    # 1-D numeric features (e.g. 'cf', 'lcoe') in case the YAML uses them.
    # NOTE: the SCORING (per-cluster RMSE) is always against the full profile
    # centroid -- we're measuring "how well does each cluster's centroid
    # profile represent its member CPAs", regardless of what feature the
    # clustering optimized for.
    cluster_specs = entry.get("cluster") or []
    if not cluster_specs:
        return []
    csp = cluster_specs[0]
    cluster_feat = csp.get("feature", "profile")
    n_per_bin = int(csp.get("n_clusters", 1))

    if cluster_feat != "profile":
        if cluster_feat not in region_md.columns:
            raise ValueError(f"Cluster feature '{cluster_feat}' not in metadata for "
                             f"{tech}/{region}. Available: {list(region_md.columns)[:20]}")
        # 1-D feature vector aligned to the same row order as profile_matrix
        cluster_input_full = region_md[cluster_feat].to_numpy(dtype=np.float64)
    else:
        cluster_input_full = None  # use profile_matrix

    # For each bin, run ward linkage and cut
    rows = []
    global_cluster_id = 0
    unique_bins = sorted(set(bin_strs))
    for bin_label in unique_bins:
        mask = np.array([b == bin_label for b in bin_strs])
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        members_profile = profile_matrix[idx]
        members_caps = cpa_caps[idx]
        members_iccap = cpa_iccap[idx]

        # Clamp per-bin n_clusters to bin size (PG's min(n, len))
        n_actual = min(n_per_bin, len(idx))
        if n_actual < 1:
            continue
        if n_actual == 1 or len(idx) == 1:
            cluster_labels = np.ones(len(idx), dtype=int)
        else:
            if cluster_feat == "profile":
                cluster_input = members_profile.astype(np.float32)
            else:
                # 1-D feature: drop NaNs (assign them to a single fallback cluster)
                vals = cluster_input_full[idx]
                cluster_input = vals.reshape(-1, 1).astype(np.float32)
                # If any NaN, replace with mean (avoids linkage NaN errors)
                if np.isnan(cluster_input).any():
                    finite_mean = np.nanmean(cluster_input)
                    cluster_input = np.where(np.isnan(cluster_input),
                                              finite_mean, cluster_input)
            Z = linkage(cluster_input, method="ward")
            cluster_labels = fcluster(Z, t=n_actual, criterion="maxclust")

        # Score each cluster
        for cid in np.unique(cluster_labels):
            cmask = cluster_labels == cid
            mem_profile = members_profile[cmask]
            mem_caps    = members_caps[cmask]
            mem_iccap   = members_iccap[cmask]
            total_w = mem_caps.sum()
            if total_w == 0:
                continue
            centroid = (mem_profile * mem_caps[:, None]).sum(axis=0) / total_w
            diff = mem_profile - centroid[None, :]
            per_member_rmse = np.sqrt((diff * diff).mean(axis=1))

            # iccap stats
            if not np.isnan(mem_iccap).any():
                wt_mean_ic = float((mem_iccap * mem_caps).sum() / total_w)
                wt_var = float(((mem_iccap - wt_mean_ic) ** 2 * mem_caps).sum() / total_w)
                std_ic = math.sqrt(max(wt_var, 0.0))
                min_ic = float(mem_iccap.min())
                max_ic = float(mem_iccap.max())
            else:
                wt_mean_ic = std_ic = min_ic = max_ic = None

            rows.append({
                "tech":         tech,
                "region":       region,
                "cluster_feature": cluster_feat,
                "bin_label":    bin_label,
                "cluster":      global_cluster_id,
                "n_members":    int(cmask.sum()),
                "cluster_cap_mw": float(total_w),
                "centroid_mean_cf": float(centroid.mean()),
                "mean_member_rmse": float(per_member_rmse.mean()),
                "std_member_rmse":  float(per_member_rmse.std()),
                "max_member_rmse":  float(per_member_rmse.max()),
                "wt_mean_iccap_per_mw": wt_mean_ic,
                "std_iccap_per_mw":     std_ic,
                "min_iccap_per_mw":     min_ic,
                "max_iccap_per_mw":     max_ic,
                "N_total_for_region": n_per_bin * len(unique_bins),  # target
            })
            global_cluster_id += 1

    return rows


# ============================================================================
# Driver
# ============================================================================
def run_sweep(yml_path: Path, scenario_name: str, n_jobs: int,
              strict_jobs: int, resume: bool, subsample_cap: int | None):
    from joblib import Parallel, delayed

    print("=" * 64)
    print(f"clustering_sweep_v3 — scenario: {scenario_name}")
    print(f"  YAML:        {yml_path}")
    print(f"  n_jobs:      {n_jobs}  (strict floor: {strict_jobs})")
    print(f"  subsample:   {subsample_cap or 'none'}")
    print(f"  output dir:  {OUT_BASE}")
    print("=" * 64)

    if n_jobs < strict_jobs:
        print(f"\nERROR: n_jobs={n_jobs} below strict floor of {strict_jobs}.")
        sys.exit(2)

    out_dir = OUT_BASE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"per_cluster_rmse_{scenario_name}.csv"
    if not resume and out_csv.exists():
        print(f"Removing existing {out_csv}")
        out_csv.unlink()

    entries = load_scenario(yml_path)
    print(f"\nLoaded {len(entries)} renewables_clusters entries "
          f"(filtered to landbasedwind/utilitypv)")

    feats_used = referenced_features(entries)
    needs_geo = bool(feats_used & _GEO_FEATURES)
    print(f"Bin/cluster features referenced: {sorted(feats_used) or '(none)'}")
    if needs_geo:
        print(f"Geo features required; will merge centroids from {DEFAULT_CENTROIDS}")
        if not DEFAULT_CENTROIDS.exists():
            print(f"\nERROR: geo features required but centroid parquet missing: "
                  f"{DEFAULT_CENTROIDS}\nGenerate cpa_centroids.parquet first.")
            sys.exit(3)
        centroids = pd.read_parquet(DEFAULT_CENTROIDS)
        centroids["cpa_id"] = pd.to_numeric(centroids["cpa_id"], errors="coerce").astype("Int64")
    else:
        centroids = None

    # Load tech metadata once. Optionally merge in centroids.
    print(f"\nLoading metadata...")
    metadata = {}
    site_maps = {}
    for tech, cfg in TECHS.items():
        md = pd.read_parquet(RG_DIR / cfg["metadata"])
        if "mw" not in md.columns:
            md = md.rename(columns={"capacity_mw": "mw"})
        # PG accepts 'capacity_mw' as a weight alias; expose both columns
        if "capacity_mw" not in md.columns:
            md["capacity_mw"] = md["mw"]
        md["cpa_id"] = pd.to_numeric(md["cpa_id"], errors="coerce").astype("Int64")
        if needs_geo and centroids is not None:
            sub = centroids[centroids["tech"] == cfg["shp_tech"]][
                ["cpa_id", "x_m", "y_m", "lat", "lon"]
            ]
            md = md.merge(sub, on="cpa_id", how="left")
        sm = pd.read_parquet(PROFILES_DIR / cfg["site_map"]).set_index("CPA_ID")["Site"]
        metadata[tech] = md
        site_maps[tech] = sm
        print(f"  {tech:14s} {len(md):>7,} CPAs, "
              f"{md['region'].nunique():>2} regions, "
              f"{md['mw'].sum()/1e3:>8,.1f} GW total")

    # Resume support
    done_keys = set()
    if resume and out_csv.exists():
        prev = pd.read_csv(out_csv)
        done_keys = set(zip(prev["tech"], prev["region"]))
        print(f"\nResume: {len(done_keys)} (tech, region) tasks already done")

    # Build task list
    tasks = []
    for e in entries:
        tech = e["technology"]
        region = e["region"]
        if region == "all":
            # region:all entries aren't supported in this methodology test
            print(f"  (skipping region:all entry for {tech})")
            continue
        if (tech, region) in done_keys:
            continue
        tasks.append((e, tech, region))

    # Sort by CPA count desc so big jobs start first
    cpa_counts = {(t, r): int((metadata[t]["region"] == r).sum())
                  for (_, t, r) in tasks}
    tasks.sort(key=lambda x: -cpa_counts.get((x[1], x[2]), 0))

    print(f"\nDispatching {len(tasks)} (tech, region) jobs to {n_jobs} workers")
    print(f"Largest 5 regions:")
    for e, t, r in tasks[:5]:
        bin_summary = ""
        for b in e.get("bin") or []:
            bin_summary += f" bin({b['feature']},q={b.get('q')})"
        ncl = e.get("cluster", [{}])[0].get("n_clusters", "?")
        print(f"  {t:14s} {r:35s} {cpa_counts[(t,r)]:>6} CPAs"
              f"  n_per_bin={ncl}{bin_summary}")

    def _wrap(entry, tech, region):
        try:
            t0 = time.time()
            rows = process_entry(
                entry, tech, region,
                metadata[tech],
                PROFILES_DIR / TECHS[tech]["profiles"],
                site_maps[tech],
                subsample_cap=subsample_cap,
            )
            elapsed = time.time() - t0
            return {"ok": True, "tech": tech, "region": region,
                    "n_cpas": cpa_counts.get((tech, region), 0),
                    "elapsed": elapsed, "rows": rows}
        except Exception:
            return {"ok": False, "tech": tech, "region": region,
                    "tb": traceback.format_exc()}

    write_header = not (resume and out_csv.exists())
    n_done = n_err = 0
    with Parallel(n_jobs=n_jobs, return_as="generator", verbose=10) as parallel:
        for res in parallel(delayed(_wrap)(e, t, r) for (e, t, r) in tasks):
            if res["ok"]:
                df = pd.DataFrame(res["rows"])
                df["scenario"] = scenario_name
                df.to_csv(out_csv, mode="a", header=write_header, index=False)
                write_header = False
                n_done += 1
                print(f"  [{n_done}/{len(tasks)}] {res['tech']:14s} {res['region']:35s} "
                      f"{res['n_cpas']:>6} CPAs in {res['elapsed']:>6.1f}s "
                      f"-> {len(res['rows']):>4} cluster rows")
            else:
                n_err += 1
                print(f"  ERROR in {res['tech']}/{res['region']}:\n{res['tb']}")

    print(f"\nDone. {n_done} succeeded, {n_err} errored.")
    print(f"Output: {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", required=True, type=Path,
                        help="Path to resources.yml for this scenario")
    parser.add_argument("--scenario", required=True,
                        help="Scenario name (used as CSV suffix and 'scenario' column)")
    parser.add_argument("--n-jobs", type=int, required=True,
                        help="Parallel workers")
    parser.add_argument("--strict-jobs", type=int, default=1,
                        help="Abort if n_jobs < this. Use 16+ on HPC.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--subsample-cap", type=int, default=None,
                        help="MW-weighted subsample threshold (laptop testing)")
    args = parser.parse_args()
    run_sweep(args.yaml, args.scenario, args.n_jobs, args.strict_jobs,
              args.resume, args.subsample_cap)


if __name__ == "__main__":
    main()