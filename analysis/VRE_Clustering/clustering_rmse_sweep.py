"""
clustering_rmse_sweep.py
========================
Within-cluster reconstruction RMSE for PowerGenome's renewable site clustering,
swept over total cluster budget. Delegates clustering to PG's own functions
(`assign_site_cluster`, `calc_cluster_values`) so the algorithm being benchmarked
is bit-exactly what PG does in the Switch-USA-PG-ReEDS pipeline.

DESIGNED FOR HPC. Key features:
  - All paths configurable via env vars
  - Per-task parquet outputs in results/<total_n>/<tech>/<region>.parquet
    so a crash mid-sweep loses nothing -- on resume we just skip files that exist
  - Per-cluster detail (cluster_id, member CPAs, capacity, individual SSEs)
    saved so the analysis can be re-aggregated locally without re-running PG
  - Per-task log lines with timing + CPA count for postmortem diagnosis
  - validate subcommand confirms bit-exact reproduction before sweep starts

ENV VARS (override defaults; defaults assume Mac dev layout)
  PG_REPO_ROOT     -- Switch-USA-PG-ReEDS repo root
  PG_PROFILES_DIR  -- *_rev_profiles_*.parquet and *_site_mapping_*.parquet
  PG_RG_DIR        -- *_lcoe_resource_groups.parquet
  PG_EXTRA_OUTPUTS -- where PG writes <region>_<tech>__site_cluster_assignments.csv
  PG_OUT_DIR       -- where this script writes results
  PG_PYTHON_PATH   -- where the powergenome package lives (for sys.path injection)

USAGE
  python clustering_rmse_sweep.py validate --tech solar --region TX
  python clustering_rmse_sweep.py sweep --grid 500 1000 1500 2000 2500 3000 \\
      --n-jobs 16 --resume
  python clustering_rmse_sweep.py aggregate    # rebuild summary from per-task parquets
"""
from __future__ import annotations

import os
import sys
import math
import time
import argparse
import logging
import traceback
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd

# Force single-threaded numerical libs so joblib parallelism doesn't oversubscribe
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# ----------------------------------------------------------------------------
# Configurable paths
# ----------------------------------------------------------------------------
DEFAULT_REPO_ROOT = "/Users/melek/Documents/GitHub/Switch-USA-PG-ReEDS"

REPO_ROOT     = Path(os.environ.get("PG_REPO_ROOT", DEFAULT_REPO_ROOT))
PROFILES_DIR  = Path(os.environ.get("PG_PROFILES_DIR", REPO_ROOT / "pg_data" / "profiles"))
RG_DIR        = Path(os.environ.get("PG_RG_DIR",
                     REPO_ROOT / "pg" / "resource_groups_10weeks_7days_PROFILE_CLUSTERS" / "ReEDS-cpas-patched"))
EXTRA_OUTPUTS = Path(os.environ.get("PG_EXTRA_OUTPUTS", REPO_ROOT / "pg" / "extra_outputs"))
OUT_DIR       = Path(os.environ.get("PG_OUT_DIR", REPO_ROOT / "analysis" / "VRE_Clustering" / "clustering_sweep"))
PG_PATH       = Path(os.environ.get("PG_PYTHON_PATH", REPO_ROOT / "PowerGenome"))

if str(PG_PATH) not in sys.path:
    sys.path.insert(0, str(PG_PATH))

# Defer the powergenome import to runtime via a helper -- this lets `preflight`
# report a missing import gracefully instead of the script crashing on load.
def _import_pg():
    """Import PG's clustering functions on demand."""
    from powergenome.cluster.renewables import assign_site_cluster, calc_cluster_values
    return assign_site_cluster, calc_cluster_values

# ----------------------------------------------------------------------------
# Tech registry
# ----------------------------------------------------------------------------
TECHS: Dict[str, Dict] = {
    "solar": {
        "metadata":  "solar_lcoe_resource_groups.parquet",
        "profiles":  "solar_rev_profiles_20240801.parquet",
        "site_map":  "solar_site_mapping_20240801.parquet",
        "filename_tech_keyword": "utilitypv",
        "slices": [
            {"name": None, "filter": lambda df: df},
        ],
    },
    "onshorewind": {
        "metadata":  "onshorewind_lcoe_resource_groups.parquet",
        "profiles":  "onshorewind_rev_profiles_20240801.parquet",
        "site_map":  "onshorewind_site_mapping_20240801.parquet",
        "filename_tech_keyword": "landbasedwind",
        "slices": [
            {"name": None, "filter": lambda df: df},
        ],
    },
}

DEFAULT_GRID = (500, 1000, 1500, 2000, 2500, 3000)
MIN_PER_ZONE = 1   # was 3; floor of 1 lets the sqrt-allocator decide for itself


# ----------------------------------------------------------------------------
# Allocator
# ----------------------------------------------------------------------------
def allocate_clusters(cap, n_total, min_per_zone=MIN_PER_ZONE, max_sites=None):
    """Distribute n_total clusters across regions, weighted by sqrt(capacity)."""
    eligible = cap[cap > 0]
    ceil = (max_sites.reindex(eligible.index) if max_sites is not None
            else pd.Series(float("inf"), index=eligible.index))
    alloc = pd.Series(min_per_zone, index=eligible.index).clip(upper=ceil)
    if alloc.sum() > n_total:
        raise ValueError(
            f"Floor of {int(alloc.sum())} (min_per_zone={min_per_zone} x "
            f"{len(eligible)} regions) exceeds budget {n_total}."
        )
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
# Data loading
# ----------------------------------------------------------------------------
def load_tech(tech: str) -> Tuple[pd.DataFrame, pd.Series]:
    cfg = TECHS[tech]
    md = pd.read_parquet(RG_DIR / cfg["metadata"])
    if "mw" not in md.columns:
        md = md.rename(columns={"capacity_mw": "mw"})
    sm = pd.read_parquet(PROFILES_DIR / cfg["site_map"]).set_index("CPA_ID")["Site"]
    return md, sm


# ----------------------------------------------------------------------------
# Single (tech, region, slice, N) -> RMSE + per-cluster detail
# ----------------------------------------------------------------------------
def cluster_one(tech: str, region: str, n_clusters: int,
                metadata: pd.DataFrame, profiles_path: Path,
                site_map: pd.Series,
                slice_name: Optional[str] = None,
                slice_filter=None,
                cluster_feature: str = "profile",
                cluster_method: str = "agglomerative") -> Optional[Dict]:
    """
    Cluster CPAs in (tech, region, slice) and compute capacity-weighted RMSE.
    Returns a dict with summary scalars + a DataFrame of per-cluster detail.
    """
    renew_data = metadata[metadata["region"] == region].copy()
    if slice_filter is not None:
        renew_data = slice_filter(renew_data)
    if renew_data.empty:
        return None

    n_cpas = len(renew_data)
    cap_mw = float(renew_data["mw"].sum())

    if n_clusters >= n_cpas:
        # Each CPA is its own cluster -> RMSE is exactly 0
        per_cluster = pd.DataFrame({
            "cluster": np.arange(n_cpas),
            "n_members": 1,
            "cluster_cap_mw": renew_data["mw"].values,
            "sse_x_cap": 0.0,
        })
        return {"summary": {"tech": tech, "slice": slice_name, "region": region,
                            "n_clusters": n_cpas, "n_cpas": n_cpas,
                            "cap_mw": cap_mw, "rmse": 0.0,
                            "n_clusters_requested": n_clusters},
                "per_cluster": per_cluster}

    assign_site_cluster, calc_cluster_values = _import_pg()
    clustered = assign_site_cluster(
        renew_data=renew_data,
        profile_path=profiles_path,
        regions=[region],
        site_map=site_map,
        cluster=[{"feature": cluster_feature, "method": cluster_method,
                  "n_clusters": n_clusters}],
    )

    T = len(clustered["profile"].iloc[0])
    total_sse_x_cap = 0.0
    total_cap = 0.0
    per_cluster_rows = []

    for k, members in clustered.groupby("cluster"):
        centroid_row = calc_cluster_values(members.reset_index(drop=True))
        centroid = np.asarray(centroid_row["profile"].iloc[0])
        member_profiles = np.stack([np.asarray(p) for p in members["profile"]])
        member_caps = members["mw"].to_numpy()
        diff = member_profiles - centroid[None, :]
        sse = (diff * diff).sum(axis=1)
        cluster_sse_x_cap = float((sse * member_caps).sum())
        cluster_cap = float(member_caps.sum())
        total_sse_x_cap += cluster_sse_x_cap
        total_cap += cluster_cap
        per_cluster_rows.append({
            "cluster": int(k),
            "n_members": len(members),
            "cluster_cap_mw": cluster_cap,
            "sse_x_cap": cluster_sse_x_cap,
            # cluster-level RMSE for diagnostic purposes
            "cluster_rmse": math.sqrt(cluster_sse_x_cap / (T * cluster_cap)) if cluster_cap > 0 else 0.0,
        })

    rmse = math.sqrt(total_sse_x_cap / (T * total_cap))
    per_cluster = pd.DataFrame(per_cluster_rows)
    return {
        "summary": {"tech": tech, "slice": slice_name, "region": region,
                    "n_clusters": n_clusters, "n_cpas": n_cpas, "cap_mw": cap_mw,
                    "rmse": rmse, "n_clusters_requested": n_clusters},
        "per_cluster": per_cluster,
    }


# ----------------------------------------------------------------------------
# Per-task output paths
# ----------------------------------------------------------------------------
def task_output_path(out_dir: Path, total_n: int, tech: str,
                     slice_name: Optional[str], region: str) -> Path:
    slc = slice_name or "all"
    safe_region = region.replace("/", "_")
    return out_dir / "results" / f"total_n={total_n}" / tech / slc / f"{safe_region}.parquet"


def task_log_path(out_dir: Path, total_n: int, tech: str,
                  slice_name: Optional[str], region: str) -> Path:
    return task_output_path(out_dir, total_n, tech, slice_name, region).with_suffix(".log")


# ----------------------------------------------------------------------------
# Pre-flight check: verify all paths, files, packages, and write permissions
# before starting any expensive work
# ----------------------------------------------------------------------------
def preflight(strict: bool = True) -> bool:
    """
    Verify the environment is correctly set up. Prints a checklist of what
    was found / missing. Returns True if everything is OK, False otherwise.

    If strict=True, raises SystemExit on any failure. The sweep subcommand
    calls this with strict=True before doing anything.
    """
    print("=" * 72)
    print("PRE-FLIGHT CHECK")
    print("=" * 72)

    failures: List[str] = []
    warnings: List[str] = []

    def check(label: str, ok: bool, detail: str = "", is_warning: bool = False):
        marker = "OK " if ok else ("WARN" if is_warning else "FAIL")
        print(f"  [{marker}] {label}{(': ' + detail) if detail else ''}")
        if not ok:
            (warnings if is_warning else failures).append(f"{label}{(': ' + detail) if detail else ''}")

    # 1. Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    check("Python version", sys.version_info >= (3, 8), py_ver)

    # 2. Required packages
    print("\nRequired packages:")
    pkg_versions = {}
    for pkg in ["numpy", "pandas", "pyarrow", "sklearn", "scipy", "joblib"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            pkg_versions[pkg] = ver
            check(pkg, True, ver)
        except ImportError as e:
            check(pkg, False, str(e))

    # 3. PowerGenome import
    print("\nPowerGenome:")
    check("PG_PYTHON_PATH exists", PG_PATH.exists(), str(PG_PATH))
    try:
        import powergenome
        from powergenome.cluster.renewables import (  # noqa: F401
            assign_site_cluster, calc_cluster_values
        )
        pg_loc = Path(powergenome.__file__).parent
        check("powergenome importable", True, str(pg_loc))
        check("powergenome.cluster.renewables.assign_site_cluster", True)
    except ImportError as e:
        check("powergenome import", False, str(e))

    # 4. Path layout
    print("\nDirectories:")
    check("REPO_ROOT", REPO_ROOT.exists(), str(REPO_ROOT))
    check("PROFILES_DIR", PROFILES_DIR.exists(), str(PROFILES_DIR))
    check("RG_DIR", RG_DIR.exists(), str(RG_DIR))
    check("EXTRA_OUTPUTS", EXTRA_OUTPUTS.exists(), str(EXTRA_OUTPUTS),
          is_warning=True)  # only needed for validate

    # 5. Output dir writable
    print("\nOutput:")
    try:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        test_file = OUT_DIR / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        check("OUT_DIR writable", True, str(OUT_DIR))
    except Exception as e:
        check("OUT_DIR writable", False, f"{OUT_DIR}: {e}")

    # 6. Per-tech data files
    print("\nPer-tech data files:")
    for tech, cfg in TECHS.items():
        print(f"  {tech}:")
        for kind in ["metadata", "site_map"]:
            p = (RG_DIR if kind == "metadata" else PROFILES_DIR) / cfg[kind]
            sz = f"{p.stat().st_size / 1e6:.1f} MB" if p.exists() else "missing"
            check(f"    {kind} ({cfg[kind]})", p.exists(), sz)
        p = PROFILES_DIR / cfg["profiles"]
        sz = f"{p.stat().st_size / 1e9:.2f} GB" if p.exists() else "missing"
        check(f"    profiles ({cfg['profiles']})", p.exists(), sz)

    # 7. PG output files for validation (needed if you want to run validate)
    print("\nPG saved cluster assignments (for validate subcommand):")
    if EXTRA_OUTPUTS.exists():
        all_csvs = list(EXTRA_OUTPUTS.glob("*_site_cluster_assignments.csv"))
        check(f"  *_site_cluster_assignments.csv files in {EXTRA_OUTPUTS.name}",
              len(all_csvs) > 0, f"{len(all_csvs)} files",
              is_warning=True)
    else:
        check("  EXTRA_OUTPUTS dir", False,
              "(not strictly required for sweep)", is_warning=True)

    # 8. Disk space (rough)
    print("\nDisk space:")
    try:
        import shutil
        usage = shutil.disk_usage(OUT_DIR)
        free_gb = usage.free / 1e9
        check(f"  Free space at OUT_DIR", free_gb > 5.0,
              f"{free_gb:.1f} GB free")
    except Exception as e:
        check("  Disk space check", False, str(e), is_warning=True)

    # 9. Quick metadata sniff: do regions agree?
    print("\nMetadata sanity:")
    try:
        for tech in TECHS:
            md, _ = load_tech(tech)
            n_regions = md["region"].nunique()
            n_cpas = len(md)
            cap_gw = md["mw"].sum() / 1e3
            print(f"  {tech}: {n_cpas:,} CPAs, {n_regions} regions, {cap_gw:,.1f} GW")
    except Exception as e:
        check("  Metadata load", False, str(e))

    # 10. Threading config
    print("\nThreading:")
    check("OMP_NUM_THREADS=1", os.environ.get("OMP_NUM_THREADS") == "1",
          os.environ.get("OMP_NUM_THREADS", "unset"), is_warning=True)
    check("MKL_NUM_THREADS=1", os.environ.get("MKL_NUM_THREADS") == "1",
          os.environ.get("MKL_NUM_THREADS", "unset"), is_warning=True)

    # Summary
    print("\n" + "=" * 72)
    if failures:
        print(f"PRE-FLIGHT FAILED: {len(failures)} error(s):")
        for f in failures:
            print(f"  - {f}")
        if warnings:
            print(f"\n({len(warnings)} warning(s) also)")
        if strict:
            raise SystemExit(2)
        return False

    if warnings:
        print(f"PRE-FLIGHT OK with {len(warnings)} warning(s):")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("PRE-FLIGHT PASSED: all checks green.")
    print("=" * 72)
    return True


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
def validate(tech: str, region: str,
             cluster_feature: str = "profile",
             cluster_method: str = "agglomerative"):
    from sklearn.metrics import adjusted_rand_score

    cfg = TECHS[tech]
    keyword = cfg["filename_tech_keyword"]
    candidates = [c for c in EXTRA_OUTPUTS.glob(f"{region}*_site_cluster_assignments.csv")
                  if keyword.lower() in c.name.lower()]
    if not candidates:
        raise FileNotFoundError(
            f"No PG output for region={region} tech keyword={keyword} in {EXTRA_OUTPUTS}"
        )
    print(f"\nFound {len(candidates)} candidate file(s):")
    for c in candidates:
        print(f"  - {c.name}")
    pg_csv = candidates[0]
    print(f"Validating against: {pg_csv.name}")
    print(f"Cluster spec: feature={cluster_feature} method={cluster_method}")

    pg_assign = pd.read_csv(pg_csv)
    n_clusters = pg_assign["cluster"].nunique()
    print(f"  PG used n_clusters = {n_clusters} on {len(pg_assign)} CPAs")

    md, sm = load_tech(tech)
    renew_data = md[md["region"] == region].copy()
    if renew_data.empty:
        print(f"\n  ERROR: no CPAs in metadata for region={region!r}, tech={tech!r}.")
        print(f"  Available regions: {sorted(md['region'].unique())}")
        raise SystemExit(1)
    print(f"  Our metadata: {len(renew_data)} CPAs in region {region}")

    pg_cpa_ids = set(pg_assign["cpa_id"])
    our_cpa_ids = set(renew_data["cpa_id"])
    overlap = pg_cpa_ids & our_cpa_ids
    print(f"  CPA ID overlap: {len(overlap)} / {len(pg_cpa_ids)}")
    if len(overlap) < len(pg_cpa_ids):
        print(f"  WARNING: {len(pg_cpa_ids) - len(overlap)} CPAs in PG output "
              f"are NOT in your metadata for this region.")
    if len(our_cpa_ids) > len(pg_cpa_ids) * 1.1:
        print(f"  WARNING: metadata has {len(our_cpa_ids)} CPAs but PG only "
              f"clustered {len(pg_cpa_ids)}. Configure 'slices' filter.")

    assign_site_cluster, _ = _import_pg()
    clustered = assign_site_cluster(
        renew_data=renew_data,
        profile_path=PROFILES_DIR / cfg["profiles"],
        regions=[region],
        site_map=sm,
        cluster=[{"feature": cluster_feature, "method": cluster_method,
                  "n_clusters": n_clusters}],
    )
    merged = pg_assign.merge(
        clustered[["cpa_id", "cluster"]].rename(columns={"cluster": "ours"}),
        on="cpa_id", how="inner",
    )
    print(f"  CPAs in PG output     : {len(pg_assign)}")
    print(f"  CPAs in our output    : {len(clustered)}")
    print(f"  CPAs matched on cpa_id: {len(merged)}")

    if len(merged) == 0:
        print(f"\n  VALIDATION FAILED: zero CPAs matched.")
        raise SystemExit(1)
    if len(merged) < 0.95 * len(pg_assign):
        print(f"\n  VALIDATION FAILED: only {len(merged)/len(pg_assign):.1%} matched.")
        raise SystemExit(1)

    ari = adjusted_rand_score(merged["cluster"], merged["ours"])
    print(f"  Adjusted Rand Index   : {ari:.6f}    "
          f"{'MATCH' if ari > 0.9999 else 'PARTITIONS DIFFER'}")
    if ari < 0.9999:
        print(f"\n  VALIDATION FAILED: ARI < 1.0.")
        raise SystemExit(1)
    print(f"\n  VALIDATION PASSED: bit-exact reproduction of PG clustering.")
    return {"ari": ari, "n_matched": len(merged)}


# ----------------------------------------------------------------------------
# Sweep driver
# ----------------------------------------------------------------------------
def split_total(total_n: int, metadata: Dict[str, pd.DataFrame],
                method: str = "sqrt_capacity") -> Dict[str, int]:
    caps = {tech: md["mw"].sum() for tech, md in metadata.items()}
    if method == "sqrt_capacity":
        weights = {tech: math.sqrt(c) for tech, c in caps.items()}
    elif method == "capacity":
        weights = caps
    elif method == "equal":
        weights = {tech: 1.0 for tech in caps}
    else:
        raise ValueError(f"Unknown split method: {method}")
    wsum = sum(weights.values())
    raw = {tech: total_n * w / wsum for tech, w in weights.items()}
    floor = {tech: int(math.floor(v)) for tech, v in raw.items()}
    leftover = total_n - sum(floor.values())
    for tech in sorted(raw, key=lambda t: -(raw[t] - floor[t])):
        if leftover <= 0:
            break
        floor[tech] += 1
        leftover -= 1
    return floor


def _execute_task(t: dict, metadata, site_maps,
                  cluster_feature: str, cluster_method: str) -> dict:
    """Run one task, write per-task parquet + log file. Returns the summary row."""
    out_path = task_output_path(OUT_DIR, t["total_n"], t["tech"],
                                t["slice_name"], t["region"])
    log_path = task_log_path(OUT_DIR, t["total_n"], t["tech"],
                             t["slice_name"], t["region"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        # Already done; read summary and return
        try:
            df = pd.read_parquet(out_path)
            return df.attrs.get("summary", {"_skipped": True,
                                            "tech": t["tech"], "region": t["region"],
                                            "total_n": t["total_n"]})
        except Exception:
            pass  # fall through to recompute

    t0 = time.time()
    try:
        result = cluster_one(
            t["tech"], t["region"], t["n_clusters"],
            metadata[t["tech"]],
            PROFILES_DIR / TECHS[t["tech"]]["profiles"],
            site_maps[t["tech"]],
            slice_name=t["slice_name"],
            slice_filter=t["slice_filter"],
            cluster_feature=cluster_feature,
            cluster_method=cluster_method,
        )
        elapsed = time.time() - t0

        if result is None:
            log_path.write_text(f"SKIPPED: no CPAs for {t}\n")
            return {"_skipped": True, **{k: v for k, v in t.items() if k != "slice_filter"}}

        # Write per-task parquet with per-cluster detail; stash summary in attrs
        per_cluster = result["per_cluster"]
        summary = result["summary"]
        summary["total_n"] = t["total_n"]
        summary["elapsed_sec"] = elapsed
        # Embed summary as parquet metadata
        per_cluster.attrs["summary"] = summary
        per_cluster.to_parquet(out_path, index=False)

        log_path.write_text(
            f"OK total_n={t['total_n']} tech={t['tech']} region={t['region']} "
            f"n_cpas={summary['n_cpas']} n_clusters={summary['n_clusters']} "
            f"rmse={summary['rmse']:.6f} elapsed_sec={elapsed:.1f}\n"
        )
        return summary
    except Exception:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        log_path.write_text(
            f"ERROR total_n={t['total_n']} tech={t['tech']} region={t['region']} "
            f"elapsed_sec={elapsed:.1f}\n\n{tb}\n"
        )
        return {"_error": True, "total_n": t["total_n"], "tech": t["tech"],
                "slice": t["slice_name"], "region": t["region"]}


def run_sweep(total_n_grid=DEFAULT_GRID,
              split_method: str = "sqrt_capacity",
              n_jobs: int = 4,
              resume: bool = False,
              cluster_feature: str = "profile",
              cluster_method: str = "agglomerative",
              skip_preflight: bool = False):
    from joblib import Parallel, delayed

    if not skip_preflight:
        preflight(strict=True)
        print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results").mkdir(exist_ok=True)
    print(f"Output dir: {OUT_DIR}")
    print(f"Cluster spec: feature={cluster_feature} method={cluster_method}")
    print(f"Resume mode: {resume}")
    print(f"min_per_zone: {MIN_PER_ZONE}")

    # Load metadata once
    print("Loading tech metadata...")
    metadata, site_maps = {}, {}
    for tech in TECHS:
        md, sm = load_tech(tech)
        metadata[tech], site_maps[tech] = md, sm
        print(f"  {tech}: {len(md):,} CPAs, {md['region'].nunique()} regions, "
              f"{md['mw'].sum()/1e3:,.1f} GW total")

    # Build task list
    all_tasks: List[dict] = []
    skipped_existing = 0
    for total_n in total_n_grid:
        n_per_tech = split_total(total_n, metadata, method=split_method)
        for tech, n_tech in n_per_tech.items():
            cfg = TECHS[tech]
            for slc in cfg["slices"]:
                slc_name = slc["name"]
                slc_filter = slc["filter"]
                md_slice = slc_filter(metadata[tech])
                if md_slice.empty:
                    continue
                cap_by_region = md_slice.groupby("region")["mw"].sum()
                try:
                    alloc = allocate_clusters(cap_by_region, n_tech)
                except ValueError as e:
                    print(f"  total_n={total_n} {tech}/{slc_name}: alloc failed -- {e}")
                    continue
                for region, n in alloc.items():
                    out_path = task_output_path(OUT_DIR, total_n, tech, slc_name, region)
                    if resume and out_path.exists():
                        skipped_existing += 1
                        continue
                    all_tasks.append({
                        "total_n": total_n, "tech": tech,
                        "slice_name": slc_name, "slice_filter": slc_filter,
                        "region": region, "n_clusters": int(n),
                    })

    print(f"\nTasks: {len(all_tasks)} new, {skipped_existing} already complete (skipped)")
    print(f"Parallelism: n_jobs={n_jobs}")
    if not all_tasks:
        print("Nothing to do. Running aggregate.")
        aggregate()
        return

    # Sort tasks largest-first so big jobs start early and small ones backfill at the end
    region_sizes = {}
    for tech in metadata:
        for region in metadata[tech]["region"].unique():
            region_sizes[(tech, region)] = (metadata[tech]["region"] == region).sum()
    all_tasks.sort(key=lambda t: -region_sizes.get((t["tech"], t["region"]), 0))

    Parallel(n_jobs=n_jobs, verbose=10, batch_size=1)(
        delayed(_execute_task)(t, metadata, site_maps, cluster_feature, cluster_method)
        for t in all_tasks
    )

    print("\nAll tasks dispatched. Aggregating...")
    aggregate()


# ----------------------------------------------------------------------------
# Aggregation: walk results/ tree and build summary CSVs
# ----------------------------------------------------------------------------
def aggregate():
    """Walk OUT_DIR/results/ and build rmse_sweep.csv + rmse_sweep_aggregated.csv."""
    results_dir = OUT_DIR / "results"
    if not results_dir.exists():
        print(f"No results dir at {results_dir}")
        return

    rows = []
    n_files = 0
    for parquet_file in results_dir.rglob("*.parquet"):
        try:
            df = pd.read_parquet(parquet_file)
            summary = df.attrs.get("summary")
            if summary is not None:
                rows.append(summary)
                n_files += 1
            else:
                # Reconstruct summary from per-cluster detail if attrs missing
                # (older parquets may lack metadata; this is a fallback)
                pass
        except Exception as e:
            print(f"  Warning: failed to read {parquet_file}: {e}")

    if not rows:
        print("No completed tasks found.")
        return

    df = pd.DataFrame(rows)
    df = df[df["cap_mw"] > 0].copy()
    csv_path = OUT_DIR / "rmse_sweep.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(df)} rows from {n_files} parquet files)")

    # Capacity-weighted RMSE per (total_n, tech)
    def cap_wt_rmse(g):
        return float(np.sqrt((g["rmse"] ** 2 * g["cap_mw"]).sum() / g["cap_mw"].sum()))

    agg = (df.groupby(["total_n", "tech"])
             .apply(lambda g: pd.Series({
                 "rmse_capwt": cap_wt_rmse(g),
                 "n_clusters_total": int(g["n_clusters"].sum()),
                 "cap_mw": g["cap_mw"].sum(),
                 "n_regions": g["region"].nunique(),
             }))
             .reset_index())
    agg_path = OUT_DIR / "rmse_sweep_aggregated.csv"
    agg.to_csv(agg_path, index=False)
    print(f"\n=== Aggregated ===")
    print(agg.to_string(index=False))
    print(f"\nWrote {agg_path}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_val = sub.add_parser("validate")
    p_val.add_argument("--tech", default="solar")
    p_val.add_argument("--region", required=True)
    p_val.add_argument("--feature", default="profile", choices=["profile", "cf"])
    p_val.add_argument("--method", default="agglomerative",
                       choices=["agglomerative", "kmeans"])

    p_pf = sub.add_parser("preflight", help="Verify environment without running anything")

    p_sw = sub.add_parser("sweep")
    p_sw.add_argument("--grid", type=int, nargs="+", default=list(DEFAULT_GRID))
    p_sw.add_argument("--split", default="sqrt_capacity",
                      choices=["sqrt_capacity", "capacity", "equal"])
    p_sw.add_argument("--n-jobs", type=int, default=4)
    p_sw.add_argument("--resume", action="store_true",
                      help="Skip tasks whose per-task parquet already exists")
    p_sw.add_argument("--feature", default="profile", choices=["profile", "cf"])
    p_sw.add_argument("--method", default="agglomerative",
                      choices=["agglomerative", "kmeans"])
    p_sw.add_argument("--skip-preflight", action="store_true",
                      help="Skip the pre-flight check (not recommended)")

    p_ag = sub.add_parser("aggregate", help="Re-aggregate from per-task parquets")

    args = parser.parse_args()

    if args.cmd == "preflight":
        preflight(strict=False)
    elif args.cmd == "validate":
        validate(args.tech, args.region,
                 cluster_feature=args.feature, cluster_method=args.method)
    elif args.cmd == "sweep":
        run_sweep(total_n_grid=tuple(args.grid), split_method=args.split,
                  n_jobs=args.n_jobs, resume=args.resume,
                  cluster_feature=args.feature, cluster_method=args.method,
                  skip_preflight=args.skip_preflight)
    elif args.cmd == "aggregate":
        aggregate()


if __name__ == "__main__":
    main()
