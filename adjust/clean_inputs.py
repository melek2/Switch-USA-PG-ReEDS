"""
Post-processing cleanup for PowerGenome -> Switch inputs.

Runs as part of `model_adjustment_scripts` (or standalone). Passes:

* Enrich gen_info: merges extra columns (n_clusters, utility_id_eia,
  plant_id_eia, state, latitude, longitude) onto gen_info.csv from
  existing_gen_units.csv and PUDL's plants_entity_eia table, then writes
  the rows with no existing-unit match out to candidate_sites.csv.

* Geothermal: fills missing per-zone capacity caps for
  Geothermal_HydroBinary_Moderate candidates in gen_info.csv and
  candidate_sites.csv from the ReEDS-2.0 geohydro supply curve (static t=0
  no-new-exploration baseline). Drops geothermal rows in zones with zero
  effective capacity.

* Biomass: builds regional_fuel_markets.csv, zone_to_regional_fuel_market.csv,
  and fuel_supply_curves.csv for the `waste_biomass` fuel from ReEDS
  bio_supplycurve.csv + hierarchy.csv. Apportions each USDA region's supply
  across overlapping load zones by BA count.

* CCS energy load: adds gen_ccs_energy_load to gen_info.csv /
  candidate_sites.csv (0 for CCS-equipped rows, '.' otherwise) so Switch
  can construct ZoneTotalCentralDispatch. Placeholder; real ATB parasitic
  loads belong upstream.

* Manual coordinate patch: fills missing latitude / longitude in gen_info.csv
  and candidate_sites.csv for a small set of plant_id_eia values whose
  coordinates are absent from PUDL.  Sources documented in MANUAL_COORDS.

* Coordinate error fix: overwrites known-bad coordinates (and optionally
  gen_load_zone) for plant_id_eia values in COORD_OVERRIDES where PUDL has
  a coordinate but it is factually wrong (e.g. sign-flipped longitude, or
  plant placed at a completely different facility).  Also renames
  GENERATION_PROJECT entries whose zone prefix becomes stale after a zone
  reassignment.  Sources documented in COORD_OVERRIDES.

* Hydro min-flow clipping: clips hydro_min_flow_mw in hydro_timeseries.csv
  to (total predetermined capacity x gen_availability) for any project
  where aggregated min-flow exceeds the feasible dispatch ceiling. Prevents
  infeasibility between Enforce_Hydro_Min_Flow and the dispatch upper limit
  caused by EIA-vs-ReEDS aggregation mismatches. Does not clip avg_flow —
  SpillHydro absorbs avg-flow overages.

CLI:
    python adjust/clean_inputs.py <out_folder> --settings <path>
    python adjust/clean_inputs.py <out_folder> --settings <path> --reeds-inputs <path>

    If --reeds-inputs is omitted, the script looks for bundled data at
    adjust/adjust_data/reeds_data/ (relative to this script).
Missing coordinates:
   plant_id_eia 675 28.0797, -81.9228   https://www.gem.wiki/Larsen_Memorial_power_station
   plant_id_eia 955 41.371729 -89.12833 https://www.gem.wiki/Larsen_Memorial_power_station
   plant_id_eia 56610 44.396150, -96.535018 https://www.google.com/maps/place/Deer+Creek+Station,+Basin+Electric+Power+Cooperative/@44.3949283,-96.5353042,1169m/data=!3m1!1e3!4m6!3m5!1s0x87898f052065472d:0x81f311bfa7c67702!8m2!3d44.3945351!4d-96.5261797!16s%2Fg%2F11css6tvrg?hl=en-US&entry=ttu&g_ep=EgoyMDI2MDUyMC4wIKXMDSoASAFQAw%3D%3D
   plant_id_eia 6120 48.88564, -122.75164 https://www.gridinfo.com/plant/whitehorn/6120
   plant_id_eia 14840 41.371729 -89.12833 https://www.interconnection.fyi/eia/project/68751-17

Overwritten coordinates (coordinates are actually wrong)
    plant_id_eia 50973 29.888462 -93.95098 (LA_and_e_TX_natural_gas_fired_combined_cycle_12-16, LA_and_e_TX_natural_gas_fired_combustion_turbine_24) https://www.interconnection.fyi/eia/project/50973-gn42
    plant_id_eia 65768 42.2747 -71.7172 (MA_natural_gas_internal_combustion_engine_23, renamed from LAX via GEN_PROJECT_RENAMES) https://www.interconnection.fyi/eia/project/6125-4

Settings YAML usage (out_folder is prepended by model_adjustment_scripts):
    Clean inputs:
        script: adjust/clean_inputs.py
        args: "--settings ../../pg/settings_46regions"
        order: 4

    Note: relative paths in `args` are resolved from the out_folder's parent
    (the CWD that model_adjustment_scripts uses when invoking the script),
    which is typically switch/in_XX/<case>/<year>/.
"""

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

import pandas as pd

from powergenome.util import load_settings

logger = logging.getLogger("clean_inputs")
logging.basicConfig(level=logging.INFO, format="%(message)s")


# ---------------------------------------------------------------------------
# Enrich gen_info (must run BEFORE everything else so downstream passes
# see the new columns and the rewritten candidate_sites.csv)
# ---------------------------------------------------------------------------

def _most_frequent_per_resource(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """For each Resource, pick the most frequent value of `col`.
    Ties broken by smallest value of `col`."""
    valid = df.dropna(subset=[col]).copy()
    if valid.empty:
        return pd.DataFrame(columns=["Resource", col])

    counts = (
        valid.groupby(["Resource", col])
             .size()
             .reset_index(name="count")
    )
    counts = counts.sort_values(["Resource", "count", col],
                                ascending=[True, True, True])
    idx = counts.groupby("Resource")["count"].idxmax()
    chosen = counts.loc[idx, ["Resource", col]].reset_index(drop=True)

    multi = counts.groupby("Resource")[col].nunique()
    multi = multi[multi > 1]
    for res, n_vals in multi.items():
        chosen_val = chosen.loc[chosen["Resource"] == res, col].iloc[0]
        freq = int(
            counts[(counts["Resource"] == res) & (counts[col] == chosen_val)]
            ["count"].iloc[0]
        )
        logger.info(
            f"[enrich] Resource {res}: {n_vals} unique {col} values found; "
            f"selected {chosen_val!r} with frequency {freq}."
        )

    return chosen


import shutil

def enrich_gen_info(out_folder: Path, pudl_db_path: Path, extra_outputs: Path | None = None) -> None:
    """Add n_clusters, utility_id_eia, plant_id_eia, state, latitude,
    longitude to gen_info.csv (overwriting it), and write rows with no
    existing-unit match out to candidate_sites.csv."""
    gen_info_path = out_folder / "gen_info.csv"
    existing_path = out_folder / "existing_gen_units.csv"
    candidates_path = out_folder / "candidate_sites.csv"

    if not gen_info_path.exists():
        logger.warning(f"[enrich] skipping: {gen_info_path} not found.")
        return

    # If existing_gen_units.csv isn't in the case folder, try to copy it
    # from PowerGenome's extra_outputs directory.
    if not existing_path.exists():
        if extra_outputs is None:
            logger.warning(
                f"[enrich] skipping: {existing_path.name} not found and "
                f"no extra_outputs path provided."
            )
            return
        src = extra_outputs / "existing_gen_units.csv"
        if not src.exists():
            logger.warning(
                f"[enrich] skipping: {existing_path.name} not in case folder "
                f"and not found at {src}."
            )
            return
        shutil.copy(src, existing_path)
        logger.info(f"[enrich] copied {src} -> {existing_path}")

    if not pudl_db_path.exists():
        logger.warning(f"[enrich] skipping: PUDL db {pudl_db_path} not found.")
        return

    gen_info = pd.read_csv(gen_info_path)
    existing = pd.read_csv(existing_path, low_memory=False)

    # Make re-runs safe: drop any columns a previous enrich pass added,
    # so the merges below don't collide into plant_id_eia_x / plant_id_eia_y.
    ENRICH_COLS = ["n_clusters", "utility_id_eia", "plant_id_eia",
                   "state", "latitude", "longitude"]
    gen_info = gen_info.drop(columns=[c for c in ENRICH_COLS if c in gen_info.columns])

    existing["utility_id_eia"] = pd.to_numeric(existing["utility_id_eia"], errors="coerce")
    existing["plant_id_eia"] = pd.to_numeric(existing["plant_id_eia"], errors="coerce")

    cluster_count_map = (
        existing.dropna(subset=["cluster"])
                .groupby("Resource")["cluster"]
                .nunique()
                .reset_index(name="n_clusters")
    )
    utility_map = _most_frequent_per_resource(existing, "utility_id_eia")
    plant_map = _most_frequent_per_resource(existing, "plant_id_eia")

    with sqlite3.connect(pudl_db_path) as con:
        plants = pd.read_sql(
            "SELECT plant_id_eia, state, latitude, longitude FROM plants_entity_eia",
            con,
        )

    enriched = (
        gen_info
        .merge(cluster_count_map, left_on="GENERATION_PROJECT", right_on="Resource", how="left")
        .drop(columns=["Resource"])
        .merge(utility_map,       left_on="GENERATION_PROJECT", right_on="Resource", how="left")
        .drop(columns=["Resource"])
        .merge(plant_map,         left_on="GENERATION_PROJECT", right_on="Resource", how="left")
        .drop(columns=["Resource"])
        .merge(plants, on="plant_id_eia", how="left")
    )

    candidates = enriched[enriched["n_clusters"].isna()].reset_index(drop=True)

    enriched.to_csv(gen_info_path, index=False, na_rep=".")
    candidates.to_csv(candidates_path, index=False, na_rep=".")

    n_matched = int(enriched["n_clusters"].notna().sum())
    logger.info(
        f"[enrich] wrote {gen_info_path.name} ({n_matched}/{len(enriched)} matched "
        f"to existing units) and {candidates_path.name} ({len(candidates)} candidates)."
    )


# ---------------------------------------------------------------------------
# Manual coordinate patch
# ---------------------------------------------------------------------------
#
# A small number of plants have valid plant_id_eia values but no coordinates
# in the PUDL plants_entity_eia table (or the upstream data was null).  Rather
# than patching PUDL, we keep a versioned override dict here.  Sources are
# recorded as comments for reproducibility.
#
# Keys   : plant_id_eia (int)
# Values : (latitude, longitude)

MANUAL_COORDS: dict[int, tuple[float, float]] = {
    675:   (28.079700,  -81.922800),   # https://www.gem.wiki/Larsen_Memorial_power_station
    955:   (41.371729,  -89.128330),   # https://www.gem.wiki/Larsen_Memorial_power_station
    56610: (44.396150,  -96.535018),   # https://www.google.com/maps/place/Deer+Creek+Station
    6120:  (48.885640, -122.751640),   # https://www.gridinfo.com/plant/whitehorn/6120
    14840: (41.371729,  -89.12833),   # https://www.interconnection.fyi/eia/project/68751-17
}

COORD_TARGET_FILES = (
    "gen_info.csv",
    "candidate_sites.csv",
)


def patch_missing_coordinates(out_folder: Path) -> None:
    """Fill missing latitude / longitude in gen_info.csv and candidate_sites.csv
    for plant_id_eia values listed in MANUAL_COORDS.

    Only rows where *both* latitude and longitude are NaN / '.' are updated;
    rows that already have coordinates are left untouched.  All rows sharing
    a given plant_id_eia (there may be several cluster rows) are patched.
    """
    for fname in COORD_TARGET_FILES:
        path = out_folder / fname
        if not path.exists():
            continue

        df = pd.read_csv(path, dtype=str, keep_default_na=False)

        if "plant_id_eia" not in df.columns:
            logger.info(f"[coords] {fname}: no plant_id_eia column, skipping.")
            continue
        if "latitude" not in df.columns or "longitude" not in df.columns:
            logger.info(f"[coords] {fname}: latitude/longitude columns missing, skipping.")
            continue

        # Normalise: treat '.' and '' as missing for the coordinate columns.
        def _is_missing(series: pd.Series) -> pd.Series:
            return series.str.strip().eq(".") | series.str.strip().eq("")

        n_patched = 0
        for plant_id, (lat, lon) in MANUAL_COORDS.items():
            plant_mask = pd.to_numeric(df["plant_id_eia"], errors="coerce") == plant_id
            coord_missing = _is_missing(df["latitude"]) | _is_missing(df["longitude"])
            mask = plant_mask & coord_missing
            n = int(mask.sum())
            if n == 0:
                if plant_mask.any():
                    logger.info(
                        f"[coords] {fname}: plant_id_eia {plant_id} – "
                        f"coordinates already present, skipping."
                    )
                continue
            df.loc[mask, "latitude"]  = str(lat)
            df.loc[mask, "longitude"] = str(lon)
            n_patched += n
            logger.info(
                f"[coords] {fname}: plant_id_eia {plant_id} – "
                f"filled {n} row(s) with ({lat}, {lon})."
            )

        if n_patched == 0:
            logger.info(f"[coords] {fname}: no missing coordinates to patch.")
            continue

        df.to_csv(path, index=False)
        logger.info(f"[coords] {fname}: wrote {n_patched} coordinate patch(es).")


# ---------------------------------------------------------------------------
# Coordinate and zone overrides (wrong coordinates in upstream EIA/PUDL data)
# ---------------------------------------------------------------------------
#
# Distinct from MANUAL_COORDS (which fills *missing* coordinates): these
# entries overwrite coordinates that exist in PUDL but are factually wrong,
# and optionally reassign gen_load_zone when the plant belongs to a different
# region than its GENERATION_PROJECT name implies.
#
# Keys   : plant_id_eia (int)
# Values : (latitude, longitude, new_gen_load_zone_or_None)
#          new_gen_load_zone=None means keep existing zone assignment.
#
# Sources:
#   50973 – coords landed at -76.4 lon (Atlantic Ocean); correct location is
#            Cameron Parish, LA.  https://www.interconnection.fyi/eia/project/50973-gn42
#   65768 – coords landed in MA but plant was assigned to LAX region; correct
#            location and zone is MA.  https://www.interconnection.fyi/eia/project/6125-4

COORD_OVERRIDES: dict[int, tuple[float, float, str | None]] = {
    50973: (29.888462, -93.950980, None),  # LA_and_e_TX CC/CT; lon was ~-76 (Atlantic)
    65768: (42.274700, -71.717200, "MA"),  # assigned to LAX but plant is in MA
}

# ---------------------------------------------------------------------------
# Generator project renames
# ---------------------------------------------------------------------------
#
# fix_coordinate_errors renames GENERATION_PROJECT in gen_info.csv and
# candidate_sites.csv when a zone reassignment changes the zone prefix, but
# the new name must also propagate to every other CSV that references the
# generator (gen_build_costs.csv, gen_build_predetermined.csv,
# gen_emission_costs.csv, gen_retirement_costs.csv, gen_storage_info.csv,
# hydro_timeseries.csv, etc.).
#
# GEN_PROJECT_RENAMES maps old_name -> new_name.  fix_gen_project_renames
# does a literal text replace across all *.csv files in the case folder so
# no file is left with a stale generator name.
#
# Keys are derived from COORD_OVERRIDES zone reassignments; update both
# together when adding a new plant override.
#
# Sources:
#   LAX -> MA for plant_id_eia 65768: plant is physically in MA; zone prefix
#   was wrong.  https://www.interconnection.fyi/eia/project/6125-4

GEN_PROJECT_RENAMES: dict[str, str] = {
    "LAX_natural_gas_internal_combustion_engine_23":
        "MA_natural_gas_internal_combustion_engine_23",
}


def fix_coordinate_errors(out_folder: Path) -> None:
    """Overwrite known-bad coordinates (and optionally gen_load_zone) in
    gen_info.csv and candidate_sites.csv for plant_id_eia values listed in
    COORD_OVERRIDES.  Unlike patch_missing_coordinates, this always overwrites
    regardless of whether coordinates are already present.

    Also renames GENERATION_PROJECT entries whose zone prefix no longer matches
    after a zone reassignment (e.g. LAX_... -> MA_... when zone changes to MA).
    """
    for fname in COORD_TARGET_FILES:
        path = out_folder / fname
        if not path.exists():
            continue

        df = pd.read_csv(path, dtype=str, keep_default_na=False)

        if "plant_id_eia" not in df.columns:
            logger.info(f"[coord-fix] {fname}: no plant_id_eia column, skipping.")
            continue
        if "latitude" not in df.columns or "longitude" not in df.columns:
            logger.info(f"[coord-fix] {fname}: latitude/longitude columns missing, skipping.")
            continue

        eid = pd.to_numeric(df["plant_id_eia"], errors="coerce")
        n_patched_total = 0

        for plant_id, (lat, lon, new_zone) in COORD_OVERRIDES.items():
            mask = eid == plant_id
            n = int(mask.sum())
            if n == 0:
                continue

            df.loc[mask, "latitude"]  = str(lat)
            df.loc[mask, "longitude"] = str(lon)
            zone_msg = "zone unchanged"

            if new_zone is not None and "gen_load_zone" in df.columns:
                old_zones = df.loc[mask, "gen_load_zone"].unique().tolist()
                df.loc[mask, "gen_load_zone"] = new_zone
                zone_msg = f"zone {old_zones} -> {new_zone}"

                # Rename GENERATION_PROJECT prefix if it starts with an old zone
                if "GENERATION_PROJECT" in df.columns:
                    for old_zone in old_zones:
                        prefix_mask = (
                            mask
                            & df["GENERATION_PROJECT"].str.startswith(old_zone + "_")
                        )
                        if prefix_mask.any():
                            df.loc[prefix_mask, "GENERATION_PROJECT"] = (
                                df.loc[prefix_mask, "GENERATION_PROJECT"]
                                .str.replace(f"^{old_zone}_", f"{new_zone}_", regex=True)
                            )
                            n_renamed = int(prefix_mask.sum())
                            logger.info(
                                f"[coord-fix] {fname}: plant_id_eia {plant_id} – "
                                f"renamed {n_renamed} GENERATION_PROJECT(s) "
                                f"{old_zone}_... -> {new_zone}_..."
                            )

            logger.info(
                f"[coord-fix] {fname}: plant_id_eia {plant_id} – "
                f"overwrote coords for {n} row(s) with ({lat}, {lon}), {zone_msg}."
            )
            n_patched_total += n

        if n_patched_total == 0:
            logger.info(f"[coord-fix] {fname}: no coordinate overrides to apply.")
            continue

        df.to_csv(path, index=False)
        logger.info(f"[coord-fix] {fname}: wrote {n_patched_total} override(s).")


def fix_gen_project_renames(out_folder: Path) -> None:
    """Propagate GENERATION_PROJECT renames from GEN_PROJECT_RENAMES across
    all *.csv files in the case folder.

    fix_coordinate_errors renames GENERATION_PROJECT in gen_info.csv and
    candidate_sites.csv, but the new name must also reach every other file
    that references the generator (gen_build_costs.csv, gen_emission_costs.csv,
    gen_build_predetermined.csv, gen_retirement_costs.csv, hydro_timeseries.csv,
    etc.).  This function does a literal text replace so no file is missed.
    """
    if not GEN_PROJECT_RENAMES:
        return

    csvs = list(out_folder.glob("*.csv"))
    # Also catch cerf sited outputs and other subdirectory CSVs.
    csvs += [p for p in out_folder.rglob("**/*.csv") if p not in csvs]

    for path in csvs:
        try:
            text = path.read_text()
        except Exception as e:
            logger.warning(f"[gen-rename] could not read {path.name}: {e}")
            continue

        total = 0
        for old, new in GEN_PROJECT_RENAMES.items():
            n = text.count(old)
            if n:
                text = text.replace(old, new)
                total += n

        if total == 0:
            continue

        path.write_text(text)
        logger.info(
            f"[gen-rename] {path.relative_to(out_folder)}: "
            f"replaced {total} generator name(s)."
        )


# ---------------------------------------------------------------------------
# Fuel string fixup (must run BEFORE clean_geothermal so archives inherit it)
# ---------------------------------------------------------------------------
BIO_TARGET_FILES = (
    "gen_info.csv",
    "gen_info_archive.csv",
    "candidate_sites.csv",
    "candidate_sites_archive.csv",
)
BIO_OLD = ",biopower_dedicated_moderate,"
BIO_NEW = ",waste_biomass,"

# graph_tech_types.csv: energy_source is the last column, so the token appears
# at end-of-line (no trailing comma). Only replace there — leave gen_tech
# (`Biopower_Dedicated_Moderate`) untouched.
GRAPH_TECH_FILE = "graph_tech_types.csv"
GRAPH_OLD = ",biopower_dedicated_moderate\n"
GRAPH_NEW = ",waste_biomass\n"


def clean_biopower_fuel_string(out_folder: Path) -> None:
    """Replace the placeholder fuel token `biopower_dedicated_moderate` with
    `waste_biomass` in gen_info / candidate_sites / graph_tech_types CSVs.
    Literal text replace (not pandas) to avoid touching unrelated columns."""
    for fname in BIO_TARGET_FILES:
        path = out_folder / fname
        if not path.exists():
            continue
        text = path.read_text()
        n = text.count(BIO_OLD)
        if n == 0:
            logger.info(f"[biopower-fuel] {fname}: no matches.")
            continue
        path.write_text(text.replace(BIO_OLD, BIO_NEW))
        logger.info(f"[biopower-fuel] {fname}: replaced {n} occurrence(s).")

    graph_path = out_folder / GRAPH_TECH_FILE
    if graph_path.exists():
        text = graph_path.read_text()
        added_newline = not text.endswith("\n")
        if added_newline:
            text += "\n"
        n = text.count(GRAPH_OLD)
        if n == 0:
            logger.info(f"[biopower-fuel] {GRAPH_TECH_FILE}: no matches.")
        else:
            text = text.replace(GRAPH_OLD, GRAPH_NEW)
            if added_newline:
                text = text.rstrip("\n") + "\n"
            graph_path.write_text(text)
            logger.info(
                f"[biopower-fuel] {GRAPH_TECH_FILE}: replaced {n} energy_source value(s)."
            )


BIO_GEN_TECH = "Biopower_Dedicated_Moderate"
BIO_HEAT_RATE_COL = "gen_full_load_heat_rate"
BIO_HEAT_RATE_DEFAULT = "13.5"


def fix_biopower_heat_rate(out_folder: Path) -> None:
    """For rows with gen_tech == Biopower_Dedicated_Moderate, fill missing
    heat rates ('.') with BIO_HEAT_RATE_DEFAULT."""
    for fname in BIO_TARGET_FILES:
        path = out_folder / fname
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        if "gen_tech" not in df.columns or BIO_HEAT_RATE_COL not in df.columns:
            logger.info(f"[biopower-hr] {fname}: required columns missing, skipping.")
            continue
        mask = (df["gen_tech"] == BIO_GEN_TECH) & (df[BIO_HEAT_RATE_COL] == ".")
        n = int(mask.sum())
        if n == 0:
            logger.info(f"[biopower-hr] {fname}: no '.' heat rates to fill.")
            continue
        df.loc[mask, BIO_HEAT_RATE_COL] = BIO_HEAT_RATE_DEFAULT
        df.to_csv(path, index=False)
        logger.info(
            f"[biopower-hr] {fname}: filled {n} {BIO_GEN_TECH} "
            f"heat rate(s) with {BIO_HEAT_RATE_DEFAULT}."
        )


# ---------------------------------------------------------------------------
# CCS energy load column
# ---------------------------------------------------------------------------
#
# PowerGenome doesn't always emit a gen_ccs_energy_load column, but Switch's
# dispatch module requires a value for every generator with a non-'.'
# gen_ccs_capture_efficiency (the CCS_EQUIPPED_GENS set is built from that
# column, and gen_ccs_energy_load has no default, so a missing row crashes
# construction of ZoneTotalCentralDispatch).
#
# Add/fill the column: 0 for CCS-equipped rows, '.' otherwise. Zero is a
# placeholder — it understates parasitic load and overstates net generation
# per unit fuel. Real ATB 2024 values are ~0.21 MWh/MWh for coal IGCC + 90%
# CCS and ~0.15 MWh/MWh for NGCC + CCS. Fine for unblocking solves; revisit
# before taking health-damage results seriously.

CCS_CAP_COL = "gen_ccs_capture_efficiency"
CCS_LOAD_COL = "gen_ccs_energy_load"
CCS_TARGET_FILES = (
    "gen_info.csv",
    "gen_info_archive.csv",
    "candidate_sites.csv",
    "candidate_sites_archive.csv",
)


def fix_ccs_energy_load(out_folder: Path) -> None:
    """Ensure gen_ccs_energy_load exists in gen_info / candidate_sites.
    Set to 0 for CCS-equipped rows (gen_ccs_capture_efficiency != '.'),
    '.' otherwise."""
    for fname in CCS_TARGET_FILES:
        path = out_folder / fname
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        if CCS_CAP_COL not in df.columns:
            logger.info(f"[ccs-load] {fname}: no {CCS_CAP_COL} column, skipping.")
            continue

        is_ccs = df[CCS_CAP_COL].str.strip().ne(".") & df[CCS_CAP_COL].str.strip().ne("")
        new_vals = pd.Series(".", index=df.index)
        new_vals[is_ccs] = "0"

        if CCS_LOAD_COL in df.columns:
            # Only fill '.'/empty for CCS rows; leave any existing numeric
            # values untouched.
            existing = df[CCS_LOAD_COL].str.strip()
            missing = is_ccs & (existing.eq(".") | existing.eq(""))
            n = int(missing.sum())
            if n == 0:
                logger.info(f"[ccs-load] {fname}: column present, no CCS gaps to fill.")
                continue
            df.loc[missing, CCS_LOAD_COL] = "0"
            df.to_csv(path, index=False)
            logger.info(f"[ccs-load] {fname}: filled {n} missing CCS value(s) with 0.")
        else:
            # Insert next to gen_ccs_capture_efficiency for readability.
            df[CCS_LOAD_COL] = new_vals
            cols = list(df.columns)
            cols.insert(cols.index(CCS_CAP_COL) + 1, cols.pop(cols.index(CCS_LOAD_COL)))
            df = df[cols]
            df.to_csv(path, index=False)
            logger.info(
                f"[ccs-load] {fname}: added {CCS_LOAD_COL} column "
                f"({int(is_ccs.sum())} CCS rows set to 0)."
            )


# ---------------------------------------------------------------------------
# Geothermal
# ---------------------------------------------------------------------------

GEO_TECH = "Geothermal_HydroBinary_Moderate"
GEO_CAP_COL = "gen_capacity_limit_mw"
GEO_SUFFIX = "_geothermal_hydrobinary_moderate_0"


def _geothermal_zone_caps(reeds_dir: Path, region_aggregations: dict) -> pd.Series:
    """Per-zone effective geothermal cap (MW), aggregated p-region -> zone."""
    rsc = pd.read_csv(reeds_dir / "geothermal" / "geo_rsc_ATB_2023.csv")
    disc = (
        pd.read_csv(reeds_dir / "geothermal" / "geo_discovery_factor_ATB_2023.csv")
        .rename(columns={"value": "disc"})
    )

    cap = (
        rsc[rsc.sc_cat == "cap"][["*i", "r", "value"]]
        .rename(columns={"value": "cap_mw_gross"})
        .merge(disc, on=["*i", "r"], how="left")
    )
    cap["disc"] = cap["disc"].fillna(1.0)
    cap["cap_mw_effective"] = cap["cap_mw_gross"] * cap["disc"]
    cap[["tech", "rclass"]] = cap["*i"].str.extract(r"^(.+)_(\d+)$")

    p_caps = cap[cap.tech == "geohydro_allkm"].copy()

    p_to_zone = {
        p: zone
        for zone, plist in region_aggregations.items()
        for p in plist
    }
    p_caps["zone"] = p_caps["r"].map(p_to_zone)

    unmapped = p_caps[p_caps.zone.isna() & (p_caps.cap_mw_effective > 0)]
    if len(unmapped):
        total = unmapped.cap_mw_effective.sum()
        logger.warning(
            f"[geothermal] {unmapped['r'].nunique()} p-regions with non-zero "
            f"cap not in region_aggregations ({total:.0f} MW dropped)."
        )

    return (
        p_caps.dropna(subset=["zone"])
        .groupby("zone")["cap_mw_effective"]
        .sum()
        .round(1)
    )


def _archive_once(df: pd.DataFrame, path: Path) -> None:
    if path.exists():
        logger.debug(f"[geothermal] archive already exists, skipping: {path.name}")
        return
    df.to_csv(path, index=False)


def clean_geothermal(out_folder: Path, settings: dict, reeds_dir: Path) -> None:
    gen_info_path = out_folder / "gen_info.csv"
    candidates_path = out_folder / "candidate_sites.csv"
    existing_path = out_folder / "existing_gen_units.csv"

    if not gen_info_path.exists() or not candidates_path.exists():
        logger.warning(
            f"[geothermal] skipping {out_folder.name}: "
            "gen_info.csv or candidate_sites.csv not found."
        )
        return

    zone_caps = _geothermal_zone_caps(reeds_dir, settings["region_aggregations"])

    nz = zone_caps[zone_caps > 0].sort_values(ascending=False)
    logger.info(
        f"[geothermal] cap by zone (MW), {len(nz)} non-zero of "
        f"{len(settings['region_aggregations'])}:\n{nz.to_string()}"
    )

    gen_info = pd.read_csv(gen_info_path)
    candidates = pd.read_csv(candidates_path)

    if existing_path.exists():
        existing_resources = set(
            pd.read_csv(existing_path, low_memory=False, usecols=["Resource"])["Resource"]
            .dropna()
            .unique()
        )
    else:
        logger.warning(
            f"[geothermal] {existing_path.name} not found; "
            "treating ALL gen_info rows as candidates."
        )
        existing_resources = set()

    _archive_once(gen_info, out_folder / "gen_info_archive.csv")
    _archive_once(candidates, out_folder / "candidate_sites_archive.csv")

    def extract_zone(project_name):
        if isinstance(project_name, str) and project_name.lower().endswith(GEO_SUFFIX):
            return project_name[: -len(GEO_SUFFIX)]
        return None

    dropped_gens: set[str] = set()

    def update(df: pd.DataFrame, label: str, candidates_only: bool) -> pd.DataFrame:
        df = df.copy()
        is_geo = df["gen_tech"] == GEO_TECH

        if candidates_only:
            is_candidate = ~df["GENERATION_PROJECT"].isin(existing_resources)
            target = is_geo & is_candidate
        else:
            target = is_geo

        zones = df.loc[target, "GENERATION_PROJECT"].apply(extract_zone)
        df.loc[target, GEO_CAP_COL] = zones.map(zone_caps).fillna(0).values

        cap_numeric = pd.to_numeric(df[GEO_CAP_COL], errors="coerce").fillna(0)
        drop_mask = target & (cap_numeric <= 0)

        dropped_gens.update(df.loc[drop_mask, "GENERATION_PROJECT"])

        kept = is_geo.sum() - drop_mask.sum()
        logger.info(
            f"[geothermal] {label}: kept {kept} / {is_geo.sum()} "
            f"geothermal rows, dropped {drop_mask.sum()} infeasible candidate(s)."
        )
        return df[~drop_mask].reset_index(drop=True)

    update(gen_info, "gen_info", candidates_only=True).to_csv(gen_info_path, index=False, na_rep=".")
    update(candidates, "candidate_sites", candidates_only=False).to_csv(candidates_path, index=False, na_rep=".")

    # Prune the same orphan gens from downstream files keyed on GENERATION_PROJECT.
    # Without this, Pyomo loads (gen, build_year) tuples for gens we just removed
    # and fails the GEN_BLD_YRS validation rule.
    for fname in ["gen_build_costs.csv", "gen_build_predetermined.csv"]:
        path = out_folder / fname
        if not path.exists() or not dropped_gens:
            continue
        df = pd.read_csv(path)
        orphan_mask = df["GENERATION_PROJECT"].isin(dropped_gens)
        n = int(orphan_mask.sum())
        if n == 0:
            continue
        _archive_once(df, out_folder / f"{path.stem}_archive.csv")
        df[~orphan_mask].to_csv(path, index=False, na_rep=".")
        logger.info(f"[geothermal] {fname}: dropped {n} orphan row(s).")


# ---------------------------------------------------------------------------
# Biomass
# ---------------------------------------------------------------------------

BIO_FUEL_NAME = "waste_biomass"
BIO_ENERGY_CONTENT = 13.0
BIO_SW_SUPPLY = 1.0
BIO_SW_TRANSPORT_USD_PER_TON = 0.0
BIO_USD2018_PER_USD2024 = 0.80 # https://www.bls.gov/data/inflation_calculator.htm
BIO_SPLIT_METHOD = "by_ba"
# Census-division assignment for each of the 46 model zones, matching the
# AEO fuel-region groupings in fuel_cost.csv. Used to look up coal prices in
# the ReEDS coal_AEO_2025_reference.csv (columns are census divisions).
ZONE_TO_CENSUS_DIV = {
    # Pacific
    "Bay_area": "Pacific", "LAX": "Pacific",
    "San_Diego": "Pacific", "WA_and_OR": "Pacific",
    # Mountain
    "AZ": "Mountain", "CO": "Mountain", "ID_and_w_MT_and_w_WY": "Mountain",
    "NM_and_w_TX": "Mountain", "NV_and_UT_and_n_CA": "Mountain",
    "e_WY_and_w_SD": "Mountain",
    # New England
    "CT_and_RT": "New_England", "MA": "New_England",
    "ME": "New_England", "VT_and_NH": "New_England",
    # South Atlantic
    "Central_FL": "South_Atlantic", "GA": "South_Atlantic",
    "MD_and_VA_Island": "South_Atlantic", "NC": "South_Atlantic",
    "SC": "South_Atlantic", "VA": "South_Atlantic",
    "WV": "South_Atlantic", "se_FL": "South_Atlantic",
    # Mid-Atlantic
    "DE_and_NJ": "Mid_Atlantic", "NY": "Mid_Atlantic",
    "New_York_City": "Mid_Atlantic", "PA": "Mid_Atlantic",
    # West South Central
    "Houston_ERCOT": "West_South_Central", "LA_and_e_TX": "West_South_Central",
    "OK_and_e_NM_and_nw_TX": "West_South_Central",
    "e_Texas_ERCOT": "West_South_Central", "w_TX_ERCOT": "West_South_Central",
    # East South Central
    "AL_and_e_MS_and_FL_pnh": "East_South_Central",
    "TN_and_s_KY": "East_South_Central",
    "e_AR_and_w_MS": "East_South_Central", "n_KY": "East_South_Central",
    # West North Central
    "IA": "West_North_Central", "KS_and_w_MO": "West_North_Central",
    "MN_and_e_ND": "West_North_Central",
    "NE_and_e_SD_and_w_ND_and_e_MT": "West_North_Central",
    "e_MO": "West_North_Central",
    # East North Central
    "Chicago_and_sw_MI": "East_North_Central",
    "IL_no_Chicago": "East_North_Central",
    "IN_and_W_KY": "East_North_Central", "MI": "East_North_Central",
    "OH": "East_North_Central", "WI_and_w_MI": "East_North_Central",
}

def clean_biomass(out_folder: Path, settings: dict, reeds_dir: Path) -> None:
    bio_csv = reeds_dir / "supply_curve" / "bio_supplycurve.csv"
    hier_csv = reeds_dir / "hierarchy.csv"
    if not bio_csv.exists() or not hier_csv.exists():
        logger.warning(
            f"[biomass] skipping: ReEDS bio_supplycurve.csv or hierarchy.csv "
            f"not found under {reeds_dir}."
        )
        return

    zone_to_bas = settings["region_aggregations"]
    periods_path = out_folder / "periods.csv"
    if not periods_path.exists():
        raise FileNotFoundError(
            f"[biomass] periods.csv not found in {out_folder}. "
            "clean_biomass must run after pg_to_switch.py has written "
            "Switch inputs."
        )
    periods = pd.read_csv(periods_path)["INVESTMENT_PERIOD"].tolist()

    hier = pd.read_csv(hier_csv)[["ba", "usda_region"]]
    bio = pd.read_csv(bio_csv)

    bio["cap"] = bio["cap"] * 1e6 * BIO_ENERGY_CONTENT * BIO_SW_SUPPLY
    bio["price"] = (
        (bio["price"] + BIO_SW_TRANSPORT_USD_PER_TON)
        / BIO_ENERGY_CONTENT
        / BIO_USD2018_PER_USD2024
    )

    rows = []
    for lz, bas in zone_to_bas.items():
        sub = hier[hier["ba"].isin(bas)]
        missing = set(bas) - set(sub["ba"])
        if missing:
            raise ValueError(f"[biomass] BAs not in hierarchy.csv for {lz}: {missing}")
        for usda, g in sub.groupby("usda_region"):
            rows.append({"load_zone": lz, "usda_region": usda, "n_bas": len(g)})
    zone_usda = pd.DataFrame(rows)

    if BIO_SPLIT_METHOD == "equal":
        zone_usda["weight"] = 1.0
    elif BIO_SPLIT_METHOD == "by_ba":
        zone_usda["weight"] = zone_usda["n_bas"].astype(float)
    else:
        raise ValueError(f"Unknown BIO_SPLIT_METHOD: {BIO_SPLIT_METHOD}")
    zone_usda["share"] = (
        zone_usda["weight"]
        / zone_usda.groupby("usda_region")["weight"].transform("sum")
    )

    rfm = pd.DataFrame({
        "regional_fuel_market": sorted({f"{lz}-{BIO_FUEL_NAME}" for lz in zone_to_bas}),
        "fuel": BIO_FUEL_NAME,
    })
    rfm.to_csv(out_folder / "regional_fuel_markets.csv", index=False)

    # fuel_cost.csv and regional_fuel_markets.csv must be mutually exclusive
    # on fuel (Switch raises if a fuel appears in both). Strip any
    # waste_biomass rows from fuel_cost.csv, archiving the original first.
    # Also repair coal prices. AEO reports a delivered coal price of 0 in
    # census divisions where NEMS projects no coal deliveries to the power
    # sector (New England in all years; most divisions by 2050), which
    # would give coal generators there free fuel. Replace ALL coal prices
    # with the AEO2025 reference coal prices as processed in ReEDS 2.0
    # inputs (fuelprices/coal_AEO_2025_reference.csv; 2024$/MMBtu per
    # ReEDS fuelprices/dollaryear.csv, matching base_financial_year in
    # financials.csv). These are identical to the AEO values where AEO
    # reports a price, carry forward each division's last reported price
    # where the series ends, and proxy New England with South Atlantic.
    fuel_cost_path = out_folder / "fuel_cost.csv"
    if fuel_cost_path.exists():
        fc = pd.read_csv(fuel_cost_path)
        bio_mask = fc["fuel"] == BIO_FUEL_NAME
        n_bio = int(bio_mask.sum())
        has_zero_coal = ((fc["fuel"] == "coal") & (fc["fuel_cost"] == 0)).any()
        if n_bio > 0 or has_zero_coal:
            _archive_once(fc, out_folder / "fuel_cost_archive.csv")
        if n_bio > 0:
            fc = fc[~bio_mask].copy()
            logger.info(
                f"[biomass] fuel_cost.csv: dropped {n_bio} {BIO_FUEL_NAME} row(s); "
                f"archived original to fuel_cost_archive.csv."
            )
        else:
            logger.info(f"[biomass] fuel_cost.csv: no {BIO_FUEL_NAME} rows to drop.")

        coal_mask = fc["fuel"] == "coal"
        if coal_mask.any():
            coal_price_path = reeds_dir / "fuelprices" / "coal_AEO_2025_reference.csv"
            if not coal_price_path.exists():
                raise FileNotFoundError(
                    f"[biomass] ReEDS coal price file not found: {coal_price_path}. "
                    "Needed to repair AEO zero coal prices in fuel_cost.csv."
                )
            reeds_coal = pd.read_csv(coal_price_path).set_index("year")

            missing_zones = set(fc.loc[coal_mask, "load_zone"]) - set(ZONE_TO_CENSUS_DIV)
            if missing_zones:
                raise ValueError(
                    f"[biomass] load zones missing from ZONE_TO_CENSUS_DIV: "
                    f"{sorted(missing_zones)}"
                )
            missing_years = set(fc.loc[coal_mask, "period"]) - set(reeds_coal.index)
            if missing_years:
                raise ValueError(
                    f"[biomass] period(s) missing from {coal_price_path.name}: "
                    f"{sorted(missing_years)}"
                )

            before = fc.loc[coal_mask, "fuel_cost"].copy()
            fc.loc[coal_mask, "fuel_cost"] = [
                round(float(reeds_coal.loc[p, ZONE_TO_CENSUS_DIV[z]]), 5)
                for z, p in zip(fc.loc[coal_mask, "load_zone"],
                                fc.loc[coal_mask, "period"])
            ]
            n_changed = int((before != fc.loc[coal_mask, "fuel_cost"]).sum())
            n_zero_fixed = int((before == 0).sum())
            if (fc["fuel_cost"] == 0).any():
                raise ValueError(
                    "[biomass] fuel_cost.csv still contains zero prices after "
                    "ReEDS coal repair."
                )
            logger.info(
                f"[biomass] fuel_cost.csv: set {int(coal_mask.sum())} coal "
                f"price(s) from {coal_price_path.name} "
                f"({n_changed} changed, {n_zero_fixed} were zero)."
            )
        fc.to_csv(fuel_cost_path, index=False, na_rep=".")
    z2rfm = pd.DataFrame({
        "load_zone": list(zone_to_bas),
        "regional_fuel_market": [f"{lz}-{BIO_FUEL_NAME}" for lz in zone_to_bas],
    })
    z2rfm.to_csv(out_folder / "zone_to_regional_fuel_market.csv", index=False)

    merged = (
        zone_usda.merge(bio, on="usda_region")
        .assign(max_avail_at_cost=lambda d: d["cap"] * d["share"])
        .query("max_avail_at_cost > 0")
        .sort_values(["load_zone", "price", "usda_region", "bioclass"])
        .reset_index(drop=True)
    )
    merged["tier"] = merged.groupby("load_zone").cumcount() + 1
    merged["regional_fuel_market"] = merged["load_zone"] + f"-{BIO_FUEL_NAME}"
    merged = merged.rename(columns={"price": "unit_cost"})

    sc = (
        merged[["regional_fuel_market", "tier", "unit_cost", "max_avail_at_cost"]]
        .merge(pd.DataFrame({"period": periods}), how="cross")
        [["regional_fuel_market", "period", "tier", "unit_cost", "max_avail_at_cost"]]
    )
    sc.to_csv(out_folder / "fuel_supply_curves.csv", index=False)

    logger.info(
        f"[biomass] wrote regional_fuel_markets.csv ({len(rfm)} rows), "
        f"zone_to_regional_fuel_market.csv ({len(z2rfm)} rows), "
        f"fuel_supply_curves.csv ({len(sc)} rows = "
        f"{sc['regional_fuel_market'].nunique()} markets x {len(periods)} periods)."
    )

# ---------------------------------------------------------------------------
# Hydro min-flow vs. capacity consistency
# ---------------------------------------------------------------------------
#
# PowerGenome's hydro capacity (gen_build_predetermined.csv) and hydro flow
# timeseries (hydro_timeseries.csv) can come out of different aggregation
# paths — EIA existing-units sums vs. ReEDS p-region generation timeseries
# rolled up to load zones — and occasionally disagree by a few percent.
#
# Which Switch constraint bites:
#   Enforce_Hydro_Min_Flow: DispatchGen[g,t] >= hydro_min_flow_mw[g,ts]
#   Enforce_Dispatch_Upper_Limit: DispatchGen[g,t] <= cap * gen_availability[g]
# If hydro_min_flow_mw > cap * gen_availability the model is infeasible and
# diagnose_infeasibility surfaces it as a BuildGen_bounds violation on the
# affected hydro project.
#
#   Enforce_Hydro_Avg_Flow: sum(DispatchGen + SpillHydro) == avg_flow * n_tps
# SpillHydro is a free non-negative variable, so hydro_avg_flow_mw > cap is
# always feasible (the excess just spills). Do NOT clip avg_flow — that would
# silently discard water the model could legitimately spill.
#
# Switch computes gen_availability (dispatch.py init_gen_availability) as:
#   baseload:      (1 - forced_outage_rate) * (1 - scheduled_outage_rate)
#   non-baseload:  (1 - forced_outage_rate)
# Hydro is non-baseload, and both outage rates default to 0 if absent → 1.0.
#
# Fix: for each hydro project where hydro_min_flow_mw > cap * availability,
# clip min_flow down to that ceiling. No tuning factor — this is exactly
# what Switch will enforce.

HYDRO_TS_FILE = "hydro_timeseries.csv"
HYDRO_GBP_FILE = "gen_build_predetermined.csv"
HYDRO_GEN_INFO_FILE = "gen_info.csv"


def _hydro_availability(gen_info: pd.DataFrame, project: str) -> float:
    """Replicate Switch's init_gen_availability for a non-baseload project.
    Returns 1 - forced_outage_rate (default 0 if column/value missing).
    Hydro is never baseload in our configs, so the scheduled_outage term
    doesn't apply; if it ever does, extend this."""
    row = gen_info.loc[gen_info["GENERATION_PROJECT"] == project]
    if row.empty:
        return 1.0
    forced = 0.0
    if "gen_forced_outage_rate" in gen_info.columns:
        raw = row["gen_forced_outage_rate"].iloc[0]
        try:
            forced = float(raw)
        except (TypeError, ValueError):
            forced = 0.0  # '.' / empty / NaN
    return 1.0 - forced


def fix_hydro_flow_capacity(out_folder: Path) -> None:
    """Clip hydro_min_flow_mw to (total predetermined capacity x
    gen_availability) for any project where min flow exceeds its feasible
    dispatch ceiling. Leaves hydro_avg_flow_mw untouched (SpillHydro absorbs
    avg-flow overages)."""
    ts_path = out_folder / HYDRO_TS_FILE
    gbp_path = out_folder / HYDRO_GBP_FILE
    gi_path = out_folder / HYDRO_GEN_INFO_FILE

    if not ts_path.exists():
        logger.info(f"[hydro-flow] {HYDRO_TS_FILE} not found, skipping.")
        return
    if not gbp_path.exists():
        logger.warning(
            f"[hydro-flow] skipping: {HYDRO_GBP_FILE} not found "
            f"(cannot determine hydro capacities)."
        )
        return

    ht = pd.read_csv(ts_path)
    gbp = pd.read_csv(gbp_path)
    gen_info = pd.read_csv(gi_path) if gi_path.exists() else pd.DataFrame(
        columns=["GENERATION_PROJECT"]
    )

    caps = gbp.groupby("GENERATION_PROJECT")["build_gen_predetermined"].sum()
    min_flow_max = ht.groupby("hydro_project")["hydro_min_flow_mw"].max()

    check = (
        min_flow_max.rename("max_min_flow")
        .to_frame()
        .join(caps.rename("total_cap"))
        .dropna()
    )
    check["availability"] = [
        _hydro_availability(gen_info, proj) for proj in check.index
    ]
    check["ceiling"] = check["total_cap"] * check["availability"]
    bad = check[check["max_min_flow"] > check["ceiling"]]

    if bad.empty:
        logger.info(
            "[hydro-flow] all hydro_min_flow_mw values within "
            "cap * availability, no clipping."
        )
        return

    _archive_once(ht, out_folder / "hydro_timeseries_archive.csv")

    n_clipped_total = 0
    for proj, row in bad.iterrows():
        ceiling = row["ceiling"]
        mask = ht["hydro_project"] == proj
        before = ht.loc[mask, "hydro_min_flow_mw"]
        n_clipped = int((before > ceiling).sum())
        ht.loc[mask, "hydro_min_flow_mw"] = before.clip(upper=ceiling)
        n_clipped_total += n_clipped
        logger.info(
            f"[hydro-flow] {proj}: cap={row['total_cap']:.2f} MW, "
            f"availability={row['availability']:.3f}, "
            f"ceiling={ceiling:.2f} MW; "
            f"max min_flow was {row['max_min_flow']:.2f} MW; "
            f"clipped {n_clipped} value(s)."
        )

    ht.to_csv(ts_path, index=False)
    logger.info(
        f"[hydro-flow] clipped {n_clipped_total} hydro_min_flow_mw value(s) "
        f"across {len(bad)} project(s); archived original to "
        f"hydro_timeseries_archive.csv."
    )


# ---------------------------------------------------------------------------
# Coal IGCC energy source fixup
# ---------------------------------------------------------------------------
#
# gen_tech Coal_IGCC_Moderate and Coal_IGCC-90%-CCS_Moderate should both use
# gen_energy_source 'coal', but sometimes inherit the tech name as their fuel
# string. Normalize via literal text replace (lowercase tokens don't collide
# with the capitalized gen_tech column).

COAL_IGCC_REPLACEMENTS = [
    # comma-bounded (gen_energy_source column in the middle of the row)
    (",coal_igcc_moderate,",          ",coal,"),
    (",coal_igcc-90%-ccs_moderate,",  ",coal,"),
    # end-of-line (e.g. graph_tech_types.csv, where energy_source is last col)
    (",coal_igcc_moderate\n",         ",coal\n"),
    (",coal_igcc-90%-ccs_moderate\n", ",coal\n"),
]

COAL_IGCC_TARGET_FILES = (
    "gen_info.csv",
    "gen_info_archive.csv",
    "gen_info.orig.csv",
    "candidate_sites.csv",
    "candidate_sites_archive.csv",
    "candidate_sites.orig.csv",
    "cerf_candidate_sites.csv",
    "graph_tech_types.csv",
)
# CERF sited outputs live under cerf_output*/<timestamp>/outputs/
COAL_IGCC_GLOB_PATTERNS = (
    "cerf_output*/**/cerf_sited_*.csv",
)


def fix_coal_igcc_energy_source(out_folder: Path) -> None:
    """Replace `coal_igcc_moderate` / `coal_igcc-90%-ccs_moderate` with
    `coal` in the gen_energy_source column of downstream files."""
    paths = [out_folder / f for f in COAL_IGCC_TARGET_FILES
             if (out_folder / f).exists()]
    for pattern in COAL_IGCC_GLOB_PATTERNS:
        paths.extend(out_folder.glob(pattern))

    for path in paths:
        text = path.read_text()
        total = 0
        for old, new in COAL_IGCC_REPLACEMENTS:
            n = text.count(old)
            if n:
                text = text.replace(old, new)
                total += n
        rel = path.relative_to(out_folder)
        if total == 0:
            logger.info(f"[coal-igcc] {rel}: no matches.")
            continue
        path.write_text(text)
        logger.info(f"[coal-igcc] {rel}: replaced {total} occurrence(s).")



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Clean up PowerGenome -> Switch inputs (enrich + geothermal + biomass)."
    )
    parser.add_argument(
        "out_folder", type=Path,
        help="Switch case input folder (prepended automatically by "
             "model_adjustment_scripts).",
    )
    parser.add_argument(
        "--reeds-inputs", required=False, default=None, type=Path,
        help="Path to ReEDS-2.0/inputs/ (optional; defaults to "
             "adjust/adjust_data/reeds_data/ next to this script).",
    )
    parser.add_argument(
        "--settings", required=True, type=str,
        help="Path to PowerGenome settings YAML or folder of YAMLs.",
    )
    args = parser.parse_args()

    if not args.out_folder.is_dir():
        logger.error(f"out_folder does not exist: {args.out_folder}")
        sys.exit(1)

    # --- Resolve reeds_inputs: explicit flag > bundled local copy -----------
    _SCRIPT_DIR = Path(__file__).resolve().parent
    _LOCAL_REEDS = _SCRIPT_DIR / "adjust_data" / "reeds_data"

    # Files that clean_geothermal and clean_biomass actually consume.
    # Kept in sync manually with _geothermal_zone_caps and clean_biomass.
    _REQUIRED_REEDS_FILES = [
        Path("geothermal") / "geo_rsc_ATB_2023.csv",
        Path("geothermal") / "geo_discovery_factor_ATB_2023.csv",
        Path("hierarchy.csv"),
        Path("supply_curve") / "bio_supplycurve.csv",
    ]

    if args.reeds_inputs is not None:
        reeds_dir = args.reeds_inputs
        if not reeds_dir.is_dir():
            logger.error(f"--reeds-inputs path does not exist: {reeds_dir}")
            sys.exit(1)
        logger.info(f"[config] Using explicit --reeds-inputs: {reeds_dir}")
    elif _LOCAL_REEDS.is_dir():
        reeds_dir = _LOCAL_REEDS
        missing = [p for p in _REQUIRED_REEDS_FILES
                   if not (reeds_dir / p).is_file()]
        if missing:
            logger.error(
                f"Bundled reeds data at {reeds_dir} is incomplete. Missing:\n  "
                + "\n  ".join(str(p) for p in missing)
                + "\nEither complete adjust/adjust_data/reeds_data/ or pass "
                "--reeds-inputs pointing at a full ReEDS-2.0/inputs/ checkout."
            )
            sys.exit(1)
        logger.info(f"[config] Using bundled reeds data: {reeds_dir}")
    else:
        logger.error(
            f"No --reeds-inputs given and bundled data not found at "
            f"{_LOCAL_REEDS}. Provide --reeds-inputs explicitly."
        )
        sys.exit(1)

    settings = load_settings(path=args.settings)

    pudl_raw = str(settings["PUDL_DB"])
    # Accept either a plain path or a SQLAlchemy URL ("sqlite:///path").
    for prefix in ("sqlite:///", "sqlite://", "sqlite:"):
        if pudl_raw.startswith(prefix):
            pudl_raw = pudl_raw[len(prefix):]
            break
    pudl_db = Path(pudl_raw).expanduser()
    extra_outputs = Path(settings["extra_outputs"]).expanduser() \
        if settings.get("extra_outputs") else None

    try:
        enrich_gen_info(args.out_folder, pudl_db, extra_outputs)
        patch_missing_coordinates(args.out_folder)
        fix_coordinate_errors(args.out_folder)
        fix_gen_project_renames(args.out_folder)
        clean_biopower_fuel_string(args.out_folder)
        fix_biopower_heat_rate(args.out_folder)
        fix_ccs_energy_load(args.out_folder)
        # fix_coal_igcc_energy_source(args.out_folder)
        clean_geothermal(args.out_folder, settings, reeds_dir)
        clean_biomass(args.out_folder, settings, reeds_dir)
        fix_hydro_flow_capacity(args.out_folder)
    except Exception as e:
        logger.error(f"[clean_inputs] failed for {args.out_folder.name}: {e}")
        sys.exit(1)



if __name__ == "__main__":
    main()
