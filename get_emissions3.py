#!/usr/bin/env python3
"""
# get_emissions.py
# ----------------
# This script builds generator-level emission intensities and marginal damage costs 
# for Switch model input folders. It automates the workflow that was previously 
# handled by AddEmissionInputs3.py, now refactored for clarity and CLI execution.

# Main tasks:
# 1. **CLI Input** – Takes a single argument: the path to a Switch input folder 
#    (e.g., `/Users/.../base_20_week_2045/2045/base_20_week`).
# 2. **Data Loading** – Reads Switch input files (`fuels.csv`, `gen_info.csv`, 
#    `financials.csv`) and supporting emission datasets (ANL Tables 2–5, InMAP 
#    marginal values, EIA plant data, CERF candidate sites, and IPM→NERC mappings).
# 3. **Emission Factor Computation** – Calculates fuel-level emission intensities 
#    (NOx, SOx, PM2.5, PM10, VOC, CO, CH4, N2O) using ANL-20/41 Tables 2 and 4.
# 4. **NERC-Region Weighting** – Derives generator-specific emission factors by 
#    combining ANL Table 5 with NERC region assignments and inferred technologies.
# 5. **Marginal Damage Costs (MVs)** – Maps each generator (and CERF site, if present)
#    to InMAP grid cells, retrieves marginal values (PM2.5, NOx, VOC, SO2) using the
#    ISRM stack-height class appropriate to each generator's technology
#    (ground / low stack / high stack per Goodkind et al. 2019 SI), and imputes
#    missing data by zone and technology using `impute_mv_by_zone_tech`.
#    VOC always uses the ground column (per SI Table S1 footnote b: VOC stack
#    emissions are rare and not well characterized at stack height).
# 6. **Non-Emitter Handling** – Non-emitting sources (wind, solar, hydro, nuclear,
#    hydrogen, storage, heat) skip the MV lookup pipeline (no meaningful damage
#    values to compute), but are written to gen_emission_costs.csv with zeros for
#    all cost and intensity columns. This is required: Switch's DataPortal
#    implicitly uses the GENERATION_PROJECT column of this file to define
#    GENERATION_PROJECTS, so any generator absent from this file is silently
#    dropped from the model.
# 7. **Cost Conversion** – Adjusts marginal damage costs from 2011 to the financial 
#    base year using a fixed cumulative inflation rate.
# 8. **Outputs** – Produces:
#    - `gen_pm25_costs.csv` – PM2.5 marginal cost and intensity by generator.
#    - `gen_emission_costs.csv` – Combined PM2.5, NOx, and VOC costs and intensities.
#    - Archives `fuels_archive.csv` with computed fuel-level emission intensities.

# Usage example:
#     $ python get_emissions.py /path/to/switch_inputs_folder

# This version improves readability, prints progress updates for long-running steps, 
# and ensures compatibility with the latest imputation logic in 
# `EmissionConversionFunctions.py`.
# """
#### ## 
# 0) Imports (deduped) + small helpers
import os
from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import box, Point
# from powergenome.financials import inflation_price_adjustment # does not work

CumulativeRateofInflation = 0.355 # from https://www.usinflationcalculator.com/ with 2011 (original year) to 2023 (base_financial_year)
def to_fraction_from_percent(s: pd.Series) -> pd.Series:
    """'87.5%' or 87.5 -> 0.875; blanks -> NaN."""
    return (
        s.astype(str).str.strip().str.rstrip('%').replace({'': np.nan})
        .astype(float) / 100.0
    )


def stack_height_for_gen(gen_tech, gen_energy_source):
    """
    Classify a generator into an ISRM emission stack-height class: 'ground', 'low', or 'high'.

    Rationale (Goodkind et al. 2019, PNAS SI S1.4; Briggs 1972 plume rise):
      ISRM stack-height classes correspond to different effective stack heights
      (physical stack + plume rise). For the electric sector:
        - 'high' : large utility boilers and NGCC units with tall stacks
                   (coal ST, NG/oil/biomass steam, combined cycle HRSGs, IGCC)
        - 'low'  : simple-cycle combustion turbines (peakers), reciprocating
                   internal-combustion engines, and small distributed combustion
        - 'ground': non-combustion / externally accounted sources
                    (wind, solar, nuclear, hydro, geothermal, batteries,
                     hydrogen, heat, imports, distributed_generation)

    Resolution order (most specific first):
      1. Tech-string non-emitter keywords (batteries, storage, nuclear, solar,
         photovoltaic, wind, hydro, geothermal, imports, distributed)
      2. Fuel-string non-emitters (electricity, water, wind, solar, uranium,
         nuclear, hydrogen, heat, geothermal)
      3. Tech-string combustion keywords (steam/ST, CC, IGCC -> high;
         CT/GT/combustion turbine, IC/reciprocating/engine -> low)
      4. Fuel-string fallback (coal/biomass -> high; NG/oil -> low)
      5. Default low

    Non-emitting rows return 'ground' as a placeholder; they are dropped from
    gen_emission_costs.csv downstream so the stack class is only a sentinel.
    """
    fuel = (str(gen_energy_source) if pd.notna(gen_energy_source) else "").strip().lower()
    tech = (str(gen_tech) if pd.notna(gen_tech) else "").strip().lower()

    # ---------- 1) Tech-string non-emitter keywords ----------
    # Catches generators whose gen_energy_source may be ambiguous (e.g., Imports,
    # distributed_generation, or where fuel is left blank for non-combustion tech).
    non_emitting_tech_keywords = (
        "batter", "storage",
        "nuclear",
        "solar", "photovoltaic", "utilitypv", "pv",
        "wind",                      # onshore/offshore/landbased/offshorewind
        "hydro",                     # hydroelectric / hydro
        "geothermal",
        "import",                    # Imports_base_base
        "distributed",               # distributed_generation
        "hydrogen",
    )
    if any(kw in tech for kw in non_emitting_tech_keywords):
        return "ground"

    # ---------- 2) Fuel-string non-emitters ----------
    non_emitting_fuels = {
        "electricity", "water", "wind", "solar",
        "uranium", "nuclear", "hydrogen", "heat",
        "geothermal",
    }
    if fuel in non_emitting_fuels:
        return "ground"

    # ---------- 3) Tech-based combustion classification ----------
    # Steam turbine (boiler + stack): tall stacks
    if "steam" in tech or " st" in f" {tech}" or tech.endswith("_st") or "_st_" in tech:
        return "high"
    # Biopower / dedicated biomass — large steam-cycle plants
    if "biopower" in tech or "biomass" in tech:
        return "high"
    # Combined cycle: tall HRSG stacks (incl. CCS variants)
    if ("combined cycle" in tech or "combinedcycle" in tech
            or " cc" in f" {tech}" or "_cc" in tech or tech.startswith("cc")):
        return "high"
    # IGCC — integrated gasification combined cycle, tall stacks
    if "igcc" in tech:
        return "high"
    # Simple-cycle combustion turbines / peakers: shorter stacks
    if ("combustion turbine" in tech or "combustionturbine" in tech
            or " ct" in f" {tech}" or "_ct" in tech
            or ("gas turbine" in tech and "steam" not in tech and "combined" not in tech)):
        return "low"
    # Reciprocating internal combustion engines: low
    if ("internal combustion" in tech or "reciproc" in tech
            or "engine" in tech or "_ice" in tech or tech.endswith("ice")):
        return "low"

    # ---------- 4) Fuel-based fallback when tech string is uninformative ----------
    if fuel == "coal":
        return "high"
    if fuel in {"biomass", "waste_biomass"}:
        return "high"
    if fuel in {"naturalgas", "ng", "fuel"}:
        return "low"
    if fuel in {"oil", "distillate"}:
        return "low"

    # ---------- 5) Final fallback ----------
    return "low"

# Not used anymore:
# def impute_mv_by_zone_tech(gen_info: pd.DataFrame, column: str) -> pd.DataFrame:
#     """
#     For any NaN in gen_info[column], replace it with the average value from the “closest” rows:
#       1. same zone & same tech
#       2. same zone & same gen_energy_source
#       3. same zone (any tech)
#       4. same tech (any zone)
#       5. global mean fallback

#     Prints each imputation and, at the end, how many times each rule fired.
#     """
    
#     valid = gen_info.loc[gen_info[column].notna(), :]
#     if valid.empty:
#         return gen_info

#     # Precompute means for fast lookups
#     global_mean = valid[column].mean()
#     mean_zone_tech = valid.groupby(["gen_load_zone", "gen_tech"], dropna=False)[column].mean()
#     # Only build the zone+energy_source map if the column exists
#     has_energy_source = "gen_energy_source" in gen_info.columns
#     if has_energy_source:
#         mean_zone_fuel = valid.groupby(["gen_load_zone", "gen_energy_source"], dropna=False)[column].mean()
#     mean_zone = valid.groupby(["gen_load_zone"], dropna=False)[column].mean()
#     mean_tech = valid.groupby(["gen_tech"], dropna=False)[column].mean()

#     na_idx = gen_info.index[gen_info[column].isna()]

#     rule_counts = {
#         "same zone & same tech":            0,
#         "same zone & same energy source":   0,
#         "same zone (any tech)":             0,
#         "same tech (any zone)":             0,
#         "global mean fallback":             0,
#     }

#     for i in na_idx:
#         zone = gen_info.at[i, "gen_load_zone"]
#         tech = gen_info.at[i, "gen_tech"]
#         fuel = gen_info.at[i, "gen_energy_source"] if has_energy_source else None

#         # 1) same zone & same tech
#         fill_val = mean_zone_tech.get((zone, tech), None)
#         rule = None

#         # 2) same zone & same gen_energy_source
#         if fill_val is None and has_energy_source and pd.notna(fuel):
#             fill_val = mean_zone_fuel.get((zone, fuel), None)
#             if fill_val is not None:
#                 rule = "same zone & same energy source"

#         # 3) same zone (any tech)
#         if fill_val is None:
#             fill_val = mean_zone.get(zone, None)
#             if fill_val is not None:
#                 rule = "same zone (any tech)"

#         # 4) same tech (any zone)
#         if fill_val is None:
#             fill_val = mean_tech.get(tech, None)
#             if fill_val is not None:
#                 rule = "same tech (any zone)"

#         # 5) global mean fallback
#         if fill_val is None or pd.isna(fill_val):
#             fill_val = global_mean
#             rule = "global mean fallback"

#         # If rule still None, it means the first rule hit (zone+tech)
#         if rule is None:
#             rule = "same zone & same tech"

#         gen_info.at[i, column] = fill_val
#         rule_counts[rule] += 1
#         print(f"Plant {i} (zone={zone}, tech={tech}"
#               + (f", fuel={fuel}" if has_energy_source else "")
#               + f") imputed via {rule}")

#     # summary
#     print("\nImputation summary:")
#     for rule, cnt in rule_counts.items():
#         print(f"  {rule:30s}: {cnt}")

#     return gen_info
def impute_mv_by_zone_tech(gen_info: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    For any NaN in gen_info[column], replace it with the average value from the “closest” rows:
      1. same zone & same tech
      2. same zone & same gen_energy_source
      3. same zone (any tech)
      4. same tech (any zone)
      5. global mean fallback

    Prints each imputation and, at the end, how many times each rule fired.
    """

    valid = gen_info.loc[gen_info[column].notna(), :]
    if valid.empty:
        return gen_info

    # Precompute means for fast lookups
    global_mean = valid[column].mean()
    mean_zone_tech = valid.groupby(["gen_load_zone", "gen_tech"], dropna=False)[column].mean()

    has_energy_source = "gen_energy_source" in gen_info.columns
    if has_energy_source:
        mean_zone_fuel = valid.groupby(["gen_load_zone", "gen_energy_source"], dropna=False)[column].mean()

    mean_zone = valid.groupby(["gen_load_zone"], dropna=False)[column].mean()
    mean_tech = valid.groupby(["gen_tech"], dropna=False)[column].mean()

    na_idx = gen_info.index[gen_info[column].isna()]

    rule_counts = {
        "same zone & same tech":            0,
        "same zone & same energy source":   0,
        "same zone (any tech)":             0,
        "same tech (any zone)":             0,
        "global mean fallback":             0,
    }

    for i in na_idx:
        zone = gen_info.at[i, "gen_load_zone"]
        tech = gen_info.at[i, "gen_tech"]
        fuel = gen_info.at[i, "gen_energy_source"] if has_energy_source else None

        # 1) same zone & same tech
        fill_val = mean_zone_tech.get((zone, tech), None)
        rule = None

        # 2) same zone & same gen_energy_source
        if fill_val is None and has_energy_source and pd.notna(fuel):
            fill_val = mean_zone_fuel.get((zone, fuel), None)
            if fill_val is not None:
                rule = "same zone & same energy source"

        # 3) same zone (any tech)
        if fill_val is None:
            fill_val = mean_zone.get(zone, None)
            if fill_val is not None:
                rule = "same zone (any tech)"

        # 4) same tech (any zone)
        if fill_val is None:
            fill_val = mean_tech.get(tech, None)
            if fill_val is not None:
                rule = "same tech (any zone)"

        # 5) global mean fallback
        if fill_val is None or pd.isna(fill_val):
            fill_val = global_mean
            rule = "global mean fallback"

        # If rule still None, it means the first rule hit (zone+tech)
        if rule is None:
            rule = "same zone & same tech"

        gen_info.at[i, column] = fill_val
        rule_counts[rule] += 1
        print(
            f"Plant {i} (zone={zone}, tech={tech}"
            + (f", fuel={fuel}" if has_energy_source else "")
            + f") imputed via {rule}"
        )

    # summary
    print("\nImputation summary:")
    for rule, cnt in rule_counts.items():
        print(f"  {rule:30s}: {cnt}")

    return gen_info
def load_MVs_as_gdf(csv_path):
    """
    Reads a marginal values CSV file and returns a GeoDataFrame with columns:
    - cell_ID: grid cell identifier
    - geometry: polygon of the cell in WGS84 coordinates (EPSG:4326)
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Create geometry in original projection (Lambert Conformal Conic 2SP)
    df['geometry'] = df.apply(
        lambda row: box(row['Location_W'], row['Location_S'], row['Location_E'], row['Location_N']),
        axis=1
    )
    
    # Define source CRS
    lcc_crs = (
        "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-97 "
        "+x_0=0 +y_0=0 +a=6370997 +b=6370997 +units=m +no_defs"
    )
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df[['cell_ID', 'geometry']],
        crs=lcc_crs
    )
    
    # Reproject to WGS84
    gdf = gdf.to_crs(epsg=4326)
    return gdf

def load_MVs_as_df(csv_path):
    """
    Reads a marginal values CSV file and returns a DataFrame with all columns except 'geometry' collumns:
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    df.drop(columns=['Location_W', 'Location_S', 'Location_E', 'Location_N'], inplace=True)
    
    return df


#### ## 
#### ##
# 1) Paths (CLI)
import argparse
import sys

parser = argparse.ArgumentParser(
    description="Build per-generator emission intensities and costs for a Switch input folder."
)
parser.add_argument(
    "input_path",
    help="Path to the Switch input folder (e.g., /Users/.../base_20_week_2045/2045/base_20_week_test)"
)
args = parser.parse_args()
input_path = Path(args.input_path).expanduser().resolve()
# input_path =Path('/Users/melek/Desktop/Research/Air quality/Inputs/AQ_Foresight_backup/2050/base_short_commit') 
print(f"[INFO] Using input folder: {input_path}")

# 1) Paths (alternatives are commented out to flip quickly)

fuels_file       = input_path / 'fuels.csv'
gen_info_file    = input_path / 'gen_info.csv'
financials_file  = input_path / 'financials.csv'
cerf_siting_file = input_path / 'cerf_candidate_sites.csv' # From [run_cerf_conus26.py.ipnyb](/Users/melek/Desktop/Research/Siting/run_cerf_conus26.py)

input_data       = Path('/Users/melek/Documents/GitHub/Switch-USA-PG/emission_inputs')
table2_file      = input_data / 'ANL_tables/Energy_Conversion_Efficiecies_ANL_2020_Table2.csv'                     # From ANL-20/41 "TABLE 2 National and regional energy conversion efficiencies by fuel subtype and combustion technology."
table4_file      = input_data / 'ANL_tables/Emission_Factors_ANL_2020_Table_4.csv'                                  # From ANL-20/41 "TABLE 4 National generation-weighted average emission factors in g/kWh by fuel subtype and combustion technology"
table5_file      = input_data / 'ANL_tables/NERC_Regions_Emission_Factors_ANL_2020_Table_5_clean.csv'               # From ANL-20/41 "TABLE 5 NERC regional generation-weighted average emission factors in g/kWh by fuel subtype and combustion technology"
mv_file          = input_data / 'InMAP/marginal_values_updated_110819.csv'
# powerplant0_file = input_data / 'DoE/power-plants.csv'                           # From openenergyhub.ornl.gov "Power Plants - EIA"
IPM2NERC_file    = input_data / 'IPM_to_NERC_mapping.csv'                                               # a csv file that maps each load zone to a NERC region. information from '/Users/melek/Desktop/Research/Capacity Expansion/Switch/Switch-USA-PG/MIP_results_comparison/case_settings/26-zone/settings-atb2023/model_definition.yml' line 82-119
plant_emissions_file = input_data / 'SingularityEnergy_2023_plant_data_annual_us_units_/plant_data.csv' # From https://singularity.energy/data-download-page
#### ## 
# 2) Load core inputs
# switch files
fuels       = pd.read_csv(fuels_file)
gen_info    = pd.read_csv(gen_info_file)
financials  = pd.read_csv(financials_file)

# other files
cerf_candidate_sites = pd.read_csv(cerf_siting_file) if cerf_siting_file.exists() else None
EmissionFactors      = pd.read_csv(table4_file)
HeatRates            = pd.read_csv(table2_file)
EmissionFactorsNERC  = pd.read_csv(table5_file)
MV                   = pd.read_csv(mv_file)
IPM_to_NERC_mapping  = pd.read_csv(IPM2NERC_file)
plant_hist_emissions = pd.read_csv(plant_emissions_file)
# Stash the authoritative list of all GENERATION_PROJECTs before any in-memory
# filtering (coordinate validation, non-emitter drops, etc.) reduces gen_info.
# gen_emission_costs.csv MUST cover every ID in gen_info.csv on disk, otherwise
# Pyomo's DataPortal silently shrinks GENERATION_PROJECTS.
all_gen_ids = gen_info["GENERATION_PROJECT"].drop_duplicates().tolist()
print(f"[INFO] Stashed {len(all_gen_ids):,} GENERATION_PROJECT IDs for final padding.")
# scalar base year
base_financial_year = int(pd.Series(financials['base_financial_year']).iloc[0])
print(f"Loaded. base_financial_year={base_financial_year}, gen_info={len(gen_info):,} rows, fuels={len(fuels):,} rows.")
fuel_map = {
    "naturalgas":"NG", "ng":"NG",
    "distillate":"OIL", "oil":"OIL",
    "coal":"COAL",
    "biomass":"BIOMASS", "waste_biomass":"BIOMASS",
    # non-emitters
    "electricity":None, "water":None, "wind":None, "solar":None,
    "uranium":None, "nuclear":None, "hydrogen":None, "heat":None,
    "geothermal":None,
    "sun": None,"storage": None, "imports": None,
}
#### ## 
# 3) Tidy ANL Table 4 shares to fractions
for col in ['Fuel_type_share', 'Fuel_subtype_share', 'Combustion_technology_share']:
    if col in EmissionFactors.columns:
        EmissionFactors[col] = to_fraction_from_percent(EmissionFactors[col])
EmissionFactors['weight'] = (
    EmissionFactors.get('Fuel_subtype_share', 1.0) *
    EmissionFactors.get('Combustion_technology_share', 1.0)
)

# pollutants to compute
pollutants = ['NOx','SOx','PM2.5','PM10','VOC','CO','CH4','N2O']

# weighted-average EF (g/kWh) by Fuel_type
ef_per_fuel = (
    EmissionFactors
    .groupby('Fuel_type', as_index=False)
    .apply(lambda df: (df[pollutants].mul(df['weight'], axis=0).sum() / df['weight'].sum()))
    .reset_index(drop=True)
)
ef_per_fuel.head()
#### ## 
# 4) Tidy ANL Table 2 efficiencies
# Expect columns: 'Fuel type', 'Fuel subtype share (%)', 'Combustion tech share (%)', 'National'
if {'Fuel subtype share (%)','Combustion tech share (%)'}.issubset(HeatRates.columns):
    HeatRates = HeatRates.copy()
    HeatRates['weight'] = HeatRates['Fuel subtype share (%)'] * HeatRates['Combustion tech share (%)']
else:
    # fallback if already fraction columns exist without (%)
    HeatRates['weight'] = HeatRates.get('Fuel subtype share', 1.0) * HeatRates.get('Combustion tech share', 1.0)

efficiency_by_fuel = (
    HeatRates
    .groupby('Fuel type', as_index=False)
    .apply(lambda df: (df[['National']].mul(df['weight'], axis=0).sum() / df['weight'].sum()))
    .reset_index(drop=True)
    .rename(columns={'National': 'Efficiency'})
)

# Efficiency given in percent; convert to fraction
efficiency_by_fuel['Efficiency'] = efficiency_by_fuel['Efficiency'] / 100.0
efficiency_by_fuel.head()
#### ## 
# 5) Join EF + Efficiency and compute intensities (t/MMBTU)
BTU_PER_KWH = 3412.0

ef_hr = ef_per_fuel.merge(
    efficiency_by_fuel[['Fuel type','Efficiency']],
    left_on='Fuel_type', right_on='Fuel type', how='left'
)

for p in pollutants:
    col = f"f_{p.lower().replace('.','')}_intensity"
    # g/kWh * (kWh_e/kWh_th) / 3412 Btu/kWh = g/Btu = t/MMBTU numerically
    ef_hr[col] = ef_hr[p] * ef_hr['Efficiency'] / BTU_PER_KWH

intensity_cols = [c for c in ef_hr.columns if c.endswith('_intensity')]
lookup = ef_hr.set_index('Fuel_type')[intensity_cols]
lookup.head()
#### ## 
# 5a) Map intensities to fuels.csv, archive, save
fuel_to_ef = {
    'Coal':          'Coal',
    'Naturalgas':    'NG',
    'Distillate':    'Oil',
    'Waste_biomass': 'Biomass',
    'Fuel':          'NG',      # generic Fuel -> NG
    'Uranium':       None,      # zero
    'Hydrogen':      None,      # zero
}

fuels_out = fuels.copy()
# Case-insensitive lookup: the 'fuel' column in fuels.csv may arrive in any
# capitalization (coal / Coal / COAL), so normalize both sides before mapping.
_fuel_to_ef_ci = {k.lower(): v for k, v in fuel_to_ef.items()}
_fuel_key_lower = fuels_out['fuel'].astype(str).str.strip().str.lower()

# Warn loudly if any fuel in fuels.csv has no mapping entry (would silently NaN)
_unmapped = sorted(set(_fuel_key_lower) - set(_fuel_to_ef_ci))
if _unmapped:
    print(f"[WARN] fuels.csv contains fuels not in fuel_to_ef mapping: {_unmapped}. "
          f"Their intensities will be 0.0.")

for col in intensity_cols:
    mapping = {
        fuel_lc: (lookup.at[key, col] if (key is not None and key in lookup.index) else 0.0)
        for fuel_lc, key in _fuel_to_ef_ci.items()
    }
    fuels_out[col] = _fuel_key_lower.map(mapping)
#from https://publications.anl.gov/anlpubs/2006/12/58024.pdf:
# "SOx emission factors for most fuels are calculated on the basis of the assumption that all sulfur
# contained in process fuels is converted into sulfur dioxide (SO2)."
fuels_out['f_so2_intensity']=fuels_out['f_sox_intensity']
# archive then write
fuels_archive = input_path / 'fuels_archive.csv'
if fuels_archive.exists():
    print(f"[INFO] {fuels_archive.name} already exists — preserving existing archive (not overwriting).")
else:
    fuels.to_csv(fuels_archive, index=False)
    print(f"[INFO] Archived original fuels.csv -> {fuels_archive}")
# Use '.' as missing-value token to match Switch's convention
fuels_out.to_csv(fuels_file, index=False, na_rep='.')
print(f"fuels.csv ( {fuels_file} ) updated with {len(intensity_cols)} intensity columns. Archive -> {fuels_archive}")
#### ## 
# 6) Emission Factors by generator
def build_EF_with_emissions(gen_info, IPM_to_NERC_mapping, table5):
    """
    Returns EF with columns:
      ['GENERATION_PROJECT','gen_tech','gen_energy_source','gen_energy_source','gen_load_zone','NERC_Region',
       'VOC_g_per_kWh','PM2.5_g_per_kWh','NOx_g_per_kWh','SO2_g_per_kWh]

    Rules (concise):
      - Non-emitting sources → 0 for all pollutants.
      - Petroleum Liquids with gen_energy_source == 'Distillate' → Oil/DFO.
      - Weighted averages by (region,fuel[,subfuel][,tech]) using Table 5 shares, with national fallbacks.
      - Tech inference from gen_tech: CC/GT/IC/ST.
    """
    # --- Validate ---
    gi = gen_info.copy()
    req_gi = {"GENERATION_PROJECT","gen_tech","gen_energy_source","gen_load_zone"}
    if not req_gi.issubset(gi.columns): raise ValueError("gen_info missing required columns.")

    if not {"gen_load_zone","NERC_Region"}.issubset(IPM_to_NERC_mapping.columns):
        raise ValueError("IPM_to_NERC_mapping must have ['gen_load_zone','NERC_Region']")

    pollutants = ["VOC_g_per_kWh","PM2.5_g_per_kWh","NOx_g_per_kWh","SOx_g_per_kWh"]
    req_t5 = {"NERC_region","Fuel_type","Fuel_subtype","Combustion_technology",
              "Fuel_type_share_pct","Fuel_subtype_share_pct","Combustion_share_pct", *pollutants}
    if not req_t5.issubset(table5.columns):
        missing = req_t5 - set(table5.columns)
        raise ValueError(f"table5 missing: {missing}")

    # --- Normalize Table 5 ---
    t5 = table5.copy()
    for c in ["NERC_region","Fuel_type","Fuel_subtype","Combustion_technology"]:
        t5[c] = t5[c].astype(str).str.strip().str.upper()
    for c in ["Fuel_type_share_pct","Fuel_subtype_share_pct","Combustion_share_pct", *pollutants]:
        t5[c] = pd.to_numeric(t5[c], errors="coerce")
    t5["__w"] = (
        t5["Fuel_type_share_pct"].fillna(100)
        * t5["Fuel_subtype_share_pct"].fillna(100)
        * t5["Combustion_share_pct"].fillna(100)
    )

    def wmeans(df):
        sw = np.nansum(df["__w"].to_numpy(dtype=float))
        out = {}
        for p in pollutants:
            v = df[p].to_numpy(dtype=float)
            if sw > 0 and np.isfinite(sw):
                out[p] = float(np.nansum(df["__w"].to_numpy()*v) / sw)
            else:
                v = v[~np.isnan(v)]
                out[p] = float(v.mean()) if v.size else np.nan
        return pd.Series(out)

    # Aggregates
    L1 = t5.groupby(["NERC_region","Fuel_type","Fuel_subtype","Combustion_technology"]).apply(wmeans).reset_index()
    L2 = t5.groupby(["NERC_region","Fuel_type","Fuel_subtype"]).apply(wmeans).reset_index()
    L3 = t5.groupby(["NERC_region","Fuel_type","Combustion_technology"]).apply(wmeans).reset_index()
    L4 = t5.groupby(["NERC_region","Fuel_type"]).apply(wmeans).reset_index()
    t5A = t5.assign(NERC_region="ALL")
    A1 = t5A.groupby(["NERC_region","Fuel_type","Fuel_subtype","Combustion_technology"]).apply(wmeans).reset_index()
    A2 = t5A.groupby(["NERC_region","Fuel_type","Fuel_subtype"]).apply(wmeans).reset_index()
    A3 = t5A.groupby(["NERC_region","Fuel_type","Combustion_technology"]).apply(wmeans).reset_index()
    A4 = t5A.groupby(["NERC_region","Fuel_type"]).apply(wmeans).reset_index()

    # Dict lookups
    def to_dict(df, cols):
        return {tuple(r[c] for c in cols): r[pollutants].to_dict() for _, r in df.iterrows()}
    D1, D2, D3, D4 = (to_dict(L1,["NERC_region","Fuel_type","Fuel_subtype","Combustion_technology"]),
                      to_dict(L2,["NERC_region","Fuel_type","Fuel_subtype"]),
                      to_dict(L3,["NERC_region","Fuel_type","Combustion_technology"]),
                      to_dict(L4,["NERC_region","Fuel_type"]))
    DA1, DA2, DA3, DA4 = (to_dict(A1,["NERC_region","Fuel_type","Fuel_subtype","Combustion_technology"]),
                          to_dict(A2,["NERC_region","Fuel_type","Fuel_subtype"]),
                          to_dict(A3,["NERC_region","Fuel_type","Combustion_technology"]),
                          to_dict(A4,["NERC_region","Fuel_type"]))

    # Fuel / subfuel / tech parsing
    def norm(s): return str(s).strip().lower() if pd.notna(s) else s
    fuel_map = {
        "naturalgas":"NG", "ng":"NG",
        "distillate":"OIL", "oil":"OIL",
        "coal":"COAL",
        "biomass":"BIOMASS", "waste_biomass":"BIOMASS",
        # non-emitters
        "electricity":None, "water":None, "wind":None, "solar":None,
        "uranium":None, "nuclear":None, "hydrogen":None, "heat":None,
        "geothermal":None,
        "sun": None,
        "storage": None, "imports": None,
    }
    def subfuel_pref(gen_energy_source, _gen_tech):
        return "DFO" if norm(gen_energy_source) == "distillate" else None
    def tech_pref(gen_tech):
        s = norm(gen_tech or "")
        if "combined cycle" in s or " cc" in f" {s}" or "cc " in f"{s} ": return "CC"
        if "internal combustion" in s or "recip" in s or "engine" in s: return "IC"
        if "steam" in s or "conventional steam" in s: return "ST"
        if "combustion turbine" in s or " ct" in f" {s}" or "ct " in f"{s} " or ("gas turbine" in s and "steam" not in s): return "GT"
        return None

    gi = gi.merge(IPM_to_NERC_mapping[["gen_load_zone","NERC_Region"]], on="gen_load_zone", how="left")
    gi["__NER"]  = gi["NERC_Region"].astype(str).str.strip().str.upper()
    gi["__FUEL"] = gi["gen_energy_source"].map(lambda x: fuel_map.get(norm(x), None))
    gi["__SUB"]  = gi.apply(lambda r: subfuel_pref(r["gen_energy_source"], r["gen_tech"]), axis=1)
    gi["__TECH"] = gi["gen_tech"].map(tech_pref)

    # Resolver
    zero_vec = {p: 0.0 for p in pollutants}
    def resolve(n, f, sf, te):
        if f is None: return zero_vec
        if pd.notna(n):
            if sf and te and (n,f,sf,te) in D1:  return D1[(n,f,sf,te)]
            if sf and (n,f,sf) in D2:            return D2[(n,f,sf)]
            if te and (n,f,te) in D3:            return D3[(n,f,te)]
            if (n,f) in D4:                      return D4[(n,f)]
        if sf and te and ("ALL",f,sf,te) in DA1: return DA1[("ALL",f,sf,te)]
        if sf and ("ALL",f,sf) in DA2:           return DA2[("ALL",f,sf)]
        if te and ("ALL",f,te) in DA3:           return DA3[("ALL",f,te)]
        if ("ALL",f) in DA4:                     return DA4[("ALL",f)]
        return {p: np.nan for p in pollutants}

    resolved = [resolve(n,f,s,t) for n,f,s,t in zip(gi["__NER"],gi["__FUEL"],gi["__SUB"],gi["__TECH"])]
    for p in pollutants:
        gi[p] = [d[p] for d in resolved]

    EF = gi[[
        "GENERATION_PROJECT","gen_tech","gen_energy_source","gen_load_zone","NERC_Region",
        "VOC_g_per_kWh","PM2.5_g_per_kWh","NOx_g_per_kWh", "SOx_g_per_kWh"
    ]].copy()

    # --- convert g/kWh → tonne/MMBtu for all pollutants ---
    # 1 g = 1e-6 tonne; 1 kWh = 0.003412 MMBtu
    conv_gpkwh_to_t_per_mmbtu = 1e-6 / 0.003412  # ≈ 0.00029308323563892143
    EF["VOC_t_per_MMBtu"]   = EF["VOC_g_per_kWh"]   * conv_gpkwh_to_t_per_mmbtu
    EF["PM2.5_t_per_MMBtu"] = EF["PM2.5_g_per_kWh"] * conv_gpkwh_to_t_per_mmbtu
    EF["NOx_t_per_MMBtu"]   = EF["NOx_g_per_kWh"]   * conv_gpkwh_to_t_per_mmbtu
    #from https://publications.anl.gov/anlpubs/2006/12/58024.pdf:
    # "SOx emission factors for most fuels are calculated on the basis of the assumption that all sulfur
    # contained in process fuels is converted into sulfur dioxide (SO2)."
    EF["SO2_t_per_MMBtu"]   = EF["SOx_g_per_kWh"]   * conv_gpkwh_to_t_per_mmbtu

    return EF
EF = build_EF_with_emissions(gen_info, IPM_to_NERC_mapping, EmissionFactorsNERC)
#### ## 
#from https://publications.anl.gov/anlpubs/2006/12/58024.pdf:
# "SOx emission factors for most fuels are calculated on the basis of the assumption that all sulfur
# contained in process fuels is converted into sulfur dioxide (SO2)."
# 6a) CCS adjustment for NOx and SO2 intensities (Koornneef-style multipliers)
#   Several studies (e.g., Koornneef et al. 2010, Atmospheric Environment; Koornneef et al. 2010 BOLK report)
#   show that adding post-combustion CO2 capture changes co-pollutant emissions per kWh:
#       - NOx tends to INCREASE (due to the parasitic load / energy penalty),
#       - SO2 tends to DECREASE strongly (due to upstream flue-gas desulfurization and solvent protection).
#   We implement this by scaling generator-level NOx and SO2 emission intensities for units that have
#   gen_ccs_capture_efficiency set (i.e., CCS units) using technology-specific multipliers that already
#   incorporate the parasitic load (primary energy penalty). The multipliers are applied to the
#   EF-derived t/MMBtu emission intensities:
#       - PC coal with post-combustion CCS:
#             NOx_t_per_MMBtu *= 1.15   # ~15% higher NOx per kWh (energy penalty × slight per-fuel reduction)
#             SO2_t_per_MMBtu *= 0.18   # ~82% lower SO2 per kWh
#       - NGCC with post-combustion CCS:
#             NOx_t_per_MMBtu *= 1.11   # ~11% higher NOx per kWh
#             SO2_t_per_MMBtu *= 0.14   # ~86% lower SO2 per kWh (absolute SO2 is small for gas)
#   This keeps the ANL-based fuel/technology emission factors as the baseline, but adjusts CCS units so that
#   the resulting co-pollutant intensities and costs are consistent with the literature on CCS co-impacts:
#   CCS reduces SO2 significantly but can increase NOx on a net-per-kWh basis once the capture energy penalty
#   is taken into account.
#### ##
print("[INFO] Adjusting NOx and SO2 emission intensities for CCS generators...")

# Bring CCS info onto EF
ccs_cols = ["GENERATION_PROJECT", "gen_energy_source", "gen_ccs_capture_efficiency"]
missing_ccs_cols = set(ccs_cols) - set(gen_info.columns)
if missing_ccs_cols:
    raise ValueError(f"gen_info.csv is missing expected CCS columns: {missing_ccs_cols}")

EF = EF.merge(gen_info[ccs_cols], on="GENERATION_PROJECT", how="left", suffixes=("", "_gi"))

# Clean CCS flag: '.' or blanks -> NaN, numeric -> CCS
EF["gen_ccs_capture_efficiency"] = pd.to_numeric(
    EF["gen_ccs_capture_efficiency"], errors="coerce"
)
is_ccs = EF["gen_ccs_capture_efficiency"].notna()

# Normalize fuel names
def _norm(s):
    return str(s).strip().lower() if pd.notna(s) else ""

fuel = EF["gen_energy_source"].map(_norm)

# Identify CCS fuel types (lowercase to match _norm)
coal_labels = {"coal"}
ng_labels   = {"naturalgas", "fuel"}

is_coal_ccs = is_ccs & fuel.isin(coal_labels)
is_ng_ccs   = is_ccs & fuel.isin(ng_labels)


n_coal_ccs = int(is_coal_ccs.sum())
n_ng_ccs   = int(is_ng_ccs.sum())
n_ccs_total = int(is_ccs.sum())

print(f"[INFO] Found {n_ccs_total} CCS generators "
      f"({n_coal_ccs} coal, {n_ng_ccs} natural gas) with gen_ccs_capture_efficiency.")

# Sanity check: make sure the intensity columns exist
for col in ["NOx_t_per_MMBtu", "SO2_t_per_MMBtu"]:
    if col not in EF.columns:
        raise ValueError(f"Expected column '{col}' not found in EF.")

# Apply multipliers (per kWh including parasitic load)
# PC post-comb: NOx × 1.15, SO2 × 0.18
EF.loc[is_coal_ccs, "NOx_t_per_MMBtu"] *= 1.15
EF.loc[is_coal_ccs, "SO2_t_per_MMBtu"] *= 0.18

# NGCC post-comb: NOx × 1.11, SO2 × 0.14
EF.loc[is_ng_ccs, "NOx_t_per_MMBtu"] *= 1.11
EF.loc[is_ng_ccs, "SO2_t_per_MMBtu"] *= 0.14

print("[INFO] Applied Koornneef-based CCS multipliers to NOx and SO2 intensities.")
#### ##

#### ##
# # 7) Exclude CERF candidate projects from gen_info (if present)
# if cerf_candidate_sites is not None:
#     before = len(gen_info)
#     gen_info = gen_info[~gen_info['GENERATION_PROJECT'].isin(cerf_candidate_sites['GENERATION_PROJECT'])]
#     print(f"Filtered gen_info: {before} -> {len(gen_info)} (removed CERF candidates)")
# else:
#     print("No CERF candidate sites file found; keeping gen_info as-is.")
#### ## 
# 8) Attach coordinates and MV grid
# Note: get_plants_coordinates reads from file path, not df
print(f"\n[INFO] Step 8: Using existing coordinates in gen_info and assigning marginal value (MV) cells...")
print(f"       → gen_info currently has {len(gen_info):,} rows.")

# Case-insensitively resolve the longitude / latitude column names
_cols_lower = {c.lower(): c for c in gen_info.columns}
if "longitude" not in _cols_lower or "latitude" not in _cols_lower:
    missing = {"longitude", "latitude"} - set(_cols_lower)
    raise ValueError(
        f"gen_info.csv is expected to contain longitude/latitude (any case), "
        f"but is missing: {missing}"
    )
LON_COL = _cols_lower["longitude"]
LAT_COL = _cols_lower["latitude"]
print(f"[INFO] Using coordinate columns: {LON_COL!r}, {LAT_COL!r}")

# Coerce coordinates to numeric: placeholders like '.', '', 'NA' -> NaN.
# Rows with missing coordinates are dropped — in practice these are the
# non-emitting techs (batteries, wind, solar, nuclear, hydro, geothermal,
# imports, distributed_generation) that we don't assign marginal damages to
# anyway. The explicit fuel_map non-emitter filter further down provides a
# second layer of defense for any stragglers.
n_before = len(gen_info)
gen_info[LON_COL] = pd.to_numeric(gen_info[LON_COL], errors="coerce")
gen_info[LAT_COL] = pd.to_numeric(gen_info[LAT_COL], errors="coerce")

# -----------------------------------------------------------------------------
# Manual coordinate patches
# -----------------------------------------------------------------------------
# For real existing plants that arrive with missing coordinates from the
# upstream PowerGenome/ReEDS pipeline. Look up the plant in EIA-860 (Form 860
# "Plant" or "Generator" sheet) and paste lat/lon here keyed by GENERATION_PROJECT.
#
# Format: "GENERATION_PROJECT_name": (latitude, longitude)
#
# Look up by plant_id_eia at:
#   https://www.eia.gov/electricity/data/eia860/
#   (or https://atlas.eia.gov/ for a map interface)
#
# Any entry whose GENERATION_PROJECT doesn't exist in gen_info is silently
# ignored, so it's safe to leave stale entries around across runs.
# -----------------------------------------------------------------------------
MANUAL_COORD_PATCHES = {
    # Plant 675 — Larsen Memorial — Polk, FL — Operational Power Plant
    "Central_FL_natural_gas_fired_combined_cycle_11": (28.0491, -81.9238),
    "Central_FL_natural_gas_fired_combined_cycle_12": (28.0491, -81.9238),

    # Plant 56610 — Antelope Valley — Mercer, ND
    "NE_and_e_SD_and_w_ND_and_e_MT_natural_gas_fired_combined_cycle_3": (47.370542, -101.8357),

    # Plant 6120 — Mint Farm Generating Station — Cowlitz, WA 
    "WA_and_OR_natural_gas_fired_combustion_turbine_8": (46.138824, -122.9855),
    "WA_and_OR_natural_gas_fired_combustion_turbine_9": (46.138824, -122.9855),
}

if MANUAL_COORD_PATCHES:
    _patch_applied = 0
    _patch_skipped_found = 0   # GP exists but already had valid coords
    _patch_skipped_missing = 0 # GP not in gen_info at all
    for _gp, (_lat, _lon) in MANUAL_COORD_PATCHES.items():
        _mask = gen_info["GENERATION_PROJECT"] == _gp
        if not _mask.any():
            _patch_skipped_missing += 1
            continue
        # Only overwrite if the existing coords are NaN (don't clobber good data)
        _needs_patch = _mask & (gen_info[LAT_COL].isna() | gen_info[LON_COL].isna())
        if _needs_patch.any():
            gen_info.loc[_needs_patch, LAT_COL] = _lat
            gen_info.loc[_needs_patch, LON_COL] = _lon
            _patch_applied += int(_needs_patch.sum())
            print(f"[INFO] Patched coords for {_gp}: ({_lat:.4f}, {_lon:.4f})")
        else:
            _patch_skipped_found += 1
    print(
        f"[INFO] Manual coord patches: {_patch_applied} applied, "
        f"{_patch_skipped_found} skipped (already had coords), "
        f"{_patch_skipped_missing} skipped (GENERATION_PROJECT not in gen_info)."
    )

bad_coord_mask = gen_info[LON_COL].isna() | gen_info[LAT_COL].isna()
n_bad = int(bad_coord_mask.sum())
if n_bad > 0:
    print(
        f"[INFO] Dropping {n_bad:,} / {n_before:,} generators with missing/non-numeric coordinates:"
    )
    # Breakdown by gen_tech for visibility
    bad_by_tech = gen_info.loc[bad_coord_mask, "gen_tech"].value_counts()
    for tech, cnt in bad_by_tech.items():
        print(f"         - {tech}: {cnt}")

    # Sanity flag: warn if any *emitting* fuels are in the dropped set,
    # since those would be losing real damage costs.
    _non_emit_fuels = {k for k, v in fuel_map.items() if v is None}
    emitting_dropped = gen_info.loc[
        bad_coord_mask
        & ~gen_info["gen_energy_source"].astype(str).str.lower().isin(_non_emit_fuels)
    ]
    if len(emitting_dropped) > 0:
        print(
            f"[WARN] {len(emitting_dropped):,} dropped rows have emitting fuels "
            f"(not in fuel_map non-emitter list). Check these before shipping results:"
        )
        # Include EIA identifiers + state when available for upstream traceability
        _warn_cols = ["GENERATION_PROJECT", "gen_tech", "gen_energy_source", "gen_load_zone"]
        for extra in ("utility_id_eia", "plant_id_eia", "state"):
            if extra in emitting_dropped.columns:
                _warn_cols.append(extra)
        print(
            emitting_dropped[_warn_cols]
            .head(20)
            .to_string(index=False)
        )

gen_info = gen_info.loc[~bad_coord_mask].copy().reset_index(drop=True)
print(f"[INFO] gen_info now has {len(gen_info):,} rows with valid coordinates.")


print("[INFO] Loading InMAP marginal values (as GeoDataFrame and DataFrame)...")
MVgdf = load_MVs_as_gdf(str(mv_file))
MVdf  = load_MVs_as_df(str(mv_file))
print(f"[INFO] Loaded MV grid with {len(MVgdf):,} spatial cells.")
MVdf  = load_MVs_as_df(str(mv_file))
####
# If MVgdf doesn't have a CRS yet, set it:
if MVgdf.crs is None:
    MVgdf = MVgdf.set_crs(epsg=4326)   # or whatever CRS MVgdf really is

print(f"[INFO] Assigning MV cell IDs for {len(gen_info):,} generator rows. This may take time due to spatial lookups...")

gen_gdf = gpd.GeoDataFrame(
    gen_info.copy(),
    geometry=gpd.points_from_xy(gen_info[LON_COL], gen_info[LAT_COL]),
    crs="EPSG:4326",
)

# Make sure MVgdf has a CRS
if MVgdf.crs is None:
    MVgdf = MVgdf.set_crs(epsg=4326)

# Reproject if needed
if MVgdf.crs != gen_gdf.crs:
    gen_gdf = gen_gdf.to_crs(MVgdf.crs)

# Reset index so we have a stable generator index column
gen_gdf_reset = gen_gdf.reset_index().rename(columns={"index": "gen_idx"})

joined = gpd.sjoin(
    gen_gdf_reset,
    MVgdf[["cell_ID", "geometry"]],
    how="left",
    predicate="within"   # or "intersects"
)

print(f"[DEBUG] sjoin produced {len(joined)} rows for {gen_gdf_reset['gen_idx'].nunique()} generators.")

# If any generators matched multiple cells, keep just one (e.g., first by cell_ID)
joined_unique = (
    joined
    .sort_values(["gen_idx", "cell_ID"])
    .drop_duplicates("gen_idx")   # keep first cell_ID per generator
    .set_index("gen_idx")
)

# Now align back to gen_info's index
cell_ids = joined_unique["cell_ID"].reindex(gen_info.index)

# Assign
gen_info["cell_IDs"] = cell_ids.values

# --- Summary of missing cell_IDs ---
missing_mask = gen_info['cell_IDs'].isna()
n_missing = missing_mask.sum()

print(f"[INFO] Missing cell_IDs: {n_missing} out of {len(gen_info)} total generators.")

# Count by gen_tech
missing_by_tech = gen_info.loc[missing_mask, 'gen_tech'].value_counts()

print("[INFO] Missing cell_IDs by gen_tech:")
for tech, count in missing_by_tech.items():
    print(f"    - {tech}: {count}")

print("[INFO] Finished assigning cell IDs to all generators.")
gen_all = gen_info.copy()
# # Ensure each generator project appears once
gen_all = gen_all.drop_duplicates(subset='GENERATION_PROJECT', keep='first')

#### Stack-height assignment + emitter filtering
# Assign ISRM stack-height class to every generator based on gen_tech / gen_energy_source.
gen_all["stack_height"] = [
    stack_height_for_gen(t, f)
    for t, f in zip(gen_all["gen_tech"], gen_all["gen_energy_source"])
]
print(
    "[INFO] Assigned ISRM stack-height classes: "
    + ", ".join(f"{k}={v}" for k, v in gen_all["stack_height"].value_counts().items())
)

# Skip non-emitters in the MV lookup pipeline (wasteful and the values would be
# meaningless), but stash their IDs so we can pad them back with zeros at the
# end. All generators must appear in gen_emission_costs.csv to avoid Pyomo's
# DataPortal silently shrinking GENERATION_PROJECTS.
non_emitters = [k for k, v in fuel_map.items() if v is None]
is_non_emitter = gen_all["gen_energy_source"].astype(str).str.lower().isin(non_emitters)
non_emitter_ids = gen_all.loc[is_non_emitter, "GENERATION_PROJECT"].tolist()
print(
    f"[INFO] Skipping MV lookup for {len(non_emitter_ids):,} non-emitting "
    f"generators (sources: {non_emitters}); they'll be padded with zeros."
)
gen_all = gen_all.loc[~is_non_emitter].copy().reset_index(drop=True)
print(f"[INFO] {len(gen_all):,} emitting generators remain for MV lookup.")


def map_cell_marginal_values(pollutant, gen_all):
    """
    Assign the ISRM marginal damage value per generator, choosing the stack-height
    column (ground / low / high) appropriate to each row's stack_height class.

    - pollutant : one of 'PM2.5', 'NOx', 'VOC', 'SO2'
    - Output column is named MD_{pollutant} (e.g. 'MD_PM2.5') — no stack suffix,
      because per-row stack choice is already embedded.
    - VOC is a special case: per Goodkind et al. 2019 SI Table S1 footnote b,
      VOC stack emissions are rare and the paper only uses ground-level VOC
      marginal damages. We therefore use MD_VOC_ground for every VOC generator
      regardless of its stack_height classification.
    """
    out_col = f"MD_{pollutant}"

    # Per-stack source column selection
    if pollutant == "VOC":
        stack_col_map = {
            "ground": "MD_VOC_ground",
            "low":    "MD_VOC_ground",  # override: VOC stacks rare (SI S1)
            "high":   "MD_VOC_ground",  # override
        }
    else:
        stack_col_map = {
            "ground": f"MD_{pollutant}_ground",
            "low":    f"MD_{pollutant}_low",
            "high":   f"MD_{pollutant}_high",
        }

    # Ensure cell_ID types are aligned for the join
    gen_all["cell_IDs"] = gen_all["cell_IDs"].astype(float)

    gen_all[out_col] = np.nan
    for stack_class, mv_col in stack_col_map.items():
        mask = gen_all["stack_height"] == stack_class
        n_rows = int(mask.sum())
        if n_rows == 0:
            continue
        if mv_col not in MVdf.columns:
            raise KeyError(
                f"Expected column '{mv_col}' not found in marginal_values CSV. "
                f"Available MD_* columns: {[c for c in MVdf.columns if c.startswith('MD_')]}"
            )
        mv_small = MVdf[["cell_ID", mv_col]].drop_duplicates().copy()
        mv_small["cell_ID"] = mv_small["cell_ID"].astype(float)
        mv_map = mv_small.set_index("cell_ID")[mv_col]

        gen_all.loc[mask, out_col] = gen_all.loc[mask, "cell_IDs"].map(mv_map).values
        print(
            f"[INFO] {out_col}: assigned {mv_col} to {n_rows:,} generators "
            f"(stack={stack_class})."
        )

    # Impute any remaining NaNs (e.g., generators with no cell_ID, or cells not in MV)
    missing_before = int(gen_all[out_col].isna().sum())
    print(f"[INFO] {out_col}: missing before imputation = {missing_before:,}")
    if missing_before > 0:
        gen_all = impute_mv_by_zone_tech(gen_all, out_col)
    missing_after = int(gen_all[out_col].isna().sum())
    print(f"[INFO] {out_col}: missing after imputation = {missing_after:,}")

    return gen_all


# 9a-d) Stack-aware MD lookup for each pollutant
gen_all = map_cell_marginal_values("PM2.5", gen_all)
gen_all = map_cell_marginal_values("NOx",   gen_all)
gen_all = map_cell_marginal_values("VOC",   gen_all)
gen_all = map_cell_marginal_values("SO2",   gen_all)

#####
# --- Build gen_emission_costs.csv via merge on GENERATION_PROJECT ---
EF_lo = EF[[
    "GENERATION_PROJECT",
    "PM2.5_t_per_MMBtu", "NOx_t_per_MMBtu", "VOC_t_per_MMBtu", "SO2_t_per_MMBtu"
]].copy()

gen_emission_costs = (
    gen_all[[
        "GENERATION_PROJECT", "stack_height",
        "MD_PM2.5", "MD_NOx", "MD_VOC", "MD_SO2",
    ]]
    .merge(EF_lo, on="GENERATION_PROJECT", how="left")
    .rename(columns={
        "MD_PM2.5":          "pm25_cost_dollar_per_ton",
        "PM2.5_t_per_MMBtu": "gen_pm25_intensity_ton_per_MMBtu",

        "MD_NOx":            "nox_cost_dollar_per_ton",
        "NOx_t_per_MMBtu":   "gen_NOx_intensity_ton_per_MMBtu",

        "MD_VOC":            "voc_cost_dollar_per_ton",
        "VOC_t_per_MMBtu":   "gen_VOC_intensity_ton_per_MMBtu",

        "MD_SO2":            "so2_cost_dollar_per_ton",
        "SO2_t_per_MMBtu":   "gen_SO2_intensity_ton_per_MMBtu",
    })
    .drop_duplicates(subset="GENERATION_PROJECT", keep="first")
)


# Inflate marginal damages to base_financial_year
for c in [
    "pm25_cost_dollar_per_ton",
    "nox_cost_dollar_per_ton",
    "voc_cost_dollar_per_ton",
    "so2_cost_dollar_per_ton",
]:
    gen_emission_costs[c] = gen_emission_costs[c] * (1 + CumulativeRateofInflation)

# Fail fast if schema is wrong (prevents the silent-zero objective bug)
expected = {
    "pm25_cost_dollar_per_ton",
    "nox_cost_dollar_per_ton",
    "voc_cost_dollar_per_ton",
    "so2_cost_dollar_per_ton",
}
missing = expected - set(gen_emission_costs.columns)
assert not missing, f"Missing required cost columns: {missing}"
# Pad ANY missing GENERATION_PROJECTs with zeros so gen_emission_costs.csv covers
# every ID in gen_info.csv on disk. Missing IDs come from two sources:
#   (1) non-emitters skipped from the MV lookup pipeline, and
#   (2) generators dropped during coordinate validation (mostly renewable
#       cluster reps with no physical lat/lon).
# Both must appear in the output or Pyomo's DataPortal will shrink
# GENERATION_PROJECTS to the emitting+geocoded subset and break gen_tech.
covered = set(gen_emission_costs["GENERATION_PROJECT"])
missing_ids = [g for g in all_gen_ids if g not in covered]
numeric_cols = [c for c in gen_emission_costs.columns if c != "GENERATION_PROJECT"]
pad = pd.DataFrame({
    "GENERATION_PROJECT": missing_ids,
    **{c: 0.0 for c in numeric_cols},
})
if "stack_height" in pad.columns:
    pad["stack_height"] = "ground"
gen_emission_costs = pd.concat([gen_emission_costs, pad], ignore_index=True)
print(
    f"[INFO] Padded {len(missing_ids):,} missing generators with zeros "
    f"(non-emitters + coord-dropped); final row count: {len(gen_emission_costs):,}."
)

# Final sanity check against the on-disk gen_info, not the in-memory one
assert len(gen_emission_costs) == len(all_gen_ids), (
    f"gen_emission_costs row count ({len(gen_emission_costs):,}) "
    f"does not match original gen_info ({len(all_gen_ids):,}). "
    f"Switch will fail with index-validation errors."
)

out_emissions_file = input_path / "gen_emission_costs.csv"
gen_emission_costs = gen_emission_costs.sort_values("GENERATION_PROJECT").reset_index(drop=True)
gen_emission_costs.to_csv(out_emissions_file, index=False, na_rep='.')
print(f"Saved {out_emissions_file} with {len(gen_emission_costs):,} rows.")
