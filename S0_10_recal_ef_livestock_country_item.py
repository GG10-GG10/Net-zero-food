# -*- coding: utf-8 -*-
"""
Livestock EF back-calculation with strict fill order:

Per (Process, Species):
  (1) Row-wise time repair for each Country–Item (treat 0 + NaN as missing): linear interpolate -> ffill/bfill
  (2) Region mean fill (Region_agg2 × Item × Year), ignoring zeros when computing means
  (3) World mean fill (Item × Year), ignoring zeros when computing means

Process = only process (no gas in parentheses)
Species = CH4 or N2O
Unit    = Enteric -> kg/An; others -> kg/kg N

Inputs (hard-coded):
  - /mnt/data/Emissions_livestock_E_All_Data_NOFLAG.csv
  - /mnt/data/Environment_LivestockManure_E_All_Data_NOFLAG.csv
  - /mnt/data/dict_v3.xlsx
      * sheet 'region': requires columns M49_Country_Code, Region_label_new, Region_agg2
          - drop rows where Region_label_new == 'no'
      * sheet 'Emis_item': filter Item_Cat2 in {'Meat','Dairy'}, take Item_Emis

Output:
  - /mnt/data/EF_recalculated_filled_2000_2022_final_v4.csv
    Columns: M49, Area, Item, Process, Species, Unit, Y2000..Y2022
"""

import sys
from itertools import product
import numpy as np
import pandas as pd

# ---------------- Hard-coded paths ----------------
EMIS_CSV  = "../../input/Emission/Emissions_livestock_E_All_Data_NOFLAG.csv"
ENV_CSV   = "../../input/Manure_Stock/Environment_LivestockManure_E_All_Data_NOFLAG.csv"
DICT_XLSX = "../../src/dict_v3.xlsx"
OUT_CSV   = "../../src/retired_and_raw/EF_recalculated_filled_2000_2022_livestock.csv"

# ---------------- helpers: IO/cols ----------------
def pick_cols(cols):
    keep = {"M49_Country_Code", "Area", "Item", "Source", "Element", "Unit"}
    return [c for c in cols if str(c).strip() in keep or str(c).startswith("Y")]

def detect_year_cols(df):
    return [c for c in df.columns if str(c).startswith("Y")]

# ---------------- helpers: units ----------------
def mass_to_kg_factor(u: str) -> float:
    if u is None: return np.nan
    us = u.strip().lower()
    base = us.replace("(", " ").replace(")", " ").replace("/", " ").replace("-", " ").replace(".", " ")
    if us in {"kg","kg ch4","kg n2o","kg n"} or ("kg" in base and ("ch4" in base or "n2o" in base or " n " in f" {base} " or "nitrogen" in base)):
        return 1.0
    if us in {"t","tonne","tonnes","t ch4","t n2o","t n"} or (("tonne" in base or " t " in f" {base} ") and "kt" not in base):
        return 1e3
    if us in {"kt","kt ch4","kt n2o","kt n","gg","gg ch4","gg n2o","gg n"} or ("kt" in base or "gg" in base):
        return 1e6
    if "kt" in base or "gg" in base: return 1e6
    if "tonne" in base or " t " in f" {base} ": return 1e3
    if "kg" in base: return 1.0
    return np.nan

def n_to_kgN_factor(u: str) -> float:
    return mass_to_kg_factor(u)

def stock_to_animal_factor(u: str) -> float:
    if u is None: return np.nan
    us = u.strip().lower()
    base = us.replace("(", " ").replace(")", " ").replace("/", " ").replace("-", " ")
    if us in {"an","head","animal","animals","no","number","count"} or "head" in base:
        if "1000" in base or "thousand" in base: return 1e3
        return 1.0
    if "1000" in base and "head" in base: return 1e3
    if "head" in base: return 1.0
    return np.nan

def convert_by_unit_groups(df, unit_col, year_cols, factor_fn):
    dfc = df.copy()
    units = dfc[unit_col].astype(str).fillna("")
    for u in units.unique().tolist():
        mask = (units == u)
        fac = factor_fn(u)
        if np.isfinite(fac):
            dfc.loc[mask, year_cols] = dfc.loc[mask, year_cols] * fac
        else:
            dfc.loc[mask, year_cols] = np.nan
    return dfc

# ---------------- helpers: shaping ----------------
def build_wide_block(df, element_name, year_cols, conv_kind=None):
    sub = df.loc[df["Element"] == element_name].copy()
    if conv_kind == "emission_kg":
        sub = convert_by_unit_groups(sub, "Unit", year_cols, mass_to_kg_factor)
    elif conv_kind == "stock_an":
        sub = convert_by_unit_groups(sub, "Unit", year_cols, stock_to_animal_factor)
    elif conv_kind == "n_kgn":
        sub = convert_by_unit_groups(sub, "Unit", year_cols, n_to_kgN_factor)
    keep = ["M49", "Item"] + year_cols
    sub = sub[keep].groupby(["M49", "Item"], as_index=False)[year_cols].mean()
    return sub

def ratio(num_df, den_df, year_cols, process, species, unit):
    m = num_df.merge(den_df, on=["M49", "Item"], suffixes=("_NUM", "_DEN"), how="left")
    out = m[["M49", "Item"]].copy()
    for c in year_cols:
        num = m[f"{c}_NUM"]
        den = m[f"{c}_DEN"]
        out[c] = np.where((den != 0) & np.isfinite(den), num / den, np.nan)
    out.insert(2, "Process", process)
    out.insert(3, "Species", species)
    out.insert(4, "Unit", unit)
    return out

# ---------------- filling logic ----------------
def row_neighbor_fill_both(series_vals: pd.Series) -> pd.Series:
    """
    Treat BOTH zeros and NaNs as missing; fill by linear interpolation and nearest (ffill/bfill).
    Only modifies positions originally 0 or NaN.
    """
    s = series_vals.astype(float).copy()
    mask = s.isna() | (s == 0)
    if not mask.any():
        return s
    base = s.copy()
    base[mask] = np.nan
    lin = base.interpolate(method="linear", limit_direction="both", axis=0)
    nn  = base.ffill().bfill()
    out = s.copy()
    for c in s.index:
        if mask.loc[c]:
            v = lin.loc[c]
            if pd.isna(v): v = nn.loc[c]
            out.loc[c] = v
    return out

def fill_region_then_world(df_ps, cross, region_col, year_cols):
    """
    After row-wise time repair, fill remaining {0 or NaN} by:
      Region mean (Region×Item×Year, ignoring zeros) -> World mean (Item×Year, ignoring zeros).
    """
    matched = cross.merge(df_ps, on=["M49", "Item"], how="left")

    # Region means (ignore zeros)
    reg_src = matched[[region_col, "Item"] + year_cols].copy()
    reg_src[year_cols] = reg_src[year_cols].mask((reg_src[year_cols] == 0) | reg_src[year_cols].isna(), np.nan)
    reg_means = (
        reg_src.groupby([region_col, "Item"], dropna=False)[year_cols]
        .mean()
        .reset_index()
    )
    tmp = matched.merge(reg_means, on=[region_col, "Item"], how="left", suffixes=("", "_REGMEAN"))
    for c in year_cols:
        need = tmp[c].isna() | (tmp[c] == 0)
        tmp.loc[need, c] = tmp.loc[need, f"{c}_REGMEAN"]

    # World means (ignore zeros)
    wrd_src = tmp[["Item"] + year_cols].copy()
    wrd_src[year_cols] = wrd_src[year_cols].mask((wrd_src[year_cols] == 0) | wrd_src[year_cols].isna(), np.nan)
    world_means = wrd_src.groupby("Item")[year_cols].mean().reset_index()
    tmp = tmp.merge(world_means, on="Item", how="left", suffixes=("", "_WORLDMEAN"))
    for c in year_cols:
        need = tmp[c].isna() | (tmp[c] == 0)
        tmp.loc[need, c] = tmp.loc[need, f"{c}_WORLDMEAN"]

    return tmp

# ---------------- dict_v3 readers ----------------
def read_dict_v3(path_dict: str):
    """
    From dict_v3.xlsx build:
      - region_df: filtered region sheet (drop Region_label_new == 'no'), columns: Country, Region_label_new, Region_agg2
      - region_map: columns M49 (parsed from Country), Region_agg2
      - countries_m49: list of M49 from region_df
      - items: Item_Emis from Emis_item where Item_Cat2 in {'Meat','Dairy'}
    """
    try:
        xls = pd.ExcelFile(path_dict)
        region_df = pd.read_excel(xls, "region", usecols=["M49_Country_Code","Region_label_new","Region_agg2"])
        emis_item = pd.read_excel(xls, "Emis_item", usecols=["Item_Emis","Item_Cat2"])
    except Exception as e:
        raise RuntimeError(f"Failed to read dict_v3.xlsx: {e}")

    region_df.columns = [str(c).strip() for c in region_df.columns]
    emis_item.columns = [str(c).strip() for c in emis_item.columns]

    # filter region: drop Region_label_new == 'no'
    region_df = region_df.loc[region_df["Region_label_new"].astype(str).str.strip().str.lower() != "no"].copy()

    # parse M49 from explicit column
    region_df["M49"] = pd.to_numeric(region_df["M49_Country_Code"], errors="coerce").astype("Int64")
    region_map = region_df[["M49","Region_agg2"]].dropna().drop_duplicates().copy()

    # items: Emis_item where Item_Cat2 in {Meat, Dairy}
    items = (
        emis_item.loc[emis_item["Item_Cat2"].isin(["Meat","Dairy"]), "Item_Emis"]
        .dropna()
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    # countries list
    countries_m49 = (
        region_df["M49"].dropna().astype(int).drop_duplicates().sort_values().tolist()
    )

    return region_df, region_map, countries_m49, items

# ---------------- main ----------------
def main():
    # Load FAO CSVs
    emis_cols = pd.read_csv(EMIS_CSV, nrows=0).columns.tolist()
    env_cols  = pd.read_csv(ENV_CSV,  nrows=0).columns.tolist()
    emis = pd.read_csv(EMIS_CSV, usecols=pick_cols(emis_cols))
    env  = pd.read_csv(ENV_CSV,  usecols=pick_cols(env_cols))

    for df in (emis, env):
        df.columns = [str(c).strip() for c in df.columns]

    emis["M49"] = pd.to_numeric(emis["M49_Country_Code"], errors="coerce").astype("Int64")
    env["M49"]  = pd.to_numeric(env["M49_Country_Code"], errors="coerce").astype("Int64")

    year_e = detect_year_cols(emis);  year_v = detect_year_cols(env)
    for c in year_e: emis[c] = pd.to_numeric(emis[c], errors="coerce")
    for c in year_v: env[c]  = pd.to_numeric(env[c],  errors="coerce")
    years = sorted(set(year_e).intersection(set(year_v)))
    years = [c for c in years if "Y2000" <= c <= "Y2022"]
    if not years:
        print("No overlapping Y2000..Y2022.", file=sys.stderr); sys.exit(2)

    emis_keep = [
        "Enteric fermentation (Emissions CH4)",
        "Manure management (Emissions CH4)",
        "Manure management (Emissions N2O)",
        "Manure left on pasture (Emissions N2O)",
        "Manure applied to soils (Emissions N2O)",
    ]
    env_keep = [
        "Stocks",
        "Manure left on pasture (N content)",
        "Manure management (manure treated, N content)",
        "Manure applied to soils (N content)",
    ]
    emis_f = emis.loc[(emis.get("Source","")=="FAO TIER 1") & (emis["Element"].isin(emis_keep)),
                      ["M49","Area","Item","Element","Unit"] + years].copy()
    env_f  = env.loc[env["Element"].isin(env_keep),
                     ["M49","Area","Item","Element","Unit"] + years].copy()

    # Unit-normalized wide tables
    num_enteric_ch4 = build_wide_block(emis_f, "Enteric fermentation (Emissions CH4)", years, "emission_kg")
    num_mm_ch4      = build_wide_block(emis_f, "Manure management (Emissions CH4)", years, "emission_kg")
    num_mm_n2o      = build_wide_block(emis_f, "Manure management (Emissions N2O)", years, "emission_kg")
    num_pasture_n2o = build_wide_block(emis_f, "Manure left on pasture (Emissions N2O)", years, "emission_kg")
    num_applied_n2o = build_wide_block(emis_f, "Manure applied to soils (Emissions N2O)", years, "emission_kg")

    den_stocks      = build_wide_block(env_f, "Stocks", years, "stock_an")
    den_mm_n        = build_wide_block(env_f, "Manure management (manure treated, N content)", years, "n_kgn")
    den_pasture_n   = build_wide_block(env_f, "Manure left on pasture (N content)", years, "n_kgn")
    den_applied_n   = build_wide_block(env_f, "Manure applied to soils (N content)", years, "n_kgn")

    # EF ratios
    ef_enteric_ch4 = ratio(num_enteric_ch4, den_stocks,    years, "Enteric fermentation",    "CH4", "kg/An")
    ef_mm_ch4      = ratio(num_mm_ch4,      den_mm_n,      years, "Manure management",       "CH4", "kg/kg N")
    ef_mm_n2o      = ratio(num_mm_n2o,      den_mm_n,      years, "Manure management",       "N2O", "kg/kg N")
    ef_pasture_n2o = ratio(num_pasture_n2o, den_pasture_n, years, "Manure left on pasture",  "N2O", "kg/kg N")
    ef_applied_n2o = ratio(num_applied_n2o, den_applied_n, years, "Manure applied to soils", "N2O", "kg/kg N")

    ef_all = pd.concat([ef_enteric_ch4, ef_mm_ch4, ef_mm_n2o, ef_pasture_n2o, ef_applied_n2o], ignore_index=True)

    # Attach Area
    area_lut = pd.concat([emis[["M49","Area"]], env[["M49","Area"]]]).dropna().drop_duplicates()
    ef_all = ef_all.merge(area_lut, on="M49", how="left")
    ef_all = ef_all[["M49","Area","Item","Process","Species","Unit"] + years]

    # Read dict_v3.xlsx to build target Country×Item (199×15)
    region_df, region_map, countries_m49, items = read_dict_v3(DICT_XLSX)

    # Full cross-product from dict_v3 (after filtering Region_label_new != 'no' and Meat/Dairy items)
    cross = pd.DataFrame(list(product(countries_m49, items)), columns=["M49","Item"])
    cross = cross.merge(region_map, on="M49", how="left")           # Region_agg2
    cross = cross.merge(area_lut,  on="M49", how="left")            # Area (for output readability)

    # Per (Process, Species): (1) row-neighbor (0+NaN) -> (2) region mean -> (3) world mean
    blocks = []
    for (proc, species), sub in ef_all.groupby(["Process","Species"], dropna=False):
        df_ps = sub[["M49","Item","Process","Species","Unit"] + years].copy()

        # (1) row-wise: treat 0 + NaN as missing, fill by linear + nearest
        df_ps[years] = df_ps[years].apply(row_neighbor_fill_both, axis=1, result_type="expand")

        # (2)+(3) region -> world (ignore zeros when computing means)
        tmp = fill_region_then_world(df_ps, cross, "Region_agg2", years)

        # Stamp meta
        tmp["Process"] = proc
        tmp["Species"] = species
        tmp["Unit"] = "kg/An" if proc == "Enteric fermentation" else "kg/kg N"

        blocks.append(tmp[["M49","Area","Item","Process","Species","Unit"] + years])

    ef_final = pd.concat(blocks, ignore_index=True).sort_values(["Process","Species","M49","Item"]).reset_index(drop=True)
    ef_final.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # Report
    expected = len(countries_m49) * len(items) * 5
    na_pct = round(100 * ef_final[years].isna().sum().sum() / (ef_final.shape[0]*len(years)), 3)
    zero_left = int((ef_final[years] == 0).sum().sum())
    print(f"Countries: {len(countries_m49)} | Items(Meat/Dairy): {len(items)}")
    print(f"Rows: {ef_final.shape[0]}  (expected ~{expected})")
    print(f"Missing % after fills: {na_pct}")
    print(f"Zeros remaining: {zero_left}")
    print(f"Saved: {OUT_CSV}")

if __name__ == "__main__":
    main()
