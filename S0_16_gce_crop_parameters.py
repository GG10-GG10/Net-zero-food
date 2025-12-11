# -*- coding: utf-8 -*-
"""
S0_16_gce_crop_parameters
-------------------------

Back-calculates country × crop parameters for FAOSTAT-based GCE processes:

    (2) Crop residues – N₂O emission factor (Emissions / N content)
    (3) Crop residues – Residue N content per unit production
    (4) Burning crop residues – CH₄/N₂O emission factors (Emissions / Biomass DM)
    (5) Burning crop residues – Biomass DM per unit production
    (6) Rice cultivation – CH₄ emission factor (Emissions / Area harvested)
    (7) Synthetic fertilizers – N₂O emission factor (Emissions / Fertilizer N content)

Data handling rules (shared across processes):
  * Years restricted to 2000–2020 (columns Y2000..Y2020, wide format)
  * Emissions source limited to FAO TIER 1; production rows limited to Select==1
  * Item names resolved via dict_v3.xlsx (Emis_item sheet) for Item_Emis ↔︎ production /
    fertilizer aliases
  * Filling order for zero/NaN/inf results:
        1. Within each country–item row use previous/next non-zero year (ffill + bfill)
        2. Region mean (Region_agg2 × Item × Process × Element × GHG × Unit)
        3. Global mean   (Item × Process × Element × GHG × Unit)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Iterable, List

from config_paths import get_input_base, get_src_base

# -----------------------------------------------------------------------------
# Constants / paths
# -----------------------------------------------------------------------------
YEAR_START = 2000
YEAR_END = 2020
YEAR_COLS = [f"Y{year}" for year in range(YEAR_START, YEAR_END + 1)]

INPUT_BASE = Path(get_input_base())
SRC_BASE = Path(get_src_base())

EMIS_CSV = INPUT_BASE / "Emission" / "Emissions_crops_E_All_Data_NOFLAG.csv"
PROD_CSV = INPUT_BASE / "Production_Trade" / "Production_Crops_Livestock_E_All_Data_NOFLAG_yield_refilled.csv"
DICT_XLSX = SRC_BASE / "dict_v3.xlsx"
FERT_XLSX = INPUT_BASE / "Fertilizer" / "Fertilizer_efficiency.xlsx"
OUT_CSV = SRC_BASE / "GCE_crop_parameters_country_item_S0_16.csv"


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def _clean_str(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.replace({"nan": np.nan, "None": np.nan, "": np.nan})
    return cleaned


def _mass_to_kg_factor(unit: str) -> float:
    u = (unit or "").strip().lower()
    if not u or u == "nan":
        return np.nan
    if "kt" in u or "kiloton" in u:
        return 1e6
    if u in {"kg", "kg n", "kg n2o", "kg ch4", "kg dm"} or "kg" in u:
        return 1.0
    if u in {"t", "tonnes", "tonne"}:
        return 1e3
    if "1000" in u and ("t" in u or "tonne" in u or "tonnes" in u):
        return 1e6
    return np.nan


def _tonne_factor(unit: str) -> float:
    u = (unit or "").strip().lower()
    if not u or u == "nan":
        return np.nan
    if u in {"t", "tonnes", "tonne"}:
        return 1.0
    if "1000" in u and ("t" in u or "tonne" in u or "tonnes" in u):
        return 1e3
    if "kg" in u:
        return 1e-3
    return np.nan


def _area_factor(unit: str) -> float:
    u = (unit or "").strip().lower()
    if not u or u == "nan":
        return np.nan
    if "ha" in u:
        return 1.0
    if "1000 ha" in u:
        return 1e3
    return np.nan


def _convert_units(df: pd.DataFrame,
                   unit_col: str,
                   year_cols: List[str],
                   factor_fn: Callable[[str], float],
                   target_unit: str) -> pd.DataFrame:
    """Convert FAO numeric columns in-place based on unit column."""
    data = df.copy()
    for col in year_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    units = _clean_str(data[unit_col])
    unique_units = units.dropna().unique().tolist()
    for unit in unique_units:
        mask = units == unit
        factor = factor_fn(unit)
        if np.isnan(factor):
            data.loc[mask, year_cols] = np.nan
        else:
            data.loc[mask, year_cols] = data.loc[mask, year_cols] * factor
    data[unit_col] = target_unit
    return data


def _aggregate_by_item(df: pd.DataFrame,
                       year_cols: List[str],
                       item_col: str = "Item") -> pd.DataFrame:
    group_cols = ["M49_Country_Code", item_col]
    agg = (
        df[group_cols + year_cols]
        .groupby(group_cols, as_index=False)
        .sum(min_count=1)
    )
    return agg


def _compute_ratio(numerator: pd.DataFrame,
                   denominator: pd.DataFrame,
                   year_cols: List[str]) -> pd.DataFrame:
    merged = numerator.merge(
        denominator,
        on=["M49_Country_Code", "Item"],
        how="outer",
        suffixes=("_num", "_den"),
    )
    out = merged[["M49_Country_Code", "Item"]].copy()
    for yc in year_cols:
        num = merged[f"{yc}_num"]
        den = merged[f"{yc}_den"]
        ratio = np.where(
            np.isfinite(num) & np.isfinite(den) & (den != 0),
            num / den,
            np.nan,
        )
        out[yc] = ratio
    return out


def _attach_metadata(df: pd.DataFrame,
                     process: str,
                     element: str,
                     ghg: str,
                     unit: str) -> pd.DataFrame:
    data = df.copy()
    data["Process"] = process
    data["Element"] = element
    data["GHG"] = ghg
    data["Unit"] = unit
    cols = ["M49_Country_Code", "Item", "Process", "Element", "GHG", "Unit"] + YEAR_COLS
    return data[cols]


def _prepare_region_map(dict_path: Path) -> pd.DataFrame:
    region = pd.read_excel(
        dict_path,
        sheet_name="region",
        usecols=["M49_Country_Code", "Region_label_new", "Region_agg2"],
        dtype=str,
    )
    region = region[region["Region_label_new"].ne("no")]
    region = region.drop_duplicates(subset=["M49_Country_Code"])
    region["M49_Country_Code"] = _clean_str(region["M49_Country_Code"])
    region["Region_agg2"] = _clean_str(region["Region_agg2"])
    return region[["M49_Country_Code", "Region_agg2"]]


def _fill_by_rules(df: pd.DataFrame, region_map: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data[YEAR_COLS] = data[YEAR_COLS].apply(pd.to_numeric, errors="coerce")
    data[YEAR_COLS] = data[YEAR_COLS].replace([np.inf, -np.inf], np.nan)
    data[YEAR_COLS] = data[YEAR_COLS].where(data[YEAR_COLS] != 0, np.nan)

    # Step 1: within-row temporal fill (previous/next non-zero)
    neighbor = data[YEAR_COLS].ffill(axis=1).bfill(axis=1)
    data[YEAR_COLS] = data[YEAR_COLS].fillna(neighbor)

    # Step 2: region-level mean (Region_agg2)
    merged = data.merge(region_map, on="M49_Country_Code", how="left")
    non_empty = merged[~merged[YEAR_COLS].isna().all(axis=1)]
    region_nonnull = non_empty[non_empty["Region_agg2"].notna()]
    region_cols = ["Region_agg2", "Item", "Process", "Element", "GHG", "Unit"]
    if not region_nonnull.empty:
        reg_means = (
            region_nonnull.groupby(region_cols)[YEAR_COLS]
            .mean(numeric_only=True)
            .reset_index()
        )
        merged = merged.merge(
            reg_means,
            on=region_cols,
            how="left",
            suffixes=("", "_reg"),
        )
        for yc in YEAR_COLS:
            mask = merged[yc].isna()
            merged.loc[mask, yc] = merged.loc[mask, f"{yc}_reg"]
            merged.drop(columns=[f"{yc}_reg"], inplace=True)
    else:
        merged["Region_agg2"] = merged["Region_agg2"]  # keep column for symmetry

    # Step 3: global mean per (Item, Process, Element, GHG, Unit)
    global_cols = ["Item", "Process", "Element", "GHG", "Unit"]
    global_nonempty = merged[~merged[YEAR_COLS].isna().all(axis=1)]
    if not global_nonempty.empty:
        glob_means = (
            global_nonempty.groupby(global_cols)[YEAR_COLS]
            .mean(numeric_only=True)
            .reset_index()
        )
        merged = merged.merge(
            glob_means,
            on=global_cols,
            how="left",
            suffixes=("", "_glob"),
        )
        for yc in YEAR_COLS:
            mask = merged[yc].isna()
            merged.loc[mask, yc] = merged.loc[mask, f"{yc}_glob"]
            merged.drop(columns=[f"{yc}_glob"], inplace=True)

    filled = merged.drop(columns=["Region_agg2"])
    filled[YEAR_COLS] = filled[YEAR_COLS].replace([np.inf, -np.inf], np.nan)
    return filled


# -----------------------------------------------------------------------------
# Data preparation helpers
# -----------------------------------------------------------------------------
def _load_emissions(csv_path: Path) -> pd.DataFrame:
    ensure_exists(csv_path)
    usecols = ["M49_Country_Code", "Area", "Item", "Element", "Unit", "Source"] + YEAR_COLS
    df = pd.read_csv(csv_path, usecols=usecols, dtype={"M49_Country_Code": str, "Source": str})
    df["Item"] = _clean_str(df["Item"])
    df["M49_Country_Code"] = _clean_str(df["M49_Country_Code"])
    df = df[df["Source"].eq("FAO TIER 1")]
    return df.drop(columns=["Source"])


def _load_production(csv_path: Path) -> pd.DataFrame:
    ensure_exists(csv_path)
    usecols = ["M49_Country_Code", "Area", "Item", "Element", "Unit", "Select"] + YEAR_COLS
    df = pd.read_csv(csv_path, usecols=usecols, dtype={"M49_Country_Code": str})
    df["Select"] = pd.to_numeric(df["Select"], errors="coerce")
    df = df[df["Select"].eq(1)]
    df = df[df["Element"].eq("Production")]
    df["Item"] = _clean_str(df["Item"])
    df["M49_Country_Code"] = _clean_str(df["M49_Country_Code"])
    return df.drop(columns=["Element", "Select"])


def _load_emis_item(dict_path: Path) -> pd.DataFrame:
    ensure_exists(dict_path)
    cols = [
        "Process",
        "GHG",
        "Item_Emis",
        "Item_Production_Map",
        "Item_Fertilizer_Map",
        "Item_Area_Map",
    ]
    df = pd.read_excel(dict_path, sheet_name="Emis_item", usecols=cols)
    df["Process"] = _clean_str(df["Process"])
    df["GHG"] = _clean_str(df["GHG"])
    df["Item_Emis"] = _clean_str(df["Item_Emis"])
    for col in ("Item_Production_Map", "Item_Fertilizer_Map", "Item_Area_Map"):
        df[col] = _clean_str(df[col])
    return df


def _load_fertilizer(xlsx_path: Path) -> pd.DataFrame:
    """Load fertilizer N content (Y2000–Y2020) from Fertilizer_efficiency.xlsx.
    
    返回 N 含量数据，单位: kg N
    """
    ensure_exists(xlsx_path)
    year_cols = [f"N_contentModi_Y{year}" for year in range(YEAR_START, YEAR_END + 1)]
    usecols = ["M49_Country_Code", "Item", "N_content_Unit"] + year_cols
    df = pd.read_excel(xlsx_path, usecols=usecols, dtype={"M49_Country_Code": str})
    df["Item"] = _clean_str(df["Item"])
    df["M49_Country_Code"] = _clean_str(df["M49_Country_Code"])
    for col in year_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    rename_map = {f"N_contentModi_Y{year}": f"Y{year}" for year in range(YEAR_START, YEAR_END + 1)}
    df = df.rename(columns=rename_map)
    df = df[["M49_Country_Code", "Item"] + YEAR_COLS]
    return df


def _load_fertilizer_emissions(xlsx_path: Path) -> pd.DataFrame:
    """Load fertilizer N2O emissions (Y2000–Y2020) from Fertilizer_efficiency.xlsx.
    
    读取 EmisN2O_Yxxxx 列，单位: kt N2O
    返回时转换为 kg N2O (×1e6)
    """
    ensure_exists(xlsx_path)
    # 列名格式: EmisN2O_Y2000, EmisN2O_Y2001, ...
    emis_cols = [f"EmisN2O_Y{year}" for year in range(YEAR_START, YEAR_END + 1)]
    usecols = ["M49_Country_Code", "Item"] + emis_cols
    
    df = pd.read_excel(xlsx_path, usecols=usecols, dtype={"M49_Country_Code": str})
    df["Item"] = _clean_str(df["Item"])
    df["M49_Country_Code"] = _clean_str(df["M49_Country_Code"])
    
    # 转换单位: kt → kg (×1e6)
    for col in emis_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce") * 1e6  # kt → kg
    
    rename_map = {f"EmisN2O_Y{year}": f"Y{year}" for year in range(YEAR_START, YEAR_END + 1)}
    df = df.rename(columns=rename_map)
    df = df[["M49_Country_Code", "Item"] + YEAR_COLS]
    return df


def _prepare_emission_measure(df_emis: pd.DataFrame,
                              element: str,
                              items: Iterable[str] | None,
                              unit_target: str,
                              factor_fn: Callable[[str], float]) -> pd.DataFrame:
    subset = df_emis[df_emis["Element"].eq(element)].copy()
    if items is not None:
        subset = subset[subset["Item"].isin(list(items))]
    subset = _convert_units(subset, "Unit", YEAR_COLS, factor_fn, unit_target)
    return _aggregate_by_item(subset, YEAR_COLS)


def _prepare_production_measure(prod_df: pd.DataFrame,
                                mapping: pd.DataFrame) -> pd.DataFrame:
    mapping = mapping.dropna(subset=["Item_Emis", "Item_Production_Map"]).drop_duplicates()
    subset = prod_df[prod_df["Item"].isin(mapping["Item_Production_Map"])].copy()
    if subset.empty:
        return pd.DataFrame(columns=["M49_Country_Code", "Item"] + YEAR_COLS)
    subset = _convert_units(subset, "Unit", YEAR_COLS, _tonne_factor, "t")
    subset = _aggregate_by_item(subset, YEAR_COLS)
    subset = subset.merge(
        mapping,
        left_on="Item",
        right_on="Item_Production_Map",
        how="left",
    )
    subset = subset.rename(columns={"Item": "Item_Production"})
    subset = subset.rename(columns={"Item_Emis": "Item"})
    subset = subset.drop(columns=["Item_Production", "Item_Production_Map"])
    subset = subset.dropna(subset=["Item"])
    subset = subset[["M49_Country_Code", "Item"] + YEAR_COLS]
    return subset


def _prepare_area_measure(df_emis: pd.DataFrame,
                          items: Iterable[str]) -> pd.DataFrame:
    subset = df_emis[df_emis["Element"].eq("Area harvested")].copy()
    subset = subset[subset["Item"].isin(list(items))]
    if subset.empty:
        return pd.DataFrame(columns=["M49_Country_Code", "Item"] + YEAR_COLS)
    subset = _convert_units(subset, "Unit", YEAR_COLS, _area_factor, "ha")
    return _aggregate_by_item(subset, YEAR_COLS)


def _prepare_fertilizer_measure(fert_df: pd.DataFrame,
                                mapping: pd.DataFrame) -> pd.DataFrame:
    mapping = mapping.dropna(subset=["Item_Emis", "Item_Fertilizer_Map"]).drop_duplicates()
    subset = fert_df[fert_df["Item"].isin(mapping["Item_Fertilizer_Map"])].copy()
    if subset.empty:
        return pd.DataFrame(columns=["M49_Country_Code", "Item"] + YEAR_COLS)
    subset = subset.merge(
        mapping,
        left_on="Item",
        right_on="Item_Fertilizer_Map",
        how="left",
    )
    subset = subset.rename(columns={"Item": "Item_Fertilizer"})
    subset = subset.rename(columns={"Item_Emis": "Item"})
    subset = subset.drop(columns=["Item_Fertilizer", "Item_Fertilizer_Map"])
    subset = subset.groupby(["M49_Country_Code", "Item"], as_index=False)[YEAR_COLS].sum(min_count=1)
    return subset


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------
def main() -> None:
    print("Loading input tables ...")
    emis_df = _load_emissions(EMIS_CSV)
    prod_df = _load_production(PROD_CSV)
    fert_df = _load_fertilizer(FERT_XLSX)  # N含量 (kg N)
    fert_emis_df = _load_fertilizer_emissions(FERT_XLSX)  # N2O排放 (已转换为 kg)
    emis_item = _load_emis_item(DICT_XLSX)
    region_map = _prepare_region_map(DICT_XLSX)

    outputs: List[pd.DataFrame] = []

    # ---- Crop residues ----
    crop_items = emis_item[emis_item["Process"].eq("Crop residues")].copy()
    crop_emis_items = crop_items["Item_Emis"].dropna().unique()
    crop_n_content = _prepare_emission_measure(
        emis_df, "Crop residues (N content)", crop_emis_items, "kg", _mass_to_kg_factor
    )
    crop_emissions = _prepare_emission_measure(
        emis_df, "Crop residues (Emissions N2O)", crop_emis_items, "kg", _mass_to_kg_factor
    )
    crop_ef = _compute_ratio(crop_emissions, crop_n_content, YEAR_COLS)
    crop_ef = _attach_metadata(
        crop_ef,
        process="Crop residues",
        element="Emission factor",
        ghg="N2O",
        unit="kg N2O/kg N",
    )
    outputs.append(crop_ef)

    crop_prod = _prepare_production_measure(prod_df, crop_items[["Item_Emis", "Item_Production_Map"]])
    crop_residue_content = _compute_ratio(crop_n_content, crop_prod, YEAR_COLS)
    crop_residue_content = _attach_metadata(
        crop_residue_content,
        process="Crop residues",
        element="Residue N content",
        ghg="N2O",
        unit="kg N / tonne product",
    )
    outputs.append(crop_residue_content)

    # ---- Burning crop residues ----
    burning_items = emis_item[emis_item["Process"].eq("Burning crop residues")].copy()
    burn_emis_items = burning_items["Item_Emis"].dropna().unique()
    burn_biomass = _prepare_emission_measure(
        emis_df,
        "Burning crop residues (Biomass burned, dry matter)",
        burn_emis_items,
        "kg",
        _mass_to_kg_factor,
    )
    burn_prod = _prepare_production_measure(
        prod_df,
        burning_items[["Item_Emis", "Item_Production_Map"]],
    )

    for ghg, element_name, unit in [
        ("CH4", "Burning crop residues (Emissions CH4)", "kg CH4/kg DM"),
        ("N2O", "Burning crop residues (Emissions N2O)", "kg N2O/kg DM"),
    ]:
        emis = _prepare_emission_measure(
            emis_df,
            element_name,
            burn_emis_items,
            "kg",
            _mass_to_kg_factor,
        )
        ef = _compute_ratio(emis, burn_biomass, YEAR_COLS)
        ef = _attach_metadata(
            ef,
            process="Burning crop residues",
            element="Emission factor",
            ghg=ghg,
            unit=unit,
        )
        outputs.append(ef)

    burn_dm_content = _compute_ratio(burn_biomass, burn_prod, YEAR_COLS)
    burn_dm_content = _attach_metadata(
        burn_dm_content,
        process="Burning crop residues",
        element="Biomass burning DM content",
        ghg="CH4_and_N2O",
        unit="kg DM / tonne product",
    )
    outputs.append(burn_dm_content)

    # ---- Rice cultivation ----
    rice_items = emis_item[emis_item["Process"].eq("Rice cultivation")]
    rice_item_names = rice_items["Item_Emis"].dropna().unique()
    rice_emis = _prepare_emission_measure(
        emis_df,
        "Rice cultivation (Emissions CH4)",
        rice_item_names,
        "kg",
        _mass_to_kg_factor,
    )
    rice_area = _prepare_area_measure(emis_df, rice_item_names)
    rice_ef = _compute_ratio(rice_emis, rice_area, YEAR_COLS)
    rice_ef = _attach_metadata(
        rice_ef,
        process="Rice cultivation",
        element="Emission factor",
        ghg="CH4",
        unit="kg CH4/ha",
    )
    outputs.append(rice_ef)

    # ---- Synthetic fertilizers ----
    # 从 Fertilizer_efficiency.xlsx 读取数据：
    # - fert_df: N含量 (kg N)
    # - fert_emis_df: N2O排放 (已转换为 kg，原始单位 kt)
    fert_items = emis_item[emis_item["Process"].eq("Synthetic fertilizers")].copy()
    fert_item_names = fert_items["Item_Emis"].dropna().unique()
    
    # N含量 (kg N)
    fert_amount = _prepare_fertilizer_measure(
        fert_df,
        fert_items[["Item_Emis", "Item_Fertilizer_Map"]],
    )
    
    # N2O排放 (kg) - 优先使用 Fertilizer_efficiency.xlsx 中的 EmisN2O 列
    if not fert_emis_df.empty:
        fert_emis = _prepare_fertilizer_measure(
            fert_emis_df,
            fert_items[["Item_Emis", "Item_Fertilizer_Map"]],
        )
        print(f"[INFO] Synthetic fertilizers: 使用 Fertilizer_efficiency.xlsx 中的 EmisN2O 数据")
    else:
        # 回退到 FAO 排放数据
        print("[WARNING] Fertilizer_efficiency.xlsx 中没有 EmisN2O 列，回退到 FAO 排放数据")
        fert_emis = _prepare_emission_measure(
            emis_df,
            "Synthetic fertilizers (Emissions N2O)",
            items=fert_item_names,
            unit_target="kg",
            factor_fn=_mass_to_kg_factor,
        )

    if fert_emis.empty:
        print("Warning: No synthetic fertilizer emissions found.")
        fert_emis = pd.DataFrame(columns=["M49_Country_Code", "Item"] + YEAR_COLS)

    # 排放因子 = 排放量(kg N2O) / 施肥量(kg N) = kg N2O/kg N
    fert_ef = _compute_ratio(fert_emis, fert_amount, YEAR_COLS)
    fert_ef = _attach_metadata(
        fert_ef,
        process="Synthetic fertilizers",
        element="Emission factor",
        ghg="N2O",
        unit="kg N2O/kg N",
    )
    outputs.append(fert_ef)

    combined = pd.concat(outputs, ignore_index=True)
    combined = _fill_by_rules(combined, region_map)
    combined = combined.sort_values(["Process", "GHG", "Item", "M49_Country_Code"]).reset_index(drop=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(
        {
            "output_csv": str(OUT_CSV),
            "n_rows": int(combined.shape[0]),
            "n_processes": combined["Process"].nunique(),
        }
    )


if __name__ == "__main__":
    main()
