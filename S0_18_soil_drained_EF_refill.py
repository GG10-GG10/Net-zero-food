# -*- coding: utf-8 -*-
"""
S0.18 soil drained organic soils parameter back-calculation.

Outputs:
    Code/input/Emission/soil_drained_parameters.xlsx
        - sheet 'Emission factor': Cropland/Grassland organic soils EF for CO2 & N2O
        - sheet 'Area correlation': area-share parameters Cropland/Pasture
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd

from config_paths import get_input_base, get_src_base

YEAR_RANGE = (2000, 2022)
YEAR_COLS = [f"Y{y}" for y in range(YEAR_RANGE[0], YEAR_RANGE[1] + 1)]
PROCESS_NAME = "Drained organic soils"

EMISSION_ELEMENTS: Dict[str, str] = {
    "Drained organic soils (N2O)": "N2O",
    "Drained organic soils (CO2)": "CO2",
}

TARGET_SOIL_ITEMS = ("Cropland organic soils", "Grassland organic soils")

AREA_CORR_MAPPING: Dict[str, str] = {
    "Cropland organic soils": "Cropland",
    "Grassland organic soils": "Pasture land",
}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    input_base = Path(get_input_base())
    src_base = Path(get_src_base())
    ap = argparse.ArgumentParser(description="Back-calculate drained organic soil EF and area correlation")
    ap.add_argument("--emission-file", type=Path,
                    default=input_base / "Emission" / "Emissions_Drained_Organic_Soils_E_All_Data_NOFLAG.csv")
    ap.add_argument("--land-file", type=Path,
                    default=input_base / "Land" / "Inputs_LandUse_E_All_Data_NOFLAG_with_Pasture.csv")
    ap.add_argument("--dict-file", type=Path, default=src_base / "dict_v3.xlsx")
    ap.add_argument("--out-file", type=Path,
                    default=src_base / "soil_drained_parameters.xlsx")
    ap.add_argument("--year-start", type=int, default=YEAR_RANGE[0])
    ap.add_argument("--year-end", type=int, default=YEAR_RANGE[1])
    return ap.parse_args(list(argv) if argv is not None else None)


def melt_years(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    out = df.melt(
        id_vars=["M49_Country_Code", "Item"],
        value_vars=[c for c in df.columns if c.startswith("Y")],
        var_name="Year",
        value_name=value_name,
    )
    out[value_name] = pd.to_numeric(out[value_name], errors="coerce")
    return out


def load_region_map(dict_path: Path) -> pd.DataFrame:
    region = pd.read_excel(dict_path, sheet_name="region", usecols=["M49_Country_Code", "Region_agg2"])
    region = region.dropna(subset=["M49_Country_Code"]).copy()
    region["M49_Country_Code"] = region["M49_Country_Code"].astype(str).str.strip()
    return region


def load_emission_data(path: Path, year_cols: List[str]) -> pd.DataFrame:
    cols = ["M49_Country_Code", "Item", "Element", "Source", "Unit"] + year_cols
    df = pd.read_csv(path, usecols=lambda c: c in cols)
    df = df[df["Source"].str.strip().eq("FAO TIER 1")].copy()
    df["Item"] = df["Item"].str.strip()
    df["Element"] = df["Element"].str.strip()
    df["M49_Country_Code"] = df["M49_Country_Code"].astype(str).str.strip()
    return df


def compute_emission_factors(em_df: pd.DataFrame, year_cols: List[str]) -> pd.DataFrame:
    area = em_df[(em_df["Element"] == "Area") & (em_df["Item"].isin(TARGET_SOIL_ITEMS))].copy()
    factor_tables: List[pd.DataFrame] = []
    for element, ghg in EMISSION_ELEMENTS.items():
        emis = em_df[(em_df["Element"] == element) & (em_df["Item"].isin(TARGET_SOIL_ITEMS))].copy()
        if emis.empty:
            continue
        emis_years = emis[["M49_Country_Code", "Item"] + year_cols].copy()
        area_years = area[["M49_Country_Code", "Item"] + year_cols].copy()
        emis_long = melt_years(emis_years, "emission_kt")
        area_long = melt_years(area_years, "area_ha")
        merged = emis_long.merge(area_long, on=["M49_Country_Code", "Item", "Year"], how="left")
        merged["emission_t"] = merged["emission_kt"] * 1_000.0
        merged["value"] = merged["emission_t"] / merged["area_ha"]
        pivot = (
            merged.pivot_table(index=["M49_Country_Code", "Item"], columns="Year", values="value", aggfunc="mean")
            .reset_index()
        )
        for col in pivot.columns:
            if col.startswith("Y"):
                pivot[col] = pd.to_numeric(pivot[col], errors="coerce")
        pivot["Process"] = PROCESS_NAME
        pivot["Element"] = "Emission factor"
        pivot["GHG"] = ghg
        pivot["Unit"] = "t/ha"
        factor_tables.append(pivot)
    if not factor_tables:
        return pd.DataFrame(columns=["M49_Country_Code", "Item", "Process", "Element", "GHG", "Unit"] + year_cols)
    out = pd.concat(factor_tables, ignore_index=True)
    keep_cols = ["M49_Country_Code", "Item", "Process", "Element", "GHG", "Unit"] + year_cols
    return out[keep_cols]


def compute_area_correlation(
    em_df: pd.DataFrame, land_df: pd.DataFrame, year_cols: List[str]
) -> pd.DataFrame:
    area = em_df[(em_df["Element"] == "Area") & (em_df["Item"].isin(AREA_CORR_MAPPING.keys()))]
    land = land_df[
        (land_df["Element"].str.strip() == "Area") & (land_df["Item"].isin(AREA_CORR_MAPPING.values()))
    ].copy()
    land_unit = land["Unit"].dropna().unique()
    multiplier = 1_000.0 if any("1000" in str(u) for u in land_unit) else 1.0
    corr_tables: List[pd.DataFrame] = []
    for soil_item, land_item in AREA_CORR_MAPPING.items():
        soil_area = area[area["Item"] == soil_item][["M49_Country_Code", "Item"] + year_cols].copy()
        land_area = land[land["Item"] == land_item][["M49_Country_Code"] + year_cols].copy()
        if soil_area.empty or land_area.empty:
            continue
        soil_long = melt_years(soil_area, "soil_area")
        land_long = land_area.melt(id_vars=["M49_Country_Code"], value_vars=year_cols,
                                   var_name="Year", value_name="land_area")
        land_long["land_area"] = pd.to_numeric(land_long["land_area"], errors="coerce") * multiplier
        merged = soil_long.merge(land_long, on=["M49_Country_Code", "Year"], how="left")
        merged["value"] = merged["soil_area"] / merged["land_area"]
        pivot = (
            merged.pivot_table(index=["M49_Country_Code", "Item"], columns="Year", values="value", aggfunc="mean")
            .reset_index()
        )
        for col in pivot.columns:
            if col.startswith("Y"):
                pivot[col] = pd.to_numeric(pivot[col], errors="coerce")
        pivot["Process"] = PROCESS_NAME
        pivot["Element"] = "Area correlation"
        pivot["GHG"] = "N2O_CO2"
        pivot["Unit"] = "ratio"
        corr_tables.append(pivot)
    if not corr_tables:
        return pd.DataFrame(columns=["M49_Country_Code", "Item", "Process", "Element", "GHG", "Unit"] + year_cols)
    out = pd.concat(corr_tables, ignore_index=True)
    keep_cols = ["M49_Country_Code", "Item", "Process", "Element", "GHG", "Unit"] + year_cols
    return out[keep_cols]


def fill_gaps(df: pd.DataFrame, year_cols: List[str], region_map: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out = out.merge(region_map, on="M49_Country_Code", how="left")
    values = out[year_cols].apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)
    values = values.mask(values == 0)
    values = values.ffill(axis=1).bfill(axis=1)
    region_keys = ["Item", "Process", "Element", "GHG", "Region_agg2"]
    region_means = values.groupby([out[k] for k in region_keys]).transform("mean")
    values = values.fillna(region_means)
    global_keys = ["Item", "Process", "Element", "GHG"]
    global_means = values.groupby([out[k] for k in global_keys]).transform("mean")
    values = values.fillna(global_means)
    out[year_cols] = values
    return out.drop(columns=["Region_agg2"])


def trim_years(df: pd.DataFrame, year_cols: List[str]) -> pd.DataFrame:
    keep = ["M49_Country_Code", "Item", "Process", "Element", "GHG", "Unit"] + year_cols
    missing = [c for c in keep if c not in df.columns]
    if missing:
        for col in missing:
            df[col] = np.nan
    return df[keep]


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    year_start, year_end = sorted([args.year_start, args.year_end])
    year_cols = [f"Y{y}" for y in range(year_start, year_end + 1)]

    em_df = load_emission_data(args.emission_file, year_cols)
    land_cols = ["M49_Country_Code", "Item", "Element", "Unit"] + year_cols
    land_df = pd.read_csv(args.land_file, usecols=lambda c: c in land_cols)
    land_df["Item"] = land_df["Item"].str.strip()
    land_df["M49_Country_Code"] = land_df["M49_Country_Code"].astype(str).str.strip()

    region_map = load_region_map(args.dict_file)

    ef_df = compute_emission_factors(em_df, year_cols)
    ef_df = fill_gaps(ef_df, year_cols, region_map)
    ef_df = trim_years(ef_df, year_cols)

    corr_df = compute_area_correlation(em_df, land_df, year_cols)
    corr_df = fill_gaps(corr_df, year_cols, region_map)
    corr_df = trim_years(corr_df, year_cols)

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        ef_df.to_excel(writer, sheet_name="Emission factor", index=False)
        corr_df.to_excel(writer, sheet_name="Area correlation", index=False)
    print(f"[INFO] Wrote results to {out_path}")


if __name__ == "__main__":
    main()
