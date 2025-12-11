# -*- coding: utf-8 -*-
"""S0_19 – build a cleaned horizon of historical max production by region×commodity.

Valid M49_Country_Code values are those listed in `dict_v3.xlsx` → `region` sheet where
`Region_label_new` is not "no"; the corresponding Region_label_new defines each country’s region.
`Item` values come from `dict_v3.xlsx` → `Emis_item` → `Item_Production_Map`, and each row
specifies which FAO production file (column `Production_file_source`, all under
`input/Production_Trade`) to read. We read those files, convert units to tonnes, and keep
2010-2020 history where `Element == "Production"` and `Select == 1`.

The script then builds the full orthogonal set of (valid M49, Item) combinations. For each
combo, if historical data exist we keep the actual 2010-2020 maximum, otherwise we fill the
missing value using the median max from other countries in the same region that do have data.
The output lands in `input/Production_Trade/S0_19_historical_max_production.csv`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from config_paths import get_input_base, get_src_base

YEAR_START = 2010
YEAR_END = 2020
YEAR_COLS = [f"Y{year}" for year in range(YEAR_START, YEAR_END + 1)]
DICT_PATH = Path(get_src_base()) / "dict_v3.xlsx"
OUTPUT_PATH = Path(get_input_base()) / "Production_Trade" / "S0_19_historical_max_production.csv"


def _normalize_m49_value(raw: str) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _tonne_factor(unit: str) -> float:
    if not isinstance(unit, str):
        return 1.0
    u = unit.strip().lower()
    if not u or u == "nan":
        return 1.0
    if "1000" in u and ("ton" in u or "tonne" in u):
        return 1e3
    if "kiloton" in u or "kt" in u:
        return 1e6
    if "tonne" in u or ("ton" in u and not u.startswith("t")):
        return 1.0
    if u in {"t", "ton", "tons"} or u.endswith(" t"):
        return 1.0
    if "kg" in u:
        return 1e-3
    if "g" in u and "kg" not in u:
        return 1e-6
    return 1.0


def _load_valid_regions(path: Path) -> dict[str, str]:
    df = pd.read_excel(path, sheet_name="region", usecols=["M49_Country_Code", "Region_label_new"])
    df["m49_norm"] = df["M49_Country_Code"].apply(_normalize_m49_value)
    df = df[df["Region_label_new"].notna()]
    df = df[df["Region_label_new"].str.lower() != "no"]
    df = df[df["m49_norm"].notna()].drop_duplicates(subset=["m49_norm"])
    return dict(zip(df["m49_norm"], df["Region_label_new"]))


def _load_items(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Emis_item", usecols=["Item_Emis", "Item_Production_Map", "Production_file_source"])
    df = df[df["Item_Production_Map"].notna() & df["Production_file_source"].notna()].copy()
    df["Item_Production_Map"] = df["Item_Production_Map"].astype(str).str.strip()
    df["Production_file_source"] = df["Production_file_source"].astype(str).str.strip()
    df = df[df["Item_Production_Map"] != ""]
    return df.drop_duplicates(subset=["Item_Production_Map", "Production_file_source"])[["Item_Production_Map", "Production_file_source"]]


def _read_production(path: Path, year_cols: Sequence[str], items: Iterable[str] | None = None) -> pd.DataFrame:
    preview = pd.read_csv(path, nrows=0)
    available = list(preview.columns)
    need = ["M49_Country_Code", "Item", "Element", "Unit"]
    missing = [c for c in need if c not in available]
    if missing:
        raise ValueError(f"Production file {path} missing columns: {missing}")
    usecols = need + list(year_cols)
    if "Select" in available:
        usecols.append("Select")
    df = pd.read_csv(path, usecols=usecols, dtype=str, low_memory=False)
    df["Element"] = df["Element"].astype(str).str.strip()
    df = df[df["Element"] == "Production"]
    if "Select" in df.columns:
        df["Select"] = pd.to_numeric(df["Select"], errors="coerce")
        df = df[df["Select"] == 1]
    drop_cols = ["Element", "Select"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df["M49_Country_Code"] = df["M49_Country_Code"].apply(_normalize_m49_value)
    df["Item"] = df["Item"].astype(str).str.strip()
    df = df[df["M49_Country_Code"].notna() & df["Item"].astype(bool)]
    if items is not None:
        df = df[df["Item"].isin(items)]
    if df.empty:
        return df
    for col in year_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    factor = df["Unit"].map(lambda v: _tonne_factor(str(v)))
    df[year_cols] = df[year_cols].multiply(factor.fillna(1.0), axis=0)
    return df


def _aggregate_max(df: pd.DataFrame, year_cols: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["M49_Country_Code", "Item", "max_year", "max_production_t"])
    grouped = (
        df.groupby(["M49_Country_Code", "Item"], dropna=False, as_index=False)[list(year_cols)]
        .sum(min_count=1)
    )
    valid = ~grouped[list(year_cols)].isna().all(axis=1)
    grouped = grouped[valid].reset_index(drop=True)
    if grouped.empty:
        return pd.DataFrame(columns=["M49_Country_Code", "Item", "max_year", "max_production_t"])
    grouped["max_production_t"] = grouped[list(year_cols)].max(axis=1, skipna=True)
    max_year = grouped[list(year_cols)].idxmax(axis=1)
    grouped["max_year"] = max_year.str.lstrip("Y").astype(int)
    return grouped[["M49_Country_Code", "Item", "max_year", "max_production_t"]]


def _build_full_grid(m49_codes: Sequence[str], items: Sequence[str]) -> pd.DataFrame:
    df = pd.MultiIndex.from_product([m49_codes, items], names=["M49_Country_Code", "Item"]).to_frame(index=False)
    return df


def main() -> None:
    if not DICT_PATH.exists():
        raise FileNotFoundError(f"Missing dict_v3 dictionary at {DICT_PATH}")
    regions = _load_valid_regions(DICT_PATH)
    if not regions:
        raise ValueError("No valid regions found in dict_v3 region sheet.")
    items_df = _load_items(DICT_PATH)
    if items_df.empty:
        raise ValueError("No production items found in dict_v3 Emis_item sheet.")
    items = items_df["Item_Production_Map"].unique().tolist()
    item_to_file = dict(zip(items_df["Item_Production_Map"], items_df["Production_file_source"]))
    production_buffers: list[pd.DataFrame] = []
    file_to_items: dict[str, list[str]] = defaultdict(list)
    for itm, src in items_df.itertuples(index=False):
        file_to_items[src].append(itm)

    # --- Start of new logging section ---
    all_items_with_data = set()
    for file_name, item_subset in file_to_items.items():
        file_path = Path(get_input_base()) / "Production_Trade" / file_name
        if not file_path.exists():
            print(f"Warning: production file {file_path} not found; skipping these items: {', '.join(item_subset)}")
            continue
        frame = _read_production(file_path, YEAR_COLS, item_subset)
        if not frame.empty:
            production_buffers.append(frame)
            found_items = set(frame["Item"].unique())
            all_items_with_data.update(found_items)

    all_configured_items = set(items)
    items_with_no_data = all_configured_items - all_items_with_data
    if items_with_no_data:
        print(f"Warning: No production data was found for the following items: {', '.join(sorted(items_with_no_data))}")
    # --- End of new logging section ---

    if not production_buffers:
        raise FileNotFoundError("No production data could be read for the configured items.")
    all_prod = pd.concat(production_buffers, ignore_index=True, sort=False)
    max_df = _aggregate_max(all_prod, YEAR_COLS)

    valid_m49 = sorted(regions.keys())
    grid = _build_full_grid(valid_m49, sorted(items))
    grid["region_label"] = grid["M49_Country_Code"].map(regions)
    grid["Production_file_source"] = grid["Item"].map(item_to_file)
    merged = grid.merge(max_df, on=["M49_Country_Code", "Item"], how="left")
    merged["source"] = np.where(merged["max_production_t"].notna(), "observed", None)
    merged["max_year"] = merged["max_year"].astype("Int64")

    # Step 1: Use map() to get the median for each (region, item) combination
    # Filter out NaN and 0 values before computing median
    valid_mask = (merged["max_production_t"].notna()) & (merged["max_production_t"] > 0)
    region_item_medians = merged[valid_mask].groupby(["region_label", "Item"])['max_production_t'].median()
    
    # Create a multi-index lookup for region medians
    def get_region_median(row):
        try:
            return region_item_medians.loc[(row["region_label"], row["Item"])]
        except KeyError:
            return np.nan
    
    merged["region_median"] = merged.apply(get_region_median, axis=1)

    # Step 2: Fill missing values with region median where available
    mask_region_fill = (merged["max_production_t"].isna() | (merged["max_production_t"] == 0)) & merged["region_median"].notna()
    merged.loc[mask_region_fill, "max_production_t"] = merged.loc[mask_region_fill, "region_median"]
    merged.loc[mask_region_fill, "max_year"] = 2020
    merged.loc[mask_region_fill, "source"] = "region_median"

    # Step 3: Fallback - for items still missing, use the global median for that item
    # Filter out NaN and 0 values before computing global median
    global_valid_mask = (merged["max_production_t"].notna()) & (merged["max_production_t"] > 0)
    global_item_medians = merged[global_valid_mask].groupby("Item")['max_production_t'].median()
    merged["global_median"] = merged["Item"].map(global_item_medians)

    mask_global_fill = (merged["max_production_t"].isna() | (merged["max_production_t"] == 0)) & merged["global_median"].notna()
    merged.loc[mask_global_fill, "max_production_t"] = merged.loc[mask_global_fill, "global_median"]
    merged.loc[mask_global_fill, "max_year"] = 2020
    merged.loc[mask_global_fill, "source"] = "global_median"

    # Clean up temporary columns
    merged = merged.drop(columns=["region_median", "global_median"])

    merged["unit"] = "tonnes"
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    
    total_missing = merged["max_production_t"].isna().sum()
    observed_count = (merged["source"] == "observed").sum()
    region_median_count = (merged["source"] == "region_median").sum()
    global_median_count = (merged["source"] == "global_median").sum()
    
    print(f"Saved {len(merged)} rows ({len(valid_m49)} M49 × {len(items)} items) to {OUTPUT_PATH}")
    print(f"  - From observed data: {observed_count} ({observed_count*100.0/len(merged):.1f}%)")
    print(f"  - Filled with region median: {region_median_count} ({region_median_count*100.0/len(merged):.1f}%)")
    print(f"  - Filled with global median: {global_median_count} ({global_median_count*100.0/len(merged):.1f}%)")
    
    if total_missing > 0:
        print(f"\n⚠️  Warning: {total_missing} combos still missing (no data for these items globally)")
    else:
        print(f"\n✓ All missing values successfully filled!")


if __name__ == "__main__":
    main()
