# -*- coding: utf-8 -*-
"""
S0.17 LUH2 state & transition aggregator.

读取 LUH2 mask / state / transition 数据，依据国家掩膜汇总：
1) 2010-2020 年 cropland、grassland、forest 面积；
2) forest↔cropland/pasture 四类粗分类土地转移面积。

结果写出 `Code/input/Land/LUH2_data_summary.xlsx`
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from config_paths import get_input_base, get_src_base

# ---- Land-cover + transition mappings (aligned with LUH bookkeeping module) ----
LAND_COVER_GROUPS: Dict[str, Sequence[str]] = {
    "cropland": ("c3ann", "c3nfx", "c3per", "c4ann", "c4per"),
    "grassland": ("pastr", "range"),
    "forest": ("primf", "secdf"),
}

STATE_TO_CATEGORY: Dict[str, str] = {
    "primf": "forest",
    "secdf": "forest",
    "primn": "othernat",
    "secdn": "othernat",
    "secma": "forest",
    "secmb": "forest",
    "range": "pasture",  # treat range as grassland/pasture for coarse stats
    "pastr": "pasture",
    "urban": "urban",
    "crop": "cropland",
    "c3ann": "cropland",
    "c3per": "cropland",
    "c3nfx": "cropland",
    "c4ann": "cropland",
    "c4per": "cropland",
}

TRANSITION_LABELS: Dict[tuple[str, str], str] = {
    ("forest", "cropland"): "forest_to_cropland",
    ("forest", "pasture"): "forest_to_pasture",
    ("cropland", "forest"): "cropland_to_forest",
    ("pasture", "forest"): "pasture_to_forest",
}

TRANSITION_RE = re.compile(r"^(?P<from>[^_]+)_to_(?P<to>[^_]+)$")


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def _choose_existing_path(paths: Iterable[Path]) -> Path:
    candidates = [Path(p) for p in paths if p]
    if not candidates:
        raise ValueError("No candidate paths provided")
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def estimate_area_ha(ds: xr.Dataset) -> xr.DataArray:
    """Return cell area (ha) for a LUH2 grid, using `areacella` if available."""
    if "areacella" in ds:
        return ds["areacella"] * 1e-4

    R = 6_371_000.0
    lat_vals = np.asarray(ds["lat"].values, dtype=float)
    lon_vals = np.asarray(ds["lon"].values, dtype=float)
    if lat_vals.size < 2 or lon_vals.size < 2:
        raise ValueError("lat/lon dimensions are insufficient to estimate cell area")
    dlat = np.deg2rad(abs(float(lat_vals[1] - lat_vals[0])))
    dlon = np.deg2rad(abs(float(lon_vals[1] - lon_vals[0])))
    lat_r = np.deg2rad(lat_vals)
    strip = (np.sin(lat_r + dlat / 2.0) - np.sin(lat_r - dlat / 2.0)) * (R**2) * dlon
    area = np.repeat(strip[:, None], lon_vals.size, axis=1)
    return xr.DataArray(area, coords={"lat": ds["lat"], "lon": ds["lon"]}, dims=("lat", "lon")) * 1e-4


def ensure_year_dim(ds: xr.Dataset) -> xr.Dataset:
    """Ensure dataset uses `year` dimension (int)."""
    if "year" in ds.dims:
        years = [int(y) for y in ds["year"].values]
        return ds.assign_coords(year=("year", years))
    if "time" not in ds.coords:
        raise KeyError("Dataset must contain 'year' or 'time' coordinate")
    years: List[int] = []
    for t in ds["time"].values:
        year = getattr(t, "year", None)
        if year is None:
            year = pd.to_datetime(t).year
        years.append(int(year))
    ds = ds.assign_coords(year=("time", years)).swap_dims({"time": "year"}).sortby("year")
    return ds


def load_mask_array(mask_path: Path) -> np.ndarray:
    ds = xr.open_dataset(mask_path)
    try:
        if "id1" not in ds:
            raise KeyError("Mask file must contain variable 'id1'")
        mask = np.asarray(ds["id1"].values, dtype=float)
        mask = np.nan_to_num(mask, nan=0.0)
        return mask.astype(np.int64)
    finally:
        ds.close()


def load_mask_lookup(dict_path: Path) -> pd.DataFrame:
    region_df = pd.read_excel(dict_path, sheet_name="region")
    region_df.columns = [str(c).strip() for c in region_df.columns]
    expected = {"Region_maskID", "M49_Country_Code"}
    missing = expected - set(region_df.columns)
    if missing:
        raise KeyError(f"Sheet 'region' missing columns: {missing}")
    out = (
        region_df.loc[:, ["Region_maskID", "M49_Country_Code"]]
        .dropna(subset=["Region_maskID"])
        .copy()
    )
    out["mask_ID"] = pd.to_numeric(out["Region_maskID"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["mask_ID"]).drop_duplicates(subset=["mask_ID"])
    out["mask_ID"] = out["mask_ID"].astype(int)

    def _format_code(val):
        if pd.isna(val):
            return pd.NA
        s = str(val).strip()
        if not s or s.lower() in {"nan", "none"}:
            return pd.NA
        return s if s.startswith("'") else f"'{s}"

    out["M49_Country_Code"] = out["M49_Country_Code"].apply(_format_code)
    return out[["mask_ID", "M49_Country_Code"]]


def reindex_years(data: xr.DataArray, years: Sequence[int]) -> xr.DataArray:
    data = data.transpose("lat", "lon", "year")
    return data.reindex(year=list(years), fill_value=0.0)


def aggregate_by_mask(data: xr.DataArray, mask: np.ndarray, years: Sequence[int]) -> pd.DataFrame:
    data = reindex_years(data, years)
    lat_dim, lon_dim = data.sizes["lat"], data.sizes["lon"]
    if mask.shape != (lat_dim, lon_dim):
        raise ValueError("Mask grid does not match LUH2 grid")

    arr = data.transpose("lat", "lon", "year").values.astype(np.float64)
    n_years = len(years)
    flat = arr.reshape(-1, n_years)
    mask_flat = mask.reshape(-1)
    valid = (mask_flat > 0) & np.isfinite(mask_flat)
    flat = flat[valid]
    mask_ids = mask_flat[valid].astype(np.int64)
    year_cols = [f"Y{y}" for y in years]
    df = pd.DataFrame(flat, columns=year_cols)
    df.insert(0, "mask_ID", mask_ids)
    grouped = df.groupby("mask_ID", as_index=False)[year_cols].sum()
    value_cols = year_cols
    grouped = grouped[grouped[value_cols].abs().sum(axis=1) > 0.0]
    return grouped


def compute_land_cover_tables(
    ds: xr.Dataset,
    area_ha: xr.DataArray,
    mask: np.ndarray,
    years: Sequence[int],
    lookup: pd.DataFrame,
) -> pd.DataFrame:
    tables: List[pd.DataFrame] = []
    for cover, vars_ in LAND_COVER_GROUPS.items():
        missing = [v for v in vars_ if v not in ds]
        if missing:
            raise KeyError(f"State dataset missing variables for {cover}: {missing}")
        arrays = [ds[v] for v in vars_]
        frac = arrays[0].copy()
        for arr in arrays[1:]:
            frac = frac + arr
        cover_area = (frac * area_ha).astype(np.float64)
        aggregated = aggregate_by_mask(cover_area, mask, years)
        aggregated["Land cover"] = cover
        tables.append(aggregated)
    if not tables:
        return pd.DataFrame()
    land_df = pd.concat(tables, ignore_index=True)
    land_df = land_df.merge(lookup, on="mask_ID", how="left")
    land_df["Unit"] = "ha"
    cols = ["mask_ID", "M49_Country_Code", "Land cover", "Unit"] + [f"Y{y}" for y in years]
    land_df = land_df[cols].sort_values(["mask_ID", "Land cover"]).reset_index(drop=True)
    return land_df


def compute_transition_tables(
    ds: xr.Dataset,
    area_ha: xr.DataArray,
    mask: np.ndarray,
    years: Sequence[int],
    lookup: pd.DataFrame,
) -> pd.DataFrame:
    results: MutableMapping[str, xr.DataArray] = {}
    for var in ds.data_vars:
        if var == "time_bnds":
            continue
        match = TRANSITION_RE.match(var)
        if not match:
            continue
        f_state, t_state = match.group("from"), match.group("to")
        label = TRANSITION_LABELS.get((STATE_TO_CATEGORY.get(f_state), STATE_TO_CATEGORY.get(t_state)))
        if not label:
            continue
        area = (ds[var] * area_ha).astype(np.float64)
        results[label] = area if label not in results else results[label] + area

    if not results:
        return pd.DataFrame()

    tables: List[pd.DataFrame] = []
    for label, data in results.items():
        aggregated = aggregate_by_mask(data, mask, years)
        aggregated["LUC type"] = label
        tables.append(aggregated)

    luc_df = pd.concat(tables, ignore_index=True)
    luc_df = luc_df.merge(lookup, on="mask_ID", how="left")
    luc_df["Unit"] = "ha"
    cols = ["mask_ID", "M49_Country_Code", "LUC type", "Unit"] + [f"Y{y}" for y in years]
    luc_df = luc_df[cols].sort_values(["mask_ID", "LUC type"]).reset_index(drop=True)
    return luc_df


def write_excel(out_path: Path, land_df: pd.DataFrame, luc_df: pd.DataFrame) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        land_df.to_excel(writer, sheet_name="Land cover", index=False)
        luc_df.to_excel(writer, sheet_name="LUC", index=False)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    input_base = Path(get_input_base())
    src_base = Path(get_src_base())
    default_data_dirs = [
        Path(r"C:\Users\cheng\MyDrive\Work\Net-zero food\Code\input\Land\LUH2_GCB2019\data"),
        input_base / "Land" / "LUH2_GCB2019" / "data",
    ]

    ap = argparse.ArgumentParser(description="Aggregate LUH2 land-cover and transition areas by mask ID")
    ap.add_argument("--data-dir", type=Path, default=None, help="Directory containing LUH2 files")
    ap.add_argument("--mask-file", type=Path, default=None, help="mask_LUH2_025d.nc path")
    ap.add_argument("--states-file", type=Path, default=None, help="LUH2 states subset file")
    ap.add_argument("--transitions-file", type=Path, default=None, help="LUH2 transitions subset file")
    ap.add_argument("--dict-file", type=Path, default=src_base / "dict_v3.xlsx", help="dict_v3.xlsx path")
    ap.add_argument("--out-file", type=Path,
                    default=input_base / "Land" / "LUH2_data_summary.xlsx",
                    help="Output Excel path")
    ap.add_argument("--start-year", type=int, default=2010)
    ap.add_argument("--end-year", type=int, default=2020)
    args = ap.parse_args(list(argv) if argv is not None else None)

    data_dir = args.data_dir or _choose_existing_path(default_data_dirs)
    args.data_dir = data_dir
    args.mask_file = args.mask_file or data_dir / "mask_LUH2_025d.nc"
    args.states_file = args.states_file or data_dir / "LUH2_GCB2019_states_2010_2020.nc4"
    args.transitions_file = args.transitions_file or data_dir / "LUH2_GCB2019_transitions_2010_2020.nc4"
    return args


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    years = list(range(min(args.start_year, args.end_year), max(args.start_year, args.end_year) + 1))

    mask = load_mask_array(Path(args.mask_file))
    lookup = load_mask_lookup(Path(args.dict_file))

    states_vars = sorted({var for vars_ in LAND_COVER_GROUPS.values() for var in vars_})
    with xr.open_dataset(args.states_file) as ds_states:
        ds_states = ensure_year_dim(ds_states)
        states_subset = ds_states[states_vars]
        area_ha = estimate_area_ha(ds_states)
        land_df = compute_land_cover_tables(states_subset, area_ha, mask, years, lookup)

    with xr.open_dataset(args.transitions_file) as ds_trans:
        ds_trans = ensure_year_dim(ds_trans)
        transitions_df = compute_transition_tables(ds_trans, area_ha, mask, years, lookup)

    out_path = Path(args.out_file)
    write_excel(out_path, land_df, transitions_df)
    print(f"[INFO] Wrote summary to {out_path}")


if __name__ == "__main__":
    main()
