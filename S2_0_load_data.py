# -*- coding: utf-8 -*-
"""
S2.0_load_data — 数据读取与构造（对齐 dict_v3 + FAOSTAT 文件 + 情景管道）
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any
from collections import defaultdict
import re
import numpy as np
import pandas as pd
import os
import xarray as xr
from config_paths import get_input_base, get_src_base

from S1_0_schema import Universe, ScenarioConfig, Node

# -------------------- paths --------------------
@dataclass
class DataPaths:
    base: str = get_input_base()
    # config/dictionaries under src
    dict_v3_path: str = os.path.join(get_src_base(), "dict_v3.xlsx")
    scenario_config_xlsx: str = os.path.join(get_src_base(), "Scenario_config_new.xlsx")
    elasticity_xlsx: str = os.path.join(get_src_base(), "Elasticity_v3_processed_filled_by_region.xlsx")
    feed_coeff_xlsx: str = os.path.join(get_src_base(), "unit_feed_crops_per_head_region_system_2010_2020_v2_imputed.xlsx")
    # inputs
    production_faostat_csv: str = os.path.join(get_input_base(), "Production_Trade", "Production_Crops_Livestock_E_All_Data_NOFLAG.csv")
    fbs_csv: str = os.path.join(get_input_base(), "Production_Trade", "FoodBalanceSheets_E_All_Data_NOFLAG.csv")
    livestock_patterns_csv: str = os.path.join(get_input_base(), "Production_Trade", "Environment_LivestockPatterns_E_All_Data_NOFLAG.csv")
    inputs_landuse_csv: str = os.path.join(get_input_base(), "Constraint", "Inputs_LandUse_E_All_Data_NOFLAG.csv")
    historical_fert_xlsx: str = os.path.join(get_input_base(), "Fertilizer", "Historical_Fertilizer_application.xlsx")
    prices_csv: str = os.path.join(get_input_base(), "Price_Cost", "Price", "World_Production_Value_per_Unit.xlsx")
    trade_crops_xlsx: str = os.path.join(get_input_base(), "Production_Trade", "Trade_CropsLivestock_E_All_Data_NOFLAG_filtered.xlsx")
    trade_forestry_csv: str = os.path.join(get_input_base(), "Production_Trade", "Forestry_E_All_Data_NOFLAG.csv")
    luh2_states_nc: str = r"R:\Data\Food\LUH2\LUH2_GCB2019_states_2010_2020.nc4"
    luh2_transitions_nc: str = r"R:\Data\Food\LUH2\LUH2_GCB2019_transitions_2010_2020.nc4"
    luh2_mask_nc: str = r"R:\Data\Food\LUH2\mask_LUH2_025d.nc"
    luc_param_xlsx: str = os.path.join(get_src_base(), "LUCE_parameter.xlsx")
    # optional price/cost sources under Price_Cost
    faostat_prices_csv: str = os.path.join(get_input_base(), "Price_Cost", "Price", "Prices_E_All_Data_NOFLAG.csv")
    macc_pkl: str = os.path.join(get_input_base(), "Price_Cost", "Cost", "MACC-Global-US.pkl")
    # constraints
    intake_constraint_xlsx: str = os.path.join(get_input_base(), "Constraint", "Intake_constraint.xlsx")
    # optional emissions csvs
    emis_fires_csv: str = os.path.join(get_input_base(), "Emissions_Land_Use_Fires_E_All_Data_NOFLAG.csv")
    # drivers and others
    population_wpp_csv: str = os.path.join(get_input_base(), "Driver", "Population", "WPP", "Population_E_All_Data_NOFLAG.csv")
    temperature_xlsx: str = os.path.join(get_input_base(), "Driver", "Temperature", "SSP_IAM_V2_201811_Temperature.xlsx")
    income_sspdb_xlsx: str = os.path.join(get_input_base(), "Driver", "Income", "SSPD", "SSPDB_future_GDP.xlsx")
    sspdb_scenario: str = "SSP2_v9_130325"
    production_trade_fbs_csv: str = os.path.join(get_input_base(), "Production_Trade", "FoodBalanceSheets_E_All_Data_NOFLAG.csv")
    nonfood_balance_csv: str = os.path.join(get_input_base(), "Production_Trade", "CommodityBalances_(non-food)_(2010-)_E_All_Data_NOFLAG.csv")
    forestry_csv: str = os.path.join(get_input_base(), "Production_Trade", "Forestry_E_All_Data_NOFLAG.csv")

# -------------------- helpers --------------------
def _lc(df: pd.DataFrame) -> pd.DataFrame:
    z = df.copy()
    z.columns = [str(c).strip() for c in z.columns]
    return z

def _faostat_wide_to_long(df: pd.DataFrame, value_name: str = 'Value') -> pd.DataFrame:
    """Convert FAOSTAT-style wide year columns (Y1961, ...) into long format."""
    if df is None or len(df) == 0:
        return df
    year_cols = [c for c in df.columns if isinstance(c, str) and c.strip().startswith('Y') and c.strip()[1:].isdigit()]
    if not year_cols:
        return df
    id_cols = [c for c in df.columns if c not in year_cols]
    long_df = df.melt(id_vars=id_cols, value_vars=year_cols,
                      var_name='Year', value_name=value_name)
    long_df['Year'] = pd.to_numeric(long_df['Year'].astype(str).str.strip().str.lstrip('Y'), errors='coerce')
    long_df[value_name] = pd.to_numeric(long_df[value_name], errors='coerce')
    long_df = long_df.dropna(subset=['Year'])
    long_df['Year'] = long_df['Year'].astype(int)
    return long_df

def _find_col(df: pd.DataFrame, names: List[str]) -> str:
    cols = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in cols:
            return cols[n.lower()]
    # fuzzy
    for c in df.columns:
        for n in names:
            if n.lower() in str(c).lower():
                return c
    raise KeyError(f"columns {names} not found")

def _maybe_find_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    try:
        return _find_col(df, names)
    except Exception:
        return None

def _tuple_field(name: str) -> str:
    """Convert a column name to the attribute name created by DataFrame.itertuples."""
    # Replace non-word characters with underscores and prefix underscores for leading digits
    return re.sub(r'\W|^(?=\d)', '_', str(name))


def _build_elasticity_map(df: pd.DataFrame, value_col: str = 'Elasticity_mean') -> Dict[Tuple[str, str], float]:
    """Return {(country_code, commodity) -> elasticity} from a wide sheet with Elasticity_mean."""
    out: Dict[Tuple[str, str], float] = {}
    if df is None or df.empty:
        return out
    required = {'Country', 'Commodity', value_col}
    if not required.issubset(df.columns):
        return out
    for r in df[['Country', 'Commodity', value_col]].itertuples(index=False):
        country = str(getattr(r, 'Country'))
        commodity = str(getattr(r, 'Commodity'))
        val = pd.to_numeric(getattr(r, value_col), errors='coerce')
        if pd.notna(val):
            out[(country, commodity)] = float(val)
    return out


def _build_cross_elasticity_map(df: pd.DataFrame, commodity_filter: Optional[set] = None) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Return {(country_code, commodity) -> {other_commodity: elasticity}} for cross-price sheets."""
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    if df is None or df.empty:
        return out
    required = {'Country', 'Commodity'}
    if not required.issubset(df.columns):
        return out
    attr_by_col = {col: _tuple_field(col) for col in df.columns}
    base_cols = {'Country', 'Country_label', 'Commodity'}
    for row in df.itertuples(index=False):
        country = str(getattr(row, attr_by_col['Country']))
        commodity = str(getattr(row, attr_by_col['Commodity']))
        key = (country, commodity)
        cross: Dict[str, float] = {}
        for col, attr in attr_by_col.items():
            if col in base_cols:
                continue
            if commodity_filter is not None and col not in commodity_filter:
                continue
            val = pd.to_numeric(getattr(row, attr), errors='coerce')
            if pd.isna(val):
                continue
            cross[col] = float(val)
        if cross:
            out[key] = cross
    return out

def _country_by_m49(df: pd.DataFrame, universe: Universe) -> Optional[pd.Series]:
    """Return a Series of mapped country names using M49 codes if available, else None."""
    c_m49 = 'Area Code (M49)'
    if c_m49 not in df.columns:
        return None
    # Build reverse map M49->country from Universe
    code_to_country = {str(v): str(k) for k, v in (universe.m49_by_country or {}).items() if v is not None}
    if not code_to_country:
        return None
    codes = df[c_m49].astype(str).str.extract(r'(\d+)')[0]
    codes = pd.to_numeric(codes, errors='coerce').astype('Int64')
    vals = codes.astype(str).map(code_to_country)
    return vals


def _estimate_area_ha_from_grid(ds: xr.Dataset) -> np.ndarray:
    """Return grid-cell areas in hectares aligned with ds (lat, lon)."""
    if 'areacella' in ds:
        return np.asarray(ds['areacella'].values, dtype=float) * 1e-4
    R = 6_371_000.0
    lat = np.asarray(ds['lat'].values, dtype=float)
    lon = np.asarray(ds['lon'].values, dtype=float)
    if lat.size < 2 or lon.size < 2:
        raise ValueError("LUH2 dataset lat/lon dimensions are insufficient to estimate cell area")
    dlat = np.deg2rad(abs(lat[1] - lat[0]))
    dlon = np.deg2rad(abs(lon[1] - lon[0]))
    lat_r = np.deg2rad(lat)
    strip = (np.sin(lat_r + dlat / 2.0) - np.sin(lat_r - dlat / 2.0)) * (R ** 2) * dlon
    area_lat = strip  # m² per latitude band
    # broadcast to grid shape (lat, lon)
    area = np.repeat(area_lat[:, None], lon.size, axis=1)
    return area * 1e-4  # convert m² to ha


def load_luh2_land_cover(states_nc_path: str,
                         mask_nc_path: str,
                         universe: Universe,
                         years: Optional[List[int]] = None) -> pd.DataFrame:
    """Aggregate LUH2 land-cover fractions (states file) to country-level cropland/pasture/forest areas."""
    columns = ['country', 'iso3', 'year', 'land_use', 'area_ha']
    if not (states_nc_path and os.path.exists(states_nc_path)):
        return pd.DataFrame(columns=columns)
    if not (mask_nc_path and os.path.exists(mask_nc_path)):
        return pd.DataFrame(columns=columns)

    try:
        dict_path = os.path.join(get_src_base(), 'dict_v3.xlsx')
        region_df = _lc(pd.read_excel(dict_path, 'region'))
    except Exception:
        region_df = pd.DataFrame(columns=['Region_label_new', 'Region_maskID', 'ISO3 Code'])

    region_df = region_df[['Region_label_new', 'Region_maskID', 'ISO3 Code']].dropna()
    region_df['Region_maskID'] = pd.to_numeric(region_df['Region_maskID'], errors='coerce').astype('Int64')
    region_df = region_df.dropna(subset=['Region_maskID'])
    mask_id_to_country = {int(r.Region_maskID): str(r.Region_label_new).strip()
                          for r in region_df.itertuples(index=False)}
    mask_id_to_iso3 = {}
    for mid, country in mask_id_to_country.items():
        iso = universe.iso3_by_country.get(country)
        if iso:
            mask_id_to_iso3[mid] = iso
    if not mask_id_to_iso3:
        return pd.DataFrame(columns=columns)

    target_years = sorted({int(y) for y in (years if years else universe.years or [])})

    with xr.open_dataset(states_nc_path) as ds:
        all_years = [int(getattr(t, 'year', getattr(t, 'year', t))) for t in ds['time'].values]
        ds = ds.assign_coords(year=('time', all_years)).swap_dims({'time': 'year'}).sortby('year')
        available_years = set(int(y) for y in ds['year'].values.tolist())
        if not target_years:
            target_years = sorted(available_years)
        max_available_year = max(available_years)
        source_years = sorted({y for y in target_years if y in available_years} | {max_available_year})
        area_grid = _estimate_area_ha_from_grid(ds.isel(year=0))

        with xr.open_dataset(mask_nc_path) as mask_ds:
            if 'id1' not in mask_ds:
                raise KeyError("Mask NetCDF must contain variable 'id1'")
            id_array = np.asarray(mask_ds['id1'].values, dtype=float)

        id_array = np.nan_to_num(id_array, nan=0.0)
        id_int = id_array.astype(np.int64)
        flat_ids = id_int.ravel()
        valid_idx = np.where(flat_ids > 0)[0]
        if not len(valid_idx):
            return pd.DataFrame(columns=columns)
        valid_ids = flat_ids[valid_idx]
        unique_ids, inverse_idx = np.unique(valid_ids, return_inverse=True)

        category_states = {
            'cropland_area_ha': ['c3ann', 'c3per', 'c4ann', 'c4per', 'c3nfx'],
            'pasture_area_ha': ['pastr', 'range'],
            'forest_area_ha': ['primf', 'secdf'],
        }

        cache: Dict[int, Dict[str, np.ndarray]] = {}
        for year in source_years:
            cat_sums: Dict[str, np.ndarray] = {}
            for cat_name, states in category_states.items():
                total = None
                for state in states:
                    if state not in ds:
                        continue
                    arr = ds[state].sel(year=year).values
                    arr = np.nan_to_num(arr, nan=0.0, copy=False)
                    total = arr if total is None else total + arr
                if total is None:
                    cat_sums[cat_name] = np.zeros(len(unique_ids), dtype=float)
                else:
                    weighted = (total * area_grid).reshape(-1)
                    cat_sums[cat_name] = np.bincount(
                        inverse_idx,
                        weights=weighted[valid_idx],
                        minlength=len(unique_ids)
                    )
            cache[year] = cat_sums

    records: List[Dict[str, Any]] = []
    for year in target_years:
        src_year = year if year in cache else max_available_year
        cat_sums = cache.get(src_year)
        if not cat_sums:
            continue
        for idx, mask_id in enumerate(unique_ids):
            mask_id_int = int(mask_id)
            country = mask_id_to_country.get(mask_id_int)
            iso3 = mask_id_to_iso3.get(mask_id_int)
            if not country or not iso3:
                continue
            for cat_name, sums in cat_sums.items():
                if idx >= len(sums):
                    continue
                area_val = float(sums[idx])
                if area_val <= 0.0:
                    continue
                records.append({
                    'country': country,
                    'iso3': iso3,
                    'year': year,
                    'land_use': cat_name,
                    'area_ha': area_val,
                })

    if not records:
        return pd.DataFrame(columns=columns)
    out_df = pd.DataFrame.from_records(records, columns=columns)
    out_df = out_df.groupby(['country', 'iso3', 'year', 'land_use'], as_index=False)['area_ha'].sum()
    return out_df


def load_roundwood_supply(forestry_csv_path: str,
                          universe: Universe,
                          years: Optional[List[int]] = None) -> pd.DataFrame:
    """Load FAOSTAT forestry production for Roundwood as m³ per country-year."""
    columns = ['country', 'iso3', 'year', 'roundwood_m3']
    if not (forestry_csv_path and os.path.exists(forestry_csv_path)):
        return pd.DataFrame(columns=columns)
    df_raw = pd.read_csv(forestry_csv_path)
    df = _lc(_faostat_wide_to_long(df_raw))
    c_area = _find_col(df, ['Area'])
    c_year = _find_col(df, ['Year'])
    c_item = _find_col(df, ['Item'])
    c_elem = _find_col(df, ['Element'])
    c_val = _find_col(df, ['Value'])
    c_unit = _maybe_find_col(df, ['Unit'])
    if not all([c_area, c_year, c_item, c_elem, c_val]):
        return pd.DataFrame(columns=columns)
    z = df[[c_area, c_year, c_item, c_elem, c_val] + ([c_unit] if c_unit else [])].copy()
    z.columns = ['area', 'year', 'item_raw', 'element', 'value'] + (['unit'] if c_unit else [])
    m49_country = _country_by_m49(df, universe)
    z['country'] = m49_country if m49_country is not None else z['area']
    try:
        maps = load_emis_item_mappings(os.path.join(get_src_base(), 'dict_v3.xlsx'))
        z['commodity'] = z['item_raw'].map(maps.production_map).fillna(z['item_raw'])
    except Exception:
        z['commodity'] = z['item_raw']
    z = z[(z['commodity'].astype(str).str.strip().str.lower() == 'roundwood') &
          (z['country'].isin(universe.countries))]
    if z.empty:
        return pd.DataFrame(columns=columns)
    z = z[z['element'].astype(str).str.contains('production', case=False, na=False)]
    z['value'] = pd.to_numeric(z['value'], errors='coerce')
    if c_unit:
        z['unit'] = z['unit'].astype(str).str.lower()
        z['roundwood_m3'] = z.apply(
            lambda r: (r['value'] if np.isfinite(r['value']) else 0.0) *
            (1000.0 if '1000' in r['unit'] else 1.0),
            axis=1
        )
    else:
        z['roundwood_m3'] = z['value']
    z = z.dropna(subset=['roundwood_m3'])
    z['roundwood_m3'] = pd.to_numeric(z['roundwood_m3'], errors='coerce').fillna(0.0)
    if z.empty:
        return pd.DataFrame(columns=columns)
    agg = z.groupby(['country', 'year'], as_index=False)['roundwood_m3'].sum()
    agg['iso3'] = agg['country'].map(universe.iso3_by_country)
    agg = agg.dropna(subset=['iso3'])
    agg['year'] = pd.to_numeric(agg['year'], errors='coerce').astype('Int64')
    agg = agg.dropna(subset=['year'])
    agg['year'] = agg['year'].astype(int)
    if years:
        target_years = sorted({int(y) for y in years})
        if agg.empty:
            return pd.DataFrame(columns=columns)
        max_hist = int(agg['year'].max())
        frames = [agg]
        missing = [y for y in target_years if y not in agg['year'].values]
        if missing and max_hist in agg['year'].values:
            base_rows = agg[agg['year'] == max_hist]
            for y in missing:
                if y < max_hist and y in agg['year'].values:
                    continue
                tmp = base_rows.copy()
                tmp['year'] = y
                frames.append(tmp)
        agg = pd.concat(frames, ignore_index=True)
        agg = agg[agg['year'].isin(target_years)]
    return agg[['country', 'iso3', 'year', 'roundwood_m3']]

# -------------------- universe --------------------
def build_universe_from_dict_v3(path: str, config: ScenarioConfig) -> Universe:
    xls = pd.ExcelFile(path)
    region = _lc(pd.read_excel(xls, 'region'))
    try:
        emis_proc = _lc(pd.read_excel(xls, 'Emis_process'))
    except ValueError:
        emis_proc = pd.DataFrame()
    emis_item = _lc(pd.read_excel(xls, 'Emis_item'))

    c_country = _find_col(region, ['Country'])
    c_iso3 = _find_col(region, ['ISO3'])
    c_label = _find_col(region, ['Region_label_new'])
    region_clean = region[region[c_label].astype(str).str.lower() != 'no'].copy()
    region_clean['country_label'] = region_clean[c_label].astype(str).str.strip()
    region_clean['country_code'] = region_clean[c_country].astype(str).str.strip()
    region2 = region_clean[['country_label', 'country_code', c_iso3]].drop_duplicates(subset=['country_label'])
    countries = region2['country_label'].tolist()
    iso3_map = dict(zip(region2['country_label'], region2[c_iso3].astype(str)))

    # Region_aggMC map
    c_ragg = _find_col(region, ['Region_aggMC'])
    region_aggMC_by_country = dict(zip(region_clean['country_label'], region_clean[c_ragg].astype(str)))
    # SSP region map
    c_ssp = 'Region_map_SSPDB' if 'Region_map_SSPDB' in region.columns else None
    ssp_region_by_country = dict(zip(region_clean['country_label'], region_clean[c_ssp].astype(str))) if c_ssp else {}
    # M49 code map
    m49_col = 'M49 Code' if 'M49 Code' in region.columns else None
    m49_by_country = {}
    if m49_col:
        region_m49 = region_clean[['country_label', m49_col]].drop_duplicates(subset=['country_label'])
        m49_by_country = dict(zip(region_m49['country_label'], region_m49[m49_col].astype(str)))

    # Processes & meta
    process_meta = {}
    processes: List[str] = []
    if not emis_proc.empty and 'Process' in emis_proc.columns:
        c_proc = _find_col(emis_proc, ['Process'])
        processes = emis_proc[c_proc].astype(str).str.strip().dropna().unique().tolist()
        c_cat = next((nm for nm in ['category','Category','Sector'] if nm in emis_proc.columns), None)
        c_gas = next((nm for nm in ['gas','GHG','Gas'] if nm in emis_proc.columns), None)
        for r in emis_proc.itertuples(index=False):
            name = getattr(r, _tuple_field(c_proc))
            if pd.isna(name):
                continue
            proc_name = str(name).strip()
            if not proc_name:
                continue
            meta_entry = process_meta.setdefault(proc_name, {})
            if c_cat:
                meta_entry['category'] = str(getattr(r, c_cat))
            if c_gas:
                meta_entry['gas'] = str(getattr(r, c_gas))
    # Fallback to Emis_item if process list or meta missing
    c_proc_item = _find_col(emis_item, ['Process'])
    if not processes:
        processes = emis_item[c_proc_item].astype(str).str.strip().dropna().unique().tolist()
    c_gas_item = _find_col(emis_item, ['GHG']) if 'GHG' in emis_item.columns else None
    c_source_item = 'Emis_file_source' if 'Emis_file_source' in emis_item.columns else None
    proc_field = _tuple_field(c_proc_item)
    if c_gas_item or c_source_item:
        for r in emis_item.itertuples(index=False):
            proc_name = str(getattr(r, proc_field)).strip()
            if not proc_name:
                continue
            meta_entry = process_meta.setdefault(proc_name, {})
            if c_gas_item:
                gas_val = getattr(r, _tuple_field(c_gas_item))
                if not pd.isna(gas_val):
                    meta_entry.setdefault('gas', str(gas_val))
            if c_source_item:
                src_val = getattr(r, _tuple_field(c_source_item))
                if not pd.isna(src_val):
                    meta_entry.setdefault('file_source', str(src_val))

    # Commodities
    c_itemmap = _find_col(emis_item, ['Item_Production_Map'])
    commodities = sorted(pd.unique(emis_item[c_itemmap].astype(str).str.strip()))

    # Cat2 mapping
    c_cat2 = _find_col(emis_item, ['Item_Cat2'])
    item_cat2_by_commodity = {}
    c_itemmap_attr = _tuple_field(c_itemmap)
    c_cat2_attr = _tuple_field(c_cat2)
    for r in emis_item.itertuples(index=False):
        item = str(getattr(r, c_itemmap_attr)).strip()
        cat2_val = getattr(r, c_cat2_attr)
        item_cat2_by_commodity[item] = '' if pd.isna(cat2_val) else str(cat2_val)

    # years
    years_hist = list(range(config.years_hist_start, config.years_hist_end + 1))
    years_future = list(range(2030, config.years_future_end + 1, config.future_step))
    years = years_hist + years_future

    return Universe(countries=countries, iso3_by_country=iso3_map, m49_by_country=m49_by_country,
                    commodities=commodities, years=years,
                    processes=processes, process_meta=process_meta,
                    region_aggMC_by_country=region_aggMC_by_country,
                    item_cat2_by_commodity=item_cat2_by_commodity,
                    ssp_region_by_country=ssp_region_by_country)

# -------------------- nodes skeleton --------------------
def make_nodes_skeleton(universe: Universe) -> List[Node]:
    nodes: List[Node] = []
    # simple cartesian for skeleton (production/demand later fill)
    for i in universe.countries:
        iso3 = universe.iso3_by_country.get(i, '')
        for t in universe.years:
            for j in universe.commodities:
                nodes.append(Node(country=i, iso3=iso3, year=t, commodity=j))
    return nodes

# -------------------- elasticities --------------------
def apply_supply_ty_elasticity(nodes: List[Node], elasticity_xlsx: str) -> None:
    # optional; if not present, skip
    if not os.path.exists(elasticity_xlsx):
        return
    xls = pd.ExcelFile(elasticity_xlsx)
    def read(sheet: str) -> pd.DataFrame:
        return _lc(pd.read_excel(xls, sheet)) if sheet in xls.sheet_names else pd.DataFrame()

    temp_map = _build_elasticity_map(read('Supply-Temperature'))
    yield_map = _build_elasticity_map(read('Supply-Yield'))
    own_price_map = _build_elasticity_map(read('Supply-Own-Price'))
    commodity_filter = {str(n.commodity) for n in nodes}
    supply_cross_map = _build_cross_elasticity_map(read('Supply_Cross_mean'), commodity_filter=commodity_filter)

    for n in nodes:
        key = (str(n.country), str(n.commodity))
        temp = temp_map.get(key)
        if temp is not None:
            n.eps_supply_temp = float(temp)
        eta_y = yield_map.get(key)
        if eta_y is not None:
            n.eps_supply_yield = float(eta_y)
        own = own_price_map.get(key)
        if own is not None:
            n.eps_supply = float(own)
        cross = supply_cross_map.get(key)
        if cross:
            n.meta['supply_cross'] = dict(cross)

# -------------------- FBS demand --------------------
FBS_TO_UNIVERSE: Dict[str, List[str]] = {
    # Cereals & grains
    'maize': ['Maize (corn)'],
    'maize and products': ['Maize (corn)'],
    'maize germ oil': ['Maize (corn)'],
    'wheat': ['Wheat'],
    'wheat and products': ['Wheat'],
    'rice': ['Rice'],
    'rice and products': ['Rice'],
    'ricebran oil': ['Rice'],
    'barley and products': ['Barley'],
    'oats': ['Oats'],
    'millet and products': ['Millet'],
    'sorghum and products': ['Sorghum'],
    'rye and products': ['Rye'],
    # Roots & tubers
    'cassava and products': ['Cassava, fresh'],
    'sweet potatoes': ['Sweet potatoes'],
    'potatoes and products': ['Potatoes'],
    'yams': ['Cassava, fresh'],
    'roots, other': ['Cassava, fresh'],
    'starchy roots': ['Cassava, fresh', 'Potatoes', 'Sweet potatoes'],
    # Oilcrops & oils
    'soyabeans': ['Soya beans'],
    'soyabean oil': ['Oilcrops, Oil Equivalent'],
    'sunflowerseed': ['Sunflower seed'],
    'sunflowerseed oil': ['Sunflower seed'],
    'rape and mustardseed': ['Rape or colza seed'],
    'rape and mustard oil': ['Rape or colza seed'],
    'groundnuts': ['Groundnuts, excluding shelled'],
    'groundnut oil': ['Groundnuts, excluding shelled'],
    'cottonseed': ['Seed cotton, unginned'],
    'cottonseed oil': ['Seed cotton, unginned'],
    'palm oil': ['Oilcrops, Oil Equivalent'],
    'palm kernels': ['Oilcrops, Oil Equivalent'],
    'palmkernel oil': ['Oilcrops, Oil Equivalent'],
    'coconut oil': ['Oilcrops, Oil Equivalent'],
    'coconuts - incl copra': ['Oilcrops, Oil Equivalent'],
    'olive oil': ['Oilcrops, Oil Equivalent'],
    'oilcrops oil, other': ['Oilcrops, Oil Equivalent'],
    'oilcrops': ['Oilcrops, Oil Equivalent'],
    'vegetable oils': ['Oilcrops, Oil Equivalent'],
    'sesame seed': ['Oilcrops, Oil Equivalent'],
    'sesameseed oil': ['Oilcrops, Oil Equivalent'],
    'nuts and products': ['Oilcrops, Oil Equivalent'],
    'treenuts': ['Oilcrops, Oil Equivalent'],
    # Fruits
    'bananas': ['Fruit Primary'],
    'apples and products': ['Fruit Primary'],
    'plantains': ['Fruit Primary'],
    'pineapples and products': ['Fruit Primary'],
    'dates': ['Fruit Primary'],
    'grapes and products (excl wine)': ['Fruit Primary'],
    'grapefruit and products': ['Fruit Primary'],
    'oranges, mandarines': ['Fruit Primary'],
    'lemons, limes and products': ['Fruit Primary'],
    'citrus, other': ['Fruit Primary'],
    'fruits - excluding wine': ['Fruit Primary'],
    'fruits, other': ['Fruit Primary'],
    'olives (including preserved)': ['Fruit Primary'],
    # Vegetables
    'tomatoes and products': ['Vegetables Primary'],
    'onions': ['Vegetables Primary'],
    'vegetables': ['Vegetables Primary'],
    'vegetables, other': ['Vegetables Primary'],
    # Pulses & beans
    'beans': ['Beans, dry'],
    'pulses': ['Beans, dry'],
    'pulses, other and products': ['Beans, dry'],
    'peas': ['Beans, dry'],
    # Sugar
    'sugar crops': ['Sugar cane', 'Sugar beet'],
    'sugar (raw equivalent)': ['Sugar cane', 'Sugar beet'],
    'sugar & sweeteners': ['Sugar cane', 'Sugar beet'],
    'sweeteners, other': ['Sugar cane', 'Sugar beet'],
    'sugar non-centrifugal': ['Sugar cane'],
    # Animal products
    'milk - excluding butter': ['Raw milk of cattle'],
    'butter, ghee': ['Raw milk of cattle'],
    'fats, animals, raw': ['Raw milk of cattle'],
    'cream': ['Raw milk of cattle'],
    'animal fats': ['Raw milk of cattle'],
    'eggs': ['Eggs Primary'],
    'bovine meat': ['Meat of cattle with the bone, fresh or chilled', 'Meat of buffalo, fresh or chilled'],
    'pigmeat': ['Meat of pig with the bone, fresh or chilled'],
    'poultry meat': ['Meat of chickens, fresh or chilled', 'Meat of turkeys, fresh or chilled', 'Meat of ducks, fresh or chilled'],
    'mutton & goat meat': ['Meat of sheep, fresh or chilled', 'Meat of goat, fresh or chilled'],
    'meat, other': [
        'Horse meat, fresh or chilled',
        'Meat of asses, fresh or chilled',
        'Meat of mules, fresh or chilled',
        'Meat of camels, fresh or chilled',
        'Meat of other domestic camelids, fresh or chilled'
    ],
    # Fish and aquatic products
    'demersal fish': ['Fish, Seafood'],
    'pelagic fish': ['Fish, Seafood'],
    'marine fish, other': ['Fish, Seafood'],
    'freshwater fish': ['Fish, Seafood'],
    'cephalopods': ['Fish, Seafood'],
    'crustaceans': ['Fish, Seafood'],
    'molluscs, other': ['Fish, Seafood'],
    'aquatic animals, others': ['Fish, Seafood'],
    'aquatic plants': ['Fish, Seafood'],
    'aquatic products, other': ['Fish, Seafood'],
    'fish, body oil': ['Fish, Seafood'],
    'fish, liver oil': ['Fish, Seafood'],
    'meat, aquatic mammals': ['Fish, Seafood'],
}


def build_demand_components_from_fbs(fbs_csv: str,
                                     universe: Universe,
                                     *,
                                     production_lookup: Optional[Dict[Tuple[str, str, int], float]] = None,
                                     latest_hist_prod: Optional[Dict[Tuple[str, str], Tuple[int, float]]] = None) -> pd.DataFrame:
    if not os.path.exists(fbs_csv):
        return pd.DataFrame(columns=['country','iso3','year','commodity','food_t','feed_t','seed_t','demand_total_t'])
    df_raw = pd.read_csv(fbs_csv)
    df = _lc(_faostat_wide_to_long(df_raw))
    c_area = _find_col(df, ['Area'])
    c_year = _find_col(df, ['Year'])
    c_item = _find_col(df, ['Item'])
    c_elem = _find_col(df, ['Element'])
    c_val  = _find_col(df, ['Value'])
    z = df[[c_area, c_year, c_item, c_elem, c_val]].copy()
    z.columns = ['area', 'year', 'item_raw', 'element', 'value']
    m49_country = _country_by_m49(df, universe)
    z['country'] = m49_country if m49_country is not None else z['area']
    try:
        maps = load_emis_item_mappings(os.path.join(get_src_base(), 'dict_v3.xlsx'))
        z['commodity'] = z['item_raw'].map(maps.production_map).fillna(z['item_raw'])
    except Exception:
        z['commodity'] = z['item_raw']
    z = z[z['country'].isin(universe.countries)]
    if z.empty:
        return pd.DataFrame(columns=['country','iso3','year','commodity','food_t','feed_t','seed_t','demand_total_t'])

    pivot = z.pivot_table(index=['country', 'year', 'commodity'],
                          columns='element', values='value', aggfunc='sum').reset_index()
    pivot.rename(columns={c: c.lower() for c in pivot.columns}, inplace=True)
    food = pivot.get('food', pd.Series(0, index=pivot.index))
    feed = pivot.get('feed', pd.Series(0, index=pivot.index))
    seed = pivot.get('seed', pd.Series(0, index=pivot.index))
    base = pd.DataFrame({
        'country': pivot['country'],
        'year': pivot['year'],
        'commodity': pivot['commodity'],
        'food_t': food.fillna(0.0).astype(float),
        'feed_t': feed.fillna(0.0).astype(float),
        'seed_t': seed.fillna(0.0).astype(float),
    })

    universe_set = set(universe.commodities or [])
    prod_lookup = production_lookup or {}
    latest_prod = latest_hist_prod or {}
    def _norm_key(name: str) -> str:
        return str(name).strip().lower()

    def _resolve_targets(name: str) -> List[str]:
        if name in universe_set:
            return [name]
        key = _norm_key(name)
        if key in FBS_TO_UNIVERSE:
            return FBS_TO_UNIVERSE[key]
        if key.endswith(' and products'):
            base_key = key[:-len(' and products')].strip()
            if base_key in FBS_TO_UNIVERSE:
                return FBS_TO_UNIVERSE[base_key]
        if key.endswith('s') and key[:-1] in FBS_TO_UNIVERSE:
            return FBS_TO_UNIVERSE[key[:-1]]
        return []

    def _production_value(country: str, commodity: str, year: int) -> float:
        val = prod_lookup.get((country, commodity, year))
        if val is None and latest_prod:
            prev = latest_prod.get((country, commodity))
            if prev is not None:
                val = prev[1]
        return max(float(val), 0.0) if val is not None else 0.0

    rows: List[Dict[str, float]] = []
    for row in base.itertuples(index=False):
        targets = _resolve_targets(row.commodity)
        if not targets:
            continue

        if len(targets) == 1:
            weights = {targets[0]: 1.0}
        else:
            shares = [_production_value(row.country, tgt, int(row.year)) for tgt in targets]
            total = sum(shares)
            if total <= 0:
                weights = {tgt: 1.0 / len(targets) for tgt in targets}
            else:
                weights = {tgt: share / total for tgt, share in zip(targets, shares)}

        for tgt, weight in weights.items():
            if weight <= 0:
                continue
            rows.append({
                'country': row.country,
                'year': int(row.year),
                'commodity': tgt,
                'food_t': float(row.food_t) * weight,
                'feed_t': float(row.feed_t) * weight,
                'seed_t': float(row.seed_t) * weight,
            })

    if not rows:
        return pd.DataFrame(columns=['country','iso3','year','commodity','food_t','feed_t','seed_t','demand_total_t'])

    result = pd.DataFrame(rows)
    result = result.groupby(['country', 'year', 'commodity'], as_index=False)[['food_t', 'feed_t', 'seed_t']].sum()
    result['iso3'] = result['country'].map(universe.iso3_by_country)
    result['demand_total_t'] = result['food_t'] + result['feed_t'] + result['seed_t']
    return result[['country','iso3','year','commodity','food_t','feed_t','seed_t','demand_total_t']]

def _parse_m49_code(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, float) and np.isnan(value):
            return None
    except Exception:
        return None
    match = re.search(r'\d+', str(value))
    if not match:
        return None
    try:
        return int(match.group(0))
    except Exception:
        return None

def _attach_country_column(df: pd.DataFrame,
                           m49_to_country: Dict[int, str],
                           area_lower_to_country: Dict[str, str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=['country'])
    country = pd.Series(index=df.index, dtype=object)
    m49_col = None
    for cand in ['Area Code (M49)', 'Area Code']:
        if cand in df.columns:
            m49_col = cand
            break
    if m49_col:
        codes = df[m49_col].apply(_parse_m49_code)
        country = codes.map(m49_to_country)
    if 'Area' in df.columns:
        missing = country.isna()
        if missing.any():
            area_vals = df.loc[missing, 'Area'].astype(str).str.strip().str.lower()
            country.loc[missing] = area_vals.map(area_lower_to_country)
    df = df.assign(country=country)
    df = df.dropna(subset=['country'])
    df['country'] = df['country'].astype(str)
    return df

def _melt_trade_quantities(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=['country','Item','year','Element','value'])
    year_cols = [c for c in df.columns if isinstance(c, str) and c.strip().startswith('Y') and c.strip()[1:].isdigit()]
    if not year_cols:
        return pd.DataFrame(columns=['country','Item','year','Element','value'])
    df_year = df[year_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    df = df.copy()
    df[year_cols] = df_year
    long_df = df.melt(id_vars=[c for c in df.columns if c not in year_cols],
                      value_vars=year_cols,
                      var_name='year',
                      value_name='value')
    long_df['year'] = pd.to_numeric(long_df['year'].astype(str).str.strip().str.lstrip('Y'), errors='coerce')
    long_df = long_df.dropna(subset=['year'])
    long_df['year'] = long_df['year'].astype(int)
    long_df['value'] = pd.to_numeric(long_df['value'], errors='coerce').fillna(0.0)
    return long_df

def _load_trade_cropslivestock(path: str,
                               items: List[str],
                               m49_to_country: Dict[int, str],
                               area_lower_to_country: Dict[str, str]) -> pd.DataFrame:
    if not items or not os.path.exists(path):
        return pd.DataFrame(columns=['country','Item','year','import','export'])
    df = _lc(pd.read_excel(path))
    df = df[df['Item'].astype(str).str.strip().isin(items)]
    if df.empty:
        return pd.DataFrame(columns=['country','Item','year','import','export'])
    df = df[df['Element'].astype(str).str.lower().isin({'import quantity','export quantity'})]
    if df.empty:
        return pd.DataFrame(columns=['country','Item','year','import','export'])
    df = _attach_country_column(df, m49_to_country, area_lower_to_country)
    if df.empty:
        return pd.DataFrame(columns=['country','Item','year','import','export'])
    df['Item'] = df['Item'].astype(str).str.strip()
    long_df = _melt_trade_quantities(df)
    if long_df.empty:
        return pd.DataFrame(columns=['country','Item','year','import','export'])
    long_df['Element'] = long_df['Element'].astype(str).str.lower()
    agg = long_df.groupby(['country','Item','year','Element'], as_index=False)['value'].sum()
    piv = agg.pivot_table(index=['country','Item','year'], columns='Element', values='value', aggfunc='sum', fill_value=0.0).reset_index()
    import_col = 'import quantity'
    export_col = 'export quantity'
    if import_col not in piv.columns:
        piv[import_col] = 0.0
    if export_col not in piv.columns:
        piv[export_col] = 0.0
    piv.rename(columns={import_col: 'import_t', export_col: 'export_t'}, inplace=True)
    return piv[['country','Item','year','import_t','export_t']]

def _load_trade_forestry(path: str,
                         items: List[str],
                         m49_to_country: Dict[int, str],
                         area_lower_to_country: Dict[str, str]) -> pd.DataFrame:
    if not items or not os.path.exists(path):
        return pd.DataFrame(columns=['country','Item','year','import','export'])
    df = _lc(pd.read_csv(path))
    df = df[df['Item'].astype(str).str.strip().isin(items)]
    if df.empty:
        return pd.DataFrame(columns=['country','Item','year','import','export'])
    df = df[df['Element'].astype(str).str.lower().isin({'import quantity','export quantity'})]
    if df.empty:
        return pd.DataFrame(columns=['country','Item','year','import','export'])
    df = _attach_country_column(df, m49_to_country, area_lower_to_country)
    if df.empty:
        return pd.DataFrame(columns=['country','Item','year','import','export'])
    df['Item'] = df['Item'].astype(str).str.strip()
    long_df = _melt_trade_quantities(df)
    if long_df.empty:
        return pd.DataFrame(columns=['country','Item','year','import','export'])
    long_df['Element'] = long_df['Element'].astype(str).str.lower()
    agg = long_df.groupby(['country','Item','year','Element'], as_index=False)['value'].sum()
    piv = agg.pivot_table(index=['country','Item','year'], columns='Element', values='value', aggfunc='sum', fill_value=0.0).reset_index()
    import_col = 'import quantity'
    export_col = 'export quantity'
    if import_col not in piv.columns:
        piv[import_col] = 0.0
    if export_col not in piv.columns:
        piv[export_col] = 0.0
    piv.rename(columns={import_col: 'import_t', export_col: 'export_t'}, inplace=True)
    return piv[['country','Item','year','import_t','export_t']]

def _load_trade_fbs(path: str,
                    items: List[str],
                    m49_to_country: Dict[int, str],
                    area_lower_to_country: Dict[str, str]) -> pd.DataFrame:
    if not items or not os.path.exists(path):
        return pd.DataFrame(columns=['country','Item','year','import','export'])
    base_cols = ['Area Code (M49)', 'Area', 'Item', 'Element', 'Unit']
    def usecols(col: str) -> bool:
        return col in base_cols or (isinstance(col, str) and col.startswith('Y') and col[1:].isdigit())
    frames: List[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=200000):
        chunk = _lc(chunk)
        chunk['Item'] = chunk['Item'].astype(str).str.strip()
        mask_item = chunk['Item'].isin(items)
        if not mask_item.any():
            continue
        chunk = chunk.loc[mask_item]
        chunk['Element'] = chunk['Element'].astype(str).str.lower()
        chunk = chunk[chunk['Element'].isin({'import quantity','export quantity'})]
        if chunk.empty:
            continue
        chunk = _attach_country_column(chunk, m49_to_country, area_lower_to_country)
        if chunk.empty:
            continue
        year_cols = [c for c in chunk.columns if c.startswith('Y') and c[1:].isdigit()]
        if not year_cols:
            continue
        values = chunk[year_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        factors = chunk['Unit'].astype(str).str.contains('1000', case=False, na=False).replace({True: 1000.0, False: 1.0}).to_numpy()
        values = values.mul(factors.reshape(-1, 1))
        chunk = chunk.drop(columns=year_cols)
        chunk[year_cols] = values
        long_df = chunk.melt(id_vars=[c for c in chunk.columns if c not in year_cols],
                             value_vars=year_cols,
                             var_name='year',
                             value_name='value')
        long_df['year'] = pd.to_numeric(long_df['year'].astype(str).str.strip().str.lstrip('Y'), errors='coerce')
        long_df = long_df.dropna(subset=['year'])
        long_df['year'] = long_df['year'].astype(int)
        long_df['value'] = pd.to_numeric(long_df['value'], errors='coerce').fillna(0.0)
        frames.append(long_df[['country','Item','year','Element','value']])
    if not frames:
        return pd.DataFrame(columns=['country','Item','year','import','export'])
    long_all = pd.concat(frames, ignore_index=True)
    agg = long_all.groupby(['country','Item','year','Element'], as_index=False)['value'].sum()
    piv = agg.pivot_table(index=['country','Item','year'], columns='Element', values='value', aggfunc='sum', fill_value=0.0).reset_index()
    import_col = 'import quantity'
    export_col = 'export quantity'
    if import_col not in piv.columns:
        piv[import_col] = 0.0
    if export_col not in piv.columns:
        piv[export_col] = 0.0
    piv.rename(columns={import_col: 'import_t', export_col: 'export_t'}, inplace=True)
    return piv[['country','Item','year','import_t','export_t']]

def load_trade_import_export(paths: DataPaths, universe: Universe) -> Tuple[Dict[Tuple[str, str, int], float], Dict[Tuple[str, str, int], float]]:
    try:
        emis_df = _lc(pd.read_excel(paths.dict_v3_path, 'Emis_item'))
    except Exception:
        return {}, {}
    if 'Item_Trade_Map' not in emis_df.columns or 'Trade_file_source' not in emis_df.columns or 'Item_Production_Map' not in emis_df.columns:
        return {}, {}
    trade_attr = _tuple_field('Item_Trade_Map')
    trade_src_attr = _tuple_field('Trade_file_source')
    prod_attr = _tuple_field('Item_Production_Map')
    universe_commodities = set(universe.commodities or [])
    trade_mapping: Dict[str, set] = defaultdict(set)
    items_by_source: Dict[str, set] = defaultdict(set)
    for row in emis_df.itertuples(index=False):
        prod_val = getattr(row, prod_attr, None)
        trade_val = getattr(row, trade_attr, None)
        source_val = getattr(row, trade_src_attr, None)
        if pd.isna(prod_val) or pd.isna(trade_val) or pd.isna(source_val):
            continue
        prod_name = str(prod_val).strip()
        if not prod_name or prod_name.lower() in {'nan', 'no'}:
            continue
        if prod_name not in universe_commodities:
            continue
        trade_items = [item.strip() for item in str(trade_val).split(';') if item and str(item).strip().lower() not in {'nan','no'}]
        if not trade_items:
            continue
        sources = [src.strip() for src in str(source_val).split(';') if src and str(src).strip().lower() not in {'nan','no'}]
        if not sources:
            continue
        if len(sources) == 1 and len(trade_items) > 1:
            sources = sources * len(trade_items)
        elif len(sources) < len(trade_items):
            sources = sources + [sources[-1]] * (len(trade_items) - len(sources))
        for item, src in zip(trade_items, sources):
            trade_mapping[prod_name].add((item, src))
            items_by_source[src].add(item)
    if not trade_mapping:
        return {}, {}

    m49_to_country: Dict[int, str] = {}
    for country, code in (universe.m49_by_country or {}).items():
        parsed = _parse_m49_code(code)
        if parsed is not None:
            m49_to_country[parsed] = country
    area_lower_to_country = {str(cty).strip().lower(): cty for cty in universe.countries}

    source_map = {
        'Trade_CropsLivestock_E_All_Data_NOFLAG_filtered.xlsx': paths.trade_crops_xlsx,
        'Forestry_E_All_Data_NOFLAG.csv': paths.trade_forestry_csv,
        'FoodBalanceSheets_E_All_Data_NOFLAG.csv': paths.fbs_csv,
    }

    data_by_source: Dict[str, pd.DataFrame] = {}
    for source_name, items in items_by_source.items():
        path = source_map.get(source_name)
        item_list = sorted({str(it).strip() for it in items if str(it).strip()})
        if not path or not item_list:
            continue
        if source_name.endswith('.xlsx') and 'Trade_CropsLivestock' in source_name:
            table = _load_trade_cropslivestock(path, item_list, m49_to_country, area_lower_to_country)
        elif source_name.endswith('.csv') and 'Forestry' in source_name:
            table = _load_trade_forestry(path, item_list, m49_to_country, area_lower_to_country)
        elif source_name.endswith('.csv') and 'FoodBalance' in source_name:
            table = _load_trade_fbs(path, item_list, m49_to_country, area_lower_to_country)
        else:
            table = pd.DataFrame(columns=['country','Item','year','import','export'])
        if table is not None and not table.empty:
            table['Item'] = table['Item'].astype(str).str.strip()
            data_by_source[source_name] = table

    imports_by = defaultdict(float)
    exports_by = defaultdict(float)
    for prod_name, pairs in trade_mapping.items():
        for trade_item, source_name in pairs:
            table = data_by_source.get(source_name)
            if table is None or table.empty:
                continue
            subset = table[table['Item'] == trade_item]
            if subset.empty:
                continue
            for record in subset.itertuples(index=False):
                key = (record.country, prod_name, int(record.year))
                imports_by[key] += float(getattr(record, 'import_t', 0.0) or 0.0)
                exports_by[key] += float(getattr(record, 'export_t', 0.0) or 0.0)

    return dict(imports_by), dict(exports_by)

def apply_fbs_components_to_nodes(nodes: List[Node], fbs_components: pd.DataFrame, feed_efficiency: float=1.0) -> None:
    if fbs_components is None or len(fbs_components)==0:
        return
    # index for quick lookup
    key = {(r.country, r.commodity, int(r.year)): (float(r.food_t), float(r.feed_t), float(r.seed_t)) for r in fbs_components.itertuples(index=False)}
    for n in nodes:
        tpl = key.get((n.country, n.commodity, n.year))
        if tpl:
            food, feed, seed = tpl
            feed_eff = float(feed_efficiency) if feed_efficiency>0 else 1.0
            n.D0 = float(food + feed/feed_eff + seed)

# -------------------- production & activities --------------------
def build_production_from_faostat(csv_path: str, universe: Universe) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=['country','iso3','year','commodity','production_t'])
    df_raw = pd.read_csv(csv_path)
    df = _lc(_faostat_wide_to_long(df_raw))
    c_area = _find_col(df, ['Area']); c_year=_find_col(df,['Year']); c_item=_find_col(df,['Item']); c_elem=_find_col(df,['Element']); c_val=_find_col(df,['Value'])
    z = df[[c_area,c_year,c_item,c_elem,c_val]].copy()
    z.columns = ['area','year','item_raw','element','value']
    m49_country = _country_by_m49(df, universe)
    z['country'] = m49_country if m49_country is not None else z['area']
    # item mapping
    try:
        maps = load_emis_item_mappings(os.path.join(get_src_base(), 'dict_v3.xlsx'))
        z['commodity'] = z['item_raw'].map(maps.production_map).fillna(z['item_raw'])
    except Exception:
        z['commodity'] = z['item_raw']
    z = z[(z['element'].str.contains('Production', case=False)) & (z['country'].isin(universe.countries))]
    out = z.groupby(['country','year','commodity'], as_index=False)['value'].sum().rename(columns={'value':'production_t'})
    out['iso3'] = out['country'].map(universe.iso3_by_country)
    return out[['country','iso3','year','commodity','production_t']]

def build_gce_activity_tables(production_csv: str, fbs_csv: str, historical_fert_xlsx: str, universe: Universe) -> Dict[str, pd.DataFrame]:
    # For now, return minimal frames with keys expected by orchestrator
    prod = build_production_from_faostat(production_csv, universe)
    residues_df = prod.rename(columns={'production_t':'residues_feedstock_t'})[['country','iso3','year','commodity','residues_feedstock_t']]
    burning_df = prod.rename(columns={'production_t':'burning_feedstock_t'})[['country','iso3','year','commodity','burning_feedstock_t']]
    rice_df = prod.rename(columns={'production_t':'rice_area_proxy'})[['country','iso3','year','commodity','rice_area_proxy']]
    fert = pd.read_excel(historical_fert_xlsx) if os.path.exists(historical_fert_xlsx) else pd.DataFrame(columns=['country','year','n_fert_t'])
    fertilizers_df = pd.DataFrame(columns=['country','iso3','year','n_fert_t'])
    if len(fert):
        fert = _lc(fert)
        # reshape wide fertilizer workbook (N_content_Y#### columns) into country/year totals
        c_area = _maybe_find_col(fert, ['M49 Code', 'Area Code (M49)', 'Area', 'country', 'Region'])
        c_year = _maybe_find_col(fert, ['Year', 'year'])
        c_val = _maybe_find_col(fert, ['Value', 'n_fert_t', 'N_content'])
        z: pd.DataFrame
        if c_area and c_year and c_val:
            z = fert[[c_area, c_year, c_val]].copy()
            z.columns = ['country', 'year', 'n_fert_t']
        else:
            year_cols = [col for col in fert.columns if isinstance(col, str) and re.match(r'n_content_y\d{4}$', col.strip(), flags=re.IGNORECASE)]
            if not year_cols:
                year_cols = [col for col in fert.columns if isinstance(col, str) and re.search(r'\d{4}', col) and 'content' in col.lower()]
            if c_area and year_cols:
                id_cols = [c_area]
                c_item = _maybe_find_col(fert, ['Item', 'commodity'])
                if c_item:
                    id_cols.append(c_item)
                melted = fert[id_cols + year_cols].melt(id_vars=id_cols, value_vars=year_cols,
                                                        var_name='_year_col', value_name='n_fert_t')
                melted['year'] = pd.to_numeric(melted['_year_col'].str.extract(r'(\d{4})')[0], errors='coerce').astype('Int64')
                melted['n_fert_t'] = pd.to_numeric(melted['n_fert_t'], errors='coerce')
                melted = melted.dropna(subset=['year', 'n_fert_t'])
                melted['year'] = melted['year'].astype(int)
                z = melted.groupby([c_area, 'year'], as_index=False)['n_fert_t'].sum()
                z = z.rename(columns={c_area: 'country'})
            else:
                z = pd.DataFrame(columns=['country', 'year', 'n_fert_t'])
        if not z.empty:
            z['country'] = z['country'].astype(str).str.strip()
            z['n_fert_t'] = pd.to_numeric(z['n_fert_t'], errors='coerce')
            z = z.dropna(subset=['n_fert_t'])
            z['year'] = pd.to_numeric(z['year'], errors='coerce').astype('Int64')
            z = z.dropna(subset=['year'])
            z['year'] = z['year'].astype(int)
            z = z[z['country'].isin(universe.countries)]
            if not z.empty:
                z['iso3'] = z['country'].map(universe.iso3_by_country)
                z = z.dropna(subset=['iso3'])
                if not z.empty:
                    # extend future years using the latest historical year (2020 preferred)
                    hist_mask = z['year'] <= 2020
                    if hist_mask.any() and 2020 in z['year'].values:
                        base_year = 2020
                    else:
                        base_year = int(z['year'].max())
                    future_years = [y for y in universe.years if y > base_year]
                    if future_years:
                        base_rows = z[z['year'] == base_year][['country', 'iso3', 'n_fert_t']]
                        future_frames = []
                        for yr in future_years:
                            if base_rows.empty:
                                break
                            tmp = base_rows.copy()
                            tmp['year'] = yr
                            future_frames.append(tmp)
                        if future_frames:
                            z = pd.concat([z, pd.concat(future_frames, ignore_index=True)], ignore_index=True)
                    z = z[z['year'].isin(universe.years)]
                    z = z[['country', 'iso3', 'year', 'n_fert_t']].drop_duplicates()
                    z = z.sort_values(['country', 'year']).reset_index(drop=True)
                    fertilizers_df = z
    return {'residues_df': residues_df, 'burning_df': burning_df, 'rice_df': rice_df, 'fertilizers_df': fertilizers_df}

def build_livestock_stock_from_env(csv_path: str, universe: Universe) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=['country','iso3','year','commodity','headcount'])
    df_raw = pd.read_csv(csv_path)
    df = _lc(_faostat_wide_to_long(df_raw))
    c_area=_find_col(df,['Area']); c_year=_find_col(df,['Year']); c_item=_find_col(df,['Item']); c_val=_find_col(df,['Value'])
    z = df[[c_area,c_year,c_item,c_val]].copy()
    z.columns=['area','year','item_raw','headcount']
    m49_country = _country_by_m49(df, universe)
    z['country'] = m49_country if m49_country is not None else z['area']
    try:
        maps = load_emis_item_mappings(os.path.join(get_src_base(), 'dict_v3.xlsx'))
        z['commodity'] = z['item_raw'].map(maps.stock_map).fillna(z['item_raw'])
    except Exception:
        z['commodity'] = z['item_raw']
    z = z[z['country'].isin(universe.countries)]
    z['iso3'] = z['country'].map(universe.iso3_by_country)
    return z[['country','iso3','year','commodity','headcount']]

def build_gv_areas_from_inputs(csv_path: str, universe: Universe) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=['country','iso3','year','land_use','area_ha'])
    df_raw = pd.read_csv(csv_path)
    df = _lc(_faostat_wide_to_long(df_raw))
    c_area = _find_col(df, ['Area'])
    c_year = _find_col(df, ['Year'])
    c_item = _find_col(df, ['Item'])
    c_elem = _find_col(df, ['Element'])
    c_val  = _find_col(df, ['Value'])
    c_unit = _maybe_find_col(df, ['Unit'])

    df = df.copy()
    mask_area = df[c_elem].astype(str).str.contains('area', case=False, na=False)
    mask_excl = df[c_elem].astype(str).str.contains('share|per capita|value', case=False, na=False) | \
                df[c_item].astype(str).str.contains('per capita', case=False, na=False)
    df = df[mask_area & ~mask_excl]
    if df.empty:
        return pd.DataFrame(columns=['country','iso3','year','land_use','area_ha'])

    if c_unit:
        unit_series = df[c_unit].astype(str).str.strip().str.lower()
        factor = unit_series.map(lambda u: 1000.0 if '1000' in u else 1.0)
    else:
        factor = 1.0
    df[c_val] = pd.to_numeric(df[c_val], errors='coerce')
    df[c_val] = df[c_val].multiply(factor, axis=0)

    item_series = df[c_item].astype(str).str.strip().str.lower()
    item_to_category = {
        'cropland': 'cropland_area_ha',
        'arable land': 'cropland_area_ha',
        'permanent crops': 'cropland_area_ha',
        'temporary crops': 'cropland_area_ha',
        'forest land': 'forest_area_ha',
        'naturally regenerating forest': 'forest_area_ha',
        'planted forest': 'forest_area_ha',
        'permanent meadows and pastures': 'pasture_area_ha',
        'permanent meadows & pastures - nat. growing': 'pasture_area_ha',
        'temporary meadows and pastures': 'pasture_area_ha',
    }
    df['land_use'] = item_series.map(item_to_category)
    df = df[df['land_use'].notna()]
    df = df[[c_area, c_year, 'land_use', c_val]].copy()
    df.columns = ['area', 'year', 'land_use', 'area_ha']

    mapped = _country_by_m49(df, universe)
    if mapped is not None:
        mapped_series = mapped.astype(str)
        df['country'] = mapped_series.where(mapped_series.notna() & mapped_series.str.strip().ne(''),
                                            df['area'].astype(str).str.strip())
    else:
        df['country'] = df['area'].astype(str).str.strip()
    df = df[df['country'].isin(universe.countries)]
    df['iso3'] = df['country'].map(universe.iso3_by_country)

    hist_start = 2010
    hist_end = 2020
    df = df[(df['year'] >= hist_start) & (df['year'] <= hist_end)]

    needed_future = [y for y in universe.years if y > hist_end]
    if needed_future and not df.empty:
        latest = df.sort_values('year').drop_duplicates(['country','land_use'], keep='last')
        future_frames = []
        for y in needed_future:
            tmp = latest.copy()
            tmp['year'] = y
            future_frames.append(tmp)
        if future_frames:
            df = pd.concat([df] + future_frames, ignore_index=True)

    df['area_ha'] = pd.to_numeric(df['area_ha'], errors='coerce').fillna(0.0)
    df = df[df['area_ha'] > 0.0]
    return df[['country','iso3','year','land_use','area_ha']]

def build_land_use_fires_timeseries(csv_path: str, universe: Universe) -> pd.DataFrame:
    # 2010–2020 使用历史，未来保持均值
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=['country','iso3','year','commodity','co2e_kt'])
    df_raw = pd.read_csv(csv_path)
    df = _lc(_faostat_wide_to_long(df_raw))
    c_area=_find_col(df,['Area']); c_year=_find_col(df,['Year']); c_gas=_find_col(df,['Element','Gas','GHG']); c_val=_find_col(df,['Value'])
    z = df[[c_area,c_year,c_val]].copy(); z.columns=['area','year','co2e_kt']
    m49_country = _country_by_m49(df, universe)
    z['country'] = m49_country if m49_country is not None else z['area']
    z = z[z['country'].isin(universe.countries)]
    z['iso3'] = z['country'].map(universe.iso3_by_country)
    # fill future with mean of 2010-2020
    base = z[(z['year']>=2010)&(z['year']<=2020)].groupby(['country'], as_index=False)['co2e_kt'].mean().rename(columns={'co2e_kt':'mean_2010_2020'})
    fut = pd.DataFrame([(c,y) for c in universe.countries for y in universe.years if y>2020], columns=['country','year'])
    fut = fut.merge(base, on='country', how='left')
    fut['iso3'] = fut['country'].map(universe.iso3_by_country)
    fut['co2e_kt'] = fut['mean_2010_2020']
    fut['commodity'] = 'ALL'
    hist = z.copy(); hist['commodity']='ALL'
    out = pd.concat([hist[['country','iso3','year','commodity','co2e_kt']], fut[['country','iso3','year','commodity','co2e_kt']]], ignore_index=True)
    return out

# -------------------- price --------------------

# -------------------- constraints loaders --------------------
def load_intake_constraint(xlsx_path: str) -> Tuple[Dict[Tuple[str,int], float], Dict[str, float]]:
    """
    Read nutrition intake constraints and optional kcal-per-unit mapping.
    Returns:
      (rhs_map, kcal_map)
        rhs_map: {(country, year) -> required_kcal_total}
        kcal_map: {commodity -> kcal_per_unit}
    Heuristics are applied to match columns.
    """
    rhs_map: Dict[Tuple[str,int], float] = {}
    kcal_map: Dict[str, float] = {}
    if not os.path.exists(xlsx_path):
        return rhs_map, kcal_map
    xls = pd.ExcelFile(xlsx_path)
    # Primary sheet for RHS: try first sheet
    try:
        df = _lc(pd.read_excel(xls, xls.sheet_names[0]))
        c_area = _maybe_find_col(df, ['Country','Area'])
        c_year = _maybe_find_col(df, ['Year'])
        # RHS candidates
        cand = _maybe_find_col(df, ['kcal_min_pc','kcal_min','intake_kcal_pc','intake_kcal','rhs','value'])
        c_pop = _maybe_find_col(df, ['Population','Pop'])
        if c_area and c_year and cand:
            z = df[[c_area, c_year, cand] + ([c_pop] if c_pop else [])].copy()
            z.columns = ['country','year','val'] + (['pop'] if c_pop else [])
            if 'pop' in z.columns and z['pop'].notna().any():
                z['rhs'] = pd.to_numeric(z['val'], errors='coerce') * pd.to_numeric(z['pop'], errors='coerce')
            else:
                z['rhs'] = pd.to_numeric(z['val'], errors='coerce')
            for r in z.itertuples(index=False):
                try:
                    rhs_map[(str(r.country), int(r.year))] = float(r.rhs)
                except Exception:
                    pass
    except Exception:
        pass
    # Optional kcals per commodity mapping: try a sheet named like 'kcal' or with columns
    try:
        sheet = None
        for nm in xls.sheet_names:
            if 'kcal' in nm.lower():
                sheet = nm; break
        if sheet is None:
            sheet = xls.sheet_names[0]
        df2 = _lc(pd.read_excel(xls, sheet))
        c_comm = _maybe_find_col(df2, ['Commodity','Item','Product'])
        c_kcal = _maybe_find_col(df2, ['kcal_per_unit','kcal per unit','kcal/unit','kcal_per_ton'])
        if c_comm and c_kcal:
            for r in df2[[c_comm, c_kcal]].itertuples(index=False):
                try:
                    kcal_map[str(r[0])] = float(r[1])
                except Exception:
                    pass
    except Exception:
        pass
    return rhs_map, kcal_map

def load_land_area_limits(csv_path: str) -> Dict[Tuple[str,int], float]:
    """Parse FAOSTAT Inputs csv to extract 'Land area' by country-year.
    Returns {(country, year): land_area_ha}.
    """
    out: Dict[Tuple[str,int], float] = {}
    if not os.path.exists(csv_path):
        return out
    df_raw = pd.read_csv(csv_path)
    df = _lc(_faostat_wide_to_long(df_raw))
    c_area = _find_col(df, ['Area'])
    c_year = _find_col(df, ['Year'])
    c_elem = _find_col(df, ['Element'])
    c_val  = _find_col(df, ['Value'])
    z = df[[c_area,c_year,c_elem,c_val]].copy(); z.columns=['country','year','element','value']
    z = z[z['element'].astype(str).str.contains('land area', case=False, na=False)]
    for r in z[['country','year','value']].itertuples(index=False):
        try:
            out[(str(r[0]), int(r[1]))] = float(r[2])
        except Exception:
            pass
    return out

def build_energy_supply_rhs(fbs_csv: str, universe: Universe) -> Dict[Tuple[str,int], float]:
    """Build country-year total energy supply RHS from FAOSTAT FBS.
    Uses elements containing 'kcal/capita/day' and multiplies by population if present.
    Returns {(country, year): kcal_total_per_year}.
    """
    out: Dict[Tuple[str,int], float] = {}
    if not os.path.exists(fbs_csv):
        return out
    df_raw = pd.read_csv(fbs_csv)
    df = _lc(_faostat_wide_to_long(df_raw))
    c_area=_find_col(df,['Area']); c_year=_find_col(df,['Year']); c_elem=_find_col(df,['Element']); c_val=_find_col(df,['Value'])
    c_unit = _maybe_find_col(df, ['Unit'])
    # Energy per capita (kcal/capita/day)
    e = df[(df[c_elem].astype(str).str.contains('kcal/capita/day', case=False, na=False))]
    if not len(e):
        return out
    e2 = e.groupby([c_area, c_year], as_index=False)[c_val].sum().rename(columns={c_val:'kcal_pc_day'})
    # Population (if available)
    p = df[df[c_elem].astype(str).str.contains('population', case=False, na=False)].copy()
    if len(p):
        if c_unit and (p[c_unit].astype(str).str.contains('1000', case=False, na=False)).any():
            p[c_val] = pd.to_numeric(p[c_val], errors='coerce') * 1000.0
        p2 = p.groupby([c_area, c_year], as_index=False)[c_val].sum().rename(columns={c_val:'population'})
        z = e2.merge(p2, on=[c_area, c_year], how='left')
    else:
        z = e2.copy(); z['population'] = np.nan
    z['rhs'] = pd.to_numeric(z['kcal_pc_day'], errors='coerce') * 365.0 * pd.to_numeric(z['population'], errors='coerce')
    for r in z[[c_area, c_year, 'rhs']].itertuples(index=False):
        try:
            out[(str(r[0]), int(r[1]))] = float(r[2])
        except Exception:
            pass
    return out

# -------------------- nutrition (future) helpers --------------------
def load_nutrient_factors_from_dict_v3(xls_path: str, indicator: str) -> Dict[str, float]:
    """Read Emis_item sheet and derive nutrient-per-ton factors by commodity.
    indicator: 'energy' | 'protein' | 'fat'
    Returns {commodity -> factor_per_ton}
    """
    if not os.path.exists(xls_path):
        return {}
    xls = pd.ExcelFile(xls_path)
    try:
        df = _lc(pd.read_excel(xls, 'Emis_item'))
    except Exception:
        return {}
    c_comm = _find_col(df, ['Item_Production_Map'])
    c_kcal = 'kcal_per_100g' if 'kcal_per_100g' in df.columns else None
    c_prot = 'g_protein_per_100g' if 'g_protein_per_100g' in df.columns else None
    c_fat = 'g_fat_per_100g' if 'g_fat_per_100g' in df.columns else None
    out: Dict[str, float] = {}
    for r in df.itertuples(index=False):
        comm_val = getattr(r, c_comm)
        if pd.isna(comm_val):
            continue
        comm = str(comm_val).strip()
        if not comm or comm.lower() in {'nan', 'no'}:
            continue
        if indicator == 'energy' and c_kcal:
            v = getattr(r, c_kcal)
            try:
                val = float(v) * 10000.0  # 1 t = 10,000×100g
                if np.isfinite(val):
                    out[comm] = val
            except Exception:
                pass
        elif indicator == 'protein' and c_prot:
            v = getattr(r, c_prot)
            try:
                val = float(v) * 10000.0  # grams per ton
                if np.isfinite(val):
                    out[comm] = val
            except Exception:
                pass
        elif indicator == 'fat' and c_fat:
            v = getattr(r, c_fat)
            try:
                val = float(v) * 10000.0  # grams per ton
                if np.isfinite(val):
                    out[comm] = val
            except Exception:
                pass
    return out

def load_intake_targets(xlsx_path: str, indicator: str) -> Dict[str, float]:
    """Load per-capita-per-day intake targets by country from Intake_constraint.xlsx.
    Uses 'Indicator' column to filter rows for one of ['Energy supply','Protein supply','Fat supply'] and reads 'MEAN'.
    Returns {country -> mean_value_per_capita_per_day}.
    """
    if not os.path.exists(xlsx_path):
        return {}
    df = _lc(pd.read_excel(xlsx_path, sheet_name=0))
    c_cty = _maybe_find_col(df, ['Country','Area'])
    c_ind = _maybe_find_col(df, ['Indicator'])
    c_val = _maybe_find_col(df, ['MEAN','Mean'])
    if not (c_cty and c_ind and c_val):
        return {}
    key = {'energy':'energy', 'protein':'protein', 'fat':'fat'}[indicator]
    z = df[df[c_ind].astype(str).str.lower().str.contains(key)].copy()
    out: Dict[str, float] = {}
    for r in z[[c_cty, c_val]].itertuples(index=False):
        try:
            out[str(r[0])] = float(r[1])
        except Exception:
            pass
    return out

def load_population_wpp(csv_path: str, universe: Optional[Universe] = None) -> Dict[Tuple[str,int], float]:
    """Load population by country-year from WPP.
    Returns {(country, year): population} (persons). When ``universe`` is provided,
    country names are normalised to match the model naming (prefer M49 codes).
    """
    out: Dict[Tuple[str,int], float] = {}
    if not os.path.exists(csv_path):
        return out
    encodings = ['utf-8', 'utf-8-sig', 'gb18030', 'latin1']
    last_err = None
    for enc in encodings:
        try:
            raw = pd.read_csv(csv_path, encoding=enc)
            df = _lc(_faostat_wide_to_long(raw))
            break
        except UnicodeDecodeError as err:
            last_err = err
    else:
        raw = pd.read_csv(csv_path, encoding='utf-8', errors='replace')
        df = _lc(_faostat_wide_to_long(raw))
        if last_err:
            print(f"[load_population_wpp] WARNING: fallback to utf-8 with replacement due to encoding error: {last_err}")

    c_area = _maybe_find_col(df, ['Area','Country'])
    c_year = _maybe_find_col(df, ['Year'])
    c_val  = _maybe_find_col(df, ['Value','Population'])
    if not (c_area and c_year and c_val):
        return out

    c_elem = _maybe_find_col(df, ['Element'])
    if c_elem:
        mask_total = df[c_elem].astype(str).str.lower().str.contains('total population') & df[c_elem].astype(str).str.lower().str.contains('both')
        df = df.loc[mask_total].copy()
    if df.empty:
        return out

    unit_col = _maybe_find_col(df, ['Unit'])
    if unit_col:
        has_thousand = df[unit_col].astype(str).str.contains('1000', case=False, na=False)
        if has_thousand.any():
            df[c_val] = pd.to_numeric(df[c_val], errors='coerce') * 1000.0
        else:
            df[c_val] = pd.to_numeric(df[c_val], errors='coerce')
    else:
        df[c_val] = pd.to_numeric(df[c_val], errors='coerce')

    country_series = df[c_area].astype(str).str.strip()
    valid_countries = set(universe.countries) if universe is not None else None
    if universe is not None:
        mapped = _country_by_m49(df, universe)
        if mapped is not None:
            mapped = mapped.astype(str)
            country_series = mapped.where(mapped.notna() & mapped.str.strip().ne(''), country_series)

    for country, year_val, pop_val in zip(country_series, df[c_year], df[c_val]):
        try:
            country_name = str(country).strip()
            year = int(year_val)
            pop = float(pop_val)
        except Exception:
            continue
        if not np.isfinite(pop) or pop <= 0:
            continue
        if valid_countries is not None and country_name not in valid_countries:
            continue
        out[(country_name, year)] = pop
    return out

def build_nutrition_rhs_for_future(universe: Universe,
                                   pop_map: Dict[Tuple[str,int], float],
                                   intake_target_pc_day: Dict[str, float]) -> Dict[Tuple[str,int], float]:
    """Construct RHS only for future years (t>2020): rhs[i,t] = mean_pc_day(country) * 365 * population(i,t)."""
    rhs: Dict[Tuple[str,int], float] = {}
    for i in universe.countries:
        for t in universe.years:
            if t <= 2020:
                continue
            mean_pc_day = intake_target_pc_day.get(i)
            pop = pop_map.get((i, t))
            if mean_pc_day is None or pop is None:
                continue
            rhs[(i, t)] = float(mean_pc_day) * 365.0 * float(pop)
    return rhs

# -------------------- temperature driver --------------------
def apply_temperature_multiplier_to_nodes(temp_xlsx: str, nodes: List[Node]) -> None:
    """Apply temperature multiplier Tmult to nodes if file provides it.
    Expect a sheet with columns: Country/Area, Year, and either 'Tmult' or a delta used elsewhere.
    If not found, keep Tmult as-is.
    """
    if not os.path.exists(temp_xlsx):
        return
    try:
        df = _lc(pd.read_excel(temp_xlsx, sheet_name=0))
    except Exception:
        return
    c_area = _maybe_find_col(df, ['Country','Area'])
    c_year = _maybe_find_col(df, ['Year'])
    c_tmul = _maybe_find_col(df, ['Tmult','temp_mult','temperature_multiplier'])
    if not (c_area and c_year and c_tmul):
        return
    key = {(str(getattr(r, c_area)), int(getattr(r, c_year))): float(getattr(r, c_tmul))
           for r in df.itertuples(index=False) if pd.notna(getattr(r, c_tmul))}
    for n in nodes:
        v = key.get((n.country, n.year))
        if v is not None:
            n.Tmult = float(v)

# -------------------- yield calculation & assignment --------------------
def compute_yield_from_prod_area(production_csv: str, inputs_csv: str, universe: Universe) -> pd.DataFrame:
    """Compute yield_t_per_ha by country-commodity-year using FAOSTAT production and area harvested.
    Uses M49 for country alignment and Emis_item mappings for item→commodity if available.
    """
    if not os.path.exists(production_csv) or not os.path.exists(inputs_csv):
        return pd.DataFrame(columns=['country','year','commodity','yield_t_per_ha'])
    maps = None
    try:
        maps = load_emis_item_mappings(os.path.join(get_src_base(), 'dict_v3.xlsx'))
    except Exception:
        pass
    # Production
    dp_raw = pd.read_csv(production_csv)
    dp = _lc(_faostat_wide_to_long(dp_raw))
    c_area=_find_col(dp,['Area']); c_year=_find_col(dp,['Year']); c_item=_find_col(dp,['Item']); c_elem=_find_col(dp,['Element']); c_val=_find_col(dp,['Value'])
    zp = dp[[c_area,c_year,c_item,c_elem,c_val]].copy(); zp.columns=['area','year','item_raw','element','value']
    m49_country = _country_by_m49(dp, universe); zp['country'] = m49_country if m49_country is not None else zp['area']
    zp = zp[zp['element'].astype(str).str.contains('Production', case=False, na=False)]
    zp['commodity'] = zp['item_raw']
    if maps:
        zp['commodity'] = zp['item_raw'].map(maps.yield_map or {}).fillna(zp['commodity'])
    prod = zp.groupby(['country','year','commodity'], as_index=False)['value'].sum().rename(columns={'value':'production_t'})
    # Area harvested
    da_raw = pd.read_csv(inputs_csv)
    da = _lc(_faostat_wide_to_long(da_raw))
    c_area=_find_col(da,['Area']); c_year=_find_col(da,['Year']); c_elem=_find_col(da,['Element']); c_val=_find_col(da,['Value'])
    za = da[[c_area,c_year,c_elem,c_val]].copy(); za.columns=['area','year','element','value']
    m49_country2 = _country_by_m49(da, universe); za['country'] = m49_country2 if m49_country2 is not None else za['area']
    elem = None
    # element fallback chain
    elem = elem or (load_emis_item_mappings(os.path.join(get_src_base(), 'dict_v3.xlsx')).area_element if os.path.exists(os.path.join(get_src_base(), 'dict_v3.xlsx')) else None)
    if elem is None:
        elem = 'Area harvested'
    za = za[za['element'].astype(str).str.contains(str(elem), case=False, na=False)]
    area = za.groupby(['country','year'], as_index=False)['value'].sum().rename(columns={'value':'area_ha'})
    # Merge and compute yield
    z = prod.merge(area, on=['country','year'], how='left')
    z['yield_t_per_ha'] = z['production_t'] / z['area_ha'].replace(0, np.nan)
    z = z.dropna(subset=['yield_t_per_ha'])
    return z[['country','year','commodity','yield_t_per_ha']]

def assign_yield0_to_nodes(nodes: List[Node], yield_df: pd.DataFrame, *, hist_start:int=2010, hist_end:int=2020) -> None:
    """Assign baseline yield0 (t/ha) to nodes using average over historical years."""
    if yield_df is None or len(yield_df)==0:
        return
    df = _lc(yield_df)
    df = df[(df['year']>=hist_start)&(df['year']<=hist_end)]
    base = df.groupby(['country','commodity'], as_index=False)['yield_t_per_ha'].mean().rename(columns={'yield_t_per_ha':'yield0'})
    key = {(r.country, r.commodity): float(r.yield0) for r in base.itertuples(index=False)}
    for n in nodes:
        v = key.get((n.country, n.commodity))
        if v is not None and v > 0:
            n.meta['yield0'] = float(v)

# -------------------- demand elasticities (cross-price & income) --------------------
def load_demand_elasticities(elasticity_xlsx: str, universe: Universe) -> Tuple[Dict[str, float], Dict[str, float], Dict[Tuple[str, str], float], Dict[Tuple[str,str], Dict[str, float]]]:
    """Load demand-side elasticities from the processed elasticity workbook.
    Returns (income_by_country, pop_by_country, own_price_by_node, cross_price_by_node)."""
    eps_income: Dict[str, float] = {}
    eps_pop: Dict[str, float] = {}
    eps_own: Dict[Tuple[str, str], float] = {}
    cross: Dict[Tuple[str,str], Dict[str, float]] = {}
    if not os.path.exists(elasticity_xlsx):
        return eps_income, eps_pop, eps_own, cross
    xls = pd.ExcelFile(elasticity_xlsx)
    # demand income
    if 'Demand-Income' in xls.sheet_names:
        df = _lc(pd.read_excel(xls, 'Demand-Income'))
        if {'Country', 'Elasticity_mean'}.issubset(df.columns):
            for r in df[['Country', 'Elasticity_mean']].itertuples(index=False):
                val = pd.to_numeric(getattr(r, 'Elasticity_mean'), errors='coerce')
                if pd.notna(val):
                    eps_income[str(getattr(r, 'Country'))] = float(val)
    # demand population
    if 'Demand-Population' in xls.sheet_names:
        df = _lc(pd.read_excel(xls, 'Demand-Population'))
        if {'Country', 'Elasticity_mean'}.issubset(df.columns):
            for r in df[['Country', 'Elasticity_mean']].itertuples(index=False):
                val = pd.to_numeric(getattr(r, 'Elasticity_mean'), errors='coerce')
                if pd.notna(val):
                    eps_pop[str(getattr(r, 'Country'))] = float(val)
    # demand own-price
    if 'Demand-Own-Price' in xls.sheet_names:
        df = _lc(pd.read_excel(xls, 'Demand-Own-Price'))
        if {'Country', 'Commodity', 'Elasticity_mean'}.issubset(df.columns):
            for r in df[['Country', 'Commodity', 'Elasticity_mean']].itertuples(index=False):
                val = pd.to_numeric(getattr(r, 'Elasticity_mean'), errors='coerce')
                if pd.notna(val):
                    key = (str(getattr(r, 'Country')), str(getattr(r, 'Commodity')))
                    eps_own[key] = float(val)
    # demand cross-price matrix
    if 'Demand_Cross_mean' in xls.sheet_names:
        df = _lc(pd.read_excel(xls, 'Demand_Cross_mean'))
        if {'Country', 'Commodity'}.issubset(df.columns):
            target_cols = [c for c in df.columns if c not in {'Country', 'Commodity', 'Country_label'}]
            for _, r in df.iterrows():
                i = str(r['Country']); j = str(r['Commodity'])
                if i not in universe.countries or j not in universe.commodities:
                    continue
                row: Dict[str, float] = {}
                for c2 in target_cols:
                    if c2 not in universe.commodities:
                        continue
                    v = pd.to_numeric(r[c2], errors='coerce')
                    if pd.notna(v):
                        row[c2] = float(v)
                if row:
                    cross[(i, j)] = row
    return eps_income, eps_pop, eps_own, cross

def apply_demand_elasticities_to_nodes(nodes: List[Node], universe: Universe, elasticity_xlsx: str) -> None:
    eps_income, eps_pop, eps_own, cross = load_demand_elasticities(elasticity_xlsx, universe)
    for n in nodes:
        # income/population elasticity
        setattr(n, 'eps_income_demand', float(eps_income.get(n.country, 0.0)))
        setattr(n, 'eps_pop_demand', float(eps_pop.get(n.country, 0.0)))
        setattr(n, 'eps_demand', float(eps_own.get((n.country, n.commodity), 0.0)))
        # cross-price row dict
        eps_row = cross.get((n.country, n.commodity), {})
        setattr(n, 'epsD_row', dict(eps_row))
def load_prices(csv_path: str, universe: Universe) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=['country','iso3','year','commodity','price'])

    try:
        df_raw = pd.read_excel(csv_path, sheet_name=0)
    except Exception:
        df_raw = pd.read_excel(csv_path)
    df = _lc(df_raw)

    required = ['Item', 'Unit']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"load_prices: missing required columns {missing}")

    unit_norm = df['Unit'].astype(str).str.replace('–', '-').str.strip().str.lower()
    valid_units = {
        'int$ (2014-2016 const) per tonne',
        'int$ (2014-2016 const) per m3'
    }
    df = df[unit_norm.isin(valid_units)].copy()
    if 'Area' in df.columns:
        df = df[df['Area'].astype(str).str.strip().str.lower() == 'world']
    if df.empty:
        return pd.DataFrame(columns=['country','iso3','year','commodity','price'])

    year_cols = [c for c in df.columns if isinstance(c, str) and c.strip().startswith('Y') and c.strip()[1:].isdigit()]
    if not year_cols:
        return pd.DataFrame(columns=['country','iso3','year','commodity','price'])

    for col in year_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    dict_path = os.path.join(get_src_base(), 'dict_v3.xlsx')
    mapping_df = _lc(pd.read_excel(dict_path, sheet_name='Emis_item'))
    price_col = 'Item_Price_Map'
    prod_col = 'Item_Production_Map'
    price_map: Dict[str, str] = {}
    for r in mapping_df[[price_col, prod_col]].dropna().itertuples(index=False):
        price_name = str(getattr(r, price_col)).strip()
        prod_name = str(getattr(r, prod_col)).strip()
        if not price_name or not prod_name:
            continue
        if price_name.lower() in {'nan', 'no'} or prod_name.lower() in {'nan', 'no'}:
            continue
        price_map[price_name.lower()] = prod_name

    def _map_item(name: str) -> Optional[str]:
        if name is None:
            return None
        key = str(name).strip().lower()
        return price_map.get(key)

    df['commodity'] = df['Item'].map(_map_item)
    df = df[df['commodity'].isin(universe.commodities)]
    if df.empty:
        return pd.DataFrame(columns=['country','iso3','year','commodity','price'])

    long_df = df.melt(id_vars=['commodity'], value_vars=year_cols, var_name='year', value_name='price')
    long_df['year'] = pd.to_numeric(long_df['year'].astype(str).str.strip().str.lstrip('Y'), errors='coerce')
    long_df['price'] = pd.to_numeric(long_df['price'], errors='coerce')
    long_df = long_df.dropna(subset=['year', 'price'])
    long_df['year'] = long_df['year'].astype(int)

    rows: List[Dict[str, Any]] = []
    for r in long_df.itertuples(index=False):
        year = int(r.year)
        price_val = float(r.price)
        commodity = str(r.commodity)
        for country in universe.countries:
            rows.append({
                'country': country,
                'iso3': universe.iso3_by_country.get(country, ''),
                'year': year,
                'commodity': commodity,
                'price': price_val
            })

    if not rows:
        return pd.DataFrame(columns=['country','iso3','year','commodity','price'])

    out = pd.DataFrame(rows)
    return out[['country','iso3','year','commodity','price']]

# -------------------- attach emission factors via FAO modules --------------------
def attach_emission_factors_from_fao_modules(nodes: List[Node], params_wide: Optional[pd.DataFrame],
                                             production_df: pd.DataFrame,
                                             crop_activity: Dict[str, pd.DataFrame],
                                             livestock_activity: Dict[str, pd.DataFrame],
                                             soils_activity: Dict[str, pd.DataFrame],
                                             forest_activity: Optional[Dict[str, pd.DataFrame]],
                                             module_paths: Dict[str, str]) -> None:
    """
    按节点 (i,j,t) 调用上传的 *_fao.py 模块（占位接口，示范如何把结果写入 n.e0_by_proc）
    实际函数/参数名以用户上传模块为准，这里做最小可用模板：若模块可导入，就用其函数；否则 e0_by_proc 留空。
    """
    import importlib.util
    def _safe_import(path):
        spec = importlib.util.spec_from_file_location("mod_"+os.path.basename(path).replace('.py',''), path)
        if spec and spec.loader:
            m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
        return None
    mods = {k:_safe_import(v) for k,v in module_paths.items() if os.path.exists(v)}

    # 这里仅做演示：如果存在某模块且暴露 get_default_intensity(process, commodity) 就调用
    for n in nodes:
        e = {}
        for p in []:  # 若你希望先用 dict_v3 的 processes 做一个空壳：for p in universe.processes
            pass
        # 尝试模块接口：例如 gfe/gce/gle 提供的按商品默认强度
        for key, m in mods.items():
            if not m: continue
            for cand in ['get_default_intensity','get_emission_intensity','intensity_for']:
                f = getattr(m, cand, None)
                if callable(f):
                    try:
                        val = float(f(process='ALL', commodity=n.commodity))  # 假设接口
                        if val>0:
                            e[key.replace('_module_fao.py','')] = val
                    except Exception:
                        pass
        n.e0_by_proc = e
def load_income_multipliers_from_sspdb(xlsx_path: str, scenario: str, universe: Universe) -> Dict[Tuple[str,int], float]:
    """Load GDP|PPP from SSPDB_future_GDP.xlsx and compute per-country multipliers relative to 2020.
    Mapping uses dict_v3.region sheet's Region_map_SSPDB via Universe.ssp_region_by_country.
    Returns {(country, year): multiplier} where multiplier=GDP(region,year)/GDP(region,2020).
    """
    out: Dict[Tuple[str,int], float] = {}
    if not os.path.exists(xlsx_path):
        return out
    try:
        df = _lc(pd.read_excel(xlsx_path, sheet_name='SSPDB_future_GDP'))
    except Exception:
        return out
    c_scen = _maybe_find_col(df, ['SCENARIO','Scenario'])
    c_var  = _maybe_find_col(df, ['VARIABLE','Variable'])
    c_reg  = _maybe_find_col(df, ['REGION','Region'])
    c_year = _maybe_find_col(df, ['YEAR','Year'])
    c_val  = _maybe_find_col(df, ['VALUE','Value'])
    if not (c_scen and c_var and c_reg and c_year and c_val):
        return out
    g = df[(df[c_scen].astype(str)==str(scenario)) & (df[c_var].astype(str).str.contains('gdp|gdp|ppp', case=False, na=False))]
    if not len(g):
        return out
    # region-year table
    z = g[[c_reg, c_year, c_val]].copy(); z.columns=['region','year','value']
    # base 2020 by region
    base = z[z['year']==2020].set_index('region')['value'].to_dict()
    if not base:
        return out
    for r in z.itertuples(index=False):
        reg = str(r.region); y = int(r.year); v = float(r.value)
        b = float(base.get(reg, float('nan')))
        if not (b and b!=0):
            continue
        mult = v / b
        # assign to all countries mapped to this region
        for i in universe.countries:
            if (universe.ssp_region_by_country or {}).get(i) == reg:
                out[(i, y)] = mult
    # Fill <=2020 as 1.0
    for i in universe.countries:
        for y in universe.years:
            if y <= 2020 and (i, y) not in out:
                out[(i, y)] = 1.0
    return out

# -------------------- FAO modules runner (lme integration) --------------------
def run_fao_modules_and_cache(nodes: List[Node], *, livestock_stock_df: pd.DataFrame, module_paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """Run selected *_emissions_module_fao modules (notably lme_manure_module_fao) and cache results.
    Returns {module_key: df}. Keeps a copy under LAST_FAO_RUNS.
    """
    import importlib.util
    from config_paths import get_input_base, get_src_base

    def _safe_import(path: str):
        spec = importlib.util.spec_from_file_location("mod_"+os.path.basename(path).replace('.py',''), path)
        if spec and spec.loader:
            m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
        return None

    out: Dict[str, pd.DataFrame] = {}
    try:
        mods = {os.path.basename(k): _safe_import(v) for k, v in (module_paths or {}).items() if os.path.exists(v)}
    except Exception:
        mods = {}

    # lme manure
    for fname, mod in mods.items():
        if not mod: continue
        if 'lme_manure_module_fao' in fname:
            try:
                load_wide = getattr(mod, 'load_parameters_wide', None)
                run_wide = getattr(mod, 'run_lme_from_wide', None)
                if callable(load_wide) and callable(run_wide):
                    path_params = os.path.join(get_input_base(), 'Livestock_Manure_parameters.xlsx')
                    P = load_wide(path_params) if os.path.exists(path_params) else None
                    pop = livestock_stock_df.copy() if isinstance(livestock_stock_df, pd.DataFrame) else pd.DataFrame()
                    if P is not None and len(pop):
                        # AreaCode from dict_v3.xlsx::region M49
                        try:
                            xls = pd.ExcelFile(os.path.join(get_src_base(), 'dict_v3.xlsx'))
                            region = _lc(pd.read_excel(xls, 'region'))
                            c_cty = _find_col(region, ['Country'])
                            c_m49 = _maybe_find_col(region, ['M49 Code','M49','M49 Code_xxx','Area Code (M49)'])
                            m49_by_country = dict(zip(region[c_cty].astype(str), region[c_m49].astype(int))) if c_m49 else {}
                        except Exception:
                            m49_by_country = {}
                        z = _lc(pop)
                        c_cty = _find_col(z, ['country'])
                        c_year= _find_col(z, ['year'])
                        c_comm= _find_col(z, ['commodity'])
                        c_head= _find_col(z, ['headcount'])
                        z['AreaCode'] = z[c_cty].map(m49_by_country).fillna(0).astype(int)
                        z['ItemName'] = z[c_comm]
                        z = z.rename(columns={c_year:'year', c_head:'head'})[['AreaCode','year','ItemName','head']]
                        lme_df = run_wide(P, z, years=sorted(z['year'].unique().tolist()), itemname_col='ItemName', head_col='head')
                        out['lme'] = lme_df
                        try:
                            global LAST_FAO_RUNS
                            LAST_FAO_RUNS['lme'] = lme_df
                        except Exception:
                            pass
            except Exception:
                pass
    return out
@dataclass
class EmisItemMappings:
    production_map: Dict[str, str]
    fertilizer_map: Dict[str, str]
    yield_map: Dict[str, str]
    yield_element: Optional[str]
    yield_unit: Optional[str]
    area_element: Optional[str]
    slaughtered_map: Dict[str, str]
    slaughtered_element: Optional[str]
    slaughtered_unit: Optional[str]
    stock_map: Dict[str, str]
    stock_element: Optional[str]
    stock_unit: Optional[str]
    elasticity_map: Dict[str, str]
    feed_map: Dict[str, str]

def load_emis_item_mappings(xls_path: str) -> EmisItemMappings:
    """Parse dict_v3.xlsx Emis_item sheet for multi-domain item mappings and units.
    Returns a structured mapping object for consistent FAOSTAT alignment.
    """
    def series_to_map(sr) -> Dict[str, str]:
        m: Dict[str, str] = {}
        for r in df.itertuples(index=False):
            try:
                m[str(getattr(r, c_map))] = str(getattr(r, c_ref))
            except Exception:
                pass
        return m

    if not os.path.exists(xls_path):
        return EmisItemMappings({}, {}, {}, None, None, None, {}, None, None, {}, None, None, {}, {})
    xls = pd.ExcelFile(xls_path)
    df = _lc(pd.read_excel(xls, 'Emis_item'))

    # columns
    def col(name: str) -> Optional[str]:
        return name if name in df.columns else None

    c_map = col('Item_Production_Map')
    c_prod = col('Item_Production_Map')
    c_fert = col('Item_Fertilizer_Map')
    c_yield = col('Item_Yield_Map')
    c_yield_elem = col('Item_Yield_Element')
    c_yield_unit = col('Item_Yield_Unit')
    c_area_elem = col('Item_Area_Element')
    c_sl_map = col('Item_Slaughtered_Map')
    c_sl_elem = col('Item_Slaughtered_Element')
    c_sl_unit = col('Item_Slaughtered_Unit')
    c_stock_map = col('Item_Stock_Map')
    c_stock_elem = col('Item_Stock_Element')
    c_stock_unit = col('Item_Stock_Unit')
    c_elast = col('Item_Elasticity_Map')
    c_feed = col('Item_Feed_Map')

    # build maps (map columns use their own value as key; reference columns may be different)
    def build_map(c_from: Optional[str], c_to: Optional[str]) -> Dict[str, str]:
        m: Dict[str, str] = {}
        if not (c_from and c_to):
            return m
        for r in df.itertuples(index=False):
            a = getattr(r, c_from, None); b = getattr(r, c_to, None)
            if pd.isna(a) or pd.isna(b):
                continue
            a_str = str(a).strip(); b_str = str(b).strip()
            if not a_str or not b_str:
                continue
            if a_str.lower() in {'nan', 'no'} or b_str.lower() in {'nan', 'no'}:
                continue
            m[a_str] = b_str
        return m

    return EmisItemMappings(
        production_map=build_map(c_prod, c_prod),
        fertilizer_map=build_map(c_fert, c_fert),
        yield_map=build_map(c_yield, c_yield),
        yield_element=c_yield_elem,
        yield_unit=c_yield_unit,
        area_element=c_area_elem,
        slaughtered_map=build_map(c_sl_map, c_sl_map),
        slaughtered_element=c_sl_elem,
        slaughtered_unit=c_sl_unit,
        stock_map=build_map(c_stock_map, c_stock_map),
        stock_element=c_stock_elem,
        stock_unit=c_stock_unit,
        elasticity_map=build_map(c_elast, c_elast),
        feed_map=build_map(c_feed, c_feed),
    )
