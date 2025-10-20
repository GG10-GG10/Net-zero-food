# -*- coding: utf-8 -*-
"""
S2.0_load_data — 数据读取与构造（对齐 dict_v3 + FAOSTAT 文件 + 情景管道）
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
import os
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
    production_faostat_csv: str = os.path.join(get_input_base(), "Production_Crops_Livestock_E_All_Data_NOFLAG.csv")
    fbs_csv: str = os.path.join(get_input_base(), "FoodBalanceSheets_E_All_Data_NOFLAG.csv")
    livestock_patterns_csv: str = os.path.join(get_input_base(), "Environment_LivestockPatterns_E_All_Data_NOFLAG.csv")
    inputs_landuse_csv: str = os.path.join(get_input_base(), "Inputs_LandUse_E_All_Data_NOFLAG.csv")
    historical_fert_xlsx: str = os.path.join(get_input_base(), "Historical_Fertilizer_application.xlsx")
    prices_csv: str = os.path.join(get_input_base(), "Prices_E_All_Data_NOFLAG.csv")
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

def _country_by_m49(df: pd.DataFrame, universe: Universe) -> Optional[pd.Series]:
    """Return a Series of mapped country names using M49 codes if available, else None."""
    # Detect area code column for M49
    c_m49 = _maybe_find_col(df, ['Area Code (M49)','M49 Code','Area Code','Area Code M49'])
    if not c_m49:
        return None
    # Build reverse map M49->country from Universe
    code_to_country = {str(v): str(k) for k, v in (universe.m49_by_country or {}).items() if v is not None}
    if not code_to_country:
        return None
    vals = df[c_m49].astype(str).map(code_to_country)
    return vals

# -------------------- universe --------------------
def build_universe_from_dict_v3(path: str, config: ScenarioConfig) -> Universe:
    xls = pd.ExcelFile(path)
    region = _lc(pd.read_excel(xls, 'region'))
    emis_proc = _lc(pd.read_excel(xls, 'Emis_process'))
    emis_item = _lc(pd.read_excel(xls, 'Emis_item'))

    c_country = _find_col(region, ['Country'])
    c_iso3 = _find_col(region, ['ISO3'])
    c_label = _find_col(region, ['Region_label_new'])
    region2 = region[region[c_label].astype(str).str.lower() != 'no'][[c_country, c_iso3, c_label]].drop_duplicates()
    countries = region2[c_country].astype(str).tolist()
    iso3_map = dict(zip(region2[c_country].astype(str), region2[c_iso3].astype(str)))

    # Region_aggMC map
    c_ragg = _find_col(region, ['Region_aggMC'])
    region_aggMC_by_country = dict(zip(region[country:=c_country].astype(str), region[c_ragg].astype(str)))
    # SSP region map
    c_ssp = _maybe_find_col(region, ['Region_map_SSPDB','Region_map_SSP'])
    ssp_region_by_country = dict(zip(region[c_country].astype(str), region[c_ssp].astype(str))) if c_ssp else {}
    # M49 code map (if present)
    m49_col = _maybe_find_col(region, ['M49','M49 Code','M49_Code','M49 Code_xxx'])
    m49_by_country = dict(zip(region[c_country].astype(str), region[m49_col].astype(str))) if m49_col else {}

    # Processes & meta
    c_proc = _find_col(emis_proc, ['Process'])
    processes = emis_proc[c_proc].astype(str).str.strip().dropna().unique().tolist()
    process_meta = {}
    c_cat = None
    for nm in ['category','Category','Sector']:
        if nm in emis_proc.columns: c_cat = nm; break
    c_gas = None
    for nm in ['gas','GHG','Gas']:
        if nm in emis_proc.columns: c_gas = nm; break
    for r in emis_proc.itertuples(index=False):
        name = getattr(r, c_proc)
        if pd.isna(name): continue
        d = {}
        if c_cat: d['category'] = str(getattr(r, c_cat))
        if c_gas: d['gas'] = str(getattr(r, c_gas))
        process_meta[str(name)] = d

    # Commodities
    c_itemmap = _find_col(emis_item, ['Item Production Map'])
    commodities = sorted(pd.unique(emis_item[c_itemmap].astype(str)))

    # Cat2 mapping
    c_cat2 = _find_col(emis_item, ['Item Cat2'])
    item_cat2_by_commodity = {}
    for r in emis_item.itertuples(index=False):
        item_cat2_by_commodity[str(getattr(r, c_itemmap))] = str(getattr(r, c_cat2))

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
    # assume a sheet with columns: Country, Commodity, eps_supply_temp, eps_supply_yield
    df = _lc(pd.read_excel(xls, xls.sheet_names[0]))
    cn = _find_col(df, ['Country'])
    in_ = _find_col(df, ['Commodity'])
    et = _find_col(df, ['eps_supply_temp'])
    ey = _find_col(df, ['eps_supply_yield'])
    key = {(str(getattr(r, cn)), str(getattr(r, in_))): (float(getattr(r, et)), float(getattr(r, ey)))
           for r in df.itertuples(index=False)}
    for n in nodes:
        v = key.get((n.country, n.commodity))
        if v:
            n.eps_supply_temp, n.eps_supply_yield = v

# -------------------- FBS demand --------------------
def build_demand_components_from_fbs(fbs_csv: str, universe: Universe) -> pd.DataFrame:
    if not os.path.exists(fbs_csv):
        return pd.DataFrame(columns=['country','iso3','year','commodity','food_t','feed_t','seed_t','demand_total_t'])
    df = _lc(pd.read_csv(fbs_csv))
    # Very generic mapping: assume columns Area, Year, Item, Element, Value; prefer M49 alignment
    c_area = _find_col(df, ['Area'])
    c_year = _find_col(df, ['Year'])
    c_item = _find_col(df, ['Item'])
    c_elem = _find_col(df, ['Element'])
    c_val  = _find_col(df, ['Value'])
    z = df[[c_area,c_year,c_item,c_elem,c_val]].copy()
    z.columns = ['area','year','item_raw','element','value']
    m49_country = _country_by_m49(df, universe)
    z['country'] = m49_country if m49_country is not None else z['area']
    # Map item to internal commodity via dict_v3 mapping (production map) if possible
    try:
        maps = load_emis_item_mappings(os.path.join(get_src_base(), 'dict_v3.xlsx'))
        z['commodity'] = z['item_raw'].map(maps.production_map).fillna(z['item_raw'])
    except Exception:
        z['commodity'] = z['item_raw']
    z = z[z['country'].isin(universe.countries)]
    pivot = z.pivot_table(index=['country','year','commodity'], columns='element', values='value', aggfunc='sum').reset_index()
    # normalize element names
    cols = {c: c.lower() for c in pivot.columns}
    pivot.rename(columns=cols, inplace=True)
    food = pivot.get('food', pd.Series(0,index=pivot.index))
    feed = pivot.get('feed', pd.Series(0,index=pivot.index))
    seed = pivot.get('seed', pd.Series(0,index=pivot.index))
    out = pd.DataFrame({
        'country': pivot['country'],
        'iso3': [universe.iso3_by_country.get(c,'') for c in pivot['country']],
        'year': pivot['year'],
        'commodity': pivot['commodity'],
        'food_t': food.fillna(0.0).astype(float),
        'feed_t': feed.fillna(0.0).astype(float),
        'seed_t': seed.fillna(0.0).astype(float),
    })
    out['demand_total_t'] = out['food_t'] + out['feed_t'] + out['seed_t']
    return out

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
    df = _lc(pd.read_csv(csv_path))
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
    if len(fert):
        fert = _lc(fert)
        # try common columns
        c_area = _find_col(fert, ['Area','country']); c_year=_find_col(fert,['Year','year']); c_val=_find_col(fert,['Value','n_fert_t'])
        fert = fert[[c_area,c_year,c_val]]; fert.columns=['country','year','n_fert_t']
    fertilizers_df = fert
    return {'residues_df': residues_df, 'burning_df': burning_df, 'rice_df': rice_df, 'fertilizers_df': fertilizers_df}

def build_livestock_stock_from_env(csv_path: str, universe: Universe) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=['country','iso3','year','commodity','headcount'])
    df = _lc(pd.read_csv(csv_path))
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
    df = _lc(pd.read_csv(csv_path))
    c_area=_find_col(df,['Area']); c_year=_find_col(df,['Year']); c_elem=_find_col(df,['Element']); c_val=_find_col(df,['Value'])
    # land uses aggregate mapping
    z = df[[c_area,c_year,c_elem,c_val]].copy()
    z.columns=['area','year','land_use','area_ha']
    m49_country = _country_by_m49(df, universe)
    z['country'] = m49_country if m49_country is not None else z['area']
    z = z[z['country'].isin(universe.countries)]
    z['iso3'] = z['country'].map(universe.iso3_by_country)
    return z[['country','iso3','year','land_use','area_ha']]

def build_land_use_fires_timeseries(csv_path: str, universe: Universe) -> pd.DataFrame:
    # 2010–2020 使用历史，未来保持均值
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=['country','iso3','year','commodity','co2e_kt'])
    df = _lc(pd.read_csv(csv_path))
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
    df = _lc(pd.read_csv(csv_path))
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
    df = _lc(pd.read_csv(fbs_csv))
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
    c_comm = _find_col(df, ['Item Production Map'])
    # columns for nutrients
    c_kcal = _maybe_find_col(df, ['kcal_per_100g', 'kcal per 100g', 'kcal/100g'])
    c_prot = _maybe_find_col(df, ['g_protein_per_100g', 'protein_per_100g', 'protein (g/100g)'])
    c_fat  = _maybe_find_col(df, ['g_fat_per_100g', 'fat_per_100g', 'fat (g/100g)'])
    out: Dict[str, float] = {}
    for r in df.itertuples(index=False):
        comm = str(getattr(r, c_comm))
        if indicator == 'energy' and c_kcal:
            v = getattr(r, c_kcal)
            try:
                out[comm] = float(v) * 10000.0  # 1 t = 10,000×100g
            except Exception:
                pass
        elif indicator == 'protein' and c_prot:
            v = getattr(r, c_prot)
            try:
                out[comm] = float(v) * 10000.0  # grams per ton
            except Exception:
                pass
        elif indicator == 'fat' and c_fat:
            v = getattr(r, c_fat)
            try:
                out[comm] = float(v) * 10000.0  # grams per ton
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

def load_population_wpp(csv_path: str) -> Dict[Tuple[str,int], float]:
    """Load population by country-year from WPP.
    Returns {(country, year): population} (persons).
    """
    out: Dict[Tuple[str,int], float] = {}
    if not os.path.exists(csv_path):
        return out
    df = _lc(pd.read_csv(csv_path))
    c_area = _maybe_find_col(df, ['Area','Country'])
    c_year = _maybe_find_col(df, ['Year'])
    c_val  = _maybe_find_col(df, ['Value','Population'])
    if not (c_area and c_year and c_val):
        return out
    for r in df[[c_area, c_year, c_val]].itertuples(index=False):
        try:
            out[(str(r[0]), int(r[1]))] = float(r[2])
        except Exception:
            pass
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
    dp = _lc(pd.read_csv(production_csv))
    c_area=_find_col(dp,['Area']); c_year=_find_col(dp,['Year']); c_item=_find_col(dp,['Item']); c_elem=_find_col(dp,['Element']); c_val=_find_col(dp,['Value'])
    zp = dp[[c_area,c_year,c_item,c_elem,c_val]].copy(); zp.columns=['area','year','item_raw','element','value']
    m49_country = _country_by_m49(dp, universe); zp['country'] = m49_country if m49_country is not None else zp['area']
    zp = zp[zp['element'].astype(str).str.contains('Production', case=False, na=False)]
    zp['commodity'] = zp['item_raw']
    if maps:
        zp['commodity'] = zp['item_raw'].map(maps.yield_map or {}).fillna(zp['commodity'])
    prod = zp.groupby(['country','year','commodity'], as_index=False)['value'].sum().rename(columns={'value':'production_t'})
    # Area harvested
    da = _lc(pd.read_csv(inputs_csv))
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
def load_demand_elasticities(elasticity_xlsx: str, universe: Universe) -> Tuple[Dict[str, float], Dict[str, float], Dict[Tuple[str,str], Dict[str, float]]]:
    """Load demand income & population elasticities by country and demand cross-price matrix.
    Returns (eps_income_by_country, eps_pop_by_country, cross_eps_by_node) where cross_eps_by_node[(country, commodity)] = {commodity2: eps}.
    """
    eps_income: Dict[str, float] = {}
    eps_pop: Dict[str, float] = {}
    cross: Dict[Tuple[str,str], Dict[str, float]] = {}
    if not os.path.exists(elasticity_xlsx):
        return eps_income, eps_pop, cross
    xls = pd.ExcelFile(elasticity_xlsx)
    # demand income
    sheet_income = None
    for s in xls.sheet_names:
        if 'demand' in s.lower() and 'income' in s.lower(): sheet_income = s; break
    if sheet_income is not None:
        df = _lc(pd.read_excel(xls, sheet_income))
        c_cty = _maybe_find_col(df, ['Country','Area','Region_label_new','Region'])
        c_val = _maybe_find_col(df, ['Elasticity_mean','mean','value','eps'])
        if c_cty and c_val:
            for r in df[[c_cty, c_val]].itertuples(index=False):
                try:
                    eps_income[str(r[0])] = float(r[1])
                except Exception:
                    pass
    # demand population (if available)
    sheet_pop = None
    for s in xls.sheet_names:
        if 'demand' in s.lower() and ('population' in s.lower() or 'pop' in s.lower()): sheet_pop = s; break
    if sheet_pop is not None:
        df = _lc(pd.read_excel(xls, sheet_pop))
        c_cty = _maybe_find_col(df, ['Country','Area','Region_label_new','Region'])
        c_val = _maybe_find_col(df, ['Elasticity_mean','mean','value','eps'])
        if c_cty and c_val:
            for r in df[[c_cty, c_val]].itertuples(index=False):
                try:
                    eps_pop[str(r[0])] = float(r[1])
                except Exception:
                    pass
    # demand cross price wide matrix
    sheet_cross = None
    for s in xls.sheet_names:
        if 'demand' in s.lower() and 'cross' in s.lower(): sheet_cross = s; break
    if sheet_cross is not None:
        df = _lc(pd.read_excel(xls, sheet_cross))
        c_cty = _maybe_find_col(df, ['Country','Area','Region_label_new','Region'])
        c_comm = _maybe_find_col(df, ['Commodity','Item'])
        if c_cty and c_comm:
            # treat all other columns as commodities (targets), take numeric values
            cols = [c for c in df.columns if c not in (c_cty, c_comm)]
            for _, r in df.iterrows():
                i = str(r[c_cty]); j = str(r[c_comm])
                if i not in universe.countries or j not in universe.commodities:
                    continue
                row: Dict[str, float] = {}
                for c2 in cols:
                    v = pd.to_numeric(r[c2], errors='coerce')
                    if pd.notna(v) and c2 in universe.commodities:
                        row[c2] = float(v)
                if row:
                    cross[(i, j)] = row
    return eps_income, eps_pop, cross

def apply_demand_elasticities_to_nodes(nodes: List[Node], universe: Universe, elasticity_xlsx: str) -> None:
    eps_income, eps_pop, cross = load_demand_elasticities(elasticity_xlsx, universe)
    for n in nodes:
        # income/population elasticity
        setattr(n, 'eps_income_demand', float(eps_income.get(n.country, 0.0)))
        setattr(n, 'eps_pop_demand', float(eps_pop.get(n.country, 0.0)))
        # cross-price row dict
        eps_row = cross.get((n.country, n.commodity), {})
        setattr(n, 'epsD_row', dict(eps_row))
def load_prices(csv_path: str, universe: Universe) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=['country','iso3','year','commodity','price'])
    df = _lc(pd.read_csv(csv_path))
    c_area=_find_col(df,['Area']); c_year=_find_col(df,['Year']); c_item=_find_col(df,['Item']); c_val=_find_col(df,['Value','Price'])
    z = df[[c_area,c_year,c_item,c_val]].copy()
    z.columns=['country','year','commodity','price']
    z = z[z['country'].isin(universe.countries)]
    z['iso3'] = z['country'].map(universe.iso3_by_country)
    return z[['country','iso3','year','commodity','price']]

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
    def col(*names):
        return _maybe_find_col(df, list(names))

    c_map = col('Item Production Map')
    c_prod = col('Item Production Map')
    c_fert = col('Item Fertilizer Map')
    c_yield = col('Item Yield Map')
    c_yield_elem = col('Item Yield Element')
    c_yield_unit = col('Item Yield Unit')
    c_area_elem = col('Item Area Element')
    c_sl_map = col('Item Slaughtered Map')
    c_sl_elem = col('Item Slaughtered Element')
    c_sl_unit = col('Item Slaughtered Unit')
    c_stock_map = col('Item Stock Map')
    c_stock_elem = col('Item Stock Element')
    c_stock_unit = col('Item Stock Unit')
    c_elast = col('Item Elasticity Map')
    c_feed = col('Item Feed Map')

    # build maps (map columns use their own value as key; reference columns may be different)
    def build_map(c_from: Optional[str], c_to: Optional[str]) -> Dict[str, str]:
        m: Dict[str, str] = {}
        if not (c_from and c_to):
            return m
        for r in df.itertuples(index=False):
            a = getattr(r, c_from, None); b = getattr(r, c_to, None)
            if pd.notna(a) and pd.notna(b):
                m[str(a)] = str(b)
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
