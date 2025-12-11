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
import sys
import xarray as xr
from config_paths import get_input_base, get_src_base

from S1_0_schema import Universe, ScenarioConfig, Node

# -------------------- 文件检查辅助函数 --------------------
def _check_file_exists(file_path: str, file_description: str, critical: bool = True) -> bool:
    """
    检查文件是否存在，如果不存在则报错
    
    Args:
        file_path: 文件路径
        file_description: 文件描述（用于错误消息）
        critical: 是否是关键文件（True则退出程序）
    
    Returns:
        文件是否存在
    """
    if not os.path.exists(file_path):
        error_msg = f"\n{'='*80}\n❌ 错误: 找不到{file_description}\n文件路径: {file_path}\n{'='*80}\n"
        print(error_msg, file=sys.stderr)
        if critical:
            sys.exit(1)
        return False
    return True


def _norm_m49(val) -> str:
    """
    规范化M49代码为标准'xxx格式（单引号+3位数字）
    例如: "'156'" -> "'156", "0156" -> "'156", 156 -> "'156", "156" -> "'156"
    """
    try:
        s = str(val).strip().lstrip("'\"")
        if s.isdigit():
            return f"'{int(s):03d}"  # 标准'xxx格式
        return f"'{s}" if not s.startswith("'") else s
    except Exception:
        return str(val)

# -------------------- paths --------------------
@dataclass
class DataPaths:
    base: str = get_input_base()
    # config/dictionaries under src
    dict_v3_path: str = os.path.join(get_src_base(), "dict_v3.xlsx")
    scenario_config_xlsx: str = os.path.join(get_src_base(), "Scenario_config_new.xlsx")
    elasticity_xlsx: str = os.path.join(get_src_base(), "Elasticity_v3_processed_filled_by_region.xlsx")
    feed_coeff_xlsx: str = os.path.join(get_input_base(), "Land", "Feed_pasture", "Feed_need_per_head_by_country_livestcok_refilled.xlsx")
    feed_need_xlsx: str = os.path.join(get_input_base(), "Land", "Feed_pasture", "Feed_need_per_head_by_country_livestcok_refilled.xlsx")
    grass_ratio_xlsx: str = os.path.join(get_input_base(), "Land", "Feed_pasture", "Grass_feed_ratio_by_country_livestock_refilled.xlsx")
    pasture_dm_yield_xlsx: str = os.path.join(get_input_base(), "Land", "Feed_pasture", "Pasture_DM_yield_by_country.xlsx")
    # inputs
    production_faostat_csv: str = os.path.join(get_input_base(), "Production_Trade", "Production_Crops_Livestock_E_All_Data_NOFLAG_yield_refilled.csv")
    fbs_csv: str = os.path.join(get_input_base(), "Production_Trade", "FoodBalanceSheets_E_All_Data_NOFLAG.csv")
    livestock_patterns_csv: str = os.path.join(get_input_base(), "Manure_Stock", "Environment_LivestockManure_with_ratio.csv")  # ✅ 修复：指向正确的livestock stock文件
    inputs_landuse_csv: str = os.path.join(get_input_base(), "Constraint", "Inputs_LandUse_E_All_Data_NOFLAG.csv")
    fertilizer_efficiency_xlsx: str = os.path.join(get_input_base(), "Fertilizer", "Fertilizer_efficiency.xlsx")
    prices_csv: str = os.path.join(get_input_base(), "Price_Cost", "Price", "World_Production_Value_per_Unit.xlsx")
    trade_crops_xlsx: str = os.path.join(get_input_base(), "Production_Trade", "Trade_CropsLivestock_E_All_Data_NOFLAG_filtered.xlsx")
    trade_forestry_csv: str = os.path.join(get_input_base(), "Production_Trade", "Forestry_E_All_Data_NOFLAG.csv")
    luh2_states_nc: str = r"R:\Data\Food\LUH2\LUH2_GCB2019_states_2010_2020.nc4"
    luh2_transitions_nc: str = r"R:\Data\Food\LUH2\LUH2_GCB2019_transitions_2010_2020.nc4"
    luh2_mask_nc: str = r"R:\Data\Food\LUH2\mask_LUH2_025d.nc"
    luc_param_xlsx: str = os.path.join(get_src_base(), "LUCE_parameter.xlsx")
    # ✅ 新增：简化的土地覆盖基准数据文件（默认优先使用）
    land_cover_base_xlsx: str = os.path.join(get_input_base(), "Land", "Land_cover_base_refill.xlsx")
    # ✅ 新增：Forest EF预计算文件路径
    luce_forest_ef_xlsx: str = os.path.join(get_src_base(), "LUCE_parameter.xlsx")
    # ✅ 开关：是否使用简化的土地覆盖数据（默认True=从Excel读取，False=从LUH2 NetCDF提取）
    use_simplified_land_cover: bool = True
    # ✅ 开关：是否使用预计算的Forest EF（默认True=直接读取，False=从历史数据反算）
    use_precomputed_forest_ef: bool = True
    # optional price/cost sources under Price_Cost
    faostat_prices_csv: str = os.path.join(get_input_base(), "Price_Cost", "Price", "Prices_E_All_Data_NOFLAG.csv")
    macc_pkl: str = os.path.join(get_input_base(), "Price_Cost", "Cost", "MACC-Global-US.pkl")
    unit_cost_xlsx: str = os.path.join(get_input_base(), "Price_Cost", "Cost", "MACC_2080_GapFilled_Final_overZero.xlsx")
    # constraints
    intake_constraint_xlsx: str = os.path.join(get_input_base(), "Constraint", "Intake_constraint.xlsx")
    # optional emissions csvs
    emis_fires_csv: str = os.path.join(get_input_base(), "Emissions_Land_Use_Fires_E_All_Data_NOFLAG.csv")
    # drivers and others
    population_wpp_csv: str = os.path.join(get_input_base(), "Driver", "Population", "WPP", "Population_E_All_Data_NOFLAG.csv")
    temperature_xlsx: str = os.path.join(get_input_base(), "Driver", "Temperature", "SSP_IAM_V2_201811_Temperature.xlsx")
    income_sspdb_xlsx: str = os.path.join(get_input_base(), "Driver", "Income", "SSPDB_future_GDP_with_change_ratio.xlsx")
    sspdb_scenario: str = "SSP2_v9_130325"
    production_trade_fbs_csv: str = os.path.join(get_input_base(), "Production_Trade", "FoodBalanceSheets_E_All_Data_NOFLAG.csv")
    nonfood_balance_csv: str = os.path.join(get_input_base(), "Production_Trade", "CommodityBalances_(non-food)_(2010-)_E_All_Data_NOFLAG.csv")
    forestry_csv: str = os.path.join(get_input_base(), "Production_Trade", "Forestry_E_All_Data_NOFLAG.csv")
    manure_stock_csv: str = os.path.join(get_input_base(), "Manure_Stock", "Environment_LivestockManure_E_All_Data_NOFLAG.csv")
    manure_stock_with_ratio_csv: str = os.path.join(get_input_base(), "Manure_Stock", "Environment_LivestockManure_with_ratio.csv")

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
    # Exclude 'Year' from id_cols to avoid duplicate columns after melt
    id_cols = [c for c in df.columns if c not in year_cols and c != 'Year']
    long_df = df.melt(id_vars=id_cols, value_vars=year_cols,
                      var_name='Year', value_name=value_name)
    
    long_df['Year'] = pd.to_numeric(long_df['Year'].astype(str).str.strip().str.lstrip('Y'), errors='coerce')
    long_df[value_name] = pd.to_numeric(long_df[value_name], errors='coerce')
    long_df = long_df.dropna(subset=['Year'])
    long_df['Year'] = long_df['Year'].astype(int)
    return long_df

def _filter_select_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows where FAOSTAT Select flag==1 when column exists."""
    if df is None or 'Select' not in df.columns:
        return df
    mask = pd.to_numeric(df['Select'], errors='coerce')
    return df[mask == 1]

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
    """Return {(country, commodity) -> elasticity} from sheet with strict columns.
    这里只构建表内的 (country, elasticity_key) -> val；商品映射在上层处理。
    """
    out: Dict[Tuple[str, str], float] = {}
    if df is None or df.empty:
        return out
    required = {'M49_Country_Code', 'Commodity', value_col}
    if not required.issubset(df.columns):
        return out
    # M49 -> 标准国家名
    country_map: Dict[str, str] = {}
    try:
        dict_path = os.path.join(get_src_base(), 'dict_v3.xlsx')
        region_df = _lc(pd.read_excel(dict_path, sheet_name='region'))
        if {'M49 Code', 'Region_label_new'}.issubset(region_df.columns):
            for r in region_df[['M49 Code', 'Region_label_new']].dropna().itertuples(index=False):
                m49 = f"'{str(getattr(r, _tuple_field('M49 Code'))).zfill(3)}"  # ✅ 'xxx格式
                country_std = str(getattr(r, _tuple_field('Region_label_new'))).strip()
                country_map[m49] = country_std
    except Exception:
        pass

    country_col = 'Country_label' if 'Country_label' in df.columns else 'Country'
    for r in df[['M49_Country_Code', country_col, 'Commodity', value_col]].itertuples(index=False):
        m49_raw = getattr(r, _tuple_field('M49_Country_Code'))
        m49 = f"'{str(m49_raw).zfill(3)}" if m49_raw is not None and not pd.isna(m49_raw) else None  # ✅ 'xxx格式
        country = country_map.get(m49, str(getattr(r, _tuple_field(country_col))).strip())
        commodity_raw = str(getattr(r, _tuple_field('Commodity'))).strip()
        val = pd.to_numeric(getattr(r, _tuple_field(value_col)), errors='coerce')
        if pd.isna(val):
            continue
        out[(country, commodity_raw)] = float(val)

    return out


def _build_country_elasticity(df: pd.DataFrame, value_col: str = 'Elasticity_mean') -> Dict[str, float]:
    """Return {country -> elasticity} using M49 mapping to标准国家名."""
    out: Dict[str, float] = {}
    if df is None or df.empty:
        return out
    required = {'M49_Country_Code', value_col}
    if not required.issubset(df.columns):
        return out
    country_map: Dict[str, str] = {}
    try:
        dict_path = os.path.join(get_src_base(), 'dict_v3.xlsx')
        region_df = _lc(pd.read_excel(dict_path, sheet_name='region'))
        if {'M49 Code', 'Region_label_new'}.issubset(region_df.columns):
            for r in region_df[['M49 Code', 'Region_label_new']].dropna().itertuples(index=False):
                m49 = f"'{str(getattr(r, _tuple_field('M49 Code'))).zfill(3)}"  # ✅ 'xxx格式
                country_std = str(getattr(r, _tuple_field('Region_label_new'))).strip()
                country_map[m49] = country_std
    except Exception:
        pass
    country_col = 'Country_label' if 'Country_label' in df.columns else 'Country'
    for r in df[[ 'M49_Country_Code', country_col, value_col]].itertuples(index=False):
        m49_raw = getattr(r, _tuple_field('M49_Country_Code'))
        m49 = f"'{str(m49_raw).zfill(3)}" if m49_raw is not None and not pd.isna(m49_raw) else None  # ✅ 'xxx格式
        country = country_map.get(m49, str(getattr(r, _tuple_field(country_col))).strip())
        val = pd.to_numeric(getattr(r, _tuple_field(value_col)), errors='coerce')
        if pd.isna(val):
            continue
        out[country] = float(val)
    return out

# Mapping helpers between Item_Emis and Item_Elasticity_Map
def _build_emis_to_elast_map() -> Dict[str, str]:
    """Item_Emis -> Item_Elasticity_Map"""
    mapping: Dict[str, str] = {}
    try:
        dict_path = os.path.join(get_src_base(), 'dict_v3.xlsx')
        df = _lc(pd.read_excel(dict_path, sheet_name='Emis_item'))
        if {'Item_Emis', 'Item_Elasticity_Map'}.issubset(df.columns):
            for r in df[['Item_Emis', 'Item_Elasticity_Map']].dropna().itertuples(index=False):
                emis = str(getattr(r, _tuple_field('Item_Emis'))).strip()
                elast = str(getattr(r, _tuple_field('Item_Elasticity_Map'))).strip()
                if emis and elast and elast.lower() not in {'nan', 'no'}:
                    mapping[emis] = elast
    except Exception:
        pass
    return mapping

def _build_elast_to_emis_map() -> Dict[str, str]:
    """Item_Elasticity_Map -> Item_Emis"""
    mapping: Dict[str, str] = {}
    try:
        dict_path = os.path.join(get_src_base(), 'dict_v3.xlsx')
        df = _lc(pd.read_excel(dict_path, sheet_name='Emis_item'))
        if {'Item_Emis', 'Item_Elasticity_Map'}.issubset(df.columns):
            for r in df[['Item_Emis', 'Item_Elasticity_Map']].dropna().itertuples(index=False):
                emis = str(getattr(r, _tuple_field('Item_Emis'))).strip()
                elast = str(getattr(r, _tuple_field('Item_Elasticity_Map'))).strip()
                if emis and elast and elast.lower() not in {'nan', 'no'}:
                    mapping[elast] = emis
    except Exception:
        pass
    return mapping

def _build_elast_to_emis_multi() -> Dict[str, List[str]]:
    """Item_Elasticity_Map -> [Item_Emis...] (可能一对多，如 Milk -> 多个乳制品节点)."""
    mapping: Dict[str, List[str]] = {}
    try:
        dict_path = os.path.join(get_src_base(), 'dict_v3.xlsx')
        df = _lc(pd.read_excel(dict_path, sheet_name='Emis_item'))
        if {'Item_Emis', 'Item_Elasticity_Map'}.issubset(df.columns):
            for r in df[['Item_Emis', 'Item_Elasticity_Map']].dropna().itertuples(index=False):
                emis = str(getattr(r, _tuple_field('Item_Emis'))).strip()
                elast = str(getattr(r, _tuple_field('Item_Elasticity_Map'))).strip()
                if not (emis and elast) or elast.lower() in {'nan', 'no'}:
                    continue
                mapping.setdefault(elast, []).append(emis)
    except Exception:
        pass
    return mapping


def load_process_cost_mapping(dict_v3_path: str) -> Dict[str, str]:
    """
    从dict_v3.xlsx的Emis_item表读取Process到Process_Cost_Map的映射
    
    返回: {Process: Process_Cost_Map}
        - 如果Process_Cost_Map为None或'no'，则不包含在映射中
        - 如果Process_Cost_Map为'Production value'，则映射为该特殊值
    """
    mapping: Dict[str, str] = {}
    try:
        df = _lc(pd.read_excel(dict_v3_path, sheet_name='Emis_item'))
        if 'Process' not in df.columns or 'Process_Cost_Map' not in df.columns:
            print(f"Warning: dict_v3 Emis_item表缺少Process或Process_Cost_Map列")
            return mapping
        
        for r in df[['Process', 'Process_Cost_Map']].dropna().itertuples(index=False):
            process = str(getattr(r, _tuple_field('Process'))).strip()
            cost_map = str(getattr(r, _tuple_field('Process_Cost_Map'))).strip()
            
            # 跳过None或'no'标记的过程
            if not process or not cost_map or cost_map.lower() in {'none', 'no', 'nan'}:
                continue
            
            mapping[process] = cost_map
    except Exception as e:
        print(f"Warning: 无法读取Process_Cost_Map映射: {e}")
    
    return mapping


def load_unit_cost_data(cost_xlsx_path: str, dict_v3_path: str) -> Tuple[Dict[Tuple[str, str], float], Dict[str, str]]:
    """
    读取单位减排成本数据（MACC_2080_GapFilled_Final_overZero.xlsx）
    
    Args:
        cost_xlsx_path: 成本数据Excel路径
        dict_v3_path: dict_v3.xlsx路径（用于M49映射）
    
    Returns:
        (unit_costs, process_mapping)
        - unit_costs: {(country_name, process_cost_name): USD_per_tCO2e}
        - process_mapping: {process: process_cost_map} 从dict_v3读取的映射
    """
    unit_costs: Dict[Tuple[str, str], float] = {}
    process_mapping = load_process_cost_mapping(dict_v3_path)
    
    if not os.path.exists(cost_xlsx_path):
        print(f"Warning: 成本数据文件不存在: {cost_xlsx_path}")
        return unit_costs, process_mapping
    
    try:
        # 读取Gap_Filled_Data表
        df = pd.read_excel(cost_xlsx_path, sheet_name='Gap_Filled_Data')
        
        # 检查必要列
        required_cols = {'M49_Country_Code', 'Process', 'Final_Unit_Cost'}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            print(f"Error: 成本数据缺少必要列: {missing}")
            return unit_costs, process_mapping
        
        # 从dict_v3获取M49到国家名的映射
        m49_to_country: Dict[str, str] = {}
        try:
            region_df = _lc(pd.read_excel(dict_v3_path, sheet_name='region'))
            if {'M49_Country_Code', 'Region_label_new'}.issubset(region_df.columns):
                for r in region_df[['M49_Country_Code', 'Region_label_new']].dropna().itertuples(index=False):
                    m49_raw = getattr(r, _tuple_field('M49_Country_Code'))
                    country = str(getattr(r, _tuple_field('Region_label_new'))).strip()
                    # 标准化M49格式
                    m49 = _norm_m49(m49_raw)
                    m49_to_country[m49] = country
        except Exception as e:
            print(f"Warning: 无法读取dict_v3 region表: {e}")
        
        # 解析成本数据
        for _, row in df.iterrows():
            try:
                m49_raw = row['M49_Country_Code']
                process_cost = str(row['Process']).strip()
                unit_cost = float(row['Final_Unit_Cost'])
                
                # 跳过NaN或负值
                if pd.isna(unit_cost) or unit_cost < 0:
                    continue
                
                # 映射M49到国家名
                m49 = _norm_m49(m49_raw)
                country = m49_to_country.get(m49)
                if not country:
                    continue
                
                # 存储成本数据
                unit_costs[(country, process_cost)] = unit_cost
                
            except Exception as e:
                continue
        
        print(f"成功读取 {len(unit_costs)} 条单位减排成本数据")
        print(f"成本数据覆盖的Process: {sorted(set(p for _, p in unit_costs.keys()))}")
        
    except Exception as e:
        print(f"Error: 读取成本数据失败: {e}")
        import traceback
        traceback.print_exc()
    
    return unit_costs, process_mapping


def _build_cross_elasticity_map(df: pd.DataFrame, commodity_filter: Optional[set] = None) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Return {(country_code, commodity) -> {other_commodity: elasticity}} for cross-price sheets.
    
    Maps elasticity column names (Item_Elasticity_Map format, e.g., 'Beans_dry') 
    to production names (Item_Production_Map format, e.g., 'Beans, dry') using dict_v3.xlsx mapping.
    """
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    if df is None or df.empty:
        return out
    # 严格要求包含 M49_Country_Code 和 Commodity；国家名称优先 Country_label
    if 'M49_Country_Code' not in df.columns or 'Commodity' not in df.columns:
        return out
    country_col = 'Country_label' if 'Country_label' in df.columns else 'Country'
    if country_col not in df.columns:
        return out
    # M49 -> 标准国家名
    country_map: Dict[str, str] = {}
    try:
        dict_path = os.path.join(get_src_base(), 'dict_v3.xlsx')
        region_df = _lc(pd.read_excel(dict_path, sheet_name='region'))
        if {'M49 Code', 'Region_label_new'}.issubset(region_df.columns):
            for r in region_df[['M49 Code', 'Region_label_new']].dropna().itertuples(index=False):
                m49 = f"'{str(getattr(r, _tuple_field('M49 Code'))).zfill(3)}"  # ✅ 'xxx格式
                country_std = str(getattr(r, _tuple_field('Region_label_new'))).strip()
                country_map[m49] = country_std
    except Exception:
        pass
    
    # Build elasticity_name -> Item_Emis mapping from dict_v3
    elast_to_prod: Dict[str, str] = {}
    try:
        dict_path = os.path.join(get_src_base(), 'dict_v3.xlsx')
        mapping_df = _lc(pd.read_excel(dict_path, sheet_name='Emis_item'))
        if {'Item_Elasticity_Map', 'Item_Emis'}.issubset(mapping_df.columns):
            for r in mapping_df[['Item_Elasticity_Map', 'Item_Emis']].dropna().itertuples(index=False):
                elast_name = str(r.Item_Elasticity_Map).strip()
                emis_name = str(r.Item_Emis).strip()
                if elast_name and emis_name and elast_name.lower() not in {'nan', 'no'} and emis_name.lower() not in {'nan', 'no'}:
                    elast_to_prod[elast_name] = emis_name
    except Exception as e:
        print(f"Warning: Could not load commodity name mapping from dict_v3: {e}")
    
    attr_by_col = {col: _tuple_field(col) for col in df.columns}
    base_cols = {country_col, 'Country', 'Country_label', 'Commodity', 'M49_Country_Code'}
    for row in df.itertuples(index=False):
        m49_raw = getattr(row, attr_by_col.get('M49_Country_Code', ''))
        m49 = f"'{str(m49_raw).zfill(3)}" if m49_raw is not None and m49_raw != '' and not pd.isna(m49_raw) else None  # ✅ 'xxx格式
        country = country_map.get(m49, str(getattr(row, attr_by_col[country_col])).strip())
        commodity = str(getattr(row, attr_by_col['Commodity'])).strip()
        key = (country, commodity)
        cross: Dict[str, float] = {}
        for col, attr in attr_by_col.items():
            if col in base_cols:
                continue
            # Map elasticity column name to production name
            mapped_col = elast_to_prod.get(col, col)
            if commodity_filter is not None and mapped_col not in commodity_filter:
                continue
            val = pd.to_numeric(getattr(row, attr), errors='coerce')
            if pd.isna(val):
                continue
            cross[mapped_col] = float(val)
        if cross:
            out[key] = cross
    return out

def _country_by_m49(df: pd.DataFrame, universe: Universe) -> Optional[pd.Series]:
    """Return Series of mapped country names using explicit M49 codes (no fuzzy fallback)."""
    if df is None:
        return None
    if 'M49_Country_Code' not in df.columns:
        raise ValueError("Expected 'M49_Country_Code' column for country mapping.")
    codes = df['M49_Country_Code'].apply(_parse_m49_code)
    formatted = codes.apply(lambda x: f"'{int(x):03d}" if pd.notna(x) else pd.NA)  # ✅ 'xxx格式
    if universe.country_by_m49:
        code_to_country = dict(universe.country_by_m49)
    else:
        code_to_country = {f"'{int(code):03d}": str(country) for country, code in (universe.m49_by_country or {}).items() if code is not None}  # ✅ 'xxx格式
    mapped = formatted.map(code_to_country)
    return mapped

def _attach_country_from_m49(df_source: pd.DataFrame,
                             df_target: pd.DataFrame,
                             universe: Universe,
                             *,
                             context: str) -> pd.DataFrame:
    """Assign country column strictly via M49 codes."""
    m49_country = _country_by_m49(df_source, universe)
    if m49_country is None:
        raise ValueError(f"{context}: missing M49_Country_Code column; cannot map countries.")
    df_target = df_target.copy()
    df_target['country'] = m49_country.reindex(df_target.index)
    df_target = df_target.dropna(subset=['country'])
    return df_target


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


def load_land_cover_from_excel(excel_path: str, 
                               universe: Universe,
                               years: Optional[List[int]] = None) -> pd.DataFrame:
    """
    从简化的Excel文件读取土地覆盖数据（替代从LUH2 NetCDF提取）
    
    Excel文件格式（宽格式）：
    - M49_Country_Code: 国家M49代码
    - Region_label_new: 区域名称
    - Land cover: 土地类型 (cropland, forest, grassland)
    - Y2010, Y2011, ..., Y2020: 各年份面积
    - Unit: 单位 (1000 ha)
    
    返回格式与 load_luh2_land_cover 相同：
    - country, iso3, year, land_use, area_ha
    """
    columns = ['country', 'iso3', 'year', 'land_use', 'area_ha']
    
    if not (excel_path and os.path.exists(excel_path)):
        print(f"[WARN] load_land_cover_from_excel: 文件不存在 {excel_path}")
        return pd.DataFrame(columns=columns)
    
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        print(f"[ERROR] load_land_cover_from_excel: 读取Excel失败 {e}")
        return pd.DataFrame(columns=columns)
    
    # 列名规范化
    df.columns = [str(c).strip() for c in df.columns]
    
    # 查找所需列
    m49_col = None
    land_cover_col = None
    for c in df.columns:
        cl = c.lower()
        if 'm49' in cl:
            m49_col = c
        elif 'land cover' in cl or 'land_cover' in cl:
            land_cover_col = c
    
    if not m49_col or not land_cover_col:
        print(f"[ERROR] load_land_cover_from_excel: 缺少必要列 M49={m49_col}, Land_cover={land_cover_col}")
        return pd.DataFrame(columns=columns)
    
    # 获取年份列
    year_cols = [c for c in df.columns if c.startswith('Y') and c[1:].isdigit()]
    if not year_cols:
        print(f"[ERROR] load_land_cover_from_excel: 未找到年份列 (Y2010, Y2011, ...)")
        return pd.DataFrame(columns=columns)
    
    records = []
    
    for _, row in df.iterrows():
        m49_raw = row.get(m49_col)
        land_type_raw = row.get(land_cover_col)
        
        if pd.isna(m49_raw) or pd.isna(land_type_raw):
            continue
        
        # 规范化M49代码
        m49_str = _norm_m49(str(m49_raw))
        
        # 查找国家
        country = universe.country_by_m49.get(m49_str)
        if not country:
            continue
        
        iso3 = universe.iso3_by_country.get(country, '')
        
        # 映射land_cover名称到land_use
        land_type = str(land_type_raw).strip().lower()
        if 'forest' in land_type:
            land_use = 'forest'
        elif 'grass' in land_type or 'pasture' in land_type:
            land_use = 'grassland'
        elif 'crop' in land_type:
            land_use = 'cropland'
        else:
            continue  # 跳过其他类型
        
        for yc in year_cols:
            try:
                year = int(yc[1:])
            except ValueError:
                continue
            
            if years and year not in years:
                continue
            
            val = row.get(yc)
            if pd.isna(val):
                continue
            
            # ✅ 修复：Land_cover_base_refill.xlsx的Unit列是'ha'，不需要乘1000！
            # 原代码错误地假设单位是1000 ha
            area_ha = float(val)
            
            records.append({
                'country': country,
                'iso3': iso3,
                'year': year,
                'land_use': land_use,
                'area_ha': area_ha,
            })
    
    if not records:
        print(f"[WARN] load_land_cover_from_excel: 未生成任何记录")
        return pd.DataFrame(columns=columns)
    
    out_df = pd.DataFrame.from_records(records, columns=columns)
    out_df = out_df.groupby(['country', 'iso3', 'year', 'land_use'], as_index=False)['area_ha'].sum()
    
    print(f"[INFO] load_land_cover_from_excel: 加载 {len(out_df)} 条土地覆盖记录 from {os.path.basename(excel_path)}")
    return out_df


def load_forest_ef_from_excel(excel_path: str,
                              universe: Universe,
                              sheet_name: str = 'Forest_EF') -> Dict[str, Dict[int, float]]:
    """
    从LUCE_parameter.xlsx读取预计算的Forest排放因子
    
    Excel文件格式：
    - M49_Country_Code: 国家M49代码
    - Process: 固定为'Forest'
    - Unit: tCO2/ha/yr
    - Y2010, Y2011, ..., Y2020: 各年份EF值
    
    返回:
    - Dict[M49, Dict[year, EF值]]
    - EF单位：tCO2/ha/yr（负值表示碳汇）
    """
    result: Dict[str, Dict[int, float]] = {}
    
    if not (excel_path and os.path.exists(excel_path)):
        print(f"[WARN] load_forest_ef_from_excel: 文件不存在 {excel_path}")
        return result
    
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
    except Exception as e:
        print(f"[ERROR] load_forest_ef_from_excel: 读取Excel失败 {e}")
        return result
    
    # 列名规范化
    df.columns = [str(c).strip() for c in df.columns]
    
    # 查找M49列
    m49_col = None
    for c in df.columns:
        if 'm49' in c.lower():
            m49_col = c
            break
    
    if not m49_col:
        print(f"[ERROR] load_forest_ef_from_excel: 未找到M49列")
        return result
    
    # 获取年份列
    year_cols = [c for c in df.columns if c.startswith('Y') and c[1:].isdigit()]
    if not year_cols:
        print(f"[ERROR] load_forest_ef_from_excel: 未找到年份列")
        return result
    
    for _, row in df.iterrows():
        m49_raw = row.get(m49_col)
        if pd.isna(m49_raw):
            continue
        
        m49_str = _norm_m49(str(m49_raw))
        
        # 跳过World汇总行
        if m49_str == '000' or m49_str == '0':
            continue
        
        ef_by_year: Dict[int, float] = {}
        
        for yc in year_cols:
            try:
                year = int(yc[1:])
            except ValueError:
                continue
            
            val = row.get(yc)
            if pd.notna(val):
                ef_by_year[year] = float(val)
        
        if ef_by_year:
            result[m49_str] = ef_by_year
    
    print(f"[INFO] load_forest_ef_from_excel: 加载 {len(result)} 个国家的Forest EF from {os.path.basename(excel_path)}")
    return result


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
    keep_cols = [c_area, c_year, c_item, c_elem, c_val] + ([c_unit] if c_unit else [])
    if 'M49_Country_Code' in df.columns:
        keep_cols.append('M49_Country_Code')
    z = df[keep_cols].copy()
    rename_cols = {c_area: 'area', c_year: 'year', c_item: 'item_raw', c_elem: 'element', c_val: 'value'}
    if c_unit:
        rename_cols[c_unit] = 'unit'
    z = z.rename(columns=rename_cols)
    z = _attach_country_from_m49(df, z, universe, context="Roundwood supply")
    if 'M49_Country_Code' in df.columns and 'M49_Country_Code' not in z.columns:
        z['M49_Country_Code'] = df['M49_Country_Code']
    try:
        maps = load_emis_item_mappings(os.path.join(get_src_base(), 'dict_v3.xlsx'))
        z['commodity'] = z['item_raw'].map(maps.production_by_item).fillna(z['item_raw'])
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
    # Emis_item表包含Process列，用于提取过程列表
    emis_item = _lc(pd.read_excel(xls, 'Emis_item'))
    emis_proc = emis_item  # 使用同一张表

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
        m49_by_country = dict(zip(
            region_m49['country_label'],
            region_m49[m49_col].apply(lambda x: f"'{int(x):03d}" if pd.notna(x) else None)  # ✅ 'xxx格式
        ))

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

    # Commodities - Use Item_Emis for standardized model commodity names
    # Item_Production_Map is FAOSTAT CSV item name; Item_Emis is the model standard name
    c_item_emis = _find_col(emis_item, ['Item_Emis'])
    commodities = sorted(pd.unique(emis_item[c_item_emis].astype(str).str.strip()))

    # Cat2 mapping - use Item_Emis for consistency with commodities
    c_cat2 = _find_col(emis_item, ['Item_Cat2'])
    item_cat2_by_commodity = {}
    c_item_emis_attr = _tuple_field(c_item_emis)
    c_cat2_attr = _tuple_field(c_cat2)
    for r in emis_item.itertuples(index=False):
        item = str(getattr(r, c_item_emis_attr)).strip()
        cat2_val = getattr(r, c_cat2_attr)
        item_cat2_by_commodity[item] = '' if pd.isna(cat2_val) else str(cat2_val)

    # years
    years_hist = list(range(config.years_hist_start, config.years_hist_end + 1))
    years_future = config.years_future if config.years_future else []
    years = sorted(set(years_hist + years_future))

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
    emis_to_elast = _build_emis_to_elast_map()
    def read(sheet: str) -> pd.DataFrame:
        return _lc(pd.read_excel(xls, sheet)) if sheet in xls.sheet_names else pd.DataFrame()

    temp_map = _build_elasticity_map(read('Supply-Temperature'))
    yield_map = _build_elasticity_map(read('Supply-Yield'))
    own_price_map = _build_elasticity_map(read('Supply-Own-Price'))
    commodity_filter = {str(n.commodity) for n in nodes}
    supply_cross_map = _build_cross_elasticity_map(read('Supply_Cross_mean'), commodity_filter=commodity_filter)

    for n in nodes:
        elast_key = emis_to_elast.get(str(n.commodity), str(n.commodity))
        key = (str(n.country), elast_key)
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
            setattr(n, 'epsS_row', dict(cross))

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
                                     latest_hist_prod: Optional[Dict[Tuple[str, str], Tuple[int, float]]] = None,
                                     feed_override_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if not os.path.exists(fbs_csv):
        return pd.DataFrame(columns=['country','iso3','year','commodity','food_t','feed_t','seed_t','demand_total_t'])
    df_raw = pd.read_csv(fbs_csv)
    df_raw = _filter_select_rows(df_raw)
    df = _lc(_faostat_wide_to_long(df_raw))
    c_area = _find_col(df, ['Area'])
    c_year = _find_col(df, ['Year'])
    c_item = _find_col(df, ['Item'])
    c_elem = _find_col(df, ['Element'])
    c_val  = _find_col(df, ['Value'])
    keep_cols = [c_area, c_year, c_item, c_elem, c_val]
    z = df[keep_cols].copy()
    rename_map = {c_area: 'area', c_year: 'year', c_item: 'item_raw', c_elem: 'element', c_val: 'value'}
    z = z.rename(columns=rename_map)
    if 'M49_Country_Code' in df.columns:
        z['M49_Country_Code'] = df['M49_Country_Code']
    z = _attach_country_from_m49(df, z, universe, context="FBS demand")
    if 'M49_Country_Code' in df.columns and 'M49_Country_Code' not in z.columns:
        z['M49_Country_Code'] = df['M49_Country_Code']
    try:
        maps = load_emis_item_mappings(os.path.join(get_src_base(), 'dict_v3.xlsx'))
        z['commodity'] = z['item_raw'].map(maps.production_by_item).fillna(z['item_raw'])
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

    if feed_override_df is not None and not feed_override_df.empty:
        cols_needed = {'country', 'year', 'commodity', 'feed_t'}
        if cols_needed.issubset(feed_override_df.columns):
            override = feed_override_df[['country', 'year', 'commodity', 'feed_t']].copy()
            override = override.groupby(['country','year','commodity'], as_index=False)['feed_t'].sum()
            result = result.merge(override,
                                  how='outer',
                                  on=['country','year','commodity'],
                                  suffixes=('', '_override'))
            for col in ['food_t', 'seed_t']:
                if col in result.columns:
                    result[col] = result[col].fillna(0.0)
            result['feed_t'] = result['feed_t_override'].fillna(result['feed_t']).fillna(0.0)
            result = result.drop(columns=['feed_t_override'])
        else:
            pass  # silently ignore malformed overrides to avoid crashing

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
    if 'M49_Country_Code' not in df.columns:
        raise ValueError("Expected 'M49_Country_Code' column for all FAOSTAT tables.")
    codes = df['M49_Country_Code'].apply(_parse_m49_code)
    country = codes.map(m49_to_country)
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
    base_cols = ['M49_Country_Code', 'Area', 'Item', 'Element', 'Unit']
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
    maps = load_emis_item_mappings(os.path.join(get_src_base(), 'dict_v3.xlsx'))
    data = build_faostat_production_indicators(csv_path, universe, maps)
    return data['production']


def load_production_indicators(paths: DataPaths, universe: Universe) -> Dict[str, pd.DataFrame]:
    maps = load_emis_item_mappings(os.path.join(get_src_base(), 'dict_v3.xlsx'))
    return build_faostat_production_indicators(paths.production_faostat_csv, universe, maps)


def load_production_statistics(paths: DataPaths, universe: Universe) -> Dict[str, pd.DataFrame]:
    """加载生产统计数据，包括作物和畜牧业参数"""
    
    # 检查关键输入文件
    dict_v3_path = os.path.join(get_src_base(), 'dict_v3.xlsx')
    _check_file_exists(dict_v3_path, "字典文件 (dict_v3.xlsx)", critical=True)
    _check_file_exists(paths.production_faostat_csv, "生产数据文件 (Production_Crops_Livestock)", critical=True)
    _check_file_exists(paths.manure_stock_with_ratio_csv, "畜牧存栏数据文件 (Environment_LivestockManure_with_ratio)", critical=True)
    _check_file_exists(paths.feed_coeff_xlsx, "饲料需求系数文件 (Feed_need_per_head)", critical=True)
    
    maps = load_emis_item_mappings(dict_v3_path)
    stats = build_faostat_production_indicators(paths.production_faostat_csv, universe, maps)
    # WARNING: production must be extended to future years; otherwise 2080 livestock emissions cannot be computed
    stats['production'] = _extend_future_years(stats['production'], 'production_t', universe)
    stats['yield'] = _extend_future_years(stats['yield'], 'yield_t_per_ha', universe)
    stats['slaughter'] = _extend_future_years(stats['slaughter'], 'slaughter_head', universe)
    stats['livestock_yield'] = _extend_future_years(stats['livestock_yield'], 'yield_t_per_head', universe)
    
    # Load livestock stock from Environment_LivestockManure_with_ratio.csv
    print(f"[INFO] 从 Environment_LivestockManure_with_ratio.csv 加载livestock stock数据...")
    stock_from_env = build_livestock_stock_from_env(paths.manure_stock_with_ratio_csv, universe)
    if not stock_from_env.empty:
        print(f"[INFO] OK 从Environment文件加载了 {len(stock_from_env)} 行stock数据")
        # Use stock from Environment file (more complete for livestock)
        stats['stock'] = stock_from_env
    else:
        print(f"[WARNING] WARN Environment文件未找到stock数据，使用Production CSV的stock")
        stats['stock'] = _extend_future_years(stats['stock'], 'stock_head', universe)
    
    # Extend stock to future years
    stats['stock'] = _extend_future_years(stats['stock'], 'stock_head', universe)
    
    fert = load_fertilizer_statistics(paths.fertilizer_efficiency_xlsx, universe, maps)
    feed = load_feed_requirement_per_head(paths.feed_coeff_xlsx, universe, maps)
    # WARNING: use the correct manure file path
    manure = load_manure_management_ratio(paths.manure_stock_with_ratio_csv, universe, maps)
    stats.update({
        'fertilizer_efficiency': fert['efficiency'],
        'fertilizer_amount': fert['amount'],
        'feed_requirement': feed,
        'manure_management_ratio': manure,
    })
    return stats

def build_gce_activity_tables(production_csv: str,
                              fbs_csv: str,
                              fertilizer_eff_xlsx: str,
                              universe: Universe) -> Dict[str, pd.DataFrame]:
    # For now, return minimal frames with keys expected by orchestrator
    prod = build_production_from_faostat(production_csv, universe)
    residues_df = prod.rename(columns={'production_t':'residues_feedstock_t'})[['country','iso3','year','commodity','residues_feedstock_t']]
    burning_df = prod.rename(columns={'production_t':'burning_feedstock_t'})[['country','iso3','year','commodity','burning_feedstock_t']]
    rice_df = prod.rename(columns={'production_t':'rice_area_proxy'})[['country','iso3','year','commodity','rice_area_proxy']]
    fertilizers_df = pd.DataFrame(columns=['country','iso3','year','n_fert_t'])
    if os.path.exists(fertilizer_eff_xlsx):
        try:
            maps = load_emis_item_mappings(os.path.join(get_src_base(), "dict_v3.xlsx"))
            fert_stats = load_fertilizer_statistics(fertilizer_eff_xlsx, universe, maps)
            amt_df = fert_stats.get('amount', pd.DataFrame())
        except Exception:
            amt_df = pd.DataFrame()
        if isinstance(amt_df, pd.DataFrame) and not amt_df.empty:
            z = amt_df[['country','iso3','year','fertilizer_n_input_t']].copy()
            z['fertilizer_n_input_t'] = pd.to_numeric(z['fertilizer_n_input_t'], errors='coerce')
            z = z.dropna(subset=['fertilizer_n_input_t'])
            if not z.empty:
                z = z.groupby(['country','iso3','year'], as_index=False)['fertilizer_n_input_t'].sum()
                z = z.rename(columns={'fertilizer_n_input_t': 'n_fert_t'})
                z = z[z['year'].isin(universe.years)]
                fertilizers_df = z.sort_values(['country','year']).reset_index(drop=True)
    return {'residues_df': residues_df, 'burning_df': burning_df, 'rice_df': rice_df, 'fertilizers_df': fertilizers_df}

def build_livestock_stock_from_env(csv_path: str, universe: Universe) -> pd.DataFrame:
    """
    Load livestock stock from Environment_LivestockManure_with_ratio.csv
    WARNING: keep only commodities defined in dict_v3 (drop 'All Animals' and similar)
    Returns DataFrame with columns: country, iso3, year, commodity, stock_head
    """
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=['country','iso3','year','commodity','stock_head'])
    df_raw = pd.read_csv(csv_path)
    df_raw = _filter_select_rows(df_raw)
    df = _lc(_faostat_wide_to_long(df_raw))
    
    # Find column names
    c_area = _find_col(df, ['Area'])
    c_year = _find_col(df, ['Year']) 
    c_item = _find_col(df, ['Item'])
    c_elem = _find_col(df, ['Element'])
    c_val = _find_col(df, ['Value'])
    
    if not all([c_area, c_year, c_item, c_elem, c_val]):
        return pd.DataFrame(columns=['country','iso3','year','commodity','stock_head'])
    
    # Filter for "Stocks" element only
    df = df[df[c_elem].astype(str).str.strip() == 'Stocks'].copy()
    if df.empty:
        return pd.DataFrame(columns=['country','iso3','year','commodity','stock_head'])
    
    keep_cols = [c_area, c_year, c_item, c_val]
    if 'M49_Country_Code' in df.columns:
        keep_cols.append('M49_Country_Code')
    
    z = df[keep_cols].copy()
    z = z.rename(columns={c_area: 'area', c_year: 'year', c_item: 'item_raw', c_val: 'stock_head'})
    z['stock_head'] = pd.to_numeric(z['stock_head'], errors='coerce')
    z = z.dropna(subset=['stock_head'])
    
    # Attach country info
    z = _attach_country_from_m49(df, z, universe, context="Livestock stock")
    # ✅ 保留M49_Country_Code列，确保索引对齐
    if 'M49_Country_Code' in df.columns and 'M49_Country_Code' not in z.columns:
        z['M49_Country_Code'] = df.loc[z.index, 'M49_Country_Code'].values
    
    # Map item to commodity using dict_v3
    try:
        maps = load_emis_item_mappings(os.path.join(get_src_base(), 'dict_v3.xlsx'))
        z['commodity'] = z['item_raw'].map(maps.stock_item_to_comm)
        
        # WARNING: keep only commodities that map to dict_v3; drop 'All Animals' and similar
        before_filter = len(z)
        z = z.dropna(subset=['commodity'])
        after_filter = len(z)
        if before_filter > after_filter:
            print(f"[INFO] 过滤掉 {before_filter - after_filter} 行非dict_v3定义的Item (如'All Animals')")
            
    except Exception as e:
        print(f"[WARNING] 无法加载dict_v3映射: {e}")
        z['commodity'] = z['item_raw']
    
    # Filter to universe (只保留198个有效国家)
    z = z[z['country'].isin(universe.countries)]
    z['iso3'] = z['country'].map(universe.iso3_by_country)
    z = z.dropna(subset=['iso3'])
    z['year'] = pd.to_numeric(z['year'], errors='coerce').astype(int)
    z = z[z['year'].isin(universe.years)]
    
    # WARNING: note: stock_df uses Item_Stock_Map names (e.g., "Cattle, dairy")
    # universe.commodities使用Item_Production_Map名称（如"Raw milk of cattle"）
    # 所以不能用universe.commodities过滤！commodity已经在上面通过maps.stock_item_to_comm映射验证过了
    
    # ✅ 确保M49_Country_Code列存在且格式正确 ('xxx 格式)
    if 'M49_Country_Code' not in z.columns:
        # 尝试从universe.m49_by_country反向查找M49
        z['M49_Country_Code'] = z['country'].map(universe.m49_by_country)
    
    # ✅ 规范化M49格式为 'xxx（单引号+3位数字）
    def _format_m49_quote(val):
        """将M49格式化为 'xxx 格式"""
        if pd.isna(val):
            return val
        s = str(val).strip().strip("'\"")
        try:
            return f"'{int(s):03d}"
        except:
            return f"'{s}"
    z['M49_Country_Code'] = z['M49_Country_Code'].apply(_format_m49_quote)
    
    print(f"[INFO] OK 最终保留 {len(z)} 行stock数据 ({z['commodity'].nunique()} 个物种, {z['country'].nunique()} 个国家)")
    
    # ✅ 返回时包含M49_Country_Code列
    return z[['M49_Country_Code','country','iso3','year','commodity','stock_head']].reset_index(drop=True)

def build_gv_areas_from_inputs(csv_path: str, universe: Universe) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=['country','iso3','year','land_use','area_ha'])
    df_raw = pd.read_csv(csv_path)
    df_raw = _filter_select_rows(df_raw)
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
    cols = [c_area, c_year, 'land_use', c_val]
    if 'M49_Country_Code' in df.columns:
        cols.append('M49_Country_Code')
    df = df[cols].copy()
    df = df.rename(columns={c_area: 'area', c_year: 'year', c_val: 'area_ha'})
    df = _attach_country_from_m49(df, df, universe, context="GV areas")
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
    df_raw = _filter_select_rows(df_raw)
    df = _lc(_faostat_wide_to_long(df_raw))
    c_area=_find_col(df,['Area']); c_year=_find_col(df,['Year']); c_gas=_find_col(df,['Element','Gas','GHG']); c_val=_find_col(df,['Value'])
    z = df[[c_area,c_year,c_val]].copy(); z.columns=['area','year','co2e_kt']
    z = _attach_country_from_m49(df, z, universe, context="Land-use fires")
    if 'M49_Country_Code' in df.columns and 'M49_Country_Code' not in z.columns:
        z['M49_Country_Code'] = df['M49_Country_Code']
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
    Returns {(m49_code, year): land_area_1000ha} where m49_code is in 'xxx format.
    """
    out: Dict[Tuple[str,int], float] = {}
    if not os.path.exists(csv_path):
        return out
    df_raw = pd.read_csv(csv_path)
    # 不使用 wide_to_long，直接处理宽格式数据
    # 需要的列: M49_Country_Code, Item, Element, Y1961...Y2022
    df = df_raw.copy()
    df.columns = df.columns.str.strip()
    
    # 找到 M49 列
    m49_col = None
    for c in df.columns:
        if 'm49' in c.lower() and 'code' in c.lower():
            m49_col = c
            break
    if m49_col is None:
        # fallback: 用 Area Code 列
        for c in df.columns:
            if c.lower() == 'area code':
                m49_col = c
                break
    if m49_col is None:
        return out
    
    # 筛选 Item == 'Land area' 且 Element == 'Area'
    item_col = _find_col(df, ['Item'])
    elem_col = _find_col(df, ['Element'])
    
    mask = (df[item_col].astype(str).str.strip() == 'Land area') & \
           (df[elem_col].astype(str).str.strip() == 'Area')
    df_land = df[mask].copy()
    
    # 找到所有年份列 (Y1961, Y2020 等)
    year_cols = [c for c in df_land.columns if c.startswith('Y') and c[1:].isdigit()]
    
    for _, row in df_land.iterrows():
        # 标准化 M49 代码为 'xxx 格式
        raw_m49 = str(row[m49_col]).strip()
        # 去掉已有的引号，然后统一加上
        m49_clean = raw_m49.lstrip("'\"")
        try:
            m49_int = int(m49_clean)
            m49_code = f"'{m49_int:03d}"  # 'xxx 格式，如 '156, '004
        except ValueError:
            continue
        
        for yc in year_cols:
            try:
                year = int(yc[1:])  # Y2020 -> 2020
                val = float(row[yc])
                if pd.notna(val):
                    out[(m49_code, year)] = val  # 单位: 1000 ha
            except (ValueError, TypeError):
                continue
    
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
    # 以 Item_Production_Map 为主进行营养匹配（保持与生产名一致），同时额外把 Item_Emis 也映射到同一系数，避免漏匹配
    c_comm_prod = _find_col(df, ['Item_Production_Map'])
    c_comm_emis = _maybe_find_col(df, ['Item_Emis'])
    c_kcal = 'kcal_per_100g' if 'kcal_per_100g' in df.columns else None
    c_prot = 'g_protein_per_100g' if 'g_protein_per_100g' in df.columns else None
    c_fat = 'g_fat_per_100g' if 'g_fat_per_100g' in df.columns else None
    out: Dict[str, float] = {}
    for r in df.itertuples(index=False):
        keys: List[str] = []
        if c_comm_prod:
            v_prod = getattr(r, c_comm_prod)
            if pd.notna(v_prod):
                keys.append(str(v_prod).strip())
        if c_comm_emis:
            v_emis = getattr(r, c_comm_emis)
            if pd.notna(v_emis):
                keys.append(str(v_emis).strip())
        keys = [k for k in keys if k and k.lower() not in {'nan', 'no'}]
        if not keys:
            continue
        if indicator == 'energy' and c_kcal:
            v = getattr(r, c_kcal)
            try:
                val = float(v) * 10000.0  # 1 t = 10,000×100g
                if np.isfinite(val):
                    for k in keys:
                        out[k] = val
            except Exception:
                pass
        elif indicator == 'protein' and c_prot:
            v = getattr(r, c_prot)
            try:
                val = float(v) * 10000.0  # grams per ton
                if np.isfinite(val):
                    for k in keys:
                        out[k] = val
            except Exception:
                pass
        elif indicator == 'fat' and c_fat:
            v = getattr(r, c_fat)
            try:
                val = float(v) * 10000.0  # grams per ton
                if np.isfinite(val):
                    for k in keys:
                        out[k] = val
            except Exception:
                pass
    return out

def load_intake_targets(xlsx_path: str, indicator: str) -> Dict[str, float]:
    """Load per-capita-per-day intake targets by country from Intake_constraint.xlsx.
    Uses 'Indicator' column to filter rows for one of ['Energy supply','Protein supply','Fat supply'] and reads 'MEAN'. #MIN
    Returns {country -> mean_value_per_capita_per_day}.
    """
    if not os.path.exists(xlsx_path):
        return {}
    df = _lc(pd.read_excel(xlsx_path, sheet_name='extract'))
    c_cty = _maybe_find_col(df, ['Country','Area'])
    c_ind = _maybe_find_col(df, ['Indicator'])
   # c_val = _maybe_find_col(df, ['MEAN','Mean'])
    c_val = _maybe_find_col(df, ['MIN','Min'])
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
    """Backward compatible wrapper that now derives yield directly from Production NOFLAG file."""
    maps = load_emis_item_mappings(os.path.join(get_src_base(), 'dict_v3.xlsx'))
    data = build_faostat_production_indicators(production_csv, universe, maps)
    return data['yield']

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


def assign_grassland_coef_to_nodes(nodes: List[Node], 
                                    paths: DataPaths, 
                                    universe: Universe,
                                    maps: EmisItemMappings,
                                    stock_df: Optional[pd.DataFrame] = None) -> None:
    """
    计算并分配grassland系数到livestock节点的meta字段
    
    流程：
    1. 使用S3_0_ds_linear_regional.load_grassland_coefficients计算区域级系数（ha/head）
    2. 如果提供stock_df，使用gle_emissions_complete转换head→ton，得到ha/ton系数
    3. 将系数分配到节点的meta['grassland_coef']
    
    Args:
        nodes: 节点列表
        paths: 数据路径配置
        universe: 宇宙配置（包含国家、年份等）
        maps: 排放项映射（用于commodity→species转换）
        stock_df: 可选的存栏数据（用于head→ton转换）
    """
    try:
        # 导入grassland系数加载函数
        from S3_0_ds_linear_regional import load_grassland_coefficients
        
        # 计算区域-species级别的grassland系数（ha/head）
        grassland_coef_ha_per_head = load_grassland_coefficients(
            feed_need_xlsx=paths.feed_need_xlsx,
            grass_ratio_xlsx=paths.grass_ratio_xlsx,
            pasture_yield_xlsx=paths.pasture_dm_yield_xlsx,
            dict_v3_path=paths.dict_v3_path,
            years=universe.years
        )
        
        if not grassland_coef_ha_per_head:
            print("[GRASSLAND_COEF] ⚠️ 未计算到任何grassland系数，跳过分配")
            return
        
        print(f"[GRASSLAND_COEF] 已加载 {len(grassland_coef_ha_per_head)} 个区域-species grassland系数")
        
        # 打印字典中的species列表（去重）
        unique_species = sorted(set(species for (region, species) in grassland_coef_ha_per_head.keys()))
        print(f"[GRASSLAND_COEF] 字典中的species列表: {unique_species}")
        
        # 打印字典中的区域列表（去重）
        unique_regions = sorted(set(region for (region, species) in grassland_coef_ha_per_head.keys()))
        print(f"[GRASSLAND_COEF] 字典中的区域列表: {unique_regions}")
        
        # 打印grassland_coef的样例键
        sample_keys = list(grassland_coef_ha_per_head.keys())[:10]
        print(f"[GRASSLAND_COEF] 系数字典样例键: {sample_keys}")
        
        # 🔍 关键诊断：打印Argentina有哪些species
        argentina_species = sorted([species for (region, species) in grassland_coef_ha_per_head.keys() if region == 'Argentina'])
        print(f"\n[GRASSLAND_COEF] 🔍 Argentina可用的species: {argentina_species}")
        if len(argentina_species) < 10:
            print(f"[GRASSLAND_COEF] ⚠️ Argentina只有 {len(argentina_species)} 个species，数据可能不完整！")
        
        # 如果有stock_df，计算production→head转换因子
        production_to_head: Dict[Tuple[str, str, int], float] = {}
        if stock_df is not None and not stock_df.empty:
            try:
                # 简化：使用历史平均的production/stock比率
                # stock_df: [country, commodity, year, stock_head]
                # 需要production数据来计算比率
                hist_years = [y for y in universe.years if y <= 2020]
                
                # 从nodes获取历史production数据
                prod_data = []
                for n in nodes:
                    if n.year in hist_years and n.Q0 > 0:
                        prod_data.append({
                            'country': n.country,
                            'commodity': n.commodity,
                            'year': n.year,
                            'production_t': n.Q0
                        })
                
                if prod_data:
                    # pd已经在文件顶部导入，无需重新导入
                    prod_df = pd.DataFrame(prod_data)
                    # 合并stock和production数据
                    merged = prod_df.merge(
                        stock_df[['country', 'commodity', 'year', 'stock_head']],
                        on=['country', 'commodity', 'year'],
                        how='inner'
                    )
                    
                    # 计算 ton/head 比率（历史平均）
                    merged['ton_per_head'] = merged['production_t'] / merged['stock_head'].clip(lower=1e-6)
                    
                    # 按country-commodity聚合（取历史平均）
                    avg_ratio = merged.groupby(['country', 'commodity'])['ton_per_head'].mean().reset_index()
                    
                    # 构建字典（扩展到所有年份）
                    for _, row in avg_ratio.iterrows():
                        country = str(row['country'])
                        commodity = str(row['commodity'])
                        ratio = float(row['ton_per_head'])
                        for year in universe.years:
                            production_to_head[(country, commodity, year)] = ratio
                    
                    print(f"[GRASSLAND_COEF] 已计算 {len(production_to_head)} 个 production→head 转换因子")
            
            except Exception as e:
                print(f"[GRASSLAND_COEF] ⚠️ 计算production→head转换失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 构建country→region映射（从universe.region_aggMC_by_country）
        country_to_region = {}
        if hasattr(universe, 'region_aggMC_by_country'):
            country_to_region = universe.region_aggMC_by_country or {}
        
        if not country_to_region:
            # 如果没有区域映射，尝试从dict_v3加载
            try:
                # pd已经在文件顶部导入
                dict_v3 = pd.read_excel(paths.dict_v3_path, sheet_name='region')
                for _, row in dict_v3.iterrows():
                    country = str(row.get('Country', '')).strip()
                    region = str(row.get('Region_market_agg', '')).strip()
                    if country and region and region != 'no':
                        country_to_region[country] = region
            except Exception as e:
                print(f"[GRASSLAND_COEF] ⚠️ 从dict_v3加载区域映射失败: {e}")
        
        # ✅ 从dict_v3读取commodity→species标准化映射
        # nodes中commodity格式: Item_Emis (如 'Cattle, non-dairy', 'Sheep, dairy')
        # grassland_coef字典中species格式: Item_Feed_Map (如 'beef_cattle', 'dairy_sheep')
        commodity_to_species = {}
        try:
            emis_item_df = pd.read_excel(paths.dict_v3_path, sheet_name='Emis_item')
            for _, row in emis_item_df.iterrows():
                item_emis = row.get('Item_Emis')
                item_feed = row.get('Item_Feed_Map')
                if pd.notna(item_emis) and pd.notna(item_feed):
                    # ✅ 关键修复：Item_Feed_Map格式转换
                    # dict_v3中可能是 "Beef_Cattle" 或 "Beef" 等格式
                    # 需要转换为grassland_coef中的species格式（小写+下划线）
                    item_emis_clean = str(item_emis).strip()
                    item_feed_clean = str(item_feed).strip()
                    # 统一转换：空格→下划线，全部小写
                    species_name = item_feed_clean.replace(' ', '_').lower()
                    commodity_to_species[item_emis_clean] = species_name
            
            print(f"[GRASSLAND_COEF] ✅ 从dict_v3加载了 {len(commodity_to_species)} 个 Item_Emis→Item_Feed_Map 映射")
            # 打印映射样例（前15个）
            sample_mapping = list(commodity_to_species.items())[:15]
            print(f"[GRASSLAND_COEF] 映射样例:")
            for emis, feed in sample_mapping:
                print(f"  '{emis}' → '{feed}'")
        except Exception as e:
            print(f"[GRASSLAND_COEF] ⚠️ 从dict_v3加载映射失败: {e}，使用默认映射")
            # 回退到默认映射
            commodity_to_species = {
                'Cattle, dairy': 'dairy_cattle',
                'Cattle, non-dairy': 'beef_cattle',
                'Buffalo, dairy': 'dairy_buffalo',
                'Buffalo, non-dairy': 'meat_buffalo',
                'Sheep, dairy': 'dairy_sheep',
                'Sheep, non-dairy': 'meat_sheep',
                'Goats, dairy': 'dairy_goat',
                'Goats, non-dairy': 'meat_goat',
                'Camel, dairy': 'dairy_camel',
                'Camel, non-dairy': 'meat_camel',
                'Swine': 'pigs',
                'Chickens, broilers': 'broilers',
                'Chickens, layers': 'layers',
                'Ducks': 'ducks',
                'Geese and guinea fowls': 'geese_guinea',
                'Turkeys': 'turkeys',
                'Horses': 'horse',
                'Asses': 'asses',
                'Mules and hinnies': 'mules_and_hinnies',
                'Llamas': 'llamas',
            }
        
        # ✅ 修复：直接使用commodity_to_species的键作为livestock判断依据
        # 这是实际数据中存在的commodity名称，不需要依赖旧的LIVESTOCK_COMMODITIES常量
        # commodity_to_species的键已经包含了所有细分后的livestock名称：
        #   'Cattle, dairy', 'Cattle, non-dairy', 'Buffalo, dairy', 'Buffalo, non-dairy', 等
        livestock_commodities_actual = set(commodity_to_species.keys())
        
        def is_livestock(commodity: str) -> bool:
            """判断commodity是否为livestock（使用实际数据中的commodity名称）"""
            return commodity in livestock_commodities_actual
        
        # 统计匹配情况
        livestock_node_count = sum(1 for n in nodes if is_livestock(n.commodity))
        mapped_commodities = sum(1 for n in nodes if n.commodity in commodity_to_species)
        print(f"[GRASSLAND_COEF] 总节点数: {len(nodes)}, Livestock节点数: {livestock_node_count}, 可映射节点: {mapped_commodities}")
        
        # 🔍 诊断：检查Argentina所有livestock节点（所有年份）
        argentina_all_livestock = {}
        for n in nodes:
            if n.country == 'Argentina' and is_livestock(n.commodity):
                if n.commodity not in argentina_all_livestock:
                    argentina_all_livestock[n.commodity] = []
                argentina_all_livestock[n.commodity].append(n.year)
        
        if argentina_all_livestock:
            print(f"\n[GRASSLAND_COEF] 🔍 Argentina所有livestock节点:")
            for comm, years in sorted(argentina_all_livestock.items()):
                print(f"  {comm:30s} | 年份数: {len(years)}, 年份范围: {min(years)}-{max(years)}")
        
        # 分配系数到节点
        assigned_count = 0
        skipped_count = 0
        # 收集诊断信息（仅Argentina 2020）
        argentina_2020_matched = []
        argentina_2020_unmatched = []
        # 收集所有被跳过的节点（没有ton_per_head数据）
        skipped_nodes = {}  # {(country, commodity): count}
        
        for n in nodes:
            if not is_livestock(n.commodity):
                continue
            
            # 获取区域
            region = country_to_region.get(n.country, n.country)
            
            # 标准化species名称
            species = commodity_to_species.get(n.commodity, None)
            
            # 诊断Argentina 2020
            if n.country == 'Argentina' and n.year == 2020:
                if species is None:
                    argentina_2020_unmatched.append((n.commodity, "commodity未在commodity_to_species字典中"))
                    continue
                
                # 打印查询键信息（仅第一次，并记录所有commodity的映射）
                if len(argentina_2020_matched) == 0 and len(argentina_2020_unmatched) == 0:
                    print(f"\n[GRASSLAND_COEF] 🔍 Argentina commodity→species映射检查:")
                    print(f"  country_to_region['Argentina'] = '{region}'")
                    # 打印所有Argentina livestock的commodity→species映射
                    argentina_livestock = [nd for nd in nodes if nd.country == 'Argentina' and nd.year == 2020 and is_livestock(nd.commodity)]
                    for nd in argentina_livestock[:10]:  # 只打印前10个
                        sp = commodity_to_species.get(nd.commodity, 'NOT_FOUND')
                        print(f"  '{nd.commodity}' → '{sp}'")
                
                # 查找系数（ha/head）
                coef_ha_per_head = grassland_coef_ha_per_head.get((region, species), 0.0)
                
                if coef_ha_per_head <= 0:
                    # 检查是否区域名问题或species名问题
                    region_exists = any(r == region for r, s in grassland_coef_ha_per_head.keys())
                    species_exists = any(s == species for r, s in grassland_coef_ha_per_head.keys())
                    detail = f"region存在={region_exists}, species存在={species_exists}"
                    argentina_2020_unmatched.append((n.commodity, f"grassland_coef中找不到('{region}', '{species}') | {detail}"))
                    continue
                
                # 转换为 ha/ton - ✅ 必须有实际的ton_per_head，不使用fallback
                ton_per_head = production_to_head.get((n.country, n.commodity, n.year), 0.0)
                if ton_per_head > 0:
                    coef_ha_per_ton = coef_ha_per_head / ton_per_head
                    argentina_2020_matched.append((n.commodity, coef_ha_per_ton, f"species={species}, coef={coef_ha_per_head:.3f} ha/head, yield={ton_per_head:.3f} t/head"))
                    n.meta['grassland_coef'] = float(coef_ha_per_ton)
                    assigned_count += 1
                else:
                    argentina_2020_unmatched.append((n.commodity, f"production_to_head中找不到('{n.country}', '{n.commodity}', {n.year})"))
                continue
            
            # 其他节点正常处理
            if species is None:
                continue
            
            # 查找系数（ha/head）
            coef_ha_per_head = grassland_coef_ha_per_head.get((region, species), 0.0)
            
            if coef_ha_per_head <= 0:
                continue
            
            # 转换为 ha/ton - ✅ 必须有实际的ton_per_head，不使用fallback
            ton_per_head = production_to_head.get((n.country, n.commodity, n.year), 0.0)
            if ton_per_head > 0:
                coef_ha_per_ton = coef_ha_per_head / ton_per_head
                n.meta['grassland_coef'] = float(coef_ha_per_ton)
                assigned_count += 1
            else:
                # 记录被跳过的节点
                key = (n.country, n.commodity)
                skipped_nodes[key] = skipped_nodes.get(key, 0) + 1
                skipped_count += 1
        
        print(f"[GRASSLAND_COEF] ✅ 已为 {assigned_count} 个livestock节点分配grassland系数")
        print(f"[GRASSLAND_COEF] ⚠️  跳过 {skipped_count} 个节点（缺少ton_per_head数据）")
        
        # 打印跳过节点的汇总（按country-commodity分组）
        if skipped_nodes:
            print(f"\n[GRASSLAND_COEF] 📋 跳过节点汇总（所有 {len(skipped_nodes)} 个country-commodity组合）:")
            sorted_skipped = sorted(skipped_nodes.items(), key=lambda x: (x[0][0], x[0][1]))  # 按country, commodity排序
            for (country, commodity), count in sorted_skipped:
                print(f"  {country:25s} | {commodity:25s} | {count} 个年份")
        
        # 打印Argentina 2020的详细诊断
        print(f"\n[GRASSLAND_COEF] 🔍 Argentina 2020年匹配诊断:")
        print(f"  ✅ 匹配成功 ({len(argentina_2020_matched)} 个):")
        for comm, coef, detail in sorted(argentina_2020_matched):
            print(f"    {comm:25s} | {coef:10.3f} ha/ton | {detail}")
        
        print(f"\n  ❌ 匹配失败 ({len(argentina_2020_unmatched)} 个):")
        for comm, reason in sorted(argentina_2020_unmatched):
            print(f"    {comm:25s} | {reason}")
    
    except Exception as e:
        print(f"[GRASSLAND_COEF] ❌ 分配grassland系数失败: {e}")
        import traceback
        traceback.print_exc()


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
    emis_to_elast = _build_emis_to_elast_map()
    elast_to_emis_single = _build_elast_to_emis_map()
    elast_to_emis_multi = _build_elast_to_emis_multi()
    # demand income
    if 'Demand-Income' in xls.sheet_names:
        df = _lc(pd.read_excel(xls, 'Demand-Income'))
        eps_income = _build_country_elasticity(df, value_col='Elasticity_mean')
    # demand population
    if 'Demand-Population' in xls.sheet_names:
        df = _lc(pd.read_excel(xls, 'Demand-Population'))
        eps_pop = _build_country_elasticity(df, value_col='Elasticity_mean')
    # demand own-price
    if 'Demand-Own-Price' in xls.sheet_names:
        df = _lc(pd.read_excel(xls, 'Demand-Own-Price'))
        raw_own = _build_elasticity_map(df, value_col='Elasticity_mean')
        eps_own = {}
        for (country, elast_key), val in raw_own.items():
            emis_list = elast_to_emis_multi.get(elast_key, [elast_to_emis_single.get(elast_key, elast_key)])
            for emis_comm in emis_list:
                eps_own[(country, emis_comm)] = val
    # demand cross-price matrix
    if 'Demand_Cross_mean' in xls.sheet_names:
        df = _lc(pd.read_excel(xls, 'Demand_Cross_mean'))
        cross_raw = _build_cross_elasticity_map(df, commodity_filter=set(universe.commodities or []))
        # remap row key elasticity->emis (一对多)
        cross = {}
        for (country, elast_key), row in cross_raw.items():
            emis_list = elast_to_emis_multi.get(elast_key, [elast_to_emis_single.get(elast_key, elast_key)])
            for emis_comm in emis_list:
                row_mapped = {}
                for k, v in row.items():
                    emis_k_list = elast_to_emis_multi.get(k, [elast_to_emis_single.get(k, k)])
                    for emis_k in emis_k_list:
                        row_mapped[emis_k] = v
                cross[(country, emis_comm)] = row_mapped
    return eps_income, eps_pop, eps_own, cross

def apply_demand_elasticities_to_nodes(nodes: List[Node], universe: Universe, elasticity_xlsx: str) -> None:
    eps_income, eps_pop, eps_own, cross = load_demand_elasticities(elasticity_xlsx, universe)
    emis_to_elast = _build_emis_to_elast_map()
    for n in nodes:
        # income/population elasticity
        setattr(n, 'eps_income_demand', float(eps_income.get(n.country, 0.0)))
        setattr(n, 'eps_pop_demand', float(eps_pop.get(n.country, 0.0)))
        elast_key = emis_to_elast.get(n.commodity, n.commodity)
        setattr(n, 'eps_demand', float(eps_own.get((n.country, n.commodity), eps_own.get((n.country, elast_key), 0.0))))
        # cross-price row dict
        eps_row = cross.get((n.country, n.commodity), cross.get((n.country, elast_key), {}))
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
    """Load per-country GDP change ratios (relative to 2020) from SSPDB_future_GDP_with_change_ratio.xlsx."""
    out: Dict[Tuple[str,int], float] = {}
    if not os.path.exists(xlsx_path):
        return out
    try:
        df = _lc(pd.read_excel(xlsx_path, sheet_name='change_ratio_country'))
    except Exception:
        return out
    if 'M49_Country_Code' not in df.columns:
        return out
    df = df[df['SCENARIO'].astype(str) == str(scenario)]
    value_cols = [c for c in df.columns if isinstance(c, str) and c.startswith('Y') and c[1:].isdigit()]
    if not value_cols:
        return out
    for r in df.itertuples(index=False):
        code = _parse_m49_code(getattr(r, 'M49_Country_Code'))
        if code is None:
            continue
        m49 = f"'{int(code):03d}"  # ✅ 'xxx格式
        country = universe.country_by_m49.get(m49)
        if not country:
            continue
        for col in value_cols:
            year = int(col[1:])
            if year not in universe.years:
                continue
            val = pd.to_numeric(getattr(r, col), errors='coerce')
            if not np.isfinite(val):
                continue
            out[(country, year)] = float(val)
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
                    path_params = os.path.join(get_src_base(), 'Livestock_Manure_parameters.xlsx')
                    P = load_wide(path_params) if os.path.exists(path_params) else None
                    pop = livestock_stock_df.copy() if isinstance(livestock_stock_df, pd.DataFrame) else pd.DataFrame()
                    if P is not None and len(pop):
                        # AreaCode from dict_v3.xlsx::region M49
                        try:
                            xls = pd.ExcelFile(os.path.join(get_src_base(), 'dict_v3.xlsx'))
                            region = _lc(pd.read_excel(xls, 'region'))
                            c_cty = _find_col(region, ['Country'])
                            c_m49 = _maybe_find_col(region, ['M49_Country_Code','M49 Code','M49','M49 Code_xxx'])
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
    production_by_item: Dict[str, str]
    fertilizer_by_item: Dict[str, str]
    yield_item_to_comm: Dict[str, str]
    yield_element_by_item: Dict[str, str]
    yield_unit_by_item: Dict[str, str]
    area_item_to_comm: Dict[str, str]
    area_element_by_item: Dict[str, str]
    area_unit_by_item: Dict[str, str]
    slaughter_item_to_comm: Dict[str, str]
    slaughter_element_by_item: Dict[str, str]
    slaughter_unit_by_item: Dict[str, str]
    stock_item_to_comm: Dict[str, str]
    stock_element_by_item: Dict[str, str]
    stock_unit_by_item: Dict[str, str]
    elasticity_by_item: Dict[str, str]
    feed_item_to_comm: Dict[str, str]

def load_emis_item_mappings(xls_path: str) -> EmisItemMappings:
    """Parse dict_v3.xlsx Emis_item sheet for multi-domain item mappings and units.
    Returns a structured mapping object for consistent FAOSTAT alignment.
    """
    if not os.path.exists(xls_path):
        return EmisItemMappings({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})
    xls = pd.ExcelFile(xls_path)
    df = _lc(pd.read_excel(xls, 'Emis_item'))

    # columns
    def col(name: str) -> Optional[str]:
        return name if name in df.columns else None

    c_emis = col('Item_Emis')  # The model commodity name
    c_prod = col('Item_Production_Map')
    c_fert = col('Item_Fertilizer_Map')
    c_yield = col('Item_Yield_Map')
    c_yield_elem = col('Item_Yield_Element')
    c_yield_unit = col('Item_Yield_Unit')
    c_area = col('Item_Area_Map')
    c_area_elem = col('Item_Area_Element')
    c_area_unit = col('Item_Area_Unit')
    c_sl_map = col('Item_Slaughtered_Map')
    c_sl_elem = col('Item_Slaughtered_Element')
    c_sl_unit = col('Item_Slaughtered_Unit')
    c_stock_map = col('Item_Stock_Map')
    c_stock_elem = col('Item_Stock_Element')
    c_stock_unit = col('Item_Stock_Unit')
    c_elast = col('Item_Elasticity_Map')
    c_feed = col('Item_Feed_Map')

    def _clean(val: Any) -> Optional[str]:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        s = str(val).strip()
        if not s or s.lower() in {'nan', 'no'}:
            return None
        return s

    production_by_item: Dict[str, str] = {}
    fertilizer_by_item: Dict[str, str] = {}
    yield_item_to_comm: Dict[str, str] = {}
    yield_element_by_item: Dict[str, str] = {}
    yield_unit_by_item: Dict[str, str] = {}
    area_item_to_comm: Dict[str, str] = {}
    area_element_by_item: Dict[str, str] = {}
    area_unit_by_item: Dict[str, str] = {}
    slaughter_item_to_comm: Dict[str, str] = {}
    slaughter_element_by_item: Dict[str, str] = {}
    slaughter_unit_by_item: Dict[str, str] = {}
    stock_item_to_comm: Dict[str, str] = {}
    stock_element_by_item: Dict[str, str] = {}
    stock_unit_by_item: Dict[str, str] = {}
    elasticity_by_item: Dict[str, str] = {}
    feed_item_to_comm: Dict[str, str] = {}

    for r in df.itertuples(index=False):
        # Get the model commodity name (Item_Emis) - this is the target for all mappings
        commodity = _clean(getattr(r, c_emis)) if c_emis else None
        if not commodity:
            continue  # Skip rows without Item_Emis

        # Map FAOSTAT Item names to model Item_Emis
        prod_item = _clean(getattr(r, c_prod)) if c_prod else None
        if prod_item and commodity:
            production_by_item[prod_item] = commodity

        fert_item = _clean(getattr(r, c_fert)) if c_fert else None
        if fert_item and commodity:
            fertilizer_by_item[fert_item] = commodity

        yield_item = _clean(getattr(r, c_yield)) if c_yield else None
        if yield_item and commodity:
            yield_item_to_comm[yield_item] = commodity
            elem = _clean(getattr(r, c_yield_elem)) if c_yield_elem else None
            if elem:
                yield_element_by_item[yield_item] = elem
            unit = _clean(getattr(r, c_yield_unit)) if c_yield_unit else None
            if unit:
                yield_unit_by_item[yield_item] = unit

        area_item = _clean(getattr(r, c_area)) if c_area else None
        if area_item and commodity:
            area_item_to_comm[area_item] = commodity
            elem = _clean(getattr(r, c_area_elem)) if c_area_elem else None
            if elem:
                area_element_by_item[area_item] = elem
            unit = _clean(getattr(r, c_area_unit)) if c_area_unit else None
            if unit:
                area_unit_by_item[area_item] = unit

        slaughter_item = _clean(getattr(r, c_sl_map)) if c_sl_map else None
        if slaughter_item and commodity:
            slaughter_item_to_comm[slaughter_item] = commodity
            elem = _clean(getattr(r, c_sl_elem)) if c_sl_elem else None
            if elem:
                slaughter_element_by_item[slaughter_item] = elem
            unit = _clean(getattr(r, c_sl_unit)) if c_sl_unit else None
            if unit:
                slaughter_unit_by_item[slaughter_item] = unit

        stock_item = _clean(getattr(r, c_stock_map)) if c_stock_map else None
        if stock_item and commodity:
            stock_item_to_comm[stock_item] = commodity
            elem = _clean(getattr(r, c_stock_elem)) if c_stock_elem else None
            if elem:
                stock_element_by_item[stock_item] = elem
            unit = _clean(getattr(r, c_stock_unit)) if c_stock_unit else None
            if unit:
                stock_unit_by_item[stock_item] = unit

        elast_item = _clean(getattr(r, c_elast)) if c_elast else None
        if elast_item and commodity:
            elasticity_by_item[elast_item] = commodity

        feed_item = _clean(getattr(r, c_feed)) if c_feed else None
        if feed_item and commodity:
            feed_item_to_comm[feed_item] = commodity

    return EmisItemMappings(
        production_by_item=production_by_item,
        fertilizer_by_item=fertilizer_by_item,
        yield_item_to_comm=yield_item_to_comm,
        yield_element_by_item=yield_element_by_item,
        yield_unit_by_item=yield_unit_by_item,
        area_item_to_comm=area_item_to_comm,
        area_element_by_item=area_element_by_item,
        area_unit_by_item=area_unit_by_item,
        slaughter_item_to_comm=slaughter_item_to_comm,
        slaughter_element_by_item=slaughter_element_by_item,
        slaughter_unit_by_item=slaughter_unit_by_item,
        stock_item_to_comm=stock_item_to_comm,
        stock_element_by_item=stock_element_by_item,
        stock_unit_by_item=stock_unit_by_item,
        elasticity_by_item=elasticity_by_item,
        feed_item_to_comm=feed_item_to_comm,
    )
def _match_element(row_element: Any, target: Optional[str]) -> bool:
    if target is None:
        return False
    if row_element is None or (isinstance(row_element, float) and pd.isna(row_element)):
        return False
    return str(row_element).strip().lower() == str(target).strip().lower()


def _convert_yield_unit(value: float, unit: Optional[str]) -> float:
    if not np.isfinite(value):
        return np.nan
    u = (unit or '').strip().lower()
    if u in {'kg/ha', 'kg ha-1', 'kilogram per hectare'}:
        return value / 1000.0
    if u in {'hg/ha', 'hectogram per hectare'}:
        return value / 100.0
    if u in {'t/ha', 'tonne per hectare', 'tonnes per hectare', 'ton/ha'}:
        return value
    return value


def _convert_carcass_unit(value: float, unit: Optional[str]) -> float:
    if not np.isfinite(value):
        return np.nan
    u = (unit or '').strip().lower()
    if u in {'kg/an', 'kg/animal', 'kilogram per animal'}:
        return value / 1000.0
    if u in {'t/an', 'tonne per animal'}:
        return value
    return value


def _convert_livestock_yield_unit(value: float, unit: Optional[str], item_raw: str) -> float:
    """
    转换livestock yield单位为统一的t/head
    对于milk/egg类，原始单位可能是kg/head, t/head, hg/head等
    """
    if not np.isfinite(value):
        return np.nan
    u = (unit or '').strip().lower()
    item_lower = (item_raw or '').lower()
    
    # Milk类 (kg -> t)
    if 'milk' in item_lower:
        if u in {'kg/an', 'kg/animal', 'kilogram per animal', 'kg'}:
            return value / 1000.0
        if u in {'hg/an', 'hg/animal', 'hectogram per animal', 'hg'}:
            return value / 10000.0
        if u in {'t/an', 'tonne per animal', 't', 'tonne'}:
            return value
    
    # Egg类 (pieces/head -> 假设平均重量转换，或直接使用数值)
    # 注: Eggs通常用production/stock计算，这里的yield可能已经是标准化的
    if 'egg' in item_lower:
        if u in {'kg/an', 'kg/animal', 'kilogram per animal', 'kg'}:
            return value / 1000.0
        if u in {'t/an', 'tonne per animal', 't', 'tonne'}:
            return value
        # pieces -> 假设每个蛋约0.06kg
        if u in {'head', 'pieces', 'number'}:
            return value * 0.06 / 1000.0
    
    # 默认: kg -> t
    if 'kg' in u:
        return value / 1000.0
    
    return value


def _convert_area_unit(value: float, unit: Optional[str]) -> float:
    if not np.isfinite(value):
        return np.nan
    u = (unit or '').strip().lower()
    if u in {'1000 ha', 'thousand ha', '1000ha'}:
        return value * 1000.0
    return value


def build_faostat_production_indicators(production_csv: str,
                                        universe: Universe,
                                        maps: Optional[EmisItemMappings] = None) -> Dict[str, pd.DataFrame]:
    # ✅ 所有DataFrame都包含M49_Country_Code列（唯一标识符）
    empty = {
        'production': pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','commodity','production_t']),
        'yield': pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','commodity','yield_t_per_ha']),
        'area': pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','commodity','area_ha']),
        'slaughter': pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','commodity','slaughter_head']),
        'livestock_yield': pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','commodity','yield_t_per_head']),
        'stock': pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','commodity','stock_head']),
    }
    if not os.path.exists(production_csv):
        print(f"[ERROR] 关键文件不存在: {production_csv}")
        raise FileNotFoundError(f"Production CSV file not found: {production_csv}")
    if maps is None:
        maps = load_emis_item_mappings(os.path.join(get_src_base(), 'dict_v3.xlsx'))

    df_raw = pd.read_csv(production_csv)
    df_raw = _filter_select_rows(df_raw)
    df = _lc(_faostat_wide_to_long(df_raw))
    c_area = _find_col(df, ['Area'])
    c_year = _find_col(df, ['Year'])
    c_item = _find_col(df, ['Item'])
    c_elem = _find_col(df, ['Element'])
    c_val = _find_col(df, ['Value'])
    c_unit = _maybe_find_col(df, ['Unit'])
    if not all([c_area, c_year, c_item, c_elem, c_val]):
        return empty

    keep_cols = [c_area, c_year, c_item, c_elem, c_val] + ([c_unit] if c_unit else [])
    if 'M49_Country_Code' in df.columns:
        keep_cols.append('M49_Country_Code')
    z = df[keep_cols].copy()
    rename_map = {c_area: 'area', c_year: 'year', c_item: 'item_raw', c_elem: 'element', c_val: 'value'}
    if c_unit:
        rename_map[c_unit] = 'unit'
    z = z.rename(columns=rename_map)
    z['value'] = pd.to_numeric(z['value'], errors='coerce')
    z = z.dropna(subset=['value'])

    z = _attach_country_from_m49(df, z, universe, context="FAOSTAT production")
    # ✅ 保留M49_Country_Code列，确保索引对齐
    if 'M49_Country_Code' in df.columns and 'M49_Country_Code' not in z.columns:
        # 使用.loc[]确保索引对齐
        z['M49_Country_Code'] = df.loc[z.index, 'M49_Country_Code'].values
    
    # ✅ 规范化M49格式为 'xxx（单引号+3位数字）
    if 'M49_Country_Code' in z.columns:
        def _format_m49_quote(val):
            """将M49格式化为 'xxx 格式"""
            if pd.isna(val):
                return val
            s = str(val).strip().strip("'\"")
            try:
                return f"'{int(s):03d}"
            except:
                return f"'{s}"
        z['M49_Country_Code'] = z['M49_Country_Code'].apply(_format_m49_quote)
    
    z['country'] = z['country'].astype(str).str.strip()
    z = z[z['country'].isin(universe.countries)]
    z['iso3'] = z['country'].map(universe.iso3_by_country)
    z = z.dropna(subset=['iso3'])
    z['iso3'] = z['iso3'].astype(str)
    z['year'] = pd.to_numeric(z['year'], errors='coerce')
    z = z.dropna(subset=['year'])
    z['year'] = z['year'].astype(int)
    if 'unit' not in z.columns:
        z['unit'] = ''

    def _group(df_subset: pd.DataFrame, value_col: str) -> pd.DataFrame:
        if df_subset.empty:
            base_cols = ['M49_Country_Code','country','iso3','year','commodity', value_col]
            return pd.DataFrame(columns=base_cols)
        
        # ✅ 保留M49_Country_Code列（如果存在）
        group_cols = ['country','iso3','year','commodity']
        if 'M49_Country_Code' in df_subset.columns:
            group_cols = ['M49_Country_Code'] + group_cols
        
        g = df_subset.groupby(group_cols, as_index=False)['value'].sum()
        g = g.rename(columns={'value': value_col})
        # WARNING: do not filter using universe.commodities!
        # 因为slaughter/stock等使用不同的Item_XXX_Map名称
        # commodity已经通过maps.xxx_item_to_comm映射验证过了
        return g

    # Production
    prod = z[z['element'].str.contains('Production', case=False, na=False)].copy()
    prod['commodity'] = prod['item_raw'].map(maps.production_by_item).fillna(prod['item_raw'])
    production_df = _group(prod, 'production_t')

    # Area
    area_items = maps.area_item_to_comm or maps.production_by_item
    area = z[z['item_raw'].isin(area_items.keys())].copy()
    area['target_elem'] = area['item_raw'].map(maps.area_element_by_item)
    area['element_norm'] = area['element'].astype(str).str.strip().str.lower()
    area['target_norm'] = area['target_elem'].fillna('').astype(str).str.strip().str.lower()
    mask_specific = area['target_elem'].notna() & (area['element_norm'] == area['target_norm'])
    mask_default = area['target_elem'].isna() & area['element_norm'].str.contains('area harvested', na=False)
    area = area[mask_specific | mask_default]
    area['commodity'] = area['item_raw'].map(area_items).fillna(area['item_raw'])
    area['value'] = area.apply(lambda r: _convert_area_unit(float(r['value']), r['unit']), axis=1)
    area_df = _group(area, 'area_ha')

    # Yield (crop)
    yields = z[z['item_raw'].isin(maps.yield_item_to_comm.keys())].copy()
    yields['target_elem'] = yields['item_raw'].map(maps.yield_element_by_item)
    yields['element_norm'] = yields['element'].astype(str).str.strip().str.lower()
    yields['target_norm'] = yields['target_elem'].fillna('').astype(str).str.strip().str.lower()
    mask_specific = yields['target_elem'].notna() & (yields['element_norm'] == yields['target_norm'])
    mask_default = yields['target_elem'].isna() & yields['element_norm'].str.contains('yield', na=False)
    yields = yields[mask_specific | mask_default]
    yields['commodity'] = yields['item_raw'].map(maps.yield_item_to_comm).fillna(yields['item_raw'])
    yields['value'] = yields.apply(lambda r: _convert_yield_unit(float(r['value']), r['unit']), axis=1)
    yield_df = _group(yields, 'yield_t_per_ha')

    # Livestock yield (meat/milk/egg产率)
    # 对于meat类: Yield/Carcass Weight
    # 对于milk/egg类: Production/Stock (yield)
    livestock_yield_list = []
    
    # 1. Meat类: Carcass Weight (t/head)
    carcass = z[(z['item_raw'].isin(maps.yield_item_to_comm.keys())) &
                (z['element'].str.contains('carcass', case=False, na=False))].copy()
    if not carcass.empty:
        carcass['commodity'] = carcass['item_raw'].map(maps.yield_item_to_comm).fillna(carcass['item_raw'])
        carcass['value'] = carcass.apply(lambda r: _convert_carcass_unit(float(r['value']), r['unit']), axis=1)
        carcass_grouped = _group(carcass, 'yield_t_per_head')
        livestock_yield_list.append(carcass_grouped)
    
    # 2. Milk/Egg类: Yield (通过Item_Yield_Map和Item_Yield_Element匹配)
    # 找出所有有Yield_Element定义的livestock items
    livestock_yield_items = {k: v for k, v in maps.yield_item_to_comm.items() 
                            if k in maps.yield_element_by_item}
    if livestock_yield_items:
        livestock_yields = z[z['item_raw'].isin(livestock_yield_items.keys())].copy()
        livestock_yields['target_elem'] = livestock_yields['item_raw'].map(maps.yield_element_by_item)
        livestock_yields['element_norm'] = livestock_yields['element'].astype(str).str.strip().str.lower()
        livestock_yields['target_norm'] = livestock_yields['target_elem'].fillna('').astype(str).str.strip().str.lower()
        
        # 只保留element匹配的行（排除carcass，因为已经在上面处理过了）
        mask_match = (livestock_yields['target_elem'].notna() & 
                     (livestock_yields['element_norm'] == livestock_yields['target_norm']) &
                     ~livestock_yields['element'].str.contains('carcass', case=False, na=False))
        livestock_yields = livestock_yields[mask_match]
        
        if not livestock_yields.empty:
            livestock_yields['commodity'] = livestock_yields['item_raw'].map(maps.yield_item_to_comm).fillna(livestock_yields['item_raw'])
            # 根据item类型转换单位
            livestock_yields['value'] = livestock_yields.apply(
                lambda r: _convert_livestock_yield_unit(float(r['value']), r['unit'], r['item_raw']), 
                axis=1
            )
            livestock_yield_grouped = _group(livestock_yields, 'yield_t_per_head')
            livestock_yield_list.append(livestock_yield_grouped)
    
    # 合并所有livestock yield数据
    if livestock_yield_list:
        livestock_yield_df = pd.concat(livestock_yield_list, ignore_index=True)
        # 如果同一个commodity有多个来源，取平均值
        # ✅ 保留M49_Country_Code列
        group_cols = ['country','iso3','year','commodity']
        if 'M49_Country_Code' in livestock_yield_df.columns:
            group_cols = ['M49_Country_Code'] + group_cols
        livestock_yield_df = livestock_yield_df.groupby(group_cols, as_index=False)['yield_t_per_head'].mean()
    else:
        livestock_yield_df = pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','commodity','yield_t_per_head'])

    # Slaughter
    sl_items = maps.slaughter_item_to_comm
    slaughter = z[z['item_raw'].isin(sl_items.keys())].copy()
    slaughter['target_elem'] = slaughter['item_raw'].map(maps.slaughter_element_by_item)
    slaughter['target_norm'] = slaughter['target_elem'].fillna('').astype(str).str.strip().str.lower()
    slaughter['element_norm'] = slaughter['element'].astype(str).str.strip().str.lower()
    mask_slaughter = slaughter['target_norm'].str.contains('slaughter', na=False)
    slaughter = slaughter[mask_slaughter & (slaughter['element_norm'] == slaughter['target_norm'])]
    slaughter['commodity'] = slaughter['item_raw'].map(sl_items).fillna(slaughter['item_raw'])
    slaughter_df = _group(slaughter, 'slaughter_head')

    # Stock
    stock_items = maps.stock_item_to_comm
    stock = z[z['item_raw'].isin(stock_items.keys())].copy()
    stock['target_elem'] = stock['item_raw'].map(maps.stock_element_by_item)
    stock['target_norm'] = stock['target_elem'].fillna('').astype(str).str.strip().str.lower()
    stock['element_norm'] = stock['element'].astype(str).str.strip().str.lower()
    stock = stock[stock['target_elem'].notna() & (stock['element_norm'] == stock['target_norm'])]
    stock['commodity'] = stock['item_raw'].map(stock_items).fillna(stock['item_raw'])
    stock_df = _group(stock, 'stock_head')

    return {
        'production': production_df,
        'yield': yield_df,
        'area': area_df,
        'slaughter': slaughter_df,
        'livestock_yield': livestock_yield_df,  # 改名: carcass_yield -> livestock_yield
        'stock': stock_df,
    }


def _extend_future_years(df: pd.DataFrame,
                         value_col: str,
                         universe: Universe,
                         commodity_required: bool = True) -> pd.DataFrame:
    """将历史数据扩展到未来年份，保留M49_Country_Code列"""
    if df is None or df.empty:
        return df
    required_cols = {'country', 'iso3', 'year'}
    if commodity_required:
        required_cols.add('commodity')
    if not required_cols.issubset(df.columns):
        return df
    df = df.copy()
    
    # ✅ 检查是否有M49列
    has_m49 = 'M49_Country_Code' in df.columns
    
    hist_mask = df['year'] <= 2020
    if not hist_mask.any():
        return df
    base_year = 2020 if (df['year'] == 2020).any() else int(df['year'].max())
    future_years = [y for y in universe.years if y > base_year]
    if not future_years:
        return df
    
    # ✅ 关键列必须包含M49（如果存在）
    key_cols = ['country', 'iso3'] + (['commodity'] if commodity_required else [])
    if has_m49:
        key_cols = ['M49_Country_Code'] + key_cols
    
    base = df[df['year'] == base_year][key_cols + [value_col]].copy()
    frames = [df]
    for y in future_years:
        missing_keys = df[df['year'] == y][key_cols]
        if not missing_keys.empty:
            continue
        tmp = base.copy()
        tmp['year'] = y
        frames.append(tmp)
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(key_cols + ['year'])
    out = out[out['year'].isin(universe.years)]
    out = out.sort_values(key_cols + ['year']).reset_index(drop=True)
    return out


def load_fertilizer_statistics(fert_xlsx: str,
                               universe: Universe,
                               maps: EmisItemMappings) -> Dict[str, pd.DataFrame]:
    eff_cols_name = 'fertilizer_efficiency_kgN_per_ha'
    amt_cols_name = 'fertilizer_n_input_t'
    # ✅ 添加M49_Country_Code列
    empty = {
        'efficiency': pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','commodity', eff_cols_name]),
        'amount': pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','commodity', amt_cols_name]),
    }
    if not os.path.exists(fert_xlsx):
        return empty
    df = _lc(pd.read_excel(fert_xlsx))
    c_m49 = _maybe_find_col(df, ['M49 Code', 'M49'])
    c_item = _maybe_find_col(df, ['Item'])
    c_prod_item = _maybe_find_col(df, ['Production_Item', 'Production Item'])
    eff_cols = [c for c in df.columns
                if isinstance(c, str) and c.startswith('N_FertEffi_') and str(c)[-4:].isdigit()]
    amt_cols = [c for c in df.columns
                if isinstance(c, str) and c.startswith('N_contentModi_') and str(c)[-4:].isdigit()]
    if not (c_m49 and (c_item or c_prod_item) and eff_cols):
        return empty
    idx_m49 = df.columns.get_loc(c_m49)
    idx_item = df.columns.get_loc(c_item) if c_item else None
    idx_prod_item = df.columns.get_loc(c_prod_item) if c_prod_item else None
    eff_cols_idx = [(col, df.columns.get_loc(col)) for col in eff_cols]
    amt_cols_idx = [(col, df.columns.get_loc(col)) for col in amt_cols]
    m49_to_country = {}
    for country, code in (universe.m49_by_country or {}).items():
        parsed = _parse_m49_code(code)
        if parsed is not None:
            m49_to_country[str(parsed)] = country
    records_eff: List[Dict[str, Any]] = []
    records_amt: List[Dict[str, Any]] = []
    def _clean_str(val: Any) -> Optional[str]:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        s = str(val).strip()
        if not s or s.lower() in {'nan', 'no'}:
            return None
        return s

    for row in df.itertuples(index=False, name=None):
        code_raw = row[idx_m49]
        parsed = _parse_m49_code(code_raw)
        country = m49_to_country.get(str(parsed)) if parsed is not None else None
        if country is None:
            continue
        item_name = _clean_str(row[idx_item]) if idx_item is not None else None
        prod_item_name = _clean_str(row[idx_prod_item]) if idx_prod_item is not None else None
        commodity: Optional[str] = None

        def _resolve(name: Optional[str]) -> Optional[str]:
            if not name:
                return None
            cand = maps.fertilizer_by_item.get(name)
            if cand and cand in universe.commodities:
                return cand
            cand = maps.production_by_item.get(name)
            if cand and cand in universe.commodities:
                return cand
            if name in universe.commodities:
                return name
            return None

        commodity = _resolve(item_name)
        if commodity is None:
            commodity = _resolve(prod_item_name)
        if commodity is None:
            continue
        iso3 = universe.iso3_by_country.get(country)
        if not iso3:
            continue
        # ✅ 修复：标准化M49代码为'xxx格式（单引号+3位数字）
        m49_normalized = f"'{str(parsed).zfill(3)}" if parsed is not None else None
        for col, col_idx in eff_cols_idx:
            year = int(str(col)[-4:])
            if year not in universe.years:
                continue
            val = pd.to_numeric(row[col_idx], errors='coerce')
            if pd.isna(val):
                continue
            records_eff.append({
                'M49_Country_Code': m49_normalized,  # ✅ 修复：使用标准化的M49代码
                'country': country,
                'iso3': iso3,
                'commodity': commodity,
                'year': year,
                eff_cols_name: float(val),
            })
        for col, col_idx in amt_cols_idx:
            year = int(str(col)[-4:])
            if year not in universe.years:
                continue
            val = pd.to_numeric(row[col_idx], errors='coerce')
            if pd.isna(val):
                continue
            records_amt.append({
                'M49_Country_Code': m49_normalized,  # ✅ 修复：使用标准化的M49代码
                'country': country,
                'iso3': iso3,
                'commodity': commodity,
                'year': year,
                amt_cols_name: float(val) / 1000.0,  # kgN -> tN
            })
    eff_df = pd.DataFrame(records_eff)
    amt_df = pd.DataFrame(records_amt)
    eff_df = _extend_future_years(eff_df, eff_cols_name, universe)
    amt_df = _extend_future_years(amt_df, amt_cols_name, universe)
    return {'efficiency': eff_df, 'amount': amt_df}


def load_feed_requirement_per_head(feed_xlsx: str,
                                   universe: Universe,
                                   maps: EmisItemMappings) -> pd.DataFrame:
    # ✅ 添加M49_Country_Code列
    columns = ['M49_Country_Code','country','iso3','year','commodity','feed_requirement_kg_per_head']
    if not os.path.exists(feed_xlsx):
        print(f"[ERROR] 关键文件不存在: {feed_xlsx}")
        raise FileNotFoundError(f"Feed requirement file not found: {feed_xlsx}")
    try:
        df = _lc(pd.read_excel(feed_xlsx, sheet_name='total_kgDM_per_head'))
    except Exception:
        return pd.DataFrame(columns=columns)
    c_species = _maybe_find_col(df, ['Species'])
    c_area = _maybe_find_col(df, ['M49_Country_Code'])
    value_cols = [c for c in df.columns if isinstance(c, str) and c.startswith('Y') and c[1:].isdigit()]
    if not (c_species and c_area and value_cols):
        return pd.DataFrame(columns=columns)
    m49_to_country = {}
    for country, code in (universe.m49_by_country or {}).items():
        parsed = _parse_m49_code(code)
        if parsed is not None:
            m49_to_country[str(parsed)] = country
    records: List[Dict[str, Any]] = []
    for row in df.itertuples(index=False):
        code_raw = getattr(row, c_area, None)
        parsed = _parse_m49_code(code_raw)
        country = m49_to_country.get(str(parsed)) if parsed is not None else None
        if country is None:
            continue
        species = str(getattr(row, c_species)).strip()
        commodity = maps.feed_item_to_comm.get(species)
        if not commodity:
            continue
        # WARNING: do not filter using universe.commodities!
        # feed使用Item_Feed_Map名称，universe.commodities使用Item_Production_Map名称
        iso3 = universe.iso3_by_country.get(country)
        if not iso3:
            continue
        # ✅ 修复：标准化M49代码为'xxx格式（单引号+3位数字）
        m49_normalized = f"'{str(parsed).zfill(3)}" if parsed is not None else None
        for col in value_cols:
            year = int(str(col).lstrip('Y'))
            if year not in universe.years:
                continue
            val = pd.to_numeric(getattr(row, col), errors='coerce')
            if pd.isna(val):
                continue
            records.append({
                'M49_Country_Code': m49_normalized,  # ✅ 修复：使用标准化的M49代码
                'country': country,
                'iso3': iso3,
                'commodity': commodity,
                'year': year,
                'feed_requirement_kg_per_head': float(val),
            })
    df_out = pd.DataFrame(records, columns=columns)
    df_out = _extend_future_years(df_out, 'feed_requirement_kg_per_head', universe)
    return df_out


def load_manure_management_ratio(manure_csv: str,
                                 universe: Universe,
                                 maps: EmisItemMappings) -> pd.DataFrame:
    """
    加载畜牧粪便管理比例
    WARNING: keep only commodities defined in dict_v3 (drop 'All Animals' and similar)
    """
    # ✅ 添加M49_Country_Code列
    columns = ['M49_Country_Code','country','iso3','year','commodity','manure_management_ratio']
    if not os.path.exists(manure_csv):
        print(f"[ERROR] 关键文件不存在: {manure_csv}")
        raise FileNotFoundError(f"Manure CSV file not found: {manure_csv}")
    df_raw = pd.read_csv(manure_csv)
    df_raw = _filter_select_rows(df_raw)
    df = _lc(_faostat_wide_to_long(df_raw))
    c_area = _find_col(df, ['Area'])
    c_year = _find_col(df, ['Year'])
    c_item = _find_col(df, ['Item'])
    c_elem = _find_col(df, ['Element'])
    c_val = _find_col(df, ['Value'])
    if not all([c_area, c_year, c_item, c_elem, c_val]):
        return pd.DataFrame(columns=columns)
    keep_cols = [c_area, c_year, c_item, c_elem, c_val]
    if 'M49_Country_Code' in df.columns:
        keep_cols.append('M49_Country_Code')
    z = df[keep_cols].copy()
    z = z.rename(columns={c_area: 'area', c_year: 'year', c_item: 'item_raw', c_elem: 'element', c_val: 'value'})
    z['value'] = pd.to_numeric(z['value'], errors='coerce')
    z = z.dropna(subset=['value'])
    z = _attach_country_from_m49(df, z, universe, context="Manure management ratio")
    # ✅ 保留M49_Country_Code列，确保索引对齐
    if 'M49_Country_Code' in df.columns and 'M49_Country_Code' not in z.columns:
        z['M49_Country_Code'] = df.loc[z.index, 'M49_Country_Code'].values
    
    # ✅ 规范化M49格式为 'xxx（单引号+3位数字）
    if 'M49_Country_Code' in z.columns:
        def _format_m49_quote(val):
            if pd.isna(val):
                return val
            s = str(val).strip().strip("'\"")
            try:
                return f"'{int(s):03d}"
            except:
                return f"'{s}"
        z['M49_Country_Code'] = z['M49_Country_Code'].apply(_format_m49_quote)
    
    z['country'] = z['country'].astype(str).str.strip()
    
    # WARNING: keep only the 198 valid countries
    z = z[z['country'].isin(universe.countries)]
    z['iso3'] = z['country'].map(universe.iso3_by_country)
    z = z.dropna(subset=['iso3'])
    z['iso3'] = z['iso3'].astype(str)
    z['year'] = pd.to_numeric(z['year'], errors='coerce')
    z = z.dropna(subset=['year'])
    z['year'] = z['year'].astype(int)
    z['element_norm'] = z['element'].astype(str).str.strip().str.lower()
    treated_label = 'manure management (manure treated, n content)'
    excreted_label = 'amount excreted in manure (n content)'
    z = z[z['element_norm'].isin([treated_label, excreted_label])]
    if z.empty:
        return pd.DataFrame(columns=columns)
    z['item_clean'] = z['item_raw'].astype(str).str.strip()
    
    # WARNING: use dict_v3 mapping and keep only items that map
    before_map = len(z)
    z['commodity'] = z['item_clean'].map(maps.stock_item_to_comm)
    if z['commodity'].isna().any():
        z.loc[z['commodity'].isna(), 'commodity'] = z.loc[z['commodity'].isna(), 'item_clean'].map(maps.slaughter_item_to_comm)
    z = z.dropna(subset=['commodity'])
    after_map = len(z)
    if before_map > after_map:
        print(f"[INFO] 过滤掉 {before_map - after_map} 行非dict_v3定义的manure Item")
    
    # WARNING: do not filter using universe.commodities!
    # manure使用Item_Stock_Map名称，universe.commodities使用Item_Production_Map名称
    # commodity已经通过maps.stock_item_to_comm/slaughter_item_to_comm映射验证过了
    
    # ✅ 保留M49_Country_Code列在pivot中
    index_cols = ['country','iso3','year','commodity']
    if 'M49_Country_Code' in z.columns:
        index_cols = ['M49_Country_Code'] + index_cols
    
    pivot = z.pivot_table(index=index_cols,
                          columns='element_norm',
                          values='value',
                          aggfunc='sum',
                          fill_value=np.nan)
    pivot = pivot.reset_index()
    if treated_label not in pivot.columns or excreted_label not in pivot.columns:
        return pd.DataFrame(columns=columns)
    pivot['manure_management_ratio'] = pivot[treated_label] / pivot[excreted_label].replace(0, np.nan)
    pivot = pivot.replace([np.inf, -np.inf], np.nan).dropna(subset=['manure_management_ratio'])
    
    # ✅ 选择列时包含M49_Country_Code
    out_cols = ['country','iso3','year','commodity','manure_management_ratio']
    if 'M49_Country_Code' in pivot.columns:
        out_cols = ['M49_Country_Code'] + out_cols
    out = pivot[out_cols].copy()
    out = _extend_future_years(out, 'manure_management_ratio', universe)
    return out

