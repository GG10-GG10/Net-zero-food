"""
简化版供需均衡模型 - 线性弹性 + 区域聚合 + 完整排放与约束
================================================================

与原版 S3_0_ds_emis_mc_full.py 的区别：
1. 移除 PWL 约束 → 直接用线性弹性方程（避免 PWL 导致的求解失败）
2. 区域聚合 → 194 国家 → 34 区域（基于 dict_v3 Region_market_agg）
3. 简化市场清算 → 仅全球出清

完整功能（与 S3_0_ds_emis_mc_full.py 一致）：
- Emissions as e0_by_proc * Qs with abatement decisions per process driven by MACC
- Land carbon price objective term for LULUCF processes
- Optional nutrition and land constraints
- Monte Carlo simulation support (LinearModelCache, apply_linear_sample_updates, run_linear_mc)
- Scenario support (tax_unit, feed_reduction, ruminant_intake_cap)
- Supply/demand slack variables for robustness

目标：
- 快速求解（秒级而非小时级）
- 保持完整的排放和减排功能
- 支持营养和土地约束
- 支持蒙特卡洛模拟和情景分析
"""

from __future__ import annotations
import math
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass

import gurobipy as gp
import pandas as pd
import numpy as np


# =============================================================================
# 辅助函数
# =============================================================================

def _is_lulucf_process(name: str) -> bool:
    """判断是否为 LULUCF（土地利用变化）相关排放过程"""
    if not name:
        return False
    s = str(name).lower()
    keys = ['forest', 'net forest', 'afforest', 'deforest', 'savanna', 
            'drained organic', 'organic soil', 'peat', 'lulucf', 'land use']
    return any(k in s for k in keys)


def _read_macc(macc_path: Optional[str]) -> pd.DataFrame:
    """读取边际减排成本曲线（MACC）数据"""
    if not macc_path:
        return pd.DataFrame()
    try:
        return pd.read_pickle(macc_path)
    except Exception:
        try:
            return pickle.load(open(macc_path, 'rb'))
        except Exception:
            return pd.DataFrame()


# =============================================================================
# 线性模型缓存类 - 用于蒙特卡洛模拟
# =============================================================================

@dataclass
class LinearModelCache:
    """
    线性区域模型缓存，用于蒙特卡洛模拟中的原地约束更新
    
    与 S3_0_ds_emis_mc_full.py 的 ModelCache 类似，但适用于线性模型：
    - 无需 PWL 变量 (lnQs, lnQd, lnPc, lnPnet)
    - 约束系数可直接更新（线性模型中系数是常数）
    
    用法：
    1. cache = build_linear_model_cache(nodes, commodities, years, ...)
    2. apply_linear_sample_updates(cache, pop_mult=..., yield_mult=..., e0_mult=...)
    3. cache.model.optimize()
    4. 读取结果: cache.Qs[key].X, cache.Qd[key].X, ...
    """
    # Gurobi 模型
    model: gp.Model
    
    # 决策变量
    Pc: Dict[Tuple[str, int], gp.Var]           # 全球价格 {(commodity, year): var}
    Qs: Dict[Tuple[str, str, int], gp.Var]      # 区域供给 {(region, commodity, year): var}
    Qd: Dict[Tuple[str, str, int], gp.Var]      # 区域需求 {(region, commodity, year): var}
    Eij: Dict[Tuple[str, str, int], gp.Var]     # 区域排放 {(region, commodity, year): var}
    Cij: Dict[Tuple[str, str, int], gp.Var]     # 区域减排成本 {(region, commodity, year): var}
    stock: Dict[Tuple[str, int], gp.Var]        # 库存 {(commodity, year): var}
    excess: Dict[Tuple[str, int], gp.Var]       # 过剩 {(commodity, year): var}
    shortage: Dict[Tuple[str, int], gp.Var]     # 短缺 {(commodity, year): var}
    
    # 约束引用（用于原地更新 RHS）
    constr_supply: Dict[Tuple[str, str, int], gp.Constr]   # 供给约束
    constr_demand: Dict[Tuple[str, str, int], gp.Constr]   # 需求约束
    constr_Edef: Dict[Tuple[str, str, int], gp.Constr]     # 排放定义约束
    nutri_constr: Dict[Tuple[str, int], gp.Constr]         # 营养约束
    land_constr: Dict[Tuple[str, int], gp.Constr]          # 土地约束
    rumi_intake_constr: Dict[Tuple[str, int], gp.Constr]   # 反刍动物需求上限约束 (Phase 2)
    
    # MACC 相关
    abatement_vars: Dict[Tuple[str, str, int, str, int], gp.Var]      # 减排变量
    abatement_caps: Dict[Tuple[str, str, int, str, int], gp.Constr]   # 减排上限约束
    abatement_costs: Dict[Tuple[str, str, int, str, int], float]      # 减排边际成本
    proc_cap_basecoeff: Dict[Tuple[str, str, int, str, int], float]   # MACC 基准系数
    
    # 校准参数（用于 MC 更新）
    alpha_s: Dict[Tuple[str, str, int], float]  # 供给常数项
    alpha_d: Dict[Tuple[str, str, int], float]  # 需求常数项
    eps_s: Dict[Tuple[str, str, int], float]    # 供给价格弹性
    eps_d: Dict[Tuple[str, str, int], float]    # 需求价格弹性
    eps_pop: Dict[Tuple[str, str, int], float]  # 人口弹性
    eps_inc: Dict[Tuple[str, str, int], float]  # 收入弹性
    eta_y: Dict[Tuple[str, str, int], float]    # 产率弹性
    eta_temp: Dict[Tuple[str, str, int], float] # 温度弹性
    
    # 基准值（用于 MC 更新计算）
    Q0: Dict[Tuple[str, str, int], float]       # 基期供给
    D0: Dict[Tuple[str, str, int], float]       # 基期需求
    P0: Dict[Tuple[str, str, int], float]       # 基期价格
    Ymult0: Dict[Tuple[str, str, int], float]   # 基期产率乘数
    Tmult0: Dict[Tuple[str, str, int], float]   # 基期温度乘数
    pop_base: Dict[Tuple[str, str, int], float] # 基期人口
    inc_base: Dict[Tuple[str, str, int], float] # 基期收入
    
    # 排放强度
    e0_by_region: Dict[Tuple[str, str, int], Dict[str, float]]  # 排放强度 {key: {process: e0}}
    
    # 元数据
    regions: List[str]
    commodities: List[str]
    years: List[int]
    idx: Dict[Tuple[str, str, int], Dict]  # 原始聚合数据索引


# =============================================================================
# 区域定义 - 从 dict_v3 Region_market_agg 加载
# =============================================================================

# 缓存：{country_name: region} 和 {m49_code: region}
_REGION_BY_COUNTRY: Dict[str, str] = {}
_REGION_BY_M49: Dict[str, str] = {}
_REGIONS_LOADED = False

def _load_region_mapping_from_dict_v3(dict_v3_path: Optional[str] = None) -> None:
    """
    从 dict_v3.xlsx 的 region 表加载 Region_market_agg 映射
    
    34个市场区域：
    - 14个大国/独立区域：Argentina, Australia, Bangladesh, Brazil, Canada, China, 
      Ethiopia, India, Indonesia, Mexico, New Zealand, Nigeria, Pakistan, Russia, 
      Tanzania, Turkey, U.S.
    - 20个区域聚合：AFR-Central, AFR-East, AFR-Southern, AFR-West, 
      AMR-Central-Caribbean, AMR-South, ASIA-Central, ASIA-East, ASIA-South, 
      ASIA-Southeast, EUR-Atlantic, EUR-Boreal, EUR-Continental, EUR-Mediterranean,
      MENA-Gulf, MENA-Mediterranean, OCEA-Pacific
    """
    global _REGION_BY_COUNTRY, _REGION_BY_M49, _REGIONS_LOADED
    
    if _REGIONS_LOADED:
        return
    
    # 确定 dict_v3 路径
    if dict_v3_path is None:
        try:
            from config_paths import get_src_base
            dict_v3_path = str(Path(get_src_base()) / 'dict_v3.xlsx')
        except ImportError:
            # 硬编码备用路径
            dict_v3_path = r'G:\我的云端硬盘\Work\Net-zero food\Code\src\dict_v3.xlsx'
    
    logger = logging.getLogger(__name__)
    
    try:
        df = pd.read_excel(dict_v3_path, sheet_name='region')
        
        # 使用 Region_label_new 作为国家名（与 universe.countries 一致）
        # M49_Country_Code 用于 M49 匹配
        # Region_market_agg 是目标区域
        df = df[['Region_label_new', 'M49_Country_Code', 'Region_market_agg']].dropna()
        df = df[df['Region_market_agg'] != 'no']
        df = df[df['Region_label_new'] != 'no']  # 也排除无效国家
        
        for _, row in df.iterrows():
            country = str(row['Region_label_new']).strip()
            m49 = str(row['M49_Country_Code']).strip().lstrip("'")
            region = str(row['Region_market_agg']).strip()
            
            _REGION_BY_COUNTRY[country] = region
            _REGION_BY_M49[m49] = region
        
        _REGIONS_LOADED = True
        logger.info(f"[REGION] 已从 dict_v3 加载 {len(_REGION_BY_COUNTRY)} 个国家的市场区域映射")
        logger.info(f"[REGION] 共 {len(set(_REGION_BY_COUNTRY.values()))} 个市场区域")
        
    except Exception as e:
        logger.warning(f"[REGION] 无法加载 dict_v3 区域映射: {e}，使用默认区域")
        _REGIONS_LOADED = True  # 防止重复尝试


def get_region(country: str, m49: Optional[str] = None, dict_v3_path: Optional[str] = None) -> str:
    """
    获取国家对应的市场区域
    
    优先使用 M49 代码匹配，其次使用国家名匹配
    """
    global _REGIONS_LOADED
    
    if not _REGIONS_LOADED:
        _load_region_mapping_from_dict_v3(dict_v3_path)
    
    # 优先用 M49 匹配
    if m49:
        m49_clean = str(m49).strip().lstrip("'")
        if m49_clean in _REGION_BY_M49:
            return _REGION_BY_M49[m49_clean]
    
    # 用国家名匹配
    country_clean = str(country).strip()
    if country_clean in _REGION_BY_COUNTRY:
        return _REGION_BY_COUNTRY[country_clean]
    
    # 默认区域（不应该发生，如果 dict_v3 包含所有国家）
    return 'OTHER'


def get_all_regions(dict_v3_path: Optional[str] = None) -> List[str]:
    """获取所有市场区域列表（34个）"""
    if not _REGIONS_LOADED:
        _load_region_mapping_from_dict_v3(dict_v3_path)
    
    regions = set(_REGION_BY_COUNTRY.values())
    return sorted(regions)


def reset_region_cache() -> None:
    """重置区域缓存（用于测试）"""
    global _REGION_BY_COUNTRY, _REGION_BY_M49, _REGIONS_LOADED
    _REGION_BY_COUNTRY = {}
    _REGION_BY_M49 = {}
    _REGIONS_LOADED = False


# =============================================================================
# 历史最大产量加载与聚合
# =============================================================================

def load_historical_max_production(
    csv_path: str,
    dict_v3_path: Optional[str] = None,
    item_column: str = 'Item',
    production_column: str = 'max_production_t',
    m49_column: str = 'M49_Country_Code',
) -> Dict[Tuple[str, str], float]:
    """
    从 S0_19_historical_max_production.csv 加载历史最大产量，并聚合到区域级
    
    功能：
    1. 读取CSV（M49_Country_Code, Item, max_production_t）
    2. 将M49映射到34个市场区域（Region_market_agg）
    3. 对同一区域-商品的产量求和
    
    参数:
        csv_path: S0_19_historical_max_production.csv 的路径
        dict_v3_path: dict_v3.xlsx 路径（用于区域映射）
        item_column: Item 列名（S0_19 输出为 'Item'，即 Item_Production_Map 值）
        production_column: 产量列名
        m49_column: M49 代码列名
    
    返回:
        Dict[(region, commodity), max_production_t] - 区域级历史最大产量
    
    注意:
        - S0_19 输出的 Item 是 Item_Production_Map 格式（与 universe.commodities 一致）
        - 返回的 commodity 键与模型中使用的商品名一致
    """
    logger = logging.getLogger(__name__)
    
    # 确保区域映射已加载
    if not _REGIONS_LOADED:
        _load_region_mapping_from_dict_v3(dict_v3_path)
    
    path = Path(csv_path)
    if not path.exists():
        logger.warning(f"[HIST_MAX] 历史最大产量文件不存在: {csv_path}")
        return {}
    
    try:
        df = pd.read_csv(path, dtype={m49_column: str})
    except Exception as e:
        logger.error(f"[HIST_MAX] 无法读取 CSV: {e}")
        return {}
    
    # 检查必要列
    required = [m49_column, item_column, production_column]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error(f"[HIST_MAX] CSV 缺少必要列: {missing}")
        return {}
    
    # 规范化 M49 代码
    def _norm_m49(val):
        if pd.isna(val):
            return None
        s = str(val).strip().lstrip("'\"")
        try:
            return str(int(s)) if s.isdigit() else s
        except:
            return s
    
    df[m49_column] = df[m49_column].apply(_norm_m49)
    df = df[df[m49_column].notna()]
    
    # 映射 M49 → 区域
    df['region'] = df[m49_column].apply(lambda m: _REGION_BY_M49.get(m, 'OTHER'))
    df = df[df['region'] != 'OTHER']  # 排除无法映射的国家
    
    # 确保产量为数值
    df[production_column] = pd.to_numeric(df[production_column], errors='coerce').fillna(0.0)
    df = df[df[production_column] > 0]  # 排除零产量
    
    # 聚合到区域级（同区域-商品求和）
    agg = df.groupby(['region', item_column])[production_column].sum().reset_index()
    
    # 构建返回字典
    result: Dict[Tuple[str, str], float] = {}
    for _, row in agg.iterrows():
        key = (str(row['region']), str(row[item_column]))
        result[key] = float(row[production_column])
    
    logger.info(f"[HIST_MAX] 已加载 {len(result)} 个 (区域, 商品) 历史最大产量")
    logger.info(f"[HIST_MAX] 覆盖区域: {len(set(k[0] for k in result))} 个")
    logger.info(f"[HIST_MAX] 覆盖商品: {len(set(k[1] for k in result))} 种")
    
    return result


def load_grassland_coefficients(
    feed_need_xlsx: str,
    grass_ratio_xlsx: str,
    pasture_yield_xlsx: str,
    dict_v3_path: str,
    years: List[int]
) -> Dict[Tuple[str, str], float]:
    """
    计算livestock到grassland的转换系数（区域级）
    
    公式：grassland_coef = (kg_DM_per_head / stock_to_production) × grass_ratio / pasture_yield
    
    简化：假设 1头livestock = 1吨产量（近似）
    则：grassland_coef ≈ kg_DM_per_head × grass_ratio / pasture_yield_kg_per_ha
    
    单位：ha/ton (产量)
    
    Args:
        feed_need_xlsx: 饲料需求文件路径
        grass_ratio_xlsx: 草地饲料占比文件路径  
        pasture_yield_xlsx: 草地产率文件路径
        dict_v3_path: dict_v3文件路径（用于国家→区域映射）
        years: 年份列表
        
    Returns:
        {(region, livestock_commodity): ha_per_ton} 字典
    """
    logger = logging.getLogger(__name__)
    
    # 加载参数数据（使用S3_2中的加载函数）
    try:
        from S3_2_feed_demand import _load_total_dm_per_head, _load_grass_ratio, _load_pasture_yield
        
        dm_per_head = _load_total_dm_per_head(feed_need_xlsx, years)
        grass_ratio_df = _load_grass_ratio(grass_ratio_xlsx)
        pasture_yield_df = _load_pasture_yield(pasture_yield_xlsx)
        
        if dm_per_head.empty or grass_ratio_df.empty or pasture_yield_df.empty:
            logger.warning("[GRASSLAND_COEF] 参数数据加载失败，返回空系数")
            return {}
        
        # 计算国家级系数
        # dm_per_head: [species, m49_code, year, kg_dm_per_head]
        # grass_ratio: [species, m49_code, grass_ratio]
        # pasture_yield: [m49_code, pasture_yield_kg_per_ha]
        
        # Merge参数
        coef_df = dm_per_head.merge(grass_ratio_df, on=['species', 'm49_code'], how='left')
        coef_df = coef_df.merge(pasture_yield_df, on='m49_code', how='left')
        
        # 填充缺失值
        coef_df['grass_ratio'] = coef_df['grass_ratio'].fillna(0.0).clip(0, 1)
        coef_df['pasture_yield_kg_per_ha'] = coef_df['pasture_yield_kg_per_ha'].fillna(3000.0)  # 默认3000 kg/ha
        
        # 计算系数：ha_per_head = (kg_DM_per_head × grass_ratio) / pasture_yield
        coef_df['ha_per_head'] = (
            coef_df['kg_dm_per_head'] * coef_df['grass_ratio'] / 
            coef_df['pasture_yield_kg_per_ha'].clip(lower=1e-6)
        )
        
        # 聚合到区域级（按产量加权，但这里简化为直接平均）
        # 需要m49_code → country → region映射
        from S2_0_load_data import DataPaths
        paths = DataPaths()
        dict_v3 = pd.read_excel(dict_v3_path, sheet_name='region')
        m49_to_region = {}
        for _, row in dict_v3.iterrows():
            m49 = str(row.get('M49_Country_Code', '')).strip().lstrip("'\"").zfill(3)
            region_market = str(row.get('Region_market_agg', '')).strip()
            if m49 and region_market and region_market != 'no':
                m49_to_region[m49] = region_market
        
        coef_df['region'] = coef_df['m49_code'].map(m49_to_region)
        coef_df = coef_df.dropna(subset=['region'])
        
        # 按区域-species聚合（取平均值）
        regional_coef = coef_df.groupby(['region', 'species'])['ha_per_head'].mean().reset_index()
        
        # 构建返回字典 {(region, species): ha_per_head}
        result = {}
        for _, row in regional_coef.iterrows():
            key = (str(row['region']), str(row['species']))
            result[key] = float(row['ha_per_head'])
        
        logger.info(f"[GRASSLAND_COEF] 已计算 {len(result)} 个 (区域, livestock) 草地系数")
        logger.info(f"[GRASSLAND_COEF] 覆盖区域: {len(set(k[0] for k in result))} 个")
        logger.info(f"[GRASSLAND_COEF] 覆盖livestock: {len(set(k[1] for k in result))} 种")
        
        # 打印几个示例
        sample_items = ['Cattle', 'Pigs', 'Poultry', 'Sheep', 'Goats']
        sample_regions = ['China', 'U.S.', 'India', 'Brazil']
        logger.info(f"[GRASSLAND_COEF] 系数示例（ha/head）:")
        for region in sample_regions:
            for item in sample_items:
                coef = result.get((region, item))
                if coef:
                    logger.info(f"  {region:20s} | {item:15s}: {coef:.6f} ha/head")
        
        return result
        
    except Exception as e:
        logger.error(f"[GRASSLAND_COEF] 计算grassland系数失败: {e}")
        import traceback
        traceback.print_exc()
        return {}


# =============================================================================
# 数据聚合
# =============================================================================

# Livestock商品列表（需要grassland的商品）
LIVESTOCK_COMMODITIES = {
    'Cattle', 'Buffaloes', 'Sheep', 'Goats', 'Pigs', 'Chickens', 
    'Ducks', 'Geese', 'Turkeys', 'Horses', 'Asses', 'Mules',
    'Milk', 'Eggs', 'Meat'
}

def aggregate_nodes_to_regions(
    nodes: List[Any], 
    dict_v3_path: Optional[str] = None,
    population_by_country_year: Optional[Dict[Tuple[str, int], float]] = None,
    income_mult_by_country_year: Optional[Dict[Tuple[str, int], float]] = None,
    hist_end_year: int = 2020,
) -> pd.DataFrame:
    """
    将国家级节点聚合到区域级
    
    聚合规则：
    - Q0, D0: 加总
    - P0: 加权平均（按 Q0 权重）
    - 弹性: 加权平均（按 Q0/D0 权重）
    - Ymult, Tmult: 加权平均（按 Q0 权重）
    - 人口、收入: 区域加总
    
    使用 dict_v3 Region_market_agg 的 34 个市场区域
    
    完整弹性列表（与 S3_0_ds_emis_mc_full.py 一致）：
    - 供给侧: eps_supply, eps_supply_yield (η_y), eps_supply_temp (η_temp)
    - 需求侧: eps_demand, eps_pop_demand, eps_income_demand
    - 交叉价格: epsS_row, epsD_row (暂不聚合，区域模型简化处理)
    """
    # 确保区域映射已加载
    if not _REGIONS_LOADED:
        _load_region_mapping_from_dict_v3(dict_v3_path)
    
    records = []
    for n in nodes:
        m49 = getattr(n, 'm49', None) or getattr(n, 'M49_Country_Code', None)
        region = get_region(n.country, m49=m49, dict_v3_path=dict_v3_path)
        
        # 获取人口和收入数据
        pop_base = 1.0
        pop_t = 1.0
        inc_base = 1.0
        inc_t = 1.0
        if population_by_country_year:
            pop_base = float(population_by_country_year.get((n.country, hist_end_year), 1.0) or 1.0)
            pop_t = float(population_by_country_year.get((n.country, n.year), pop_base) or pop_base)
        if income_mult_by_country_year:
            inc_base = float(income_mult_by_country_year.get((n.country, hist_end_year), 1.0) or 1.0)
            inc_t = float(income_mult_by_country_year.get((n.country, n.year), inc_base) or inc_base)
        
        # 获取yield0（历史平均产率，来自meta）
        meta = getattr(n, 'meta', {}) or {}
        yield0 = float(meta.get('yield0', 0.0) or 0.0)
        
        # 获取grassland系数（仅livestock商品，用于方案A）
        # grassland_coef = ha_per_ton (产量吨数转换为草地公顷)
        # ✅ 修复：直接检查meta中是否有grassland_coef，而不是用旧的LIVESTOCK_COMMODITIES判断
        # 理由：LIVESTOCK_COMMODITIES={'Cattle', ...}不匹配实际commodity='Cattle, dairy'
        grassland_coef = float(meta.get('grassland_coef', 0.0) or 0.0)
        
        records.append({
            'region': region,
            'country': n.country,
            'commodity': n.commodity,
            'year': n.year,
            # 基础量
            'Q0': getattr(n, 'Q0', 0.0) or 0.0,
            'D0': getattr(n, 'D0', 0.0) or 0.0,
            'P0': getattr(n, 'P0', 1.0) or 1.0,
            # 产率（用于土地约束）
            'yield0': yield0,
            'grassland_coef': grassland_coef,
            # 供给弹性
            'eps_supply': getattr(n, 'eps_supply', 0.0) or 0.0,
            'eps_supply_yield': getattr(n, 'eps_supply_yield', 0.0) or 0.0,  # η_y
            'eps_supply_temp': getattr(n, 'eps_supply_temp', 0.0) or 0.0,    # η_temp
            # 供给因子
            'Ymult': getattr(n, 'Ymult', 1.0) or 1.0,  # 产率乘数
            'Tmult': getattr(n, 'Tmult', 1.0) or 1.0,  # 温度乘数
            # 需求弹性
            'eps_demand': getattr(n, 'eps_demand', 0.0) or 0.0,
            'eps_pop_demand': getattr(n, 'eps_pop_demand', 0.0) or 0.0,
            'eps_income_demand': getattr(n, 'eps_income_demand', 0.0) or 0.0,
            # 交叉价格弹性（保留原始字典，聚合时按 Q0/D0 加权）
            'epsS_row': dict(getattr(n, 'epsS_row', {}) or {}),  # 供给侧交叉价格弹性
            'epsD_row': dict(getattr(n, 'epsD_row', {}) or {}),  # 需求侧交叉价格弹性
            # 人口和收入（用于需求方程）
            'pop_base': pop_base,
            'pop_t': pop_t,
            'inc_base': inc_base,
            'inc_t': inc_t,
        })
    
    df = pd.DataFrame(records)
    
    # 按区域-商品-年份聚合
    agg_funcs = {
        # 数量加总
        'Q0': 'sum',
        'D0': 'sum',
        # 价格加权平均
        'P0': lambda x: np.average(x, weights=df.loc[x.index, 'Q0'].clip(lower=1e-6)),
        # 产率（按 Q0 加权平均）
        'yield0': lambda x: np.average(x, weights=df.loc[x.index, 'Q0'].clip(lower=1e-6)) if x.sum() > 0 else 0.0,
        # Grassland系数（按 Q0 加权平均，仅livestock商品有值）
        'grassland_coef': lambda x: np.average(x, weights=df.loc[x.index, 'Q0'].clip(lower=1e-6)) if x.sum() > 0 else 0.0,
        # 供给弹性（按 Q0 加权）
        'eps_supply': lambda x: np.average(x, weights=df.loc[x.index, 'Q0'].clip(lower=1e-6)),
        'eps_supply_yield': lambda x: np.average(x, weights=df.loc[x.index, 'Q0'].clip(lower=1e-6)),
        'eps_supply_temp': lambda x: np.average(x, weights=df.loc[x.index, 'Q0'].clip(lower=1e-6)),
        # 供给因子（按 Q0 加权）
        'Ymult': lambda x: np.average(x, weights=df.loc[x.index, 'Q0'].clip(lower=1e-6)),
        'Tmult': lambda x: np.average(x, weights=df.loc[x.index, 'Q0'].clip(lower=1e-6)),
        # 需求弹性（按 D0 加权）
        'eps_demand': lambda x: np.average(x, weights=df.loc[x.index, 'D0'].clip(lower=1e-6)),
        'eps_pop_demand': lambda x: np.average(x, weights=df.loc[x.index, 'D0'].clip(lower=1e-6)),
        'eps_income_demand': lambda x: np.average(x, weights=df.loc[x.index, 'D0'].clip(lower=1e-6)),
        # 人口和收入（加总）
        'pop_base': 'sum',
        'pop_t': 'sum',
        'inc_base': lambda x: np.average(x, weights=df.loc[x.index, 'D0'].clip(lower=1e-6)),  # 收入用加权平均
        'inc_t': lambda x: np.average(x, weights=df.loc[x.index, 'D0'].clip(lower=1e-6)),
    }
    
    regional_df = df.groupby(['region', 'commodity', 'year']).agg(agg_funcs).reset_index()
    
    # 单独聚合交叉价格弹性（需要按 Q0/D0 加权平均每个商品的弹性值）
    cross_price_data: Dict[Tuple[str, str, int], Dict[str, Dict[str, float]]] = {}
    for _, row in df.iterrows():
        key = (row['region'], row['commodity'], row['year'])
        if key not in cross_price_data:
            cross_price_data[key] = {'epsS_sum': {}, 'epsD_sum': {}, 'Q0_total': 0.0, 'D0_total': 0.0}
        
        Q0 = max(1e-9, float(row['Q0']))
        D0 = max(1e-9, float(row['D0']))
        cross_price_data[key]['Q0_total'] += Q0
        cross_price_data[key]['D0_total'] += D0
        
        # 供给侧交叉价格弹性（按 Q0 加权）
        for comm, eps_val in row.get('epsS_row', {}).items():
            if comm not in cross_price_data[key]['epsS_sum']:
                cross_price_data[key]['epsS_sum'][comm] = 0.0
            cross_price_data[key]['epsS_sum'][comm] += float(eps_val) * Q0
        
        # 需求侧交叉价格弹性（按 D0 加权）
        for comm, eps_val in row.get('epsD_row', {}).items():
            if comm not in cross_price_data[key]['epsD_sum']:
                cross_price_data[key]['epsD_sum'][comm] = 0.0
            cross_price_data[key]['epsD_sum'][comm] += float(eps_val) * D0
    
    # 计算加权平均并添加到 regional_df
    epsS_row_col = []
    epsD_row_col = []
    for _, row in regional_df.iterrows():
        key = (row['region'], row['commodity'], row['year'])
        cpd = cross_price_data.get(key, {})
        
        Q0_total = max(1e-9, cpd.get('Q0_total', 1.0))
        D0_total = max(1e-9, cpd.get('D0_total', 1.0))
        
        epsS_avg = {c: v / Q0_total for c, v in cpd.get('epsS_sum', {}).items()}
        epsD_avg = {c: v / D0_total for c, v in cpd.get('epsD_sum', {}).items()}
        
        epsS_row_col.append(epsS_avg)
        epsD_row_col.append(epsD_avg)
    
    regional_df['epsS_row'] = epsS_row_col
    regional_df['epsD_row'] = epsD_row_col
    
    return regional_df


def aggregate_emissions_to_regions(
    nodes: List[Any], 
    dict_v3_path: Optional[str] = None
) -> Dict[Tuple[str, str, int], Dict[str, float]]:
    """
    聚合国家级排放强度 e0_by_proc 到区域级
    
    聚合规则：按 Q0 加权平均
    
    返回: {(region, commodity, year): {process: e0_intensity}}
    """
    # 确保区域映射已加载
    if not _REGIONS_LOADED:
        _load_region_mapping_from_dict_v3(dict_v3_path)
    
    # 收集 (region, commodity, year) -> [(e0_by_proc, Q0), ...]
    data_by_key: Dict[Tuple[str, str, int], List[Tuple[Dict[str, float], float]]] = {}
    
    for n in nodes:
        m49 = getattr(n, 'm49', None) or getattr(n, 'M49_Country_Code', None)
        region = get_region(n.country, m49=m49, dict_v3_path=dict_v3_path)
        key = (region, n.commodity, n.year)
        
        e0_map = getattr(n, 'e0_by_proc', {}) or {}
        Q0 = float(getattr(n, 'Q0', 0.0) or 0.0)
        
        if key not in data_by_key:
            data_by_key[key] = []
        data_by_key[key].append((dict(e0_map), Q0))
    
    # 加权平均
    result: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    for key, entries in data_by_key.items():
        all_procs = set()
        for e0_map, _ in entries:
            all_procs.update(e0_map.keys())
        
        total_Q0 = sum(Q0 for _, Q0 in entries)
        if total_Q0 < 1e-9:
            total_Q0 = len(entries)  # 均分
        
        agg_e0: Dict[str, float] = {}
        for proc in all_procs:
            weighted_sum = sum(e0_map.get(proc, 0.0) * Q0 for e0_map, Q0 in entries)
            agg_e0[proc] = weighted_sum / total_Q0 if total_Q0 > 0 else 0.0
        
        result[key] = agg_e0
    
    return result


# =============================================================================
# 线性弹性模型（完整版：含排放、MACC、约束）
# =============================================================================

def build_linear_regional_model(
    nodes: List[Any],
    commodities: List[str],
    years: List[int],
    price_bounds: Tuple[float, float] = (1e-6, 1e6),  # 价格下界1e-6已足够小
    qty_bounds: Tuple[float, float] = (1e-6, 1e12),
    dict_v3_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    gurobi_log_path: Optional[str] = None,
    # ===== 人口与收入 =====
    population_by_country_year: Optional[Dict[Tuple[str, int], float]] = None,
    income_mult_by_country_year: Optional[Dict[Tuple[str, int], float]] = None,
    # ===== 排放与减排参数 =====
    macc_path: Optional[str] = None,
    land_carbon_price_by_year: Optional[Dict[int, float]] = None,
    # ===== 约束参数 =====
    nutrition_rhs: Optional[Dict[Tuple[str, int], float]] = None,
    nutrient_per_unit_by_comm: Optional[Dict[str, float]] = None,
    land_area_limits: Optional[Dict[Tuple[str, int], float]] = None,
    grass_area_by_region_year: Optional[Dict[Tuple[str, int], float]] = None,  # ✅ 草地面积 {(region, year): ha}
    forest_area_by_region_year: Optional[Dict[Tuple[str, int], float]] = None,  # ✅ 森林面积 {(region, year): ha}
    yield_t_per_ha_default: float = 3.0,
    grassland_method: str = 'dynamic',  # ✅ 'dynamic' (方案A: 优化变量) or 'static' (方案B: 迭代)
    # ===== 增长约束 =====
    max_growth_rate_per_period: Optional[float] = None,
    max_decline_rate_per_period: Optional[float] = None,
    hist_end_year: int = 2020,
    hist_max_production: Optional[Dict[Tuple[str, str], float]] = None,
    # ===== 情景参数 (Phase 2) =====
    tax_unit_adder: Optional[Dict[Tuple[str, str, int], float]] = None,
    feed_reduction_by: Optional[Dict[Tuple[str, str, int], float]] = None,
    ruminant_intake_cap: Optional[Dict[Tuple[str, int], float]] = None,
    ruminant_commodities: Optional[List[str]] = None,
    # ===== 市场失衡限制 =====
    max_slack_rate: Optional[float] = None,  # 短缺/过剩上限：占总供给的比例（例如0.01=1%，None=不限制）
    # ===== 单位成本法参数 =====
    cost_calculation_method: str = 'MACC',  # 'MACC' (默认) 或 'unit_cost'
    unit_cost_data: Optional[Dict[Tuple[str, str], float]] = None,  # {(region/country, process): USD/tCO2e}
    process_cost_mapping: Optional[Dict[str, str]] = None,  # {pipeline_process: cost_process_name}
    baseline_scenario_result: Optional[Dict[str, Any]] = None,  # BASE情景结果 {'Qs': {(region, comm, year): value}}
) -> gp.Model:
    """
    构建线性弹性区域模型（完整版）
    
    特点：
    1. 线性弹性方程（无 log 变换，无 PWL）→ 快速求解
    2. 区域聚合（34 区域，基于 dict_v3 Region_market_agg）
    3. 全球市场出清
    
    完整弹性（与 S3_0_ds_emis_mc_full.py 一致）：
    - 供给侧: ε_supply (价格), η_yield (产率), η_temp (温度)
    - 需求侧: ε_demand (价格), ε_pop (人口), ε_income (收入)
    
    完整功能：
    - Emissions as e0_by_proc * Qs with abatement decisions per process driven by MACC
    - Land carbon price objective term for LULUCF processes
    - Optional nutrition and land constraints
    - Scenario support: tax_unit, feed_reduction, ruminant_intake_cap
    - Monte Carlo simulation support via LinearModelCache
    
    参数:
        nodes: 国家级节点列表
        commodities: 商品列表
        years: 年份列表
        price_bounds: 价格边界 (min, max)
        qty_bounds: 数量边界 (min, max)
        dict_v3_path: dict_v3.xlsx 路径
        macc_path: MACC 数据路径（pickle 文件）
        land_carbon_price_by_year: 土地碳价 {year: price_per_tCO2e}
        nutrition_rhs: 营养约束右端项 {(region, year): min_calories}
        nutrient_per_unit_by_comm: 每单位商品的营养值 {commodity: value}
        land_area_limits: 土地面积上限 {(region, year): max_ha}
        yield_t_per_ha_default: 默认产量（吨/公顷）
        grassland_method: 草地处理方式
            - 'dynamic' (方案A, 默认): 草地作为优化变量表达式，grassland_ha = Σ coef × Qs
            - 'static' (方案B): 草地作为外生参数，需迭代更新直到收敛
        max_growth_rate_per_period: 最大增长率
        max_decline_rate_per_period: 最大下降率
        hist_end_year: 历史期结束年份
        hist_max_production: 历史最大产量锚定 {(region, commodity): max_t}，限制未来产量
        tax_unit_adder: 单位税 {(region, commodity, year): $/t}，影响供给侧净价格
        feed_reduction_by: 饲料减少比例 {(region, commodity, year): fraction in [0,1]}
        ruminant_intake_cap: 反刍动物需求上限 {(region, year): cap_in_t}
        ruminant_commodities: 反刍动物商品列表（默认：牛羊肉）
    """
    logger = logging.getLogger(__name__)
    
    # 聚合数据（传入人口和收入数据）
    regional_df = aggregate_nodes_to_regions(
        nodes, 
        dict_v3_path=dict_v3_path,
        population_by_country_year=population_by_country_year,
        income_mult_by_country_year=income_mult_by_country_year,
        hist_end_year=hist_end_year,
    )
    
    # 过滤商品：只保留commodities列表中的商品
    # 这会移除非商品项目（排放项、土地类别等）
    original_count = len(regional_df)
    regional_df = regional_df[regional_df['commodity'].isin(commodities)]
    filtered_count = original_count - len(regional_df)
    if filtered_count > 0:
        logger.info(f"[LINEAR] 已过滤 {filtered_count} 行非商品数据")
    
    # 进一步过滤极小Q0的商品（按区域-商品组合）
    # 策略：如果某区域-商品组合在基期年份（hist_end_year）的Q0和D0都很小，
    # 则过滤该区域-商品的所有年份数据
    MIN_Q0_THRESHOLD = 1e-3  # 最小产量阈值（kt/year）
    
    # 提取基期年份数据用于判断
    base_year_df = regional_df[regional_df['year'] == hist_end_year].copy()
    base_year_df['tiny_q0'] = (base_year_df['Q0'] < MIN_Q0_THRESHOLD) & (base_year_df['D0'] < MIN_Q0_THRESHOLD)
    
    # 标记需要过滤的区域-商品组合
    tiny_pairs = base_year_df[base_year_df['tiny_q0']][['region', 'commodity']].drop_duplicates()
    
    if len(tiny_pairs) > 0:
        logger.info(f"[LINEAR] 发现 {len(tiny_pairs)} 个区域-商品组合在基期({hist_end_year})的Q0和D0都 < {MIN_Q0_THRESHOLD:.0e}")
        
        # 创建要过滤的(region, commodity)集合
        tiny_set = set(tiny_pairs.apply(tuple, axis=1))
        
        # 创建过滤掩码
        filter_mask = regional_df.apply(
            lambda row: (row['region'], row['commodity']) in tiny_set,
            axis=1
        )
        
        filtered_rows = filter_mask.sum()
        logger.info(f"[LINEAR] 将过滤 {filtered_rows} 行数据（跨所有年份）")
        
        # 记录被过滤的商品
        filtered_comms = tiny_pairs['commodity'].unique()
        logger.info(f"[LINEAR] 涉及商品: {', '.join(filtered_comms[:10])}")
        if len(filtered_comms) > 10:
            logger.info(f"[LINEAR] ... 以及其他 {len(filtered_comms)-10} 个")
        
        # 执行过滤
        regional_df = regional_df[~filter_mask]
    
    regions = regional_df['region'].unique().tolist()
    actual_commodities = regional_df['commodity'].unique().tolist()
    
    # 聚合排放强度
    e0_by_region = aggregate_emissions_to_regions(nodes, dict_v3_path=dict_v3_path)
    
    # 加载 MACC 数据
    macc_df = _read_macc(macc_path)
    has_macc = not macc_df.empty
    
    logger.info(f"[LINEAR] 区域数: {len(regions)}, 商品数: {len(actual_commodities)}, 年份数: {len(years)}")
    logger.info(f"[LINEAR] 参与模拟的区域: {regions}")
    logger.info(f"[LINEAR] 参与模拟的商品 ({len(actual_commodities)}个): {', '.join(sorted(actual_commodities))}")
    logger.info(f"[LINEAR] MACC 数据: {'已加载' if has_macc else '无'}")
    
    # 统计未来年份参与模拟的区域-商品组合数
    future_df = regional_df[regional_df['year'] > hist_end_year]
    if len(future_df) > 0:
        future_pairs = future_df.groupby(['region', 'commodity']).size().reset_index(name='count')
        logger.info(f"[LINEAR] 未来年份参与模拟: {len(future_pairs)} 个区域-商品组合")
        logger.info(f"[LINEAR] 覆盖 {future_df['year'].nunique()} 个未来年份: {sorted(future_df['year'].unique().tolist())}")
    
    # 创建模型
    m = gp.Model("nzf_linear_regional_full")
    m.setParam('OutputFlag', 1)
    m.setParam('LogToConsole', 1)
    # 设置Gurobi日志文件
    if gurobi_log_path:
        m.Params.LogFile = str(gurobi_log_path)
    # 提高数值稳定性（避免大系数范围导致的假性不可行）
    m.setParam('NumericFocus', 3)  # 0=auto, 1=moderate, 2=aggressive, 3=very aggressive
    m.setParam('ScaleFlag', 2)  # 自动缩放系数矩阵以改善数值范围
    m.setParam('FeasibilityTol', 1e-6)  # 放松可行性容差（默认 1e-6，可以试 1e-5 或 1e-4）
    m.setParam('OptimalityTol', 1e-6)   # 放松最优性容差
    
    Pmin, Pmax = price_bounds
    Qmin, Qmax = qty_bounds
    
    # 索引
    idx = {}
    for _, row in regional_df.iterrows():
        key = (row['region'], row['commodity'], row['year'])
        idx[key] = row.to_dict()
    
    # ==========================================================================
    # 变量
    # ==========================================================================
    
    # 全球商品价格 Pc[j,t]
    Pc: Dict[Tuple[str, int], gp.Var] = {}
    for j in commodities:
        for t in years:
            Pc[j, t] = m.addVar(lb=Pmin, ub=Pmax, name=f"Pc[{j},{t}]")
    
    # 区域供给 Qs[r,j,t] 和需求 Qd[r,j,t]
    Qs: Dict[Tuple[str, str, int], gp.Var] = {}
    Qd: Dict[Tuple[str, str, int], gp.Var] = {}
    for r in regions:
        for j in commodities:
            for t in years:
                key = (r, j, t)
                if key in idx:
                    # 历史年份：变量下界必须允许等于实际Q0/D0（因为会被固定为这些值）
                    # 未来年份：使用基期Q0/D0设置较大下界以改善数值范围
                    if t <= hist_end_year:
                        # 历史年份：允许变量取实际值（但设置合理上界）
                        Q0_val = idx[key]['Q0']
                        D0_val = idx[key]['D0']
                        # 下界：允许略小于实际值（容许舍入误差），但不低于1e-9
                        Q0_lb = max(1e-9, Q0_val * 0.1)  
                        D0_lb = max(1e-9, D0_val * 0.1)
                        # 上界：确保能容纳Q0_val且允许适度增长（至少是Q0_val的1.1倍）
                        Q0_ub = max(Q0_val * 1.1, Q0_val * 10, 1e-3)  # 确保 ub >= Q0_val
                        D0_ub = max(D0_val * 1.1, D0_val * 10, 1e-3)
                    else:
                        # 未来年份：
                        # - 下界：仅设极小值（1e-9）防止数值问题，允许产品退出生产/需求降为0
                        # - 上界：设置足够大的上界，不在变量界限层面限制增长
                        #         实际增长约束通过 growth_constraints 和 hist_max_production 来控制
                        Q0_lb = 1e-9  # 仅防止数值问题，允许停产
                        D0_lb = 1e-9  # 允许需求降至接近0
                        Q0_ub = 1e12  # 供给上界：由 enable_growth_constraints 控制实际增长
                        D0_ub = 1e12  # 需求上界：不限制（允许饮食结构任意转变）
                    
                    Qs[key] = m.addVar(lb=Q0_lb, ub=Q0_ub, name=f"Qs[{r},{j},{t}]")
                    Qd[key] = m.addVar(lb=D0_lb, ub=D0_ub, name=f"Qd[{r},{j},{t}]")
    
    # 库存和 slack
    stock: Dict[Tuple[str, int], gp.Var] = {}
    excess: Dict[Tuple[str, int], gp.Var] = {}
    shortage: Dict[Tuple[str, int], gp.Var] = {}
    for j in commodities:
        for t in years:
            stock[j, t] = m.addVar(lb=0, ub=1e12, name=f"stock[{j},{t}]")
            excess[j, t] = m.addVar(lb=0, name=f"excess[{j},{t}]")
            shortage[j, t] = m.addVar(lb=0, name=f"shortage[{j},{t}]")
    
    # ==========================================================================
    # 排放变量
    # ==========================================================================
    
    # 区域排放 E[r,j,t] 和减排成本 C[r,j,t]
    Eij: Dict[Tuple[str, str, int], gp.Var] = {}
    Cij: Dict[Tuple[str, str, int], gp.Var] = {}
    
    # MACC 减排变量：a[r,j,t,proc,seg] - 每个过程每个 segment 的减排量
    abatement_vars: Dict[Tuple[str, str, int, str, int], gp.Var] = {}
    abatement_caps: Dict[Tuple[str, str, int, str, int], gp.Constr] = {}
    abatement_costs: Dict[Tuple[str, str, int, str, int], float] = {}  # 边际成本
    
    for key in Qs.keys():
        r, j, t = key
        Eij[key] = m.addVar(lb=0.0, name=f"E[{r},{j},{t}]")
        Cij[key] = m.addVar(lb=0.0, name=f"C[{r},{j},{t}]")
    
    # ==========================================================================
    # 约束
    # ==========================================================================
    
    # 约束引用字典（用于 MC 更新）
    constr_supply: Dict[Tuple[str, str, int], gp.Constr] = {}
    constr_demand: Dict[Tuple[str, str, int], gp.Constr] = {}
    constr_Edef: Dict[Tuple[str, str, int], gp.Constr] = {}
    
    # 校准参数字典（用于 MC 更新）
    alpha_s_cache: Dict[Tuple[str, str, int], float] = {}
    alpha_d_cache: Dict[Tuple[str, str, int], float] = {}
    eps_s_cache: Dict[Tuple[str, str, int], float] = {}
    eps_d_cache: Dict[Tuple[str, str, int], float] = {}
    eps_pop_cache: Dict[Tuple[str, str, int], float] = {}
    eps_inc_cache: Dict[Tuple[str, str, int], float] = {}
    eta_y_cache: Dict[Tuple[str, str, int], float] = {}
    eta_temp_cache: Dict[Tuple[str, str, int], float] = {}
    Q0_cache: Dict[Tuple[str, str, int], float] = {}
    D0_cache: Dict[Tuple[str, str, int], float] = {}
    P0_cache: Dict[Tuple[str, str, int], float] = {}
    Ymult0_cache: Dict[Tuple[str, str, int], float] = {}
    Tmult0_cache: Dict[Tuple[str, str, int], float] = {}
    pop_base_cache: Dict[Tuple[str, str, int], float] = {}
    inc_base_cache: Dict[Tuple[str, str, int], float] = {}
    proc_cap_basecoeff: Dict[Tuple[str, str, int, str, int], float] = {}
    
    # ==========================================================================
    # 1. 供给方程（线性化的 log-log 弹性）
    # ==========================================================================
    # Full 版本 (log-log，对数空间相加 = 原始空间相乘):
    #   ln(Qs) = α_s + ε_s·ln(Pnet) + η_y·ln(Ymult) + η_temp·ln(Tmult) + Σ(ε_sj·ln(Pj))
    #   
    # 等价于原始空间:
    #   Qs = A · Pnet^ε_s · Ymult^η_y · Tmult^η_temp · ∏(Pj^ε_sj)
    #
    # 线性化（一阶泰勒展开，在基期点展开）:
    #   ln(Qs) ≈ ln(Q0) + ε_s·[(P-P0)/P0] + η_y·[(Y-1)/1] + η_temp·[(T-1)/1] + Σ(ε_sj·[(Pj-P0j)/P0j])
    #
    # 整理得（乘法形式的线性近似）:
    #   Qs ≈ Q0 · exp(ε_s·(P-P0)/P0 + η_y·(Ymult-1) + η_temp·(Tmult-1) + Σ(ε_sj·(Pj-P0j)/P0j))
    #
    # 进一步线性化 exp(x) ≈ 1 + x（当 x 较小时）:
    #   Qs ≈ Q0 · [1 + ε_s·(P-P0)/P0 + η_y·(Ymult-1) + η_temp·(Tmult-1) + Σ(ε_sj·(Pj-P0j)/P0j)]
    #
    # 注意：这里所有效应都是【加法叠加】的，不是相乘！
    # ==========================================================================
    
    # 统计历史/未来年份约束
    n_hist_fixed = 0
    n_future_supply = 0
    
    for key in Qs.keys():
        r, j, t = key
        data = idx[key]
        Q0 = max(1e-6, data['Q0'])
        P0 = max(1e-6, data['P0'])
        
        # ===== 历史年份：固定 Qs = Q0，不应用弹性方程 =====
        if t <= hist_end_year:
            # 固定供给量为历史观测值
            m.addConstr(Qs[key] == Q0, name=f"supply_fixed[{r},{j},{t}]")
            # 仍然缓存参数用于其他用途
            alpha_s_cache[key] = Q0
            eps_s_cache[key] = 0.0
            eta_y_cache[key] = 0.0
            eta_temp_cache[key] = 0.0
            Q0_cache[key] = Q0
            P0_cache[key] = P0
            Ymult0_cache[key] = 1.0
            Tmult0_cache[key] = 1.0
            n_hist_fixed += 1
            continue
        
        # ===== 未来年份：应用弹性方程 =====
        n_future_supply += 1
        
        # 关键修复：未来年份使用历史基期（hist_end_year）的 Q0 作为基准
        # 这确保供给方程与历史固定值一致，避免与增长约束冲突
        base_key = (r, j, hist_end_year)
        if base_key in idx:
            Q0_base = max(1e-6, idx[base_key].get('Q0', 1e-6))
            P0_base = max(1e-6, idx[base_key].get('P0', 1.0))
        else:
            # 如果没有历史基期数据，使用当前年份数据（fallback）
            Q0_base = max(1e-6, data['Q0'])
            P0_base = max(1e-6, data['P0'])
        
        # 使用基期的 Q0 和 P0
        Q0 = Q0_base
        P0 = P0_base
        
        # 供给弹性
        eps_s = data.get('eps_supply', 0.0) or 0.0
        eta_y = data.get('eps_supply_yield', 0.0) or 0.0      # 产率弹性
        eta_temp = data.get('eps_supply_temp', 0.0) or 0.0    # 温度弹性
        
        # 产率和温度乘数（外生给定的情景因子）
        Ymult = data.get('Ymult', 1.0) or 1.0
        Tmult = data.get('Tmult', 1.0) or 1.0
        
        # 交叉价格弹性
        epsS_row = data.get('epsS_row', {}) or {}
        
        # 计算常数项（非价格因素的效应）
        # 这些在模型求解前就已经确定，作为 Q0 的调整
        yield_adj = eta_y * (Ymult - 1.0)      # 产率效应
        temp_adj = eta_temp * (Tmult - 1.0)    # 温度效应
        
        # Q0 调整后的基准（包含产率和温度效应）
        # Qs = Q0 * [1 + yield_adj + temp_adj + price_effects]
        Q0_const = Q0 * (1.0 + yield_adj + temp_adj)
        
        # 自身价格效应: ε_s * Q0 * (Pc - P0) / P0 = ε_s * Q0 / P0 * Pc - ε_s * Q0
        # 整理: Qs = Q0_const + ε_s * Q0 * (Pc - P0) / P0 + Σ(ε_sj * Q0 * (Pj - P0j) / P0j)
        #     = Q0_const - ε_s * Q0 + ε_s * Q0 / P0 * Pc + Σ(...)
        #     = [Q0_const - ε_s * Q0 - Σ(ε_sj * Q0)] + ε_s * Q0 / P0 * Pc + Σ(ε_sj * Q0 / P0j * Pj)
        
        # 收集所有交叉价格弹性之和
        sum_cross_eps = sum(float(v) for v in epsS_row.values())
        
        # 常数项 a_s = Q0 * (1 + yield_adj + temp_adj) - Q0 * ε_s - Q0 * Σε_sj
        #           = Q0 * (1 + yield_adj + temp_adj - ε_s - Σε_sj)
        a_s = Q0 * (1.0 + yield_adj + temp_adj - eps_s - sum_cross_eps)
        
        # 自身价格系数: b_s = Q0 * ε_s / P0
        b_s = Q0 * eps_s / P0
        
        # 过滤极小系数（避免数值不稳定）
        COEFF_THRESHOLD = 1e-6  # 提高阈值以改善数值范围
        if abs(b_s) < COEFF_THRESHOLD:
            b_s = 0.0
        
        # ===== Phase 2: 单位税 tax_unit_adder =====
        # 供给响应净价格 Pnet = Pc - tau，所以:
        # Qs = a_s + b_s * (Pc - tau) = (a_s - b_s * tau) + b_s * Pc
        # 调整常数项 a_s 以包含税的效应
        tau = 0.0
        if tax_unit_adder:
            tau = float(tax_unit_adder.get(key, 0.0) or 0.0)
            a_s = a_s - b_s * tau
        
        # 构建交叉价格项: Σ(ε_sj * Q0 / P0j * Pc_j)
        cross_terms_s = gp.LinExpr(0.0)
        for other_comm, cross_eps in epsS_row.items():
            if other_comm == j:  # 跳过自身
                continue
            if (other_comm, t) in Pc:
                other_key = (r, other_comm, t)
                P0_other = idx.get(other_key, {}).get('P0', P0) or P0
                P0_other = max(1e-6, P0_other)
                b_cross = Q0 * float(cross_eps) / P0_other
                # 过滤极小交叉弹性系数
                if abs(b_cross) < COEFF_THRESHOLD:
                    continue
                cross_terms_s += b_cross * Pc[other_comm, t]
        
        cs = m.addConstr(Qs[key] == a_s + b_s * Pc[j, t] + cross_terms_s, name=f"supply[{r},{j},{t}]")
        constr_supply[key] = cs
        
        # 缓存校准参数
        alpha_s_cache[key] = a_s
        eps_s_cache[key] = eps_s
        eta_y_cache[key] = eta_y
        eta_temp_cache[key] = eta_temp
        Q0_cache[key] = Q0
        P0_cache[key] = P0
        Ymult0_cache[key] = Ymult
        Tmult0_cache[key] = Tmult
    
    # ==========================================================================
    # 2. 需求方程（线性化的 log-log 弹性 + 人口/收入效应）
    # ==========================================================================
    # Full 版本 (log-log，对数空间相加 = 原始空间相乘):
    #   ln(Qd) = α_d + ε_d·ln(Pc) + ε_pop·ln(Pop/Pop0) + ε_inc·ln(Inc/Inc0) + Σ(ε_dj·ln(Pj))
    #
    # 等价于原始空间:
    #   Qd = A · Pc^ε_d · (Pop/Pop0)^ε_pop · (Inc/Inc0)^ε_inc · ∏(Pj^ε_dj)
    #
    # 线性化（一阶泰勒展开）:
    #   ln(Qd) ≈ ln(D0) + ε_d·(Pc-P0)/P0 + ε_pop·ln(Pop/Pop0) + ε_inc·ln(Inc/Inc0) + Σ(ε_dj·(Pj-P0j)/P0j)
    #
    # 注意：人口和收入的对数项是【常数】（在模型求解前已知），不需要线性化
    # 但价格项需要线性化
    #
    # 整理得:
    #   Qd ≈ D0 · exp(ε_pop·ln(Pop/Pop0) + ε_inc·ln(Inc/Inc0)) · [1 + ε_d·(Pc-P0)/P0 + Σ(ε_dj·(Pj-P0j)/P0j)]
    #      = D0 · (Pop/Pop0)^ε_pop · (Inc/Inc0)^ε_inc · [1 + price_effects]
    #
    # 所以：人口/收入效应是【乘法】（因为是常数，直接用幂函数）
    #       价格效应是【加法线性化】
    # ==========================================================================
    
    n_hist_demand_fixed = 0
    n_future_demand = 0
    
    for key in Qd.keys():
        r, j, t = key
        data = idx[key]
        D0 = max(1e-6, data['D0'])
        P0 = max(1e-6, data['P0'])
        
        # ===== 历史年份：固定 Qd = D0，不应用弹性方程 =====
        if t <= hist_end_year:
            # 固定需求量为历史观测值
            m.addConstr(Qd[key] == D0, name=f"demand_fixed[{r},{j},{t}]")
            # 缓存参数
            alpha_d_cache[key] = D0
            eps_d_cache[key] = 0.0
            eps_pop_cache[key] = 0.0
            eps_inc_cache[key] = 0.0
            D0_cache[key] = D0
            pop_base_cache[key] = 1.0
            inc_base_cache[key] = 1.0
            n_hist_demand_fixed += 1
            continue
        
        # ===== 未来年份：应用弹性方程 =====
        n_future_demand += 1
        
        # 关键修复：未来年份使用历史基期（hist_end_year）的 D0 和 P0 作为基准
        base_key = (r, j, hist_end_year)
        if base_key in idx:
            D0_base = max(1e-6, idx[base_key].get('D0', 1e-6))
            P0_base = max(1e-6, idx[base_key].get('P0', 1.0))
        else:
            # 如果没有历史基期数据，使用当前年份数据（fallback）
            D0_base = max(1e-6, data['D0'])
            P0_base = max(1e-6, data['P0'])
        
        D0 = D0_base
        P0 = P0_base
        
        # 需求弹性
        eps_d = data.get('eps_demand', 0.0) or 0.0
        eps_pop = data.get('eps_pop_demand', 0.0) or 0.0      # 人口弹性
        eps_inc = data.get('eps_income_demand', 0.0) or 0.0   # 收入弹性
        
        # 人口和收入（这些是外生常数，在模型求解前已知）
        pop_base = max(1e-6, data.get('pop_base', 1.0) or 1.0)
        pop_t = max(1e-6, data.get('pop_t', pop_base) or pop_base)
        inc_base = max(1e-6, data.get('inc_base', 1.0) or 1.0)
        inc_t = max(1e-6, data.get('inc_t', inc_base) or inc_base)
        
        # 人口和收入效应（使用幂函数，因为这些是常数）
        # 这里保持乘法形式是正确的！
        pop_ratio = pop_t / pop_base
        inc_ratio = inc_t / inc_base
        pop_effect = pop_ratio ** eps_pop if eps_pop != 0 else 1.0
        inc_effect = inc_ratio ** eps_inc if eps_inc != 0 else 1.0
        
        # 交叉价格弹性
        epsD_row = data.get('epsD_row', {}) or {}
        sum_cross_eps = sum(float(v) for v in epsD_row.values())
        
        # D0 调整后的基准（包含人口和收入效应 - 乘法）
        D0_adjusted = D0 * pop_effect * inc_effect
        
        # ===== Phase 2: 饲料减少 feed_reduction_by =====
        # 如果有 feed reduction scenario，对饲料用途的商品需求乘以 (1 - reduction_rate)
        # 这里简化处理：将 reduction 直接应用到 D0_adjusted
        feed_mult = 1.0
        if feed_reduction_by:
            red = float(feed_reduction_by.get(key, 0.0) or 0.0)
            red = max(0.0, min(1.0, red))  # 限制在 [0, 1]
            feed_mult = 1.0 - red
            D0_adjusted = D0_adjusted * feed_mult
        
        # 价格效应（加法线性化）
        # Qd = D0_adj * [1 + ε_d*(Pc-P0)/P0 + Σ(ε_dj*(Pj-P0j)/P0j)]
        #    = D0_adj * (1 - ε_d - Σε_dj) + D0_adj * ε_d / P0 * Pc + Σ(D0_adj * ε_dj / P0j * Pj)
        
        a_d = D0_adjusted * (1.0 - eps_d - sum_cross_eps)
        b_d = D0_adjusted * eps_d / P0
        
        # 过滤极小系数（避免数值不稳定）
        COEFF_THRESHOLD = 1e-6  # 提高阈值以改善数值范围
        if abs(b_d) < COEFF_THRESHOLD:
            b_d = 0.0
        
        # 构建交叉价格项
        cross_terms_d = gp.LinExpr(0.0)
        for other_comm, cross_eps in epsD_row.items():
            if other_comm == j:  # 跳过自身
                continue
            if (other_comm, t) in Pc:
                other_key = (r, other_comm, t)
                P0_other = idx.get(other_key, {}).get('P0', P0) or P0
                P0_other = max(1e-6, P0_other)
                b_cross = D0_adjusted * float(cross_eps) / P0_other
                # 过滤极小交叉弹性系数
                if abs(b_cross) < COEFF_THRESHOLD:
                    continue
                cross_terms_d += b_cross * Pc[other_comm, t]
        
        cd = m.addConstr(Qd[key] == a_d + b_d * Pc[j, t] + cross_terms_d, name=f"demand[{r},{j},{t}]")
        constr_demand[key] = cd
        
        # 缓存校准参数
        alpha_d_cache[key] = a_d
        eps_d_cache[key] = eps_d
        eps_pop_cache[key] = eps_pop
        eps_inc_cache[key] = eps_inc
        D0_cache[key] = D0
        pop_base_cache[key] = pop_base
        inc_base_cache[key] = inc_base
    
    # 日志：历史/未来年份约束统计
    logger.info(f"[LINEAR] 供给约束: 历史固定={n_hist_fixed}, 未来弹性={n_future_supply}")
    logger.info(f"[LINEAR] 需求约束: 历史固定={n_hist_demand_fixed}, 未来弹性={n_future_demand}")
    
    # ==========================================================================
    # 2.5 反刍动物需求上限约束 (Phase 2: ruminant_intake_cap)
    # ==========================================================================
    # 如果提供了 ruminant_intake_cap，则对反刍动物商品的需求施加上限
    # Σ(Qd[r,j,t] for j in ruminant_comms) <= cap[r,t]
    
    rumi_intake_constr: Dict[Tuple[str, int], gp.Constr] = {}
    if ruminant_intake_cap:
        # 默认反刍动物商品列表
        rumi_comms = ruminant_commodities or [
            'Meat of cattle with the bone, fresh or chilled',
            'Meat of buffalo, fresh or chilled',
            'Meat of sheep, fresh or chilled',
            'Meat of goat, fresh or chilled',
        ]
        rumi_set = set(rumi_comms)
        
        for (r_key, t_key), cap_val in ruminant_intake_cap.items():
            if cap_val is None or cap_val <= 0:
                continue
            
            # 构建该区域-年份的反刍动物需求之和
            rumi_demand = gp.quicksum(
                Qd[r_key, j, t_key]
                for j in rumi_set
                if (r_key, j, t_key) in Qd
            )
            
            if isinstance(rumi_demand, gp.LinExpr) and rumi_demand.size() > 0:
                con = m.addConstr(
                    rumi_demand <= float(cap_val),
                    name=f"rumi_cap[{r_key},{t_key}]"
                )
                rumi_intake_constr[(r_key, t_key)] = con
                logger.debug(f"[LINEAR] 反刍动物需求上限约束: {r_key}, {t_key} <= {cap_val:.0f}")
    
    # 3. 全球市场清算
    for j in commodities:
        for t in years:
            supply_sum = gp.quicksum(Qs[r, j, t] for r in regions if (r, j, t) in Qs)
            demand_sum = gp.quicksum(Qd[r, j, t] for r in regions if (r, j, t) in Qd)
            
            m.addConstr(
                supply_sum + shortage[j, t] == demand_sum + excess[j, t] + stock[j, t],
                name=f"clear[{j},{t}]"
            )
    
    # ==========================================================================
    # 3b. 短缺和过剩硬约束（可选）：基于全球总能量供给的约束
    # ==========================================================================
    # 改进：改为全球总能量约束，而非分商品约束
    # shortage_energy ≤ max_slack_rate × total_energy_supply
    # 其中 energy = Σ(kcal_per_ton[j] × quantity[j])
    
    slack_limit_constr: Dict[int, Tuple[gp.Constr, gp.Constr]] = {}
    if max_slack_rate is not None and max_slack_rate > 0:
        if nutrient_per_unit_by_comm:  # 需要营养系数才能转换为能量
            for t in years:
                # ✅ 只对未来年份应用短缺限制
                if t <= hist_end_year:
                    continue
                
                # 计算全球总能量供给（kcal）
                total_energy_supply = gp.LinExpr(0.0)
                for j in commodities:
                    kcal_per_ton = float(nutrient_per_unit_by_comm.get(j, 0.0) or 0.0)
                    if kcal_per_ton > 0:
                        supply_sum_j = gp.quicksum(Qs[r, j, t] for r in regions if (r, j, t) in Qs)
                        if isinstance(supply_sum_j, gp.LinExpr) and supply_sum_j.size() > 0:
                            total_energy_supply += kcal_per_ton * supply_sum_j
                
                # 计算能量短缺（kcal）
                energy_shortage = gp.LinExpr(0.0)
                for j in commodities:
                    kcal_per_ton = float(nutrient_per_unit_by_comm.get(j, 0.0) or 0.0)
                    if kcal_per_ton > 0:
                        energy_shortage += kcal_per_ton * shortage[j, t]
                
                # 计算能量过剩（kcal）
                energy_excess = gp.LinExpr(0.0)
                for j in commodities:
                    kcal_per_ton = float(nutrient_per_unit_by_comm.get(j, 0.0) or 0.0)
                    if kcal_per_ton > 0:
                        energy_excess += kcal_per_ton * excess[j, t]
                
                # 添加约束：能量短缺/过剩 ≤ max_slack_rate × 总能量供给
                if isinstance(total_energy_supply, gp.LinExpr) and total_energy_supply.size() > 0:
                    cn_shortage = m.addConstr(
                        energy_shortage <= max_slack_rate * total_energy_supply,
                        name=f"slack_limit_energy_short[{t}]"
                    )
                    cn_excess = m.addConstr(
                        energy_excess <= max_slack_rate * total_energy_supply,
                        name=f"slack_limit_energy_excess[{t}]"
                    )
                    slack_limit_constr[t] = (cn_shortage, cn_excess)
            
            logger.info(f"[LINEAR] 短缺/过剩能量约束: {len(slack_limit_constr)} 个未来年份 (限制≤{max_slack_rate:.1%}全球总能量供给，历史年份不限制)")
        else:
            logger.warning(f"[LINEAR] 未提供营养系数，跳过能量约束")
    
    # ==========================================================================
    # 4. 排放约束: E = Σ(e0_proc * Qs) - Σ(abatement)
    # ==========================================================================
    
    total_abatement_cost = gp.LinExpr(0.0)
    total_E_land = gp.LinExpr(0.0)
    total_E_other = gp.LinExpr(0.0)
    
    # 选择成本计算方法
    if cost_calculation_method == 'unit_cost':
        # ==========================================================================
        # 方法二：基于给定单位成本的减排成本计筗
        # 成本 = Σ (unit_cost * 减排量)
        # 减排量 = E_baseline - E_current (相对于BASE情景)
        # ==========================================================================
        logger.info("[LINEAR] 使用单位成本方法计算减排成本")
        
        # 构建BASELINE排放查找表
        baseline_emissions: Dict[Tuple[str, str, int, str], float] = {}  # {(region, comm, year, process): tCO2e}
        
        if baseline_scenario_result and 'Qs' in baseline_scenario_result:
            baseline_Qs = baseline_scenario_result['Qs']
            for key in baseline_Qs.keys():
                r, j, t = key
                Q_base = baseline_Qs[key]
                e0_map = e0_by_region.get(key, {})
                for proc, e0p in e0_map.items():
                    e_base = float(e0p) * float(Q_base)
                    baseline_emissions[(r, j, t, proc)] = e_base
            logger.info(f"[LINEAR] 加载了 {len(baseline_emissions)} 条BASELINE排放数据")
        else:
            logger.warning("[LINEAR] 未提供BASELINE情景结果，无法计算减排量和成本")
        
        # 计算每个节点的排放和成本
        for key in Qs.keys():
            r, j, t = key
            e0_map = e0_by_region.get(key, {})
            sum_e0 = sum(float(v) for v in e0_map.values())
            
            # 当前情景的总排放
            emis_expr = sum_e0 * Qs[key]
            
            # 按过程计算减排量和成本
            cost_expr = gp.LinExpr(0.0)
            
            for proc, e0p in e0_map.items():
                e0p = float(e0p)
                if e0p <= 0:
                    continue
                
                # 当前过程排放
                e_current = e0p * Qs[key]
                
                # 从dict_v3获取process到cost process的映射
                process_cost_name = None
                if process_cost_mapping:
                    process_cost_name = process_cost_mapping.get(proc)
                
                # 处理不同类型的成本映射
                if process_cost_name is None:
                    # 无成本数据，跳过
                    continue
                elif process_cost_name == 'Production value':
                    # 特殊处理：De/Reforestation过程的经济损失
                    # 计算为：相对BASE情景，由于土地面积约束导致的产出减少 × 市场价格
                    # TODO: 这需要在solve之后才能计算（需要产量变化和价格）
                    # 暂时在模型中不处理，留在S4_0_main后处理
                    pass
                else:
                    # 使用单位成本数据
                    unit_cost = 0.0
                    if unit_cost_data:
                        # 尝试匹配 (region, process_cost_name)
                        unit_cost = unit_cost_data.get((r, process_cost_name), 0.0)
                    
                    if unit_cost > 0 and baseline_emissions:
                        # 计算减排量：baseline - current
                        e_baseline = baseline_emissions.get((r, j, t, proc), 0.0)
                        # abatement = max(0, e_baseline - e_current)
                        # 成本 = unit_cost * abatement
                        # 因为e_current = e0p * Qs, 所以：
                        # abatement = max(0, e_baseline - e0p * Qs)
                        # 但在优化中我们不能直接用max，因此简化为：
                        # 如果e_baseline > 0，则成本 = unit_cost * (e_baseline - e0p * Qs)
                        # 这在目标函数中会激励减少排放
                        if e_baseline > 0:
                            # 成本项：unit_cost * (baseline - current)
                            # = unit_cost * baseline - unit_cost * e0p * Qs
                            # 在目标函数中，我们需要minimize cost
                            # 故cost = -unit_cost * (e_baseline - e0p * Qs) = unit_cost * e0p * Qs - unit_cost * e_baseline
                            # 但成本应该是正的，所以我们需要重新理解：
                            # 减排成本 = unit_cost * 减排量 = unit_cost * max(0, e_baseline - e_current)
                            # 在线性优化中，我们用辅助变量来表示减排量
                            # abat_var >= 0
                            # abat_var >= e_baseline - e_current
                            # cost += unit_cost * abat_var
                            # 但这会增加变量数。为简化，我们直接用：
                            # 如果我们假设当前情景排放总是 <= baseline（因为有约束）
                            # 那么 abatement = e_baseline - e_current 总是 >= 0
                            # 成本 = unit_cost * (e_baseline - e0p * Qs)
                            # 在目标函数中，这是一个常数 - 系数*Qs
                            # 优化会倾向于增加Qs来减少成本，这与我们的目标相反
                            # 
                            # 正确的做法是：成本 = unit_cost * max(0, e_baseline - e_current)
                            # 但这需要辅助变量。暂时用简化版本：
                            # 当 e_current < e_baseline 时，成本 = unit_cost * (e_baseline - e_current)
                            # 当 e_current >= e_baseline 时，成本 = 0
                            # 由于我们在minimize总成本，而成本随着e_current增加而减少，
                            # 这与直觉相反。
                            # 
                            # 重新理解：减排成本应该是“为了减少排放而付出的代价”
                            # 如果我们的目标是最小化成本，那么：
                            # - 如果减排需要成本，则应该在目标函数中加上这个成本
                            # - 成本应该与减排量成正比
                            # 
                            # 让我们用辅助变量 a_proc 表示减排量：
                            # a_proc >= 0
                            # e_current + a_proc = e_baseline  (如果减排)
                            # 或 e_current = e_baseline - a_proc
                            # cost += unit_cost * a_proc
                            # 
                            # 这样的话，a_proc 越大，成本越高，优化会倾向于减小 a_proc
                            # 但同时 e_current = e0p * Qs = e_baseline - a_proc
                            # 即 Qs = (e_baseline - a_proc) / e0p
                            # 这意味着减小 a_proc 会增加 Qs，与减排目标相反
                            #
                            # 正确的理解是：
                            # 我们应该在目标函数中加上“减排成本”，同时添加排放限制约束
                            # 或者在目标函数中加上“排放罚金项”：penalty * E_current
                            # 
                            # 由于这里的逻辑较为复杂，我们采用以下简化方案：
                            # 在目标函数中加上：unit_cost * max(0, e_baseline - e_current)
                            # 用辅助变量 abat >= 0, abat >= e_baseline - e_current
                            # 然后 cost += unit_cost * abat
                            #
                            # 创建辅助变量
                            abat_var = m.addVar(lb=0.0, name=f"abat_unit[{r},{j},{t},{proc}]")
                            # abat >= e_baseline - e_current = e_baseline - e0p * Qs
                            m.addConstr(abat_var >= e_baseline - e0p * Qs[key], 
                                        name=f"abat_def_unit[{r},{j},{t},{proc}]")
                            # 成本项
                            cost_expr += unit_cost * abat_var
                            
                            # 保存到abatement_vars用于后续提取
                            # key: (region, commodity, year, process)
                            abatement_vars[(r, j, t, proc)] = abat_var
                            # 保存单位成本信息
                            abatement_costs[(r, j, t, proc)] = unit_cost
            
            # 排放定义
            ce = m.addConstr(Eij[key] == emis_expr, name=f"E_def[{r},{j},{t}]")
            constr_Edef[key] = ce
            
            # 成本定义
            if cost_expr.size() > 0:
                m.addConstr(Cij[key] == cost_expr, name=f"C_def[{r},{j},{t}]")
                total_abatement_cost += Cij[key]
            else:
                m.addConstr(Cij[key] == 0.0, name=f"C_def[{r},{j},{t}]")
            
            # 区分 LULUCF 和其他排放
            e_land = sum(float(v) for p, v in e0_map.items() if _is_lulucf_process(p))
            e_other = sum_e0 - e_land
            
            total_E_land += e_land * Qs[key]
            total_E_other += e_other * Qs[key]
    
    else:
        # ==========================================================================
        # 方法一：原有的MACC减排成本计算
        # ==========================================================================
        logger.info("[LINEAR] 使用MACC方法计算减排成本")
        
        for key in Qs.keys():
            r, j, t = key
            e0_map = e0_by_region.get(key, {})
            sum_e0 = sum(float(v) for v in e0_map.values())
            
            # 基础排放表达式
            emis_expr = sum_e0 * Qs[key]
            
            # MACC 减排变量（如果有 MACC 数据）
            abat_vars_this_node: List[gp.Var] = []
            cost_terms: List[gp.LinExpr] = []
            
            if has_macc and sum_e0 > 0:
                for proc, e0p in e0_map.items():
                    e0p = float(e0p)
                    if e0p <= 0:
                        continue
                    
                    # 查找 MACC 数据（按区域或全局）
                    dfp = pd.DataFrame()
                    if 'Country' in macc_df.columns:
                        dfp = macc_df[(macc_df['Country'] == r) & (macc_df['Process'] == proc)]
                    if dfp.empty and 'Process' in macc_df.columns:
                        dfp = macc_df[macc_df['Process'] == proc]
                    
                    if dfp.empty:
                        continue
                    
                    # 解析 MACC 曲线
                    if 'cumulative_fraction_of_process' in dfp.columns and 'marginal_cost_$per_tco2e' in dfp.columns:
                        dfp = dfp[['cumulative_fraction_of_process', 'marginal_cost_$per_tco2e']].dropna()
                        dfp = dfp.sort_values('cumulative_fraction_of_process')
                        
                        prev_frac = 0.0
                        for seg_idx, (_, row) in enumerate(dfp.iterrows()):
                            frac = float(row['cumulative_fraction_of_process'])
                            mu = float(row['marginal_cost_$per_tco2e'])
                            
                            if frac <= prev_frac:
                                continue
                            
                            # 创建减排变量 a[r,j,t,proc,seg]
                            # 约束: a <= (delta_frac * e0p) * Qs
                            delta_frac = frac - prev_frac
                            coeff = delta_frac * e0p
                            
                            a_var = m.addVar(lb=0.0, name=f"a[{r},{j},{t},{proc},{seg_idx}]")
                            cap_con = m.addConstr(a_var <= coeff * Qs[key], name=f"cap[{r},{j},{t},{proc},{seg_idx}]")
                            
                            abatement_vars[(r, j, t, proc, seg_idx)] = a_var
                            abatement_caps[(r, j, t, proc, seg_idx)] = cap_con
                            abatement_costs[(r, j, t, proc, seg_idx)] = mu
                            proc_cap_basecoeff[(r, j, t, proc, seg_idx)] = coeff  # 保存基准系数
                            
                            abat_vars_this_node.append(a_var)
                            cost_terms.append(mu * a_var)
                            
                            prev_frac = frac
            
            # 排放定义: E = e0 * Qs - 减排量
            if abat_vars_this_node:
                emis_expr -= gp.quicksum(abat_vars_this_node)
            ce = m.addConstr(Eij[key] == emis_expr, name=f"E_def[{r},{j},{t}]")
            constr_Edef[key] = ce
            
            # 成本定义
            if cost_terms:
                m.addConstr(Cij[key] == gp.quicksum(cost_terms), name=f"C_def[{r},{j},{t}]")
                total_abatement_cost += Cij[key]
            else:
                m.addConstr(Cij[key] == 0.0, name=f"C_def[{r},{j},{t}]")
            
            # 区分 LULUCF 和其他排放（用于土地碳价）
            e_land = sum(float(v) for p, v in e0_map.items() if _is_lulucf_process(p))
            e_other = sum_e0 - e_land
            
            # LULUCF 减排量
            abat_land = gp.quicksum(
                abatement_vars.get((r, j, t, p, s), 0)
                for (rr, jj, tt, p, s) in abatement_vars.keys()
                if (rr, jj, tt) == (r, j, t) and _is_lulucf_process(p)
            ) if abatement_vars else 0.0
            
            total_E_land += e_land * Qs[key]
            if isinstance(abat_land, gp.LinExpr) and abat_land.size() > 0:
                total_E_land -= abat_land
            total_E_other += e_other * Qs[key]
    
    # ==========================================================================
    # 5. 营养约束（可选）
    # ==========================================================================
    
    nutri_constr: Dict[Tuple[str, int], gp.Constr] = {}
    if nutrition_rhs and nutrient_per_unit_by_comm:
        for r in regions:
            for t in years:
                if t <= hist_end_year:  # 仅未来年份
                    continue
                rhs = nutrition_rhs.get((r, t))
                if rhs is None:
                    continue
                # ✅ 跳过 NaN 值，避免 Gurobi 报错
                try:
                    rhs_float = float(rhs)
                    if np.isnan(rhs_float) or np.isinf(rhs_float):
                        continue
                except (ValueError, TypeError):
                    continue
                
                expr = gp.LinExpr(0.0)
                for j in commodities:
                    if (r, j, t) in Qd:
                        v = float(nutrient_per_unit_by_comm.get(j, 0.0) or 0.0)
                        if v > 0:
                            expr += v * Qd[r, j, t]
                
                if expr.size() > 0:
                    cn = m.addConstr(expr >= rhs_float, name=f"nutri[{r},{t}]")
                    nutri_constr[(r, t)] = cn
        
        # ✅ 诊断：显示营养约束的RHS值（前5个样本）
        if nutri_constr:
            logger.info(f"[LINEAR] 营养约束: {len(nutri_constr)} 个")
            sample_count = 0
            for (r, t), rhs_val in sorted(nutrition_rhs.items())[:5]:
                if (r, t) in nutri_constr:
                    logger.info(f"  - 样本: {r[:20]}, {t}年: RHS={rhs_val:.2e} kcal")
                    sample_count += 1
                    if sample_count >= 5:
                        break
    
    # ==========================================================================
    # 6. 土地约束（可选）
    # ==========================================================================
    # 约束含义：耕地面积 + 草地面积 + 森林面积 ≤ 总土地上限
    # - 耕地面积 = Σ Qs[r,j,t] / yield（从优化变量计算）
    # - 草地面积 = grass_area_by_region_year[(r, t)]（从 grass_requirement_df 获取）
    # - 森林面积 = forest_area_by_region_year[(r, t)]（从历史数据或LUC模块获取）
    # 未来年份统一使用2020年的土地上限数据
    
    land_constr: Dict[Tuple[str, int], gp.Constr] = {}
    land_constr_diagnostics = []  # 收集约束创建诊断信息
    if land_area_limits:
        BASE_YEAR_FOR_LAND = 2020
        for r in regions:
            # ✅ 统一使用2020年的土地上限数据
            limit = land_area_limits.get((r, BASE_YEAR_FOR_LAND))
            if limit is None:
                continue
            # ✅ 跳过 NaN 值，避免 Gurobi 报错
            try:
                limit_float = float(limit)
                if np.isnan(limit_float) or np.isinf(limit_float):
                    continue
            except (ValueError, TypeError):
                continue
            
            # ✅ 单位转换 - land_area_limits 是 1000 ha，转换为 ha
            limit_ha = limit_float * 1000.0  # 1000 ha -> ha
            
            for t in years:
                # ✅ 获取该区域-年份的草地面积（如果有）
                grass_area_ha = 0.0
                if grass_area_by_region_year:
                    grass_area_ha = float(grass_area_by_region_year.get((r, t), 0.0) or 0.0)
                
                # ✅ 获取该区域-年份的森林面积（如果有）
                forest_area_ha = 0.0
                if forest_area_by_region_year:
                    forest_area_ha = float(forest_area_by_region_year.get((r, t), 0.0) or 0.0)
                
                # ===== 土地约束表达式 =====
                # cropland + grassland ≤ limit - forest
                # cropland = Σ Qs[crop] / yield[crop]  ← 优化变量表达式
                # grassland = Σ Qs[livestock] × grassland_coef[livestock]  ← 优化变量表达式
                land_expr = gp.LinExpr(0.0)
                
                # Cropland部分：使用商品特定产率
                for j in commodities:
                    if (r, j, t) not in Qs:
                        continue
                    
                    # 获取该商品的区域平均产率yield0
                    yield_j = yield_t_per_ha_default  # 默认值
                    node_data = idx.get((r, j, t))
                    if node_data is not None:
                        yield0_val = node_data.get('yield0', 0.0)
                        if yield0_val and yield0_val > 0:
                            yield_j = yield0_val
                    
                    # Qs是吨，coef_crop是ha/吨，结果是ha
                    coef_crop = 1.0 / max(1e-6, yield_j)
                    land_expr += coef_crop * Qs[r, j, t]
                
                # ===== Grassland部分：方案A (dynamic) vs 方案B (static) =====
                if grassland_method == 'dynamic':
                    # 方案A：草地作为优化变量表达式
                    # grassland_ha = Σ grassland_coef[r,livestock] × Qs[r,livestock,t]
                    # ✅ 修复：遍历所有有grassland_coef的commodity，而不是用LIVESTOCK_COMMODITIES
                    for (reg, comm, yr) in Qs.keys():
                        if reg != r or yr != t:
                            continue
                        node_data = idx.get((r, comm, t))
                        if node_data is not None:
                            coef_grass = node_data.get('grassland_coef', 0.0)
                            if coef_grass > 0:
                                # coef_grass 单位：ha/ton (区域Q0加权平均)
                                # Qs 单位：ton
                                land_expr += coef_grass * Qs[r, comm, t]
                else:
                    # 方案B：草地作为外生参数（不随Qs变化）
                    # 在迭代框架中会更新 grass_area_ha
                    pass  # grass_area_ha already loaded from grass_area_by_region_year
                
                if land_expr.size() > 0 or grassland_method == 'static':
                    # 约束：cropland + grassland + forest ≤ limit
                    # 
                    # 方案A (dynamic): 
                    #   land_expr = cropland_expr + grassland_expr (都是优化变量)
                    #   约束: land_expr ≤ limit - forest
                    # 
                    # 方案B (static):
                    #   land_expr = cropland_expr (优化变量)
                    #   约束: land_expr + grass_area_ha ≤ limit - forest
                    #   即: land_expr ≤ limit - forest - grass_area_ha
                    
                    if grassland_method == 'static':
                        available_for_land = limit_ha - forest_area_ha - grass_area_ha
                    else:
                        available_for_land = limit_ha - forest_area_ha
                    
                    # [DIAGNOSTIC] 收集约束创建信息
                    diag_info = {
                        'region': r, 'year': t,
                        'land_limit_ha': limit_ha,
                        'forest_ha': forest_area_ha,
                        'available_for_land_ha': available_for_land,
                        'constraint_added': available_for_land > 0
                    }
                    land_constr_diagnostics.append(diag_info)
                    
                    if available_for_land > 0:
                        cl = m.addConstr(land_expr <= available_for_land, name=f"landU[{r},{t}]")
                        land_constr[(r, t)] = cl
                    else:
                        # 如果available_for_land <= 0，约束未创建，记录警告
                        if len(land_constr_diagnostics) <= 20:  # 只记录前20个警告
                            print(f"    ⚠️ 土地约束未创建: {r} {t}年, available={available_for_land:,.0f} ha <= 0")
                            print(f"       (limit={limit_ha:,.0f}, forest={forest_ha:,.0f})")
    
    # [DIAGNOSTIC] 输出商品产率使用情况
    if idx:
        print(f"\n[Cropland产率诊断]")
        sample_commodities = ['Wheat', 'Maize', 'Rice', 'Soybeans', 'Cattle', 'Pigs', 'Poultry']
        sample_regions_for_yield = ['China', 'U.S.', 'India', 'Brazil']
        sample_year_for_yield = 2020
        
        print(f"  区域-商品产率示例（{sample_year_for_yield}年）：")
        for r in sample_regions_for_yield:
            for j in sample_commodities:
                node_data = idx.get((r, j, sample_year_for_yield))
                if node_data:
                    yield0 = node_data.get('yield0', 0.0)
                    if yield0 and yield0 > 0:
                        print(f"    {r:20s} | {j:15s}: {yield0:.2f} t/ha")
                    elif (r, j, sample_year_for_yield) in idx:
                        print(f"    {r:20s} | {j:15s}: 使用默认 {yield_t_per_ha_default} t/ha")
    
    # [DIAGNOSTIC] 输出土地约束创建汇总
    if land_constr_diagnostics:
        total_attempts = len(land_constr_diagnostics)
        constraints_added = sum(1 for d in land_constr_diagnostics if d['constraint_added'])
        constraints_skipped = total_attempts - constraints_added
        
        print(f"\n[土地约束创建汇总]")
        print(f"  尝试创建: {total_attempts} 个 (区域, 年份) 组合")
        print(f"  成功创建: {constraints_added} 个约束")
        print(f"  跳过: {constraints_skipped} 个 (available_cropland <= 0)")
        
        # 打印几个示例约束
        sample_regions = ['China', 'U.S.', 'India', 'Brazil', 'EUR-Continental']
        sample_years = [2020, 2080]
        print(f"\n[约束示例]")
        for diag in land_constr_diagnostics:
            if diag['region'] in sample_regions and diag['year'] in sample_years:
                r, t = diag['region'], diag['year']
                status = "✅" if diag['constraint_added'] else "❌"
                print(f"  {status} {r} {t}: land_limit={diag['land_limit_ha']:,.0f} ha, "
                      f"forest={diag['forest_ha']:,.0f} ha, "
                      f"available_for_land={diag['available_for_land_ha']:,.0f} ha")
    
    # ==========================================================================
    # 7. 增长约束（可选）- 仅应用于未来年份
    # ==========================================================================
    
    growth_constr: Dict[Tuple[str, str, int], gp.Constr] = {}
    if max_growth_rate_per_period is not None or max_decline_rate_per_period is not None:
        growth_rate = max_growth_rate_per_period if max_growth_rate_per_period is not None else 0.5
        decline_rate = max_decline_rate_per_period if max_decline_rate_per_period is not None else 0.5
        
        for r in regions:
            for j in commodities:
                years_sorted = sorted([t for t in years if (r, j, t) in idx])
                for i_t in range(1, len(years_sorted)):
                    t_prev = years_sorted[i_t - 1]
                    t_curr = years_sorted[i_t]
                    
                    # 只对未来年份应用约束（t_curr > hist_end_year）
                    # 历史年份之间不应用增长约束
                    if t_curr <= hist_end_year:
                        continue
                    
                    year_diff = t_curr - t_prev
                    
                    if (r, j, t_prev) not in Qs or (r, j, t_curr) not in Qs:
                        continue
                    
                    Q0_prev = idx.get((r, j, t_prev), {}).get('Q0', 0.0)
                    if Q0_prev < 1e-3:
                        continue
                    
                    # 上限约束：不能增长太快
                    if max_growth_rate_per_period is not None:
                        max_mult_upper = (1.0 + growth_rate) ** year_diff
                        m.addConstr(
                            Qs[r, j, t_curr] <= Qs[r, j, t_prev] * max_mult_upper,
                            name=f"growth_upper[{r},{j},{t_curr}]"
                        )
                    
                    # 下限约束：不能下降太快
                    if max_decline_rate_per_period is not None:
                        max_mult_lower = (1.0 - decline_rate) ** year_diff
                        m.addConstr(
                            Qs[r, j, t_curr] >= Qs[r, j, t_prev] * max_mult_lower,
                            name=f"growth_lower[{r},{j},{t_curr}]"
                        )
    
    # ==========================================================================
    # 7b. 历史最大产量锚定约束（可选）
    # ==========================================================================
    # 约束未来产量不超过历史最大产量 × (1+growth_rate)^(t-hist_end_year)
    
    hist_max_constr: Dict[Tuple[str, str, int], gp.Constr] = {}
    if hist_max_production:
        # 如果未指定增长率，使用默认值 5%
        anchor_growth_rate = max_growth_rate_per_period if max_growth_rate_per_period is not None else 0.05
        n_hist_constrs = 0
        
        for r in regions:
            for j in commodities:
                hist_max = hist_max_production.get((r, j))
                if hist_max is None or hist_max <= 0:
                    continue
                
                # 只约束未来年份（> hist_end_year）
                future_years = [t for t in years if t > hist_end_year and (r, j, t) in Qs]
                for t in future_years:
                    years_since = t - hist_end_year
                    max_allowed = hist_max * ((1.0 + anchor_growth_rate) ** years_since)
                    
                    cn = m.addConstr(
                        Qs[r, j, t] <= max_allowed,
                        name=f"hist_max_anchor[{r},{j},{t}]"
                    )
                    hist_max_constr[(r, j, t)] = cn
                    n_hist_constrs += 1
        
        if n_hist_constrs > 0:
            logger.info(f"[LINEAR] 历史最大产量约束: {n_hist_constrs} 个 (growth_rate={anchor_growth_rate:.2%})")
            
            # ✅ 诊断：显示典型商品的历史最大产量锚定值
            future_year_sample = max(years) if years else hist_end_year + 60
            years_ahead = future_year_sample - hist_end_year
            sample_count = 0
            for (r, j), hist_max in (hist_max_production.items() if hist_max_production else []):
                if sample_count >= 5:  # 只显示前5个样本
                    break
                if hist_max > 0:
                    max_allowed = hist_max * ((1.0 + anchor_growth_rate) ** years_ahead)
                    logger.info(f"  - 样本: {r[:20]}, {j[:30]}: 历史最大={hist_max:.0f}, {future_year_sample}年上限={max_allowed:.0f} (×{max_allowed/hist_max:.1f})")
                    sample_count += 1
    
    # ==========================================================================
    # 目标函数
    # ==========================================================================
    
    # 基础目标：最小化 slack（市场失衡）
    SLACK_PENALTY = 1e6
    obj = gp.quicksum(
        SLACK_PENALTY * (excess[j, t] + shortage[j, t]) + 0.01 * stock[j, t]
        for j in commodities for t in years
    )
    
    # ✅ 添加生产成本项：防止需求无限增长
    # 生产成本 = Σ(price × Qs)，使用基准价格P0作为近似成本
    # 权重设置为较小值（1.0），使得成本项在合理范围内约束增产
    PRODUCTION_COST_WEIGHT = 1.0
    for key in Qs.keys():
        r, j, t = key
        # 使用基准价格P0作为生产成本代理
        P0_val = P0_cache.get(key, 0.0)
        if P0_val > 0:
            obj += PRODUCTION_COST_WEIGHT * P0_val * Qs[key]
    
    # 加入减排成本
    obj += total_abatement_cost
    
    # 加入土地碳价项：cp * E_land（激励减少 LULUCF 排放）
    if land_carbon_price_by_year:
        for key in Qs.keys():
            r, j, t = key
            cp = float(land_carbon_price_by_year.get(t, 0.0) or 0.0)
            if cp > 0:
                e0_map = e0_by_region.get(key, {})
                e_land = sum(float(v) for p, v in e0_map.items() if _is_lulucf_process(p))
                if e_land > 0:
                    obj += cp * e_land * Qs[key]
    
    m.setObjective(obj, gp.GRB.MINIMIZE)
    
    # ==========================================================================
    # 缓存
    # ==========================================================================
    
    m._nzf_cache = {
        # 变量
        'Pc': Pc, 'Qs': Qs, 'Qd': Qd,
        'Eij': Eij, 'Cij': Cij,
        'stock': stock, 'excess': excess, 'shortage': shortage,
        # 约束引用
        'constr_supply': constr_supply,
        'constr_demand': constr_demand,
        'constr_Edef': constr_Edef,
        'nutri_constr': nutri_constr,
        'land_constr': land_constr,
        'rumi_intake_constr': rumi_intake_constr,  # Phase 2: 反刍动物需求上限约束
        'hist_max_constr': hist_max_constr,  # 历史最大产量锚定约束
        # MACC
        'abatement_vars': abatement_vars,
        'abatement_caps': abatement_caps,
        'abatement_costs': abatement_costs,
        'proc_cap_basecoeff': proc_cap_basecoeff,
        # 校准参数
        'alpha_s': alpha_s_cache,
        'alpha_d': alpha_d_cache,
        'eps_s': eps_s_cache,
        'eps_d': eps_d_cache,
        'eps_pop': eps_pop_cache,
        'eps_inc': eps_inc_cache,
        'eta_y': eta_y_cache,
        'eta_temp': eta_temp_cache,
        'Q0': Q0_cache,
        'D0': D0_cache,
        'P0': P0_cache,
        'Ymult0': Ymult0_cache,
        'Tmult0': Tmult0_cache,
        'pop_base': pop_base_cache,
        'inc_base': inc_base_cache,
        # 元数据
        'regions': regions, 'commodities': commodities, 'years': years,
        'idx': idx,
        'e0_by_region': e0_by_region,
        'nutrient_per_unit_by_comm': nutrient_per_unit_by_comm,  # 用于计算能量短缺
    }
    
    m.update()  # 更新模型以获取正确的变量/约束数
    logger.info(f"[LINEAR] 模型构建完成: 变量={m.NumVars}, 约束={m.NumConstrs}")
    if has_macc:
        logger.info(f"[LINEAR] 减排变量数: {len(abatement_vars)}")
    
    return m


def solve_linear_regional(
    nodes: List[Any],
    commodities: List[str],
    years: List[int],
    time_limit: float = 300.0,
    dict_v3_path: Optional[str] = None,
    output_dir: Optional[str] = None,  # IIS 文件输出目录
    gurobi_log_path: Optional[str] = None,  # Gurobi日志文件路径
    # ===== 人口与收入 =====
    population_by_country_year: Optional[Dict[Tuple[str, int], float]] = None,
    income_mult_by_country_year: Optional[Dict[Tuple[str, int], float]] = None,
    # ===== 排放与减排参数 =====
    macc_path: Optional[str] = None,
    land_carbon_price_by_year: Optional[Dict[int, float]] = None,
    # ===== 约束参数 =====
    nutrition_rhs: Optional[Dict[Tuple[str, int], float]] = None,
    nutrient_per_unit_by_comm: Optional[Dict[str, float]] = None,
    land_area_limits: Optional[Dict[Tuple[str, int], float]] = None,
    grass_area_by_region_year: Optional[Dict[Tuple[str, int], float]] = None,  # ✅ 草地面积 {(region, year): ha}
    forest_area_by_region_year: Optional[Dict[Tuple[str, int], float]] = None,  # ✅ 森林面积 {(region, year): ha}
    yield_t_per_ha_default: float = 3.0,
    grassland_method: str = 'dynamic',  # ✅ 'dynamic' (方案A) or 'static' (方案B)
    max_growth_rate_per_period: Optional[float] = None,
    max_decline_rate_per_period: Optional[float] = None,
    hist_end_year: int = 2020,
    hist_max_production: Optional[Dict[Tuple[str, str], float]] = None,
    # ===== 情景参数 (Phase 2) =====
    tax_unit_adder: Optional[Dict[Tuple[str, str, int], float]] = None,
    feed_reduction_by: Optional[Dict[Tuple[str, str, int], float]] = None,
    ruminant_intake_cap: Optional[Dict[Tuple[str, int], float]] = None,
    ruminant_commodities: Optional[List[str]] = None,
    # ===== 市场失衡限制 =====
    max_slack_rate: Optional[float] = None,
    # ===== 单位成本法参数 =====
    unit_cost_data: Optional[Dict[Tuple[str, str], float]] = None,
    baseline_scenario_result: Optional[Dict[str, Any]] = None,
    process_cost_mapping: Optional[Dict[str, str]] = None,
    cost_calculation_method: str = 'MACC',
) -> Dict[str, Any]:
    """
    求解线性区域模型（完整版）
    
    返回：
    - status: Gurobi 状态码
    - Pc: 价格结果 {(j, t): value}
    - Qs: 区域供给 {(r, j, t): value}
    - Qd: 区域需求 {(r, j, t): value}
    - Eij: 区域排放 {(r, j, t): value}
    - Cij: 区域减排成本 {(r, j, t): value}
    - abatement: 减排量 {(r, j, t, proc, seg): value}
    """
    logger = logging.getLogger(__name__)
    
    # 构建完整模型
    m = build_linear_regional_model(
        nodes, commodities, years,
        dict_v3_path=dict_v3_path,
        output_dir=output_dir,
        gurobi_log_path=gurobi_log_path,
        population_by_country_year=population_by_country_year,
        income_mult_by_country_year=income_mult_by_country_year,
        macc_path=macc_path,
        land_carbon_price_by_year=land_carbon_price_by_year,
        nutrition_rhs=nutrition_rhs,
        nutrient_per_unit_by_comm=nutrient_per_unit_by_comm,
        land_area_limits=land_area_limits,
        grass_area_by_region_year=grass_area_by_region_year,  # ✅ 传递草地面积
        forest_area_by_region_year=forest_area_by_region_year,  # ✅ 传递森林面积
        yield_t_per_ha_default=yield_t_per_ha_default,
        grassland_method=grassland_method,  # ✅ 传递草地处理方式
        max_growth_rate_per_period=max_growth_rate_per_period,
        max_decline_rate_per_period=max_decline_rate_per_period,
        hist_end_year=hist_end_year,
        hist_max_production=hist_max_production,
        # Phase 2 情景参数
        tax_unit_adder=tax_unit_adder,
        feed_reduction_by=feed_reduction_by,
        ruminant_intake_cap=ruminant_intake_cap,
        ruminant_commodities=ruminant_commodities,
        # 市场失衡限制
        max_slack_rate=max_slack_rate,
        # 单位成本法参数
        unit_cost_data=unit_cost_data,
        baseline_scenario_result=baseline_scenario_result,
        process_cost_mapping=process_cost_mapping,
        cost_calculation_method=cost_calculation_method,
    )
    # 移除时间限制（用户要求：'另外去掉求解时限 300.0 秒'）
    # m.setParam('TimeLimit', time_limit)
    
    logger.info(f"[LINEAR] 开始求解（无时限）...")
    m.optimize()
    
    status = m.Status
    logger.info(f"[LINEAR] 求解完成，状态={status}")
    
    result = {'status': status, 'model': m}
    
    # ===== IIS 分析（当模型不可行时）=====
    if status == gp.GRB.INFEASIBLE:
        logger.warning("[LINEAR] 模型不可行，开始 IIS 分析...")
        
        # 先保存模型供手动分析（在 IIS 之前，因为 IIS 可能因数值问题失败）
        try:
            if output_dir:
                model_path = Path(output_dir) / "linear_model_infeasible.lp"
            else:
                model_path = Path("linear_model_infeasible.lp")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            m.write(str(model_path))
            logger.info(f"[LINEAR] 不可行模型已保存到: {model_path}")
        except Exception as e:
            logger.warning(f"[LINEAR] 无法保存模型文件: {e}")
        
        try:
            m.computeIIS()
            
            # 收集 IIS 中的约束
            iis_constrs = []
            iis_bounds = []
            
            for c in m.getConstrs():
                if c.IISConstr:
                    iis_constrs.append(c.ConstrName)
            
            for v in m.getVars():
                if v.IISLB:
                    iis_bounds.append(f"{v.VarName} (lower bound)")
                if v.IISUB:
                    iis_bounds.append(f"{v.VarName} (upper bound)")
            
            logger.error(f"[LINEAR] IIS 包含 {len(iis_constrs)} 个约束, {len(iis_bounds)} 个变量边界")
            
            # 分类统计 IIS 约束
            iis_by_type = {}
            for cname in iis_constrs:
                # 提取约束类型（方括号前的部分）
                ctype = cname.split('[')[0] if '[' in cname else cname
                iis_by_type[ctype] = iis_by_type.get(ctype, 0) + 1
            
            logger.error("[LINEAR] IIS 约束类型统计:")
            for ctype, count in sorted(iis_by_type.items(), key=lambda x: -x[1]):
                logger.error(f"  {ctype}: {count} 个")
            
            # 显示前10个 IIS 约束详情
            logger.error("[LINEAR] IIS 约束示例 (前20个):")
            for cname in iis_constrs[:20]:
                logger.error(f"  - {cname}")
            
            # 显示 IIS 变量边界
            if iis_bounds:
                logger.error("[LINEAR] IIS 变量边界 (前10个):")
                for vname in iis_bounds[:10]:
                    logger.error(f"  - {vname}")
            
            # 保存 IIS 到结果
            result['iis_constrs'] = iis_constrs
            result['iis_bounds'] = iis_bounds
            result['iis_by_type'] = iis_by_type
            
            # 写入 IIS 文件
            try:
                if output_dir:
                    iis_path = Path(output_dir) / "linear_model_iis.ilp"
                else:
                    iis_path = Path("linear_model_iis.ilp")
                iis_path.parent.mkdir(parents=True, exist_ok=True)
                m.write(str(iis_path))
                logger.info(f"[LINEAR] IIS 已保存到: {iis_path}")
            except Exception as e:
                logger.warning(f"[LINEAR] 无法保存 IIS 文件: {e}")
                
        except Exception as e:
            logger.error(f"[LINEAR] IIS 分析失败: {e}")
    
    if status in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL, gp.GRB.TIME_LIMIT):
        cache = m._nzf_cache
        
        # 基础结果
        result['Pc'] = {k: v.X for k, v in cache['Pc'].items()}
        result['Qs'] = {k: v.X for k, v in cache['Qs'].items()}
        result['Qd'] = {k: v.X for k, v in cache['Qd'].items()}
        result['excess'] = {k: v.X for k, v in cache['excess'].items()}
        result['shortage'] = {k: v.X for k, v in cache['shortage'].items()}
        
        # 排放结果
        result['Eij'] = {k: v.X for k, v in cache.get('Eij', {}).items()}
        result['Cij'] = {k: v.X for k, v in cache.get('Cij', {}).items()}
        
        # 减排结果
        abat_vars = cache.get('abatement_vars', {})
        result['abatement'] = {k: v.X for k, v in abat_vars.items()}
        
        # 汇总统计
        total_supply = sum(result['Qs'].values())
        total_demand = sum(result['Qd'].values())
        total_excess = sum(result['excess'].values())
        total_shortage = sum(result['shortage'].values())
        total_emissions = sum(result['Eij'].values())
        total_abat_cost = sum(result['Cij'].values())
        total_abatement = sum(result['abatement'].values())
        
        # 计算能量短缺（如果有营养系数）
        nutrient_dict = cache.get('nutrient_per_unit_by_comm', {})
        total_energy_supply = 0.0
        total_energy_shortage = 0.0
        if nutrient_dict:
            for (r, j, t), val in result['Qs'].items():
                kcal_per_ton = float(nutrient_dict.get(j, 0.0) or 0.0)
                if kcal_per_ton > 0:
                    total_energy_supply += kcal_per_ton * val
            
            for (j, t), val in result['shortage'].items():
                kcal_per_ton = float(nutrient_dict.get(j, 0.0) or 0.0)
                if kcal_per_ton > 0:
                    total_energy_shortage += kcal_per_ton * val
        
        energy_shortage_ratio = (total_energy_shortage / total_energy_supply * 100) if total_energy_supply > 0 else 0
        
        logger.info(f"[LINEAR] 总供给={total_supply:.2e} kt (跨{len(result['Qs'])}个区域-商品-年份), 总需求={total_demand:.2e} kt")
        logger.info(f"[LINEAR] 总过剩={total_excess:.2e} kt (跨{len(result['excess'])}个商品-年份), 总短缺={total_shortage:.2e} kt")
        if nutrient_dict:
            logger.info(f"[LINEAR] 能量供给={total_energy_supply:.2e} kcal, 能量短缺={total_energy_shortage:.2e} kcal ({energy_shortage_ratio:.1f}%)")
        logger.info(f"[LINEAR] 总排放={total_emissions:.2e} tCO2e (跨{len(result['Eij'])}个区域-商品-年份), 总减排={total_abatement:.2e} tCO2e, 减排成本={total_abat_cost:.2e} USD")
        
        # ✅ 诊断：如果总短缺异常大，输出前10大短缺商品
        if total_shortage > 1e6:  # 短缺 > 1 Mt
            # 计算短缺占总供给的比例（重量）
            shortage_ratio = (total_shortage / total_supply * 100) if total_supply > 0 else 0
            
            shortage_by_comm = {}
            supply_by_comm = {}
            energy_shortage_by_comm = {}
            energy_supply_by_comm = {}
            
            for (j, t), val in result['shortage'].items():
                if val > 0:
                    if j not in shortage_by_comm:
                        shortage_by_comm[j] = 0
                    shortage_by_comm[j] += val
                    
                    # 计算能量短缺
                    if nutrient_dict:
                        kcal_per_ton = float(nutrient_dict.get(j, 0.0) or 0.0)
                        if kcal_per_ton > 0:
                            if j not in energy_shortage_by_comm:
                                energy_shortage_by_comm[j] = 0
                            energy_shortage_by_comm[j] += kcal_per_ton * val
            
            # 同时统计各商品的总供给
            for (r, j, t), val in result['Qs'].items():
                if j not in supply_by_comm:
                    supply_by_comm[j] = 0
                supply_by_comm[j] += val
                
                # 计算能量供给
                if nutrient_dict:
                    kcal_per_ton = float(nutrient_dict.get(j, 0.0) or 0.0)
                    if kcal_per_ton > 0:
                        if j not in energy_supply_by_comm:
                            energy_supply_by_comm[j] = 0
                        energy_supply_by_comm[j] += kcal_per_ton * val
            
            if shortage_by_comm:
                top_shortages = sorted(shortage_by_comm.items(), key=lambda x: x[1], reverse=True)[:10]
                logger.warning(f"[LINEAR] ⚠️ 总短缺={total_shortage:.2e} ({shortage_ratio:.1f}%重量, {energy_shortage_ratio:.1f}%能量) 异常大！前10大短缺商品:")
                for j, val in top_shortages:
                    supply_j = supply_by_comm.get(j, 0)
                    ratio_j = (val / supply_j * 100) if supply_j > 0 else 0
                    
                    # 能量占比
                    energy_short_j = energy_shortage_by_comm.get(j, 0)
                    energy_supply_j = energy_supply_by_comm.get(j, 0)
                    energy_ratio_j = (energy_short_j / energy_supply_j * 100) if energy_supply_j > 0 else 0
                    
                    if nutrient_dict and energy_short_j > 0:
                        logger.warning(f"  - {j}: 短缺={val:.2e} kt ({ratio_j:.1f}%重量, {energy_ratio_j:.1f}%能量)")
                    else:
                        logger.warning(f"  - {j}: 短缺={val:.2e} kt, 供给={supply_j:.2e} kt, 占比={ratio_j:.1f}%")
    
    return result


# =============================================================================
# 蒙特卡洛模拟支持
# =============================================================================

def build_linear_model_cache(
    nodes: List[Any],
    commodities: List[str],
    years: List[int],
    dict_v3_path: Optional[str] = None,
    # ===== 人口与收入 =====
    population_by_country_year: Optional[Dict[Tuple[str, int], float]] = None,
    income_mult_by_country_year: Optional[Dict[Tuple[str, int], float]] = None,
    # ===== 排放与减排参数 =====
    macc_path: Optional[str] = None,
    land_carbon_price_by_year: Optional[Dict[int, float]] = None,
    # ===== 约束参数 =====
    nutrition_rhs: Optional[Dict[Tuple[str, int], float]] = None,
    nutrient_per_unit_by_comm: Optional[Dict[str, float]] = None,
    land_area_limits: Optional[Dict[Tuple[str, int], float]] = None,
    yield_t_per_ha_default: float = 3.0,
    max_growth_rate_per_period: Optional[float] = None,
    max_decline_rate_per_period: Optional[float] = None,
    hist_end_year: int = 2020,
    hist_max_production: Optional[Dict[Tuple[str, str], float]] = None,
    grassland_method: str = 'dynamic',
) -> LinearModelCache:
    """
    构建线性区域模型并返回缓存对象，用于蒙特卡洛模拟
    
    与 S3_0_ds_emis_mc_full.py 的 build_model_cache() 类似：
    1. 构建完整的线性区域模型
    2. 提取所有变量、约束和校准参数
    3. 返回 LinearModelCache 对象
    
    用法：
        cache = build_linear_model_cache(nodes, commodities, years, ...)
        apply_linear_sample_updates(cache, pop_mult=..., yield_mult=..., e0_mult=...)
        cache.model.optimize()
        # 读取结果: cache.Qs[key].X, cache.Qd[key].X, ...
    
    参数：
        nodes: 国家级节点列表
        commodities: 商品列表
        years: 年份列表
        dict_v3_path: dict_v3.xlsx 路径
        population_by_country_year: 人口数据
        income_mult_by_country_year: 收入乘数数据
        macc_path: MACC 数据路径
        land_carbon_price_by_year: 土地碳价
        nutrition_rhs: 营养约束右端项
        nutrient_per_unit_by_comm: 每单位商品营养值
        land_area_limits: 土地面积上限
        yield_t_per_ha_default: 默认产量
        max_growth_rate_per_period: 最大增长率
        max_decline_rate_per_period: 最大下降率
        hist_end_year: 历史期结束年份
        hist_max_production: 历史最大产量锚定 {(region, commodity): max_t}
    
    返回：
        LinearModelCache 对象
    """
    logger = logging.getLogger(__name__)
    
    # 构建模型
    m = build_linear_regional_model(
        nodes=nodes,
        commodities=commodities,
        years=years,
        dict_v3_path=dict_v3_path,
        population_by_country_year=population_by_country_year,
        income_mult_by_country_year=income_mult_by_country_year,
        macc_path=macc_path,
        land_carbon_price_by_year=land_carbon_price_by_year,
        nutrition_rhs=nutrition_rhs,
        nutrient_per_unit_by_comm=nutrient_per_unit_by_comm,
        land_area_limits=land_area_limits,
        yield_t_per_ha_default=yield_t_per_ha_default,
        max_growth_rate_per_period=max_growth_rate_per_period,
        max_decline_rate_per_period=max_decline_rate_per_period,
        hist_end_year=hist_end_year,
        hist_max_production=hist_max_production,
        grassland_method=grassland_method,
    )
    
    # 提取缓存
    c = m._nzf_cache
    
    # 构建 LinearModelCache
    cache = LinearModelCache(
        model=m,
        # 变量
        Pc=c['Pc'],
        Qs=c['Qs'],
        Qd=c['Qd'],
        Eij=c['Eij'],
        Cij=c['Cij'],
        stock=c['stock'],
        excess=c['excess'],
        shortage=c['shortage'],
        # 约束引用
        constr_supply=c['constr_supply'],
        constr_demand=c['constr_demand'],
        constr_Edef=c['constr_Edef'],
        nutri_constr=c['nutri_constr'],
        land_constr=c['land_constr'],
        rumi_intake_constr=c.get('rumi_intake_constr', {}),  # Phase 2: 反刍动物需求上限约束
        # MACC
        abatement_vars=c['abatement_vars'],
        abatement_caps=c['abatement_caps'],
        abatement_costs=c['abatement_costs'],
        proc_cap_basecoeff=c['proc_cap_basecoeff'],
        # 校准参数
        alpha_s=c['alpha_s'],
        alpha_d=c['alpha_d'],
        eps_s=c['eps_s'],
        eps_d=c['eps_d'],
        eps_pop=c['eps_pop'],
        eps_inc=c['eps_inc'],
        eta_y=c['eta_y'],
        eta_temp=c['eta_temp'],
        Q0=c['Q0'],
        D0=c['D0'],
        P0=c['P0'],
        Ymult0=c['Ymult0'],
        Tmult0=c['Tmult0'],
        pop_base=c['pop_base'],
        inc_base=c['inc_base'],
        # 排放强度
        e0_by_region=c['e0_by_region'],
        # 元数据
        regions=c['regions'],
        commodities=c['commodities'],
        years=c['years'],
        idx=c['idx'],
    )
    
    logger.info(f"[LINEAR_MC] 模型缓存构建完成: 变量={m.NumVars}, 约束={m.NumConstrs}")
    logger.info(f"[LINEAR_MC] 区域数={len(c['regions'])}, 商品数={len(c['commodities'])}, 年份数={len(c['years'])}")
    
    return cache


def apply_linear_sample_updates(
    cache: LinearModelCache,
    *,
    pop_mult_by_region: Optional[Dict[str, float]] = None,
    income_mult_by_region: Optional[Dict[str, float]] = None,
    yield_mult_by_region_comm: Optional[Dict[Tuple[str, str], float]] = None,
    temp_mult_by_region_comm: Optional[Dict[Tuple[str, str], float]] = None,
    e0_mult_by_region_comm_proc: Optional[Dict[Tuple[str, str, int, str], float]] = None,
    land_cp_by_year: Optional[Dict[int, float]] = None,
    nutrition_rhs: Optional[Dict[Tuple[str, int], float]] = None,
    land_limits: Optional[Dict[Tuple[str, int], float]] = None,
) -> None:
    """
    原地更新线性模型约束，用于蒙特卡洛采样
    
    与 S3_0_ds_emis_mc_full.py 的 apply_sample_updates() 类似，但适用于线性模型。
    线性模型的约束形式为：
    
    供给约束：Qs = a_s + b_s * Pc + Σ(b_sj * Pc_j)
    需求约束：Qd = a_d + b_d * Pc + Σ(b_dj * Pc_j)
    
    其中 a_s, a_d 包含了产率、温度、人口、收入等效应。
    MC 更新需要修改这些常数项。
    
    更新方式：
    1. 人口乘数 → 修改需求约束的 a_d (RHS)
    2. 收入乘数 → 修改需求约束的 a_d (RHS)
    3. 产率乘数 → 修改供给约束的 a_s (RHS)
    4. 温度乘数 → 修改供给约束的 a_s (RHS)
    5. 排放因子乘数 → 修改 E_def 约束中 Qs 的系数
    
    参数：
        cache: LinearModelCache 对象
        pop_mult_by_region: 人口乘数 {region: mult}，相对于基期
        income_mult_by_region: 收入乘数 {region: mult}，相对于基期
        yield_mult_by_region_comm: 产率乘数 {(region, commodity): mult}
        temp_mult_by_region_comm: 温度乘数 {(region, commodity): mult}
        e0_mult_by_region_comm_proc: 排放因子乘数 {(region, commodity, year, process): mult}
        land_cp_by_year: 土地碳价 {year: price}
        nutrition_rhs: 营养约束右端项 {(region, year): value}
        land_limits: 土地约束 {(region, year): value}
    """
    m = cache.model
    
    # =========================================================================
    # 1. 更新供给约束（产率和温度乘数）
    # =========================================================================
    # 线性供给方程：Qs = Q0 * (1 + η_y*(Ymult-1) + η_temp*(Tmult-1) - ε_s - Σε_sj) + ε_s*Q0/P0*Pc + ...
    # MC 更新产率：Ymult_new = Ymult0 * yield_mult
    # 新的常数项：a_s_new = Q0 * (1 + η_y*(Ymult_new-1) + η_temp*(Tmult_new-1) - ε_s - Σε_sj)
    
    if yield_mult_by_region_comm or temp_mult_by_region_comm:
        for key, con in cache.constr_supply.items():
            r, j, t = key
            
            Q0 = cache.Q0.get(key, 1.0)
            P0 = cache.P0.get(key, 1.0)
            eps_s = cache.eps_s.get(key, 0.0)
            eta_y = cache.eta_y.get(key, 0.0)
            eta_temp = cache.eta_temp.get(key, 0.0)
            Ymult0 = cache.Ymult0.get(key, 1.0)
            Tmult0 = cache.Tmult0.get(key, 1.0)
            
            # 获取乘数
            y_mult = 1.0
            t_mult = 1.0
            if yield_mult_by_region_comm:
                y_mult = float(yield_mult_by_region_comm.get((r, j), 1.0) or 1.0)
            if temp_mult_by_region_comm:
                t_mult = float(temp_mult_by_region_comm.get((r, j), 1.0) or 1.0)
            
            # 新的产率和温度乘数
            Ymult_new = Ymult0 * y_mult
            Tmult_new = Tmult0 * t_mult
            
            # 新的常数项
            yield_adj_new = eta_y * (Ymult_new - 1.0)
            temp_adj_new = eta_temp * (Tmult_new - 1.0)
            
            # 获取原始交叉价格弹性之和
            idx_data = cache.idx.get(key, {})
            epsS_row = idx_data.get('epsS_row', {}) or {}
            sum_cross_eps = sum(float(v) for v in epsS_row.values())
            
            # 新的常数项
            a_s_new = Q0 * (1.0 + yield_adj_new + temp_adj_new - eps_s - sum_cross_eps)
            
            # 更新约束 RHS（线性约束形式：Qs - b_s*Pc - ... = a_s）
            # Gurobi 中线性约束的 RHS 就是常数项
            con.RHS = a_s_new
    
    # =========================================================================
    # 2. 更新需求约束（人口和收入乘数）
    # =========================================================================
    # 线性需求方程：Qd = D0 * pop_effect * inc_effect * (1 - ε_d - Σε_dj) + ε_d*D0_adj/P0*Pc + ...
    # MC 更新：pop_effect_new = (pop_t * pop_mult / pop_base) ^ ε_pop
    
    if pop_mult_by_region or income_mult_by_region:
        for key, con in cache.constr_demand.items():
            r, j, t = key
            
            D0 = cache.D0.get(key, 1.0)
            P0 = cache.P0.get(key, 1.0)
            eps_d = cache.eps_d.get(key, 0.0)
            eps_pop = cache.eps_pop.get(key, 0.0)
            eps_inc = cache.eps_inc.get(key, 0.0)
            pop_base = cache.pop_base.get(key, 1.0)
            inc_base = cache.inc_base.get(key, 1.0)
            
            # 获取乘数
            p_mult = 1.0
            i_mult = 1.0
            if pop_mult_by_region:
                p_mult = float(pop_mult_by_region.get(r, 1.0) or 1.0)
            if income_mult_by_region:
                i_mult = float(income_mult_by_region.get(r, 1.0) or 1.0)
            
            # 获取原始人口和收入比例
            idx_data = cache.idx.get(key, {})
            pop_t = float(idx_data.get('pop_t', pop_base) or pop_base)
            inc_t = float(idx_data.get('inc_t', inc_base) or inc_base)
            
            # 新的人口和收入效应
            pop_ratio_new = (pop_t * p_mult) / max(1e-9, pop_base)
            inc_ratio_new = (inc_t * i_mult) / max(1e-9, inc_base)
            pop_effect_new = pop_ratio_new ** eps_pop if eps_pop != 0 else 1.0
            inc_effect_new = inc_ratio_new ** eps_inc if eps_inc != 0 else 1.0
            
            # 获取交叉价格弹性之和
            epsD_row = idx_data.get('epsD_row', {}) or {}
            sum_cross_eps = sum(float(v) for v in epsD_row.values())
            
            # 新的调整后 D0
            D0_adjusted_new = D0 * pop_effect_new * inc_effect_new
            
            # 新的常数项
            a_d_new = D0_adjusted_new * (1.0 - eps_d - sum_cross_eps)
            
            # 更新约束 RHS
            con.RHS = a_d_new
    
    # =========================================================================
    # 3. 更新排放定义约束（排放因子乘数）
    # =========================================================================
    # E_def: Eij = Σ(e0_p * Qs) - Σ(abatement)
    # MC 更新：e0_p_new = e0_p * mult
    
    if e0_mult_by_region_comm_proc:
        for key, con in cache.constr_Edef.items():
            r, j, t = key
            
            e0_map = cache.e0_by_region.get(key, {})
            if not e0_map:
                continue
            
            # 计算新的总排放强度
            new_sum_e0 = 0.0
            for proc, e0p in e0_map.items():
                mult = float(e0_mult_by_region_comm_proc.get((r, j, t, proc), 1.0) or 1.0)
                new_sum_e0 += float(e0p) * mult
            
            # 更新 Qs 变量的系数
            qs_var = cache.Qs.get(key)
            if qs_var is not None:
                m.chgCoeff(con, qs_var, new_sum_e0)
        
        # 同时更新 MACC 减排上限约束的系数
        for cap_key, cap_con in cache.abatement_caps.items():
            r, j, t, proc, seg = cap_key
            
            base_coeff = cache.proc_cap_basecoeff.get(cap_key, 0.0)
            mult = float(e0_mult_by_region_comm_proc.get((r, j, t, proc), 1.0) or 1.0)
            new_coeff = base_coeff * mult
            
            qs_var = cache.Qs.get((r, j, t))
            if qs_var is not None:
                # MACC 约束形式: a - coeff * Qs <= 0，所以系数是负的
                m.chgCoeff(cap_con, qs_var, -new_coeff)
    
    # =========================================================================
    # 4. 更新营养约束 RHS
    # =========================================================================
    if nutrition_rhs:
        for (r, t), con in cache.nutri_constr.items():
            rhs = nutrition_rhs.get((r, t))
            if rhs is not None:
                con.RHS = float(rhs)
    
    # =========================================================================
    # 5. 更新土地约束 RHS
    # =========================================================================
    if land_limits:
        for (r, t), con in cache.land_constr.items():
            lim = land_limits.get((r, t))
            if lim is not None:
                con.RHS = float(lim)
    
    # =========================================================================
    # 5.5 更新反刍动物需求上限约束 RHS (Phase 2)
    # =========================================================================
    # rumi_intake_constr 的 RHS 可以通过 MC 采样更新
    # 注意：这里假设 cache 中有 rumi_intake_constr 字段
    # 如果需要更新，可以传入 ruminant_cap_by_region_year 参数
    
    # =========================================================================
    # 6. 更新目标函数（土地碳价）
    # =========================================================================
    if land_cp_by_year is not None:
        # 重建目标函数
        obj = gp.LinExpr(0.0)
        
        # 保持原有的 slack 惩罚
        SLACK_PENALTY = 1e6
        for j in cache.commodities:
            for t in cache.years:
                if (j, t) in cache.excess:
                    obj += SLACK_PENALTY * cache.excess[j, t]
                if (j, t) in cache.shortage:
                    obj += SLACK_PENALTY * cache.shortage[j, t]
                if (j, t) in cache.stock:
                    obj += 0.01 * cache.stock[j, t]
        
        # 保持减排成本
        for key, cij_var in cache.Cij.items():
            obj += cij_var
        
        # 新的土地碳价
        for key, qs_var in cache.Qs.items():
            r, j, t = key
            cp = float(land_cp_by_year.get(t, 0.0) or 0.0)
            if cp > 0:
                e0_map = cache.e0_by_region.get(key, {})
                e_land = sum(float(v) for p, v in e0_map.items() if _is_lulucf_process(p))
                if e_land > 0:
                    obj += cp * e_land * qs_var
        
        m.setObjective(obj, gp.GRB.MINIMIZE)
    
    # 更新模型
    m.update()


def run_linear_mc(
    nodes: List[Any],
    commodities: List[str],
    years: List[int],
    specs_df: pd.DataFrame,
    universe: Any,
    *,
    n_samples: int = 100,
    seed: int = 42,
    dict_v3_path: Optional[str] = None,
    macc_path: Optional[str] = None,
    land_carbon_price_by_year: Optional[Dict[int, float]] = None,
    population_by_country_year: Optional[Dict[Tuple[str, int], float]] = None,
    income_mult_by_country_year: Optional[Dict[Tuple[str, int], float]] = None,
    save_prefix: str = 'mc_linear_results',
    time_limit_per_sample: float = 60.0,
    grassland_method: str = 'dynamic',
) -> pd.DataFrame:
    """
    运行线性区域模型的蒙特卡洛模拟
    
    与 S3_0_ds_emis_mc_full.py 的 run_mc() 类似，但使用线性区域模型。
    
    参数：
        nodes: 国家级节点列表
        commodities: 商品列表
        years: 年份列表
        specs_df: MC 规格表（来自 Scenario_config.xlsx 的 MC sheet）
        universe: Universe 对象（包含商品分类等信息）
        n_samples: 采样数
        seed: 随机种子
        dict_v3_path: dict_v3.xlsx 路径
        macc_path: MACC 数据路径
        land_carbon_price_by_year: 土地碳价
        population_by_country_year: 人口数据
        income_mult_by_country_year: 收入数据
        save_prefix: 输出文件前缀
        time_limit_per_sample: 每个样本的求解时限
    
    返回：
        DataFrame 包含所有样本的结果
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"[LINEAR_MC] 开始蒙特卡洛模拟: n_samples={n_samples}, seed={seed}")
    
    # 构建模型缓存
    cache = build_linear_model_cache(
        nodes=nodes,
        commodities=commodities,
        years=years,
        dict_v3_path=dict_v3_path,
        macc_path=macc_path,
        land_carbon_price_by_year=land_carbon_price_by_year,
        population_by_country_year=population_by_country_year,
        income_mult_by_country_year=income_mult_by_country_year,
        grassland_method=grassland_method,
    )
    
    m = cache.model
    m.setParam('OutputFlag', 0)
    m.setParam('TimeLimit', time_limit_per_sample)
    
    # 解析 MC 规格
    df = specs_df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    
    # 必需列
    c_elem = 'Element'
    c_proc = 'Process' if 'Process' in df.columns else None
    c_item = 'Item' if 'Item' in df.columns else None
    c_min = 'Min_bound'
    c_max = 'Max_bound'
    c_region = 'Region_cat' if 'Region_cat' in df.columns else None
    
    if not all(c in df.columns for c in [c_elem, c_min, c_max]):
        logger.warning("[LINEAR_MC] 规格表缺少必要列")
        return pd.DataFrame()
    
    rng = np.random.default_rng(seed)
    recs = []
    
    for s in range(1, n_samples + 1):
        logger.info(f"[LINEAR_MC] 样本 {s}/{n_samples}")
        
        # 为本次采样构建乘数字典
        yield_mult: Dict[Tuple[str, str], float] = {}
        e0_mult: Dict[Tuple[str, str, int, str], float] = {}
        
        for r in df.itertuples(index=False):
            elem = str(getattr(r, c_elem, '')).lower()
            proc = str(getattr(r, c_proc, 'All')) if c_proc else 'All'
            item = str(getattr(r, c_item, 'All')) if c_item else 'All'
            lo = float(getattr(r, c_min, 1.0))
            hi = float(getattr(r, c_max, 1.0))
            regc = str(getattr(r, c_region, 'All')) if c_region else 'All'
            
            if hi < lo:
                lo, hi = hi, lo
            draw = float(rng.uniform(lo, hi))
            
            # 确定目标区域
            if regc and regc.lower() != 'all':
                target_regions = [regc]
            else:
                target_regions = cache.regions
            
            # 确定目标商品
            if item and item.lower() != 'all':
                # 检查是否是分类（crop/meat/dairy/other）
                if item.lower() in ('crop', 'meat', 'dairy', 'other'):
                    cat2 = getattr(universe, 'item_cat2_by_commodity', {}) or {}
                    target_comms = [c for c in cache.commodities 
                                   if cat2.get(c, '').lower() == item.lower()]
                else:
                    target_comms = [item] if item in cache.commodities else []
            else:
                target_comms = cache.commodities
            
            # 确定目标过程
            if proc and proc.lower() != 'all':
                target_procs = [proc]
            else:
                target_procs = list(getattr(universe, 'processes', []) or [])
            
            # 应用抽样值
            if 'yield' in elem or 'productivity' in elem:
                for reg in target_regions:
                    for comm in target_comms:
                        yield_mult[(reg, comm)] = draw
            
            elif 'ef' in elem or 'emission' in elem:
                for reg in target_regions:
                    for comm in target_comms:
                        for p in target_procs:
                            for t in cache.years:
                                e0_mult[(reg, comm, t, p)] = draw
        
        # 应用更新
        apply_linear_sample_updates(
            cache,
            yield_mult_by_region_comm=yield_mult if yield_mult else None,
            e0_mult_by_region_comm_proc=e0_mult if e0_mult else None,
        )
        
        # 求解
        m.optimize()
        
        status = m.Status
        row = {'sample': s, 'status': status}
        
        if status == gp.GRB.OPTIMAL:
            # 汇总结果
            tot_E = sum(v.X for v in cache.Eij.values())
            tot_C = sum(v.X for v in cache.Cij.values())
            tot_Qs = sum(v.X for v in cache.Qs.values())
            tot_Qd = sum(v.X for v in cache.Qd.values())
            
            row['E_total'] = tot_E
            row['C_total'] = tot_C
            row['Qs_total'] = tot_Qs
            row['Qd_total'] = tot_Qd
            
            # 价格
            for (j, t), v in cache.Pc.items():
                row[f'{j}__Pc_{t}'] = v.X
            
            # 可选：保存详细区域结果
            # for (r, j, t), v in cache.Qs.items():
            #     row[f'{r}::{j}::{t}__Qs'] = v.X
        
        recs.append(row)
        
        # 重置约束（恢复基准值）
        # 注意：这里简化处理，实际应该重建模型或保存/恢复原始系数
    
    out = pd.DataFrame(recs)
    
    # 保存结果
    if save_prefix:
        out.to_csv(f'{save_prefix}__samples.csv', index=False, encoding='utf-8-sig')
        
        # 计算统计量
        num = out.drop(columns=['sample', 'status'], errors='ignore').apply(pd.to_numeric, errors='coerce')
        if num.shape[1] > 0 and num.shape[0] > 0:
            qs = num.quantile([0.5, 0.05, 0.95])
            qs.index = ['median', 'p05', 'p95']
            qs.to_csv(f'{save_prefix}__summary.csv', encoding='utf-8-sig')
    
    logger.info(f"[LINEAR_MC] 蒙特卡洛模拟完成: 成功样本={sum(1 for r in recs if r['status'] == gp.GRB.OPTIMAL)}/{n_samples}")
    
    return out


# =============================================================================
# 区域结果分解回国家级
# =============================================================================

def disaggregate_to_countries(
    nodes: List[Any],
    regional_result: Dict[str, Any],
    dict_v3_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    将区域级结果分解回国家级
    
    分解规则：
    - 价格 Pc[j,t]: 全球统一，直接赋值给所有国家
    - 区域供给 Qs[r,j,t] → 国家供给 Qs[i,j,t]: 按基期 Q0 比例分配
    - 区域需求 Qd[r,j,t] → 国家需求 Qd[i,j,t]: 按基期 D0 比例分配
    - 区域排放 E[r,j,t] → 国家排放 E[i,j,t]: 按基期 Q0 比例分配
    - 区域减排成本 C[r,j,t] → 国家成本 C[i,j,t]: 按基期 Q0 比例分配
    
    参数:
        nodes: 原始国家级节点列表
        regional_result: solve_linear_regional() 的返回结果
        dict_v3_path: dict_v3.xlsx 路径（用于区域映射）
        
    返回:
        country_result: {
            'Pc': {(country, commodity, year): price},
            'Qs': {(country, commodity, year): supply},
            'Qd': {(country, commodity, year): demand},
            'Eij': {(country, commodity, year): emissions},
            'Cij': {(country, commodity, year): abatement_cost},
        }
    """
    if regional_result.get('status') not in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL, gp.GRB.TIME_LIMIT):
        return {'status': regional_result.get('status'), 'Pc': {}, 'Qs': {}, 'Qd': {}, 'Eij': {}, 'Cij': {}}
    
    Pc_regional = regional_result.get('Pc', {})
    Qs_regional = regional_result.get('Qs', {})
    Qd_regional = regional_result.get('Qd', {})
    Eij_regional = regional_result.get('Eij', {})
    Cij_regional = regional_result.get('Cij', {})
    
    # 1. 计算每个区域内各国的基期份额
    supply_shares: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    demand_shares: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    
    for n in nodes:
        m49 = getattr(n, 'm49', None) or getattr(n, 'M49_Country_Code', None)
        region = get_region(n.country, m49=m49, dict_v3_path=dict_v3_path)
        key = (region, n.commodity, n.year)
        
        if key not in supply_shares:
            supply_shares[key] = {}
        q0 = getattr(n, 'Q0', 0.0) or 0.0
        supply_shares[key][n.country] = supply_shares[key].get(n.country, 0.0) + q0
        
        if key not in demand_shares:
            demand_shares[key] = {}
        d0 = getattr(n, 'D0', 0.0) or 0.0
        demand_shares[key][n.country] = demand_shares[key].get(n.country, 0.0) + d0
    
    # 2. 分解到国家级
    Pc_country: Dict[Tuple[str, str, int], float] = {}
    Qs_country: Dict[Tuple[str, str, int], float] = {}
    Qd_country: Dict[Tuple[str, str, int], float] = {}
    Eij_country: Dict[Tuple[str, str, int], float] = {}
    Cij_country: Dict[Tuple[str, str, int], float] = {}
    
    for n in nodes:
        m49 = getattr(n, 'm49', None) or getattr(n, 'M49_Country_Code', None)
        region = get_region(n.country, m49=m49, dict_v3_path=dict_v3_path)
        key = (region, n.commodity, n.year)
        country_key = (n.country, n.commodity, n.year)
        
        # 价格：全球统一
        price_key = (n.commodity, n.year)
        if price_key in Pc_regional:
            Pc_country[country_key] = Pc_regional[price_key]
        
        # 供给分解
        regional_key = (region, n.commodity, n.year)
        if regional_key in Qs_regional:
            regional_Qs = Qs_regional[regional_key]
            total_Q0 = sum(supply_shares.get(key, {}).values())
            if total_Q0 > 1e-9:
                country_Q0 = supply_shares.get(key, {}).get(n.country, 0.0)
                share = country_Q0 / total_Q0
                Qs_country[country_key] = regional_Qs * share
                
                # 排放和成本也按供给份额分解
                if regional_key in Eij_regional:
                    Eij_country[country_key] = Eij_regional[regional_key] * share
                if regional_key in Cij_regional:
                    Cij_country[country_key] = Cij_regional[regional_key] * share
            else:
                n_countries = len(supply_shares.get(key, {}))
                if n_countries > 0:
                    Qs_country[country_key] = regional_Qs / n_countries
                    if regional_key in Eij_regional:
                        Eij_country[country_key] = Eij_regional[regional_key] / n_countries
                    if regional_key in Cij_regional:
                        Cij_country[country_key] = Cij_regional[regional_key] / n_countries
        
        # 需求分解
        if regional_key in Qd_regional:
            regional_Qd = Qd_regional[regional_key]
            total_D0 = sum(demand_shares.get(key, {}).values())
            if total_D0 > 1e-9:
                country_D0 = demand_shares.get(key, {}).get(n.country, 0.0)
                share = country_D0 / total_D0
                Qd_country[country_key] = regional_Qd * share
            else:
                n_countries = len(demand_shares.get(key, {}))
                if n_countries > 0:
                    Qd_country[country_key] = regional_Qd / n_countries
    
    return {
        'status': regional_result.get('status'),
        'Pc': Pc_country,
        'Qs': Qs_country,
        'Qd': Qd_country,
        'Eij': Eij_country,
        'Cij': Cij_country,
    }


def apply_results_to_nodes(
    nodes: List[Any],
    country_result: Dict[str, Any],
) -> None:
    """
    将国家级结果应用到节点上（原地修改）
    
    设置节点的以下属性：
    - P: 出清价格
    - Q: 出清供给量
    - D: 出清需求量
    - E: 排放量
    - abatement_cost: 减排成本
    """
    Pc = country_result.get('Pc', {})
    Qs = country_result.get('Qs', {})
    Qd = country_result.get('Qd', {})
    Eij = country_result.get('Eij', {})
    Cij = country_result.get('Cij', {})
    
    for n in nodes:
        key = (n.country, n.commodity, n.year)
        
        if key in Pc:
            n.P = Pc[key]
        if key in Qs:
            n.Q = Qs[key]
        if key in Qd:
            n.D = Qd[key]
        if key in Eij:
            n.E = Eij[key]
        if key in Cij:
            n.abatement_cost = Cij[key]


# =============================================================================
# 便捷函数：完整流程
# =============================================================================

def run_linear_regional_model(
    nodes: List[Any],
    commodities: List[str],
    years: List[int],
    dict_v3_path: Optional[str] = None,
    time_limit: float = 300.0,
    # ===== 排放与减排参数 =====
    macc_path: Optional[str] = None,
    land_carbon_price_by_year: Optional[Dict[int, float]] = None,
    # ===== 约束参数 =====
    nutrition_rhs: Optional[Dict[Tuple[str, int], float]] = None,
    nutrient_per_unit_by_comm: Optional[Dict[str, float]] = None,
    land_area_limits: Optional[Dict[Tuple[str, int], float]] = None,
    yield_t_per_ha_default: float = 3.0,
    max_growth_rate_per_period: Optional[float] = None,
    max_decline_rate_per_period: Optional[float] = None,
    hist_end_year: int = 2020,
    apply_to_nodes: bool = True,
) -> Dict[str, Any]:
    """
    运行完整的线性区域模型流程
    
    步骤：
    1. 构建并求解区域级模型
    2. 分解结果到国家级
    3. （可选）应用结果到节点
    
    返回：
    - regional_result: 区域级结果
    - country_result: 国家级结果
    - status: 求解状态
    """
    logger = logging.getLogger(__name__)
    
    # 1. 求解区域模型
    logger.info("[LINEAR] 步骤 1: 求解区域模型...")
    regional_result = solve_linear_regional(
        nodes=nodes,
        commodities=commodities,
        years=years,
        time_limit=time_limit,
        dict_v3_path=dict_v3_path,
        macc_path=macc_path,
        land_carbon_price_by_year=land_carbon_price_by_year,
        nutrition_rhs=nutrition_rhs,
        nutrient_per_unit_by_comm=nutrient_per_unit_by_comm,
        land_area_limits=land_area_limits,
        yield_t_per_ha_default=yield_t_per_ha_default,
        max_growth_rate_per_period=max_growth_rate_per_period,
        max_decline_rate_per_period=max_decline_rate_per_period,
        hist_end_year=hist_end_year,
    )
    
    status = regional_result.get('status')
    if status not in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL, gp.GRB.TIME_LIMIT):
        logger.warning(f"[LINEAR] 求解失败，状态={status}")
        return {'status': status, 'regional_result': regional_result, 'country_result': None}
    
    # 2. 分解到国家级
    logger.info("[LINEAR] 步骤 2: 分解到国家级...")
    country_result = disaggregate_to_countries(
        nodes=nodes,
        regional_result=regional_result,
        dict_v3_path=dict_v3_path,
    )
    
    # 3. 应用到节点
    if apply_to_nodes:
        logger.info("[LINEAR] 步骤 3: 应用结果到节点...")
        apply_results_to_nodes(nodes, country_result)
    
    logger.info("[LINEAR] 完成！")
    
    return {
        'status': status,
        'regional_result': regional_result,
        'country_result': country_result,
    }


# =============================================================================
# 方案B: 草地迭代框架（grassland_method='static'）
# =============================================================================

def solve_with_grassland_iteration(
    nodes: List[Any],
    commodities: List[str],
    years: List[int],
    time_limit: float = 300.0,
    dict_v3_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    gurobi_log_path: Optional[str] = None,
    population_by_country_year: Optional[Dict[Tuple[str, int], float]] = None,
    income_mult_by_country_year: Optional[Dict[Tuple[str, int], float]] = None,
    macc_path: Optional[str] = None,
    land_carbon_price_by_year: Optional[Dict[int, float]] = None,
    nutrition_rhs: Optional[Dict[Tuple[str, int], float]] = None,
    nutrient_per_unit_by_comm: Optional[Dict[str, float]] = None,
    land_area_limits: Optional[Dict[Tuple[str, int], float]] = None,
    grass_area_by_region_year_initial: Optional[Dict[Tuple[str, int], float]] = None,
    forest_area_by_region_year: Optional[Dict[Tuple[str, int], float]] = None,
    yield_t_per_ha_default: float = 3.0,
    max_growth_rate_per_period: Optional[float] = None,
    max_decline_rate_per_period: Optional[float] = None,
    hist_end_year: int = 2020,
    hist_max_production: Optional[Dict[Tuple[str, str], float]] = None,
    max_iterations: int = 10,
    convergence_tolerance: float = 0.01,
    damping_factor: float = 0.5,
) -> Dict:
    """
    方案B：草地面积作为外生参数的迭代求解框架
    
    迭代流程：
    1. Round 1: 使用初始grassland面积求解模型 → 得到优化后的 Qs
    2. 计算新的grassland需求：通过 livestock产量 → stock → feed demand → grassland_ha
    3. Round 2: 使用更新后的grassland重新求解 → 得到新的 Qs
    4. 重复步骤2-3直到收敛（grassland变化 < tolerance 或达到max_iterations）
    
    收敛判据：
    - max(Δ_grassland / grassland_prev) < convergence_tolerance (相对变化 < 1%)
    - max(Δ_objective / objective_prev) < convergence_tolerance (目标函数变化 < 1%)
    
    阻尼策略：
    - grassland_new = damping_factor * grassland_calculated + (1-damping) * grassland_old
    - 防止振荡，提高收敛稳定性
    
    Args:
        nodes: 国家级节点列表
        ... (其他参数与 solve_linear_regional 相同)
        grass_area_by_region_year_initial: 初始草地面积 {(region, year): ha}
        max_iterations: 最大迭代次数（默认10）
        convergence_tolerance: 收敛容差（默认1%）
        damping_factor: 阻尼系数（默认0.5）
    
    Returns:
        与 solve_linear_regional 相同的结果字典，额外包含：
        - 'iterations': 实际迭代次数
        - 'converged': 是否收敛（True/False）
        - 'grassland_history': 每轮迭代的草地面积历史
    """
    logger = logging.getLogger(__name__)
    logger.info(f"[GRASSLAND_ITER] 开始方案B迭代求解（最大迭代次数={max_iterations}，收敛容差={convergence_tolerance*100:.1f}%）")
    
    # 初始化草地面积
    grass_area_current = dict(grass_area_by_region_year_initial or {})
    grass_area_history = [dict(grass_area_current)]
    
    objective_prev = None
    converged = False
    
    for iteration in range(1, max_iterations + 1):
        logger.info(f"\n[GRASSLAND_ITER] ===== 迭代 {iteration}/{max_iterations} =====")
        
        # 求解线性模型（使用当前grassland面积）
        result = solve_linear_regional(
            nodes=nodes,
            commodities=commodities,
            years=years,
            time_limit=time_limit,
            dict_v3_path=dict_v3_path,
            output_dir=output_dir,
            gurobi_log_path=gurobi_log_path,
            population_by_country_year=population_by_country_year,
            income_mult_by_country_year=income_mult_by_country_year,
            macc_path=macc_path,
            land_carbon_price_by_year=land_carbon_price_by_year,
            nutrition_rhs=nutrition_rhs,
            nutrient_per_unit_by_comm=nutrient_per_unit_by_comm,
            land_area_limits=land_area_limits,
            grass_area_by_region_year=grass_area_current,  # 使用当前草地面积
            forest_area_by_region_year=forest_area_by_region_year,
            yield_t_per_ha_default=yield_t_per_ha_default,
            grassland_method='static',  # 强制使用方案B
            max_growth_rate_per_period=max_growth_rate_per_period,
            max_decline_rate_per_period=max_decline_rate_per_period,
            hist_end_year=hist_end_year,
            hist_max_production=hist_max_production,
        )
        
        if result['status'] != 2:  # Not OPTIMAL
            logger.warning(f"[GRASSLAND_ITER] 迭代{iteration}未达到最优状态（status={result['status']}），终止迭代")
            result['iterations'] = iteration
            result['converged'] = False
            result['grassland_history'] = grass_area_history
            return result
        
        objective_current = result.get('objective', 0.0)
        logger.info(f"[GRASSLAND_ITER] 迭代{iteration}完成，目标函数={objective_current:.2f}")
        
        # 检查目标函数收敛
        if objective_prev is not None:
            obj_change = abs(objective_current - objective_prev) / max(abs(objective_prev), 1e-6)
            logger.info(f"[GRASSLAND_ITER] 目标函数变化: {obj_change*100:.3f}%")
            if obj_change < convergence_tolerance:
                logger.info(f"[GRASSLAND_ITER] ✅ 目标函数收敛（变化 < {convergence_tolerance*100:.1f}%）")
                converged = True
        
        # 从优化结果计算新的grassland需求
        # 需要：1) 优化后的livestock产量Qs；2) livestock→stock转换；3) stock→grassland转换
        try:
            # 提取livestock商品的优化产量
            from S3_2_feed_demand import build_feed_demand_from_stock
            from gle_emissions_complete import calculate_stock_from_optimized_production
            from S2_0_load_data import DataPaths, load_emis_item_mappings
            from S1_0_schema import Universe
            
            # 构建universe（简化版）
            universe = Universe(
                countries=sorted(set(n.country for n in nodes)),
                iso3_by_country={n.country: getattr(n, 'iso3', '') for n in nodes},
                commodities=sorted(set(n.commodity for n in nodes)),
                years=sorted(set(n.year for n in nodes)),
            )
            
            paths = DataPaths()
            maps = load_emis_item_mappings(dict_v3_path or paths.dict_v3_path)
            
            # 计算新的stock
            country_result = result.get('country_result', {})
            qs_optimized = country_result.get('Qs', {})
            
            stock_df = calculate_stock_from_optimized_production(
                optimized_qs=qs_optimized,
                nodes=nodes,
                universe=universe,
                maps=maps,
                paths=paths
            )
            
            # 从stock计算feed demand（包括grassland需求）
            feed_outputs = build_feed_demand_from_stock(
                stock_df=stock_df,
                universe=universe,
                maps=maps,
                paths=paths,
                years=years
            )
            
            grass_req_df = feed_outputs.grass_requirement
            
            if grass_req_df.empty:
                logger.warning(f"[GRASSLAND_ITER] 迭代{iteration}：未计算到grassland需求，使用旧值")
                grass_area_new = dict(grass_area_current)
            else:
                # 聚合到区域级
                # grass_req_df: [country, iso3, year, grass_tdm, grass_area_need_ha]
                grass_req_df['region'] = grass_req_df['country'].apply(
                    lambda c: get_region(c, dict_v3_path=dict_v3_path)
                )
                
                grass_area_new_df = grass_req_df.groupby(['region', 'year'])['grass_area_need_ha'].sum().reset_index()
                grass_area_new = {
                    (str(row['region']), int(row['year'])): float(row['grass_area_need_ha'])
                    for _, row in grass_area_new_df.iterrows()
                }
                
                # 阻尼更新（防止振荡）
                grass_area_damped = {}
                for key in set(grass_area_current.keys()) | set(grass_area_new.keys()):
                    old_val = grass_area_current.get(key, 0.0)
                    new_val = grass_area_new.get(key, 0.0)
                    damped_val = damping_factor * new_val + (1 - damping_factor) * old_val
                    grass_area_damped[key] = damped_val
                
                grass_area_new = grass_area_damped
            
            # 检查grassland收敛
            grass_changes = []
            for key in grass_area_current.keys():
                old_val = grass_area_current[key]
                new_val = grass_area_new.get(key, 0.0)
                if old_val > 1e-6:
                    rel_change = abs(new_val - old_val) / old_val
                    grass_changes.append(rel_change)
            
            if grass_changes:
                max_grass_change = max(grass_changes)
                avg_grass_change = sum(grass_changes) / len(grass_changes)
                logger.info(f"[GRASSLAND_ITER] Grassland变化: max={max_grass_change*100:.3f}%, avg={avg_grass_change*100:.3f}%")
                
                if max_grass_change < convergence_tolerance and converged:
                    logger.info(f"[GRASSLAND_ITER] ✅✅ Grassland和目标函数均收敛，迭代完成")
                    result['iterations'] = iteration
                    result['converged'] = True
                    result['grassland_history'] = grass_area_history
                    return result
            
            # 更新草地面积供下一轮迭代
            grass_area_current = grass_area_new
            grass_area_history.append(dict(grass_area_current))
            objective_prev = objective_current
            
        except Exception as e:
            logger.error(f"[GRASSLAND_ITER] 迭代{iteration}：计算新grassland失败: {e}")
            import traceback
            traceback.print_exc()
            # 继续使用旧grassland值
            grass_area_history.append(dict(grass_area_current))
    
    # 达到最大迭代次数
    logger.warning(f"[GRASSLAND_ITER] ⚠️ 达到最大迭代次数{max_iterations}，未完全收敛")
    result['iterations'] = max_iterations
    result['converged'] = False
    result['grassland_history'] = grass_area_history
    return result


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    
    # 创建测试数据
    from dataclasses import dataclass
    
    @dataclass
    class MockNode:
        country: str
        commodity: str
        year: int
        Q0: float = 1000.0
        D0: float = 1000.0
        P0: float = 100.0
        eps_supply: float = 0.3
        eps_demand: float = -0.5
        eps_pop_demand: float = 0.0
        eps_income_demand: float = 0.0
        e0_by_proc: Dict = None
        
        def __post_init__(self):
            if self.e0_by_proc is None:
                self.e0_by_proc = {'Enteric fermentation': 0.5, 'Manure management': 0.2}
    
    # 生成测试节点
    test_countries = ['United States of America', 'China', 'Brazil', 'Germany', 'India']
    test_commodities = ['Wheat', 'Rice', 'Maize (corn)']
    test_years = [2020, 2050]
    
    nodes = []
    for c in test_countries:
        for j in test_commodities:
            for t in test_years:
                nodes.append(MockNode(
                    country=c, commodity=j, year=t,
                    Q0=1000 + np.random.rand() * 500,
                    D0=1000 + np.random.rand() * 500,
                ))
    
    print(f"测试节点数: {len(nodes)}")
    
    # 求解
    result = solve_linear_regional(nodes, test_commodities, test_years, time_limit=60)
    
    print(f"\n状态: {result['status']}")
    if 'Pc' in result:
        print("\n价格结果:")
        for (j, t), p in sorted(result['Pc'].items()):
            print(f"  {j}, {t}: {p:.2f}")
