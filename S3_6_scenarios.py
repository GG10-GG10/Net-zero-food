
# -*- coding: utf-8 -*-
"""
S3.6_scenarios.py — 读取/应用“情景配置”表（Scenario_config_new.xlsx）
支持维度：Country（All/Region_aggMC/逐国），Commodity（All/crop/meat/dairy/other/逐commodity），Emis process（All/逐过程）
支持 Unit: 'rate'（2080 相对 2020 的比例变化；线性插值到各未来年）、'amount'（绝对值）
示例类型：
- Feed efficiency （rate）：2080 相较 2020 提升 30% → 未来按线性从 0% 到 30% 降低饲料需求（用于 feed 量乘以 (1 - r(t))）
- Land carbon price （amount，$/tCO2e）：用于 LUC 集约化代理与供应侧“单位税” tax_unit（乘以 LULUCF 强度）
- Ruminant intake decreasing ratio （rate）：对 ruminant 肉类建立需求上限 Qd ≤ (1 - r(t))·Qd_2020（按国、年）
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

from S1_0_schema import Node, Universe

RUMINANT_COMMS = [
    'Meat of cattle with the bone, fresh or chilled',
    'Meat of buffalo, fresh or chilled',
    'Meat of sheep, fresh or chilled',
    'Meat of goat, fresh or chilled'
]

@dataclass
class ScenarioEffect:
    scenario_id: str
    kind: str                 # 'feed_efficiency', 'land_carbon_price', 'ruminant_intake_ratio'
    unit: str                 # 'rate' or 'amount'
    value_2080: float
    country_sel: str          # 'All' or Region_aggMC or specific country (name)
    commodity_sel: str        # 'All' or Cat2 bucket ('crop'/'meat'/'dairy'/'other') or exact commodity
    process_sel: str          # 'All' or exact process
    # 解析后的集合
    countries: List[str] = None
    commodities: List[str] = None
    processes: List[str] = None

def _linear_path_2020_2080(value_2080: float, unit: str, years: List[int]) -> Dict[int, float]:
    # 返回 {year: f(year)}；2020=0（rate）或基准（amount）→按线性到 2080
    out = {}
    for y in years:
        if y <= 2020:
            out[y] = 0.0 if unit=='rate' else np.nan
        else:
            frac = (y - 2020) / (2080 - 2020)
            out[y] = value_2080 * frac if unit=='rate' else value_2080  # amount 用恒定值
    return out

def _select_countries(universe: Universe, key: str) -> List[str]:
    if not key or str(key).lower()=='all':
        return list(universe.countries)
    # Region_aggMC
    inv = {}
    for c, r in (universe.region_aggMC_by_country or {}).items():
        inv.setdefault(str(r), []).append(c)
    if key in inv:
        return inv[key]
    # 单国
    return [key] if key in universe.countries else []

def _select_commodities(universe: Universe, key: str) -> List[str]:
    if not key or str(key).lower()=='all':
        return list(universe.commodities)
    k = str(key).lower()
    if k in ('crop','meat','dairy','other'):
        return [c for c in universe.commodities if (universe.item_cat2_by_commodity or {}).get(c, '').lower()==k]
    # 单个商品名
    return [key] if key in universe.commodities else []

def _select_processes(universe: Universe, key: str) -> List[str]:
    if not key or str(key).lower()=='all':
        return list(universe.processes)
    return [key] if key in universe.processes else []

def load_scenario_config(xlsx_path: str, universe: Universe) -> List[ScenarioEffect]:
    try:
        df = pd.read_excel(xlsx_path, sheet_name=0)
    except Exception:
        return []
    df.columns = [str(c).strip() for c in df.columns]
    reqs = ['Scenario ID','Scenario Type','Scenario Unit','Value','Country','Commodity','Emis process']
    miss = [c for c in reqs if c not in df.columns]
    if miss:
        # 容错：尽量匹配大小写
        cols = {c.lower(): c for c in df.columns}
        reqs2 = [cols.get(c.lower(), c) for c in reqs]
        df = df.rename(columns={cols.get(c.lower(), c): c for c in df.columns})
    effects: List[ScenarioEffect] = []
    for r in df.itertuples(index=False):
        sid  = getattr(r, 'Scenario ID')
        kind = str(getattr(r, 'Scenario Type')).strip().lower().replace(' ', '_')
        unit = str(getattr(r, 'Scenario Unit')).strip().lower()
        val  = float(getattr(r, 'Value'))
        cty  = getattr(r, 'Country')
        comm = getattr(r, 'Commodity')
        proc = getattr(r, 'Emis process', 'All')
        eff = ScenarioEffect(sid, kind, unit, val, cty, comm, proc)
        eff.countries   = _select_countries(universe, cty)
        eff.commodities = _select_commodities(universe, comm)
        eff.processes   = _select_processes(universe, proc)
        effects.append(eff)
    return effects

def apply_scenario_to_data(effects: List[ScenarioEffect], scenario_id: str, universe: Universe, nodes: List[Node],
                           *, base_year:int=2020) -> Dict[str, Dict]:
    """返回按年份的情景参数字典，供 S4.0_main 下游模块使用。
    keys:
      - feed_reduction_by[(country,commodity,year)] = fraction in [0,1]
      - land_carbon_price_by_year[year] = $/tCO2e
      - ruminant_intake_cap[(country,year)] = cap_in_t  （按 2020 年基准总需求 × (1 - ratio_t)）
      - tax_unit_adder[(country,commodity,year)] = $/t （用于供应侧最小生产者价）
    """
    out = {
        'feed_reduction_by': {},
        'land_carbon_price_by_year': {},
        'ruminant_intake_cap': {},
        'tax_unit_adder': {}
    }
    # 预处理：r 动物基准需求（2020）
    base_rumi = {}
    for n in nodes:
        if n.year==base_year and n.commodity in RUMINANT_COMMS:
            base_rumi[n.country] = base_rumi.get(n.country, 0.0) + float(n.D0)

    for eff in [e for e in effects if e.scenario_id==scenario_id]:
        timeline = _linear_path_2020_2080(eff.value_2080, eff.unit, universe.years)
        if eff.kind in ('feed_efficiency','feed_efficiency_rate','feed_efficiency_improve'):
            for y,val in timeline.items():
                if eff.unit=='rate':
                    for i in eff.countries:
                        for j in eff.commodities:
                            out['feed_reduction_by'][i,j,y] = max(0.0, min(1.0, val))
        elif eff.kind in ('land_carbon_price','land_co2_price'):
            for y,val in timeline.items():
                if eff.unit=='amount':
                    out['land_carbon_price_by_year'][y] = float(val)
        elif eff.kind in ('ruminant_intake_decreasing_ratio','ruminant_intake_ratio','ruminant_reduction'):
            for y,val in timeline.items():
                if eff.unit=='rate':
                    for i in eff.countries:
                        cap = (1.0 - val) * base_rumi.get(i, 0.0)
                        out['ruminant_intake_cap'][i,y] = max(0.0, cap)

    # 根据 land carbon price 生成供应侧税额 adder：等于 价格 ×（该节点的 LULUCF 强度 e0_land）
    # 这里不直接读取 process_meta；在 S4.0_main 里根据 universe.process_meta 的 'category' 识别 LULUCF 过程后填入
    return out


def load_scenarios(xlsx_path: str, universe: Universe, sheet: str='Scenario') -> List[ScenarioEffect]:
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    # 适配列名
    col_id   = 'Scenario ID'
    col_type = 'Scenario Element' if 'Scenario Element' in df.columns else 'Scenario Type'
    col_unit = 'Scenario Unit'
    col_proc = 'Emis Process' if 'Emis Process' in df.columns else 'Emis process'
    col_reg  = 'Region' if 'Region' in df.columns else 'Country'
    col_comm = 'Commodity'
    col_val  = 'Value'
    effects = []
    for r in df.itertuples(index=False):
        sid  = getattr(r, col_id)
        kind = str(getattr(r, col_type)).strip().lower().replace(' ', '_')
        unit = str(getattr(r, col_unit)).strip().lower()
        val  = float(getattr(r, col_val))
        reg  = getattr(r, col_reg, 'All')
        com  = getattr(r, col_comm, 'All')
        proc = getattr(r, col_proc, 'All')
        eff = ScenarioEffect(sid, kind, unit, val, reg, com, proc)
        eff.countries   = _select_countries(universe, reg)
        eff.commodities = _select_commodities(universe, com)
        eff.processes   = _select_processes(universe, proc)
        effects.append(eff)
    return effects

def load_mc_specs(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name='MC')
    df.columns = [str(c).strip() for c in df.columns]
    # 需要列：Element, Element unit, Process, Item, GHG, Min_bound, Max_bound, Region_cat
    return df

def draw_mc_to_params(df_specs: pd.DataFrame, universe: Universe, *, seed: int, draw_idx: int) -> dict:
    rng = np.random.default_rng(seed + draw_idx)
    ef_mult = {}   # ((country, commodity, process, year) -> factor)
    for r in df_specs.itertuples(index=False):
        elem = str(getattr(r, 'Element')).lower()
        unit = str(getattr(r, 'Element unit')).lower() if 'Element unit' in df_specs.columns else ''
        proc = str(getattr(r, 'Process')) if 'Process' in df_specs.columns else 'All'
        item = str(getattr(r, 'Item')) if 'Item' in df_specs.columns else 'All'
        ghg  = str(getattr(r, 'GHG')) if 'GHG' in df_specs.columns else ''
        lo   = float(getattr(r, 'Min_bound', 1.0))
        hi   = float(getattr(r, 'Max_bound', 1.0))
        regc = str(getattr(r, 'Region_cat', 'All'))

        # 仅支持“排放因子/强度”类不确定性：识别关键字
        if 'ef' in elem or 'emission' in elem:
            procs = _select_processes(universe, proc)
            # 物品选择
            if item and item.lower()!='all':
                comms = _select_commodities(universe, item)
            else:
                comms = list(universe.commodities)
            # 区域选择
            if regc and regc.lower()!='all':
                # Region_aggMC 分类
                inv = {}
                for c, ragg in (universe.region_aggMC_by_country or {}).items():
                    inv.setdefault(str(ragg), []).append(c)
                countries = inv.get(regc, [])
            else:
                countries = list(universe.countries)
            # 抽样
            factor = float(rng.uniform(lo, hi))
            for i in countries:
                for j in comms:
                    for p in procs:
                        for y in universe.years:
                            ef_mult[i,j,p,y] = factor
    return {'ef_multiplier_by': ef_mult}
