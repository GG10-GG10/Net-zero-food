# -*- coding: utf-8 -*-
"""
S1.0_schema — 基础数据结构（无硬编码 PROCESSES；由 dict_v3 提供）
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class Node:
    country: str
    iso3: str
    year: int
    commodity: str
    Q0: float = 0.0
    D0: float = 0.0
    P0: float = 1.0
    eps_supply: float = 0.0
    eps_demand: float = 0.0
    # 需求侧弹性
    eps_pop_demand: float = 0.0
    eps_income_demand: float = 0.0
    epsD_row: Dict[str, float] = field(default_factory=dict)
    eps_supply_temp: float = 0.0
    eps_supply_yield: float = 0.0
    Tmult: float = 1.0
    Ymult: float = 1.0
    # 排放强度（tCO2e / t product），分过程
    e0_by_proc: Dict[str, float] = field(default_factory=dict)
    # 税费/加价（$/t）
    tax_unit: float = 0.0
    # 区域标签、附加信息
    region_label_new: Optional[str] = None
    meta: Dict[str, float] = field(default_factory=dict)
    def q0_with_ty(self) -> float:
        return float(self.Q0) * (self.Tmult ** self.eps_supply_temp) * (self.Ymult ** self.eps_supply_yield)

@dataclass
class Universe:
    countries: List[str]
    iso3_by_country: Dict[str, str]
    m49_by_country: Dict[str, str] = field(default_factory=dict)
    commodities: List[str]
    years: List[int]
    processes: List[str] = field(default_factory=list)
    process_meta: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # 供情景/匹配使用
    region_aggMC_by_country: Dict[str, str] = field(default_factory=dict)
    item_cat2_by_commodity: Dict[str, str] = field(default_factory=dict)
    ssp_region_by_country: Dict[str, str] = field(default_factory=dict)

@dataclass
class ScenarioConfig:
    years_hist_start: int = 2010
    years_hist_end: int = 2020
    years_future_end: int = 2080
    future_step: int = 10
    feed_efficiency: float = 1.0
    seed_rate_default: float = 0.05

@dataclass
class ScenarioData:
    nodes: List[Node]
    universe: Universe
    config: ScenarioConfig = field(default_factory=ScenarioConfig)
