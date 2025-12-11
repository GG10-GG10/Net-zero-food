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
    commodities: List[str]
    years: List[int]
    m49_by_country: Dict[str, str] = field(default_factory=dict)
    country_by_m49: Dict[str, str] = field(default_factory=dict)
    processes: List[str] = field(default_factory=list)
    process_meta: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # 供情景/匹配使用
    region_aggMC_by_country: Dict[str, str] = field(default_factory=dict)
    item_cat2_by_commodity: Dict[str, str] = field(default_factory=dict)
    ssp_region_by_country: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Build reverse M49->country lookup when not provided explicitly."""
        if not self.country_by_m49 and self.m49_by_country:
            reverse: Dict[str, str] = {}
            for country, code in self.m49_by_country.items():
                if code is None:
                    continue
                code_str = str(code).strip()
                if not code_str:
                    continue
                try:
                    lowered = code_str.lower()
                except AttributeError:
                    lowered = ''
                if lowered in {'nan', 'inf', '-inf'}:
                    continue
                try:
                    code_key = f"'{int(float(code_str)):03d}"  # ✅ 'xxx格式
                except (ValueError, TypeError):
                    code_key = f"'{code_str}" if not code_str.startswith("'") else code_str
                reverse.setdefault(code_key, country)
            self.country_by_m49 = reverse

@dataclass
class ScenarioConfig:
    years_hist_start: int = 2010
    years_hist_end: int = 2020
    years_future: List[int] = field(default_factory=lambda: [2080])  # 恢复：只跑2080年
    feed_efficiency: float = 1.0
    seed_rate_default: float = 0.05

@dataclass
class ScenarioData:
    nodes: List[Node]
    universe: Universe
    config: ScenarioConfig = field(default_factory=ScenarioConfig)
