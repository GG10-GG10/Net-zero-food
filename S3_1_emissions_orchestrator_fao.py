# -*- coding: utf-8 -*-
"""
S3.1_emissions_orchestrator_fao — 简化版：把活动表/火情直接转为可汇总的排放表结构
注：真正的排放计算由 *_emissions_module_fao.py 提供；此处主要为汇总管道提供统一输出结构。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd

@dataclass
class FAOPaths:
    gce_params: Optional[str] = None
    gfe_params: Optional[str] = None
    gle_params: Optional[str] = None
    gos_params: Optional[str] = None
    lme_params: Optional[str] = None

class EmissionsFAO:
    def __init__(self, paths: FAOPaths):
        self.paths = paths

    def _to_emis_df(self, df: pd.DataFrame, label: str) -> pd.DataFrame:
        if df is None or len(df)==0:
            return pd.DataFrame(columns=['country','iso3','year','commodity','co2e_kt'])
        z = df.copy()
        for c in ['country','iso3','year']:
            if c not in z.columns:
                z[c] = None
        if 'commodity' not in z.columns:
            z['commodity'] = 'ALL'
        # 这里不做真实排放计算，仅生成 0 占位；真实数值应由各模块计算并写到 df 中。
        if 'co2e_kt' not in z.columns:
            z['co2e_kt'] = 0.0
        return z[['country','iso3','year','commodity','co2e_kt']]

    def run_all(self, *, crop_activity: Dict[str, pd.DataFrame], livestock_stock: pd.DataFrame,
                gv_areas: pd.DataFrame, fires_df: Optional[pd.DataFrame],
                iso3_list: Optional[list], years: list) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        # GCE/GFE/GLE/GOS 最小占位（真正数值应由前面的 attach_emission_factors 与各模块完成，或在这里接模块计算）
        out['GCE'] = {
            'Residues': self._to_emis_df(crop_activity.get('residues_df'), 'Residues'),
            'Burning':  self._to_emis_df(crop_activity.get('burning_df'),  'Burning'),
            'Rice':     self._to_emis_df(crop_activity.get('rice_df'),     'Rice'),
            'Fertilizers': self._to_emis_df(crop_activity.get('fertilizers_df'), 'Fertilizers'),
        }
        out['GLE'] = []
        out['GOS'] = []
        out['GFE'] = {}
        out['LUF'] = self._to_emis_df(fires_df, 'Land-use fires') if fires_df is not None else None
        return out
