# -*- coding: utf-8 -*-
"""
S3.5_land_use_change — 土地利用变化（含“土地碳价”集约化代理）
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd

@dataclass
class LUCConfig:
    yield_t_per_ha_default: float = 3.0
    grass_intensity_tdm_per_ha: float = 5.0
    cropland_restore_share: float = 0.5
    # 土地碳价情景（$ / tCO2e）
    land_carbon_price_per_tco2: float = 0.0
    carbon_stock_tco2_per_ha: dict = None  # {'forest':150,'cropland':10,'grassland':30}
    intensification_per_usd: float = 0.0005  # 每 $/tCO2e 触发的需地降低比例（示意）
    intensification_cap: float = 0.3       # 需地最多下降 30%

def _lc(df: pd.DataFrame) -> pd.DataFrame:
    z = df.copy(); z.columns = [str(c).strip() for c in z.columns]; return z

def compute_luc_areas(*, demand_df: pd.DataFrame, production_df: Optional[pd.DataFrame]=None,
                      crop_yield_df: Optional[pd.DataFrame]=None, grass_requirement_df: Optional[pd.DataFrame]=None,
                      base_area_df: Optional[pd.DataFrame]=None, cfg: LUCConfig = LUCConfig()) -> Dict[str, pd.DataFrame]:
    d = _lc(demand_df)
    cn = [c for c in d.columns if 'country' in c.lower()][0]
    yn = [c for c in d.columns if 'year' in c.lower()][0]
    in_ = [c for c in d.columns if 'comm' in c.lower() or 'item' in c.lower()][0]
    dn = [c for c in d.columns if 'demand' in c.lower() or 'qty' in c.lower()][0]
    d = d.rename(columns={cn:'country', yn:'year', in_:'commodity', dn:'demand_t'})[['country','year','commodity','demand_t']]

    if crop_yield_df is not None and len(crop_yield_df):
        y = _lc(crop_yield_df)
        cn = [c for c in y.columns if 'country' in c.lower()][0]
        yn = [c for c in y.columns if 'year' in c.lower()][0]
        in_ = [c for c in y.columns if 'comm' in c.lower() or 'item' in c.lower()][0]
        yv = [c for c in y.columns if 'yield' in c.lower()][0]
        y = y.rename(columns={cn:'country', yn:'year', in_:'commodity', yv:'yield_t_per_ha'})[['country','year','commodity','yield_t_per_ha']]
    else:
        y = d[['country','year','commodity']].drop_duplicates()
        y['yield_t_per_ha'] = cfg.yield_t_per_ha_default

    z = d.merge(y, on=['country','year','commodity'], how='left')
    z['yield_t_per_ha'] = z['yield_t_per_ha'].replace(0, np.nan).fillna(cfg.yield_t_per_ha_default)
    z['crop_area_need_ha'] = z['demand_t'] / z['yield_t_per_ha']
    crop_need = z.groupby(['country','year'], as_index=False)['crop_area_need_ha'].sum()

    # 土地碳价触发的“集约化”：减少需地（不改变物质量）
    if cfg.land_carbon_price_per_tco2 and cfg.land_carbon_price_per_tco2>0:
        red = min(cfg.intensification_per_usd * cfg.land_carbon_price_per_tco2, cfg.intensification_cap)
        crop_need['crop_area_need_ha'] *= (1.0 - red)

    # 草地需求
    if grass_requirement_df is not None and len(grass_requirement_df):
        g = _lc(grass_requirement_df)
        cn = [c for c in g.columns if 'country' in c.lower()][0]
        yn = [c for c in g.columns if 'year' in c.lower()][0]
        gv = [c for c in g.columns if 'dem' in c.lower() or 'req' in c.lower()][0]
        g = g.rename(columns={cn:'country', yn:'year', gv:'grass_tdm'})[['country','year','grass_tdm']]
        g['grass_area_need_ha'] = g['grass_tdm'] / max(cfg.grass_intensity_tdm_per_ha, 1e-9)
    else:
        g = crop_need[['country','year']].copy()
        g['grass_area_need_ha'] = 0.0

    need = crop_need.merge(g, on=['country','year'], how='outer').fillna(0.0)
    need['target_cropland_ha'] = need['crop_area_need_ha']
    need['target_grassland_ha'] = need['grass_area_need_ha']

    # 基准库存
    if base_area_df is not None and len(base_area_df):
        b = _lc(base_area_df)
        cn = [c for c in b.columns if 'country' in c.lower()][0]
        yn = [c for c in b.columns if 'year' in c.lower()][0]
        cc = [c for c in b.columns if 'crop' in c.lower()][0]
        fc = [c for c in b.columns if 'forest' in c.lower()][0]
        gc = [c for c in b.columns if 'grass' in c.lower()][0]
        b = b.rename(columns={cn:'country', yn:'year', cc:'cropland_ha', fc:'forest_ha', gc:'grassland_ha'})[['country','year','cropland_ha','forest_ha','grassland_ha']]
    else:
        b = need[['country','year']].copy()
        b['cropland_ha'] = need['target_cropland_ha'].values
        b['grassland_ha'] = need['target_grassland_ha'].values
        b['forest_ha'] = 0.0

    out = b.merge(need[['country','year','target_cropland_ha','target_grassland_ha']], on=['country','year'], how='left')
    out['new_cropland_ha']  = out['target_cropland_ha'].fillna(out['cropland_ha'])
    out['new_grassland_ha'] = out['target_grassland_ha'].fillna(out['grassland_ha'])

    total0 = out['cropland_ha'] + out['forest_ha'] + out['grassland_ha']
    out['new_forest_ha'] = (total0 - out['new_cropland_ha'] - out['new_grassland_ha']).clip(lower=0.0)

    deltas = out[['country','year']].copy()
    deltas['d_cropland_ha']  = out['new_cropland_ha']  - out['cropland_ha']
    deltas['d_grassland_ha'] = out['new_grassland_ha'] - out['grassland_ha']
    deltas['d_forest_ha']    = out['new_forest_ha']    - out['forest_ha']

    # 碳库变化与碳价成本
    cs = (cfg.carbon_stock_tco2_per_ha or {'forest':150.0,'cropland':10.0,'grassland':30.0})
    before = out['cropland_ha']*cs['cropland'] + out['grassland_ha']*cs['grassland'] + out['forest_ha']*cs['forest']
    after  = out['new_cropland_ha']*cs['cropland'] + out['new_grassland_ha']*cs['grassland'] + out['new_forest_ha']*cs['forest']
    deltas['d_carbon_stock_tco2'] = after - before
    deltas['carbon_price_cost_$'] = - deltas['d_carbon_stock_tco2'] * float(cfg.land_carbon_price_per_tco2 or 0.0)

    luc_area = out[['country','year','new_cropland_ha','new_forest_ha','new_grassland_ha']].rename(
        columns={'new_cropland_ha':'cropland_ha','new_forest_ha':'forest_ha','new_grassland_ha':'grassland_ha'})

    return {'luc_area': luc_area, 'deltas': deltas}
