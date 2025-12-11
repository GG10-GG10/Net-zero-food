# -*- coding: utf-8 -*-
"""
S3.5_land_use_change — 土地利用变化（含"土地碳价"集约化代理）

逐期比较逻辑：
- 每个年份的土地变化(deltas)相对于上一期计算
- 例如：2020与基准年(LUH2 2020)比较，2040与2020比较，2080与2040比较
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
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
    """
    计算土地利用变化面积。
    
    逐期比较逻辑：
    - 基准年(如2020)：与LUH2基准数据比较
    - 后续年份：与前一期的模型计算结果(new_*_ha)比较
    - 例如：2040与2020的new_*_ha比较，2080与2040的new_*_ha比较
    """
    d = _lc(demand_df)
    cn = [c for c in d.columns if 'country' in c.lower() and 'm49' not in c.lower()][0]
    yn = [c for c in d.columns if 'year' in c.lower()][0]
    in_ = [c for c in d.columns if 'comm' in c.lower() or 'item' in c.lower()][0]
    dn = [c for c in d.columns if 'demand' in c.lower() or 'qty' in c.lower()][0]
    d = d.rename(columns={cn:'country', yn:'year', in_:'commodity', dn:'demand_t'})
    # ✅ 保留M49_Country_Code列（如果存在）
    select_cols = ['country','year','commodity','demand_t']
    if 'M49_Country_Code' in d.columns:
        select_cols = ['M49_Country_Code'] + select_cols
    d = d[select_cols]

    if crop_yield_df is not None and len(crop_yield_df):
        y = _lc(crop_yield_df)
        cn = [c for c in y.columns if 'country' in c.lower() and 'm49' not in c.lower()][0]
        yn = [c for c in y.columns if 'year' in c.lower()][0]
        in_ = [c for c in y.columns if 'comm' in c.lower() or 'item' in c.lower()][0]
        yv = [c for c in y.columns if 'yield' in c.lower()][0]
        y = y.rename(columns={cn:'country', yn:'year', in_:'commodity', yv:'yield_t_per_ha'})
        # ✅ 保留M49_Country_Code列（如果存在）
        select_cols = ['country','year','commodity','yield_t_per_ha']
        if 'M49_Country_Code' in y.columns:
            select_cols = ['M49_Country_Code'] + select_cols
        y = y[select_cols]
    else:
        # ✅ 从d复制所有列（包括M49）
        base_cols = ['country','year','commodity']
        if 'M49_Country_Code' in d.columns:
            base_cols = ['M49_Country_Code'] + base_cols
        y = d[base_cols].drop_duplicates()
        y['yield_t_per_ha'] = cfg.yield_t_per_ha_default

    # ✅ merge使用所有公共key列
    merge_keys = ['country','year','commodity']
    if 'M49_Country_Code' in d.columns and 'M49_Country_Code' in y.columns:
        merge_keys = ['M49_Country_Code'] + merge_keys
    z = d.merge(y, on=merge_keys, how='left')
    z['yield_t_per_ha'] = z['yield_t_per_ha'].replace(0, np.nan).fillna(cfg.yield_t_per_ha_default)
    
    # ✅ DEBUG: 检查yield数据源
    yield_na_count = z['yield_t_per_ha'].isna().sum()
    yield_default_count = (z['yield_t_per_ha'] == cfg.yield_t_per_ha_default).sum()
    print(f"[S3_5 DEBUG] yield数据统计: 总行数={len(z)}, NA后填默认值={yield_na_count}, 使用默认值{cfg.yield_t_per_ha_default}的行数={yield_default_count}")
    if len(z) > 0:
        sample_yields = z[['country', 'year', 'commodity', 'yield_t_per_ha']].drop_duplicates().head(5)
        print(f"[S3_5 DEBUG] yield样本:\n{sample_yields.to_string(index=False)}")
    
    z['crop_area_need_ha'] = z['demand_t'] / z['yield_t_per_ha']
    # ✅ groupby包含M49（如果存在）
    group_cols = ['country','year']
    if 'M49_Country_Code' in z.columns:
        group_cols = ['M49_Country_Code'] + group_cols
    crop_need = z.groupby(group_cols, as_index=False)['crop_area_need_ha'].sum()
    
    # ✅ DEBUG: 检查全球总面积
    for year in [2020, 2080]:
        year_total = crop_need[crop_need['year'] == year]['crop_area_need_ha'].sum()
        print(f"[S3_5 DEBUG] {year}年全球耕地面积需求: {year_total:,.0f} ha")

    # 土地碳价触发的"集约化"：减少需地（不改变物质量）
    if cfg.land_carbon_price_per_tco2 and cfg.land_carbon_price_per_tco2>0:
        red = min(cfg.intensification_per_usd * cfg.land_carbon_price_per_tco2, cfg.intensification_cap)
        crop_need['crop_area_need_ha'] *= (1.0 - red)

    # 草地需求
    if grass_requirement_df is not None and len(grass_requirement_df):
        print(f"[S3_5 DEBUG] grass_requirement_df 输入: {len(grass_requirement_df)} 行, 列={list(grass_requirement_df.columns)}")
        g = _lc(grass_requirement_df)
        cn = [c for c in g.columns if 'country' in c.lower() and 'm49' not in c.lower()][0]
        yn = [c for c in g.columns if 'year' in c.lower()][0]
        area_col = next((c for c in g.columns if 'area' in c.lower()), None)
        dm_col = next((c for c in g.columns if 'grass_t' in c.lower() or 'dem' in c.lower()), None)
        rename_map = {cn: 'country', yn: 'year'}
        if area_col:
            rename_map[area_col] = 'grass_area_need_ha'
        if dm_col:
            rename_map[dm_col] = 'grass_tdm'
        g = g.rename(columns=rename_map)
        # ✅ 保留M49_Country_Code列
        keep_cols = ['country','year']
        if 'M49_Country_Code' in g.columns:
            keep_cols = ['M49_Country_Code'] + keep_cols
        if 'grass_tdm' in g.columns:
            keep_cols.append('grass_tdm')
        if 'grass_area_need_ha' in g.columns:
            keep_cols.append('grass_area_need_ha')
        g = g[keep_cols]
        if 'grass_area_need_ha' not in g.columns and 'grass_tdm' in g.columns:
            g['grass_area_need_ha'] = g['grass_tdm'] / max(cfg.grass_intensity_tdm_per_ha, 1e-9)
        if 'grass_tdm' not in g.columns and 'grass_area_need_ha' in g.columns:
            g['grass_tdm'] = g['grass_area_need_ha'] * max(cfg.grass_intensity_tdm_per_ha, 1e-9)
        # ✅ DEBUG: 检查草地需求
        for year in [2020, 2080]:
            year_total = g[g['year'] == year]['grass_area_need_ha'].sum() if 'grass_area_need_ha' in g.columns else 0
            print(f"[S3_5 DEBUG] {year}年全球草地面积需求: {year_total:,.0f} ha")
    else:
        print(f"[S3_5 DEBUG] ⚠️ grass_requirement_df 为空或None！草地需求将设为0")
        # ✅ 从crop_need复制所有key列（包括M49）
        base_cols = ['country','year']
        if 'M49_Country_Code' in crop_need.columns:
            base_cols = ['M49_Country_Code'] + base_cols
        g = crop_need[base_cols].copy()
        g['grass_area_need_ha'] = 0.0

    # ✅ merge使用所有公共key列（包括M49）
    merge_keys = ['country','year']
    if 'M49_Country_Code' in crop_need.columns and 'M49_Country_Code' in g.columns:
        merge_keys = ['M49_Country_Code'] + merge_keys
    need = crop_need.merge(g, on=merge_keys, how='outer').fillna(0.0)
    need['target_cropland_ha'] = need['crop_area_need_ha']
    need['target_grassland_ha'] = need['grass_area_need_ha']

    # 基准库存（只使用基准年数据作为初始值）
    has_m49 = False
    if base_area_df is not None and len(base_area_df):
        b = _lc(base_area_df)
        cn = [c for c in b.columns if 'country' in c.lower() and 'm49' not in c.lower()][0]
        yn = [c for c in b.columns if 'year' in c.lower()][0]
        cc = [c for c in b.columns if 'crop' in c.lower()][0]
        fc = [c for c in b.columns if 'forest' in c.lower()][0]
        gc = [c for c in b.columns if ('grass' in c.lower() or 'pasture' in c.lower())][0]
        b = b.rename(columns={cn:'country', yn:'year', cc:'cropland_ha', fc:'forest_ha', gc:'grassland_ha'})
        # ✅ 保留M49_Country_Code列
        select_cols = ['country','year','cropland_ha','forest_ha','grassland_ha']
        if 'M49_Country_Code' in b.columns:
            select_cols = ['M49_Country_Code'] + select_cols
            has_m49 = True
        b = b[select_cols]
    else:
        # ✅ 从need复制key列（包括M49）
        base_cols = ['country','year']
        if 'M49_Country_Code' in need.columns:
            base_cols = ['M49_Country_Code'] + base_cols
            has_m49 = True
        b = need[base_cols].copy()
        b['cropland_ha'] = need['target_cropland_ha'].values
        b['grassland_ha'] = need['target_grassland_ha'].values
        b['forest_ha'] = 0.0

    # ========== 逐期比较逻辑 ==========
    # 获取所有年份并排序
    all_years = sorted(need['year'].unique())
    print(f"[S3_5 DEBUG] 所有年份: {all_years}")
    print(f"[S3_5 DEBUG] 年份数量: {len(all_years)}, 最小年份: {min(all_years)}, 最大年份: {max(all_years)}")
    
    # ✅ 关键修复：确定基准年（使用2020作为基准年，而非最小年份）
    # 原因：最小年份可能是2010（历史数据），但LUC变化应以2020为基准
    #      （因为未来情景从2020年开始，且Qs优化结果也是从2020开始）
    base_year = 2020  # ✅ 固定使用2020作为基准年
    if base_year not in all_years:
        # 如果2020不在数据中，才回退到最小年份
        base_year = min(all_years)
        print(f"[S3_5 WARNING] 2020年不在数据中，使用最小年份 {base_year} 作为基准")
    print(f"[S3_5 DEBUG] 基准年: {base_year}")
    
    # 获取所有国家
    countries = need['country'].unique()
    
    # 准备输出数据
    out_records: List[Dict] = []
    delta_records: List[Dict] = []
    
    # 碳库参数
    cs = (cfg.carbon_stock_tco2_per_ha or {'forest':150.0,'cropland':10.0,'grassland':30.0})
    
    for country in countries:
        # 获取该国的基准面积（从base_area_df中取基准年数据）
        country_base = b[(b['country'] == country) & (b['year'] == base_year)]
        if country_base.empty:
            # 如果没有基准年数据，尝试使用该国最早年份的数据
            country_base = b[b['country'] == country].sort_values('year').head(1)
            if country_base.empty:
                continue
        
        # 初始化：上一期的面积 = 基准年的LUH2数据
        prev_cropland_ha = float(country_base['cropland_ha'].iloc[0])
        prev_grassland_ha = float(country_base['grassland_ha'].iloc[0])
        prev_forest_ha = float(country_base['forest_ha'].iloc[0])
        total_land = prev_cropland_ha + prev_grassland_ha + prev_forest_ha
        
        # 获取M49（如果存在）
        m49_val = None
        if has_m49 and 'M49_Country_Code' in country_base.columns:
            m49_val = country_base['M49_Country_Code'].iloc[0]
        
        # 按年份顺序处理
        for year in all_years:
            # 获取该年的需求目标
            country_year_need = need[(need['country'] == country) & (need['year'] == year)]
            if country_year_need.empty:
                continue
            
            target_cropland = float(country_year_need['target_cropland_ha'].iloc[0])
            target_grassland_raw = float(country_year_need['target_grassland_ha'].iloc[0])
            
            # ✅ 关键修复：如果目标草地需求为0或缺失（表示没有预测数据），
            # 则保持上一期的草地面积，而不是设为0导致森林面积爆增
            if target_grassland_raw <= 0 and prev_grassland_ha > 0:
                # 没有草地需求预测数据，保持上一期面积
                target_grassland = prev_grassland_ha
                print(f"[S3_5 DEBUG] {country} {year}: 草地需求缺失，保持上一期面积={prev_grassland_ha:,.0f} ha")
            else:
                target_grassland = target_grassland_raw
            
            # ========== ✅ 关键修复：正确的土地转换逻辑 ==========
            # 土地转换规则：
            # 1. 耕地/草地需求增加时，优先从森林转换（毁林）
            # 2. 耕地/草地需求减少时，释放的土地可以恢复为森林（造林）
            # 3. 约束：forest + cropland + grassland ≤ total_land（不是等于！）
            # 4. 森林面积不能为负
            
            # 计算耕地和草地的需求变化
            d_cropland_demand = target_cropland - prev_cropland_ha
            d_grassland_demand = target_grassland - prev_grassland_ha
            
            # 计算总的土地需求变化
            total_demand_change = d_cropland_demand + d_grassland_demand
            
            # 新的耕地和草地面积直接等于目标需求
            new_cropland_ha = target_cropland
            new_grassland_ha = target_grassland
            
            # ✅ 森林面积变化逻辑：
            # - 如果总需求增加（扩张），森林减少（毁林）
            # - 如果总需求减少（收缩），森林增加（造林/恢复）
            # - 森林面积 = 上期森林 - 净毁林量
            # - 净毁林量 = 耕地扩张 + 草地扩张（正值表示毁林，负值表示造林）
            
            new_forest_ha = prev_forest_ha - total_demand_change
            
            # ✅ 约束1：森林面积不能为负
            if new_forest_ha < 0:
                # 森林不够砍了，需要限制扩张
                available_forest = prev_forest_ha
                # 按比例缩减耕地和草地的扩张
                if total_demand_change > 0 and total_demand_change > available_forest:
                    scale_factor = available_forest / total_demand_change if total_demand_change > 0 else 1.0
                    # 只缩减扩张部分，不缩减原有面积
                    if d_cropland_demand > 0:
                        d_cropland_demand = d_cropland_demand * scale_factor
                    if d_grassland_demand > 0:
                        d_grassland_demand = d_grassland_demand * scale_factor
                    new_cropland_ha = prev_cropland_ha + d_cropland_demand
                    new_grassland_ha = prev_grassland_ha + d_grassland_demand
                    print(f"[S3_5 WARN] {country} {year}: 森林不足，限制扩张 scale={scale_factor:.2%}")
                new_forest_ha = 0.0
            
            # ✅ 约束2：总面积不能超过初始总土地面积
            new_total = new_cropland_ha + new_grassland_ha + new_forest_ha
            if new_total > total_land * 1.001:  # 允许0.1%误差
                print(f"[S3_5 WARN] {country} {year}: 总面积超限 {new_total:,.0f} > {total_land:,.0f}")
            
            # 计算实际的变化量（逐期比较）
            d_cropland = new_cropland_ha - prev_cropland_ha
            d_grassland = new_grassland_ha - prev_grassland_ha
            d_forest = new_forest_ha - prev_forest_ha
            
            # 计算碳库变化
            before_carbon = prev_cropland_ha * cs['cropland'] + prev_grassland_ha * cs['grassland'] + prev_forest_ha * cs['forest']
            after_carbon = new_cropland_ha * cs['cropland'] + new_grassland_ha * cs['grassland'] + new_forest_ha * cs['forest']
            d_carbon_stock = after_carbon - before_carbon
            carbon_price_cost = -d_carbon_stock * float(cfg.land_carbon_price_per_tco2 or 0.0)
            
            # 记录输出
            out_row = {
                'country': country,
                'year': year,
                'cropland_ha': prev_cropland_ha,  # 期初面积（上一期的结果）
                'forest_ha': prev_forest_ha,
                'grassland_ha': prev_grassland_ha,
                'target_cropland_ha': target_cropland,
                'target_grassland_ha': target_grassland,
                'new_cropland_ha': new_cropland_ha,
                'new_forest_ha': new_forest_ha,
                'new_grassland_ha': new_grassland_ha,
            }
            if has_m49:
                out_row['M49_Country_Code'] = m49_val
            out_records.append(out_row)
            
            delta_row = {
                'country': country,
                'year': year,
                'd_cropland_ha': d_cropland,
                'd_grassland_ha': d_grassland,
                'd_forest_ha': d_forest,
                'd_carbon_stock_tco2': d_carbon_stock,
                'carbon_price_cost_$': carbon_price_cost,
            }
            if has_m49:
                delta_row['M49_Country_Code'] = m49_val
            delta_records.append(delta_row)
            
            # 更新上一期面积为本期的新面积（为下一期做准备）
            prev_cropland_ha = new_cropland_ha
            prev_grassland_ha = new_grassland_ha
            prev_forest_ha = new_forest_ha
    
    # 构建输出DataFrame
    out = pd.DataFrame(out_records)
    deltas = pd.DataFrame(delta_records)
    
    # ✅ DEBUG: 检查 deltas 的数量级
    if not deltas.empty:
        us_deltas = deltas[deltas['country'] == 'United States of America']
        if not us_deltas.empty:
            for year in all_years:
                year_data = us_deltas[us_deltas['year'] == year]
                if not year_data.empty:
                    d_crop = year_data['d_cropland_ha'].iloc[0]
                    d_grass = year_data['d_grassland_ha'].iloc[0]
                    d_forest = year_data['d_forest_ha'].iloc[0]
                    print(f"[S3_5 DEBUG] U.S. {year}: d_cropland={d_crop:,.0f} ha, d_grassland={d_grass:,.0f} ha, d_forest={d_forest:,.0f} ha")
            # 检查 out 中的原始数据
            us_out = out[out['country'] == 'United States of America']
            for year in all_years:
                year_data = us_out[us_out['year'] == year]
                if not year_data.empty:
                    crop_ha = year_data['cropland_ha'].iloc[0]
                    new_crop_ha = year_data['new_cropland_ha'].iloc[0]
                    target_crop_ha = year_data['target_cropland_ha'].iloc[0] if 'target_cropland_ha' in year_data.columns else 0
                    print(f"[S3_5 DEBUG] U.S. {year}: 期初cropland={crop_ha:,.0f}, 期末new_cropland={new_crop_ha:,.0f}, target={target_crop_ha:,.0f}")

    # 构建输出DataFrames
    if not out.empty:
        # 期初面积 (period_start)
        period_start_cols = ['country', 'year', 'cropland_ha', 'forest_ha', 'grassland_ha']
        if has_m49:
            period_start_cols = ['M49_Country_Code'] + period_start_cols
        period_start = out[period_start_cols].copy()
        
        # 期末面积 (period_end)
        period_end_cols = ['country', 'year', 'new_cropland_ha', 'new_forest_ha', 'new_grassland_ha']
        if has_m49:
            period_end_cols = ['M49_Country_Code'] + period_end_cols
        period_end = out[period_end_cols].rename(
            columns={'new_cropland_ha': 'cropland_ha', 'new_forest_ha': 'forest_ha', 'new_grassland_ha': 'grassland_ha'})
        
        # 兼容旧接口：luc_area = period_end
        luc_area = period_end.copy()
    else:
        empty_cols = ['country', 'year', 'cropland_ha', 'forest_ha', 'grassland_ha']
        period_start = pd.DataFrame(columns=empty_cols)
        period_end = pd.DataFrame(columns=empty_cols)
        luc_area = pd.DataFrame(columns=empty_cols)

    return {
        'luc_area': luc_area,           # 期末面积（兼容旧接口）
        'period_start': period_start,   # 期初面积
        'period_end': period_end,       # 期末面积
        'deltas': deltas                # 逐期变化量
    }
