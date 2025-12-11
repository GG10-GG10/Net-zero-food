#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LUC历史排放读取与处理模块
从Emission_LULUCF_Historical_updated.xlsx读取历史排放，
过滤出指定过程（Wood harvest, Forest, De/Reforestation_crop, De/Reforestation_pasture）
并格式化为与未来排放兼容的输出格式。
"""
from __future__ import annotations
from typing import Optional, Dict
import pandas as pd
import numpy as np


TARGET_PROCESSES = ['Wood harvest', 'Forest', 'De/Reforestation_crop', 'De/Reforestation_pasture']


def read_luc_historical_emissions(
    hist_file: str,
    dict_v3_path: Optional[str] = None,
    years: Optional[list] = None,
) -> pd.DataFrame:
    """
    从Emission_LULUCF_Historical_updated.xlsx读取历史LUC排放。
    
    参数
    ----
    hist_file : str
        Emission_LULUCF_Historical_updated.xlsx路径
    dict_v3_path : str, optional
        dict_v3.xlsx路径，用于M49映射与region_label_new
    years : list, optional
        限制的年份列表（通常2000-2020）；如None则读取全部
    
    返回
    ----
    DataFrame
        列: M49_Country_Code, Region_label_new, year, Process, GHG, value(tCO2/年)
        仅包含Select=1的目标过程记录
    """
    # 读取历史排放数据
    df = pd.read_excel(hist_file, sheet_name='LULUCF_updated')
    df.columns = [str(c).strip() for c in df.columns]
    
    # 过滤Select=1（选中行）
    if 'Select' in df.columns:
        df = df[df['Select'] == 1]
    
    # 过滤目标过程
    if 'Land Category' in df.columns:
        df = df[df['Land Category'].isin(TARGET_PROCESSES)]
    elif 'Process' in df.columns:
        df = df[df['Process'].isin(TARGET_PROCESSES)]
    else:
        print("[WARN] 无法找到Land Category或Process列，将不过滤过程")
    
    if df.empty:
        return pd.DataFrame(columns=[
            'M49_Country_Code', 'Region_label_new', 'year', 'Process', 'GHG', 'value'
        ])
    
    # 确保M49_Country_Code列存在且格式正确（标准化M49格式）
    if 'M49_Country_Code' not in df.columns:
        raise KeyError("M49_Country_Code列缺失")
    
    # ✅ 关键修复：过滤掉全球汇总行（M49='000' 或 Region_label_new='World'）
    # 历史数据中包含World汇总行，会导致重复计算
    df = df[~df['M49_Country_Code'].astype(str).str.strip().str.lstrip("'\"").isin(['0', '00', '000', '1'])]
    if 'Region_label_new' in df.columns:
        df = df[df['Region_label_new'] != 'World']
    print(f"[LUC历史] 过滤World汇总行后: {len(df)} 行")
    
    # 标准化M49：格式化为'xxx（单引号+3位数字）
    def _normalize_m49(val):
        s = str(val).strip().lstrip("'\"")
        try:
            # 转换为整数再格式化为'xxx格式（单引号+3位数字）
            return f"'{int(s):03d}"  # ✅ 'xxx格式
        except:
            return f"'{s}" if not s.startswith("'") else s
    
    df['M49_Country_Code'] = df['M49_Country_Code'].apply(_normalize_m49)
    
    # 确定年份列和过程列
    year_cols = [c for c in df.columns if c.startswith('Y') and len(c) == 5]  # Y2000, Y2020等
    process_col = 'Land Category' if 'Land Category' in df.columns else 'Process'
    
    # 确定GHG列（假设'Species'或直接在过程名中）
    if 'Species' in df.columns:
        ghg_col = 'Species'
    else:
        ghg_col = None
    
    # 将宽格式转换为长格式
    id_cols = ['M49_Country_Code', process_col]
    if ghg_col:
        id_cols.append(ghg_col)
    if 'Region_label_new' in df.columns:
        id_cols.append('Region_label_new')
    
    # 熔融年份列
    df_long = df[id_cols + year_cols].melt(
        id_vars=id_cols,
        value_vars=year_cols,
        var_name='year_str',
        value_name='value'
    )
    
    # 解析年份（Y2000 -> 2000）
    df_long['year'] = df_long['year_str'].str.replace('Y', '').astype(int)
    
    # 重命名列
    df_long = df_long.rename(columns={
        process_col: 'Process',
        ghg_col: 'GHG' if ghg_col else None
    })
    
    # 移除NaN值与0值（简化）
    df_long = df_long.dropna(subset=['value'])
    df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce')
    df_long = df_long.dropna(subset=['value'])
    
    # ✅ 关键修复：单位转换
    # 历史数据单位是 MtCO2/yr（百万吨CO2/年），需要转换为 kt CO2/yr（千吨CO2/年）用于汇总
    # MtCO2 → kt: 乘以 1e3（1 MtCO2 = 1000 ktCO2）
    df_long['value'] = df_long['value'] * 1e3  # MtCO2/yr → kt CO2/yr
    
    # 如果没有GHG列，默认为CO2
    if 'GHG' not in df_long.columns or df_long['GHG'].isna().all():
        df_long['GHG'] = 'CO2'
    
    # 如果没有Region_label_new，从dict_v3加载
    if 'Region_label_new' not in df_long.columns or df_long['Region_label_new'].isna().all():
        if dict_v3_path:
            try:
                region_df = pd.read_excel(dict_v3_path, sheet_name='region',
                                         usecols=['M49_Country_Code', 'Region_label_new'])
                region_df['M49_Country_Code'] = region_df['M49_Country_Code'].astype(str).str.strip()
                region_map = dict(zip(region_df['M49_Country_Code'], region_df['Region_label_new']))
                df_long['Region_label_new'] = df_long['M49_Country_Code'].map(region_map)
            except Exception as e:
                print(f"[WARN] 无法从dict_v3加载Region_label_new: {e}")
                df_long['Region_label_new'] = 'Unknown'
        else:
            df_long['Region_label_new'] = 'Unknown'
    
    # 添加Item列：根据Process映射到对应的Item
    process_to_item_map = {
        'Wood harvest': 'Roundwood',
        'Forest': 'Forestland',
        'De/Reforestation_crop': 'De/Reforestation_crop area',
        'De/Reforestation_pasture': 'De/Reforestation_pasture area'
    }
    df_long['Item'] = df_long['Process'].map(process_to_item_map)
    
    # 过滤年份（如果提供）
    if years:
        years_set = set(int(y) for y in years)
        df_long = df_long[df_long['year'].isin(years_set)]
    
    # 选择和排序列
    cols_out = ['M49_Country_Code', 'Region_label_new', 'year', 'Process', 'Item', 'GHG', 'value']
    df_long = df_long[cols_out].reset_index(drop=True)
    
    # 聚合（可能有重复）
    df_long = df_long.groupby(cols_out[:-1], as_index=False)['value'].sum()
    
    return df_long
