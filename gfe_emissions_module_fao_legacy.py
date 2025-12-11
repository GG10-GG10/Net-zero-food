
# -*- coding: utf-8 -*-
"""
gfe_emissions_module_fao_forest_only.py

仅保留“Forest Land Remaining Forest Land（FL-FL）”的CO2排放/汇计算（负排放为森林碳汇），
完全移除所有LUC相关（如“Net forest conversion / Land Converted to Forest Land”）的计算。
实现遵循 IPCC 2006 指南 + 2019 Refinement（Volume 4, Ch.4 Forest Land）中 Tier 1 的指导：

- 生物量（Biomass）：采用“存量差法（Stock-Difference）”计算生物量碳库变化。
  ΔC_biomass = C_t - C_(t-1)，CO2eq = - ΔC_biomass × 44/12 × 1e-3  (单位：ktCO2；负号表示“碳汇为负排放”)
- 死有机质（DOM: dead wood, litter）：Tier 1 对 FL-FL 不要求计算（视同 0）。
- 矿质土壤（Mineral soil）：Tier 1 下，若活动数据采用 Approach 2/3，则 FL-FL 的 ΔSOC=0；
  若仅有 Approach 1 且需要闭合土地基数，可按 Eq.2.25 估算期初/期末 SOC 再取差；本模块默认 ΔSOC=0。
- 有机土壤（Organic soil）：2019 Refinement 对 FL-FL“无新增细化”，本模块默认 ΔSOC_organic=0。

输入/接口假定：
- params 是一个“宽表/长表均可”的 DataFrame，但必须可被 get_param(...) 取到下列关键字段：
    process="Forests"（或"Forest Land"）, item="Forests total"（或同义）, param="B_total_tC"
  即：每个国家-年份的森林总碳存量（tC，总生物量；若包含DOM/土壤请确保这里只是“活生物量”或保持口径一致）。
- 若你已有更细的碳库字段（AGB/BGB/HWP等），可在 get_param(...) 中切换到对应字段。

输出：
- 返回一个 DataFrame，仅包含一个“process = Forest land remaining forest land (FL-FL)”；
  gas 固定为“CO2”；value_ktCO2 为年度汇（负值）；
  保持列名与原版模块风格一致（例如：['iso3','AreaCode','year','process','gas','value_ktCO2']）。

注意：
- 本实现默认 ΔSOC=0（Tier 1 & Approach 2/3）；如需在 Approach 1 下做SOC闭合，请将
  `ENABLE_SOIL_APPROACH1` 设为 True，并在 TODO 区域按 Eq.2.25 引入 SOCREF/FLU/FMG/FI/FND 等参数。
- 为防止“重复计数”，本模块**不再**计算 “Net forest conversion” 或 “Land converted to forest land”。

作者：NET-ZERO FOOD 项目
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Iterable

# ---------------------- 常量 ----------------------
_KG_CO2_PER_KG_C = 44.0 / 12.0              # 1 tC -> 44/12 tCO2
_TCO2_TO_KTCO2 = 1.0 / 1e3                  # tCO2 -> ktCO2
_TC_TO_KTCO2 = _KG_CO2_PER_KG_C * _TCO2_TO_KTCO2  # tC -> ktCO2

# 切换：是否在 Approach 1 下尝试对 SOC 做“起止库存法”计算（默认 False -> ΔSOC=0）
ENABLE_SOIL_APPROACH1 = False


# ---------------------- 工具函数 ----------------------
def _standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    统一列名，兼容大小写/下划线差异；不做强制重排，仅做映射存在的列。
    需要至少包含：['iso3','AreaCode','year'] + 用于 get_param 的键值列。
    """
    mapper = {
        'ISO3': 'iso3',
        'iso': 'iso3',
        'areacode': 'AreaCode',
        'Year': 'year',
        'YEAR': 'year',
        'PROCESS': 'process',
        'ITEM': 'item',
        'PARAM': 'param',
        'ParamName': 'param',
        'PARAMNAME': 'param',
        'value': 'value',
        'VALUE': 'value',
        'Val': 'value',
    }
    cols = {c: mapper.get(c, c) for c in df.columns}
    out = df.rename(columns=cols)
    return out


def get_param(params: pd.DataFrame,
              process: str,
              item: str,
              param: str,
              value_col: str = 'value') -> pd.DataFrame:
    """
    从参数表中抽取指定 process-item-param 的数值时间序列。
    要求存在列：['iso3','AreaCode','year','process','item','param', value_col]
    若 'process'/'item'/'param' 字段不存在，则尝试以更宽松的规则匹配（仅 param）。
    """
    p = _standardize_cols(params).copy()

    # 宽表容错：如果没有 'process'/'item'/'param' 列，但有直接的列名 param
    if 'process' not in p.columns or 'item' not in p.columns or 'param' not in p.columns:
        if param in p.columns:
            # 假定这是国别-年份的列
            sel = ['iso3', 'AreaCode', 'year', param]
            missing = [c for c in ['iso3', 'AreaCode', 'year'] if c not in p.columns]
            if missing:
                raise KeyError(f"参数表缺少必要列: {missing}")
            df = p[sel].rename(columns={param: value_col}).copy()
            df['process'] = process
            df['item'] = item
            df['param'] = param
            return df
        else:
            raise KeyError("参数表缺少 'process'/'item'/'param' 列，且未找到同名参数列。")

    # 确保 value 列存在
    if value_col not in p.columns:
        # 常见备选名
        for alt in ['Val', 'VAL', 'val', 'Value']:
            if alt in p.columns:
                p = p.rename(columns={alt: value_col})
                break
        else:
            raise KeyError(f"参数表缺少数值列 '{value_col}'")

    mask = (
        (p['process'].astype(str).str.lower() == str(process).lower()) &
        (p['item'].astype(str).str.lower() == str(item).lower()) &
        (p['param'].astype(str).str.lower() == str(param).lower())
    )
    df = p.loc[mask, ['iso3', 'AreaCode', 'year', 'process', 'item', 'param', value_col]].copy()
    if df.empty:
        # 放宽：只按 param 匹配（有时 process/item 命名略有差异）
        df = p.loc[p['param'].astype(str).str.lower() == str(param).lower(),
                   ['iso3', 'AreaCode', 'year', 'process', 'item', 'param', value_col]].copy()
        if df.empty:
            raise KeyError(f"未在参数表中找到指定条目：process='{process}', item='{item}', param='{param}'")

    return df


# ---------------------- 计算核心 ----------------------
def compute_forest_fl_fl_biomass(params: pd.DataFrame,
                                 process_hint: Iterable[str] = ('Forests', 'Forest Land'),
                                 item_hint: Iterable[str] = ('Forests total', 'Forest total', 'ForestLand total'),
                                 param_name: str = 'B_total_tC') -> pd.DataFrame:
    """
    计算 “Forest Land Remaining Forest Land（FL-FL）- 生物量碳库” 的年度 CO2（负值为碳汇）。
    - 采用存量差法：CO2_kt = - (B_t - B_{t-1}) * 44/12 / 1e3
    - 输入为“总生物量碳存量”随年份的时间序列（单位 tC）。

    返回列：['iso3','AreaCode','year','process','gas','value_ktCO2']
    其中 process 固定为 "Forest land remaining forest land (FL-FL)".
    """
    # 优先匹配 process/item 提示词；若失败则放宽到仅 param 匹配
    df_list = []
    last_err = None
    for proc in process_hint:
        for item in item_hint:
            try:
                df_list.append(get_param(params, proc, item, param_name))
            except Exception as e:
                last_err = e
                continue
    if not df_list:
        # 退化为仅 param 匹配
        df_list.append(get_param(params, process_hint[0], item_hint[0], param_name))

    df = pd.concat(df_list, ignore_index=True).drop_duplicates(subset=['iso3','year','param'], keep='last')
    df = df.rename(columns={'value': 'B_total_tC'}).sort_values(['iso3','year'])

    # 逐国计算年度差分
    def _diff_group(g):
        g = g.sort_values('year').copy()
        g['dB_tC'] = g['B_total_tC'].diff()  # tC
        # 第一年的差分为空，按 0 处理（或丢弃第一年）
        g['dB_tC'] = g['dB_tC'].fillna(0.0)
        # 负排放（碳汇）：存量增加 => 负值
        g['value_ktCO2'] = - g['dB_tC'] * _TC_TO_KTCO2
        return g

    out = df.groupby(['iso3', 'AreaCode'], as_index=False, group_keys=False).apply(_diff_group)

    # 组装结果
    out_df = out[['iso3','AreaCode','year']].copy()
    out_df['process'] = 'Forest land remaining forest land (FL-FL)'
    out_df['gas'] = 'CO2'
    out_df['value_ktCO2'] = out['value_ktCO2'].astype(float)

    return out_df


def compute_forest_fl_fl_soil_tier1_zero(params: pd.DataFrame) -> pd.DataFrame:
    """
    Tier 1 & Approach 2/3：矿质土壤 ΔSOC = 0；有机土壤：无细化 => 记 0。
    如需在 Approach 1 下进行 SOC 起止库存估算，请开启 ENABLE_SOIL_APPROACH1 并在 TODO 区域实现。
    返回与 biomass 同结构的 0 向量，便于与生物量结果拼接或校验。
    """
    base = params.copy()
    base = _standardize_cols(base)
    cols_need = [c for c in ['iso3','AreaCode','year'] if c not in base.columns]
    if cols_need:
        raise KeyError(f"参数表缺少必要列: {cols_need}")

    soil = base[['iso3','AreaCode','year']].drop_duplicates().copy()
    soil['process'] = 'Forest land remaining forest land (FL-FL)'
    soil['gas'] = 'CO2'
    soil['value_ktCO2'] = 0.0

    if ENABLE_SOIL_APPROACH1:
        # TODO: 在此实现基于 Eq.2.25 的 SOC 起止库存法（需要：SOCREF/FLU/FMG/FI/FND/分区气候土壤/土地面积等）
        # 目前保持为 0，以免与土地利用变化模块重复或引入不闭合的土地基数。
        pass

    return soil


# ---------------------- 主入口 ----------------------
def run_gfe_forest_only(params: pd.DataFrame) -> pd.DataFrame:
    """
    仅计算 “Forest Land Remaining Forest Land（FL-FL）” 的 CO2（负排放）。
    - 生物量：按存量差法；
    - DOM：Tier 1 视同 0；
    - 土壤：Tier 1 & Approach 2/3 视同 0（默认）；

    返回 DataFrame，列：['iso3','AreaCode','year','process','gas','value_ktCO2']
    """
    biomass = compute_forest_fl_fl_biomass(params)

    # Soil/DOM 均为 0（若后续启用更高 Tier，可在此拼接）
    soil = compute_forest_fl_fl_soil_tier1_zero(params)

    # 由于 soil 全 0，拼接与否对总量无影响；保留 biomass 即可。
    out = biomass.copy()

    # 若希望显式返回“生物量 + 土壤(0)”两行，可改为：
    # out = pd.concat([biomass, soil], ignore_index=True)

    # 排序/重排
    out = out.sort_values(['iso3','year']).reset_index(drop=True)
    return out


# 兼容旧接口名
def run_gfe(params: pd.DataFrame) -> pd.DataFrame:
    """
    向后兼容：保留旧入口名 `run_gfe`，但仅返回 FL-FL 结果。
    """
    return run_gfe_forest_only(params)


if __name__ == "__main__":
    # 简单自测（示例）：构造 2 国的 B_total_tC 时间序列，检查负排放方向是否正确。
    demo = pd.DataFrame({
        'iso3': ['AAA']*3 + ['BBB']*3,
        'AreaCode': [1]*3 + [2]*3,
        'year': [2000, 2001, 2002, 2000, 2001, 2002],
        'process': ['Forests']*6,
        'item': ['Forests total']*6,
        'param': ['B_total_tC']*6,
        'value': [1000, 1100, 1200, 2000, 1950, 1900],  # AAA在增长(碳汇), BBB在下降(碳源)
    })
    res = run_gfe(demo)
    print(res)
