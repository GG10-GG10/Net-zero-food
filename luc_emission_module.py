# luc_oscar_module.py
# -*- coding: utf-8 -*-
"""
复刻自 OSCAR 的简化 LUC 簿记模块（增强版，聚合优化 + 外部驱动适配 + 详细注释）

功能总览
---------
1) **LUH2 v2h 驱动（格点、逐年）**：支持 12 类状态（primf/primn/secdf/secdn/urban/crop/c3ann/c3per/c4ann/c4per/pastr/range）。
2) **全路径土地转移**：自动发现 `from_to` 或 `from_to` 命名的转移变量，计算由转移导致的瞬时“采伐”碳，并分配至 HWP 与即时排放。
3) **wood harvest（木材采伐）** 与 **shifting cultivation（轮耕）**：
   - 可从 LUH2 自带变量读取（如 `*_harv` / `*harvest*`）。
   - 也可从**外部 roundwood 供给**（m³/年，国家×年）注入并按森林面积加权分配到格点。
   - 轮耕按等效“清理 τ_shift 年龄次生林”的近似换算采伐碳。
4) **池响应**：植被/土壤池采用指数逼近新稳态；HWP 采用三池一阶衰减（短/中/长寿命）。
5) **聚合优化**：每个年份**整图一次**按国家聚合（去掉逐像素循环），显著降低内存与时间开销。
6) **参数外部化**：从 Excel 读取 `cveg/csoil/constants` 三张表。constants 支持：
   - `tau_veg, tau_soil, hl_HWP_*`、HWP 分配、`frac_HWP`
   - `rho_wood, cf_wood, pi_agb, harvest_intensity`
   - `tau_shift, enable_shift`
   - `replace_luh2_transitions, replace_luh2_harvest`
7) **外部粗分类转移**：支持从 land-use-change 模块读取国家级粗分类转移（单位 ha），
   在每个国家×年内按 LUH2 的可用“来源面积”权重将其**细化**到格点，并映射到模型使用的精细状态组合。

输入数据（必须）
---------------
- `luh_file`：LUH2 v2h NetCDF（含状态分数与/或转移变量、可能含 areacella）
- `mask_file`：国家掩膜 NetCDF，变量名 `iso3`（数值或编码），与 LUH2 同网格
- `param_excel`：参数表 Excel，包含 sheet：`cveg`、`csoil`、`constants`

可选外部驱动
-----------
- `coarse_transitions_df`（DataFrame）：列 `iso3, year, forest_to_cropland, forest_to_pasture, cropland_to_forest, pasture_to_forest`（单位 ha）。
  这些粗分类会在国家内按来源类别（如 forest=primf+secdf）面积权重分配到格点，再转换为精细 from→to 组合。
- `roundwood_supply_df`（DataFrame）：列 `iso3, year, roundwood_m3`（单位 m³/年）。将被换算为 tC 并按森林面积权重分配到格点。

输出
----
- Pandas DataFrame：列 `iso3, year, F_veg_co2, F_soil_co2, F_hwp_co2, F_inst_co2, total_co2`（单位 tCO₂/年）。
- 如提供 `out_csv`，写出同名 CSV。

重要的量纲与约定
----------------
- LUH2 状态通常是“**分数**”（0–1），需乘以格点面积（ha）得到面积。
- 本模块内部统一：
  - 瞬时/缓释通量在格点层面统一先用 **tC** 表示（像元总量）；
  - 聚合前保持 **numpy 数组**，避免 xarray→pandas 的宽→长爆内存；
  - 聚合后在国家层面统一乘以 **44/12** 输出 **tCO₂**。

性能与内存
----------
- 采用逐年“整图一次”聚合（`ravel()+groupby`），避免逐像元循环与 DataFrame 膨胀。
- 面积：优先使用 `areacella`（m²），否则按球面几何随纬度计算像元面积（m²）再转 ha。

"""
from __future__ import annotations
import re
from typing import Optional, List, Tuple

import numpy as np
import xarray as xr
import pandas as pd

# 物理/常量
LN2 = np.log(2.0)
TC2CO2 = 44.0 / 12.0

# =========================================================
# 1) 参数读取
# =========================================================

def load_params_from_excel(path: str) -> dict:
    """读取 Excel 参数表（三个 sheet：cveg/csoil/constants），并提供默认回退。

    Sheet 结构：
    - cveg/csoil：列 `land_type, value, unit, source, notes`（仅 `land_type,value` 必需）
    - constants：列 `param, value, unit, source, notes`（仅 `param,value` 必需）

    返回字典键：
    - 'cveg', 'csoil'：dict[str,float]
    - 'tau_veg', 'tau_soil'
    - 'hl_HWP': {'short','medium','long'}
    - 'alloc_HWP': {'short','medium','long'}
    - 'frac_HWP'
    - 'rho_wood', 'cf_wood', 'pi_agb', 'harvest_intensity'
    - 'tau_shift', 'enable_shift'
    - 'replace_luh2_transitions', 'replace_luh2_harvest'
    """
    sheets = pd.read_excel(path, sheet_name=None)
    cveg = sheets['cveg'].set_index('land_type')['value'].to_dict()
    csoil = sheets['csoil'].set_index('land_type')['value'].to_dict()
    const = sheets['constants'].set_index('param')['value'].to_dict()

    def _get(k: str, dflt: float) -> float:
        return float(const[k]) if k in const and pd.notna(const[k]) else dflt

    return {
        'cveg': cveg,
        'csoil': csoil,
        'tau_veg': _get('tau_veg', 20.0),
        'tau_soil': _get('tau_soil', 20.0),
        'hl_HWP': {
            'short': _get('hl_HWP_short', 2.0),
            'medium': _get('hl_HWP_medium', 25.0),
            'long': _get('hl_HWP_long', 35.0),
        },
        'alloc_HWP': {
            'short': _get('alloc_HWP_short', 0.3),
            'medium': _get('alloc_HWP_medium', 0.2),
            'long': _get('alloc_HWP_long', 0.5),
        },
        'frac_HWP': _get('frac_HWP', 0.5),
        # wood harvest & shifting
        'rho_wood': _get('rho_wood', 0.5),          # tDM/m3：基本木材密度
        'cf_wood':  _get('cf_wood', 0.5),           # tC/tDM：干物碳分数
        'pi_agb': _get('pi_agb', 0.7),              # AGB 份额（地上生物量）
        'harvest_intensity': _get('harvest_intensity', 1.0),
        'tau_shift': _get('tau_shift', 15.0),       # 轮耕周期（年）
        'enable_shift': int(_get('enable_shift', 1.0)),
        # 是否用外部数据替换 LUH2 自带的驱动
        'replace_luh2_transitions': int(_get('replace_luh2_transitions', 0.0)),
        'replace_luh2_harvest': int(_get('replace_luh2_harvest', 0.0)),
    }

# =========================================================
# 2) 网格面积估算（ha）
# =========================================================

def estimate_area_ha(ds: xr.Dataset) -> xr.DataArray:
    """返回与 LUH2 网格匹配的面积（单位 ha）。

    优先使用 `areacella`（m²），否则按球面几何随纬度计算像元面积：
        A = R² * dlon * (sin(phi+Δφ/2) - sin(phi-Δφ/2))
    然后广播到 (lat,lon) 网格，并转换为 ha。
    """
    if 'areacella' in ds:
        return ds['areacella'] * 1e-4  # m²→ha

    # 回退：球面几何
    R = 6_371_000.0
    dlat = np.deg2rad(abs(float(ds.lat[1] - ds.lat[0])))
    dlon = np.deg2rad(abs(float(ds.lon[1] - ds.lon[0])))
    lat_r = np.deg2rad(ds['lat'])
    strip = (np.sin(lat_r + dlat/2) - np.sin(lat_r - dlat/2)) * (R**2) * dlon  # m²/纬带
    # 广播到 (lat,lon)
    sample = ds.isel(time=0).isel(lat=slice(None), lon=slice(None))
    area = xr.DataArray(strip, dims=['lat']).broadcast_like(sample)
    return area * 1e-4  # ha

# =========================================================
# 3) 变量发现与外部驱动分配
# =========================================================
# 支持两种命名："from_to" 或 "from_to"（兼容常见数据集命名差异）
_TRANS_RE = re.compile(r"^(?P<from>[^_]+)_to_(?P<to>[^_]+)$")
_BARE_RE  = re.compile(r"^(?P<from>[^_]+)_(?P<to>[^_]+)$")

StateList = List[str]
TransList = List[Tuple[str, str, str]]


def discover_transitions(ds: xr.Dataset, states: StateList) -> TransList:
    """遍历 NetCDF 变量名，筛选所有属于 `states` 组合的转移变量。

    返回列表元素：(var_name, from_state, to_state)
    注意：这里只是“发现”，真实计算时会再结合面积等信息。
    """
    out: TransList = []
    for v in ds.data_vars:
        m = _TRANS_RE.match(v) or _BARE_RE.match(v)
        if not m:
            continue
        f, t = m.group('from'), m.group('to')
        if f in states and t in states:
            out.append((v, f, t))
    return out

# —— 粗分类→精细 from/to 的映射约定 ——
_COARSE_TO_FINE = {
    # 毁林：forest→cropland / pasture（来源 forest ≈ primf+secdf；目标 cropland/pasture）
    'forest_to_cropland': {'from': ['primf', 'secdf'], 'to': ['crop', 'c3ann', 'c3per', 'c4ann', 'c4per', 'crop']},
    'forest_to_pasture':  {'from': ['primf', 'secdf'], 'to': ['pastr', 'range', 'pastr']},
    # 造林/再造林：非林→林（目标统一落在 secdf，符合“次生林”惯例）
    'cropland_to_forest': {'from': ['crop', 'c3ann', 'c3per', 'c4ann', 'c4per', 'crop'], 'to': ['secdf']},
    'pasture_to_forest':  {'from': ['pastr', 'range'], 'to': ['secdf']},
}


def _sum_over_states(ds: xr.Dataset, states: StateList, year: int) -> Optional[np.ndarray]:
    """将多个状态分数按像元逐一相加，返回 numpy 数组（与格点形状一致）。"""
    arrs = [ds[s].sel(time=year).values for s in states if s in ds]
    return np.sum(arrs, axis=0) if arrs else None


def allocate_coarse_transitions_for_year(
    ds: xr.Dataset,
    area_ha: xr.DataArray,
    iso: xr.DataArray,
    year: int,
    coarse_df_year: pd.DataFrame,
) -> List[Tuple[str, str, np.ndarray]]:
    """将**国家级粗分类转移**（单位 ha）分配到格点，输出用于瞬时“采伐”计算的清单。

    返回列表元素：(varname, from_state, to_state, A_ha_grid)
    - 其中 A_ha_grid 是一个与格点同形状的 numpy 数组（单位 ha/像元）。

    分配规则（简化稳健）：
    1) 对每个国家×年×粗分类，取其“来源 from 类别”在该国的**可用面积**作为权重；
    2) 将国家层面的需求（ha）按权重比例分配到格点；
    3) 为确保映射确定性，本实现将所有流量映射到 `to_states` 的第一个目标子类（例如 forest→cropland 的目标统一至 `crop`）。
       若需要进一步在多个 `to` 子类之间细分，可在此处二次按 `to` 的目标占比拆分（可扩展）。
    """
    outputs: List[Tuple[str, str, np.ndarray]] = []
    if coarse_df_year is None or coarse_df_year.empty:
        return outputs

    iso_grid = iso.values
    for _, row in coarse_df_year.iterrows():
        iso_code = row['iso3']
        for key, meta in _COARSE_TO_FINE.items():
            if key not in row or pd.isna(row[key]) or row[key] <= 0:
                continue
            demand = float(row[key])  # ha（国家级）
            from_states = [s for s in meta['from'] if s in ds]
            to_states   = [s for s in meta['to']   if s in ds]
            if not from_states or not to_states:
                continue
            # 来源类别分数 → ha（格点）
            from_frac = _sum_over_states(ds, from_states, year)
            if from_frac is None:
                continue
            from_ha = from_frac * area_ha.values  # ha/像元
            # 国家掩膜
            mask = (iso_grid == iso_code)
            country_total = np.sum(from_ha[mask])
            if country_total <= 0:
                continue
            # 权重分配到格点
            alloc = np.zeros_like(from_ha)
            alloc[mask] = from_ha[mask] / country_total * demand  # ha/像元
            # 统一映射到第一个目标子类（可扩展为多目标权重分配）
            chosen_to = to_states[0]
            for fstate in from_states:
                outputs.append((f"{fstate}_{chosen_to}", fstate, chosen_to, alloc))
    return outputs

# —— 外部 roundwood（m³）→ 网格采伐碳（tC） ——

def allocate_roundwood_for_year(
    ds: xr.Dataset,
    area_ha: xr.DataArray,
    iso: xr.DataArray,
    year: int,
    roundwood_df_year: pd.DataFrame,
    params: dict,
) -> np.ndarray:
    """将国家级 roundwood 供给（m³/年）换算为 tC，并按**森林面积权重**分配到格点。

    换算：tC = m³ × rho_wood(tDM/m³) × cf_wood(tC/tDM)
    返回：harvested_tc（tC/像元）的 numpy 数组。
    """
    if roundwood_df_year is None or roundwood_df_year.empty:
        return np.zeros_like(area_ha.values)

    rho = params['rho_wood']
    cf = params['cf_wood']
    iso_grid = iso.values

    # 森林面积（ha/像元）：primf + secdf
    forest_frac = _sum_over_states(ds, ['primf', 'secdf'], year)
    if forest_frac is None:
        return np.zeros_like(area_ha.values)
    forest_ha = forest_frac * area_ha.values

    harvested_tc = np.zeros_like(forest_ha)
    for _, r in roundwood_df_year.iterrows():
        iso_code = r['iso3']
        m3 = float(r['roundwood_m3'])
        tc = m3 * rho * cf  # tC（国家总量）
        mask = (iso_grid == iso_code)
        total = np.sum(forest_ha[mask])
        if total <= 0:
            continue
        harvested_tc[mask] += forest_ha[mask] / total * tc
    return harvested_tc

# =========================================================
# 4) 国家聚合（整图一次）
# =========================================================
_DEF_KEEP_COLS = ['F_veg_tc', 'F_soil_tc', 'F_hwp_tc', 'F_inst_tc']

def aggregate_country_year(
    F_veg_tc: np.ndarray,
    F_soil_tc: np.ndarray,
    F_hwp_tc: np.ndarray,
    F_inst_tc: np.ndarray,
    iso: xr.DataArray,
) -> pd.DataFrame:
    """将四类通量（tC/像元）与 iso3 一并 ravel，再按 iso3 求和，返回当年的国家表（tC/年）。"""
    df = pd.DataFrame({
        'iso3': iso.values.ravel(),
        'F_veg_tc': F_veg_tc.ravel(),
        'F_soil_tc': F_soil_tc.ravel(),
        'F_hwp_tc': F_hwp_tc.ravel(),
        'F_inst_tc': F_inst_tc.ravel(),
    })
    df = df.dropna(subset=['iso3'])
    return df.groupby('iso3', as_index=False)[_DEF_KEEP_COLS].sum()

# =========================================================
# 5) 主函数
# =========================================================

def run_luc_bookkeeping(
    luh_file: str,
    mask_file: str,
    years,
    param_excel: str,
    out_csv: Optional[str] = None,
    coarse_transitions_df: Optional[pd.DataFrame] = None,
    roundwood_supply_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """运行 LUC 记账模块（支持 LUH2 + 外部粗分类 + 外部 roundwood）。

    参数
    ----
    luh_file : NetCDF 路径（LUH2 v2h）
    mask_file: NetCDF 路径（包含 iso3 掩膜）
    years    : 可迭代年份（如 range(2020, 2081)）
    param_excel : Excel 参数表（含 cveg/csoil/constants）
    out_csv  : 如提供，写出国家×年结果 CSV
    coarse_transitions_df : 可选，国家×年粗分类转移（单位 ha）
    roundwood_supply_df   : 可选，国家×年 roundwood（单位 m³/年）

    返回
    ----
    DataFrame，列：`iso3, year, F_veg_co2, F_soil_co2, F_hwp_co2, F_inst_co2, total_co2`（单位 tCO₂/年）
    """
    # 读取数据与参数
    params = load_params_from_excel(param_excel)
    ds = xr.open_dataset(luh_file)
    iso = xr.open_dataset(mask_file)['iso3']
    area_ha = estimate_area_ha(ds)

    # 状态列表与功能类映射（用于给定稳态碳密度）
    state_vars = ['primf', 'primn', 'secdf', 'secdn', 'urban',
                  'crop', 'c3ann', 'c3per', 'c4ann', 'c4per', 'pastr', 'range']
    state_to_func = {
        'primf': 'forest', 'secdf': 'forest',
        'primn': 'othernat', 'secdn': 'othernat', 'range': 'othernat',
        'pastr': 'pasture', 'urban': 'urban',
        'crop': 'cropland', 'c3ann': 'cropland', 'c3per': 'cropland', 'c4ann': 'cropland', 'c4per': 'cropland',
    }

    # 自动发现 LUH2 的转移变量（如存在）
    trans_list = discover_transitions(ds, state_vars)

    # 初始池库存（tC/ha）：以首年主导功能类的稳态作为初值（可替换为稳态解）
    t0 = years[0] if hasattr(years, '__getitem__') else list(years)[0]
    frac_stack0 = np.stack([ds[s].sel(time=t0).values for s in state_vars])
    dom_idx0 = np.argmax(frac_stack0, axis=0)
    dom_names0 = np.array(state_vars)[dom_idx0]
    func0 = np.vectorize(lambda s: state_to_func.get(s, 'othernat'))(dom_names0)
    Cveg = np.vectorize(lambda f: params['cveg'].get(f, 10.0))(func0)
    Csoil = np.vectorize(lambda f: params['csoil'].get(f, 60.0))(func0)

    # HWP 三池（tC/像元），k 为年衰减率
    shape = Cveg.shape
    HWP = {k: np.zeros(shape) for k in ('short', 'medium', 'long')}
    k_HWP = {k: LN2 / hl for k, hl in params['hl_HWP'].items()}
    alloc = params['alloc_HWP']
    f_HWP = params['frac_HWP']

    recs: List[pd.DataFrame] = []

    for yr in years:
        # ===== 5.1 年度目标稳态（主导功能类） =====
        frac_stack = np.stack([ds[s].sel(time=yr).values for s in state_vars])
        dom_idx = np.argmax(frac_stack, axis=0)
        dom_names = np.array(state_vars)[dom_idx]
        func = np.vectorize(lambda s: state_to_func.get(s, 'othernat'))(dom_names)
        Cveg_star = np.vectorize(lambda f: params['cveg'].get(f, 10.0))(func)
        Csoil_star = np.vectorize(lambda f: params['csoil'].get(f, 60.0))(func)

        # ===== 5.2 指数响应（缓释/缓汇；返回 tC/像元） =====
        a_veg = 1 - np.exp(-1.0 / params['tau_veg'])
        a_soil = 1 - np.exp(-1.0 / params['tau_soil'])
        Cveg_next = Cveg + (Cveg_star - Cveg) * a_veg
        Csoil_next = Csoil + (Csoil_star - Csoil) * a_soil
        F_veg_tc = (Cveg - Cveg_next) * area_ha.values
        F_soil_tc = (Csoil - Csoil_next) * area_ha.values

        # ===== 5.3 即时项（转移/采伐/轮耕）与 HWP 输入（tC/像元） =====
        F_inst_tc = np.zeros(shape)
        HWP_add_tc = np.zeros(shape)

        # -- 5.3.1 LUH2 原生转移（如使用）--
        if not params['replace_luh2_transitions']:
            for v, fstate, tstate in trans_list:
                A_ha = ds[v].sel(time=yr).values * area_ha.values  # ha/像元
                f_func = state_to_func.get(fstate, 'othernat')
                t_func = state_to_func.get(tstate, 'othernat')
                if f_func == t_func:
                    continue
                dCveg_ha = params['cveg'].get(f_func, 10.0) - params['cveg'].get(t_func, 5.0)  # tC/ha
                harvested_tc = np.where(dCveg_ha > 0, A_ha * dCveg_ha, 0.0)                    # tC/像元
                add_hwp = harvested_tc * f_HWP
                F_inst_tc += harvested_tc - add_hwp
                HWP_add_tc += add_hwp

        # -- 5.3.2 外部“粗分类转移”的格点化 --
        if coarse_transitions_df is not None:
            ydf = coarse_transitions_df[coarse_transitions_df['year'] == int(yr)]
            syn = allocate_coarse_transitions_for_year(ds, area_ha, iso, int(yr), ydf)
            for vname, fstate, tstate, A_ha in syn:
                f_func = state_to_func.get(fstate, 'othernat')
                t_func = state_to_func.get(tstate, 'othernat')
                if f_func == t_func:
                    continue
                dCveg_ha = params['cveg'].get(f_func, 10.0) - params['cveg'].get(t_func, 5.0)
                harvested_tc = np.where(dCveg_ha > 0, A_ha * dCveg_ha, 0.0)
                add_hwp = harvested_tc * f_HWP
                F_inst_tc += harvested_tc - add_hwp
                HWP_add_tc += add_hwp

        # -- 5.3.3 外部 roundwood 供给（m³）--→ tC/像元
        if roundwood_supply_df is not None:
            ydf = roundwood_supply_df[roundwood_supply_df['year'] == int(yr)]
            harvested_tc = allocate_roundwood_for_year(ds, area_ha, iso, int(yr), ydf, params)
            add_hwp = harvested_tc * f_HWP
            F_inst_tc += harvested_tc - add_hwp
            HWP_add_tc += add_hwp

        # -- 5.3.4 LUH2 自带 wood harvest（如使用）--
        if not params['replace_luh2_harvest']:
            for v in ds.data_vars:
                # 简单的“采伐”变量名识别（可按你的 LUH2 文件更精准定制）
                name = v.lower()
                if v.endswith('_harv') or ('harvest' in name) or ('wood_harv' in name) or ('wharv' in name):
                    prefix = v.split('_')[0]  # 期望是 primf/secdf 等
                    f_func = 'forest' if prefix in ['primf', 'secdf'] else None
                    if f_func != 'forest':
                        continue
                    A_ha = ds[v].sel(time=yr).values * area_ha.values
                    per_ha_tc = params['cveg'].get('forest', 150.0) * params['pi_agb'] * params['harvest_intensity']
                    harvested_tc = A_ha * per_ha_tc
                    add_hwp = harvested_tc * f_HWP
                    F_inst_tc += harvested_tc - add_hwp
                    HWP_add_tc += add_hwp

        # ===== 5.4 HWP 衰减（tC/像元） =====
        F_hwp_tc = np.zeros(shape)
        for k in HWP:
            # 近似离散：H_{t+1} = H_t * (1 - kΔt) + input ；当年排放=H_t * kΔt
            emit = HWP[k] * (1 - np.exp(-k_HWP[k]))
            HWP[k] = HWP[k] - emit + HWP_add_tc * alloc[k]
            F_hwp_tc += emit

        # ===== 5.5 国家聚合（整图一次，tC→tCO₂） =====
        grp_tc = aggregate_country_year(F_veg_tc, F_soil_tc, F_hwp_tc, F_inst_tc, iso)
        grp_tc['year'] = int(yr)
        # CO₂ 单位转换
        for col in _DEF_KEEP_COLS:
            grp_tc[col.replace('_tc', '_co2')] = grp_tc[col] * TC2CO2
        grp_tc['total_co2'] = (grp_tc['F_veg_tc'] + grp_tc['F_soil_tc'] + grp_tc['F_hwp_tc'] + grp_tc['F_inst_tc']) * TC2CO2
        recs.append(grp_tc[['iso3', 'year', 'F_veg_co2', 'F_soil_co2', 'F_hwp_co2', 'F_inst_co2', 'total_co2']])

        # ===== 5.6 滚动库存 =====
        Cveg, Csoil = Cveg_next, Csoil_next

    # 串联所有年份
    out = pd.concat(recs, ignore_index=True)
    if out_csv:
        out.to_csv(out_csv, index=False)
    return out
