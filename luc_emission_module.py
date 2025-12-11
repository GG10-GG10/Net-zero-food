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
import os
import re
from typing import Optional, List, Tuple, Dict
import logging

import numpy as np
import xarray as xr
import pandas as pd
from config_paths import get_src_base

# 物理/常量
LN2 = np.log(2.0)
TC2CO2 = 44.0 / 12.0

# ===== 诊断日志辅助函数 =====
def _log_to_model(msg: str) -> None:
    """将诊断信息写入model.log"""
    try:
        log_path = os.path.join(os.getcwd(), 'model.log')
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
    except Exception:
        pass  # 静默失败，不影响主流程
    # 同时输出到控制台
    print(msg)

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
    import os
    if not os.path.exists(path):
        print(f"[WARN] 参数文件不存在: {path}，使用默认参数")
        return {
            'cveg': {'forest': 150.0, 'cropland': 5.0, 'pasture': 10.0},
            'csoil': {'forest': 80.0, 'cropland': 50.0, 'pasture': 70.0},
            'rho_wood': 0.5,
            'cf_wood': 0.5,
            'pi_agb': 0.7,
        }
    
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
        # ✅ 森林碳汇速率（tC/ha/年），负值表示碳吸收
        # 按森林类型区分：Tropical=-6, Temperate=-3, Boreal=-0.8 (IPCC数据)
        'forest_c_sink_rate_Tropical': _get('forest_c_sink_rate_Tropical', -6.0),   # tC/ha/年（热带森林）
        'forest_c_sink_rate_Temperate': _get('forest_c_sink_rate_Temperate', -3.0), # tC/ha/年（温带森林）
        'forest_c_sink_rate_Boreal': _get('forest_c_sink_rate_Boreal', -0.8),       # tC/ha/年（寒带森林）
        'forest_c_sink_rate_Average': _get('forest_c_sink_rate_Average', -2.5),     # tC/ha/年（全球平均）
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
    lat_vals = np.asarray(ds['lat'].values, dtype=float)
    lon_vals = np.asarray(ds['lon'].values, dtype=float)
    if lat_vals.size < 2 or lon_vals.size < 2:
        raise ValueError("LUH2 dataset lat/lon dimensions are insufficient to estimate cell area")
    dlat = np.deg2rad(abs(float(lat_vals[1] - lat_vals[0])))
    dlon = np.deg2rad(abs(float(lon_vals[1] - lon_vals[0])))
    lat_r = np.deg2rad(lat_vals)
    strip = (np.sin(lat_r + dlat / 2.0) - np.sin(lat_r - dlat / 2.0)) * (R ** 2) * dlon  # m²/纬带
    area = np.repeat(strip[:, None], lon_vals.size, axis=1)
    return xr.DataArray(area, coords={'lat': ds['lat'], 'lon': ds['lon']}, dims=('lat', 'lon')) * 1e-4


def _ensure_year_dim(ds: xr.Dataset, target_years: Optional[List[int]] = None) -> xr.Dataset:
    """确保 LUH2 数据集使用整数年份维度，并按需重建年份坐标。"""
    if 'year' not in ds.dims:
        if 'time' not in ds.dims:
            raise KeyError("LUH2 dataset must contain 'time' or 'year' dimension")
        years = [int(getattr(t, 'year', getattr(t, 'year', t))) for t in ds['time'].values]
        ds = ds.assign_coords(year=('time', years)).swap_dims({'time': 'year'}).sortby('year')
    if target_years:
        target_years = sorted({int(y) for y in target_years})
        ds = ds.reindex(year=target_years, method='nearest')
    return ds


def _sel_year(arr: xr.DataArray, year: int) -> xr.DataArray:
    """选择指定年份数据，缺失时回退至最近年份。"""
    if 'year' not in arr.coords:
        raise KeyError("DataArray missing 'year' coordinate")
    if year in arr['year']:
        return arr.sel(year=year)
    return arr.sel(year=year, method='nearest')


def _build_iso_mask(mask_ds: xr.Dataset) -> xr.DataArray:
    """从掩膜文件生成 iso3 DataArray，必要时利用 id1→ISO3 映射。"""
    if 'iso3' in mask_ds:
        iso = mask_ds['iso3']
        return iso.astype(str)
    if 'id1' not in mask_ds:
        raise KeyError("Mask dataset must contain 'iso3' or 'id1'")
    id_array = np.asarray(mask_ds['id1'].values, dtype=float)
    id_array = np.nan_to_num(id_array, nan=0.0)
    id_int = id_array.astype(np.int64)
    try:
        region_df = pd.read_excel(os.path.join(get_src_base(), 'dict_v3.xlsx'), 'region')
        region_df.columns = [str(c).strip() for c in region_df.columns]
        region_df = region_df[['Region_maskID', 'ISO3 Code']].dropna()
        region_df['Region_maskID'] = pd.to_numeric(region_df['Region_maskID'], errors='coerce').astype('Int64')
        region_df = region_df.dropna(subset=['Region_maskID'])
        id_to_iso = {int(r.Region_maskID): str(r['ISO3 Code']).strip()
                     for r in region_df.itertuples(index=False)}
    except Exception:
        id_to_iso = {}
    iso_vals = np.full(id_int.shape, '', dtype=object)
    for mask_id, iso in id_to_iso.items():
        iso_vals[id_int == mask_id] = iso
    return xr.DataArray(iso_vals, coords=mask_ds['id1'].coords, dims=mask_ds['id1'].dims, name='iso3')

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


def _sum_states_cached(state_cache: Dict[str, np.ndarray], states: StateList, year_idx: int) -> Optional[np.ndarray]:
    """在缓存中聚合指定年份的状态分数。"""
    arrs = [state_cache[s][year_idx] for s in states if s in state_cache]
    if not arrs:
        return None
    return np.sum(arrs, axis=0)


def allocate_coarse_transitions_for_year(
    state_cache: Dict[str, np.ndarray],
    area_ha_array: np.ndarray,
    iso_grid: np.ndarray,
    year_idx: int,
    year_val: int,
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

    for _, row in coarse_df_year.iterrows():
        iso_code = str(row['iso3']).strip()
        for key, meta in _COARSE_TO_FINE.items():
            if key not in row or pd.isna(row[key]) or row[key] <= 0:
                continue
            demand = float(row[key])  # ha（国家级）
            from_states = [s for s in meta['from'] if s in state_cache]
            to_states   = [s for s in meta['to']   if s in state_cache]
            if not from_states or not to_states:
                continue
            # 来源类别分数 → ha（格点）
            from_frac = _sum_states_cached(state_cache, from_states, year_idx)
            if from_frac is None:
                continue
            from_ha = from_frac * area_ha_array  # ha/像元
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
    state_cache: Dict[str, np.ndarray],
    area_ha_array: np.ndarray,
    iso_grid: np.ndarray,
    year_idx: int,
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

    # 森林面积（ha/像元）：primf + secdf
    forest_frac = _sum_states_cached(state_cache, ['primf', 'secdf'], year_idx)
    if forest_frac is None:
        return np.zeros_like(area_ha_array, dtype=float)
    forest_ha = forest_frac * area_ha_array

    harvested_tc = np.zeros_like(forest_ha, dtype=float)
    for _, r in roundwood_df_year.iterrows():
        iso_code = str(r['iso3']).strip()
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

STATE_TO_CATEGORY = {
    'primf': 'forest', 'secdf': 'forest',
    'primn': 'othernat', 'secdn': 'othernat', 'range': 'othernat',
    'pastr': 'pasture', 'urban': 'urban',
    'crop': 'cropland', 'c3ann': 'cropland', 'c3per': 'cropland',
    'c4ann': 'cropland', 'c4per': 'cropland', 'c3nfx': 'cropland',
}

TRANSITION_LABELS = {
    ('forest', 'cropland'): 'forest_to_cropland',
    ('forest', 'pasture'): 'forest_to_pasture',
    ('cropland', 'forest'): 'cropland_to_forest',
    ('pasture', 'forest'): 'pasture_to_forest',
}


def _aggregate_area_by_iso(A_ha: np.ndarray, iso: xr.DataArray) -> pd.DataFrame:
    df = pd.DataFrame({
        'iso3': iso.values.ravel(),
        'area_ha': A_ha.ravel(),
    })
    df = df.dropna(subset=['iso3'])
    df['area_ha'] = pd.to_numeric(df['area_ha'], errors='coerce')
    df = df[df['area_ha'] > 0.0]
    if df.empty:
        return df
    df['iso3'] = df['iso3'].astype(str).str.strip()
    df = df[df['iso3'].astype(str).str.len() > 0]
    return df.groupby('iso3', as_index=False)['area_ha'].sum()


def _to_year_lat_lon(arr: xr.DataArray, allow_year: bool = True) -> np.ndarray:
    dims = arr.dims
    target = []
    if allow_year and 'year' in dims:
        target.append('year')
    for axis in ('lat', 'latitude'):
        if axis in dims:
            target.append(axis)
            break
    for axis in ('lon', 'longitude'):
        if axis in dims:
            target.append(axis)
            break
    if allow_year and 'year' not in dims:
        raise ValueError("DataArray 缺少 year 维度")
    if len(target) < (3 if allow_year else 2):
        raise ValueError(f"DataArray 缺少 lat/lon 维度: {dims}")
    arr_t = arr.transpose(*target)
    data = np.asarray(arr_t.values)
    if allow_year:
        if data.ndim != 3:
            raise ValueError("期望三维数组 (year, lat, lon)")
    else:
        if data.ndim != 2:
            raise ValueError("期望二维数组 (lat, lon)")
    return data


def run_luc_bookkeeping(
    luh_file: str,
    mask_file: str,
    years,
    param_excel: str,
    out_csv: Optional[str] = None,
    coarse_transitions_df: Optional[pd.DataFrame] = None,
    roundwood_supply_df: Optional[pd.DataFrame] = None,
    transitions_file: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    transitions_file      : 可选，LUH2 transitions NetCDF（若与 luh_file 分离）

    返回
    ----
    (emissions_df, transitions_df)
      emissions_df: `iso3, year, F_veg_co2, F_soil_co2, F_hwp_co2, F_inst_co2, total_co2`
      transitions_df: `iso3, year, transition, area_ha`
    """
    # 读取数据与参数
    params = load_params_from_excel(param_excel)
    try:
        year_list = [int(y) for y in years]
    except TypeError:
        year_list = [int(y) for y in list(years)]
    if not year_list:
        raise ValueError("years must contain at least one value")
    year_list = sorted(set(year_list))

    load_all = len(year_list) <= 20
    open_kwargs = {'chunks': {'year': len(year_list)}} if load_all else {}
    ds_raw = xr.open_dataset(luh_file, **open_kwargs)
    try:
        if 'year' in ds_raw.coords:
            hist_year_max_states = int(np.max(ds_raw['year'].values))
        else:
            hist_year_max_states = int(max(year_list))
        ds = _ensure_year_dim(ds_raw, year_list)
        ds = ds.sel(year=year_list)
        if load_all:
            ds = ds.load()
    finally:
        try:
            ds_raw.close()
        except Exception:
            pass
    area_ha = estimate_area_ha(ds)
    area_array = _to_year_lat_lon(area_ha, allow_year=False)
    if transitions_file and os.path.exists(transitions_file):
        trans_open_kwargs = {'chunks': {'year': len(year_list)}} if load_all else {}
        trans_raw = xr.open_dataset(transitions_file, **trans_open_kwargs)
        try:
            if 'year' in trans_raw.coords:
                hist_year_max = int(np.max(trans_raw['year'].values))
            else:
                hist_year_max = hist_year_max_states
            trans_ds = _ensure_year_dim(trans_raw, year_list).sel(year=year_list)
            if load_all:
                trans_ds = trans_ds.load()
        finally:
            try:
                trans_raw.close()
            except Exception:
                pass
    else:
        trans_ds = ds
        hist_year_max = hist_year_max_states
    mask_ds = xr.open_dataset(mask_file)
    try:
        iso = _build_iso_mask(mask_ds)
    finally:
        mask_ds.close()
    iso_grid = np.asarray(iso.values)
    iso_grid = np.where(pd.isna(iso_grid), '', iso_grid).astype(str)

    # 状态列表与功能类映射（用于给定稳态碳密度）
    state_vars_all = ['primf', 'primn', 'secdf', 'secdn', 'urban',
                      'crop', 'c3ann', 'c3per', 'c4ann', 'c4per', 'pastr', 'range']
    state_to_func = {
        'primf': 'forest', 'secdf': 'forest',
        'primn': 'othernat', 'secdn': 'othernat', 'range': 'othernat',
        'pastr': 'pasture', 'urban': 'urban',
        'crop': 'cropland', 'c3ann': 'cropland', 'c3per': 'cropland', 'c4ann': 'cropland', 'c4per': 'cropland',
    }
    state_vars = [s for s in state_vars_all if s in ds]
    if not state_vars:
        raise ValueError("LUH2 dataset missing expected state variables")

    # 自动发现 LUH2 的转移变量（如存在）
    trans_list = discover_transitions(trans_ds, state_vars)

    years_arr = np.asarray(ds['year'].values, dtype=int)
    year_index = {int(y): idx for idx, y in enumerate(years_arr)}
    state_cache: Dict[str, np.ndarray] = {}
    for s in state_vars:
        state_cache[s] = _to_year_lat_lon(ds[s])
    trans_years_arr = np.asarray(trans_ds['year'].values, dtype=int)
    trans_year_index = {int(y): idx for idx, y in enumerate(trans_years_arr)}
    hist_year_max = int(np.max(trans_years_arr)) if trans_years_arr.size else hist_year_max
    trans_cache: Dict[str, np.ndarray] = {}
    for v, _, _ in trans_list:
        if v in trans_ds:
            trans_cache[v] = _to_year_lat_lon(trans_ds[v])

    # 初始池库存（tC/ha）：以首年主导功能类的稳态作为初值（可替换为稳态解）
    t0 = year_list[0]
    idx0 = year_index.get(t0)
    if idx0 is None:
        raise ValueError(f"Year {t0} not available in LUH2 dataset")
    frac_stack0 = np.stack([state_cache[s][idx0] for s in state_vars])
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
    transition_records: List[pd.DataFrame] = []

    for yr in year_list:
        # ===== 5.1 年度目标稳态（主导功能类） =====
        year_idx = year_index.get(yr)
        if year_idx is None:
            continue
        frac_stack = np.stack([state_cache[s][year_idx] for s in state_vars])
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
        F_veg_tc = (Cveg - Cveg_next) * area_array
        F_soil_tc = (Csoil - Csoil_next) * area_array

        # ===== 5.3 即时项（转移/采伐/轮耕）与 HWP 输入（tC/像元） =====
        F_inst_tc = np.zeros_like(area_array, dtype=float)
        HWP_add_tc = np.zeros_like(area_array, dtype=float)

        # -- 5.3.1 LUH2 原生转移（如使用）--
        if (not params['replace_luh2_transitions']) and (yr <= hist_year_max):
            for v, fstate, tstate in trans_list:
                cache = trans_cache.get(v)
                if cache is None:
                    continue
                t_idx = trans_year_index.get(yr)
                if t_idx is None:
                    continue
                A_ha = cache[t_idx] * area_array  # ha/像元
                f_func = state_to_func.get(fstate, 'othernat')
                t_func = state_to_func.get(tstate, 'othernat')
                if f_func == t_func:
                    continue
                dCveg_ha = params['cveg'].get(f_func, 10.0) - params['cveg'].get(t_func, 5.0)  # tC/ha
                harvested_tc = np.where(dCveg_ha > 0, A_ha * dCveg_ha, 0.0)                    # tC/像元
                add_hwp = harvested_tc * f_HWP
                F_inst_tc += harvested_tc - add_hwp
                HWP_add_tc += add_hwp
                label = TRANSITION_LABELS.get((STATE_TO_CATEGORY.get(fstate), STATE_TO_CATEGORY.get(tstate)))
                if label:
                    agg = _aggregate_area_by_iso(np.maximum(A_ha, 0.0), iso)
                    if not agg.empty:
                        agg['transition'] = label
                        agg['year'] = int(yr)
                        transition_records.append(agg)

        # -- 5.3.2 外部“粗分类转移”的格点化 --
        if coarse_transitions_df is not None:
            ydf = coarse_transitions_df[coarse_transitions_df['year'] == int(yr)]
            syn = allocate_coarse_transitions_for_year(state_cache, area_array, iso_grid, year_idx, int(yr), ydf)
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
            if yr > hist_year_max and not ydf.empty:
                for col in ['forest_to_cropland', 'forest_to_pasture', 'cropland_to_forest', 'pasture_to_forest']:
                    if col in ydf.columns:
                        tmp = ydf[['iso3', col]].copy()
                        tmp = tmp.rename(columns={col: 'area_ha'})
                        tmp = tmp.dropna(subset=['iso3'])
                        tmp['transition'] = col
                        tmp['year'] = int(yr)
                        tmp['iso3'] = tmp['iso3'].astype(str)
                        tmp = tmp[tmp['area_ha'] > 0.0]
                        if len(tmp):
                            transition_records.append(tmp)

        # -- 5.3.3 外部 roundwood 供给（m³）--→ tC/像元
        if roundwood_supply_df is not None:
            ydf = roundwood_supply_df[roundwood_supply_df['year'] == int(yr)]
            harvested_tc = allocate_roundwood_for_year(state_cache, area_array, iso_grid, year_idx, ydf, params)
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
                    if v not in state_cache and v in ds:
                        state_cache[v] = _to_year_lat_lon(ds[v])
                    cache = state_cache.get(v)
                    if cache is None:
                        continue
                    A_ha = cache[year_idx] * area_array
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
    emissions_df = pd.concat(recs, ignore_index=True)
    transitions_df = pd.concat(transition_records, ignore_index=True) if transition_records else pd.DataFrame(columns=['iso3', 'year', 'transition', 'area_ha'])
    if not transitions_df.empty:
        transitions_df = transitions_df[['iso3', 'year', 'transition', 'area_ha']].reset_index(drop=True)
    if out_csv:
        emissions_df.to_csv(out_csv, index=False)
    if transitions_file and trans_ds is not ds:
        trans_ds.close()
    return emissions_df, transitions_df


# =========================================================
# 6) 仅计算未来年份的简化LUC排放模块（与GLE/GCE集成）
# =========================================================

def normalize_m49(val) -> str:
    """标准化 M49 代码：格式化为'xxx（单引号+3位数字）"""
    s = str(val).strip()
    if s.startswith("'"):
        s = s[1:]
    try:
        return f"'{int(s):03d}"  # ✅ 'xxx格式
    except (ValueError, TypeError):
        return f"'{s}" if not s.startswith("'") else s

def run_luc_emissions_future(
    param_excel: str,
    luc_area_df: Optional[pd.DataFrame] = None,
    roundwood_change_df: Optional[pd.DataFrame] = None,
    forest_area_df: Optional[pd.DataFrame] = None,  # ✅ 森林绝对面积数据（用于Forest碳汇计算）
    years: Optional[list] = None,
    dict_v3_path: Optional[str] = None,
    historical_wood_harvest_ef: Optional[Dict[str, float]] = None,
    historical_forest_sink_ef: Optional[Dict[str, float]] = None,  # ✅ 历史森林碳汇EF (kt CO2/ha/yr)
    use_exponential_response: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    计算仅未来年份(>2020)的LUC排放，支持两种模式：
    1. 即时转换模式（use_exponential_response=False）：碳即时释放/吸收
    2. 指数响应模式（use_exponential_response=True）：碳随时间按τ参数衰减
    
    指数响应模型使用 constants 表中的参数：
    - tau_veg: 植被碳库响应时间尺度（默认20年）
    - tau_soil: 土壤碳库响应时间尺度（默认20年）
    
    参数
    ----
    param_excel : str
        LUC参数Excel文件路径（含 cveg/csoil/constants 三个sheet）
    luc_area_df : DataFrame, optional
        土地利用面积变化数据（含cropland_ha, forest_ha, grassland_ha等，单位ha）
    roundwood_change_df : DataFrame, optional
        Roundwood产量数据（含roundwood_m3列）
    forest_area_df : DataFrame, optional
        森林绝对面积数据（用于Forest碳汇计算）
    years : list, optional
        要计算的年份列表
    dict_v3_path : str, optional
        dict_v3.xlsx路径
    historical_wood_harvest_ef : dict, optional
        历史木材采伐排放因子：{M49_Country_Code: kt_CO2_per_m3}
    historical_forest_sink_ef : dict, optional
        历史森林碳汇排放因子：{M49_Country_Code: kt_CO2_per_ha_per_yr}
    use_exponential_response : bool
        是否使用指数响应模型（默认True）
    """
    params = load_params_from_excel(param_excel)
    
    # ✅ 从 constants 表加载指数响应参数
    tau_veg = params.get('tau_veg', 20.0)     # 植被碳库响应时间尺度（年）
    tau_soil = params.get('tau_soil', 20.0)   # 土壤碳库响应时间尺度（年）
    
    # 计算年度响应系数（指数衰减）
    # a = 1 - exp(-1/τ) 表示每年向稳态转移的比例
    a_veg = 1.0 - np.exp(-1.0 / tau_veg) if use_exponential_response else 1.0
    a_soil = 1.0 - np.exp(-1.0 / tau_soil) if use_exponential_response else 1.0
    
    print(f"[LUC] 使用{'指数响应' if use_exponential_response else '即时转换'}模型")
    if use_exponential_response:
        print(f"[LUC]   tau_veg={tau_veg:.1f}年 → a_veg={a_veg:.4f}")
        print(f"[LUC]   tau_soil={tau_soil:.1f}年 → a_soil={a_soil:.4f}")
    
    # ✅ HWP池参数
    hl_HWP = params.get('hl_HWP', {'short': 2.0, 'medium': 25.0, 'long': 35.0})
    alloc_HWP = params.get('alloc_HWP', {'short': 0.3, 'medium': 0.2, 'long': 0.5})
    frac_HWP = params.get('frac_HWP', 0.5)  # 进入HWP池的碳比例
    
    # HWP衰减率: k = ln(2) / 半衰期
    k_HWP = {k: LN2 / hl for k, hl in hl_HWP.items()}
    
    print(f"[LUC] HWP池参数:")
    print(f"[LUC]   半衰期: short={hl_HWP['short']:.0f}yr, medium={hl_HWP['medium']:.0f}yr, long={hl_HWP['long']:.0f}yr")
    print(f"[LUC]   分配: short={alloc_HWP['short']:.1%}, medium={alloc_HWP['medium']:.1%}, long={alloc_HWP['long']:.1%}")
    print(f"[LUC]   frac_HWP={frac_HWP:.1%} (进入HWP的碳比例)")
    
    # ✅ 森林碳汇速率（按类型）
    forest_sink_rates = {
        'Tropical': params.get('forest_c_sink_rate_Tropical', -6.0),
        'Temperate': params.get('forest_c_sink_rate_Temperate', -3.0),
        'Boreal': params.get('forest_c_sink_rate_Boreal', -0.8),
    }
    default_sink_rate = params.get('forest_c_sink_rate_Average', -2.5)
    print(f"[LUC] 森林碳汇速率 (tC/ha/yr): Tropical={forest_sink_rates['Tropical']}, "
          f"Temperate={forest_sink_rates['Temperate']}, Boreal={forest_sink_rates['Boreal']}, "
          f"Default={default_sink_rate}")
    
    if years is None:
        years = []
    else:
        years = sorted([int(y) for y in years if int(y) > 2020])
    
    if not years:
        return {'future': pd.DataFrame(columns=[
            'M49_Country_Code', 'Region_label_new', 'year', 'Process', 'GHG', 'value'
        ])}
    
    # 加载国家<->M49映射 + 森林类型映射
    m49_to_iso3 = {}
    iso3_to_m49 = {}
    m49_to_forest_type = {}  # ✅ 新增：M49 -> 森林类型映射
    if dict_v3_path:
        try:
            region_df = pd.read_excel(dict_v3_path, sheet_name='region', 
                                     usecols=['M49_Country_Code', 'ISO3 Code', 'Region_Forest_type'])
            # 使用 normalize_m49 统一格式
            region_df['M49_Country_Code'] = region_df['M49_Country_Code'].apply(normalize_m49)
            region_df['ISO3 Code'] = region_df['ISO3 Code'].astype(str).str.strip()
            m49_to_iso3 = dict(zip(region_df['M49_Country_Code'], region_df['ISO3 Code']))
            iso3_to_m49 = dict(zip(region_df['ISO3 Code'], region_df['M49_Country_Code']))
            
            # ✅ 加载森林类型映射
            region_df['Region_Forest_type'] = region_df['Region_Forest_type'].fillna('').astype(str).str.strip()
            m49_to_forest_type = dict(zip(region_df['M49_Country_Code'], region_df['Region_Forest_type']))
            # 统计森林类型分布
            forest_type_counts = region_df['Region_Forest_type'].value_counts().to_dict()
            print(f"[LUC] 森林类型分布: {forest_type_counts}")
        except Exception as e:
            print(f"[WARN] 无法加载dict_v3中的M49/森林类型映射: {e}")
    
    records = []
    
    # ✅ HWP池状态：{m49: {short: tC, medium: tC, long: tC}}
    # 用于跟踪每个国家的HWP池碳库存
    hwp_pools: Dict[str, Dict[str, float]] = {}
    
    def get_hwp_pool(m49: str) -> Dict[str, float]:
        if m49 not in hwp_pools:
            hwp_pools[m49] = {'short': 0.0, 'medium': 0.0, 'long': 0.0}
        return hwp_pools[m49]
    
    # ===== 处理土地利用面积驱动 =====
    if luc_area_df is not None and not luc_area_df.empty:
        df_luc = luc_area_df.copy()
        df_luc.columns = [str(c).strip() for c in df_luc.columns]
        
        # 识别年份列
        year_col = next((c for c in df_luc.columns if 'year' in c.lower()), 'year')
        df_luc = df_luc.rename(columns={year_col: 'year'})
        
        # 确保M49_Country_Code列存在
        if 'M49_Country_Code' not in df_luc.columns:
            # 尝试从其他国家列识别
            country_col = next((c for c in df_luc.columns if 'country' in c.lower()), None)
            if country_col:
                # 假设是ISO3，尝试映射到M49
                df_luc['M49_Country_Code'] = df_luc[country_col].apply(
                    lambda x: iso3_to_m49.get(str(x).strip(), str(x).strip())
                )
            else:
                print("[WARN] luc_area_df中无法识别国家标识")
                df_luc['M49_Country_Code'] = 'UNK'
        
        # 确保M49_Country_Code为标准化字符串
        df_luc['M49_Country_Code'] = df_luc['M49_Country_Code'].apply(normalize_m49)
        
        # 过滤年份
        df_luc = df_luc[df_luc['year'].astype(int) > 2020]
        
        # ✅ DEBUG: 打印输入数据统计
        print(f"[LUC DEBUG] 未来年份LUC面积变化数据: {len(df_luc)} 行")
        # 标准化列名：移除'd_'前缀（如果存在）
        rename_map = {}
        if 'd_cropland_ha' in df_luc.columns and 'cropland_ha' not in df_luc.columns:
            rename_map['d_cropland_ha'] = 'cropland_ha'
        if 'd_grassland_ha' in df_luc.columns and 'grassland_ha' not in df_luc.columns:
            rename_map['d_grassland_ha'] = 'grassland_ha'
        if 'd_forest_ha' in df_luc.columns and 'forest_ha' not in df_luc.columns:
            rename_map['d_forest_ha'] = 'forest_ha'
        
        if rename_map:
            df_luc = df_luc.rename(columns=rename_map)
            print(f"[LUC DEBUG] 列名标准化: {rename_map}")
        
        if not df_luc.empty:
            print(f"[LUC DEBUG] 列名: {list(df_luc.columns)}")
            for yr in df_luc['year'].unique():
                yr_data = df_luc[df_luc['year'] == yr]
                d_crop_sum = yr_data['cropland_ha'].sum() if 'cropland_ha' in yr_data.columns else 0
                d_grass_sum = yr_data['grassland_ha'].sum() if 'grassland_ha' in yr_data.columns else 0
                d_forest_sum = yr_data['forest_ha'].sum() if 'forest_ha' in yr_data.columns else 0
                print(f"[LUC DEBUG] {yr}年全球汇总: d_cropland={d_crop_sum:,.0f} ha, d_grassland={d_grass_sum:,.0f} ha, d_forest={d_forest_sum:,.0f} ha")
        
        if not df_luc.empty:
            # ✅ DEBUG: 统计输入数据
            emis_debug_stats = {}
            
            # 碳密度参数 (tC/ha) - 从 cveg/csoil 表读取
            forest_c_ha = params['cveg'].get('forest', 150.0)       # tC/ha 植被
            cropland_c_ha = params['cveg'].get('cropland', 5.0)     # tC/ha 植被
            pasture_c_ha = params['cveg'].get('pasture', 10.0)      # tC/ha 植被
            
            forest_soil_c_ha = params['csoil'].get('forest', 80.0)   # tC/ha 土壤
            cropland_soil_c_ha = params['csoil'].get('cropland', 50.0)
            pasture_soil_c_ha = params['csoil'].get('pasture', 70.0)
            
            # ===== 诊断日志：碳密度参数 =====
            _log_to_model(f"\n[LUC诊断] 碳密度参数 (tC/ha):")
            _log_to_model(f"  植被层: 森林={forest_c_ha:.1f}, 耕地={cropland_c_ha:.1f}, 草地={pasture_c_ha:.1f}")
            _log_to_model(f"  土壤层: 森林={forest_soil_c_ha:.1f}, 耕地={cropland_soil_c_ha:.1f}, 草地={pasture_soil_c_ha:.1f}")
            _log_to_model(f"  碳密度差(森林→耕地): 植被={forest_c_ha-cropland_c_ha:.1f}, 土壤={forest_soil_c_ha-cropland_soil_c_ha:.1f} tC/ha")
            _log_to_model(f"  响应系数: a_veg={a_veg:.4f} ({a_veg*100:.2f}%/年), a_soil={a_soil:.4f} ({a_soil*100:.2f}%/年)")
            
            # ✅ 指数响应模型：维护每个国家的碳库待释放/待吸收池
            # 结构: {m49: {'veg_crop': float, 'soil_crop': float, 'veg_pasture': float, 'soil_pasture': float}}
            # 正值=待释放（毁林），负值=待吸收（造林）
            carbon_pools: Dict[str, Dict[str, float]] = {}
            
            def get_pool(m49: str) -> Dict[str, float]:
                if m49 not in carbon_pools:
                    carbon_pools[m49] = {
                        'veg_crop': 0.0, 'soil_crop': 0.0,
                        'veg_pasture': 0.0, 'soil_pasture': 0.0
                    }
                return carbon_pools[m49]
            
            # 按年份排序处理（确保碳库状态正确累积）
            years_in_data = sorted(df_luc['year'].unique())
            
            # ===== 诊断日志：初始化年度统计结构 =====
            _log_to_model(f"\n[LUC诊断] 开始处理 {len(years_in_data)} 个年份: {years_in_data}")
            
            for year_val in years_in_data:
                year_data = df_luc[df_luc['year'] == year_val]
                
                if year_val not in emis_debug_stats:
                    emis_debug_stats[year_val] = {
                        'd_cropland_sum': 0, 'd_forest_sum': 0, 
                        'emis_crop_sum': 0, 'emis_pasture_sum': 0, 'count': 0,
                        # 新增：碳库累积统计
                        'pool_veg_crop_total': 0, 'pool_soil_crop_total': 0,
                        'pool_veg_pasture_total': 0, 'pool_soil_pasture_total': 0,
                        # 新增：排放组分
                        'emit_veg_crop': 0, 'emit_soil_crop': 0,
                        'emit_veg_pasture': 0, 'emit_soil_pasture': 0
                    }
                
                for _, row in year_data.iterrows():
                    m49 = str(row['M49_Country_Code']).strip()
                    
                    # ✅ 这些值是面积变化量(delta)，单位: ha
                    # 使用pd.isna处理NaN值
                    import pandas as pd
                    d_cropland = 0.0 if pd.isna(row.get('cropland_ha')) else float(row.get('cropland_ha', 0.0))
                    d_forest = 0.0 if pd.isna(row.get('forest_ha')) else float(row.get('forest_ha', 0.0))
                    
                    # 优先使用grassland_ha，如果不存在则用pasture_ha
                    grass_val = row.get('grassland_ha', row.get('pasture_ha', 0.0))
                    d_pasture = 0.0 if pd.isna(grass_val) else float(grass_val)
                    
                    emis_debug_stats[year_val]['d_cropland_sum'] += d_cropland
                    emis_debug_stats[year_val]['d_forest_sum'] += d_forest
                    emis_debug_stats[year_val]['d_pasture_sum'] = emis_debug_stats[year_val].get('d_pasture_sum', 0) + d_pasture
                    emis_debug_stats[year_val]['count'] += 1
                    
                    # ✅ DEBUG: 以美国(M49='840')为例打印草地变化
                    if m49 in ['840', "'840"]:
                        print(f"[LUC FUTURE DEBUG] {year_val}年 U.S. (M49={m49}): d_cropland={d_cropland:,.0f}, d_pasture={d_pasture:,.0f}, d_forest={d_forest:,.0f} ha")
                    
                    pool = get_pool(m49)
                    
                    # ========== De/Reforestation_crop 计算 ==========
                    if abs(d_cropland) > 0:
                        # 碳密度差（tC/ha）= 森林碳 - 耕地碳
                        delta_veg_c = (forest_c_ha - cropland_c_ha) * d_cropland      # tC
                        delta_soil_c = (forest_soil_c_ha - cropland_soil_c_ha) * d_cropland  # tC
                        
                        # 添加到碳库池（正值=毁林待释放，负值=造林待吸收）
                        pool['veg_crop'] += delta_veg_c
                        pool['soil_crop'] += delta_soil_c
                    
                    # ========== De/Reforestation_pasture 计算 ==========
                    if abs(d_pasture) > 0:
                        delta_veg_p = (forest_c_ha - pasture_c_ha) * d_pasture
                        delta_soil_p = (forest_soil_c_ha - pasture_soil_c_ha) * d_pasture
                        
                        pool['veg_pasture'] += delta_veg_p
                        pool['soil_pasture'] += delta_soil_p
                    
                    # ========== 指数响应：本年释放/吸收的碳 ==========
                    # 使用指数衰减模型：本年排放 = 碳库 × a（响应系数）
                    # 排放后碳库减少: 碳库 = 碳库 × (1 - a)
                    
                    # Crop转换的排放
                    emit_veg_crop = pool['veg_crop'] * a_veg      # tC
                    emit_soil_crop = pool['soil_crop'] * a_soil   # tC
                    total_emit_crop = (emit_veg_crop + emit_soil_crop) * TC2CO2 / 1000.0  # kt CO2
                    
                    # 更新碳库（剩余部分）
                    pool['veg_crop'] -= emit_veg_crop
                    pool['soil_crop'] -= emit_soil_crop
                    
                    # ===== 诊断日志：记录碳库和排放组分 =====
                    emis_debug_stats[year_val]['pool_veg_crop_total'] += pool['veg_crop']
                    emis_debug_stats[year_val]['pool_soil_crop_total'] += pool['soil_crop']
                    emis_debug_stats[year_val]['emit_veg_crop'] += emit_veg_crop * TC2CO2 / 1000.0
                    emis_debug_stats[year_val]['emit_soil_crop'] += emit_soil_crop * TC2CO2 / 1000.0
                    
                    if abs(total_emit_crop) > 0.001:  # 忽略极小值
                        emis_debug_stats[year_val]['emis_crop_sum'] += total_emit_crop
                        records.append({
                            'M49_Country_Code': m49,
                            'year': year_val,
                            'Process': 'De/Reforestation_crop',
                            'Item': 'De/Reforestation_crop area',
                            'GHG': 'CO2',
                            'value': total_emit_crop,  # 正值=毁林排放，负值=造林碳汇
                        })
                    
                    # Pasture转换的排放
                    emit_veg_pasture = pool['veg_pasture'] * a_veg
                    emit_soil_pasture = pool['soil_pasture'] * a_soil
                    total_emit_pasture = (emit_veg_pasture + emit_soil_pasture) * TC2CO2 / 1000.0
                    
                    pool['veg_pasture'] -= emit_veg_pasture
                    pool['soil_pasture'] -= emit_soil_pasture
                    
                    # ===== 诊断日志：记录碳库和排放组分 =====
                    emis_debug_stats[year_val]['pool_veg_pasture_total'] += pool['veg_pasture']
                    emis_debug_stats[year_val]['pool_soil_pasture_total'] += pool['soil_pasture']
                    emis_debug_stats[year_val]['emit_veg_pasture'] += emit_veg_pasture * TC2CO2 / 1000.0
                    emis_debug_stats[year_val]['emit_soil_pasture'] += emit_soil_pasture * TC2CO2 / 1000.0
                    
                    if abs(total_emit_pasture) > 0.001:
                        emis_debug_stats[year_val]['emis_pasture_sum'] += total_emit_pasture
                        records.append({
                            'M49_Country_Code': m49,
                            'year': year_val,
                            'Process': 'De/Reforestation_pasture',
                            'Item': 'De/Reforestation_pasture area',
                            'GHG': 'CO2',
                            'value': total_emit_pasture,
                        })
                    
                    # ========== Forest 过程（校验用，不重复计算碳） ==========
                    # 这里只记录森林面积变化，排放已在上面计算
                    # 不再重复计算，仅作为辅助信息
            
            # ===== 诊断日志：每年详细统计（写入model.log） =====
            _log_to_model("\n" + "="*100)
            _log_to_model("[De/Reforestation_crop 诊断报告] 年度排放与碳库状态")
            _log_to_model("="*100)
            
            for yr, stats in sorted(emis_debug_stats.items()):
                emis_pasture = stats.get('emis_pasture_sum', 0)
                d_pasture = stats.get('d_pasture_sum', 0)
                
                # 碳库状态
                pool_veg_crop = stats.get('pool_veg_crop_total', 0)
                pool_soil_crop = stats.get('pool_soil_crop_total', 0)
                pool_veg_pasture = stats.get('pool_veg_pasture_total', 0)
                pool_soil_pasture = stats.get('pool_soil_pasture_total', 0)
                
                # 排放组分
                emit_veg_crop = stats.get('emit_veg_crop', 0)
                emit_soil_crop = stats.get('emit_soil_crop', 0)
                emit_veg_pasture = stats.get('emit_veg_pasture', 0)
                emit_soil_pasture = stats.get('emit_soil_pasture', 0)
                
                _log_to_model(f"\n[{yr}年] 国家数={stats['count']}")
                _log_to_model(f"  面积变化(全球汇总):")
                _log_to_model(f"    耕地: {stats['d_cropland_sum']:>15,.0f} ha")
                _log_to_model(f"    草地: {d_pasture:>15,.0f} ha")
                _log_to_model(f"    森林: {stats['d_forest_sum']:>15,.0f} ha")
                _log_to_model(f"  碳库状态(待释放/吸收, kt CO2):")
                _log_to_model(f"    Crop-植被池: {pool_veg_crop:>12,.0f}  |  Crop-土壤池: {pool_soil_crop:>12,.0f}")
                _log_to_model(f"    Pasture-植被池: {pool_veg_pasture:>12,.0f}  |  Pasture-土壤池: {pool_soil_pasture:>12,.0f}")
                _log_to_model(f"  本年排放组分(kt CO2):")
                _log_to_model(f"    Crop-植被排放: {emit_veg_crop:>12,.0f}  |  Crop-土壤排放: {emit_soil_crop:>12,.0f}")
                _log_to_model(f"    Pasture-植被排放: {emit_veg_pasture:>12,.0f}  |  Pasture-土壤排放: {emit_soil_pasture:>12,.0f}")
                _log_to_model(f"  总排放(kt CO2):")
                _log_to_model(f"    De/Reforestation_crop: {stats['emis_crop_sum']:>15,.0f}")
                _log_to_model(f"    De/Reforestation_pasture: {emis_pasture:>15,.0f}")
                _log_to_model(f"    合计: {stats['emis_crop_sum'] + emis_pasture:>15,.0f}")
            
            # ===== 诊断日志：碳库剩余状态与关键指标分析 =====
            if use_exponential_response:
                total_remaining_veg = sum(abs(p['veg_crop']) + abs(p['veg_pasture']) for p in carbon_pools.values())
                total_remaining_soil = sum(abs(p['soil_crop']) + abs(p['soil_pasture']) for p in carbon_pools.values())
                total_remaining_kt_co2 = (total_remaining_veg + total_remaining_soil) * TC2CO2 / 1000.0
                
                _log_to_model("\n" + "="*100)
                _log_to_model("[碳库剩余状态] 指数响应模型累积效应")
                _log_to_model("="*100)
                _log_to_model(f"  全球碳库剩余(待释放/吸收):")
                _log_to_model(f"    植被池: {total_remaining_veg:>15,.0f} tC ({total_remaining_veg * TC2CO2 / 1000:>12,.0f} kt CO2)")
                _log_to_model(f"    土壤池: {total_remaining_soil:>15,.0f} tC ({total_remaining_soil * TC2CO2 / 1000:>12,.0f} kt CO2)")
                _log_to_model(f"    合计:   {total_remaining_veg + total_remaining_soil:>15,.0f} tC ({total_remaining_kt_co2:>12,.0f} kt CO2)")
                _log_to_model(f"\n  说明: 碳库剩余表示尚未释放完的历史毁林碳，未来年份将继续释放")
            
            # ===== 诊断日志：排放倍增分析 =====
            if len(emis_debug_stats) >= 2:
                years_sorted = sorted(emis_debug_stats.keys())
                first_year = years_sorted[0]
                last_year = years_sorted[-1]
                
                first_emis = emis_debug_stats[first_year]['emis_crop_sum']
                last_emis = emis_debug_stats[last_year]['emis_crop_sum']
                
                if first_emis > 0:
                    ratio = last_emis / first_emis
                    _log_to_model("\n" + "="*100)
                    _log_to_model("[排放倍增分析] De/Reforestation_crop")
                    _log_to_model("="*100)
                    _log_to_model(f"  基准年份({first_year}): {first_emis:>15,.0f} kt CO2")
                    _log_to_model(f"  目标年份({last_year}): {last_emis:>15,.0f} kt CO2")
                    _log_to_model(f"  倍增比率: {ratio:>15.2f}x")
                    _log_to_model(f"\n  解释:")
                    if ratio > 5:
                        _log_to_model(f"    ⚠️ 排放显著增长! 可能原因:")
                        _log_to_model(f"       1. 毁林速率加速（检查d_cropland面积变化趋势）")
                        _log_to_model(f"       2. 碳库持续累积效应（指数响应模型特性）")
                        _log_to_model(f"       3. 基准年数据异常（检查{first_year}年输入数据）")
                    elif ratio > 2:
                        _log_to_model(f"    ✓ 排放增长在合理范围内（碳库累积效应）")
                    else:
                        _log_to_model(f"    ✓ 排放基本稳定或减少")
            
            _log_to_model("\n" + "="*100)
    
    # ===== 处理木材采伐驱动 - 完整HWP池模型 =====
    # HWP池模型：采伐碳按frac_HWP进入三个产品池（short/medium/long），其余即时排放
    # 每年池中碳按指数衰减排放: emit = pool × (1 - exp(-k))
    if roundwood_change_df is not None and not roundwood_change_df.empty:
        df_rw = roundwood_change_df.copy()
        df_rw.columns = [str(c).strip() for c in df_rw.columns]
        
        # 确保M49_Country_Code列存在
        if 'M49_Country_Code' not in df_rw.columns:
            # 尝试从其他国家列识别
            if 'iso3' in df_rw.columns:
                # 假设iso3列存在，映射到M49
                df_rw['M49_Country_Code'] = df_rw['iso3'].apply(
                    lambda x: iso3_to_m49.get(str(x).strip(), str(x).strip())
                )
            elif 'country' in df_rw.columns:
                df_rw['M49_Country_Code'] = df_rw['country'].apply(
                    lambda x: iso3_to_m49.get(str(x).strip(), str(x).strip())
                )
            else:
                print("[WARN] roundwood_change_df中无法识别国家标识")
                df_rw['M49_Country_Code'] = 'UNK'
        
        # 确保M49_Country_Code为标准化字符串
        df_rw['M49_Country_Code'] = df_rw['M49_Country_Code'].apply(normalize_m49)
        
        df_rw = df_rw[df_rw['year'].astype(int) > 2020]
        
        if not df_rw.empty:
            # Wood harvest参数
            rho = params.get('rho_wood', 0.5)       # tDM/m³
            cf = params.get('cf_wood', 0.5)         # tC/tDM
            pi_agb = params.get('pi_agb', 0.7)      # AGB份额
            
            use_hist_ef = historical_wood_harvest_ef is not None and len(historical_wood_harvest_ef) > 0
            if use_hist_ef:
                print(f"[INFO] 使用历史排放因子计算未来Wood harvest (共{len(historical_wood_harvest_ef)}个国家)")
            else:
                print(f"[INFO] 使用理论公式计算Wood harvest (rho={rho}, cf={cf}, pi_agb={pi_agb})")
            
            print(f"[LUC] HWP池模型: frac_HWP={frac_HWP:.1%}进入产品池, {1-frac_HWP:.1%}即时排放")
            
            # 按年份排序处理（HWP池需要状态累积）
            years_in_rw = sorted(df_rw['year'].unique())
            hwp_emit_total = 0.0
            inst_emit_total = 0.0
            
            for year_val in years_in_rw:
                year_data = df_rw[df_rw['year'] == year_val]
                year_hwp_emit = 0.0
                year_inst_emit = 0.0
                
                for _, row in year_data.iterrows():
                    m49 = str(row['M49_Country_Code']).strip()
                    roundwood_m3 = float(row.get('roundwood_m3', 0.0))
                    
                    if roundwood_m3 > 0:
                        # 计算排放 - 两种模式
                        if use_hist_ef and historical_wood_harvest_ef and m49 in historical_wood_harvest_ef:
                            # ===== 历史EF模式 =====
                            # 历史EF = 历史排放(kt CO2) / 历史产量(m³)
                            # 这个EF已经隐含了实际的排放模式，不需要再进行HWP分割
                            # 直接用: 未来产量 × EF = 未来排放
                            ef_kt_per_m3 = historical_wood_harvest_ef[m49]
                            instant_emit_kt = roundwood_m3 * ef_kt_per_m3  # 直接得到kt CO2
                            year_inst_emit += instant_emit_kt
                            # 注意：历史EF模式下不计算HWP池（因为EF已包含完整排放模式）
                        else:
                            # ===== 理论公式模式 =====
                            # 需要进行HWP池分割
                            # 理论公式: m³ × tDM/m³ × tC/tDM × AGB比例 = tC
                            harvested_tc = roundwood_m3 * rho * cf * pi_agb
                            
                            # 一部分进入HWP池，一部分即时排放
                            to_hwp_tc = harvested_tc * frac_HWP        # 进入产品池的碳
                            instant_tc = harvested_tc * (1 - frac_HWP) # 即时排放的碳
                            
                            # 将to_hwp_tc分配到三个池
                            hwp_pool = get_hwp_pool(m49)
                            for pool_name, alloc_ratio in alloc_HWP.items():
                                hwp_pool[pool_name] += to_hwp_tc * alloc_ratio
                            
                            # 即时排放 (tC → kt CO2)
                            inst_emit_kt = instant_tc * TC2CO2 / 1000.0
                            year_inst_emit += inst_emit_kt
                
                # ===== HWP池衰减排放 =====
                # 所有国家的HWP池衰减（每年处理一次）
                # ✅ 关键修复：HWP池排放合并到Wood harvest Process，不单独列为HWP decay
                hwp_by_country = {}  # 存储每个国家的HWP池排放
                for m49, hwp_pool in hwp_pools.items():
                    hwp_emit_m49 = 0.0
                    for pool_name in ['short', 'medium', 'long']:
                        # 本年排放 = 池量 × (1 - exp(-k))
                        k = k_HWP[pool_name]
                        pool_val = hwp_pool[pool_name]
                        emit_tc = pool_val * (1 - np.exp(-k))
                        hwp_pool[pool_name] -= emit_tc  # 更新池量
                        hwp_emit_m49 += emit_tc
                    
                    if hwp_emit_m49 > 0.001:  # tC
                        hwp_emit_kt = hwp_emit_m49 * TC2CO2 / 1000.0  # kt CO2
                        year_hwp_emit += hwp_emit_kt
                        hwp_by_country[m49] = hwp_emit_kt  # 暂存，稍后合并到Wood harvest
                
                # ✅ 关键修复：记录Wood harvest（即时排放 + HWP池排放）
                # 先收集所有国家的即时排放
                instant_by_country = {}
                for _, row in year_data.iterrows():
                    m49 = str(row['M49_Country_Code']).strip()
                    roundwood_m3 = float(row.get('roundwood_m3', 0.0))
                    
                    if roundwood_m3 > 0:
                        # 计算该国家的排放 - 与上面的逻辑保持一致
                        if use_hist_ef and historical_wood_harvest_ef and m49 in historical_wood_harvest_ef:
                            # 历史EF模式：直接用EF计算排放
                            ef_kt_per_m3 = historical_wood_harvest_ef[m49]
                            instant_kt = roundwood_m3 * ef_kt_per_m3  # 直接得到kt CO2
                        else:
                            # 理论公式模式：需要HWP分割
                            harvested_tc = roundwood_m3 * rho * cf * pi_agb
                            instant_kt = harvested_tc * (1 - frac_HWP) * TC2CO2 / 1000.0
                        
                        if instant_kt > 0.001:
                            instant_by_country[m49] = instant_kt
                
                # 合并即时排放和HWP池排放，统一记录为Wood harvest
                all_countries = set(instant_by_country.keys()) | set(hwp_by_country.keys())
                for m49 in all_countries:
                    instant_kt = instant_by_country.get(m49, 0.0)
                    hwp_kt = hwp_by_country.get(m49, 0.0)
                    total_kt = instant_kt + hwp_kt
                    
                    if total_kt > 0.001:
                        records.append({
                            'M49_Country_Code': m49,
                            'year': year_val,
                            'Process': 'Wood harvest',  # 即时排放 + HWP池排放
                            'Item': 'Roundwood',
                            'GHG': 'CO2',
                            'value': total_kt,
                        })
                
                hwp_emit_total += year_hwp_emit
                inst_emit_total += year_inst_emit
                print(f"[LUC] {year_val}年 Wood harvest: 即时排放={year_inst_emit:,.0f} kt, HWP池排放={year_hwp_emit:,.0f} kt")
            
            # 打印HWP池剩余量
            total_hwp_remaining = sum(sum(p.values()) for p in hwp_pools.values())
            print(f"[LUC] HWP池最终剩余: {total_hwp_remaining:,.0f} tC (将在后续年份继续释放)")
    
    # ===== 处理森林碳汇（Forest process）- 按国家森林类型匹配碳汇速率 =====
    # Forest 碳汇表示现有森林每年吸收的碳（负排放）
    # 计算公式: 森林面积(ha) × 碳汇速率(tC/ha/年) × 44/12 / 1000 = kt CO2/年
    
    if forest_area_df is not None and not forest_area_df.empty:
        df_forest = forest_area_df.copy()
        df_forest.columns = [str(c).strip() for c in df_forest.columns]
        
        # 确保M49_Country_Code列存在
        if 'M49_Country_Code' not in df_forest.columns:
            if 'iso3' in df_forest.columns:
                df_forest['M49_Country_Code'] = df_forest['iso3'].apply(
                    lambda x: iso3_to_m49.get(str(x).strip(), str(x).strip())
                )
            elif 'country' in df_forest.columns:
                df_forest['M49_Country_Code'] = df_forest['country'].apply(
                    lambda x: iso3_to_m49.get(str(x).strip(), str(x).strip())
                )
            else:
                print("[WARN] forest_area_df中无法识别国家标识")
                df_forest['M49_Country_Code'] = 'UNK'
        
        df_forest['M49_Country_Code'] = df_forest['M49_Country_Code'].apply(normalize_m49)
        
        # 过滤未来年份
        if 'year' in df_forest.columns:
            df_forest = df_forest[df_forest['year'].astype(int) > 2020]
        
        if not df_forest.empty:
            # 识别森林面积列
            forest_col = None
            for col_name in ['forest_ha', 'forest_area_ha', 'forestland_ha', 'forest']:
                if col_name in df_forest.columns:
                    forest_col = col_name
                    break
            
            if forest_col:
                use_hist_sink_ef = historical_forest_sink_ef is not None and len(historical_forest_sink_ef) > 0
                if use_hist_sink_ef:
                    print(f"[INFO] 使用历史排放因子计算未来Forest碳汇 (共{len(historical_forest_sink_ef)}个国家)")
                    # ✅ DEBUG: 打印历史EF样本
                    sample_efs = list(historical_forest_sink_ef.items())[:5]
                    print(f"[DEBUG] 历史Forest EF样本: {sample_efs}")
                else:
                    print(f"[INFO] 按国家森林类型计算Forest碳汇:")
                    print(f"[INFO]   Tropical={forest_sink_rates['Tropical']}, Temperate={forest_sink_rates['Temperate']}, "
                          f"Boreal={forest_sink_rates['Boreal']}, Default={default_sink_rate} tC/ha/yr")
                
                # 统计各类型使用情况
                forest_type_usage = {'Tropical': 0, 'Temperate': 0, 'Boreal': 0, 'Default': 0, 'HistEF': 0}
                total_forest_area = 0
                total_forest_area_hist_ef = 0  # ✅ 使用历史EF的面积
                total_forest_area_default = 0  # ✅ 使用默认速率的面积
                unmatched_m49s = set()  # ✅ 调试：记录未匹配的M49
                total_sink_hist_ef = 0  # ✅ 历史EF计算的碳汇
                total_sink_default = 0  # ✅ 默认速率计算的碳汇
                
                for _, row in df_forest.iterrows():
                    year_val = int(row['year'])
                    m49 = str(row['M49_Country_Code']).strip()
                    forest_area = float(row.get(forest_col, 0.0))
                    
                    if forest_area > 0:
                        total_forest_area += forest_area
                        
                        # 优先使用历史排放因子
                        if use_hist_sink_ef and historical_forest_sink_ef and m49 in historical_forest_sink_ef:
                            # 历史EF单位: kt CO2/ha/年
                            ef_kt_per_ha = historical_forest_sink_ef[m49]
                            forest_sink_kt = forest_area * ef_kt_per_ha
                            forest_type_usage['HistEF'] += 1
                            total_forest_area_hist_ef += forest_area
                            total_sink_hist_ef += forest_sink_kt
                        else:
                            # ✅ 调试：记录未匹配的M49
                            if use_hist_sink_ef and historical_forest_sink_ef:
                                unmatched_m49s.add(m49)
                            # ✅ 按国家森林类型选择碳汇速率
                            forest_type = m49_to_forest_type.get(m49, '')
                            if forest_type in forest_sink_rates:
                                sink_rate = forest_sink_rates[forest_type]
                                forest_type_usage[forest_type] += 1
                            else:
                                sink_rate = default_sink_rate
                                forest_type_usage['Default'] += 1
                            
                            # 理论公式: ha × tC/ha/年 × 44/12 / 1000 = kt CO2/年
                            forest_sink_kt = forest_area * sink_rate * TC2CO2 / 1000.0
                            total_forest_area_default += forest_area
                            total_sink_default += forest_sink_kt
                        
                        # 森林碳汇是负值（吸收CO2）
                        records.append({
                            'M49_Country_Code': m49,
                            'year': year_val,
                            'Process': 'Forest',
                            'Item': 'Forestland',
                            'GHG': 'CO2',
                            'value': forest_sink_kt,  # 负值表示碳汇
                        })
                
                # 打印森林类型使用统计
                print(f"[LUC] 森林类型使用统计: {forest_type_usage}")
                print(f"[LUC] 总森林面积(所有年份累计): {total_forest_area:,.0f} ha")
                print(f"[LUC]   其中: 使用历史EF={total_forest_area_hist_ef:,.0f} ha, 使用默认速率={total_forest_area_default:,.0f} ha")
                print(f"[LUC] 碳汇分解: 历史EF={total_sink_hist_ef:,.0f} kt, 默认速率={total_sink_default:,.0f} kt")
                
                # ✅ DEBUG: 打印M49匹配情况
                if use_hist_sink_ef and historical_forest_sink_ef and unmatched_m49s:
                    print(f"[LUC DEBUG] 未匹配历史EF的M49数量: {len(unmatched_m49s)}")
                    hist_ef_m49s = set(historical_forest_sink_ef.keys())
                    print(f"[LUC DEBUG] 历史EF M49样本: {list(hist_ef_m49s)[:5]}")
                    print(f"[LUC DEBUG] 未匹配M49样本: {list(unmatched_m49s)[:5]}")
                
                # DEBUG: 打印森林碳汇统计
                forest_records = [r for r in records if r['Process'] == 'Forest']
                if forest_records:
                    total_sink = sum(r['value'] for r in forest_records)
                    print(f"[LUC] Forest碳汇计算完成: {len(forest_records)} 条记录, 总计 {total_sink:,.0f} kt CO2")
            else:
                print(f"[WARN] forest_area_df中无森林面积列，可用列: {list(df_forest.columns)}")
    
    # 转换为标准格式
    if records:
        df_future = pd.DataFrame(records)
    else:
        df_future = pd.DataFrame(columns=[
            'M49_Country_Code', 'year', 'Process', 'Item', 'GHG', 'value'
        ])
    
    # 聚合（可能多个转换贡献到同一个过程）
    if not df_future.empty:
        df_future = df_future.groupby(['M49_Country_Code', 'year', 'Process', 'Item', 'GHG'], 
                                      as_index=False)['value'].sum()
    
    # 添加Region_label_new列（从M49映射）
    if dict_v3_path:
        try:
            region_df = pd.read_excel(dict_v3_path, sheet_name='region',
                                     usecols=['M49_Country_Code', 'Region_label_new'])
            region_df['M49_Country_Code'] = region_df['M49_Country_Code'].apply(normalize_m49)
            region_map = dict(zip(region_df['M49_Country_Code'], region_df['Region_label_new']))
            df_future['Region_label_new'] = df_future['M49_Country_Code'].map(region_map).fillna('Unknown')
        except Exception as e:
            print(f"[WARN] 无法加载Region_label_new映射: {e}")
            df_future['Region_label_new'] = 'Unknown'
    else:
        df_future['Region_label_new'] = 'Unknown'
    
    # 保证M49_Country_Code为字符串格式（保持原格式）
    df_future['M49_Country_Code'] = df_future['M49_Country_Code'].astype(str)
    
    return {'future': df_future}
