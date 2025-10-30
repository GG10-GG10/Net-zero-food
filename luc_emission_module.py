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

import numpy as np
import xarray as xr
import pandas as pd
from config_paths import get_src_base

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
