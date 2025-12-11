# -*- coding: utf-8 -*-
"""
按 mask code 计算各国家/地区的平均 AGB（kg/ha），输出 Excel。

"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
import xarray as xr

# ======= 硬编码路径（如需换本地路径，直接改这里） =======
MASK_NC  = r"R:\\Data\\Food\\LUH2\\mask_pastureYield_0083d.nc"
AGB_TIF  = "../../input/Land/Feed_pasture/pastures_coi_AGB_kg_ha.tif"
AREA_TIF = "../../input/Land/Feed_pasture/pastures_coi_Area_ha.tif"
OUT_XLSX = "../../input/Land/Feed_pasture/Pasture_DM_yield_by_country.xlsx"
# ======================================================

def _load_agb_grid(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        nod = src.nodata
        if nod is not None and not np.isnan(nod):
            arr[arr == nod] = np.nan
        tr = src.transform
        crs = src.crs
        rows, cols = src.height, src.width
        res_x, res_y = src.res
    return arr, tr, crs, rows, cols, res_x, res_y

def _load_area_to_agb_grid(path, dst_transform, dst_crs, shape_like):
    with rasterio.open(path) as src:
        src_arr = src.read(1).astype(np.float64)
        src_tr  = src.transform
        src_crs = src.crs
        src_nod = src.nodata
    dst = np.full(shape_like, np.nan, dtype=np.float64)
    reproject(
        source=src_arr,
        destination=dst,
        src_transform=src_tr,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
        src_nodata=src_nod,
        dst_nodata=np.nan
    )
    return dst

def _find_lat_lon_names(ds):
    lat_name = None; lon_name = None
    for cand in ["lat", "latitude", "Lat", "Latitude", "y"]:
        if cand in ds.coords:
            lat_name = cand; break
    for cand in ["lon", "longitude", "Lon", "Longitude", "x"]:
        if cand in ds.coords:
            lon_name = cand; break
    if lat_name is None or lon_name is None:
        for d in ds.dims:
            if d.lower() in ["lat","latitude","y"]:
                lat_name = d
            if d.lower() in ["lon","longitude","x"]:
                lon_name = d
    if not lat_name or not lon_name:
        raise RuntimeError("mask NC 中找不到 lat/lon 坐标。")
    return lat_name, lon_name

def _pick_mask_var(ds, lat_name, lon_name):
    # 优先选 2D 整型变量，维度等于 {lat,lon}
    for v in ds.data_vars:
        da = ds[v]
        if da.ndim == 2 and set(da.dims) == set([lat_name, lon_name]):
            if np.issubdtype(da.dtype, np.integer):
                return v
    # 退化：选第一个满足 2D 且维度匹配的变量
    for v in ds.data_vars:
        da = ds[v]
        if da.ndim == 2 and set(da.dims) == set([lat_name, lon_name]):
            return v
    raise RuntimeError("mask NC 中没有可用的二维掩码变量。")

def _interp_mask_to_agb_grid(mask_nc, rows, cols, res_x, res_y):
    ds = xr.open_dataset(mask_nc)
    lat_name, lon_name = _find_lat_lon_names(ds)
    var = _pick_mask_var(ds, lat_name, lon_name)
    da = ds[var]

    # AGB 网格像元中心（行0在北半球）
    target_lats = 90 - (np.arange(rows) + 0.5) * abs(res_y)
    target_lons = -180 + (np.arange(cols) + 0.5) * abs(res_x)

    # 纬度从北到南排序
    if da[lat_name][0] < da[lat_name][-1]:
        da = da.sortby(lat_name, ascending=False)

    # 经度包裹到 [-180, 180) 并排序
    lon_vals = da[lon_name].values
    if (lon_vals.min() >= 0) and (lon_vals.max() <= 360):
        lon_wrapped = (((da[lon_name] + 180) % 360) - 180)
        da = da.assign_coords({lon_name: lon_wrapped}).sortby(lon_name)
    else:
        da = da.sortby(lon_name)

    # 最近邻插值：避免边界混码
    mask_interp = da.interp({lat_name: target_lats, lon_name: target_lons}, method="nearest")
    mask_np = mask_interp.values
    mask_codes = np.rint(np.nan_to_num(mask_np, nan=0.0)).astype(np.int64)
    return mask_codes

def _compute_by_code(mask_codes, agb, area):
    valid = (mask_codes > 0) & np.isfinite(agb) & np.isfinite(area) & (area > 0)
    codes   = mask_codes[valid].ravel()
    vals    = agb[valid].ravel()
    weights = area[valid].ravel()

    df = pd.DataFrame({"code": codes, "val": vals, "wt": weights})
    sum_w  = df.groupby("code")["wt"].sum()
    sum_vw = (df["val"] * df["wt"]).groupby(df["code"]).sum()
    wmean  = sum_vw / sum_w

    smean  = df.groupby("code")["val"].mean()
    npx    = df.groupby("code").size()
    vmin   = df.groupby("code")["val"].min()
    vmax   = df.groupby("code")["val"].max()

    out = pd.DataFrame({
        "code": wmean.index,
        "mean_AGB_kg_ha_weighted_by_area": wmean.values,
        "mean_AGB_kg_ha_simple": smean.reindex(wmean.index).values,
        "n_pixels": npx.reindex(wmean.index).values,
        "sum_area_ha": sum_w.reindex(wmean.index).values,
        "min_AGB_kg_ha": vmin.reindex(wmean.index).values,
        "max_AGB_kg_ha": vmax.reindex(wmean.index).values,
    }).sort_values("code").reset_index(drop=True)
    return out

def main():
    agb, tr, crs, rows, cols, res_x, res_y = _load_agb_grid(AGB_TIF)
    area = _load_area_to_agb_grid(AREA_TIF, tr, crs, agb.shape)
    mask_codes = _interp_mask_to_agb_grid(MASK_NC, rows, cols, res_x, res_y)

    out = _compute_by_code(mask_codes, agb, area)
    out.to_excel(OUT_XLSX, index=False)
    print(f"完成，已写出：{OUT_XLSX}")
    print(out.head(10))

if __name__ == "__main__":
    main()
