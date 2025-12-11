# -*- coding: utf-8 -*-
"""
与 LUH2 网格完全一致的国家掩膜（使用 Shapefile 中的 ID1 字段）
输出 NetCDF 维度为 (lat, lon)，变量为 id1（int32, 0 表示海洋/无国）
"""

import os
import tempfile
import numpy as np
import xarray as xr
import netCDF4 as nc
import geopandas as gpd
from osgeo import gdal, osr
from shapely.ops import transform as shp_transform
from shapely.geometry import base as shp_base

# ------------ 路径（按需修改） ------------
luh2_nc_path = r"R:\Data\Food\LUH2\LUH2_GCB2019_states.nc4"
shp_path     = r"..\..\src\World_map\polygon\World_polygon.shp"
out_nc       = r"..\..\src\mask_LUH2_025d.nc"
# -----------------------------------------

os.environ['PROJ_LIB']  = r'D:\Anaconda3-202402\Library\share\proj'
os.environ['GDAL_DATA'] = r'D:\Anaconda3-202402\Library\share'
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
gdal.UseExceptions()

# ---------- 工具函数 ----------
def _pick_1D(ds, names, kind='lat'):
    for n in names:
        if n in ds and ds[n].ndim == 1 and ds[n].size > 1:
            arr = np.asarray(ds[n].values, dtype=float)
            if kind == 'lat' and np.nanmin(arr) >= -90-1e-6 and np.nanmax(arr) <= 90+1e-6:
                return arr
            if kind == 'lon' and np.nanmin(arr) >= -360-1e-6 and np.nanmax(arr) <= 360+1e-6:
                return arr
    return None

def _from_2D(ds, lat2d_name, lon2d_name):
    lat2d = np.asarray(ds[lat2d_name].values, float)
    lon2d = np.asarray(ds[lon2d_name].values, float)
    if np.allclose(lat2d, lat2d[:, [0]], equal_nan=True):
        lat = lat2d[:, 0]
    elif np.allclose(lat2d, lat2d[[0], :], equal_nan=True):
        lat = lat2d[0, :]
    else:
        lat = np.nanmean(lat2d, axis=1)
    if np.allclose(lon2d, lon2d[[0], :], equal_nan=True):
        lon = lon2d[0, :]
    elif np.allclose(lon2d, lon2d[:, [0]], equal_nan=True):
        lon = lon2d[:, 0]
    else:
        lon = np.nanmean(lon2d, axis=0)
    return np.asarray(lat, float), np.asarray(lon, float)

def get_lat_lon(ds):
    lat = _pick_1D(ds, ['lat','latitude','y','LAT','Latitude'], 'lat')
    lon = _pick_1D(ds, ['lon','longitude','x','LON','Longitude'], 'lon')
    if lat is not None and lon is not None:
        return lat, lon
    for la in ['latitude','lat','LAT','Latitude']:
        for lo in ['longitude','lon','LON','Longitude']:
            if la in ds and lo in ds and ds[la].ndim == 2 and ds[lo].ndim == 2:
                return _from_2D(ds, la, lo)
    raise RuntimeError("未能解析 LUH2 的经纬度坐标（支持 1D/2D）。")

def geotransform_from_centers(lat, lon):
    dlon = float(np.min(np.abs(np.diff(lon))))
    dlat = float(np.min(np.abs(np.diff(lat))))
    x_origin = float(lon.min() - dlon/2.0)
    y_origin = float(lat.max() + dlat/2.0)
    return (x_origin, dlon, 0.0, y_origin, 0.0, -dlat), dlat, dlon

def _make_valid(g):
    try:
        # Shapely 2.0+
        from shapely.validation import make_valid
        return make_valid(g)
    except Exception:
        # 旧版：用 buffer(0) 修复常见自相交
        try:
            return g.buffer(0) if (g and not g.is_empty) else g
        except Exception:
            return g
# -----------------------------

def main():
    # 1) LUH2 网格
    ds = xr.open_dataset(luh2_nc_path, engine="netcdf4")
    lat, lon = get_lat_lon(ds)
    nlat, nlon = len(lat), len(lon)
    gt, dlat, dlon = geotransform_from_centers(lat, lon)
    lat_ascending = bool(lat[0] < lat[-1])

    # 2) 读取 shp，确保有 ID1，并投影为 WGS84（用 pyproj+shapely，避免 to_crs/VectorTranslate 问题）
    gdf = gpd.read_file(shp_path)
    if 'ID1' not in gdf.columns:
        raise KeyError("Shapefile 中未找到字段 'ID1'。")
    gdf['ID1'] = gdf['ID1'].fillna(0).astype(np.int32)

    from pyproj import CRS, Transformer
    from pyproj.exceptions import ProjError

    target_crs = CRS.from_epsg(4326)

    if gdf.crs is None:
        # 无 CRS：若边界看似经纬度，则直接设为 4326；否则报错
        minx, miny, maxx, maxy = gdf.total_bounds
        looks_lonlat = (-180-1e-6 <= minx <= 180+1e-6 and
                        -180-1e-6 <= maxx <= 180+1e-6 and
                         -90-1e-6 <= miny <=  90+1e-6 and
                         -90-1e-6 <= maxy <=  90+1e-6)
        if not looks_lonlat:
            raise ValueError("Shapefile 无 CRS 且坐标范围不像经纬度，无法假设为 WGS84。")
        gdf.set_crs(target_crs, inplace=True)
    else:
        src = CRS.from_user_input(gdf.crs)
        needs_transform = False
        tr = None
        if not src.equals(target_crs):
            try:
                tr = Transformer.from_crs(src, target_crs, always_xy=True)
                needs_transform = True
            except ProjError as exc:
                name_lower = (src.name or "").lower()
                proj4_lower = ""
                try:
                    proj4_lower = src.to_proj4().lower()
                except Exception:
                    proj4_lower = ""
                looks_wgs84 = (
                    src.is_geographic and (
                        "wgs84" in name_lower or "wgs 84" in name_lower or
                        "+datum=wgs84" in proj4_lower or "+ellps=wgs84" in proj4_lower or
                        "+a=6378137" in proj4_lower
                    )
                )
                if not looks_wgs84:
                    raise RuntimeError("无法构建到 WGS84 的坐标转换，请检查 Shapefile 的坐标系定义。") from exc
        if needs_transform and tr is not None:
            def _trf(geom: shp_base.BaseGeometry):
                if geom is None or geom.is_empty:
                    return geom
                geom2 = _make_valid(geom)
                return shp_transform(tr.transform, geom2) if geom2 and not geom2.is_empty else geom2
            gdf = gdf.set_geometry(gdf.geometry.apply(_trf), crs=target_crs)
        else:
            gdf = gdf.set_crs(target_crs, allow_override=True)

    # 清理空/无效几何
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()].copy()

    # 3) 写到本地临时 GPKG（ASCII 路径），让 GDAL 打开矢量层以栅格化
    tmp_dir = tempfile.mkdtemp(prefix="mask_luh2_")
    tmp_gpkg = os.path.join(tmp_dir, "src4326.gpkg")
    gdf.to_file(tmp_gpkg, driver='GPKG')  # 会带上 EPSG:4326

    vds = gdal.OpenEx(tmp_gpkg, gdal.OF_VECTOR)
    if vds is None:
        raise RuntimeError(f"无法打开临时 GPKG：{tmp_gpkg}")
    vlyr = vds.GetLayer(0)

    # 4) 准备目标栅格（EPSG:4326，与 LUH2 对齐）
    srs = osr.SpatialReference(); srs.ImportFromEPSG(4326)
    mem = gdal.GetDriverByName('MEM').Create('', nlon, nlat, 1, gdal.GDT_Int32)
    mem.SetProjection(srs.ExportToWkt())
    mem.SetGeoTransform(gt)
    band = mem.GetRasterBand(1)
    band.Fill(0); band.SetNoDataValue(0)

    # 5) 栅格化（ATTRIBUTE=ID1）
    gdal.RasterizeLayer(mem, [1], vlyr, options=["ATTRIBUTE=ID1", "ALL_TOUCHED=TRUE"])
    arr_n2s = mem.ReadAsArray()
    id1_grid = np.flipud(arr_n2s) if lat_ascending else arr_n2s

    # 6) 写 NetCDF（CF 友好）
    if os.path.exists(out_nc):
        os.remove(out_nc)
    root = nc.Dataset(out_nc, 'w', format='NETCDF4')
    try:
        root.createDimension('lat', nlat)
        root.createDimension('lon', nlon)
        vlat = root.createVariable('lat','f4',('lat',)); vlat[:] = lat
        vlon = root.createVariable('lon','f4',('lon',)); vlon[:] = lon
        vlat.standard_name='latitude';  vlat.units='degrees_north'
        vlon.standard_name='longitude'; vlon.units='degrees_east'

        vid1 = root.createVariable('id1','i4',('lat','lon'), zlib=True, complevel=4, fill_value=0)
        vid1[:, :] = id1_grid
        vid1.long_name = "Country/region mask from Shapefile attribute ID1 (0=ocean/no-country)"
        vid1.grid_mapping = 'crs'

        crs = root.createVariable('crs','i4')
        crs.grid_mapping_name = "latitude_longitude"
        crs.epsg_code = "EPSG:4326"
        crs.semi_major_axis = 6378137.0
        crs.inverse_flattening = 298.257223563

        root.title = "Mask aligned to LUH2 grid using ID1 attribute"
        root.source_luh2 = os.path.basename(luh2_nc_path)
        root.history = "created by script; ATTRIBUTE=ID1; ALL_TOUCHED=TRUE"
        root.author = "Jing Cheng"
    finally:
        root.close()
        # 清理临时文件
        try:
            import shutil; shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    print(f"Done: {out_nc}")
    print(f"Grid: lat({nlat}) x lon({nlon}), dlat={dlat}, dlon={dlon}")
    print(f"lat[0]={lat[0]:.6f}, lat[-1]={lat[-1]:.6f}; lon[0]={lon[0]:.6f}, lon[-1]={lon[-1]:.6f}")

if __name__ == "__main__":
    main()
