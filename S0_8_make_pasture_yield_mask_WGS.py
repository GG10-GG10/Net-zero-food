# -*- coding: utf-8 -*-
"""
按“原脚本的掩码生成逻辑（矢量→栅格化）”输出与参考GeoTIFF完全一致的坐标系/分辨率/范围的mask。
- 参考栅格：ref_tif_path（如 pastures_coi_Area_ha.tif，1/12°）
- 矢量边界：shp_path（含 ID1 字段；若没有则回退为0/1掩码）
- 输出：与参考栅格一致的GTiff（Int32，0为NoData/海洋/无国）

依赖：GDAL（osgeo.gdal/ogr/osr）。建议：>=3.4。
"""

import os
import tempfile
from osgeo import gdal, ogr, osr

# ---------------- 用户需检查的路径 ----------------
# 参考栅格（要求：目标像元大小=1/12°, CRS/范围/分辨率以它为准）
ref_tif_path = r"..\\..\\input\\Land\\Feed_pasture\\pastures_coi_Area_ha.tif"  
# 世界边界（含ID1字段的矢量，Shapefile/GPKG等皆可）
# 与原脚本一致：使用ID1作为掩码值；若没有ID1字段则退化为0/1掩码（陆地=1，海洋=0）
shp_path = r"..\\..\\src\\World_map\\polygon\\World_polygon.shp"  # 改为你的实际路径

# 输出路径（支持 GTiff/NetCDF，根据扩展名自动选择）
out_mask_path = r"..\\..\\src\\mask_pastureYield_0083d.nc"
# -------------------------------------------------

ATTRIBUTE_FIELD = "ID1"       # 与原脚本一致，优先使用ID1作为掩码值
ALL_TOUCHED = True            # 栅格化时是否“触碰即算”（边界更饱满）
NODATA_VALUE = 0              # 海洋/无国等填0（与原脚本一致的语义）
SHAPE_ENCODING = "CP936"      # 若矢量属性表为国标编码，可写 "CP936"/"GBK"；置为 None 则保持默认
FORCE_UTF8 = False            # 设为 False 可关闭“非 UTF-8”警告

def _open_vector(vpath):
    vds = gdal.OpenEx(vpath, gdal.OF_VECTOR)
    if vds is None:
        raise RuntimeError(f"无法打开矢量文件：{vpath}")
    lyr = vds.GetLayer(0)
    if lyr is None:
        raise RuntimeError(f"矢量文件无有效图层：{vpath}")
    return vds, lyr

def _srs_from_wkt(wkt):
    srs = osr.SpatialReference()
    if wkt is None or len(wkt.strip()) == 0:
        return None
    srs.ImportFromWkt(wkt)
    return srs

def _srs_equivalent(src_srs, target_srs):
    """判断两个 SRS 是否实质等价（容忍缺失 EPSG authority 的情况）。"""
    if src_srs is None or target_srs is None:
        return False
    try:
        if src_srs.IsSame(target_srs):
            return True
    except Exception:
        pass
    try:
        src_clone = src_srs.Clone()
        tgt_clone = target_srs.Clone()
        src_clone.AutoIdentifyEPSG()
        tgt_clone.AutoIdentifyEPSG()
        src_code = src_clone.GetAuthorityCode(None)
        tgt_code = tgt_clone.GetAuthorityCode(None)
        if src_code and tgt_code and src_code == tgt_code:
            return True
    except Exception:
        pass
    try:
        src_proj = (src_srs.ExportToProj4() or "").strip()
        tgt_proj = (target_srs.ExportToProj4() or "").strip()
        if src_proj and tgt_proj and src_proj == tgt_proj:
            return True
    except Exception:
        pass
    name_src = (src_srs.GetAttrValue("GEOGCS") or "").lower()
    name_tgt = (target_srs.GetAttrValue("GEOGCS") or "").lower()
    if "wgs" in name_src and "84" in name_src and "wgs" in name_tgt and "84" in name_tgt:
        return True
    return False

def _vector_reproject_to(vds, target_srs):
    """使用 GDAL VectorTranslate 将矢量重投影到 target_srs（返回临时GPKG路径与Layer）。"""
    lyr = vds.GetLayer(0)
    src_srs = lyr.GetSpatialRef() if lyr is not None else None
    tmp_dir = tempfile.mkdtemp(prefix="mask_vec_")
    gpkg_path = os.path.join(tmp_dir, "vec.gpkg")

    target_wkt = target_srs.ExportToWkt()
    common_kwargs = dict(geometryType="MULTIPOLYGON")
    if _srs_equivalent(src_srs, target_srs):
        # 只需覆盖 SRS 元数据，避免触发 PROJ 变换。
        opts = gdal.VectorTranslateOptions(srcSRS=target_wkt, dstSRS=target_wkt, reproject=False, **common_kwargs)
    else:
        opts = gdal.VectorTranslateOptions(dstSRS=target_wkt, **common_kwargs)

    vds2 = gdal.VectorTranslate(gpkg_path, vds, options=opts)
    if vds2 is None:
        raise RuntimeError("VectorTranslate 失败（重投影矢量失败）。")
    lyr2 = vds2.GetLayer(0)
    if lyr2 is None:
        raise RuntimeError("重投影后的矢量缺少图层。")
    return gpkg_path, vds2, lyr2, tmp_dir

def _layer_has_field(lyr, field_name):
    defn = lyr.GetLayerDefn()
    for i in range(defn.GetFieldCount()):
        if defn.GetFieldDefn(i).GetName().lower() == field_name.lower():
            return True
    return False

def _choose_driver(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        return "GTiff", [
            "COMPRESS=DEFLATE", "PREDICTOR=2", "ZLEVEL=6",
            "TILED=YES", "BIGTIFF=IF_SAFER", "NUM_THREADS=ALL_CPUS"
        ]
    if ext == ".nc":
        return "NetCDF", [
            "FORMAT=NC4", "COMPRESS=DEFLATE", "ZLEVEL=4"
        ]
    raise RuntimeError(f"暂不支持的输出格式：{path}")

def main():
    if not FORCE_UTF8:
        gdal.SetConfigOption("OGR_FORCE_UTF8", "NO")
    if SHAPE_ENCODING:
        gdal.SetConfigOption("SHAPE_ENCODING", SHAPE_ENCODING)

    # 1) 打开参考栅格，读取网格参数
    rds = gdal.Open(ref_tif_path, gdal.GA_ReadOnly)
    if rds is None:
        raise RuntimeError(f"无法打开参考栅格：{ref_tif_path}")
    gt = rds.GetGeoTransform()
    proj_wkt = rds.GetProjection()
    cols, rows = rds.RasterXSize, rds.RasterYSize
    px_w, px_h = gt[1], abs(gt[5])
    srs_raster = _srs_from_wkt(proj_wkt)
    if srs_raster is None:
        raise RuntimeError("参考栅格缺少投影信息（Projection WKT 为空）。")

    # 简单提示像元是否接近 1/12°
    try:
        approx_1_12 = abs(px_w - (1.0/12.0)) < 1e-6 and abs(px_h - (1.0/12.0)) < 1e-6
        if not approx_1_12:
            print(f"[警告] 参考栅格像元大小为 ({px_w}, {px_h})，并非严格1/12°；将仍以参考栅格为准对齐输出。")
    except Exception:
        pass

    # 2) 打开矢量，投影到与栅格一致的SRS
    vds, lyr = _open_vector(shp_path)
    gpkg_path, vds2, lyr2, tmp_dir = _vector_reproject_to(vds, srs_raster)
    if lyr2 is None:
        raise RuntimeError("矢量图层打开失败。")

    # 3) 创建输出栅格（与参考一致）
    drv_name, creation_opts = _choose_driver(out_mask_path)
    drv = gdal.GetDriverByName(drv_name)
    ods = drv.Create(out_mask_path, cols, rows, 1, gdal.GDT_Int32, options=creation_opts)
    if ods is None:
        raise RuntimeError(f"无法创建输出：{out_mask_path}")
    ods.SetGeoTransform(gt)
    ods.SetProjection(proj_wkt)
    band = ods.GetRasterBand(1)
    band.SetNoDataValue(NODATA_VALUE)
    band.Fill(NODATA_VALUE)

    # 4) 栅格化（优先用ID1字段；若没有则退化为0/1掩码）
    use_attribute = _layer_has_field(lyr2, ATTRIBUTE_FIELD)
    if use_attribute:
        ropts = [f"ATTRIBUTE={ATTRIBUTE_FIELD}"]
        if ALL_TOUCHED:
            ropts.append("ALL_TOUCHED=TRUE")
        err = gdal.RasterizeLayer(ods, [1], lyr2, options=ropts)
        mode = f"ATTRIBUTE={ATTRIBUTE_FIELD}"
    else:
        ropts = [f"BURN_VALUE=1"]
        if ALL_TOUCHED:
            ropts.append("ALL_TOUCHED=TRUE")
        err = gdal.RasterizeLayer(ods, [1], lyr2, burn_values=[1], options=ropts)
        mode = "BINARY(0/1)"

    if err != 0:
        raise RuntimeError(f"RasterizeLayer 失败（mode={mode}）。")

    # 5) 可选：构建金字塔（快速浏览）
    if drv_name == "GTiff":
        try:
            ods.BuildOverviews("NEAREST", [2, 4, 8, 16, 32])
        except Exception:
            pass

    # 6) 收尾
    band = None; ods = None
    vds = None
    vds2 = None
    rds = None
    if tmp_dir:
        try:
            import shutil; shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    print("完成：", out_mask_path)
    print(f"与参考一致：size=({cols},{rows}), pixel=({px_w},{px_h}), CRS={srs_raster.GetAttrValue('AUTHORITY',1) or 'WKT'}")
    print(f"栅格化方式：{mode}；NoData={NODATA_VALUE}；ALL_TOUCHED={ALL_TOUCHED}")

if __name__ == "__main__":
    # 可按需设置 GDAL/PROJ 数据路径（Windows 中文路径环境常见问题）：
    # import os
    # os.environ.setdefault('PROJ_LIB', r'D:\\Anaconda3\\Library\\share\\proj')
    # os.environ.setdefault('GDAL_DATA', r'D:\\Anaconda3\\Library\\share')
    main()
