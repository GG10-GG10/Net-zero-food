# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 21:54:08 2025

@author: cheng
"""

# -*- coding: utf-8 -*-
"""
脚本功能：
1. 读取 mask_LUH2_025d.nc
2. 针对列表中未知的 Mask ID，计算其几何中心坐标
3. 使用 geopy 在线定位，自动获取并打印国家名称
"""

import numpy as np
import netCDF4 as nc
import time
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

# ------------ 设置 ------------
nc_path = r"..\..\src\mask_LUH2_025d.nc"  # 您的NC文件路径
# 这里填入您 Excel 截图中所有 #N/A 的 ID
target_ids = [2, 5, 8, 16, 21, 33, 36, 49, 79, 
              131, 132, 146, 147, 165, 173, 181, 187, 212]
# -----------------------------

def get_location_name(geolocator, lat, lon, retries=3):
    """尝试获取经纬度对应的地址信息"""
    for i in range(retries):
        try:
            # language='en' 输出英文名，方便和 ISO 标准对比
            # language='zh-cn' 可输出中文名
            location = geolocator.reverse((lat, lon), language='en', timeout=10)
            if location:
                return location.raw.get('address', {})
            return None
        except (GeocoderTimedOut, Exception) as e:
            print(f"    (连接超时，正在重试 {i+1}/{retries}...)")
            time.sleep(1)
    return None

def main():
    print(f"正在读取文件: {nc_path} ...")
    ds = nc.Dataset(nc_path)
    
    # 读取变量
    lat_arr = ds.variables['lat'][:]
    lon_arr = ds.variables['lon'][:]
    mask_arr = ds.variables['id1'][:]  # (lat, lon)
    
    # 创建地理编码器 (使用 OpenStreetMap 数据)
    geolocator = Nominatim(user_agent="geo_checker_script_v1")

    print("-" * 95)
    print(f"{'Mask_ID':<8} | {'Center Lat':<10} | {'Center Lon':<10} | {'Identified Country / Territory'}")
    print("-" * 95)

    # 网格化经纬度，方便索引
    # 注意：mask_arr 形状是 (lat, lon)，lon_grid 和 lat_grid 也要对应
    lon_grid, lat_grid = np.meshgrid(lon_arr, lat_arr)

    for uid in target_ids:
        # 1. 找到该 ID 所有的网格点
        rows, cols = np.where(mask_arr == uid)
        
        if len(rows) == 0:
            print(f"{uid:<8} | {'Empty':<10} | {'Empty':<10} | (该 ID 在地图中未找到像素)")
            continue

        # 2. 计算中心点 (简单的算术平均)
        # 对于群岛国家，平均点可能在海里，但 OpenStreetMap 通常能识别附近海域归属
        center_lat = np.mean(lat_grid[rows, cols])
        center_lon = np.mean(lon_grid[rows, cols])

        # 3. 在线查询国家名
        address = get_location_name(geolocator, center_lat, center_lon)
        
        country_name = "Unknown"
        cc = ""
        
        if address:
            # 优先获取 'country', 其次 'territory' (针对属地), 最后 'state'
            country_name = address.get('country', address.get('territory', 'Unknown'))
            cc = address.get('country_code', '').upper()
        
        # 4. 打印结果
        print(f"{uid:<8} | {center_lat:<10.4f} | {center_lon:<10.4f} | {country_name} ({cc})")
        
        # 礼貌性延时，避免 API 限制
        time.sleep(0.5)

    ds.close()
    print("-" * 95)
    print("完成。请根据上述英文名更新您的 Excel 映射表。")

if __name__ == "__main__":
    main()