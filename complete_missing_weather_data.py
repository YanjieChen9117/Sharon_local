#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
补充获取缺失的NTA天气数据
"""

import requests
import pandas as pd
import numpy as np
import re
import os
import sys
import time
from datetime import datetime

# 配置
NTA_CSV_FILE = "data/2020_Neighborhood_Tabulation_Areas_(NTAs)_20251115.csv"
EXISTING_FILE = "data/nta_weather_data.csv"
OUTPUT_FILE = "data/nta_weather_data.csv"
START_DATE = "2023-01-01"
END_DATE = "2025-10-15"

# 缺失的NTA列表
MISSING_NTAS = [
     'MN0801', 'MN0802', 'MN0803', 'MN0901', 'MN0902', 'MN0903', 'MN1001', 'MN1002',
    'MN1101', 'MN1102', 'MN1191', 'MN1202', 'MN1203', 'MN1291', 'MN1292'
]


def parse_multipolygon(geom_string):
    """解析MULTIPOLYGON字符串，提取所有坐标点"""
    if pd.isna(geom_string) or not geom_string:
        return None, None
    
    pattern = r'(-?\d+\.?\d+)\s+(-?\d+\.?\d+)'
    matches = re.findall(pattern, str(geom_string))
    
    if not matches:
        return None, None
    
    lons = [float(match[0]) for match in matches]
    lats = [float(match[1]) for match in matches]
    
    centroid_lat = np.mean(lats)
    centroid_lon = np.mean(lons)
    
    return centroid_lat, centroid_lon


def get_nta_coords(nta_list, csv_file):
    """获取NTA坐标"""
    df = pd.read_csv(csv_file)
    nta_coords = {}
    
    for nta_code in nta_list:
        nta_row = df[df['NTA2020'] == nta_code]
        if nta_row.empty:
            continue
        
        row = nta_row.iloc[0]
        nta_name = row.get('NTAName', '')
        geom = row['the_geom']
        
        lat, lon = parse_multipolygon(geom)
        if lat and lon:
            nta_coords[nta_code] = {
                'name': nta_name,
                'latitude': lat,
                'longitude': lon
            }
    
    return nta_coords


def fetch_weather(lat, lon, start_date, end_date):
    """获取天气数据"""
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,precipitation,wind_speed_10m,relative_humidity_2m"
        f"&timezone=America/New_York"
    )
    
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        
        dt_index = pd.to_datetime(data["hourly"]["time"]).floor("h")
        df = pd.DataFrame({
            "datetime": dt_index,
            "temperature": data["hourly"]["temperature_2m"],
            "precipitation": data["hourly"]["precipitation"],
            "windspeed": data["hourly"]["wind_speed_10m"],
            "humidity": data["hourly"]["relative_humidity_2m"]
        })
        return df
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        return None


def main():
    print("=" * 60)
    print("补充获取缺失的NTA天气数据")
    print("=" * 60)
    
    # 获取缺失NTA的坐标
    print(f"\n正在获取 {len(MISSING_NTAS)} 个缺失NTA的坐标...")
    nta_coords = get_nta_coords(MISSING_NTAS, NTA_CSV_FILE)
    print(f"成功获取 {len(nta_coords)} 个NTA坐标")
    
    # 获取天气数据
    print(f"\n正在获取天气数据...")
    weather_frames = []
    
    for idx, (nta_code, info) in enumerate(nta_coords.items(), 1):
        lat = info['latitude']
        lon = info['longitude']
        name = info['name']
        
        print(f"[{idx}/{len(nta_coords)}] {nta_code} ({name})...")
        df_temp = fetch_weather(lat, lon, START_DATE, END_DATE)
        
        if df_temp is not None and len(df_temp) > 0:
            df_temp["nta_code"] = nta_code
            df_temp["nta_name"] = name
            weather_frames.append(df_temp)
            print(f"  ✓ 成功获取 {len(df_temp)} 条记录")
        else:
            print(f"  ✗ 获取失败")
        
        if idx < len(nta_coords):
            time.sleep(0.5)
    
    if not weather_frames:
        print("\n未获取到任何数据")
        return
    
    # 读取现有数据
    print(f"\n正在合并数据...")
    if os.path.exists(EXISTING_FILE):
        df_existing = pd.read_csv(EXISTING_FILE)
        print(f"  现有数据: {len(df_existing):,} 条记录")
    else:
        df_existing = pd.DataFrame()
        print(f"  现有数据: 0 条记录")
    
    # 合并新数据
    df_new = pd.concat(weather_frames, ignore_index=True)
    df_new["datetime"] = pd.to_datetime(df_new["datetime"]).dt.tz_localize(None)
    print(f"  新数据: {len(df_new):,} 条记录")
    
    # 合并所有数据
    if len(df_existing) > 0:
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    # 重新排列列
    columns_order = ['datetime', 'nta_code', 'nta_name', 'temperature', 
                     'precipitation', 'windspeed', 'humidity']
    df_combined = df_combined[columns_order]
    
    # 保存
    df_combined.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✓ 成功保存 {len(df_combined):,} 条记录到 {OUTPUT_FILE}")
    print(f"  NTA区域数: {df_combined['nta_code'].nunique()}")


if __name__ == "__main__":
    main()

