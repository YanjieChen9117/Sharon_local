#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
天气特征工作流脚本

功能：
1. 从NTA_lookup.md文件中读取需要被计算的NTA zone list
2. 计算每个区域的质心坐标
3. 为每个区域获取2023-01-01到2025-10-15的天气数据
4. 将结果存储在data/文件夹中

数据分辨率：每小时（hourly）
"""

import requests
import pandas as pd
import numpy as np
import re
import os
import sys
from datetime import datetime
import time

# 配置
NTA_LOOKUP_FILE = "documents/NTA_lookup.md"
NTA_CSV_FILE = "data/2020_Neighborhood_Tabulation_Areas_(NTAs)_20251115.csv"
OUTPUT_FILE = "data/nta_weather_data.csv"
START_DATE = "2023-01-01"
END_DATE = "2025-10-15"


def read_nta_list_from_lookup(lookup_file):
    """
    从NTA_lookup.md文件中读取NTA代码列表
    
    格式示例：
    MN0191 - 12, 104
    MN0101 - 13, 261, 37, 38,  209
    """
    nta_list = []
    
    if not os.path.exists(lookup_file):
        print(f"错误: 文件不存在 {lookup_file}")
        return nta_list
    
    print(f"正在读取NTA列表: {lookup_file}")
    with open(lookup_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # 提取NTA代码（行首到第一个空格或"-"之前）
            match = re.match(r'^([A-Z]{2}\d{4})', line)
            if match:
                nta_code = match.group(1)
                nta_list.append(nta_code)
    
    print(f"成功读取 {len(nta_list)} 个NTA代码")
    return nta_list


def parse_multipolygon(geom_string):
    """
    解析MULTIPOLYGON字符串，提取所有坐标点
    格式: MULTIPOLYGON (((lon lat, lon lat, ...), ...))
    """
    if pd.isna(geom_string) or not geom_string:
        return None, None
    
    # 提取所有坐标对
    pattern = r'(-?\d+\.?\d+)\s+(-?\d+\.?\d+)'
    matches = re.findall(pattern, str(geom_string))
    
    if not matches:
        return None, None
    
    # 提取所有经度和纬度
    lons = [float(match[0]) for match in matches]
    lats = [float(match[1]) for match in matches]
    
    # 计算质心（所有点的平均值）
    centroid_lat = np.mean(lats)
    centroid_lon = np.mean(lons)
    
    return centroid_lat, centroid_lon


def calculate_nta_centroids(nta_list, csv_file):
    """
    从CSV文件计算指定NTA列表的质心坐标
    """
    print(f"\n正在读取NTA CSV文件: {csv_file}")
    if not os.path.exists(csv_file):
        print(f"错误: 文件不存在 {csv_file}")
        return {}
    
    df = pd.read_csv(csv_file)
    print(f"成功读取 {len(df)} 条记录")
    
    # 检查必要的列
    if 'NTA2020' not in df.columns or 'the_geom' not in df.columns:
        print("错误: CSV文件缺少必要的列 (NTA2020 或 the_geom)")
        return {}
    
    # 计算质心
    print(f"\n正在计算 {len(nta_list)} 个NTA区域的质心坐标...")
    nta_coords = {}
    
    for nta_code in nta_list:
        # 在CSV中查找对应的NTA
        nta_row = df[df['NTA2020'] == nta_code]
        
        if nta_row.empty:
            print(f"  警告: 未找到NTA代码 {nta_code} 在CSV文件中")
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
            print(f"  ✓ {nta_code}: ({lat:.6f}, {lon:.6f}) - {nta_name}")
        else:
            print(f"  ✗ {nta_code}: 无法解析几何数据")
    
    print(f"\n成功计算 {len(nta_coords)} 个NTA区域的质心坐标")
    return nta_coords


def fetch_weather(lat, lon, start_date, end_date, nta_code=None):
    """
    从Open-Meteo API获取天气数据
    
    参数:
        lat: 纬度
        lon: 经度
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        nta_code: NTA代码（用于日志）
    
    返回:
        DataFrame: 包含每小时天气数据的DataFrame
    """
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
        print(f"  ✗ 获取天气数据失败: {e}")
        return None


def fetch_weather_for_all_ntas(nta_coords, start_date, end_date):
    """
    为所有NTA区域获取天气数据
    """
    print(f"\n正在为 {len(nta_coords)} 个NTA区域获取天气数据...")
    print(f"日期范围: {start_date} 到 {end_date}")
    print(f"数据分辨率: 每小时 (hourly)\n")
    
    weather_frames = []
    total_ntas = len(nta_coords)
    
    for idx, (nta_code, info) in enumerate(nta_coords.items(), 1):
        lat = info['latitude']
        lon = info['longitude']
        name = info['name']
        
        print(f"[{idx}/{total_ntas}] 正在获取 {nta_code} ({name}) 的天气数据...")
        print(f"  坐标: ({lat:.6f}, {lon:.6f})")
        
        df_temp = fetch_weather(lat, lon, start_date, end_date, nta_code)
        
        if df_temp is not None and len(df_temp) > 0:
            df_temp["nta_code"] = nta_code
            df_temp["nta_name"] = name
            weather_frames.append(df_temp)
            print(f"  ✓ 成功获取 {len(df_temp)} 条小时级记录")
        else:
            print(f"  ✗ 未获取到数据")
        
        # 添加延迟以避免API限流
        if idx < total_ntas:
            time.sleep(0.5)  # 每次请求间隔0.5秒
    
    if not weather_frames:
        print("\n错误: 没有成功获取任何天气数据")
        return None
    
    # 合并所有数据
    print(f"\n正在合并 {len(weather_frames)} 个NTA区域的数据...")
    df_weather = pd.concat(weather_frames, ignore_index=True)
    df_weather["datetime"] = pd.to_datetime(df_weather["datetime"]).dt.tz_localize(None)
    
    # 重新排列列的顺序
    columns_order = ['datetime', 'nta_code', 'nta_name', 'temperature', 'precipitation', 
                     'windspeed', 'humidity']
    df_weather = df_weather[columns_order]
    
    return df_weather


def save_weather_data(df_weather, output_file):
    """
    保存天气数据到CSV文件
    """
    print(f"\n正在保存天气数据到: {output_file}")
    
    # 确保data目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df_weather.to_csv(output_file, index=False)
    print(f"✓ 成功保存 {len(df_weather):,} 条记录到 {output_file}")
    
    # 显示数据统计
    print(f"\n数据统计:")
    print(f"  总记录数: {len(df_weather):,}")
    print(f"  NTA区域数: {df_weather['nta_code'].nunique()}")
    print(f"  日期范围: {df_weather['datetime'].min()} 到 {df_weather['datetime'].max()}")
    print(f"  数据分辨率: 每小时 (hourly)")
    print(f"\n数据预览:")
    print(df_weather.head(10))
    print(f"\n数据列:")
    print(df_weather.dtypes)


def main():
    """
    主函数：执行完整的天气特征工作流
    """
    print("=" * 80)
    print("天气特征工作流")
    print("=" * 80)
    
    start_time = datetime.now()
    
    # 步骤1: 从NTA_lookup.md读取NTA列表
    print("\n步骤 1/4: 读取NTA列表")
    nta_list = read_nta_list_from_lookup(NTA_LOOKUP_FILE)
    
    if not nta_list:
        print("错误: 未读取到任何NTA代码")
        return 1
    
    # 步骤2: 计算每个NTA区域的质心坐标
    print("\n步骤 2/4: 计算NTA质心坐标")
    nta_coords = calculate_nta_centroids(nta_list, NTA_CSV_FILE)
    
    if not nta_coords:
        print("错误: 未计算出任何NTA坐标")
        return 1
    
    # 步骤3: 获取天气数据
    print("\n步骤 3/4: 获取天气数据")
    df_weather = fetch_weather_for_all_ntas(nta_coords, START_DATE, END_DATE)
    
    if df_weather is None or len(df_weather) == 0:
        print("错误: 未获取到天气数据")
        return 1
    
    # 步骤4: 保存数据
    print("\n步骤 4/4: 保存数据")
    save_weather_data(df_weather, OUTPUT_FILE)
    
    # 完成
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 80)
    print("工作流完成！")
    print("=" * 80)
    print(f"总耗时: {duration}")
    print(f"输出文件: {OUTPUT_FILE}")
    print(f"数据分辨率: 每小时 (hourly)")
    print(f"每条记录代表一个NTA区域在一个小时的天气数据")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n工作流被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

