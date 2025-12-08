#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的天气数据工作流脚本

功能：
1. 使用NTA代码列表直接控制需要计算的区域
2. 计算每个区域的质心坐标
3. 为每个区域获取2023-01-01到2025-10-15的天气数据（包括降雪数据）
4. 合并现有数据（如果存在）
5. 将结果存储在data/文件夹中

数据分辨率：每小时（hourly）
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

# NTA区域列表 - 直接在这里配置需要计算的区域
NTA_LIST = [
    'MN0191', 'MN0101', 'MN0102', 'MN0301', 'MN0201', 'MN0302', 'MN0203',
    'MN0202', 'MN0303', 'MN0401', 'MN0501', 'MN0602', 'MN0601', 'MN0603',
    'MN0402', 'MN0502', 'MN0604', 'MN0701', 'MN0702', 'MN0703', 'MN6491',
    'MN0802', 'MN0801', 'MN0803', 'MN1101', 'MN1102', 'MN1001', 'MN0901',
    'MN0902', 'MN1002', 'MN0903', 'MN1202', 'MN1291', 'MN1292', 'MN1203',
    'MN1191'
]


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


def fetch_weather(lat, lon, start_date, end_date, nta_code=None, max_retries=3, retry_delay=65):
    """
    从Open-Meteo API获取天气数据（包括降雪数据）
    
    参数:
        lat: 纬度
        lon: 经度
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        nta_code: NTA代码（用于日志）
        max_retries: 最大重试次数（默认3次）
        retry_delay: 重试延迟时间（秒，默认65秒，略大于1分钟）
    
    返回:
        DataFrame: 包含每小时天气数据的DataFrame
    """
    # 构建API请求URL，包含降雪相关参数
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,precipitation,wind_speed_10m,relative_humidity_2m,snowfall,snow_depth"
        f"&timezone=America/New_York"
    )
    
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=60)
            
            # 检查是否是429错误（限流）
            if resp.status_code == 429:
                if attempt < max_retries - 1:
                    wait_time = retry_delay
                    print(f"  ⚠️  遇到API限流（429错误），等待 {wait_time} 秒后重试... (尝试 {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  ✗ 达到最大重试次数，仍然遇到API限流")
                    if hasattr(resp, 'text'):
                        print(f"  响应内容: {resp.text[:200]}")
                    return None
            
            resp.raise_for_status()
            data = resp.json()
            
            dt_index = pd.to_datetime(data["hourly"]["time"]).floor("h")
            
            # 构建DataFrame，包含所有天气参数
            df_dict = {
                "datetime": dt_index,
                "temperature": data["hourly"]["temperature_2m"],
                "precipitation": data["hourly"]["precipitation"],
                "windspeed": data["hourly"]["wind_speed_10m"],
                "humidity": data["hourly"]["relative_humidity_2m"]
            }
            
            # 尝试添加降雪数据（如果API返回）
            if "snowfall" in data["hourly"]:
                df_dict["snowfall"] = data["hourly"]["snowfall"]
            else:
                # 如果API不支持snowfall，使用None填充
                df_dict["snowfall"] = [None] * len(dt_index)
                print(f"  警告: API未返回snowfall数据，将使用None填充")
            
            if "snow_depth" in data["hourly"]:
                df_dict["snow_depth"] = data["hourly"]["snow_depth"]
            else:
                # 如果API不支持snow_depth，使用None填充
                df_dict["snow_depth"] = [None] * len(dt_index)
                print(f"  警告: API未返回snow_depth数据，将使用None填充")
            
            df = pd.DataFrame(df_dict)
            
            return df
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                # 429错误已经在上面处理了，这里不应该到达
                continue
            else:
                print(f"  ✗ HTTP错误: {e}")
                if hasattr(e.response, 'text'):
                    print(f"  响应内容: {e.response.text[:200]}")
                if attempt < max_retries - 1:
                    print(f"  等待 {retry_delay} 秒后重试... (尝试 {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                return None
        except Exception as e:
            print(f"  ✗ 获取天气数据失败: {e}")
            if attempt < max_retries - 1:
                print(f"  等待 {retry_delay} 秒后重试... (尝试 {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                continue
            return None
    
    return None


def fetch_weather_for_ntas(nta_coords, start_date, end_date, request_delay=10):
    """
    为所有NTA区域获取天气数据
    
    参数:
        nta_coords: NTA坐标字典
        start_date: 开始日期
        end_date: 结束日期
        request_delay: 每次请求之间的延迟时间（秒，默认10秒）
    """
    print(f"\n正在为 {len(nta_coords)} 个NTA区域获取天气数据...")
    print(f"日期范围: {start_date} 到 {end_date}")
    print(f"数据分辨率: 每小时 (hourly)")
    print(f"天气参数: temperature, precipitation, windspeed, humidity, snowfall, snow_depth")
    print(f"请求延迟: {request_delay} 秒/请求（以避免API限流）\n")
    
    weather_frames = []
    total_ntas = len(nta_coords)
    failed_ntas = []
    
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
            failed_ntas.append(nta_code)
        
        # 添加延迟以避免API限流（最后一个请求不需要延迟）
        if idx < total_ntas:
            print(f"  等待 {request_delay} 秒后继续下一个请求...")
            time.sleep(request_delay)
    
    if not weather_frames:
        print("\n错误: 没有成功获取任何天气数据")
        if failed_ntas:
            print(f"\n失败的NTA区域 ({len(failed_ntas)} 个):")
            for nta in failed_ntas:
                print(f"  - {nta}")
        return None
    
    # 合并所有数据
    print(f"\n正在合并 {len(weather_frames)} 个NTA区域的数据...")
    df_weather = pd.concat(weather_frames, ignore_index=True)
    df_weather["datetime"] = pd.to_datetime(df_weather["datetime"]).dt.tz_localize(None)
    
    # 显示成功和失败的统计
    print(f"\n数据获取统计:")
    print(f"  成功: {len(weather_frames)} 个NTA区域")
    if failed_ntas:
        print(f"  失败: {len(failed_ntas)} 个NTA区域")
        print(f"  失败的NTA: {', '.join(failed_ntas)}")
        print(f"\n提示: 可以稍后重新运行脚本，只处理失败的NTA区域")
    
    return df_weather


def merge_with_existing_data(df_new, existing_file):
    """
    合并新数据与现有数据
    """
    if os.path.exists(existing_file):
        print(f"\n正在读取现有数据文件: {existing_file}")
        df_existing = pd.read_csv(existing_file)
        df_existing["datetime"] = pd.to_datetime(df_existing["datetime"])
        print(f"  现有数据: {len(df_existing):,} 条记录")
        print(f"  现有NTA区域数: {df_existing['nta_code'].nunique()}")
        
        # 合并数据
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        
        # 去重：如果有相同datetime和nta_code的记录，保留新数据
        df_combined = df_combined.drop_duplicates(
            subset=['datetime', 'nta_code'],
            keep='last'
        )
        
        print(f"  合并后数据: {len(df_combined):,} 条记录")
        print(f"  合并后NTA区域数: {df_combined['nta_code'].nunique()}")
        
        return df_combined
    else:
        print(f"\n未找到现有数据文件，将创建新文件")
        return df_new


def save_weather_data(df_weather, output_file):
    """
    保存天气数据到CSV文件
    """
    print(f"\n正在保存天气数据到: {output_file}")
    
    # 确保data目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 重新排列列的顺序
    # 检查哪些列存在，按顺序排列
    base_columns = ['datetime', 'nta_code', 'nta_name']
    weather_columns = ['temperature', 'precipitation', 'windspeed', 'humidity']
    snow_columns = ['snowfall', 'snow_depth']
    
    columns_order = base_columns.copy()
    for col in weather_columns:
        if col in df_weather.columns:
            columns_order.append(col)
    for col in snow_columns:
        if col in df_weather.columns:
            columns_order.append(col)
    
    # 只保留存在的列
    available_columns = [col for col in columns_order if col in df_weather.columns]
    df_weather = df_weather[available_columns]
    
    df_weather.to_csv(output_file, index=False)
    print(f"✓ 成功保存 {len(df_weather):,} 条记录到 {output_file}")
    
    # 显示数据统计
    print(f"\n数据统计:")
    print(f"  总记录数: {len(df_weather):,}")
    print(f"  NTA区域数: {df_weather['nta_code'].nunique()}")
    print(f"  日期范围: {df_weather['datetime'].min()} 到 {df_weather['datetime'].max()}")
    print(f"  数据分辨率: 每小时 (hourly)")
    print(f"\n数据列:")
    print(df_weather.dtypes)
    print(f"\n数据预览:")
    print(df_weather.head(10))


def main():
    """
    主函数：执行完整的天气数据工作流
    """
    print("=" * 80)
    print("完整的天气数据工作流")
    print("=" * 80)
    
    start_time = datetime.now()
    
    # 步骤1: 计算每个NTA区域的质心坐标
    print("\n步骤 1/4: 计算NTA质心坐标")
    print(f"需要处理的NTA区域数: {len(NTA_LIST)}")
    nta_coords = calculate_nta_centroids(NTA_LIST, NTA_CSV_FILE)
    
    if not nta_coords:
        print("错误: 未计算出任何NTA坐标")
        return 1
    
    # 步骤2: 获取天气数据
    print("\n步骤 2/4: 获取天气数据")
    df_new = fetch_weather_for_ntas(nta_coords, START_DATE, END_DATE)
    
    if df_new is None or len(df_new) == 0:
        print("错误: 未获取到天气数据")
        return 1
    
    # 步骤3: 合并现有数据（如果存在）
    print("\n步骤 3/4: 合并现有数据")
    df_combined = merge_with_existing_data(df_new, EXISTING_FILE)
    
    # 步骤4: 保存数据
    print("\n步骤 4/4: 保存数据")
    save_weather_data(df_combined, OUTPUT_FILE)
    
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

