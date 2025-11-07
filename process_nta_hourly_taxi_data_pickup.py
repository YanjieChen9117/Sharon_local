#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NYC出租车数据按小时+NTA区域聚合处理脚本

功能：
1. 循环读取data/文件夹中的每月出租车数据（Yellow & Green，同时读取并合并）
2. 进行数据清洗和特征工程
3. 按小时+NTA出发区域聚合数据
4. 保存所有月份的聚合数据供后续分析

数据源：
- Yellow Taxi: yellow_tripdata_*.parquet
- Green Taxi: green_tripdata_*.parquet

输出：
- 每行代表某个小时中某个NTA区域出发的所有行程的聚合数据

作者: AI Assistant
日期: 2025-01-08
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# 节假日列表（来自is_holiday.md）
HOLIDAYS = {
    '2023-01-02', '2023-01-16', '2023-02-13', '2023-02-20', '2023-05-29',
    '2023-06-19', '2023-07-04', '2023-09-04', '2023-10-09', '2023-11-07',
    '2023-11-10', '2023-11-23', '2023-12-25', '2024-01-01', '2024-01-15',
    '2024-02-12', '2024-02-19', '2024-05-27', '2024-06-19', '2024-07-04',
    '2024-09-02', '2024-10-14', '2024-11-05', '2024-11-11', '2024-11-28',
    '2024-12-25', '2025-01-01', '2025-01-20', '2025-02-12', '2025-02-17',
    '2025-05-26', '2025-06-19', '2025-07-04', '2025-09-01', '2025-10-13',
    '2025-11-04', '2025-11-11', '2025-11-27', '2025-12-25'
}

# 政策实施日期
POLICY_DATE = pd.Timestamp('2025-01-05')

# NTA映射字典（从NTA_lookup.md解析）
NTA_MAPPING = {
    12: 'MN0191', 104: 'MN0191',
    13: 'MN0101', 261: 'MN0101', 37: 'MN0101', 38: 'MN0101', 209: 'MN0101',
    231: 'MN0102',
    45: 'MN0301',
    125: 'MN0201', 211: 'MN0201', 144: 'MN0201',
    148: 'MN0302', 232: 'MN0302',
    158: 'MN0203', 249: 'MN0203',
    113: 'MN0202', 114: 'MN0202',
    4: 'MN0303', 79: 'MN0303',
    246: 'MN0401', 68: 'MN0401', 90: 'MN0401',
    186: 'MN0501', 234: 'MN0501', 164: 'MN0501', 100: 'MN0501',
    107: 'MN0602',
    224: 'MN0601',
    137: 'MN0603', 233: 'MN0603', 170: 'MN0603',
    50: 'MN0402', 48: 'MN0402',
    230: 'MN0502', 163: 'MN0502', 161: 'MN0502',
    162: 'MN0604', 299: 'MN0604',
    142: 'MN0701', 143: 'MN0701',
    238: 'MN0702', 239: 'MN0702',
    154: 'MN0703', 21: 'MN0703',
    43: 'MN6491',
    236: 'MN0802', 237: 'MN0802',
    140: 'MN0801', 141: 'MN0801', 202: 'MN0801',
    262: 'MN0803', 263: 'MN0803',
    75: 'MN1101',
    74: 'MN1102',
    41: 'MN1001',
    166: 'MN0901',
    152: 'MN0902',
    42: 'MN1002',
    116: 'MN0903',
    244: 'MN1202',
    120: 'MN1291',
    243: 'MN1202',
    128: 'MN1292',
    127: 'MN1203',
    194: 'MN1191'
}


class NTAHourlyTaxiDataProcessor:
    """按小时+NTA区域处理出租车数据的处理器（合并Yellow & Green Taxi）"""
    
    def __init__(self, data_dir):
        """
        初始化处理器
        
        参数:
            data_dir: parquet文件所在目录
        """
        self.data_dir = data_dir
        self.aggregated_stats = []  # 存储所有聚合数据
        self.weather_data = self._load_weather_data()  # 加载天气数据
    
    def _load_weather_data(self):
        """
        加载天气数据
        
        返回:
            DataFrame: 处理后的天气数据，包含日期、降水和降雪信息
        """
        try:
            weather_file = os.path.join(self.data_dir, 'central_park_weather_records.csv')
            if not os.path.exists(weather_file):
                print(f"⚠ 警告: 天气数据文件不存在: {weather_file}")
                return None
            
            # 读取天气数据
            weather_df = pd.read_csv(weather_file)
            
            # 转换日期格式
            weather_df['DATE'] = pd.to_datetime(weather_df['DATE'])
            
            # 处理降水量和降雪量，将字符串转换为数值
            weather_df['PRCP'] = pd.to_numeric(weather_df['PRCP'], errors='coerce').fillna(0)
            weather_df['SNOW'] = pd.to_numeric(weather_df['SNOW'], errors='coerce').fillna(0)
            
            # 创建天气特征
            weather_df['is_rain'] = weather_df['PRCP'] > 0  # 有降水
            weather_df['is_snow'] = weather_df['SNOW'] > 0  # 有降雪
            
            # 只保留需要的列
            weather_df = weather_df[['DATE', 'PRCP', 'SNOW', 'is_rain', 'is_snow']].copy()
            
            print(f"✓ 成功加载天气数据: {len(weather_df)} 天的记录")
            print(f"  日期范围: {weather_df['DATE'].min().date()} 至 {weather_df['DATE'].max().date()}")
            print(f"  降雨天数: {weather_df['is_rain'].sum()} 天")
            print(f"  降雪天数: {weather_df['is_snow'].sum()} 天")
            
            return weather_df
            
        except Exception as e:
            print(f"❌ 加载天气数据失败: {str(e)}")
            return None
    
    def normalize_taxi_data(self, df, taxi_type):
        """
        标准化不同类型出租车的字段名
        
        参数:
            df: 原始数据框
            taxi_type: 出租车类型 ('yellow' 或 'green')
            
        返回:
            DataFrame: 标准化后的数据框
        """
        df = df.copy()
        
        if taxi_type == 'green':
            # Green taxi使用lpep前缀，需要重命名为标准字段名
            df = df.rename(columns={
                'lpep_pickup_datetime': 'pickup_datetime',
                'lpep_dropoff_datetime': 'dropoff_datetime'
            })
        elif taxi_type == 'yellow':
            # Yellow taxi使用tpep前缀，需要重命名为标准字段名
            df = df.rename(columns={
                'tpep_pickup_datetime': 'pickup_datetime',
                'tpep_dropoff_datetime': 'dropoff_datetime'
            })
        
        return df
        
    def clean_data(self, df, year_month=None):
        """
        数据清洗：移除异常值
        
        参数:
            df: 原始数据框
            year_month: 文件名中的年月字符串 (如: "2025-01")，用于验证数据日期
            
        返回:
            DataFrame: 清洗后的数据
        """
        print(f"  原始数据: {len(df):,} 条记录")
        
        # 如果提供了year_month，检查pickup_datetime是否在合理范围内
        if year_month:
            try:
                # 解析年月
                year, month = map(int, year_month.split('-'))
                
                # 计算该月的开始和结束时间
                month_start = pd.Timestamp(year=year, month=month, day=1)
                
                # 计算下个月的第一天作为结束时间
                if month == 12:
                    month_end = pd.Timestamp(year=year+1, month=1, day=1)
                else:
                    month_end = pd.Timestamp(year=year, month=month+1, day=1)
                
                # 统计日期范围外的记录
                date_valid = (df['pickup_datetime'] >= month_start) & \
                            (df['pickup_datetime'] < month_end)
                invalid_dates = (~date_valid).sum()
                
                if invalid_dates > 0:
                    print(f"  ⚠ 发现 {invalid_dates:,} 条记录的pickup时间不在 {year_month} 范围内，将被移除")
                    df = df[date_valid]
                else:
                    print(f"  ✓ 所有记录的pickup时间都在 {year_month} 范围内")
                    
            except Exception as e:
                print(f"  ⚠ 日期验证失败: {str(e)}")
        
        # 数据清洗条件
        df_clean = df[
            (df['trip_distance'] > 0) &          # 距离必须大于0
            (df['trip_distance'] < 100) &        # 距离小于100英里
            (df['fare_amount'] > 0) &            # 费用必须大于0
            (df['total_amount'] > 0) &           # 总费用必须大于0
            (df['total_amount'] < 500) &         # 总费用小于500美元
            (df['passenger_count'] > 0) &        # 乘客数必须大于0
            (df['passenger_count'] <= 6)         # 乘客数合理范围
        ]
        
        print(f"  清洗后数据: {len(df_clean):,} 条记录 (保留率: {len(df_clean)/len(df)*100:.1f}%)")
        
        return df_clean
    
    def remove_unnecessary_columns(self, df):
        """
        删除不需要的列
        
        参数:
            df: 数据框
            
        返回:
            DataFrame: 删除列后的数据框
        """
        columns_to_remove = ['VendorID', 'RatecodeID', 'store_and_fwd_flag']
        
        # 只删除存在的列
        existing_cols_to_remove = [col for col in columns_to_remove if col in df.columns]
        
        if existing_cols_to_remove:
            df = df.drop(columns=existing_cols_to_remove)
            print(f"  已删除列: {', '.join(existing_cols_to_remove)}")
        
        return df
    
    def feature_engineering(self, df):
        """
        特征工程：计算派生特征和时间标签
        
        参数:
            df: 数据框
            
        返回:
            DataFrame: 添加了派生特征的数据框
        """
        # 计算行程时长（分钟）
        df['trip_duration_min'] = (
            df['dropoff_datetime'] - df['pickup_datetime']
        ).dt.total_seconds() / 60
        
        # 移除时长异常的记录（小于0或大于180分钟）
        df = df[(df['trip_duration_min'] > 0) & (df['trip_duration_min'] < 180)]
        
        # 计算平均速度（英里/小时）
        df['avg_speed_mph'] = df['trip_distance'] / (df['trip_duration_min'] / 60)
        df['avg_speed_mph'] = df['avg_speed_mph'].replace([np.inf, -np.inf], np.nan)
        
        # 移除速度异常的记录（小于0或大于100mph）
        df = df[(df['avg_speed_mph'] > 0) & (df['avg_speed_mph'] < 100)]
        
        # 计算小费率（仅针对信用卡支付，payment_type=1）
        df['tip_rate'] = np.where(
            df['payment_type'] == 1,
            df['tip_amount'] / df['fare_amount'],
            np.nan
        )
        df['tip_rate'] = df['tip_rate'].replace([np.inf, -np.inf], np.nan)
        
        # 提取时间特征 - 向下取整到小时（用于聚合）
        df['pickup_hour'] = df['pickup_datetime'].dt.floor('H')
        df['year'] = df['pickup_datetime'].dt.year
        df['month'] = df['pickup_datetime'].dt.month
        df['day'] = df['pickup_datetime'].dt.day  # 月份中的第几天
        df['day_of_week'] = df['pickup_datetime'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['hour_of_day'] = df['pickup_datetime'].dt.hour
        
        # 添加政策状态标签
        df['is_pre_policy'] = df['pickup_datetime'] < POLICY_DATE
        df['policy_status'] = df['is_pre_policy'].map({True: 'pre-policy', False: 'post-policy'})
        
        # 添加周末状态标签
        df['is_weekday'] = df['day_of_week'] < 5  # Monday-Friday = 0-4
        df['weekend_status'] = df['is_weekday'].map({True: 'weekday', False: 'weekend'})
        
        # 添加节假日标签
        df['date_str'] = df['pickup_datetime'].dt.strftime('%Y-%m-%d')
        df['is_holiday_date'] = df['date_str'].isin(HOLIDAYS)
        df['is_weekend'] = ~df['is_weekday']
        df['holiday'] = df['is_holiday_date'] | df['is_weekend']  # 节假日或周末
        
        # 添加天气特征
        df = self._add_weather_features(df)
        
        # 添加NTA映射
        df = self._add_nta_mapping(df)
        
        # 只保留有NTA映射的记录（因为我们按NTA聚合）
        df = df[df['PUNTA'].notna()].copy()
        
        # 清理临时列
        df = df.drop(columns=['is_pre_policy', 'is_weekday', 'date_str', 'is_holiday_date', 'is_weekend'])
        
        return df
    
    def _add_weather_features(self, df):
        """
        添加天气特征到出租车数据
        
        参数:
            df: 出租车数据框
            
        返回:
            DataFrame: 添加了天气特征的数据框
        """
        if self.weather_data is None:
            print("  ⚠ 天气数据不可用，跳过天气特征")
            # 添加默认值
            df['is_rain'] = False
            df['is_snow'] = False
            return df
        
        # 提取日期（不包含时间）
        df['pickup_date'] = df['pickup_datetime'].dt.date
        
        # 创建天气数据的日期副本用于匹配
        weather_lookup = self.weather_data.copy()
        weather_lookup['pickup_date'] = weather_lookup['DATE'].dt.date
        
        # 合并天气数据
        df = df.merge(
            weather_lookup[['pickup_date', 'is_rain', 'is_snow']], 
            on='pickup_date', 
            how='left'
        )
        
        # 处理缺失值（对于没有天气数据的日期）
        df['is_rain'] = df['is_rain'].fillna(False)
        df['is_snow'] = df['is_snow'].fillna(False)
        
        # 清理临时列
        df = df.drop(columns=['pickup_date'])
        
        # 统计天气情况
        rain_trips = df['is_rain'].sum()
        snow_trips = df['is_snow'].sum()
        total_trips = len(df)
        
        if total_trips > 0:
            print(f"  天气特征统计: 雨天行程={rain_trips:,} ({rain_trips/total_trips:.1%}), "
                  f"雪天行程={snow_trips:,} ({snow_trips/total_trips:.1%})")
        
        return df
    
    def _add_nta_mapping(self, df):
        """
        添加NTA区域映射到出租车数据
        
        参数:
            df: 出租车数据框
            
        返回:
            DataFrame: 添加了PUNTA列的数据框
        """
        # 为上车地点添加NTA映射
        df['PUNTA'] = df['PULocationID'].map(NTA_MAPPING)
        
        # 统计NTA映射情况
        punta_mapped = df['PUNTA'].notna().sum()
        total_trips = len(df)
        
        print(f"  NTA映射统计: PUNTA映射={punta_mapped:,} ({punta_mapped/total_trips:.1%})")
        
        return df
    
    def aggregate_by_hour_nta(self, df):
        """
        按小时+NTA出发区域聚合数据
        
        参数:
            df: 处理后的数据框
            
        返回:
            DataFrame: 按小时+NTA聚合的数据框
        """
        if df is None or len(df) == 0:
            return None
        
        # 按小时和NTA区域分组
        grouped = df.groupby(['pickup_hour', 'PUNTA'])
        
        # 计算聚合统计
        aggregated = pd.DataFrame({
            # 时间特征（取每组的第一个值，因为同一组内这些值相同）
            'pickup_hour': grouped['pickup_hour'].first(),
            'PUNTA': grouped['PUNTA'].first(),
            'year': grouped['year'].first(),
            'month': grouped['month'].first(),
            'day': grouped['day'].first(),
            'day_of_week': grouped['day_of_week'].first(),
            'hour_of_day': grouped['hour_of_day'].first(),
            
            # 状态标签（取每组的第一个值）
            'policy_status': grouped['policy_status'].first(),
            'weekend_status': grouped['weekend_status'].first(),
            'holiday': grouped['holiday'].first(),
            
            # 天气特征（取每组的第一个值，因为同一天内天气相同）
            'is_rain': grouped['is_rain'].first(),
            'is_snow': grouped['is_snow'].first(),
            
            # 聚合特征
            'total_trips': grouped.size(),
            'total_distance': grouped['trip_distance'].sum(),
            'total_passengers': grouped['passenger_count'].sum(),
            'total_duration': grouped['trip_duration_min'].sum(),
            
            # 平均值特征
            # avg_speed: 使用总距离/总时间计算，更准确
            'avg_tip_rate': grouped['tip_rate'].mean(),
            'avg_fare': grouped['fare_amount'].mean(),
        })
        
        # 计算平均速度：总距离（英里）/ 总时间（小时）
        aggregated['avg_speed'] = aggregated['total_distance'] / (aggregated['total_duration'] / 60)
        aggregated['avg_speed'] = aggregated['avg_speed'].replace([np.inf, -np.inf], np.nan)
        
        # 重置索引，使pickup_hour和PUNTA成为普通列
        aggregated = aggregated.reset_index(drop=True)
        
        return aggregated
    
    def load_and_merge_monthly_data(self, year_month):
        """
        加载并合并某个月份的Yellow和Green出租车数据
        
        参数:
            year_month: 年月字符串，格式为 "YYYY-MM"
            
        返回:
            DataFrame: 合并后的数据框，如果文件不存在则返回None
        """
        # 构建文件路径
        yellow_file = os.path.join(self.data_dir, f"yellow_tripdata_{year_month}.parquet")
        green_file = os.path.join(self.data_dir, f"green_tripdata_{year_month}.parquet")
        
        dfs = []
        
        # 读取Yellow taxi数据
        if os.path.exists(yellow_file):
            print(f"  读取 Yellow Taxi 数据: {os.path.basename(yellow_file)}")
            yellow_df = pd.read_parquet(yellow_file)
            yellow_df = self.normalize_taxi_data(yellow_df, 'yellow')
            dfs.append(yellow_df)
        else:
            print(f"  ⚠ Yellow Taxi 文件不存在: {yellow_file}")
        
        # 读取Green taxi数据
        if os.path.exists(green_file):
            print(f"  读取 Green Taxi 数据: {os.path.basename(green_file)}")
            green_df = pd.read_parquet(green_file)
            green_df = self.normalize_taxi_data(green_df, 'green')
            dfs.append(green_df)
        else:
            print(f"  ⚠ Green Taxi 文件不存在: {green_file}")
        
        # 合并数据
        if dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            print(f"  合并后总记录数: {len(merged_df):,}")
            return merged_df
        else:
            print(f"  ❌ 未找到 {year_month} 的任何数据文件")
            return None
    
    def process_monthly_data(self, year_month):
        """
        处理单个月份的数据（Yellow & Green Taxi合并）
        
        参数:
            year_month: 年月字符串，格式为 "YYYY-MM"
            
        返回:
            DataFrame: 该月的聚合数据
        """
        try:
            print(f"\n{'='*80}")
            print(f"正在处理: {year_month}")
            print(f"{'='*80}")
            
            # 1. 读取并合并数据
            print("步骤 1/6: 读取并合并Yellow和Green数据...")
            df = self.load_and_merge_monthly_data(year_month)
            
            if df is None or len(df) == 0:
                print("  ⚠ 没有数据可处理")
                return None
            
            # 2. 数据清洗（包含日期验证）
            print("步骤 2/6: 数据清洗与日期验证...")
            df = self.clean_data(df, year_month=year_month)
            
            # 3. 删除不必要的列
            print("步骤 3/6: 删除不必要的列...")
            df = self.remove_unnecessary_columns(df)
            
            # 4. 特征工程
            print("步骤 4/6: 特征工程...")
            df = self.feature_engineering(df)
            print(f"  特征工程后: {len(df):,} 条记录")
            
            # 5. 按小时+NTA聚合
            print("步骤 5/6: 按小时+NTA区域聚合...")
            aggregated_data = self.aggregate_by_hour_nta(df)
            
            if aggregated_data is not None:
                print(f"  ✓ 生成了 {len(aggregated_data)} 条聚合记录")
                print(f"  NTA区域数: {aggregated_data['PUNTA'].nunique()}")
                return aggregated_data
            else:
                print("  ⚠ 未生成聚合数据")
                return None
                
        except Exception as e:
            print(f"  ❌ 处理文件出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_all_files(self):
        """
        处理data目录下的所有parquet文件（Yellow & Green Taxi）
        """
        print("\n" + "="*80)
        print("NYC出租车数据按小时+NTA区域聚合处理 (Yellow & Green Taxi合并)")
        print("="*80)
        
        # 获取所有yellow和green出租车数据文件
        yellow_pattern = os.path.join(self.data_dir, "yellow_tripdata_*.parquet")
        green_pattern = os.path.join(self.data_dir, "green_tripdata_*.parquet")
        
        yellow_files = glob.glob(yellow_pattern)
        green_files = glob.glob(green_pattern)
        
        # 提取所有唯一的年月
        year_months = set()
        for file in yellow_files + green_files:
            basename = os.path.basename(file)
            if 'yellow_tripdata_' in basename:
                year_month = basename.replace('yellow_tripdata_', '').replace('.parquet', '')
                year_months.add(year_month)
            elif 'green_tripdata_' in basename:
                year_month = basename.replace('green_tripdata_', '').replace('.parquet', '')
                year_months.add(year_month)
        
        # 排序年月
        year_months = sorted(year_months)
        
        if not year_months:
            print(f"\n❌ 错误: 在 {self.data_dir} 目录下未找到任何出租车数据文件")
            print("请确保数据文件已放置在正确的目录中。")
            return
        
        print(f"\n找到 {len(year_months)} 个唯一月份的数据:")
        for i, ym in enumerate(year_months, 1):
            print(f"  {i}. {ym}")
        
        # 处理每个月的数据
        for i, year_month in enumerate(year_months, 1):
            print(f"\n进度: [{i}/{len(year_months)}]")
            
            aggregated_data = self.process_monthly_data(year_month)
            
            if aggregated_data is not None:
                self.aggregated_stats.append(aggregated_data)
        
        print("\n" + "="*80)
        print(f"✅ 处理完成！成功处理 {len(self.aggregated_stats)} 个月份的数据")
        print("="*80)
    
    def save_results(self, output_file='nta_hourly_taxi_summary.csv'):
        """
        保存所有月份的聚合数据
        
        参数:
            output_file: 输出文件名
        """
        if not self.aggregated_stats:
            print("\n❌ 没有数据可以保存！")
            return None
        
        print("\n正在整合所有月份的聚合数据...")
        
        # 合并所有月份的聚合数据
        all_aggregated_data = pd.concat(self.aggregated_stats, ignore_index=True)
        
        # 按时间排序
        all_aggregated_data = all_aggregated_data.sort_values(['pickup_hour', 'PUNTA']).reset_index(drop=True)
        
        # 保存为CSV
        output_path = os.path.join(self.data_dir, output_file)
        all_aggregated_data.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"\n✓ 聚合数据已保存到: {output_path}")
        print(f"  总记录数: {len(all_aggregated_data):,} 行")
        print(f"  时间范围: {all_aggregated_data['pickup_hour'].min()} 至 {all_aggregated_data['pickup_hour'].max()}")
        print(f"  NTA区域数: {all_aggregated_data['PUNTA'].nunique()}")
        print(f"  文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        
        # 显示数据预览
        print("\n数据预览 (前5行):")
        print(all_aggregated_data.head())
        
        # 显示数据统计摘要
        print("\n" + "="*80)
        print("数据统计摘要")
        print("="*80)
        print(f"总记录数: {len(all_aggregated_data):,}")
        print(f"总行程数: {all_aggregated_data['total_trips'].sum():,.0f}")
        print(f"总距离: {all_aggregated_data['total_distance'].sum():,.2f} 英里")
        print(f"总乘客数: {all_aggregated_data['total_passengers'].sum():,.0f}")
        print(f"平均每小时每NTA行程数: {all_aggregated_data['total_trips'].mean():.2f}")
        print(f"平均行程距离: {all_aggregated_data['total_distance'].sum() / all_aggregated_data['total_trips'].sum():.2f} 英里")
        print(f"平均速度: {all_aggregated_data['avg_speed'].mean():.2f} mph")
        print(f"平均小费率: {all_aggregated_data['avg_tip_rate'].mean():.2%}")
        print(f"平均费用: {all_aggregated_data['avg_fare'].mean():.2f} 美元")
        print(f"\n状态标签分布:")
        print(f"  政策前期比例: {(all_aggregated_data['policy_status'] == 'pre-policy').mean():.2%}")
        print(f"  政策后期比例: {(all_aggregated_data['policy_status'] == 'post-policy').mean():.2%}")
        print(f"  工作日比例: {(all_aggregated_data['weekend_status'] == 'weekday').mean():.2%}")
        print(f"  周末比例: {(all_aggregated_data['weekend_status'] == 'weekend').mean():.2%}")
        print(f"  节假日比例: {all_aggregated_data['holiday'].mean():.2%}")
        print(f"\n天气特征分布:")
        print(f"  雨天比例: {all_aggregated_data['is_rain'].mean():.2%}")
        print(f"  雪天比例: {all_aggregated_data['is_snow'].mean():.2%}")
        print(f"\nNTA区域统计:")
        top_ntas = all_aggregated_data.groupby('PUNTA')['total_trips'].sum().sort_values(ascending=False).head(10)
        print("  前10个NTA区域（按总行程数）:")
        for nta, trips in top_ntas.items():
            print(f"    {nta}: {trips:,.0f} 行程")
        print("="*80)
        
        return all_aggregated_data


def main():
    """主函数"""
    # 设置数据目录
    data_dir = "/Users/yanjiechen/Documents/Github/Sharon_local/data"
    
    # 检查目录是否存在
    if not os.path.exists(data_dir):
        print(f"❌ 错误: 数据目录不存在: {data_dir}")
        print("请创建data目录并将parquet文件放入其中。")
        return
    
    # 创建处理器实例
    processor = NTAHourlyTaxiDataProcessor(data_dir)
    
    # 处理所有文件
    processor.process_all_files()
    
    # 保存结果
    if processor.aggregated_stats:
        aggregated_df = processor.save_results('nta_hourly_taxi_summary.csv')
        print("\n✅ 所有任务完成！")
    else:
        print("\n⚠ 警告: 没有成功处理任何数据文件")


if __name__ == "__main__":
    main()

