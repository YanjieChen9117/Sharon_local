#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NYC黄色出租车数据按小时汇总处理脚本

功能：
1. 循环读取data/文件夹中的每月出租车数据
2. 进行数据清洗和特征工程
3. 为每个行程打上CBD交互标签（inside, in, out, non）
4. 按小时汇总数据
5. 保存所有月份的小时汇总数据供后续分析

作者: AI Assistant
日期: 2025-10-01
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# CBD区域ID列表（来自is_cbd.md）
CBD_ZONES = {
     50,  48, 163, 162, 229, 230, 161, 233, 246,  68,
    100, 186,  90, 164, 234, 107, 170, 137, 224, 158,
    249, 113, 114,  79,   4, 125, 211, 144, 148, 232,
    231,  45, 209,  13, 261,  87,  12,  88
}


class HourlyTaxiDataProcessor:
    """按小时处理出租车数据的处理器"""
    
    def __init__(self, data_dir):
        """
        初始化处理器
        
        参数:
            data_dir: parquet文件所在目录
        """
        self.data_dir = data_dir
        self.hourly_stats = []  # 存储所有小时的汇总数据
        
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
                date_valid = (df['tpep_pickup_datetime'] >= month_start) & \
                            (df['tpep_pickup_datetime'] < month_end)
                invalid_dates = (~date_valid).sum()
                
                if invalid_dates > 0:
                    print(f"  ⚠ 发现 {invalid_dates:,} 条记录的pickup时间不在 {year_month} 范围内，将被移除")
                    df = df[date_valid]
                else:
                    print(f"  ✓ 所有记录的pickup时间都在 {year_month} 范围内")
                    
            except Exception as e:
                print(f"  ⚠ 日期验证失败: {str(e)}")
        
        # 数据清洗条件（参考数据字段说明.md）
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
    
    def add_cbd_labels(self, df):
        """
        为每个行程添加CBD交互标签
        
        CBD交互类型:
        - 'inside': 上车和下车都在CBD区域内
        - 'in':     上车不在CBD，下车在CBD（进入CBD）
        - 'out':    上车在CBD，下车不在CBD（离开CBD）
        - 'non':    上车和下车都不在CBD
        
        参数:
            df: 数据框
            
        返回:
            DataFrame: 添加了cbd_interaction列的数据框
        """
        # 判断上车和下车地点是否在CBD
        pickup_in_cbd = df['PULocationID'].isin(CBD_ZONES)
        dropoff_in_cbd = df['DOLocationID'].isin(CBD_ZONES)
        
        # 根据组合情况分配标签
        conditions = [
            (pickup_in_cbd & dropoff_in_cbd),      # 都在CBD内
            (~pickup_in_cbd & dropoff_in_cbd),     # 进入CBD
            (pickup_in_cbd & ~dropoff_in_cbd),     # 离开CBD
            (~pickup_in_cbd & ~dropoff_in_cbd)     # 都不在CBD
        ]
        
        choices = ['inside', 'in', 'out', 'non']
        
        df['cbd_interaction'] = np.select(conditions, choices, default='unknown')
        
        # 打印CBD交互统计
        cbd_counts = df['cbd_interaction'].value_counts()
        print(f"  CBD交互分布: inside={cbd_counts.get('inside', 0):,}, "
              f"in={cbd_counts.get('in', 0):,}, "
              f"out={cbd_counts.get('out', 0):,}, "
              f"non={cbd_counts.get('non', 0):,}")
        
        return df
    
    def feature_engineering(self, df):
        """
        特征工程：计算派生特征
        
        参数:
            df: 数据框
            
        返回:
            DataFrame: 添加了派生特征的数据框
        """
        # 计算行程时长（分钟）
        df['trip_duration_min'] = (
            df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
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
        
        # 提取小时信息
        df['pickup_hour'] = df['tpep_pickup_datetime'].dt.floor('H')  # 向下取整到小时
        
        return df
    
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
    
    def aggregate_by_hour(self, df):
        """
        按小时汇总数据
        
        参数:
            df: 处理后的数据框
            
        返回:
            DataFrame: 按小时汇总的数据框
        """
        if df is None or len(df) == 0:
            return None
        
        # 按小时分组
        hourly_groups = df.groupby('pickup_hour')
        
        # 计算汇总统计
        hourly_summary = pd.DataFrame({
            # 基本统计
            'total_trips': hourly_groups.size(),
            'avg_distance': hourly_groups['trip_distance'].mean(),
            'avg_passengers': hourly_groups['passenger_count'].mean(),
            'avg_duration': hourly_groups['trip_duration_min'].mean(),
            'avg_speed': hourly_groups['avg_speed_mph'].mean(),
            'avg_tip_rate': hourly_groups['tip_rate'].mean(),
            
            # CBD交互比例
            'cbd_inside_ratio': hourly_groups.apply(
                lambda x: (x['cbd_interaction'] == 'inside').sum() / len(x)
            ),
            'cbd_in_ratio': hourly_groups.apply(
                lambda x: (x['cbd_interaction'] == 'in').sum() / len(x)
            ),
            'cbd_out_ratio': hourly_groups.apply(
                lambda x: (x['cbd_interaction'] == 'out').sum() / len(x)
            ),
            'cbd_non_ratio': hourly_groups.apply(
                lambda x: (x['cbd_interaction'] == 'non').sum() / len(x)
            ),
            
            # 额外的有用指标
            'total_revenue': hourly_groups['total_amount'].sum(),
            'avg_fare': hourly_groups['fare_amount'].mean(),
        })
        
        # 重置索引，使pickup_hour成为一列
        hourly_summary = hourly_summary.reset_index()
        
        return hourly_summary
    
    def process_monthly_file(self, file_path):
        """
        处理单个月份的数据文件
        
        参数:
            file_path: parquet文件路径
            
        返回:
            DataFrame: 该月的小时汇总数据
        """
        try:
            filename = os.path.basename(file_path)
            print(f"\n{'='*80}")
            print(f"正在处理: {filename}")
            print(f"{'='*80}")
            
            # 从文件名提取年月信息（如: yellow_tripdata_2025-01.parquet -> 2025-01）
            year_month = None
            try:
                year_month = filename.replace('yellow_tripdata_', '').replace('.parquet', '')
                print(f"文件对应月份: {year_month}")
            except:
                print("⚠ 无法从文件名提取年月信息，将跳过日期验证")
            
            # 1. 读取数据
            print("步骤 1/6: 读取数据...")
            df = pd.read_parquet(file_path)
            
            # 2. 数据清洗（包含日期验证）
            print("步骤 2/6: 数据清洗与日期验证...")
            df = self.clean_data(df, year_month=year_month)
            
            # 3. 删除不必要的列
            print("步骤 3/6: 删除不必要的列...")
            df = self.remove_unnecessary_columns(df)
            
            # 4. 添加CBD标签
            print("步骤 4/6: 添加CBD交互标签...")
            df = self.add_cbd_labels(df)
            
            # 5. 特征工程
            print("步骤 5/6: 特征工程...")
            df = self.feature_engineering(df)
            print(f"  特征工程后: {len(df):,} 条记录")
            
            # 6. 按小时汇总
            print("步骤 6/6: 按小时汇总...")
            hourly_summary = self.aggregate_by_hour(df)
            
            if hourly_summary is not None:
                print(f"  ✓ 生成了 {len(hourly_summary)} 个小时的汇总数据")
                return hourly_summary
            else:
                print("  ⚠ 未生成汇总数据")
                return None
                
        except Exception as e:
            print(f"  ❌ 处理文件出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_all_files(self):
        """
        处理data目录下的所有parquet文件
        """
        print("\n" + "="*80)
        print("NYC黄色出租车数据按小时汇总处理")
        print("="*80)
        
        # 获取所有parquet文件
        pattern = os.path.join(self.data_dir, "yellow_tripdata_*.parquet")
        files = sorted(glob.glob(pattern))
        
        if not files:
            print(f"\n❌ 错误: 在 {self.data_dir} 目录下未找到任何 yellow_tripdata_*.parquet 文件")
            print("请确保数据文件已放置在正确的目录中。")
            return
        
        print(f"\n找到 {len(files)} 个数据文件:")
        for i, file in enumerate(files, 1):
            print(f"  {i}. {os.path.basename(file)}")
        
        # 处理每个文件
        for i, file_path in enumerate(files, 1):
            print(f"\n进度: [{i}/{len(files)}]")
            
            hourly_data = self.process_monthly_file(file_path)
            
            if hourly_data is not None:
                self.hourly_stats.append(hourly_data)
        
        print("\n" + "="*80)
        print(f"✅ 处理完成！成功处理 {len(self.hourly_stats)} 个月份的数据")
        print("="*80)
    
    def save_results(self, output_file='hourly_taxi_summary.csv'):
        """
        保存所有月份的小时汇总数据
        
        参数:
            output_file: 输出文件名
        """
        if not self.hourly_stats:
            print("\n❌ 没有数据可以保存！")
            return None
        
        print("\n正在整合所有月份的小时数据...")
        
        # 合并所有月份的小时数据
        all_hourly_data = pd.concat(self.hourly_stats, ignore_index=True)
        
        # 按时间排序
        all_hourly_data = all_hourly_data.sort_values('pickup_hour').reset_index(drop=True)
        
        # 保存为CSV
        output_path = os.path.join(self.data_dir, output_file)
        all_hourly_data.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"\n✓ 小时汇总数据已保存到: {output_path}")
        print(f"  总记录数: {len(all_hourly_data):,} 行")
        print(f"  时间范围: {all_hourly_data['pickup_hour'].min()} 至 {all_hourly_data['pickup_hour'].max()}")
        print(f"  文件大小: {os.path.getsize(output_path) / 1024:.2f} KB")
        
        # 显示数据预览
        print("\n数据预览 (前5行):")
        print(all_hourly_data.head())
        
        # 显示数据统计摘要
        print("\n" + "="*80)
        print("数据统计摘要")
        print("="*80)
        print(f"总小时数: {len(all_hourly_data):,}")
        print(f"总行程数: {all_hourly_data['total_trips'].sum():,.0f}")
        print(f"平均每小时行程数: {all_hourly_data['total_trips'].mean():.0f}")
        print(f"平均行程距离: {all_hourly_data['avg_distance'].mean():.2f} 英里")
        print(f"平均行程时长: {all_hourly_data['avg_duration'].mean():.2f} 分钟")
        print(f"平均速度: {all_hourly_data['avg_speed'].mean():.2f} mph")
        print(f"平均小费率: {all_hourly_data['avg_tip_rate'].mean():.2%}")
        print(f"\nCBD交互平均比例:")
        print(f"  CBD内部 (inside): {all_hourly_data['cbd_inside_ratio'].mean():.2%}")
        print(f"  进入CBD (in): {all_hourly_data['cbd_in_ratio'].mean():.2%}")
        print(f"  离开CBD (out): {all_hourly_data['cbd_out_ratio'].mean():.2%}")
        print(f"  非CBD (non): {all_hourly_data['cbd_non_ratio'].mean():.2%}")
        print("="*80)
        
        return all_hourly_data


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
    processor = HourlyTaxiDataProcessor(data_dir)
    
    # 处理所有文件
    processor.process_all_files()
    
    # 保存结果
    if processor.hourly_stats:
        hourly_df = processor.save_results('hourly_taxi_summary.csv')
        print("\n✅ 所有任务完成！")
    else:
        print("\n⚠ 警告: 没有成功处理任何数据文件")


if __name__ == "__main__":
    main()

