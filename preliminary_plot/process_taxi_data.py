#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理NYC黄色出租车数据的脚本
功能：
1. 按月份读取yellow tripdata数据
2. 提取关键特征并进行数据清洗
3. 将月度统计数据存储为CSV格式
4. 绘制月度总里程数据折线图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class TaxiDataProcessor:
    def __init__(self, data_dir):
        """
        初始化处理器
        
        参数:
            data_dir: parquet文件所在目录
        """
        self.data_dir = data_dir
        self.monthly_stats = []
        
    def read_monthly_data(self, file_path):
        """
        读取单个月份的数据并提取特征
        
        参数:
            file_path: parquet文件路径
            
        返回:
            DataFrame: 提取特征后的数据
        """
        try:
            print(f"\n正在读取: {os.path.basename(file_path)}")
            
            # 读取parquet文件
            df = pd.read_parquet(file_path)
            print(f"  原始数据: {len(df):,} 条记录")
            
            # 数据清洗 - 移除异常值
            df = df[
                (df['trip_distance'] > 0) &  # 距离必须大于0
                (df['trip_distance'] < 100) &  # 距离小于100英里
                (df['fare_amount'] > 0) &  # 费用必须大于0
                (df['total_amount'] > 0) &  # 总费用必须大于0
                (df['total_amount'] < 500) &  # 总费用小于500美元
                (df['passenger_count'] > 0) &  # 乘客数必须大于0
                (df['passenger_count'] <= 6)  # 乘客数合理范围
            ]
            print(f"  清洗后数据: {len(df):,} 条记录")
            
            # 提取关键特征
            features_df = pd.DataFrame({
                # 时间特征
                '上车时间': df['tpep_pickup_datetime'],
                '下车时间': df['tpep_dropoff_datetime'],
                
                # 行程特征
                '行程距离_英里': df['trip_distance'],
                '乘客数量': df['passenger_count'],
                
                # 位置特征
                '上车地点ID': df['PULocationID'],
                '下车地点ID': df['DOLocationID'],
                
                # 费用特征
                '基本费用': df['fare_amount'],
                '小费金额': df['tip_amount'],
                '通行费': df['tolls_amount'],
                '总金额': df['total_amount'],
                
                # 支付方式 (1=信用卡, 2=现金)
                '支付方式': df['payment_type']
            })
            
            # 计算派生特征
            # 行程时长（分钟）
            features_df['行程时长_分钟'] = (
                (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
            )
            
            # 平均速度（英里/小时）
            features_df['平均速度_英里每小时'] = (
                features_df['行程距离_英里'] / (features_df['行程时长_分钟'] / 60)
            )
            features_df['平均速度_英里每小时'] = features_df['平均速度_英里每小时'].replace([np.inf, -np.inf], np.nan)
            
            # 小费率
            features_df['小费率'] = (features_df['小费金额'] / features_df['基本费用']).replace([np.inf, -np.inf], np.nan)
            
            # 提取时间特征
            features_df['小时'] = df['tpep_pickup_datetime'].dt.hour
            features_df['星期几'] = df['tpep_pickup_datetime'].dt.dayofweek  # 0=周一, 6=周日
            features_df['是否周末'] = features_df['星期几'].isin([5, 6])
            
            return features_df
            
        except Exception as e:
            print(f"  ❌ 读取文件出错: {str(e)}")
            return None
    
    def calculate_monthly_summary(self, df, year_month):
        """
        计算月度汇总统计
        
        参数:
            df: 特征数据框
            year_month: 年月字符串 (如: "2023-01")
            
        返回:
            dict: 月度统计信息
        """
        if df is None or len(df) == 0:
            return None
        
        summary = {
            '年月': year_month,
            '总行程数': len(df),
            '总里程_英里': df['行程距离_英里'].sum(),
            '平均行程距离_英里': df['行程距离_英里'].mean(),
            '总收入_美元': df['总金额'].sum(),
            '平均票价_美元': df['总金额'].mean(),
            '平均乘客数': df['乘客数量'].mean(),
            '平均行程时长_分钟': df['行程时长_分钟'].mean(),
            '平均速度_英里每小时': df['平均速度_英里每小时'].mean(),
            '总小费_美元': df['小费金额'].sum(),
            '平均小费率': df['小费率'].mean(),
            '信用卡支付比例': (df['支付方式'] == 1).sum() / len(df),
            '现金支付比例': (df['支付方式'] == 2).sum() / len(df),
            '周末行程比例': df['是否周末'].sum() / len(df)
        }
        
        print(f"  ✓ 月度汇总: 总里程 {summary['总里程_英里']:,.0f} 英里, "
              f"总收入 ${summary['总收入_美元']:,.0f}")
        
        return summary
    
    def process_all_months(self, start_year=2023, end_year=2025):
        """
        处理所有月份的数据
        
        参数:
            start_year: 起始年份
            end_year: 结束年份
        """
        print("=" * 80)
        print("开始处理NYC黄色出租车数据")
        print("=" * 80)
        
        # 获取所有parquet文件
        pattern = os.path.join(self.data_dir, "yellow_tripdata_*.parquet")
        files = sorted(glob.glob(pattern))
        
        print(f"\n找到 {len(files)} 个数据文件")
        
        for file_path in files:
            # 从文件名提取年月信息
            filename = os.path.basename(file_path)
            # yellow_tripdata_2023-01.parquet
            try:
                year_month = filename.replace('yellow_tripdata_', '').replace('.parquet', '')
                year = int(year_month.split('-')[0])
                
                # 过滤年份范围
                if year < start_year or year > end_year:
                    continue
                
                # 读取和处理数据
                features_df = self.read_monthly_data(file_path)
                
                if features_df is not None:
                    # 计算月度汇总
                    summary = self.calculate_monthly_summary(features_df, year_month)
                    if summary:
                        self.monthly_stats.append(summary)
                        
            except Exception as e:
                print(f"  ❌ 处理文件失败: {str(e)}")
                continue
        
        print("\n" + "=" * 80)
        print(f"处理完成！共处理 {len(self.monthly_stats)} 个月份的数据")
        print("=" * 80)
    
    def save_to_csv(self, output_file='月度出租车数据统计.csv'):
        """
        将月度统计数据保存为CSV文件
        
        参数:
            output_file: 输出文件名
        """
        if not self.monthly_stats:
            print("没有数据可以保存！")
            return
        
        # 转换为DataFrame
        df = pd.DataFrame(self.monthly_stats)
        
        # 保存为CSV
        output_path = os.path.join(os.path.dirname(self.data_dir), output_file)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"\n✓ 月度统计数据已保存到: {output_path}")
        print(f"  文件大小: {os.path.getsize(output_path) / 1024:.2f} KB")
        
        # 显示数据预览
        print("\n数据预览 (前5行):")
        print(df.head())
        
        return df
    
    def plot_monthly_mileage(self, save_path='月度总里程折线图.png'):
        """
        绘制月度总里程折线图
        
        参数:
            save_path: 图片保存路径
        """
        if not self.monthly_stats:
            print("没有数据可以绘图！")
            return
        
        # 准备数据
        df = pd.DataFrame(self.monthly_stats)
        df['日期'] = pd.to_datetime(df['年月'])
        df = df.sort_values('日期')
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # 绘制折线图
        ax.plot(df['日期'], df['总里程_英里'], 
                marker='o', linewidth=2, markersize=8, 
                color='#2E86AB', label='月度总里程')
        
        # 添加数据标签
        for i, row in df.iterrows():
            if i % 3 == 0:  # 每3个月显示一次标签，避免拥挤
                ax.annotate(f'{row["总里程_英里"]/1e6:.1f}M',
                           xy=(row['日期'], row['总里程_英里']),
                           xytext=(0, 10),
                           textcoords='offset points',
                           ha='center',
                           fontsize=9,
                           color='#333333')
        
        # 设置标题和标签
        ax.set_title('NYC黄色出租车月度总里程趋势图', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('月份', fontsize=12, fontweight='bold')
        ax.set_ylabel('总里程 (英里)', fontsize=12, fontweight='bold')
        
        # 格式化y轴刻度
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
        
        # 添加网格
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 设置背景颜色
        ax.set_facecolor('#F9F9F9')
        fig.patch.set_facecolor('white')
        
        # 添加图例
        ax.legend(fontsize=11, loc='best')
        
        # 自动调整日期格式
        fig.autofmt_xdate()
        
        # 添加统计信息文本框
        stats_text = (
            f"数据时间段: {df['年月'].min()} 至 {df['年月'].max()}\n"
            f"总月份数: {len(df)} 个月\n"
            f"累计总里程: {df['总里程_英里'].sum()/1e6:.1f} 百万英里\n"
            f"平均月里程: {df['总里程_英里'].mean()/1e6:.1f} 百万英里\n"
            f"最高月里程: {df['总里程_英里'].max()/1e6:.1f} 百万英里 ({df.loc[df['总里程_英里'].idxmax(), '年月']})\n"
            f"最低月里程: {df['总里程_英里'].min()/1e6:.1f} 百万英里 ({df.loc[df['总里程_英里'].idxmin(), '年月']})"
        )
        
        ax.text(0.02, 0.98, stats_text,
               transform=ax.transAxes,
               fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 保存图片
        plt.tight_layout()
        save_full_path = os.path.join(os.path.dirname(self.data_dir), save_path)
        plt.savefig(save_full_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ 月度总里程折线图已保存到: {save_full_path}")
        
        # 显示图表
        plt.show()
        
        # 打印统计摘要
        print("\n" + "=" * 80)
        print("月度里程统计摘要")
        print("=" * 80)
        print(stats_text)
        print("=" * 80)


def main():
    """主函数"""
    # 设置数据目录
    data_dir = "/Users/yanjiechen/Documents/Github/Sharon_local/data"
    
    # 创建处理器实例
    processor = TaxiDataProcessor(data_dir)
    
    # 处理所有月份数据
    processor.process_all_months(start_year=2023, end_year=2025)
    
    # 保存月度统计到CSV
    monthly_df = processor.save_to_csv('月度出租车数据统计.csv')
    
    # 绘制月度总里程折线图
    processor.plot_monthly_mileage('月度总里程折线图.png')
    
    print("\n✅ 所有任务完成！")


if __name__ == "__main__":
    main()

