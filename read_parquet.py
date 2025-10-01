#!/usr/bin/env python3
"""
读取Parquet文件并显示前几行数据的脚本
"""

import pandas as pd
import os

def read_parquet_sample(file_path, n_rows=10):
    """
    读取parquet文件并显示前n行
    
    参数:
        file_path: parquet文件路径
        n_rows: 要显示的行数，默认10行
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"错误: 文件 '{file_path}' 不存在")
            return
        
        # 获取文件大小
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"文件大小: {file_size_mb:.2f} MB")
        print("-" * 80)
        
        # 读取parquet文件（pandas会智能处理，不会一次性加载所有数据到内存）
        print("正在读取parquet文件...")
        df = pd.read_parquet(file_path)
        
        # 显示基本信息
        print(f"\n数据集形状: {df.shape[0]} 行 × {df.shape[1]} 列")
        print("-" * 80)
        
        # 显示列名和数据类型
        print("\n列信息:")
        print(df.dtypes)
        print("-" * 80)
        
        # 显示前n行
        print(f"\n前 {n_rows} 行数据:")
        print(df.head(n_rows))
        print("-" * 80)
        
        # 显示基本统计信息（仅数值列）
        print("\n数值列的基本统计信息:")
        print(df.describe())
        print("-" * 80)
        
        # 显示内存使用情况
        print(f"\n内存使用: {df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")
        
    except Exception as e:
        print(f"读取文件时发生错误: {str(e)}")

if __name__ == "__main__":
    # 指定parquet文件路径
    parquet_file = "/Users/yanjiechen/Documents/Github/Sharon_local/data/yellow_tripdata_2024-01.parquet"
    
    # 读取并显示前10行
    read_parquet_sample(parquet_file, n_rows=10)

