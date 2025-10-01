#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制额外的出租车数据分析图表
包括：平均速度、周末行程比例、平均行程距离
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def read_monthly_stats(csv_path):
    """读取月度统计数据"""
    df = pd.read_csv(csv_path)
    df['日期'] = pd.to_datetime(df['年月'])
    df = df.sort_values('日期')
    return df

def plot_average_speed(df, save_path='月度平均速度折线图.png'):
    """绘制平均速度折线图"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # 绘制折线图
    ax.plot(df['日期'], df['平均速度_英里每小时'], 
            marker='o', linewidth=2, markersize=8, 
            color='#E63946', label='平均速度')
    
    # 添加平均线
    avg_speed = df['平均速度_英里每小时'].mean()
    ax.axhline(y=avg_speed, color='gray', linestyle='--', 
               linewidth=1.5, alpha=0.7, label=f'总体平均: {avg_speed:.2f} mph')
    
    # 添加数据标签
    for i, row in df.iterrows():
        if i % 3 == 0:  # 每3个月显示一次
            ax.annotate(f'{row["平均速度_英里每小时"]:.1f}',
                       xy=(row['日期'], row['平均速度_英里每小时']),
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center',
                       fontsize=9,
                       color='#333333')
    
    # 设置标题和标签
    ax.set_title('NYC黄色出租车月度平均速度趋势图', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('月份', fontsize=12, fontweight='bold')
    ax.set_ylabel('平均速度 (英里/小时)', fontsize=12, fontweight='bold')
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 设置背景颜色
    ax.set_facecolor('#F9F9F9')
    fig.patch.set_facecolor('white')
    
    # 添加图例
    ax.legend(fontsize=11, loc='best')
    
    # 自动调整日期格式
    fig.autofmt_xdate()
    
    # 添加统计信息
    stats_text = (
        f"数据时间段: {df['年月'].min()} 至 {df['年月'].max()}\n"
        f"平均速度: {df['平均速度_英里每小时'].mean():.2f} mph\n"
        f"最高速度: {df['平均速度_英里每小时'].max():.2f} mph ({df.loc[df['平均速度_英里每小时'].idxmax(), '年月']})\n"
        f"最低速度: {df['平均速度_英里每小时'].min():.2f} mph ({df.loc[df['平均速度_英里每小时'].idxmin(), '年月']})\n"
        f"速度标准差: {df['平均速度_英里每小时'].std():.2f} mph"
    )
    
    ax.text(0.02, 0.98, stats_text,
           transform=ax.transAxes,
           fontsize=10,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 平均速度折线图已保存到: {save_path}")
    plt.close()

def plot_weekend_ratio(df, save_path='周末行程比例折线图.png'):
    """绘制周末行程比例折线图"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # 转换为百分比
    weekend_pct = df['周末行程比例'] * 100
    
    # 绘制折线图
    ax.plot(df['日期'], weekend_pct, 
            marker='s', linewidth=2, markersize=8, 
            color='#457B9D', label='周末行程比例')
    
    # 添加平均线
    avg_ratio = weekend_pct.mean()
    ax.axhline(y=avg_ratio, color='gray', linestyle='--', 
               linewidth=1.5, alpha=0.7, label=f'总体平均: {avg_ratio:.1f}%')
    
    # 添加数据标签
    for i, row in df.iterrows():
        if i % 3 == 0:  # 每3个月显示一次
            ax.annotate(f'{row["周末行程比例"]*100:.1f}%',
                       xy=(row['日期'], row['周末行程比例']*100),
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center',
                       fontsize=9,
                       color='#333333')
    
    # 设置标题和标签
    ax.set_title('NYC黄色出租车周末行程比例趋势图', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('月份', fontsize=12, fontweight='bold')
    ax.set_ylabel('周末行程比例 (%)', fontsize=12, fontweight='bold')
    
    # 设置y轴范围
    ax.set_ylim(20, 35)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 设置背景颜色
    ax.set_facecolor('#F9F9F9')
    fig.patch.set_facecolor('white')
    
    # 添加图例
    ax.legend(fontsize=11, loc='best')
    
    # 自动调整日期格式
    fig.autofmt_xdate()
    
    # 添加统计信息
    stats_text = (
        f"数据时间段: {df['年月'].min()} 至 {df['年月'].max()}\n"
        f"平均周末比例: {df['周末行程比例'].mean()*100:.2f}%\n"
        f"最高比例: {df['周末行程比例'].max()*100:.2f}% ({df.loc[df['周末行程比例'].idxmax(), '年月']})\n"
        f"最低比例: {df['周末行程比例'].min()*100:.2f}% ({df.loc[df['周末行程比例'].idxmin(), '年月']})\n"
        f"标准差: {df['周末行程比例'].std()*100:.2f}%"
    )
    
    ax.text(0.02, 0.98, stats_text,
           transform=ax.transAxes,
           fontsize=10,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 周末行程比例折线图已保存到: {save_path}")
    plt.close()

def plot_average_distance(df, save_path='月度平均行程距离折线图.png'):
    """绘制平均行程距离折线图"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # 绘制折线图
    ax.plot(df['日期'], df['平均行程距离_英里'], 
            marker='D', linewidth=2, markersize=8, 
            color='#2A9D8F', label='平均行程距离')
    
    # 添加平均线
    avg_distance = df['平均行程距离_英里'].mean()
    ax.axhline(y=avg_distance, color='gray', linestyle='--', 
               linewidth=1.5, alpha=0.7, label=f'总体平均: {avg_distance:.2f} 英里')
    
    # 添加数据标签
    for i, row in df.iterrows():
        if i % 3 == 0:  # 每3个月显示一次
            ax.annotate(f'{row["平均行程距离_英里"]:.2f}',
                       xy=(row['日期'], row['平均行程距离_英里']),
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center',
                       fontsize=9,
                       color='#333333')
    
    # 设置标题和标签
    ax.set_title('NYC黄色出租车月度平均行程距离趋势图', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('月份', fontsize=12, fontweight='bold')
    ax.set_ylabel('平均行程距离 (英里)', fontsize=12, fontweight='bold')
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 设置背景颜色
    ax.set_facecolor('#F9F9F9')
    fig.patch.set_facecolor('white')
    
    # 添加图例
    ax.legend(fontsize=11, loc='best')
    
    # 自动调整日期格式
    fig.autofmt_xdate()
    
    # 添加统计信息
    stats_text = (
        f"数据时间段: {df['年月'].min()} 至 {df['年月'].max()}\n"
        f"平均距离: {df['平均行程距离_英里'].mean():.3f} 英里\n"
        f"最长平均: {df['平均行程距离_英里'].max():.3f} 英里 ({df.loc[df['平均行程距离_英里'].idxmax(), '年月']})\n"
        f"最短平均: {df['平均行程距离_英里'].min():.3f} 英里 ({df.loc[df['平均行程距离_英里'].idxmin(), '年月']})\n"
        f"标准差: {df['平均行程距离_英里'].std():.3f} 英里"
    )
    
    ax.text(0.02, 0.98, stats_text,
           transform=ax.transAxes,
           fontsize=10,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 平均行程距离折线图已保存到: {save_path}")
    plt.close()

def plot_combined_view(df, save_path='综合趋势对比图.png'):
    """绘制综合对比图 - 三个指标在一张图上"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 1. 平均速度
    ax1 = axes[0]
    ax1.plot(df['日期'], df['平均速度_英里每小时'], 
            marker='o', linewidth=2, markersize=6, color='#E63946')
    ax1.set_title('平均速度趋势', fontsize=14, fontweight='bold', pad=10)
    ax1.set_ylabel('速度 (mph)', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_facecolor('#F9F9F9')
    avg_speed = df['平均速度_英里每小时'].mean()
    ax1.axhline(y=avg_speed, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.text(0.98, 0.95, f'平均: {avg_speed:.2f} mph', 
            transform=ax1.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. 周末行程比例
    ax2 = axes[1]
    weekend_pct = df['周末行程比例'] * 100
    ax2.plot(df['日期'], weekend_pct, 
            marker='s', linewidth=2, markersize=6, color='#457B9D')
    ax2.set_title('周末行程比例趋势', fontsize=14, fontweight='bold', pad=10)
    ax2.set_ylabel('比例 (%)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_facecolor('#F9F9F9')
    avg_ratio = weekend_pct.mean()
    ax2.axhline(y=avg_ratio, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.text(0.98, 0.95, f'平均: {avg_ratio:.1f}%', 
            transform=ax2.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. 平均行程距离
    ax3 = axes[2]
    ax3.plot(df['日期'], df['平均行程距离_英里'], 
            marker='D', linewidth=2, markersize=6, color='#2A9D8F')
    ax3.set_title('平均行程距离趋势', fontsize=14, fontweight='bold', pad=10)
    ax3.set_xlabel('月份', fontsize=11, fontweight='bold')
    ax3.set_ylabel('距离 (英里)', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_facecolor('#F9F9F9')
    avg_distance = df['平均行程距离_英里'].mean()
    ax3.axhline(y=avg_distance, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax3.text(0.98, 0.95, f'平均: {avg_distance:.2f} 英里', 
            transform=ax3.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 调整日期显示
    for ax in axes:
        fig.autofmt_xdate()
    
    # 总标题
    fig.suptitle('NYC黄色出租车关键指标综合趋势分析', 
                fontsize=18, fontweight='bold', y=0.995)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 综合趋势对比图已保存到: {save_path}")
    plt.close()

def main():
    """主函数"""
    print("=" * 80)
    print("开始绘制额外的数据分析图表")
    print("=" * 80)
    
    # 读取数据
    csv_path = '/Users/yanjiechen/Documents/Github/Sharon_local/preliminary_plot/月度出租车数据统计.csv'
    print(f"\n读取数据: {csv_path}")
    df = read_monthly_stats(csv_path)
    print(f"数据行数: {len(df)}")
    
    # 输出目录
    output_dir = '/Users/yanjiechen/Documents/Github/Sharon_local/preliminary_plot'
    
    print("\n" + "-" * 80)
    print("开始绘制图表...")
    print("-" * 80)
    
    # 绘制各种图表
    plot_average_speed(df, os.path.join(output_dir, '月度平均速度折线图.png'))
    plot_weekend_ratio(df, os.path.join(output_dir, '周末行程比例折线图.png'))
    plot_average_distance(df, os.path.join(output_dir, '月度平均行程距离折线图.png'))
    plot_combined_view(df, os.path.join(output_dir, '综合趋势对比图.png'))
    
    print("\n" + "=" * 80)
    print("✅ 所有图表绘制完成！")
    print("=" * 80)
    
    # 打印统计摘要
    print("\n数据摘要:")
    print(f"  时间范围: {df['年月'].min()} 至 {df['年月'].max()}")
    print(f"  平均速度: {df['平均速度_英里每小时'].mean():.2f} mph (最高: {df['平均速度_英里每小时'].max():.2f}, 最低: {df['平均速度_英里每小时'].min():.2f})")
    print(f"  周末行程比例: {df['周末行程比例'].mean()*100:.2f}% (最高: {df['周末行程比例'].max()*100:.2f}%, 最低: {df['周末行程比例'].min()*100:.2f}%)")
    print(f"  平均行程距离: {df['平均行程距离_英里'].mean():.3f} 英里 (最高: {df['平均行程距离_英里'].max():.3f}, 最低: {df['平均行程距离_英里'].min():.3f})")
    print("=" * 80)

if __name__ == "__main__":
    main()

