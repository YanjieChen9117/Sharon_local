#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NYC出租车数据DiD (Difference-in-Differences) 分析

功能：
1. 使用pre-policy (2024-01-05 到 2024-08-31) 和 post-policy (2025-01-05 到 2025-08-31) 数据
2. 按holiday/not-holiday分组分析
3. 分析政策对以下指标的影响：
   - 平均出租车速度 (avg_speed)
   - 行程总量 (total_trips)
   - CBD内部行程量 (cbd_inside_ratio)
   - CBD neighbor行程量 (cbd_neighbor_inside_ratio)
   - CBD-out行程速度 (avg_speed_out_CBD)

DiD模型：
Y = β₀ + β₁*Treatment + β₂*Post + β₃*(Treatment*Post) + β₄*Controls + ε

其中：
- Treatment: 1 if CBD-related trips, 0 if non-CBD trips
- Post: 1 if post-policy period, 0 if pre-policy period
- Treatment*Post: 交互项，捕捉政策效应

作者: AI Assistant
日期: 2025-01-27
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.stattools import durbin_watson
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class DIDAnalyzer:
    """DiD分析器"""
    
    def __init__(self, data_path):
        """
        初始化分析器
        
        参数:
            data_path: 数据文件路径
        """
        self.data_path = data_path
        self.df = None
        self.pre_data = None
        self.post_data = None
        self.results = {}
        
    def load_and_prepare_data(self):
        """加载和准备数据"""
        print("Loading and preparing data...")
        
        # 读取数据
        self.df = pd.read_csv(self.data_path)
        self.df['pickup_hour'] = pd.to_datetime(self.df['pickup_hour'])
        
        # 定义时间范围
        pre_start = pd.Timestamp('2024-01-05')
        pre_end = pd.Timestamp('2024-08-31')
        post_start = pd.Timestamp('2025-01-05')
        post_end = pd.Timestamp('2025-08-31')
        
        # 筛选目标时间段
        self.pre_data = self.df[(self.df['pickup_hour'] >= pre_start) & 
                               (self.df['pickup_hour'] <= pre_end)].copy()
        self.post_data = self.df[(self.df['pickup_hour'] >= post_start) & 
                                (self.df['pickup_hour'] <= post_end)].copy()
        
        print(f"Pre-policy data: {len(self.pre_data)} hours")
        print(f"Post-policy data: {len(self.post_data)} hours")
        
        # 合并数据并创建DiD变量
        self.pre_data['post'] = 0
        self.post_data['post'] = 1
        
        self.did_data = pd.concat([self.pre_data, self.post_data], ignore_index=True)
        
        # 创建treatment变量 - 基于CBD相关行程比例
        # 如果CBD相关行程比例高，则认为是treatment group
        cbd_threshold = self.did_data['cbd_inside_ratio'].median()
        self.did_data['treatment'] = (self.did_data['cbd_inside_ratio'] > cbd_threshold).astype(int)
        
        # 创建交互项
        self.did_data['treatment_post'] = self.did_data['treatment'] * self.did_data['post']
        
        # 添加控制变量
        self.did_data['hour_of_day_sq'] = self.did_data['hour_of_day'] ** 2
        self.did_data['day_of_week'] = self.did_data['pickup_hour'].dt.dayofweek
        
        print(f"Treatment group size: {self.did_data['treatment'].sum()}")
        print(f"Control group size: {(self.did_data['treatment'] == 0).sum()}")
        
    def run_did_analysis(self, outcome_var, holiday_filter=None):
        """
        运行DiD分析
        
        参数:
            outcome_var: 结果变量名
            holiday_filter: 节假日筛选 ('holiday', 'not_holiday', None)
        """
        print(f"\nRunning DiD analysis for {outcome_var}...")
        
        # 数据筛选
        if holiday_filter == 'holiday':
            analysis_data = self.did_data[self.did_data['holiday'] == True].copy()
            print(f"Analyzing holiday data: {len(analysis_data)} observations")
        elif holiday_filter == 'not_holiday':
            analysis_data = self.did_data[self.did_data['holiday'] == False].copy()
            print(f"Analyzing non-holiday data: {len(analysis_data)} observations")
        else:
            analysis_data = self.did_data.copy()
            print(f"Analyzing all data: {len(analysis_data)} observations")
        
        # 移除缺失值
        analysis_data = analysis_data.dropna(subset=[outcome_var])
        
        # 准备回归变量
        X_vars = ['treatment', 'post', 'treatment_post']
        
        # 添加控制变量
        controls = ['hour_of_day', 'hour_of_day_sq', 'day_of_week', 'is_rain', 'is_snow']
        for control in controls:
            if control in analysis_data.columns:
                X_vars.append(control)
        
        X = analysis_data[X_vars].astype(float)
        y = analysis_data[outcome_var].astype(float)
        
        # 添加常数项
        X = sm.add_constant(X)
        
        # 运行回归
        model = sm.OLS(y, X).fit()
        
        # 存储结果
        result_key = f"{outcome_var}_{holiday_filter if holiday_filter else 'all'}"
        self.results[result_key] = {
            'model': model,
            'data': analysis_data,
            'outcome_var': outcome_var,
            'holiday_filter': holiday_filter,
            'n_obs': len(analysis_data),
            'treatment_effect': model.params.get('treatment_post', np.nan),
            'treatment_effect_pvalue': model.pvalues.get('treatment_post', np.nan)
        }
        
        return model
    
    def generate_summary_statistics(self):
        """生成描述性统计"""
        print("\nGenerating summary statistics...")
        
        # 按treatment和post分组统计
        summary_stats = []
        
        for holiday_type in [None, 'holiday', 'not_holiday']:
            if holiday_type == 'holiday':
                data = self.did_data[self.did_data['holiday'] == True]
                label = 'Holiday'
            elif holiday_type == 'not_holiday':
                data = self.did_data[self.did_data['holiday'] == False]
                label = 'Non-Holiday'
            else:
                data = self.did_data
                label = 'All'
            
            for treatment in [0, 1]:
                for post in [0, 1]:
                    subset = data[(data['treatment'] == treatment) & (data['post'] == post)]
                    
                    if len(subset) > 0:
                        stats_row = {
                            'Period': 'Pre-policy' if post == 0 else 'Post-policy',
                            'Group': 'Control' if treatment == 0 else 'Treatment',
                            'Holiday_Type': label,
                            'N': len(subset),
                            'avg_speed': subset['avg_speed'].mean(),
                            'total_trips': subset['total_trips'].mean(),
                            'cbd_inside_ratio': subset['cbd_inside_ratio'].mean(),
                            'cbd_neighbor_inside_ratio': subset['cbd_neighbor_inside_ratio'].mean(),
                            'avg_speed_out_CBD': subset['avg_speed_out_CBD'].mean()
                        }
                        summary_stats.append(stats_row)
        
        self.summary_stats = pd.DataFrame(summary_stats)
        return self.summary_stats
    
    def create_visualizations(self):
        """创建可视化图表"""
        print("\nCreating visualizations...")
        
        # 设置图表样式
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('DiD Analysis: Policy Impact on Taxi Operations', fontsize=16, fontweight='bold')
        
        # 定义结果变量
        outcome_vars = ['avg_speed', 'total_trips', 'cbd_inside_ratio', 
                       'cbd_neighbor_inside_ratio', 'avg_speed_out_CBD']
        
        for i, var in enumerate(outcome_vars):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            # 计算分组均值
            plot_data = []
            for holiday_type in ['holiday', 'not_holiday']:
                if holiday_type == 'holiday':
                    data = self.did_data[self.did_data['holiday'] == True]
                else:
                    data = self.did_data[self.did_data['holiday'] == False]
                
                for treatment in [0, 1]:
                    for post in [0, 1]:
                        subset = data[(data['treatment'] == treatment) & (data['post'] == post)]
                        if len(subset) > 0:
                            mean_val = subset[var].mean()
                            std_val = subset[var].std()
                            plot_data.append({
                                'Holiday_Type': 'Holiday' if holiday_type == 'holiday' else 'Non-Holiday',
                                'Group': 'Control' if treatment == 0 else 'Treatment',
                                'Period': 'Pre-policy' if post == 0 else 'Post-policy',
                                'Mean': mean_val,
                                'Std': std_val
                            })
            
            plot_df = pd.DataFrame(plot_data)
            
            # 创建分组柱状图
            x_pos = np.arange(2)  # 2个时间段
            width = 0.35
            
            for j, holiday_type in enumerate(['Holiday', 'Non-Holiday']):
                holiday_data = plot_df[plot_df['Holiday_Type'] == holiday_type]
                
                control_pre = holiday_data[(holiday_data['Group'] == 'Control') & 
                                         (holiday_data['Period'] == 'Pre-policy')]['Mean'].iloc[0]
                control_post = holiday_data[(holiday_data['Group'] == 'Control') & 
                                          (holiday_data['Period'] == 'Post-policy')]['Mean'].iloc[0]
                treatment_pre = holiday_data[(holiday_data['Group'] == 'Treatment') & 
                                           (holiday_data['Period'] == 'Pre-policy')]['Mean'].iloc[0]
                treatment_post = holiday_data[(holiday_data['Group'] == 'Treatment') & 
                                            (holiday_data['Period'] == 'Post-policy')]['Mean'].iloc[0]
                
                # 绘制柱状图
                ax.bar(x_pos + j*width, [control_pre, control_post], width, 
                      label=f'Control ({holiday_type})', alpha=0.7)
                ax.bar(x_pos + j*width + width, [treatment_pre, treatment_post], width, 
                      label=f'Treatment ({holiday_type})', alpha=0.7)
            
            ax.set_xlabel('Period')
            ax.set_ylabel(var.replace('_', ' ').title())
            ax.set_title(f'{var.replace("_", " ").title()}')
            ax.set_xticks(x_pos + width/2)
            ax.set_xticklabels(['Pre-policy', 'Post-policy'])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 移除最后一个空的子图
        axes[1, 2].remove()
        
        plt.tight_layout()
        plt.savefig('/Users/yanjiechen/Documents/Github/Sharon_local/figures/did_analysis_overview.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # 创建时间序列图
        self._create_time_series_plots()
    
    def _create_time_series_plots(self):
        """创建时间序列图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Time Series Analysis: Policy Impact Trends', fontsize=16, fontweight='bold')
        
        # 按周聚合数据
        weekly_data = self.did_data.copy()
        weekly_data['week'] = weekly_data['pickup_hour'].dt.to_period('W')
        
        outcome_vars = ['avg_speed', 'total_trips', 'cbd_inside_ratio', 'avg_speed_out_CBD']
        
        for i, var in enumerate(outcome_vars):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            # 按周和组计算均值
            weekly_means = weekly_data.groupby(['week', 'treatment', 'holiday'])[var].mean().reset_index()
            
            # 分别绘制holiday和non-holiday
            for holiday in [True, False]:
                holiday_data = weekly_means[weekly_means['holiday'] == holiday]
                
                for treatment in [0, 1]:
                    subset = holiday_data[holiday_data['treatment'] == treatment]
                    if len(subset) > 0:
                        label = f"{'Treatment' if treatment else 'Control'} ({'Holiday' if holiday else 'Non-Holiday'})"
                        ax.plot(subset['week'].astype(str), subset[var], 
                               marker='o', label=label, linewidth=2)
            
            # 添加政策实施线
            policy_date = pd.Timestamp('2025-01-05')
            policy_week = policy_date.to_period('W')
            # 找到政策实施周在x轴上的位置
            week_labels = weekly_means['week'].astype(str).unique()
            if str(policy_week) in week_labels:
                policy_idx = list(week_labels).index(str(policy_week))
                ax.axvline(x=policy_idx, color='red', linestyle='--', alpha=0.7, label='Policy Implementation')
            
            ax.set_xlabel('Week')
            ax.set_ylabel(var.replace('_', ' ').title())
            ax.set_title(f'{var.replace("_", " ").title()} - Weekly Trends')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('/Users/yanjiechen/Documents/Github/Sharon_local/figures/did_time_series.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self):
        """生成分析报告"""
        print("\nGenerating analysis report...")
        
        report = []
        report.append("# DiD Analysis Report: Policy Impact on NYC Taxi Operations")
        report.append("")
        report.append("## Executive Summary")
        report.append("")
        report.append("This report presents a Difference-in-Differences (DiD) analysis examining the impact of a policy implemented on January 5, 2025, on NYC taxi operations. The analysis compares pre-policy (January 5 - August 31, 2024) and post-policy (January 5 - August 31, 2025) periods, with separate analyses for holiday and non-holiday periods.")
        report.append("")
        
        # 模型说明
        report.append("## DiD Model Specification")
        report.append("")
        report.append("The DiD model is specified as:")
        report.append("")
        report.append("**Y = β₀ + β₁×Treatment + β₂×Post + β₃×(Treatment×Post) + β₄×Controls + ε**")
        report.append("")
        report.append("Where:")
        report.append("- **Y**: Outcome variable (speed, trip volume, CBD interactions)")
        report.append("- **Treatment**: 1 if high CBD interaction (treatment group), 0 otherwise (control group)")
        report.append("- **Post**: 1 if post-policy period (2025-01-05 to 2025-08-31), 0 if pre-policy period (2024-01-05 to 2024-08-31)")
        report.append("- **Treatment×Post**: Interaction term capturing the policy effect")
        report.append("- **Controls**: Hour of day, day of week, weather conditions")
        report.append("")
        
        # 数据描述
        report.append("## Data Description")
        report.append("")
        report.append(f"- **Total observations**: {len(self.did_data):,}")
        report.append(f"- **Pre-policy period**: {len(self.pre_data):,} hours")
        report.append(f"- **Post-policy period**: {len(self.post_data):,} hours")
        report.append(f"- **Treatment group**: {self.did_data['treatment'].sum():,} observations")
        report.append(f"- **Control group**: {(self.did_data['treatment'] == 0).sum():,} observations")
        report.append("")
        
        # 结果变量
        outcome_vars = ['avg_speed', 'total_trips', 'cbd_inside_ratio', 
                       'cbd_neighbor_inside_ratio', 'avg_speed_out_CBD']
        outcome_names = ['Average Taxi Speed', 'Total Trip Volume', 'CBD Internal Trips', 
                        'CBD Neighbor Trips', 'CBD Exit Speed']
        
        report.append("## Analysis Results")
        report.append("")
        
        for var, name in zip(outcome_vars, outcome_names):
            report.append(f"### {name}")
            report.append("")
            
            # 全样本结果
            if f"{var}_all" in self.results:
                model = self.results[f"{var}_all"]['model']
                treatment_effect = self.results[f"{var}_all"]['treatment_effect']
                p_value = self.results[f"{var}_all"]['treatment_effect_pvalue']
                
                report.append("**All Data:**")
                report.append(f"- Treatment Effect (β₃): {treatment_effect:.4f}")
                report.append(f"- P-value: {p_value:.4f}")
                report.append(f"- Significance: {'***' if p_value < 0.01 else '**' if p_value < 0.05 else '*' if p_value < 0.1 else 'Not significant'}")
                report.append("")
            
            # 节假日结果
            if f"{var}_holiday" in self.results:
                model = self.results[f"{var}_holiday"]['model']
                treatment_effect = self.results[f"{var}_holiday"]['treatment_effect']
                p_value = self.results[f"{var}_holiday"]['treatment_effect_pvalue']
                
                report.append("**Holiday Periods:**")
                report.append(f"- Treatment Effect (β₃): {treatment_effect:.4f}")
                report.append(f"- P-value: {p_value:.4f}")
                report.append(f"- Significance: {'***' if p_value < 0.01 else '**' if p_value < 0.05 else '*' if p_value < 0.1 else 'Not significant'}")
                report.append("")
            
            # 非节假日结果
            if f"{var}_not_holiday" in self.results:
                model = self.results[f"{var}_not_holiday"]['model']
                treatment_effect = self.results[f"{var}_not_holiday"]['treatment_effect']
                p_value = self.results[f"{var}_not_holiday"]['treatment_effect_pvalue']
                
                report.append("**Non-Holiday Periods:**")
                report.append(f"- Treatment Effect (β₃): {treatment_effect:.4f}")
                report.append(f"- P-value: {p_value:.4f}")
                report.append(f"- Significance: {'***' if p_value < 0.01 else '**' if p_value < 0.05 else '*' if p_value < 0.1 else 'Not significant'}")
                report.append("")
        
        # 模型诊断
        report.append("## Model Diagnostics")
        report.append("")
        report.append("### Regression Statistics")
        report.append("")
        
        # 选择一个代表性模型进行诊断
        if self.results:
            sample_key = list(self.results.keys())[0]
            model = self.results[sample_key]['model']
            
            report.append(f"**Sample Model ({sample_key}):**")
            report.append(f"- R-squared: {model.rsquared:.4f}")
            report.append(f"- Adjusted R-squared: {model.rsquared_adj:.4f}")
            report.append(f"- F-statistic: {model.fvalue:.4f}")
            report.append(f"- F-statistic p-value: {model.f_pvalue:.4f}")
            report.append("")
        
        # 结论
        report.append("## Conclusions")
        report.append("")
        report.append("The DiD analysis reveals the following key findings:")
        report.append("")
        report.append("1. **Policy Impact**: The treatment effect (β₃ coefficient) measures the differential impact of the policy on the treatment group relative to the control group.")
        report.append("")
        report.append("2. **Holiday vs Non-Holiday Effects**: Separate analyses for holiday and non-holiday periods allow for understanding how policy effects vary across different time contexts.")
        report.append("")
        report.append("3. **Statistical Significance**: Results are evaluated at 1%, 5%, and 10% significance levels.")
        report.append("")
        report.append("4. **Robustness**: The analysis includes multiple control variables to account for confounding factors such as time of day, weather conditions, and day of week effects.")
        report.append("")
        
        # 保存报告
        report_text = "\n".join(report)
        with open('/Users/yanjiechen/Documents/Github/Sharon_local/reports/did_analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("Report saved to: /Users/yanjiechen/Documents/Github/Sharon_local/reports/did_analysis_report.md")
        
        return report_text

def main():
    """主函数"""
    print("Starting DiD Analysis...")
    
    # 初始化分析器
    analyzer = DIDAnalyzer('/Users/yanjiechen/Documents/Github/Sharon_local/data/hourly_taxi_summary.csv')
    
    # 加载和准备数据
    analyzer.load_and_prepare_data()
    
    # 生成描述性统计
    summary_stats = analyzer.generate_summary_statistics()
    print("\nSummary Statistics:")
    print(summary_stats.to_string(index=False))
    
    # 定义分析变量
    outcome_vars = ['avg_speed', 'total_trips', 'cbd_inside_ratio', 
                   'cbd_neighbor_inside_ratio', 'avg_speed_out_CBD']
    
    # 运行DiD分析
    for var in outcome_vars:
        # 全样本分析
        analyzer.run_did_analysis(var, holiday_filter=None)
        
        # 节假日分析
        analyzer.run_did_analysis(var, holiday_filter='holiday')
        
        # 非节假日分析
        analyzer.run_did_analysis(var, holiday_filter='not_holiday')
    
    # 创建可视化
    analyzer.create_visualizations()
    
    # 生成报告
    report = analyzer.generate_report()
    
    print("\nDiD Analysis completed successfully!")
    print("Results saved to:")
    print("- Report: /Users/yanjiechen/Documents/Github/Sharon_local/reports/did_analysis_report.md")
    print("- Figures: /Users/yanjiechen/Documents/Github/Sharon_local/figures/")

if __name__ == "__main__":
    main()
