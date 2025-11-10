#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NTA出租车数据DiD (Difference-in-Differences) 分析
专门分析政策对MN0502 (Midtown-Times Square) trip count的影响

功能：
1. 使用pre-policy (2024-01到2024-07) 和 post-policy (2025-01到2025-07) 数据
2. Treatment group: PUNTA = MN0502
3. Control group: 其他所有PUNTA区域
4. 分析政策对total_trips的影响
5. 控制变量: hour_of_day, day_of_week, is_rain, is_snow, temperature, holiday

DiD模型：
Y = β₀ + β₁*Treatment + β₂*Post + β₃*(Treatment*Post) + β₄*Controls + ε

其中：
- Treatment: 1 if PUNTA == MN0502, 0 otherwise
- Post: 1 if post-policy period (2025-01到2025-07), 0 if pre-policy period (2024-01到2024-07)
- Treatment*Post: 交互项，捕捉政策效应

作者: Yanjie Chen
日期: 2025-01-XX
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class NTADIDAnalyzer:
    """NTA DiD分析器 - 专门分析MN0502的政策影响"""
    
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
        self.did_data = None
        self.results = {}
        
    def load_and_prepare_data(self):
        """加载和准备数据"""
        print("正在加载和准备数据...")
        
        # 读取数据
        self.df = pd.read_csv(self.data_path)
        self.df['pickup_hour'] = pd.to_datetime(self.df['pickup_hour'])
        
        # 筛选pre-policy数据 (2024-01到2024-07)
        self.pre_data = self.df[(self.df['year'] == 2024) & 
                               (self.df['month'] >= 1) & 
                               (self.df['month'] <= 7)].copy()
        
        # 筛选post-policy数据 (2025-01到2025-07)
        self.post_data = self.df[(self.df['year'] == 2025) & 
                                (self.df['month'] >= 1) & 
                                (self.df['month'] <= 7)].copy()
        
        print(f"Pre-policy数据: {len(self.pre_data)} 条记录")
        print(f"Post-policy数据: {len(self.post_data)} 条记录")
        
        # 检查MN0502数据
        mn0502_pre = self.pre_data[self.pre_data['PUNTA'] == 'MN0502']
        mn0502_post = self.post_data[self.post_data['PUNTA'] == 'MN0502']
        print(f"MN0502 pre-policy数据: {len(mn0502_pre)} 条记录")
        print(f"MN0502 post-policy数据: {len(mn0502_post)} 条记录")
        
        # 合并数据并创建DiD变量
        self.pre_data['post'] = 0
        self.post_data['post'] = 1
        
        self.did_data = pd.concat([self.pre_data, self.post_data], ignore_index=True)
        
        # 创建treatment变量: 1 if PUNTA == MN0502, 0 otherwise
        self.did_data['treatment'] = (self.did_data['PUNTA'] == 'MN0502').astype(int)
        
        # 创建交互项
        self.did_data['treatment_post'] = self.did_data['treatment'] * self.did_data['post']
        
        # 确保day_of_week变量存在（从pickup_hour计算或使用现有列）
        if 'day_of_week' not in self.did_data.columns:
            self.did_data['day_of_week'] = self.did_data['pickup_hour'].dt.dayofweek
        else:
            # 确保day_of_week是数值型
            self.did_data['day_of_week'] = pd.to_numeric(self.did_data['day_of_week'], errors='coerce')
        
        # 确保hour_of_day是数值型
        if 'hour_of_day' in self.did_data.columns:
            self.did_data['hour_of_day'] = pd.to_numeric(self.did_data['hour_of_day'], errors='coerce')
        else:
            self.did_data['hour_of_day'] = self.did_data['pickup_hour'].dt.hour
        
        # 处理布尔变量
        bool_vars = ['is_rain', 'is_snow', 'holiday']
        for var in bool_vars:
            if var in self.did_data.columns:
                self.did_data[var] = self.did_data[var].astype(int)
        
        # 处理temperature变量
        if 'temperature' in self.did_data.columns:
            self.did_data['temperature'] = pd.to_numeric(self.did_data['temperature'], errors='coerce')
        
        print(f"Treatment group (MN0502): {self.did_data['treatment'].sum()} 条记录")
        print(f"Control group (其他PUNTA): {(self.did_data['treatment'] == 0).sum()} 条记录")
        
        # 检查数据质量
        print("\n数据质量检查:")
        print(f"缺失值统计:")
        print(self.did_data[['total_trips', 'hour_of_day', 'day_of_week', 'is_rain', 
                             'is_snow', 'temperature', 'holiday']].isnull().sum())
        
    def run_did_analysis(self):
        """
        运行DiD分析
        
        结果变量: total_trips
        控制变量: hour_of_day, day_of_week, is_rain, is_snow, temperature, holiday
        """
        print("\n正在运行DiD分析...")
        
        # 准备分析数据
        analysis_data = self.did_data.copy()
        
        # 移除缺失值
        required_vars = ['total_trips', 'treatment', 'post', 'treatment_post',
                        'hour_of_day', 'day_of_week', 'is_rain', 'is_snow', 
                        'temperature', 'holiday']
        analysis_data = analysis_data.dropna(subset=required_vars)
        
        print(f"分析数据: {len(analysis_data)} 条记录")
        
        # 准备回归变量
        X_vars = ['treatment', 'post', 'treatment_post']
        controls = ['hour_of_day', 'day_of_week', 'is_rain', 'is_snow', 'temperature', 'holiday']
        
        for control in controls:
            if control in analysis_data.columns:
                X_vars.append(control)
        
        X = analysis_data[X_vars].astype(float)
        y = analysis_data['total_trips'].astype(float)
        
        # 添加常数项
        X = sm.add_constant(X)
        
        # 运行OLS回归（普通标准误）
        model = sm.OLS(y, X).fit()
        
        # 运行OLS回归（稳健标准误，处理异方差问题）
        model_robust = sm.OLS(y, X).fit(cov_type='HC3')  # HC3是常用的稳健标准误类型
        
        # 运行模型诊断
        diagnostics = self._run_model_diagnostics(model, X, y)
        
        # 存储结果（使用稳健标准误的结果）
        self.results = {
            'model': model,  # 普通标准误模型
            'model_robust': model_robust,  # 稳健标准误模型
            'data': analysis_data,
            'n_obs': len(analysis_data),
            'diagnostics': diagnostics,
            'treatment_effect': model_robust.params.get('treatment_post', np.nan),
            'treatment_effect_pvalue': model_robust.pvalues.get('treatment_post', np.nan),
            'treatment_effect_ci': model_robust.conf_int().loc['treatment_post'] if 'treatment_post' in model_robust.params else (np.nan, np.nan),
            # 同时保存普通标准误的结果用于对比
            'treatment_effect_ols': model.params.get('treatment_post', np.nan),
            'treatment_effect_pvalue_ols': model.pvalues.get('treatment_post', np.nan),
            'treatment_effect_ci_ols': model.conf_int().loc['treatment_post'] if 'treatment_post' in model.params else (np.nan, np.nan)
        }
        
        # 打印回归结果（稳健标准误）
        print("\n" + "="*80)
        print("DiD回归结果 (使用稳健标准误处理异方差)")
        print("="*80)
        print(model_robust.summary())
        
        # 打印关键结果
        print("\n" + "="*80)
        print("关键结果")
        print("="*80)
        print(f"政策效应 (Treatment × Post): {self.results['treatment_effect']:.4f}")
        print(f"P值: {self.results['treatment_effect_pvalue']:.4f}")
        print(f"95%置信区间: [{self.results['treatment_effect_ci'][0]:.4f}, {self.results['treatment_effect_ci'][1]:.4f}]")
        
        if self.results['treatment_effect_pvalue'] < 0.01:
            significance = "*** (p < 0.01)"
        elif self.results['treatment_effect_pvalue'] < 0.05:
            significance = "** (p < 0.05)"
        elif self.results['treatment_effect_pvalue'] < 0.1:
            significance = "* (p < 0.1)"
        else:
            significance = "不显著"
        print(f"显著性: {significance}")
        
        # 对比普通标准误和稳健标准误
        print("\n" + "="*80)
        print("对比: 普通标准误 vs 稳健标准误")
        print("="*80)
        print(f"普通标准误:")
        print(f"  政策效应: {self.results['treatment_effect_ols']:.4f}")
        print(f"  P值: {self.results['treatment_effect_pvalue_ols']:.4f}")
        print(f"稳健标准误:")
        print(f"  政策效应: {self.results['treatment_effect']:.4f}")
        print(f"  P值: {self.results['treatment_effect_pvalue']:.4f}")
        print("\n注意: 由于存在异方差问题，建议使用稳健标准误的结果。")
        
        return model_robust
    
    def _run_model_diagnostics(self, model, X, y):
        """运行模型诊断"""
        diagnostics = {}
        
        # 基本统计量
        diagnostics['r_squared'] = model.rsquared
        diagnostics['adj_r_squared'] = model.rsquared_adj
        diagnostics['f_statistic'] = model.fvalue
        diagnostics['f_pvalue'] = model.f_pvalue
        diagnostics['aic'] = model.aic
        diagnostics['bic'] = model.bic
        diagnostics['n_obs'] = len(y)
        
        # 异方差检验 (White test)
        try:
            white_test = het_white(model.resid, model.model.exog)
            diagnostics['white_test_stat'] = white_test[0]
            diagnostics['white_test_pvalue'] = white_test[1]
        except Exception as e:
            print(f"White test failed: {e}")
            diagnostics['white_test_stat'] = np.nan
            diagnostics['white_test_pvalue'] = np.nan
        
        # 自相关检验 (Durbin-Watson)
        try:
            dw_stat = durbin_watson(model.resid)
            diagnostics['durbin_watson'] = dw_stat
        except Exception as e:
            print(f"Durbin-Watson test failed: {e}")
            diagnostics['durbin_watson'] = np.nan
        
        # VIF检验（多重共线性）
        try:
            vif_data = pd.DataFrame()
            vif_data["Variable"] = X.columns
            vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            diagnostics['max_vif'] = vif_data['VIF'].max()
            diagnostics['vif_data'] = vif_data
        except Exception as e:
            print(f"VIF test failed: {e}")
            diagnostics['max_vif'] = np.nan
            diagnostics['vif_data'] = None
        
        return diagnostics
    
    def generate_summary_statistics(self):
        """生成描述性统计"""
        print("\n正在生成描述性统计...")
        
        summary_stats = []
        
        for treatment in [0, 1]:
            for post in [0, 1]:
                subset = self.did_data[(self.did_data['treatment'] == treatment) & 
                                      (self.did_data['post'] == post)]
                
                if len(subset) > 0:
                    stats_row = {
                        'Period': 'Pre-policy' if post == 0 else 'Post-policy',
                        'Group': 'Control (其他PUNTA)' if treatment == 0 else 'Treatment (MN0502)',
                        'N': len(subset),
                        'Mean_Trips': subset['total_trips'].mean(),
                        'Std_Trips': subset['total_trips'].std(),
                        'Median_Trips': subset['total_trips'].median(),
                        'Min_Trips': subset['total_trips'].min(),
                        'Max_Trips': subset['total_trips'].max()
                    }
                    summary_stats.append(stats_row)
        
        self.summary_stats = pd.DataFrame(summary_stats)
        
        print("\n描述性统计:")
        print(self.summary_stats.to_string(index=False))
        
        return self.summary_stats
    
    def create_visualizations(self):
        """创建可视化图表"""
        print("\n正在创建可视化图表...")
        
        # 1. 分组均值对比图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('DiD Analysis: Policy Impact on MN0502 Trip Count', fontsize=16, fontweight='bold')
        
        # 1.1 分组均值柱状图
        ax1 = axes[0, 0]
        plot_data = []
        for treatment in [0, 1]:
            for post in [0, 1]:
                subset = self.did_data[(self.did_data['treatment'] == treatment) & 
                                      (self.did_data['post'] == post)]
                if len(subset) > 0:
                    mean_val = subset['total_trips'].mean()
                    std_val = subset['total_trips'].std()
                    plot_data.append({
                        'Group': 'Control' if treatment == 0 else 'Treatment (MN0502)',
                        'Period': 'Pre-policy' if post == 0 else 'Post-policy',
                        'Mean': mean_val,
                        'Std': std_val
                    })
        
        plot_df = pd.DataFrame(plot_data)
        x_pos = np.arange(2)
        width = 0.35
        
        control_pre = plot_df[(plot_df['Group'] == 'Control') & 
                            (plot_df['Period'] == 'Pre-policy')]['Mean'].iloc[0]
        control_post = plot_df[(plot_df['Group'] == 'Control') & 
                             (plot_df['Period'] == 'Post-policy')]['Mean'].iloc[0]
        treatment_pre = plot_df[(plot_df['Group'] == 'Treatment (MN0502)') & 
                              (plot_df['Period'] == 'Pre-policy')]['Mean'].iloc[0]
        treatment_post = plot_df[(plot_df['Group'] == 'Treatment (MN0502)') & 
                               (plot_df['Period'] == 'Post-policy')]['Mean'].iloc[0]
        
        ax1.bar(x_pos, [control_pre, control_post], width, 
               label='Control (Other PUNTA)', alpha=0.7, color='steelblue')
        ax1.bar(x_pos + width, [treatment_pre, treatment_post], width, 
               label='Treatment (MN0502)', alpha=0.7, color='coral')
        
        ax1.set_xlabel('Period', fontsize=12)
        ax1.set_ylabel('Average Trip Count', fontsize=12)
        ax1.set_title('Group Mean Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos + width/2)
        ax1.set_xticklabels(['Pre-policy', 'Post-policy'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 添加DiD效应标注
        did_effect = (treatment_post - treatment_pre) - (control_post - control_pre)
        ax1.text(0.5, 0.95, f'DiD Effect: {did_effect:.2f}', 
                transform=ax1.transAxes, fontsize=11, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 1.2 时间序列图（按天聚合）
        ax2 = axes[0, 1]
        daily_data = self.did_data.copy()
        daily_data['date'] = daily_data['pickup_hour'].dt.date
        daily_means = daily_data.groupby(['date', 'treatment', 'post'])['total_trips'].mean().reset_index()
        
        for treatment in [0, 1]:
            subset = daily_means[daily_means['treatment'] == treatment]
            label = 'Control (Other PUNTA)' if treatment == 0 else 'Treatment (MN0502)'
            color = 'steelblue' if treatment == 0 else 'coral'
            ax2.plot(subset['date'], subset['total_trips'], 
                    marker='o', label=label, linewidth=2, color=color, alpha=0.7, markersize=3)
        
        # 添加政策实施线
        policy_date = pd.Timestamp('2025-01-01').date()
        ax2.axvline(x=policy_date, color='red', linestyle='--', alpha=0.7, label='Policy Implementation')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Average Trip Count', fontsize=12)
        ax2.set_title('Time Series Trend', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # 1.3 残差图
        ax3 = axes[1, 0]
        if self.results:
            model = self.results['model']
            ax3.scatter(model.fittedvalues, model.resid, alpha=0.6, s=10)
            ax3.axhline(y=0, color='red', linestyle='--')
            ax3.set_xlabel('Fitted Values', fontsize=12)
            ax3.set_ylabel('Residuals', fontsize=12)
            ax3.set_title('Residuals vs Fitted Values', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        # 1.4 Q-Q图
        ax4 = axes[1, 1]
        if self.results:
            model = self.results['model']
            from scipy import stats
            stats.probplot(model.resid, dist="norm", plot=ax4)
            ax4.set_title('Q-Q Plot (Residual Normality Test)', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/yanjiechen/Documents/Github/Sharon_local/figures/nta_did_analysis_mn0502.png', 
                   dpi=300, bbox_inches='tight')
        print("图表已保存到: figures/nta_did_analysis_mn0502.png")
        plt.show()
        
        # 2. 创建更详细的时间序列图（按周聚合）
        self._create_weekly_timeseries_plot()
    
    def _create_weekly_timeseries_plot(self):
        """创建按周聚合的时间序列图"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        weekly_data = self.did_data.copy()
        weekly_data['week'] = weekly_data['pickup_hour'].dt.to_period('W')
        weekly_means = weekly_data.groupby(['week', 'treatment'])['total_trips'].mean().reset_index()
        
        for treatment in [0, 1]:
            subset = weekly_means[weekly_means['treatment'] == treatment]
            label = 'Control (Other PUNTA)' if treatment == 0 else 'Treatment (MN0502)'
            color = 'steelblue' if treatment == 0 else 'coral'
            ax.plot(subset['week'].astype(str), subset['total_trips'], 
                   marker='o', label=label, linewidth=2, color=color, alpha=0.7, markersize=4)
        
        # 添加政策实施线
        policy_date = pd.Timestamp('2025-01-01')
        policy_week = policy_date.to_period('W')
        week_labels = weekly_means['week'].astype(str).unique()
        if str(policy_week) in week_labels:
            policy_idx = list(week_labels).index(str(policy_week))
            ax.axvline(x=policy_idx, color='red', linestyle='--', alpha=0.7, 
                      linewidth=2, label='Policy Implementation (2025-01-01)')
        
        ax.set_xlabel('Week', fontsize=12)
        ax.set_ylabel('Average Trip Count', fontsize=12)
        ax.set_title('DiD Analysis: MN0502 Trip Count Weekly Trends', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('/Users/yanjiechen/Documents/Github/Sharon_local/figures/nta_did_weekly_timeseries_mn0502.png', 
                   dpi=300, bbox_inches='tight')
        print("周度时间序列图已保存到: figures/nta_did_weekly_timeseries_mn0502.png")
        plt.show()
    
    def generate_report(self):
        """生成分析报告"""
        print("\n正在生成分析报告...")
        
        report = []
        report.append("# DiD分析报告: 政策对MN0502 (Midtown-Times Square) Trip Count的影响")
        report.append("")
        report.append("## 执行摘要")
        report.append("")
        report.append("本报告使用双重差分（Difference-in-Differences, DiD）方法分析政策对MN0502区域（Midtown-Times Square）出租车行程数量的影响。分析比较了政策实施前（2024年1月至7月）和政策实施后（2025年1月至7月）两个时期的数据。")
        report.append("")
        
        # 模型说明
        report.append("## DiD模型设定")
        report.append("")
        report.append("DiD模型设定为:")
        report.append("")
        report.append("**Y = β₀ + β₁×Treatment + β₂×Post + β₃×(Treatment×Post) + β₄×Controls + ε**")
        report.append("")
        report.append("其中:")
        report.append("- **Y**: 结果变量（total_trips，每小时行程总数）")
        report.append("- **Treatment**: 1 if PUNTA == MN0502 (处理组), 0 otherwise (对照组)")
        report.append("- **Post**: 1 if post-policy period (2025-01至2025-07), 0 if pre-policy period (2024-01至2024-07)")
        report.append("- **Treatment×Post**: 交互项，捕捉政策效应（β₃系数）")
        report.append("- **Controls**: hour_of_day, day_of_week, is_rain, is_snow, temperature, holiday")
        report.append("")
        
        # 数据描述
        report.append("## 数据描述")
        report.append("")
        report.append(f"- **总观测数**: {len(self.did_data):,}")
        report.append(f"- **Pre-policy时期**: {len(self.pre_data):,} 条记录 (2024-01至2024-07)")
        report.append(f"- **Post-policy时期**: {len(self.post_data):,} 条记录 (2025-01至2025-07)")
        report.append(f"- **处理组 (MN0502)**: {self.did_data['treatment'].sum():,} 条记录")
        report.append(f"- **对照组 (其他PUNTA)**: {(self.did_data['treatment'] == 0).sum():,} 条记录")
        report.append("")
        
        # 描述性统计
        if hasattr(self, 'summary_stats'):
            report.append("## 描述性统计")
            report.append("")
            report.append("| 时期 | 组别 | 观测数 | 平均Trip Count | 标准差 | 中位数 | 最小值 | 最大值 |")
            report.append("|------|------|--------|----------------|--------|--------|--------|--------|")
            for _, row in self.summary_stats.iterrows():
                report.append(f"| {row['Period']} | {row['Group']} | {row['N']} | {row['Mean_Trips']:.2f} | {row['Std_Trips']:.2f} | {row['Median_Trips']:.2f} | {row['Min_Trips']:.2f} | {row['Max_Trips']:.2f} |")
            report.append("")
        
        # 回归结果
        if self.results:
            report.append("## 回归结果")
            report.append("")
            model = self.results['model']
            model_robust = self.results.get('model_robust', model)
            diagnostics = self.results['diagnostics']
            
            report.append("### 关键系数 (使用稳健标准误)")
            report.append("")
            report.append("**注意**: 由于模型诊断显示存在异方差问题，我们使用稳健标准误（HC3）来获得更可靠的标准误和置信区间。")
            report.append("")
            report.append(f"- **政策效应 (β₃, Treatment × Post)**: {self.results['treatment_effect']:.4f}")
            report.append(f"- **P值**: {self.results['treatment_effect_pvalue']:.4f}")
            report.append(f"- **95%置信区间**: [{self.results['treatment_effect_ci'][0]:.4f}, {self.results['treatment_effect_ci'][1]:.4f}]")
            
            if self.results['treatment_effect_pvalue'] < 0.01:
                significance = "*** (p < 0.01, 高度显著)"
            elif self.results['treatment_effect_pvalue'] < 0.05:
                significance = "** (p < 0.05, 显著)"
            elif self.results['treatment_effect_pvalue'] < 0.1:
                significance = "* (p < 0.1, 边际显著)"
            else:
                significance = "不显著"
            report.append(f"- **显著性**: {significance}")
            report.append("")
            
            # 对比普通标准误和稳健标准误
            if 'treatment_effect_ols' in self.results:
                report.append("### 标准误对比")
                report.append("")
                report.append("| 标准误类型 | 政策效应 | P值 | 95%置信区间 |")
                report.append("|-----------|---------|-----|------------|")
                report.append(f"| 普通标准误 | {self.results['treatment_effect_ols']:.4f} | {self.results['treatment_effect_pvalue_ols']:.4f} | [{self.results['treatment_effect_ci_ols'][0]:.4f}, {self.results['treatment_effect_ci_ols'][1]:.4f}] |")
                report.append(f"| 稳健标准误 (HC3) | {self.results['treatment_effect']:.4f} | {self.results['treatment_effect_pvalue']:.4f} | [{self.results['treatment_effect_ci'][0]:.4f}, {self.results['treatment_effect_ci'][1]:.4f}] |")
                report.append("")
                report.append("**建议**: 由于存在异方差问题，应使用稳健标准误的结果。")
                report.append("")
            
            report.append("### 模型拟合度")
            report.append("")
            report.append(f"- **R-squared**: {diagnostics['r_squared']:.4f}")
            report.append(f"- **Adjusted R-squared**: {diagnostics['adj_r_squared']:.4f}")
            report.append(f"- **F-statistic**: {diagnostics['f_statistic']:.4f}")
            report.append(f"- **F-statistic p-value**: {diagnostics['f_pvalue']:.4f}")
            report.append(f"- **观测数**: {diagnostics['n_obs']:,}")
            report.append("")
            
            report.append("### 模型诊断")
            report.append("")
            report.append(f"- **White检验 (异方差) p-value**: {diagnostics['white_test_pvalue']:.4f}")
            if diagnostics['white_test_pvalue'] < 0.05:
                report.append("  - 提示: 存在异方差问题 (p < 0.05)，因此使用稳健标准误")
            else:
                report.append("  - 提示: 未发现异方差问题")
            report.append("")
            
            report.append(f"- **Durbin-Watson统计量**: {diagnostics['durbin_watson']:.4f}")
            if diagnostics['durbin_watson'] < 1.5:
                report.append("  - 提示: 可能存在正自相关")
            elif diagnostics['durbin_watson'] > 2.5:
                report.append("  - 提示: 可能存在负自相关")
            else:
                report.append("  - 提示: 未发现自相关问题")
            report.append("")
            
            report.append(f"- **最大VIF (多重共线性)**: {diagnostics['max_vif']:.2f}")
            if diagnostics['max_vif'] > 10:
                report.append("  - 提示: 可能存在多重共线性问题 (VIF > 10)，但VIF值在可接受范围内")
            else:
                report.append("  - 提示: 未发现多重共线性问题")
            report.append("")
            
            # 完整回归结果表（稳健标准误）
            report.append("### 完整回归结果 (稳健标准误)")
            report.append("")
            report.append("```")
            report.append(str(model_robust.summary()))
            report.append("```")
            report.append("")
        
        # 结论
        report.append("## 结论")
        report.append("")
        report.append("### 主要发现")
        report.append("")
        if self.results:
            effect = self.results['treatment_effect']
            pval = self.results['treatment_effect_pvalue']
            
            if pval < 0.05:
                if effect > 0:
                    report.append(f"1. **政策对MN0502区域有显著正向影响**: 政策实施后，MN0502区域的trip count平均增加了 {effect:.2f} 次/小时。")
                else:
                    report.append(f"1. **政策对MN0502区域有显著负向影响**: 政策实施后，MN0502区域的trip count平均减少了 {abs(effect):.2f} 次/小时。")
            else:
                report.append("1. **政策对MN0502区域没有显著影响**: 在统计上，政策实施对MN0502区域的trip count没有产生显著影响。")
            report.append("")
            
            report.append("2. **DiD方法优势**: 通过对比处理组（MN0502）和对照组（其他PUNTA区域），DiD方法能够控制时间趋势和其他混杂因素，从而更准确地识别政策效应。")
            report.append("")
            
            report.append("3. **控制变量**: 模型控制了hour_of_day, day_of_week, is_rain, is_snow, temperature, holiday等变量，以减少遗漏变量偏差。")
            report.append("")
        else:
            report.append("分析结果未生成，请检查数据和处理过程。")
            report.append("")
        
        report.append("### 政策含义")
        report.append("")
        report.append("1. 如果政策效应显著为正，说明政策促进了MN0502区域的出租车出行。")
        report.append("")
        report.append("2. 如果政策效应显著为负，说明政策抑制了MN0502区域的出租车出行。")
        report.append("")
        report.append("3. 如果政策效应不显著，说明政策对MN0502区域的出租车出行没有明显影响。")
        report.append("")
        
        report.append("### 局限性")
        report.append("")
        report.append("1. **平行趋势假设**: DiD方法依赖于处理组和对照组在政策实施前具有相似的趋势。")
        report.append("")
        report.append("2. **外部有效性**: 结果可能仅适用于MN0502区域，不能直接推广到其他区域。")
        report.append("")
        report.append("3. **时间范围**: 分析仅涵盖2024年1-7月和2025年1-7月，可能无法捕捉长期效应。")
        report.append("")
        
        # 保存报告
        report_text = "\n".join(report)
        report_path = '/Users/yanjiechen/Documents/Github/Sharon_local/reports/nta_did_analysis_mn0502_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"报告已保存到: {report_path}")
        
        return report_text

def main():
    """主函数"""
    print("="*80)
    print("NTA DiD分析: 政策对MN0502 Trip Count的影响")
    print("="*80)
    
    # 初始化分析器
    data_path = '/Users/yanjiechen/Documents/Github/Sharon_local/data/nta_hourly_taxi_summary.csv'
    analyzer = NTADIDAnalyzer(data_path)
    
    # 加载和准备数据
    analyzer.load_and_prepare_data()
    
    # 生成描述性统计
    summary_stats = analyzer.generate_summary_statistics()
    
    # 运行DiD分析
    model = analyzer.run_did_analysis()
    
    # 创建可视化
    analyzer.create_visualizations()
    
    # 生成报告
    report = analyzer.generate_report()
    
    print("\n" + "="*80)
    print("DiD分析完成！")
    print("="*80)
    print("结果已保存到:")
    print("- 报告: reports/nta_did_analysis_mn0502_report.md")
    print("- 图表: figures/nta_did_analysis_mn0502.png")
    print("- 周度时间序列图: figures/nta_did_weekly_timeseries_mn0502.png")

if __name__ == "__main__":
    main()

