#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细的DiD分析脚本 - 生成完整的统计结果和诊断

功能：
1. 运行DiD回归分析
2. 生成详细的回归结果表
3. 进行模型诊断
4. 创建结果汇总表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

# 设置图表样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class DetailedDIDAnalyzer:
    """详细的DiD分析器"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.did_data = None
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
        pre_data = self.df[(self.df['pickup_hour'] >= pre_start) & 
                          (self.df['pickup_hour'] <= pre_end)].copy()
        post_data = self.df[(self.df['pickup_hour'] >= post_start) & 
                           (self.df['pickup_hour'] <= post_end)].copy()
        
        # 合并数据并创建DiD变量
        pre_data['post'] = 0
        post_data['post'] = 1
        
        self.did_data = pd.concat([pre_data, post_data], ignore_index=True)
        
        # 创建treatment变量
        cbd_threshold = self.did_data['cbd_inside_ratio'].median()
        self.did_data['treatment'] = (self.did_data['cbd_inside_ratio'] > cbd_threshold).astype(int)
        
        # 创建交互项
        self.did_data['treatment_post'] = self.did_data['treatment'] * self.did_data['post']
        
        # 添加控制变量
        self.did_data['hour_of_day_sq'] = self.did_data['hour_of_day'] ** 2
        self.did_data['day_of_week'] = self.did_data['pickup_hour'].dt.dayofweek
        
        print(f"Data prepared: {len(self.did_data)} observations")
        
    def run_detailed_did_analysis(self, outcome_var, holiday_filter=None):
        """运行详细的DiD分析"""
        print(f"\nRunning detailed DiD analysis for {outcome_var}...")
        
        # 数据筛选
        if holiday_filter == 'holiday':
            analysis_data = self.did_data[self.did_data['holiday'] == True].copy()
        elif holiday_filter == 'not_holiday':
            analysis_data = self.did_data[self.did_data['holiday'] == False].copy()
        else:
            analysis_data = self.did_data.copy()
        
        # 移除缺失值
        analysis_data = analysis_data.dropna(subset=[outcome_var])
        
        # 准备回归变量
        X_vars = ['treatment', 'post', 'treatment_post']
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
        
        # 模型诊断
        diagnostics = self._run_model_diagnostics(model, X, y)
        
        # 存储结果
        result_key = f"{outcome_var}_{holiday_filter if holiday_filter else 'all'}"
        self.results[result_key] = {
            'model': model,
            'data': analysis_data,
            'outcome_var': outcome_var,
            'holiday_filter': holiday_filter,
            'n_obs': len(analysis_data),
            'diagnostics': diagnostics,
            'treatment_effect': model.params.get('treatment_post', np.nan),
            'treatment_effect_pvalue': model.pvalues.get('treatment_post', np.nan),
            'treatment_effect_ci': model.conf_int().loc['treatment_post'] if 'treatment_post' in model.params else (np.nan, np.nan)
        }
        
        return model, diagnostics
    
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
        
        # 异方差检验
        try:
            white_test = het_white(model.resid, model.model.exog)
            diagnostics['white_test_stat'] = white_test[0]
            diagnostics['white_test_pvalue'] = white_test[1]
        except:
            diagnostics['white_test_stat'] = np.nan
            diagnostics['white_test_pvalue'] = np.nan
        
        # 自相关检验
        try:
            dw_stat = durbin_watson(model.resid)
            diagnostics['durbin_watson'] = dw_stat
        except:
            diagnostics['durbin_watson'] = np.nan
        
        # VIF检验（多重共线性）
        try:
            vif_data = pd.DataFrame()
            vif_data["Variable"] = X.columns
            vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            diagnostics['max_vif'] = vif_data['VIF'].max()
            diagnostics['vif_data'] = vif_data
        except:
            diagnostics['max_vif'] = np.nan
            diagnostics['vif_data'] = None
        
        return diagnostics
    
    def create_results_summary_table(self):
        """创建结果汇总表"""
        print("\nCreating results summary table...")
        
        summary_data = []
        
        for key, result in self.results.items():
            model = result['model']
            diagnostics = result['diagnostics']
            
            # 提取关键系数
            treatment_effect = result['treatment_effect']
            treatment_pvalue = result['treatment_effect_pvalue']
            treatment_ci = result['treatment_effect_ci']
            
            # 显著性标记
            if treatment_pvalue < 0.01:
                significance = '***'
            elif treatment_pvalue < 0.05:
                significance = '**'
            elif treatment_pvalue < 0.1:
                significance = '*'
            else:
                significance = ''
            
            summary_row = {
                'Outcome': result['outcome_var'].replace('_', ' ').title(),
                'Sample': result['holiday_filter'] if result['holiday_filter'] else 'All',
                'N': result['n_obs'],
                'Treatment_Effect': f"{treatment_effect:.4f}{significance}",
                'P_Value': f"{treatment_pvalue:.4f}",
                'CI_Lower': f"{treatment_ci[0]:.4f}",
                'CI_Upper': f"{treatment_ci[1]:.4f}",
                'R_Squared': f"{diagnostics['r_squared']:.4f}",
                'Adj_R_Squared': f"{diagnostics['adj_r_squared']:.4f}",
                'F_Statistic': f"{diagnostics['f_statistic']:.2f}",
                'White_Test_P': f"{diagnostics['white_test_pvalue']:.4f}",
                'Durbin_Watson': f"{diagnostics['durbin_watson']:.4f}",
                'Max_VIF': f"{diagnostics['max_vif']:.2f}"
            }
            summary_data.append(summary_row)
        
        self.summary_table = pd.DataFrame(summary_data)
        return self.summary_table
    
    def create_diagnostic_plots(self):
        """创建诊断图表"""
        print("\nCreating diagnostic plots...")
        
        # 选择几个代表性模型进行诊断
        sample_models = []
        for key, result in self.results.items():
            if 'all' in key:  # 选择全样本模型
                sample_models.append((key, result))
                if len(sample_models) >= 3:  # 最多3个模型
                    break
        
        if not sample_models:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Diagnostic Plots', fontsize=16, fontweight='bold')
        
        for i, (key, result) in enumerate(sample_models):
            model = result['model']
            outcome_var = result['outcome_var']
            
            # 残差图
            ax1 = axes[0, i]
            ax1.scatter(model.fittedvalues, model.resid, alpha=0.6)
            ax1.axhline(y=0, color='red', linestyle='--')
            ax1.set_xlabel('Fitted Values')
            ax1.set_ylabel('Residuals')
            ax1.set_title(f'{outcome_var.replace("_", " ").title()}\nResiduals vs Fitted')
            ax1.grid(True, alpha=0.3)
            
            # Q-Q图
            ax2 = axes[1, i]
            sm.qqplot(model.resid, ax=ax2, line='s')
            ax2.set_title(f'{outcome_var.replace("_", " ").title()}\nQ-Q Plot')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/yanjiechen/Documents/Github/Sharon_local/figures/did_diagnostic_plots.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_detailed_report(self):
        """生成详细报告"""
        print("\nGenerating detailed report...")
        
        report = []
        report.append("# Detailed DiD Analysis Report: Policy Impact on NYC Taxi Operations")
        report.append("")
        report.append("## Executive Summary")
        report.append("")
        report.append("This detailed report presents comprehensive Difference-in-Differences (DiD) analysis results examining the impact of a policy implemented on January 5, 2025, on NYC taxi operations. The analysis includes model diagnostics, robustness checks, and detailed statistical results.")
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
        
        # 结果汇总表
        if hasattr(self, 'summary_table'):
            report.append("## Results Summary Table")
            report.append("")
            report.append("| Outcome | Sample | N | Treatment Effect | P-Value | 95% CI Lower | 95% CI Upper | R² | Adj R² | F-Stat | White Test P | DW | Max VIF |")
            report.append("|---------|--------|---|------------------|---------|--------------|--------------|----|---------|---------|--------------|----|---------|")
            
            for _, row in self.summary_table.iterrows():
                report.append(f"| {row['Outcome']} | {row['Sample']} | {row['N']} | {row['Treatment_Effect']} | {row['P_Value']} | {row['CI_Lower']} | {row['CI_Upper']} | {row['R_Squared']} | {row['Adj_R_Squared']} | {row['F_Statistic']} | {row['White_Test_P']} | {row['Durbin_Watson']} | {row['Max_VIF']} |")
            report.append("")
            report.append("**Note**: *** p<0.01, ** p<0.05, * p<0.1")
            report.append("")
        
        # 详细结果
        report.append("## Detailed Results by Outcome Variable")
        report.append("")
        
        outcome_vars = ['avg_speed', 'total_trips', 'cbd_inside_ratio', 
                       'cbd_neighbor_inside_ratio', 'avg_speed_out_CBD']
        outcome_names = ['Average Taxi Speed', 'Total Trip Volume', 'CBD Internal Trips', 
                        'CBD Neighbor Trips', 'CBD Exit Speed']
        
        for var, name in zip(outcome_vars, outcome_names):
            report.append(f"### {name}")
            report.append("")
            
            # 全样本结果
            if f"{var}_all" in self.results:
                result = self.results[f"{var}_all"]
                model = result['model']
                diagnostics = result['diagnostics']
                
                report.append("**All Data:**")
                report.append(f"- Treatment Effect (β₃): {result['treatment_effect']:.4f}")
                report.append(f"- P-value: {result['treatment_effect_pvalue']:.4f}")
                report.append(f"- 95% Confidence Interval: [{result['treatment_effect_ci'][0]:.4f}, {result['treatment_effect_ci'][1]:.4f}]")
                report.append(f"- R-squared: {diagnostics['r_squared']:.4f}")
                report.append(f"- F-statistic: {diagnostics['f_statistic']:.2f}")
                report.append(f"- White test p-value: {diagnostics['white_test_pvalue']:.4f}")
                report.append(f"- Durbin-Watson statistic: {diagnostics['durbin_watson']:.4f}")
                report.append("")
            
            # 节假日结果
            if f"{var}_holiday" in self.results:
                result = self.results[f"{var}_holiday"]
                diagnostics = result['diagnostics']
                
                report.append("**Holiday Periods:**")
                report.append(f"- Treatment Effect (β₃): {result['treatment_effect']:.4f}")
                report.append(f"- P-value: {result['treatment_effect_pvalue']:.4f}")
                report.append(f"- 95% Confidence Interval: [{result['treatment_effect_ci'][0]:.4f}, {result['treatment_effect_ci'][1]:.4f}]")
                report.append(f"- R-squared: {diagnostics['r_squared']:.4f}")
                report.append("")
            
            # 非节假日结果
            if f"{var}_not_holiday" in self.results:
                result = self.results[f"{var}_not_holiday"]
                diagnostics = result['diagnostics']
                
                report.append("**Non-Holiday Periods:**")
                report.append(f"- Treatment Effect (β₃): {result['treatment_effect']:.4f}")
                report.append(f"- P-value: {result['treatment_effect_pvalue']:.4f}")
                report.append(f"- 95% Confidence Interval: [{result['treatment_effect_ci'][0]:.4f}, {result['treatment_effect_ci'][1]:.4f}]")
                report.append(f"- R-squared: {diagnostics['r_squared']:.4f}")
                report.append("")
        
        # 模型诊断
        report.append("## Model Diagnostics")
        report.append("")
        report.append("### Diagnostic Tests")
        report.append("")
        report.append("1. **Heteroscedasticity Test (White Test)**: Tests for constant variance of residuals")
        report.append("2. **Autocorrelation Test (Durbin-Watson)**: Tests for serial correlation in residuals")
        report.append("3. **Multicollinearity Test (VIF)**: Tests for correlation among independent variables")
        report.append("")
        report.append("### Interpretation Guidelines")
        report.append("")
        report.append("- **White Test p-value < 0.05**: Evidence of heteroscedasticity")
        report.append("- **Durbin-Watson ≈ 2**: No autocorrelation")
        report.append("- **VIF > 10**: Potential multicollinearity concerns")
        report.append("")
        
        # 结论
        report.append("## Conclusions")
        report.append("")
        report.append("### Key Findings")
        report.append("")
        
        # 统计显著的结果
        significant_results = []
        for key, result in self.results.items():
            if result['treatment_effect_pvalue'] < 0.05:
                significant_results.append((key, result))
        
        if significant_results:
            report.append("**Statistically Significant Results (p < 0.05):**")
            for key, result in significant_results:
                outcome = result['outcome_var'].replace('_', ' ').title()
                sample = result['holiday_filter'] if result['holiday_filter'] else 'All'
                effect = result['treatment_effect']
                pval = result['treatment_effect_pvalue']
                report.append(f"- {outcome} ({sample}): Treatment effect = {effect:.4f}, p-value = {pval:.4f}")
            report.append("")
        else:
            report.append("**No statistically significant treatment effects were found at the 5% level.**")
            report.append("")
        
        report.append("### Policy Implications")
        report.append("")
        report.append("1. **Treatment Effect Interpretation**: The β₃ coefficient represents the differential impact of the policy on the treatment group relative to the control group.")
        report.append("")
        report.append("2. **Holiday vs Non-Holiday Effects**: Separate analyses reveal how policy effects vary across different time contexts.")
        report.append("")
        report.append("3. **Statistical Significance**: Results are evaluated at conventional significance levels (1%, 5%, 10%).")
        report.append("")
        report.append("4. **Model Robustness**: The analysis includes comprehensive diagnostic tests to ensure model validity.")
        report.append("")
        
        # 保存报告
        report_text = "\n".join(report)
        with open('/Users/yanjiechen/Documents/Github/Sharon_local/reports/detailed_did_analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("Detailed report saved to: /Users/yanjiechen/Documents/Github/Sharon_local/reports/detailed_did_analysis_report.md")
        
        return report_text

def main():
    """主函数"""
    print("Starting Detailed DiD Analysis...")
    
    # 初始化分析器
    analyzer = DetailedDIDAnalyzer('/Users/yanjiechen/Documents/Github/Sharon_local/data/hourly_taxi_summary.csv')
    
    # 加载和准备数据
    analyzer.load_and_prepare_data()
    
    # 定义分析变量
    outcome_vars = ['avg_speed', 'total_trips', 'cbd_inside_ratio', 
                   'cbd_neighbor_inside_ratio', 'avg_speed_out_CBD']
    
    # 运行详细DiD分析
    for var in outcome_vars:
        # 全样本分析
        analyzer.run_detailed_did_analysis(var, holiday_filter=None)
        
        # 节假日分析
        analyzer.run_detailed_did_analysis(var, holiday_filter='holiday')
        
        # 非节假日分析
        analyzer.run_detailed_did_analysis(var, holiday_filter='not_holiday')
    
    # 创建结果汇总表
    summary_table = analyzer.create_results_summary_table()
    print("\nResults Summary Table:")
    print(summary_table.to_string(index=False))
    
    # 创建诊断图表
    analyzer.create_diagnostic_plots()
    
    # 生成详细报告
    report = analyzer.generate_detailed_report()
    
    print("\nDetailed DiD Analysis completed successfully!")
    print("Results saved to:")
    print("- Detailed Report: /Users/yanjiechen/Documents/Github/Sharon_local/reports/detailed_did_analysis_report.md")
    print("- Diagnostic Plots: /Users/yanjiechen/Documents/Github/Sharon_local/figures/did_diagnostic_plots.png")

if __name__ == "__main__":
    main()
