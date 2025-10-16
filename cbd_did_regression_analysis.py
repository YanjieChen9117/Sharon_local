#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NYC CBD Congestion Pricing Policy Impact Analysis - Difference-in-Differences Regression Model
使用标准DiD回归模型：Y_it = α + βD_i + γPost_t + δ(D_i × Post_t) + ε_it

Author: Senior Data Scientist
Date: 2025-01-15
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, date
from pathlib import Path
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(20250115)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plotting parameters
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# Create output directories
for dir_name in ['figures', 'tables', 'reports']:
    Path(dir_name).mkdir(exist_ok=True)

print("🚀 Starting NYC CBD Congestion Pricing Policy Impact Analysis - DiD Regression Model")
print("=" * 80)

# ============================================================================
# 1. Data Loading and Preparation
# ============================================================================

print("\n📊 1. Data Loading and Preparation")
print("-" * 50)

# Load data
data_path = '/Users/sophiayeung/Desktop/sharon/Sharon_local/data/hourly_taxi_summary.csv'
print("Loading data from: {}".format(data_path))

try:
    df_raw = pd.read_csv(data_path)
    print("✅ Data loaded successfully, shape: {}".format(df_raw.shape))
except Exception as e:
    print("❌ Data loading failed: {}".format(e))
    exit(1)

# Parse datetime column
df_raw['pickup_hour'] = pd.to_datetime(df_raw['pickup_hour'])
df_raw = df_raw.sort_values('pickup_hour').reset_index(drop=True)

print("\n🔄 Merging Yellow and Green Taxi data by hour...")
print("Original data has {} rows (2 rows per hour - Yellow and Green)".format(len(df_raw)))

# Aggregate data by hour with weighted averaging
grouped = df_raw.groupby('pickup_hour')

# Calculate weighted aggregations
df = pd.DataFrame({
    # Sum metrics
    'total_trips': grouped['total_trips'].sum(),
    'total_revenue': grouped['total_revenue'].sum(),
    
    # Weighted averages (weighted by total_trips)
    'avg_distance': grouped.apply(lambda x: np.average(x['avg_distance'], weights=x['total_trips'])),
    'avg_passengers': grouped.apply(lambda x: np.average(x['avg_passengers'], weights=x['total_trips'])),
    'avg_duration': grouped.apply(lambda x: np.average(x['avg_duration'], weights=x['total_trips'])),
    'avg_speed': grouped.apply(lambda x: np.average(x['avg_speed'], weights=x['total_trips'])),
    'avg_tip_rate': grouped.apply(lambda x: np.average(x['avg_tip_rate'], weights=x['total_trips'])),
    'avg_fare': grouped.apply(lambda x: np.average(x['avg_fare'], weights=x['total_trips'])),
    
    # CBD interaction ratios (weighted by total_trips)
    'cbd_inside_ratio': grouped.apply(lambda x: np.average(x['cbd_inside_ratio'], weights=x['total_trips'])),
    'cbd_in_ratio': grouped.apply(lambda x: np.average(x['cbd_in_ratio'], weights=x['total_trips'])),
    'cbd_out_ratio': grouped.apply(lambda x: np.average(x['cbd_out_ratio'], weights=x['total_trips'])),
    'cbd_non_ratio': grouped.apply(lambda x: np.average(x['cbd_non_ratio'], weights=x['total_trips'])),
    
    # CBD neighbor interaction ratios (weighted by total_trips)
    'cbd_neighbor_inside_ratio': grouped.apply(lambda x: np.average(x['cbd_neighbor_inside_ratio'], weights=x['total_trips'])),
    'cbd_neighbor_in_ratio': grouped.apply(lambda x: np.average(x['cbd_neighbor_in_ratio'], weights=x['total_trips'])),
    'cbd_neighbor_out_ratio': grouped.apply(lambda x: np.average(x['cbd_neighbor_out_ratio'], weights=x['total_trips'])),
    'cbd_neighbor_non_ratio': grouped.apply(lambda x: np.average(x['cbd_neighbor_non_ratio'], weights=x['total_trips'])),
    
    # Taxi type distribution
    'yellow_ratio': grouped.apply(lambda x: np.average(x['yellow_ratio'], weights=x['total_trips'])),
    'green_ratio': grouped.apply(lambda x: np.average(x['green_ratio'], weights=x['total_trips'])),
}).reset_index()

print("✅ Merged data shape: {}".format(df.shape))

# Create analysis variables
policy_date = pd.Timestamp('2025-01-05')
df['post'] = df['pickup_hour'] >= policy_date
df['hour'] = df['pickup_hour'].dt.hour
df['dow'] = df['pickup_hour'].dt.day_name()
df['is_peak'] = df['hour'].isin([7, 8, 9, 10, 16, 17, 18, 19])
df['is_weekend'] = df['pickup_hour'].dt.dayofweek.isin([5, 6])
df['month'] = df['pickup_hour'].dt.month
df['date'] = df['pickup_hour'].dt.date

# Define holidays
holidays_2024_2025 = [
    date(2024, 1, 1), date(2024, 1, 15), date(2024, 2, 19), date(2024, 5, 27),
    date(2024, 6, 19), date(2024, 7, 4), date(2024, 9, 2), date(2024, 10, 14),
    date(2024, 11, 11), date(2024, 11, 28), date(2024, 12, 25),
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17), date(2025, 5, 26),
    date(2025, 6, 19), date(2025, 7, 4), date(2025, 9, 1), date(2025, 10, 13),
    date(2025, 11, 11), date(2025, 11, 27), date(2025, 12, 25),
]

df['is_holiday'] = df['date'].isin(holidays_2024_2025)

# Create day type classification
def get_day_type(row):
    if row['is_holiday']:
        return 'Holiday'
    elif row['is_weekend']:
        return 'Weekend'
    else:
        return 'Weekday'

df['day_type'] = df.apply(get_day_type, axis=1)

# Create CBD exposure variable
df['cbd_exposure'] = df['cbd_in_ratio'] + df['cbd_out_ratio'] + df['cbd_inside_ratio']

# Create treatment groups for DiD analysis
# D_i = 1 for high CBD exposure areas, 0 for low CBD exposure areas
cbd_median = df['cbd_exposure'].median()
df['treatment_group'] = (df['cbd_exposure'] > cbd_median).astype(int)

# Create peak hour treatment group
df['peak_treatment'] = df['is_peak'].astype(int)

# Create weekday treatment group
df['weekday_treatment'] = (df['day_type'] == 'Weekday').astype(int)

print("📅 Data summary:")
print(f"   • Time period: {df['pickup_hour'].min().strftime('%Y-%m-%d')} to {df['pickup_hour'].max().strftime('%Y-%m-%d')}")
print(f"   • Total observations: {len(df):,} hours")
print(f"   • Pre-policy: {(~df['post']).sum():,} hours")
print(f"   • Post-policy: {df['post'].sum():,} hours")
print(f"   • Treatment group (high CBD exposure): {df['treatment_group'].sum():,} hours")
print(f"   • Control group (low CBD exposure): {(df['treatment_group'] == 0).sum():,} hours")

# ============================================================================
# 2. DiD Regression Functions
# ============================================================================

def run_did_regression(data, outcome_var, treatment_var, post_var, covariates=None, robust=True):
    """
    运行标准DiD回归模型: Y_it = α + βD_i + γPost_t + δ(D_i × Post_t) + ε_it
    
    Parameters:
    - data: DataFrame with the data
    - outcome_var: 结果变量名称
    - treatment_var: 处理组变量名称 (D_i)
    - post_var: 政策后时期变量名称 (Post_t)
    - covariates: 协变量列表 (可选)
    - robust: 是否使用稳健标准误
    
    Returns:
    - fitted model object
    """
    
    # Prepare variables and ensure they are numeric
    y = pd.to_numeric(data[outcome_var], errors='coerce')
    
    # Create interaction term
    treatment = pd.to_numeric(data[treatment_var], errors='coerce')
    post = pd.to_numeric(data[post_var], errors='coerce')
    treatment_post = treatment * post
    
    # Prepare X matrix
    X = pd.DataFrame({
        'treatment': treatment,
        'post': post,
        'treatment_post': treatment_post
    })
    
    # Add covariates if provided
    if covariates:
        for cov in covariates:
            if cov in data.columns:
                X[cov] = pd.to_numeric(data[cov], errors='coerce')
    
    # Remove any rows with NaN values
    valid_idx = ~(y.isna() | X.isna().any(axis=1))
    y = y[valid_idx]
    X = X[valid_idx]
    
    # Add constant
    X = sm.add_constant(X)
    
    # Run regression
    if robust:
        model = sm.OLS(y, X).fit(cov_type='HC1')  # White's robust standard errors
    else:
        model = sm.OLS(y, X).fit()
    
    return model, ['treatment', 'post', 'treatment_post']

def format_regression_results(model, outcome_var, treatment_var, post_var):
    """
    格式化回归结果为易读的表格
    """
    results = {
        'outcome_variable': outcome_var,
        'treatment_variable': treatment_var,
        'post_variable': post_var,
        'n_obs': model.nobs,
        'r_squared': model.rsquared,
        'f_statistic': model.fvalue,
        'f_pvalue': model.f_pvalue,
    }
    
    # Extract coefficients
    for var in ['const', 'treatment', 'post', 'treatment_post']:
        if var in model.params.index:
            results[f'{var}_coef'] = model.params[var]
            results[f'{var}_se'] = model.bse[var]
            results[f'{var}_tstat'] = model.tvalues[var]
            results[f'{var}_pvalue'] = model.pvalues[var]
            results[f'{var}_ci_lower'] = model.conf_int().loc[var, 0]
            results[f'{var}_ci_upper'] = model.conf_int().loc[var, 1]
    
    # DiD estimate is the interaction coefficient
    results['did_estimate'] = model.params.get('treatment_post', np.nan)
    results['did_se'] = model.bse.get('treatment_post', np.nan)
    results['did_pvalue'] = model.pvalues.get('treatment_post', np.nan)
    results['did_ci_lower'] = model.conf_int().loc.get('treatment_post', (np.nan, np.nan))[0]
    results['did_ci_upper'] = model.conf_int().loc.get('treatment_post', (np.nan, np.nan))[1]
    
    return results

def run_diagnostic_tests(model, data):
    """
    运行回归诊断测试
    """
    diagnostics = {}
    
    try:
        # Heteroscedasticity tests
        white_test = het_white(model.resid, model.model.exog)
        diagnostics['white_stat'] = white_test[0]
        diagnostics['white_pvalue'] = white_test[1]
        
        bp_test = het_breuschpagan(model.resid, model.model.exog)
        diagnostics['bp_stat'] = bp_test[0]
        diagnostics['bp_pvalue'] = bp_test[1]
        
        # Autocorrelation test
        dw_stat = durbin_watson(model.resid)
        diagnostics['dw_statistic'] = dw_stat
        
        # Residuals statistics
        diagnostics['residual_mean'] = model.resid.mean()
        diagnostics['residual_std'] = model.resid.std()
        diagnostics['residual_min'] = model.resid.min()
        diagnostics['residual_max'] = model.resid.max()
        
    except Exception as e:
        print(f"Warning: Some diagnostic tests failed: {e}")
        diagnostics['error'] = str(e)
    
    return diagnostics

# ============================================================================
# 3. DiD Regression Analysis
# ============================================================================

print("\n📊 2. Difference-in-Differences Regression Analysis")
print("-" * 60)

# Define outcome variables to analyze
outcome_variables = {
    'avg_speed': 'Average Speed (mph)',
    'total_trips': 'Total Trips per Hour',
    'cbd_inside_ratio': 'CBD Inside Ratio',
    'cbd_in_ratio': 'CBD In Ratio',
    'cbd_out_ratio': 'CBD Out Ratio',
    'cbd_neighbor_in_ratio': 'CBD Neighbor In Ratio',
    'cbd_neighbor_out_ratio': 'CBD Neighbor Out Ratio',
    'avg_duration': 'Average Trip Duration (minutes)',
    'avg_distance': 'Average Trip Distance (miles)',
    'total_revenue': 'Total Revenue per Hour'
}

# Define treatment scenarios
treatment_scenarios = {
    'cbd_exposure': {
        'treatment_var': 'treatment_group',
        'description': 'High vs Low CBD Exposure Areas'
    },
    'peak_hours': {
        'treatment_var': 'peak_treatment', 
        'description': 'Peak vs Off-Peak Hours'
    },
    'weekday': {
        'treatment_var': 'weekday_treatment',
        'description': 'Weekday vs Weekend'
    }
}

# Store all regression results
all_results = []
all_models = {}

print("Running DiD regressions for all outcome variables and treatment scenarios...")

for scenario_name, scenario_info in treatment_scenarios.items():
    treatment_var = scenario_info['treatment_var']
    description = scenario_info['description']
    
    print(f"\n🔍 Treatment Scenario: {description}")
    print(f"   Treatment variable: {treatment_var}")
    
    scenario_results = []
    
    for outcome_var, outcome_label in outcome_variables.items():
        print(f"   Analyzing: {outcome_label}")
        
        # Run DiD regression
        try:
            model, X_vars = run_did_regression(
                data=df,
                outcome_var=outcome_var,
                treatment_var=treatment_var,
                post_var='post',
                covariates=None,  # Start with basic model
                robust=True
            )
            
            # Format results
            results = format_regression_results(model, outcome_var, treatment_var, 'post')
            results['scenario'] = scenario_name
            results['scenario_description'] = description
            results['outcome_label'] = outcome_label
            
            # Run diagnostic tests
            diagnostics = run_diagnostic_tests(model, df)
            results.update(diagnostics)
            
            scenario_results.append(results)
            all_results.append(results)
            
            # Store model for later use
            all_models[f"{scenario_name}_{outcome_var}"] = model
            
            # Print key results
            did_coef = results['did_estimate']
            did_pval = results['did_pvalue']
            significance = "***" if did_pval < 0.01 else "**" if did_pval < 0.05 else "*" if did_pval < 0.1 else ""
            
            print(f"     DiD Estimate: {did_coef:.4f} (p={did_pval:.4f}) {significance}")
            
        except Exception as e:
            print(f"     Error in regression: {e}")
            continue
    
    # Summary for this scenario
    scenario_df = pd.DataFrame(scenario_results)
    if not scenario_df.empty:
        significant_effects = scenario_df[scenario_df['did_pvalue'] < 0.05]
        print(f"   Summary: {len(significant_effects)} out of {len(scenario_df)} effects are statistically significant")

# Convert all results to DataFrame
results_df = pd.DataFrame(all_results)

print(f"\n✅ Completed DiD analysis for {len(results_df)} regression specifications")

# ============================================================================
# 4. Results Summary and Visualization
# ============================================================================

print("\n📊 3. Results Summary and Visualization")
print("-" * 50)

# Check if we have any results
if results_df.empty:
    print("❌ No successful regression results to summarize")
    print("   This may be due to data issues or model convergence problems")
    exit(1)

# Create summary tables
print("Creating summary tables...")

# Summary by scenario
scenario_summary = results_df.groupby('scenario').agg({
    'did_estimate': ['count', 'mean', 'std'],
    'did_pvalue': lambda x: (x < 0.05).sum(),
    'r_squared': 'mean'
}).round(4)

scenario_summary.columns = ['N_Models', 'Mean_DiD_Estimate', 'Std_DiD_Estimate', 'Significant_Effects', 'Mean_R_Squared']
scenario_summary.to_csv('tables/did_scenario_summary.csv')

# Summary by outcome variable
outcome_summary = results_df.groupby('outcome_variable').agg({
    'did_estimate': ['count', 'mean', 'std'],
    'did_pvalue': lambda x: (x < 0.05).sum(),
    'r_squared': 'mean'
}).round(4)

outcome_summary.columns = ['N_Models', 'Mean_DiD_Estimate', 'Std_DiD_Estimate', 'Significant_Effects', 'Mean_R_Squared']
outcome_summary.to_csv('tables/did_outcome_summary.csv')

# Detailed results table
detailed_results = results_df[[
    'scenario', 'scenario_description', 'outcome_variable', 'outcome_label',
    'n_obs', 'did_estimate', 'did_se', 'did_pvalue', 'did_ci_lower', 'did_ci_upper',
    'r_squared', 'f_statistic', 'f_pvalue'
]].copy()

detailed_results.to_csv('tables/did_detailed_results.csv', index=False)

print("✅ Created summary tables:")
print("   • tables/did_scenario_summary.csv")
print("   • tables/did_outcome_summary.csv") 
print("   • tables/did_detailed_results.csv")

# Create visualizations
print("Creating visualizations...")

# 1. DiD Estimates by Scenario and Outcome
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for i, scenario in enumerate(['cbd_exposure', 'peak_hours', 'weekday']):
    scenario_data = results_df[results_df['scenario'] == scenario].copy()
    
    if not scenario_data.empty:
        # Sort by DiD estimate for better visualization
        scenario_data = scenario_data.sort_values('did_estimate', ascending=True)
        
        # Create bar plot with confidence intervals
        y_pos = range(len(scenario_data))
        estimates = scenario_data['did_estimate']
        errors = [estimates - scenario_data['did_ci_lower'], 
                 scenario_data['did_ci_upper'] - estimates]
        
        # Color bars by significance
        colors = ['red' if p < 0.01 else 'orange' if p < 0.05 else 'lightblue' 
                 for p in scenario_data['did_pvalue']]
        
        axes[i].barh(y_pos, estimates, xerr=errors, color=colors, alpha=0.7)
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels(scenario_data['outcome_label'], fontsize=8)
        axes[i].set_xlabel('DiD Estimate')
        axes[i].set_title(f'{scenario_data.iloc[0]["scenario_description"]}\nDiD Estimates with 95% CI')
        axes[i].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/did_estimates_by_scenario.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Significance Heatmap
pivot_data = results_df.pivot(index='outcome_variable', columns='scenario', values='did_pvalue')
pivot_data = pivot_data < 0.05  # Convert to significance boolean

plt.figure(figsize=(8, 10))
sns.heatmap(pivot_data, annot=True, cmap='RdYlGn', cbar_kws={'label': 'Significant (p < 0.05)'})
plt.title('Statistical Significance of DiD Estimates\n(Red = Not Significant, Green = Significant)')
plt.xlabel('Treatment Scenario')
plt.ylabel('Outcome Variable')
plt.tight_layout()
plt.savefig('figures/did_significance_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Effect Size Distribution
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.hist(results_df['did_estimate'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('DiD Estimate')
plt.ylabel('Frequency')
plt.title('Distribution of DiD Estimates')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.scatter(results_df['did_estimate'], results_df['did_pvalue'], alpha=0.6)
plt.xlabel('DiD Estimate')
plt.ylabel('P-value')
plt.title('DiD Estimates vs P-values')
plt.axhline(y=0.05, color='red', linestyle='--', label='p = 0.05')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.scatter(results_df['r_squared'], results_df['did_pvalue'], alpha=0.6)
plt.xlabel('R-squared')
plt.ylabel('P-value')
plt.title('Model Fit vs Significance')
plt.axhline(y=0.05, color='red', linestyle='--', label='p = 0.05')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
significant = results_df['did_pvalue'] < 0.05
plt.pie([significant.sum(), (~significant).sum()], 
        labels=['Significant', 'Not Significant'], 
        autopct='%1.1f%%', 
        colors=['lightgreen', 'lightcoral'])
plt.title('Proportion of Significant Effects')

plt.tight_layout()
plt.savefig('figures/did_effect_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Created visualizations:")
print("   • figures/did_estimates_by_scenario.png")
print("   • figures/did_significance_heatmap.png")
print("   • figures/did_effect_analysis.png")

# ============================================================================
# 5. Generate Comprehensive Report
# ============================================================================

print("\n📊 4. Generating Comprehensive DiD Report")
print("-" * 50)

# Key findings summary
significant_results = results_df[results_df['did_pvalue'] < 0.05].copy()
key_findings = []

if not significant_results.empty:
    # Group by scenario for key findings
    for scenario in significant_results['scenario'].unique():
        scenario_results = significant_results[significant_results['scenario'] == scenario]
        
        findings = {
            'scenario': scenario,
            'scenario_description': scenario_results.iloc[0]['scenario_description'],
            'significant_effects': len(scenario_results),
            'largest_effect': scenario_results.loc[scenario_results['did_estimate'].abs().idxmax()],
            'most_significant': scenario_results.loc[scenario_results['did_pvalue'].idxmin()]
        }
        key_findings.append(findings)

# Create comprehensive report
report_content = f"""# NYC CBD Congestion Pricing Policy Impact Analysis - DiD Regression Model

**Analysis Date:** {datetime.now().strftime('%Y-%m-%d')}  
**Policy Implementation Date:** January 5, 2025  
**Data Source:** NYC Yellow & Green Taxi Hourly Summary Data (Combined)  
**Methodology:** Difference-in-Differences Regression Analysis

## Model Specification

The analysis uses the standard DiD regression specification:

**Y_it = α + βD_i + γPost_t + δ(D_i × Post_t) + ε_it**

Where:
- **Y_it**: Outcome variable for observation i at time t
- **D_i**: Treatment group indicator (1 if treated, 0 if control)
- **Post_t**: Post-policy period indicator (1 if after policy, 0 if before)
- **δ**: DiD estimator (the coefficient of interest)
- **ε_it**: Error term

## Treatment Scenarios

1. **CBD Exposure Analysis**: High vs Low CBD exposure areas
2. **Peak Hours Analysis**: Peak vs Off-peak hours
3. **Weekday Analysis**: Weekday vs Weekend patterns

## Data Description

- **Dataset:** Hourly aggregated NYC Yellow & Green Taxi data (weighted combination)
- **Time Period:** {df['pickup_hour'].min().strftime('%Y-%m-%d')} to {df['pickup_hour'].max().strftime('%Y-%m-%d')}
- **Total Observations:** {len(df):,} hours
- **Pre-policy Period:** {(~df['post']).sum():,} hours
- **Post-policy Period:** {df['post'].sum():,} hours
- **Treatment Groups:** {len(treatment_scenarios)} scenarios analyzed

## Key Findings

### Overall Results Summary

- **Total Regression Models:** {len(results_df)}
- **Statistically Significant Effects:** {len(significant_results)} ({len(significant_results)/len(results_df)*100:.1f}%)
- **Mean R-squared:** {results_df['r_squared'].mean():.3f}

### Significant Treatment Effects (p < 0.05)

"""

if key_findings:
    for finding in key_findings:
        report_content += f"""
#### {finding['scenario_description']}
- **Number of significant effects:** {finding['significant_effects']}
- **Largest effect:** {finding['largest_effect']['outcome_label']} (DiD = {finding['largest_effect']['did_estimate']:.4f}, p = {finding['largest_effect']['did_pvalue']:.4f})
- **Most significant effect:** {finding['most_significant']['outcome_label']} (DiD = {finding['most_significant']['did_estimate']:.4f}, p = {finding['most_significant']['did_pvalue']:.4f})
"""
else:
    report_content += "\nNo statistically significant treatment effects found at the 5% level.\n"

# Add detailed results table
report_content += f"""

## Detailed Results

The following table shows all DiD regression results:

| Scenario | Outcome Variable | DiD Estimate | Standard Error | P-value | 95% CI Lower | 95% CI Upper | R-squared |
|----------|------------------|--------------|----------------|---------|--------------|--------------|-----------|
"""

for _, row in results_df.iterrows():
    significance = "***" if row['did_pvalue'] < 0.01 else "**" if row['did_pvalue'] < 0.05 else "*" if row['did_pvalue'] < 0.1 else ""
    report_content += f"| {row['scenario']} | {row['outcome_label']} | {row['did_estimate']:.4f}{significance} | {row['did_se']:.4f} | {row['did_pvalue']:.4f} | {row['did_ci_lower']:.4f} | {row['did_ci_upper']:.4f} | {row['r_squared']:.3f} |\n"

report_content += f"""

**Note:** *** p<0.01, ** p<0.05, * p<0.1

## Model Diagnostics

### Heteroscedasticity Tests
- **White Test:** {results_df['white_pvalue'].mean():.4f} (average p-value across models)
- **Breusch-Pagan Test:** {results_df['bp_pvalue'].mean():.4f} (average p-value across models)

### Autocorrelation
- **Durbin-Watson Statistic:** {results_df['dw_statistic'].mean():.4f} (average across models)

## Interpretation

### CBD Exposure Analysis
"""

cbd_results = results_df[results_df['scenario'] == 'cbd_exposure']
if not cbd_results.empty:
    significant_cbd = cbd_results[cbd_results['did_pvalue'] < 0.05]
    if not significant_cbd.empty:
        report_content += f"- Found {len(significant_cbd)} statistically significant effects in CBD exposure analysis\n"
        for _, row in significant_cbd.iterrows():
            direction = "increase" if row['did_estimate'] > 0 else "decrease"
            report_content += f"- {row['outcome_label']}: {direction} of {abs(row['did_estimate']):.4f} (p = {row['did_pvalue']:.4f})\n"
    else:
        report_content += "- No statistically significant effects found in CBD exposure analysis\n"
else:
    report_content += "- CBD exposure analysis not available\n"

report_content += """
### Peak Hours Analysis
"""

peak_results = results_df[results_df['scenario'] == 'peak_hours']
if not peak_results.empty:
    significant_peak = peak_results[peak_results['did_pvalue'] < 0.05]
    if not significant_peak.empty:
        report_content += f"- Found {len(significant_peak)} statistically significant effects in peak hours analysis\n"
        for _, row in significant_peak.iterrows():
            direction = "increase" if row['did_estimate'] > 0 else "decrease"
            report_content += f"- {row['outcome_label']}: {direction} of {abs(row['did_estimate']):.4f} (p = {row['did_pvalue']:.4f})\n"
    else:
        report_content += "- No statistically significant effects found in peak hours analysis\n"
else:
    report_content += "- Peak hours analysis not available\n"

report_content += """
### Weekday vs Weekend Analysis
"""

weekday_results = results_df[results_df['scenario'] == 'weekday']
if not weekday_results.empty:
    significant_weekday = weekday_results[weekday_results['did_pvalue'] < 0.05]
    if not significant_weekday.empty:
        report_content += f"- Found {len(significant_weekday)} statistically significant effects in weekday analysis\n"
        for _, row in significant_weekday.iterrows():
            direction = "increase" if row['did_estimate'] > 0 else "decrease"
            report_content += f"- {row['outcome_label']}: {direction} of {abs(row['did_estimate']):.4f} (p = {row['did_pvalue']:.4f})\n"
    else:
        report_content += "- No statistically significant effects found in weekday analysis\n"
else:
    report_content += "- Weekday analysis not available\n"

report_content += f"""

## Conclusions

Based on the DiD regression analysis:

1. **Policy Effectiveness:** {'The policy shows significant effects on multiple outcomes' if len(significant_results) > 0 else 'Limited evidence of significant policy effects'}
2. **Treatment Heterogeneity:** Different treatment scenarios show {'varying' if len(set(significant_results['scenario'])) > 1 else 'consistent'} effects
3. **Model Fit:** Average R-squared of {results_df['r_squared'].mean():.3f} indicates {'good' if results_df['r_squared'].mean() > 0.5 else 'moderate' if results_df['r_squared'].mean() > 0.2 else 'limited'} explanatory power

## Limitations

1. **Parallel Trends Assumption:** DiD methodology assumes parallel trends between treatment and control groups
2. **Confounding Factors:** Other concurrent changes (weather, holidays, etc.) not controlled for
3. **Short-term Data:** Limited post-policy observation period
4. **Selection Bias:** Treatment group assignment based on CBD exposure may introduce selection bias
5. **Heteroscedasticity:** Some models show evidence of heteroscedasticity in residuals

## Files Generated

- **Tables:** `tables/did_*.csv`
- **Figures:** `figures/did_*.png`
- **Report:** `reports/cbd_did_regression_analysis.md`

---

**Analysis conducted by:** Senior Data Scientist  
**Methodology:** Difference-in-Differences Regression  
**Statistical Software:** Python with statsmodels
"""

# Save report
with open('reports/cbd_did_regression_analysis.md', 'w', encoding='utf-8') as f:
    f.write(report_content)

print("✅ Generated comprehensive DiD report: reports/cbd_did_regression_analysis.md")

# ============================================================================
# 6. Final Summary
# ============================================================================

print("\n" + "="*80)
print("🎯 DIFFERENCE-IN-DIFFERENCES REGRESSION ANALYSIS COMPLETE")
print("="*80)

print(f"\n📊 ANALYSIS SUMMARY:")
print(f"   • Total regression models: {len(results_df)}")
print(f"   • Statistically significant effects: {len(significant_results)} ({len(significant_results)/len(results_df)*100:.1f}%)")
print(f"   • Treatment scenarios analyzed: {len(treatment_scenarios)}")
print(f"   • Outcome variables analyzed: {len(outcome_variables)}")

print(f"\n📈 KEY FINDINGS:")
if not significant_results.empty:
    for scenario in significant_results['scenario'].unique():
        scenario_sig = significant_results[significant_results['scenario'] == scenario]
        print(f"   • {scenario_sig.iloc[0]['scenario_description']}: {len(scenario_sig)} significant effects")
        
        # Show the most significant effect
        most_sig = scenario_sig.loc[scenario_sig['did_pvalue'].idxmin()]
        print(f"     - Most significant: {most_sig['outcome_label']} (DiD = {most_sig['did_estimate']:.4f}, p = {most_sig['did_pvalue']:.4f})")
else:
    print("   • No statistically significant effects found at 5% level")

print(f"\n📁 OUTPUT FILES:")
print(f"   • Report: reports/cbd_did_regression_analysis.md")
print(f"   • Tables: tables/did_*.csv")
print(f"   • Figures: figures/did_*.png")

print(f"\n✅ DiD REGRESSION ANALYSIS COMPLETE!")
print("   • Standard DiD specification implemented")
print("   • Robust standard errors used")
print("   • Multiple treatment scenarios analyzed")
print("   • Comprehensive diagnostic tests performed")
print("   • Results formatted for academic publication")
print("="*80)
