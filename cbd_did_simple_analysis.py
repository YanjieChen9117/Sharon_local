#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NYC CBD Congestion Pricing Policy Impact Analysis - Simplified DiD Regression
使用标准DiD回归模型：Y_it = α + βD_i + γPost_t + δ(D_i × Post_t) + ε_it

Author: Senior Data Scientist
Date: 2025-01-15
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, date
from pathlib import Path
import statsmodels.api as sm
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

print("🚀 Starting NYC CBD Congestion Pricing Policy Impact Analysis - Simplified DiD")
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

print("\n🔄 Aggregating Yellow and Green Taxi data by hour...")

# Simple aggregation by hour - sum for counts, mean for ratios
df = df_raw.groupby('pickup_hour').agg({
    'total_trips': 'sum',
    'total_revenue': 'sum',
    'avg_distance': 'mean',
    'avg_passengers': 'mean',
    'avg_duration': 'mean',
    'avg_speed': 'mean',
    'avg_tip_rate': 'mean',
    'avg_fare': 'mean',
    'cbd_inside_ratio': 'mean',
    'cbd_in_ratio': 'mean',
    'cbd_out_ratio': 'mean',
    'cbd_non_ratio': 'mean',
    'cbd_neighbor_inside_ratio': 'mean',
    'cbd_neighbor_in_ratio': 'mean',
    'cbd_neighbor_out_ratio': 'mean',
    'cbd_neighbor_non_ratio': 'mean',
    'yellow_ratio': 'mean',
    'green_ratio': 'mean'
}).reset_index()

print("✅ Aggregated data shape: {}".format(df.shape))

# Create analysis variables
policy_date = pd.Timestamp('2025-01-05')
df['post'] = (df['pickup_hour'] >= policy_date).astype(int)
df['hour'] = df['pickup_hour'].dt.hour
df['is_peak'] = df['hour'].isin([7, 8, 9, 10, 16, 17, 18, 19]).astype(int)
df['is_weekend'] = df['pickup_hour'].dt.dayofweek.isin([5, 6]).astype(int)
df['month'] = df['pickup_hour'].dt.month

# Define holidays
holidays_2024_2025 = [
    date(2024, 1, 1), date(2024, 1, 15), date(2024, 2, 19), date(2024, 5, 27),
    date(2024, 6, 19), date(2024, 7, 4), date(2024, 9, 2), date(2024, 10, 14),
    date(2024, 11, 11), date(2024, 11, 28), date(2024, 12, 25),
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17), date(2025, 5, 26),
    date(2025, 6, 19), date(2025, 7, 4), date(2025, 9, 1), date(2025, 10, 13),
    date(2025, 11, 11), date(2025, 11, 27), date(2025, 12, 25),
]

df['is_holiday'] = df['pickup_hour'].dt.date.isin(holidays_2024_2025).astype(int)

# Create treatment groups for DiD analysis
# CBD exposure treatment
df['cbd_exposure'] = df['cbd_in_ratio'] + df['cbd_out_ratio'] + df['cbd_inside_ratio']
cbd_median = df['cbd_exposure'].median()
df['treatment_cbd'] = (df['cbd_exposure'] > cbd_median).astype(int)

# Peak hours treatment
df['treatment_peak'] = df['is_peak']

# Weekday treatment
df['treatment_weekday'] = (1 - df['is_weekend'] - df['is_holiday']).astype(int)

print("📅 Data summary:")
print(f"   • Time period: {df['pickup_hour'].min().strftime('%Y-%m-%d')} to {df['pickup_hour'].max().strftime('%Y-%m-%d')}")
print(f"   • Total observations: {len(df):,} hours")
print(f"   • Pre-policy: {(df['post'] == 0).sum():,} hours")
print(f"   • Post-policy: {df['post'].sum():,} hours")
print(f"   • CBD treatment group: {df['treatment_cbd'].sum():,} hours")
print(f"   • Peak hours treatment: {df['treatment_peak'].sum():,} hours")
print(f"   • Weekday treatment: {df['treatment_weekday'].sum():,} hours")

# ============================================================================
# 2. DiD Regression Analysis
# ============================================================================

print("\n📊 2. Difference-in-Differences Regression Analysis")
print("-" * 60)

# Define outcome variables
outcomes = {
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

# Define treatments
treatments = {
    'treatment_cbd': 'High vs Low CBD Exposure',
    'treatment_peak': 'Peak vs Off-Peak Hours',
    'treatment_weekday': 'Weekday vs Weekend'
}

# Store results
results = []

print("Running DiD regressions...")

for treatment_name, treatment_desc in treatments.items():
    print(f"\n🔍 Treatment: {treatment_desc}")
    
    for outcome_name, outcome_desc in outcomes.items():
        try:
            # Prepare data for regression
            y = df[outcome_name].values
            treatment = df[treatment_name].values
            post = df['post'].values
            interaction = treatment * post
            
            # Create design matrix
            X = np.column_stack([
                np.ones(len(y)),  # constant
                treatment,        # D_i
                post,            # Post_t
                interaction      # D_i × Post_t
            ])
            
            # Run OLS regression
            model = sm.OLS(y, X).fit(cov_type='HC1')  # Robust standard errors
            
            # Extract results
            did_estimate = model.params[3]  # interaction coefficient
            did_se = model.bse[3]
            did_pvalue = model.pvalues[3]
            did_ci = model.conf_int()[3]  # Get confidence interval for the 4th parameter (interaction)
            
            result = {
                'treatment': treatment_name,
                'treatment_description': treatment_desc,
                'outcome': outcome_name,
                'outcome_description': outcome_desc,
                'n_obs': model.nobs,
                'did_estimate': did_estimate,
                'did_se': did_se,
                'did_pvalue': did_pvalue,
                'did_ci_lower': did_ci[0],
                'did_ci_upper': did_ci[1],
                'r_squared': model.rsquared,
                'f_statistic': model.fvalue,
                'f_pvalue': model.f_pvalue,
                'treatment_coef': model.params[1],
                'post_coef': model.params[2],
                'constant': model.params[0]
            }
            
            results.append(result)
            
            # Print key result
            significance = ""
            if did_pvalue < 0.01:
                significance = "***"
            elif did_pvalue < 0.05:
                significance = "**"
            elif did_pvalue < 0.1:
                significance = "*"
            
            print(f"   {outcome_desc}: DiD = {did_estimate:.4f} (p = {did_pvalue:.4f}) {significance}")
            
        except Exception as e:
            print(f"   {outcome_desc}: Error - {e}")
            continue

# Convert to DataFrame
results_df = pd.DataFrame(results)

print(f"\n✅ Completed DiD analysis: {len(results_df)} successful regressions")

# ============================================================================
# 3. Results Summary and Tables
# ============================================================================

print("\n📊 3. Results Summary and Tables")
print("-" * 50)

if results_df.empty:
    print("❌ No successful regressions to summarize")
    exit(1)

# Create summary tables
print("Creating summary tables...")

# Summary by treatment
treatment_summary = results_df.groupby('treatment').agg({
    'did_estimate': ['count', 'mean', 'std'],
    'did_pvalue': lambda x: (x < 0.05).sum(),
    'r_squared': 'mean'
}).round(4)

treatment_summary.columns = ['N_Models', 'Mean_DiD_Estimate', 'Std_DiD_Estimate', 'Significant_Effects', 'Mean_R_Squared']
treatment_summary.to_csv('tables/did_treatment_summary.csv')

# Summary by outcome
outcome_summary = results_df.groupby('outcome').agg({
    'did_estimate': ['count', 'mean', 'std'],
    'did_pvalue': lambda x: (x < 0.05).sum(),
    'r_squared': 'mean'
}).round(4)

outcome_summary.columns = ['N_Models', 'Mean_DiD_Estimate', 'Std_DiD_Estimate', 'Significant_Effects', 'Mean_R_Squared']
outcome_summary.to_csv('tables/did_outcome_summary.csv')

# Detailed results
detailed_results = results_df[[
    'treatment', 'treatment_description', 'outcome', 'outcome_description',
    'n_obs', 'did_estimate', 'did_se', 'did_pvalue', 'did_ci_lower', 'did_ci_upper',
    'r_squared', 'f_statistic', 'f_pvalue', 'treatment_coef', 'post_coef', 'constant'
]].copy()

detailed_results.to_csv('tables/did_detailed_results.csv', index=False)

print("✅ Created summary tables:")
print("   • tables/did_treatment_summary.csv")
print("   • tables/did_outcome_summary.csv")
print("   • tables/did_detailed_results.csv")

# ============================================================================
# 4. Visualizations
# ============================================================================

print("\n📊 4. Creating Visualizations")
print("-" * 50)

# 1. DiD Estimates by Treatment and Outcome
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for i, treatment in enumerate(['treatment_cbd', 'treatment_peak', 'treatment_weekday']):
    treatment_data = results_df[results_df['treatment'] == treatment].copy()
    
    if not treatment_data.empty:
        # Sort by DiD estimate
        treatment_data = treatment_data.sort_values('did_estimate', ascending=True)
        
        # Create bar plot with confidence intervals
        y_pos = range(len(treatment_data))
        estimates = treatment_data['did_estimate']
        errors = [estimates - treatment_data['did_ci_lower'], 
                 treatment_data['did_ci_upper'] - estimates]
        
        # Color bars by significance
        colors = ['red' if p < 0.01 else 'orange' if p < 0.05 else 'lightblue' 
                 for p in treatment_data['did_pvalue']]
        
        axes[i].barh(y_pos, estimates, xerr=errors, color=colors, alpha=0.7)
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels(treatment_data['outcome_description'], fontsize=8)
        axes[i].set_xlabel('DiD Estimate')
        axes[i].set_title(f'{treatment_data.iloc[0]["treatment_description"]}\nDiD Estimates with 95% CI')
        axes[i].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/did_estimates_by_treatment.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Significance Heatmap
pivot_data = results_df.pivot(index='outcome_description', columns='treatment_description', values='did_pvalue')
pivot_data = pivot_data < 0.05  # Convert to significance boolean

plt.figure(figsize=(10, 8))
sns.heatmap(pivot_data, annot=True, cmap='RdYlGn', cbar_kws={'label': 'Significant (p < 0.05)'})
plt.title('Statistical Significance of DiD Estimates\n(Red = Not Significant, Green = Significant)')
plt.xlabel('Treatment Scenario')
plt.ylabel('Outcome Variable')
plt.tight_layout()
plt.savefig('figures/did_significance_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Effect Size Distribution
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.hist(results_df['did_estimate'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('DiD Estimate')
plt.ylabel('Frequency')
plt.title('Distribution of DiD Estimates')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 2)
plt.scatter(results_df['did_estimate'], results_df['did_pvalue'], alpha=0.6)
plt.xlabel('DiD Estimate')
plt.ylabel('P-value')
plt.title('DiD Estimates vs P-values')
plt.axhline(y=0.05, color='red', linestyle='--', label='p = 0.05')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 3)
plt.scatter(results_df['r_squared'], results_df['did_pvalue'], alpha=0.6)
plt.xlabel('R-squared')
plt.ylabel('P-value')
plt.title('Model Fit vs Significance')
plt.axhline(y=0.05, color='red', linestyle='--', label='p = 0.05')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 4)
significant = results_df['did_pvalue'] < 0.05
plt.pie([significant.sum(), (~significant).sum()], 
        labels=['Significant', 'Not Significant'], 
        autopct='%1.1f%%', 
        colors=['lightgreen', 'lightcoral'])
plt.title('Proportion of Significant Effects')

plt.subplot(2, 3, 5)
# DiD estimates by treatment type
for treatment in results_df['treatment'].unique():
    treatment_data = results_df[results_df['treatment'] == treatment]
    plt.scatter([treatment] * len(treatment_data), treatment_data['did_estimate'], 
               alpha=0.6, label=treatment_data.iloc[0]['treatment_description'])
plt.xlabel('Treatment Type')
plt.ylabel('DiD Estimate')
plt.title('DiD Estimates by Treatment Type')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 6)
# R-squared by treatment type
for treatment in results_df['treatment'].unique():
    treatment_data = results_df[results_df['treatment'] == treatment]
    plt.scatter([treatment] * len(treatment_data), treatment_data['r_squared'], 
               alpha=0.6, label=treatment_data.iloc[0]['treatment_description'])
plt.xlabel('Treatment Type')
plt.ylabel('R-squared')
plt.title('Model Fit by Treatment Type')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/did_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Created visualizations:")
print("   • figures/did_estimates_by_treatment.png")
print("   • figures/did_significance_heatmap.png")
print("   • figures/did_comprehensive_analysis.png")

# ============================================================================
# 5. Generate Report
# ============================================================================

print("\n📊 5. Generating Comprehensive DiD Report")
print("-" * 50)

# Key findings
significant_results = results_df[results_df['did_pvalue'] < 0.05].copy()

# Create report
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

- **Dataset:** Hourly aggregated NYC Yellow & Green Taxi data
- **Time Period:** {df['pickup_hour'].min().strftime('%Y-%m-%d')} to {df['pickup_hour'].max().strftime('%Y-%m-%d')}
- **Total Observations:** {len(df):,} hours
- **Pre-policy Period:** {(df['post'] == 0).sum():,} hours
- **Post-policy Period:** {df['post'].sum():,} hours
- **Treatment Groups:** {len(treatments)} scenarios analyzed

## Key Findings

### Overall Results Summary

- **Total Regression Models:** {len(results_df)}
- **Statistically Significant Effects:** {len(significant_results)} ({len(significant_results)/len(results_df)*100:.1f}%)
- **Mean R-squared:** {results_df['r_squared'].mean():.3f}

### Significant Treatment Effects (p < 0.05)

"""

if not significant_results.empty:
    for _, row in significant_results.iterrows():
        significance = "***" if row['did_pvalue'] < 0.01 else "**" if row['did_pvalue'] < 0.05 else "*"
        report_content += f"""
#### {row['treatment_description']} - {row['outcome_description']}
- **DiD Estimate:** {row['did_estimate']:.4f} {significance}
- **Standard Error:** {row['did_se']:.4f}
- **P-value:** {row['did_pvalue']:.4f}
- **95% Confidence Interval:** [{row['did_ci_lower']:.4f}, {row['did_ci_upper']:.4f}]
- **R-squared:** {row['r_squared']:.3f}
"""
else:
    report_content += "\nNo statistically significant treatment effects found at the 5% level.\n"

report_content += f"""

## Detailed Results

The following table shows all DiD regression results:

| Treatment | Outcome Variable | DiD Estimate | Standard Error | P-value | 95% CI Lower | 95% CI Upper | R-squared |
|-----------|------------------|--------------|----------------|---------|--------------|--------------|-----------|
"""

for _, row in results_df.iterrows():
    significance = "***" if row['did_pvalue'] < 0.01 else "**" if row['did_pvalue'] < 0.05 else "*" if row['did_pvalue'] < 0.1 else ""
    report_content += f"| {row['treatment_description']} | {row['outcome_description']} | {row['did_estimate']:.4f}{significance} | {row['did_se']:.4f} | {row['did_pvalue']:.4f} | {row['did_ci_lower']:.4f} | {row['did_ci_upper']:.4f} | {row['r_squared']:.3f} |\n"

report_content += f"""

**Note:** *** p<0.01, ** p<0.05, * p<0.1

## Interpretation

### CBD Exposure Analysis
"""

cbd_results = results_df[results_df['treatment'] == 'treatment_cbd']
if not cbd_results.empty:
    significant_cbd = cbd_results[cbd_results['did_pvalue'] < 0.05]
    if not significant_cbd.empty:
        report_content += f"- Found {len(significant_cbd)} statistically significant effects in CBD exposure analysis\n"
        for _, row in significant_cbd.iterrows():
            direction = "increase" if row['did_estimate'] > 0 else "decrease"
            report_content += f"- {row['outcome_description']}: {direction} of {abs(row['did_estimate']):.4f} (p = {row['did_pvalue']:.4f})\n"
    else:
        report_content += "- No statistically significant effects found in CBD exposure analysis\n"

report_content += """
### Peak Hours Analysis
"""

peak_results = results_df[results_df['treatment'] == 'treatment_peak']
if not peak_results.empty:
    significant_peak = peak_results[peak_results['did_pvalue'] < 0.05]
    if not significant_peak.empty:
        report_content += f"- Found {len(significant_peak)} statistically significant effects in peak hours analysis\n"
        for _, row in significant_peak.iterrows():
            direction = "increase" if row['did_estimate'] > 0 else "decrease"
            report_content += f"- {row['outcome_description']}: {direction} of {abs(row['did_estimate']):.4f} (p = {row['did_pvalue']:.4f})\n"
    else:
        report_content += "- No statistically significant effects found in peak hours analysis\n"

report_content += """
### Weekday vs Weekend Analysis
"""

weekday_results = results_df[results_df['treatment'] == 'treatment_weekday']
if not weekday_results.empty:
    significant_weekday = weekday_results[weekday_results['did_pvalue'] < 0.05]
    if not significant_weekday.empty:
        report_content += f"- Found {len(significant_weekday)} statistically significant effects in weekday analysis\n"
        for _, row in significant_weekday.iterrows():
            direction = "increase" if row['did_estimate'] > 0 else "decrease"
            report_content += f"- {row['outcome_description']}: {direction} of {abs(row['did_estimate']):.4f} (p = {row['did_pvalue']:.4f})\n"
    else:
        report_content += "- No statistically significant effects found in weekday analysis\n"

report_content += f"""

## Conclusions

Based on the DiD regression analysis:

1. **Policy Effectiveness:** {'The policy shows significant effects on multiple outcomes' if len(significant_results) > 0 else 'Limited evidence of significant policy effects'}
2. **Treatment Heterogeneity:** Different treatment scenarios show {'varying' if len(set(significant_results['treatment'])) > 1 else 'consistent'} effects
3. **Model Fit:** Average R-squared of {results_df['r_squared'].mean():.3f} indicates {'good' if results_df['r_squared'].mean() > 0.5 else 'moderate' if results_df['r_squared'].mean() > 0.2 else 'limited'} explanatory power

## Limitations

1. **Parallel Trends Assumption:** DiD methodology assumes parallel trends between treatment and control groups
2. **Confounding Factors:** Other concurrent changes (weather, holidays, etc.) not controlled for
3. **Short-term Data:** Limited post-policy observation period
4. **Selection Bias:** Treatment group assignment based on CBD exposure may introduce selection bias

## Files Generated

- **Tables:** `tables/did_*.csv`
- **Figures:** `figures/did_*.png`
- **Report:** `reports/cbd_did_simple_analysis.md`

---

**Analysis conducted by:** Senior Data Scientist  
**Methodology:** Difference-in-Differences Regression  
**Statistical Software:** Python with statsmodels
"""

# Save report
with open('reports/cbd_did_simple_analysis.md', 'w', encoding='utf-8') as f:
    f.write(report_content)

print("✅ Generated comprehensive DiD report: reports/cbd_did_simple_analysis.md")

# ============================================================================
# 6. Final Summary
# ============================================================================

print("\n" + "="*80)
print("🎯 DIFFERENCE-IN-DIFFERENCES REGRESSION ANALYSIS COMPLETE")
print("="*80)

print(f"\n📊 ANALYSIS SUMMARY:")
print(f"   • Total regression models: {len(results_df)}")
print(f"   • Statistically significant effects: {len(significant_results)} ({len(significant_results)/len(results_df)*100:.1f}%)")
print(f"   • Treatment scenarios analyzed: {len(treatments)}")
print(f"   • Outcome variables analyzed: {len(outcomes)}")

print(f"\n📈 KEY FINDINGS:")
if not significant_results.empty:
    for treatment in significant_results['treatment'].unique():
        treatment_sig = significant_results[significant_results['treatment'] == treatment]
        print(f"   • {treatment_sig.iloc[0]['treatment_description']}: {len(treatment_sig)} significant effects")
        
        # Show the most significant effect
        most_sig = treatment_sig.loc[treatment_sig['did_pvalue'].idxmin()]
        print(f"     - Most significant: {most_sig['outcome_description']} (DiD = {most_sig['did_estimate']:.4f}, p = {most_sig['did_pvalue']:.4f})")
else:
    print("   • No statistically significant effects found at 5% level")

print(f"\n📁 OUTPUT FILES:")
print(f"   • Report: reports/cbd_did_simple_analysis.md")
print(f"   • Tables: tables/did_*.csv")
print(f"   • Figures: figures/did_*.png")

print(f"\n✅ DiD REGRESSION ANALYSIS COMPLETE!")
print("   • Standard DiD specification implemented")
print("   • Robust standard errors used")
print("   • Multiple treatment scenarios analyzed")
print("   • Comprehensive results tables and visualizations generated")
print("="*80)
