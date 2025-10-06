#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NYC CBD Congestion Pricing Policy Impact Analysis - Simplified Version
Focus on descriptive statistics and basic visualizations

Author: Senior Data Scientist
Date: 2025-10-06
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
from pathlib import Path
from scipy import stats

# Set random seed for reproducibility
np.random.seed(20251006)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plotting parameters
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Create output directories
for dir_name in ['figures', 'tables', 'reports']:
    Path(dir_name).mkdir(exist_ok=True)

print("🚀 Starting NYC CBD Congestion Pricing Policy Impact Analysis")
print("=" * 60)

# ============================================================================
# 1. Data Loading and Quality Check
# ============================================================================

print("\n📊 1. Data Loading and Quality Check")
print("-" * 40)

# Load data
data_path = '/Users/yanjiechen/Documents/Github/Sharon_local/data/hourly_taxi_summary.csv'
print("Loading data from: {}".format(data_path))

try:
    df = pd.read_csv(data_path)
    print("✅ Data loaded successfully, shape: {}".format(df.shape))
except Exception as e:
    print("❌ Data loading failed: {}".format(e))
    exit(1)

# Parse datetime column
df['pickup_hour'] = pd.to_datetime(df['pickup_hour'])
df = df.sort_values('pickup_hour').reset_index(drop=True)

# Basic info
print("\nData time range: {} to {}".format(df['pickup_hour'].min(), df['pickup_hour'].max()))
print("Total records: {:,}".format(len(df)))

# Create analysis flags
policy_date = pd.Timestamp('2025-01-05')
df['post'] = df['pickup_hour'] >= policy_date
df['hour'] = df['pickup_hour'].dt.hour
df['dow'] = df['pickup_hour'].dt.day_name()
df['is_peak'] = df['hour'].isin([7, 8, 9, 10, 16, 17, 18, 19])
df['is_weekend'] = df['pickup_hour'].dt.dayofweek.isin([5, 6])
df['month'] = df['pickup_hour'].dt.month
df['date'] = df['pickup_hour'].dt.date

print("Pre-policy data: {:,} hours".format((~df['post']).sum()))
print("Post-policy data: {:,} hours".format(df['post'].sum()))

# ============================================================================
# 2. Descriptive Statistics
# ============================================================================

print("\n📊 2. Descriptive Statistics")
print("-" * 40)

# Key metrics
key_metrics = ['avg_speed', 'total_trips', 'avg_duration', 'avg_distance',
               'cbd_inside_ratio', 'cbd_in_ratio', 'cbd_out_ratio', 'cbd_non_ratio',
               'total_revenue', 'avg_fare']

# Overall pre/post comparison
pre_post_stats = df.groupby('post')[key_metrics].mean()
pre_post_stats.index = ['Pre-policy', 'Post-policy']
pct_change = ((pre_post_stats.loc['Post-policy'] / pre_post_stats.loc['Pre-policy'] - 1) * 100).round(2)

print("Overall pre/post comparison:")
comparison_table = pre_post_stats.round(4)
comparison_table.loc['Change (%)'] = pct_change
print(comparison_table)

# Save table
comparison_table.to_csv('tables/pre_post_comparison.csv')

# Peak vs off-peak comparison
peak_comparison = []
for is_peak in [True, False]:
    peak_label = "Peak hours" if is_peak else "Off-peak hours"
    peak_data = df[df['is_peak'] == is_peak]
    
    pre_peak = peak_data[~peak_data['post']][key_metrics].mean()
    post_peak = peak_data[peak_data['post']][key_metrics].mean()
    change_peak = ((post_peak / pre_peak - 1) * 100).round(2)
    
    peak_comparison.append({
        'Period': peak_label,
        'Pre_avg_speed': pre_peak['avg_speed'],
        'Post_avg_speed': post_peak['avg_speed'],
        'avg_speed_change_%': change_peak['avg_speed'],
        'Pre_cbd_inside_ratio': pre_peak['cbd_inside_ratio'],
        'Post_cbd_inside_ratio': post_peak['cbd_inside_ratio'],
        'cbd_inside_ratio_change_%': change_peak['cbd_inside_ratio']
    })

peak_comparison_df = pd.DataFrame(peak_comparison)
print("\nPeak hours comparison:")
print(peak_comparison_df.round(4))
peak_comparison_df.to_csv('tables/peak_comparison.csv', index=False)

# Simple t-tests for statistical significance
print("\n📈 Statistical Tests:")

# Test for speed difference
pre_speed = df[~df['post']]['avg_speed']
post_speed = df[df['post']]['avg_speed']
t_stat_speed, p_val_speed = stats.ttest_ind(pre_speed, post_speed)
print("Average Speed - t-statistic: {:.4f}, p-value: {:.4f}".format(t_stat_speed, p_val_speed))

# Test for CBD inside ratio difference
pre_cbd = df[~df['post']]['cbd_inside_ratio']
post_cbd = df[df['post']]['cbd_inside_ratio']
t_stat_cbd, p_val_cbd = stats.ttest_ind(pre_cbd, post_cbd)
print("CBD Inside Ratio - t-statistic: {:.4f}, p-value: {:.4f}".format(t_stat_cbd, p_val_cbd))

# Test for total trips difference
pre_trips = df[~df['post']]['total_trips']
post_trips = df[df['post']]['total_trips']
t_stat_trips, p_val_trips = stats.ttest_ind(pre_trips, post_trips)
print("Total Trips - t-statistic: {:.4f}, p-value: {:.4f}".format(t_stat_trips, p_val_trips))

# ============================================================================
# 3. Simple Difference-in-Differences
# ============================================================================

print("\n📊 3. Simple Difference-in-Differences Analysis")
print("-" * 40)

# Peak vs Off-peak DiD for avg_speed
print("Peak vs Off-peak DiD for Average Speed:")

# Calculate means for each group
peak_pre = df[(df['is_peak'] == True) & (df['post'] == False)]['avg_speed'].mean()
peak_post = df[(df['is_peak'] == True) & (df['post'] == True)]['avg_speed'].mean()
offpeak_pre = df[(df['is_peak'] == False) & (df['post'] == False)]['avg_speed'].mean()
offpeak_post = df[(df['is_peak'] == False) & (df['post'] == True)]['avg_speed'].mean()

# Calculate DiD estimate
peak_change = peak_post - peak_pre
offpeak_change = offpeak_post - offpeak_pre
did_estimate = peak_change - offpeak_change

print("Peak hours - Pre: {:.3f}, Post: {:.3f}, Change: {:.3f}".format(peak_pre, peak_post, peak_change))
print("Off-peak hours - Pre: {:.3f}, Post: {:.3f}, Change: {:.3f}".format(offpeak_pre, offpeak_post, offpeak_change))
print("DiD Estimate (Peak effect): {:.3f} mph".format(did_estimate))

# CBD exposure analysis
print("\nCBD Exposure Analysis:")
df['cbd_exposure'] = df['cbd_in_ratio'] + df['cbd_out_ratio'] + df['cbd_inside_ratio']

# Divide into high and low CBD exposure
cbd_median = df['cbd_exposure'].median()
df['high_cbd_exposure'] = df['cbd_exposure'] > cbd_median

# Calculate DiD for CBD exposure
high_cbd_pre = df[(df['high_cbd_exposure'] == True) & (df['post'] == False)]['avg_speed'].mean()
high_cbd_post = df[(df['high_cbd_exposure'] == True) & (df['post'] == True)]['avg_speed'].mean()
low_cbd_pre = df[(df['high_cbd_exposure'] == False) & (df['post'] == False)]['avg_speed'].mean()
low_cbd_post = df[(df['high_cbd_exposure'] == False) & (df['post'] == True)]['avg_speed'].mean()

high_cbd_change = high_cbd_post - high_cbd_pre
low_cbd_change = low_cbd_post - low_cbd_pre
cbd_did_estimate = high_cbd_change - low_cbd_change

print("High CBD exposure - Pre: {:.3f}, Post: {:.3f}, Change: {:.3f}".format(high_cbd_pre, high_cbd_post, high_cbd_change))
print("Low CBD exposure - Pre: {:.3f}, Post: {:.3f}, Change: {:.3f}".format(low_cbd_pre, low_cbd_post, low_cbd_change))
print("DiD Estimate (CBD exposure effect): {:.3f} mph".format(cbd_did_estimate))

# ============================================================================
# 4. Visualizations
# ============================================================================

print("\n📊 4. Creating Visualizations")
print("-" * 40)

# Create daily aggregated data for cleaner time series
print("Creating daily aggregated data for cleaner visualization...")
df['date'] = df['pickup_hour'].dt.date
daily_df = df.groupby('date').agg({
    'avg_speed': 'mean',
    'total_trips': 'sum',
    'cbd_inside_ratio': 'mean',
    'cbd_in_ratio': 'mean', 
    'cbd_out_ratio': 'mean',
    'cbd_non_ratio': 'mean',
    'pickup_hour': 'first'  # Keep one datetime for plotting
}).reset_index()

# Convert date back to datetime for plotting
daily_df['date_dt'] = pd.to_datetime(daily_df['date'])

# Create separate plots for better readability
# Plot 1: Daily Time Series Overview
fig, axes = plt.subplots(2, 2, figsize=(20, 12))

# Plot 1: Average Speed (Daily)
axes[0, 0].plot(daily_df['date_dt'], daily_df['avg_speed'], linewidth=2, color='blue', alpha=0.8)
axes[0, 0].axvline(policy_date, color='red', linestyle='--', linewidth=3, label='Policy Implementation')
axes[0, 0].set_title('Daily Average Speed Time Series', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Speed (mph)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].tick_params(axis='x', rotation=45)

# Plot 2: Total Trips (Daily)
axes[0, 1].plot(daily_df['date_dt'], daily_df['total_trips'], linewidth=2, color='green', alpha=0.8)
axes[0, 1].axvline(policy_date, color='red', linestyle='--', linewidth=3, label='Policy Implementation')
axes[0, 1].set_title('Daily Total Trips Time Series', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('Total Trips per Day')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].tick_params(axis='x', rotation=45)

# Plot 3: CBD Inside Ratio (Daily)
axes[1, 0].plot(daily_df['date_dt'], daily_df['cbd_inside_ratio'], linewidth=2, color='orange', alpha=0.8)
axes[1, 0].axvline(policy_date, color='red', linestyle='--', linewidth=3, label='Policy Implementation')
axes[1, 0].set_title('Daily CBD Inside Ratio Time Series', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('CBD Inside Ratio')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].tick_params(axis='x', rotation=45)

# Plot 4: CBD Interaction Ratios (Daily Stacked Area)
cbd_ratios = ['cbd_inside_ratio', 'cbd_in_ratio', 'cbd_out_ratio', 'cbd_non_ratio']
colors = ['#ff7f0e', '#2ca02c', '#d62728', '#1f77b4']
labels = ['CBD Inside', 'CBD In', 'CBD Out', 'CBD Non']

# Create stacked area plot
bottom = np.zeros(len(daily_df))
for i, ratio in enumerate(cbd_ratios):
    axes[1, 1].fill_between(daily_df['date_dt'], bottom, bottom + daily_df[ratio], 
                           alpha=0.8, label=labels[i], color=colors[i])
    bottom += daily_df[ratio]

axes[1, 1].axvline(policy_date, color='black', linestyle='--', linewidth=3, label='Policy Implementation')
axes[1, 1].set_title('Daily CBD Interaction Ratios', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Cumulative Ratio')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('figures/time_series_overview_daily.png', dpi=300, bbox_inches='tight')
plt.show()

# Create weekly moving averages for even smoother trends
print("Creating weekly moving averages for trend analysis...")
daily_df_sorted = daily_df.sort_values('date_dt')
window_size = 7  # 7-day moving average

daily_df_sorted['speed_ma7'] = daily_df_sorted['avg_speed'].rolling(window=window_size, center=True).mean()
daily_df_sorted['trips_ma7'] = daily_df_sorted['total_trips'].rolling(window=window_size, center=True).mean()
daily_df_sorted['cbd_inside_ma7'] = daily_df_sorted['cbd_inside_ratio'].rolling(window=window_size, center=True).mean()

# Plot 2: Weekly Moving Averages
fig, axes = plt.subplots(1, 3, figsize=(21, 7))

# Speed with moving average
axes[0].plot(daily_df_sorted['date_dt'], daily_df_sorted['avg_speed'], alpha=0.3, color='lightblue', label='Daily')
axes[0].plot(daily_df_sorted['date_dt'], daily_df_sorted['speed_ma7'], linewidth=3, color='blue', label='7-day MA')
axes[0].axvline(policy_date, color='red', linestyle='--', linewidth=3, label='Policy Implementation')
axes[0].set_title('Average Speed: Daily vs 7-Day Moving Average', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Speed (mph)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(axis='x', rotation=45)

# Trips with moving average
axes[1].plot(daily_df_sorted['date_dt'], daily_df_sorted['total_trips'], alpha=0.3, color='lightgreen', label='Daily')
axes[1].plot(daily_df_sorted['date_dt'], daily_df_sorted['trips_ma7'], linewidth=3, color='green', label='7-day MA')
axes[1].axvline(policy_date, color='red', linestyle='--', linewidth=3, label='Policy Implementation')
axes[1].set_title('Total Trips: Daily vs 7-Day Moving Average', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Total Trips per Day')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(axis='x', rotation=45)

# CBD Inside Ratio with moving average
axes[2].plot(daily_df_sorted['date_dt'], daily_df_sorted['cbd_inside_ratio'], alpha=0.3, color='lightsalmon', label='Daily')
axes[2].plot(daily_df_sorted['date_dt'], daily_df_sorted['cbd_inside_ma7'], linewidth=3, color='orange', label='7-day MA')
axes[2].axvline(policy_date, color='red', linestyle='--', linewidth=3, label='Policy Implementation')
axes[2].set_title('CBD Inside Ratio: Daily vs 7-Day Moving Average', fontsize=14, fontweight='bold')
axes[2].set_ylabel('CBD Inside Ratio')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('figures/time_series_moving_averages.png', dpi=300, bbox_inches='tight')
plt.show()

# Create monthly aggregation for long-term trends
print("Creating monthly aggregated data for long-term trend analysis...")
df['year_month'] = df['pickup_hour'].dt.to_period('M')
monthly_df = df.groupby('year_month').agg({
    'avg_speed': 'mean',
    'total_trips': 'sum',
    'cbd_inside_ratio': 'mean',
    'cbd_in_ratio': 'mean',
    'cbd_out_ratio': 'mean',
    'cbd_non_ratio': 'mean'
}).reset_index()

# Convert period to datetime for plotting
monthly_df['date_dt'] = monthly_df['year_month'].dt.to_timestamp()

# Plot 3: Monthly Trends
fig, axes = plt.subplots(1, 3, figsize=(21, 7))

# Monthly Speed
axes[0].plot(monthly_df['date_dt'], monthly_df['avg_speed'], 'o-', linewidth=3, markersize=8, color='blue')
axes[0].axvline(policy_date, color='red', linestyle='--', linewidth=3, label='Policy Implementation')
axes[0].set_title('Monthly Average Speed Trend', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Speed (mph)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(axis='x', rotation=45)

# Monthly Trips
axes[1].plot(monthly_df['date_dt'], monthly_df['total_trips'], 'o-', linewidth=3, markersize=8, color='green')
axes[1].axvline(policy_date, color='red', linestyle='--', linewidth=3, label='Policy Implementation')
axes[1].set_title('Monthly Total Trips Trend', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Total Trips per Month')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(axis='x', rotation=45)

# Monthly CBD Inside Ratio
axes[2].plot(monthly_df['date_dt'], monthly_df['cbd_inside_ratio'], 'o-', linewidth=3, markersize=8, color='orange')
axes[2].axvline(policy_date, color='red', linestyle='--', linewidth=3, label='Policy Implementation')
axes[2].set_title('Monthly CBD Inside Ratio Trend', fontsize=14, fontweight='bold')
axes[2].set_ylabel('CBD Inside Ratio')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('figures/time_series_monthly_trends.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Created multiple time series visualizations:")
print("   - Daily aggregated time series")
print("   - 7-day moving averages")
print("   - Monthly trend analysis")

# Box plots for before/after comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Speed comparison
speed_data = [df[~df['post']]['avg_speed'], df[df['post']]['avg_speed']]
axes[0].boxplot(speed_data, labels=['Pre-policy', 'Post-policy'])
axes[0].set_title('Average Speed Distribution')
axes[0].set_ylabel('Speed (mph)')
axes[0].grid(True, alpha=0.3)

# Trips comparison
trips_data = [df[~df['post']]['total_trips'], df[df['post']]['total_trips']]
axes[1].boxplot(trips_data, labels=['Pre-policy', 'Post-policy'])
axes[1].set_title('Total Trips Distribution')
axes[1].set_ylabel('Trips per Hour')
axes[1].grid(True, alpha=0.3)

# CBD inside ratio comparison
cbd_data = [df[~df['post']]['cbd_inside_ratio'], df[df['post']]['cbd_inside_ratio']]
axes[2].boxplot(cbd_data, labels=['Pre-policy', 'Post-policy'])
axes[2].set_title('CBD Inside Ratio Distribution')
axes[2].set_ylabel('Ratio')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/distribution_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 5. Generate Report
# ============================================================================

print("\n📊 5. Generating Analysis Report")
print("-" * 40)

# Create executive summary
exec_summary = {
    'speed_change_pct': pct_change['avg_speed'],
    'trips_change_pct': pct_change['total_trips'],
    'cbd_inside_change_pct': pct_change['cbd_inside_ratio'],
    'did_peak_estimate': did_estimate,
    'did_cbd_estimate': cbd_did_estimate,
    'speed_pvalue': p_val_speed,
    'cbd_pvalue': p_val_cbd,
    'trips_pvalue': p_val_trips
}

# Create markdown report
report_content = f"""# NYC CBD Congestion Pricing Policy Impact Analysis - Simplified

**Analysis Date:** {datetime.now().strftime('%Y-%m-%d')}  
**Policy Implementation Date:** January 5, 2025  
**Data Source:** NYC Yellow Taxi Hourly Summary Data  

## Executive Summary

### Key Findings

1. **Average Speed Impact:**
   - Overall change: {exec_summary['speed_change_pct']:.2f}%
   - Statistical significance: {'Significant' if exec_summary['speed_pvalue'] < 0.05 else 'Not significant'} (p = {exec_summary['speed_pvalue']:.4f})
   - Peak hours DiD effect: {exec_summary['did_peak_estimate']:.4f} mph

2. **Trip Volume Impact:**
   - Total trips change: {exec_summary['trips_change_pct']:.2f}%
   - Statistical significance: {'Significant' if exec_summary['trips_pvalue'] < 0.05 else 'Not significant'} (p = {exec_summary['trips_pvalue']:.4f})

3. **CBD Usage Patterns:**
   - CBD internal trips change: {exec_summary['cbd_inside_change_pct']:.2f}%
   - Statistical significance: {'Significant' if exec_summary['cbd_pvalue'] < 0.05 else 'Not significant'} (p = {exec_summary['cbd_pvalue']:.4f})
   - CBD exposure DiD effect: {exec_summary['did_cbd_estimate']:.4f} mph

## Data Description

- **Dataset:** Hourly aggregated NYC Yellow Taxi data
- **Time Period:** {df['pickup_hour'].min().strftime('%Y-%m-%d')} to {df['pickup_hour'].max().strftime('%Y-%m-%d')}
- **Total Observations:** {len(df):,} hours
- **Pre-policy Period:** {(~df['post']).sum():,} hours
- **Post-policy Period:** {df['post'].sum():,} hours

## Methodology

### 1. Descriptive Analysis
- Simple before/after comparisons
- Peak vs off-peak hour analysis
- Statistical t-tests for significance

### 2. Difference-in-Differences
- Peak intensity comparison (peak vs off-peak hours)
- CBD exposure comparison (high vs low CBD interaction)

## Results

### Pre/Post Comparison
{comparison_table.to_string()}

### Peak vs Off-Peak Analysis
{peak_comparison_df.to_string()}

## Policy Impact Assessment

### Speed Effects
The analysis shows {'statistically significant' if exec_summary['speed_pvalue'] < 0.05 else 'no statistically significant'} changes in average speeds following the CBD congestion pricing implementation. The overall speed change was {exec_summary['speed_change_pct']:.2f}%.

### CBD Usage Patterns
CBD internal trip ratios show a {'decrease' if exec_summary['cbd_inside_change_pct'] < 0 else 'increase'} of {abs(exec_summary['cbd_inside_change_pct']):.2f}%, which is {'statistically significant' if exec_summary['cbd_pvalue'] < 0.05 else 'not statistically significant'}.

### Trip Volume Effects
Overall taxi trip volumes {'decreased' if exec_summary['trips_change_pct'] < 0 else 'increased'} by {abs(exec_summary['trips_change_pct']):.2f}%, suggesting {'demand reduction' if exec_summary['trips_change_pct'] < 0 else 'increased demand'}.

## Difference-in-Differences Results

1. **Peak Hours Effect:** {exec_summary['did_peak_estimate']:.4f} mph differential impact during peak hours
2. **CBD Exposure Effect:** {exec_summary['did_cbd_estimate']:.4f} mph differential impact for high CBD exposure areas

## Conclusions

Based on the simplified analysis of NYC Yellow Taxi hourly data:

1. **Speed Effects:** {'Evidence of speed improvements' if exec_summary['speed_change_pct'] > 0 else 'No evidence of speed improvements'} following policy implementation
2. **Usage Patterns:** {'Evidence of reduced CBD usage' if exec_summary['cbd_inside_change_pct'] < 0 else 'No evidence of reduced CBD usage'} as intended by the policy
3. **Overall Impact:** The policy shows {'mixed results' if abs(exec_summary['speed_change_pct']) < 1 else 'clear impacts'} in the initial implementation period

## Limitations

1. **Simplified Analysis:** This analysis uses basic statistical methods and may not capture complex causal relationships
2. **Confounding Factors:** Weather, holidays, and other concurrent changes not controlled for
3. **Short-term Data:** Limited post-policy observation period
4. **Data Scope:** Only Yellow Taxi data, excluding other transportation modes

---

**Analysis conducted by:** Yanjie Chen
**Files generated:** Tables in `tables/`, Figures in `figures/`
"""

# Save report
with open('reports/cbd_taxi_simplified_analysis.md', 'w', encoding='utf-8') as f:
    f.write(report_content)

# Save executive summary as CSV
exec_summary_df = pd.DataFrame([exec_summary])
exec_summary_df.to_csv('tables/executive_summary.csv', index=False)

print("✅ Report generated: reports/cbd_taxi_simplified_analysis.md")

# ============================================================================
# Final Executive Summary to Console
# ============================================================================

print("\n" + "="*80)
print("🎯 EXECUTIVE SUMMARY - NYC CBD CONGESTION PRICING ANALYSIS")
print("="*80)

print(f"\n📊 DATA OVERVIEW:")
print(f"   • Time Period: {df['pickup_hour'].min().strftime('%Y-%m-%d')} to {df['pickup_hour'].max().strftime('%Y-%m-%d')}")
print(f"   • Total Hours: {len(df):,}")
print(f"   • Pre-policy: {(~df['post']).sum():,} hours | Post-policy: {df['post'].sum():,} hours")

print(f"\n🚗 KEY FINDINGS:")
print(f"   • Average Speed: {exec_summary['speed_change_pct']:+.2f}% change (p = {exec_summary['speed_pvalue']:.4f})")
print(f"   • Total Trips: {exec_summary['trips_change_pct']:+.2f}% change (p = {exec_summary['trips_pvalue']:.4f})")
print(f"   • CBD Internal Trips: {exec_summary['cbd_inside_change_pct']:+.2f}% change (p = {exec_summary['cbd_pvalue']:.4f})")

print(f"\n📈 DIFFERENCE-IN-DIFFERENCES:")
print(f"   • Peak Hours Effect: {exec_summary['did_peak_estimate']:+.4f} mph")
print(f"   • CBD Exposure Effect: {exec_summary['did_cbd_estimate']:+.4f} mph")

print(f"\n📁 OUTPUT FILES:")
print(f"   • Tables: tables/")
print(f"   • Figures: figures/")
print(f"   • Report: reports/cbd_taxi_simplified_analysis.md")

print(f"\n✅ SIMPLIFIED ANALYSIS COMPLETE!")
print("="*80)
