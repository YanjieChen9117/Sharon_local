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
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，不显示窗口
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, date
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
# For each hour, we need to combine Yellow and Green taxi data
# Weighted average for ratios and speeds, sum for counts and revenues

# Group by pickup_hour
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
print("   Each hour now has 1 row combining Yellow and Green taxi data")
print("   Yellow taxi average proportion: {:.2%}".format(df['yellow_ratio'].mean()))
print("   Green taxi average proportion: {:.2%}".format(df['green_ratio'].mean()))

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

# 定义2024-2025年美国主要假期
holidays_2024_2025 = [
    # 2024年假期
    date(2024, 1, 1),   # New Year's Day
    date(2024, 1, 15),  # Martin Luther King Jr. Day
    date(2024, 2, 19),  # Presidents' Day
    date(2024, 5, 27),  # Memorial Day
    date(2024, 6, 19),  # Juneteenth
    date(2024, 7, 4),   # Independence Day
    date(2024, 9, 2),   # Labor Day
    date(2024, 10, 14), # Columbus Day
    date(2024, 11, 11), # Veterans Day
    date(2024, 11, 28), # Thanksgiving Day
    date(2024, 12, 25), # Christmas Day
    
    # 2025年假期
    date(2025, 1, 1),   # New Year's Day
    date(2025, 1, 20),  # Martin Luther King Jr. Day
    date(2025, 2, 17),  # Presidents' Day
    date(2025, 5, 26),  # Memorial Day
    date(2025, 6, 19),  # Juneteenth
    date(2025, 7, 4),   # Independence Day
    date(2025, 9, 1),   # Labor Day
    date(2025, 10, 13), # Columbus Day
    date(2025, 11, 11), # Veterans Day
    date(2025, 11, 27), # Thanksgiving Day
    date(2025, 12, 25), # Christmas Day
]

# 创建假期标识
df['is_holiday'] = df['date'].isin(holidays_2024_2025)

# 创建时段分类：工作日、周末、假期
def get_day_type(row):
    if row['is_holiday']:
        return 'Holiday'
    elif row['is_weekend']:
        return 'Weekend'
    else:
        return 'Weekday'

df['day_type'] = df.apply(get_day_type, axis=1)

print("📅 Day type distribution:")
print(df['day_type'].value_counts())
print(f"Holiday days: {df['is_holiday'].sum()} hours")
print(f"Weekend days: {df['is_weekend'].sum()} hours")
print(f"Weekday days: {(~df['is_weekend'] & ~df['is_holiday']).sum()} hours")

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
               'cbd_neighbor_inside_ratio', 'cbd_neighbor_in_ratio', 'cbd_neighbor_out_ratio', 'cbd_neighbor_non_ratio',
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

# Day type comparison (工作日、周末、假期)
print("\n📅 Day Type Analysis (Weekday/Weekend/Holiday)")
print("-" * 50)

day_type_comparison = []
for day_type in ['Weekday', 'Weekend', 'Holiday']:
    day_data = df[df['day_type'] == day_type]
    
    if len(day_data) > 0:  # 确保有数据
        pre_day = day_data[~day_data['post']][key_metrics].mean()
        post_day = day_data[day_data['post']][key_metrics].mean()
        change_day = ((post_day / pre_day - 1) * 100).round(2)
        
        day_type_comparison.append({
            'Day_Type': day_type,
            'Pre_avg_speed': pre_day['avg_speed'],
            'Post_avg_speed': post_day['avg_speed'],
            'avg_speed_change_%': change_day['avg_speed'],
            'Pre_total_trips': pre_day['total_trips'],
            'Post_total_trips': post_day['total_trips'],
            'total_trips_change_%': change_day['total_trips'],
            'Pre_cbd_inside_ratio': pre_day['cbd_inside_ratio'],
            'Post_cbd_inside_ratio': post_day['cbd_inside_ratio'],
            'cbd_inside_ratio_change_%': change_day['cbd_inside_ratio'],
            'Pre_cbd_neighbor_in_ratio': pre_day['cbd_neighbor_in_ratio'],
            'Post_cbd_neighbor_in_ratio': post_day['cbd_neighbor_in_ratio'],
            'cbd_neighbor_in_ratio_change_%': change_day['cbd_neighbor_in_ratio'],
            'Pre_cbd_neighbor_out_ratio': pre_day['cbd_neighbor_out_ratio'],
            'Post_cbd_neighbor_out_ratio': post_day['cbd_neighbor_out_ratio'],
            'cbd_neighbor_out_ratio_change_%': change_day['cbd_neighbor_out_ratio'],
            'Sample_Size_Pre': (~day_data['post']).sum(),
            'Sample_Size_Post': day_data['post'].sum()
        })

day_type_comparison_df = pd.DataFrame(day_type_comparison)
print("\nDay type comparison:")
print(day_type_comparison_df.round(4))
day_type_comparison_df.to_csv('tables/day_type_comparison.csv', index=False)

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

# Test for CBD neighbor in ratio difference (substitution effect)
pre_neighbor_in = df[~df['post']]['cbd_neighbor_in_ratio']
post_neighbor_in = df[df['post']]['cbd_neighbor_in_ratio']
t_stat_neighbor_in, p_val_neighbor_in = stats.ttest_ind(pre_neighbor_in, post_neighbor_in)
print("CBD Neighbor In Ratio - t-statistic: {:.4f}, p-value: {:.4f}".format(t_stat_neighbor_in, p_val_neighbor_in))

# Test for CBD neighbor out ratio difference (substitution effect)
pre_neighbor_out = df[~df['post']]['cbd_neighbor_out_ratio']
post_neighbor_out = df[df['post']]['cbd_neighbor_out_ratio']
t_stat_neighbor_out, p_val_neighbor_out = stats.ttest_ind(pre_neighbor_out, post_neighbor_out)
print("CBD Neighbor Out Ratio - t-statistic: {:.4f}, p-value: {:.4f}".format(t_stat_neighbor_out, p_val_neighbor_out))

# Statistical tests by day type
print("\n📈 Statistical Tests by Day Type:")
print("-" * 40)

day_type_tests = {}
for day_type in ['Weekday', 'Weekend', 'Holiday']:
    day_data = df[df['day_type'] == day_type]
    
    if len(day_data) > 50:  # 确保有足够的数据进行统计检验
        print(f"\n{day_type} Analysis (n={len(day_data)} hours):")
        
        # Speed test
        pre_speed = day_data[~day_data['post']]['avg_speed']
        post_speed = day_data[day_data['post']]['avg_speed']
        if len(pre_speed) > 1 and len(post_speed) > 1:
            t_stat_speed, p_val_speed = stats.ttest_ind(pre_speed, post_speed)
            print("  Average Speed - t-statistic: {:.4f}, p-value: {:.4f}".format(t_stat_speed, p_val_speed))
            day_type_tests[f'{day_type}_speed'] = {'t_stat': t_stat_speed, 'p_val': p_val_speed}
        
        # CBD inside ratio test
        pre_cbd = day_data[~day_data['post']]['cbd_inside_ratio']
        post_cbd = day_data[day_data['post']]['cbd_inside_ratio']
        if len(pre_cbd) > 1 and len(post_cbd) > 1:
            t_stat_cbd, p_val_cbd = stats.ttest_ind(pre_cbd, post_cbd)
            print("  CBD Inside Ratio - t-statistic: {:.4f}, p-value: {:.4f}".format(t_stat_cbd, p_val_cbd))
            day_type_tests[f'{day_type}_cbd'] = {'t_stat': t_stat_cbd, 'p_val': p_val_cbd}
        
        # Total trips test
        pre_trips = day_data[~day_data['post']]['total_trips']
        post_trips = day_data[day_data['post']]['total_trips']
        if len(pre_trips) > 1 and len(post_trips) > 1:
            t_stat_trips, p_val_trips = stats.ttest_ind(pre_trips, post_trips)
            print("  Total Trips - t-statistic: {:.4f}, p-value: {:.4f}".format(t_stat_trips, p_val_trips))
            day_type_tests[f'{day_type}_trips'] = {'t_stat': t_stat_trips, 'p_val': p_val_trips}
        
        # CBD neighbor substitution tests
        pre_neighbor_in = day_data[~day_data['post']]['cbd_neighbor_in_ratio']
        post_neighbor_in = day_data[day_data['post']]['cbd_neighbor_in_ratio']
        if len(pre_neighbor_in) > 1 and len(post_neighbor_in) > 1:
            t_stat_neighbor_in, p_val_neighbor_in = stats.ttest_ind(pre_neighbor_in, post_neighbor_in)
            print("  CBD Neighbor In - t-statistic: {:.4f}, p-value: {:.4f}".format(t_stat_neighbor_in, p_val_neighbor_in))
            day_type_tests[f'{day_type}_neighbor_in'] = {'t_stat': t_stat_neighbor_in, 'p_val': p_val_neighbor_in}
        
        pre_neighbor_out = day_data[~day_data['post']]['cbd_neighbor_out_ratio']
        post_neighbor_out = day_data[day_data['post']]['cbd_neighbor_out_ratio']
        if len(pre_neighbor_out) > 1 and len(post_neighbor_out) > 1:
            t_stat_neighbor_out, p_val_neighbor_out = stats.ttest_ind(pre_neighbor_out, post_neighbor_out)
            print("  CBD Neighbor Out - t-statistic: {:.4f}, p-value: {:.4f}".format(t_stat_neighbor_out, p_val_neighbor_out))
            day_type_tests[f'{day_type}_neighbor_out'] = {'t_stat': t_stat_neighbor_out, 'p_val': p_val_neighbor_out}
    else:
        print(f"\n{day_type}: Insufficient data for statistical testing (n={len(day_data)})")

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

# Day type DiD analysis
print("\nDay Type Difference-in-Differences Analysis:")
print("-" * 50)

# Weekday vs Weekend DiD for avg_speed
print("Weekday vs Weekend DiD for Average Speed:")

weekday_pre = df[(df['day_type'] == 'Weekday') & (df['post'] == False)]['avg_speed'].mean()
weekday_post = df[(df['day_type'] == 'Weekday') & (df['post'] == True)]['avg_speed'].mean()
weekend_pre = df[(df['day_type'] == 'Weekend') & (df['post'] == False)]['avg_speed'].mean()
weekend_post = df[(df['day_type'] == 'Weekend') & (df['post'] == True)]['avg_speed'].mean()

weekday_change = weekday_post - weekday_pre
weekend_change = weekend_post - weekend_pre
weekday_weekend_did = weekday_change - weekend_change

print("Weekday - Pre: {:.3f}, Post: {:.3f}, Change: {:.3f}".format(weekday_pre, weekday_post, weekday_change))
print("Weekend - Pre: {:.3f}, Post: {:.3f}, Change: {:.3f}".format(weekend_pre, weekend_post, weekend_change))
print("DiD Estimate (Weekday effect): {:.3f} mph".format(weekday_weekend_did))

# Weekday vs Holiday DiD for avg_speed (if sufficient holiday data)
if df[df['day_type'] == 'Holiday'].shape[0] > 100:
    print("\nWeekday vs Holiday DiD for Average Speed:")
    
    holiday_pre = df[(df['day_type'] == 'Holiday') & (df['post'] == False)]['avg_speed'].mean()
    holiday_post = df[(df['day_type'] == 'Holiday') & (df['post'] == True)]['avg_speed'].mean()
    
    holiday_change = holiday_post - holiday_pre
    weekday_holiday_did = weekday_change - holiday_change
    
    print("Holiday - Pre: {:.3f}, Post: {:.3f}, Change: {:.3f}".format(holiday_pre, holiday_post, holiday_change))
    print("DiD Estimate (Weekday vs Holiday effect): {:.3f} mph".format(weekday_holiday_did))
else:
    print("\nInsufficient holiday data for DiD analysis")
    weekday_holiday_did = None

# CBD usage DiD by day type
print("\nCBD Inside Ratio DiD by Day Type:")

weekday_cbd_pre = df[(df['day_type'] == 'Weekday') & (df['post'] == False)]['cbd_inside_ratio'].mean()
weekday_cbd_post = df[(df['day_type'] == 'Weekday') & (df['post'] == True)]['cbd_inside_ratio'].mean()
weekend_cbd_pre = df[(df['day_type'] == 'Weekend') & (df['post'] == False)]['cbd_inside_ratio'].mean()
weekend_cbd_post = df[(df['day_type'] == 'Weekend') & (df['post'] == True)]['cbd_inside_ratio'].mean()

weekday_cbd_change = weekday_cbd_post - weekday_cbd_pre
weekend_cbd_change = weekend_cbd_post - weekend_cbd_pre
cbd_daytype_did = weekday_cbd_change - weekend_cbd_change

print("Weekday CBD - Pre: {:.4f}, Post: {:.4f}, Change: {:.4f}".format(weekday_cbd_pre, weekday_cbd_post, weekday_cbd_change))
print("Weekend CBD - Pre: {:.4f}, Post: {:.4f}, Change: {:.4f}".format(weekend_cbd_pre, weekend_cbd_post, weekend_cbd_change))
print("DiD Estimate (CBD usage weekday effect): {:.4f}".format(cbd_daytype_did))

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
    'cbd_neighbor_inside_ratio': 'mean',
    'cbd_neighbor_in_ratio': 'mean',
    'cbd_neighbor_out_ratio': 'mean',
    'cbd_neighbor_non_ratio': 'mean',
    'pickup_hour': 'first',  # Keep one datetime for plotting
    'day_type': 'first',     # Keep day type for coloring
    'is_holiday': 'first',   # Keep holiday flag
    'is_weekend': 'first'    # Keep weekend flag
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
plt.close()

# Create CBD Neighbor visualization
print("Creating CBD Neighbor interaction visualization...")
fig, axes = plt.subplots(2, 2, figsize=(20, 12))

# Plot 1: CBD Neighbor In Ratio (Daily)
axes[0, 0].plot(daily_df['date_dt'], daily_df['cbd_neighbor_in_ratio'], linewidth=2, color='purple', alpha=0.8)
axes[0, 0].axvline(policy_date, color='red', linestyle='--', linewidth=3, label='Policy Implementation')
axes[0, 0].set_title('Daily CBD Neighbor In Ratio Time Series', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('CBD Neighbor In Ratio')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].tick_params(axis='x', rotation=45)

# Plot 2: CBD Neighbor Out Ratio (Daily)
axes[0, 1].plot(daily_df['date_dt'], daily_df['cbd_neighbor_out_ratio'], linewidth=2, color='brown', alpha=0.8)
axes[0, 1].axvline(policy_date, color='red', linestyle='--', linewidth=3, label='Policy Implementation')
axes[0, 1].set_title('Daily CBD Neighbor Out Ratio Time Series', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('CBD Neighbor Out Ratio')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].tick_params(axis='x', rotation=45)

# Plot 3: CBD Neighbor Inside Ratio (Daily)
axes[1, 0].plot(daily_df['date_dt'], daily_df['cbd_neighbor_inside_ratio'], linewidth=2, color='teal', alpha=0.8)
axes[1, 0].axvline(policy_date, color='red', linestyle='--', linewidth=3, label='Policy Implementation')
axes[1, 0].set_title('Daily CBD Neighbor Inside Ratio Time Series', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('CBD Neighbor Inside Ratio')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].tick_params(axis='x', rotation=45)

# Plot 4: CBD Neighbor Interaction Ratios (Daily Stacked Area)
neighbor_ratios = ['cbd_neighbor_inside_ratio', 'cbd_neighbor_in_ratio', 'cbd_neighbor_out_ratio', 'cbd_neighbor_non_ratio']
neighbor_colors = ['#17becf', '#9467bd', '#8c564b', '#e377c2']
neighbor_labels = ['CBD Neighbor Inside', 'CBD Neighbor In', 'CBD Neighbor Out', 'CBD Neighbor Non']

# Create stacked area plot
bottom = np.zeros(len(daily_df))
for i, ratio in enumerate(neighbor_ratios):
    axes[1, 1].fill_between(daily_df['date_dt'], bottom, bottom + daily_df[ratio], 
                           alpha=0.8, label=neighbor_labels[i], color=neighbor_colors[i])
    bottom += daily_df[ratio]

axes[1, 1].axvline(policy_date, color='black', linestyle='--', linewidth=3, label='Policy Implementation')
axes[1, 1].set_title('Daily CBD Neighbor Interaction Ratios', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Cumulative Ratio')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('figures/cbd_neighbor_time_series.png', dpi=300, bbox_inches='tight')
plt.close()

# Create weekly moving averages for even smoother trends
print("Creating weekly moving averages for trend analysis...")
daily_df_sorted = daily_df.sort_values('date_dt')
window_size = 7  # 7-day moving average

daily_df_sorted['speed_ma7'] = daily_df_sorted['avg_speed'].rolling(window=window_size, center=True).mean()
daily_df_sorted['trips_ma7'] = daily_df_sorted['total_trips'].rolling(window=window_size, center=True).mean()
daily_df_sorted['cbd_inside_ma7'] = daily_df_sorted['cbd_inside_ratio'].rolling(window=window_size, center=True).mean()
daily_df_sorted['cbd_neighbor_in_ma7'] = daily_df_sorted['cbd_neighbor_in_ratio'].rolling(window=window_size, center=True).mean()
daily_df_sorted['cbd_neighbor_out_ma7'] = daily_df_sorted['cbd_neighbor_out_ratio'].rolling(window=window_size, center=True).mean()

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
plt.close()

# CBD Neighbor Moving Averages
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# CBD Neighbor In with moving average
axes[0].plot(daily_df_sorted['date_dt'], daily_df_sorted['cbd_neighbor_in_ratio'], alpha=0.3, color='lavender', label='Daily')
axes[0].plot(daily_df_sorted['date_dt'], daily_df_sorted['cbd_neighbor_in_ma7'], linewidth=3, color='purple', label='7-day MA')
axes[0].axvline(policy_date, color='red', linestyle='--', linewidth=3, label='Policy Implementation')
axes[0].set_title('CBD Neighbor In Ratio: Daily vs 7-Day Moving Average', fontsize=14, fontweight='bold')
axes[0].set_ylabel('CBD Neighbor In Ratio')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(axis='x', rotation=45)

# CBD Neighbor Out with moving average
axes[1].plot(daily_df_sorted['date_dt'], daily_df_sorted['cbd_neighbor_out_ratio'], alpha=0.3, color='wheat', label='Daily')
axes[1].plot(daily_df_sorted['date_dt'], daily_df_sorted['cbd_neighbor_out_ma7'], linewidth=3, color='brown', label='7-day MA')
axes[1].axvline(policy_date, color='red', linestyle='--', linewidth=3, label='Policy Implementation')
axes[1].set_title('CBD Neighbor Out Ratio: Daily vs 7-Day Moving Average', fontsize=14, fontweight='bold')
axes[1].set_ylabel('CBD Neighbor Out Ratio')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('figures/cbd_neighbor_moving_averages.png', dpi=300, bbox_inches='tight')
plt.close()

# Create monthly aggregation for long-term trends
print("Creating monthly aggregated data for long-term trend analysis...")
df['year_month'] = df['pickup_hour'].dt.to_period('M')
monthly_df = df.groupby('year_month').agg({
    'avg_speed': 'mean',
    'total_trips': 'sum',
    'cbd_inside_ratio': 'mean',
    'cbd_in_ratio': 'mean',
    'cbd_out_ratio': 'mean',
    'cbd_non_ratio': 'mean',
    'cbd_neighbor_inside_ratio': 'mean',
    'cbd_neighbor_in_ratio': 'mean',
    'cbd_neighbor_out_ratio': 'mean',
    'cbd_neighbor_non_ratio': 'mean'
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
plt.close()

# CBD Neighbor Monthly Trends
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Monthly CBD Neighbor In Ratio
axes[0].plot(monthly_df['date_dt'], monthly_df['cbd_neighbor_in_ratio'], 'o-', linewidth=3, markersize=8, color='purple')
axes[0].axvline(policy_date, color='red', linestyle='--', linewidth=3, label='Policy Implementation')
axes[0].set_title('Monthly CBD Neighbor In Ratio Trend', fontsize=14, fontweight='bold')
axes[0].set_ylabel('CBD Neighbor In Ratio')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(axis='x', rotation=45)

# Monthly CBD Neighbor Out Ratio
axes[1].plot(monthly_df['date_dt'], monthly_df['cbd_neighbor_out_ratio'], 'o-', linewidth=3, markersize=8, color='brown')
axes[1].axvline(policy_date, color='red', linestyle='--', linewidth=3, label='Policy Implementation')
axes[1].set_title('Monthly CBD Neighbor Out Ratio Trend', fontsize=14, fontweight='bold')
axes[1].set_ylabel('CBD Neighbor Out Ratio')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('figures/cbd_neighbor_monthly_trends.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Created multiple time series visualizations:")
print("   - Daily aggregated time series")
print("   - 7-day moving averages")
print("   - Monthly trend analysis")
print("   - CBD Neighbor interaction trends")

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
plt.close()

# Day type specific visualizations
print("Creating day type specific visualizations...")

# Plot 1: Average Speed by Day Type with colored points
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Speed by day type (colored by day type)
colors = {'Weekday': 'blue', 'Weekend': 'green', 'Holiday': 'red'}
for day_type in ['Weekday', 'Weekend', 'Holiday']:
    day_data = daily_df[daily_df['day_type'] == day_type]
    if len(day_data) > 0:
        axes[0].scatter(day_data['date_dt'], day_data['avg_speed'], 
                       c=colors[day_type], alpha=0.6, label=day_type, s=30)

axes[0].axvline(policy_date, color='black', linestyle='--', linewidth=3, label='Policy Implementation')
axes[0].set_title('Daily Average Speed by Day Type', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Speed (mph)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(axis='x', rotation=45)

# CBD Inside Ratio by day type
for day_type in ['Weekday', 'Weekend', 'Holiday']:
    day_data = daily_df[daily_df['day_type'] == day_type]
    if len(day_data) > 0:
        axes[1].scatter(day_data['date_dt'], day_data['cbd_inside_ratio'], 
                       c=colors[day_type], alpha=0.6, label=day_type, s=30)

axes[1].axvline(policy_date, color='black', linestyle='--', linewidth=3, label='Policy Implementation')
axes[1].set_title('Daily CBD Inside Ratio by Day Type', fontsize=14, fontweight='bold')
axes[1].set_ylabel('CBD Inside Ratio')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('figures/day_type_scatter.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Box plots by day type and policy period
fig, axes = plt.subplots(1, 3, figsize=(21, 7))

# Speed box plots
speed_data_by_type = []
labels_by_type = []
for day_type in ['Weekday', 'Weekend', 'Holiday']:
    for period in [False, True]:
        period_label = 'Post-policy' if period else 'Pre-policy'
        data = df[(df['day_type'] == day_type) & (df['post'] == period)]['avg_speed']
        if len(data) > 0:
            speed_data_by_type.append(data)
            labels_by_type.append(f'{day_type}\n{period_label}')

if speed_data_by_type:
    axes[0].boxplot(speed_data_by_type, labels=labels_by_type)
    axes[0].set_title('Average Speed Distribution by Day Type and Period', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Speed (mph)')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)

# CBD Inside Ratio box plots
cbd_data_by_type = []
cbd_labels_by_type = []
for day_type in ['Weekday', 'Weekend', 'Holiday']:
    for period in [False, True]:
        period_label = 'Post-policy' if period else 'Pre-policy'
        data = df[(df['day_type'] == day_type) & (df['post'] == period)]['cbd_inside_ratio']
        if len(data) > 0:
            cbd_data_by_type.append(data)
            cbd_labels_by_type.append(f'{day_type}\n{period_label}')

if cbd_data_by_type:
    axes[1].boxplot(cbd_data_by_type, labels=cbd_labels_by_type)
    axes[1].set_title('CBD Inside Ratio Distribution by Day Type and Period', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('CBD Inside Ratio')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)

# Total Trips box plots
trips_data_by_type = []
trips_labels_by_type = []
for day_type in ['Weekday', 'Weekend', 'Holiday']:
    for period in [False, True]:
        period_label = 'Post-policy' if period else 'Pre-policy'
        data = df[(df['day_type'] == day_type) & (df['post'] == period)]['total_trips']
        if len(data) > 0:
            trips_data_by_type.append(data)
            trips_labels_by_type.append(f'{day_type}\n{period_label}')

if trips_data_by_type:
    axes[2].boxplot(trips_data_by_type, labels=trips_labels_by_type)
    axes[2].set_title('Total Trips Distribution by Day Type and Period', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Total Trips per Hour')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/day_type_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Time series with different colors for different day types
fig, axes = plt.subplots(2, 2, figsize=(20, 12))

# Average Speed time series by day type
for day_type in ['Weekday', 'Weekend', 'Holiday']:
    day_data = daily_df[daily_df['day_type'] == day_type]
    if len(day_data) > 0:
        axes[0, 0].plot(day_data['date_dt'], day_data['avg_speed'], 
                       'o-', linewidth=2, markersize=4, color=colors[day_type], 
                       alpha=0.8, label=day_type)

axes[0, 0].axvline(policy_date, color='black', linestyle='--', linewidth=3, label='Policy Implementation')
axes[0, 0].set_title('Daily Average Speed by Day Type', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Speed (mph)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].tick_params(axis='x', rotation=45)

# CBD Inside Ratio time series by day type
for day_type in ['Weekday', 'Weekend', 'Holiday']:
    day_data = daily_df[daily_df['day_type'] == day_type]
    if len(day_data) > 0:
        axes[0, 1].plot(day_data['date_dt'], day_data['cbd_inside_ratio'], 
                       'o-', linewidth=2, markersize=4, color=colors[day_type], 
                       alpha=0.8, label=day_type)

axes[0, 1].axvline(policy_date, color='black', linestyle='--', linewidth=3, label='Policy Implementation')
axes[0, 1].set_title('Daily CBD Inside Ratio by Day Type', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('CBD Inside Ratio')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].tick_params(axis='x', rotation=45)

# CBD Neighbor In Ratio time series by day type
for day_type in ['Weekday', 'Weekend', 'Holiday']:
    day_data = daily_df[daily_df['day_type'] == day_type]
    if len(day_data) > 0:
        axes[1, 0].plot(day_data['date_dt'], day_data['cbd_neighbor_in_ratio'], 
                       'o-', linewidth=2, markersize=4, color=colors[day_type], 
                       alpha=0.8, label=day_type)

axes[1, 0].axvline(policy_date, color='black', linestyle='--', linewidth=3, label='Policy Implementation')
axes[1, 0].set_title('Daily CBD Neighbor In Ratio by Day Type', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('CBD Neighbor In Ratio')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].tick_params(axis='x', rotation=45)

# CBD Neighbor Out Ratio time series by day type
for day_type in ['Weekday', 'Weekend', 'Holiday']:
    day_data = daily_df[daily_df['day_type'] == day_type]
    if len(day_data) > 0:
        axes[1, 1].plot(day_data['date_dt'], day_data['cbd_neighbor_out_ratio'], 
                       'o-', linewidth=2, markersize=4, color=colors[day_type], 
                       alpha=0.8, label=day_type)

axes[1, 1].axvline(policy_date, color='black', linestyle='--', linewidth=3, label='Policy Implementation')
axes[1, 1].set_title('Daily CBD Neighbor Out Ratio by Day Type', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('CBD Neighbor Out Ratio')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('figures/day_type_time_series.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Created day type specific visualizations:")
print("   - Scatter plots colored by day type")
print("   - Box plots by day type and policy period")
print("   - Time series with different colors for each day type")

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
    'cbd_neighbor_in_change_pct': pct_change['cbd_neighbor_in_ratio'],
    'cbd_neighbor_out_change_pct': pct_change['cbd_neighbor_out_ratio'],
    'did_peak_estimate': did_estimate,
    'did_cbd_estimate': cbd_did_estimate,
    'did_weekday_weekend_estimate': weekday_weekend_did,
    'did_weekday_holiday_estimate': weekday_holiday_did if weekday_holiday_did is not None else None,
    'did_cbd_daytype_estimate': cbd_daytype_did,
    'speed_pvalue': p_val_speed,
    'cbd_pvalue': p_val_cbd,
    'trips_pvalue': p_val_trips,
    'neighbor_in_pvalue': p_val_neighbor_in,
    'neighbor_out_pvalue': p_val_neighbor_out
}

# Add day type test results to executive summary
for key, value in day_type_tests.items():
    exec_summary[f'{key}_t_stat'] = value['t_stat']
    exec_summary[f'{key}_p_val'] = value['p_val']

# Create markdown report using string concatenation to avoid f-string issues
# Determine significance and changes
speed_sig = 'Significant' if exec_summary['speed_pvalue'] < 0.05 else 'Not significant'
trips_sig = 'Significant' if exec_summary['trips_pvalue'] < 0.05 else 'Not significant'
cbd_sig = 'Significant' if exec_summary['cbd_pvalue'] < 0.05 else 'Not significant'
neighbor_in_sig = 'Significant' if exec_summary['neighbor_in_pvalue'] < 0.05 else 'Not significant'
neighbor_out_sig = 'Significant' if exec_summary['neighbor_out_pvalue'] < 0.05 else 'Not significant'

# Substitution effect interpretation
has_substitution = ((exec_summary['cbd_neighbor_in_change_pct'] > 1 or exec_summary['cbd_neighbor_out_change_pct'] > 1) and 
                   (exec_summary['neighbor_in_pvalue'] < 0.05 or exec_summary['neighbor_out_pvalue'] < 0.05))
substitution_interp = ('Evidence of substitution effect - passengers avoiding CBD by using neighboring areas' if has_substitution 
                      else 'No strong evidence of substitution effect')
substitution_detailed = ('There is evidence of a substitution effect. Passengers appear to be avoiding the CBD congestion charge by shifting their trips to neighboring areas. This suggests the policy is having its intended deterrent effect on CBD usage, but may be displacing congestion to adjacent areas.' 
                        if has_substitution else 
                        'There is no strong evidence of a substitution effect. The changes in CBD neighbor area usage are not statistically significant or substantial, suggesting passengers are not systematically avoiding the CBD by using neighboring areas.')

# Speed effects
speed_effect_text = 'statistically significant' if exec_summary['speed_pvalue'] < 0.05 else 'no statistically significant'
cbd_change_dir = 'decrease' if exec_summary['cbd_inside_change_pct'] < 0 else 'increase'
cbd_stat_sig = 'statistically significant' if exec_summary['cbd_pvalue'] < 0.05 else 'not statistically significant'
trips_change_dir = 'decreased' if exec_summary['trips_change_pct'] < 0 else 'increased'
trips_suggest = 'demand reduction' if exec_summary['trips_change_pct'] < 0 else 'increased demand'

# Conclusions
speed_conclusion = 'Evidence of speed improvements' if exec_summary['speed_change_pct'] > 0 else 'No evidence of speed improvements'
usage_conclusion = 'Evidence of reduced CBD usage' if exec_summary['cbd_inside_change_pct'] < 0 else 'No evidence of reduced CBD usage'
substitution_conclusion = ('Evidence suggests passengers are avoiding CBD by using neighboring areas' if has_substitution 
                          else 'No strong evidence of substitution to neighboring areas')
impact_conclusion = 'mixed results' if abs(exec_summary['speed_change_pct']) < 1 else 'clear impacts'

# Store dataframe values to avoid quote issues in .format()
time_min = df['pickup_hour'].min().strftime('%Y-%m-%d')
time_max = df['pickup_hour'].max().strftime('%Y-%m-%d')
total_hours = len(df)
pre_hours = (~df['post']).sum()
post_hours = df['post'].sum()
yellow_prop = df['yellow_ratio'].mean()
green_prop = df['green_ratio'].mean()
hours_ratio = pre_hours / post_hours

report_content = """# NYC CBD Congestion Pricing Policy Impact Analysis - Comprehensive

**Analysis Date:** {}  
**Policy Implementation Date:** January 5, 2025  
**Data Source:** NYC Yellow & Green Taxi Hourly Summary Data (Combined)  

## Executive Summary

### Key Findings

1. **Average Speed Impact:**
   - Overall change: {:.2f}%
   - Statistical significance: {} (p = {:.4f})
   - Peak hours DiD effect: {:.4f} mph

2. **Trip Volume Impact:**
   - Total trips change: {:.2f}%
   - Statistical significance: {} (p = {:.4f})

3. **CBD Usage Patterns:**
   - CBD internal trips change: {:.2f}%
   - Statistical significance: {} (p = {:.4f})
   - CBD exposure DiD effect: {:.4f} mph

4. **Substitution Effect (CBD Neighbor Areas):**
   - CBD Neighbor In ratio change: {:.2f}%
   - Statistical significance: {} (p = {:.4f})
   - CBD Neighbor Out ratio change: {:.2f}%
   - Statistical significance: {} (p = {:.4f})
   - **Interpretation:** {}

5. **Day Type Analysis (Weekday/Weekend/Holiday):**
   - Weekday vs Weekend DiD effect: {:.4f} mph
   - CBD usage weekday vs weekend DiD: {:.4f}
   - Holiday data availability: {} ({} hours)

## Data Description

- **Dataset:** Hourly aggregated NYC Yellow & Green Taxi data (weighted combination)
- **Time Period:** {} to {}
- **Total Observations:** {:,} hours
- **Pre-policy Period:** {:,} hours
- **Post-policy Period:** {:,} hours
- **Yellow Taxi Proportion:** {:.2%}
- **Green Taxi Proportion:** {:.2%}

## Methodology

### 1. Data Aggregation
Yellow and Green taxi data are combined using weighted averaging:
- **Sum metrics** (total_trips, total_revenue): Direct summation
- **Average metrics** (avg_speed, avg_distance, etc.): Weighted by total_trips
- **Ratio metrics** (cbd_inside_ratio, etc.): Weighted by total_trips

This ensures that taxi types with more trips have proportionally more influence on the combined metrics.

### 2. Descriptive Analysis
- Simple before/after comparisons using t-tests
- Peak vs off-peak hour analysis
- Statistical significance testing at α = 0.05

### 3. Difference-in-Differences (DiD) Analysis

#### 3.1 DiD Methodology Overview

The Difference-in-Differences (DiD) method estimates the causal effect of a policy by comparing the change in outcomes over time between a treatment group and a control group.

**Basic DiD Formula:**

$$
\\text{{DiD}} = (\\bar{{Y}}_{{\\text{{treatment, post}}}} - \\bar{{Y}}_{{\\text{{treatment, pre}}}}) - (\\bar{{Y}}_{{\\text{{control, post}}}} - \\bar{{Y}}_{{\\text{{control, pre}}}})
$$

Where:
- $\\bar{{Y}}_{{\\text{{treatment, post}}}}$ = Average outcome for treatment group after policy
- $\\bar{{Y}}_{{\\text{{treatment, pre}}}}$ = Average outcome for treatment group before policy
- $\\bar{{Y}}_{{\\text{{control, post}}}}$ = Average outcome for control group after policy
- $\\bar{{Y}}_{{\\text{{control, pre}}}}$ = Average outcome for control group before policy

**Key Assumption:** Parallel trends - without the policy, treatment and control groups would have followed similar trends.

#### 3.2 DiD Implementation in This Analysis

**Analysis 1: Peak vs Off-Peak Hours**

- **Treatment Group:** Peak hours (7-10 AM, 4-7 PM) - more affected by congestion pricing
- **Control Group:** Off-peak hours - less affected by congestion pricing
- **Outcome Variable:** Average speed (mph)

Formula:

$$
\\text{{DiD}}_{{\\text{{peak}}}} = (\\text{{Speed}}_{{\\text{{peak, post}}}} - \\text{{Speed}}_{{\\text{{peak, pre}}}}) - (\\text{{Speed}}_{{\\text{{offpeak, post}}}} - \\text{{Speed}}_{{\\text{{offpeak, pre}}}})
$$

**Result:** DiD = {:.4f} mph

**Analysis 2: CBD Exposure**

- **Treatment Group:** High CBD exposure hours (above median CBD interaction)
- **Control Group:** Low CBD exposure hours (below median CBD interaction)
- **Outcome Variable:** Average speed (mph)

Formula:

$$
\\text{{DiD}}_{{\\text{{CBD}}}} = (\\text{{Speed}}_{{\\text{{high CBD, post}}}} - \\text{{Speed}}_{{\\text{{high CBD, pre}}}}) - (\\text{{Speed}}_{{\\text{{low CBD, post}}}} - \\text{{Speed}}_{{\\text{{low CBD, pre}}}})
$$

**Result:** DiD = {:.4f} mph

#### 3.3 Discussion: Equal Sample Size Requirement

**Professor's Concern:** "DiD requires equal number of observations in pre- and post-policy periods."

**Evaluation:**

1. **Theoretical Requirement:** DiD does NOT strictly require equal sample sizes. The method is valid as long as:
   - The parallel trends assumption holds
   - Both periods have sufficient observations for reliable estimation
   - Standard errors are properly calculated

2. **Our Data:**
   - Pre-policy: {:,} hours
   - Post-policy: {:,} hours
   - Ratio: {:.2f}:1

3. **Implications:**
   - **Unequal sample sizes are acceptable** in DiD analysis
   - Larger pre-policy sample provides better baseline estimation
   - Standard t-tests and regression-based DiD handle unequal samples naturally
   - **Concern:** If sample sizes are very imbalanced, statistical power may be reduced
   - **Mitigation:** We have substantial observations in both periods (5,000+ hours each)

4. **Best Practices:**
   - Use regression-based DiD for formal inference: $Y_{{it}} = \\beta_0 + \\beta_1 \\text{{Post}}_t + \\beta_2 \\text{{Treatment}}_i + \\beta_3 (\\text{{Post}}_t \\times \\text{{Treatment}}_i) + \\epsilon_{{it}}$
   - Where $\\beta_3$ is the DiD estimate
   - This approach properly accounts for unequal sample sizes and provides standard errors
   - Consider time-varying confounders and seasonality

**Conclusion:** The equal sample size "requirement" is a misconception. Our analysis is methodologically sound with unequal sample sizes.

## Results

### Pre/Post Comparison
{}

### Peak vs Off-Peak Analysis
{}

### Day Type Analysis (Weekday/Weekend/Holiday)
{}

**Statistical Tests by Day Type:**

**Weekday Analysis:**
- Average Speed: t-statistic: {:.4f}, p-value: {:.4f}
- CBD Inside Ratio: t-statistic: {:.4f}, p-value: {:.4f}
- Total Trips: t-statistic: {:.4f}, p-value: {:.4f}

**Weekend Analysis:**
- Average Speed: t-statistic: {:.4f}, p-value: {:.4f}
- CBD Inside Ratio: t-statistic: {:.4f}, p-value: {:.4f}
- Total Trips: t-statistic: {:.4f}, p-value: {:.4f}

**Holiday Analysis:**
- Average Speed: t-statistic: {:.4f}, p-value: {:.4f}
- CBD Inside Ratio: t-statistic: {:.4f}, p-value: {:.4f}
- Total Trips: t-statistic: {:.4f}, p-value: {:.4f}

## Policy Impact Assessment

### Speed Effects
The analysis shows {} changes in average speeds following the CBD congestion pricing implementation. The overall speed change was {:.2f}%.

### CBD Usage Patterns
CBD internal trip ratios show a {} of {:.2f}%, which is {}.

### Trip Volume Effects
Overall taxi trip volumes {} by {:.2f}%, suggesting {}.

### Substitution Effect Analysis (CBD Neighbor Areas)

To detect whether passengers are avoiding the congestion charge by using neighboring areas instead of CBD:

**CBD Neighbor In Ratio (trips entering CBD neighbor areas):**
- Change: {:+.2f}%
- Statistical significance: {} (p = {:.4f})

**CBD Neighbor Out Ratio (trips leaving CBD neighbor areas):**
- Change: {:+.2f}%
- Statistical significance: {} (p = {:.4f})

**Interpretation:**
{}

## Difference-in-Differences Results

1. **Peak Hours Effect:** {:.4f} mph differential impact during peak hours
2. **CBD Exposure Effect:** {:.4f} mph differential impact for high CBD exposure areas
3. **Weekday vs Weekend Effect:** {:.4f} mph differential impact for weekdays vs weekends
4. **CBD Usage Weekday Effect:** {:.4f} differential impact for CBD usage on weekdays vs weekends
5. **Weekday vs Holiday Effect:** {} mph differential impact (if sufficient holiday data)

## Conclusions

Based on the comprehensive analysis of NYC Yellow & Green Taxi hourly data:

1. **Speed Effects:** {} following policy implementation
2. **Usage Patterns:** {} as intended by the policy
3. **Substitution Effect:** {}
4. **Overall Impact:** The policy shows {} in the initial implementation period

## Limitations

1. **Simplified Analysis:** This analysis uses basic statistical methods and may not capture complex causal relationships
2. **Confounding Factors:** Weather, holidays, and other concurrent changes not controlled for
3. **Short-term Data:** Limited post-policy observation period (approximately 8 months)
4. **Data Scope:** Only Yellow & Green Taxi data, excluding other transportation modes (Uber, Lyft, public transit, private vehicles)
5. **Parallel Trends Assumption:** DiD assumes parallel trends, which we have not formally tested
6. **Seasonality:** Monthly and seasonal patterns may confound the policy effect

---

**Analysis conducted by:** Yanjie Chen
**Files generated:** Tables in `tables/`, Figures in `figures/`
""".format(
    datetime.now().strftime('%Y-%m-%d'),
    exec_summary['speed_change_pct'], speed_sig, exec_summary['speed_pvalue'], exec_summary['did_peak_estimate'],
    exec_summary['trips_change_pct'], trips_sig, exec_summary['trips_pvalue'],
    exec_summary['cbd_inside_change_pct'], cbd_sig, exec_summary['cbd_pvalue'], exec_summary['did_cbd_estimate'],
    exec_summary['cbd_neighbor_in_change_pct'], neighbor_in_sig, exec_summary['neighbor_in_pvalue'],
    exec_summary['cbd_neighbor_out_change_pct'], neighbor_out_sig, exec_summary['neighbor_out_pvalue'],
    substitution_interp,
    exec_summary['did_weekday_weekend_estimate'], exec_summary['did_cbd_daytype_estimate'],
    'Sufficient' if weekday_holiday_did is not None else 'Insufficient',
    df[df['day_type'] == 'Holiday'].shape[0] if df[df['day_type'] == 'Holiday'].shape[0] > 0 else 0,
    time_min, time_max,
    total_hours, pre_hours, post_hours,
    yellow_prop, green_prop,
    exec_summary['did_peak_estimate'], exec_summary['did_cbd_estimate'],
    pre_hours, post_hours, hours_ratio,
    comparison_table.to_string(), peak_comparison_df.to_string(), day_type_comparison_df.to_string(),
    speed_effect_text, exec_summary['speed_change_pct'],
    cbd_change_dir, abs(exec_summary['cbd_inside_change_pct']), cbd_stat_sig,
    trips_change_dir, abs(exec_summary['trips_change_pct']), trips_suggest,
    exec_summary['cbd_neighbor_in_change_pct'], neighbor_in_sig, exec_summary['neighbor_in_pvalue'],
    exec_summary['cbd_neighbor_out_change_pct'], neighbor_out_sig, exec_summary['neighbor_out_pvalue'],
    substitution_detailed,
    exec_summary['did_peak_estimate'], exec_summary['did_cbd_estimate'],
    exec_summary['did_weekday_weekend_estimate'], exec_summary['did_cbd_daytype_estimate'],
    f"{exec_summary['did_weekday_holiday_estimate']:.4f}" if exec_summary['did_weekday_holiday_estimate'] is not None else "N/A (insufficient data)",
    # Day type statistical tests
    exec_summary.get('Weekday_speed_t_stat', 0), exec_summary.get('Weekday_speed_p_val', 1),
    exec_summary.get('Weekday_cbd_t_stat', 0), exec_summary.get('Weekday_cbd_p_val', 1),
    exec_summary.get('Weekday_trips_t_stat', 0), exec_summary.get('Weekday_trips_p_val', 1),
    exec_summary.get('Weekend_speed_t_stat', 0), exec_summary.get('Weekend_speed_p_val', 1),
    exec_summary.get('Weekend_cbd_t_stat', 0), exec_summary.get('Weekend_cbd_p_val', 1),
    exec_summary.get('Weekend_trips_t_stat', 0), exec_summary.get('Weekend_trips_p_val', 1),
    exec_summary.get('Holiday_speed_t_stat', 0), exec_summary.get('Holiday_speed_p_val', 1),
    exec_summary.get('Holiday_cbd_t_stat', 0), exec_summary.get('Holiday_cbd_p_val', 1),
    exec_summary.get('Holiday_trips_t_stat', 0), exec_summary.get('Holiday_trips_p_val', 1),
    speed_conclusion, usage_conclusion, substitution_conclusion, impact_conclusion
)

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

print(f"\n🔄 SUBSTITUTION EFFECT (CBD NEIGHBOR):")
print(f"   • CBD Neighbor In: {exec_summary['cbd_neighbor_in_change_pct']:+.2f}% change (p = {exec_summary['neighbor_in_pvalue']:.4f})")
print(f"   • CBD Neighbor Out: {exec_summary['cbd_neighbor_out_change_pct']:+.2f}% change (p = {exec_summary['neighbor_out_pvalue']:.4f})")

print(f"\n📅 DAY TYPE ANALYSIS:")
print(f"   • Weekday vs Weekend DiD: {exec_summary['did_weekday_weekend_estimate']:+.4f} mph")
print(f"   • CBD Usage Weekday DiD: {exec_summary['did_cbd_daytype_estimate']:+.4f}")
if exec_summary['did_weekday_holiday_estimate'] is not None:
    print(f"   • Weekday vs Holiday DiD: {exec_summary['did_weekday_holiday_estimate']:+.4f} mph")
else:
    print(f"   • Weekday vs Holiday DiD: N/A (insufficient holiday data)")

print(f"\n📈 DIFFERENCE-IN-DIFFERENCES:")
print(f"   • Peak Hours Effect: {exec_summary['did_peak_estimate']:+.4f} mph")
print(f"   • CBD Exposure Effect: {exec_summary['did_cbd_estimate']:+.4f} mph")
print(f"   • Weekday vs Weekend Effect: {exec_summary['did_weekday_weekend_estimate']:+.4f} mph")

print(f"\n📁 OUTPUT FILES:")
print(f"   • Tables: tables/")
print(f"   • Figures: figures/")
print(f"   • Report: reports/cbd_taxi_simplified_analysis.md")

print(f"\n✅ COMPREHENSIVE ANALYSIS COMPLETE!")
print(f"   • Combined Yellow & Green Taxi data")
print(f"   • CBD Neighbor substitution effect analysis included")
print(f"   • Day type analysis (Weekday/Weekend/Holiday) included")
print(f"   • DiD methodology fully documented")
print(f"   • Holiday and weekend/weekday separate processing implemented")
print("="*80)
