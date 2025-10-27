#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NTA Area Taxi Data Analysis Visualization Script

Features:
1. Read hourly_taxi_summary.csv data
2. Analyze MN0401 area pickup and MN0502 area dropoff trip count changes
3. Plot moving average charts to show time series trends
4. Generate multiple visualization charts

Author: AI Assistant
Date: 2025-01-08
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to use non-interactive backend
import matplotlib
matplotlib.use('Agg')

# Set font settings
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Set chart style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class NTAAnalysisPlotter:
    """NTA Area Taxi Data Visualization Analyzer"""
    
    def __init__(self, data_file):
        """
        Initialize analyzer
        
        Parameters:
            data_file: Hourly summary data file path
        """
        self.data_file = data_file
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load data"""
        try:
            print("Loading data...")
            self.df = pd.read_csv(self.data_file)
            
            # Convert time column
            self.df['pickup_hour'] = pd.to_datetime(self.df['pickup_hour'])
            
            # Sort by time
            self.df = self.df.sort_values('pickup_hour').reset_index(drop=True)
            
            print(f"✓ Successfully loaded data: {len(self.df):,} records")
            print(f"  Time range: {self.df['pickup_hour'].min()} to {self.df['pickup_hour'].max()}")
            
        except Exception as e:
            print(f"❌ Failed to load data: {str(e)}")
            raise
    
    def calculate_moving_averages(self, window_hours=24):
        """
        Calculate moving averages
        
        Parameters:
            window_hours: Moving average window size (hours)
            
        Returns:
            DataFrame: DataFrame containing moving averages
        """
        df = self.df.copy()
        
        # Calculate moving averages
        df['mn0401_pickup_ma'] = df['mn0401_pickup_trips'].rolling(
            window=window_hours, min_periods=1, center=True
        ).mean()
        
        df['mn0502_dropoff_ma'] = df['mn0502_dropoff_trips'].rolling(
            window=window_hours, min_periods=1, center=True
        ).mean()
        
        # Calculate total trips moving average as reference
        df['total_trips_ma'] = df['total_trips'].rolling(
            window=window_hours, min_periods=1, center=True
        ).mean()
        
        return df
    
    def plot_nta_trends(self, window_hours=24, save_path=None):
        """
        Plot NTA area trip trends
        
        Parameters:
            window_hours: Moving average window size
            save_path: Save path
        """
        df_ma = self.calculate_moving_averages(window_hours)
        
        # Create charts
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle(f'NTA Area Taxi Trip Trend Analysis (Moving Average Window: {window_hours} hours)', 
                     fontsize=16, fontweight='bold')
        
        # 1. MN0401 area pickup trends
        axes[0].plot(df_ma['pickup_hour'], df_ma['mn0401_pickup_trips'], 
                     alpha=0.3, color='lightblue', label='Raw Data')
        axes[0].plot(df_ma['pickup_hour'], df_ma['mn0401_pickup_ma'], 
                     color='blue', linewidth=2, label=f'{window_hours}-hour Moving Average')
        axes[0].set_title('MN0401 Area Pickup Trip Count Trends', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Trip Count')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. MN0502 area dropoff trends
        axes[1].plot(df_ma['pickup_hour'], df_ma['mn0502_dropoff_trips'], 
                     alpha=0.3, color='lightcoral', label='Raw Data')
        axes[1].plot(df_ma['pickup_hour'], df_ma['mn0502_dropoff_ma'], 
                     color='red', linewidth=2, label=f'{window_hours}-hour Moving Average')
        axes[1].set_title('MN0502 Area Dropoff Trip Count Trends', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Trip Count')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Total trips trends (as reference)
        axes[2].plot(df_ma['pickup_hour'], df_ma['total_trips'], 
                     alpha=0.3, color='lightgreen', label='Raw Data')
        axes[2].plot(df_ma['pickup_hour'], df_ma['total_trips_ma'], 
                     color='green', linewidth=2, label=f'{window_hours}-hour Moving Average')
        axes[2].set_title('Total Trip Count Trends (Reference)', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Trip Count')
        axes[2].set_xlabel('Time')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Chart saved to: {save_path}")
        
        plt.close()
    
    def plot_daily_patterns(self, save_path=None):
        """
        Plot daily pattern analysis
        
        Parameters:
            save_path: Save path
        """
        df = self.df.copy()
        
        # Add date and hour information
        df['date'] = df['pickup_hour'].dt.date
        df['hour'] = df['pickup_hour'].dt.hour
        df['day_of_week'] = df['pickup_hour'].dt.day_name()
        
        # Aggregate by date
        daily_data = df.groupby('date').agg({
            'mn0401_pickup_trips': 'sum',
            'mn0502_dropoff_trips': 'sum',
            'total_trips': 'sum'
        }).reset_index()
        
        # Create charts
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('NTA Area Daily Pattern Analysis', fontsize=16, fontweight='bold')
        
        # 1. Daily MN0401 pickup count
        axes[0, 0].plot(daily_data['date'], daily_data['mn0401_pickup_trips'], 
                       marker='o', linewidth=2, markersize=4)
        axes[0, 0].set_title('MN0401 Area Daily Pickup Count')
        axes[0, 0].set_ylabel('Trip Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Daily MN0502 dropoff count
        axes[0, 1].plot(daily_data['date'], daily_data['mn0502_dropoff_trips'], 
                       marker='s', linewidth=2, markersize=4, color='red')
        axes[0, 1].set_title('MN0502 Area Daily Dropoff Count')
        axes[0, 1].set_ylabel('Trip Count')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Average pattern by day of week
        hourly_pattern = df.groupby(['day_of_week', 'hour']).agg({
            'mn0401_pickup_trips': 'mean',
            'mn0502_dropoff_trips': 'mean'
        }).reset_index()
        
        # Reorder days of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        hourly_pattern['day_of_week'] = pd.Categorical(hourly_pattern['day_of_week'], categories=day_order, ordered=True)
        hourly_pattern = hourly_pattern.sort_values(['day_of_week', 'hour'])
        
        # MN0401 hourly pattern
        for day in day_order:
            day_data = hourly_pattern[hourly_pattern['day_of_week'] == day]
            axes[1, 0].plot(day_data['hour'], day_data['mn0401_pickup_trips'], 
                           marker='o', label=day, alpha=0.7)
        axes[1, 0].set_title('MN0401 Area Average Hourly Pattern (by Day of Week)')
        axes[1, 0].set_xlabel('Hour')
        axes[1, 0].set_ylabel('Average Trip Count')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].grid(True, alpha=0.3)
        
        # MN0502 hourly pattern
        for day in day_order:
            day_data = hourly_pattern[hourly_pattern['day_of_week'] == day]
            axes[1, 1].plot(day_data['hour'], day_data['mn0502_dropoff_trips'], 
                           marker='s', label=day, alpha=0.7)
        axes[1, 1].set_title('MN0502 Area Average Hourly Pattern (by Day of Week)')
        axes[1, 1].set_xlabel('Hour')
        axes[1, 1].set_ylabel('Average Trip Count')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Daily pattern chart saved to: {save_path}")
        
        plt.close()
    
    def plot_correlation_analysis(self, save_path=None):
        """
        Plot correlation analysis
        
        Parameters:
            save_path: Save path
        """
        # Select numeric columns for correlation analysis
        numeric_cols = [
            'mn0401_pickup_trips', 'mn0502_dropoff_trips', 'total_trips',
            'avg_distance', 'avg_duration', 'avg_speed', 'avg_fare',
            'cbd_inside_ratio', 'cbd_in_ratio', 'cbd_out_ratio', 'cbd_non_ratio'
        ]
        
        # Filter existing columns
        available_cols = [col for col in numeric_cols if col in self.df.columns]
        corr_data = self.df[available_cols]
        
        # Calculate correlation matrix
        correlation_matrix = corr_data.corr()
        
        # Create charts
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle('NTA Area Data Correlation Analysis', fontsize=16, fontweight='bold')
        
        # 1. Heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=axes[0], cbar_kws={'shrink': 0.8})
        axes[0].set_title('Correlation Heatmap')
        
        # 2. MN0401 and MN0502 scatter plots with other variables
        if 'mn0401_pickup_trips' in corr_data.columns and 'mn0502_dropoff_trips' in corr_data.columns:
            # Select key variables for scatter plot analysis
            key_vars = ['total_trips', 'avg_distance', 'avg_duration', 'avg_speed']
            available_key_vars = [var for var in key_vars if var in corr_data.columns]
            
            if available_key_vars:
                n_vars = len(available_key_vars)
                n_cols = 2
                n_rows = (n_vars + 1) // 2
                
                # Create subplots
                fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
                if n_rows == 1:
                    axes2 = axes2.reshape(1, -1)
                elif n_cols == 1:
                    axes2 = axes2.reshape(-1, 1)
                
                fig2.suptitle('MN0401 Pickup vs MN0502 Dropoff Scatter Plot Analysis', fontsize=14, fontweight='bold')
                
                for i, var in enumerate(available_key_vars):
                    row = i // n_cols
                    col = i % n_cols
                    ax = axes2[row, col] if n_rows > 1 else axes2[col]
                    
                    # Create scatter plot
                    scatter = ax.scatter(corr_data['mn0401_pickup_trips'], 
                                       corr_data['mn0502_dropoff_trips'],
                                       c=corr_data[var], cmap='viridis', alpha=0.6)
                    ax.set_xlabel('MN0401 Pickup Count')
                    ax.set_ylabel('MN0502 Dropoff Count')
                    ax.set_title(f'Colored by {var}')
                    plt.colorbar(scatter, ax=ax)
                    ax.grid(True, alpha=0.3)
                
                # Hide extra subplots
                for i in range(n_vars, n_rows * n_cols):
                    row = i // n_cols
                    col = i % n_cols
                    if n_rows > 1:
                        axes2[row, col].set_visible(False)
                    else:
                        axes2[col].set_visible(False)
                
                plt.tight_layout()
                
                if save_path:
                    scatter_path = save_path.replace('.png', '_scatter.png')
                    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
                    print(f"✓ Scatter plot saved to: {scatter_path}")
                
                plt.close()
        
        # Adjust layout
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Correlation analysis chart saved to: {save_path}")
        
        plt.close()
    
    def generate_summary_statistics(self):
        """Generate summary statistics"""
        print("\n" + "="*80)
        print("NTA Area Data Analysis Summary Statistics")
        print("="*80)
        
        # Basic statistics
        print(f"Data time range: {self.df['pickup_hour'].min()} to {self.df['pickup_hour'].max()}")
        print(f"Total records: {len(self.df):,} hours")
        
        # MN0401 statistics
        mn0401_total = self.df['mn0401_pickup_trips'].sum()
        mn0401_avg = self.df['mn0401_pickup_trips'].mean()
        mn0401_max = self.df['mn0401_pickup_trips'].max()
        mn0401_std = self.df['mn0401_pickup_trips'].std()
        
        print(f"\nMN0401 Area Pickup Statistics:")
        print(f"  Total trips: {mn0401_total:,}")
        print(f"  Average per hour: {mn0401_avg:.1f}")
        print(f"  Maximum per hour: {mn0401_max}")
        print(f"  Standard deviation: {mn0401_std:.1f}")
        
        # MN0502 statistics
        mn0502_total = self.df['mn0502_dropoff_trips'].sum()
        mn0502_avg = self.df['mn0502_dropoff_trips'].mean()
        mn0502_max = self.df['mn0502_dropoff_trips'].max()
        mn0502_std = self.df['mn0502_dropoff_trips'].std()
        
        print(f"\nMN0502 Area Dropoff Statistics:")
        print(f"  Total trips: {mn0502_total:,}")
        print(f"  Average per hour: {mn0502_avg:.1f}")
        print(f"  Maximum per hour: {mn0502_max}")
        print(f"  Standard deviation: {mn0502_std:.1f}")
        
        # Correlation
        correlation = self.df['mn0401_pickup_trips'].corr(self.df['mn0502_dropoff_trips'])
        print(f"\nMN0401 Pickup vs MN0502 Dropoff Correlation: {correlation:.3f}")
        
        # Percentage of total trips
        total_trips = self.df['total_trips'].sum()
        mn0401_ratio = mn0401_total / total_trips * 100
        mn0502_ratio = mn0502_total / total_trips * 100
        
        print(f"\nPercentage of Total Trips:")
        print(f"  MN0401 Pickup: {mn0401_ratio:.2f}%")
        print(f"  MN0502 Dropoff: {mn0502_ratio:.2f}%")
        
        print("="*80)


def main():
    """Main function"""
    # Data file path
    data_file = "/Users/yanjiechen/Documents/Github/Sharon_local/data/hourly_taxi_summary.csv"
    
    try:
        # Create analyzer
        plotter = NTAAnalysisPlotter(data_file)
        
        # Generate summary statistics
        plotter.generate_summary_statistics()
        
        # Plot trend charts (24-hour moving average)
        print("\nGenerating trend charts...")
        plotter.plot_nta_trends(window_hours=24, save_path="figures/nta_trends_24h.png")
        
        # Plot trend charts (168-hour moving average, i.e., 7 days)
        print("\nGenerating 7-day moving average trend charts...")
        plotter.plot_nta_trends(window_hours=168, save_path="figures/nta_trends_7d.png")
        
        # Plot daily patterns
        print("\nGenerating daily pattern analysis...")
        plotter.plot_daily_patterns(save_path="figures/nta_daily_patterns.png")
        
        # Plot correlation analysis
        print("\nGenerating correlation analysis...")
        plotter.plot_correlation_analysis(save_path="figures/nta_correlation.png")
        
        print("\n✅ All analysis completed!")
        
    except Exception as e:
        print(f"❌ Error occurred during analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
