# NYC CBD拥堵费政策影响分析 - 项目结构说明

## 📁 项目概览

本项目对NYC CBD拥堵费政策（2025年1月5日实施）对黄色出租车出行模式的影响进行了全面分析。

## 🗂️ 文件结构

### 📊 核心分析文件
- `cbd_analysis_simplified.py` - 主要分析脚本（包含NaN处理和多层次可视化）
- `process_hourly_taxi_data.py` - 数据预处理脚本

### 📋 数据文档
- `hourly_data_documentation.md` - 小时级数据详细说明文档
- `is_cbd.md` - CBD区域定义说明
- `data_structure_document.md` - 原始数据字段说明

### 📈 分析结果

#### 报告
- `reports/cbd_taxi_clean_analysis.md` - **最终分析报告**（包含NaN处理和统计稳健性检查）

#### 可视化图表
- `figures/time_series_overview_daily.png` - 日度聚合时间序列图（主要图表）
- `figures/time_series_moving_averages.png` - 7天移动平均趋势图
- `figures/time_series_monthly_trends.png` - 月度长期趋势图
- `figures/distribution_comparison.png` - 政策前后分布对比箱线图

#### 数据表格
- `tables/executive_summary_clean.csv` - 执行摘要数据
- `tables/pre_post_comparison_clean.csv` - 政策前后对比统计
- `tables/peak_comparison_clean.csv` - 峰值时段对比分析
- `tables/weekend_comparison.csv` - 工作日/周末对比分析

### 📂 数据文件
- `data/hourly_taxi_summary.csv` - 小时级汇总数据（主要数据源）

### 🗺️ 地理数据
- `taxi_zone_lookup/` - 出租车区域查找表和地理边界文件

### 📁 历史分析
- `preliminary_plot/` - 初步探索性分析文件（保留作为参考）

## 🎯 主要发现

### 关键结果
- **平均速度变化**: -0.68% (统计上不显著, p = 0.1466)
- **总出行量变化**: -3.40% (统计显著, p < 0.001)
- **CBD内部出行变化**: +1.14% (统计显著, p < 0.001)

### 双重差分分析
- **峰值时段效应**: -0.0425 mph
- **CBD暴露度效应**: -0.7941 mph

## 🔧 技术特性

### 数据质量保证
- ✅ 全面的NaN值检测和处理
- ✅ 稳健的统计计算方法
- ✅ 适当的显著性检验
- ✅ 清晰的置信度指示

### 可视化改进
- ✅ 多层次时间聚合（小时→日→周→月）
- ✅ 移动平均平滑处理
- ✅ 清晰的政策实施标记
- ✅ 专业的图表格式

### 分析方法
- 描述性统计分析
- 双重差分设计（DiD）
- 统计显著性检验
- 稳健性检查

## 📝 使用说明

### 运行主要分析
```bash
python3 cbd_analysis.py
```

### 生成的输出
- 报告: `reports/cbd_taxi_clean_analysis.md`
- 图表: `figures/` 目录
- 数据表: `tables/` 目录

## 🏷️ 版本信息

- **最后更新**: 2025-10-06
- **分析师**: Yanjie Chen
- **数据版本**: 2023-01-01 至 2025-08-31
- **政策日期**: 2025-01-05
