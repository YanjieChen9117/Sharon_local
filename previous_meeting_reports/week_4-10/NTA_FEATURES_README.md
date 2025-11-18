# NTA区域分析功能说明

## 概述

本功能为 `process_hourly_taxi_data.py` 脚本添加了NTA（Neighborhood Tabulation Area）区域分析功能，可以分析特定NTA区域的出租车行程模式。

## 新增功能

### 1. NTA映射功能
- 基于 `documents/NTA_lookup.md` 文件创建了PULocationID/DOLocationID到NTA区域的映射
- 为每个行程添加了 `PUNTA`（上车NTA区域）和 `DONTA`（下车NTA区域）列
- 如果LocationID不在映射表中，对应值为NA

### 2. 特定NTA区域统计
- **MN0401区域上车统计**：统计每小时在MN0401区域上车的行程数量
- **MN0502区域下车统计**：统计每小时在MN0502区域下车的行程数量

### 3. 可视化分析
- 创建了 `nta_analysis_plots.py` 脚本，提供多种可视化分析
- 支持移动平均趋势分析
- 提供每日模式分析
- 包含相关性分析

## 文件说明

### 修改的文件
- `process_hourly_taxi_data.py`：添加了NTA映射和统计功能

### 新增的文件
- `nta_analysis_plots.py`：NTA区域数据可视化分析脚本
- `test_nta_features.py`：功能测试脚本
- `NTA_FEATURES_README.md`：本说明文档

## 使用方法

### 1. 运行数据处理
```bash
python3 process_hourly_taxi_data.py
```

这将生成包含NTA分析功能的 `hourly_taxi_summary.csv` 文件。

### 2. 运行可视化分析
```bash
python3 nta_analysis_plots.py
```

这将生成以下图表：
- `figures/nta_trends_24h.png`：24小时移动平均趋势图
- `figures/nta_trends_7d.png`：7天移动平均趋势图
- `figures/nta_daily_patterns.png`：每日模式分析图
- `figures/nta_correlation.png`：相关性分析图

### 3. 运行功能测试
```bash
python3 test_nta_features.py
```

## 新增数据列

### 在原始数据中新增的列
- `PUNTA`：上车地点的NTA区域代码
- `DONTA`：下车地点的NTA区域代码

### 在小时汇总数据中新增的列
- `mn0401_pickup_trips`：每小时在MN0401区域上车的行程数量
- `mn0502_dropoff_trips`：每小时在MN0502区域下车的行程数量

## NTA区域映射

根据 `documents/NTA_lookup.md` 文件，主要NTA区域包括：

- **MN0401**：包含LocationID 246, 68, 90
- **MN0502**：包含LocationID 230, 163, 161
- 其他区域详见 `documents/NTA_lookup.md`

## 可视化功能

### 1. 趋势分析
- 原始数据与移动平均对比
- 支持不同时间窗口的移动平均（24小时、168小时等）

### 2. 每日模式分析
- 按日期显示行程数量变化
- 按星期几和小时显示平均模式

### 3. 相关性分析
- NTA区域行程与其他变量的相关性热力图
- 散点图分析

## 注意事项

1. 确保 `documents/NTA_lookup.md` 文件存在且格式正确
2. 天气数据文件 `central_park_weather_records.csv` 是可选的
3. 生成的图表将保存在 `figures/` 目录中
4. 建议使用Python 3运行脚本

## 扩展功能

如需分析其他NTA区域，可以：
1. 修改 `process_hourly_taxi_data.py` 中的 `aggregate_by_hour` 方法
2. 添加新的NTA区域统计列
3. 在可视化脚本中添加相应的分析功能
