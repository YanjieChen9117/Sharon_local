# 更新日志

## 2025-10-08: Green Taxi数据整合

### 更新内容
已成功将Green Taxi数据整合到`process_hourly_taxi_data.py`脚本中。

### 关键改进
1. ✅ **支持多种出租车类型**: 现在可以同时处理Yellow和Green Taxi数据
2. ✅ **自动字段标准化**: 自动将不同前缀的时间字段（tpep_/lpep_）统一为标准字段名
3. ✅ **数据来源追踪**: 新增`taxi_type`字段和`yellow_ratio`/`green_ratio`统计指标
4. ✅ **完全兼容**: 所有现有特征都可以从Green Taxi数据中提取

### 数据规模
- **Yellow Taxi**: 32个文件 (~45MB/文件)
- **Green Taxi**: 32个文件 (~1.4MB/文件)
- **总计**: 64个月的数据

### 特征兼容性
**所有核心特征都可以从Green Taxi提取：**
- ✅ CBD交互标签 (inside/in/out/non)
- ✅ 行程时长 (trip_duration_min)
- ✅ 平均速度 (avg_speed_mph)
- ✅ 小费率 (tip_rate)
- ✅ 所有小时汇总指标

### 使用方法
直接运行脚本即可处理所有Yellow和Green Taxi数据：
```bash
python3 process_hourly_taxi_data.py
```

输出文件`data/hourly_taxi_summary.csv`将包含两种出租车类型的汇总数据。

### 详细文档
请查看 `green_taxi_integration.md` 获取完整的技术文档和字段兼容性分析。

