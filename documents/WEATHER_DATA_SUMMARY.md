# NTA天气数据总结

## 数据文件
- **文件路径**: `data/nta_weather_data.csv`
- **生成日期**: 2025年
- **数据源**: Open-Meteo Archive API

## 数据统计

### 基本信息
- **总记录数**: 880,416 条
- **NTA区域数**: 36 个
- **日期范围**: 2023-01-01 00:00:00 到 2025-10-15 23:00:00
- **时间跨度**: 约2年9个月

### 数据分辨率
**每小时 (hourly)**

每条记录代表：**一个NTA区域在一个小时的天气数据**

每个NTA区域有 24,456 条记录（约1,019天的每小时数据）

## 数据列

| 列名 | 类型 | 说明 |
|------|------|------|
| `datetime` | object | 日期时间（每小时） |
| `nta_code` | object | NTA区域代码（如MN0101） |
| `nta_name` | object | NTA区域名称 |
| `temperature` | float64 | 温度（摄氏度，2米高度） |
| `precipitation` | float64 | 降水量（毫米） |
| `windspeed` | float64 | 风速（公里/小时，10米高度） |
| `humidity` | int64 | 相对湿度（百分比） |

## NTA区域列表

数据包含以下36个Manhattan NTA区域：

1. MN0101 - Financial District-Battery Park City
2. MN0102 - Tribeca-Civic Center
3. MN0191 - The Battery-Governors Island-Ellis Island-Liberty Island
4. MN0201 - SoHo-Little Italy-Hudson Square
5. MN0202 - Greenwich Village
6. MN0203 - West Village
7. MN0301 - Chinatown-Two Bridges
8. MN0302 - Lower East Side
9. MN0303 - East Village
10. MN0401 - Chelsea-Hudson Yards
11. MN0402 - Hell's Kitchen
12. MN0501 - Midtown South-Flatiron-Union Square
13. MN0502 - Midtown-Times Square
14. MN0601 - Stuyvesant Town-Peter Cooper Village
15. MN0602 - Gramercy
16. MN0603 - Murray Hill-Kips Bay
17. MN0604 - East Midtown-Turtle Bay
18. MN0701 - Upper West Side-Lincoln Square
19. MN0702 - Upper West Side (Central)
20. MN0703 - Upper West Side-Manhattan Valley
21. MN0801 - Upper East Side-Lenox Hill-Roosevelt Island
22. MN0802 - Upper East Side-Carnegie Hill
23. MN0803 - Upper East Side-Yorkville
24. MN0901 - Morningside Heights
25. MN0902 - Manhattanville-West Harlem
26. MN0903 - Hamilton Heights-Sugar Hill
27. MN1001 - Harlem (South)
28. MN1002 - Harlem (North)
29. MN1101 - East Harlem (South)
30. MN1102 - East Harlem (North)
31. MN1191 - Randall's Island
32. MN1202 - Washington Heights (North)
33. MN1203 - Inwood
34. MN1291 - Highbridge Park
35. MN1292 - Inwood Hill Park
36. MN6491 - Central Park

## 数据使用示例

```python
import pandas as pd

# 读取数据
df = pd.read_csv('data/nta_weather_data.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

# 查看特定NTA区域的数据
mn0502_data = df[df['nta_code'] == 'MN0502']

# 查看特定日期的数据
date_data = df[df['datetime'].dt.date == pd.to_datetime('2024-01-01').date()]

# 按日期聚合（如果需要每日数据）
daily_data = df.groupby([df['datetime'].dt.date, 'nta_code']).agg({
    'temperature': 'mean',
    'precipitation': 'sum',
    'windspeed': 'mean',
    'humidity': 'mean'
}).reset_index()
```

## 生成脚本

数据由以下脚本生成：
- `weather_feature_workflow.py` - 主要工作流脚本
- `complete_missing_weather_data.py` - 补充获取缺失数据

## 注意事项

1. **数据分辨率**: 每小时数据，如需每日数据需要聚合
2. **时区**: 数据使用 America/New_York 时区
3. **坐标来源**: NTA质心坐标从NYC Open Data的shapefile计算得出
4. **API限制**: 获取数据时已添加延迟以避免API限流

