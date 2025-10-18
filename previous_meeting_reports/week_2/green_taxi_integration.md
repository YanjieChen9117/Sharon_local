# Green Taxi数据整合说明

## 更新日期
2025-10-08

## 概述
已成功将Green Taxi数据整合到现有的`process_hourly_taxi_data.py`脚本中。该脚本现在可以同时处理Yellow Taxi和Green Taxi数据，并将它们合并到同一个小时汇总数据集中。

## 数据源
- **Yellow Taxi**: `yellow_tripdata_YYYY-MM.parquet`
- **Green Taxi**: `green_tripdata_YYYY-MM.parquet`

## 字段兼容性分析

### ✅ 完全兼容的核心字段
以下字段在Yellow和Green Taxi中都存在，且含义相同：

| 字段名 | 说明 | 用途 |
|--------|------|------|
| `VendorID` | 供应商ID | 删除（不需要） |
| `PULocationID` | 上车地点ID | **CBD标签判断** |
| `DOLocationID` | 下车地点ID | **CBD标签判断** |
| `passenger_count` | 乘客数量 | 数据清洗、特征统计 |
| `trip_distance` | 行程距离 | 数据清洗、特征统计 |
| `fare_amount` | 车费金额 | 数据清洗、特征统计 |
| `total_amount` | 总金额 | 数据清洗、收入统计 |
| `payment_type` | 支付方式 | 小费率计算 |
| `tip_amount` | 小费金额 | 小费率计算 |
| `RatecodeID` | 费率代码 | 删除（不需要） |
| `store_and_fwd_flag` | 存储转发标志 | 删除（不需要） |

### 🔄 需要标准化的时间字段
不同出租车类型使用不同的前缀，已通过`normalize_taxi_data()`函数统一：

| Yellow Taxi | Green Taxi | 标准化为 |
|-------------|------------|----------|
| `tpep_pickup_datetime` | `lpep_pickup_datetime` | `pickup_datetime` |
| `tpep_dropoff_datetime` | `lpep_dropoff_datetime` | `dropoff_datetime` |

### ⚠️ 特有字段（不影响核心功能）

**Yellow Taxi独有：**
- `airport_fee`: 机场费用（Green Taxi没有此字段，因为Green Taxi不服务机场）

**Green Taxi独有：**
- `ehail_fee`: 电子叫车费用
- `trip_type`: 行程类型（1=街头扬招, 2=派单）

这些特有字段不影响核心分析功能，因为它们不在数据处理流程中使用。

## 提取的特征

### ✅ 所有特征都可以从Green Taxi提取

| 特征名 | 说明 | 数据来源 |
|--------|------|----------|
| `cbd_interaction` | CBD交互类型 | PULocationID + DOLocationID |
| `trip_duration_min` | 行程时长（分钟） | pickup_datetime - dropoff_datetime |
| `avg_speed_mph` | 平均速度（英里/小时） | trip_distance / trip_duration_min |
| `tip_rate` | 小费率 | tip_amount / fare_amount |
| `pickup_hour` | 上车小时 | pickup_datetime |
| `taxi_type` | 出租车类型 | 新增字段（'yellow' 或 'green'） |

### 小时汇总指标

所有以下汇总指标都可以从Green Taxi数据计算：

- `total_trips`: 总行程数
- `avg_distance`: 平均行程距离
- `avg_passengers`: 平均乘客数
- `avg_duration`: 平均行程时长
- `avg_speed`: 平均速度
- `avg_tip_rate`: 平均小费率
- `cbd_inside_ratio`: CBD内部行程比例
- `cbd_in_ratio`: 进入CBD行程比例
- `cbd_out_ratio`: 离开CBD行程比例
- `cbd_non_ratio`: 非CBD行程比例
- `total_revenue`: 总收入
- `avg_fare`: 平均车费
- **`yellow_ratio`**: Yellow Taxi行程比例（新增）
- **`green_ratio`**: Green Taxi行程比例（新增）

## 脚本更新内容

### 1. 新增函数
- `normalize_taxi_data(df, taxi_type)`: 标准化不同类型出租车的字段名，统一处理逻辑

### 2. 更新的函数
- `process_monthly_file()`: 
  - 自动识别文件类型（yellow或green）
  - 调用标准化函数
  - 更新处理步骤为7步（增加标准化步骤）

- `process_all_files()`:
  - 搜索并处理两种类型的文件
  - 显示每种类型的文件数量
  - 在文件列表中标注类型

- `clean_data()`, `feature_engineering()`:
  - 使用标准化后的字段名（`pickup_datetime`, `dropoff_datetime`）

- `aggregate_by_hour()`:
  - 新增`yellow_ratio`和`green_ratio`统计

### 3. 文档更新
- 更新脚本顶部注释，说明支持两种出租车类型
- 更新日期为2025-10-08

## 使用方法

### 运行脚本
```bash
cd /Users/yanjiechen/Documents/Github/Sharon_local
python3 process_hourly_taxi_data.py
```

脚本会自动：
1. 扫描`data/`目录下的所有`yellow_tripdata_*.parquet`和`green_tripdata_*.parquet`文件
2. 识别每个文件的类型
3. 标准化字段名
4. 应用相同的清洗、标签和特征工程流程
5. 按小时汇总所有数据
6. 保存到`data/hourly_taxi_summary.csv`

### 输出文件
- `data/hourly_taxi_summary.csv`: 包含所有Yellow和Green Taxi的小时汇总数据

## 数据质量说明

### Green Taxi vs Yellow Taxi
- **服务区域差异**: Green Taxi主要服务于曼哈顿上城和外围区域，Yellow Taxi主要服务曼哈顿下城和CBD
- **数据量差异**: Green Taxi的数据量通常小于Yellow Taxi
- **CBD交互模式**: Green Taxi的CBD交互比例可能与Yellow Taxi不同

### 建议分析方向
1. **对比分析**: 比较Yellow和Green Taxi在CBD交互模式上的差异
2. **时空分布**: 分析两种出租车的服务时间和空间分布
3. **综合分析**: 合并两种数据以获得更全面的NYC出租车服务画像

## 无法提取的特征
**✅ 无！所有核心特征都可以从Green Taxi数据中成功提取。**

唯一的差异是某些特有字段（如`airport_fee`、`trip_type`）在不同类型出租车中的存在性，但这些字段不影响当前的分析流程。

## 测试验证
已测试以下功能：
- ✅ Green Taxi数据字段名标准化
- ✅ `pickup_datetime`和`dropoff_datetime`正确生成
- ✅ `taxi_type`列正确添加
- ✅ 无linter错误

## 下一步
现在您可以运行更新后的脚本来处理所有Yellow和Green Taxi数据：

```bash
python3 process_hourly_taxi_data.py
```

脚本将自动处理data目录中的所有出租车数据文件。

