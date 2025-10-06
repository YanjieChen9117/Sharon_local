# NYC黄色出租车按小时汇总数据说明文档

## 📋 数据集概览

- **数据文件**: `hourly_taxi_summary.csv`
- **数据粒度**: 按小时汇总
- **数据来源**: NYC黄色出租车行程数据（经过清洗和特征工程处理）
- **生成脚本**: `process_hourly_taxi_data.py`
- **更新日期**: 2025-10-01

---

## 🎯 数据用途

本数据集专门设计用于以下分析：

1. **时间序列分析**
   - 小时级别的出租车需求预测
   - 交通模式的时间变化趋势
   - 周期性模式识别（小时、日、周、月）

2. **CBD拥堵费影响评估** 🚨
   - **政策实施日期**: 2025年1月5日
   - **分析重点**: 评估CBD拥堵费政策对出租车出行模式的影响
   - **对比分析**: 2025年1月5日前后的数据对比
   - **关键指标**: CBD交互比例、平均距离、平均速度等的变化

3. **交通流量分析**
   - CBD区域与非CBD区域的交通流向
   - 不同时段的出行特征
   - 收入和需求模式

---

## 📊 数据结构

### 字段详细说明

#### 1. pickup_hour (上车小时)
- **数据类型**: datetime
- **格式**: YYYY-MM-DD HH:00:00
- **含义**: 该小时的起始时间戳（向下取整到小时）
- **示例**: `2025-01-01 08:00:00` 表示2025年1月1日8:00-8:59的数据
- **用途**: 
  - 时间序列分析的时间索引
  - 识别高峰时段和低谷时段
  - 区分工作日和周末模式
- **注意事项**: 
  - 每个小时对应一条记录
  - 时间戳为该小时的开始时间

---

#### 2. total_trips (总行程数)
- **数据类型**: int
- **含义**: 该小时内完成的总行程数量
- **用途**:
  - 衡量出租车需求强度
  - 识别高峰时段
  - 评估CBD拥堵费对出行量的影响
- **典型值范围**: 1,000 - 10,000 trips/hour（根据时段和季节变化）
- **分析建议**:
  ```python
  # 对比政策前后的总行程数变化
  pre_policy = df[df['pickup_hour'] < '2025-01-05']
  post_policy = df[df['pickup_hour'] >= '2025-01-05']
  
  print(f"政策前平均每小时: {pre_policy['total_trips'].mean():.0f} trips")
  print(f"政策后平均每小时: {post_policy['total_trips'].mean():.0f} trips")
  print(f"变化率: {(post_policy['total_trips'].mean() / pre_policy['total_trips'].mean() - 1) * 100:.2f}%")
  ```

---

#### 3. avg_distance (平均行程距离)
- **数据类型**: float
- **单位**: 英里 (miles)
- **含义**: 该小时内所有行程的平均距离
- **用途**:
  - 评估出行距离模式
  - 分析CBD拥堵费对行程距离的影响（用户可能选择更短/更长的替代路线）
  - 识别长途和短途出行的时间模式
- **典型值范围**: 2-4 miles
- **分析重点**:
  - CBD拥堵费可能导致短途CBD内部行程减少
  - 用户可能选择绕行以避免CBD区域

---

#### 4. avg_passengers (平均乘客数量)
- **数据类型**: float
- **含义**: 该小时内所有行程的平均乘客数量
- **用途**:
  - 分析拼车模式
  - 评估出租车容量利用率
- **典型值范围**: 1.0 - 2.0 passengers
- **注意事项**: 
  - 数据由司机手动输入，可能存在误差
  - 绝大多数行程为单人或双人出行

---

#### 5. avg_duration (平均行程时长)
- **数据类型**: float
- **单位**: 分钟 (minutes)
- **含义**: 该小时内所有行程的平均时长
- **计算公式**: `dropoff_time - pickup_time`
- **用途**:
  - 评估交通拥堵程度
  - 分析行程效率
  - CBD拥堵费可能改变行程时长（避开拥堵区域）
- **典型值范围**: 10-30 minutes
- **分析建议**:
  - 结合 `avg_distance` 和 `avg_speed` 综合评估交通状况
  - 对比相同时段不同日期的变化趋势

---

#### 6. avg_speed (平均速度)
- **数据类型**: float
- **单位**: 英里/小时 (mph)
- **含义**: 该小时内所有行程的平均速度
- **计算公式**: `trip_distance / (trip_duration / 60)`
- **用途**:
  - **关键指标**: 评估交通拥堵程度
  - 分析CBD拥堵费对交通流畅度的影响
  - 识别拥堵时段
- **典型值范围**: 8-15 mph（市区）
- **分析重点** 🎯:
  - **假设**: CBD拥堵费实施后，CBD区域内的平均速度可能提高（车流量减少）
  - **对比分析**: 2025年1月5日前后，CBD内部行程的速度变化
  ```python
  # 分析CBD内部行程的速度变化
  pre_cbd_speed = pre_policy['avg_speed'].mean()
  post_cbd_speed = post_policy['avg_speed'].mean()
  ```

---

#### 7. avg_tip_rate (平均小费率)
- **数据类型**: float
- **含义**: 该小时内所有信用卡支付行程的平均小费率
- **计算公式**: `tip_amount / fare_amount`
- **用途**:
  - 评估服务质量
  - 分析司机收入
  - 间接反映乘客满意度
- **典型值范围**: 0.15 - 0.25 (15% - 25%)
- **注意事项**: 
  - ⚠️ 仅包含信用卡支付的小费
  - 现金小费未记录在数据中
  - CBD拥堵费可能影响乘客的小费行为

---

#### 8-11. CBD交互比例字段 🏙️

这是分析CBD拥堵费影响的**核心指标**！

##### 8. cbd_inside_ratio (CBD内部行程比例)
- **数据类型**: float
- **取值范围**: 0.0 - 1.0
- **含义**: 上车和下车都在CBD区域内的行程占比
- **计算公式**: `cbd_inside_trips / total_trips`
- **用途**:
  - **最重要的政策影响指标**
  - 评估CBD拥堵费对CBD内部短途出行的抑制效果
- **分析假设**:
  - 拥堵费实施后，此比例可能**显著下降**
  - 用户可能改用步行、自行车或地铁进行CBD内部短途出行

##### 9. cbd_in_ratio (进入CBD行程比例)
- **数据类型**: float
- **取值范围**: 0.0 - 1.0
- **含义**: 从非CBD区域进入CBD区域的行程占比
- **用途**:
  - 评估外围区域到CBD的通勤需求变化
  - 分析拥堵费对通勤模式的影响

##### 10. cbd_out_ratio (离开CBD行程比例)
- **数据类型**: float
- **取值范围**: 0.0 - 1.0
- **含义**: 从CBD区域离开到非CBD区域的行程占比
- **用途**:
  - 评估CBD到外围区域的出行需求
  - 分析拥堵费的非对称影响（进入vs离开）

##### 11. cbd_non_ratio (非CBD行程比例)
- **数据类型**: float
- **取值范围**: 0.0 - 1.0
- **含义**: 上车和下车都不在CBD区域的行程占比
- **用途**:
  - 作为对照组，评估拥堵费的间接影响
  - 分析是否有替代效应（用户转向非CBD区域出行）

**四个比例之和 = 1.0** ✓

##### CBD区域定义
CBD区域包含以下LocationID（共38个区域）:
```
50, 48, 163, 162, 229, 230, 161, 233, 246, 68, 100, 186,
90, 164, 234, 107, 170, 137, 224, 158, 249, 113, 114, 79,
4, 125, 211, 144, 148, 232, 231, 45, 209, 13, 261, 87, 12, 88
```

---

#### 12. total_revenue (总收入)
- **数据类型**: float
- **单位**: 美元 ($)
- **含义**: 该小时内所有行程的总收入
- **包含**: 所有费用总和（基本费用、小费、通行费、拥堵费等）
- **用途**:
  - 评估出租车行业收入趋势
  - 分析CBD拥堵费对司机收入的影响
- **分析建议**:
  - 结合 `total_trips` 计算单位时间收入效率
  - 评估拥堵费是否导致司机收入下降

---

#### 13. avg_fare (平均票价)
- **数据类型**: float
- **单位**: 美元 ($)
- **含义**: 该小时内所有行程的平均基本票价（不含小费和附加费）
- **用途**:
  - 评估出行成本
  - 分析票价与距离、时长的关系
- **典型值范围**: $10 - $20
- **注意**: CBD拥堵费会增加总费用，但不影响基本票价

---

## 📈 CBD拥堵费政策分析框架

### 政策背景

- **实施日期**: 2025年1月5日
- **费用标准**: $0.75（根据数据字段说明.md中的cbd_congestion_fee字段）
- **适用范围**: CBD区域（38个LocationID）
- **政策目标**: 减少CBD区域交通拥堵，鼓励使用公共交通

### 关键研究问题

#### 1. 需求影响 (Demand Impact)
```python
# 总行程数变化
# CBD内部行程变化
# 各类CBD交互比例变化
```

**预期结果**:
- ✓ CBD内部行程 (`cbd_inside_ratio`) 显著下降
- ✓ 总行程数可能略有下降
- ✓ 非CBD行程 (`cbd_non_ratio`) 可能增加（替代效应）

#### 2. 出行模式影响 (Travel Pattern Impact)
```python
# 平均距离变化
# 行程时长变化
# 出行时段分布变化
```

**预期结果**:
- ✓ CBD内部短途行程减少
- ✓ 用户可能选择更长路线以避开CBD
- ✓ 高峰时段的影响可能更明显

#### 3. 交通流畅度影响 (Traffic Flow Impact)
```python
# 平均速度变化（关键指标）
# 拥堵时段的速度改善
```

**预期结果**:
- ✓ CBD区域平均速度提高（政策有效性的直接证据）
- ✓ 高峰时段的拥堵缓解更明显

#### 4. 经济影响 (Economic Impact)
```python
# 总收入变化
# 司机收入效率
# 平均票价变化
```

**预期结果**:
- ⚠️ 总收入可能下降（行程数减少）
- ⚠️ 司机单位时间收入效率可能下降
- ✓ 乘客出行成本增加

---

## 📊 分析方法建议

### 1. 时间序列可视化

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('hourly_taxi_summary.csv', parse_dates=['pickup_hour'])

# 定义政策实施日期
policy_date = pd.Timestamp('2025-01-05')

# 绘制总行程数时间序列
plt.figure(figsize=(14, 6))
plt.plot(df['pickup_hour'], df['total_trips'], alpha=0.7)
plt.axvline(policy_date, color='red', linestyle='--', label='CBD拥堵费实施')
plt.title('总行程数时间序列 - CBD拥堵费影响分析')
plt.xlabel('时间')
plt.ylabel('每小时行程数')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 2. 政策前后对比分析

```python
# 分割数据
pre_policy = df[df['pickup_hour'] < policy_date]
post_policy = df[df['pickup_hour'] >= policy_date]

# 对比关键指标
metrics = ['total_trips', 'avg_distance', 'avg_speed', 
           'cbd_inside_ratio', 'cbd_in_ratio', 'cbd_out_ratio']

comparison = pd.DataFrame({
    '政策前均值': pre_policy[metrics].mean(),
    '政策后均值': post_policy[metrics].mean(),
    '变化率(%)': (post_policy[metrics].mean() / pre_policy[metrics].mean() - 1) * 100
})

print(comparison)
```

### 3. 按小时模式分析

```python
# 提取小时信息
df['hour_of_day'] = df['pickup_hour'].dt.hour

# 按小时对比政策前后的CBD内部行程比例
pre_hourly = pre_policy.groupby('hour_of_day')['cbd_inside_ratio'].mean()
post_hourly = post_policy.groupby('hour_of_day')['cbd_inside_ratio'].mean()

plt.figure(figsize=(12, 6))
plt.plot(pre_hourly.index, pre_hourly.values, 'o-', label='政策前')
plt.plot(post_hourly.index, post_hourly.values, 's-', label='政策后')
plt.title('CBD内部行程比例 - 按小时对比')
plt.xlabel('小时')
plt.ylabel('CBD内部行程比例')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 4. 统计检验

```python
from scipy import stats

# t检验：检验政策前后CBD内部行程比例是否有显著差异
t_stat, p_value = stats.ttest_ind(
    pre_policy['cbd_inside_ratio'],
    post_policy['cbd_inside_ratio']
)

print(f"t统计量: {t_stat:.4f}")
print(f"p值: {p_value:.4f}")
print(f"显著性: {'显著' if p_value < 0.05 else '不显著'} (α=0.05)")
```

### 5. 差异中的差异分析 (DID)

```python
# 构建双重差分模型
# 处理组：CBD相关行程
# 对照组：非CBD行程
# 处理时间：2025-01-05

df['post_policy'] = (df['pickup_hour'] >= policy_date).astype(int)
df['treatment_cbd'] = (df['cbd_inside_ratio'] + df['cbd_in_ratio'] + df['cbd_out_ratio'])

# DID估计
# 这是一个简化示例，实际分析需要使用回归模型
```

---

## ⚠️ 注意事项与局限性

### 数据局限性

1. **现金小费未记录**
   - `avg_tip_rate` 仅基于信用卡支付
   - 可能低估实际小费水平

2. **乘客数量可能不准确**
   - 由司机手动输入
   - 存在输入错误或遗漏

3. **仅包含黄色出租车**
   - 不包含Uber、Lyft等网约车
   - 不包含绿色出租车（主要服务外围区域）

4. **短期数据**
   - 政策实施后的长期效应需要更长时间的数据
   - 可能存在季节性混淆因素

### 分析建议

1. **控制混淆因素**
   - 考虑天气、节假日、特殊事件的影响
   - 使用同比对比（如2024年1月vs2025年1月）

2. **多维度分析**
   - 不要仅依赖单一指标
   - 综合考虑需求、速度、收入等多个维度

3. **时段分析**
   - 区分高峰时段和非高峰时段
   - 工作日vs周末
   - 不同小时的异质性影响

4. **空间异质性**
   - 不同CBD子区域可能有不同影响
   - 边缘区域vs核心区域

---

## 📚 相关文件

- **原始数据字段说明**: `数据字段说明.md`
- **CBD区域定义**: `is_cbd.md`
- **数据处理脚本**: `process_hourly_taxi_data.py`
- **原始数据目录**: `data/`

---

## 🔄 数据更新

本数据集随着新月份数据的添加而持续更新。
- 运行 `process_hourly_taxi_data.py` 可重新生成最新的汇总数据
- 建议定期备份历史版本以进行纵向比较

---

## 📧 问题反馈

如有数据质量问题或分析建议，请检查：
1. 数据清洗参数是否合理
2. CBD区域定义是否准确
3. 时间序列是否连续完整

---

**文档版本**: v1.0  
**创建日期**: 2025-10-01  
**最后更新**: 2025-10-01  
**维护者**: 数据分析团队

---

## 附录：快速开始代码

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv('data/hourly_taxi_summary.csv', parse_dates=['pickup_hour'])

# 查看基本信息
print("数据形状:", df.shape)
print("\n数据列名:")
print(df.columns.tolist())
print("\n数据预览:")
print(df.head())

# 定义政策日期
policy_date = pd.Timestamp('2025-01-05')

# 创建政策标识
df['is_post_policy'] = df['pickup_hour'] >= policy_date

# 基本统计
print("\n基本统计:")
print(df.describe())

# 快速可视化：CBD内部行程比例的变化
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. 总行程数时间序列
axes[0, 0].plot(df['pickup_hour'], df['total_trips'], alpha=0.5)
axes[0, 0].axvline(policy_date, color='red', linestyle='--', linewidth=2, label='政策实施日')
axes[0, 0].set_title('总行程数时间序列', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('时间')
axes[0, 0].set_ylabel('每小时行程数')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. CBD内部行程比例时间序列
axes[0, 1].plot(df['pickup_hour'], df['cbd_inside_ratio'], alpha=0.5)
axes[0, 1].axvline(policy_date, color='red', linestyle='--', linewidth=2, label='政策实施日')
axes[0, 1].set_title('CBD内部行程比例时间序列', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('时间')
axes[0, 1].set_ylabel('比例')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. 平均速度时间序列
axes[1, 0].plot(df['pickup_hour'], df['avg_speed'], alpha=0.5)
axes[1, 0].axvline(policy_date, color='red', linestyle='--', linewidth=2, label='政策实施日')
axes[1, 0].set_title('平均速度时间序列', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('时间')
axes[1, 0].set_ylabel('速度 (mph)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. CBD交互比例堆叠面积图
axes[1, 1].fill_between(df['pickup_hour'], 0, df['cbd_inside_ratio'], 
                         alpha=0.5, label='CBD内部')
axes[1, 1].fill_between(df['pickup_hour'], df['cbd_inside_ratio'], 
                         df['cbd_inside_ratio'] + df['cbd_in_ratio'],
                         alpha=0.5, label='进入CBD')
axes[1, 1].fill_between(df['pickup_hour'], 
                         df['cbd_inside_ratio'] + df['cbd_in_ratio'],
                         df['cbd_inside_ratio'] + df['cbd_in_ratio'] + df['cbd_out_ratio'],
                         alpha=0.5, label='离开CBD')
axes[1, 1].axvline(policy_date, color='red', linestyle='--', linewidth=2, label='政策实施日')
axes[1, 1].set_title('CBD交互比例分布', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('时间')
axes[1, 1].set_ylabel('累积比例')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cbd_policy_impact_overview.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ 可视化完成！图表已保存为 'cbd_policy_impact_overview.png'")
```

**祝分析顺利！** 🎉

