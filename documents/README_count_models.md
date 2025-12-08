# CBD拥堵收费政策DiD分析 - 计数模型实现

## 概述

本项目包含多个R脚本，用于分析NYC CBD拥堵收费政策对出租车行程的影响。针对教授的建议，新增了**Poisson回归**和**Negative Binomial回归**模型，专门用于分析计数数据（trips）。

## 文件结构

### R分析脚本

| 文件 | 用途 | 推荐使用场景 |
|------|------|-------------|
| `cbd_did_analysis.R` | 原始OLS DiD分析 | 分析连续变量（speed）或参考对比 |
| `cbd_did_analysis_count_models.R` | 完整计数模型分析 | 需要详细诊断和统计信息 |
| `compare_did_models.R` | **快速模型对比** | **日常分析推荐** ✓ |

### 文档和报告

| 文件 | 内容 |
|------|------|
| `documents/count_models_analysis_guide.md` | 详细使用指南和方法学说明 |
| `reports/count_models_results_summary.md` | 完整结果总结和解释 |
| `README_count_models.md` | 本文件（快速开始指南） |

## 快速开始

### 1. 安装依赖包

在R中运行：
```r
install.packages(c("dplyr", "lubridate", "fixest", "MASS", "lmtest", "sandwich"))
```

### 2. 运行分析

**推荐方式** - 使用对比脚本：
```bash
Rscript compare_did_models.R
```

完整分析（包含详细诊断）：
```bash
Rscript cbd_did_analysis_count_models.R
```

原始OLS分析（仅供参考）：
```bash
Rscript cbd_did_analysis.R
```

## 主要发现

### 基于Negative Binomial模型的结果

#### 流出行程数 (Outflow Trips)
- **政策效应**: +2.10%
- **统计显著性**: p = 0.012 (**)
- **解释**: CBD拥堵收费使得CBD区域流出行程相对增加2.1%

#### 流入行程数 (Inflow Trips)
- **政策效应**: +2.87%
- **统计显著性**: p < 0.001 (***)
- **解释**: CBD拥堵收费使得CBD区域流入行程相对增加2.9%

### 关键技术发现

1. **严重过度离散**:
   - 流出行程: 过度离散比率 = 50.8
   - 流入行程: 过度离散比率 = 35.8
   - 结论: 必须使用Negative Binomial模型

2. **模型选择**:
   - ✓ **推荐**: Negative Binomial（处理过度离散）
   - ⚠️ 备选: Poisson（过度离散下结果不可靠）
   - ✗ 不推荐: OLS（不适合计数数据）

3. **Poisson vs NB差异**:
   - 流出行程: 两者一致（+2.10%）
   - 流入行程: **完全不同** (Poisson: -0.94%不显著; NB: +2.87%显著)
   - 原因: 严重过度离散导致Poisson失效

## 模型对比示例

### 流入行程数 - 三种模型对比

| 模型 | 系数 | 标准误 | P值 | 百分比效应 | 推荐度 |
|------|------|--------|-----|-----------|-------|
| OLS | -1.84 | 0.132 | <0.001*** | N/A | ✗ 不适合 |
| Poisson | -0.009 | 0.0076 | 0.218 (不显著) | -0.94% | ✗ 过度离散 |
| **NB** | **0.028** | **0.0051** | **<0.001***| **+2.87%** | **✓ 推荐** |

**关键观察**: Poisson和NB结果方向相反，证明过度离散的严重性！

## 为什么需要计数模型？

### OLS的问题
- ✗ 假设数据连续正态分布
- ✗ 可能预测负值（trips不能为负）
- ✗ 忽略数据的离散性
- ✗ 不能处理过度离散

### Poisson的问题
- ✓ 适合计数数据
- ✗ 假设方差=均值（本数据严重违反）
- ✗ 存在过度离散时标准误被低估

### Negative Binomial的优势
- ✓ 适合计数数据
- ✓ 允许过度离散（方差 > 均值）
- ✓ 更灵活，更稳健
- ✓ **本分析的最佳选择**

## 系数解释

### OLS
```
trips_change = β₃ = -0.41
```
解释: CBD区域行程绝对减少0.41次/小时

### Poisson / Negative Binomial
```
log(trips) = ... + β₃ × (Treatment × Post) + ...
百分比效应 = (exp(β₃) - 1) × 100%
```
例如: β₃ = 0.028 → exp(0.028) - 1 = 0.0287 → **+2.87%**

解释: CBD区域行程相对增加2.87%

## 技术细节

### 模型设定

**DiD框架**:
```
Y = f(Treatment, Post, Treatment×Post, Controls)
```

**控制变量**:
- 时间: day_of_week, hour_of_day, holiday
- 天气: temperature, precipitation, windspeed, snow
- 财务: total_tip, total_tolls, total_fare

**标准误**: HC3稳健标准误（处理异方差）

### 过度离散检验

**统计量**: Dispersion Ratio = Pearson χ² / df

**判断标准**:
- < 1.5: 无过度离散，Poisson适用
- 1.5 - 3.0: 中度过度离散，建议NB
- \> 3.0: 严重过度离散，必须用NB

**本数据**: 35.8 和 50.8 → **严重过度离散**

## 学术报告建议

### 论文中应报告

1. **主要结果表**: Negative Binomial的系数、标准误、P值、百分比效应
2. **稳健性检验**: Poisson结果作为对比
3. **模型诊断**: 过度离散检验结果和AIC比较
4. **方法学说明**: 为什么选择NB而非OLS或Poisson

### 建议表格格式

```
Table 1: Policy Effects on Taxi Trips (Negative Binomial Models)

                    Outflow Trips    Inflow Trips
Treatment × Post      0.0208**         0.0283***
                     (0.0083)         (0.0051)
                     
Marginal Effect       +2.10%           +2.87%

Dispersion Ratio      50.83            35.84
AIC                   25,120,858       3,867,869
Observations          412,812          412,812

Note: *** p<0.01, ** p<0.05, * p<0.1
Robust standard errors in parentheses.
Marginal effect = (exp(β)-1)×100%.
```

## 下一步分析建议

### 稳健性检验
- [ ] 改变时间窗口
- [ ] 安慰剂检验（假政策）
- [ ] 平行趋势检验
- [ ] 不同控制组定义

### 异质性分析
- [ ] 时段异质性（高峰vs非高峰）
- [ ] 空间异质性（不同NTA区域）
- [ ] 天气调节效应

### 扩展模型
- [ ] Zero-Inflated模型（如果零值过多）
- [ ] 固定效应模型（zone FE, time FE）
- [ ] 动态效应（event study）

## 常见问题

### Q: 为什么OLS和NB结果方向不同？
**A**: OLS不适合计数数据，假设违反导致估计有偏。计数模型（NB）考虑了数据的真实分布特征，结果更可靠。

### Q: Poisson和NB差异很大怎么办？
**A**: 说明存在严重过度离散。**必须使用NB的结果**，Poisson在过度离散下失效。

### Q: AIC怎么比OLS的还大？
**A**: OLS和计数模型的似然函数定义不同，AIC不能直接比较。只能在Poisson和NB之间用AIC比较。

### Q: 为什么trips增加了而不是减少？
**A**: 可能原因：
- 价格信号导致出行优化
- 替代效应（其他交通方式转为出租车）
- 供给侧调整（司机行为变化）
- 需要进一步分析机制

## 数据说明

- **数据源**: `/Users/yanjiechen/Documents/Github/Sharon_local/data/nta_zone_hourly_taxi_summary.csv`
- **时间范围**:
  - Pre-policy: 2024-01-06 至 2024-08-31
  - Post-policy: 2025-01-06 至 2025-08-31
- **Treatment组**: 16个CBD NTA区域
- **Control组**: 所有其他NTA区域
- **观测数**: 412,812 (zone × hour × date)

## 依赖包版本

建议使用以下版本（或更新）：
- R >= 4.0.0
- dplyr >= 1.0.0
- fixest >= 0.10.0
- MASS >= 7.3-50
- lmtest >= 0.9-38
- sandwich >= 3.0-0

## 脚本运行时间

- `compare_did_models.R`: ~2分钟
- `cbd_did_analysis_count_models.R`: ~3分钟
- `cbd_did_analysis.R`: ~1分钟

## 联系方式

- **作者**: Yanjie Chen
- **日期**: 2025-12-08
- **项目**: NYC CBD Congestion Pricing Analysis

## 参考文献

1. Cameron, A. C., & Trivedi, P. K. (2013). *Regression analysis of count data*. Cambridge University Press.
2. Hilbe, J. M. (2011). *Negative binomial regression*. Cambridge University Press.
3. Wooldridge, J. M. (2010). *Econometric analysis of cross section and panel data*. MIT Press.

---

**注**: 本分析使用Negative Binomial模型作为主要方法，专门处理计数数据的过度离散问题，比OLS和Poisson更加稳健和可靠。

