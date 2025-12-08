# 计数模型DiD分析使用指南

## 概述

本指南介绍如何使用Poisson和Negative Binomial回归模型分析出租车行程计数数据。

## 文件说明

### 1. `cbd_did_analysis.R` (原始脚本)
- **用途**: 使用OLS (最小二乘法) 进行DiD分析
- **适用**: 所有结果变量，包括trips和speed
- **特点**: 线性模型，简单直观

### 2. `cbd_did_analysis_count_models.R` (新建)
- **用途**: 使用Poisson和Negative Binomial模型分析计数数据
- **适用**: 仅用于trips数据 (outflow_trips, inflow_trips)
- **特点**: 
  - 专门为计数数据设计
  - 包含过度离散检验
  - 提供详细的模型诊断
  - 输出完整的统计结果

### 3. `compare_did_models.R` (新建)
- **用途**: 并排对比OLS、Poisson和Negative Binomial三种模型
- **适用**: trips数据
- **特点**: 
  - 快速对比三种模型
  - 清晰的对比表格
  - 提供模型选择建议

## 快速开始

### 运行完整分析
```bash
# 在终端运行
Rscript cbd_did_analysis_count_models.R
```

### 运行对比分析（推荐）
```bash
# 更快速，重点关注模型对比
Rscript compare_did_models.R
```

## 模型选择指南

### 什么时候使用哪种模型？

#### OLS (普通最小二乘法)
- ✅ **适用场景**:
  - 连续变量 (如平均速度)
  - 结果变量可以为负值或小数
  - 数据近似正态分布
- ❌ **不适用**:
  - 计数数据 (trips)
  - 结果变量只能为非负整数

#### Poisson回归
- ✅ **适用场景**:
  - 计数数据 (trips)
  - 方差约等于均值
  - 没有过度离散
- ❌ **不适用**:
  - 存在显著过度离散 (方差 >> 均值)
  - 过多的零值

#### Negative Binomial回归
- ✅ **适用场景**:
  - 计数数据 (trips)
  - 存在过度离散 (方差 > 均值)
  - Poisson模型过度离散检验失败
- ✅ **优势**:
  - 更灵活，可处理过度离散
  - 适合大多数真实世界的计数数据
  - 包含额外的离散参数 θ

## 关键概念

### 1. 过度离散 (Overdispersion)

**定义**: 数据的方差大于均值

**检验方法**:
```
过度离散比率 = Pearson χ² / 自由度
```

**判断标准**:
- 比率 < 1.5: 无显著过度离散，使用Poisson
- 比率 > 1.5: 存在过度离散，考虑Negative Binomial

### 2. 系数解释

#### OLS
```
Y = β₀ + β₃×(Treatment×Post) + ...
```
- β₃ = 政策效应的**绝对变化**
- 例如: β₃ = 10 → CBD区域相比对照组增加10次trips

#### Poisson/Negative Binomial
```
log(Y) = β₀ + β₃×(Treatment×Post) + ...
```
- β₃ = 政策效应的**对数变化**
- **百分比效应** = (exp(β₃) - 1) × 100%
- 例如: β₃ = 0.10 → exp(0.10) - 1 = 0.105 → 10.5%增长

### 3. 模型评估指标

#### AIC (Akaike Information Criterion)
- 衡量模型拟合优度和复杂度
- **越小越好**
- 用于模型选择

#### BIC (Bayesian Information Criterion)
- 类似AIC，但对复杂度惩罚更重
- **越小越好**

#### Dispersion参数 θ (仅NB模型)
- 衡量过度离散程度
- **θ越小，过度离散越严重**
- θ → ∞ 时，NB模型接近Poisson

## 输出结果解读

### 对比表格示例

```
指标                  OLS        Poisson      Negative_Binomial
治疗效应系数         15.2340     0.0856        0.0892
标准误               2.3450      0.0123        0.0145
P值                  0.000001    0.000000      0.000000
显著性               ***         ***           ***
百分比效应           N/A         8.94%         9.33%

模型拟合
AIC                  45678.23    45234.56      45123.45
BIC                  45890.12    45446.78      45335.67
过度离散比率         N/A         2.3456        N/A
Theta (θ)            N/A         N/A           12.3456
```

### 关键信息

1. **治疗效应系数**: 
   - OLS: 绝对变化量
   - Poisson/NB: 对数尺度的变化

2. **百分比效应**: 
   - 仅Poisson/NB有
   - 直接表示百分比变化，易于解释

3. **过度离散比率**:
   - > 1.5 建议使用NB
   - < 1.5 可使用Poisson

4. **模型选择**:
   - 比较AIC，选择最小的
   - 考虑过度离散情况
   - 参考统计显著性

## 实际应用建议

### 对于您的分析

1. **对于trips数据** (outflow_trips, inflow_trips):
   - ✅ 首选: Negative Binomial
   - ✅ 备选: Poisson (如果无过度离散)
   - ⚠️ 参考: OLS (仅用于对比)

2. **对于speed数据** (outflow_avg_speed, inflow_avg_speed):
   - ✅ 使用: OLS (原始脚本)
   - ❌ 不使用: Poisson/NB (不适合连续数据)

3. **报告建议**:
   - 同时报告Poisson和NB的结果
   - 说明模型选择理由 (基于过度离散检验)
   - 提供百分比效应的解释
   - 对比OLS结果以展示稳健性

## 常见问题

### Q1: 为什么不同模型的系数大小差异很大？
**A**: OLS是绝对值，Poisson/NB是对数尺度。关注百分比效应更合理。

### Q2: 三个模型的显著性不同怎么办？
**A**: 优先相信计数模型(Poisson/NB)的结果，因为它们更适合trips数据的分布特征。

### Q3: Negative Binomial模型拟合失败？
**A**: 可能原因：
- 数据没有过度离散，用Poisson即可
- 数据质量问题
- 模型过于复杂，尝试减少控制变量

### Q4: 过度离散比率在什么范围是正常的？
**A**: 
- 0.8-1.2: 理想，使用Poisson
- 1.2-1.5: 轻微过度离散，Poisson可接受
- > 1.5: 明显过度离散，使用NB
- > 3.0: 严重过度离散，必须使用NB

## 进一步分析

### 可能的扩展

1. **Zero-Inflated模型**:
   - 如果数据有过多零值
   - 使用ZIP或ZINB模型

2. **固定效应**:
   - 添加NTA_zone固定效应
   - 控制时间固定效应

3. **交互效应**:
   - 分析不同时段的异质性效应
   - 周末vs工作日的差异

4. **稳健性检验**:
   - 改变时间窗口
   - 剔除异常值
   - 使用不同的控制组定义

## 参考文献

1. Cameron, A. C., & Trivedi, P. K. (2013). *Regression analysis of count data* (Vol. 53). Cambridge University Press.

2. Hilbe, J. M. (2011). *Negative binomial regression*. Cambridge University Press.

3. Wooldridge, J. M. (2010). *Econometric analysis of cross section and panel data*. MIT Press.

## 联系方式

如有问题或需要进一步协助，请联系：
- 作者: Yanjie Chen
- 日期: 2025-12-08

