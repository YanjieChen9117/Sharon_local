# 平行趋势假设检验结果 / Parallel Trends Assumption Test Results

本文件夹包含CBD拥堵收费政策DiD分析的平行趋势假设检验结果。

This folder contains the parallel trends assumption test results for the CBD congestion pricing policy DiD analysis.

---

## 文件清单 / File List

### 报告文件 / Reports

1. **`parallel_trends_report_zh.md`** - 中文详细报告
   - 完整的检验方法说明
   - 所有检验结果的详细解读
   - 结论和建议

2. **`parallel_trends_report_en.md`** - English detailed report
   - Complete testing methods explanation
   - Detailed interpretation of all test results
   - Conclusions and recommendations

### 可视化图表 / Visualizations

**注意 / Note**: 所有图表使用英文标签。趋势图使用连续的周索引，移除了数据缺失期间的gap，以便更清晰地展示趋势。

**Note**: All plots use English labels. Trend plots use continuous week indices, removing the data gap period for clearer trend visualization.

#### 趋势图 / Trend Plots
- `trend_log_outflow_trips.png` - 流出行程数时间趋势 / Time trend of outflow trips
- `trend_log_inflow_trips.png` - 流入行程数时间趋势 / Time trend of inflow trips
- `trend_log_outflow_avg_speed.png` - 流出平均速度时间趋势 / Time trend of outflow average speed
- `trend_log_inflow_avg_speed.png` - 流入平均速度时间趋势 / Time trend of inflow average speed

#### Event Study图 / Event Study Plots
- `event_study_log_outflow_trips.png` - 流出行程数Event Study / Event study of outflow trips
- `event_study_log_inflow_trips.png` - 流入行程数Event Study / Event study of inflow trips
- `event_study_log_outflow_avg_speed.png` - 流出平均速度Event Study / Event study of outflow average speed
- `event_study_log_inflow_avg_speed.png` - 流入平均速度Event Study / Event study of inflow average speed

### 数据文件 / Data Files

1. **`parallel_trends_summary.csv`** - 检验结果汇总表
   - 包含所有四种检验方法的关键统计量
   - 可用于进一步分析或制表

2. **`parallel_trends_detailed_results.rds`** - R详细结果对象
   - 包含所有回归模型的完整输出
   - 可在R中加载进行进一步分析
   - 使用方法: `results <- readRDS("parallel_trends_detailed_results.rds")`

---

## 快速结论 / Quick Summary

### 📊 重要发现 / Key Finding

**检验结果显示混合结论：视觉检验基本满足平行趋势假设，但formal统计检验检测到统计显著的趋势差异。**

**Test results show mixed conclusions: Visual tests generally satisfy parallel trends assumption, but formal statistical tests detect statistically significant trend differences.**

### 检验结果概览 / Test Results Overview

| 变量 / Variable | 可视化<br/>Visual | 时间趋势检验<br/>Time Trend Test | Event Study | Placebo测试<br/>Placebo Test |
|----------------|----------------|-------------------------------|------------|---------------------------|
| Log(流出行程数)<br/>Log(Outflow Trips) | ✓ 基本满足<br/>Generally Satisfied | ❌ p=0.0089** | ❌ 5/7月显著 | ❌ p=0.0106** |
| Log(流入行程数)<br/>Log(Inflow Trips) | ✓ 基本满足<br/>Generally Satisfied | ❌ p=0.0369** | ❌ 6/7月显著 | ❌ p=0.0330** |
| Log(流出速度)<br/>Log(Outflow Speed) | ✓ 基本满足<br/>Generally Satisfied | ❌ p=0.0229** | ❌ 5/7月显著 | ❌ p=0.0345** |
| Log(流入速度)<br/>Log(Inflow Speed) | ✓ 基本满足<br/>Generally Satisfied | ❌ p=0.0002*** | ❌ 7/7月显著 | ❌ p=0.0017*** |

**注 / Note**: ** p<0.05, *** p<0.01

### 主要发现 / Main Findings

#### 1. **视觉检验vs统计检验的差异** / Visual vs Statistical Test Differences

**视觉检验（Informal）/ Visual Test (Informal)**:
- ✓ 趋势线基本平行，满足平行趋势假设
- ✓ Trend lines are basically parallel, satisfying parallel trends assumption
- ✓ 主要是水平差异，而非趋势差异
- ✓ Mainly level differences, not trend differences

**统计检验（Formal）/ Statistical Tests (Formal)**:
- ❌ 检测到统计显著的趋势差异
- ❌ Statistically significant trend differences detected
- ⚠️ 但效应大小很小（系数约0.0001-0.0003/天）
- ⚠️ But effect sizes are very small (coefficients ~0.0001-0.0003/day)

#### 2. **关键洞察** / Key Insights

1. **大样本效应** / Large Sample Effect
   - 样本量很大（20万+观测），微小差异也会统计显著
   - Very large sample size (200,000+ observations), tiny differences become statistically significant
   - **统计显著性 ≠ 实际重要性**
   - **Statistical significance ≠ Practical importance**

2. **DiD结果仍然有参考价值** / DiD Results Still Have Reference Value
   - 不应完全否定DiD估计
   - Should not completely reject DiD estimates
   - 但需要谨慎解释，承认一定的不确定性
   - But need cautious interpretation, acknowledging some uncertainty

3. **平衡的评估** / Balanced Assessment
   - 从实践角度：趋势差异在可接受范围内
   - From practical perspective: Trend differences within acceptable range
   - 从统计角度：存在显著差异，建议进行稳健性检查
   - From statistical perspective: Significant differences exist, robustness checks recommended

---

## 建议 / Recommendations

### 对原始DiD分析的影响 / Implications for Original DiD Analysis

#### 平衡的结论 / Balanced Conclusion

**DiD结果仍然有效，但需要谨慎解释** / DiD results are still valid but require cautious interpretation

1. **乐观视角（基于视觉检验）** / Optimistic View (Based on Visual Test)
   - ✓ 趋势基本平行，DiD假设大致成立
   - ✓ Trends are basically parallel, DiD assumptions roughly hold
   - ✓ 结果可以作为政策效应的近似估计
   - ✓ Results can serve as approximate estimates of policy effects
   - ✓ 偏误程度可能较小
   - ✓ Degree of bias may be small

2. **谨慎视角（基于统计检验）** / Cautious View (Based on Statistical Tests)
   - ⚠️ 存在统计显著的趋势差异
   - ⚠️ Statistically significant trend differences exist
   - ⚠️ 建议进行稳健性检查
   - ⚠️ Robustness checks recommended
   - ⚠️ 需要承认不确定性
   - ⚠️ Need to acknowledge uncertainty

#### 报告建议 / Reporting Recommendations

1. **推荐的表述方式** / Recommended Phrasing
   - ✅ "结果显示政策与...相关联"
   - ✅ "Results show policy is associated with..."
   - ✅ "DiD估计提供了政策效应的近似评估"
   - ✅ "DiD estimates provide approximate assessment of policy effects"
   - ✅ "在承认平行趋势假设存在轻微偏离的前提下..."
   - ✅ "Acknowledging slight deviations from parallel trends assumption..."

2. **避免的表述** / Phrasing to Avoid
   - ❌ "明确证明了因果关系"
   - ❌ "Definitively proves causal relationship"
   - ❌ "完全满足DiD的所有假设"
   - ❌ "Completely satisfies all DiD assumptions"

3. **稳健性检查建议** / Robustness Check Recommendations
   - 趋势调整的DiD (Trend-adjusted DiD)
   - Event Study框架
   - 倾向得分加权 (Propensity Score Weighting)
   - 不同对照组定义 (Different control group definitions)
   - 子样本分析 (Subsample analysis)

### 进一步分析建议 / Suggestions for Further Analysis

1. **子样本分析** / Subsample analysis
   - 按时间段分析（工作日/周末、高峰/非高峰）
   - Analyze by time periods (weekday/weekend, peak/off-peak)

2. **不同对照组** / Different control groups
   - 尝试使用不同的对照组定义
   - Try different control group definitions

3. **稳健性检查** / Robustness checks
   - 不同时间窗口
   - Different time windows
   - 不同控制变量组合
   - Different combinations of control variables

---

## 如何使用这些结果 / How to Use These Results

### 对于研究者 / For Researchers

1. **阅读详细报告** - 从`parallel_trends_report_zh.md`或`parallel_trends_report_en.md`开始
2. **查看可视化** - 观察趋势图和Event Study图，理解违反的模式
3. **检查数据** - 使用`parallel_trends_summary.csv`获取关键统计量
4. **深入分析** - 在R中加载`parallel_trends_detailed_results.rds`进行进一步探索

### 对于决策者 / For Policymakers

1. **关注关键发现** - 本README的"快速结论"部分
2. **理解局限性** - 原始DiD估计的因果解释可能不可靠
3. **考虑额外证据** - 需要其他研究方法来支持因果推断

---

## 技术细节 / Technical Details

### 检验方法 / Testing Methods

1. **可视化趋势检验** (Informal)
   - 按周汇总，绘制时间趋势图

2. **时间趋势差异检验** (Formal)
   - 回归: `Y ~ Treatment + Time + Treatment×Time + Controls`
   - 使用pre-policy数据

3. **Event Study分析** (Formal)
   - 回归: `Y ~ Treatment + Σ(Treatment×Month_m) + Controls`
   - 检验pre-policy月份系数

4. **Placebo测试** (Informal)
   - 假设虚假政策日期 (2024-05-01)
   - 检验是否有显著"效应"

### 控制变量 / Control Variables

所有检验都控制了：
- 星期、小时、假期
- Day of week, hour, holiday
- 天气（温度、降水、风速、降雪）
- Weather (temperature, precipitation, wind speed, snowfall)

### 稳健标准误 / Robust Standard Errors

所有回归使用HC3稳健标准误。
All regressions use HC3 robust standard errors.

---

## 生成信息 / Generation Information

- **生成日期** / Generated on: 2025-12-09
- **脚本文件** / Script: `parallel_trends_tests.r`
- **数据来源** / Data source: `data/nta_zone_hourly_taxi_summary.csv`
- **R版本** / R version: 根据系统安装 / According to system installation

---

## 联系方式 / Contact

如有问题，请联系 / For questions, please contact:

**Yanjie Chen**
- 项目 / Project: NYC CBD拥堵收费政策评估 / NYC CBD Congestion Pricing Policy Evaluation

