# NYC CBD Congestion Pricing Policy Impact Analysis - Comprehensive

**Analysis Date:** 2025-10-08  
**Policy Implementation Date:** January 5, 2025  
**Data Source:** NYC Yellow & Green Taxi Hourly Summary Data (Combined)  

## Executive Summary

### Key Findings

1. **Average Speed Impact:**
   - Overall change: -0.64%
   - Statistical significance: Not significant (p = 0.1700)
   - Peak hours DiD effect: -0.0598 mph

2. **Trip Volume Impact:**
   - Total trips change: -3.19%
   - Statistical significance: Significant (p = 0.0003)

3. **CBD Usage Patterns:**
   - CBD internal trips change: 1.89%
   - Statistical significance: Significant (p = 0.0000)
   - CBD exposure DiD effect: -0.7958 mph

4. **Substitution Effect (CBD Neighbor Areas):**
   - CBD Neighbor In ratio change: 1.22%
   - Statistical significance: Significant (p = 0.0009)
   - CBD Neighbor Out ratio change: -0.84%
   - Statistical significance: Not significant (p = 0.1192)
   - **Interpretation:** Evidence of substitution effect - passengers avoiding CBD by using neighboring areas

## Data Description

- **Dataset:** Hourly aggregated NYC Yellow & Green Taxi data (weighted combination)
- **Time Period:** 2023-01-01 to 2025-08-31
- **Total Observations:** 23,373 hours
- **Pre-policy Period:** 17,638 hours
- **Post-policy Period:** 5,735 hours
- **Yellow Taxi Proportion:** 97.96%
- **Green Taxi Proportion:** 2.04%

## Methodology

### 1. Data Aggregation
Yellow and Green taxi data are combined using weighted averaging:
- **Sum metrics** (total_trips, total_revenue): Direct summation
- **Average metrics** (avg_speed, avg_distance, etc.): Weighted by total_trips
- **Ratio metrics** (cbd_inside_ratio, etc.): Weighted by total_trips

This ensures that taxi types with more trips have proportionally more influence on the combined metrics.

### 2. Descriptive Analysis
- Simple before/after comparisons using t-tests
- Peak vs off-peak hour analysis
- Statistical significance testing at α = 0.05

### 3. Difference-in-Differences (DiD) Analysis

#### 3.1 DiD Methodology Overview

The Difference-in-Differences (DiD) method estimates the causal effect of a policy by comparing the change in outcomes over time between a treatment group and a control group.

**Basic DiD Formula:**

$$
\text{DiD} = (\bar{Y}_{\text{treatment, post}} - \bar{Y}_{\text{treatment, pre}}) - (\bar{Y}_{\text{control, post}} - \bar{Y}_{\text{control, pre}})
$$

Where:
- $\bar{Y}_{\text{treatment, post}}$ = Average outcome for treatment group after policy
- $\bar{Y}_{\text{treatment, pre}}$ = Average outcome for treatment group before policy
- $\bar{Y}_{\text{control, post}}$ = Average outcome for control group after policy
- $\bar{Y}_{\text{control, pre}}$ = Average outcome for control group before policy

**Key Assumption:** Parallel trends - without the policy, treatment and control groups would have followed similar trends.

#### 3.2 DiD Implementation in This Analysis

**Analysis 1: Peak vs Off-Peak Hours**

- **Treatment Group:** Peak hours (7-10 AM, 4-7 PM) - more affected by congestion pricing
- **Control Group:** Off-peak hours - less affected by congestion pricing
- **Outcome Variable:** Average speed (mph)

Formula:
$$
\text{DiD}_{\text{peak}} = (\text{Speed}_{\text{peak, post}} - \text{Speed}_{\text{peak, pre}}) - (\text{Speed}_{\text{offpeak, post}} - \text{Speed}_{\text{offpeak, pre}})
$$

**Result:** DiD = -0.0598 mph

**Analysis 2: CBD Exposure**

- **Treatment Group:** High CBD exposure hours (above median CBD interaction)
- **Control Group:** Low CBD exposure hours (below median CBD interaction)
- **Outcome Variable:** Average speed (mph)

Formula:
$$
\text{DiD}_{\text{CBD}} = (\text{Speed}_{\text{high CBD, post}} - \text{Speed}_{\text{high CBD, pre}}) - (\text{Speed}_{\text{low CBD, post}} - \text{Speed}_{\text{low CBD, pre}})
$$

**Result:** DiD = -0.7958 mph

#### 3.3 Discussion: Equal Sample Size Requirement

**Professor's Concern:** "DiD requires equal number of observations in pre- and post-policy periods."

**Evaluation:**

1. **Theoretical Requirement:** DiD does NOT strictly require equal sample sizes. The method is valid as long as:
   - The parallel trends assumption holds
   - Both periods have sufficient observations for reliable estimation
   - Standard errors are properly calculated

2. **Our Data:**
   - Pre-policy: 17,638 hours
   - Post-policy: 5,735 hours
   - Ratio: 3.08:1

3. **Implications:**
   - **Unequal sample sizes are acceptable** in DiD analysis
   - Larger pre-policy sample provides better baseline estimation
   - Standard t-tests and regression-based DiD handle unequal samples naturally
   - **Concern:** If sample sizes are very imbalanced, statistical power may be reduced
   - **Mitigation:** We have substantial observations in both periods (5,000+ hours each)

4. **Best Practices:**
   - Use regression-based DiD for formal inference: $Y_{it} = \beta_0 + \beta_1 \text{Post}_t + \beta_2 \text{Treatment}_i + \beta_3 (\text{Post}_t \times \text{Treatment}_i) + \epsilon_{it}$
   - Where $\beta_3$ is the DiD estimate
   - This approach properly accounts for unequal sample sizes and provides standard errors
   - Consider time-varying confounders and seasonality

**Conclusion:** The equal sample size "requirement" is a misconception. Our analysis is methodologically sound with unequal sample sizes.

## Results

### Pre/Post Comparison
             avg_speed  total_trips  avg_duration  avg_distance  cbd_inside_ratio  cbd_in_ratio  cbd_out_ratio  cbd_non_ratio  cbd_neighbor_inside_ratio  cbd_neighbor_in_ratio  cbd_neighbor_out_ratio  cbd_neighbor_non_ratio  total_revenue  avg_fare
Pre-policy     13.1434    4124.6105       15.8364        3.9013            0.4002        0.1347         0.1889         0.2763                     0.0241                 0.0938                  0.0918                  0.7903    118968.3263   20.7284
Post-policy    13.0595    3993.1238       15.6937        3.8632            0.4077        0.1348         0.1794         0.2780                     0.0250                 0.0950                  0.0910                  0.7890    115708.0487   20.5318
Change (%)     -0.6400      -3.1900       -0.9000       -0.9800            1.8900        0.1000        -5.0100         0.6400                     3.6400                 1.2200                 -0.8400                 -0.1600        -2.7400   -0.9500

### Peak vs Off-Peak Analysis
           Period  Pre_avg_speed  Post_avg_speed  avg_speed_change_%  Pre_cbd_inside_ratio  Post_cbd_inside_ratio  cbd_inside_ratio_change_%
0      Peak hours      11.198451       11.074730               -1.10              0.384719               0.394651                       2.58
1  Off-peak hours      14.116005       14.052134               -0.45              0.407882               0.414276                       1.57

## Policy Impact Assessment

### Speed Effects
The analysis shows no statistically significant changes in average speeds following the CBD congestion pricing implementation. The overall speed change was -0.64%.

### CBD Usage Patterns
CBD internal trip ratios show a increase of 1.89%, which is statistically significant.

### Trip Volume Effects
Overall taxi trip volumes decreased by 3.19%, suggesting demand reduction.

### Substitution Effect Analysis (CBD Neighbor Areas)

To detect whether passengers are avoiding the congestion charge by using neighboring areas instead of CBD:

**CBD Neighbor In Ratio (trips entering CBD neighbor areas):**
- Change: +1.22%
- Statistical significance: Significant (p = 0.0009)

**CBD Neighbor Out Ratio (trips leaving CBD neighbor areas):**
- Change: -0.84%
- Statistical significance: Not significant (p = 0.1192)

**Interpretation:**
There is evidence of a substitution effect. Passengers appear to be avoiding the CBD congestion charge by shifting their trips to neighboring areas. This suggests the policy is having its intended deterrent effect on CBD usage, but may be displacing congestion to adjacent areas.

## Difference-in-Differences Results

1. **Peak Hours Effect:** -0.0598 mph differential impact during peak hours
2. **CBD Exposure Effect:** -0.7958 mph differential impact for high CBD exposure areas

## Conclusions

Based on the comprehensive analysis of NYC Yellow & Green Taxi hourly data:

1. **Speed Effects:** No evidence of speed improvements following policy implementation
2. **Usage Patterns:** No evidence of reduced CBD usage as intended by the policy
3. **Substitution Effect:** Evidence suggests passengers are avoiding CBD by using neighboring areas
4. **Overall Impact:** The policy shows mixed results in the initial implementation period

## Limitations

1. **Simplified Analysis:** This analysis uses basic statistical methods and may not capture complex causal relationships
2. **Confounding Factors:** Weather, holidays, and other concurrent changes not controlled for
3. **Short-term Data:** Limited post-policy observation period (approximately 8 months)
4. **Data Scope:** Only Yellow & Green Taxi data, excluding other transportation modes (Uber, Lyft, public transit, private vehicles)
5. **Parallel Trends Assumption:** DiD assumes parallel trends, which we have not formally tested
6. **Seasonality:** Monthly and seasonal patterns may confound the policy effect

---

**Analysis conducted by:** Yanjie Chen
**Files generated:** Tables in `tables/`, Figures in `figures/`
