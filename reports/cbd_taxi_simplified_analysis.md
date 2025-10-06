# NYC CBD Congestion Pricing Policy Impact Analysis - Simplified

**Analysis Date:** 2025-10-06  
**Policy Implementation Date:** January 5, 2025  
**Data Source:** NYC Yellow Taxi Hourly Summary Data  

## Executive Summary

### Key Findings

1. **Average Speed Impact:**
   - Overall change: -0.68%
   - Statistical significance: Not significant (p = 0.1466)
   - Peak hours DiD effect: -0.0425 mph

2. **Trip Volume Impact:**
   - Total trips change: -3.40%
   - Statistical significance: Significant (p = 0.0001)

3. **CBD Usage Patterns:**
   - CBD internal trips change: 1.14%
   - Statistical significance: Significant (p = 0.0001)
   - CBD exposure DiD effect: -0.7941 mph

## Data Description

- **Dataset:** Hourly aggregated NYC Yellow Taxi data
- **Time Period:** 2023-01-01 to 2025-08-31
- **Total Observations:** 23,277 hours
- **Pre-policy Period:** 17,542 hours
- **Post-policy Period:** 5,735 hours

## Methodology

### 1. Descriptive Analysis
- Simple before/after comparisons
- Peak vs off-peak hour analysis
- Statistical t-tests for significance

### 2. Difference-in-Differences
- Peak intensity comparison (peak vs off-peak hours)
- CBD exposure comparison (high vs low CBD interaction)

## Results

### Pre/Post Comparison
             avg_speed  total_trips  avg_duration  avg_distance  cbd_inside_ratio  cbd_in_ratio  cbd_out_ratio  cbd_non_ratio  total_revenue  avg_fare
Pre-policy     13.1465    4072.4320       15.8773        3.9187            0.4091        0.1358         0.1931         0.2620    117860.7829   20.7801
Post-policy    13.0570    3934.1643       15.7237        3.8739            0.4138        0.1356         0.1820         0.2686    114272.3260   20.5568
Change (%)     -0.6800      -3.4000       -0.9700       -1.1400            1.1400       -0.1600        -5.7200         2.5100        -3.0400   -1.0700

### Peak vs Off-Peak Analysis
           Period  Pre_avg_speed  Post_avg_speed  avg_speed_change_%  Pre_cbd_inside_ratio  Post_cbd_inside_ratio  cbd_inside_ratio_change_%
0      Peak hours      11.182637       11.064828               -1.05              0.394420               0.401519                       1.80
1  Off-peak hours      14.128668       14.053347               -0.53              0.416436               0.419893                       0.83

## Policy Impact Assessment

### Speed Effects
The analysis shows no statistically significant changes in average speeds following the CBD congestion pricing implementation. The overall speed change was -0.68%.

### CBD Usage Patterns
CBD internal trip ratios show a increase of 1.14%, which is statistically significant.

### Trip Volume Effects
Overall taxi trip volumes decreased by 3.40%, suggesting demand reduction.

## Difference-in-Differences Results

1. **Peak Hours Effect:** -0.0425 mph differential impact during peak hours
2. **CBD Exposure Effect:** -0.7941 mph differential impact for high CBD exposure areas

## Conclusions

Based on the simplified analysis of NYC Yellow Taxi hourly data:

1. **Speed Effects:** No evidence of speed improvements following policy implementation
2. **Usage Patterns:** No evidence of reduced CBD usage as intended by the policy
3. **Overall Impact:** The policy shows mixed results in the initial implementation period

## Limitations

1. **Simplified Analysis:** This analysis uses basic statistical methods and may not capture complex causal relationships
2. **Confounding Factors:** Weather, holidays, and other concurrent changes not controlled for
3. **Short-term Data:** Limited post-policy observation period
4. **Data Scope:** Only Yellow Taxi data, excluding other transportation modes

---

**Analysis conducted by:** Yanjie Chen
**Files generated:** Tables in `tables/`, Figures in `figures/`
