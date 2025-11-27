# Difference-in-Differences Analysis Report: Policy Impact on CBD Areas

**Analysis Date:** November 17, 2025  
**Policy Implementation Date:** January 5, 2025  
**Data Source:** NYC NTA Zone Hourly Taxi Summary Data

---

## Executive Summary

This report presents a Difference-in-Differences (DiD) analysis examining the impact of a policy implemented on January 5, 2025, on taxi operations in CBD (Central Business District) areas in New York City. The analysis compares pre-policy (January 6 - August 31, 2024) and post-policy (January 6 - August 31, 2025) periods, with CBD areas as the treatment group and all other areas as the control group.

### Key Findings

1. **Trip Volumes**: The policy had **statistically significant negative effects** on trip volumes:
   - **Outflow Trips**: Treatment effect = **-0.3956** (p < 0.001, 95% CI: [-0.6156, -0.1756]) ***
   - **Inflow Trips**: Treatment effect = **-1.3669** (p < 0.001, 95% CI: [-1.6194, -1.1144]) ***

2. **Average Speeds**: The policy had **statistically significant positive effects** on average speeds:
   - **Outflow Average Speed**: Treatment effect = **0.1109 mph** (p = 0.001, 95% CI: [0.0439, 0.1780]) ***
   - **Inflow Average Speed**: Treatment effect = **0.2149 mph** (p < 0.001, 95% CI: [0.1481, 0.2816]) ***

---

## Methodology

### DiD Model Specification

The Difference-in-Differences model is specified as:

**Y = β₀ + β₁×Treatment + β₂×Post + β₃×(Treatment×Post) + β₄×Controls + ε**

Where:
- **Y**: Outcome variable (trip volumes or average speeds)
- **Treatment**: 1 if CBD area, 0 if others
- **Post**: 1 if post-policy period (2025-01-06 to 2025-08-31), 0 if pre-policy period (2024-01-06 to 2024-08-31)
- **Treatment×Post**: Interaction term capturing the policy effect (β₃ is the DiD estimate)
- **Controls**: Day of week (factor), hour of day (factor), holiday, weather_temperature, weather_precipitation, weather_windspeed, weather_humidity, and financial variables:
  - For outflow analyses: outflow_total_tip, outflow_total_tolls, outflow_total_fare
  - For inflow analyses: inflow_total_tip, inflow_total_tolls, inflow_total_fare

### Data Description

- **Total observations**: 470,147 valid observations
- **Pre-policy period**: 235,586 observations
- **Post-policy period**: 234,561 observations
- **Treatment group (CBD)**: 194,939 observations
- **Control group (Others)**: 275,208 observations

### CBD Zones

**CBD Zones (Treatment Group):**
MN0191, MN0101, MN0102, MN0301, MN0201, MN0302, MN0203, MN0202, MN0303, MN0401, MN0501, MN0601, MN0602, MN0603, MN0402, MN0502, MN0604

**Control Group:** All other NTA zones in New York City

### Statistical Methods

- **Regression Method**: Ordinary Least Squares (OLS) regression
- **Standard Errors**: Heteroskedasticity-robust standard errors (HC3)
- **Significance Levels**: *** p<0.01, ** p<0.05, * p<0.1

---

## Descriptive Statistics

### Summary Statistics by Group and Period

| Group | Period | N | Outflow Trips (Mean) | Inflow Trips (Mean) | Outflow Avg Speed (Mean) | Inflow Avg Speed (Mean) |
|-------|--------|---|----------------------|---------------------|--------------------------|-------------------------|
| Control (Others) | Pre-policy | 137,904 | 77.43 | 84.21 | 10.20 | 13.38 |
| Control (Others) | Post-policy | 137,304 | 77.43 | 84.65 | 10.08 | 13.16 |
| Treatment (CBD) | Pre-policy | 97,682 | 129.90 | 122.93 | 12.00 | 12.55 |
| Treatment (CBD) | Post-policy | 97,257 | 130.60 | 123.23 | 11.97 | 12.50 |

### Simple DiD Estimates (Without Controls)

| Outcome Variable | Treatment Change | Control Change | DiD Estimate |
|------------------|------------------|----------------|--------------|
| Outflow Trips | 0.6927 | 0.0007 | **0.6920** |
| Inflow Trips | 0.3036 | 0.4393 | **-0.1356** |
| Outflow Avg Speed | -0.0250 | -0.1165 | **0.0916** |
| Inflow Avg Speed | -0.0511 | -0.2234 | **0.1723** |

---

## Detailed Results

### 1. Outflow Trips (流出行程数) ⭐ **SIGNIFICANT**

**Treatment Effect (β₃)**: **-0.3956**  
**Standard Error**: 0.1122  
**P-value**: **< 0.001***  
**95% Confidence Interval**: **[-0.6156, -0.1756]**  
**Significance**: **Highly significant (***)**

**Interpretation**: The policy had a **statistically significant negative effect** on the number of outflow trips from CBD areas relative to other areas. After controlling for day of week, hour of day, holidays, weather conditions, and financial variables (outflow_total_tip, outflow_total_tolls, outflow_total_fare), the policy decreased outflow trips by approximately **0.40 trips per hour** in CBD areas.

**Model Statistics**:
- R-squared: 0.9879
- Adjusted R-squared: 0.9879
- Residual standard error: 17.95
- F-statistic: 9.576e+05 (p < 2.2e-16)

---

### 2. Inflow Trips (流入行程数) ⭐ **SIGNIFICANT**

**Treatment Effect (β₃)**: **-1.3669**  
**Standard Error**: 0.1288  
**P-value**: **< 0.001***  
**95% Confidence Interval**: **[-1.6194, -1.1144]**  
**Significance**: **Highly significant (***)**

**Interpretation**: The policy had a **statistically significant negative effect** on the number of inflow trips to CBD areas relative to other areas. After controlling for day of week, hour of day, holidays, weather conditions, and financial variables (inflow_total_tip, inflow_total_tolls, inflow_total_fare), the policy decreased inflow trips by approximately **1.37 trips per hour** in CBD areas.

**Model Statistics**:
- R-squared: 0.9809
- Adjusted R-squared: 0.9809
- Residual standard error: 20.51
- F-statistic: 6.041e+05 (p < 2.2e-16)

---

### 3. Outflow Average Speed (流出平均速度) ⭐ **SIGNIFICANT**

**Treatment Effect (β₃)**: **0.1109**  
**Standard Error**: 0.0342  
**P-value**: **0.001***  
**95% Confidence Interval**: **[0.0439, 0.1780]**  
**Significance**: **Highly significant (***)**

**Interpretation**: The policy had a **statistically significant positive effect** on outflow average speed in CBD areas. After controlling for day of week, hour of day, holidays, weather conditions, and financial variables (outflow_total_tip, outflow_total_tolls, outflow_total_fare), the policy increased outflow average speed by approximately **0.11 mph** in CBD areas relative to other areas.

**Magnitude**: This represents approximately a **0.9% increase** in outflow average speed (0.1109 / 12.00 baseline speed).

**Model Statistics**:
- R-squared: 0.1279
- Adjusted R-squared: 0.1278
- Residual standard error: 6.098
- F-statistic: 1724 (p < 2.2e-16)

---

### 4. Inflow Average Speed (流入平均速度) ⭐ **SIGNIFICANT**

**Treatment Effect (β₃)**: **0.2149**  
**Standard Error**: 0.0341  
**P-value**: **< 0.001***  
**95% Confidence Interval**: **[0.1481, 0.2816]**  
**Significance**: **Highly significant (***)**

**Interpretation**: The policy had a **statistically significant positive effect** on inflow average speed in CBD areas. After controlling for day of week, hour of day, holidays, weather conditions, and financial variables (inflow_total_tip, inflow_total_tolls, inflow_total_fare), the policy increased inflow average speed by approximately **0.21 mph** in CBD areas relative to other areas.

**Magnitude**: This represents approximately a **1.7% increase** in inflow average speed (0.2149 / 12.55 baseline speed).

**Model Statistics**:
- R-squared: 0.1443
- Adjusted R-squared: 0.1442
- Residual standard error: 6.23
- F-statistic: 1982 (p < 2.2e-16)

---

## Results Summary Table

| Outcome Variable | Treatment Effect | Std. Error | P-value | 95% CI Lower | 95% CI Upper | Significance | N Obs |
|------------------|------------------|------------|---------|--------------|--------------|---------------|-------|
| **Outflow Trips** | **-0.3956** | **0.1122** | **< 0.001** | **-0.6156** | **-0.1756** | ***** | **470,147** |
| **Inflow Trips** | **-1.3669** | **0.1288** | **< 0.001** | **-1.6194** | **-1.1144** | ***** | **470,147** |
| **Outflow Avg Speed** | **0.1109** | **0.0342** | **0.001** | **0.0439** | **0.1780** | ***** | **470,147** |
| **Inflow Avg Speed** | **0.2149** | **0.0341** | **< 0.001** | **0.1481** | **0.2816** | ***** | **470,147** |

**Significance codes**: *** p<0.01, ** p<0.05, * p<0.1

---

## Discussion

### Main Findings

1. **Speed Improvements**: The policy had statistically significant positive effects on both outflow and inflow average speeds in CBD areas. This suggests that the policy may have improved traffic flow in CBD areas, potentially due to reduced congestion or improved traffic management. The effect on inflow speed (0.21 mph) is approximately twice as large as the effect on outflow speed (0.11 mph).

2. **Trip Volume Reductions**: The policy had statistically significant negative effects on both outflow and inflow trip volumes in CBD areas. This suggests that the policy may have reduced travel demand to and from CBD areas. The reduction in inflow trips (-1.37 trips/hour) is more than three times larger than the reduction in outflow trips (-0.40 trips/hour).

3. **Comprehensive Effects**: Unlike the previous analysis that only found significant effects on outflow speed, the updated analysis with expanded control group and additional financial control variables reveals significant effects across all four outcome variables. This suggests that the policy had broader impacts on CBD traffic patterns than initially observed.

### Limitations and Considerations

1. **Parallel Trends Assumption**: The DiD method assumes that, in the absence of the policy, treatment and control groups would have followed similar trends. This assumption should be validated through pre-policy trend analysis.

2. **Control Variables**: The analysis controls for day of week, hour of day, holidays, weather conditions, and financial variables (total tips, total tolls, total fare). The financial variables are matched to the outcome variable (outflow variables for outflow analyses, inflow variables for inflow analyses). However, other unobserved factors (e.g., economic conditions, special events) may still influence the results.

3. **Time Period**: The analysis covers January 6 to August 31 for both pre- and post-policy periods, ensuring comparable seasonal patterns. However, longer-term effects may differ.

4. **Spatial Heterogeneity**: The analysis aggregates all CBD zones and compares them to all other zones. There may be heterogeneity within these groups that is not captured in the aggregate analysis.

5. **Control Group Expansion**: The control group has been expanded from CBD neighbor zones to all other zones in New York City. This provides a larger and more diverse control group, which may improve the robustness of the estimates but also introduces greater heterogeneity.

---

## Conclusions

The DiD analysis reveals that the policy implemented on January 5, 2025, had **statistically significant effects on all four outcome variables** in CBD areas relative to other areas:

1. **Speed Improvements**: The policy increased both outflow and inflow average speeds:
   - Outflow average speed: **+0.11 mph** (p = 0.001)
   - Inflow average speed: **+0.21 mph** (p < 0.001)

2. **Trip Volume Reductions**: The policy decreased both outflow and inflow trip volumes:
   - Outflow trips: **-0.40 trips/hour** (p < 0.001)
   - Inflow trips: **-1.37 trips/hour** (p < 0.001)

These findings suggest that the policy had a **comprehensive impact** on CBD traffic patterns: while it improved traffic flow (as measured by average speeds), it also reduced travel demand (as measured by trip volumes). The larger reduction in inflow trips compared to outflow trips, combined with the larger increase in inflow speed compared to outflow speed, suggests that the policy may have had a particularly strong effect on traffic entering CBD areas.

Further analysis, including pre-policy trend validation and examination of potential mechanisms, would strengthen these conclusions.

---

## Technical Notes

- **Analysis Script**: `cbd_did_analysis.R`
- **Data File**: `data/nta_zone_hourly_taxi_summary.csv`
- **Software**: R (with packages: dplyr, lubridate, sandwich, lmtest)
- **Standard Errors**: Heteroskedasticity-robust (HC3)
- **Date Range**: 
  - Pre-policy: 2024-01-06 to 2024-08-31
  - Post-policy: 2025-01-06 to 2025-08-31

---

**Report Generated**: November 17, 2025  
**Analyst**: Yanjie Chen

