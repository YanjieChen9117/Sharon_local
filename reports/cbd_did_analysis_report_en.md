# Difference-in-Differences Analysis Report: Policy Impact on CBD Areas

**Analysis Date:** December 8, 2025  
**Policy Implementation Date:** January 5, 2025  
**Data Source:** NYC NTA Zone Hourly Taxi Summary Data

---

## Executive Summary

This report presents a Difference-in-Differences (DiD) analysis examining the impact of a policy implemented on January 5, 2025, on taxi operations in CBD (Central Business District) areas in New York City. The analysis compares pre-policy (January 6 - August 31, 2024) and post-policy (January 6 - August 31, 2025) periods, with CBD areas as the treatment group and all other areas as the control group.

### Key Findings

1. **Trip Volumes**: The policy had **statistically significant negative effects** on trip volumes:
   - **Outflow Trips**: Treatment effect = **-0.4089** (p = 0.0003, 95% CI: [-0.6324, -0.1855]) ***
   - **Inflow Trips**: Treatment effect = **-1.8402** (p < 0.001, 95% CI: [-2.0988, -1.5817]) ***

2. **Average Speeds**: The policy had **statistically significant positive effects** on average speeds:
   - **Outflow Average Speed**: Treatment effect = **0.1043 mph** (p = 0.005, 95% CI: [0.0323, 0.1764]) ***
   - **Inflow Average Speed**: Treatment effect = **0.2446 mph** (p < 0.001, 95% CI: [0.1701, 0.3192]) ***

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
- **Controls**: Day of week (factor), hour of day (factor), holiday, weather_temperature, weather_precipitation, weather_windspeed, weather_snow, and financial variables:
  - For outflow analyses: outflow_total_tip, outflow_total_tolls, outflow_total_fare
  - For inflow analyses: inflow_total_tip, inflow_total_tolls, inflow_total_fare

### Data Description

- **Total observations**: 412,812 valid observations
- **Pre-policy period**: 206,856 observations
- **Post-policy period**: 205,956 observations
- **Treatment group (CBD)**: 183,472 observations
- **Control group (Others)**: 229,340 observations

### CBD Zones

**CBD Zones (Treatment Group):**
MN0101, MN0102, MN0301, MN0201, MN0302, MN0601, MN0203, MN0202, MN0303, MN0401, MN0501, MN0602, MN0603, MN0402, MN0502, MN0604

**Control Group:** All other NTA zones in New York City

### Statistical Methods

- **Regression Method**: Ordinary Least Squares (OLS) regression (using fixest package)
- **Standard Errors**: Heteroskedasticity-robust standard errors (HC3)
- **Collinearity Check**: Using fixest package's collinearity function to check for collinearity among control variables
- **Significance Levels**: *** p<0.01, ** p<0.05, * p<0.1

---

## Descriptive Statistics

### Summary Statistics by Group and Period

| Group | Period | N | Outflow Trips (Mean) | Inflow Trips (Mean) | Outflow Avg Speed (Mean) | Inflow Avg Speed (Mean) |
|-------|--------|---|----------------------|---------------------|--------------------------|-------------------------|
| Control (Others) | Pre-policy | 114,920 | 58.74 | 64.37 | 9.37 | 12.93 |
| Control (Others) | Post-policy | 114,420 | 58.21 | 64.30 | 9.26 | 12.71 |
| Treatment (CBD) | Pre-policy | 91,936 | 137.91 | 130.45 | 12.36 | 12.88 |
| Treatment (CBD) | Post-policy | 91,536 | 138.66 | 130.79 | 12.35 | 12.84 |

### Simple DiD Estimates (Without Controls)

| Outcome Variable | Treatment Change | Control Change | DiD Estimate |
|------------------|------------------|----------------|--------------|
| Outflow Trips | 0.7502 | -0.5230 | **1.2732** |
| Inflow Trips | 0.3425 | -0.0701 | **0.4126** |
| Outflow Avg Speed | -0.0110 | -0.1012 | **0.0902** |
| Inflow Avg Speed | -0.0383 | -0.2127 | **0.1744** |

---

## Detailed Results

### 1. Outflow Trips ⭐ **SIGNIFICANT**

**Treatment Effect (β₃)**: **-0.4089**  
**Standard Error**: 0.1140  
**P-value**: **0.0003***  
**95% Confidence Interval**: **[-0.6324, -0.1855]**  
**Significance**: **Highly significant (***)**

**Interpretation**: The policy had a **statistically significant negative effect** on the number of outflow trips from CBD areas relative to other areas. After controlling for day of week, hour of day, holidays, weather conditions, and financial variables (outflow_total_tip, outflow_total_tolls, outflow_total_fare), the policy decreased outflow trips by approximately **0.41 trips per hour** in CBD areas.

**Model Statistics**:
- Adjusted R-squared: 0.9877
- Residual standard error (RMSE): 17.5
- N observations: 412,812

---

### 2. Inflow Trips ⭐ **SIGNIFICANT**

**Treatment Effect (β₃)**: **-1.8402**  
**Standard Error**: 0.1319  
**P-value**: **< 0.001***  
**95% Confidence Interval**: **[-2.0988, -1.5817]**  
**Significance**: **Highly significant (***)**

**Interpretation**: The policy had a **statistically significant negative effect** on the number of inflow trips to CBD areas relative to other areas. After controlling for day of week, hour of day, holidays, weather conditions, and financial variables (inflow_total_tip, inflow_total_tolls, inflow_total_fare), the policy decreased inflow trips by approximately **1.84 trips per hour** in CBD areas.

**Model Statistics**:
- Adjusted R-squared: 0.9796
- Residual standard error (RMSE): 20.1
- N observations: 412,812

---

### 3. Outflow Average Speed ⭐ **SIGNIFICANT**

**Treatment Effect (β₃)**: **0.1043**  
**Standard Error**: 0.0368  
**P-value**: **0.005***  
**95% Confidence Interval**: **[0.0323, 0.1764]**  
**Significance**: **Highly significant (***)**

**Interpretation**: The policy had a **statistically significant positive effect** on outflow average speed in CBD areas. After controlling for day of week, hour of day, holidays, weather conditions, and financial variables (outflow_total_tip, outflow_total_tolls, outflow_total_fare), the policy increased outflow average speed by approximately **0.10 mph** in CBD areas relative to other areas.

**Magnitude**: This represents approximately a **0.8% increase** in outflow average speed (0.1043 / 12.36 baseline speed).

**Model Statistics**:
- Adjusted R-squared: 0.1426
- Residual standard error (RMSE): 6.19
- N observations: 412,812

---

### 4. Inflow Average Speed ⭐ **SIGNIFICANT**

**Treatment Effect (β₃)**: **0.2446**  
**Standard Error**: 0.0380  
**P-value**: **< 0.001***  
**95% Confidence Interval**: **[0.1701, 0.3192]**  
**Significance**: **Highly significant (***)**

**Interpretation**: The policy had a **statistically significant positive effect** on inflow average speed in CBD areas. After controlling for day of week, hour of day, holidays, weather conditions, and financial variables (inflow_total_tip, inflow_total_tolls, inflow_total_fare), the policy increased inflow average speed by approximately **0.24 mph** in CBD areas relative to other areas.

**Magnitude**: This represents approximately a **1.9% increase** in inflow average speed (0.2446 / 12.88 baseline speed).

**Model Statistics**:
- Adjusted R-squared: 0.1208
- Residual standard error (RMSE): 6.53
- N observations: 412,812

---

## Results Summary Table

| Outcome Variable | Treatment Effect | Std. Error | P-value | 95% CI Lower | 95% CI Upper | Significance | N Obs |
|------------------|------------------|------------|---------|--------------|--------------|---------------|-------|
| **Outflow Trips** | **-0.4089** | **0.1140** | **0.0003** | **-0.6324** | **-0.1855** | ***** | **412,812** |
| **Inflow Trips** | **-1.8402** | **0.1319** | **< 0.001** | **-2.0988** | **-1.5817** | ***** | **412,812** |
| **Outflow Avg Speed** | **0.1043** | **0.0368** | **0.005** | **0.0323** | **0.1764** | ***** | **412,812** |
| **Inflow Avg Speed** | **0.2446** | **0.0380** | **< 0.001** | **0.1701** | **0.3192** | ***** | **412,812** |

**Significance codes**: *** p<0.01, ** p<0.05, * p<0.1

---

## Discussion

### Main Findings

1. **Speed Improvements**: The policy had statistically significant positive effects on both outflow and inflow average speeds in CBD areas. This suggests that the policy may have improved traffic flow in CBD areas, potentially due to reduced congestion or improved traffic management. The effect on inflow speed (0.24 mph) is approximately 2.4 times as large as the effect on outflow speed (0.10 mph).

2. **Trip Volume Reductions**: The policy had statistically significant negative effects on both outflow and inflow trip volumes in CBD areas. This suggests that the policy may have reduced travel demand to and from CBD areas. The reduction in inflow trips (-1.84 trips/hour) is 4.5 times larger than the reduction in outflow trips (-0.41 trips/hour).

3. **Comprehensive Effects**: Unlike the previous analysis that only found significant effects on outflow speed, the updated analysis with expanded control group and additional financial control variables reveals significant effects across all four outcome variables. This suggests that the policy had broader impacts on CBD traffic patterns than initially observed.

### Limitations and Considerations

1. **Parallel Trends Assumption**: The DiD method assumes that, in the absence of the policy, treatment and control groups would have followed similar trends. This assumption should be validated through pre-policy trend analysis.

2. **Control Variables**: The analysis controls for day of week, hour of day, holidays, weather conditions (temperature, precipitation, windspeed, snow), and financial variables (total tips, total tolls, total fare). The financial variables are matched to the outcome variable (outflow variables for outflow analyses, inflow variables for inflow analyses). Collinearity checks were performed on control variables using the fixest package, and no obvious collinearity problems were detected. However, other unobserved factors (e.g., economic conditions, special events) may still influence the results.

3. **Time Period**: The analysis covers January 6 to August 31 for both pre- and post-policy periods, ensuring comparable seasonal patterns. However, longer-term effects may differ.

4. **Spatial Heterogeneity**: The analysis aggregates all CBD zones and compares them to all other zones. There may be heterogeneity within these groups that is not captured in the aggregate analysis.

5. **Control Group Expansion**: The control group has been expanded from CBD neighbor zones to all other zones in New York City. This provides a larger and more diverse control group, which may improve the robustness of the estimates but also introduces greater heterogeneity.

---

## Conclusions

The DiD analysis reveals that the policy implemented on January 5, 2025, had **statistically significant effects on all four outcome variables** in CBD areas relative to other areas:

1. **Speed Improvements**: The policy increased both outflow and inflow average speeds:
   - Outflow average speed: **+0.10 mph** (p = 0.005)
   - Inflow average speed: **+0.24 mph** (p < 0.001)

2. **Trip Volume Reductions**: The policy decreased both outflow and inflow trip volumes:
   - Outflow trips: **-0.41 trips/hour** (p = 0.0003)
   - Inflow trips: **-1.84 trips/hour** (p < 0.001)

These findings suggest that the policy had a **comprehensive impact** on CBD traffic patterns: while it improved traffic flow (as measured by average speeds), it also reduced travel demand (as measured by trip volumes). The much larger reduction in inflow trips (-1.84 trips/hour) compared to outflow trips (-0.41 trips/hour), combined with the larger increase in inflow speed (+0.24 mph) compared to outflow speed (+0.10 mph), suggests that the policy may have had a particularly strong effect on traffic entering CBD areas.

Further analysis, including pre-policy trend validation and examination of potential mechanisms, would strengthen these conclusions.

---

## Technical Notes

- **Analysis Script**: `cbd_did_analysis.R`
- **Data File**: `data/nta_zone_hourly_taxi_summary.csv`
- **Software**: R (with packages: dplyr, lubridate, fixest)
- **Regression Method**: OLS regression using feols function from fixest package
- **Standard Errors**: Heteroskedasticity-robust (HC3)
- **Collinearity Check**: Using collinearity function from fixest package to check for collinearity among control variables
- **Date Range**: 
  - Pre-policy: 2024-01-06 to 2024-08-31
  - Post-policy: 2025-01-06 to 2025-08-31

---

**Report Generated**: December 8, 2025  
**Analyst**: Yanjie Chen

