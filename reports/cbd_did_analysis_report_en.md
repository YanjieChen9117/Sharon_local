# Difference-in-Differences Analysis Report: Policy Impact on CBD and CBD Neighbor Areas

**Analysis Date:** November 17, 2025  
**Policy Implementation Date:** January 5, 2025  
**Data Source:** NYC NTA Zone Hourly Taxi Summary Data

---

## Executive Summary

This report presents a Difference-in-Differences (DiD) analysis examining the impact of a policy implemented on January 5, 2025, on taxi operations in CBD (Central Business District) areas and CBD neighbor areas in New York City. The analysis compares pre-policy (January 6 - August 31, 2024) and post-policy (January 6 - August 31, 2025) periods.

### Key Findings

1. **Outflow Average Speed**: The policy had a **statistically significant positive effect** on outflow average speed in CBD areas. The treatment effect is **0.2059 mph** (p < 0.001, 95% CI: [0.1278, 0.2840]).

2. **Trip Volumes**: The policy did not have statistically significant effects on trip volumes:
   - **Outflow Trips**: Treatment effect = -0.4670 (p = 0.787, not significant)
   - **Inflow Trips**: Treatment effect = -2.2979 (p = 0.158, not significant)

3. **Inflow Average Speed**: The policy did not have a statistically significant effect on inflow average speed (treatment effect = 0.0014, p = 0.971).

---

## Methodology

### DiD Model Specification

The Difference-in-Differences model is specified as:

**Y = β₀ + β₁×Treatment + β₂×Post + β₃×(Treatment×Post) + β₄×Controls + ε**

Where:
- **Y**: Outcome variable (trip volumes or average speeds)
- **Treatment**: 1 if CBD area, 0 if CBD neighbor area
- **Post**: 1 if post-policy period (2025-01-06 to 2025-08-31), 0 if pre-policy period (2024-01-06 to 2024-08-31)
- **Treatment×Post**: Interaction term capturing the policy effect (β₃ is the DiD estimate)
- **Controls**: Day of week (factor), hour of day (factor), holiday, weather_temperature, weather_precipitation, weather_windspeed, weather_humidity

### Data Description

- **Total observations**: 240,807 valid observations
- **Pre-policy period**: 120,666 observations
- **Post-policy period**: 120,141 observations
- **Treatment group (CBD)**: 194,939 observations
- **Control group (CBD neighbor)**: 45,868 observations

### CBD and CBD Neighbor Zones

**CBD Zones (Treatment Group):**
MN0191, MN0101, MN0102, MN0301, MN0201, MN0302, MN0203, MN0202, MN0303, MN0401, MN0501, MN0601, MN0602, MN0603, MN0402, MN0502, MN0604

**CBD Neighbor Zones (Control Group):**
MN0701, MN6491, MN0802, MN0801

### Statistical Methods

- **Regression Method**: Ordinary Least Squares (OLS) regression
- **Standard Errors**: Heteroskedasticity-robust standard errors (HC3)
- **Significance Levels**: *** p<0.01, ** p<0.05, * p<0.1

---

## Descriptive Statistics

### Summary Statistics by Group and Period

| Group | Period | N | Outflow Trips (Mean) | Inflow Trips (Mean) | Outflow Avg Speed (Mean) | Inflow Avg Speed (Mean) |
|-------|--------|---|----------------------|---------------------|--------------------------|-------------------------|
| Control (CBD Neighbor) | Pre-policy | 22,984 | 191.23 | 189.48 | 12.58 | 12.51 |
| Control (CBD Neighbor) | Post-policy | 22,884 | 192.29 | 192.01 | 12.34 | 12.46 |
| Treatment (CBD) | Pre-policy | 97,682 | 129.90 | 122.93 | 12.00 | 12.55 |
| Treatment (CBD) | Post-policy | 97,257 | 130.60 | 123.23 | 11.97 | 12.50 |

### Simple DiD Estimates (Without Controls)

| Outcome Variable | Treatment Change | Control Change | DiD Estimate |
|------------------|------------------|----------------|--------------|
| Outflow Trips | 0.6927 | 1.0597 | **-0.3670** |
| Inflow Trips | 0.3036 | 2.5251 | **-2.2214** |
| Outflow Avg Speed | -0.0250 | -0.2350 | **0.2100** |
| Inflow Avg Speed | -0.0511 | -0.0521 | **0.0011** |

---

## Detailed Results

### 1. Outflow Trips (流出行程数)

**Treatment Effect (β₃)**: -0.4670  
**Standard Error**: 1.7289  
**P-value**: 0.787  
**95% Confidence Interval**: [-3.8557, 2.9217]  
**Significance**: Not significant

**Interpretation**: The policy did not have a statistically significant effect on the number of outflow trips from CBD areas relative to CBD neighbor areas.

**Model Statistics**:
- R-squared: 0.1957
- Adjusted R-squared: 0.1956
- Residual standard error: 165.8
- F-statistic: 1584 (p < 2.2e-16)

---

### 2. Inflow Trips (流入行程数)

**Treatment Effect (β₃)**: -2.2979  
**Standard Error**: 1.6294  
**P-value**: 0.158  
**95% Confidence Interval**: [-5.4916, 0.8957]  
**Significance**: Not significant

**Interpretation**: The policy did not have a statistically significant effect on the number of inflow trips to CBD areas relative to CBD neighbor areas, though the effect is negative and approaches significance at the 10% level.

**Model Statistics**:
- R-squared: 0.2485
- Adjusted R-squared: 0.2484
- Residual standard error: 139.6
- F-statistic: 2152 (p < 2.2e-16)

---

### 3. Outflow Average Speed (流出平均速度) ⭐ **SIGNIFICANT**

**Treatment Effect (β₃)**: **0.2059**  
**Standard Error**: 0.0398  
**P-value**: **< 0.001***  
**95% Confidence Interval**: **[0.1278, 0.2840]**  
**Significance**: **Highly significant (***)**

**Interpretation**: The policy had a **statistically significant positive effect** on outflow average speed in CBD areas. After controlling for day of week, hour of day, holidays, and weather conditions, the policy increased outflow average speed by approximately **0.21 mph** in CBD areas relative to CBD neighbor areas.

**Magnitude**: This represents approximately a **1.7% increase** in outflow average speed (0.2059 / 12.00 baseline speed).

**Model Statistics**:
- R-squared: 0.2805
- Adjusted R-squared: 0.2804
- Residual standard error: 4.618
- F-statistic: 2537 (p < 2.2e-16)

---

### 4. Inflow Average Speed (流入平均速度)

**Treatment Effect (β₃)**: 0.0014  
**Standard Error**: 0.0374  
**P-value**: 0.971  
**95% Confidence Interval**: [-0.0719, 0.0746]  
**Significance**: Not significant

**Interpretation**: The policy did not have a statistically significant effect on inflow average speed in CBD areas relative to CBD neighbor areas.

**Model Statistics**:
- R-squared: 0.2366
- Adjusted R-squared: 0.2364
- Residual standard error: 4.244
- F-statistic: 2016 (p < 2.2e-16)

---

## Results Summary Table

| Outcome Variable | Treatment Effect | Std. Error | P-value | 95% CI Lower | 95% CI Upper | Significance | N Obs |
|------------------|------------------|------------|---------|--------------|--------------|---------------|-------|
| Outflow Trips | -0.4670 | 1.7289 | 0.787 | -3.8557 | 2.9217 | | 240,807 |
| Inflow Trips | -2.2979 | 1.6294 | 0.158 | -5.4916 | 0.8957 | | 240,807 |
| **Outflow Avg Speed** | **0.2059** | **0.0398** | **< 0.001** | **0.1278** | **0.2840** | ***** | **240,807** |
| Inflow Avg Speed | 0.0014 | 0.0374 | 0.971 | -0.0719 | 0.0746 | | 240,807 |

**Significance codes**: *** p<0.01, ** p<0.05, * p<0.1

---

## Discussion

### Main Findings

1. **Speed Improvement**: The most notable finding is the statistically significant increase in outflow average speed in CBD areas. This suggests that the policy may have improved traffic flow for vehicles leaving CBD areas, potentially due to reduced congestion or improved traffic management.

2. **No Significant Impact on Trip Volumes**: The policy did not significantly affect trip volumes (both inflow and outflow), suggesting that the policy may not have substantially changed travel demand patterns between CBD and CBD neighbor areas.

3. **Asymmetric Effects**: The policy appears to have different effects on outflow vs. inflow speeds. While outflow speed increased significantly, inflow speed was unaffected. This asymmetry may reflect differences in traffic patterns, congestion levels, or policy implementation effects for entering vs. leaving CBD areas.

### Limitations and Considerations

1. **Parallel Trends Assumption**: The DiD method assumes that, in the absence of the policy, treatment and control groups would have followed similar trends. This assumption should be validated through pre-policy trend analysis.

2. **Control Variables**: The analysis controls for day of week, hour of day, holidays, and weather conditions. However, other unobserved factors (e.g., economic conditions, special events) may still influence the results.

3. **Time Period**: The analysis covers January 6 to August 31 for both pre- and post-policy periods, ensuring comparable seasonal patterns. However, longer-term effects may differ.

4. **Spatial Heterogeneity**: The analysis aggregates all CBD zones and CBD neighbor zones. There may be heterogeneity within these groups that is not captured in the aggregate analysis.

---

## Conclusions

The DiD analysis reveals that the policy implemented on January 5, 2025, had a **statistically significant positive effect on outflow average speed** in CBD areas relative to CBD neighbor areas. Specifically, the policy increased outflow average speed by approximately **0.21 mph** (1.7% relative to baseline), with this effect being highly statistically significant (p < 0.001).

However, the policy did not have statistically significant effects on:
- Outflow trip volumes
- Inflow trip volumes  
- Inflow average speed

These findings suggest that the policy may have improved traffic flow for vehicles leaving CBD areas without substantially altering trip demand patterns. Further analysis, including pre-policy trend validation and examination of potential mechanisms, would strengthen these conclusions.

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

