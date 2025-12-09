# Difference-in-Differences Analysis Report: Policy Impact on CBD Areas

**Analysis Date:** December 9, 2025  
**Policy Implementation Date:** January 5, 2025  
**Data Source:** NYC NTA Zone Hourly Taxi Summary Data

---

## Executive Summary

This report presents a Difference-in-Differences (DiD) analysis examining the impact of a policy implemented on January 5, 2025, on taxi operations in CBD (Central Business District) areas in New York City. The analysis compares pre-policy (January 6 - August 31, 2024) and post-policy (January 6 - August 31, 2025) periods, with CBD areas as the treatment group and all other areas as the control group.

### Key Findings

**Note**: All outcome variables were log-transformed before regression, so treatment effects represent changes on the log scale, approximately equal to percentage changes.

1. **Trip Volumes**: The policy had **statistically significant positive effects** on trip volumes:
   - **Outflow Trips**: Treatment effect = **0.0418** (p < 0.001, 95% CI: [0.0269, 0.0567]) ***, approximately **4.2% increase**
   - **Inflow Trips**: Treatment effect = **0.0260** (p < 0.001, 95% CI: [0.0132, 0.0388]) ***, approximately **2.6% increase**

2. **Average Speeds**: The policy had **statistically significant positive effects** on average speeds:
   - **Outflow Average Speed**: Treatment effect = **0.0154** (p = 0.0055, 95% CI: [0.0045, 0.0262]) ***, approximately **1.5% increase**
   - **Inflow Average Speed**: Treatment effect = **0.0221** (p < 0.001, 95% CI: [0.0125, 0.0316]) ***, approximately **2.2% increase**

---

## Methodology

### DiD Model Specification

The Difference-in-Differences model is specified as:

**Y = β₀ + β₁×Treatment + β₂×Post + β₃×(Treatment×Post) + β₄×Controls + ε**

Where:
- **Y**: Outcome variable (log-transformed trip volumes or average speeds)
- **Treatment**: 1 if CBD area, 0 if others
- **Post**: 1 if post-policy period (2025-01-06 to 2025-08-31), 0 if pre-policy period (2024-01-06 to 2024-08-31)
- **Treatment×Post**: Interaction term capturing the policy effect (β₃ is the DiD estimate, representing changes on the log scale)
- **Controls**: Day of week (factor), hour of day (factor), holiday, weather_temperature, weather_precipitation, weather_windspeed, weather_snow, and financial variables:
  - For outflow analyses: outflow_total_tip, outflow_total_tolls, outflow_total_fare
  - For inflow analyses: inflow_total_tip, inflow_total_tolls, inflow_total_fare

**Important Notes**:
- All outcome variables (outflow_trips, inflow_trips, outflow_avg_speed, inflow_avg_speed) were log-transformed before regression
- For trip count variables, log(1+x) was used to handle zero values
- For speed variables, log(x) or log(1+x) was used depending on the data
- On the log scale, coefficients approximately represent percentage changes (e.g., 0.0418 ≈ 4.18% increase)

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

- **Regression Method**: Ordinary Least Squares (OLS) regression (using standard lm function)
- **Standard Errors**: Heteroskedasticity-robust standard errors (HC3, using sandwich package)
- **Collinearity Check**: Identified perfectly collinear variables by checking for NA values in model coefficients
- **Variable Transformation**: All outcome variables were log-transformed before regression
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

### Simple DiD Estimates (Without Controls, Log-Transformed)

| Outcome Variable | Treatment Change | Control Change | DiD Estimate |
|------------------|------------------|----------------|--------------|
| Outflow Trips (log) | -0.0105 | -0.0174 | **0.0069** |
| Inflow Trips (log) | -0.0081 | -0.0356 | **0.0275** |
| Outflow Avg Speed (log) | 0.0036 | -0.0030 | **0.0065** |
| Inflow Avg Speed (log) | -0.0047 | -0.0190 | **0.0143** |

**Note**: These are DiD estimates on the log scale, approximately representing percentage changes.

---

## Detailed Results

### 1. Outflow Trips ⭐ **SIGNIFICANT**

**Treatment Effect (β₃)**: **0.0418** (log scale)  
**Standard Error**: 0.0076  
**P-value**: **< 0.001***  
**95% Confidence Interval**: **[0.0269, 0.0567]**  
**Significance**: **Highly significant (***)**

**Interpretation**: The policy had a **statistically significant positive effect** on the number of outflow trips from CBD areas relative to other areas. After controlling for day of week, hour of day, holidays, weather conditions, and financial variables (outflow_total_tip, outflow_total_tolls, outflow_total_fare), the policy increased outflow trips by approximately **4.2%** (on the log scale, coefficient 0.0418 approximately equals a 4.18% percentage change).

**Model Statistics**:
- N observations: 412,812
- Robust standard errors (HC3) used

---

### 2. Inflow Trips ⭐ **SIGNIFICANT**

**Treatment Effect (β₃)**: **0.0260** (log scale)  
**Standard Error**: 0.0065  
**P-value**: **< 0.001***  
**95% Confidence Interval**: **[0.0132, 0.0388]**  
**Significance**: **Highly significant (***)**

**Interpretation**: The policy had a **statistically significant positive effect** on the number of inflow trips to CBD areas relative to other areas. After controlling for day of week, hour of day, holidays, weather conditions, and financial variables (inflow_total_tip, inflow_total_tolls, inflow_total_fare), the policy increased inflow trips by approximately **2.6%** (on the log scale, coefficient 0.0260 approximately equals a 2.60% percentage change).

**Model Statistics**:
- N observations: 412,812
- Robust standard errors (HC3) used

---

### 3. Outflow Average Speed ⭐ **SIGNIFICANT**

**Treatment Effect (β₃)**: **0.0154** (log scale)  
**Standard Error**: 0.0055  
**P-value**: **0.0055***  
**95% Confidence Interval**: **[0.0045, 0.0262]**  
**Significance**: **Highly significant (***)**

**Interpretation**: The policy had a **statistically significant positive effect** on outflow average speed in CBD areas. After controlling for day of week, hour of day, holidays, weather conditions, and financial variables (outflow_total_tip, outflow_total_tolls, outflow_total_fare), the policy increased outflow average speed by approximately **1.5%** (on the log scale, coefficient 0.0154 approximately equals a 1.54% percentage change).

**Model Statistics**:
- N observations: 412,812
- Robust standard errors (HC3) used

---

### 4. Inflow Average Speed ⭐ **SIGNIFICANT**

**Treatment Effect (β₃)**: **0.0221** (log scale)  
**Standard Error**: 0.0049  
**P-value**: **< 0.001***  
**95% Confidence Interval**: **[0.0125, 0.0316]**  
**Significance**: **Highly significant (***)**

**Interpretation**: The policy had a **statistically significant positive effect** on inflow average speed in CBD areas. After controlling for day of week, hour of day, holidays, weather conditions, and financial variables (inflow_total_tip, inflow_total_tolls, inflow_total_fare), the policy increased inflow average speed by approximately **2.2%** (on the log scale, coefficient 0.0221 approximately equals a 2.21% percentage change).

**Model Statistics**:
- N observations: 412,812
- Robust standard errors (HC3) used

---

## Results Summary Table

| Outcome Variable | Treatment Effect (log scale) | Std. Error | P-value | 95% CI Lower | 95% CI Upper | Significance | N Obs |
|------------------|------------------------------|------------|---------|--------------|--------------|---------------|-------|
| **Outflow Trips** | **0.0418** | **0.0076** | **< 0.001** | **0.0269** | **0.0567** | ***** | **412,812** |
| **Inflow Trips** | **0.0260** | **0.0065** | **< 0.001** | **0.0132** | **0.0388** | ***** | **412,812** |
| **Outflow Avg Speed** | **0.0154** | **0.0055** | **0.0055** | **0.0045** | **0.0262** | ***** | **412,812** |
| **Inflow Avg Speed** | **0.0221** | **0.0049** | **< 0.001** | **0.0125** | **0.0316** | ***** | **412,812** |

**Note**: All outcome variables were log-transformed before regression. Treatment effects represent changes on the log scale, approximately equal to percentage changes (e.g., 0.0418 ≈ 4.18% increase).

**Significance codes**: *** p<0.01, ** p<0.05, * p<0.1

---

## Discussion

### Main Findings

1. **Speed Improvements**: The policy had statistically significant positive effects on both outflow and inflow average speeds in CBD areas. On the log scale, the effect on inflow speed (2.2%) is slightly larger than the effect on outflow speed (1.5%). This suggests that the policy may have improved traffic flow in CBD areas, potentially due to reduced congestion or improved traffic management.

2. **Trip Volume Increases**: The policy had statistically significant positive effects on both outflow and inflow trip volumes in CBD areas. On the log scale, the increase in outflow trips (4.2%) is larger than the increase in inflow trips (2.6%). This suggests that the policy may have increased travel demand to and from CBD areas.

3. **Comprehensive Effects**: The analysis using log transformation and standard regression methods reveals statistically significant positive effects across all four outcome variables. This suggests that the policy had comprehensive positive impacts on CBD traffic patterns.

### Limitations and Considerations

1. **Parallel Trends Assumption**: The DiD method assumes that, in the absence of the policy, treatment and control groups would have followed similar trends. This assumption should be validated through pre-policy trend analysis.

2. **Control Variables**: The analysis controls for day of week, hour of day, holidays, weather conditions (temperature, precipitation, windspeed, snow), and financial variables (total tips, total tolls, total fare). The financial variables are matched to the outcome variable (outflow variables for outflow analyses, inflow variables for inflow analyses). Collinearity was checked by identifying NA values in model coefficients, and no obvious collinearity problems were detected. However, other unobserved factors (e.g., economic conditions, special events) may still influence the results.

3. **Log Transformation**: All outcome variables were log-transformed before regression. For trip count variables, log(1+x) was used to handle zero values; for speed variables, log(x) or log(1+x) was used depending on the data. On the log scale, coefficients approximately represent percentage changes, which helps interpret the economic significance of the results.

4. **Time Period**: The analysis covers January 6 to August 31 for both pre- and post-policy periods, ensuring comparable seasonal patterns. However, longer-term effects may differ.

5. **Spatial Heterogeneity**: The analysis aggregates all CBD zones and compares them to all other zones. There may be heterogeneity within these groups that is not captured in the aggregate analysis.

6. **Control Group Expansion**: The control group has been expanded from CBD neighbor zones to all other zones in New York City. This provides a larger and more diverse control group, which may improve the robustness of the estimates but also introduces greater heterogeneity.

---

## Conclusions

The DiD analysis reveals that the policy implemented on January 5, 2025, had **statistically significant positive effects on all four outcome variables** in CBD areas relative to other areas (all outcome variables were log-transformed before regression):

1. **Speed Improvements**: The policy increased both outflow and inflow average speeds:
   - Outflow average speed: **+1.5%** (p = 0.0055, log scale coefficient = 0.0154)
   - Inflow average speed: **+2.2%** (p < 0.001, log scale coefficient = 0.0221)

2. **Trip Volume Increases**: The policy increased both outflow and inflow trip volumes:
   - Outflow trips: **+4.2%** (p < 0.001, log scale coefficient = 0.0418)
   - Inflow trips: **+2.6%** (p < 0.001, log scale coefficient = 0.0260)

These findings suggest that the policy had a **comprehensive positive impact** on CBD traffic patterns: it both improved traffic flow (as measured by average speeds) and increased travel demand (as measured by trip volumes). The larger increase in outflow trips (4.2%) compared to inflow trips (2.6%), combined with the slightly larger increase in inflow speed (2.2%) compared to outflow speed (1.5%), suggests that the policy had positive and comprehensive effects on CBD area traffic.

Further analysis, including pre-policy trend validation and examination of potential mechanisms, would strengthen these conclusions.

---

## Technical Notes

- **Analysis Script**: `cbd_did_analysis.R`
- **Data File**: `data/nta_zone_hourly_taxi_summary.csv`
- **Software**: R (with packages: dplyr, lubridate, lmtest, sandwich)
- **Regression Method**: OLS regression using standard lm function
- **Standard Errors**: Heteroskedasticity-robust (HC3, using sandwich package)
- **Variable Transformation**: All outcome variables were log-transformed before regression
  - Trip count variables: log(1+x) used to handle zero values
  - Speed variables: log(x) or log(1+x) used depending on the data
- **Collinearity Check**: Identified perfectly collinear variables by checking for NA values in model coefficients
- **Date Range**: 
  - Pre-policy: 2024-01-06 to 2024-08-31
  - Post-policy: 2025-01-06 to 2025-08-31

---

**Report Generated**: December 9, 2025  
**Analyst**: Yanjie Chen

