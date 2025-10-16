# NYC CBD Congestion Pricing Policy Impact Analysis - DiD Regression Model

**Analysis Date:** 2025-10-12  
**Policy Implementation Date:** January 5, 2025  
**Data Source:** NYC Yellow & Green Taxi Hourly Summary Data (Combined)  
**Methodology:** Difference-in-Differences Regression Analysis

## Model Specification

The analysis uses the standard DiD regression specification:

**Y_it = α + βD_i + γPost_t + δ(D_i × Post_t) + ε_it**

Where:
- **Y_it**: Outcome variable for observation i at time t
- **D_i**: Treatment group indicator (1 if treated, 0 if control)
- **Post_t**: Post-policy period indicator (1 if after policy, 0 if before)
- **δ**: DiD estimator (the coefficient of interest)
- **ε_it**: Error term

## Treatment Scenarios

1. **CBD Exposure Analysis**: High vs Low CBD exposure areas
2. **Peak Hours Analysis**: Peak vs Off-peak hours  
3. **Weekday Analysis**: Weekday vs Weekend patterns

## Data Description

- **Dataset:** Hourly aggregated NYC Yellow & Green Taxi data
- **Time Period:** 2023-01-01 to 2025-08-31
- **Total Observations:** 23,373 hours
- **Pre-policy Period:** 17,638 hours
- **Post-policy Period:** 5,735 hours
- **Treatment Groups:** 3 scenarios analyzed

## Key Findings

### Overall Results Summary

- **Total Regression Models:** 30
- **Statistically Significant Effects:** 14 (46.7%)
- **Mean R-squared:** 0.058

### Significant Treatment Effects (p < 0.05)


#### High vs Low CBD Exposure - Average Speed (mph)
- **DiD Estimate:** -0.7358 ***
- **Standard Error:** 0.1074
- **P-value:** 0.0000
- **95% Confidence Interval:** [-0.9463, -0.5253]
- **R-squared:** 0.008

#### High vs Low CBD Exposure - Total Trips per Hour
- **DiD Estimate:** 480.1498 ***
- **Standard Error:** 71.5498
- **P-value:** 0.0000
- **95% Confidence Interval:** [339.9147, 620.3848]
- **R-squared:** 0.003

#### High vs Low CBD Exposure - CBD In Ratio
- **DiD Estimate:** 0.0070 ***
- **Standard Error:** 0.0012
- **P-value:** 0.0000
- **95% Confidence Interval:** [0.0047, 0.0094]
- **R-squared:** 0.021

#### High vs Low CBD Exposure - CBD Out Ratio
- **DiD Estimate:** -0.0052 ***
- **Standard Error:** 0.0009
- **P-value:** 0.0000
- **95% Confidence Interval:** [-0.0069, -0.0035]
- **R-squared:** 0.097

#### High vs Low CBD Exposure - CBD Neighbor Out Ratio
- **DiD Estimate:** 0.0023 ***
- **Standard Error:** 0.0005
- **P-value:** 0.0000
- **95% Confidence Interval:** [0.0013, 0.0033]
- **R-squared:** 0.074

#### High vs Low CBD Exposure - Average Trip Duration (minutes)
- **DiD Estimate:** 0.3466 ***
- **Standard Error:** 0.0669
- **P-value:** 0.0000
- **95% Confidence Interval:** [0.2155, 0.4776]
- **R-squared:** 0.034

#### High vs Low CBD Exposure - Average Trip Distance (miles)
- **DiD Estimate:** -0.1225 ***
- **Standard Error:** 0.0287
- **P-value:** 0.0000
- **95% Confidence Interval:** [-0.1787, -0.0662]
- **R-squared:** 0.001

#### High vs Low CBD Exposure - Total Revenue per Hour
- **DiD Estimate:** 14888.8769 ***
- **Standard Error:** 2064.4775
- **P-value:** 0.0000
- **95% Confidence Interval:** [10842.5754, 18935.1784]
- **R-squared:** 0.004

#### Peak vs Off-Peak Hours - Average Speed (mph)
- **DiD Estimate:** -0.2414 **
- **Standard Error:** 0.0939
- **P-value:** 0.0101
- **95% Confidence Interval:** [-0.4255, -0.0574]
- **R-squared:** 0.119

#### Peak vs Off-Peak Hours - CBD In Ratio
- **DiD Estimate:** -0.0030 ***
- **Standard Error:** 0.0011
- **P-value:** 0.0075
- **95% Confidence Interval:** [-0.0052, -0.0008]
- **R-squared:** 0.040

#### Peak vs Off-Peak Hours - CBD Out Ratio
- **DiD Estimate:** 0.0060 ***
- **Standard Error:** 0.0007
- **P-value:** 0.0000
- **95% Confidence Interval:** [0.0046, 0.0074]
- **R-squared:** 0.212

#### Peak vs Off-Peak Hours - Average Trip Distance (miles)
- **DiD Estimate:** -0.0681 ***
- **Standard Error:** 0.0248
- **P-value:** 0.0061
- **95% Confidence Interval:** [-0.1168, -0.0194]
- **R-squared:** 0.097

#### Weekday vs Weekend - CBD Neighbor Out Ratio
- **DiD Estimate:** 0.0011 **
- **Standard Error:** 0.0005
- **P-value:** 0.0483
- **95% Confidence Interval:** [0.0000, 0.0021]
- **R-squared:** 0.044

#### Weekday vs Weekend - Average Trip Duration (minutes)
- **DiD Estimate:** 0.1618 **
- **Standard Error:** 0.0654
- **P-value:** 0.0134
- **95% Confidence Interval:** [0.0336, 0.2901]
- **R-squared:** 0.026


## Detailed Results

The following table shows all DiD regression results:

| Treatment | Outcome Variable | DiD Estimate | Standard Error | P-value | 95% CI Lower | 95% CI Upper | R-squared |
|-----------|------------------|--------------|----------------|---------|--------------|--------------|-----------|
| High vs Low CBD Exposure | Average Speed (mph) | -0.7358*** | 0.1074 | 0.0000 | -0.9463 | -0.5253 | 0.008 |
| High vs Low CBD Exposure | Total Trips per Hour | 480.1498*** | 71.5498 | 0.0000 | 339.9147 | 620.3848 | 0.003 |
| High vs Low CBD Exposure | CBD Inside Ratio | 0.0021* | 0.0011 | 0.0637 | -0.0001 | 0.0042 | 0.263 |
| High vs Low CBD Exposure | CBD In Ratio | 0.0070*** | 0.0012 | 0.0000 | 0.0047 | 0.0094 | 0.021 |
| High vs Low CBD Exposure | CBD Out Ratio | -0.0052*** | 0.0009 | 0.0000 | -0.0069 | -0.0035 | 0.097 |
| High vs Low CBD Exposure | CBD Neighbor In Ratio | 0.0006 | 0.0009 | 0.5546 | -0.0013 | 0.0024 | 0.061 |
| High vs Low CBD Exposure | CBD Neighbor Out Ratio | 0.0023*** | 0.0005 | 0.0000 | 0.0013 | 0.0033 | 0.074 |
| High vs Low CBD Exposure | Average Trip Duration (minutes) | 0.3466*** | 0.0669 | 0.0000 | 0.2155 | 0.4776 | 0.034 |
| High vs Low CBD Exposure | Average Trip Distance (miles) | -0.1225*** | 0.0287 | 0.0000 | -0.1787 | -0.0662 | 0.001 |
| High vs Low CBD Exposure | Total Revenue per Hour | 14888.8769*** | 2064.4775 | 0.0000 | 10842.5754 | 18935.1784 | 0.004 |
| Peak vs Off-Peak Hours | Average Speed (mph) | -0.2414** | 0.0939 | 0.0101 | -0.4255 | -0.0574 | 0.119 |
| Peak vs Off-Peak Hours | Total Trips per Hour | 32.3465 | 67.1405 | 0.6300 | -99.2465 | 163.9395 | 0.101 |
| Peak vs Off-Peak Hours | CBD Inside Ratio | 0.0019* | 0.0011 | 0.0902 | -0.0003 | 0.0042 | 0.016 |
| Peak vs Off-Peak Hours | CBD In Ratio | -0.0030*** | 0.0011 | 0.0075 | -0.0052 | -0.0008 | 0.040 |
| Peak vs Off-Peak Hours | CBD Out Ratio | 0.0060*** | 0.0007 | 0.0000 | 0.0046 | 0.0074 | 0.212 |
| Peak vs Off-Peak Hours | CBD Neighbor In Ratio | 0.0014* | 0.0008 | 0.0998 | -0.0003 | 0.0030 | 0.088 |
| Peak vs Off-Peak Hours | CBD Neighbor Out Ratio | 0.0001 | 0.0004 | 0.8224 | -0.0007 | 0.0009 | 0.189 |
| Peak vs Off-Peak Hours | Average Trip Duration (minutes) | -0.0412 | 0.0665 | 0.5352 | -0.1716 | 0.0891 | 0.011 |
| Peak vs Off-Peak Hours | Average Trip Distance (miles) | -0.0681*** | 0.0248 | 0.0061 | -0.1168 | -0.0194 | 0.097 |
| Peak vs Off-Peak Hours | Total Revenue per Hour | 417.2238 | 2006.4384 | 0.8353 | -3515.3233 | 4349.7708 | 0.095 |
| Weekday vs Weekend | Average Speed (mph) | -0.1496 | 0.1088 | 0.1693 | -0.3629 | 0.0637 | 0.022 |
| Weekday vs Weekend | Total Trips per Hour | 113.3975 | 70.5001 | 0.1077 | -24.7801 | 251.5751 | 0.007 |
| Weekday vs Weekend | CBD Inside Ratio | -0.0028* | 0.0017 | 0.0913 | -0.0060 | 0.0004 | 0.040 |
| Weekday vs Weekend | CBD In Ratio | -0.0014 | 0.0013 | 0.2473 | -0.0039 | 0.0010 | 0.002 |
| Weekday vs Weekend | CBD Out Ratio | 0.0003 | 0.0009 | 0.7189 | -0.0014 | 0.0021 | 0.005 |
| Weekday vs Weekend | CBD Neighbor In Ratio | 0.0016* | 0.0009 | 0.0809 | -0.0002 | 0.0035 | 0.033 |
| Weekday vs Weekend | CBD Neighbor Out Ratio | 0.0011** | 0.0005 | 0.0483 | 0.0000 | 0.0021 | 0.044 |
| Weekday vs Weekend | Average Trip Duration (minutes) | 0.1618** | 0.0654 | 0.0134 | 0.0336 | 0.2901 | 0.026 |
| Weekday vs Weekend | Average Trip Distance (miles) | -0.0109 | 0.0298 | 0.7139 | -0.0694 | 0.0475 | 0.007 |
| Weekday vs Weekend | Total Revenue per Hour | 2656.4673 | 1994.4688 | 0.1829 | -1252.6197 | 6565.5542 | 0.011 |


**Note:** *** p<0.01, ** p<0.05, * p<0.1

## Interpretation

### CBD Exposure Analysis
- Found 8 statistically significant effects in CBD exposure analysis
- Average Speed (mph): decrease of 0.7358 (p = 0.0000)
- Total Trips per Hour: increase of 480.1498 (p = 0.0000)
- CBD In Ratio: increase of 0.0070 (p = 0.0000)
- CBD Out Ratio: decrease of 0.0052 (p = 0.0000)
- CBD Neighbor Out Ratio: increase of 0.0023 (p = 0.0000)
- Average Trip Duration (minutes): increase of 0.3466 (p = 0.0000)
- Average Trip Distance (miles): decrease of 0.1225 (p = 0.0000)
- Total Revenue per Hour: increase of 14888.8769 (p = 0.0000)

### Peak Hours Analysis
- Found 4 statistically significant effects in peak hours analysis
- Average Speed (mph): decrease of 0.2414 (p = 0.0101)
- CBD In Ratio: decrease of 0.0030 (p = 0.0075)
- CBD Out Ratio: increase of 0.0060 (p = 0.0000)
- Average Trip Distance (miles): decrease of 0.0681 (p = 0.0061)

### Weekday vs Weekend Analysis
- Found 2 statistically significant effects in weekday analysis
- CBD Neighbor Out Ratio: increase of 0.0011 (p = 0.0483)
- Average Trip Duration (minutes): increase of 0.1618 (p = 0.0134)


## Conclusions

Based on the DiD regression analysis:

1. **Policy Effectiveness:** The policy shows significant effects on multiple outcomes
2. **Treatment Heterogeneity:** Different treatment scenarios show varying effects
3. **Model Fit:** Average R-squared of 0.058 indicates limited explanatory power

## Limitations

1. **Parallel Trends Assumption:** DiD methodology assumes parallel trends between treatment and control groups
2. **Confounding Factors:** Other concurrent changes (weather, holidays, etc.) not controlled for
3. **Short-term Data:** Limited post-policy observation period
4. **Selection Bias:** Treatment group assignment based on CBD exposure may introduce selection bias

## Files Generated

- **Tables:** `tables/did_*.csv`
- **Figures:** `figures/did_*.png`
- **Report:** `reports/cbd_did_simple_analysis.md`

---

**Analysis conducted by:** Senior Data Scientist  
**Methodology:** Difference-in-Differences Regression  
**Statistical Software:** Python with statsmodels
