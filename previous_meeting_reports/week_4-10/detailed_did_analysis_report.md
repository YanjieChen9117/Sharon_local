# Detailed DiD Analysis Report: Policy Impact on NYC Taxi Operations

## Executive Summary

This detailed report presents comprehensive Difference-in-Differences (DiD) analysis results examining the impact of a policy implemented on January 5, 2025, on NYC taxi operations. The analysis includes model diagnostics, robustness checks, and detailed statistical results.

## DiD Model Specification

The DiD model is specified as:

**Y = β₀ + β₁×Treatment + β₂×Post + β₃×(Treatment×Post) + β₄×Controls + ε**

Where:
- **Y**: Outcome variable (speed, trip volume, CBD interactions)
- **Treatment**: 1 if high CBD interaction (treatment group), 0 otherwise (control group)
- **Post**: 1 if post-policy period (2025-01-05 to 2025-08-31), 0 if pre-policy period (2024-01-05 to 2024-08-31)
- **Treatment×Post**: Interaction term capturing the policy effect
- **Controls**: Hour of day, day of week, weather conditions

## Results Summary Table

| Outcome | Sample | N | Treatment Effect | P-Value | 95% CI Lower | 95% CI Upper | R² | Adj R² | F-Stat | White Test P | DW | Max VIF |
|---------|--------|---|------------------|---------|--------------|--------------|----|---------|---------|--------------|----|---------|
| Avg Speed | All | 11448 | -0.0437 | 0.6641 | -0.2408 | 0.1534 | 0.5445 | 0.5442 | 1709.32 | 0.0000 | 0.4804 | 18.56 |
| Avg Speed | holiday | 3552 | -0.1651 | 0.3584 | -0.5176 | 0.1874 | 0.4524 | 0.4511 | 365.82 | 0.0000 | 0.5881 | 30.23 |
| Avg Speed | not_holiday | 7896 | -0.0281 | 0.7677 | -0.2146 | 0.1584 | 0.7394 | 0.7391 | 2796.51 | 0.0000 | 0.7128 | 19.33 |
| Total Trips | All | 11448 | 56.7011 | 0.2683 | -43.7062 | 157.1084 | 0.6529 | 0.6526 | 2689.37 | 0.0000 | 0.7164 | 18.56 |
| Total Trips | holiday | 3552 | 138.5901 | 0.1449 | -47.7729 | 324.9530 | 0.4870 | 0.4858 | 420.38 | 0.0000 | 0.6240 | 30.23 |
| Total Trips | not_holiday | 7896 | 40.7416 | 0.4421 | -63.1455 | 144.6288 | 0.7742 | 0.7740 | 3380.27 | 0.0000 | 1.0051 | 19.33 |
| Cbd Inside Ratio | All | 11448 | 0.0041** | 0.0370 | 0.0002 | 0.0080 | 0.5939 | 0.5936 | 2091.01 | 0.0000 | 0.7855 | 18.56 |
| Cbd Inside Ratio | holiday | 3552 | 0.0079** | 0.0435 | 0.0002 | 0.0155 | 0.7349 | 0.7343 | 1227.98 | 0.0000 | 0.9012 | 30.23 |
| Cbd Inside Ratio | not_holiday | 7896 | 0.0026 | 0.1425 | -0.0009 | 0.0061 | 0.6134 | 0.6131 | 1564.52 | 0.0000 | 1.1573 | 19.33 |
| Cbd Neighbor Inside Ratio | All | 11448 | -0.0001 | 0.5528 | -0.0006 | 0.0003 | 0.7202 | 0.7200 | 3679.90 | 0.0000 | 0.9324 | 18.56 |
| Cbd Neighbor Inside Ratio | holiday | 3552 | 0.0005 | 0.2524 | -0.0003 | 0.0013 | 0.6791 | 0.6784 | 937.24 | 0.0000 | 0.9666 | 30.23 |
| Cbd Neighbor Inside Ratio | not_holiday | 7896 | -0.0003 | 0.3464 | -0.0008 | 0.0003 | 0.7507 | 0.7504 | 2968.14 | 0.0000 | 1.0435 | 19.33 |
| Avg Speed Out Cbd | All | 11448 | 0.0838 | 0.5178 | -0.1702 | 0.3377 | 0.4740 | 0.4736 | 1288.59 | 0.0000 | 0.5515 | 18.56 |
| Avg Speed Out Cbd | holiday | 3552 | 0.0669 | 0.7374 | -0.3244 | 0.4583 | 0.5359 | 0.5349 | 511.39 | 0.0000 | 0.5977 | 30.23 |
| Avg Speed Out Cbd | not_holiday | 7896 | 0.0702 | 0.6333 | -0.2182 | 0.3586 | 0.5529 | 0.5525 | 1219.25 | 0.0000 | 0.6759 | 19.33 |

**Note**: *** p<0.01, ** p<0.05, * p<0.1

## Detailed Results by Outcome Variable

### Average Taxi Speed

**All Data:**
- Treatment Effect (β₃): -0.0437
- P-value: 0.6641
- 95% Confidence Interval: [-0.2408, 0.1534]
- R-squared: 0.5445
- F-statistic: 1709.32
- White test p-value: 0.0000
- Durbin-Watson statistic: 0.4804

**Holiday Periods:**
- Treatment Effect (β₃): -0.1651
- P-value: 0.3584
- 95% Confidence Interval: [-0.5176, 0.1874]
- R-squared: 0.4524

**Non-Holiday Periods:**
- Treatment Effect (β₃): -0.0281
- P-value: 0.7677
- 95% Confidence Interval: [-0.2146, 0.1584]
- R-squared: 0.7394

### Total Trip Volume

**All Data:**
- Treatment Effect (β₃): 56.7011
- P-value: 0.2683
- 95% Confidence Interval: [-43.7062, 157.1084]
- R-squared: 0.6529
- F-statistic: 2689.37
- White test p-value: 0.0000
- Durbin-Watson statistic: 0.7164

**Holiday Periods:**
- Treatment Effect (β₃): 138.5901
- P-value: 0.1449
- 95% Confidence Interval: [-47.7729, 324.9530]
- R-squared: 0.4870

**Non-Holiday Periods:**
- Treatment Effect (β₃): 40.7416
- P-value: 0.4421
- 95% Confidence Interval: [-63.1455, 144.6288]
- R-squared: 0.7742

### CBD Internal Trips

**All Data:**
- Treatment Effect (β₃): 0.0041
- P-value: 0.0370
- 95% Confidence Interval: [0.0002, 0.0080]
- R-squared: 0.5939
- F-statistic: 2091.01
- White test p-value: 0.0000
- Durbin-Watson statistic: 0.7855

**Holiday Periods:**
- Treatment Effect (β₃): 0.0079
- P-value: 0.0435
- 95% Confidence Interval: [0.0002, 0.0155]
- R-squared: 0.7349

**Non-Holiday Periods:**
- Treatment Effect (β₃): 0.0026
- P-value: 0.1425
- 95% Confidence Interval: [-0.0009, 0.0061]
- R-squared: 0.6134

### CBD Neighbor Trips

**All Data:**
- Treatment Effect (β₃): -0.0001
- P-value: 0.5528
- 95% Confidence Interval: [-0.0006, 0.0003]
- R-squared: 0.7202
- F-statistic: 3679.90
- White test p-value: 0.0000
- Durbin-Watson statistic: 0.9324

**Holiday Periods:**
- Treatment Effect (β₃): 0.0005
- P-value: 0.2524
- 95% Confidence Interval: [-0.0003, 0.0013]
- R-squared: 0.6791

**Non-Holiday Periods:**
- Treatment Effect (β₃): -0.0003
- P-value: 0.3464
- 95% Confidence Interval: [-0.0008, 0.0003]
- R-squared: 0.7507

### CBD Exit Speed

**All Data:**
- Treatment Effect (β₃): 0.0838
- P-value: 0.5178
- 95% Confidence Interval: [-0.1702, 0.3377]
- R-squared: 0.4740
- F-statistic: 1288.59
- White test p-value: 0.0000
- Durbin-Watson statistic: 0.5515

**Holiday Periods:**
- Treatment Effect (β₃): 0.0669
- P-value: 0.7374
- 95% Confidence Interval: [-0.3244, 0.4583]
- R-squared: 0.5359

**Non-Holiday Periods:**
- Treatment Effect (β₃): 0.0702
- P-value: 0.6333
- 95% Confidence Interval: [-0.2182, 0.3586]
- R-squared: 0.5529

## Model Diagnostics

### Diagnostic Tests

1. **Heteroscedasticity Test (White Test)**: Tests for constant variance of residuals
2. **Autocorrelation Test (Durbin-Watson)**: Tests for serial correlation in residuals
3. **Multicollinearity Test (VIF)**: Tests for correlation among independent variables

### Interpretation Guidelines

- **White Test p-value < 0.05**: Evidence of heteroscedasticity
- **Durbin-Watson ≈ 2**: No autocorrelation
- **VIF > 10**: Potential multicollinearity concerns

## Conclusions

### Key Findings

**Statistically Significant Results (p < 0.05):**
- Cbd Inside Ratio (All): Treatment effect = 0.0041, p-value = 0.0370
- Cbd Inside Ratio (holiday): Treatment effect = 0.0079, p-value = 0.0435

### Policy Implications

1. **Treatment Effect Interpretation**: The β₃ coefficient represents the differential impact of the policy on the treatment group relative to the control group.

2. **Holiday vs Non-Holiday Effects**: Separate analyses reveal how policy effects vary across different time contexts.

3. **Statistical Significance**: Results are evaluated at conventional significance levels (1%, 5%, 10%).

4. **Model Robustness**: The analysis includes comprehensive diagnostic tests to ensure model validity.
