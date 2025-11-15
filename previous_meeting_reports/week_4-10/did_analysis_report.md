# DiD Analysis Report: Policy Impact on NYC Taxi Operations

## Executive Summary

This report presents a Difference-in-Differences (DiD) analysis examining the impact of a policy implemented on January 5, 2025, on NYC taxi operations. The analysis compares pre-policy (January 5 - August 31, 2024) and post-policy (January 5 - August 31, 2025) periods, with separate analyses for holiday and non-holiday periods.

## DiD Model Specification

The DiD model is specified as:

**Y = β₀ + β₁×Treatment + β₂×Post + β₃×(Treatment×Post) + β₄×Controls + ε**

Where:
- **Y**: Outcome variable (speed, trip volume, CBD interactions)
- **Treatment**: 1 if high CBD interaction (treatment group), 0 otherwise (control group)
- **Post**: 1 if post-policy period (2025-01-05 to 2025-08-31), 0 if pre-policy period (2024-01-05 to 2024-08-31)
- **Treatment×Post**: Interaction term capturing the policy effect
- **Controls**: Hour of day, day of week, weather conditions

## Data Description

- **Total observations**: 11,448
- **Pre-policy period**: 5,736 hours
- **Post-policy period**: 5,712 hours
- **Treatment group**: 5,724 observations
- **Control group**: 5,724 observations

## Analysis Results

### Average Taxi Speed

**All Data:**
- Treatment Effect (β₃): -0.0437
- P-value: 0.6641
- Significance: Not significant

**Holiday Periods:**
- Treatment Effect (β₃): -0.1651
- P-value: 0.3584
- Significance: Not significant

**Non-Holiday Periods:**
- Treatment Effect (β₃): -0.0281
- P-value: 0.7677
- Significance: Not significant

### Total Trip Volume

**All Data:**
- Treatment Effect (β₃): 56.7011
- P-value: 0.2683
- Significance: Not significant

**Holiday Periods:**
- Treatment Effect (β₃): 138.5901
- P-value: 0.1449
- Significance: Not significant

**Non-Holiday Periods:**
- Treatment Effect (β₃): 40.7416
- P-value: 0.4421
- Significance: Not significant

### CBD Internal Trips

**All Data:**
- Treatment Effect (β₃): 0.0041
- P-value: 0.0370
- Significance: **

**Holiday Periods:**
- Treatment Effect (β₃): 0.0079
- P-value: 0.0435
- Significance: **

**Non-Holiday Periods:**
- Treatment Effect (β₃): 0.0026
- P-value: 0.1425
- Significance: Not significant

### CBD Neighbor Trips

**All Data:**
- Treatment Effect (β₃): -0.0001
- P-value: 0.5528
- Significance: Not significant

**Holiday Periods:**
- Treatment Effect (β₃): 0.0005
- P-value: 0.2524
- Significance: Not significant

**Non-Holiday Periods:**
- Treatment Effect (β₃): -0.0003
- P-value: 0.3464
- Significance: Not significant

### CBD Exit Speed

**All Data:**
- Treatment Effect (β₃): 0.0838
- P-value: 0.5178
- Significance: Not significant

**Holiday Periods:**
- Treatment Effect (β₃): 0.0669
- P-value: 0.7374
- Significance: Not significant

**Non-Holiday Periods:**
- Treatment Effect (β₃): 0.0702
- P-value: 0.6333
- Significance: Not significant

## Model Diagnostics

### Regression Statistics

**Sample Model (avg_speed_all):**
- R-squared: 0.5445
- Adjusted R-squared: 0.5442
- F-statistic: 1709.3223
- F-statistic p-value: 0.0000

## Conclusions

The DiD analysis reveals the following key findings:

1. **Policy Impact**: The treatment effect (β₃ coefficient) measures the differential impact of the policy on the treatment group relative to the control group.

2. **Holiday vs Non-Holiday Effects**: Separate analyses for holiday and non-holiday periods allow for understanding how policy effects vary across different time contexts.

3. **Statistical Significance**: Results are evaluated at 1%, 5%, and 10% significance levels.

4. **Robustness**: The analysis includes multiple control variables to account for confounding factors such as time of day, weather conditions, and day of week effects.
