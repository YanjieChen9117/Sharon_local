# ============================================================================
# DiD Model Implementation: Policy Impact on Trip Count
# ============================================================================

import pandas as pd
import numpy as np
import statsmodels.api as sm

# Step 1: Prepare Data and Create DiD Variables
# ----------------------------------------------------------------------------

# Load and filter data
df = pd.read_csv('nta_hourly_taxi_summary.csv')
df['pickup_hour'] = pd.to_datetime(df['pickup_hour'])

# Pre-policy period: 2024-01 to 2024-07
pre_data = df[(df['year'] == 2024) & (df['month'] >= 1) & (df['month'] <= 7)].copy()
pre_data['post'] = 0

# Post-policy period: 2025-01 to 2025-07
post_data = df[(df['year'] == 2025) & (df['month'] >= 1) & (df['month'] <= 7)].copy()
post_data['post'] = 1

# Combine pre and post data
did_data = pd.concat([pre_data, post_data], ignore_index=True)

# Create Treatment variable: 1 if PUNTA == MN0502, 0 otherwise
did_data['treatment'] = (did_data['PUNTA'] == 'MN0502').astype(int)

# Create Interaction term: Treatment × Post
did_data['treatment_post'] = did_data['treatment'] * did_data['post']

# Step 2: Prepare Regression Variables
# ----------------------------------------------------------------------------

# Remove missing values
analysis_data = did_data.dropna(subset=['total_trips', 'treatment', 'post', 
                                         'treatment_post', 'hour_of_day', 
                                         'day_of_week', 'is_rain', 'is_snow', 
                                         'temperature', 'holiday'])

# DiD model variables
X_vars = ['treatment', 'post', 'treatment_post']

# Control variables
controls = ['hour_of_day', 'day_of_week', 'is_rain', 'is_snow', 
            'temperature', 'holiday']
X_vars.extend(controls)

# Prepare regression matrices
X = analysis_data[X_vars].astype(float)
X = sm.add_constant(X)  # Add intercept
y = analysis_data['total_trips'].astype(float)

# Step 3: Estimate DiD Model
# ----------------------------------------------------------------------------

# OLS regression with robust standard errors (HC3) to handle heteroscedasticity
model = sm.OLS(y, X).fit(cov_type='HC3')

# Step 4: Extract Policy Effect
# ----------------------------------------------------------------------------

# Policy effect is the coefficient on 'treatment_post'
treatment_effect = model.params['treatment_post']
treatment_se = model.bse['treatment_post']
treatment_pvalue = model.pvalues['treatment_post']
treatment_ci = model.conf_int().loc['treatment_post']

print(f"Policy Effect (β₃): {treatment_effect:.4f}")
print(f"Standard Error: {treatment_se:.4f}")
print(f"P-value: {treatment_pvalue:.4f}")
print(f"95% CI: [{treatment_ci[0]:.4f}, {treatment_ci[1]:.4f}]")

# Print full regression results
print(model.summary())