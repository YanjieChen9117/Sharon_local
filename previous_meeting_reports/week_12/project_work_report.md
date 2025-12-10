# Project Work Report: Analyzing the Impact of NYC Congestion Pricing Policy on Taxi Operations

**Project Period:** September 2025 - December 2025  
**Policy Implementation Date:** January 5, 2025  
**Report Date:** December 8, 2025  
**Author:** Yanjie Chen

---

## Table of Contents

1. [The Science Problem](#1-the-science-problem)
2. [Our Method](#2-our-method)
3. [The Results](#3-the-results)
4. [Conclusions and Future Work](#4-conclusions-and-future-work)

---

## 1. The Science Problem

### 1.1 Research Context

On January 5, 2025, New York City implemented a congestion pricing policy for the Central Business District (CBD). This policy aims to reduce traffic congestion, improve air quality, and generate revenue for public transportation improvements. Understanding the causal impact of such policies is crucial for evidence-based urban transportation policy-making.

### 1.2 Research Question

**Primary Question:** What is the causal effect of the NYC congestion pricing policy on taxi operations in CBD areas?

**Specific Research Objectives:**

1. **Traffic Volume Impact:** How does the policy affect the number of taxi trips flowing into and out of CBD areas?

2. **Traffic Speed Impact:** Does the policy improve average travel speeds for taxis entering and exiting CBD areas?

3. **Temporal and Spatial Patterns:** Are there heterogeneous effects across different times of day, days of week, or weather conditions?

4. **Comparative Analysis:** How do CBD areas (treatment group) compare to non-CBD areas (control group) in terms of policy impact?

### 1.3 Methodological Challenge

The key challenge in causal inference is to isolate the policy effect from other confounding factors such as:
- Seasonal variations in travel demand
- Weather conditions (temperature, precipitation, snowfall)
- Day-of-week and hour-of-day patterns
- Holidays and special events
- Economic conditions and general trends

A quasi-experimental approach is needed because:
- Random assignment is impossible (policy applies to all CBD areas)
- Simple before-after comparison would confound policy effects with time trends
- Need a control group to establish counterfactual outcomes

### 1.4 Data Requirements

To answer these questions, we need:
- **High-resolution temporal data:** Hourly granularity to capture within-day patterns
- **Spatial granularity:** Neighborhood-level analysis (NTA zones)
- **Comprehensive coverage:** Data spanning both pre- and post-policy periods
- **Rich control variables:** Weather, temporal patterns, and financial indicators
- **Large sample size:** Sufficient statistical power to detect policy effects

---

## 2. Our Method

### 2.1 Data Sources and Collection

#### 2.1.1 NYC Taxi Trip Records

**Source:** [NYC Taxi & Limousine Commission TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

**Data Description:**
- **Types:** Yellow and Green taxi trip records
- **Format:** Monthly parquet files
- **Time Coverage:** Complete data through October 2025
- **Key Fields:** Pickup/dropoff datetime, location IDs, trip distance, fare amount, passenger count, etc.

**Data Dictionary:** Available at the TLC website, documenting all fields including the new `cbd_congestion_fee` column added in 2025.

#### 2.1.2 Weather Data

**Source:** [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api#data_sources)

**Data Description:**
- **Resolution:** Hourly data based on latitude/longitude coordinates
- **Variables:** Temperature (°F), precipitation (mm), wind speed (mph), snowfall (mm), snow depth (cm), relative humidity (%)
- **Time Coverage:** 2023-01-01 to 2025-10-15
- **Spatial Coverage:** Weather data for each NTA zone's centroid

**API Features:**
- Archive API for historical data
- Time zone: America/New_York
- Free access with rate limiting (handled by our automated workflow)

#### 2.1.3 Geographic Boundaries

**Source:** [NYC 2020 Neighborhood Tabulation Areas (NTAs)](https://data.cityofnewyork.us/City-Government/2020-Neighborhood-Tabulation-Areas-NTAs-Mapped/4hft-v355)

**Data Description:**
- **Spatial Unit:** NTA zones (neighborhood-level aggregation)
- **Format:** Shapefile with multipolygon geometries
- **Key Fields:** NTA2020 code, NTA name, borough, geometry
- **Coverage:** Complete NYC geographic coverage

**Zone Classification:**
- **CBD Zones (Treatment Group):** 16 NTA zones in Manhattan's Central Business District
  - MN0101, MN0102, MN0301, MN0201, MN0302, MN0601, MN0203, MN0202, MN0303, MN0401, MN0501, MN0602, MN0603, MN0402, MN0502, MN0604
- **Control Group:** All other NTA zones (19 additional zones)

### 2.2 Data Processing Pipeline

#### 2.2.1 Automated Weather Data Collection (`complete_weather_data_workflow.py`)

**Purpose:** Automate the retrieval of weather data for all NTA zones.

**Key Features:**
1. **Centroid Calculation:**
   - Parse multipolygon geometries from NTA shapefile
   - Calculate geographic centroid (mean latitude/longitude) for each NTA zone
   - Store coordinates for API queries

2. **API Management:**
   - Automated retry mechanism for failed requests (max 3 retries)
   - Rate limiting compliance (10-second delay between requests)
   - Handle 429 errors (API throttling) with 65-second wait times

3. **Data Quality:**
   - Validate API responses
   - Handle missing data (fill with None/0 as appropriate)
   - Merge with existing data to avoid duplication

4. **Output:**
   - Hourly weather data for 35 NTA zones
   - File: `data/nta_weather_data.csv`
   - Variables: datetime, nta_code, temperature, precipitation, windspeed, snowfall, snow_depth

**Technical Implementation:**
```python
# Example workflow steps:
1. Load NTA CSV file
2. Calculate centroids for specified NTA_LIST
3. For each NTA zone:
   - Query Open-Meteo API with centroid coordinates
   - Retrieve hourly data (2023-01-01 to 2025-10-15)
   - Handle rate limiting and errors
4. Merge with existing data (if any)
5. Save to CSV with proper data types
```

#### 2.2.2 Data Cleaning and Feature Engineering (`process_nta_hourly_taxi_data_zone.py`)

**Purpose:** Transform raw taxi trip records into analysis-ready hourly NTA-level aggregations.

**Stage 1: Data Loading and Normalization**
- Read monthly Yellow and Green taxi parquet files
- Normalize field names (e.g., `tpep_pickup_datetime` → `pickup_datetime`)
- Merge Yellow and Green data for each month

**Stage 2: Data Cleaning**
- **Date Validation:** Ensure pickup dates match the file month
- **Outlier Removal:**
  - Trip distance: 0 < distance < 100 miles
  - Fare amount: 0 < fare < $500
  - Passenger count: 0 < passengers ≤ 6
  - Trip duration: 0 < duration < 180 minutes
  - Average speed: 0 < speed < 100 mph
- **Retention Rate:** Typically 95-98% of records pass cleaning

**Stage 3: Feature Engineering**

*Temporal Features:*
- `hour_index`: Pickup hour (floor to hour)
- `dropoff_hour`: Dropoff hour (floor to hour)
- `year`, `month`, `day`: Calendar components
- `day_of_week`: 0=Monday, 6=Sunday
- `hour_of_day`: Hour of day (0-23)

*Policy Indicators:*
- `policy_status`: "pre-policy" (before 2025-01-05) or "post-policy" (after)
- `weekend_status`: "weekday" or "weekend"
- `holiday`: Boolean combining weekends and official holidays

*Trip Characteristics:*
- `trip_duration_min`: Trip duration in minutes
- `avg_speed_mph`: Average speed (miles/hour)

*Spatial Mapping:*
- Map pickup location (`PULocationID`) to `PUNTA` (pickup NTA)
- Map dropoff location (`DOLocationID`) to `DONTA` (dropoff NTA)
- Based on predefined taxi zone → NTA mapping

**Stage 4: Hourly NTA-Level Aggregation**

*Key Innovation:* Distinguish between **outflow** (departures from NTA) and **inflow** (arrivals to NTA).

*Outflow Features (based on pickup in NTA):*
- `outflow_trips`: Count of trips departing from NTA
- `outflow_total_distance`: Sum of trip distances
- `outflow_total_passengers`: Sum of passengers
- `outflow_total_duration`: Sum of trip durations
- `outflow_avg_speed`: Average speed (distance/duration)
- `outflow_total_fare`, `outflow_total_tip`, `outflow_total_tolls`: Financial metrics

*Inflow Features (based on dropoff in NTA):*
- `inflow_trips`: Count of trips arriving to NTA
- `inflow_total_distance`: Sum of trip distances
- `inflow_total_passengers`: Sum of passengers
- `inflow_total_duration`: Sum of trip durations
- `inflow_avg_speed`: Average speed (distance/duration)
- `inflow_total_fare`, `inflow_total_tip`, `inflow_total_tolls`: Financial metrics

*Complete Time-Space Grid:*
- Create all combinations of (hour × NTA zone)
- Fill missing combinations with zeros (no trips)
- Ensures balanced panel structure

**Stage 5: Weather Data Integration**
- Merge weather data by `(hour_index, NTA_zone)`
- Handle missing weather data:
  - Forward fill within each NTA zone
  - Use hourly average across all NTAs if still missing
  - Fill precipitation/snow with 0 if missing
- Diagnostic output for data quality assessment

**Output:**
- File: `data/nta_zone_hourly_taxi_summary.csv`
- **Structure:** Each row = one hour × one NTA zone
- **Variables:** Time features, spatial features, outflow metrics, inflow metrics, weather data
- **Size:** ~412,812 observations for the analysis period

### 2.3 Statistical Analysis

#### 2.3.1 Difference-in-Differences (DiD) Framework

**Conceptual Foundation:**

The DiD estimator compares the change in outcomes for the treatment group (CBD areas) before and after the policy to the change in outcomes for the control group (non-CBD areas) over the same period.

**DiD Logic:**
```
Policy Effect = (Treatment_Post - Treatment_Pre) - (Control_Post - Control_Pre)
```

This "difference of differences" isolates the policy effect from:
- Time trends affecting all areas
- Permanent differences between CBD and non-CBD areas

**Key Assumption:** *Parallel Trends*  
In the absence of the policy, treatment and control groups would have followed similar trends.

#### 2.3.2 Regression Specification

**Main DiD Model (`cbd_did_analysis.R`):**

```
Y_it = β₀ + β₁·Treatment_i + β₂·Post_t + β₃·(Treatment_i × Post_t) + β₄·X_it + ε_it
```

**Where:**
- `Y_it`: Outcome variable for NTA zone i at hour t
  - `outflow_trips`: Number of trips departing from zone
  - `inflow_trips`: Number of trips arriving to zone
  - `outflow_avg_speed`: Average speed of outbound trips
  - `inflow_avg_speed`: Average speed of inbound trips
  
- `Treatment_i`: Binary indicator (1 = CBD zone, 0 = non-CBD)
  
- `Post_t`: Binary indicator (1 = post-policy period, 0 = pre-policy period)
  - Pre-policy: 2024-01-06 to 2024-08-31
  - Post-policy: 2025-01-06 to 2025-08-31
  
- `Treatment_i × Post_t`: **Interaction term** (β₃ is the DiD estimate of policy effect)
  
- `X_it`: Control variables
  - **Temporal:** `factor(day_of_week)`, `factor(hour_of_day)`, `holiday`
  - **Weather:** `weather_temperature`, `weather_precipitation`, `weather_windspeed`, `weather_snow`
  - **Financial (outcome-specific):**
    - For outflow: `outflow_total_tip`, `outflow_total_tolls`, `outflow_total_fare`
    - For inflow: `inflow_total_tip`, `inflow_total_tolls`, `inflow_total_fare`

**Statistical Implementation:**
- **Software:** R with `fixest` package
- **Estimation:** OLS with heteroskedasticity-robust standard errors (HC3)
- **Collinearity Check:** Automatic detection and handling of collinear variables
- **Inference:** Two-sided t-tests with significance levels: * p<0.1, ** p<0.05, *** p<0.01

#### 2.3.3 Count Model Extensions (`compare_did_models.R`)

**Motivation:** Trip counts are non-negative integers, potentially violating OLS assumptions.

**Alternative Models:**

1. **Poisson Regression:**
   - Assumes mean = variance
   - Log-link function: log(E[Y]) = Xβ
   - Marginal effect: (exp(β) - 1) × 100%

2. **Negative Binomial Regression:**
   - Allows overdispersion (variance > mean)
   - Additional dispersion parameter θ
   - More flexible than Poisson

**Model Comparison:**
- **Overdispersion Test:** Pearson χ² / df
  - Ratio < 1.5: Poisson appropriate
  - Ratio > 1.5: Use Negative Binomial
- **Model Selection:** AIC, BIC, dispersion statistics
- **Robust Standard Errors:** HC3 for all models

**Key Findings from Model Comparison:**
- Severe overdispersion detected (ratio = 35.8 to 50.8)
- Negative Binomial model strongly preferred for trip counts
- OLS and NB results differ in sign for some outcomes
- Recommendation: Use Negative Binomial for trips, OLS for speeds

### 2.4 Data Summary Statistics

**Final Analysis Dataset:**
- **Total observations:** 412,812
- **Pre-policy period:** 206,856 observations (2024-01-06 to 2024-08-31)
- **Post-policy period:** 205,956 observations (2025-01-06 to 2025-08-31)
- **Treatment group (CBD):** 183,472 observations (16 NTA zones)
- **Control group (Others):** 229,340 observations (19 NTA zones)
- **Temporal coverage:** 8 months × 2 periods = comparable seasonal patterns
- **Spatial coverage:** 35 NTA zones in Manhattan

**Descriptive Statistics by Group:**

| Group | Period | Avg Outflow Trips | Avg Inflow Trips | Avg Outflow Speed (mph) | Avg Inflow Speed (mph) |
|-------|--------|-------------------|------------------|-------------------------|------------------------|
| Control | Pre | 58.74 | 64.37 | 9.37 | 12.93 |
| Control | Post | 58.21 | 64.30 | 9.26 | 12.71 |
| Treatment (CBD) | Pre | 137.91 | 130.45 | 12.36 | 12.88 |
| Treatment (CBD) | Post | 138.66 | 130.79 | 12.35 | 12.84 |

**Observations:**
- CBD areas have ~2.3× more trips than non-CBD areas
- CBD outflow speeds are higher, but inflow speeds similar to control
- Both groups show slight decreases in most metrics (highlighting need for DiD)

---

## 3. The Results

### 3.1 Main DiD Results (OLS Models)

#### 3.1.1 Trip Volume Effects

**Outflow Trips:**
- **Treatment Effect (β₃):** -0.4089 trips/hour
- **Standard Error:** 0.1140
- **P-value:** 0.0003 ***
- **95% Confidence Interval:** [-0.6324, -0.1855]
- **Interpretation:** The policy **significantly reduced** outflow trips from CBD areas by approximately 0.41 trips per hour relative to control areas.
- **Model Fit:** Adjusted R² = 0.9877, RMSE = 17.5

**Inflow Trips:**
- **Treatment Effect (β₃):** -1.8402 trips/hour
- **Standard Error:** 0.1319
- **P-value:** < 0.001 ***
- **95% Confidence Interval:** [-2.0988, -1.5817]
- **Interpretation:** The policy **significantly reduced** inflow trips to CBD areas by approximately 1.84 trips per hour relative to control areas.
- **Magnitude:** The inflow effect is **4.5× larger** than the outflow effect
- **Model Fit:** Adjusted R² = 0.9796, RMSE = 20.1

**Key Insight:** The policy had asymmetric effects on trip volumes, with much stronger impacts on trips entering CBD areas compared to trips leaving CBD areas.

#### 3.1.2 Average Speed Effects

**Outflow Average Speed:**
- **Treatment Effect (β₃):** +0.1043 mph
- **Standard Error:** 0.0368
- **P-value:** 0.005 ***
- **95% Confidence Interval:** [0.0323, 0.1764]
- **Interpretation:** The policy **significantly increased** outflow average speed by 0.10 mph in CBD areas.
- **Percentage Impact:** ~0.8% increase (0.1043 / 12.36 baseline)
- **Model Fit:** Adjusted R² = 0.1426, RMSE = 6.19

**Inflow Average Speed:**
- **Treatment Effect (β₃):** +0.2446 mph
- **Standard Error:** 0.0380
- **P-value:** < 0.001 ***
- **95% Confidence Interval:** [0.1701, 0.3192]
- **Interpretation:** The policy **significantly increased** inflow average speed by 0.24 mph in CBD areas.
- **Percentage Impact:** ~1.9% increase (0.2446 / 12.88 baseline)
- **Magnitude:** The inflow effect is **2.3× larger** than the outflow effect
- **Model Fit:** Adjusted R² = 0.1208, RMSE = 6.53

**Key Insight:** The policy improved traffic speeds for both inbound and outbound trips, with stronger improvements for inbound traffic.

### 3.2 Summary of DiD Results

| Outcome Variable | Treatment Effect | Std Error | P-value | 95% CI | Significance |
|------------------|------------------|-----------|---------|--------|--------------|
| **Outflow Trips** | **-0.4089** | 0.1140 | 0.0003 | [-0.63, -0.19] | *** |
| **Inflow Trips** | **-1.8402** | 0.1319 | <0.001 | [-2.10, -1.58] | *** |
| **Outflow Speed** | **+0.1043** | 0.0368 | 0.005 | [0.03, 0.18] | *** |
| **Inflow Speed** | **+0.2446** | 0.0380 | <0.001 | [0.17, 0.32] | *** |

**Significance:** All four outcomes show highly significant policy effects (p < 0.01)

### 3.3 Count Model Results (Negative Binomial)

**Motivation:** Trip counts are discrete, non-negative, and show severe overdispersion.

**Overdispersion Statistics:**
- Outflow trips: Dispersion ratio = 50.83 (severe overdispersion)
- Inflow trips: Dispersion ratio = 35.84 (severe overdispersion)
- **Conclusion:** Negative Binomial model strongly preferred

**Negative Binomial DiD Results:**

**Outflow Trips:**
- **Coefficient (β₃):** +0.0208
- **Standard Error:** 0.0083
- **P-value:** 0.012 **
- **Percentage Effect:** +2.10% [(exp(0.0208) - 1) × 100]
- **Interpretation:** The policy increased outflow trips by 2.1% in CBD areas

**Inflow Trips:**
- **Coefficient (β₃):** +0.0283
- **Standard Error:** 0.0051
- **P-value:** < 0.001 ***
- **Percentage Effect:** +2.87% [(exp(0.0283) - 1) × 100]
- **Interpretation:** The policy increased inflow trips by 2.9% in CBD areas

**Model Comparison:**

| Model | Outflow Trips Effect | Inflow Trips Effect | Conclusion |
|-------|---------------------|---------------------|------------|
| **OLS** | -0.41 trips/hour | -1.84 trips/hour | Not appropriate for counts |
| **Poisson** | +2.10% (significant) | -0.94% (not significant) | Fails under overdispersion |
| **Negative Binomial** | **+2.10%** (**significant**) | **+2.87%** (**significant**) | **Recommended** ✓ |

**Critical Observation:** OLS suggests negative effects, while Negative Binomial shows positive effects. This reversal demonstrates the importance of using appropriate models for count data.

### 3.4 Interpretation and Implications

#### 3.4.1 Speed Improvements (Consistent Across Models)

**Finding:** The policy significantly increased average speeds for both inbound and outbound taxi trips in CBD areas.

**Magnitude:**
- Inflow speed: +0.24 mph (+1.9%)
- Outflow speed: +0.10 mph (+0.8%)

**Implications:**
1. **Reduced Congestion:** Higher speeds suggest less traffic congestion in CBD areas
2. **Improved Traffic Flow:** Policy successfully improved traffic efficiency
3. **Asymmetric Impact:** Stronger effects on inbound traffic (entering CBD)
4. **Modest Scale:** Effects are statistically significant but operationally modest (~1-2% improvements)

#### 3.4.2 Trip Volume Changes (Model-Dependent)

**OLS Findings:** Significant decreases in trip volumes (-0.41 and -1.84 trips/hour)

**Negative Binomial Findings:** Significant increases in trip volumes (+2.1% and +2.9%)

**Reconciliation:**
1. **Scale Matters:** OLS measures absolute changes; NB measures percentage changes
2. **Baseline Differences:** CBD areas have much higher baseline trip volumes (138 vs 58 trips/hour)
3. **Statistical Appropriateness:** Count models are theoretically correct for discrete outcomes
4. **Overdispersion:** Severe overdispersion (ratio >35) invalidates OLS assumptions

**Preferred Interpretation (Based on Negative Binomial):**
- Policy increased outflow trips by **2.1%** (p=0.012)
- Policy increased inflow trips by **2.9%** (p<0.001)
- Suggests policy did not deter taxi demand; possible substitution from other modes

#### 3.4.3 Comprehensive Policy Effects

**Synthesis:**
1. **Traffic Efficiency:** Speed improvements indicate reduced congestion
2. **Demand Response:** Modest increases in taxi trips (2-3%)
3. **Spatial Asymmetry:** Stronger effects on inbound traffic
4. **Economic Behavior:** Positive trip effects suggest taxis may have competitive advantage over personal vehicles under congestion pricing

**Potential Mechanisms:**
1. **Reduced Congestion:** Fewer private vehicles → faster speeds
2. **Mode Substitution:** Drivers shift from personal cars to taxis
3. **Price Elasticity:** Taxi riders less sensitive to congestion charges
4. **Supply Response:** Taxi drivers may be attracted to CBD due to higher demand/prices

### 3.5 Robustness and Validity

**Strengths:**
1. **Large Sample:** 412,812 observations provide high statistical power
2. **High R²:** Models explain 98% of variance in trip counts
3. **Rich Controls:** Extensive temporal, weather, and financial controls
4. **Balanced Design:** Comparable pre- and post-policy periods (8 months each)
5. **Multiple Models:** Results robust across model specifications (for speeds)
6. **Appropriate Methods:** Count models for count outcomes, OLS for continuous outcomes

**Limitations:**
1. **Parallel Trends:** Assumption not formally tested (requires pre-policy trend analysis)
2. **Control Group:** Non-CBD areas may also be affected by policy (spillover effects)
3. **Short Post-Period:** Only 8 months of post-policy data (long-term effects unknown)
4. **Model Sensitivity:** Trip volume results depend on model choice (OLS vs NB)
5. **Confounders:** Unobserved factors (special events, economic shocks) not controlled
6. **Spatial Aggregation:** NTA-level analysis may mask within-zone heterogeneity

**Data Quality:**
1. **Weather Coverage:** 100% match for temperature and wind data
2. **Missing Data:** Minimal after forward-filling (< 1%)
3. **Outlier Removal:** 2-5% of trips removed in cleaning
4. **Temporal Consistency:** Date validation ensures data integrity

---

## 4. Conclusions and Future Work

### 4.1 Key Findings Summary

**Research Question:** What is the causal effect of NYC congestion pricing on taxi operations?

**Answer:**

1. **Speed Improvements (High Confidence):**
   - Policy increased inflow speeds by **1.9%** (0.24 mph)
   - Policy increased outflow speeds by **0.8%** (0.10 mph)
   - Effects are statistically significant and consistent across models
   - Suggests successful congestion reduction

2. **Trip Volume Changes (Model-Dependent):**
   - **OLS Model:** Decreases of 0.41 (outflow) and 1.84 (inflow) trips/hour
   - **Negative Binomial Model (Preferred):** Increases of 2.1% (outflow) and 2.9% (inflow)
   - Count model results theoretically more appropriate
   - Suggests policy did not deter taxi demand

3. **Asymmetric Effects:**
   - Inbound traffic (entering CBD) more affected than outbound
   - Inflow effects 2-4× larger than outflow effects
   - May reflect commuting patterns and demand asymmetries

### 4.2 Policy Implications

**For NYC Transportation Policy:**
1. **Congestion Reduction:** Policy achieved its primary goal (faster speeds)
2. **Taxi Industry:** No evidence of demand decline; potential beneficiary of policy
3. **Mode Substitution:** Possible shift from private vehicles to taxis
4. **Revenue Implications:** Increased taxi trips may offset revenue from fewer private vehicles

**For Other Cities:**
1. **Replicability:** Congestion pricing can improve traffic speeds
2. **Unintended Benefits:** Taxi industry may benefit rather than suffer
3. **Data Infrastructure:** Importance of comprehensive, high-resolution data
4. **Analytical Framework:** DiD with count models provides robust causal estimates

### 4.3 Methodological Contributions

**Data Engineering:**
1. **Automated Weather API Integration:** Reproducible workflow for multi-location weather data
2. **Comprehensive Feature Engineering:** Bidirectional flow metrics (inflow/outflow)
3. **Scalable Pipeline:** Handles millions of trip records efficiently
4. **Data Quality Controls:** Multi-stage validation and cleaning

**Statistical Analysis:**
1. **Appropriate Model Selection:** Count models for count data
2. **Overdispersion Testing:** Demonstrates importance of model diagnostics
3. **Rich Control Strategy:** Temporal, weather, and financial controls
4. **Robust Inference:** Heteroskedasticity-robust standard errors

**Reproducibility:**
1. **Documented Code:** Commented Python and R scripts
2. **Modular Design:** Separate scripts for data processing and analysis
3. **Version Control:** Git repository with clear commit history
4. **Open Data:** Public data sources enable replication

### 4.4 Future Research Directions

#### 4.4.1 Validity Enhancements

**Parallel Trends Testing:**
- Plot pre-policy trends for treatment vs control groups
- Formal statistical tests (e.g., interaction terms for pre-policy months)
- Placebo tests with fake policy dates

**Alternative Specifications:**
- Different control group definitions (e.g., CBD-adjacent zones only)
- Varying time windows (exclude immediate post-policy weeks)
- Subgroup analysis (high vs low baseline congestion areas)

**Robustness Checks:**
- Exclude extreme weather days
- Test for composition effects (passenger types, trip purposes)
- Sensitivity to outlier definition

#### 4.4.2 Mechanism Analysis

**Why Did Speeds Increase?**
- Analyze total traffic volume (all vehicle types, not just taxis)
- Examine private vehicle counts using other data sources
- Test for time-of-day heterogeneity (peak vs off-peak)

**Why Did Taxi Trips Increase?**
- Investigate other transportation modes (subway, rideshare)
- Analyze fare changes and price elasticity
- Examine taxi supply responses (driver behavior)

**Spatial Spillovers:**
- Effects on zones bordering CBD
- Traffic displacement to non-CBD areas
- Changes in trip routing patterns

#### 4.4.3 Model Extensions

**Dynamic Effects:**
- Event study design (monthly treatment effects)
- Test for anticipation effects (pre-policy behavior changes)
- Long-term equilibrium vs short-term adjustment

**Heterogeneous Treatment Effects:**
- By time of day (morning rush, evening rush, overnight)
- By day of week (weekdays vs weekends)
- By weather conditions (rain, snow, extreme temperatures)
- By zone characteristics (population density, employment)

**Advanced Count Models:**
- Zero-inflated models (if excess zeros present)
- Fixed effects Poisson/Negative Binomial
- Generalized additive models for non-linear time trends

#### 4.4.4 Data Expansion

**Extended Time Period:**
- Add more post-policy months as data becomes available
- Examine if effects persist, grow, or diminish over time
- Test for seasonal variations in policy effects

**Additional Variables:**
- Economic indicators (employment, retail sales)
- Rideshare data (Uber, Lyft) for mode substitution analysis
- Public transit ridership (subway, bus)
- Air quality measurements (PM2.5, NO2)

**Spatial Detail:**
- Finer geographic units (taxi zones instead of NTA zones)
- Network analysis (specific routes, corridors)
- Spatial econometrics (account for spatial autocorrelation)

#### 4.4.5 Policy Extensions

**Comparative Analysis:**
- Compare NYC results to other cities with congestion pricing (London, Singapore, Stockholm)
- Meta-analysis of congestion pricing effectiveness

**Cost-Benefit Analysis:**
- Monetize speed improvements (value of time savings)
- Environmental benefits (emissions reductions)
- Health impacts (air quality, physical activity)
- Revenue generation and use

**Equity Analysis:**
- Distributional effects across income groups
- Access to jobs and services
- Burden on taxi drivers vs passengers

### 4.5 Technical Documentation

**Data Files:**
- `data/nta_weather_data.csv`: Weather data for all NTA zones
- `data/nta_zone_hourly_taxi_summary.csv`: Aggregated analysis dataset
- Monthly parquet files: `yellow_tripdata_YYYY-MM.parquet`, `green_tripdata_YYYY-MM.parquet`

**Code Files:**
- `complete_weather_data_workflow.py`: Weather API automation
- `process_nta_hourly_taxi_data_zone.py`: Data processing pipeline
- `cbd_did_analysis.R`: Main DiD analysis (OLS)
- `compare_did_models.R`: Count model comparison
- `cbd_did_analysis_count_models.R`: Detailed count model analysis

**Reports:**
- `reports/cbd_did_analysis_report_en.md`: Detailed English report
- `reports/cbd_did_analysis_report_zh.md`: Detailed Chinese report
- `reports/count_models_results_summary_en.md`: Count model results
- `reports/count_models_results_summary_zh.md`: Count model results (Chinese)

**Software Requirements:**
- Python 3.8+: pandas, numpy, requests
- R 4.0+: dplyr, lubridate, fixest, MASS, lmtest, sandwich

### 4.6 Acknowledgments

**Data Providers:**
- NYC Taxi & Limousine Commission for taxi trip records
- Open-Meteo for historical weather data API
- NYC Open Data for geographic boundary files

**Methodological Guidance:**
- Professor Sharon Di for suggesting count model approaches
- Research team for feedback on analysis design

---

## References

### Data Sources

1. NYC Taxi & Limousine Commission. (2025). *TLC Trip Record Data*. Retrieved from https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

2. Open-Meteo. (2025). *Historical Weather API*. Retrieved from https://open-meteo.com/en/docs/historical-weather-api

3. NYC Department of City Planning. (2020). *2020 Neighborhood Tabulation Areas (NTAs)*. Retrieved from https://data.cityofnewyork.us/City-Government/2020-Neighborhood-Tabulation-Areas-NTAs-Mapped/4hft-v355

### Methodological References

4. Angrist, J. D., & Pischke, J. S. (2009). *Mostly harmless econometrics: An empiricist's companion*. Princeton University Press.

5. Cameron, A. C., & Trivedi, P. K. (2013). *Regression analysis of count data* (2nd ed.). Cambridge University Press.

6. Hilbe, J. M. (2011). *Negative binomial regression* (2nd ed.). Cambridge University Press.

7. Wooldridge, J. M. (2010). *Econometric analysis of cross section and panel data* (2nd ed.). MIT Press.

8. Bertrand, M., Duflo, E., & Mullainathan, S. (2004). How much should we trust differences-in-differences estimates? *The Quarterly Journal of Economics*, 119(1), 249-275.

### Policy Context

9. NYC Mayor's Office. (2025). *Congestion Pricing in New York City*. 

10. Duranton, G., & Turner, M. A. (2011). The fundamental law of road congestion: Evidence from US cities. *American Economic Review*, 101(6), 2616-2652.

---

**Report End**

*For questions or additional analysis requests, please contact:*  
**Yanjie Chen**  
Email: yc4594@columbia.edu
GitHub: https://github.com/YanjieChen9117/NYC_taxi_trip_analysis

*Last Updated: December 8, 2025*

