#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-
# ============================================================================
# CBD区域DiD分析 - 计数模型 (Poisson & Negative Binomial)
# ============================================================================
#
# 功能：
# 1. 使用Poisson和Negative Binomial回归分析计数数据（trips）
# 2. 对比OLS, Poisson, Negative Binomial三种模型的结果
# 3. 分析政策对以下指标的影响：
#    - outflow_trips (流出行程数) - 计数数据，适合Poisson/NB
#    - inflow_trips (流入行程数) - 计数数据，适合Poisson/NB
#    - outflow_avg_speed (流出平均速度) - 连续数据，仅用OLS
#    - inflow_avg_speed (流入平均速度) - 连续数据，仅用OLS
#
# 模型说明：
# 1. Poisson回归: 假设均值=方差，适合没有过度离散的计数数据
# 2. Negative Binomial回归: 允许过度离散（方差>均值），更灵活
# 3. 使用过度离散检验来选择最优模型
#
# 作者: Yanjie Chen
# 日期: 2025-12-08
# ============================================================================

# 加载必要的库
required_packages <- c("dplyr", "lubridate", "fixest", "MASS", "lmtest", "sandwich")
missing_packages <- required_packages[!(required_packages %in% installed.packages()[, "Package"])]

if (length(missing_packages) > 0) {
  cat("正在安装缺失的R包:", paste(missing_packages, collapse = ", "), "\n")
  install.packages(missing_packages, repos = "https://cran.rstudio.com/")
}

library(dplyr)
library(lubridate)
library(fixest)      # 用于OLS with fixed effects
library(MASS)        # 用于Negative Binomial
library(lmtest)      # 用于模型检验
library(sandwich)    # 用于稳健标准误

# ============================================================================
# 1. 数据加载和准备
# ============================================================================

cat("=== 开始分析 ===\n")
cat("开始时间:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")

# 读取数据
data_path <- "/Users/yanjiechen/Documents/Github/Sharon_local/data/nta_zone_hourly_taxi_summary.csv"
df <- read.csv(data_path, stringsAsFactors = FALSE)

# 确保year, month, day是数值型
df$year <- as.numeric(df$year)
df$month <- as.numeric(df$month)
df$day <- as.numeric(df$day)

# 创建日期列
df$date <- as.Date(paste(df$year, df$month, df$day, sep = "-"))

# 定义CBD区域
cbd_zones <- c(
  "MN0101", "MN0102", "MN0301", "MN0201", "MN0302", "MN0601",
  "MN0203", "MN0202", "MN0303", "MN0401", "MN0501",
  "MN0602", "MN0603", "MN0402", "MN0502", "MN0604"
)

# 创建区域类型变量
df$is_cbd <- ifelse(df$NTA_zone %in% cbd_zones, 1, 0)

# 定义时间范围
pre_start <- as.Date("2024-01-06")
pre_end <- as.Date("2024-08-31")
post_start <- as.Date("2025-01-06")
post_end <- as.Date("2025-08-31")

# 筛选并合并数据
pre_data <- df %>%
  filter(date >= pre_start & date <= pre_end) %>%
  mutate(post = 0)

post_data <- df %>%
  filter(date >= post_start & date <= post_end) %>%
  mutate(post = 1)

did_data <- bind_rows(pre_data, post_data)

# 创建treatment和交互项
did_data$treatment <- did_data$is_cbd
did_data$treatment_post <- did_data$treatment * did_data$post

# 转换控制变量类型
if (is.logical(did_data$holiday)) {
  did_data$holiday <- as.numeric(did_data$holiday)
} else if (is.character(did_data$holiday)) {
  did_data$holiday <- as.numeric(did_data$holiday == "True" | did_data$holiday == "TRUE")
}

did_data$day_of_week <- as.numeric(did_data$day_of_week)
did_data$hour_of_day <- as.numeric(did_data$hour_of_day)
did_data$weather_temperature <- as.numeric(did_data$weather_temperature)
did_data$weather_precipitation <- as.numeric(did_data$weather_precipitation)
did_data$weather_windspeed <- as.numeric(did_data$weather_windspeed)
did_data$weather_snow <- as.numeric(did_data$weather_snow)

# 转换财务变量
if ("outflow_total_tip" %in% names(did_data)) {
  did_data$outflow_total_tip <- as.numeric(did_data$outflow_total_tip)
  did_data$outflow_total_tolls <- as.numeric(did_data$outflow_total_tolls)
  did_data$outflow_total_fare <- as.numeric(did_data$outflow_total_fare)
}
if ("inflow_total_tip" %in% names(did_data)) {
  did_data$inflow_total_tip <- as.numeric(did_data$inflow_total_tip)
  did_data$inflow_total_tolls <- as.numeric(did_data$inflow_total_tolls)
  did_data$inflow_total_fare <- as.numeric(did_data$inflow_total_fare)
}

# 打印数据摘要
cat("=== 数据摘要 ===\n")
cat(sprintf("Pre-policy数据: %d 条记录\n", nrow(pre_data)))
cat(sprintf("Post-policy数据: %d 条记录\n", nrow(post_data)))
cat(sprintf("Treatment group (CBD): %d 条记录\n", sum(did_data$treatment == 1)))
cat(sprintf("Control group (Others): %d 条记录\n", sum(did_data$treatment == 0)))
cat("\n")

# ============================================================================
# 2. 辅助函数
# ============================================================================

# 准备分析数据
prepare_analysis_data <- function(data, outcome_var) {
  # 确定财务控制变量
  if (grepl("^outflow", outcome_var)) {
    candidate_financial_vars <- c("outflow_total_tip", "outflow_total_tolls", "outflow_total_fare")
  } else if (grepl("^inflow", outcome_var)) {
    candidate_financial_vars <- c("inflow_total_tip", "inflow_total_tolls", "inflow_total_fare")
  } else {
    candidate_financial_vars <- character(0)
  }
  
  # 检查哪些财务变量存在
  financial_vars <- character(0)
  if (length(candidate_financial_vars) > 0) {
    for (var in candidate_financial_vars) {
      if (var %in% names(data)) {
        financial_vars <- c(financial_vars, var)
      }
    }
  }
  
  # 筛选缺失值
  analysis_data <- data %>%
    filter(
      !is.na(.data[[outcome_var]]),
      !is.na(treatment),
      !is.na(post),
      !is.na(day_of_week),
      !is.na(hour_of_day),
      !is.na(holiday),
      !is.na(weather_temperature),
      !is.na(weather_precipitation),
      !is.na(weather_windspeed),
      !is.na(weather_snow)
    )
  
  # 筛选财务变量缺失值
  for (var in financial_vars) {
    if (var %in% names(analysis_data)) {
      analysis_data <- analysis_data %>% filter(!is.na(.data[[var]]))
    }
  }
  
  # 对于计数数据，确保非负且为整数
  if (grepl("trips$", outcome_var)) {
    analysis_data <- analysis_data %>%
      filter(.data[[outcome_var]] >= 0) %>%
      mutate(!!outcome_var := round(.data[[outcome_var]]))
  }
  
  return(list(data = analysis_data, financial_vars = financial_vars))
}

# 构建模型公式
build_formula <- function(outcome_var, financial_vars) {
  # 基础控制变量
  base_controls <- "factor(day_of_week) + factor(hour_of_day) + holiday"
  
  # 天气变量
  weather_controls <- "weather_temperature + weather_precipitation + weather_windspeed + weather_snow"
  all_controls <- paste(base_controls, weather_controls, sep = " + ")
  
  # 添加财务变量
  if (length(financial_vars) > 0) {
    financial_controls <- paste(financial_vars, collapse = " + ")
    all_controls <- paste(all_controls, financial_controls, sep = " + ")
  }
  
  # 完整公式
  formula_str <- sprintf(
    "%s ~ treatment + post + treatment_post + %s",
    outcome_var, all_controls
  )
  
  return(as.formula(formula_str))
}

# 过度离散检验
test_overdispersion <- function(model) {
  # 计算Pearson卡方统计量
  pearson_residuals <- residuals(model, type = "pearson")
  pearson_chi2 <- sum(pearson_residuals^2)
  df_residual <- model$df.residual
  dispersion_ratio <- pearson_chi2 / df_residual
  
  # 计算p值
  p_value <- pchisq(pearson_chi2, df_residual, lower.tail = FALSE)
  
  return(list(
    dispersion_ratio = dispersion_ratio,
    pearson_chi2 = pearson_chi2,
    df_residual = df_residual,
    p_value = p_value,
    overdispersed = dispersion_ratio > 1.5 && p_value < 0.05
  ))
}

# ============================================================================
# 3. 计数模型分析函数
# ============================================================================

run_count_models_analysis <- function(data, outcome_var, outcome_name) {
  cat("\n")
  cat(paste0(rep("=", 80), collapse = ""), "\n")
  cat(sprintf("=== %s - 计数模型分析 ===\n", outcome_name))
  cat(paste0(rep("=", 80), collapse = ""), "\n")
  
  # 准备数据
  prepared <- prepare_analysis_data(data, outcome_var)
  analysis_data <- prepared$data
  financial_vars <- prepared$financial_vars
  
  cat(sprintf("\n有效观测数: %d\n", nrow(analysis_data)))
  
  if (nrow(analysis_data) == 0) {
    stop("错误: 数据集中没有有效观测值")
  }
  
  # 构建公式
  formula_obj <- build_formula(outcome_var, financial_vars)
  cat("\n模型公式:\n")
  print(formula_obj)
  
  # 1. OLS模型 (使用fixest)
  cat("\n--- (1) OLS模型 ---\n")
  ols_model <- feols(formula_obj, data = analysis_data, vcov = "HC3")
  
  ols_coef <- coef(ols_model)
  ols_se <- se(ols_model)
  ols_treatment_effect <- ols_coef["treatment_post"]
  ols_treatment_se <- ols_se["treatment_post"]
  ols_t_stat <- ols_treatment_effect / ols_treatment_se
  ols_pvalue <- 2 * (1 - pnorm(abs(ols_t_stat)))
  
  cat(sprintf("治疗效应 (β₃): %.4f\n", ols_treatment_effect))
  cat(sprintf("标准误: %.4f\n", ols_treatment_se))
  cat(sprintf("P值: %.6f\n", ols_pvalue))
  
  # 2. Poisson模型
  cat("\n--- (2) Poisson模型 ---\n")
  poisson_model <- glm(formula_obj, data = analysis_data, family = poisson(link = "log"))
  
  # 使用稳健标准误
  poisson_coef <- coef(poisson_model)
  poisson_vcov <- vcovHC(poisson_model, type = "HC3")
  poisson_se <- sqrt(diag(poisson_vcov))
  
  poisson_treatment_effect <- poisson_coef["treatment_post"]
  poisson_treatment_se <- poisson_se["treatment_post"]
  poisson_z_stat <- poisson_treatment_effect / poisson_treatment_se
  poisson_pvalue <- 2 * (1 - pnorm(abs(poisson_z_stat)))
  
  cat(sprintf("治疗效应 (对数尺度): %.4f\n", poisson_treatment_effect))
  cat(sprintf("标准误: %.4f\n", poisson_treatment_se))
  cat(sprintf("P值: %.6f\n", poisson_pvalue))
  
  # 计算边际效应 (exp(β) - 1 = 百分比变化)
  poisson_pct_effect <- (exp(poisson_treatment_effect) - 1) * 100
  cat(sprintf("边际效应 (百分比变化): %.2f%%\n", poisson_pct_effect))
  
  # 过度离散检验
  cat("\n过度离散检验:\n")
  overdispersion_test <- test_overdispersion(poisson_model)
  cat(sprintf("  Dispersion ratio: %.4f\n", overdispersion_test$dispersion_ratio))
  cat(sprintf("  Pearson χ²: %.2f (df = %d)\n", 
              overdispersion_test$pearson_chi2, overdispersion_test$df_residual))
  cat(sprintf("  P值: %.6f\n", overdispersion_test$p_value))
  if (overdispersion_test$overdispersed) {
    cat("  结论: 存在显著的过度离散，建议使用Negative Binomial模型\n")
  } else {
    cat("  结论: 无显著过度离散，Poisson模型适用\n")
  }
  
  # 3. Negative Binomial模型
  cat("\n--- (3) Negative Binomial模型 ---\n")
  nb_model <- tryCatch({
    glm.nb(formula_obj, data = analysis_data)
  }, error = function(e) {
    cat("警告: Negative Binomial模型拟合失败\n")
    cat("错误信息:", e$message, "\n")
    return(NULL)
  })
  
  nb_treatment_effect <- NA
  nb_treatment_se <- NA
  nb_pvalue <- NA
  nb_pct_effect <- NA
  theta <- NA
  
  if (!is.null(nb_model)) {
    # 使用稳健标准误
    nb_coef <- coef(nb_model)
    nb_vcov <- vcovHC(nb_model, type = "HC3")
    nb_se <- sqrt(diag(nb_vcov))
    
    nb_treatment_effect <- nb_coef["treatment_post"]
    nb_treatment_se <- nb_se["treatment_post"]
    nb_z_stat <- nb_treatment_effect / nb_treatment_se
    nb_pvalue <- 2 * (1 - pnorm(abs(nb_z_stat)))
    
    cat(sprintf("治疗效应 (对数尺度): %.4f\n", nb_treatment_effect))
    cat(sprintf("标准误: %.4f\n", nb_treatment_se))
    cat(sprintf("P值: %.6f\n", nb_pvalue))
    
    # 边际效应
    nb_pct_effect <- (exp(nb_treatment_effect) - 1) * 100
    cat(sprintf("边际效应 (百分比变化): %.2f%%\n", nb_pct_effect))
    
    # Dispersion参数
    theta <- nb_model$theta
    cat(sprintf("\nDispersion参数 (θ): %.4f\n", theta))
    cat("  (θ越小，过度离散越严重)\n")
  }
  
  # 4. 模型比较
  cat("\n--- (4) 模型比较 ---\n")
  
  # AIC和BIC
  ols_aic <- AIC(ols_model)
  poisson_aic <- AIC(poisson_model)
  
  cat("\n信息准则 (仅用于参考，不能直接比较OLS和计数模型):\n")
  cat(sprintf("  OLS AIC: %.2f (注: 不能与计数模型直接比较)\n", ols_aic))
  cat(sprintf("  Poisson AIC: %.2f\n", poisson_aic))
  
  if (!is.null(nb_model)) {
    nb_aic <- AIC(nb_model)
    cat(sprintf("  Negative Binomial AIC: %.2f\n", nb_aic))
    
    # 在Poisson和NB之间选择
    if (nb_aic < poisson_aic) {
      best_model_name <- "Negative Binomial"
      cat(sprintf("\n在计数模型中，基于AIC和过度离散检验，推荐: %s\n", best_model_name))
    } else {
      best_model_name <- "Poisson"
      cat(sprintf("\n在计数模型中，基于AIC，推荐: %s\n", best_model_name))
    }
  } else {
    best_model_name <- "Poisson"
    cat(sprintf("\n在计数模型中，推荐: %s (NB拟合失败)\n", best_model_name))
  }
  
  # 5. 结果汇总表
  cat("\n--- (5) 治疗效应汇总 ---\n")
  summary_df <- data.frame(
    Model = c("OLS", "Poisson", "Negative Binomial"),
    Coefficient = c(ols_treatment_effect, poisson_treatment_effect, nb_treatment_effect),
    Std_Error = c(ols_treatment_se, poisson_treatment_se, nb_treatment_se),
    P_Value = c(ols_pvalue, poisson_pvalue, nb_pvalue),
    Pct_Effect = c(NA, poisson_pct_effect, nb_pct_effect),
    stringsAsFactors = FALSE
  )
  
  # 添加显著性标记
  summary_df$Significance <- sapply(summary_df$P_Value, function(p) {
    if (is.na(p)) return("")
    ifelse(p < 0.01, "***",
           ifelse(p < 0.05, "**",
                  ifelse(p < 0.1, "*", "")))
  })
  
  print(summary_df)
  cat("\n显著性标记: *** p<0.01, ** p<0.05, * p<0.1\n")
  cat("注: OLS系数为绝对变化，Poisson/NB系数为对数尺度，Pct_Effect为百分比变化\n")
  
  # 返回结果
  return(list(
    outcome_var = outcome_var,
    outcome_name = outcome_name,
    ols_model = ols_model,
    poisson_model = poisson_model,
    nb_model = nb_model,
    summary_table = summary_df,
    overdispersion_test = overdispersion_test,
    best_model = best_model_name,
    n_obs = nrow(analysis_data)
  ))
}

# ============================================================================
# 4. 运行分析
# ============================================================================

results <- list()

# 分析outflow_trips (计数数据)
results$outflow_trips <- run_count_models_analysis(
  did_data,
  "outflow_trips",
  "流出行程数 (Outflow Trips)"
)

# 分析inflow_trips (计数数据)
results$inflow_trips <- run_count_models_analysis(
  did_data,
  "inflow_trips",
  "流入行程数 (Inflow Trips)"
)

# ============================================================================
# 5. 最终结果汇总
# ============================================================================

cat("\n\n")
cat(paste0(rep("=", 80), collapse = ""), "\n")
cat("=== 最终结果汇总 ===\n")
cat(paste0(rep("=", 80), collapse = ""), "\n")

for (result_name in names(results)) {
  result <- results[[result_name]]
  cat(sprintf("\n%s:\n", result$outcome_name))
  cat(sprintf("  观测数: %d\n", result$n_obs))
  cat(sprintf("  推荐模型: %s\n", result$best_model))
  cat(sprintf("  过度离散比率: %.4f\n", result$overdispersion_test$dispersion_ratio))
  cat("\n")
  print(result$summary_table)
}

cat("\n")
cat(paste0(rep("=", 80), collapse = ""), "\n")
cat("=== 分析完成 ===\n")
cat("结束时间:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat(paste0(rep("=", 80), collapse = ""), "\n")

