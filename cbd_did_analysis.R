#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-
# ============================================================================
# CBD区域DiD (Difference-in-Differences) 分析
# ============================================================================
#
# 功能：
# 1. 使用pre-policy (2024-01-06 到 2024-08-31) 和 post-policy (2025-01-06 到 2025-08-31) 数据
# 2. Treatment group: CBD区域
# 3. Control group: 所有其他区域
# 4. 分析政策对以下指标的影响：
#    - outflow_trips (流出行程数)
#    - inflow_trips (流入行程数)
#    - outflow_avg_speed (流出平均速度)
#    - inflow_avg_speed (流入平均速度)
#
# DiD模型：
# Y = β₀ + β₁*Treatment + β₂*Post + β₃*(Treatment*Post) + β₄*Controls + ε
#
# 其中：
# - Treatment: 1 if CBD区域, 0 if others
# - Post: 1 if post-policy period, 0 if pre-policy period
# - Treatment*Post: 交互项，捕捉政策效应
#
# 控制变量：
# - day_of_week, hour_of_day, holiday
# - weather_temperature, weather_precipitation, weather_windspeed, weather_snow
# - 对于outflow分析: outflow_total_tip, outflow_total_tolls, outflow_total_fare
# - 对于inflow分析: inflow_total_tip, inflow_total_tolls, inflow_total_fare
#
# 作者: Yanjie Chen
# 日期: 2025-12-08
# ============================================================================

# 加载必要的库
# 检查并安装缺失的包
required_packages <- c("dplyr", "lubridate", "lmtest", "sandwich")
missing_packages <- required_packages[!(required_packages %in% installed.packages()[, "Package"])]

if (length(missing_packages) > 0) {
  cat("正在安装缺失的R包:", paste(missing_packages, collapse = ", "), "\n")
  install.packages(missing_packages, repos = "https://cran.rstudio.com/")
}

library(dplyr)
library(lubridate)
library(lmtest)
library(sandwich)

# ============================================================================
# 1. 数据加载和准备
# ============================================================================

# 读取数据
data_path <- "/Users/yanjiechen/Documents/Github/Sharon_local/data/nta_zone_hourly_taxi_summary.csv"
df <- read.csv(data_path, stringsAsFactors = FALSE)

# 确保year, month, day是数值型
df$year <- as.numeric(df$year)
df$month <- as.numeric(df$month)
df$day <- as.numeric(df$day)

# 创建日期列（使用已有的year, month, day列）
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

# 筛选pre-policy数据
pre_data <- df %>%
  filter(date >= pre_start & date <= pre_end) %>%
  mutate(post = 0)

# 筛选post-policy数据
post_data <- df %>%
  filter(date >= post_start & date <= post_end) %>%
  mutate(post = 1)

# 合并数据
did_data <- bind_rows(pre_data, post_data)

# 创建treatment变量: 1 if CBD区域, 0 if others
did_data$treatment <- did_data$is_cbd

# 创建交互项
did_data$treatment_post <- did_data$treatment * did_data$post

# 确保控制变量类型正确
# holiday转换为数值型（TRUE/FALSE -> 1/0）
if (is.logical(did_data$holiday)) {
  did_data$holiday <- as.numeric(did_data$holiday)
} else if (is.character(did_data$holiday)) {
  did_data$holiday <- as.numeric(did_data$holiday == "True" | did_data$holiday == "TRUE")
}

# 确保day_of_week和hour_of_day是数值型
did_data$day_of_week <- as.numeric(did_data$day_of_week)
did_data$hour_of_day <- as.numeric(did_data$hour_of_day)

# 确保天气变量是数值型
did_data$weather_temperature <- as.numeric(did_data$weather_temperature)
did_data$weather_precipitation <- as.numeric(did_data$weather_precipitation)
did_data$weather_windspeed <- as.numeric(did_data$weather_windspeed)
did_data$weather_snow <- as.numeric(did_data$weather_snow)

# 确保财务变量是数值型（用于控制变量）
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
# 1.5. 对dependent variables进行log转换
# ============================================================================

cat("=== 对因变量进行log转换 ===\n")

# 定义需要转换的变量
dependent_vars <- c("outflow_trips", "inflow_trips", "outflow_avg_speed", "inflow_avg_speed")

for (var in dependent_vars) {
  if (var %in% names(did_data)) {
    # 检查数据范围
    var_data <- did_data[[var]]
    var_data_numeric <- as.numeric(var_data)
    var_data_clean <- var_data_numeric[!is.na(var_data_numeric)]
    
    # 统计信息
    n_total <- length(var_data_numeric)
    n_valid <- length(var_data_clean)
    n_zero <- sum(var_data_clean == 0, na.rm = TRUE)
    n_negative <- sum(var_data_clean < 0, na.rm = TRUE)
    n_positive <- sum(var_data_clean > 0, na.rm = TRUE)
    
    cat(sprintf("\n%s:\n", var))
    cat(sprintf("  总观测数: %d\n", n_total))
    cat(sprintf("  有效观测数: %d\n", n_valid))
    cat(sprintf("  零值: %d\n", n_zero))
    cat(sprintf("  负值: %d\n", n_negative))
    cat(sprintf("  正值: %d\n", n_positive))
    
    # 创建log转换后的变量名
    log_var_name <- paste0("log_", var)
    
    # 进行log转换
    # 对于trip count，如果有零值，使用log(1+x)
    # 对于speed，应该都是正值，直接使用log
    if (grepl("trips", var)) {
      # trip count: 使用 log(1+x) 处理零值
      did_data[[log_var_name]] <- log1p(var_data_numeric)
      cat(sprintf("  转换方法: log(1+x) (处理零值)\n"))
    } else {
      # speed: 直接使用log，但需要处理零值和负值
      if (n_zero > 0 || n_negative > 0) {
        # 如果有零值或负值，使用log(1+x)
        did_data[[log_var_name]] <- log1p(pmax(var_data_numeric, 0))
        cat(sprintf("  转换方法: log(1+x) (处理零值和负值)\n"))
      } else {
        # 如果都是正值，直接使用log
        did_data[[log_var_name]] <- log(var_data_numeric)
        cat(sprintf("  转换方法: log(x)\n"))
      }
    }
    
    # 检查转换后的数据
    log_data <- did_data[[log_var_name]]
    log_data_clean <- log_data[!is.na(log_data)]
    cat(sprintf("  转换后有效观测数: %d\n", length(log_data_clean)))
    cat(sprintf("  转换后范围: [%.4f, %.4f]\n", min(log_data_clean), max(log_data_clean)))
  } else {
    cat(sprintf("警告: 变量 %s 不在数据集中\n", var))
  }
}

cat("\n")


# ============================================================================
# 2. DiD分析函数
# ============================================================================

run_did_analysis <- function(data, outcome_var, outcome_name) {
  # 运行DiD分析（使用标准lm函数）
  #
  # 参数:
  #   data: 分析数据框
  #   outcome_var: 结果变量名（原始变量名，函数内部会使用log转换后的变量）
  #   outcome_name: 结果变量显示名称

  cat(sprintf("\n=== DiD分析: %s (log转换后) ===\n", outcome_name))
  
  # 使用log转换后的变量
  log_outcome_var <- paste0("log_", outcome_var)
  
  # 检查log转换后的变量是否存在
  if (!log_outcome_var %in% names(data)) {
    stop(sprintf("错误: log转换后的变量 %s 不存在。请先进行log转换。", log_outcome_var))
  }
  # 根据outcome_var确定使用哪些财务控制变量
  # 先定义候选变量，然后检查哪些在数据中存在
  if (grepl("^outflow", outcome_var)) {
    # outflow相关分析：候选outflow财务变量
    candidate_financial_vars <- c("outflow_total_tip", "outflow_total_tolls", "outflow_total_fare")
  } else if (grepl("^inflow", outcome_var)) {
    # inflow相关分析：候选inflow财务变量
    candidate_financial_vars <- c("inflow_total_tip", "inflow_total_tolls", "inflow_total_fare")
  } else {
    # 默认情况：不使用财务变量
    candidate_financial_vars <- character(0)
  }

  # 检查哪些财务变量在数据中存在
  financial_vars <- character(0)
  if (length(candidate_financial_vars) > 0) {
    for (var in candidate_financial_vars) {
      if (var %in% names(data)) {
        financial_vars <- c(financial_vars, var)
      } else {
        cat(sprintf("注意: 财务变量 %s 不在数据集中，将跳过\n", var))
      }
    }
  }

  # 构建财务控制变量公式
  if (length(financial_vars) > 0) {
    financial_controls <- paste(financial_vars, collapse = " + ")
  } else {
    financial_controls <- ""
    cat("注意: 未找到可用的财务控制变量，将不使用财务变量\n")
  }

  # 诊断缺失值情况
  cat("\n=== 缺失值诊断 ===\n")
  cat(sprintf("原始数据观测数: %d\n", nrow(data)))

  # 检查每个变量的缺失值
  check_vars <- c(
    log_outcome_var, "treatment", "post", "day_of_week", "hour_of_day",
    "holiday", "weather_temperature", "weather_precipitation",
    "weather_windspeed", "weather_snow"
  )
  for (var in check_vars) {
    if (var %in% names(data)) {
      na_count <- sum(is.na(data[[var]]))
      na_pct <- na_count / nrow(data) * 100
      cat(sprintf("  %s: 缺失 %d (%.2f%%)\n", var, na_count, na_pct))
    }
  }

  # 检查财务变量的缺失值
  for (var in financial_vars) {
    if (var %in% names(data)) {
      na_count <- sum(is.na(data[[var]]))
      na_pct <- na_count / nrow(data) * 100
      cat(sprintf("  %s: 缺失 %d (%.2f%%)\n", var, na_count, na_pct))
    }
  }

  # 移除缺失值（逐步筛选，以便诊断）
  analysis_data <- data %>%
    filter(
      !is.na(.data[[log_outcome_var]]),
      !is.na(treatment),
      !is.na(post),
      !is.na(day_of_week),
      !is.na(hour_of_day),
      !is.na(holiday)
    )

  cat(sprintf("筛选基础变量后: %d 观测\n", nrow(analysis_data)))

  # 筛选天气变量（只筛选存在的且有数据的变量）
  weather_var_list <- c("weather_temperature", "weather_precipitation", "weather_windspeed", "weather_snow")
  available_weather_vars <- character(0)
  for (var in weather_var_list) {
    if (var %in% names(analysis_data)) {
      # 检查是否有非缺失值
      if (sum(!is.na(analysis_data[[var]])) > 0) {
        available_weather_vars <- c(available_weather_vars, var)
      }
    }
  }

  # 只筛选存在的天气变量
  if (length(available_weather_vars) > 0) {
    for (var in available_weather_vars) {
      analysis_data <- analysis_data %>% filter(!is.na(.data[[var]]))
    }
    cat(sprintf(
      "筛选天气变量后: %d 观测 (使用了 %d 个天气变量: %s)\n",
      nrow(analysis_data), length(available_weather_vars),
      paste(available_weather_vars, collapse = ", ")
    ))
  } else {
    cat("警告: 没有可用的天气变量，跳过天气变量筛选\n")
  }

  # 添加财务变量的缺失值检查
  for (var in financial_vars) {
    if (var %in% names(analysis_data)) {
      na_before <- nrow(analysis_data)
      analysis_data <- analysis_data %>% filter(!is.na(.data[[var]]))
      if (nrow(analysis_data) < na_before) {
        cat(sprintf(
          "筛选%s后: %d 观测 (删除了 %d 个缺失观测)\n",
          var, nrow(analysis_data), na_before - nrow(analysis_data)
        ))
      }
    }
  }

  cat(sprintf("\n最终有效观测数: %d\n", nrow(analysis_data)))
  if (nrow(analysis_data) == 0) {
    stop("错误: 数据集中没有有效观测值。请检查数据中的缺失值情况。")
  }

  # 准备回归公式
  # 基础控制变量：day_of_week, hour_of_day, holiday
  base_controls <- "factor(day_of_week) + factor(hour_of_day) + holiday"

  # 添加天气变量（使用之前确定的可用变量）
  if (length(available_weather_vars) > 0) {
    weather_controls <- paste(available_weather_vars, collapse = " + ")
    base_controls <- paste(base_controls, weather_controls, sep = " + ")
  }

  # 如果有财务控制变量，添加到公式中
  if (financial_controls != "") {
    all_controls <- paste(base_controls, financial_controls, sep = " + ")
  } else {
    all_controls <- base_controls
  }

  # 使用标准lm函数进行回归
  formula_str <- sprintf(
    "%s ~ treatment + post + treatment_post + %s",
    log_outcome_var, all_controls
  )

  formula_obj <- as.formula(formula_str)

  # 运行标准线性回归
  cat("\n运行线性回归模型...\n")
  model <- lm(formula_obj, data = analysis_data)

  # 检查共线性（使用模型摘要中的信息）
  cat("\n=== 模型诊断 ===\n")
  
  # 检查是否有变量被删除（由于完全共线性）
  if (any(is.na(coef(model)))) {
    cat("警告: 检测到完全共线变量，以下变量被自动删除:\n")
    na_coefs <- names(coef(model))[is.na(coef(model))]
    for (var in na_coefs) {
      cat(sprintf("  - %s\n", var))
    }
  } else {
    cat("未检测到完全共线变量（所有变量都被保留）\n")
  }

  # 显示模型中实际使用的变量
  final_coefs <- names(coef(model))
  cat("\n完整模型中实际估计的系数:\n")
  cat(sprintf("  核心变量: treatment, post, treatment_post\n"))
  control_coefs <- setdiff(final_coefs, c("treatment", "post", "treatment_post", "(Intercept)"))
  if (length(control_coefs) > 0) {
    cat(sprintf("  控制变量: %s\n", paste(control_coefs, collapse = ", ")))
  }

  # 使用稳健标准误（HC3）
  robust_se <- sqrt(diag(vcovHC(model, type = "HC3")))
  
  # 提取treatment_post系数（政策效应）
  coef_table <- coef(model)
  se_table <- robust_se

  if ("treatment_post" %in% names(coef_table)) {
    treatment_effect <- coef_table["treatment_post"]
    treatment_se <- se_table["treatment_post"]

    # 计算p值（使用t分布，自由度为n-k）
    df_residual <- model$df.residual
    t_stat <- treatment_effect / treatment_se
    treatment_pvalue <- 2 * (1 - pt(abs(t_stat), df = df_residual))

    # 计算95%置信区间（使用t分布的临界值）
    t_critical <- qt(0.975, df = df_residual)
    treatment_ci_lower <- treatment_effect - t_critical * treatment_se
    treatment_ci_upper <- treatment_effect + t_critical * treatment_se

    # 打印结果
    cat(sprintf("\n政策效应 (β₃): %.4f\n", treatment_effect))
    cat(sprintf("标准误: %.4f\n", treatment_se))
    cat(sprintf("P值: %.6f\n", treatment_pvalue))
    cat(sprintf("95%% 置信区间: [%.4f, %.4f]\n", treatment_ci_lower, treatment_ci_upper))

    # 显著性标记
    significance <- ifelse(treatment_pvalue < 0.01, "***",
      ifelse(treatment_pvalue < 0.05, "**",
        ifelse(treatment_pvalue < 0.1, "*", "")
      )
    )
    cat(sprintf("显著性: %s\n", significance))
  } else {
    stop("未找到treatment_post系数，请检查模型设定")
  }

  # 打印模型摘要（使用稳健标准误）
  cat("\n=== 回归模型摘要（稳健标准误）===\n")
  print(coeftest(model, vcov = vcovHC(model, type = "HC3")))

  # 返回结果
  return(list(
    model = model,
    treatment_effect = treatment_effect,
    treatment_se = treatment_se,
    treatment_pvalue = treatment_pvalue,
    treatment_ci = c(treatment_ci_lower, treatment_ci_upper),
    n_obs = nrow(analysis_data),
    outcome_var = outcome_var,
    log_outcome_var = log_outcome_var,
    outcome_name = outcome_name
  ))
}

# ============================================================================
# 3. 描述性统计
# ============================================================================

cat("\n=== 描述性统计 ===\n")

# 按treatment和post分组计算均值
desc_stats <- did_data %>%
  group_by(treatment, post) %>%
  summarise(
    n = n(),
    outflow_trips_mean = mean(outflow_trips, na.rm = TRUE),
    inflow_trips_mean = mean(inflow_trips, na.rm = TRUE),
    outflow_avg_speed_mean = mean(outflow_avg_speed, na.rm = TRUE),
    inflow_avg_speed_mean = mean(inflow_avg_speed, na.rm = TRUE),
    .groups = "drop"
  )

cat("\n按组别和时间段的均值:\n")
print(desc_stats)

# 计算DiD的简单估计（未控制其他变量，使用log转换后的变量）
cat("\n=== 简单DiD估计（未控制其他变量，log转换后）===\n")

for (var in c("outflow_trips", "inflow_trips", "outflow_avg_speed", "inflow_avg_speed")) {
  log_var <- paste0("log_", var)
  
  if (log_var %in% names(did_data)) {
    # Treatment group
    treatment_pre <- mean(did_data[did_data$treatment == 1 & did_data$post == 0, log_var], na.rm = TRUE)
    treatment_post <- mean(did_data[did_data$treatment == 1 & did_data$post == 1, log_var], na.rm = TRUE)
    treatment_change <- treatment_post - treatment_pre

    # Control group
    control_pre <- mean(did_data[did_data$treatment == 0 & did_data$post == 0, log_var], na.rm = TRUE)
    control_post <- mean(did_data[did_data$treatment == 0 & did_data$post == 1, log_var], na.rm = TRUE)
    control_change <- control_post - control_pre

    # DiD estimate
    did_estimate <- treatment_change - control_change

    cat(sprintf("\n%s (log转换后):\n", var))
    cat(sprintf(
      "  Treatment group - Pre: %.4f, Post: %.4f, Change: %.4f\n",
      treatment_pre, treatment_post, treatment_change
    ))
    cat(sprintf(
      "  Control group - Pre: %.4f, Post: %.4f, Change: %.4f\n",
      control_pre, control_post, control_change
    ))
    cat(sprintf("  DiD估计: %.4f\n", did_estimate))
    cat(sprintf("  注意: 这是log尺度上的DiD估计\n"))
  } else {
    cat(sprintf("\n警告: %s 的log转换变量不存在\n", var))
  }
}

# ============================================================================
# 4. 运行DiD分析
# ============================================================================

results <- list()

# 分析outflow_trips
results$outflow_trips <- run_did_analysis(
  did_data,
  "outflow_trips",
  "流出行程数 (Outflow Trips)"
)

# 分析inflow_trips
results$inflow_trips <- run_did_analysis(
  did_data,
  "inflow_trips",
  "流入行程数 (Inflow Trips)"
)

# 分析outflow_avg_speed
results$outflow_avg_speed <- run_did_analysis(
  did_data,
  "outflow_avg_speed",
  "流出平均速度 (Outflow Average Speed)"
)

# 分析inflow_avg_speed
results$inflow_avg_speed <- run_did_analysis(
  did_data,
  "inflow_avg_speed",
  "流入平均速度 (Inflow Average Speed)"
)

# ============================================================================
# 5. 结果汇总
# ============================================================================

cat("\n\n=== DiD分析结果汇总 ===\n")
cat(paste0(rep("=", 80), collapse = ""), "\n")

summary_table <- data.frame(
  Outcome = character(),
  Treatment_Effect = numeric(),
  Std_Error = numeric(),
  P_Value = numeric(),
  CI_Lower = numeric(),
  CI_Upper = numeric(),
  Significance = character(),
  N_Obs = integer(),
  stringsAsFactors = FALSE
)

for (result_name in names(results)) {
  result <- results[[result_name]]

  significance <- ifelse(result$treatment_pvalue < 0.01, "***",
    ifelse(result$treatment_pvalue < 0.05, "**",
      ifelse(result$treatment_pvalue < 0.1, "*", "")
    )
  )

  summary_table <- rbind(summary_table, data.frame(
    Outcome = result$outcome_name,
    Treatment_Effect = result$treatment_effect,
    Std_Error = result$treatment_se,
    P_Value = result$treatment_pvalue,
    CI_Lower = result$treatment_ci[1],
    CI_Upper = result$treatment_ci[2],
    Significance = significance,
    N_Obs = result$n_obs,
    stringsAsFactors = FALSE
  ))
}

print(summary_table)

cat("\n显著性标记: *** p<0.01, ** p<0.05, * p<0.1\n")

cat("\n=== 分析完成 ===\n")
cat("结束时间:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
