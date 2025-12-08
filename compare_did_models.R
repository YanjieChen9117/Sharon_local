#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-
# ============================================================================
# DiD模型对比分析：OLS vs Poisson vs Negative Binomial
# ============================================================================
#
# 功能：
# 1. 并排比较三种模型的结果
# 2. 生成综合对比表格
# 3. 提供模型选择建议
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
library(fixest)
library(MASS)
library(lmtest)
library(sandwich)

# ============================================================================
# 1. 数据加载和准备 (与原脚本相同)
# ============================================================================

cat("=== DiD模型对比分析 ===\n")
cat("开始时间:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")

# 读取数据
data_path <- "/Users/yanjiechen/Documents/Github/Sharon_local/data/nta_zone_hourly_taxi_summary.csv"
df <- read.csv(data_path, stringsAsFactors = FALSE)

# 数据准备
df$year <- as.numeric(df$year)
df$month <- as.numeric(df$month)
df$day <- as.numeric(df$day)
df$date <- as.Date(paste(df$year, df$month, df$day, sep = "-"))

# 定义CBD区域
cbd_zones <- c(
  "MN0101", "MN0102", "MN0301", "MN0201", "MN0302", "MN0601",
  "MN0203", "MN0202", "MN0303", "MN0401", "MN0501",
  "MN0602", "MN0603", "MN0402", "MN0502", "MN0604"
)

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

# 创建变量
did_data$treatment <- did_data$is_cbd
did_data$treatment_post <- did_data$treatment * did_data$post

# 转换变量类型
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

# ============================================================================
# 2. 对比分析函数
# ============================================================================

compare_models <- function(data, outcome_var, outcome_name) {
  cat("\n")
  cat(paste0(rep("=", 80), collapse = ""), "\n")
  cat(sprintf("对比分析: %s\n", outcome_name))
  cat(paste0(rep("=", 80), collapse = ""), "\n")
  
  # 确定财务控制变量
  if (grepl("^outflow", outcome_var)) {
    financial_vars <- intersect(c("outflow_total_tip", "outflow_total_tolls", "outflow_total_fare"), names(data))
  } else if (grepl("^inflow", outcome_var)) {
    financial_vars <- intersect(c("inflow_total_tip", "inflow_total_tolls", "inflow_total_fare"), names(data))
  } else {
    financial_vars <- character(0)
  }
  
  # 准备数据
  analysis_data <- data %>%
    filter(
      !is.na(.data[[outcome_var]]),
      !is.na(treatment), !is.na(post),
      !is.na(day_of_week), !is.na(hour_of_day), !is.na(holiday),
      !is.na(weather_temperature), !is.na(weather_precipitation),
      !is.na(weather_windspeed), !is.na(weather_snow)
    )
  
  for (var in financial_vars) {
    analysis_data <- analysis_data %>% filter(!is.na(.data[[var]]))
  }
  
  # 对于计数数据，确保非负整数
  if (grepl("trips$", outcome_var)) {
    analysis_data <- analysis_data %>%
      filter(.data[[outcome_var]] >= 0) %>%
      mutate(!!outcome_var := round(.data[[outcome_var]]))
  }
  
  cat(sprintf("\n观测数: %d\n", nrow(analysis_data)))
  
  # 构建公式
  base_controls <- "factor(day_of_week) + factor(hour_of_day) + holiday"
  weather_controls <- "weather_temperature + weather_precipitation + weather_windspeed + weather_snow"
  all_controls <- paste(base_controls, weather_controls, sep = " + ")
  
  if (length(financial_vars) > 0) {
    financial_controls <- paste(financial_vars, collapse = " + ")
    all_controls <- paste(all_controls, financial_controls, sep = " + ")
  }
  
  formula_obj <- as.formula(sprintf("%s ~ treatment + post + treatment_post + %s", 
                                    outcome_var, all_controls))
  
  # 拟合三种模型
  cat("\n正在拟合模型...\n")
  
  # OLS
  ols_model <- feols(formula_obj, data = analysis_data, vcov = "HC3")
  
  # Poisson
  poisson_model <- glm(formula_obj, data = analysis_data, family = poisson(link = "log"))
  
  # Negative Binomial
  nb_model <- tryCatch({
    glm.nb(formula_obj, data = analysis_data)
  }, error = function(e) NULL)
  
  # 提取系数和标准误
  # OLS
  ols_coef <- coef(ols_model)["treatment_post"]
  ols_se <- se(ols_model)["treatment_post"]
  ols_t <- ols_coef / ols_se
  ols_p <- 2 * (1 - pnorm(abs(ols_t)))
  ols_ci_lower <- ols_coef - 1.96 * ols_se
  ols_ci_upper <- ols_coef + 1.96 * ols_se
  
  # Poisson
  poisson_coef <- coef(poisson_model)["treatment_post"]
  poisson_vcov <- vcovHC(poisson_model, type = "HC3")
  poisson_se <- sqrt(diag(poisson_vcov))["treatment_post"]
  poisson_z <- poisson_coef / poisson_se
  poisson_p <- 2 * (1 - pnorm(abs(poisson_z)))
  poisson_pct <- (exp(poisson_coef) - 1) * 100
  poisson_ci_lower <- poisson_coef - 1.96 * poisson_se
  poisson_ci_upper <- poisson_coef + 1.96 * poisson_se
  
  # Negative Binomial
  if (!is.null(nb_model)) {
    nb_coef <- coef(nb_model)["treatment_post"]
    nb_vcov <- vcovHC(nb_model, type = "HC3")
    nb_se <- sqrt(diag(nb_vcov))["treatment_post"]
    nb_z <- nb_coef / nb_se
    nb_p <- 2 * (1 - pnorm(abs(nb_z)))
    nb_pct <- (exp(nb_coef) - 1) * 100
    nb_ci_lower <- nb_coef - 1.96 * nb_se
    nb_ci_upper <- nb_coef + 1.96 * nb_se
    nb_theta <- nb_model$theta
  } else {
    nb_coef <- nb_se <- nb_p <- nb_pct <- nb_theta <- NA
    nb_ci_lower <- nb_ci_upper <- NA
  }
  
  # 模型拟合统计
  ols_aic <- AIC(ols_model)
  ols_bic <- BIC(ols_model)
  poisson_aic <- AIC(poisson_model)
  poisson_bic <- BIC(poisson_model)
  
  # 过度离散检验
  pearson_resid <- residuals(poisson_model, type = "pearson")
  dispersion_ratio <- sum(pearson_resid^2) / poisson_model$df.residual
  
  if (!is.null(nb_model)) {
    nb_aic <- AIC(nb_model)
    nb_bic <- BIC(nb_model)
  } else {
    nb_aic <- nb_bic <- NA
  }
  
  # 创建对比表
  cat("\n")
  cat(paste0(rep("-", 80), collapse = ""), "\n")
  cat("模型对比结果\n")
  cat(paste0(rep("-", 80), collapse = ""), "\n")
  
  comparison <- data.frame(
    Metric = c(
      "Treatment Effect",
      "Std Error",
      "P-value",
      "95% CI Lower",
      "95% CI Upper",
      "Significance",
      "Pct Effect",
      "",
      "Model Fit",
      "AIC",
      "BIC",
      "Dispersion Ratio",
      "Theta"
    ),
    OLS = c(
      sprintf("%.4f", ols_coef),
      sprintf("%.4f", ols_se),
      sprintf("%.6f", ols_p),
      sprintf("%.4f", ols_ci_lower),
      sprintf("%.4f", ols_ci_upper),
      ifelse(ols_p < 0.01, "***", ifelse(ols_p < 0.05, "**", ifelse(ols_p < 0.1, "*", ""))),
      "N/A (绝对值)",
      "",
      "",
      sprintf("%.2f", ols_aic),
      sprintf("%.2f", ols_bic),
      "N/A",
      "N/A"
    ),
    Poisson = c(
      sprintf("%.4f", poisson_coef),
      sprintf("%.4f", poisson_se),
      sprintf("%.6f", poisson_p),
      sprintf("%.4f", poisson_ci_lower),
      sprintf("%.4f", poisson_ci_upper),
      ifelse(poisson_p < 0.01, "***", ifelse(poisson_p < 0.05, "**", ifelse(poisson_p < 0.1, "*", ""))),
      sprintf("%.2f%%", poisson_pct),
      "",
      "",
      sprintf("%.2f", poisson_aic),
      sprintf("%.2f", poisson_bic),
      sprintf("%.4f", dispersion_ratio),
      "N/A"
    ),
    Negative_Binomial = c(
      ifelse(is.na(nb_coef), "拟合失败", sprintf("%.4f", nb_coef)),
      ifelse(is.na(nb_se), "", sprintf("%.4f", nb_se)),
      ifelse(is.na(nb_p), "", sprintf("%.6f", nb_p)),
      ifelse(is.na(nb_ci_lower), "", sprintf("%.4f", nb_ci_lower)),
      ifelse(is.na(nb_ci_upper), "", sprintf("%.4f", nb_ci_upper)),
      ifelse(is.na(nb_p), "", ifelse(nb_p < 0.01, "***", ifelse(nb_p < 0.05, "**", ifelse(nb_p < 0.1, "*", "")))),
      ifelse(is.na(nb_pct), "", sprintf("%.2f%%", nb_pct)),
      "",
      "",
      ifelse(is.na(nb_aic), "", sprintf("%.2f", nb_aic)),
      ifelse(is.na(nb_bic), "", sprintf("%.2f", nb_bic)),
      "N/A",
      ifelse(is.na(nb_theta), "", sprintf("%.4f", nb_theta))
    ),
    stringsAsFactors = FALSE
  )
  
  print(comparison, row.names = FALSE)
  
  cat("\n")
  cat("注释:\n")
  cat("- 显著性: *** p<0.01, ** p<0.05, * p<0.1\n")
  cat("- OLS系数表示绝对变化量\n")
  cat("- Poisson/NB系数为对数尺度，百分比效应 = (exp(β)-1)×100%\n")
  cat("- 过度离散比率 > 1.5 建议使用Negative Binomial\n")
  cat("- Theta (θ)越小表示过度离散越严重\n")
  cat("\n")
  
  # 模型选择建议
  cat(paste0(rep("-", 80), collapse = ""), "\n")
  cat("模型选择建议:\n")
  cat(paste0(rep("-", 80), collapse = ""), "\n")
  
  if (dispersion_ratio > 1.5) {
    cat(sprintf("• 过度离散比率 = %.4f > 1.5，数据存在过度离散\n", dispersion_ratio))
    if (!is.null(nb_model)) {
      if (nb_aic < poisson_aic) {
        cat("• Negative Binomial的AIC更小\n")
        cat("✓ 推荐使用: Negative Binomial模型\n")
        best_model <- "Negative Binomial"
      } else {
        cat("• 尽管存在过度离散，但Poisson的AIC更小\n")
        cat("✓ 推荐使用: Poisson模型 (但需注意标准误可能被低估)\n")
        best_model <- "Poisson"
      }
    } else {
      cat("• Negative Binomial模型拟合失败\n")
      cat("✓ 推荐使用: Poisson模型 (使用稳健标准误)\n")
      best_model <- "Poisson"
    }
  } else {
    cat(sprintf("• 过度离散比率 = %.4f < 1.5，无显著过度离散\n", dispersion_ratio))
    cat("✓ 推荐使用: Poisson模型\n")
    best_model <- "Poisson"
  }
  
  cat("\n")
  
  return(list(
    outcome_name = outcome_name,
    comparison_table = comparison,
    best_model = best_model,
    dispersion_ratio = dispersion_ratio,
    n_obs = nrow(analysis_data)
  ))
}

# ============================================================================
# 3. 运行对比分析
# ============================================================================

results <- list()

# 分析outflow_trips
results$outflow_trips <- compare_models(
  did_data,
  "outflow_trips",
  "流出行程数 (Outflow Trips)"
)

# 分析inflow_trips
results$inflow_trips <- compare_models(
  did_data,
  "inflow_trips",
  "流入行程数 (Inflow Trips)"
)

# ============================================================================
# 4. 综合总结
# ============================================================================

cat("\n\n")
cat(paste0(rep("=", 80), collapse = ""), "\n")
cat("=== 综合总结 ===\n")
cat(paste0(rep("=", 80), collapse = ""), "\n")

for (name in names(results)) {
  result <- results[[name]]
  cat(sprintf("\n%s:\n", result$outcome_name))
  cat(sprintf("  • 观测数: %d\n", result$n_obs))
  cat(sprintf("  • 过度离散比率: %.4f\n", result$dispersion_ratio))
  cat(sprintf("  • 推荐模型: %s\n", result$best_model))
}

cat("\n")
cat(paste0(rep("=", 80), collapse = ""), "\n")
cat("关键发现:\n")
cat(paste0(rep("=", 80), collapse = ""), "\n")
cat("\n1. 对于计数数据 (trips):\n")
cat("   - 如果存在过度离散，优先使用Negative Binomial模型\n")
cat("   - 如果过度离散不明显，Poisson模型更简洁高效\n")
cat("   - OLS仅用于参考，不适合计数数据\n")
cat("\n2. 系数解释:\n")
cat("   - Poisson/NB的treatment_post系数是对数尺度\n")
cat("   - 政策的百分比效应 = (exp(β)-1)×100%\n")
cat("   - 例如：β=0.10 → 效应约为10.5%的增长\n")
cat("\n3. 标准误:\n")
cat("   - 使用HC3稳健标准误处理异方差\n")
cat("   - Negative Binomial自动调整过度离散\n")
cat("\n")
cat(paste0(rep("=", 80), collapse = ""), "\n")
cat("分析完成时间:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat(paste0(rep("=", 80), collapse = ""), "\n")

