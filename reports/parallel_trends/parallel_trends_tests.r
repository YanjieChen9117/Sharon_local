#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-
# ============================================================================
# 平行趋势假设检验 (Parallel Trends Assumption Tests)
# ============================================================================
#
# 功能：
# 1. 可视化检验：绘制treatment组和control组在pre-policy期间的趋势图
# 2. Formal检验1：时间趋势差异检验（只使用pre-policy数据）
# 3. Formal检验2：Event Study分析（lead-lag分析）
# 4. Informal检验：Placebo测试（假设虚假政策日期）
#
# 作者: Yanjie Chen
# 日期: 2025-12-09
# ============================================================================

# 加载必要的库
required_packages <- c("dplyr", "lubridate", "lmtest", "sandwich", "ggplot2", "tidyr")
missing_packages <- required_packages[!(required_packages %in% installed.packages()[, "Package"])]

if (length(missing_packages) > 0) {
  cat("正在安装缺失的R包:", paste(missing_packages, collapse = ", "), "\n")
  install.packages(missing_packages, repos = "https://cran.rstudio.com/")
}

library(dplyr)
library(lubridate)
library(lmtest)
library(sandwich)
library(ggplot2)
library(tidyr)

# ============================================================================
# 1. 数据加载和准备
# ============================================================================

cat("=== 加载数据 ===\n")

# 读取数据
data_path <- "/Users/yanjiechen/Documents/Github/Sharon_local/data/nta_zone_hourly_taxi_summary.csv"
df <- read.csv(data_path, stringsAsFactors = FALSE)

# 数据类型转换
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

# 筛选数据
pre_data <- df %>%
  filter(date >= pre_start & date <= pre_end) %>%
  mutate(post = 0)

post_data <- df %>%
  filter(date >= post_start & date <= post_end) %>%
  mutate(post = 1)

did_data <- bind_rows(pre_data, post_data)
did_data$treatment <- did_data$is_cbd

# 数据类型转换
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

# Log转换
cat("\n=== 进行log转换 ===\n")
dependent_vars <- c("outflow_trips", "inflow_trips", "outflow_avg_speed", "inflow_avg_speed")

for (var in dependent_vars) {
  if (var %in% names(did_data)) {
    log_var_name <- paste0("log_", var)
    if (grepl("trips", var)) {
      did_data[[log_var_name]] <- log1p(as.numeric(did_data[[var]]))
    } else {
      var_data <- as.numeric(did_data[[var]])
      did_data[[log_var_name]] <- log1p(pmax(var_data, 0))
    }
    cat(sprintf("  已转换: %s -> %s\n", var, log_var_name))
  }
}

cat(sprintf("\n总观测数: %d\n", nrow(did_data)))
cat(sprintf("Pre-policy观测数: %d\n", sum(did_data$post == 0)))
cat(sprintf("Post-policy观测数: %d\n", sum(did_data$post == 1)))
cat(sprintf("Treatment组观测数: %d\n", sum(did_data$treatment == 1)))
cat(sprintf("Control组观测数: %d\n", sum(did_data$treatment == 0)))

# ============================================================================
# 2. 可视化检验：趋势图
# ============================================================================

cat("\n\n" , paste0(rep("=", 80), collapse = ""), "\n")
cat("=== 方法1: 可视化趋势检验 ===\n")
cat(paste0(rep("=", 80), collapse = ""), "\n\n")

# 创建输出目录
output_dir <- "/Users/yanjiechen/Documents/Github/Sharon_local/reports/parallel_trends"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# 按周汇总数据（用于可视化）
cat("正在生成趋势可视化...\n")

weekly_data <- did_data %>%
  mutate(
    week = floor_date(date, "week"),
    treatment_group = ifelse(treatment == 1, "Treatment (CBD)", "Control (Non-CBD)")
  ) %>%
  group_by(week, treatment_group, post) %>%
  summarise(
    log_outflow_trips = mean(log_outflow_trips, na.rm = TRUE),
    log_inflow_trips = mean(log_inflow_trips, na.rm = TRUE),
    log_outflow_avg_speed = mean(log_outflow_avg_speed, na.rm = TRUE),
    log_inflow_avg_speed = mean(log_inflow_avg_speed, na.rm = TRUE),
    .groups = "drop"
  )

# 创建时间索引来处理gap（移除数据缺失期间）
# 为每个唯一的周分配一个连续的索引
unique_weeks <- sort(unique(weekly_data$week))
week_index_map <- data.frame(
  week = unique_weeks,
  week_index = seq_along(unique_weeks)
)

weekly_data <- weekly_data %>%
  left_join(week_index_map, by = "week")

# 为每个outcome变量创建趋势图
outcome_vars <- list(
  "log_outflow_trips" = "Log(Outflow Trips)",
  "log_inflow_trips" = "Log(Inflow Trips)",
  "log_outflow_avg_speed" = "Log(Outflow Avg Speed)",
  "log_inflow_avg_speed" = "Log(Inflow Avg Speed)"
)

for (var_name in names(outcome_vars)) {
  var_label <- outcome_vars[[var_name]]
  
  # 找到政策实施时间点在连续索引中的位置
  # pre_end是2024-08-31，找到最接近的week
  policy_week <- max(weekly_data$week[weekly_data$post == 0])
  policy_index <- weekly_data$week_index[weekly_data$week == policy_week][1]
  
  # 创建用于x轴标签的数据（选择部分周显示标签）
  # 选择每个月的第一周
  label_weeks <- weekly_data %>%
    group_by(year_month = format(week, "%Y-%m")) %>%
    slice(1) %>%
    ungroup() %>%
    select(week, week_index) %>%
    distinct()
  
  p <- ggplot(weekly_data, aes(x = week_index, y = .data[[var_name]], 
                                color = treatment_group, 
                                linetype = treatment_group)) +
    geom_line(linewidth = 1) +
    geom_vline(xintercept = policy_index + 0.5, linetype = "dashed", 
               color = "red", linewidth = 0.8) +
    annotate("text", x = policy_index + 0.5, y = Inf, label = "Policy Implementation", 
             vjust = 1.5, hjust = 1, color = "red", size = 4) +
    labs(
      title = paste("Parallel Trends Test:", var_label),
      subtitle = "Weekly aggregated time trends (red dashed line indicates policy implementation)",
      x = "Date",
      y = var_label,
      color = "Group",
      linetype = "Group"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 11),
      legend.position = "bottom",
      legend.title = element_text(size = 11),
      legend.text = element_text(size = 10),
      axis.text.x = element_text(angle = 45, hjust = 1)
    ) +
    scale_color_manual(values = c("Treatment (CBD)" = "#E74C3C", 
                                   "Control (Non-CBD)" = "#3498DB")) +
    scale_x_continuous(
      breaks = label_weeks$week_index,
      labels = format(label_weeks$week, "%Y-%m")
    )
  
  # 保存图片
  filename <- file.path(output_dir, paste0("trend_", var_name, ".png"))
  ggsave(filename, plot = p, width = 12, height = 6, dpi = 300)
  cat(sprintf("  已保存: %s\n", filename))
}

cat("\n可视化检验说明：\n")
cat("- 如果两条线在pre-policy期间（红线之前）基本平行，说明平行趋势假设成立\n")
cat("- 如果两条线在pre-policy期间有明显的趋势差异，说明平行趋势假设可能不成立\n")

# ============================================================================
# 3. Formal检验1：时间趋势差异检验（Pre-policy数据）
# ============================================================================

cat("\n\n", paste0(rep("=", 80), collapse = ""), "\n")
cat("=== 方法2: 时间趋势差异检验（仅使用Pre-policy数据）===\n")
cat(paste0(rep("=", 80), collapse = ""), "\n\n")

cat("原理：\n")
cat("在pre-policy期间运行回归：Y = β₀ + β₁*Treatment + β₂*Time + β₃*(Treatment×Time) + Controls\n")
cat("如果β₃显著不为0，说明treatment组和control组有不同的时间趋势，违反平行趋势假设\n\n")

# 只使用pre-policy数据
pre_only_data <- did_data %>% filter(post == 0)

# 创建时间趋势变量（从第一天开始的天数）
pre_only_data$time_trend <- as.numeric(pre_only_data$date - min(pre_only_data$date))

# 创建交互项
pre_only_data$treatment_time <- pre_only_data$treatment * pre_only_data$time_trend

# 存储结果
trend_test_results <- list()

for (var_name in names(outcome_vars)) {
  var_label <- outcome_vars[[var_name]]
  
  cat(sprintf("\n--- %s ---\n", var_label))
  
  # 准备分析数据（移除缺失值）
  analysis_vars <- c(var_name, "treatment", "time_trend", "treatment_time",
                     "day_of_week", "hour_of_day", "holiday",
                     "weather_temperature", "weather_precipitation",
                     "weather_windspeed", "weather_snow")
  
  analysis_data <- pre_only_data
  for (v in analysis_vars) {
    if (v %in% names(analysis_data)) {
      analysis_data <- analysis_data %>% filter(!is.na(.data[[v]]))
    }
  }
  
  cat(sprintf("有效观测数: %d\n", nrow(analysis_data)))
  
  # 构建回归公式
  formula_str <- sprintf(
    "%s ~ treatment + time_trend + treatment_time + factor(day_of_week) + factor(hour_of_day) + holiday + weather_temperature + weather_precipitation + weather_windspeed + weather_snow",
    var_name
  )
  
  # 运行回归
  model <- lm(as.formula(formula_str), data = analysis_data)
  
  # 使用稳健标准误
  robust_vcov <- vcovHC(model, type = "HC3")
  robust_test <- coeftest(model, vcov = robust_vcov)
  
  # 提取treatment_time系数
  if ("treatment_time" %in% rownames(robust_test)) {
    coef_val <- robust_test["treatment_time", "Estimate"]
    se_val <- robust_test["treatment_time", "Std. Error"]
    t_val <- robust_test["treatment_time", "t value"]
    p_val <- robust_test["treatment_time", "Pr(>|t|)"]
    
    cat(sprintf("Treatment×Time系数: %.6f\n", coef_val))
    cat(sprintf("标准误: %.6f\n", se_val))
    cat(sprintf("t值: %.4f\n", t_val))
    cat(sprintf("P值: %.6f\n", p_val))
    
    # 判断
    if (p_val < 0.05) {
      cat("结论: ❌ 拒绝平行趋势假设（p < 0.05）\n")
      conclusion <- "拒绝"
    } else if (p_val < 0.1) {
      cat("结论: ⚠️  弱证据反对平行趋势假设（0.05 < p < 0.1）\n")
      conclusion <- "弱拒绝"
    } else {
      cat("结论: ✓ 未能拒绝平行趋势假设（p > 0.1）\n")
      conclusion <- "接受"
    }
    
    trend_test_results[[var_name]] <- list(
      coefficient = coef_val,
      se = se_val,
      t_value = t_val,
      p_value = p_val,
      conclusion = conclusion,
      n_obs = nrow(analysis_data)
    )
  }
}

# ============================================================================
# 4. Formal检验2：Event Study分析
# ============================================================================

cat("\n\n", paste0(rep("=", 80), collapse = ""), "\n")
cat("=== 方法3: Event Study 分析 ===\n")
cat(paste0(rep("=", 80), collapse = ""), "\n\n")

cat("原理：\n")
cat("将pre-policy期间分成多个时期，创建每个时期与treatment的交互项\n")
cat("如果所有pre-policy时期的系数都不显著，说明平行趋势假设成立\n\n")

# 创建月份变量（相对于政策实施月份）
# Pre-policy: 2024-01 到 2024-08 -> relative months: -7 to 0
# Post-policy: 2025-01 到 2025-08 -> relative months: 1 to 8

did_data$year_month <- as.Date(format(did_data$date, "%Y-%m-01"))

# 定义基准月（政策实施前一个月，即2024-08）
reference_month <- as.Date("2024-08-01")

did_data$relative_month <- interval(reference_month, did_data$year_month) %/% months(1)

# 确保relative_month是整数
did_data$relative_month <- as.integer(did_data$relative_month)

cat(sprintf("Relative month 范围: %d 到 %d\n", 
            min(did_data$relative_month, na.rm = TRUE),
            max(did_data$relative_month, na.rm = TRUE)))

# Event study结果存储
event_study_results <- list()

for (var_name in names(outcome_vars)) {
  var_label <- outcome_vars[[var_name]]
  
  cat(sprintf("\n--- %s ---\n", var_label))
  
  # 准备分析数据
  analysis_data <- did_data %>%
    filter(!is.na(.data[[var_name]]),
           !is.na(relative_month),
           !is.na(treatment),
           !is.na(day_of_week),
           !is.na(hour_of_day),
           !is.na(holiday),
           !is.na(weather_temperature),
           !is.na(weather_precipitation),
           !is.na(weather_windspeed),
           !is.na(weather_snow))
  
  cat(sprintf("有效观测数: %d\n", nrow(analysis_data)))
  
  # 创建月份虚拟变量（排除基准月，即relative_month = 0）
  unique_months <- sort(unique(analysis_data$relative_month))
  unique_months <- unique_months[unique_months != 0]  # 排除基准月
  
  cat(sprintf("月份范围: %d 到 %d (排除基准月 0)\n", 
              min(unique_months), max(unique_months)))
  
  # 为每个月创建treatment交互项（使用简单的变量名）
  for (m in unique_months) {
    # 使用正数作为变量名（避免负号问题）
    var_name_m <- paste0("tm_", abs(m), ifelse(m < 0, "pre", "post"))
    analysis_data[[var_name_m]] <- ifelse(
      analysis_data$treatment == 1 & analysis_data$relative_month == m, 1, 0
    )
  }
  
  # 构建交互项列表
  interaction_vars <- character()
  for (m in unique_months) {
    var_name_m <- paste0("tm_", abs(m), ifelse(m < 0, "pre", "post"))
    interaction_vars <- c(interaction_vars, var_name_m)
  }
  
  interaction_terms <- paste(interaction_vars, collapse = " + ")
  
  # 构建回归公式（使用relative_month作为连续变量而不是因子）
  formula_str <- sprintf(
    "%s ~ treatment + relative_month + %s + factor(day_of_week) + factor(hour_of_day) + holiday + weather_temperature + weather_precipitation + weather_windspeed + weather_snow",
    var_name, interaction_terms
  )
  
  cat(sprintf("\n运行回归模型...\n"))
  cat(sprintf("交互项数量: %d\n", length(interaction_vars)))
  
  # 运行回归
  model <- lm(as.formula(formula_str), data = analysis_data)
  
  # 使用稳健标准误
  robust_vcov <- vcovHC(model, type = "HC3")
  robust_test <- coeftest(model, vcov = robust_vcov)
  
  # 提取所有交互项系数
  interaction_coefs <- grep("^tm_", rownames(robust_test), value = TRUE)
  
  # 分离pre-policy和post-policy系数
  pre_months <- unique_months[unique_months < 0]
  post_months <- unique_months[unique_months > 0]
  
  cat("\nPre-policy期间的treatment效应:\n")
  pre_significant <- 0
  for (m in pre_months) {
    coef_name <- paste0("tm_", abs(m), "pre")
    if (coef_name %in% rownames(robust_test)) {
      coef_val <- robust_test[coef_name, "Estimate"]
      p_val <- robust_test[coef_name, "Pr(>|t|)"]
      sig_mark <- ifelse(p_val < 0.05, "**", ifelse(p_val < 0.1, "*", ""))
      cat(sprintf("  Month %d: %.6f (p=%.4f) %s\n", m, coef_val, p_val, sig_mark))
      if (p_val < 0.05) pre_significant <- pre_significant + 1
    }
  }
  
  cat("\nPost-policy期间的treatment效应:\n")
  for (m in post_months) {
    coef_name <- paste0("tm_", abs(m), "post")
    if (coef_name %in% rownames(robust_test)) {
      coef_val <- robust_test[coef_name, "Estimate"]
      p_val <- robust_test[coef_name, "Pr(>|t|)"]
      sig_mark <- ifelse(p_val < 0.05, "**", ifelse(p_val < 0.1, "*", ""))
      cat(sprintf("  Month %d: %.6f (p=%.4f) %s\n", m, coef_val, p_val, sig_mark))
    }
  }
  
  # 判断平行趋势
  if (pre_significant == 0) {
    cat("\n结论: ✓ Pre-policy期间无显著treatment效应，支持平行趋势假设\n")
    conclusion <- "支持"
  } else {
    cat(sprintf("\n结论: ❌ Pre-policy期间有%d个月显示显著treatment效应，可能违反平行趋势假设\n", 
                pre_significant))
    conclusion <- "不支持"
  }
  
  # 保存结果
  event_study_results[[var_name]] <- list(
    n_obs = nrow(analysis_data),
    pre_months = pre_months,
    post_months = post_months,
    pre_significant_count = pre_significant,
    conclusion = conclusion
  )
  
  # 创建event study图
  coef_data <- data.frame()
  for (m in unique_months) {
    coef_name <- paste0("tm_", abs(m), ifelse(m < 0, "pre", "post"))
    if (coef_name %in% rownames(robust_test)) {
      coef_val <- robust_test[coef_name, "Estimate"]
      se_val <- robust_test[coef_name, "Std. Error"]
      ci_lower <- coef_val - 1.96 * se_val
      ci_upper <- coef_val + 1.96 * se_val
      
      coef_data <- rbind(coef_data, data.frame(
        month = m,
        coefficient = coef_val,
        ci_lower = ci_lower,
        ci_upper = ci_upper
      ))
    }
  }
  
  # 添加基准月（系数为0）
  coef_data <- rbind(coef_data, data.frame(
    month = 0,
    coefficient = 0,
    ci_lower = 0,
    ci_upper = 0
  ))
  
  coef_data <- coef_data %>% arrange(month)
  
  # 绘制event study图
  p <- ggplot(coef_data, aes(x = month, y = coefficient)) +
    geom_point(size = 3) +
    geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), width = 0.2) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    geom_vline(xintercept = 0.5, linetype = "dashed", color = "red", linewidth = 0.8) +
    labs(
      title = paste("Event Study:", var_label),
      subtitle = "Treatment effects over time (relative to one month before policy)",
      x = "Relative Month (0 = One Month Before Policy)",
      y = "Treatment Coefficient"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 11)
    )
  
  filename <- file.path(output_dir, paste0("event_study_", var_name, ".png"))
  ggsave(filename, plot = p, width = 12, height = 6, dpi = 300)
  cat(sprintf("\n已保存event study图: %s\n", filename))
}

# ============================================================================
# 5. Informal检验：Placebo测试
# ============================================================================

cat("\n\n", paste0(rep("=", 80), collapse = ""), "\n")
cat("=== 方法4: Placebo测试 ===\n")
cat(paste0(rep("=", 80), collapse = ""), "\n\n")

cat("原理：\n")
cat("在pre-policy期间选择一个虚假的政策日期（例如2024-05-01）\n")
cat("运行DiD分析，如果发现显著效应，说明可能存在其他因素影响结果\n\n")

# 定义placebo政策日期（pre-policy期间的中点）
placebo_date <- as.Date("2024-05-01")

cat(sprintf("Placebo政策日期: %s\n", placebo_date))

# 只使用pre-policy数据
placebo_data <- did_data %>% filter(post == 0)

# 创建placebo post变量
placebo_data$placebo_post <- ifelse(placebo_data$date >= placebo_date, 1, 0)
placebo_data$placebo_treatment_post <- placebo_data$treatment * placebo_data$placebo_post

# Placebo结果存储
placebo_results <- list()

for (var_name in names(outcome_vars)) {
  var_label <- outcome_vars[[var_name]]
  
  cat(sprintf("\n--- %s ---\n", var_label))
  
  # 准备分析数据
  analysis_data <- placebo_data %>%
    filter(!is.na(.data[[var_name]]),
           !is.na(treatment),
           !is.na(placebo_post),
           !is.na(day_of_week),
           !is.na(hour_of_day),
           !is.na(holiday),
           !is.na(weather_temperature),
           !is.na(weather_precipitation),
           !is.na(weather_windspeed),
           !is.na(weather_snow))
  
  cat(sprintf("有效观测数: %d\n", nrow(analysis_data)))
  cat(sprintf("  Placebo pre: %d\n", sum(analysis_data$placebo_post == 0)))
  cat(sprintf("  Placebo post: %d\n", sum(analysis_data$placebo_post == 1)))
  
  # 构建回归公式
  formula_str <- sprintf(
    "%s ~ treatment + placebo_post + placebo_treatment_post + factor(day_of_week) + factor(hour_of_day) + holiday + weather_temperature + weather_precipitation + weather_windspeed + weather_snow",
    var_name
  )
  
  # 运行回归
  model <- lm(as.formula(formula_str), data = analysis_data)
  
  # 使用稳健标准误
  robust_vcov <- vcovHC(model, type = "HC3")
  robust_test <- coeftest(model, vcov = robust_vcov)
  
  # 提取placebo treatment效应
  if ("placebo_treatment_post" %in% rownames(robust_test)) {
    coef_val <- robust_test["placebo_treatment_post", "Estimate"]
    se_val <- robust_test["placebo_treatment_post", "Std. Error"]
    t_val <- robust_test["placebo_treatment_post", "t value"]
    p_val <- robust_test["placebo_treatment_post", "Pr(>|t|)"]
    
    cat(sprintf("Placebo Treatment效应: %.6f\n", coef_val))
    cat(sprintf("标准误: %.6f\n", se_val))
    cat(sprintf("t值: %.4f\n", t_val))
    cat(sprintf("P值: %.6f\n", p_val))
    
    # 判断
    if (p_val < 0.05) {
      cat("结论: ❌ 发现显著的placebo效应（p < 0.05），可能违反平行趋势假设\n")
      conclusion <- "违反"
    } else if (p_val < 0.1) {
      cat("结论: ⚠️  弱证据显示placebo效应（0.05 < p < 0.1）\n")
      conclusion <- "弱违反"
    } else {
      cat("结论: ✓ 未发现显著placebo效应（p > 0.1），支持平行趋势假设\n")
      conclusion <- "支持"
    }
    
    placebo_results[[var_name]] <- list(
      coefficient = coef_val,
      se = se_val,
      t_value = t_val,
      p_value = p_val,
      conclusion = conclusion,
      n_obs = nrow(analysis_data)
    )
  }
}

# ============================================================================
# 6. 汇总所有检验结果
# ============================================================================

cat("\n\n", paste0(rep("=", 80), collapse = ""), "\n")
cat("=== 平行趋势假设检验结果汇总 ===\n")
cat(paste0(rep("=", 80), collapse = ""), "\n\n")

summary_results <- data.frame(
  Outcome = character(),
  Trend_Test_P = numeric(),
  Trend_Test_Conclusion = character(),
  Event_Study_PreSig = integer(),
  Event_Study_Conclusion = character(),
  Placebo_P = numeric(),
  Placebo_Conclusion = character(),
  Overall_Assessment = character(),
  stringsAsFactors = FALSE
)

for (var_name in names(outcome_vars)) {
  var_label <- outcome_vars[[var_name]]
  
  # 时间趋势检验
  trend_p <- trend_test_results[[var_name]]$p_value
  trend_conclusion <- trend_test_results[[var_name]]$conclusion
  
  # Event study
  event_presig <- event_study_results[[var_name]]$pre_significant_count
  event_conclusion <- event_study_results[[var_name]]$conclusion
  
  # Placebo
  placebo_p <- placebo_results[[var_name]]$p_value
  placebo_conclusion <- placebo_results[[var_name]]$conclusion
  
  # 综合评估
  issues <- 0
  if (trend_conclusion == "拒绝") issues <- issues + 1
  if (event_conclusion == "不支持") issues <- issues + 1
  if (placebo_conclusion == "违反") issues <- issues + 1
  
  if (issues == 0) {
    overall <- "✓ 支持平行趋势假设"
  } else if (issues == 1) {
    overall <- "⚠️ 弱证据支持平行趋势假设"
  } else {
    overall <- "❌ 可能违反平行趋势假设"
  }
  
  summary_results <- rbind(summary_results, data.frame(
    Outcome = var_label,
    Trend_Test_P = trend_p,
    Trend_Test_Conclusion = trend_conclusion,
    Event_Study_PreSig = event_presig,
    Event_Study_Conclusion = event_conclusion,
    Placebo_P = placebo_p,
    Placebo_Conclusion = placebo_conclusion,
    Overall_Assessment = overall,
    stringsAsFactors = FALSE
  ))
}

print(summary_results)

# 保存汇总结果到CSV
summary_file <- file.path(output_dir, "parallel_trends_summary.csv")
write.csv(summary_results, summary_file, row.names = FALSE, fileEncoding = "UTF-8")
cat(sprintf("\n已保存汇总结果: %s\n", summary_file))

# ============================================================================
# 7. 保存详细结果到RDS文件
# ============================================================================

detailed_results <- list(
  trend_test = trend_test_results,
  event_study = event_study_results,
  placebo_test = placebo_results,
  summary = summary_results
)

rds_file <- file.path(output_dir, "parallel_trends_detailed_results.rds")
saveRDS(detailed_results, rds_file)
cat(sprintf("已保存详细结果: %s\n", rds_file))

cat("\n=== 所有平行趋势检验完成 ===\n")
cat(sprintf("结束时间: %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))
cat(sprintf("\n结果保存在: %s\n", output_dir))

