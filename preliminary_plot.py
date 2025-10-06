import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv('data/hourly_taxi_summary.csv', parse_dates=['pickup_hour'])

# 查看基本信息
print("数据形状:", df.shape)
print("\n数据列名:")
print(df.columns.tolist())
print("\n数据预览:")
print(df.head())

# 定义政策日期
policy_date = pd.Timestamp('2025-01-05')

# 创建政策标识
df['is_post_policy'] = df['pickup_hour'] >= policy_date

# 基本统计
print("\n基本统计:")
print(df.describe())

# 可视化：四张单独的图，并调整尺寸与线宽

# 1) 总行程数时间序列
plt.figure(figsize=(20, 6))
plt.plot(df['pickup_hour'], df['total_trips'], alpha=0.7, linewidth=0.7)
plt.axvline(policy_date, color='red', linestyle='--', linewidth=2, label='政策实施日')
plt.title('总行程数时间序列', fontsize=16, fontweight='bold')
plt.xlabel('时间')
plt.ylabel('每小时行程数')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('total_trips_timeseries.png', dpi=300, bbox_inches='tight')

# 2) CBD内部行程比例时间序列
plt.figure(figsize=(20, 6))
plt.plot(df['pickup_hour'], df['cbd_inside_ratio'], alpha=0.7, linewidth=0.7)
plt.axvline(policy_date, color='red', linestyle='--', linewidth=2, label='政策实施日')
plt.title('CBD内部行程比例时间序列', fontsize=16, fontweight='bold')
plt.xlabel('时间')
plt.ylabel('比例')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cbd_inside_ratio_timeseries.png', dpi=300, bbox_inches='tight')

# 3) 平均速度时间序列
plt.figure(figsize=(20, 6))
plt.plot(df['pickup_hour'], df['avg_speed'], alpha=0.7, linewidth=0.7)
plt.axvline(policy_date, color='red', linestyle='--', linewidth=2, label='政策实施日')
plt.title('平均速度时间序列', fontsize=16, fontweight='bold')
plt.xlabel('时间')
plt.ylabel('速度 (mph)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('avg_speed_timeseries.png', dpi=300, bbox_inches='tight')

# 4) CBD交互比例堆叠面积图
plt.figure(figsize=(20, 6))
plt.fill_between(df['pickup_hour'], 0, df['cbd_inside_ratio'],
                 alpha=0.5, linewidth=0.0, label='CBD内部')
plt.fill_between(df['pickup_hour'], df['cbd_inside_ratio'],
                 df['cbd_inside_ratio'] + df['cbd_in_ratio'],
                 alpha=0.5, linewidth=0.0, label='进入CBD')
plt.fill_between(df['pickup_hour'],
                 df['cbd_inside_ratio'] + df['cbd_in_ratio'],
                 df['cbd_inside_ratio'] + df['cbd_in_ratio'] + df['cbd_out_ratio'],
                 alpha=0.5, linewidth=0.0, label='离开CBD')
plt.axvline(policy_date, color='red', linestyle='--', linewidth=2, label='政策实施日')
plt.title('CBD交互比例分布', fontsize=16, fontweight='bold')
plt.xlabel('时间')
plt.ylabel('累积比例')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cbd_interaction_stacked_area.png', dpi=300, bbox_inches='tight')

plt.show()

print("\n✅ 可视化完成！已生成以下图片：")
print("- total_trips_timeseries.png")
print("- cbd_inside_ratio_timeseries.png")
print("- avg_speed_timeseries.png")
print("- cbd_interaction_stacked_area.png")