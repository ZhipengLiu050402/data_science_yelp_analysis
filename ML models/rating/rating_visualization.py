import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from math import pi
from matplotlib.lines import Line2D

# ==========================================
# 0. 整体风格与文件夹设置
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

folder_name = 'rating_visualization'
os.makedirs(folder_name, exist_ok=True)
print(f"初始化完成，Rating 任务的图片将保存至: {folder_name}/")

# ==========================================
# 1. 核心数据准备 (已从你的图片中准确提取！)
# ==========================================
models = ['Logistic Regression', 'Random Forest', 'Linear SVM', 'XGBoost']
categories = ['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars']
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']  # 统一模型颜色

# Accuracy 数据
acc_standard = [0.712, 0.588, 0.701, 0.688]
acc_tolerance = [0.972, 0.894, 0.954, 0.923]

# F1-score 数据
f1_data = {
    'Logistic Regression': [0.82, 0.46, 0.52, 0.59, 0.82],
    'Random Forest': [0.75, 0.01, 0.13, 0.49, 0.72],
    'Linear SVM': [0.81, 0.36, 0.43, 0.55, 0.84],
    'XGBoost': [0.79, 0.34, 0.36, 0.49, 0.82]
}

# Precision 数据 (从结果截图提取)
precision_data = {
    'Logistic Regression': [0.83, 0.47, 0.49, 0.52, 0.89],
    'Random Forest': [0.73, 0.47, 0.42, 0.36, 0.82],
    'Linear SVM': [0.78, 0.46, 0.46, 0.52, 0.83],
    'XGBoost': [0.75, 0.50, 0.50, 0.52, 0.76]
}

# Recall 数据 (从结果截图提取)
recall_data = {
    'Logistic Regression': [0.82, 0.44, 0.54, 0.67, 0.77],
    'Random Forest': [0.77, 0.00, 0.08, 0.78, 0.65],
    'Linear SVM': [0.85, 0.30, 0.41, 0.58, 0.84],
    'XGBoost': [0.84, 0.26, 0.28, 0.47, 0.89]
}

f1_matrix = np.array(list(f1_data.values()))

# ==========================================
# 图表 1: 标准准确率 vs 容忍准确率对比
# ==========================================
plt.figure(figsize=(10, 6))
x = np.arange(len(models))
width = 0.35
plt.bar(x - width / 2, acc_standard, width, label='Standard Accuracy', color='#4C72B0', edgecolor='black')
plt.bar(x + width / 2, acc_tolerance, width, label='Tolerance Accuracy (±1 Star)', color='#55A868', edgecolor='black')

plt.ylabel('Accuracy', fontweight='bold')
plt.title('Performance: Standard vs Tolerance Accuracy', fontsize=16, fontweight='bold', pad=15)
plt.xticks(x, models, fontweight='bold')
plt.ylim(0, 1.15)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(folder_name, '01_rating_accuracy_comparison.png'), dpi=300)
plt.close()

# ==========================================
# 图表 2: 预测误差分布堆叠图 (强有力支撑容忍误差指标)
# ==========================================
plt.figure(figsize=(10, 6))
exact_match = np.array(acc_standard)
tol_match = np.array(acc_tolerance) - np.array(acc_standard)
missed = 1.0 - np.array(acc_tolerance)

plt.bar(models, exact_match, label='Exact Match (0 Error)', color='#4C72B0', edgecolor='white')
plt.bar(models, tol_match, bottom=exact_match, label='±1 Star Error (Tolerated)', color='#F2B705', edgecolor='white')
plt.bar(models, missed, bottom=exact_match + tol_match, label='>1 Star Error (Severe)', color='#C44E52',
        edgecolor='white')

plt.ylabel('Proportion', fontweight='bold')
plt.title('Prediction Error Distribution', fontsize=16, fontweight='bold', pad=15)
plt.ylim(0, 1.05)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
sns.despine(top=True, right=True, left=True)
plt.tight_layout()
plt.savefig(os.path.join(folder_name, '02_rating_error_distribution.png'), dpi=300)
plt.close()

# ==========================================
# 图表 3: F1-Score 星级热力图
# ==========================================
plt.figure(figsize=(9, 6))
sns.heatmap(f1_matrix, annot=True, fmt=".2f", cmap='Purples',
            xticklabels=categories, yticklabels=models, vmin=0, vmax=0.9,
            linewidths=1, linecolor='white', annot_kws={"weight": "bold"})
plt.title('F1-Score Heatmap across Ratings', fontsize=16, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(folder_name, '03_rating_f1_heatmap.png'), dpi=300)
plt.close()

# ==========================================
# 图表 4 (新颖): “U型”性能趋势线 (The U-Shape Dip)
# ==========================================
plt.figure(figsize=(10, 6))
for i, (model_name, values) in enumerate(f1_data.items()):
    # 画带标记的折线
    plt.plot(categories, values, marker='o', markersize=9, linewidth=3, label=model_name, color=colors[i], alpha=0.85)

plt.title('The "Middle Rating Dip": F1-Score Trend', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('F1-Score', fontweight='bold')
plt.xlabel('Star Rating', fontweight='bold')
plt.ylim(0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 添加一个高亮区域标示出 2-4 星属于 "Hard to Predict Zone"
plt.axvspan(0.8, 3.2, color='gray', alpha=0.1)
plt.text(2, 0.92, 'Hard to Predict Zone (Middle Ratings)', ha='center', va='center', color='gray', fontweight='bold',
         style='italic')

sns.despine()
plt.legend(title='Models', loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(folder_name, '04_rating_ushape_trend.png'), dpi=300)
plt.close()

# ==========================================
# 图表 5 (新颖): 五维雷达图 (Pentagon Radar Chart)
# ==========================================
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]  # 闭合

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
plt.xticks(angles[:-1], categories, fontweight='bold', size=11)
ax.set_rlabel_position(30)
plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=10)
plt.ylim(0, 0.9)

for i, (model_name, values) in enumerate(f1_data.items()):
    values_closed = values + values[:1]
    ax.plot(angles, values_closed, linewidth=2, label=model_name, color=colors[i])
    ax.fill(angles, values_closed, color=colors[i], alpha=0.05)

plt.title('Rating Capability Radar', size=16, fontweight='bold', y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig(os.path.join(folder_name, '05_rating_radar.png'), dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# 图表 6 (高阶): 精确率-召回率 散点图 (1-5星级)
# ==========================================
plt.figure(figsize=(11, 7))

# 为1-5星定义不同形状，方便区分
markers = ['o', 'v', 's', 'D', '*']

for i, model_name in enumerate(models):
    p_vals = precision_data[model_name]
    r_vals = recall_data[model_name]

    # 虚线连接同一个模型
    plt.plot(r_vals, p_vals, color=colors[i], alpha=0.3, linestyle='-', linewidth=1)

    for j in range(len(categories)):
        # 为了让5星(星形)稍微大一点
        s_size = 300 if j == 4 else 180
        plt.scatter(r_vals[j], p_vals[j], color=colors[i], marker=markers[j],
                    s=s_size, alpha=0.9, edgecolor='white', linewidth=1.5, zorder=5)

plt.title('Precision-Recall Trade-off (Stars 1-5)', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Recall', fontsize=13, fontweight='bold')
plt.ylabel('Precision', fontsize=13, fontweight='bold')

# 等高线背景 (F1=0.2, 0.4, 0.6, 0.8)
x = np.linspace(0.01, 1.0, 100)
y = np.linspace(0.01, 1.0, 100)
X, Y = np.meshgrid(x, y)
F1 = 2 * (X * Y) / (X + Y)
cs = plt.contour(X, Y, F1, levels=[0.2, 0.4, 0.6, 0.8], colors='gray', alpha=0.3, linestyles='dotted')
plt.clabel(cs, inline=True, fontsize=10, fmt='F1=%.1f')

plt.xlim(-0.05, 1.0)
plt.ylim(0.2, 1.0)  # 根据数据，Precision 最低在 0.36

# --- 双图例配置 ---
# 1. 模型颜色
color_lines = [Line2D([0], [0], color=colors[i], lw=4) for i in range(len(models))]
legend1 = plt.legend(color_lines, models, title='Models', loc='lower right')
plt.gca().add_artist(legend1)

# 2. 星级形状
shape_lines = [Line2D([0], [0], marker=markers[j], color='gray', linestyle='None', markersize=10) for j in
               range(len(categories))]
plt.legend(shape_lines, categories, title='Star Rating', loc='upper left')

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(folder_name, '06_rating_pr_scatter.png'), dpi=300)
plt.close()

print("\n全部 6 张图表已成功生成并保存在 rating_visualization 文件夹中！")