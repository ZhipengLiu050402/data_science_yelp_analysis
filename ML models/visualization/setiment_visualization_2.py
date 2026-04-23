import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from math import pi
from matplotlib.lines import Line2D  # 用于高级图例自定义

# ==========================================
# 0. 整体风格与文件夹设置
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

folder_name = 'sentiment_visualization'
os.makedirs(folder_name, exist_ok=True)
print(f"初始化完成，图片将保存至: {folder_name}/")

# ==========================================
# 1. 核心数据准备
# ==========================================
models = ['Logistic Regression', 'Random Forest', 'Linear SVM', 'SGB']
categories = ['Negative (0)', 'Neutral (1)', 'Positive (2)']
colors = sns.color_palette("muted", len(models))  # 统一模型颜色

# 准确率数据
accuracies = [0.812, 0.751, 0.806, 0.772]

# F1-score 数据 (格式: [Negative, Neutral, Positive])
f1_data = {
    'Logistic Regression': [0.78, 0.80, 0.85],
    'Random Forest': [0.67, 0.76, 0.79],
    'Linear SVM': [0.77, 0.80, 0.84],
    'SGB': [0.70, 0.77, 0.81]
}

# 数据对应顺序必须是: [Negative(0), Neutral(1), Positive(2)]
precision_data = {
    'Logistic Regression': [0.85, 0.54, 0.91],
    'Random Forest': [0.87, 0.67, 0.84],
    'Linear SVM': [0.85, 0.47, 0.95],
    'SGB': [0.85, 0.55, 0.89]
}

recall_data = {
    'Logistic Regression': [0.88, 0.30, 0.96],
    'Random Forest': [0.79, 0.01, 0.99],
    'Linear SVM': [0.88, 0.49, 0.93],
    'SGB': [0.85, 0.25, 0.97]
}

# 将字典转换为热力图所需的 numpy 矩阵
f1_matrix = np.array(list(f1_data.values()))

# ==========================================
# 图表 1: 模型测试准确率对比 (Bar Chart)
# ==========================================
plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracies, color=colors, edgecolor='black', alpha=0.85, width=0.5)
plt.title('Test Accuracy Comparison', fontsize=16, fontweight='bold', pad=15)
plt.ylabel('Test Accuracy', fontweight='bold')
plt.ylim(0, 1.0)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f'{height:.3f}', ha='center', fontweight='bold')
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(folder_name, '01_accuracy_bar.png'), dpi=300)
plt.close()

# ==========================================
# 图表 2: 各情感类别 F1-Score 热力图 (Heatmap)
# ==========================================
plt.figure(figsize=(9, 6))
sns.heatmap(f1_matrix, annot=True, fmt=".2f", cmap='Blues',
            xticklabels=categories, yticklabels=models,
            vmin=0.6, vmax=1.0, linewidths=1, linecolor='white', annot_kws={"weight": "bold"})
plt.title('F1-Score Heatmap', fontsize=16, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(folder_name, '02_f1_heatmap.png'), dpi=300)
plt.close()

# ==========================================
# 图表 3: 模型能力雷达图 (Radar Chart)
# ==========================================
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]  # 闭合

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
plt.xticks(angles[:-1], categories, fontweight='bold')
ax.set_rlabel_position(0)
plt.yticks([0.6, 0.7, 0.8], ["0.6", "0.7", "0.8"], color="grey", size=10)
plt.ylim(0.55, 0.9)

for i, (model_name, values) in enumerate(f1_data.items()):
    values_closed = values + values[:1]
    ax.plot(angles, values_closed, linewidth=2, label=model_name, color=colors[i])
    ax.fill(angles, values_closed, color=colors[i], alpha=0.1)

plt.title('Model Capability Profile (F1-Score)', size=16, fontweight='bold', y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig(os.path.join(folder_name, '03_f1_radar.png'), dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# 图表 4: 类别性能斜率图 (Slope Graph)
# ==========================================
plt.figure(figsize=(9, 6))
short_categories = ['Neg', 'Neu', 'Pos']  # 简写以防拥挤
for i, (model_name, values) in enumerate(f1_data.items()):
    plt.plot(short_categories, values, marker='o', markersize=8, linewidth=2.5, label=model_name, color=colors[i])
    for j, val in enumerate(values):
        offset = 0.005 if i % 2 == 0 else -0.015
        plt.text(j, val + offset, f'{val:.2f}', color=colors[i], fontweight='bold', ha='center', va='bottom',
                 fontsize=10)

plt.title('Performance Trend across Sentiment', fontsize=16, fontweight='bold', pad=15)
plt.ylabel('F1-Score', fontweight='bold')
plt.ylim(0.60, 0.90)
sns.despine()
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(folder_name, '04_f1_slope.png'), dpi=300)
plt.close()

# ==========================================
# 图表 5: 精确率-召回率 权衡散点图 (P-R Scatter Plot)
# ==========================================
plt.figure(figsize=(10, 7))

# 定义三个类别的特定形状
markers = ['o', 's', '^']  # 圆圈(Neg), 方块(Neu), 三角(Pos)

# 绘制散点和连线
for i, model_name in enumerate(models):
    p_vals = precision_data[model_name]
    r_vals = recall_data[model_name]

    # 用虚线连接同一个模型的三个点，展示模型的整体特征区域
    plt.plot(r_vals, p_vals, color=colors[i], alpha=0.4, linestyle='--', linewidth=1.5)

    for j in range(len(categories)):
        plt.scatter(r_vals[j], p_vals[j], color=colors[i], marker=markers[j],
                    s=180, alpha=0.9, edgecolor='white', linewidth=1.5, zorder=5)

plt.title('Precision-Recall Trade-off Analysis', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Recall', fontsize=13, fontweight='bold')
plt.ylabel('Precision', fontsize=13, fontweight='bold')

# 添加等值线背景 (F1=0.7, 0.8) 提升高级感
x = np.linspace(0.5, 1.0, 100)
y = np.linspace(0.5, 1.0, 100)
X, Y = np.meshgrid(x, y)
F1 = 2 * (X * Y) / (X + Y)
cs = plt.contour(X, Y, F1, levels=[0.7, 0.75, 0.8, 0.85], colors='gray', alpha=0.2, linestyles='dotted')
plt.clabel(cs, inline=True, fontsize=10, fmt='F1=%.2f')

plt.xlim(0.6, 0.9)
plt.ylim(0.6, 0.9)

# --- 制作双图例 (高级技巧) ---
# 1. 模型颜色图例
color_lines = [Line2D([0], [0], color=colors[i], lw=4) for i in range(len(models))]
legend1 = plt.legend(color_lines, models, title='Models', loc='lower right', bbox_to_anchor=(0.95, 0.05))
plt.gca().add_artist(legend1)  # 保留第一个图例

# 2. 类别形状图例
shape_lines = [Line2D([0], [0], marker=markers[j], color='gray', linestyle='None', markersize=10) for j in
               range(len(categories))]
plt.legend(shape_lines, categories, title='Sentiment Class', loc='upper left', bbox_to_anchor=(0.05, 0.95))

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(folder_name, '05_pr_scatter.png'), dpi=300)
plt.close()

print("\n全部 5 张图表已成功生成并保存在 sentiment_visualization 文件夹中！")