import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ==========================================
# 1. 整体风格设置 (保留之前的精美设置)
# ==========================================
# 使用内置的 seabron 风格，干净且专业
plt.style.use('seaborn-v0_8-whitegrid')
# 设置全局字体大小，确保清晰度
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

# ==========================================
# 2. 数据准备 (直接使用你图片中的结果)
# ==========================================
models = ['Logistic Regression', 'Random Forest', 'Linear SVM', 'SGB (XGBoost)']
accuracies = [0.812, 0.751, 0.806, 0.772]

# 各类别的 F1-score (行: 模型, 列: 类别 0, 1, 2)
# 类别 0: Negative, 1: Neutral, 2: Positive
# 数据取自你的 Classification Report 中的 F1-score 列
f1_data = np.array([
    [0.78, 0.80, 0.85], # Logistic Regression
    [0.67, 0.76, 0.79], # Random Forest
    [0.77, 0.80, 0.84], # Linear SVM
    [0.70, 0.77, 0.81]  # SGB (XGBoost)
])

# 定义情感标签
sentiment_labels = ['Negative (0)', 'Neutral (1)', 'Positive (2)']

# ==========================================
# 3. 创建目标文件夹 (如果不存在)
# ==========================================
folder_name = 'sentiment_visualization'
# exist_ok=True 防止文件夹已存在时报错
os.makedirs(folder_name, exist_ok=True)
print(f"准备保存图片到文件夹: {folder_name}")

# ==========================================
# 4. 绘图 1: 模型测试准确率对比 并保存
# ==========================================
plt.figure(figsize=(12, 7))

# 定义颜色 (使用 Seaborn 的 color_palette，看起来更高级)
colors = sns.color_palette("muted", len(models))

# 绘制柱状图，设置边缘颜色和透明度
bars = plt.bar(models, accuracies, color=colors, edgecolor='black', alpha=0.85, width=0.6)

# --- 细节优化 ---
# 添加标题和坐标轴标签 (加粗)
plt.title('Sentiment Task: Test Accuracy Comparison', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Test Accuracy (0-1)', fontsize=14, fontweight='bold')
plt.xlabel('Model', fontsize=14, fontweight='bold')

# 设置 Y 轴范围和刻度，留出顶部空间显示数值标签
plt.ylim(0, 1.0)
plt.yticks(np.arange(0, 1.1, 0.1))

# 在柱状图顶部添加具体的数值标签 (精细调整位置)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.015,
             f'{height:.3f}', # 格式化为三位小数
             ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

# 去掉顶部和右侧的边框线，让图表更清爽
sns.despine(top=True, right=True)

# 调整布局以适应所有元素
plt.tight_layout()

# --- 保存图片到指定文件夹 ---
# 构建包含文件夹路径的文件名 (使用 os.path.join 确保跨平台兼容性)
accuracy_plot_path = os.path.join(folder_name, 'sentiment_accuracy_comparison.png')
plt.savefig(accuracy_plot_path, dpi=300)
print(f"准确率对比图已保存至: {accuracy_plot_path}")

# 绘制完成后关闭 figure，释放内存，避免下一张图受影响
plt.close()

# ==========================================
# 5. 绘图 2: 各情感类别 F1-Score 热力图 并保存
# ==========================================
plt.figure(figsize=(10, 7))

# 使用 Seaborn 的 heatmap 绘制热力图
# cmap: 颜色映射 ("Blues" 渐变，看起来很舒服)
# annot: 是否在格子内显示数值, fmt: 数值格式
# vmin/vmax: 颜色映射的最小/最大值
ax = sns.heatmap(f1_data, annot=True, fmt=".2f", cmap='Blues',
                 xticklabels=sentiment_labels, yticklabels=models,
                 vmin=0.6, vmax=1.0, # 聚焦在 F1 所在的区间，让颜色对比更明显
                 linewidths=1, linecolor='white', annot_kws={"size": 13, "fontweight": "bold"}) # 细化格子边框和字体

# --- 细节优化 ---
# 添加标题和坐标轴标签 (加粗)
plt.title('Sentiment Task: F1-Score across Sentiment Classes', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('True Sentiment Class', fontsize=14, fontweight='bold')
plt.ylabel('Model', fontsize=14, fontweight='bold')

# 旋转 X 轴和 Y 轴标签，确保不重叠且易读
plt.xticks(rotation=15, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)

# 调整布局
plt.tight_layout()



# --- 保存图片到指定文件夹 ---
# 构建包含文件夹路径的文件名
heatmap_plot_path = os.path.join(folder_name, 'sentiment_f1_heatmap.png')
plt.savefig(heatmap_plot_path, dpi=300)
print(f"F1-Score 热力图已保存至: {heatmap_plot_path}")

# 绘制完成后关闭 figure
plt.close()

print("\n情感分析任务的可视化绘图与保存已完成！")