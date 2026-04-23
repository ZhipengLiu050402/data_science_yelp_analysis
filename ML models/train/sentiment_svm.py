import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import joblib
import warnings
warnings.filterwarnings('ignore')

# =========================
# 1. 读取数据
# =========================
df = pd.read_csv("processed_data.csv")

# 可选：减少数据量（强烈建议先用20万调试）
# 如果希望用全量数据，注释下一行即可
# df = df.sample(200000, random_state=42)

df = df.dropna()
print("Data shape:", df.shape)

# =========================
# 2. 特征 & 标签
# =========================
X_text = df['text_clean']
y = df['sentiment']

numeric_features = df[[
    'text_length',
    'year',
    'month',
    'hour',
    'user_review_count',
    'user_avg_stars',
    'business_stars',
    'business_review_count',
    'checkin_count',
    'tip_count'
]]

# =========================
# 3. 划分数据
# =========================
X_train_text, X_test_text, X_train_num, X_test_num, y_train, y_test = train_test_split(
    X_text, numeric_features, y, test_size=0.2, random_state=42, stratify=y  # 分层抽样，保持类别比例
)

# =========================
# 4. TF-IDF 向量化
# =========================
tfidf = TfidfVectorizer(
    max_features=8000,
    stop_words='english',
    ngram_range=(1, 2)  # 加入二元词组，捕获更多上下文
)

X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)

# =========================
# 5. 数值特征标准化
# =========================
scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num)
X_test_num_scaled = scaler.transform(X_test_num)

# =========================
# 6. 特征融合
# =========================
from scipy.sparse import csr_matrix
X_train_final = hstack([X_train_tfidf, X_train_num_scaled])
X_test_final = hstack([X_test_tfidf, X_test_num_scaled])

print(f"训练集特征维度: {X_train_final.shape}")
print(f"测试集特征维度: {X_test_final.shape}")

# =========================
# 7. SVM 模型（优化参数）
# =========================
model = LinearSVC(
    max_iter=2000,               # 增加迭代次数，确保收敛
    class_weight='balanced',     # 自动处理类别不平衡
    dual=False,                  # 样本数 > 特征数时设为 False
    random_state=42,
    verbose=1                    # 显示训练进度
)

print("开始训练 SVM 模型...")
model.fit(X_train_final, y_train)
print("训练完成！")

# =========================
# 8. 预测
# =========================
y_pred = model.predict(X_test_final)

# =========================
# 9. 评估
# =========================
print("\n" + "="*50)
print("测试集准确率: {:.4f}".format(accuracy_score(y_test, y_pred)))
print("="*50)
print("\n分类报告:\n")
print(classification_report(y_test, y_pred))

# 混淆矩阵
print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred))

# =========================
# 10. 保存模型与预处理组件
# =========================
joblib.dump(model, 'svm_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(scaler, 'numerical_scaler.pkl')
print("\n模型已保存至 svm_model.pkl, tfidf_vectorizer.pkl, numerical_scaler.pkl")