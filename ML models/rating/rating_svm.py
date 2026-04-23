import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

# =========================
# 1. 数据
# =========================
df = pd.read_csv("processed_data.csv").dropna()

print("Data shape:", df.shape)

X_text = df['text_clean']
y = df['stars']

numeric_features = df[[
    'text_length', 'year', 'month', 'hour',
    'user_review_count', 'user_avg_stars',
    'business_stars', 'business_review_count',
    'checkin_count', 'tip_count'
]]

# =========================
# 2. 划分
# =========================
X_train_text, X_test_text, X_train_num, X_test_num, y_train, y_test = train_test_split(
    X_text, numeric_features, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 3. TF-IDF（平衡速度+效果）
# =========================
print("TF-IDF processing...")

tfidf = TfidfVectorizer(
    max_features=10000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.9
)

X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)

# =========================
# 4. 数值特征
# =========================
scaler = StandardScaler()

X_train_num = scaler.fit_transform(X_train_num)
X_test_num = scaler.transform(X_test_num)

# =========================
# 5. 融合
# =========================
X_train_final = hstack([X_train_tfidf, X_train_num])
X_test_final = hstack([X_test_tfidf, X_test_num])

# =========================
# 6. 类别权重（优化2-4星）
# =========================
class_weights = {
    1: 2.0,
    2: 3.0,
    3: 3.5,   #  中间类重点
    4: 2.5,
    5: 1.0
}

# =========================
# 7. SVM（LinearSVC）
# =========================
print("Training Linear SVM...")

model = LinearSVC(
    class_weight=class_weights,
    max_iter=2000
)

model.fit(X_train_final, y_train)

print("Training finished!")

# =========================
# 8. 预测
# =========================
y_pred = model.predict(X_test_final)

# =========================
# 9. 标准 Accuracy
# =========================
print("\n=== Standard Accuracy ===")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# 10. 容忍误差 Accuracy
# =========================
tolerance_acc = np.mean(np.abs(y_pred - y_test) <= 1)

print("\n=== Tolerance Accuracy (±1 star) ===")
print("Tolerance Accuracy:", round(tolerance_acc, 4))