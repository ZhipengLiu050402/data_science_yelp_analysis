import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

# =========================
# 1. 读取数据
# =========================
df = pd.read_csv("processed_data.csv")
df = df.dropna()

print("Data shape:", df.shape)

# =========================
# 2. 特征 & 标签
# =========================
X_text = df['text_clean']
y = df['stars']   # ⭐ 1~5 作为分类标签

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
    X_text, numeric_features, y,
    test_size=0.2,
    random_state=42,
    stratify=y   # 很关键（防止类别不均衡）
)

# =========================
# 4. TF-IDF
# =========================
print("TF-IDF processing...")

tfidf = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 3),
    min_df=5,
    max_df=0.9
)

X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)

# =========================
# 5. 数值特征
# =========================
scaler = StandardScaler()

X_train_num_scaled = scaler.fit_transform(X_train_num)
X_test_num_scaled = scaler.transform(X_test_num)

# =========================
# 6. 特征融合
# =========================
X_train_final = hstack([X_train_tfidf, X_train_num_scaled])
X_test_final = hstack([X_test_tfidf, X_test_num_scaled])

# =========================
# 7. Logistic Regression（多分类）
# =========================
print("Training Logistic Regression for rating prediction...")

class_weights = {
    1: 2.0,
    2: 3.0,
    3: 3.5,
    4: 2.5,
    5: 1.0
}

model = LogisticRegression(
    max_iter=1000,
    solver='lbfgs',
    class_weight=class_weights
)

model.fit(X_train_final, y_train)

print("Training finished!")

# =========================
# 8. 预测
# =========================
y_pred = model.predict(X_test_final)

# =========================
# 9. 评估
# =========================
tolerance_acc = np.mean(np.abs(y_pred - y_test) <= 1)
print("Tolerance Accuracy:", tolerance_acc)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))