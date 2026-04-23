import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from tqdm import tqdm

# =========================
# 1. 数据
# =========================
df = pd.read_csv("processed_data.csv").dropna()

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
# 3. TF-IDF（降维加速）
# =========================
print("TF-IDF processing...")

tfidf = TfidfVectorizer(
    max_features=8000,        # 🔥 降维
    stop_words='english',
    ngram_range=(1, 2),       # 🔥 降复杂度
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
# 6. class_weight
# =========================
class_weights = {
    1: 2.0,
    2: 3.0,
    3: 3.5,
    4: 2.5,
    5: 1.0
}

# =========================
# 7. Random Forest（进度条版）
# =========================
print("Training Random Forest...")

n_estimators = 200
batch_size = 20

model = RandomForestClassifier(
    n_estimators=batch_size,
    max_depth=20,             # 🔥 限制深度
    min_samples_leaf=2,       # 🔥 防过拟合 + 加速
    class_weight=class_weights,
    warm_start=True,          # 🔥 支持增量训练
    n_jobs=-1,
    random_state=42
)

for i in tqdm(range(batch_size, n_estimators + 1, batch_size)):
    model.n_estimators = i
    model.fit(X_train_final, y_train)

print("Training finished!")

# =========================
# 8. 预测
# =========================
y_pred = model.predict(X_test_final)

# =========================
# 9. 评估
# =========================
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# 10. 容忍误差
# =========================
tolerance_acc = np.mean(np.abs(y_pred - y_test) <= 1)

print("\nTolerance Accuracy (±1 star):", round(tolerance_acc, 4))