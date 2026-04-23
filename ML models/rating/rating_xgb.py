import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack
from xgboost import XGBClassifier

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
# 3. TF-IDF（关键优化）
# =========================
print("TF-IDF processing...")

tfidf = TfidfVectorizer(
    max_features=10000,       # 平衡速度与效果
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
# 5. 融合特征
# =========================
X_train_final = hstack([X_train_tfidf, X_train_num])
X_test_final = hstack([X_test_tfidf, X_test_num])

# =========================
# 6. 标签编码（XGB需要）
# =========================
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# =========================
# 7. XGBoost（高性能参数）
# =========================
print("Training XGBoost...")

model = XGBClassifier(
    n_estimators=200,        # 控制速度
    max_depth=6,             # 防过拟合
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',      #  关键加速
    eval_metric='mlogloss',
    verbosity=1,
    n_jobs=-1
)

# 加验证集（可看到训练过程）
model.fit(
    X_train_final,
    y_train_enc,
    eval_set=[(X_test_final, y_test_enc)],
    verbose=True             # 🔥 显示进度
)

print("Training finished!")

# =========================
# 8. 预测
# =========================
y_pred_enc = model.predict(X_test_final)
y_pred = le.inverse_transform(y_pred_enc)

# =========================
# 9. 标准 Accuracy
# =========================
print("\n=== Standard Accuracy ===")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# 10. ⭐ 容忍误差 Accuracy（重点）
# =========================
tolerance_acc = np.mean(np.abs(y_pred - y_test) <= 1)

print("\n=== Tolerance Accuracy (±1 star) ===")
print("Tolerance Accuracy:", round(tolerance_acc, 4))