import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.sparse import hstack
from xgboost import XGBClassifier

# =========================
# 1. 读取数据
# =========================
df = pd.read_csv("processed_data.csv")

df = df.sample(200000, random_state=42)
df = df.dropna()

print("Data shape:", df.shape)

# =========================
# 2. 标签编码
# =========================
le = LabelEncoder()
df['sentiment_label'] = le.fit_transform(df['sentiment'])

# =========================
# 3. 特征
# =========================
X_text = df['text_clean']
y = df['sentiment_label']

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
# 4. 划分数据
# =========================
X_train_text, X_test_text, X_train_num, X_test_num, y_train, y_test = train_test_split(
    X_text, numeric_features, y, test_size=0.2, random_state=42
)

# =========================
# 5. TF-IDF
# =========================
tfidf = TfidfVectorizer(
    max_features=8000,
    stop_words='english'
)

X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)

# =========================
# 6. 数值特征
# =========================
scaler = StandardScaler()

X_train_num_scaled = scaler.fit_transform(X_train_num)
X_test_num_scaled = scaler.transform(X_test_num)

# =========================
# 7. 特征融合
# =========================
X_train_final = hstack([X_train_tfidf, X_train_num_scaled])
X_test_final = hstack([X_test_tfidf, X_test_num_scaled])

# =========================
# 8. XGBoost（升级版）
# =========================
model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    verbosity=1  # 控制日志
)

print("Start training XGBoost...")

model.fit(
    X_train_final,
    y_train,
    eval_set=[(X_test_final, y_test)],  # 👈 关键：显示每轮结果
    verbose=True  # 👈 显示训练进度
)

print("Training finished!")

# =========================
# 9. 预测
# =========================
y_pred = model.predict(X_test_final)

# =========================
# 10. 评估
# =========================
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))