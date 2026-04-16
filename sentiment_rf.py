import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack, vstack
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import numpy as np

# =========================
# 1. 读取数据
# =========================
print("Loading data...")
df = pd.read_csv("processed_data.csv")

df = df.sample(200000, random_state=42)
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
    X_text, numeric_features, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 4. TF-IDF
# =========================
print("TF-IDF processing...")

tfidf = TfidfVectorizer(
    max_features=15000,
    stop_words='english',
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)

print("TF-IDF done.")

# =========================
# 5. 特征融合
# =========================
X_train_final = hstack([X_train_tfidf, X_train_num])
X_test_final = hstack([X_test_tfidf, X_test_num])

# =========================
# 6. Random Forest
# =========================
print("Start training Random Forest...")

n_estimators = 300
batch_size = 20   # 每次训练多少棵树

model = RandomForestClassifier(
    n_estimators=batch_size,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    warm_start=True,   #  关键：允许逐步增加树
    n_jobs=-1,
    random_state=42
)

# 分批训练
for i in tqdm(range(batch_size, n_estimators + 1, batch_size)):
    model.n_estimators = i
    model.fit(X_train_final, y_train)

print("Training finished!")

# =========================
# 7. 预测
# =========================
y_pred = model.predict(X_test_final)

# =========================
# 8. 评估
# =========================
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))