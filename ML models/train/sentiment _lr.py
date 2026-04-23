import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

# =========================
# 1. 读取数据
# =========================
import pandas as pd

df = pd.read_csv("processed_data.csv")
# 清理数据
df = df.dropna()
print("Data shape:", df.shape)

# =========================
# 2. 特征 & 标签
# =========================
X_text = df['text_clean']
y = df['sentiment']

# 数值特征
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
# 3. 划分数据集
# =========================
X_train_text, X_test_text, X_train_num, X_test_num, y_train, y_test = train_test_split(
    X_text, numeric_features, y, test_size=0.2, random_state=42
)

# =========================
# 4. TF-IDF
# =========================
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words='english'
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
# 6. 特征融合（关键点）
# =========================
X_train_final = hstack([X_train_tfidf, X_train_num_scaled])
X_test_final = hstack([X_test_tfidf, X_test_num_scaled])

# =========================
# 7. 训练模型
# =========================
model = LogisticRegression(max_iter=1000)

model.fit(X_train_final, y_train)

# =========================
# 8. 预测
# =========================
y_pred = model.predict(X_test_final)

# =========================
# 9. 评估
# =========================
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))