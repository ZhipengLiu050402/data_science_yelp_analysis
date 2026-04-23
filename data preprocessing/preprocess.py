import json
import pandas as pd
import re
from tqdm import tqdm
import random

# =========================
# 1. 读取 JSON
# =========================
def load_json_lines(file_path, nrows=None):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if nrows and i >= nrows:
                break
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                # 跳过损坏的 JSON 行
                continue
    return pd.DataFrame(data)


# =========================
# 2. 随机采样
# =========================
def sample_review(file_path, sample_size=1000000):
    print("Sampling reviews...")

    # 先统计总行数（快速遍历一遍，仅读取行数）
    print("Counting lines...")
    with open(file_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    # 生成随机行索引（不放回抽样）
    sample_indices = set(random.sample(range(total_lines), min(sample_size, total_lines)))

    # 第二次遍历，仅保留选中行
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, total=total_lines, desc="Reading sampled lines")):
            if i in sample_indices:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    return pd.DataFrame(data)


# =========================
# 3. 文本清洗
# =========================
def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# =========================
# 4. 情感标签
# =========================
def get_sentiment(stars):
    if stars >= 4:
        return "positive"
    elif stars == 3:
        return "neutral"
    else:
        return "negative"


# =========================
# 5. 主流程
# =========================
def preprocess():

    # ========= 文件路径 =========
    review_path = "yelp_academic_dataset_review.json"
    user_path = "yelp_academic_dataset_user.json"
    business_path = "yelp_academic_dataset_business.json"
    tip_path = "yelp_academic_dataset_tip.json"
    checkin_path = "yelp_academic_dataset_checkin.json"

    # ========= Step 1: 采样 review =========
    review_df = sample_review(review_path, 1000000)

    # ========= Step 2: 清洗 review =========
    print("Cleaning reviews...")

    review_df = review_df.dropna(subset=['text'])

    review_df['text_clean'] = review_df['text'].apply(clean_text)

    review_df = review_df[review_df['text_clean'].str.len() > 20]

    review_df['sentiment'] = review_df['stars'].apply(get_sentiment)

    # 文本长度
    review_df['text_length'] = review_df['text_clean'].apply(lambda x: len(x.split()))

    # 时间特征
    review_df['date'] = pd.to_datetime(review_df['date'])
    review_df['year'] = review_df['date'].dt.year
    review_df['month'] = review_df['date'].dt.month
    review_df['hour'] = review_df['date'].dt.hour

    # ========= Step 3: 提取 ID =========
    user_ids = set(review_df['user_id'])
    business_ids = set(review_df['business_id'])

    # ========= Step 4: 加载 user =========
    print("Loading users...")
    user_df = load_json_lines(user_path)

    user_df = user_df[user_df['user_id'].isin(user_ids)]

    user_df = user_df[['user_id', 'review_count', 'average_stars']]

    user_df.rename(columns={
        'review_count': 'user_review_count',
        'average_stars': 'user_avg_stars'
    }, inplace=True)

    # ========= Step 5: 加载 business =========
    print("Loading businesses...")
    business_df = load_json_lines(business_path)

    business_df = business_df[business_df['business_id'].isin(business_ids)]

    business_df = business_df[[
        'business_id',
        'stars',
        'review_count',
        'categories'
    ]]

    business_df.rename(columns={
        'stars': 'business_stars',
        'review_count': 'business_review_count'
    }, inplace=True)

    # 修正 categories 字段处理：可能是列表或字符串
    def extract_main_category(cat):
        if isinstance(cat, list):
            if len(cat) > 0:
                return cat[0]
            else:
                return ""
        elif isinstance(cat, str):
            return cat.split(',')[0].strip()
        else:
            return ""

    business_df['main_category'] = business_df['categories'].apply(extract_main_category)

    # ========= Step 6: checkin =========
    print("Processing checkin...")
    checkin_df = load_json_lines(checkin_path)

    checkin_df = checkin_df[checkin_df['business_id'].isin(business_ids)]

    # 修正：处理 date 字段可能为空或非字符串的情况
    def safe_checkin_count(date_val):
        if isinstance(date_val, str):
            return len(date_val.split(','))
        else:
            return 0

    checkin_df['checkin_count'] = checkin_df['date'].apply(safe_checkin_count)

    # ========= Step 7: tip =========
    print("Processing tips...")
    tip_df = load_json_lines(tip_path)

    tip_df = tip_df[tip_df['business_id'].isin(business_ids)]

    tip_count = tip_df.groupby('business_id').size().reset_index(name='tip_count')

    # ========= Step 8: merge =========
    print("Merging data...")

    df = review_df.merge(user_df, on='user_id', how='left')
    df = df.merge(business_df[['business_id', 'business_stars', 'business_review_count', 'main_category']],
                  on='business_id', how='left')
    df = df.merge(checkin_df[['business_id', 'checkin_count']], on='business_id', how='left')
    df = df.merge(tip_count, on='business_id', how='left')

    # ========= Step 9: 清理缺失值 =========
    # 修正：对衍生字段填充默认值，而非直接 dropna 丢弃所有含 NaN 的行
    df['checkin_count'] = df['checkin_count'].fillna(0).astype(int)
    df['tip_count'] = df['tip_count'].fillna(0).astype(int)

    # 仅删除核心字段缺失的行（这些本应有值）
    df = df.dropna(subset=['text_clean', 'stars', 'user_id', 'business_id'])

    # ========= Step 10: 最终字段 =========
    df_final = df[[
        'text_clean',
        'stars',
        'sentiment',
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

    print("Final shape:", df_final.shape)

    # ========= Step 11: 保存 =========
    df_final.to_csv("processed_data.csv", index=False)
    print("Saved to processed_data.csv")


# =========================
# 运行
# =========================
if __name__ == "__main__":
    preprocess()