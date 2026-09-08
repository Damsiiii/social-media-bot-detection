import os
import json
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_instagram_data(data_dir="data/instagram"):
    """
    Loads and cleans Instagram dataset from train.csv and test.csv or single csv.
    Returns cleaned DataFrame.
    """
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    if os.path.exists(train_path) and os.path.exists(test_path):
        tr = pd.read_csv(train_path)
        te = pd.read_csv(test_path)
        df = pd.concat([tr, te], ignore_index=True)
    else:
        # Fallback to any csv
        csvs = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if not csvs:
            raise FileNotFoundError(f"No CSV found in {data_dir}")
        df = pd.read_csv(os.path.join(data_dir, csvs[0]))
    
    df.columns = [c.strip() for c in df.columns]
    df = df.drop_duplicates().reset_index(drop=True)
    
    target_col = 'fake' if 'fake' in df.columns else df.columns[-1]
    df[target_col] = df[target_col].astype(int)
    
    # Impute missing values if any
    for col in df.columns:
        if col != target_col and df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    return df, target_col

def load_twitter_data(data_dir="data/twitter"):
    """
    Loads raw Gilani-2017 TSV and JSON, resolves conflicting labels transparently,
    merges user profile metadata, and performs feature engineering.
    """
    tsv_path = os.path.join(data_dir, "gilani-2017.tsv")
    json_path = os.path.join(data_dir, "gilani-2017_tweets.json")
    
    if not os.path.exists(tsv_path) or not os.path.exists(json_path):
        clean_cand = os.path.join(data_dir, "clean_twitter_dataset.csv")
        if os.path.exists(clean_cand):
            df = pd.read_csv(clean_cand)
            return df, 'target'
        raise FileNotFoundError(f"Raw Twitter files not found in {data_dir}")
        
    tsv_df = pd.read_csv(tsv_path, sep='\t', header=None, names=['user_id', 'raw_label'])
    
    # Identify conflicting IDs
    dup_rows = tsv_df[tsv_df['user_id'].duplicated(keep=False)]
    conflict_group = dup_rows.groupby('user_id')['raw_label'].nunique()
    conflicting_ids = conflict_group[conflict_group > 1].index.tolist()
    
    tsv_clean = tsv_df[~tsv_df['user_id'].isin(conflicting_ids)].copy()
    tsv_clean = tsv_clean.drop_duplicates(subset=['user_id']).reset_index(drop=True)
    tsv_clean['target'] = tsv_clean['raw_label'].str.strip().str.lower().map({'human': 0, 'bot': 1})
    tsv_clean['user_id'] = tsv_clean['user_id'].astype(np.int64)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        
    records = []
    for item in json_data:
        if isinstance(item, dict):
            if 'user' in item and isinstance(item['user'], dict):
                records.append(item['user'])
            elif 'id' in item and 'followers_count' in item:
                records.append(item)
                
    df_u = pd.DataFrame(records).drop_duplicates(subset=['id']).reset_index(drop=True)
    df_u['id'] = df_u['id'].astype(np.int64)
    
    merged = pd.merge(df_u, tsv_clean[['user_id', 'target']], left_on='id', right_on='user_id', how='inner')
    
    # Feature engineering
    merged['followers_friends_ratio'] = (merged['followers_count'] + 1.0) / (merged['friends_count'] + 1.0)
    merged['favourites_statuses_ratio'] = (merged['favourites_count'] + 1.0) / (merged['statuses_count'] + 1.0)
    merged['statuses_followers_ratio'] = (merged['statuses_count'] + 1.0) / (merged['followers_count'] + 1.0)
    
    merged['description'] = merged['description'].fillna('').astype(str)
    merged['name'] = merged['name'].fillna('').astype(str)
    merged['screen_name'] = merged['screen_name'].fillna('').astype(str)
    
    merged['description_length'] = merged['description'].str.len()
    merged['username_length'] = merged['screen_name'].str.len()
    merged['username_digit_count'] = merged['screen_name'].apply(lambda s: sum(c.isdigit() for c in s))
    merged['username_digit_ratio'] = merged['username_digit_count'] / merged['username_length'].replace(0, 1)
    merged['fullname_length'] = merged['name'].str.len()
    merged['fullname_word_count'] = merged['name'].apply(lambda s: len(s.split()))
    merged['name_equals_username'] = (merged['name'].str.lower().str.strip() == merged['screen_name'].str.lower().str.strip()).astype(int)
    
    for b in ['verified', 'protected', 'default_profile', 'default_profile_image']:
        if b in merged.columns:
            merged[b] = merged[b].fillna(False).astype(int)
            
    return merged, 'target'

def clean_text_for_nlp(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
