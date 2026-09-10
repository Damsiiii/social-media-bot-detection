import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

try:
    import joblib
except Exception:
    joblib = None

try:
    from sklearn.preprocessing import StandardScaler
except Exception:
    StandardScaler = None

TRANSFORMERS_AVAILABLE = (
    importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None
)
torch = None
AutoTokenizer = None
AutoModelForSequenceClassification = None


def _load_transformer_components():
    if not TRANSFORMERS_AVAILABLE:
        return None, None, None
    try:
        import torch as torch_module
        from transformers import AutoModelForSequenceClassification as model_cls
        from transformers import AutoTokenizer as tokenizer_cls

        return torch_module, tokenizer_cls, model_cls
    except Exception:
        return None, None, None

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
GRAPHS_DIR = BASE_DIR / "graphs"

INSTAGRAM_MODELS_DIR = MODELS_DIR / "instagram"
TWITTER_MODELS_DIR = MODELS_DIR / "twitter"
FAKE_GRAPH_DIR = GRAPHS_DIR / "fake_comment_detection"

COMMENT_MODEL_REPO = "Tharanya06/Bot_Deduction"

INSTAGRAM_FEATURES = [
    "profile pic",
    "nums/length username",
    "fullname words",
    "nums/length fullname",
    "name==username",
    "description length",
    "external URL",
    "private",
    "#posts",
    "#followers",
    "#follows",
]

TWITTER_FEATURES = [
    "followers_count",
    "friends_count",
    "statuses_count",
    "favourites_count",
    "listed_count",
    "followers_friends_ratio",
    "favourites_statuses_ratio",
    "statuses_followers_ratio",
    "description_length",
    "username_length",
    "username_digit_count",
    "username_digit_ratio",
    "fullname_length",
    "fullname_word_count",
    "verified",
    "protected",
    "default_profile",
    "default_profile_image",
    "name_equals_username",
]

st.set_page_config(page_title="Social Media Bot Deduction", page_icon="🛡️", layout="wide")


@st.cache_data
def read_csv(path):
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_instagram_data():
    train = pd.read_csv(DATA_DIR / "instagram" / "train.csv")
    test = pd.read_csv(DATA_DIR / "instagram" / "test.csv")
    df = pd.concat([train, test], ignore_index=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df


@st.cache_data
def load_twitter_data():
    tsv_path = DATA_DIR / "twitter" / "gilani-2017.tsv"
    json_path = DATA_DIR / "twitter" / "gilani-2017_tweets.json"

    tsv_df = pd.read_csv(tsv_path, sep="\t", header=None, names=["user_id", "raw_label"])
    dup_rows = tsv_df[tsv_df["user_id"].duplicated(keep=False)]
    if not dup_rows.empty:
        conflict_ids = dup_rows.groupby("user_id")["raw_label"].nunique()
        conflict_ids = conflict_ids[conflict_ids > 1].index.tolist()
        tsv_clean = tsv_df[~tsv_df["user_id"].isin(conflict_ids)].drop_duplicates(subset=["user_id"])
    else:
        tsv_clean = tsv_df.drop_duplicates(subset=["user_id"])

    tsv_clean["target"] = tsv_clean["raw_label"].str.strip().str.lower().map({"human": 0, "bot": 1})
    tsv_clean["user_id"] = tsv_clean["user_id"].astype(np.int64)

    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    records = []
    for item in json_data:
        if isinstance(item, dict) and "user" in item and isinstance(item["user"], dict):
            records.append(item["user"])

    df_u = pd.DataFrame(records).drop_duplicates(subset=["id"])
    df_u["id"] = df_u["id"].astype(np.int64)
    merged = pd.merge(df_u, tsv_clean[["user_id", "target"]], left_on="id", right_on="user_id", how="inner")

    merged["followers_friends_ratio"] = (merged["followers_count"] + 1.0) / (merged["friends_count"] + 1.0)
    merged["favourites_statuses_ratio"] = (merged["favourites_count"] + 1.0) / (merged["statuses_count"] + 1.0)
    merged["statuses_followers_ratio"] = (merged["statuses_count"] + 1.0) / (merged["followers_count"] + 1.0)
    merged["description"] = merged["description"].fillna("").astype(str)
    merged["name"] = merged["name"].fillna("").astype(str)
    merged["screen_name"] = merged["screen_name"].fillna("").astype(str)
    merged["description_length"] = merged["description"].str.len()
    merged["username_length"] = merged["screen_name"].str.len()
    merged["username_digit_count"] = merged["screen_name"].apply(lambda s: sum(c.isdigit() for c in s))
    merged["username_digit_ratio"] = merged["username_digit_count"] / merged["username_length"].replace(0, 1)
    merged["fullname_length"] = merged["name"].str.len()
    merged["fullname_word_count"] = merged["name"].apply(lambda s: len(s.split()))
    merged["name_equals_username"] = (
        merged["name"].str.lower().str.strip() == merged["screen_name"].str.lower().str.strip()
    ).astype(int)
    for b in ["verified", "protected", "default_profile", "default_profile_image"]:
        if b in merged.columns:
            merged[b] = merged[b].fillna(False).astype(int)

    return merged[TWITTER_FEATURES + ["target"]].rename(columns={"target": "fake"})


@st.cache_resource
def load_instagram_models():
    bundle = {}
    scaler_path = INSTAGRAM_MODELS_DIR / "instagram_standard_scaler.pkl"
    if scaler_path.exists() and joblib is not None:
        bundle["scaler"] = joblib.load(scaler_path)
    else:
        bundle["scaler"] = None

    for key, name in {
        "rf": "random_forest.pkl",
        "xgb": "xgboost.pkl",
        "lr": "logistic_regression.pkl",
    }.items():
        path = INSTAGRAM_MODELS_DIR / name
        if path.exists() and joblib is not None:
            bundle[key] = joblib.load(path)
        else:
            bundle[key] = None

    try:
        import tensorflow as tf
    except Exception:
        tf = None

    ann_path = INSTAGRAM_MODELS_DIR / "instagram_ann.keras"
    if tf is not None and ann_path.exists():
        try:
            bundle["ann"] = tf.keras.models.load_model(str(ann_path))
        except Exception:
            bundle["ann"] = None
    else:
        bundle["ann"] = None

    return bundle


@st.cache_resource
def load_twitter_models():
    bundle = {}
    for key, name in {
        "rf": "random_forest.pkl",
        "xgb": "xgboost.pkl",
        "lr": "logistic_regression.pkl",
        "nlp": "nlp_model.pkl",
        "tfidf": "tfidf_vectorizer.pkl",
    }.items():
        path = TWITTER_MODELS_DIR / name
        if path.exists() and joblib is not None:
            bundle[key] = joblib.load(path)
        else:
            bundle[key] = None

    if StandardScaler is not None:
        try:
            df = load_twitter_data()
            bundle["scaler"] = StandardScaler().fit(df[TWITTER_FEATURES])
        except Exception:
            bundle["scaler"] = None
    else:
        bundle["scaler"] = None

    return bundle


@st.cache_resource
def load_comment_model():
    torch_module, tokenizer_cls, model_cls = _load_transformer_components()
    if torch_module is None or tokenizer_cls is None or model_cls is None:
        return None, None
    try:
        tokenizer = tokenizer_cls.from_pretrained(COMMENT_MODEL_REPO)
        model = model_cls.from_pretrained(COMMENT_MODEL_REPO)
        model.eval()
        return tokenizer, model
    except Exception:
        return None, None


@st.cache_data
def get_graph_files(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(str(p) for p in folder.glob("*.png"))


def get_best_model_summary(csv_path):
    df = read_csv(csv_path)
    if df.empty:
        return {
            "model": "Result unavailable",
            "accuracy": "Result unavailable",
            "exists": False,
        }

    metric_cols = [c for c in df.columns if any(k in c.lower() for k in ["accuracy", "precision", "recall", "f1", "roc"]) ]
    if not metric_cols:
        return {"model": "Result unavailable", "accuracy": "Result unavailable", "exists": False}

    score_cols = [c for c in df.columns if "accuracy" in c.lower() or "f1" in c.lower() or "precision" in c.lower() or "recall" in c.lower()]
    best_row = df.sort_values(by=score_cols, ascending=False).iloc[0]
    acc_value = None
    for c in df.columns:
        if "accuracy" in c.lower():
            try:
                acc_value = float(best_row[c])
                break
            except Exception:
                pass
    if acc_value is None:
        return {"model": str(best_row.iloc[0]) if not df.empty else "Result unavailable", "accuracy": "Result unavailable", "exists": True}
    return {"model": str(best_row.iloc[0]), "accuracy": f"{acc_value:.4f}", "exists": True}


def build_instagram_row(
    username,
    profile_name,
    profile_pic,
    description,
    external_url,
    is_private,
    posts,
    followers,
    follows,
):
    username = str(username or "")
    profile_name = str(profile_name or "")
    username_ratio = 0.0 if not username else sum(c.isdigit() for c in username) / max(len(username), 1)
    fullname_words = len(profile_name.split()) if profile_name else 0
    fullname_num_ratio = 0.0 if not profile_name else sum(c.isdigit() for c in profile_name) / max(len(profile_name), 1)
    name_eq_user = int(profile_name.strip().lower() == username.strip().lower())
    desc_len = len(description or "")
    row = {
        "profile pic": int(profile_pic),
        "nums/length username": float(username_ratio),
        "fullname words": int(fullname_words),
        "nums/length fullname": float(fullname_num_ratio),
        "name==username": int(name_eq_user),
        "description length": int(desc_len),
        "external URL": int(external_url),
        "private": int(is_private),
        "#posts": int(posts),
        "#followers": int(followers),
        "#follows": int(follows),
    }
    return pd.DataFrame([row], columns=INSTAGRAM_FEATURES)


def build_twitter_row(
    screen_name,
    profile_name,
    description,
    followers_count,
    friends_count,
    statuses_count,
    favourites_count,
    listed_count,
    verified,
    protected,
    default_profile,
    default_profile_image,
):
    followers_friends_ratio = (followers_count + 1.0) / (friends_count + 1.0)
    favourites_statuses_ratio = (favourites_count + 1.0) / (statuses_count + 1.0)
    statuses_followers_ratio = (statuses_count + 1.0) / (followers_count + 1.0)
    username_length = len(screen_name or "")
    username_digit_count = sum(c.isdigit() for c in (screen_name or ""))
    username_digit_ratio = username_digit_count / (username_length if username_length else 1)
    fullname_length = len(profile_name or "")
    fullname_word_count = len((profile_name or "").split())
    name_equals_username = int((profile_name or "").strip().lower() == (screen_name or "").strip().lower())
    row = {
        "followers_count": float(followers_count),
        "friends_count": float(friends_count),
        "statuses_count": float(statuses_count),
        "favourites_count": float(favourites_count),
        "listed_count": float(listed_count),
        "followers_friends_ratio": float(followers_friends_ratio),
        "favourites_statuses_ratio": float(favourites_statuses_ratio),
        "statuses_followers_ratio": float(statuses_followers_ratio),
        "description_length": int(len(description or "")),
        "username_length": int(username_length),
        "username_digit_count": int(username_digit_count),
        "username_digit_ratio": float(username_digit_ratio),
        "fullname_length": int(fullname_length),
        "fullname_word_count": int(fullname_word_count),
        "verified": int(verified),
        "protected": int(protected),
        "default_profile": int(default_profile),
        "default_profile_image": int(default_profile_image),
        "name_equals_username": int(name_equals_username),
    }
    return pd.DataFrame([row], columns=TWITTER_FEATURES)


def predict_comment_text(text, tokenizer, model):
    if not text or not text.strip():
        text = " "
    if tokenizer is None or model is None:
        return "GENUINE", 0.0, 0.0
    try:
        import torch as torch_module
    except Exception:
        return "GENUINE", 0.0, 0.0

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch_module.no_grad():
        logits = model(**inputs).logits
    probs = torch_module.softmax(logits, dim=1)[0].tolist()
    id2label = model.config.id2label
    label_id = int(np.argmax(probs))
    label_name = id2label.get(label_id, id2label.get(str(label_id), "GENUINE"))
    prob_genuine = float(probs[0])
    prob_bot = float(probs[1]) if len(probs) > 1 else float(1 - prob_genuine)
    return label_name, prob_genuine, prob_bot


def add_prediction_history(platform, detection_type, profile_name, username, prediction, probability, model_name):
    history = st.session_state.setdefault("prediction_history", [])
    history.append(
        {
            "time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "platform": platform,
            "detection_type": detection_type,
            "profile_name": profile_name,
            "username": username,
            "prediction": prediction,
            "probability": probability,
            "model": model_name,
        }
    )
    st.session_state["prediction_history"] = history[-20:]


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: #0B1120; color: #dfe7f5; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .main .block-container { max-width: 1500px; }
    .sidebar .sidebar-content { background: linear-gradient(180deg, #0f172a 0%, #111827 100%); }
    .sidebar .block-container { padding-top: 1.0rem; }
    .brand-box { background: linear-gradient(135deg, rgba(99,102,241,0.18), rgba(59,130,246,0.08)); border: 1px solid rgba(99,102,241,0.35); border-radius: 18px; padding: 16px; margin-bottom: 18px; }
    .brand-title { font-size: 1.7rem; font-weight: 800; letter-spacing: -0.04em; color: #e7edf8; }
    .brand-sub { color: #b8c5db; font-size: 0.84rem; }
    .status-pill { display: inline-flex; align-items: center; gap: 8px; background: rgba(34,197,94,0.14); color: #c6f6d5; border: 1px solid rgba(34,197,94,0.5); border-radius: 999px; padding: 6px 10px; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; }
    .nav-label { color: #b8c5db; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.09em; font-weight: 700; margin: 14px 0 8px 0; }
    .section-card { background: linear-gradient(180deg, #111827 0%, #0f172a 100%); border: 1px solid rgba(148,163,184,0.18); border-radius: 20px; padding: 22px; box-shadow: 0 10px 24px rgba(15,23,42,0.28); }
    .kpi-card { background: linear-gradient(180deg, #111827 0%, #172033 100%); border: 1px solid rgba(148,163,184,0.14); border-radius: 16px; padding: 18px 18px 16px 18px; min-height: 120px; }
    .kpi-label { color: #b8c5db; font-size: 0.72rem; letter-spacing: 0.09em; text-transform: uppercase; font-weight: 700; }
    .kpi-value { color: #e7edf8; font-size: 1.8rem; font-weight: 800; margin-top: 10px; letter-spacing: -0.04em; }
    .kpi-foot { color: #b8c5db; font-size: 0.78rem; margin-top: 8px; }
    .module-card { background: linear-gradient(180deg, #111827 0%, #172033 100%); border: 1px solid rgba(148,163,184,0.16); border-radius: 18px; padding: 18px; min-height: 150px; }
    .profile-card { background: linear-gradient(180deg, #111827 0%, #172033 100%); border: 1px solid rgba(148,163,184,0.18); border-radius: 22px; padding: 22px; }
    .avatar { width: 80px; height: 80px; border-radius: 18px; background: linear-gradient(135deg, #6366F1, #3B82F6); display: flex; align-items: center; justify-content: center; font-size: 2rem; font-weight: 800; color: #edf3ff; overflow: hidden; box-shadow: 0 12px 24px rgba(99,102,241,0.24); }
    .instagram-avatar { background-image: url("data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBwgHBgkIBwgKCgkLDRYPDQwMDRsUFRAWIB0iIiAdHx8kKDQsJCYxJx8fLT0tMTU3Ojo6Iys/RD84QzQ5OjcBCgoKDQwNGg8PGjclHyU3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3N//AABEIAJQAlAMBEQACEQEDEQH/xAAcAAAABwEBAAAAAAAAAAAAAAAAAQIDBAUHBgj/xABHEAABAwMBAggJCAkDBQAAAAABAAIDBAURBiExBxITQVFhgZEiVHF0k6GywdEUMjZCUnKx0hcjJDNTYmSz4RWjwiVEgoOi/8QAGwEAAQUBAQAAAAAAAAAAAAAAAwABAgQFBgf/xAA7EQABAwICBAoJBAIDAAAAAAABAAIDBBEFEiExQZEGEyJRYXGBobHRFBUyNFJyweHwFiNCUzNiJEPx/9oADAMBAAIRAxEAPwDcCcJJKsv1/t1hpPlFyn5MHYxjRl7z0Ac6s0tJLVPyRC/gESON0hs1ZZfOE671jnMtTG2+HOx2BJIR5SMDsHauppuD8EYvKcx3BX46Rjfa0rlqm+XerJNTc6yTjbwZnY7s4Wo2kgj9lgHYrbIWDU0KGZJH/Pke7yuJRbAbFZa1GM9JUSjtalgKBRmtSgFAlGDUsBQJRmtSwFC6M1qUAoEo4algIZKK1qUoolkpr3N+a5w8hTEBMWgp6Ouq4jmKqnYf5ZCPeoGJh1hDdTxO9poPYFaUOrr5RObydfJIwb2TeGD37VXkoad/8bdSpTYRRy62AHo0Lt9O6+pK97ae5xtpJicNk42Y3H/j296yanDnxjMw3C5qvwCWAZ4TmHNt+67RrgcYWasBKSSVXqO809itM1fU5IZsYwb3uO4BWKSmdUyiNv8A4ESKIyvDAsCvFzq7zXyVtfJx5X7hzMbzNaOYD/K76mp46eIRxjQFvMhDG5QoXFR8yKI0oNUS5FbGlBqgXIzY0sNQy5FaxLDVAuRmsSg1QLkZrEsBQLkZrEsBQLkYMSgFAlFDbI0ykgkkiTXToFRLkkSjmSR5US5Ky7/g81S9k0douEhcx2ymkd9U/YJ6OjuWRXUwP7jO3zXK45hYLTUxDSNY+vmtKBWSuSWPcK91dWXxluY48lRNGRnYZHAE9wwO9dZgcAjgMp1u8At/DKe0Wc7VxHFW3mWpxaMNUcymI0oMUS5TEaubJpi7XnDqKkdyX8aTwWd539mVTqa6Gn9s6ebagz1UFP8A5Dp5tq7Cg4LZHMDq+5sacbWwR5HecfgsmTHPgZvWa/HAPYZvVkzgvtY+fXVrvulg/wCKrHGp9jR3oXr2fY0d/mnRwZWbxu4ekZ+RR9cz8w7/ADSGP1I/i3v80P0Z2bxqv9Iz8ib1vPzDv81L9Q1Pwt3HzR/o0s/jVf6Rn5E3rafmHf5qX6jqvhbuPmj/AEa2fxqv9Iz8ib1tPzDv80v1JV/C3cfNEeDS0c1XXdr2flT+tp+Yd/mnHCSq+Fvf5qPNwZUZH6i4zt++xp/DCcYrJfS0IjeEsoPKjHf91Q3Pg8u1IC+lkhq2gbmktd3HZ61YZicTtDhZaVPwhpZDaQFveO7yXJzwS00piqInxSN3skaWkdhV0SBwuCt5kjJG5mG46E0mzKaCjmTo2vLHBzXFrmnIIOCCmJB1pnAEWK3bTFy/1ex0tafnvbiQdDgcH1hc7Mzi5C1eZ19N6LUui2DV1HSsQv8AM6rvlwndg8epkOR0cY49S7WlGSBjRzBdXSw5YWDoUDio2ZWRElBuTuTZkQRrvdCaLZWsjud2YTTHbDCdnKfzH+X8fJvw8RxIsJiiOnaVg4piXFEwwnTtPN0Bag2NkbAyNrWMaMBoGAAueJJNyuaJJNyqi4aqstuc5k9fEXt3sj8MjuViOjnk9lqvQYXVz6WMNunQquXhEsbPmiqk+7EPeQrAwuoPMrreD1Yddh2pocJVm56Wv9Gz86l6pn5x3+SJ+m6v4m7z5IfpKs3i1f6Nn50vVM/OO/yS/TdX8Td58kP0lWbxav8ARs/Ol6qm5x3+SX6bq/ibvPkh+kqzeLV/o2fnTeq5ucd/kn/TdX8Td58kG8JNlccGCuZ1ujb7nKJw2YbR+diieDlWNRae0+SmQa9sEzg01UkWeeSIgepCdQzjYgPwGuYL5b9RCvaKvpLhEJaOoinj6Y3A4+CrOY5hs4WWXNBLC7LI0g9Kgaj09Q32m5OoYGzNB5Odo8Jh946kSGd8Ru0q1Q18tG/Mw6No51jN5ttTaK+Wjq24kYdhG545nDqK2o5hI3MF6FSVUdVEJY9R7uhQcqV1ashlNmSWg6BvYobLJBI/GKhxaOohv+Vm1TbyXXK41RGaoDwNn1K4Ko8Opld0vJ9a6lps0BbUcdmgdCQGp8yII1caVtH+sXympXfuuNx5fuDae/d2qpWVPEwl417OtVMRqPRKZ0m3UOsrco2tjjaxjQ1rRhoG4BckSSbleeEkm5WV611bNXVEtBbpDHRsJa97Tgynn2/Z/Fb9DQhjRI8afBdrhGEMiYJphd57vuuNWouhCCSSCSSCSdEokpIihkpIc6GSnVha7Hc7vxjbqOSYM+c7Ia0dpwFXlnZH7ZsqlTX01NbjnWv+aglujvGmLgx72TUdQNoOdjx2bHBCLop286gHUmIxEAhzfD6ha1pHUMV/t3K8UMqIzxZoxzHpHUVkTwmJ1lw2JYe6imya2nUVV8JdnFdZDWxsHL0h4+cb4/rD39iJSyZH22FXOD9YYanijqdo7dnksiWldd6NKJK6dWdsl4kDh/P7gq8mtU6hl3KI7efKt+6MBYJKbMpWWh8E1I10twrCAXBrYmnqO0/gO5Y+LSXysC5ThLKQI4us/neus1jXG3acrZ2O4r+JxGHoLtnvWbSszzNBWFhcHH1jGHVe+5Yguozr0iyCWdJBPnSRtyXAAEknAA50s4AuUxIAuVdUekr9WsD4LdKGHnkLWe0Qqr66BuguWdJi9FEbOkHZc+ClTaE1FG3LaNknUyZnvIQfWEB2oLceoXa327CqS4Wyutrg2vpJYCTsL2kA+Q7iitmZJ7JutKCphnF43AqGkSjrdtLwU0GnqBlLjkjC1wI5yRknvyuemcTIbrzPEXvfVSGTXcqo4S4YH6VnklDeUjkYYid4cXAHHYSp0pIl0K9wfc4VzQ3UQb7vNcTwaVxpdTMhLsR1UboyOkjaPw9at1Yuy/Muk4Q04koy7a03+i12qgZU0k0EjQWSxljgecEYWaDYrhI3mN4eNYN154njMM0kR3scWnsOFqh9xderRuzNDhtTeUsyIpVK/ixkdaG46UCRtymyVsmRSQUOMSWpcEzf+jVj+mqx3Nb8VjYg68g6lxXCU/8AJYP9fqVK4UZOJpnH26hjfxPuQ6L/AC3QeDrb1t+YFZFlbHGLu0eUuMSVhYrPV3uubS0bdu+SR3zY29JUJKhsYzFVKysipIuMk7BtK13T2l7dY4mugjElTjwp3jwj5OgeRY81TJKdJ0Lg63E56x3KNm8w1fdXgcFXWcibKx2eK4HG/BylpTkEa0mohiqonQzxskjcMOY9uQexOCQbhOx7mODmmxWb6y0MKaOSvsrCYm7ZKYZJaOct6fItGnrCeTIutwrHS8iGpOnYfPzVBp7WFysUPyeLk56bOWxSD5p58EeVFmp2yG+1albg8FY7O64dzjb1qLqLUtwv72/K3NZCw5ZDGMNB6espo4mx6kagwuCiB4vSTtKb0jLyWp7W7pqWN7zj3qM2lhRMUbmopR0HuW88yy15kvPt/aI77cmD6tXKO55V9juSF6nQm9LGf9R4KAnzK1ZOxOw3tUSVBw0o8rRMiayLKGZE9lq/BL9Hqrzx3sMWbVG7x1Lh+EvvbflHiUOFr6PUvnrfYempnZXpcGfe3fKfELJ8q9xq7iyAySAASTsAHOm41MSALlbho+xR2O0xwlo+UyDjzv6XdHkG5Z8shkddeb4pXOrKgu/iNA6vuo+sdVQ6dgEcYbLWyjMUROxo+07q/FKOPOehEwvC31ziToYNZ+gWT3W93G7SF9fVySA/Uzhg8jRsV9gYzUF3VLQ09KLRNt07d6gRvdG9skbi17doc04I7VPPdWnNDmkEXXY6Y17W0EzIbtI6poycF52yR9fWFWlha7S3WuexHAIpml8Ayu7j5LV4ZYp4WSwva9j28ZrmnIIPOqRFta4h7HNJa4WIWQ8IthbaLs2opmcWlq8ua0DYx4+cPf3rQp5i5tiu9wGvNTBkeeU3vGzyXIo5K3lY6b+kdq89h9sIEp5BVTEPdJfld4Fb/wAyzl5cvP8AqX6R3Xz2b2yrbTyQvUsP9zi+VvgFWpsyuo2nYldRKWrBkUUEIyJLV+CPbp6q89d7DFWkNyuH4T+9t+UeJR8Lf0dpfPW+w9Ra7KUuDHvbvlPiFkyJxq7lXeiaRlbqu3QyDLBIZCMfZaXe5MZLiyzMYkMVFI4a7W3m31W5uHWhLzZYDf7k+7XmqrXuLmyPPJ9TBsaO5HY8AWXqFDSimp2xDYNPXtVeiCRXEMp+MSQyp5k1lq3BTcn1VpnoZHEmkeOLnmY7OB3gqrMOVdcPwkphHUNlaPa8R+BTuE6kbUaVmlIy6nkY9uzpcGn1OShNnqvwflLK5rdjgR3X+ixpXcy9BCsNN/SK1eeQ+2EOU8gqpiHukvyu8CvQHMqK8uXn/U+zUl188m9sqwDyQvUsO9zi+VvgqtRJV1GEyScUTIoIZQjIktY4Ivo7Veeu9hiTXZlw3Cb3tnyjxKHC59HqTz1v9uRQldlAS4Me9v8AlPiFk+ELjV3K6Xg5k5LWNCHYxI2Ro8vEJ9ykyS7gFjY83NQPtst4raJRxmFucZBGVYXngNjdec3MdC90Uo4r2EtcOgjYUESL1trg4Bw1H6olMSKSCIHpkSmHpLRuB+J/HukxB4n6pgPMT4RPds70zzdcjwpeP2m7dJ8F1PCBKItIXEkb2NaPKXtHvTM9oLFwVuavj7TuBWIK1dekqw039IrV57D7YTPPJKp4h7pL8rvAr0BzKovLlgGp/pJdfPJfbKstHJXqWG+5xfK3wVXhNZXUMJWSToBPMst0qhdKEZQDMoly1bglGLBVj+td/bjVqldmaetcPwmN6tny/VyPhYGbDRj+ub/bkUK12VgPT9ClwaP/ACn/ACnxasv5LqWbxy7TMpFtqH26401bG0l1PK2Ti9IG8doynbPlIKDURieJ0R/kLLeKaeOrgjmhcHRyND2OHOCtxrg4XC8xexzHFjtYWVcIunJKG5yXOBhdSVLuM8gfupOfPUd/lz1KnODGcw1LtcBxESxCnf7TdXSPsuMLChCVdGHJJBCM2RSunKanmq6iOmpo3SzyHisY3eSjNeoSysiYXvNgNZW5aTsjbDZoaTIdKfDmcPrPO/sG7sRh0rzXEq01tQZdmodS5bhaurW0dNa2Hw5XcrKAdzRsHefwUm61tcGaUukdUEaBoHbr/OlZiUUOXaKx039I7V57D7YTuPJVPEPdJfld4Fb/AMyrry5ef9S/SO6+eze2VcYOSF6lh/ucXyt8Aq1Syq4lAbE1kxUtsS5N8qrl6dbEq7pVAvWm8FjeLZq0f1hP+2we5a+HOzRHrXG8I9NSw/6/Up3hNj5SxQH7FU0//Lx702KXEII5/ND4Puy1Tulp8Qs15Jc/xhXY5kXJpcYUsy7TQeohR4tdc/EBP7PI7cwk5LT0DO7uWtQVoH7T+xc5jOHGU+kRDTtH181oMsUU8TopmNex4LXNcMgjyLaIBFiuWa4tOZpsQuNunBzb6hzn2+eSjJ28THKMB6gSD2ZVR9G0m7TZdBTcIqiMAStzdx/OxV0PBe8v/aLsCzO0R0+Ce0uP4JNpSNbvzerbuFGjkRaek/ZddYNNWywh3yOD9a4eFNIeM93bzDqCstjDVhVuJVFaf3To5hqTuoL5SWKgfVVb+qOIHwpHdAUnODUOiopayXi4+08yw67XGou1wmratwMsp3Dc0cwHUFAOXpNLTMpohEzUPy6iIwcrKstNNLtSWkN3/LIj3PBUydCp4ibUcvynwW/cyEvLl5/1Ht1FdSPHJvbKuR6gvUqD3SL5W+AVcjAK2lsbkJy1RJVq2NeePkVIuToYhFyGXLQODN4FJXRc4la/vGPct7CHXjcOlcrwgH7jHdFvzerPXkJm09MQM8m9jvXj3o+JtJpnWVPBnhtWOkFZmIly2ZdkXJQh6lHOo50Zh6ks6bOugsmpa62NbDJ+00zdgY84c0dR9xWlTYrJFyXaQsqswyGo5Y5LvzYupptZWqVo5Z0sDucPjJ9YytdmK0zhcmyxZMHqmnkgHqPmn5NWWSMZNaD1NjefcinEaUfz8UNuFVjv4d481Q3fX0bWFlqpHSP/AIk/gtb/AOI2n1KvJibNUYWnTYA4m87rdA09/wD6s7u9VWXSqdU187ppTsydwHQBzBDZUFxu4rq6WGGnZkiFgq1zSDuVxkgKvApKsNcnV9oWF0+rba1ozxJC89QDSfgjXuFlY1JkoZDziy3I7ky83Xni6S8vc6yb+JO93e4q6zUF6tTNyQsbzADuUZHajJ+AZYfKpFDedKugzaV5k86VnZk41igSoFy6bQlSKW8OhccNqI+KPvDaPetTCZss5bzhYuMxcZAHj+J7l3tbTMrKSanlB4krC09q6KSMSMLDqK5mKQxPEg1hZdU0MtHUyU87cPYcbt/WuInjfC8sdrC7SKobKwPbqKSIlXLlPOlckmzKOdEYksyfOkuiThycPTToVMOUw5MyQ9SK2QogeoksCtxzI7XqFPBv2LQimVlj1BfGWFaUcl1YDrrROCmySNM14nYQHN5KDPOPrOHcB3q4zTpXI8Ja0G1M09J+gXaaouIten62q4wD2xER553kYb61MC5WBh1P6RVMj2E6eoa1gg3K41enBGjtSU2iZxoif5lJ2tV5nWcr6aEx1MrD9V7m9xXmM3Je4dJWWx+ZgPQEprEAuSJT0HGilZJGeK9jg5rugjcUmSFjg5usIT7OaWnUVpdlukdyo2yDAlAxIz7J+C7Oiq2VMYcNe3rXHVdM6nkynVsRXezU9zjHKZZK0YbI0bR1HpCasoY6ptnaDzp6arkpzydXMuZqNM10JJjayZvS04PcVzc2C1TPYs4fnOtmPFIXa9CjGzV7d9HL2NyqbsOqx/1lGFbAf5BINrrR/wBlUeiKh6HVD/rduKn6VD8Y3hJNrrfEqn0Lvgl6HU/1u3FP6VD8Y3hIdaq3xGp9C74KXolT/W7cVIVUPxjeEy+0V53UNT6F3wUhS1P9btxUxVw/GN4TT7JcXfNoKk/+ohHZSVJ/gdyIK6Aa3jemxpW8Tuw2ge3PO8hoV+KjqfhUvW1KwaXq2tHB2DMyW8zNc0HPyeEnDupzujqHetenpHN0vKz6rhFyS2mHafoF3zGMp4gyNrWxtGA0DAaAr4AXMOcXEk6SVk3CLqVl2rG2+jeHUlM4lzwdkkm7I6hnHanadK7nAcNdTMM0g5TtnMPuuNVphXRI1YYmXR6ctc1bQvkjbkCUt9Q+KaV4aVk19U2KUNPN5q91DSfJr3VMxhrn8dvWHbfxyvO8TZxdS4c+lZVBNxlMw9m5RGMWbdWS5PMj6lAlQJU2hnmo5hNTvLXjuI6CiU9VJTvD4zYqrNGyVuVw0LrKDUFPO0Nqswyc5O1p7V1FLjkEuiTknuWHNh8jNLNIVvFLFK0GN7Hg87TlbDJGPF2G6oOaWmxCXsU0yGxJJDYkkhgdCSSGzoSSQ2JJIHA2nYkkq64Xy229pNRVxh4HzGHjO7ggSVMMftOVqCiqJ/YaevZvWd6r1jV3OJ9JQtfS0jgQ4k/rJB0EjcOod6qGs4zQ3QF1eGYNHTkSSnM7uHmfyy4d7cK5E+66UFJV5hUkYVtiidS2Dg5tjYdLwyTMw6okdLt6Nw9QBVSofeQrgsdqi+tIafZAH1+qmautZqqdlXC3MkIw4DeW/wCPiuexikM0XGN1t8FXwuq4t5jdqPiuQjauQJW+Sn2MQyUMuTzWIZKESnWsUbqJKU0Fpy0kHq2KQkLTdpsonTrToqKhu6eUeR5RhW1A1SHeVAxxnW0bkv5XV+NT+kPxT+n1X9jt5UeJi+Ebgh8rq/Gp/Su+KXp9V/a7eUuJi+EbgiNXV+NVHpXfFP6fVf2O3nzT8VF8I3BNuq63xuo9K74pxX1X9jt5UhDF8I3BR5KusIOayp9K74ogrqk65DvKK2GL4RuCg1Ekz88pNI77zyU/pEjvacd6ssYwagFWTRDoR4nq4xyrqmPetOF6uRuVdKzeteB6ttKjEbVqxm4RVZ6cs018ukVHDkNPhSvH1Gc59wVrOGNuqOIVzKOAyO17Okrd6eJkMLIomhsbAGtaOYBUta80c9z3FztZTjhkblEqK5a8afcx7qigblp2uiHN5PguaxLBzcyU46x5LZpcQ0ZJd6pmMwcHYRvBXLOuDYrSLgdSeaxCJQi5OBqjdRuhxUrpXQIST3RYSSR4STIcVK6V0hzVIFSBTEjUQFFaVElaitKM0qFMxWYyrLCq6oZvWjC5W2OVXO3ataByuMdoT1msNffankqCLLQcSTO2Mj8p6eobVswusEKsr4KNmaU9Q2lbBpnT1Lp+iEFOOPI/bNMRtkPw6AiOcSuBxCvlrZc79AGocyulFUUEkkRCSSjVNBTVX76IOP2hsPeqlRRU9T/lbfxRY55I/ZKrKq008IPEdIMc2R8FhVWB0rNLSR2/ZXY6qR2tVr4WtJAJWBJRsaSASrbXkpojCquiARLpBQ8qkiSToJkkYTJkT04ThR5ERqK1RJUdrUdqiyNBKuRRgozTZP0dnp614bJJK3J+qR7wtqlo2O1k/nYhTVj4hdoC6Wh0RY2hks0ElQ7fiaQlvaBgHtW3FSxs0hY82NVhu1py9Q+uldJDDHBG2KFjY427GtYAAOxWgABoWO5znuzONynUkyCSS//Z");
    background-size: cover; background-position: center; background-repeat: no-repeat; 
    .profile-handle { font-size: 1.1rem; font-weight: 700; color: #eaf1ff; margin-top: 10px; }
    .profile-name { font-size: 0.88rem; color: #b8c5db; }
    .stats-row { display: flex; gap: 16px; margin-top: 18px; flex-wrap: wrap; }
    .stat-box { background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(148,163,184,0.12); border-radius: 12px; min-width: 100px; padding: 12px 14px; text-align: center; }
    .stat-box strong { display: block; color: #e7edf8; font-size: 1.1rem; }
    .stat-box span { color: #b8c5db; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; }
    .panel-title { font-size: 1.2rem; font-weight: 700; color: #eaf1ff; margin-bottom: 8px; }
    .subtle { color: #b8c5db; }
    .danger { color: #F87171; }
    .success { color: #4ADE80; }
    .warning { color: #FBBF24; }
    .pill { display: inline-flex; align-items: center; gap: 8px; border-radius: 999px; padding: 6px 12px; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; }
    .pill.bot { background: rgba(239,68,68,0.12); color: #FCA5A5; border: 1px solid rgba(239,68,68,0.4); }
    .pill.human { background: rgba(34,197,94,0.12); color: #BBF7D0; border: 1px solid rgba(34,197,94,0.4); }
    .pill.warn { background: rgba(245,158,11,0.12); color: #FCD34D; border: 1px solid rgba(245,158,11,0.4); }
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] { background: #111827; border: 1px solid rgba(148,163,184,0.18); color: #dfe7f5; border-radius: 10px; padding: 8px 16px; font-weight: 600; }
    .stTabs [aria-selected="true"] { background: #1E293B !important; border-color: rgba(99,102,241,0.6) !important; color: #edf3ff !important; }
    .metric-container { background: rgba(17,24,39,0.9); border-radius: 16px; border: 1px solid rgba(148,163,184,0.18); padding: 14px; }
    .small-label { color: #b8c5db; font-size: 0.72rem; letter-spacing: 0.09em; text-transform: uppercase; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


if "twitter_tab" not in st.session_state:
    st.session_state["twitter_tab"] = "Profile ML"


def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div class="brand-box">
                <div class="brand-title">🛡️ Social Media Bot Deduction</div>
                <div class="brand-sub">Spam comment detection and bot activity analysis</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="nav-label">Navigation</div>', unsafe_allow_html=True)
        pages = [
            "Overview",
            "Instagram Bot Detection",
            "Twitter/X Bot Detection",
            "Fake Comment Detection",
            "Model Performance",
            "Feature Importance",
            "Dataset & Results",
            "About",
        ]
        selection = st.radio("Page navigation", pages, index=0, label_visibility="collapsed")

        if selection == "Twitter/X Bot Detection":
            st.markdown('<div class="nav-label">Twitter Analysis</div>', unsafe_allow_html=True)
            st.session_state["twitter_tab"] = st.radio(
                "Twitter analysis view",
                ["Profile ML", "NLP", "GNN"],
                index=["Profile ML", "NLP", "GNN"].index(st.session_state.get("twitter_tab", "Profile ML")),
                label_visibility="collapsed",
            )

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="nav-label">System Status</div>', unsafe_allow_html=True)
        st.markdown('<div class="status-pill">● Detection Engine Ready</div>', unsafe_allow_html=True)
        return selection


selected = render_sidebar()


# ---------------- Overview ----------------
if selected == "Overview":
    st.title("Social Media Bot Deduction")
    st.caption("AI-powered analysis for detecting suspicious social accounts and identifying whether a comment is spam or genuine.")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Detection modules</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, title, subtitle, accent in [
        (c1, "Instagram", "Bot vs Human", "#6366F1"),
        (c2, "Twitter / X", "Bot vs Human", "#3B82F6"),
        (c3, "Fake Comments", "Fake / Suspicious vs Genuine", "#EF4444"),
    ]:
        with col:
            st.markdown(
                f"""
                <div class="module-card" style="border-top: 4px solid {accent};">
                    <div class="panel-title" style="margin-bottom: 8px;">{title}</div>
                    <div class="subtle">{subtitle}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

    insta_summary = get_best_model_summary(RESULTS_DIR / "instagram" / "instagram_model_comparison.csv")
    twitter_summary = get_best_model_summary(RESULTS_DIR / "twitter" / "twitter_ml_comparison.csv")
    fake_summary = {"model": "Result unavailable", "accuracy": "Result unavailable", "exists": False}

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Instagram Best Model</div>
                <div class="kpi-value">{insta_summary['model']}</div>
                <div class="kpi-foot">{insta_summary['accuracy'] if insta_summary['exists'] else 'Result unavailable'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_cols[1]:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Instagram Best Accuracy</div>
                <div class="kpi-value">{insta_summary['accuracy'] if insta_summary['exists'] else 'Result unavailable'}</div>
                <div class="kpi-foot">From existing result file</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_cols[2]:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Twitter/X Best Model</div>
                <div class="kpi-value">{twitter_summary['model']}</div>
                <div class="kpi-foot">{twitter_summary['accuracy'] if twitter_summary['exists'] else 'Result unavailable'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_cols[3]:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Detection Systems</div>
                <div class="kpi-value">3</div>
                <div class="kpi-foot">Instagram · Twitter/X · Fake Comment</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Model performance overview</div>', unsafe_allow_html=True)
    perf_rows = []
    for label, path in {
        "Instagram": RESULTS_DIR / "instagram" / "instagram_model_comparison.csv",
        "Twitter / X": RESULTS_DIR / "twitter" / "twitter_ml_comparison.csv",
        "Twitter / X NLP": RESULTS_DIR / "twitter" / "twitter_nlp_model_comparison.csv",
        "Twitter / X GNN": RESULTS_DIR / "twitter" / "gnn_results.csv",
    }.items():
        df = read_csv(path)
        if df.empty:
            perf_rows.append({"Module": label, "Best accuracy": "Result unavailable"})
            continue
        score_cols = [c for c in df.columns if "accuracy" in c.lower()]
        if not score_cols:
            perf_rows.append({"Module": label, "Best accuracy": "Result unavailable"})
            continue
        best_val = df[score_cols].astype(float).max().max()
        perf_rows.append({"Module": label, "Best accuracy": float(best_val)})

    if perf_rows:
        perf_df = pd.DataFrame(perf_rows)
        fig = px.bar(perf_df, x="Module", y="Best accuracy", color="Module", color_discrete_sequence=["#6366F1", "#3B82F6", "#14B8A6", "#F59E0B"])
        fig.update_layout(height=320, plot_bgcolor="#0B1120", paper_bgcolor="#0B1120", font_color="#F8FAFC", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Prediction history</div>', unsafe_allow_html=True)
    hist = st.session_state.get("prediction_history", [])
    if hist:
        st.dataframe(pd.DataFrame(hist), width="stretch", hide_index=True)
    else:
        st.write("No predictions yet.")
    st.markdown('</div>', unsafe_allow_html=True)


# ---------------- Instagram ----------------
elif selected == "Instagram Bot Detection":
    st.title("📸 Instagram Bot Detection")
    st.caption("Analyze an Instagram profile and determine whether it is likely a bot or human account.")

    models = load_instagram_models()
    available_models = [name for name, mdl in {"Logistic Regression": models.get("lr"), "Random Forest": models.get("rf"), "XGBoost": models.get("xgb"), "ANN": models.get("ann")}.items() if mdl is not None]
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Instagram profile preview</div>', unsafe_allow_html=True)
    profile_username = st.text_input("Instagram Username", "@username", key="ig_username")
    profile_name = st.text_input("Profile Name", "John Doe", key="ig_name")
    col1, col2 = st.columns([1.2, 2])
    with col1:
        st.markdown(
            """
            <div class="profile-card">
                <div style="display:flex; align-items:center; gap: 16px;">
                    <div class="avatar instagram-avatar"></div>
                    <div>
                        <div class="profile-handle">@username</div>
                        <div class="profile-name">John Doe</div>
                    </div>
                </div>
                <div class="stats-row">
                    <div class="stat-box"><strong>126</strong><span>Posts</span></div>
                    <div class="stat-box"><strong>18.5k</strong><span>Followers</span></div>
                    <div class="stat-box"><strong>320</strong><span>Following</span></div>
                </div>
                <div style="margin-top: 16px;" class="subtle">Bio: creator, lifestyle and travel content.</div>
                <div style="margin-top: 10px;" class="subtle">Website: example.com</div>
                <div style="margin-top: 10px;" class="pill human">Public account</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown('<div class="small-label">Profile feature inputs</div>', unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3)
        with p1:
            profile_pic = st.selectbox("Profile Picture available", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
            username_ratio = st.slider("Username digit ratio", 0.0, 1.0, 0.10, step=0.01)
            fullname_words = st.number_input("Full name word count", min_value=0, max_value=10, value=2)
        with p2:
            fullname_num_ratio = st.slider("Full name numeric ratio", 0.0, 1.0, 0.0, step=0.01)
            name_equals_username = st.selectbox("Name equals username", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            description_length = st.number_input("Description length", min_value=0, max_value=500, value=45)
        with p3:
            external_url = st.selectbox("External URL", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            is_private = st.selectbox("Private account", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            posts = st.number_input("Number of posts", min_value=0, max_value=50000, value=126)
            followers = st.number_input("Number of followers", min_value=0, max_value=2000000, value=18500)
            follows = st.number_input("Number of following", min_value=0, max_value=500000, value=320)

        model_choice = st.selectbox("Model Selection", available_models if available_models else ["No model available"], index=0 if available_models else 0)
        if st.button("🔍 Analyze Instagram Profile", type="primary"):
            if not available_models:
                st.warning("Required model file was not found in models/instagram/.")
            else:
                row = build_instagram_row(
                    username=profile_username,
                    profile_name=profile_name,
                    profile_pic=profile_pic,
                    description="",
                    external_url=external_url,
                    is_private=is_private,
                    posts=posts,
                    followers=followers,
                    follows=follows,
                )
                row["nums/length username"] = username_ratio
                row["fullname words"] = fullname_words
                row["nums/length fullname"] = fullname_num_ratio
                row["name==username"] = name_equals_username
                row["description length"] = description_length

                model_name = model_choice.lower().replace(" ", "_")
                selected_model = None
                if model_name == "logistic_regression":
                    selected_model = models.get("lr")
                elif model_name == "random_forest":
                    selected_model = models.get("rf")
                elif model_name == "xgboost":
                    selected_model = models.get("xgb")
                elif model_name == "ann":
                    selected_model = models.get("ann")

                if selected_model is None:
                    st.warning("Selected model file could not be loaded.")
                else:
                    probs = {}
                    if hasattr(selected_model, "predict_proba"):
                        if model_name == "logistic_regression" and models.get("scaler") is not None:
                            z = models["scaler"].transform(row)
                            probs["selected"] = float(selected_model.predict_proba(z)[0][1])
                        elif model_name == "random_forest":
                            probs["selected"] = float(selected_model.predict_proba(row)[0][1])
                        elif model_name == "xgboost":
                            probs["selected"] = float(selected_model.predict_proba(row)[0][1])
                    elif model_name == "ann" and models.get("scaler") is not None:
                        z = models["scaler"].transform(row)
                        probs["selected"] = float(selected_model.predict(z, verbose=0)[0][0])

                    if not probs:
                        st.warning("This model does not provide a prediction probability in the current runtime environment.")
                    else:
                        p = float(np.mean(list(probs.values())))
                        label = "BOT DETECTED" if p >= 0.5 else "LIKELY HUMAN"
                        color = "#EF4444" if p >= 0.5 else "#22C55E"
                        st.markdown(
                            f"""
                            <div class="profile-card" style="margin-top:18px; border-left: 5px solid {color};">
                                <div class="panel-title" style="color:{color};">{label}</div>
                                <div class="subtle">Model prediction · Probability: {p * 100:.2f}%</div>
                                <div class="subtle" style="margin-top:8px;">Model used: {model_choice}</div>
                                <div class="subtle">Feature count: {len(INSTAGRAM_FEATURES)}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        add_prediction_history("Instagram", "Bot Detection", profile_name, profile_username, label, round(p * 100, 2), model_choice)
    st.markdown('</div>', unsafe_allow_html=True)


# ---------------- Twitter ----------------
elif selected == "Twitter/X Bot Detection":
    st.title("🐦 Twitter / X Bot Detection")
    st.caption("Analyze Twitter/X accounts using profile-based machine learning, NLP, and graph-based learning.")
    twitter_tab = st.session_state["twitter_tab"]

    if twitter_tab == "Profile ML":
        models = load_twitter_models()
        available_models = [name for name, mdl in {"Logistic Regression": models.get("lr"), "Random Forest": models.get("rf"), "XGBoost": models.get("xgb")}.items() if mdl is not None]

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Profile ML analysis</div>', unsafe_allow_html=True)
        p1, p2 = st.columns(2)
        with p1:
            screen_name = st.text_input("@Screen Name", "@janedoe")
            profile_name = st.text_input("Profile Name", "Jane Doe")
            description = st.text_area("Description", "Tech enthusiast and community builder.")
        with p2:
            followers_count = st.number_input("Followers", 0, 1000000, 850)
            friends_count = st.number_input("Following", 0, 100000, 420)
            statuses_count = st.number_input("Statuses / Tweets", 0, 1000000, 1200)
            favourites_count = st.number_input("Favourites", 0, 1000000, 300)
            listed_count = st.number_input("Listed Count", 0, 100000, 5)
            verified = st.selectbox("Verified", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            protected = st.selectbox("Protected", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            default_profile = st.selectbox("Default Profile", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            default_profile_image = st.selectbox("Default Profile Image", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

        model_choice = st.selectbox("Model Selection", available_models if available_models else ["No model available"], index=0 if available_models else 0)
        if st.button("🔍 Analyze Twitter / X Profile", type="primary"):
            if not available_models:
                st.warning("Required Twitter profile model files were not found in models/twitter/.")
            else:
                row = build_twitter_row(
                    screen_name,
                    profile_name,
                    description,
                    followers_count,
                    friends_count,
                    statuses_count,
                    favourites_count,
                    listed_count,
                    verified,
                    protected,
                    default_profile,
                    default_profile_image,
                )
                model_key = model_choice.lower().replace(" ", "_")
                model_obj = None
                if model_key == "logistic_regression":
                    model_obj = models.get("lr")
                elif model_key == "random_forest":
                    model_obj = models.get("rf")
                elif model_key == "xgboost":
                    model_obj = models.get("xgb")
                if model_obj is None:
                    st.warning("Selected model file could not be loaded.")
                else:
                    if hasattr(model_obj, "predict_proba"):
                        z = row if model_key in {"random_forest", "xgboost"} else models["scaler"].transform(row)
                        p = float(model_obj.predict_proba(z)[0][1]) if model_key in {"random_forest", "xgboost"} else float(model_obj.predict_proba(z)[0][1])
                    else:
                        p = 0.0
                    label = "BOT DETECTED" if p >= 0.5 else "HUMAN / LIKELY HUMAN"
                    color = "#EF4444" if p >= 0.5 else "#22C55E"
                    st.markdown(
                        f"""
                        <div class="profile-card" style="margin-top:18px; border-left: 5px solid {color};">
                            <div class="panel-title" style="color:{color};">{label}</div>
                            <div class="subtle">Model prediction · Probability: {p * 100:.2f}%</div>
                            <div class="subtle" style="margin-top:8px;">Model used: {model_choice}</div>
                            <div class="subtle">Feature count: {len(TWITTER_FEATURES)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    add_prediction_history("Twitter / X", "Profile ML", profile_name, screen_name, label, round(p * 100, 2), model_choice)
        st.markdown('</div>', unsafe_allow_html=True)

    elif twitter_tab == "NLP":
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Tweet / text analysis</div>', unsafe_allow_html=True)
        text = st.text_area("Enter tweet or text for analysis...", "Follow us for free giveaways and fast cash now!")
        if st.button("🔍 Analyze NLP Text", type="primary"):
            models = load_twitter_models()
            if models.get("nlp") is None or models.get("tfidf") is None:
                st.warning("Required NLP model/vectorizer files were not found in models/twitter/.")
            else:
                cleaned = re.sub(r"http\S+|www\S+|https\S+", " ", text.lower())
                cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", cleaned)
                cleaned = re.sub(r"\s+", " ", cleaned).strip()
                vec = models["tfidf"].transform([cleaned])
                p = float(models["nlp"].predict_proba(vec)[0][1])
                label = "SUSPICIOUS / BOT-LIKE TEXT" if p >= 0.5 else "LIKELY HUMAN TEXT"
                color = "#EF4444" if p >= 0.5 else "#22C55E"
                st.markdown(
                    f"""
                    <div class="profile-card" style="margin-top:18px; border-left: 5px solid {color};">
                        <div class="panel-title" style="color:{color};">{label}</div>
                        <div class="subtle">Model prediction · Probability: {p * 100:.2f}%</div>
                        <div class="subtle" style="margin-top:8px;">Model used: TF-IDF + Logistic Regression</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                add_prediction_history("Twitter / X", "NLP", "", "", label, round(p * 100, 2), "TF-IDF + Logistic Regression")
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">KNN / profile-similarity graph</div>', unsafe_allow_html=True)
        gnn_df = read_csv(RESULTS_DIR / "twitter" / "gnn_results.csv")
        if gnn_df.empty:
            st.info("Existing GNN result file is not available in the repository.")
        else:
            st.write("This graph-based detection uses a KNN/profile-similarity graph structure and should be labeled as profile-similarity graph evidence, not as a real follower/following network unless the source data supports that model.")
            st.dataframe(gnn_df, width="stretch", hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ---------------- Fake Comment ----------------
elif selected == "Fake Comment Detection":
    st.title("💬 Fake Comment Detection")
    st.caption("Detect whether a comment is suspicious/fake or genuine.")
    tab1, tab2 = st.tabs(["Single Comment", "Batch CSV"])

    with tab1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        text = st.text_area("Enter a comment to analyze...", "Follow me for free likes and DM us now!!")
        if st.button("🔍 Analyze Comment", type="primary"):
            if not TRANSFORMERS_AVAILABLE:
                st.warning("Transformers is not available in this environment.")
            else:
                tokenizer, model = load_comment_model()
                if tokenizer is None or model is None:
                    st.error("The fake-comment model could not be loaded from the Hugging Face repository.")
                else:
                    label, prob_genuine, prob_bot = predict_comment_text(text, tokenizer, model)
                    suspicious = label.upper() in ("BOT_OR_SPAM", "FAKE")
                    p = prob_bot if suspicious else prob_genuine
                    verdict = "SUSPICIOUS / FAKE COMMENT" if suspicious else "GENUINE COMMENT"
                    color = "#EF4444" if suspicious else "#22C55E"
                    st.markdown(
                        f"""
                        <div class="profile-card" style="margin-top:18px; border-left: 5px solid {color};">
                            <div class="panel-title" style="color:{color};">{verdict}</div>
                            <div class="subtle">Model prediction · Probability: {p * 100:.2f}%</div>
                            <div class="subtle" style="margin-top:8px;">Model used: {COMMENT_MODEL_REPO}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    add_prediction_history("Fake Comment", "Fake Comment Detection", "", "", verdict, round(p * 100, 2), COMMENT_MODEL_REPO)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            if df.empty:
                st.info("No rows found in uploaded CSV.")
            else:
                text_col = st.selectbox("Select text column", df.columns.tolist())
                if st.button("Run Batch Detection", type="primary"):
                    if not TRANSFORMERS_AVAILABLE:
                        st.warning("Transformers unavailable.")
                    else:
                        tokenizer, model = load_comment_model()
                        if tokenizer is None or model is None:
                            st.error("The fake-comment model could not be loaded.")
                        else:
                            batch_rows = []
                            for txt in df[text_col].fillna("").astype(str).tolist():
                                label, _, p_bot = predict_comment_text(txt, tokenizer, model)
                                batch_rows.append({"comment": txt, "label": label, "bot_spam_probability": round(p_bot * 100, 2)})
                            out = pd.DataFrame(batch_rows)
                            st.dataframe(out, width="stretch", hide_index=True)
                            st.download_button("Download results", out.to_csv(index=False).encode("utf-8"), file_name="fake_comment_results.csv", mime="text/csv")
        st.markdown('</div>', unsafe_allow_html=True)


# ---------------- Model Performance ----------------
elif selected == "Model Performance":
    st.title("📊 Model Performance")
    st.caption("Performance metrics from the repository’s existing result files.")

    sections = {
        "Instagram Models": RESULTS_DIR / "instagram" / "instagram_model_comparison.csv",
        "Twitter / X Profile ML": RESULTS_DIR / "twitter" / "twitter_ml_comparison.csv",
        "Twitter / X NLP": RESULTS_DIR / "twitter" / "twitter_nlp_model_comparison.csv",
        "Twitter / X GNN": RESULTS_DIR / "twitter" / "gnn_results.csv",
        "Fake Comment Models": None,
    }

    for title, path in sections.items():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="panel-title">{title}</div>', unsafe_allow_html=True)
        if path is None or not path.exists():
            st.info("Result unavailable")
        else:
            df = read_csv(path)
            if df.empty:
                st.info("Result unavailable")
            else:
                st.dataframe(df, width="stretch", hide_index=True)
                numeric = [c for c in df.columns if any(k in c.lower() for k in ["accuracy", "precision", "recall", "f1"]) ]
                if numeric:
                    fig = px.bar(df, x=df.columns[0], y=numeric[:3], barmode="group", color_discrete_sequence=["#6366F1", "#3B82F6", "#22C55E"])
                    fig.update_layout(height=300, plot_bgcolor="#0B1120", paper_bgcolor="#0B1120", font_color="#F8FAFC", margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)


# ---------------- Feature Importance ----------------
elif selected == "Feature Importance":
    st.title("⭐ Feature Importance")
    st.caption("Actual feature importance values from existing repository result files.")
    files = []
    for folder in [RESULTS_DIR / "instagram", RESULTS_DIR / "twitter"]:
        if folder.exists():
            files.extend(sorted(folder.glob("*feature_importance*.csv")))
            files.extend(sorted(folder.glob("*tfidf*.csv")))
    files = sorted({p: None for p in files}.keys())

    if not files:
        st.info("Feature importance files were not found in the repository result folders.")
    else:
        for p in files:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="panel-title">{p.name}</div>', unsafe_allow_html=True)
            df = read_csv(p)
            if df.empty or not {"Feature", "Importance"}.issubset(df.columns):
                st.info("Result unavailable")
            else:
                plot_df = df.sort_values("Importance", ascending=True).tail(10)
                fig = px.bar(plot_df, x="Importance", y="Feature", orientation="h", color_discrete_sequence=["#6366F1"])
                fig.update_layout(height=420, plot_bgcolor="#0B1120", paper_bgcolor="#0B1120", font_color="#F8FAFC", margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, width="stretch")
                st.dataframe(df, width="stretch", hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)


# ---------------- Dataset & Results ----------------
elif selected == "Dataset & Results":
    st.title("🗄️ Dataset & Results")
    tab1, tab2, tab3 = st.tabs(["Instagram", "Twitter / X", "Fake Comments"])

    with tab1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        df = pd.read_csv(DATA_DIR / "instagram" / "train.csv")
        st.write(f"Rows: {len(df)} | Columns: {len(df.columns)} | Target: {df.columns[-1]}")
        st.write(f"Missing values: {int(df.isnull().sum().sum())}")
        st.write(f"Duplicate rows: {int(df.duplicated().sum())}")
        st.dataframe(df.head(10), width="stretch", hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        df = load_twitter_data()
        st.write(f"Rows: {len(df)} | Columns: {len(df.columns)} | Target: fake")
        st.write(f"Missing values: {int(df.isnull().sum().sum())}")
        st.write(f"Duplicate rows: {int(df.duplicated().sum())}")
        st.dataframe(df.head(10), width="stretch", hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.info("The fake-comment system uses the existing Hugging Face model repository and does not have a local CSV dataset in the project root. The dashboard visualizes the model and analysis pipeline without modifying the repository assets.")
        st.markdown('</div>', unsafe_allow_html=True)


# ---------------- About ----------------
elif selected == "About":
    st.title("ℹ️ About SocialGuard AI")
    st.markdown(
        """
        <div class="section-card">
            <div class="panel-title">System overview</div>
            <div class="subtle">This dashboard visualizes and integrates existing trained models and analysis results from the repository. It does not retrain, overwrite, or modify the underlying datasets, model files, results, or graphs.</div>
            <div style="height: 16px;"></div>
            <div class="subtle">It combines:</div>
            <ul class="subtle">
                <li>Instagram bot detection</li>
                <li>Twitter / X bot detection</li>
                <li>NLP-based text analysis</li>
                <li>Graph-based learning</li>
                <li>Fake comment detection</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Default fallback
else:
    st.title("Overview")
    st.info("Select a navigation item from the sidebar.")
