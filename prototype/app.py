import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from PIL import Image

# Fake Comment Detector (RoBERTa) - loaded lazily, only when that tab is opened
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# Hosted on Hugging Face Hub - downloaded & cached automatically at runtime,
# so the 300+ MB weights file never needs to be committed to GitHub.
COMMENT_MODEL_REPO = "Tharanya06/Bot-Deduction"
# The model is stored in the main repo (no subfolder needed) OR in a subfolder
# The transformers library will automatically resolve the correct path
COMMENT_MODEL_SUBFOLDER = None  # Will load from main repo root

# Official Logos (SVG URLs)
INSTAGRAM_LOGO_URL = "https://upload.wikimedia.org/wikipedia/commons/e/e7/Instagram_logo_2016.svg"
TWITTER_LOGO_URL = "https://upload.wikimedia.org/wikipedia/commons/6/6f/Logo_of_Twitter.svg"

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Multi-Platform Bot Detection | CCS4310 Project",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Path to graphs directory (in the parent directory of prototype)
GRAPHS_DIR = os.path.join(os.path.dirname(BASE_DIR), "graphs")
INSTAGRAM_GRAPHS_DIR = os.path.join(GRAPHS_DIR, "instagram")
TWITTER_GRAPHS_DIR = os.path.join(GRAPHS_DIR, "twitter")
FAKE_COMMENT_GRAPHS_DIR = os.path.join(GRAPHS_DIR, "fake_comment_detection")

# Helper function to display graph images
@st.cache_data
def load_graph_image(graph_path):
    """Load and return a graph image from file."""
    if os.path.exists(graph_path):
        return Image.open(graph_path)
    return None

def display_graph_from_file(graph_path, caption="", width=None):
    """Display a graph image file in Streamlit."""
    img = load_graph_image(graph_path)
    if img is not None:
        st.image(img, caption=caption, use_container_width=True if width is None else False, width=width)
    else:
        st.warning(f"Graph not found: {graph_path}")

def get_available_graphs(graphs_dir):
    """Get list of available PNG graphs in a directory."""
    if os.path.exists(graphs_dir):
        graphs = sorted([f for f in os.listdir(graphs_dir) if f.endswith('.png')])
        return graphs
    return []

INSTAGRAM_FEATURES = [
    "profile pic", "nums/length username", "fullname words",
    "nums/length fullname", "name==username", "description length",
    "external URL", "private", "#posts", "#followers", "#follows",
]

TWITTER_FEATURES = [
    "followers_count", "friends_count", "statuses_count", "favourites_count",
    "listed_count", "followers_friends_ratio", "favourites_statuses_ratio",
    "statuses_followers_ratio", "description_length", "username_length",
    "username_digit_count", "username_digit_ratio", "fullname_length",
    "fullname_word_count", "verified", "protected", "default_profile",
    "default_profile_image", "name_equals_username",
]

# Custom Base CSS
st.markdown("""
<style>
    .metric-card {
        background: #1E1E2E;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.05);
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #1A1C23;
        border-radius: 8px;
        color: #B0B3C6;
        font-weight: 600;
    }
    .header-logo-container {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 10px;
    }
    .header-logo {
        width: 45px;
        height: 45px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar - Platform Switcher Only
# ---------------------------------------------------------
with st.sidebar:
    st.title("Bot Detection")
    st.markdown("---")

    # Platform Switcher
    platform = st.radio(
        "Choose Social Media Platform:",
        ["Instagram Bot Detection", "Twitter / X Bot Detection", "Fake Comment Detection (NLP)"]
    )

# ---------------------------------------------------------
# REAL DATA LOADERS
# (loads the actual project datasets instead of synthetic random data)
# ---------------------------------------------------------
@st.cache_data
def load_instagram_data():
    tr = pd.read_csv(os.path.join(DATA_DIR, "instagram", "train.csv"))
    te = pd.read_csv(os.path.join(DATA_DIR, "instagram", "test.csv"))
    df = pd.concat([tr, te], ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    return df


@st.cache_data
def load_twitter_data():
    tsv_path = os.path.join(DATA_DIR, "twitter", "gilani-2017.tsv")
    json_path = os.path.join(DATA_DIR, "twitter", "gilani-2017_tweets.json")

    tsv_df = pd.read_csv(tsv_path, sep="\t", header=None, names=["user_id", "raw_label"])
    dup_rows = tsv_df[tsv_df["user_id"].duplicated(keep=False)]
    conflict_ids = dup_rows.groupby("user_id")["raw_label"].nunique()
    conflict_ids = conflict_ids[conflict_ids > 1].index.tolist()
    tsv_clean = tsv_df[~tsv_df["user_id"].isin(conflict_ids)].drop_duplicates(subset=["user_id"])
    tsv_clean["target"] = tsv_clean["raw_label"].str.strip().str.lower().map({"human": 0, "bot": 1})
    tsv_clean["user_id"] = tsv_clean["user_id"].astype(np.int64)

    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    records = [item["user"] for item in json_data
               if isinstance(item, dict) and "user" in item and isinstance(item["user"], dict)]
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
    merged["name_equals_username"] = (merged["name"].str.lower().str.strip() == merged["screen_name"].str.lower().str.strip()).astype(int)
    for b in ["verified", "protected", "default_profile", "default_profile_image"]:
        if b in merged.columns:
            merged[b] = merged[b].fillna(False).astype(int)

    return merged[TWITTER_FEATURES + ["target"]].rename(columns={"target": "fake"})


# ---------------------------------------------------------
# REAL MODEL LOADERS
# (loads the actual trained .pkl / .keras files instead of a heuristic score)
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading Instagram models...")
def load_instagram_models():
    bundle = {}
    ig_dir = os.path.join(MODELS_DIR, "instagram")
    
    # Try to load ANN model
    try:
        import tensorflow as tf
        bundle["ann"] = tf.keras.models.load_model(os.path.join(ig_dir, "instagram_ann.keras"))
    except ImportError:
        st.warning("TensorFlow not available - ANN model skipped")
        bundle["ann"] = None
    except Exception as e:
        st.warning(f"Could not load ANN model: {e}")
        bundle["ann"] = None
    
    # Load traditional ML models
    for key, fname in [("xgb", "xgboost.pkl"), ("rf", "random_forest.pkl"), ("lr", "logistic_regression.pkl")]:
        try:
            model_path = os.path.join(ig_dir, fname)
            if os.path.exists(model_path):
                bundle[key] = joblib.load(model_path)
            else:
                st.warning(f"Instagram {key.upper()} model file not found: {fname}")
                bundle[key] = None
        except Exception as e:
            st.warning(f"Could not load Instagram {key.upper()} model: {e}")
            bundle[key] = None
    
    # Load scaler for feature normalization
    try:
        df = load_instagram_data()
        bundle["scaler"] = StandardScaler().fit(df[INSTAGRAM_FEATURES])
    except Exception as e:
        st.warning(f"Could not create scaler: {e}")
        bundle["scaler"] = None
    
    return bundle


@st.cache_resource(show_spinner="Loading Twitter models...")
def load_twitter_models():
    bundle = {}
    tw_dir = os.path.join(MODELS_DIR, "twitter")
    
    # Load ML models
    for key, fname in [("xgb", "xgboost.pkl"), ("rf", "random_forest.pkl"), ("lr", "logistic_regression.pkl"),
                        ("nlp", "nlp_model.pkl"), ("tfidf", "tfidf_vectorizer.pkl")]:
        try:
            model_path = os.path.join(tw_dir, fname)
            if os.path.exists(model_path):
                bundle[key] = joblib.load(model_path)
            else:
                st.warning(f"Twitter {key.upper()} model file not found: {fname}")
                bundle[key] = None
        except Exception as e:
            st.warning(f"Could not load Twitter {key.upper()} model: {e}")
            bundle[key] = None
    
    # Load scaler for feature normalization
    try:
        df = load_twitter_data()
        bundle["scaler"] = StandardScaler().fit(df[TWITTER_FEATURES])
    except Exception as e:
        st.warning(f"Could not create Twitter scaler: {e}")
        bundle["scaler"] = None
    
    return bundle


def engineer_twitter_row(screen_name, name, description, followers_count, friends_count,
                          statuses_count, favourites_count, listed_count,
                          verified, protected, default_profile, default_profile_image):
    followers_friends_ratio = (followers_count + 1.0) / (friends_count + 1.0)
    favourites_statuses_ratio = (favourites_count + 1.0) / (statuses_count + 1.0)
    statuses_followers_ratio = (statuses_count + 1.0) / (followers_count + 1.0)
    username_length = len(screen_name)
    username_digit_count = sum(c.isdigit() for c in screen_name)
    username_digit_ratio = username_digit_count / (username_length if username_length else 1)
    fullname_length = len(name)
    fullname_word_count = len(name.split()) if name else 0
    name_equals_username = int(name.strip().lower() == screen_name.strip().lower())

    row = {
        "followers_count": followers_count, "friends_count": friends_count,
        "statuses_count": statuses_count, "favourites_count": favourites_count,
        "listed_count": listed_count, "followers_friends_ratio": followers_friends_ratio,
        "favourites_statuses_ratio": favourites_statuses_ratio,
        "statuses_followers_ratio": statuses_followers_ratio,
        "description_length": len(description or ""), "username_length": username_length,
        "username_digit_count": username_digit_count, "username_digit_ratio": username_digit_ratio,
        "fullname_length": fullname_length, "fullname_word_count": fullname_word_count,
        "verified": int(verified), "protected": int(protected),
        "default_profile": int(default_profile), "default_profile_image": int(default_profile_image),
        "name_equals_username": name_equals_username,
    }
    return pd.DataFrame([row])[TWITTER_FEATURES]


@st.cache_resource(show_spinner="Downloading & loading Fake Comment Detection model (RoBERTa, first run only)...")
def load_comment_model():
    if not TRANSFORMERS_AVAILABLE:
        return None, None
    try:
        # Try loading without subfolder first (model at repo root)
        if COMMENT_MODEL_SUBFOLDER is None:
            tokenizer = AutoTokenizer.from_pretrained(COMMENT_MODEL_REPO, trust_remote_code=True)
            model = AutoModelForSequenceClassification.from_pretrained(COMMENT_MODEL_REPO, trust_remote_code=True)
        else:
            # Try loading with subfolder
            tokenizer = AutoTokenizer.from_pretrained(COMMENT_MODEL_REPO, subfolder=COMMENT_MODEL_SUBFOLDER, trust_remote_code=True)
            model = AutoModelForSequenceClassification.from_pretrained(COMMENT_MODEL_REPO, subfolder=COMMENT_MODEL_SUBFOLDER, trust_remote_code=True)
        
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Could not load the fake comment model from Hugging Face: {e}")
        st.info(f"Model repo: https://huggingface.co/{COMMENT_MODEL_REPO}\nMake sure you have internet access for first-time model download.")
        return None, None


def predict_comment(text: str, tokenizer, model):
    """Returns (label_str, prob_genuine, prob_bot_or_spam)."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0].tolist()
    id2label = model.config.id2label
    label_id = int(np.argmax(probs))
    label = id2label[label_id] if label_id in id2label else id2label[str(label_id)]
    prob_genuine = probs[0]
    prob_bot = probs[1]
    return label, prob_genuine, prob_bot


def run_ensemble(preds: dict):
    """preds: {model_name: prob_fake}. Returns (avg_prob, votes_fake, total)."""
    if not preds:
        return None, 0, 0
    avg_prob = float(np.mean(list(preds.values())))
    votes_fake = sum(1 for v in preds.values() if v >= 0.5)
    return avg_prob, votes_fake, len(preds)


def render_prediction_result(avg_prob, votes_fake, total, preds):
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        if avg_prob > 0.5:
            st.error(f"Prediction: FAKE / BOT ACCOUNT\n\nProbability: {avg_prob*100:.1f}%  ·  {votes_fake}/{total} models agree")
        else:
            st.success(f"Prediction: REAL / AUTHENTIC ACCOUNT\n\nProbability: {(1-avg_prob)*100:.1f}%  ·  {total-votes_fake}/{total} models agree")
    with res_col2:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_prob * 100,
            title={'text': "Bot Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#E1306C"},
                'steps': [
                    {'range': [0, 50], 'color': "#2E7D32"},
                    {'range': [50, 100], 'color': "#C2185B"}
                ]
            }
        ))
        fig_gauge.update_layout(height=250, margin=dict(t=30, b=10, l=30, r=30))
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("#### Model-by-model breakdown")
    breakdown = pd.DataFrame(
        [{"Model": k, "Fake probability": f"{v*100:.1f}%", "Verdict": "Bot" if v >= 0.5 else "Real"} for k, v in preds.items()]
    )
    st.dataframe(breakdown, use_container_width=True, hide_index=True)


# =========================================================
# PLATFORM 1: INSTAGRAM DASHBOARD
# =========================================================
if "Instagram" in platform:
    # Instagram Dynamic Styling
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #833AB4 0%, #FD1D1D 50%, #FCB045 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 2.5rem;
            margin: 0;
        }
        .stTabs [aria-selected="true"] {
            background-color: #E1306C !important;
            color: #FFFFFF !important;
        }
    </style>
    """, unsafe_allow_html=True)

    df_insta = load_instagram_data()
    ig_models = load_instagram_models()

    # Header with Official Logo
    st.markdown(f"""
    <div class="header-logo-container">
        <img src="{INSTAGRAM_LOGO_URL}" class="header-logo">
        <h1 class="main-header">Instagram Bot Detection Analysis Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)
    st.write("Comprehensive machine learning & deep learning pipeline analysis for identifying fake and automated Instagram profiles.")

    # Main Tabs
    tab_summary, tab_eda, tab_models, tab_inference = st.tabs([
        "Executive Summary",
        "Exploratory Data Analysis",
        "Machine Learning Benchmarks",
        "Real-Time Bot Predictor"
    ])

    # --- TAB 1: EXECUTIVE SUMMARY ---
    with tab_summary:
        st.subheader("Dataset Metrics & Class Distribution")

        n_total = len(df_insta)
        n_real = int((df_insta["fake"] == 0).sum())
        n_fake = int((df_insta["fake"] == 1).sum())

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Total Cleaned Records", value=f"{n_total}")
        with col2:
            st.metric(label="Real Accounts (Class 0)", value=f"{n_real}", delta=f"{n_real/n_total*100:.1f}%")
        with col3:
            st.metric(label="Bot Accounts (Class 1)", value=f"{n_fake}", delta=f"{n_fake/n_total*100:.1f}%")
        with col4:
            st.metric(label="Total Features", value="11 Inputs")

        st.markdown("---")

        col_left, col_right = st.columns([1.2, 1])

        with col_left:
            st.subheader("Target Class Distribution")
            target_counts = df_insta['fake'].value_counts().reset_index()
            target_counts.columns = ['Account Type', 'Count']
            target_counts['Account Type'] = target_counts['Account Type'].map({0: 'Real Account (0)', 1: 'Fake/Bot Account (1)'})

            fig_pie = px.pie(
                target_counts, values='Count', names='Account Type',
                hole=0.4,
                color_discrete_sequence=['#4CAF50', '#E1306C']
            )
            fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=350)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_right:
            st.subheader("Feature Audit Table")
            feature_data = pd.DataFrame({
                "Feature Name": [
                    "profile pic", "nums/length username", "fullname words",
                    "nums/length fullname", "name==username", "description length",
                    "external URL", "private", "#posts", "#followers", "#follows"
                ],
                "Data Type": ["int64", "float64", "int64", "float64", "int64", "int64", "int64", "int64", "int64", "int64", "int64"],
                "Description": [
                    "Has profile picture (1/0)", "Digit to length ratio (username)", "Word count in full name",
                    "Digit ratio in full name", "Full name matches username", "Bio character length",
                    "Has external URL (1/0)", "Private account status (1/0)", "Total published posts",
                    "Follower count", "Following count"
                ]
            })
            st.dataframe(feature_data, use_container_width=True, hide_index=True)

    # --- TAB 2: EDA ---
    with tab_eda:
        st.subheader("Feature Analysis & Distributions")

        # Display available distribution graphs
        st.markdown("#### Key Distributions")
        eda_col1, eda_col2 = st.columns(2)
        
        with eda_col1:
            img = load_graph_image(os.path.join(INSTAGRAM_GRAPHS_DIR, "01_target_distribution.png"))
            if img:
                st.image(img, caption="Target Distribution (Real vs Bot)", use_container_width=True)
        
        with eda_col2:
            img = load_graph_image(os.path.join(INSTAGRAM_GRAPHS_DIR, "distribution_followers.png"))
            if img:
                st.image(img, caption="Follower Count Distribution", use_container_width=True)

        eda_col3, eda_col4 = st.columns(2)
        
        with eda_col3:
            img = load_graph_image(os.path.join(INSTAGRAM_GRAPHS_DIR, "distribution_follows.png"))
            if img:
                st.image(img, caption="Following Count Distribution", use_container_width=True)
        
        with eda_col4:
            img = load_graph_image(os.path.join(INSTAGRAM_GRAPHS_DIR, "distribution_posts.png"))
            if img:
                st.image(img, caption="Posts Count Distribution", use_container_width=True)

        eda_col5, eda_col6 = st.columns(2)
        
        with eda_col5:
            img = load_graph_image(os.path.join(INSTAGRAM_GRAPHS_DIR, "distribution_description_length.png"))
            if img:
                st.image(img, caption="Description Length Distribution", use_container_width=True)
        
        with eda_col6:
            img = load_graph_image(os.path.join(INSTAGRAM_GRAPHS_DIR, "distribution_username_digits.png"))
            if img:
                st.image(img, caption="Username Digits Distribution", use_container_width=True)

        st.markdown("---")
        st.subheader("Binary Feature Analysis")
        
        binary_col1, binary_col2 = st.columns(2)
        
        with binary_col1:
            img = load_graph_image(os.path.join(INSTAGRAM_GRAPHS_DIR, "binary_profile_pic.png"))
            if img:
                st.image(img, caption="Profile Picture Feature", use_container_width=True)
        
        with binary_col2:
            img = load_graph_image(os.path.join(INSTAGRAM_GRAPHS_DIR, "binary_external_url.png"))
            if img:
                st.image(img, caption="External URL Feature", use_container_width=True)

        binary_col3, binary_col4 = st.columns(2)
        
        with binary_col3:
            img = load_graph_image(os.path.join(INSTAGRAM_GRAPHS_DIR, "binary_private.png"))
            if img:
                st.image(img, caption="Private Account Feature", use_container_width=True)
        
        with binary_col4:
            img = load_graph_image(os.path.join(INSTAGRAM_GRAPHS_DIR, "binary_name_equals_username.png"))
            if img:
                st.image(img, caption="Name Equals Username Feature", use_container_width=True)

        st.markdown("---")
        st.subheader("Feature Correlation Heatmap")
        img = load_graph_image(os.path.join(INSTAGRAM_GRAPHS_DIR, "correlation_heatmap.png"))
        if img:
            st.image(img, caption="Pearson Correlation Matrix", use_container_width=True)

    # --- TAB 3: BENCHMARKS (real held-out test set numbers from the training notebooks) ---
    with tab_models:
        st.subheader("Model Evaluation & Benchmarks")

        results_df = pd.DataFrame({
            "Model": ["Logistic Regression", "Random Forest", "XGBoost", "Artificial Neural Network (ANN)"],
            "Accuracy": [0.8921, 0.9209, 0.9281, 0.9137],
            "F1-Score": [0.8819, 0.9160, 0.9242, 0.9077],
            "Precision": [0.9655, 0.9677, 0.9683, 0.9672],
            "Recall": [0.8116, 0.8696, 0.8841, 0.8551],
        })

        col_bench1, col_bench2 = st.columns([1.2, 1])

        with col_bench1:
            fig_bench = px.bar(
                results_df, x='Model', y=['Accuracy', 'F1-Score'],
                barmode='group',
                title="Model Performance Metrics Comparison (held-out test set)",
                color_discrete_sequence=['#3F51B5', '#009688']
            )
            fig_bench.update_layout(yaxis=dict(range=[0.7, 1.0]))
            st.plotly_chart(fig_bench, use_container_width=True)

        with col_bench2:
            st.markdown("### Leaderboard Summary")
            st.dataframe(results_df.style.highlight_max(axis=0, subset=["Accuracy", "F1-Score", "Precision", "Recall"], color='#2E7D32'),
                         use_container_width=True, hide_index=True)
            st.success("Best Model: XGBoost achieved the highest performance — 92.8% accuracy, 92.4% F1-score.")

        st.markdown("---")
        st.subheader("Model Comparison Visualizations")
        
        # Display model performance graphs
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            img = load_graph_image(os.path.join(INSTAGRAM_GRAPHS_DIR, "model_accuracy_comparison.png"))
            if img:
                st.image(img, caption="Accuracy Comparison", use_container_width=True)
        
        with comp_col2:
            img = load_graph_image(os.path.join(INSTAGRAM_GRAPHS_DIR, "model_f1_comparison.png"))
            if img:
                st.image(img, caption="F1-Score Comparison", use_container_width=True)

        st.markdown("---")
        st.subheader("Confusion Matrices")
        
        cf_col1, cf_col2 = st.columns(2)
        
        with cf_col1:
            img = load_graph_image(os.path.join(INSTAGRAM_GRAPHS_DIR, "random_forest_confusion_matrix.png"))
            if img:
                st.image(img, caption="Random Forest Confusion Matrix", use_container_width=True)
        
        with cf_col2:
            img = load_graph_image(os.path.join(INSTAGRAM_GRAPHS_DIR, "xgboost_confusion_matrix.png"))
            if img:
                st.image(img, caption="XGBoost Confusion Matrix", use_container_width=True)

        cf_col3, cf_col4 = st.columns(2)
        
        with cf_col3:
            img = load_graph_image(os.path.join(INSTAGRAM_GRAPHS_DIR, "logistic_confusion_matrix.png"))
            if img:
                st.image(img, caption="Logistic Regression Confusion Matrix", use_container_width=True)
        
        with cf_col4:
            img = load_graph_image(os.path.join(INSTAGRAM_GRAPHS_DIR, "ann_confusion_matrix.png"))
            if img:
                st.image(img, caption="ANN Confusion Matrix", use_container_width=True)

        st.markdown("---")
        st.subheader("Feature Importance Analysis")
        
        fi_col1, fi_col2 = st.columns(2)
        
        with fi_col1:
            img = load_graph_image(os.path.join(INSTAGRAM_GRAPHS_DIR, "random_forest_feature_importance.png"))
            if img:
                st.image(img, caption="Random Forest Feature Importance", use_container_width=True)
        
        with fi_col2:
            img = load_graph_image(os.path.join(INSTAGRAM_GRAPHS_DIR, "xgboost_feature_importance.png"))
            if img:
                st.image(img, caption="XGBoost Feature Importance", use_container_width=True)

        st.markdown("---")
        st.subheader("ROC Curve & Training Analysis")
        
        roc_col1, roc_col2 = st.columns(2)
        
        with roc_col1:
            img = load_graph_image(os.path.join(INSTAGRAM_GRAPHS_DIR, "roc_curve_comparison.png"))
            if img:
                st.image(img, caption="ROC Curve Comparison", use_container_width=True)
        
        with roc_col2:
            img = load_graph_image(os.path.join(INSTAGRAM_GRAPHS_DIR, "ann_loss.png"))
            if img:
                st.image(img, caption="ANN Training Loss", use_container_width=True)
        
        ann_acc_img = load_graph_image(os.path.join(INSTAGRAM_GRAPHS_DIR, "ann_accuracy.png"))
        if ann_acc_img:
            st.image(ann_acc_img, caption="ANN Training Accuracy", use_container_width=True)

    # --- TAB 4: PREDICTOR (wired to the real trained models) ---
    with tab_inference:
        st.subheader("Real-Time Account Classification Engine")
        st.write("Input profile characteristics below to predict if an account is Authentic or a Bot.")

        with st.form("prediction_form"):
            col_in1, col_in2, col_in3 = st.columns(3)

            with col_in1:
                profile_pic = st.selectbox("Profile Picture Present?", [1, 0], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
                username_num_ratio = st.slider("Username Numerical Ratio", 0.0, 1.0, 0.1)
                fullname_words = st.number_input("Full Name Word Count", 0, 10, 2)
                fullname_num_ratio = st.slider("Full Name Numerical Ratio", 0.0, 1.0, 0.0)

            with col_in2:
                name_eq_user = st.selectbox("Name Equals Username?", [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
                bio_len = st.number_input("Description Length (Chars)", 0, 150, 45)
                ext_url = st.selectbox("External URL Included?", [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
                is_private = st.selectbox("Private Account?", [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")

            with col_in3:
                num_posts = st.number_input("Total Posts", 0, 10000, 120)
                num_followers = st.number_input("Follower Count", 0, 1000000, 850)
                num_follows = st.number_input("Following Count", 0, 100000, 400)

            submit_btn = st.form_submit_button("Run Classification Model")

        if submit_btn:
            row = pd.DataFrame([{
                "profile pic": profile_pic,
                "nums/length username": username_num_ratio,
                "fullname words": fullname_words,
                "nums/length fullname": fullname_num_ratio,
                "name==username": name_eq_user,
                "description length": bio_len,
                "external URL": ext_url,
                "private": is_private,
                "#posts": num_posts,
                "#followers": num_followers,
                "#follows": num_follows,
            }])[INSTAGRAM_FEATURES]

            preds = {}
            if ig_models.get("rf") is not None:
                preds["Random Forest"] = ig_models["rf"].predict_proba(row)[0][1]
            if ig_models.get("xgb") is not None:
                preds["XGBoost"] = ig_models["xgb"].predict_proba(row)[0][1]
            if ig_models.get("scaler") is not None:
                row_scaled = ig_models["scaler"].transform(row)
                if ig_models.get("lr") is not None:
                    preds["Logistic Regression"] = ig_models["lr"].predict_proba(row_scaled)[0][1]
                if ig_models.get("ann") is not None:
                    preds["ANN"] = float(ig_models["ann"].predict(row_scaled, verbose=0)[0][0])

            st.markdown("---")
            st.subheader("Prediction Result")

            if preds:
                avg_prob, votes_fake, total = run_ensemble(preds)
                render_prediction_result(avg_prob, votes_fake, total, preds)
            else:
                st.warning("No Instagram models could be loaded — check that the .pkl / .keras files are in `models/instagram/`.")

# =========================================================
# PLATFORM 2: TWITTER / X DASHBOARD
# =========================================================
elif "Twitter" in platform:
    # Twitter Dynamic Styling
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #1DA1F2 0%, #00D2FF 50%, #1A8CD8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 2.5rem;
            margin: 0;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1DA1F2 !important;
            color: #FFFFFF !important;
        }
    </style>
    """, unsafe_allow_html=True)

    df_twitter = load_twitter_data()
    tw_models = load_twitter_models()

    # Header with Official Logo
    st.markdown(f"""
    <div class="header-logo-container">
        <img src="{TWITTER_LOGO_URL}" class="header-logo">
        <h1 class="main-header">Twitter / X Bot Detection Analysis Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)
    st.write("Comprehensive machine learning & deep learning pipeline analysis for identifying fake and automated Twitter profiles.")

    # Main Tabs
    tab_summary, tab_eda, tab_models, tab_inference = st.tabs([
        "Executive Summary",
        "Exploratory Data Analysis",
        "Machine Learning Benchmarks",
        "Real-Time Bot Predictor"
    ])

    # --- TAB 1: EXECUTIVE SUMMARY ---
    with tab_summary:
        st.subheader("Dataset Metrics & Class Distribution")

        n_total = len(df_twitter)
        n_real = int((df_twitter["fake"] == 0).sum())
        n_fake = int((df_twitter["fake"] == 1).sum())

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Total Cleaned Records", value=f"{n_total}")
        with col2:
            st.metric(label="Human Accounts (Class 0)", value=f"{n_real}", delta=f"{n_real/n_total*100:.1f}%")
        with col3:
            st.metric(label="Bot Accounts (Class 1)", value=f"{n_fake}", delta=f"{n_fake/n_total*100:.1f}%")
        with col4:
            st.metric(label="Total Features", value="19 Inputs")

        st.markdown("---")

        col_left, col_right = st.columns([1.2, 1])

        with col_left:
            st.subheader("Target Class Distribution")
            target_counts = df_twitter['fake'].value_counts().reset_index()
            target_counts.columns = ['Account Type', 'Count']
            target_counts['Account Type'] = target_counts['Account Type'].map({0: 'Human Account (0)', 1: 'Bot Account (1)'})

            fig_pie = px.pie(
                target_counts, values='Count', names='Account Type',
                hole=0.4,
                color_discrete_sequence=['#1DA1F2', '#E0245E']
            )
            fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=350)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_right:
            st.subheader("Feature Audit Table")
            feature_data = pd.DataFrame({
                "Feature Name": TWITTER_FEATURES,
                "Data Type": ["int64", "int64", "int64", "int64", "int64", "float64", "float64", "float64",
                              "int64", "int64", "int64", "float64", "int64", "int64", "int64", "int64", "int64",
                              "int64", "int64"],
                "Description": [
                    "Follower count", "Following (friends) count", "Total tweets posted", "Total likes given",
                    "Number of lists the account appears on", "Followers-to-following ratio",
                    "Favourites-to-statuses ratio", "Statuses-to-followers ratio", "Bio character length",
                    "Username character length", "Digit count in username", "Digit-to-length ratio in username",
                    "Full display-name character length", "Word count in display name", "Verified badge (1/0)",
                    "Protected/private account (1/0)", "Uses default profile theme (1/0)",
                    "Uses default (egg) profile picture (1/0)", "Display name matches username (1/0)"
                ]
            })
            st.dataframe(feature_data, use_container_width=True, hide_index=True)

    # --- TAB 2: EDA ---
    with tab_eda:
        st.subheader("Feature Analysis & Distributions")

        # Display available distribution graphs
        st.markdown("#### Key Distributions")
        eda_col1, eda_col2 = st.columns(2)
        
        with eda_col1:
            img = load_graph_image(os.path.join(TWITTER_GRAPHS_DIR, "01_target_distribution.png"))
            if img:
                st.image(img, caption="Target Distribution (Human vs Bot)", use_container_width=True)
        
        with eda_col2:
            img = load_graph_image(os.path.join(TWITTER_GRAPHS_DIR, "distribution_followers.png"))
            if img:
                st.image(img, caption="Follower Count Distribution", use_container_width=True)

        eda_col3, eda_col4 = st.columns(2)
        
        with eda_col3:
            img = load_graph_image(os.path.join(TWITTER_GRAPHS_DIR, "distribution_friends.png"))
            if img:
                st.image(img, caption="Following Count Distribution", use_container_width=True)
        
        with eda_col4:
            img = load_graph_image(os.path.join(TWITTER_GRAPHS_DIR, "distribution_statuses.png"))
            if img:
                st.image(img, caption="Tweet Count Distribution", use_container_width=True)

        eda_col5, eda_col6 = st.columns(2)
        
        with eda_col5:
            img = load_graph_image(os.path.join(TWITTER_GRAPHS_DIR, "distribution_favourites.png"))
            if img:
                st.image(img, caption="Favorites/Likes Distribution", use_container_width=True)
        
        with eda_col6:
            img = load_graph_image(os.path.join(TWITTER_GRAPHS_DIR, "distribution_description_length.png"))
            if img:
                st.image(img, caption="Bio Length Distribution", use_container_width=True)

        st.markdown("---")
        st.subheader("Binary Feature Analysis")
        
        binary_col1, binary_col2 = st.columns(2)
        
        with binary_col1:
            img = load_graph_image(os.path.join(TWITTER_GRAPHS_DIR, "binary_verified.png"))
            if img:
                st.image(img, caption="Verified Badge Feature", use_container_width=True)
        
        with binary_col2:
            img = load_graph_image(os.path.join(TWITTER_GRAPHS_DIR, "binary_protected.png"))
            if img:
                st.image(img, caption="Protected Account Feature", use_container_width=True)

        binary_col3, binary_col4 = st.columns(2)
        
        with binary_col3:
            img = load_graph_image(os.path.join(TWITTER_GRAPHS_DIR, "binary_default_profile.png"))
            if img:
                st.image(img, caption="Default Profile Theme Feature", use_container_width=True)
        
        with binary_col4:
            img = load_graph_image(os.path.join(TWITTER_GRAPHS_DIR, "binary_default_profile_image.png"))
            if img:
                st.image(img, caption="Default Profile Picture Feature", use_container_width=True)

        st.markdown("---")
        st.subheader("Feature Correlation Heatmap")
        img = load_graph_image(os.path.join(TWITTER_GRAPHS_DIR, "correlation_heatmap.png"))
        if img:
            st.image(img, caption="Pearson Correlation Matrix", use_container_width=True)

        st.markdown("---")
        st.subheader("Advanced NLP Analysis")
        img = load_graph_image(os.path.join(TWITTER_GRAPHS_DIR, "nlp_top_tfidf_features.png"))
        if img:
            st.image(img, caption="Top TF-IDF Features (Bio Text Analysis)", use_container_width=True)

    # --- TAB 3: BENCHMARKS (real held-out test set numbers from the training notebooks) ---
    with tab_models:
        st.subheader("Model Evaluation & Benchmarks")

        results_df = pd.DataFrame({
            "Model": ["Random Forest", "XGBoost", "Logistic Regression", "GNN (graph)", "TF-IDF + Log. Reg. (bio text)"],
            "Accuracy": [0.7586, 0.7525, 0.7221, 0.7221, 0.6004],
            "F1-Score": [0.6972, 0.6823, 0.6347, 0.6347, 0.4012],
            "Precision": [0.7654, 0.7706, 0.7391, 0.7391, 0.5739],
            "Recall": [0.6402, 0.6121, 0.5561, 0.5561, 0.3084],
        })

        col_bench1, col_bench2 = st.columns([1.2, 1])

        with col_bench1:
            fig_bench = px.bar(
                results_df, x='Model', y=['Accuracy', 'F1-Score'],
                barmode='group',
                title="Model Performance Metrics Comparison (held-out test set)",
                color_discrete_sequence=['#1DA1F2', '#17BF63']
            )
            fig_bench.update_layout(yaxis=dict(range=[0.0, 1.0]))
            st.plotly_chart(fig_bench, use_container_width=True)

        with col_bench2:
            st.markdown("### Leaderboard Summary")
            st.dataframe(results_df.style.highlight_max(axis=0, subset=["Accuracy", "F1-Score", "Precision", "Recall"], color='#17BF63'),
                         use_container_width=True, hide_index=True)
            st.info("Best Model: Random Forest — 75.9% accuracy, 69.7% F1-score. Twitter bot detection is "
                    "noticeably harder than Instagram here; bio text alone (TF-IDF) is close to chance (60%).")

        st.markdown("---")
        st.subheader("Model Comparison Visualizations")
        
        # Display model performance graphs
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            img = load_graph_image(os.path.join(TWITTER_GRAPHS_DIR, "twitter_accuracy_comparison.png"))
            if img:
                st.image(img, caption="Accuracy Comparison", use_container_width=True)
        
        with comp_col2:
            img = load_graph_image(os.path.join(TWITTER_GRAPHS_DIR, "twitter_f1_comparison.png"))
            if img:
                st.image(img, caption="F1-Score Comparison", use_container_width=True)

        st.markdown("---")
        st.subheader("Confusion Matrices")
        
        cf_col1, cf_col2 = st.columns(2)
        
        with cf_col1:
            img = load_graph_image(os.path.join(TWITTER_GRAPHS_DIR, "random_forest_confusion_matrix.png"))
            if img:
                st.image(img, caption="Random Forest Confusion Matrix", use_container_width=True)
        
        with cf_col2:
            img = load_graph_image(os.path.join(TWITTER_GRAPHS_DIR, "xgboost_confusion_matrix.png"))
            if img:
                st.image(img, caption="XGBoost Confusion Matrix", use_container_width=True)

        cf_col3, cf_col4 = st.columns(2)
        
        with cf_col3:
            img = load_graph_image(os.path.join(TWITTER_GRAPHS_DIR, "logistic_confusion_matrix.png"))
            if img:
                st.image(img, caption="Logistic Regression Confusion Matrix", use_container_width=True)
        
        with cf_col4:
            img = load_graph_image(os.path.join(TWITTER_GRAPHS_DIR, "nlp_confusion_matrix.png"))
            if img:
                st.image(img, caption="NLP TF-IDF Confusion Matrix", use_container_width=True)

        st.markdown("---")
        st.subheader("Feature Importance Analysis")
        
        fi_col1, fi_col2 = st.columns(2)
        
        with fi_col1:
            img = load_graph_image(os.path.join(TWITTER_GRAPHS_DIR, "random_forest_feature_importance.png"))
            if img:
                st.image(img, caption="Random Forest Feature Importance", use_container_width=True)
        
        with fi_col2:
            img = load_graph_image(os.path.join(TWITTER_GRAPHS_DIR, "xgboost_feature_importance.png"))
            if img:
                st.image(img, caption="XGBoost Feature Importance", use_container_width=True)

        st.markdown("---")
        st.subheader("ROC Curve & Deep Learning Analysis")
        
        roc_col1, roc_col2 = st.columns(2)
        
        with roc_col1:
            img = load_graph_image(os.path.join(TWITTER_GRAPHS_DIR, "roc_curve_comparison.png"))
            if img:
                st.image(img, caption="ROC Curve Comparison", use_container_width=True)
        
        with roc_col2:
            img = load_graph_image(os.path.join(TWITTER_GRAPHS_DIR, "gnn_training_loss.png"))
            if img:
                st.image(img, caption="GNN Model Training Loss", use_container_width=True)

    # --- TAB 4: PREDICTOR (redesigned to match the real 19-feature schema the models were trained on) ---
    with tab_inference:
        st.subheader("Real-Time Account Classification Engine")
        st.write("Input Twitter/X profile characteristics below to predict if an account is Authentic or a Bot.")
        st.caption(
            "Note: these fields match the 19 features the Twitter models were actually trained on "
            "(profile metadata, not tweet-content signals like retweet ratio)."
        )

        with st.form("twitter_prediction_form"):
            col_in1, col_in2, col_in3 = st.columns(3)

            with col_in1:
                screen_name = st.text_input("Username / handle", "real_user123")
                name = st.text_input("Display name", "Real User")
                description = st.text_area("Bio / description", height=90)
                verified = st.selectbox("Verified Badge?", [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")

            with col_in2:
                followers_count = st.number_input("Followers Count", 0, 1000000, 800)
                friends_count = st.number_input("Following Count", 0, 100000, 400)
                statuses_count = st.number_input("Total Tweets", 0, 1000000, 1200)
                favourites_count = st.number_input("Total Likes (favourites)", 0, 1000000, 300)

            with col_in3:
                listed_count = st.number_input("Listed Count", 0, 100000, 5)
                protected = st.selectbox("Protected Account?", [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
                default_profile = st.selectbox("Default Profile Theme?", [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
                default_profile_image = st.selectbox("Default (Egg) Profile Picture?", [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")

            submit_btn = st.form_submit_button("Run Classification Model")

        if submit_btn:
            row = engineer_twitter_row(
                screen_name, name, description, followers_count, friends_count,
                statuses_count, favourites_count, listed_count,
                verified, protected, default_profile, default_profile_image
            )

            preds = {}
            if tw_models.get("rf") is not None:
                preds["Random Forest (profile)"] = tw_models["rf"].predict_proba(row)[0][1]
            if tw_models.get("xgb") is not None:
                preds["XGBoost (profile)"] = tw_models["xgb"].predict_proba(row)[0][1]
            if tw_models.get("scaler") is not None and tw_models.get("lr") is not None:
                row_scaled = tw_models["scaler"].transform(row)
                preds["Logistic Regression (profile)"] = tw_models["lr"].predict_proba(row_scaled)[0][1]
            if description and tw_models.get("nlp") is not None and tw_models.get("tfidf") is not None:
                cleaned = str(description).lower()
                cleaned = re.sub(r"http\S+|www\S+|https\S+", "", cleaned)
                cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", cleaned)
                cleaned = re.sub(r"\s+", " ", cleaned).strip()
                vec = tw_models["tfidf"].transform([cleaned])
                preds["TF-IDF + Log. Reg. (bio text)"] = tw_models["nlp"].predict_proba(vec)[0][1]

            st.markdown("---")
            st.subheader("Prediction Result")

            if preds:
                avg_prob, votes_fake, total = run_ensemble(preds)
                render_prediction_result(avg_prob, votes_fake, total, preds)
                st.caption(
                    "The Graph Convolutional Network (GNN) model is not included here — it requires the "
                    "full follower/following network graph, not a single manually-entered account. "
                    "Its offline test accuracy (72.2%) is shown on the Machine Learning Benchmarks tab."
                )
            else:
                st.warning("No Twitter models could be loaded — check that the .pkl files are in `models/twitter/`.")

# =========================================================
# PLATFORM 3: FAKE COMMENT DETECTION (NLP - RoBERTa)
# =========================================================
else:
    # Fake Comment Dynamic Styling (kept in the same dark-card / gradient-header
    # language as the Instagram + Twitter tabs above, own accent colour only)
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #7B2FF7 0%, #B84FCE 50%, #F7B733 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 2.5rem;
            margin: 0;
        }
        .stTabs [aria-selected="true"] {
            background-color: #7B2FF7 !important;
            color: #FFFFFF !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="header-logo-container">
        <h1 class="main-header">💬 Fake Comment Detection Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)
    st.write("Transformer-based (RoBERTa) NLP model that classifies a social media comment as "
             "**GENUINE** or **BOT / SPAM** based on its text content.")

    tab_about, tab_single, tab_batch = st.tabs([
        "Model Overview",
        "Single Comment Checker",
        "Batch CSV Checker"
    ])

    # --- TAB 1: MODEL OVERVIEW ---
    with tab_about:
        st.subheader("About This Model")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Architecture", value="RoBERTa (base)")
        with col2:
            st.metric(label="Encoder Layers", value="6")
        with col3:
            st.metric(label="Output Classes", value="2")

        st.markdown("---")
        st.markdown(
            "- **Task:** binary sequence classification on raw comment text\n"
            "- **Labels:** `GENUINE` (0) vs `BOT_OR_SPAM` (1)\n"
            "- **Tokenizer:** Byte-level BPE (RoBERTa tokenizer, 50,265-token vocabulary, 512 max tokens)\n"
            "- **Hosting:** weights are downloaded on first use from the project's Hugging Face model repo "
            "and cached locally, so they don't need to live inside the GitHub repo."
        )
        st.markdown("---")
        st.subheader("Training & Performance Analysis")
        
        st.caption(f"Source: huggingface.co/{COMMENT_MODEL_REPO}")

        if not TRANSFORMERS_AVAILABLE:
            st.warning("`transformers` / `torch` are not installed in this environment — add them to "
                       "`requirements.txt` to enable this tab (see updated requirements.txt).")

        st.markdown("---")
        st.subheader("Training & Performance Analysis")
        
        # Display training graphs
        train_col1, train_col2 = st.columns(2)
        
        with train_col1:
            img = load_graph_image(os.path.join(FAKE_COMMENT_GRAPHS_DIR, "01_class_distribution.png"))
            if img:
                st.image(img, caption="Training Dataset Class Distribution", use_container_width=True)
        
        with train_col2:
            img = load_graph_image(os.path.join(FAKE_COMMENT_GRAPHS_DIR, "02_word_count_distribution.png"))
            if img:
                st.image(img, caption="Comment Word Count Distribution", use_container_width=True)

        train_col3, train_col4 = st.columns([1.2, 1])
        
        with train_col3:
            img = load_graph_image(os.path.join(FAKE_COMMENT_GRAPHS_DIR, "03_training_validation_loss.png"))
            if img:
                st.image(img, caption="Training & Validation Loss", use_container_width=True)

    # --- TAB 2: SINGLE COMMENT CHECKER ---
    with tab_single:
        st.subheader("Classify a Single Comment")

        comment_text = st.text_area(
            "Paste a comment to analyze:",
            height=120,
            placeholder="e.g. 'Follow me for free followers!! link in bio 🔥🔥🔥'"
        )
        check_btn = st.button("Run Fake Comment Detection", type="primary")

        if check_btn:
            if not comment_text.strip():
                st.warning("Please enter a comment first.")
            elif not TRANSFORMERS_AVAILABLE:
                st.error("`transformers` / `torch` are not installed — see Model Overview tab.")
            else:
                tokenizer, model = load_comment_model()
                if tokenizer is None or model is None:
                    st.error("Fake comment model could not be loaded. Check your internet connection "
                             "(model downloads from Hugging Face on first run).")
                else:
                    label, prob_genuine, prob_bot = predict_comment(comment_text, tokenizer, model)

                    st.markdown("---")
                    st.subheader("Prediction Result")

                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        if label == "BOT_OR_SPAM":
                            st.error(f"Prediction: BOT / SPAM COMMENT\n\nConfidence: {prob_bot*100:.1f}%")
                        else:
                            st.success(f"Prediction: GENUINE COMMENT\n\nConfidence: {prob_genuine*100:.1f}%")
                    with res_col2:
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=prob_bot * 100,
                            title={'text': "Bot/Spam Probability (%)"},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "#7B2FF7"},
                                'steps': [
                                    {'range': [0, 50], 'color': "#2E7D32"},
                                    {'range': [50, 100], 'color': "#C2185B"}
                                ]
                            }
                        ))
                        fig_gauge.update_layout(height=250, margin=dict(t=30, b=10, l=30, r=30))
                        st.plotly_chart(fig_gauge, use_container_width=True)

    # --- TAB 3: BATCH CSV CHECKER ---
    with tab_batch:
        st.subheader("Classify a Batch of Comments")
        st.caption("Upload a CSV with a single text column (e.g. `comment`) to classify many comments at once.")

        uploaded_csv = st.file_uploader("Upload CSV", type=["csv"])
        text_col = st.text_input("Column name containing the comment text", value="comment")
        run_batch_btn = st.button("Run Batch Detection")

        if run_batch_btn:
            if uploaded_csv is None:
                st.warning("Please upload a CSV file first.")
            elif not TRANSFORMERS_AVAILABLE:
                st.error("`transformers` / `torch` are not installed — see Model Overview tab.")
            else:
                batch_df = pd.read_csv(uploaded_csv)
                if text_col not in batch_df.columns:
                    st.error(f"Column `{text_col}` not found. Available columns: {list(batch_df.columns)}")
                else:
                    tokenizer, model = load_comment_model()
                    if tokenizer is None or model is None:
                        st.error("Fake comment model could not be loaded.")
                    else:
                        progress = st.progress(0, text="Classifying comments...")
                        labels, probs_bot = [], []
                        texts = batch_df[text_col].fillna("").astype(str).tolist()
                        for i, t in enumerate(texts):
                            lbl, _, p_bot = predict_comment(t if t.strip() else " ", tokenizer, model)
                            labels.append(lbl)
                            probs_bot.append(round(p_bot * 100, 1))
                            progress.progress((i + 1) / len(texts), text=f"Classifying comments... {i+1}/{len(texts)}")
                        progress.empty()

                        batch_df["prediction"] = labels
                        batch_df["bot_spam_probability_%"] = probs_bot

                        st.markdown("---")
                        st.subheader("Results")

                        n_bot = int((batch_df["prediction"] == "BOT_OR_SPAM").sum())
                        n_genuine = len(batch_df) - n_bot
                        mcol1, mcol2 = st.columns(2)
                        with mcol1:
                            st.metric("Genuine Comments", n_genuine)
                        with mcol2:
                            st.metric("Bot / Spam Comments", n_bot)

                        st.dataframe(batch_df, use_container_width=True, hide_index=True)
                        st.download_button(
                            "Download Results as CSV",
                            batch_df.to_csv(index=False).encode("utf-8"),
                            "fake_comment_results.csv",
                            "text/csv"
                        )
