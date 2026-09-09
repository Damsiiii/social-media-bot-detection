# 🛡️ Bot Sentinel — Social Media Bot Detection Dashboard

CCS4310 Final Project — Instagram + Twitter fake/bot account checker.

## 📁 Folder structure (ready-made — check nothing is missing)
```
bot_detection_dashboard/
├── app.py
├── requirements.txt
├── models/
│   ├── instagram/
│   │   ├── instagram_ann.keras
│   │   ├── xgboost.pkl              ⚠️ ADD THIS (see note below)
│   │   ├── random_forest.pkl        ⚠️ ADD THIS
│   │   └── logistic_regression.pkl  ⚠️ ADD THIS
│   └── twitter/
│       ├── xgboost.pkl
│       ├── random_forest.pkl
│       ├── logistic_regression.pkl
│       ├── nlp_model.pkl
│       ├── tfidf_vectorizer.pkl
│       └── gnn_model.pt
└── data/
    ├── instagram/
    │   ├── train.csv
    │   └── test.csv
    └── twitter/
        ├── gilani-2017.tsv
        └── gilani-2017_tweets.json
```

> ⚠️ **වැදගත්:** Instagram සහ Twitter වල `xgboost.pkl`, `random_forest.pkl`,
> `logistic_regression.pkl` file නම් **එකම නම්** තියෙන නිසා, ඔයා upload කරද්දී
> Instagram versions ටික accidentally overwrite උනා Twitter versions වලින්.
> ඒ නිසා **Instagram xgboost.pkl / random_forest.pkl / logistic_regression.pkl**
> Google Drive එකේ `Bot_Detection_Project/models/` folder එකෙන් ආයෙත් download
> කරලා `models/instagram/` folder එකට දාන්න (Twitter files වලින් වෙනස් කරලා).

## ▶️ Local එකේ Run කරන විදිහ
```bash
pip install -r requirements.txt
streamlit run app.py
```
Browser එකේ automatic විදිහට `http://localhost:8501` open වෙනවා.

## 🚀 Deploy කරන step-by-step (Streamlit Community Cloud — free)
1. GitHub account එකක් හදාගන්න (නැත්නම්).
2. `bot-detection-dashboard` කියලා **public repository** එකක් අලුතින් හදන්න.
3. මේ folder එකම (app.py, requirements.txt, models/, data/) GitHub repo එකට upload කරන්න.
4. https://share.streamlit.io වෙත ගිහින් GitHub account එකෙන් login වෙන්න.
5. **"New app"** click කරලා ඔයාගේ repo එක select කරන්න, main file path එකට `app.py` දෙන්න.
6. **Deploy** click කරන්න — විනාඩි කිහිපයකින් `xxx.streamlit.app` කියලා live link එකක් ලැබෙනවා.
7. ඒ link එක group එකට / lecturer ට share කරන්න.

## 📊 Models Included
| Platform  | Models |
|---|---|
| Instagram | Logistic Regression, Random Forest, XGBoost, ANN |
| Twitter   | Logistic Regression, Random Forest, XGBoost, TF-IDF+LR (NLP), GNN (offline results only) |

Note: GNN model එක dashboard එකේ live predict කරන්නේ නෑ (follower/following graph
data ඕන නිසා) — ඒ වෙනුවට Model Comparison page එකේ එහි offline accuracy score එක
(72.2%) පෙන්නනවා.
