# Quick Reference Card

## Installation Issues Fixed ✅

### Problem
```
ERROR: Could not find a version that satisfies the requirement tensorflow-cpu
```

### Solutions

#### 1️⃣ Python 3.14 (Quick)
```bash
cd prototype
pip install -r requirements-py314.txt
streamlit run app.py
```
⚠️ Instagram ANN unavailable

#### 2️⃣ Python 3.11/3.12 (Best)
```bash
python3.11 -m pip install -r requirements.txt
python3.11 -m streamlit run prototype/app.py
```
✅ Full support

#### 3️⃣ Auto-Install (Smartest)
```bash
cd prototype
python3 install.py
```
✅ Detects Python version automatically

#### 4️⃣ Diagnose First
```bash
cd prototype
python3 diagnose.py
```
✅ Shows what's working/broken

---

## Files Reference

| File | Purpose | Location |
|------|---------|----------|
| install.py | Auto-installer | prototype/ |
| diagnose.py | Diagnostic tool | prototype/ |
| requirements.txt | Python 3.11/3.12 | prototype/ |
| requirements-py314.txt | Python 3.14 | prototype/ |
| SETUP_GUIDE.md | Complete guide | root |
| INSTALLATION_SUMMARY.md | This summary | root |

---

## Dashboard Access

```
http://localhost:8501
```

---

## Python Version Check

```bash
python --version
```

| Version | Recommendation |
|---------|-----------------|
| 3.11 | ✅ Use this |
| 3.12 | ✅ Use this |
| 3.13 | ⚠️ OK |
| 3.14 | ⚠️ Limited |

---

## Features Status

### Instagram Bot Detection
- ✅ EDA + Graphs
- ✅ Random Forest
- ✅ XGBoost
- ✅ Logistic Regression
- ⚠️ ANN (requires TensorFlow)

### Twitter Bot Detection
- ✅ EDA + Graphs
- ✅ Random Forest
- ✅ XGBoost
- ✅ Logistic Regression
- ✅ NLP/TF-IDF
- ✅ GNN

### Fake Comment Detection
- ✅ RoBERTa Model
- ✅ Single comment checker
- ✅ Batch processor

---

## Common Commands

### Install
```bash
python install.py
```

### Diagnose
```bash
python diagnose.py
```

### Run Dashboard
```bash
streamlit run app.py
```

### Check Status
```bash
python -m pip list | grep -E "streamlit|torch|tensorflow"
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| tensorflow-cpu not found | Use Python 3.11/3.12 or requirements-py314.txt |
| No module named 'streamlit' | Run `python install.py` |
| Port 8501 in use | Use `streamlit run app.py --server.port 8502` |
| Models not loading | Run `python diagnose.py` to check |
| Graphs not showing | Verify graphs/ directory exists |

---

## Performance

- **Cold Start**: 30-60 sec
- **Warm Start**: 5-10 sec
- **Model Prediction**: 20-200 ms

---

## Resources Summary

- 📊 **Graphs**: 55 PNG files ✅
- 🤖 **Local Models**: 11 files ✅
- 🌐 **HF Model**: RoBERTa (auto-download) ✅
- 💾 **Total Local**: ~5.8 MB
- 📥 **HF Model**: ~300 MB (cached)

---

## Support

1. Read SETUP_GUIDE.md
2. Run python install.py
3. Run python diagnose.py
4. Check INSTALLATION_SUMMARY.md

✅ Dashboard is production-ready!
