# Installation Checklist

## Pre-Installation

- [ ] Check Python version: `python --version`
- [ ] Verify internet connection (needed for dependencies)
- [ ] Ensure you have at least 500MB free disk space
- [ ] Note: First run will download ~300MB for Hugging Face model

## Choose Your Path

### Path A: Auto-Install (Recommended)
- [ ] Navigate to prototype directory: `cd prototype`
- [ ] Run: `python3 install.py`
- [ ] Follow on-screen prompts
- [ ] Press Enter when ready
- [ ] Wait for installation to complete
- [ ] Check for ✅ All checks passed message

### Path B: Diagnose First
- [ ] Navigate to prototype directory: `cd prototype`
- [ ] Run: `python3 diagnose.py`
- [ ] Review the diagnostic report
- [ ] Note any ✗ failures
- [ ] Follow recommendations
- [ ] Run install.py
- [ ] Verify again with diagnose.py

### Path C: Manual Install (Python 3.11/3.12)
- [ ] Verify Python version is 3.11 or 3.12
- [ ] Navigate to prototype directory: `cd prototype`
- [ ] Create virtual environment: `python -m venv venv`
- [ ] Activate: `source venv/bin/activate`
- [ ] Upgrade pip: `pip install --upgrade pip`
- [ ] Install requirements: `pip install -r requirements.txt`
- [ ] Verify: `python -c "import streamlit; print('OK')"`

### Path D: Manual Install (Python 3.14)
- [ ] Navigate to prototype directory: `cd prototype`
- [ ] Create virtual environment: `python -m venv venv`
- [ ] Activate: `source venv/bin/activate`
- [ ] Upgrade pip: `pip install --upgrade pip`
- [ ] Install requirements: `pip install -r requirements-py314.txt`
- [ ] Note: ANN model will be unavailable
- [ ] Verify: `python -c "import streamlit; print('OK')"`

## Verification

- [ ] Run diagnostic: `cd prototype && python3 diagnose.py`
- [ ] Check output for all ✅ marks
- [ ] Verify Python version is supported
- [ ] Confirm all critical imports are available
- [ ] Verify models directory exists
- [ ] Verify graphs directory exists
- [ ] Network test should show Hugging Face reachable

## Dashboard Launch

- [ ] Navigate to prototype directory: `cd prototype`
- [ ] Ensure virtual environment is activated (if using one)
- [ ] Run: `streamlit run app.py`
- [ ] Wait for "You can now view your Streamlit app" message
- [ ] Note the URL (usually http://localhost:8501)
- [ ] Open URL in web browser
- [ ] Check all three tabs load:
  - [ ] Instagram Bot Detection
  - [ ] Twitter Bot Detection
  - [ ] Fake Comment Detection

## First Run Tests

### Instagram Tab
- [ ] Click on "Exploratory Data Analysis" tab
- [ ] Check 10+ graphs load
- [ ] Click on "Machine Learning Benchmarks" tab
- [ ] Check model performance charts
- [ ] Click on "Real-Time Predictor" tab
- [ ] Enter sample data and run prediction
- [ ] Verify result displays

### Twitter Tab
- [ ] Click on "Exploratory Data Analysis" tab
- [ ] Check 10+ graphs load
- [ ] Click on "Machine Learning Benchmarks" tab
- [ ] Check 5 model comparison charts
- [ ] Click on "Real-Time Predictor" tab
- [ ] Enter sample Twitter data and predict
- [ ] Verify ensemble voting works

### Fake Comment Tab
- [ ] Click on "Model Overview" tab
- [ ] Check 3 training graphs load
- [ ] Click on "Single Comment Checker" tab
- [ ] Enter test comment: "Follow me for free followers! 🔥"
- [ ] Check prediction shows "BOT / SPAM"
- [ ] Click on "Batch CSV Checker" tab
- [ ] Check upload interface appears

## Troubleshooting Checklist

If Installation Fails:
- [ ] Check Python version: `python --version`
- [ ] Run: `python3 diagnose.py` to identify issue
- [ ] Check internet connection
- [ ] Try upgrading pip: `pip install --upgrade pip`
- [ ] Clear pip cache: `pip cache purge`
- [ ] Try again with specific requirements file

If Dashboard Won't Start:
- [ ] Check for error message in terminal
- [ ] Verify port 8501 is available
- [ ] Try different port: `streamlit run app.py --server.port 8502`
- [ ] Run: `python3 diagnose.py` to check imports
- [ ] Restart terminal and try again

If Models Don't Load:
- [ ] Check models directory: `ls models/instagram/ models/twitter/`
- [ ] Run: `python3 diagnose.py` to verify
- [ ] Check error message in Streamlit UI
- [ ] Try to import specific model: `python -c "import joblib; joblib.load('models/instagram/random_forest.pkl')"`

If Graphs Don't Display:
- [ ] Check graphs directory: `ls ../graphs/instagram/ ../graphs/twitter/`
- [ ] Verify PNG files exist
- [ ] Check for file permission issues
- [ ] Run: `python3 diagnose.py` to verify paths

If Hugging Face Model Won't Download:
- [ ] Check internet connection
- [ ] Test Hugging Face: Visit https://huggingface.co
- [ ] Clear HF cache: `rm -rf ~/.cache/huggingface/`
- [ ] Try again (will re-download)

## Performance Verification

- [ ] First load time: 30-60 seconds
- [ ] Second load time: 5-10 seconds
- [ ] Model prediction: < 500ms
- [ ] Graph display: Instant (< 1 second)
- [ ] Check browser developer tools for errors
- [ ] Check terminal for warnings

## Cleanup (Optional)

If using virtual environment:
- [ ] To deactivate: `deactivate`
- [ ] To remove: `rm -rf venv` (requires reactivation next time)

If reinstalling:
- [ ] Clear pip cache: `pip cache purge`
- [ ] Remove venv: `rm -rf venv`
- [ ] Clear HF cache: `rm -rf ~/.cache/huggingface/`
- [ ] Start fresh installation

## Success Checklist

When everything is working, you should have:

- [ ] Python 3.11, 3.12, 3.13, or 3.14 installed
- [ ] All dependencies installed via pip
- [ ] Streamlit dashboard accessible at http://localhost:8501
- [ ] Instagram tab showing graphs and models
- [ ] Twitter tab showing graphs and models
- [ ] Fake Comment tab showing detector
- [ ] All predictions working correctly
- [ ] No error messages in dashboard
- [ ] Logs show all models loaded successfully

## Support

If you're still having issues:

1. **Read Documentation**
   - SETUP_GUIDE.md (comprehensive)
   - INSTALLATION_SUMMARY.md (detailed)
   - prototype/QUICK_START.md (quick ref)

2. **Run Diagnostics**
   - `python3 diagnose.py` (automated checks)
   - `python -m pip list` (check installed packages)
   - `python -c "import <package>"` (test imports)

3. **Check Common Issues**
   - Python version < 3.11? Update to 3.11+
   - Permission denied? Use `sudo` or check permissions
   - Port in use? Use `--server.port 8502`
   - Missing models? Run `ls prototype/models/*/`
   - Missing graphs? Run `ls graphs/*/`

4. **Contact or Report**
   - Check error message in terminal
   - Run diagnostic and save output
   - Review all setup documentation
   - Check if issue is with your system or dashboard

---

## Final Notes

✅ **Installation is straightforward** - Most users complete in 5-10 minutes  
✅ **Auto-installer handles most issues** - Use `python3 install.py`  
✅ **Diagnostic tool provides specific fixes** - Use `python3 diagnose.py`  
✅ **Full documentation available** - See SETUP_GUIDE.md  
✅ **Dashboard is production-ready** - Once installed, it just works!

Good luck! 🚀
