#!/usr/bin/env python3
"""
Bot Detection Dashboard - Diagnostic Tool

Run this to diagnose installation and environment issues.

Usage:
    python3 diagnose.py
"""

import sys
import os
import subprocess
import json
from pathlib import Path

def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def check_python_info():
    """Check Python information."""
    print_section("Python Environment")
    
    print(f"Executable: {sys.executable}")
    print(f"Version: {sys.version}")
    print(f"Version Info: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print(f"Platform: {sys.platform}")
    
    if sys.version_info >= (3, 14):
        print("\n⚠️  WARNING: Python 3.14 has limited package support")
        print("   TensorFlow is not available for Python 3.14 yet")
        print("   Consider using Python 3.11 or 3.12 for full support")
    elif sys.version_info >= (3, 11):
        print("\n✓ Python version fully supported")
    else:
        print(f"\n⚠️  WARNING: Python {sys.version_info.major}.{sys.version_info.minor} is quite old")
        print("   Consider upgrading to Python 3.11 or newer")

def check_pip():
    """Check pip version and installation."""
    print_section("Pip Information")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True, text=True, timeout=10
        )
        print(result.stdout)
        
        if result.returncode != 0:
            print(f"✗ Pip error: {result.stderr}")
            return False
        return True
        
    except Exception as e:
        print(f"✗ Error checking pip: {e}")
        return False

def check_installed_packages():
    """Check installed packages."""
    print_section("Installed Packages")
    
    critical = [
        "streamlit", "pandas", "numpy", "scikit-learn",
        "xgboost", "joblib", "plotly", "transformers",
        "torch", "PIL", "huggingface_hub"
    ]
    
    optional = [
        "tensorflow", "tensorflow-intel", "tensorflow-gpu"
    ]
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format", "json"],
            capture_output=True, text=True, timeout=30
        )
        
        if result.returncode != 0:
            print("✗ Could not get package list")
            return
        
        installed = {pkg["name"].lower(): pkg["version"] for pkg in json.loads(result.stdout)}
        
        print("Critical packages:")
        all_present = True
        for pkg in critical:
            if pkg.lower() in installed:
                print(f"  ✓ {pkg}: {installed[pkg.lower()]}")
            else:
                print(f"  ✗ {pkg}: NOT INSTALLED")
                all_present = False
        
        print("\nOptional packages:")
        for pkg in optional:
            if pkg.lower() in installed:
                print(f"  ✓ {pkg}: {installed[pkg.lower()]}")
            else:
                print(f"  ⚠️  {pkg}: not installed")
        
        return all_present
        
    except Exception as e:
        print(f"✗ Error checking packages: {e}")
        return False

def check_imports():
    """Check if critical imports work."""
    print_section("Module Imports")
    
    imports = {
        "streamlit": "Streamlit",
        "pandas": "Pandas",
        "numpy": "NumPy",
        "sklearn": "Scikit-learn",
        "joblib": "Joblib",
        "plotly.express": "Plotly",
        "xgboost": "XGBoost",
        "transformers": "Transformers",
        "torch": "PyTorch",
        "PIL": "Pillow",
        "huggingface_hub": "Hugging Face Hub",
        "tensorflow": "TensorFlow (optional)",
    }
    
    all_ok = True
    for module, name in imports.items():
        try:
            __import__(module)
            optional = "(optional)" in name
            prefix = "⚠️ " if optional else "✓"
            print(f"  {prefix} {name}")
        except ImportError as e:
            optional = "(optional)" in name
            prefix = "⚠️ " if optional else "✗"
            print(f"  {prefix} {name}: {e}")
            if not optional:
                all_ok = False
    
    return all_ok

def check_directory_structure():
    """Check project directory structure."""
    print_section("Directory Structure")
    
    # Try to find project root
    current = Path.cwd()
    project_root = None
    
    for parent in [current, current.parent, current.parent.parent]:
        if (parent / "prototype" / "models").exists():
            project_root = parent
            break
    
    if not project_root:
        print("✗ Could not find project root directory")
        print(f"  Current: {current}")
        print("  Expected: directory containing 'prototype/models/'")
        return False
    
    print(f"Project root: {project_root}")
    
    dirs_to_check = [
        ("prototype", "Main app directory"),
        ("prototype/models", "Models directory"),
        ("prototype/models/instagram", "Instagram models"),
        ("prototype/models/twitter", "Twitter models"),
        ("prototype/data", "Data directory"),
        ("graphs", "Graphs directory"),
        ("graphs/instagram", "Instagram graphs"),
        ("graphs/twitter", "Twitter graphs"),
        ("graphs/fake_comment_detection", "Fake comment graphs"),
    ]
    
    all_ok = True
    for path, desc in dirs_to_check:
        full_path = project_root / path
        if full_path.exists():
            if full_path.is_dir():
                count = len(list(full_path.glob("*")))
                print(f"  ✓ {desc}: {count} items")
            else:
                print(f"  ✓ {desc}")
        else:
            print(f"  ✗ {desc}: NOT FOUND")
            all_ok = False
    
    return all_ok

def check_app_py():
    """Check if app.py is valid."""
    print_section("Application File")
    
    try:
        app_path = Path("app.py")
        if not app_path.exists():
            app_path = Path("prototype/app.py")
        
        if not app_path.exists():
            print("✗ app.py not found")
            return False
        
        print(f"✓ Found: {app_path}")
        print(f"  Size: {app_path.stat().st_size} bytes")
        
        # Check syntax
        with open(app_path, "r") as f:
            code = f.read()
        compile(code, str(app_path), "exec")
        print("✓ Syntax valid")
        
        # Count lines
        lines = code.count("\n")
        print(f"  Lines: {lines}")
        
        return True
        
    except SyntaxError as e:
        print(f"✗ Syntax error in app.py: {e}")
        return False
    except Exception as e:
        print(f"✗ Error checking app.py: {e}")
        return False

def check_models():
    """Check model files."""
    print_section("Model Files")
    
    models_base = Path("prototype/models") if Path("prototype/models").exists() else Path("models")
    
    if not models_base.exists():
        print(f"✗ Models directory not found: {models_base}")
        return False
    
    all_ok = True
    for platform in ["instagram", "twitter"]:
        platform_dir = models_base / platform
        if platform_dir.exists():
            models = list(platform_dir.glob("*"))
            total_size = sum(m.stat().st_size for m in models) / (1024 * 1024)
            print(f"✓ {platform.capitalize()}: {len(models)} files ({total_size:.1f} MB)")
        else:
            print(f"✗ {platform.capitalize()}: directory not found")
            all_ok = False
    
    return all_ok

def check_graphs():
    """Check graph files."""
    print_section("Graph Files")
    
    graphs_base = Path("graphs") if Path("graphs").exists() else Path("../graphs")
    
    if not graphs_base.exists():
        print(f"✗ Graphs directory not found: {graphs_base}")
        return False
    
    all_ok = True
    for platform in ["instagram", "twitter", "fake_comment_detection"]:
        platform_dir = graphs_base / platform
        if platform_dir.exists():
            pngs = list(platform_dir.glob("*.png"))
            print(f"✓ {platform}: {len(pngs)} PNG files")
        else:
            print(f"✗ {platform}: directory not found")
            all_ok = False
    
    return all_ok

def check_network():
    """Check network connectivity for Hugging Face."""
    print_section("Network Connectivity")
    
    try:
        result = subprocess.run(
            ["ping", "-c", "1", "huggingface.co"],
            capture_output=True, timeout=5
        )
        if result.returncode == 0:
            print("✓ Hugging Face reachable (huggingface.co ping success)")
        else:
            print("⚠️  Could not reach huggingface.co (ping failed)")
            
    except Exception as e:
        print(f"⚠️  Network check failed: {e}")
    
    try:
        import urllib.request
        urllib.request.urlopen("https://huggingface.co", timeout=5)
        print("✓ HTTPS connection to Hugging Face works")
    except Exception as e:
        print(f"✗ Could not connect to Hugging Face: {e}")

def generate_recommendation():
    """Generate recommendation based on checks."""
    print_section("Recommendations")
    
    major, minor = sys.version_info.major, sys.version_info.minor
    
    if major == 3 and minor >= 14:
        print("Current Python: 3.14+")
        print("\n⚠️  RECOMMENDATION:")
        print("  Use Python 3.11 or 3.12 for full support")
        print("\n  To switch versions:")
        print("    python3.11 -m pip install -r requirements.txt")
        print("    python3.11 -m streamlit run prototype/app.py")
    elif major == 3 and minor >= 11:
        print("Current Python: 3.11+")
        print("\n✓ Your Python version is fully supported!")
        print("\nTo install:")
        print("  pip install -r requirements.txt")
        print("  streamlit run prototype/app.py")
    else:
        print("Current Python: 3.10 or older")
        print("\n⚠️  RECOMMENDATION:")
        print("  Upgrade to Python 3.11 or newer")

def main():
    """Run all diagnostics."""
    print("\n" + "=" * 70)
    print("  Bot Detection Dashboard - Diagnostic Report")
    print("=" * 70)
    
    try:
        # Run all checks
        check_python_info()
        check_pip()
        pkg_ok = check_installed_packages()
        imp_ok = check_imports()
        dir_ok = check_directory_structure()
        app_ok = check_app_py()
        mod_ok = check_models()
        graph_ok = check_graphs()
        check_network()
        
        # Summary
        print_section("Summary")
        
        checks = [
            ("Packages", pkg_ok),
            ("Imports", imp_ok),
            ("Directory Structure", dir_ok),
            ("app.py", app_ok),
            ("Models", mod_ok),
            ("Graphs", graph_ok),
        ]
        
        passed = sum(1 for _, ok in checks if ok)
        total = len(checks)
        
        print(f"\nChecks passed: {passed}/{total}")
        
        if passed == total:
            print("\n✅ All checks passed! Your installation is ready.")
            print("\nTo run the dashboard:")
            print("  cd prototype")
            print("  streamlit run app.py")
        else:
            print("\n⚠️  Some checks failed. See above for details.")
            print("\nTo fix issues:")
            print("  1. Check the errors above")
            print("  2. Run: python3 install.py")
            print("  3. Then try again")
        
        generate_recommendation()
        
    except Exception as e:
        print(f"\n✗ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
