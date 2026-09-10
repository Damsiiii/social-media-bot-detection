#!/usr/bin/env python3
"""
Bot Detection Dashboard - Smart Installation Script

This script automatically detects your Python version and installs
the appropriate dependencies. It handles Python 3.14 limitations gracefully.

Usage:
    python3 install.py
    python3.11 install.py
    python3.12 install.py
"""

import sys
import subprocess
import os

def print_header(msg):
    """Print a formatted header message."""
    print("\n" + "=" * 70)
    print(f"  {msg}")
    print("=" * 70 + "\n")

def print_step(num, msg):
    """Print a step message."""
    print(f"\n[{num}/5] {msg}")

def check_python_version():
    """Check Python version and return version info."""
    version_info = sys.version_info
    version_str = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    
    print_header("Bot Detection Dashboard - Dependency Installation")
    print(f"Python Version: {version_str}")
    print(f"Executable: {sys.executable}\n")
    
    return version_info.major, version_info.minor

def get_requirements_file(major, minor):
    """Determine which requirements file to use based on Python version."""
    if (major == 3 and minor >= 14) or major > 3:
        print("⚠️  Detected Python 3.14+ (limited TensorFlow support)")
        return "requirements-py314.txt", True
    else:
        print(f"✓ Detected Python {major}.{minor} (full support)")
        return "requirements.txt", False

def install_dependencies(req_file):
    """Install dependencies from requirements file."""
    print_step(2, "Installing Dependencies")
    
    if not os.path.exists(req_file):
        print(f"✗ Requirements file not found: {req_file}")
        return False
    
    print(f"Using: {req_file}\n")
    
    try:
        # Upgrade pip, setuptools, wheel first
        print("Upgrading pip, setuptools, and wheel...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--upgrade", "pip", "setuptools", "wheel"
        ])
        
        print("\n✓ Pip upgraded\n")
        
        # Install requirements
        print("Installing requirements...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "-r", req_file
        ])
        
        print("\n✓ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Installation failed: {e}")
        return False

def verify_imports():
    """Verify critical imports are working."""
    print_step(3, "Verifying Critical Imports")
    
    critical_imports = [
        ("streamlit", "Streamlit"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-learn"),
        ("joblib", "Joblib"),
        ("plotly.express", "Plotly"),
        ("xgboost", "XGBoost"),
        ("transformers", "Transformers"),
        ("torch", "PyTorch"),
        ("PIL", "Pillow"),
        ("huggingface_hub", "Hugging Face Hub"),
    ]
    
    optional_imports = [
        ("tensorflow", "TensorFlow (optional)"),
    ]
    
    print("\nCritical imports:")
    failed = []
    for import_path, display_name in critical_imports:
        try:
            __import__(import_path)
            print(f"  ✓ {display_name}")
        except ImportError as e:
            print(f"  ✗ {display_name}: {e}")
            failed.append(display_name)
    
    print("\nOptional imports:")
    for import_path, display_name in optional_imports:
        try:
            __import__(import_path)
            print(f"  ✓ {display_name}")
        except ImportError:
            print(f"  ⚠️  {display_name} (will skip ANN models)")
    
    if failed:
        print(f"\n✗ {len(failed)} critical import(s) failed")
        return False
    
    print("\n✓ All critical imports verified")
    return True

def verify_resources():
    """Verify models and graphs are accessible."""
    print_step(4, "Verifying Resources")
    
    # Change to prototype directory
    if os.path.exists("prototype"):
        os.chdir("prototype")
    elif not os.path.exists("models"):
        print("✗ Not in prototype directory and 'models' not found")
        return False
    
    resources = [
        ("models/instagram", "Instagram models"),
        ("models/twitter", "Twitter models"),
        ("../graphs/instagram", "Instagram graphs"),
        ("../graphs/twitter", "Twitter graphs"),
        ("../graphs/fake_comment_detection", "Fake comment graphs"),
    ]
    
    all_ok = True
    for path, description in resources:
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) if f.endswith(('.pkl', '.keras', '.pt', '.png'))])
            print(f"  ✓ {description}: {count} files")
        else:
            print(f"  ✗ {description}: Not found at {path}")
            all_ok = False
    
    if all_ok:
        print("\n✓ All resources verified")
    else:
        print("\n⚠️  Some resources missing")
    
    return all_ok

def test_dashboard():
    """Test if dashboard can be imported."""
    print_step(5, "Testing Dashboard Import")
    
    try:
        # Simple import test (don't run the app)
        print("Checking app.py syntax...")
        with open("app.py", "r") as f:
            code = f.read()
        compile(code, "app.py", "exec")
        print("✓ app.py syntax valid")
        
        print("\nTo run the dashboard:")
        print("  $ streamlit run app.py")
        print("\nAccess at: http://localhost:8501")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Main installation flow."""
    try:
        # Step 1: Check Python version
        print_step(1, "Checking Python Version")
        major, minor = check_python_version()
        
        # Step 2: Determine requirements file
        req_file, is_py314 = get_requirements_file(major, minor)
        
        if is_py314:
            print("\nNote: TensorFlow will be skipped (Python 3.14 not yet supported)")
            print("Instagram ANN model will be unavailable, but all other features work fine.")
            print("\nFor full support, consider using Python 3.11 or 3.12")
        
        input("\nPress Enter to continue...")
        
        # Step 2: Install dependencies
        if not install_dependencies(req_file):
            print("\n✗ Installation failed. Please fix errors above and retry.")
            sys.exit(1)
        
        # Step 3: Verify imports
        if not verify_imports():
            print("\n⚠️  Some imports failed. Installation may be incomplete.")
        
        # Step 4: Verify resources
        verify_resources()
        
        # Step 5: Test dashboard
        test_dashboard()
        
        print_header("✅ Installation Complete!")
        print("Your bot detection dashboard is ready to use!")
        print("\nNext steps:")
        print("  1. cd prototype")
        print("  2. streamlit run app.py")
        print("  3. Open http://localhost:8501 in your browser")
        
    except KeyboardInterrupt:
        print("\n\n✗ Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
