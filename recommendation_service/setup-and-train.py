"""
setup-and-train.py
🚀 Complete Setup and Training Pipeline

This script will:
1. Check and install missing dependencies
2. Fix MongoDB index conflicts
3. Run the complete training pipeline
4. Generate a comprehensive report

Usage:
    python setup-and-train.py
    python setup-and-train.py --skip-install  # Skip dependency installation
    python setup-and-train.py --fix-indexes-only  # Only fix indexes
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def print_section(title):
    """Print a section header"""
    print("\n" + "="*80)
    print(f"🎯 {title}")
    print("="*80 + "\n")

def check_package(package_name):
    """Check if a package is installed"""
    try:
        __import__(package_name.replace("-", "_"))
        return True
    except ImportError:
        return False

def install_dependencies():
    """Install missing dependencies"""
    print_section("STEP 1: Installing Dependencies")
    
    # Core dependencies
    core_packages = [
        "pymongo",
        "python-dotenv",
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "tqdm",
        "joblib"
    ]
    
    # ML dependencies
    ml_packages = [
        "sentence-transformers",
        "torch",
        "torchvision",
        "transformers",
        "pillow",
        "requests"
    ]
    
    # Optional (can fail without breaking)
    optional_packages = [
        "spacy",
        "clip-by-openai"
    ]
    
    all_packages = core_packages + ml_packages
    missing = [pkg for pkg in all_packages if not check_package(pkg)]
    
    if not missing:
        print("✅ All core dependencies already installed!")
        return True
    
    print(f"📦 Missing packages: {', '.join(missing)}")
    print(f"📥 Installing {len(missing)} packages...\n")
    
    try:
        # Install in one go for efficiency
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + missing
        subprocess.check_call(cmd)
        print("\n✅ Core dependencies installed successfully!")
        
        # Try optional packages
        print("\n📦 Installing optional packages...")
        for pkg in optional_packages:
            try:
                print(f"  Installing {pkg}...")
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", pkg],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print(f"  ✅ {pkg} installed")
            except:
                print(f"  ⚠️  {pkg} installation failed (optional, continuing...)")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Installation failed: {e}")
        return False

def fix_mongodb_indexes():
    """Fix MongoDB index conflicts"""
    print_section("STEP 2: Fixing MongoDB Indexes")
    
    fix_script = Path(__file__).parent / "fix-indexes.py"
    
    if not fix_script.exists():
        print(f"⚠️  fix-indexes.py not found at {fix_script}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(fix_script)],
            check=True,
            capture_output=False
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Index fix failed: {e}")
        return False

def run_training_pipeline(args):
    """Run the training pipeline"""
    print_section("STEP 3: Running Training Pipeline")
    
    train_script = Path(__file__).parent / "train_all_models.py"
    
    if not train_script.exists():
        print(f"❌ train_all_models.py not found at {train_script}")
        return False
    
    # Build command
    cmd = [sys.executable, str(train_script)]
    
    if args.validate:
        cmd.append("--validate")
    if args.skip_text:
        cmd.append("--skip-text")
    if args.skip_hybrid:
        cmd.append("--skip-hybrid")
    if args.skip_images:
        cmd.append("--skip-images")
    
    print(f"🚀 Running: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Training pipeline failed: {e}")
        return False

def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description="Complete setup and training pipeline for recommendation system"
    )
    
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip dependency installation"
    )
    parser.add_argument(
        "--fix-indexes-only",
        action="store_true",
        help="Only fix MongoDB indexes and exit"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation after each training stage"
    )
    parser.add_argument(
        "--skip-text",
        action="store_true",
        help="Skip text embeddings generation"
    )
    parser.add_argument(
        "--skip-hybrid",
        action="store_true",
        help="Skip collaborative model training"
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip image embeddings generation"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🚀 RECOMMENDATION SYSTEM - COMPLETE SETUP & TRAINING")
    print("="*80)
    print("This script will:")
    print("  1. ✅ Install missing dependencies")
    print("  2. 🔧 Fix MongoDB index conflicts")
    print("  3. 🎯 Run complete training pipeline")
    print("="*80)
    
    try:
        # Step 1: Install dependencies
        if not args.skip_install:
            if not install_dependencies():
                print("\n❌ Dependency installation failed")
                return 1
        else:
            print("\n⏭️  Skipping dependency installation (--skip-install)")
        
        # Step 2: Fix indexes
        if not fix_mongodb_indexes():
            print("\n❌ Index fix failed")
            return 1
        
        if args.fix_indexes_only:
            print("\n✅ Index fix completed. Exiting (--fix-indexes-only)")
            return 0
        
        # Step 3: Run training
        if not run_training_pipeline(args):
            print("\n❌ Training pipeline failed")
            return 1
        
        # Success!
        print("\n" + "="*80)
        print("🎉 SETUP AND TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\n📋 Next steps:")
        print("  1. Check training_summary.json for results")
        print("  2. Verify output directories:")
        print("     - text_embeddings/")
        print("     - models/")
        print("     - image_embeddings/")
        print("  3. Restart your recommendation service:")
        print("     uvicorn recommendation_service_enhanced:app --reload")
        print("  4. Test with: http://localhost:8000/health?deep=true")
        print("="*80 + "\n")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
