"""
train_all_models.py
✅ Master Training Orchestrator - ONE COMMAND TO TRAIN THEM ALL!

Purpose:
- Run all 3 training scripts in optimal sequence
- Handle dependencies and errors gracefully
- Show progress and statistics
- Generate comprehensive training report
- Validate all outputs

Training Pipeline:
1. 📝 Generate Text Embeddings (SentenceTransformer)
2. 🤝 Train Collaborative Model (SVD)
3. 🖼️ Generate Image Embeddings (CLIP)

Features:
- Sequential execution with error handling
- Progress tracking for each stage
- Dependency checking
- Comprehensive validation
- Training summary report
- Environment variable support
- Dry-run mode for testing

Usage:
    # Full training pipeline
    python train_all_models.py
    
    # Dry run (check without training)
    python train_all_models.py --dry-run
    
    # Skip specific stages
    python train_all_models.py --skip-text --skip-images
    
    # With validation
    python train_all_models.py --validate

Output:
    text_embeddings/      (text embeddings)
    models/               (collaborative model)
    image_embeddings/     (image embeddings)
    training_summary.json (complete report)
"""

import os
import sys
import argparse
import logging
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from dotenv import load_dotenv

# ==========================================
# Configuration
# ==========================================
load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('master_training.log')
    ]
)
logger = logging.getLogger("MasterTrainer")

# ==========================================
# Training Stage Configuration
# ==========================================
STAGES = {
    "text_embeddings": {
        "name": "Text Embeddings Generation",
        "script": "generate_text_embeddings.py",
        "output_dir": "text_embeddings",
        "output_files": ["manifest.json"],
        "icon": "📝",
        "dependencies": ["sentence-transformers", "pymongo", "numpy", "tqdm"]
    },
    "hybrid_model": {
        "name": "Collaborative Model Training",
        "script": "train_hybrid_model.py",
        "output_dir": "models",
        "output_files": ["hybrid_model.joblib", "training_report.json"],
        "icon": "🤝",
        "dependencies": ["scikit-learn", "scipy", "pandas", "pymongo", "joblib"]
    },
    "image_embeddings": {
        "name": "Image Embeddings Generation",
        "script": "generate_image_embeddings.py",
        "output_dir": "image_embeddings",
        "output_files": ["manifest.json"],
        "icon": "🖼️",
        "dependencies": ["torch", "torchvision", "clip", "pillow", "requests"]
    }
}

# ==========================================
# Dependency Checking
# ==========================================
def check_dependencies() -> Dict[str, bool]:
    """Check if required Python packages are installed"""
    logger.info("🔍 Checking dependencies...")
    
    all_deps = set()
    for stage_config in STAGES.values():
        all_deps.update(stage_config["dependencies"])
    
    results = {}
    missing = []
    
    for package in sorted(all_deps):
        try:
            __import__(package.replace("-", "_"))
            results[package] = True
            logger.info(f"  ✓ {package}")
        except ImportError:
            results[package] = False
            missing.append(package)
            logger.warning(f"  ✗ {package} - NOT INSTALLED")
    
    if missing:
        logger.warning(f"\n⚠️  Missing packages: {', '.join(missing)}")
        logger.info(f"\n💡 Install with: pip install {' '.join(missing)}")
        return results
    
    logger.info("✅ All dependencies satisfied!")
    return results


def check_environment_variables() -> Dict[str, bool]:
    """Check if required environment variables are set"""
    logger.info("\n🔍 Checking environment variables...")
    
    required_vars = [
        "MONGO_URI",
        "MONGO_DB_NAME",
        "MONGO_COLLECTION_PRODUCTS",
        "MONGO_COLLECTION_INTERACTIONS"
    ]
    
    optional_vars = [
        "TEXT_MODEL_NAME",
        "IMAGE_MODEL_NAME",
        "LATENT_FEATURES",
        "MIN_INTERACTIONS"
    ]
    
    results = {}
    
    # Check required
    for var in required_vars:
        value = os.getenv(var)
        results[var] = bool(value)
        if value:
            # Mask sensitive parts
            display_value = value[:20] + "..." if len(value) > 20 else value
            if "URI" in var or "PASSWORD" in var:
                display_value = "***" + value[-10:] if len(value) > 10 else "***"
            logger.info(f"  ✓ {var}: {display_value}")
        else:
            logger.warning(f"  ✗ {var}: NOT SET")
    
    # Check optional
    for var in optional_vars:
        value = os.getenv(var)
        results[var] = bool(value)
        if value:
            logger.info(f"  ℹ {var}: {value}")
    
    missing = [var for var in required_vars if not results[var]]
    if missing:
        logger.error(f"\n❌ Missing required variables: {', '.join(missing)}")
        logger.info("💡 Create a .env file with these variables")
        return results
    
    logger.info("✅ All required environment variables set!")
    return results


def check_mongodb_connection() -> bool:
    """Test MongoDB connection"""
    logger.info("\n🔍 Testing MongoDB connection...")
    
    try:
        from pymongo import MongoClient
        from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure
        
        mongo_uri = os.getenv("MONGO_URI")
        if not mongo_uri:
            logger.error("❌ MONGO_URI not set")
            return False
        
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        
        # Check collections
        db_name = os.getenv("MONGO_DB_NAME", "futurist_ecommerce")
        db = client[db_name]
        
        products_col = os.getenv("MONGO_COLLECTION_PRODUCTS", "products")
        interactions_col = os.getenv("MONGO_COLLECTION_INTERACTIONS", "interaction_logs")
        
        product_count = db[products_col].count_documents({})
        interaction_count = db[interactions_col].count_documents({})
        
        logger.info(f"  ✓ MongoDB connection: OK")
        logger.info(f"  ✓ Database: {db_name}")
        logger.info(f"  ✓ Products: {product_count:,}")
        logger.info(f"  ✓ Interactions: {interaction_count:,}")
        
        client.close()
        
        if product_count == 0:
            logger.warning("⚠️  No products found in database!")
            return False
        
        logger.info("✅ MongoDB connection successful!")
        return True
        
    except Exception as e:
        logger.error(f"❌ MongoDB connection failed: {e}")
        return False


# ==========================================
# Stage Execution
# ==========================================
def run_training_stage(
    stage_key: str,
    validate: bool = False,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Run a single training stage
    
    Returns:
        Dictionary with stage results
    """
    stage = STAGES[stage_key]
    icon = stage["icon"]
    name = stage["name"]
    script = stage["script"]
    
    logger.info("\n" + "=" * 80)
    logger.info(f"{icon} STAGE: {name}")
    logger.info("=" * 80)
    
    result = {
        "stage": stage_key,
        "name": name,
        "script": script,
        "success": False,
        "duration": 0,
        "output_files": [],
        "error": None
    }
    
    if dry_run:
        logger.info("🔍 DRY RUN - Skipping actual execution")
        result["success"] = True
        result["dry_run"] = True
        return result
    
    # Check if script exists
    script_path = Path(__file__).parent / script
    if not script_path.exists():
        error_msg = f"Script not found: {script_path}"
        logger.error(f"❌ {error_msg}")
        result["error"] = error_msg
        return result
    
    # Build command
    cmd = [sys.executable, str(script_path)]
    if validate:
        cmd.append("--validate")
    
    logger.info(f"🚀 Running: {' '.join(cmd)}")
    logger.info(f"📂 Output directory: {stage['output_dir']}")
    
    # Execute script
    start_time = time.time()
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        duration = time.time() - start_time
        result["duration"] = duration
        
        if process.returncode == 0:
            logger.info(f"✅ {name} completed in {duration:.2f} seconds")
            result["success"] = True
            
            # Check output files
            output_dir = Path(stage["output_dir"])
            if output_dir.exists():
                for filename in stage["output_files"]:
                    file_path = output_dir / filename
                    if file_path.exists():
                        result["output_files"].append(str(file_path))
                        logger.info(f"  ✓ Created: {file_path}")
                    else:
                        logger.warning(f"  ⚠️  Expected file not found: {file_path}")
        else:
            error_msg = f"Script exited with code {process.returncode}"
            logger.error(f"❌ {name} failed: {error_msg}")
            result["error"] = error_msg
            
    except Exception as e:
        duration = time.time() - start_time
        result["duration"] = duration
        error_msg = str(e)
        logger.error(f"❌ {name} error: {error_msg}")
        result["error"] = error_msg
    
    return result


# ==========================================
# Training Pipeline
# ==========================================
def run_training_pipeline(
    skip_stages: List[str] = None,
    validate: bool = False,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Run the complete training pipeline
    
    Args:
        skip_stages: List of stage keys to skip
        validate: Run validation after each stage
        dry_run: Check setup without running
        
    Returns:
        Complete training report
    """
    skip_stages = skip_stages or []
    
    logger.info("\n" + "=" * 80)
    logger.info("🚀 MASTER TRAINING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'FULL TRAINING'}")
    logger.info(f"Validation: {'ENABLED' if validate else 'DISABLED'}")
    if skip_stages:
        logger.info(f"Skipping: {', '.join(skip_stages)}")
    logger.info("=" * 80)
    
    pipeline_start = time.time()
    
    report = {
        "pipeline": "Master Training Pipeline",
        "started_at": datetime.utcnow().isoformat(),
        "mode": "dry_run" if dry_run else "full_training",
        "validation_enabled": validate,
        "stages": {},
        "summary": {}
    }
    
    # Stage execution order
    stage_order = ["text_embeddings", "hybrid_model", "image_embeddings"]
    
    for stage_key in stage_order:
        if stage_key in skip_stages:
            logger.info(f"\n⏭️  Skipping {STAGES[stage_key]['icon']} {STAGES[stage_key]['name']}")
            report["stages"][stage_key] = {"skipped": True}
            continue
        
        result = run_training_stage(stage_key, validate=validate, dry_run=dry_run)
        report["stages"][stage_key] = result
        
        if not result["success"] and not dry_run:
            logger.error(f"\n❌ Pipeline stopped due to failure in stage: {stage_key}")
            break
    
    pipeline_duration = time.time() - pipeline_start
    
    # Generate summary
    total_stages = len(stage_order)
    skipped_stages = len([s for s in skip_stages if s in stage_order])
    successful_stages = len([r for r in report["stages"].values() if r.get("success")])
    failed_stages = len([r for r in report["stages"].values() if r.get("error")])
    
    report["summary"] = {
        "total_stages": total_stages,
        "skipped": skipped_stages,
        "executed": total_stages - skipped_stages,
        "successful": successful_stages,
        "failed": failed_stages,
        "total_duration": pipeline_duration,
        "completed_at": datetime.utcnow().isoformat()
    }
    
    return report


# ==========================================
# Report Generation
# ==========================================
def print_training_report(report: Dict[str, Any]):
    """Print a beautiful training summary"""
    logger.info("\n" + "=" * 80)
    logger.info("📊 TRAINING PIPELINE SUMMARY")
    logger.info("=" * 80)
    
    summary = report["summary"]
    
    logger.info(f"Started: {report['started_at']}")
    logger.info(f"Completed: {summary['completed_at']}")
    logger.info(f"Total Duration: {summary['total_duration']:.2f} seconds")
    logger.info("")
    
    logger.info("Stage Results:")
    for stage_key, result in report["stages"].items():
        stage = STAGES[stage_key]
        icon = stage["icon"]
        name = stage["name"]
        
        if result.get("skipped"):
            logger.info(f"  ⏭️  {icon} {name}: SKIPPED")
        elif result.get("success"):
            duration = result.get("duration", 0)
            logger.info(f"  ✅ {icon} {name}: SUCCESS ({duration:.2f}s)")
            if result.get("output_files"):
                for file in result["output_files"]:
                    logger.info(f"      → {file}")
        else:
            error = result.get("error", "Unknown error")
            logger.info(f"  ❌ {icon} {name}: FAILED - {error}")
    
    logger.info("")
    logger.info("Summary:")
    logger.info(f"  Total Stages: {summary['total_stages']}")
    logger.info(f"  Executed: {summary['executed']}")
    logger.info(f"  Successful: {summary['successful']}")
    logger.info(f"  Failed: {summary['failed']}")
    logger.info(f"  Skipped: {summary['skipped']}")
    logger.info("")
    
    if summary['failed'] == 0 and summary['successful'] > 0:
        logger.info("🎉 ALL STAGES COMPLETED SUCCESSFULLY!")
    elif summary['failed'] > 0:
        logger.warning("⚠️  SOME STAGES FAILED - Check logs for details")
    
    logger.info("=" * 80)


def save_training_report(report: Dict[str, Any], output_path: Path = Path("training_summary.json")):
    """Save training report to JSON"""
    try:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"📄 Training report saved: {output_path}")
    except Exception as e:
        logger.error(f"⚠️  Failed to save report: {e}")


# ==========================================
# Pre-flight Checks
# ==========================================
def run_preflight_checks() -> bool:
    """Run all pre-flight checks before training"""
    logger.info("\n" + "=" * 80)
    logger.info("🔍 PRE-FLIGHT CHECKS")
    logger.info("=" * 80)
    
    all_passed = True
    
    # Check 1: Dependencies
    deps_result = check_dependencies()
    if not all(deps_result.values()):
        all_passed = False
    
    # Check 2: Environment Variables
    env_result = check_environment_variables()
    required_vars = ["MONGO_URI", "MONGO_DB_NAME", "MONGO_COLLECTION_PRODUCTS", "MONGO_COLLECTION_INTERACTIONS"]
    if not all(env_result.get(var, False) for var in required_vars):
        all_passed = False
    
    # Check 3: MongoDB Connection
    if not check_mongodb_connection():
        all_passed = False
    
    logger.info("\n" + "=" * 80)
    if all_passed:
        logger.info("✅ ALL PRE-FLIGHT CHECKS PASSED!")
        logger.info("🚀 Ready to start training!")
    else:
        logger.error("❌ SOME PRE-FLIGHT CHECKS FAILED!")
        logger.info("💡 Fix the issues above before running training")
    logger.info("=" * 80)
    
    return all_passed


# ==========================================
# Main Function
# ==========================================
def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Master training orchestrator for recommendation system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full training pipeline
  python train_all_models.py
  
  # Dry run (check setup without training)
  python train_all_models.py --dry-run
  
  # Skip specific stages
  python train_all_models.py --skip-text
  python train_all_models.py --skip-images
  
  # With validation
  python train_all_models.py --validate
  
  # Skip checks and run directly (not recommended)
  python train_all_models.py --skip-checks
        """
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pre-flight checks without executing training"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation after each stage"
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip pre-flight checks (not recommended)"
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
    parser.add_argument(
        "--output-report",
        type=str,
        default="training_summary.json",
        help="Output path for training report"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("🎯 MASTER TRAINING ORCHESTRATOR")
    logger.info("=" * 80)
    logger.info("Version: 1.0.0")
    logger.info("Purpose: Train all recommendation models in one command")
    logger.info("=" * 80)
    
    try:
        # Pre-flight checks
        if not args.skip_checks:
            checks_passed = run_preflight_checks()
            
            if not checks_passed and not args.dry_run:
                logger.error("\n❌ Pre-flight checks failed. Fix issues before training.")
                logger.info("💡 Use --dry-run to see what would happen")
                logger.info("💡 Use --skip-checks to bypass (not recommended)")
                return 1
        else:
            logger.warning("⚠️  Skipping pre-flight checks (--skip-checks)")
        
        if args.dry_run:
            logger.info("\n✅ Dry run completed. Ready for actual training!")
            logger.info("💡 Run without --dry-run to start training")
            return 0
        
        # Build skip list
        skip_stages = []
        if args.skip_text:
            skip_stages.append("text_embeddings")
        if args.skip_hybrid:
            skip_stages.append("hybrid_model")
        if args.skip_images:
            skip_stages.append("image_embeddings")
        
        # Run training pipeline
        report = run_training_pipeline(
            skip_stages=skip_stages,
            validate=args.validate,
            dry_run=False
        )
        
        # Print and save report
        print_training_report(report)
        save_training_report(report, Path(args.output_report))
        
        # Return exit code
        if report["summary"]["failed"] > 0:
            logger.error("\n❌ Training pipeline completed with errors")
            return 1
        else:
            logger.info("\n🎉 Training pipeline completed successfully!")
            logger.info("\n🎯 Next steps:")
            logger.info("  1. Upload trained models to your server:")
            logger.info("     - text_embeddings/")
            logger.info("     - models/")
            logger.info("     - image_embeddings/")
            logger.info("  2. Set LOAD_ML_MODELS=true on server")
            logger.info("  3. Restart recommendation service")
            logger.info("  4. Test with /health?deep=true")
            return 0
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Training interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
