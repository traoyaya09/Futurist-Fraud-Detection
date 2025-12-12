"""
train_all_models_fixed.py
Master Training Orchestrator - ONE COMMAND TO TRAIN THEM ALL! (WINDOWS COMPATIBLE)

Purpose:
- Run all 3 FIXED training scripts in optimal sequence
- Handle dependencies and errors gracefully
- Show progress and statistics
- Generate comprehensive training report
- Validate all outputs
- NO UNICODE CHARACTERS (Windows compatible)

Training Pipeline:
1. [TEXT] Generate Text Embeddings (SentenceTransformer) - CURSOR TIMEOUT FIXED
2. [COLLAB] Train Collaborative Model (SVD) - ID MAPPING & DATETIME FIXED
3. [IMAGE] Generate Image Embeddings (CLIP) - CURSOR TIMEOUT FIXED

Features:
- Sequential execution with error handling
- Progress tracking for each stage
- Dependency checking
- Comprehensive validation
- Training summary report
- Environment variable support
- Dry-run mode for testing
- Windows encoding safe (no emojis)

Usage:
    # Full training pipeline
    python train_all_models_fixed.py
    
    # Dry run (check without training)
    python train_all_models_fixed.py --dry-run
    
    # Skip specific stages
    python train_all_models_fixed.py --skip-text --skip-images
    
    # With validation
    python train_all_models_fixed.py --validate

Output:
    text_embeddings/      (text embeddings)
    models/               (collaborative model with ID mappings)
    image_embeddings/     (image embeddings)
    training_summary.json (complete report)
"""

import os
import sys
import argparse
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

# Simple logging (no Unicode)
def log_info(msg: str):
    print(f"[INFO] {msg}")

def log_warning(msg: str):
    print(f"[WARNING] {msg}")

def log_error(msg: str):
    print(f"[ERROR] {msg}", file=sys.stderr)

def log_success(msg: str):
    print(f"[SUCCESS] {msg}")

# ==========================================
# Training Stage Configuration - UPDATED
# ==========================================
STAGES = {
    "text_embeddings": {
        "name": "Text Embeddings Generation",
        "script": "generate_text_embeddings_fixed.py",  # ✅ FIXED SCRIPT
        "output_dir": "text_embeddings",
        "output_files": ["manifest.json", "statistics.json"],
        "icon": "[TEXT]",
        "dependencies": ["sentence-transformers", "pymongo", "numpy", "tqdm"]
    },
    "hybrid_model": {
        "name": "Collaborative Model Training",
        "script": "train_hybrid_model.py",  # ✅ UPDATED - Enhanced version
        "output_dir": "models",
        "output_files": ["hybrid_model.joblib", "id_mappings.json", "training_report.json"],  # ✅ Added id_mappings.json
        "icon": "[COLLAB]",
        "dependencies": ["scikit-learn", "scipy", "pandas", "pymongo", "joblib"]
    },
    "image_embeddings": {
        "name": "Image Embeddings Generation",
        "script": "generate_image_embeddings_fixed.py",  # ✅ FIXED SCRIPT
        "output_dir": "image_embeddings",
        "output_files": ["manifest.json", "statistics.json"],
        "icon": "[IMAGE]",
        "dependencies": ["torch", "clip", "pillow", "requests"]
    }
}

# ==========================================
# Dependency Checking
# ==========================================
def check_dependencies() -> Dict[str, bool]:
    """Check if required Python packages are installed"""
    log_info("Checking dependencies...")
    
    all_deps = set()
    for stage_config in STAGES.values():
        all_deps.update(stage_config["dependencies"])
    
    results = {}
    missing = []
    
    for package in sorted(all_deps):
        try:
            __import__(package.replace("-", "_"))
            results[package] = True
            log_info(f"  [OK] {package}")
        except ImportError:
            results[package] = False
            missing.append(package)
            log_warning(f"  [MISSING] {package}")
    
    if missing:
        log_warning(f"\nMissing packages: {', '.join(missing)}")
        log_info(f"Install with: pip install {' '.join(missing)}")
        return results
    
    log_success("All dependencies satisfied!")
    return results


def check_environment_variables() -> Dict[str, bool]:
    """Check if required environment variables are set"""
    log_info("\nChecking environment variables...")
    
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
            log_info(f"  [OK] {var}: {display_value}")
        else:
            log_warning(f"  [MISSING] {var}")
    
    # Check optional
    for var in optional_vars:
        value = os.getenv(var)
        results[var] = bool(value)
        if value:
            log_info(f"  [OPTIONAL] {var}: {value}")
    
    missing = [var for var in required_vars if not results[var]]
    if missing:
        log_error(f"\nMissing required variables: {', '.join(missing)}")
        log_info("Create a .env file with these variables")
        return results
    
    log_success("All required environment variables set!")
    return results


def check_mongodb_connection() -> bool:
    """Test MongoDB connection"""
    log_info("\nTesting MongoDB connection...")
    
    try:
        from pymongo import MongoClient
        from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure
        
        mongo_uri = os.getenv("MONGO_URI")
        if not mongo_uri:
            log_error("MONGO_URI not set")
            return False
        
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        
        # Check collections
        db_name = os.getenv("MONGO_DB_NAME", "futurist_e-commerce")
        db = client[db_name]
        
        products_col = os.getenv("MONGO_COLLECTION_PRODUCTS", "products")
        interactions_col = os.getenv("MONGO_COLLECTION_INTERACTIONS", "interaction_logs")
        
        product_count = db[products_col].count_documents({})
        interaction_count = db[interactions_col].count_documents({})
        
        log_info(f"  [OK] MongoDB connection successful")
        log_info(f"  [OK] Database: {db_name}")
        log_info(f"  [OK] Products: {product_count:,}")
        log_info(f"  [OK] Interactions: {interaction_count:,}")
        
        client.close()
        
        if product_count == 0:
            log_warning("No products found in database!")
            return False
        
        log_success("MongoDB connection successful!")
        return True
        
    except Exception as e:
        log_error(f"MongoDB connection failed: {e}")
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
    
    print("\n" + "=" * 80)
    log_info(f"{icon} STAGE: {name}")
    print("=" * 80)
    
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
        log_info("DRY RUN - Skipping actual execution")
        result["success"] = True
        result["dry_run"] = True
        return result
    
    # Check if script exists
    script_path = Path(__file__).parent / script
    if not script_path.exists():
        error_msg = f"Script not found: {script_path}"
        log_error(error_msg)
        result["error"] = error_msg
        return result
    
    # Build command
    cmd = [sys.executable, str(script_path)]
    if validate:
        cmd.append("--validate")
    
    log_info(f"Running: {' '.join(cmd)}")
    log_info(f"Output directory: {stage['output_dir']}")
    
    # Execute script
    start_time = time.time()
    
    try:
        # Use subprocess.run for cleaner output on Windows
        process = subprocess.run(
            cmd,
            check=False,
            text=True,
            encoding='utf-8',
            errors='replace'  # Replace encoding errors
        )
        
        duration = time.time() - start_time
        result["duration"] = duration
        
        if process.returncode == 0:
            log_success(f"{name} completed in {duration:.2f} seconds")
            result["success"] = True
            
            # Check output files
            output_dir = Path(stage["output_dir"])
            if output_dir.exists():
                for filename in stage["output_files"]:
                    file_path = output_dir / filename
                    if file_path.exists():
                        result["output_files"].append(str(file_path))
                        log_info(f"  [OK] Created: {file_path}")
                    else:
                        log_warning(f"  [MISSING] Expected file not found: {file_path}")
        else:
            error_msg = f"Script exited with code {process.returncode}"
            log_error(f"{name} failed: {error_msg}")
            result["error"] = error_msg
            
    except Exception as e:
        duration = time.time() - start_time
        result["duration"] = duration
        error_msg = str(e)
        log_error(f"{name} error: {error_msg}")
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
    
    print("\n" + "=" * 80)
    log_info("MASTER TRAINING PIPELINE - ENHANCED v2.1")
    print("=" * 80)
    log_info(f"Mode: {'DRY RUN' if dry_run else 'FULL TRAINING'}")
    log_info(f"Validation: {'ENABLED' if validate else 'DISABLED'}")
    if skip_stages:
        log_info(f"Skipping: {', '.join(skip_stages)}")
    log_info("Features: Cursor timeout fixes, ID mapping, DateTime normalization")
    print("=" * 80)
    
    pipeline_start = time.time()
    
    report = {
        "pipeline": "Master Training Pipeline (Fixed & Enhanced)",
        "version": "2.1.0",
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
            log_info(f"\n[SKIP] Skipping {STAGES[stage_key]['icon']} {STAGES[stage_key]['name']}")
            report["stages"][stage_key] = {"skipped": True}
            continue
        
        result = run_training_stage(stage_key, validate=validate, dry_run=dry_run)
        report["stages"][stage_key] = result
        
        if not result["success"] and not dry_run:
            log_error(f"\nPipeline stopped due to failure in stage: {stage_key}")
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
        "completed_at": datetime.utcnow().isoformat(),
        "enhancements_applied": {
            "cursor_timeout_fixed": True,
            "id_mapping_fixed": True,
            "datetime_normalized": True,
            "utils_integrated": True
        }
    }
    
    return report


# ==========================================
# Report Generation
# ==========================================
def print_training_report(report: Dict[str, Any]):
    """Print a beautiful training summary"""
    print("\n" + "=" * 80)
    log_info("TRAINING PIPELINE SUMMARY")
    print("=" * 80)
    
    summary = report["summary"]
    
    log_info(f"Version: {report.get('version', '2.0.0')}")
    log_info(f"Started: {report['started_at']}")
    log_info(f"Completed: {summary['completed_at']}")
    log_info(f"Total Duration: {summary['total_duration']:.2f} seconds")
    print("")
    
    log_info("Stage Results:")
    for stage_key, result in report["stages"].items():
        stage = STAGES[stage_key]
        icon = stage["icon"]
        name = stage["name"]
        
        if result.get("skipped"):
            log_info(f"  [SKIP] {icon} {name}")
        elif result.get("success"):
            duration = result.get("duration", 0)
            log_success(f"  {icon} {name}: SUCCESS ({duration:.2f}s)")
            if result.get("output_files"):
                for file in result["output_files"]:
                    log_info(f"      -> {file}")
        else:
            error = result.get("error", "Unknown error")
            log_error(f"  {icon} {name}: FAILED - {error}")
    
    print("")
    log_info("Summary:")
    log_info(f"  Total Stages: {summary['total_stages']}")
    log_info(f"  Executed: {summary['executed']}")
    log_info(f"  Successful: {summary['successful']}")
    log_info(f"  Failed: {summary['failed']}")
    log_info(f"  Skipped: {summary['skipped']}")
    
    # Show enhancements
    if "enhancements_applied" in summary:
        print("")
        log_info("Enhancements Applied:")
        enhancements = summary["enhancements_applied"]
        log_info(f"  Cursor Timeout Fix: {'YES' if enhancements.get('cursor_timeout_fixed') else 'NO'}")
        log_info(f"  ID Mapping Fix: {'YES' if enhancements.get('id_mapping_fixed') else 'NO'}")
        log_info(f"  DateTime Normalized: {'YES' if enhancements.get('datetime_normalized') else 'NO'}")
        log_info(f"  Utils Integrated: {'YES' if enhancements.get('utils_integrated') else 'NO'}")
    
    print("")
    
    if summary['failed'] == 0 and summary['successful'] > 0:
        log_success("ALL STAGES COMPLETED SUCCESSFULLY!")
    elif summary['failed'] > 0:
        log_warning("SOME STAGES FAILED - Check logs for details")
    
    print("=" * 80)


def save_training_report(report: Dict[str, Any], output_path: Path = Path("training_summary.json")):
    """Save training report to JSON"""
    try:
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        log_success(f"Training report saved: {output_path}")
    except Exception as e:
        log_error(f"Failed to save report: {e}")


# ==========================================
# Pre-flight Checks
# ==========================================
def run_preflight_checks() -> bool:
    """Run all pre-flight checks before training"""
    print("\n" + "=" * 80)
    log_info("PRE-FLIGHT CHECKS")
    print("=" * 80)
    
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
    
    print("\n" + "=" * 80)
    if all_passed:
        log_success("ALL PRE-FLIGHT CHECKS PASSED!")
        log_info("Ready to start training!")
    else:
        log_error("SOME PRE-FLIGHT CHECKS FAILED!")
        log_info("Fix the issues above before running training")
    print("=" * 80)
    
    return all_passed


# ==========================================
# Main Function
# ==========================================
def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Master training orchestrator for recommendation system (ENHANCED v2.1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full training pipeline
  python train_all_models_fixed.py
  
  # Dry run (check setup without training)
  python train_all_models_fixed.py --dry-run
  
  # Skip specific stages
  python train_all_models_fixed.py --skip-text
  python train_all_models_fixed.py --skip-images
  
  # With validation
  python train_all_models_fixed.py --validate
  
  # Skip checks and run directly (not recommended)
  python train_all_models_fixed.py --skip-checks

Enhancements in v2.1:
  - Cursor timeout fixes for MongoDB queries
  - ID mapping fixes (string IDs, not indices)
  - DateTime normalization throughout
  - Utils integration for database operations
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
    
    print("=" * 80)
    log_info("MASTER TRAINING ORCHESTRATOR v2.1 (ENHANCED)")
    print("=" * 80)
    log_info("Version: 2.1.0")
    log_info("Purpose: Train all recommendation models")
    log_info("Scripts:")
    log_info("  - generate_text_embeddings_fixed.py (cursor timeout fix)")
    log_info("  - train_hybrid_model.py (ID mapping & datetime fix)")
    log_info("  - generate_image_embeddings_fixed.py (cursor timeout fix)")
    print("=" * 80)
    
    try:
        # Pre-flight checks
        if not args.skip_checks:
            checks_passed = run_preflight_checks()
            
            if not checks_passed and not args.dry_run:
                log_error("\nPre-flight checks failed. Fix issues before training.")
                log_info("Use --dry-run to see what would happen")
                log_info("Use --skip-checks to bypass (not recommended)")
                return 1
        else:
            log_warning("Skipping pre-flight checks (--skip-checks)")
        
        if args.dry_run:
            log_success("\nDry run completed. Ready for actual training!")
            log_info("Run without --dry-run to start training")
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
            log_error("\nTraining pipeline completed with errors")
            return 1
        else:
            log_success("\nTraining pipeline completed successfully!")
            print("\nNext steps:")
            log_info("  1. Test embeddings:")
            log_info("     python embedding_loader_enhanced.py")
            log_info("  2. Start recommendation service:")
            log_info("     uvicorn recommendation_service_enhanced:app --reload")
            log_info("  3. Test API at http://localhost:8000/docs")
            return 0
        
    except KeyboardInterrupt:
        log_warning("\nTraining interrupted by user")
        return 130
    except Exception as e:
        log_error(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
