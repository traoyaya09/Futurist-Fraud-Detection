"""
train_endpoint.py
  API endpoints for training and model management - ENHANCED v2.1

Provides endpoints to:
- Trigger model training (uses train_hybrid_model.py)
- Check training status
- View training history
- Manage model versions
- Reload models after training

Integration:
- Works with train_hybrid_model.py (enhanced with ID mapping & datetime fixes)
- Compatible with train_all_models_fixed.py
- Supports both CLI and API training triggers
"""

import os
import subprocess
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, status
from pydantic import BaseModel, Field

logger = logging.getLogger("TrainEndpoint")

router = APIRouter(prefix="/train", tags=["Training"])


# ==========================================
# Request/Response Models
# ==========================================

class TrainRequest(BaseModel):
    force_full: bool = Field(False, description="Force full training (ignore incremental mode)")
    cleanup: bool = Field(False, description="Cleanup old embedding files before training")
    validate: bool = Field(True, description="Run validation after training")
    min_interactions: int = Field(3, description="Minimum interactions per user", ge=1, le=100)
    days_back: int = Field(90, description="Number of days of history to use", ge=1, le=365)
    notify_url: Optional[str] = Field(None, description="Webhook URL to notify when training completes")


class TrainResponse(BaseModel):
    status: str = Field(..., description="success or error")
    message: str = Field(..., description="Status message")
    training_id: Optional[str] = Field(None, description="Training job ID")
    estimated_duration: Optional[int] = Field(None, description="Estimated duration in seconds")


class TrainStatusResponse(BaseModel):
    status: str = Field(..., description="pending, running, completed, failed, cancelled")
    training_id: str = Field(..., description="Training job ID")
    started_at: Optional[str] = Field(None, description="Training start time (ISO 8601)")
    completed_at: Optional[str] = Field(None, description="Training completion time (ISO 8601)")
    duration: Optional[float] = Field(None, description="Training duration in seconds")
    users_trained: Optional[int] = Field(None, description="Number of users in model")
    products_trained: Optional[int] = Field(None, description="Number of products trained")
    error: Optional[str] = Field(None, description="Error message if failed")


class TrainingHistoryItem(BaseModel):
    timestamp: str
    type: str  # full or incremental
    users_count: int
    products_count: int
    duration_seconds: float
    success: bool
    datetime_normalized: bool = False
    utils_integrated: bool = False


class TrainingHistoryResponse(BaseModel):
    status: str
    history: list[TrainingHistoryItem]
    last_full_train: Optional[str]
    last_incremental_train: Optional[str]
    total_users_trained: int
    total_products_trained: int


# ==========================================
# Training State Management
# ==========================================

training_jobs: Dict[str, Dict] = {}  # In-memory job tracking


def get_training_history() -> Dict:
    """Load training history from file"""
    history_file = os.getenv("TRAINING_HISTORY_FILE", "training_history.json")
    
    if Path(history_file).exists():
        try:
            with open(history_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load training history: {e}")
    
    return {
        "runs": [],
        "last_full_train": None,
        "last_incremental_train": None,
        "total_users_trained": 0,
        "total_products_trained": 0
    }


def save_training_history(history: Dict):
    """Save training history to file"""
    history_file = os.getenv("TRAINING_HISTORY_FILE", "training_history.json")
    
    try:
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save training history: {e}")


async def run_training(
    training_id: str,
    force_full: bool,
    cleanup: bool,
    validate: bool,
    min_interactions: int,
    days_back: int,
    notify_url: Optional[str] = None
):
    """
    Execute training script in background
    
    Uses train_hybrid_model.py (enhanced version with ID mapping & datetime fixes)
    """
    try:
        # Update job status
        training_jobs[training_id]["status"] = "running"
        training_jobs[training_id]["started_at"] = datetime.utcnow().isoformat()
        
        logger.info(
            f"Starting training job {training_id} "
            f"(force_full={force_full}, cleanup={cleanup}, validate={validate})"
        )
        
        # Build command - USE ENHANCED TRAINING SCRIPT  
        cmd = ["python", "train_hybrid_model.py"]
        
        # Add arguments
        cmd.extend(["--min-interactions", str(min_interactions)])
        cmd.extend(["--days-back", str(days_back)])
        
        if validate:
            cmd.append("--validate")
        
        # Run training script
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(Path(__file__).parent)  # Run from recommendation_service/
        )
        
        stdout, stderr = await process.communicate()
        
        # Parse output
        output = stdout.decode('utf-8', errors='replace')
        error_output = stderr.decode('utf-8', errors='replace')
        
        # Check result
        if process.returncode == 0:
            training_jobs[training_id]["status"] = "completed"
            training_jobs[training_id]["error"] = None
            logger.info(f"Training job {training_id} completed successfully")
            
            # Parse training output for metrics
            try:
                # Look for training report
                report_path = Path("models/training_report.json")
                if report_path.exists():
                    with open(report_path, 'r') as f:
                        report = json.load(f)
                    
                    users_count = report.get("model_stats", {}).get("total_users", 0)
                    products_count = report.get("data_stats", {}).get("unique_products", 0)
                    
                    training_jobs[training_id]["users_trained"] = users_count
                    training_jobs[training_id]["products_trained"] = products_count
                    
                    # Update history
                    history = get_training_history()
                    history["runs"].append({
                        "timestamp": training_jobs[training_id]["started_at"],
                        "type": "full" if force_full else "incremental",
                        "users_count": users_count,
                        "products_count": products_count,
                        "duration_seconds": 0,  # Will be updated below
                        "success": True,
                        "datetime_normalized": report.get("datetime_normalized", False),
                        "utils_integrated": report.get("utils_integrated", False)
                    })
                    history["last_full_train"] = training_jobs[training_id]["started_at"]
                    history["total_users_trained"] = users_count
                    history["total_products_trained"] = products_count
                    save_training_history(history)
                    
            except Exception as e:
                logger.warning(f"Failed to parse training metrics: {e}")
        else:
            error_msg = error_output or output or "Training failed"
            training_jobs[training_id]["status"] = "failed"
            training_jobs[training_id]["error"] = error_msg[:500]  # Truncate long errors
            logger.error(f"Training job {training_id} failed: {error_msg[:200]}")
        
        # Update completion time and duration
        completed_at = datetime.utcnow()
        training_jobs[training_id]["completed_at"] = completed_at.isoformat()
        
        started_at = datetime.fromisoformat(training_jobs[training_id]["started_at"])
        duration = (completed_at - started_at).total_seconds()
        training_jobs[training_id]["duration"] = duration
        
        # Update history with duration
        history = get_training_history()
        if history["runs"]:
            history["runs"][-1]["duration_seconds"] = duration
            save_training_history(history)
        
        # Notify webhook if provided
        if notify_url:
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        notify_url,
                        json={
                            "training_id": training_id,
                            "status": training_jobs[training_id]["status"],
                            "duration": duration,
                            "users_trained": training_jobs[training_id].get("users_trained"),
                            "products_trained": training_jobs[training_id].get("products_trained")
                        },
                        timeout=aiohttp.ClientTimeout(total=10)
                    )
            except Exception as e:
                logger.warning(f"Failed to notify webhook: {e}")
        
    except Exception as e:
        logger.error(f"Training job {training_id} crashed: {e}")
        training_jobs[training_id]["status"] = "failed"
        training_jobs[training_id]["error"] = str(e)
        training_jobs[training_id]["completed_at"] = datetime.utcnow().isoformat()


# ==========================================
# API Endpoints
# ==========================================

@router.post("/start", response_model=TrainResponse)
async def start_training(
    request: TrainRequest,
    background_tasks: BackgroundTasks
):
    """
    Start a training job
    
    **Training Script:** Uses `train_hybrid_model.py` (enhanced with ID mapping & datetime fixes)
    
    **Note**: Training runs in the background. Use `/train/status/{training_id}` to check progress.
    
    **Features:**
    -   Proper ID mapping (string IDs, not indices)
    -   DateTime normalization (ISO strings)
    -   Utils integration
    -   Comprehensive validation
    """
    # Check if training script exists
    train_script = Path(__file__).parent / "train_hybrid_model.py"
    if not train_script.exists():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training script not found: {train_script}"
        )
    
    # Generate training ID with datetime
    training_id = f"train_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize job tracking
    training_jobs[training_id] = {
        "status": "pending",
        "force_full": request.force_full,
        "cleanup": request.cleanup,
        "validate": request.validate,
        "min_interactions": request.min_interactions,
        "days_back": request.days_back,
        "created_at": datetime.utcnow().isoformat(),
        "started_at": None,
        "completed_at": None,
        "duration": None,
        "users_trained": None,
        "products_trained": None,
        "error": None
    }
    
    # Add to background tasks
    background_tasks.add_task(
        run_training,
        training_id,
        request.force_full,
        request.cleanup,
        request.validate,
        request.min_interactions,
        request.days_back,
        request.notify_url
    )
    
    # Estimate duration based on mode and data size
    estimated_duration = 300 if request.force_full else 180  # 5 min full, 3 min incremental
    
    return TrainResponse(
        status="success",
        message=f"Training job started: {training_id}",
        training_id=training_id,
        estimated_duration=estimated_duration
    )


@router.get("/status/{training_id}", response_model=TrainStatusResponse)
async def get_training_status(training_id: str):
    """
    Get status of a training job
    
    **Returns:**
    - Job status (pending, running, completed, failed, cancelled)
    - Start/completion times (ISO 8601 format)
    - Duration, users trained, products trained
    - Error message if failed
    """
    if training_id not in training_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job not found: {training_id}"
        )
    
    job = training_jobs[training_id]
    
    return TrainStatusResponse(
        status=job["status"],
        training_id=training_id,
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        duration=job.get("duration"),
        users_trained=job.get("users_trained"),
        products_trained=job.get("products_trained"),
        error=job.get("error")
    )


@router.get("/history", response_model=TrainingHistoryResponse)
async def get_training_history_endpoint(limit: int = Query(50, ge=1, le=200)):
    """
    Get training history
    
    **Returns:**
    - List of past training runs
    - Last full/incremental train timestamps
    - Total users/products trained
    - DateTime normalization status
    """
    history = get_training_history()
    
    # Convert to response format
    history_items = [
        TrainingHistoryItem(**run)
        for run in history["runs"][-limit:]
    ]
    
    return TrainingHistoryResponse(
        status="success",
        history=history_items,
        last_full_train=history.get("last_full_train"),
        last_incremental_train=history.get("last_incremental_train"),
        total_users_trained=history.get("total_users_trained", 0),
        total_products_trained=history.get("total_products_trained", 0)
    )


@router.get("/jobs", response_model=Dict)
async def list_training_jobs():
    """
    List all training jobs (in-memory)
    
    **Note**: Jobs are stored in memory and will be lost on service restart.
    Use `/train/history` for persistent training history.
    """
    return {
        "status": "success",
        "jobs": training_jobs,
        "count": len(training_jobs)
    }


@router.delete("/jobs/{training_id}")
async def cancel_training_job(training_id: str):
    """
    Cancel a running training job
    
    **Note**: This marks the job as cancelled but doesn't kill the process.
    The training script will continue running until completion.
    """
    if training_id not in training_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job not found: {training_id}"
        )
    
    job = training_jobs[training_id]
    
    if job["status"] not in ["pending", "running"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel job with status: {job['status']}"
        )
    
    # Mark as cancelled
    training_jobs[training_id]["status"] = "cancelled"
    training_jobs[training_id]["completed_at"] = datetime.utcnow().isoformat()
    training_jobs[training_id]["error"] = "Cancelled by user"
    
    return {
        "status": "success",
        "message": f"Training job cancelled: {training_id}"
    }


@router.post("/reload-models")
async def reload_models():
    """
    Reload models from disk after training
    
    **This endpoint should be called after training completes to load new models.**
    
    **Reloads:**
    - Hybrid model (collaborative filtering)
    - Text embeddings (if changed)
    - Image embeddings (if changed)
    - ID mappings (user & product)
    """
    try:
        # Import load_models function from main service
        from recommendation_service_enhanced import load_models
        
        success = await asyncio.to_thread(load_models)
        
        if success:
            return {
                "status": "success",
                "message": "Models reloaded successfully",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to reload models - check logs for details"
            )
    
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model reload failed: {str(e)}"
        )


@router.get("/model-info")
async def get_model_info():
    """
    Get information about currently loaded model
    
    **Returns:**
    - Model file paths
    - Model sizes
    - Last modified times
    - Training report metadata
    """
    try:
        models_dir = Path("models")
        
        if not models_dir.exists():
            return {
                "status": "no_models",
                "message": "Models directory not found - training required"
            }
        
        info = {
            "status": "success",
            "models_directory": str(models_dir.absolute()),
            "files": {}
        }
        
        # Check for model files
        model_files = [
            "hybrid_model.joblib",
            "id_mappings.json",
            "training_report.json"
        ]
        
        for filename in model_files:
            file_path = models_dir / filename
            if file_path.exists():
                stat = file_path.stat()
                info["files"][filename] = {
                    "exists": True,
                    "size_bytes": stat.st_size,
                    "size_mb": round(stat.st_size / 1024 / 1024, 2),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
            else:
                info["files"][filename] = {
                    "exists": False
                }
        
        # Load training report if available
        report_path = models_dir / "training_report.json"
        if report_path.exists():
            try:
                with open(report_path, 'r') as f:
                    report = json.load(f)
                
                info["training_info"] = {
                    "training_date": report.get("training_date"),
                    "duration_seconds": report.get("duration_seconds"),
                    "total_users": report.get("model_stats", {}).get("total_users"),
                    "total_products": report.get("data_stats", {}).get("unique_products"),
                    "datetime_normalized": report.get("datetime_normalized", False),
                    "utils_integrated": report.get("utils_integrated", False),
                    "product_model_used": report.get("product_model_used", False)
                }
            except Exception as e:
                logger.warning(f"Failed to load training report: {e}")
        
        return info
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )


@router.post("/validate-model")
async def validate_current_model():
    """
    Validate currently loaded model
    
    **Checks:**
    - Model file exists and can be loaded
    - ID mappings are correct
    - Score ranges are valid
    - No missing data
    """
    try:
        import joblib
        
        models_dir = Path("models")
        model_path = models_dir / "hybrid_model.joblib"
        mappings_path = models_dir / "id_mappings.json"
        
        if not model_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model file not found - training required"
            )
        
        # Load model
        model = joblib.load(model_path)
        
        # Load mappings
        if mappings_path.exists():
            with open(mappings_path, 'r') as f:
                mappings = json.load(f)
        else:
            mappings = {}
        
        # Validate
        validation_results = {
            "status": "success",
            "model_loaded": True,
            "checks": {}
        }
        
        # Check 1: Model not empty
        validation_results["checks"]["not_empty"] = len(model) > 0
        
        # Check 2: User IDs are strings
        sample_users = list(model.keys())[:3]
        validation_results["checks"]["user_ids_are_strings"] = all(
            isinstance(uid, str) for uid in sample_users
        )
        
        # Check 3: Product IDs are strings
        if sample_users:
            sample_products = list(model[sample_users[0]].keys())[:3]
            validation_results["checks"]["product_ids_are_strings"] = all(
                isinstance(pid, str) for pid in sample_products
            )
        
        # Check 4: Scores in valid range [0, 1]
        all_scores = []
        for user_scores in list(model.values())[:10]:
            all_scores.extend(list(user_scores.values())[:10])
        
        if all_scores:
            validation_results["checks"]["scores_in_range"] = all(
                0 <= score <= 1 for score in all_scores
            )
            validation_results["checks"]["score_stats"] = {
                "min": min(all_scores),
                "max": max(all_scores),
                "mean": sum(all_scores) / len(all_scores)
            }
        
        # Check 5: Mappings present
        validation_results["checks"]["has_mappings"] = bool(mappings)
        validation_results["checks"]["datetime_normalized"] = mappings.get("datetime_normalized", False)
        
        # Overall status
        all_checks_passed = all(
            v for k, v in validation_results["checks"].items() 
            if isinstance(v, bool)
        )
        
        validation_results["all_checks_passed"] = all_checks_passed
        validation_results["total_users"] = len(model)
        validation_results["sample_users"] = sample_users
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model validation failed: {str(e)}"
        )
