"""
train_endpoint.py
API endpoints for training and model management

Provides endpoints to:
- Trigger model training
- Check training status
- View training history
- Manage model versions
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
    notify_url: Optional[str] = Field(None, description="Webhook URL to notify when training completes")


class TrainResponse(BaseModel):
    status: str = Field(..., description="success or error")
    message: str = Field(..., description="Status message")
    training_id: Optional[str] = Field(None, description="Training job ID")
    estimated_duration: Optional[int] = Field(None, description="Estimated duration in seconds")


class TrainStatusResponse(BaseModel):
    status: str = Field(..., description="pending, running, completed, failed")
    training_id: str = Field(..., description="Training job ID")
    started_at: Optional[str] = Field(None, description="Training start time")
    completed_at: Optional[str] = Field(None, description="Training completion time")
    duration: Optional[float] = Field(None, description="Training duration in seconds")
    products_trained: Optional[int] = Field(None, description="Number of products trained")
    error: Optional[str] = Field(None, description="Error message if failed")


class TrainingHistoryItem(BaseModel):
    timestamp: str
    type: str  # full or incremental
    products_count: int
    duration_seconds: float
    success: bool


class TrainingHistoryResponse(BaseModel):
    status: str
    history: list[TrainingHistoryItem]
    last_full_train: Optional[str]
    last_incremental_train: Optional[str]
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
        "total_products_trained": 0
    }


async def run_training(
    training_id: str,
    force_full: bool,
    cleanup: bool,
    notify_url: Optional[str] = None
):
    """
    Execute training script in background
    """
    try:
        # Update job status
        training_jobs[training_id]["status"] = "running"
        training_jobs[training_id]["started_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"Starting training job {training_id} (force_full={force_full}, cleanup={cleanup})")
        
        # Build command
        cmd = ["python", "train_hybrid_enhanced.py"]
        if force_full:
            cmd.append("--force-full")
        if cleanup:
            cmd.append("--cleanup")
        
        # Run training script
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd="recommendation_service"
        )
        
        stdout, stderr = await process.communicate()
        
        # Check result
        if process.returncode == 0:
            training_jobs[training_id]["status"] = "completed"
            training_jobs[training_id]["error"] = None
            logger.info(f"Training job {training_id} completed successfully")
            
            # Parse training output for metrics
            output = stdout.decode()
            if "Products trained:" in output:
                try:
                    products_line = [l for l in output.split("\n") if "Products trained:" in l][0]
                    products_count = int(products_line.split(":")[1].strip())
                    training_jobs[training_id]["products_trained"] = products_count
                except:
                    pass
        else:
            error_msg = stderr.decode() or "Training failed"
            training_jobs[training_id]["status"] = "failed"
            training_jobs[training_id]["error"] = error_msg
            logger.error(f"Training job {training_id} failed: {error_msg}")
        
        # Update completion time and duration
        completed_at = datetime.utcnow()
        training_jobs[training_id]["completed_at"] = completed_at.isoformat()
        
        started_at = datetime.fromisoformat(training_jobs[training_id]["started_at"])
        duration = (completed_at - started_at).total_seconds()
        training_jobs[training_id]["duration"] = duration
        
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
    
    **Note**: Training runs in the background. Use `/train/status/{training_id}` to check progress.
    """
    # Check if training script exists
    train_script = Path("recommendation_service/train_hybrid_enhanced.py")
    if not train_script.exists():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Training script not found"
        )
    
    # Generate training ID
    training_id = f"train_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize job tracking
    training_jobs[training_id] = {
        "status": "pending",
        "force_full": request.force_full,
        "cleanup": request.cleanup,
        "created_at": datetime.utcnow().isoformat(),
        "started_at": None,
        "completed_at": None,
        "duration": None,
        "products_trained": None,
        "error": None
    }
    
    # Add to background tasks
    background_tasks.add_task(
        run_training,
        training_id,
        request.force_full,
        request.cleanup,
        request.notify_url
    )
    
    # Estimate duration based on mode
    estimated_duration = 300 if request.force_full else 120  # 5 min full, 2 min incremental
    
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
        products_trained=job.get("products_trained"),
        error=job.get("error")
    )


@router.get("/history", response_model=TrainingHistoryResponse)
async def get_training_history(limit: int = Query(50, ge=1, le=200)):
    """
    Get training history
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
        total_products_trained=history.get("total_products_trained", 0)
    )


@router.get("/jobs", response_model=Dict)
async def list_training_jobs():
    """
    List all training jobs (in-memory)
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
    
    **Note**: This will attempt to kill the training process.
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
    
    # TODO: Implement process killing
    # For now, just mark as failed
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
    
    This endpoint should be called after training completes to load new models.
    """
    try:
        # Import load_models function from main service
        from recommendation_service_enhanced import load_models
        
        success = await asyncio.to_thread(load_models)
        
        if success:
            return {
                "status": "success",
                "message": "Models reloaded successfully"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to reload models"
            )
    
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model reload failed: {str(e)}"
        )
