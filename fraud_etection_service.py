"""
Fraud Detection API Service
============================

Production-ready FastAPI service for real-time credit card fraud detection.

Architecture:
- Singleton pattern for model loading (load once at startup)
- Stacking Ensemble (Logistic + RF + XGBoost + Isolation Forest)
- < 50ms response time target
- Comprehensive error handling and logging

Author: Fraud Detection System v2.0.1
Date: 2026-03-16
"""

import os
import sys
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import settings
from models.requests import TransactionRequest
from models.responses import FraudPredictionResponse, RiskLevel, ModelInfo, HealthResponse
from services.stacking_ensemble import load_stacking_ensemble_from_parts
from utils.preprocessing import FeatureScaler
from utils.model_utils import load_model

# ═══════════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Configure loguru
logger.remove()  # Remove default handler
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/fraud_api_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="30 days",
    level="INFO"
)

# ═══════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY (SINGLETON PATTERN)
# ═══════════════════════════════════════════════════════════════════════════

class ModelRegistry:
    """
    Singleton class to load ML models once at startup and reuse across requests.
    
    This pattern is critical for performance:
    - Loading models per request: ~50-100ms per prediction
    - Loading models once (singleton): ~5-10ms per prediction
    
    Performance improvement: 10x faster predictions! 🚀
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize models (only once)."""
        if not ModelRegistry._initialized:
            self.scaler = None
            self.ensemble = None
            self.model_metadata = {}
            self.load_timestamp = None
            self.prediction_count = 0
            self.total_inference_time = 0.0
            ModelRegistry._initialized = True
    
    def load_models(self) -> None:
        """
        Load all models and scaler from disk.
        
        This happens once at startup. Subsequent calls are no-ops.
        """
        if self.scaler is not None and self.ensemble is not None:
            logger.info("Models already loaded, skipping...")
            return
        
        logger.info("=" * 80)
        logger.info("LOADING FRAUD DETECTION MODELS")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # 1. Load Feature Scaler (CRITICAL!)
            logger.info("Loading feature scaler...")
            scaler_path = os.path.join(settings.MODEL_PATH, "feature_scaler.pkl")
            
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(
                    f"Scaler not found at {scaler_path}. "
                    "Please run 'python scripts/train_models.py' first."
                )
            self.scaler = FeatureScaler(scaler_path)
            logger.info(f"✓ Feature scaler loaded from {scaler_path}")
            
            # 2. Load Stacking Ensemble
            logger.info("Loading stacking ensemble...")
            ensemble_path = os.path.join(settings.MODEL_PATH, "stacking_ensemble.pkl")
            
            if not os.path.exists(ensemble_path):
                raise FileNotFoundError(
                    f"Ensemble not found at {ensemble_path}. "
                    "Please run 'python scripts/train_models.py' first."
                )
            
            self.ensemble = load_stacking_ensemble_from_parts(ensemble_path)
            logger.info(f"✓ Stacking ensemble loaded from {ensemble_path}")
            
            # 3. Load Model Metadata
            logger.info("Loading model metadata...")
            self._load_metadata()
            
            # 4. Warm up models (dummy prediction)
            logger.info("Warming up models with dummy prediction...")
            dummy_features = np.zeros((1, 30))
            dummy_scaled = self.scaler.transform(dummy_features)
            _ = self.ensemble.predict_proba(dummy_scaled)
            logger.info("✓ Models warmed up")
            
            # 5. Record load time
            load_time = time.time() - start_time
            self.load_timestamp = datetime.now()
            
            logger.info("=" * 80)
            logger.info(f"✓ ALL MODELS LOADED SUCCESSFULLY")
            logger.info(f"  Load Time: {load_time:.2f}s")
            logger.info(f"  Timestamp: {self.load_timestamp.isoformat()}")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ FAILED TO LOAD MODELS: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _load_metadata(self) -> None:
        """Load model metadata from JSON files."""
        import json
        
        metadata_files = [
            "logistic_regression_metadata.json",
            "random_forest_metadata.json",
            "xgboost_metadata.json",
            "isolation_forest_metadata.json",
            "stacking_ensemble_metadata.json"
        ]
        
        for filename in metadata_files:
            path = os.path.join(settings.MODEL_PATH, filename)
            if os.path.exists(path):
                with open(path, 'r') as f:
                    model_name = filename.replace('_metadata.json', '')
                    self.model_metadata[model_name] = json.load(f)
                logger.info(f"  ✓ Loaded metadata: {model_name}")
    
    def predict(self, transaction: TransactionRequest) -> Dict:
        """
        Make fraud prediction for a single transaction.
        
        Args:
            transaction: Validated transaction request
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        try:
            # 1. Convert to feature array (correct order!)
            features = transaction.to_array()
            
            # 2. Scale features (features is already 2D: shape (1, 30))
            scaled_features = self.scaler.transform(features)
            
            # 3. Get prediction
            fraud_probability = float(self.ensemble.predict_proba(scaled_features)[0])
            
            # 4. Get base model contributions
            base_predictions = self.ensemble.get_base_predictions(scaled_features)
            
            # 5. Determine risk level
            risk_level = self._calculate_risk_level(fraud_probability)
            
            # 6. Track performance
            inference_time = time.time() - start_time
            self.prediction_count += 1
            self.total_inference_time += inference_time
            
            logger.info(
                f"Prediction #{self.prediction_count}: "
                f"P(fraud)={fraud_probability:.4f}, "
                f"Risk={risk_level.value}, "
                f"Time={inference_time*1000:.2f}ms"
            )
            
            return {
                'fraud_probability': fraud_probability,
                'risk_level': risk_level,
                'base_model_contributions': base_predictions,
                'inference_time_ms': inference_time * 1000,
                'model_version': '2.0.1'
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _calculate_risk_level(self, probability: float) -> RiskLevel:
        """
        Calculate risk level from fraud probability.
        
        Thresholds:
        - LOW:      0.0 - 0.3  (0-30% fraud probability)
        - MEDIUM:   0.3 - 0.6  (30-60% fraud probability)
        - HIGH:     0.6 - 0.8  (60-80% fraud probability)
        - CRITICAL: 0.8 - 1.0  (80-100% fraud probability)
        """
        if probability < 0.3:
            return RiskLevel.LOW
        elif probability < 0.6:
            return RiskLevel.MEDIUM
        elif probability < 0.8:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def get_stats(self) -> Dict:
        """Get runtime statistics."""
        avg_time = (
            self.total_inference_time / self.prediction_count 
            if self.prediction_count > 0 
            else 0
        )
        
        return {
            'models_loaded': self.scaler is not None and self.ensemble is not None,
            'load_timestamp': self.load_timestamp.isoformat() if self.load_timestamp else None,
            'prediction_count': self.prediction_count,
            'avg_inference_time_ms': avg_time * 1000,
            'model_version': '2.0.1'
        }

# ═══════════════════════════════════════════════════════════════════════════
# FASTAPI APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    
    Startup: Load models once
    Shutdown: Log statistics
    """
    # Startup
    logger.info("🚀 Starting Fraud Detection API...")
    
    try:
        registry = ModelRegistry()
        registry.load_models()
        logger.info("✓ Startup complete")
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Fraud Detection API...")
    registry = ModelRegistry()
    stats = registry.get_stats()
    logger.info(f"📊 Final stats: {stats}")
    logger.info("✓ Shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection using Stacking Ensemble ML",
    version="2.0.1",
    lifespan=lifespan
)

# CORS middleware (configure based on your needs)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════════
# MIDDLEWARE
# ═══════════════════════════════════════════════════════════════════════════

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = time.time()
    
    # Log request
    logger.info(f"→ {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(
        f"← {request.method} {request.url.path} "
        f"Status={response.status_code} Time={process_time*1000:.2f}ms"
    )
    
    # Add custom header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# ═══════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API information."""
    return {
        "name": "Fraud Detection API",
        "version": "2.0.1",
        "status": "operational",
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health",
            "models": "GET /models/info",
            "metrics": "GET /models/metrics"
        },
        "documentation": "/docs"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status and model loading state
    """
    registry = ModelRegistry()
    stats = registry.get_stats()
    
    return HealthResponse(
        status="healthy" if stats['models_loaded'] else "unhealthy",
        models_loaded=stats['models_loaded'],
        model_version=stats['model_version'],
        uptime_seconds=None  # TODO: Calculate actual uptime
    )

@app.post("/predict", response_model=FraudPredictionResponse, tags=["Prediction"])
async def predict_fraud(transaction: TransactionRequest):
    """
    Predict fraud probability for a transaction.
    
    Args:
        transaction: Transaction data with Time, V1-V28, and Amount
        
    Returns:
        Fraud prediction with probability, risk level, and base model contributions
        
    Raises:
        HTTPException: If models not loaded or prediction fails
    """
    registry = ModelRegistry()
    
    # Check if models are loaded
    if registry.scaler is None or registry.ensemble is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded. Please wait for startup to complete."
        )
    
    try:
        # Make prediction
        result = registry.predict(transaction)
        
        # Build response
        return FraudPredictionResponse(
            fraud_probability=result['fraud_probability'],
            risk_level=result['risk_level'],
            base_model_contributions=result['base_model_contributions'],
            model_version=result['model_version'],
            timestamp=datetime.now(),
            inference_time_ms=result['inference_time_ms']
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch", tags=["Prediction"])
async def predict_fraud_batch(transactions: List[TransactionRequest]):
    """
    Predict fraud probability for multiple transactions (batch processing).
    
    Args:
        transactions: List of transactions
        
    Returns:
        List of fraud predictions
    """
    registry = ModelRegistry()
    
    if registry.scaler is None or registry.ensemble is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded."
        )
    
    try:
        results = []
        
        for transaction in transactions:
            result = registry.predict(transaction)
            results.append(FraudPredictionResponse(
                fraud_probability=result['fraud_probability'],
                risk_level=result['risk_level'],
                base_model_contributions=result['base_model_contributions'],
                model_version=result['model_version'],
                timestamp=datetime.now(),
                inference_time_ms=result['inference_time_ms']
            ))
        
        return {
            "predictions": results,
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.get("/models/info", response_model=ModelInfo, tags=["Models"])
async def get_model_info():
    """
    Get information about loaded models.
    
    Returns:
        Model metadata and performance metrics
    """
    registry = ModelRegistry()
    
    if registry.ensemble is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded."
        )
    
    return ModelInfo(
        model_version='2.0.1',
        models_loaded=True,
        load_timestamp=registry.load_timestamp,
        base_models=['logistic_regression', 'random_forest', 'xgboost', 'isolation_forest'],
        meta_model='logistic_regression',
        feature_count=30,
        metadata=registry.model_metadata
    )

@app.get("/models/metrics", tags=["Models"])
async def get_model_metrics():
    """
    Get runtime performance metrics.
    
    Returns:
        Statistics about API usage and performance
    """
    registry = ModelRegistry()
    stats = registry.get_stats()
    
    return {
        "runtime_statistics": stats,
        "metadata": registry.model_metadata
    }

# ═══════════════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ═══════════════════════════════════════════════════════════════════════════

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unexpected errors."""
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "error": str(exc),
            "path": request.url.path
        }
    )

# ═══════════════════════════════════════════════════════════════════════════
# MAIN (for local development)
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Fraud Detection API in development mode...")
    
    uvicorn.run(
        "fraud_detection_service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )

