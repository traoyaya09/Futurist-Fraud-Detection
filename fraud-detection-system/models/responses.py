
"""
Response models for Fraud Detection API.

Pydantic models for structuring API responses with type-safe enums.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class RiskLevel(str, Enum):
    """
    Risk level classification for fraud detection.
    
    Type-safe enum to prevent magic strings in Node.js integration.
    
    Thresholds (suggested):
    - LOW: fraud_probability < 0.3
    - MEDIUM: 0.3 <= fraud_probability < 0.6
    - HIGH: 0.6 <= fraud_probability < 0.8
    - CRITICAL: fraud_probability >= 0.8
    """
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class PredictionResponse(BaseModel):
    """
    Single transaction prediction response.
    
    Attributes:
        transaction_id: Unique identifier for this prediction
        is_fraud: Binary prediction (0: legitimate, 1: fraud)
        fraud_probability: Probability of fraud (0.0 to 1.0)
        risk_level: Risk category (LOW, MEDIUM, HIGH, CRITICAL)
        model_used: Name of the model used for prediction
        confidence_score: Model confidence in the prediction
        timestamp: When the prediction was made
        processing_time_ms: Time taken to process request in milliseconds
    """
    
    transaction_id: str = Field(
        ...,
        description="Unique identifier for this prediction"
    )
    is_fraud: int = Field(
        ...,
        ge=0,
        le=1,
        description="Binary fraud prediction (0: legitimate, 1: fraud)"
    )
    fraud_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of fraud (0.0 to 1.0)"
    )
    risk_level: RiskLevel = Field(
        ...,
        description="Risk category: LOW, MEDIUM, HIGH, CRITICAL"
    )
    model_used: str = Field(
        ...,
        description="Model used for prediction"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence in prediction"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Prediction timestamp"
    )
    processing_time_ms: float = Field(
        ...,
        ge=0,
        description="Processing time in milliseconds"
    )
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "transaction_id": "txn_1234567890",
                "is_fraud": 0,
                "fraud_probability": 0.0234,
                "risk_level": "LOW",
                "model_used": "stacking_ensemble",
                "confidence_score": 0.9766,
                "timestamp": "2024-03-16T10:30:00",
                "processing_time_ms": 12.5
            }
        }


class BatchPredictionResponse(BaseModel):
    """
    Batch transaction predictions response.
    
    Attributes:
        predictions: List of individual predictions
        total_transactions: Total number of transactions processed
        fraud_detected: Number of fraudulent transactions detected
        fraud_rate: Percentage of fraudulent transactions
        model_used: Name of the model used
        total_processing_time_ms: Total processing time in milliseconds
        timestamp: When the batch was processed
    """
    
    predictions: List[PredictionResponse] = Field(
        ...,
        description="List of individual predictions"
    )
    total_transactions: int = Field(
        ...,
        ge=0,
        description="Total transactions processed"
    )
    fraud_detected: int = Field(
        ...,
        ge=0,
        description="Number of fraudulent transactions"
    )
    fraud_rate: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of fraudulent transactions"
    )
    model_used: str = Field(
        ...,
        description="Model used for predictions"
    )
    total_processing_time_ms: float = Field(
        ...,
        ge=0,
        description="Total processing time in milliseconds"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Batch processing timestamp"
    )
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "transaction_id": "txn_001",
                        "is_fraud": 0,
                        "fraud_probability": 0.0234,
                        "risk_level": "LOW",
                        "model_used": "stacking_ensemble",
                        "confidence_score": 0.9766,
                        "timestamp": "2024-03-16T10:30:00",
                        "processing_time_ms": 5.2
                    }
                ],
                "total_transactions": 100,
                "fraud_detected": 3,
                "fraud_rate": 3.0,
                "model_used": "stacking_ensemble",
                "total_processing_time_ms": 523.4,
                "timestamp": "2024-03-16T10:30:00"
            }
        }


class ModelMetricsResponse(BaseModel):
    """
    Model performance metrics response.
    
    Attributes:
        model_name: Name of the model
        accuracy: Overall accuracy
        precision: Precision score
        recall: Recall score
        f1_score: F1 score
        roc_auc: ROC-AUC score
        confusion_matrix: Confusion matrix values
        additional_metrics: Any additional metrics
        training_date: When the model was trained
        dataset_size: Size of training dataset
    """
    
    model_name: str = Field(
        ...,
        description="Name of the model"
    )
    accuracy: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall accuracy"
    )
    precision: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Precision score"
    )
    recall: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Recall score"
    )
    f1_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="F1 score"
    )
    roc_auc: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="ROC-AUC score"
    )
    confusion_matrix: Dict[str, int] = Field(
        ...,
        description="Confusion matrix: TN, FP, FN, TP"
    )
    additional_metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional model-specific metrics"
    )
    training_date: Optional[str] = Field(
        default=None,
        description="Model training date"
    )
    dataset_size: Optional[int] = Field(
        default=None,
        description="Training dataset size"
    )
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "model_name": "xgboost",
                "accuracy": 0.9995,
                "precision": 0.8840,
                "recall": 0.8211,
                "f1_score": 0.8514,
                "roc_auc": 0.9717,
                "confusion_matrix": {
                    "TN": 56849,
                    "FP": 13,
                    "FN": 17,
                    "TP": 78
                },
                "additional_metrics": {
                    "log_loss": 0.0123,
                    "average_precision": 0.8567
                },
                "training_date": "2024-03-16",
                "dataset_size": 284807
            }
        }


class FraudPredictionResponse(BaseModel):
    """Response model for fraud prediction."""
    
    fraud_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of fraud (0.0 to 1.0)"
    )
    risk_level: RiskLevel = Field(
        ...,
        description="Categorical risk level"
    )
    base_model_contributions: Dict[str, float] = Field(
        default_factory=dict,
        description="Contribution from each base model"
    )
    model_version: str = Field(
        default="2.0.1",
        description="Model version used for prediction"
    )
    timestamp: Optional[datetime] = Field(
        default_factory=datetime.now,
        description="Prediction timestamp"
    )
    inference_time_ms: Optional[float] = Field(
        None,
        description="Inference time in milliseconds"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "fraud_probability": 0.1234,
                "risk_level": "LOW",
                "base_model_contributions": {
                    "logistic_regression": 0.205,
                    "random_forest": 0.142,
                    "xgboost": 0.168,
                    "isolation_forest": 0.089
                },
                "model_version": "2.0.1",
                "timestamp": "2026-03-16T10:30:00",
                "inference_time_ms": 12.34
            }
        }
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    
    status: str = Field(
        ...,
        description="Health status (healthy/unhealthy)"
    )
    models_loaded: bool = Field(
        ...,
        description="Whether ML models are loaded"
    )
    model_version: str = Field(
        ...,
        description="Current model version"
    )
    uptime_seconds: Optional[float] = Field(
        None,
        description="API uptime in seconds"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "models_loaded": True,
                "model_version": "2.0.1",
                "uptime_seconds": 3600.5
            }
        }
    )


class ModelInfo(BaseModel):
    """Response model for model information endpoint."""
    
    model_version: str = Field(
        ...,
        description="Current model version"
    )
    models_loaded: bool = Field(
        ...,
        description="Whether models are loaded"
    )
    load_timestamp: Optional[datetime] = Field(
        None,
        description="When models were loaded"
    )
    base_models: List[str] = Field(
        default_factory=list,
        description="List of base models in ensemble"
    )
    meta_model: str = Field(
        ...,
        description="Meta-learner model name"
    )
    feature_count: int = Field(
        ...,
        description="Number of input features"
    )
    metadata: Dict = Field(
        default_factory=dict,
        description="Additional model metadata"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_version": "2.0.1",
                "models_loaded": True,
                "load_timestamp": "2026-03-16T10:00:00",
                "base_models": [
                    "logistic_regression",
                    "random_forest",
                    "xgboost",
                    "isolation_forest"
                ],
                "meta_model": "logistic_regression",
                "feature_count": 30,
                "metadata": {}
            }
        }
    )


class ErrorResponse(BaseModel):
    """
    Error response.
    
    Attributes:
        error: Error message
        detail: Detailed error information
        timestamp: When the error occurred
        request_id: Request identifier for tracking
    """
    
    error: str = Field(
        ...,
        description="Error message"
    )
    detail: Optional[str] = Field(
        default=None,
        description="Detailed error information"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Error timestamp"
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Request identifier"
    )
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "error": "Invalid transaction data",
                "detail": "Amount must be greater than or equal to 0",
                "timestamp": "2024-03-16T10:30:00",
                "request_id": "req_abc123"
            }
        }

