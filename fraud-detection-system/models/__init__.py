"""
API Models for Fraud Detection System.

This module contains Pydantic models for API request/response validation.
"""

from .requests import (
    TransactionRequest,
    BatchTransactionRequest,
    ModelSelectionRequest
)
from .responses import (
    RiskLevel,
    PredictionResponse,
    BatchPredictionResponse,
    ModelMetricsResponse,
    HealthResponse,
    ErrorResponse
)

__all__ = [
    # Request models
    "TransactionRequest",
    "BatchTransactionRequest",
    "ModelSelectionRequest",
    # Response models
    "RiskLevel",
    "PredictionResponse",
    "BatchPredictionResponse",
    "ModelMetricsResponse",
    "HealthResponse",
    "ErrorResponse",
]

