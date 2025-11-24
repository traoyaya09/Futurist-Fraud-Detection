"""
Pydantic Models for Recommendation Service
"""

from .requests import (
    RecommendationRequest,
    BatchRecommendationRequest,
    InteractionRequest,
    EmbeddingRequest,
    HealthCheckRequest,
    FeedbackRequest
)

from .responses import (
    ProductResponse,
    ScoreBreakdown,
    RecommendationItem,
    RecommendationResponse,
    BatchRecommendationResponse,
    InteractionResponse,
    EmbeddingData,
    EmbeddingResponse,
    HealthStatus,
    ErrorDetail,
    ErrorResponse,
    FeedbackResponse,
    MetricsResponse,
    RateLimitResponse
)

__all__ = [
    # Requests
    "RecommendationRequest",
    "BatchRecommendationRequest",
    "InteractionRequest",
    "EmbeddingRequest",
    "HealthCheckRequest",
    "FeedbackRequest",
    # Responses
    "ProductResponse",
    "ScoreBreakdown",
    "RecommendationItem",
    "RecommendationResponse",
    "BatchRecommendationResponse",
    "InteractionResponse",
    "EmbeddingData",
    "EmbeddingResponse",
    "HealthStatus",
    "ErrorDetail",
    "ErrorResponse",
    "FeedbackResponse",
    "MetricsResponse",
    "RateLimitResponse"
]
