"""
Pydantic Response Models
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ProductResponse(BaseModel):
    """
    Product data structure returned in recommendations
    """
    id: str = Field(..., alias="_id", description="Product ID")
    name: str = Field(..., description="Product name")
    description: str = Field(..., description="Product description")
    price: Optional[float] = Field(None, description="Product price")
    discountPrice: Optional[float] = Field(None, description="Discounted price")
    category: str = Field(..., description="Product category")
    subCategory: Optional[str] = Field(None, description="Product subcategory")
    brand: Optional[str] = Field(None, description="Product brand")
    stock: int = Field(..., description="Available stock")
    imageUrl: str = Field(..., description="Product image URL")
    rating: float = Field(..., description="Average rating")
    reviewsCount: int = Field(..., description="Number of reviews")
    reviews: List[Dict[str, Any]] = Field(default_factory=list, description="Product reviews")
    isFeatured: bool = Field(default=False, description="Whether product is featured")
    promotion: Optional[Dict[str, Any]] = Field(None, description="Active promotion")
    createdAt: Optional[str] = Field(None, description="Creation timestamp")
    
    class Config:
        populate_by_name = True


class ScoreBreakdown(BaseModel):
    """
    Detailed score breakdown for debugging
    """
    hybrid: float = Field(..., description="Collaborative filtering score")
    text: float = Field(..., description="Text similarity score")
    image: float = Field(..., description="Image similarity score")
    interaction: float = Field(..., description="User interaction boost")
    field: float = Field(..., description="Field matching boost")
    diversity: float = Field(default=0.0, description="Diversity penalty/boost")
    final: float = Field(..., description="Final combined score")
    
    weights: Dict[str, float] = Field(..., description="Weights used in scoring")
    explanation: Optional[str] = Field(None, description="Human-readable explanation")


class RecommendationItem(BaseModel):
    """
    Single recommendation with product and score
    """
    product: ProductResponse = Field(..., description="Recommended product")
    score: float = Field(..., ge=0.0, le=1.0, description="Recommendation score (0-1)")
    rank: Optional[int] = Field(None, description="Ranking position")
    scoreBreakdown: Optional[ScoreBreakdown] = Field(None, description="Detailed scoring (debug mode)")
    reason: Optional[str] = Field(None, description="Recommendation reason")


class RecommendationResponse(BaseModel):
    """
    Response for single recommendation request
    """
    status: str = Field(default="success", description="Response status")
    data: List[RecommendationItem] = Field(..., description="List of recommendations")
    meta: Optional[Dict[str, Any]] = Field(None, description="Metadata about the response")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Response timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "data": [
                    {
                        "product": {
                            "_id": "prod123",
                            "name": "Premium Sneakers",
                            "description": "High-quality athletic sneakers",
                            "price": 129.99,
                            "discountPrice": 99.99,
                            "category": "shoes",
                            "subCategory": "sneakers",
                            "brand": "Nike",
                            "stock": 50,
                            "imageUrl": "https://example.com/image.jpg",
                            "rating": 4.5,
                            "reviewsCount": 120,
                            "reviews": [],
                            "isFeatured": True,
                            "promotion": None,
                            "createdAt": "2024-01-01T00:00:00Z"
                        },
                        "score": 0.95,
                        "rank": 1
                    }
                ],
                "meta": {
                    "query": "running shoes",
                    "limit": 20,
                    "page": 1,
                    "total": 1
                },
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }


class BatchRecommendationResponse(BaseModel):
    """
    Response for batch recommendation requests
    """
    status: str = Field(default="success", description="Response status")
    data: List[List[RecommendationItem]] = Field(..., description="List of recommendation lists")
    meta: Optional[Dict[str, Any]] = Field(None, description="Metadata")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class InteractionResponse(BaseModel):
    """
    Response for interaction logging
    """
    status: str = Field(default="success", description="Response status")
    message: Optional[str] = Field(None, description="Response message")
    interactionId: Optional[str] = Field(None, description="ID of logged interaction")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class EmbeddingData(BaseModel):
    """
    Embedding data for a single product
    """
    text: Optional[List[float]] = Field(None, description="Text embedding vector")
    image: Optional[List[float]] = Field(None, description="Image embedding vector")


class EmbeddingResponse(BaseModel):
    """
    Response for embedding lookup
    """
    status: str = Field(default="success", description="Response status")
    data: Dict[str, EmbeddingData] = Field(..., description="Embeddings keyed by product ID")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class HealthStatus(BaseModel):
    """
    Health check response
    """
    status: str = Field(..., description="Overall health status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    uptime: float = Field(..., description="Service uptime in seconds")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Component health
    mongodb: Optional[str] = Field(None, description="MongoDB status")
    redis: Optional[str] = Field(None, description="Redis status (if enabled)")
    models: Optional[Dict[str, str]] = Field(None, description="Model loading status")
    embeddings: Optional[Dict[str, str]] = Field(None, description="Embedding status")
    
    # Performance metrics
    metrics: Optional[Dict[str, Any]] = Field(None, description="Performance metrics")


class ErrorDetail(BaseModel):
    """
    Error detail structure
    """
    field: Optional[str] = Field(None, description="Field that caused the error")
    message: str = Field(..., description="Error message")
    type: Optional[str] = Field(None, description="Error type")


class ErrorResponse(BaseModel):
    """
    Error response structure
    """
    status: str = Field(default="error", description="Response status")
    error: str = Field(..., description="Error message")
    details: Optional[List[ErrorDetail]] = Field(None, description="Detailed error information")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    path: Optional[str] = Field(None, description="Request path that caused the error")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "error",
                "error": "Validation error",
                "details": [
                    {
                        "field": "userId",
                        "message": "userId must not exceed 50 characters",
                        "type": "value_error"
                    }
                ],
                "timestamp": "2024-01-01T12:00:00Z",
                "path": "/recommendations"
            }
        }


class FeedbackResponse(BaseModel):
    """
    Response for recommendation feedback
    """
    status: str = Field(default="success", description="Response status")
    message: str = Field(default="Feedback recorded successfully", description="Response message")
    feedbackId: Optional[str] = Field(None, description="ID of recorded feedback")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class MetricsResponse(BaseModel):
    """
    Response for metrics endpoint
    """
    status: str = Field(default="success", description="Response status")
    metrics: Dict[str, Any] = Field(..., description="Service metrics")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class RateLimitResponse(BaseModel):
    """
    Response when rate limit is exceeded
    """
    status: str = Field(default="error", description="Response status")
    error: str = Field(default="Rate limit exceeded", description="Error message")
    limit: int = Field(..., description="Rate limit threshold")
    window: str = Field(..., description="Time window (e.g., '1 minute')")
    retryAfter: int = Field(..., description="Seconds to wait before retry")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
