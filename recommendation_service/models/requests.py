"""
Pydantic Request Models with Enhanced Validation
"""

from typing import Optional, List
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime
import base64


class RecommendationRequest(BaseModel):
    """
    Request model for single product recommendation
    """
    userId: Optional[str] = Field(
        default=None,
        max_length=50,
        description="User ID for personalized recommendations"
    )
    query: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Text search query"
    )
    image: Optional[str] = Field(
        default=None,
        description="Base64 encoded image or image URL"
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of recommendations to return"
    )
    page: int = Field(
        default=1,
        ge=1,
        description="Page number for pagination"
    )
    normalize: Optional[bool] = Field(
        default=None,
        description="Whether to normalize scores to 0-1 range"
    )
    debug: Optional[bool] = Field(
        default=False,
        description="Enable debug mode with detailed scoring breakdown"
    )
    diversityWeight: Optional[float] = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Weight for diversity in recommendations (0-1)"
    )
    
    @field_validator("userId")
    @classmethod
    def validate_user_id(cls, v):
        """Validate user ID format"""
        if v is not None:
            v = v.strip()
            if len(v) == 0:
                return None
            if len(v) > 50:
                raise ValueError("userId must not exceed 50 characters")
            # Allow alphanumeric, hyphens, and underscores
            if not all(c.isalnum() or c in ['-', '_'] for c in v):
                raise ValueError("userId must contain only alphanumeric characters, hyphens, and underscores")
        return v
    
    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        """Validate and clean query string"""
        if v is not None:
            v = v.strip()
            if len(v) == 0:
                return None
            if len(v) > 500:
                raise ValueError("query must not exceed 500 characters")
        return v
    
    @field_validator("image")
    @classmethod
    def validate_image(cls, v):
        """Validate image format and size"""
        if v is not None:
            v = v.strip()
            if len(v) == 0:
                return None
            
            # Check if it's a URL
            if v.startswith("http://") or v.startswith("https://"):
                if len(v) > 2048:
                    raise ValueError("Image URL must not exceed 2048 characters")
                return v
            
            # Check if it's base64
            if v.startswith("data:image"):
                try:
                    # Extract base64 part
                    base64_data = v.split(",")[-1]
                    decoded = base64.b64decode(base64_data)
                    
                    # Check size (10 MB limit)
                    size_mb = len(decoded) / (1024 * 1024)
                    if size_mb > 10:
                        raise ValueError(f"Image size ({size_mb:.2f}MB) exceeds 10MB limit")
                    
                    return v
                except Exception as e:
                    raise ValueError(f"Invalid base64 image format: {str(e)}")
            else:
                raise ValueError("Image must be a valid URL or base64-encoded data URI")
        
        return v
    
    @model_validator(mode='after')
    def validate_request(self):
        """Cross-field validation"""
        # At least one of query or image must be provided for meaningful recommendations
        if not self.query and not self.image and not self.userId:
            raise ValueError("At least one of userId, query, or image must be provided")
        
        return self


class BatchRecommendationRequest(BaseModel):
    """
    Request model for batch recommendations
    """
    requests: List[RecommendationRequest] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="List of recommendation requests (max 10)"
    )
    
    @field_validator("requests")
    @classmethod
    def validate_requests(cls, v):
        """Validate batch size"""
        if len(v) > 10:
            raise ValueError("Batch size must not exceed 10 requests")
        if len(v) == 0:
            raise ValueError("Batch must contain at least 1 request")
        return v


class InteractionRequest(BaseModel):
    """
    Request model for logging user interactions
    """
    userId: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="User ID who performed the action"
    )
    productId: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Product ID that was interacted with"
    )
    action: str = Field(
        ...,
        description="Type of interaction action"
    )
    timestamp: Optional[str] = Field(
        default=None,
        description="ISO format timestamp of interaction"
    )
    metadata: Optional[dict] = Field(
        default=None,
        description="Additional metadata about the interaction"
    )
    
    @field_validator("userId", "productId")
    @classmethod
    def validate_ids(cls, v):
        """Validate ID format"""
        v = v.strip()
        if len(v) == 0:
            raise ValueError("ID must not be empty")
        if len(v) > 50:
            raise ValueError("ID must not exceed 50 characters")
        # Allow alphanumeric, hyphens, and underscores
        if not all(c.isalnum() or c in ['-', '_'] for c in v):
            raise ValueError("ID must contain only alphanumeric characters, hyphens, and underscores")
        return v
    
    @field_validator("action")
    @classmethod
    def validate_action(cls, v):
        """Validate action type"""
        allowed_actions = [
            "view", "click", "add_to_cart", "remove_from_cart",
            "purchase", "wishlist_add", "wishlist_remove",
            "review", "rating", "share"
        ]
        v = v.lower().strip()
        if v not in allowed_actions:
            raise ValueError(f"action must be one of: {', '.join(allowed_actions)}")
        return v
    
    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v):
        """Validate timestamp format"""
        if v is not None:
            try:
                # Try to parse ISO format
                datetime.fromisoformat(v.replace('Z', '+00:00'))
            except ValueError:
                raise ValueError("timestamp must be in ISO format (e.g., 2024-01-01T12:00:00Z)")
        return v
    
    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v):
        """Validate metadata size"""
        if v is not None:
            # Limit metadata size to prevent abuse
            import json
            metadata_str = json.dumps(v)
            if len(metadata_str) > 1024:  # 1KB limit
                raise ValueError("metadata must not exceed 1KB")
        return v


class EmbeddingRequest(BaseModel):
    """
    Request model for embedding lookup
    """
    productIds: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of product IDs to fetch embeddings for"
    )
    embeddingType: str = Field(
        default="both",
        pattern="^(text|image|both)$",
        description="Type of embeddings to fetch"
    )
    
    @field_validator("productIds")
    @classmethod
    def validate_product_ids(cls, v):
        """Validate product IDs"""
        if len(v) > 100:
            raise ValueError("Cannot request embeddings for more than 100 products at once")
        
        validated = []
        for pid in v:
            pid = pid.strip()
            if len(pid) == 0:
                continue
            if len(pid) > 50:
                raise ValueError(f"Product ID '{pid}' exceeds 50 characters")
            validated.append(pid)
        
        if len(validated) == 0:
            raise ValueError("At least one valid product ID must be provided")
        
        return validated


class HealthCheckRequest(BaseModel):
    """
    Request model for health check
    """
    deep: bool = Field(
        default=False,
        description="Perform deep health check (includes DB and model checks)"
    )


class FeedbackRequest(BaseModel):
    """
    Request model for recommendation feedback
    """
    userId: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="User ID providing feedback"
    )
    productId: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Product ID being rated"
    )
    relevant: bool = Field(
        ...,
        description="Whether the recommendation was relevant"
    )
    clicked: bool = Field(
        default=False,
        description="Whether the user clicked the recommendation"
    )
    purchased: bool = Field(
        default=False,
        description="Whether the user purchased the product"
    )
    rating: Optional[int] = Field(
        default=None,
        ge=1,
        le=5,
        description="User rating (1-5 stars)"
    )
    comment: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional feedback comment"
    )
    
    @field_validator("comment")
    @classmethod
    def validate_comment(cls, v):
        """Clean and validate comment"""
        if v is not None:
            v = v.strip()
            if len(v) == 0:
                return None
            if len(v) > 500:
                raise ValueError("comment must not exceed 500 characters")
        return v
