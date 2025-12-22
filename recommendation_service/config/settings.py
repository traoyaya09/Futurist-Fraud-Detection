"""
Configuration Management for Recommendation Service
Handles all environment variables and settings using Pydantic BaseSettings
"""

import os
import json
from typing import Optional, List, Union, Any
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    All sensitive data should be stored in .env file
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )
    
    # ==========================================
    # Application Settings
    # ==========================================
    APP_NAME: str = "Hybrid Product Recommendation Service"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"
    
    # ==========================================
    # Server Settings
    # ==========================================
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    RELOAD: bool = False
    
    # ==========================================
    # MongoDB Settings
    # ==========================================
    MONGO_URI: str = Field(
        default="mongodb://localhost:27017",
        description="MongoDB connection URI"
    )
    MONGO_DB_NAME: str = Field(
        default="futurist_e-commerce",
        description="MongoDB database name"
    )
    MONGO_TIMEOUT_MS: int = 30000
    MONGO_CONNECT_TIMEOUT_MS: int = 30000
    MONGO_MAX_RETRY_ATTEMPTS: int = 5
    MONGO_RETRY_DELAY_SECONDS: int = 5
    
    # ==========================================
    # Security Settings
    # ==========================================
    SECRET_KEY: str = Field(
        default="your-secret-key-change-in-production",
        description="Secret key for JWT and encryption"
    )
    API_KEY: Optional[str] = Field(
        default=None,
        description="Optional API key for authentication"
    )
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    
    # ==========================================
    # CORS Settings
    # ==========================================
    CORS_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        description="Allowed CORS origins as comma-separated string"
    )
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: str = "GET,POST,PUT,DELETE,OPTIONS"
    CORS_ALLOW_HEADERS: str = "*"
    
    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> str:
        """Parse CORS_ORIGINS and ensure it's a string"""
        if v is None or v == "":
            return "http://localhost:3000,http://localhost:5173"
        
        if isinstance(v, list):
            return ",".join(str(x) for x in v)
        
        if isinstance(v, str):
            v = v.strip()
            if v == "":
                return "http://localhost:3000,http://localhost:5173"
            
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return ",".join(str(x) for x in parsed)
            except (json.JSONDecodeError, ValueError):
                pass
            
            return v
        
        return str(v)
    
    def get_cors_origins_list(self) -> List[str]:
        """Get CORS origins as a list"""
        if not self.CORS_ORIGINS:
            return ["http://localhost:3000", "http://localhost:5173"]
        origins = [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]
        return origins if origins else ["http://localhost:3000", "http://localhost:5173"]
    
    def get_cors_allow_methods_list(self) -> List[str]:
        """Get CORS allow methods as a list"""
        if not self.CORS_ALLOW_METHODS:
            return ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        methods = [method.strip() for method in self.CORS_ALLOW_METHODS.split(",") if method.strip()]
        return methods if methods else ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    
    def get_cors_allow_headers_list(self) -> List[str]:
        """Get CORS allow headers as a list"""
        if not self.CORS_ALLOW_HEADERS or self.CORS_ALLOW_HEADERS == "*":
            return ["*"]
        headers = [header.strip() for header in self.CORS_ALLOW_HEADERS.split(",") if header.strip()]
        return headers if headers else ["*"]
    
    # ==========================================
    # Rate Limiting Settings
    # ==========================================
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 100
    RATE_LIMIT_PER_HOUR: int = 1000
    RATE_LIMIT_PER_DAY: int = 10000
    
    # ==========================================
    # Model Settings
    # ==========================================
    TEXT_MODEL_NAME: str = "all-MiniLM-L6-v2"
    IMAGE_MODEL_NAME: str = "ViT-B/32"
    COLLABORATIVE_MODEL_PATH: str = "models_storage/collaborative_model.pkl"
    TEXT_EMBEDDINGS_PATH: str = "text_embeddings"
    IMAGE_EMBEDDINGS_PATH: str = "image_embeddings"
    
    # ==========================================
    # Recommendation Settings
    # ==========================================
    DEFAULT_LIMIT: int = 20
    MAX_LIMIT: int = 100
    DEFAULT_NORMALIZE: bool = True
    
    # Scoring weights
    WEIGHT_HYBRID: float = 0.5
    WEIGHT_TEXT: float = 0.2
    WEIGHT_IMAGE: float = 0.2
    WEIGHT_INTERACTION: float = 0.05
    WEIGHT_FIELD: float = 0.05
    
    @field_validator("WEIGHT_HYBRID", "WEIGHT_TEXT", "WEIGHT_IMAGE", "WEIGHT_INTERACTION", "WEIGHT_FIELD")
    @classmethod
    def validate_weights(cls, v: float) -> float:
        """Ensure weights are between 0 and 1"""
        if not 0 <= v <= 1:
            raise ValueError("Weights must be between 0 and 1")
        return v
    
    # ==========================================
    # Validation Settings
    # ==========================================
    MAX_USER_ID_LENGTH: int = 50
    MAX_QUERY_LENGTH: int = 500
    MAX_IMAGE_SIZE_MB: int = 10
    MAX_IMAGE_SIZE_BYTES: int = 10 * 1024 * 1024
    
    ALLOWED_INTERACTION_ACTIONS: str = "view,click,add_to_cart,remove_from_cart,purchase,wishlist_add,wishlist_remove,review,rating,share"
    
    def get_allowed_interaction_actions_list(self) -> List[str]:
        """Get allowed interaction actions as a list"""
        if not self.ALLOWED_INTERACTION_ACTIONS:
            return ["view", "click", "add_to_cart", "remove_from_cart", "purchase", "wishlist_add", "wishlist_remove", "review", "rating", "share"]
        actions = [action.strip() for action in self.ALLOWED_INTERACTION_ACTIONS.split(",") if action.strip()]
        return actions if actions else ["view", "click", "add_to_cart", "remove_from_cart", "purchase"]
    
    # ==========================================
    # Redis Settings (Optional)
    # ==========================================
    REDIS_ENABLED: bool = False
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    REDIS_TTL_SECONDS: int = 300
    
    # ==========================================
    # Metrics & Monitoring
    # ==========================================
    METRICS_ENABLED: bool = True
    CSV_LOGGING_ENABLED: bool = True
    CSV_LOG_PATH: str = "performance_metrics.csv"
    
    PROMETHEUS_ENABLED: bool = False
    PROMETHEUS_PORT: int = 9090
    
    # ==========================================
    # Fallback Settings
    # ==========================================
    USE_FALLBACK_CATALOG: bool = False
    FALLBACK_PRODUCTS_COUNT: int = 10
    
    # ==========================================
    # Performance Settings
    # ==========================================
    ENABLE_CACHING: bool = True
    CACHE_TTL_SECONDS: int = 300
    MAX_BATCH_SIZE: int = 10
    
    # ==========================================
    # Logging Settings
    # ==========================================
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    
    def get_mongodb_collections(self):
        """Get MongoDB collection names"""
        return {
            "products": "products",
            "recommendation_logs": "recommendation_logs",
            "interaction_logs": "interaction_logs",
            "users": "users"
        }
    
    def get_scoring_weights(self):
        """Get normalized scoring weights"""
        weights = {
            "hybrid": self.WEIGHT_HYBRID,
            "text": self.WEIGHT_TEXT,
            "image": self.WEIGHT_IMAGE,
            "interaction": self.WEIGHT_INTERACTION,
            "field": self.WEIGHT_FIELD
        }
        total = sum(weights.values())
        if total == 0:
            return {k: 0.2 for k in weights.keys()}
        return {k: v / total for k, v in weights.items()}
    
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.ENVIRONMENT == "production"
    
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.ENVIRONMENT == "development"


settings = Settings()


def validate_settings():
    """Validate critical settings"""
    errors = []
    
    if not settings.MONGO_URI or settings.MONGO_URI == "":
        errors.append("MONGO_URI is not set")
    
    if settings.is_production() and settings.SECRET_KEY == "your-secret-key-change-in-production":
        errors.append("SECRET_KEY must be changed in production")
    
    cors_list = settings.get_cors_origins_list()
    if settings.is_production() and "*" in cors_list:
        errors.append("CORS should not allow all origins in production")
    
    weights = settings.get_scoring_weights()
    if abs(sum(weights.values()) - 1.0) > 0.01:
        errors.append(f"Scoring weights sum to {sum(weights.values())}, should be 1.0")
    
    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))


try:
    validate_settings()
except ValueError as e:
    if not settings.DEBUG:
        raise e
    else:
        print(f"   Configuration Warning: {e}")
