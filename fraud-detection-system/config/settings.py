
"""
Settings configuration for Fraud Detection System.

Uses Pydantic Settings for environment variable management with validation.
"""

from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    
    All settings can be overridden via environment variables.
    Create a .env file in the project root with your custom values.
    """
    
    # ================================
    # DATA CONFIGURATION
    # ================================
    DATA_PATH: str = Field(
        default="data/raw/creditcard.csv",
        description="Path to the raw credit card fraud dataset"
    )
    PROCESSED_DATA_PATH: str = Field(
        default="data/processed",
        description="Directory for processed data files"
    )
    MODELS_PATH: str = Field(
        default="trained_models",
        description="Directory to save trained models"
    )
    
    # Alias for MODEL_PATH (used in some configs)
    @property
    def MODEL_PATH(self) -> str:
        """Alias for MODELS_PATH for backward compatibility."""
        return self.MODELS_PATH
    
    RESULTS_PATH: str = Field(
        default="results",
        description="Directory to save evaluation results"
    )
    
    # ================================
    # MODEL CONFIGURATION
    # ================================
    OVERSAMPLING_METHOD: str = Field(
        default="smote",
        description="Oversampling method: 'smote' or 'adasyn'"
    )
    TEST_SIZE: float = Field(
        default=0.2,
        ge=0.1,
        le=0.5,
        description="Test set size (0.1 to 0.5)"
    )
    RANDOM_STATE: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    CONTAMINATION: float = Field(
        default=0.0017,
        ge=0.0001,
        le=0.1,
        description="Contamination parameter for Isolation Forest"
    )
    CV_FOLDS: int = Field(
        default=5,
        ge=2,
        le=10,
        description="Number of cross-validation folds"
    )
    
    # ================================
    # API CONFIGURATION
    # ================================
    API_HOST: str = Field(
        default="0.0.0.0",
        description="API host address"
    )
    API_PORT: int = Field(
        default=8001,
        ge=1024,
        le=65535,
        description="API port number"
    )
    API_DEBUG: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    API_RELOAD: bool = Field(
        default=False,
        description="Enable auto-reload on code changes"
    )
    API_WORKERS: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Number of API workers"
    )
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = Field(
        default=True,
        description="Enable rate limiting"
    )
    RATE_LIMIT_REQUESTS: int = Field(
        default=100,
        ge=1,
        description="Maximum requests per period"
    )
    RATE_LIMIT_PERIOD: int = Field(
        default=60,
        ge=1,
        description="Rate limit period in seconds"
    )
    
    # ================================
    # LOGGING CONFIGURATION
    # ================================
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL"
    )
    LOG_FILE: str = Field(
        default="fraud_detection.log",
        description="Log file path"
    )
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    
    # ================================
    # TRAINING CONFIGURATION
    # ================================
    ENABLE_EDA: bool = Field(
        default=False,
        description="Enable Exploratory Data Analysis visualizations"
    )
    SAVE_INTERMEDIATE_RESULTS: bool = Field(
        default=True,
        description="Save intermediate training results"
    )
    ENABLE_NESTED_CV: bool = Field(
        default=False,
        description="Enable nested cross-validation"
    )
    ENABLE_BAYESIAN_OPT: bool = Field(
        default=False,
        description="Enable Bayesian optimization for hyperparameters"
    )
    
    # ================================
    # DEPLOYMENT CONFIGURATION
    # ================================
    ENVIRONMENT: str = Field(
        default="development",
        description="Environment: development, staging, production"
    )
    CORS_ORIGINS: str = Field(
        default="*",
        description="CORS allowed origins (comma-separated or '*')"
    )
    CORS_ALLOW_CREDENTIALS: bool = Field(
        default=True,
        description="Allow credentials in CORS"
    )
    CORS_ALLOW_METHODS: str = Field(
        default="*",
        description="CORS allowed methods"
    )
    CORS_ALLOW_HEADERS: str = Field(
        default="*",
        description="CORS allowed headers"
    )
    
    # ================================
    # MONITORING CONFIGURATION
    # ================================
    PROMETHEUS_ENABLED: bool = Field(
        default=True,
        description="Enable Prometheus metrics"
    )
    PROMETHEUS_PORT: int = Field(
        default=9090,
        ge=1024,
        le=65535,
        description="Prometheus metrics port"
    )
    
    # ================================
    # SECURITY CONFIGURATION
    # ================================
    API_KEY_REQUIRED: bool = Field(
        default=False,
        description="Require API key for requests"
    )
    API_KEY: str = Field(
        default="your-secret-api-key-here",
        description="API key for authentication"
    )
    
    # ================================
    # FEATURE FLAGS
    # ================================
    ENABLE_BATCH_PREDICTIONS: bool = Field(
        default=True,
        description="Enable batch prediction endpoint"
    )
    ENABLE_MODEL_EXPLANATION: bool = Field(
        default=False,
        description="Enable model explanation features (SHAP, LIME)"
    )
    ENABLE_REAL_TIME_MONITORING: bool = Field(
        default=False,
        description="Enable real-time prediction monitoring"
    )
    
    class Config:
        """Pydantic config class."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
    
    def get_cors_origins(self) -> List[str]:
        """
        Parse CORS origins from string to list.
        
        Returns:
            List of allowed origins
        """
        if self.CORS_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
    
    def is_production(self) -> bool:
        """
        Check if running in production environment.
        
        Returns:
            True if production, False otherwise
        """
        return self.ENVIRONMENT.lower() == "production"
    
    def is_development(self) -> bool:
        """
        Check if running in development environment.
        
        Returns:
            True if development, False otherwise
        """
        return self.ENVIRONMENT.lower() == "development"


# Global settings instance
settings = Settings()


# Convenience function for getting settings
def get_settings() -> Settings:
    """
    Get the global settings instance.
    
    Returns:
        Settings instance
    """
    return settings

