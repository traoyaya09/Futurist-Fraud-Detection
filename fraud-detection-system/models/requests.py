"""
Request models for Fraud Detection API.

Pydantic models for validating incoming API requests.

CRITICAL: Feature order in to_array() MUST match training data order:
[Time, V1, V2, ..., V28, Amount]
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
import numpy as np


class TransactionRequest(BaseModel):
    """
    Single transaction prediction request.
    
    CRITICAL: Feature order matches creditcard.csv column order:
    Time, V1-V28, Amount (30 features total)
    
    Attributes:
        time: Seconds elapsed since first transaction in dataset
        v1-v28: PCA-transformed features from original transaction data
        amount: Transaction amount in dollars
        model_name: Optional model selection (default: stacking_ensemble)
    """
    
    # Time feature (column 0 in creditcard.csv)
    time: float = Field(
        ...,
        ge=0,
        description="Seconds elapsed since first transaction"
    )
    
    # PCA features V1-V28 (columns 1-28 in creditcard.csv)
    v1: float = Field(..., description="PCA feature 1")
    v2: float = Field(..., description="PCA feature 2")
    v3: float = Field(..., description="PCA feature 3")
    v4: float = Field(..., description="PCA feature 4")
    v5: float = Field(..., description="PCA feature 5")
    v6: float = Field(..., description="PCA feature 6")
    v7: float = Field(..., description="PCA feature 7")
    v8: float = Field(..., description="PCA feature 8")
    v9: float = Field(..., description="PCA feature 9")
    v10: float = Field(..., description="PCA feature 10")
    v11: float = Field(..., description="PCA feature 11")
    v12: float = Field(..., description="PCA feature 12")
    v13: float = Field(..., description="PCA feature 13")
    v14: float = Field(..., description="PCA feature 14")
    v15: float = Field(..., description="PCA feature 15")
    v16: float = Field(..., description="PCA feature 16")
    v17: float = Field(..., description="PCA feature 17")
    v18: float = Field(..., description="PCA feature 18")
    v19: float = Field(..., description="PCA feature 19")
    v20: float = Field(..., description="PCA feature 20")
    v21: float = Field(..., description="PCA feature 21")
    v22: float = Field(..., description="PCA feature 22")
    v23: float = Field(..., description="PCA feature 23")
    v24: float = Field(..., description="PCA feature 24")
    v25: float = Field(..., description="PCA feature 25")
    v26: float = Field(..., description="PCA feature 26")
    v27: float = Field(..., description="PCA feature 27")
    v28: float = Field(..., description="PCA feature 28")
    
    # Amount feature (column 29 in creditcard.csv)
    amount: float = Field(
        ...,
        ge=0,
        description="Transaction amount in dollars"
    )
    
    # Model selection
    model_name: Optional[str] = Field(
        default="stacking_ensemble",
        description="Model to use for prediction"
    )
    
    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """
        Validate model name is supported.
        
        Updated to Pydantic v2 field_validator syntax.
        """
        allowed_models = [
            "logistic_regression",
            "random_forest",
            "xgboost",
            "isolation_forest",
            "stacking_ensemble"
        ]
        if v not in allowed_models:
            raise ValueError(
                f"Model '{v}' not supported. Choose from: {', '.join(allowed_models)}"
            )
        return v
    
    def to_array(self) -> np.ndarray:
        """
        Convert transaction to numpy array for model prediction.
        
        CRITICAL: This order MUST match the column order in creditcard.csv:
        [Time, V1, V2, V3, ..., V28, Amount]
        
        DO NOT CHANGE THIS ORDER unless you retrain all models.
        
        Returns:
            2D numpy array (1, 30) with features in correct order
        """
        # EXACT order from creditcard.csv training data
        features = [
            self.time,    # Column 0
            self.v1,      # Column 1
            self.v2,      # Column 2
            self.v3,      # Column 3
            self.v4,      # Column 4
            self.v5,      # Column 5
            self.v6,      # Column 6
            self.v7,      # Column 7
            self.v8,      # Column 8
            self.v9,      # Column 9
            self.v10,     # Column 10
            self.v11,     # Column 11
            self.v12,     # Column 12
            self.v13,     # Column 13
            self.v14,     # Column 14
            self.v15,     # Column 15
            self.v16,     # Column 16
            self.v17,     # Column 17
            self.v18,     # Column 18
            self.v19,     # Column 19
            self.v20,     # Column 20
            self.v21,     # Column 21
            self.v22,     # Column 22
            self.v23,     # Column 23
            self.v24,     # Column 24
            self.v25,     # Column 25
            self.v26,     # Column 26
            self.v27,     # Column 27
            self.v28,     # Column 28
            self.amount   # Column 29
        ]
        return np.array(features).reshape(1, -1)
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "time": 0.0,
                "v1": -1.3598071336738,
                "v2": -0.0727811733098497,
                "v3": 2.53634673796914,
                "v4": 1.37815522427443,
                "v5": -0.338320769942518,
                "v6": 0.462387777762292,
                "v7": 0.239598554061257,
                "v8": 0.0986979012610507,
                "v9": 0.363786969611213,
                "v10": 0.0907941719789316,
                "v11": -0.551599533260813,
                "v12": -0.617800855762348,
                "v13": -0.991389847235408,
                "v14": -0.311169353699879,
                "v15": 1.46817697209427,
                "v16": -0.470400525259478,
                "v17": 0.207971241929242,
                "v18": 0.0257905801985591,
                "v19": 0.403992960255733,
                "v20": 0.251412098239705,
                "v21": -0.018306777944153,
                "v22": 0.277837575558899,
                "v23": -0.110473910188767,
                "v24": 0.0669280749146731,
                "v25": 0.128539358273528,
                "v26": -0.189114843888824,
                "v27": 0.133558376740387,
                "v28": -0.0210530534538215,
                "amount": 149.62,
                "model_name": "stacking_ensemble"
            }
        }


class BatchTransactionRequest(BaseModel):
    """
    Batch transaction prediction request.
    
    Attributes:
        transactions: List of transactions to predict (max 1000)
        model_name: Optional model selection (default: stacking_ensemble)
    """
    
    transactions: List[TransactionRequest] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of transactions (max 1000)"
    )
    model_name: Optional[str] = Field(
        default="stacking_ensemble",
        description="Model to use for predictions"
    )
    
    @field_validator("transactions")
    @classmethod
    def validate_batch_size(cls, v: List[TransactionRequest]) -> List[TransactionRequest]:
        """Validate batch size doesn't exceed maximum."""
        if len(v) > 1000:
            raise ValueError("Maximum batch size is 1000 transactions")
        return v
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "transactions": [
                    {
                        "time": 0.0,
                        "v1": -1.36, "v2": -0.07, "v3": 2.54,
                        "v4": 1.38, "v5": -0.34, "v6": 0.46,
                        "v7": 0.24, "v8": 0.10, "v9": 0.36,
                        "v10": 0.09, "v11": -0.55, "v12": -0.62,
                        "v13": -0.99, "v14": -0.31, "v15": 1.47,
                        "v16": -0.47, "v17": 0.21, "v18": 0.03,
                        "v19": 0.40, "v20": 0.25, "v21": -0.02,
                        "v22": 0.28, "v23": -0.11, "v24": 0.07,
                        "v25": 0.13, "v26": -0.19, "v27": 0.13,
                        "v28": -0.02,
                        "amount": 149.62
                    }
                ],
                "model_name": "stacking_ensemble"
            }
        }


class ModelSelectionRequest(BaseModel):
    """
    Request to select which model to use for predictions.
    
    Attributes:
        model_name: Name of the model to use
    """
    
    model_name: str = Field(
        ...,
        description="Model name to select"
    )
    
    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model name is supported."""
        allowed_models = [
            "logistic_regression",
            "random_forest",
            "xgboost",
            "isolation_forest",
            "stacking_ensemble"
        ]
        if v not in allowed_models:
            raise ValueError(
                f"Model '{v}' not supported. Choose from: {', '.join(allowed_models)}"
            )
        return v
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "model_name": "xgboost"
            }
        }
