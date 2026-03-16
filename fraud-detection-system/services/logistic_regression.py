"""
Logistic Regression Service.

Baseline linear model for fraud detection.
Fast inference, good for high-volume transactions.
"""

import numpy as np
from typing import Optional, Dict, Any
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from utils.model_utils import load_model, save_model, create_model_metadata

logger = logging.getLogger(__name__)


class LogisticRegressionService:
    """
    Logistic Regression service for fraud detection.
    
    Characteristics:
    - Fast inference (~0.1ms per prediction)
    - Linear decision boundary
    - Good for well-separated data
    - Baseline model for comparison
    
    Use Case:
    - High-volume, low-latency predictions
    - Initial fraud screening
    - Feature importance analysis (coefficients)
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Logistic Regression service.
        
        Args:
            model_path: Path to trained model (if loading existing)
            **kwargs: Hyperparameters for new model
        """
        if model_path:
            self.model = load_model(model_path)
            logger.info(f"Loaded Logistic Regression from {model_path}")
        else:
            # Default hyperparameters for fraud detection
            default_params = {
                'penalty': 'l2',
                'C': 1.0,
                'solver': 'lbfgs',
                'max_iter': 1000,
                'class_weight': 'balanced',  # Handle imbalanced data
                'random_state': 42,
                'n_jobs': -1
            }
            default_params.update(kwargs)
            
            self.model = LogisticRegression(**default_params)
            logger.info(f"Initialized new Logistic Regression: {default_params}")
        
        self.model_name = "Logistic Regression"
        self.model_type = "logistic_regression"
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train Logistic Regression model.
        
        Args:
            X_train: Training features (scaled)
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dict with training results
        """
        logger.info(f"Training {self.model_name}...")
        logger.info(f"Training samples: {len(X_train)}")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Calculate training metrics
        train_pred_proba = self.model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_pred_proba)
        
        results = {
            'model_name': self.model_name,
            'train_auc': float(train_auc),
            'n_samples': len(X_train),
            'n_features': X_train.shape[1]
        }
        
        # Validation metrics if provided
        if X_val is not None and y_val is not None:
            val_pred_proba = self.model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_pred_proba)
            results['val_auc'] = float(val_auc)
            logger.info(f"Validation AUC: {val_auc:.4f}")
        
        logger.info(f"Training complete - Train AUC: {train_auc:.4f}")
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict fraud labels (0 or 1).
        
        Args:
            X: Feature matrix (scaled)
            
        Returns:
            Binary predictions (0=legitimate, 1=fraud)
        """
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict fraud probabilities.
        
        Args:
            X: Feature matrix (scaled)
            
        Returns:
            Fraud probabilities [0.0-1.0]
        """
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature coefficients (importance).
        
        Returns:
            Array of coefficients for each feature
        """
        return self.model.coef_[0]
    
    def get_top_features(self, feature_names: list, top_n: int = 10) -> Dict[str, float]:
        """
        Get top N most important features.
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            Dict mapping feature names to coefficients
        """
        coefs = np.abs(self.model.coef_[0])
        top_indices = np.argsort(coefs)[-top_n:][::-1]
        
        top_features = {
            feature_names[i]: float(self.model.coef_[0][i])
            for i in top_indices
        }
        
        return top_features
    
    def save(
        self,
        filepath: str,
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Save model to disk.
        
        Args:
            filepath: Path to save model
            metrics: Performance metrics to save with model
        """
        # Create metadata
        metadata = create_model_metadata(
            model_name=self.model_name,
            model_type=self.model_type,
            metrics=metrics or {},
            hyperparameters=self.model.get_params()
        )
        
        # Save model with metadata
        save_model(self.model, filepath, metadata)
        logger.info(f"Saved {self.model_name} to {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dict with model details
        """
        return {
            'name': self.model_name,
            'type': self.model_type,
            'hyperparameters': self.model.get_params(),
            'is_fitted': hasattr(self.model, 'coef_'),
            'n_features': len(self.model.coef_[0]) if hasattr(self.model, 'coef_') else None
        }


def create_logistic_regression_service(
    penalty: str = 'l2',
    C: float = 1.0,
    solver: str = 'lbfgs',
    max_iter: int = 1000
) -> LogisticRegressionService:
    """
    Factory function to create Logistic Regression service.
    
    Args:
        penalty: Regularization type ('l1', 'l2', 'elasticnet', 'none')
        C: Inverse of regularization strength (smaller = stronger)
        solver: Optimization algorithm
        max_iter: Maximum iterations
        
    Returns:
        Configured LogisticRegressionService
    """
    return LogisticRegressionService(
        penalty=penalty,
        C=C,
        solver=solver,
        max_iter=max_iter,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
