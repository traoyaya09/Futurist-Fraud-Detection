"""
Random Forest Service.

Tree-based ensemble for fraud detection.
Good balance of accuracy and interpretability.
"""

import numpy as np
from typing import Optional, Dict, Any
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from utils.model_utils import load_model, save_model, create_model_metadata

logger = logging.getLogger(__name__)


class RandomForestService:
    """
    Random Forest service for fraud detection.
    
    Characteristics:
    - Ensemble of decision trees
    - Robust to outliers
    - Feature importance via Gini/entropy
    - Medium inference speed (~1-5ms)
    
    Use Case:
    - Balanced accuracy-speed trade-off
    - Feature importance analysis
    - Non-linear pattern detection
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Random Forest service.
        
        Args:
            model_path: Path to trained model (if loading existing)
            **kwargs: Hyperparameters for new model
        """
        if model_path:
            self.model = load_model(model_path)
            logger.info(f"Loaded Random Forest from {model_path}")
        else:
            # Default hyperparameters for fraud detection
            default_params = {
                'n_estimators': 100,
                'max_depth': 20,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'max_features': 'sqrt',
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1,
                'verbose': 0
            }
            default_params.update(kwargs)
            
            self.model = RandomForestClassifier(**default_params)
            logger.info(f"Initialized new Random Forest: {default_params}")
        
        self.model_name = "Random Forest"
        self.model_type = "random_forest"
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train Random Forest model.
        
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
        logger.info(f"Number of trees: {self.model.n_estimators}")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Calculate training metrics
        train_pred_proba = self.model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_pred_proba)
        
        results = {
            'model_name': self.model_name,
            'train_auc': float(train_auc),
            'n_samples': len(X_train),
            'n_features': X_train.shape[1],
            'n_trees': self.model.n_estimators
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
        Get feature importance scores.
        
        Based on mean decrease in impurity (Gini importance).
        
        Returns:
            Array of importance scores for each feature
        """
        return self.model.feature_importances_
    
    def get_top_features(self, feature_names: list, top_n: int = 10) -> Dict[str, float]:
        """
        Get top N most important features.
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            Dict mapping feature names to importance scores
        """
        importances = self.model.feature_importances_
        top_indices = np.argsort(importances)[-top_n:][::-1]
        
        top_features = {
            feature_names[i]: float(importances[i])
            for i in top_indices
        }
        
        return top_features
    
    def get_tree_count(self) -> int:
        """
        Get number of trees in the forest.
        
        Returns:
            Number of decision trees
        """
        return len(self.model.estimators_)
    
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
        
        # Add tree count to metadata
        metadata['n_trees'] = self.get_tree_count()
        
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
            'is_fitted': hasattr(self.model, 'estimators_'),
            'n_trees': self.get_tree_count() if hasattr(self.model, 'estimators_') else None,
            'n_features': self.model.n_features_in_ if hasattr(self.model, 'n_features_in_') else None
        }


def create_random_forest_service(
    n_estimators: int = 100,
    max_depth: Optional[int] = 20,
    min_samples_split: int = 10,
    min_samples_leaf: int = 5
) -> RandomForestService:
    """
    Factory function to create Random Forest service.
    
    Args:
        n_estimators: Number of trees
        max_depth: Maximum depth of trees (None for unlimited)
        min_samples_split: Minimum samples to split node
        min_samples_leaf: Minimum samples in leaf node
        
    Returns:
        Configured RandomForestService
    """
    return RandomForestService(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
