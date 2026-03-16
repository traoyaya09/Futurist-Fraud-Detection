





"""
Isolation Forest Service.

Unsupervised anomaly detection for fraud.
Detects outliers without labeled training data.
"""

import numpy as np
from typing import Optional, Dict, Any
import logging
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from utils.model_utils import load_model, save_model, create_model_metadata

logger = logging.getLogger(__name__)


class IsolationForestService:
    """
    Isolation Forest service for fraud detection.
    
    Characteristics:
    - Unsupervised anomaly detection
    - Finds outliers by isolation
    - No need for labeled fraud examples
    - Fast training and inference
    
    Use Case:
    - Detecting novel fraud patterns
    - Complementary to supervised models
    - When labeled data is scarce
    
    CRITICAL:
    - Returns anomaly scores (negative = outlier)
    - Need to normalize to [0, 1] for consistency
    - See models/responses.py for normalization
    """
    
    def __init__(self, model_path: str = None, metadata_path: str = None, **model_params):
        """
        Initialize Isolation Forest service.
        
        Args:
            model_path: Path to saved model (optional)
            metadata_path: Path to metadata JSON (optional, for quick metadata access)
            **model_params: Parameters for IsolationForest
        """
        self.model = None
        self.score_min = None
        self.score_max = None
        
        if model_path:
            # Load from pickle
            logger.info(f"Loading Isolation Forest from {model_path}")
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.score_min = model_data.get('score_min')
            self.score_max = model_data.get('score_max')
            
            logger.info(f"✓ Model loaded (score range: [{self.score_min:.4f}, {self.score_max:.4f}])")
        elif metadata_path:
            # Load ONLY metadata (faster, no model unpickling)
            logger.info(f"Loading Isolation Forest metadata from {metadata_path}")
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.score_min = metadata['score_min']
            self.score_max = metadata['score_max']
            
            # Model must be loaded separately
            logger.info(f"✓ Metadata loaded (score range: [{self.score_min:.4f}, {self.score_max:.4f}])")
            logger.warning("Model not loaded - call load_model() or provide model_path")
        else:
            # Create new model
            default_params = {
                'contamination': 0.002,  # Expected fraud rate (~0.2%)
                'n_estimators': 100,
                'max_samples': 'auto',
                'random_state': 42,
                'n_jobs': -1
            }
            default_params.update(model_params)
            self.model = IsolationForest(**default_params)
            logger.info("✓ New Isolation Forest created")
        
        self.model_name = "Isolation Forest"
        self.model_type = "isolation_forest"
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train Isolation Forest model.
        
        Note: Isolation Forest is unsupervised, but we can use labels
        for evaluation if available.
        
        Args:
            X_train: Training features (scaled)
            y_train: Training labels (optional, for evaluation only)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dict with training results
        """
        logger.info(f"Training {self.model_name}...")
        logger.info(f"Training samples: {len(X_train)}")
        
        # Train model (unsupervised)
        self.model.fit(X_train)
        
        # CRITICAL FIX: Calculate and store normalization parameters from training data
        train_scores = self.model.score_samples(X_train)
        flipped_scores = -train_scores
        self.score_min = float(flipped_scores.min())
        self.score_max = float(flipped_scores.max())
        
        logger.info(f"Normalization parameters: min={self.score_min:.4f}, max={self.score_max:.4f}")
        
        results = {
            'model_name': self.model_name,
            'n_samples': len(X_train),
            'n_features': X_train.shape[1],
            'n_estimators': self.model.n_estimators,
            'contamination': self.model.contamination,
            'score_min': self.score_min,
            'score_max': self.score_max
        }
        
        # Evaluate if labels provided
        if y_train is not None:
            train_scores = self.score_samples(X_train)
            train_pred_proba = self.normalize_scores(train_scores)
            train_auc = roc_auc_score(y_train, train_pred_proba)
            results['train_auc'] = float(train_auc)
            logger.info(f"Training AUC: {train_auc:.4f}")
        
        # Validation metrics if provided
        if X_val is not None and y_val is not None:
            val_scores = self.score_samples(X_val)
            val_pred_proba = self.normalize_scores(val_scores)
            val_auc = roc_auc_score(y_val, val_pred_proba)
            results['val_auc'] = float(val_auc)
            logger.info(f"Validation AUC: {val_auc:.4f}")
        
        logger.info(f"Training complete")
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.
        
        Args:
            X: Feature matrix (scaled)
            
        Returns:
            Predictions: 1 for inliers, -1 for outliers/frauds
        """
        return self.model.predict(X)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores.
        
        CRITICAL: Scores are negative for outliers.
        More negative = more anomalous = higher fraud risk
        
        Args:
            X: Feature matrix (scaled)
            
        Returns:
            Anomaly scores (negative = outlier)
        """
        return self.model.score_samples(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict fraud probabilities (normalized scores).
        
        Converts anomaly scores to probabilities [0, 1]
        where 1 = high fraud risk.
        
        Args:
            X: Feature matrix (scaled)
            
        Returns:
            Fraud probabilities [0.0-1.0]
        """
        scores = self.score_samples(X)
        return self.normalize_scores(scores)
    
    def normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize anomaly scores to [0, 1] probabilities.
        
        CRITICAL FIX: Uses training-time min/max for consistent normalization.
        This ensures:
        - Same transaction always gets same score
        - Works for single transactions
        - No division by zero
        
        CRITICAL: Isolation Forest returns negative scores for outliers.
        We need to flip and scale to match supervised models.
        
        Method: Min-Max normalization + flip (using training statistics)
        - Most negative score → probability 1.0 (fraud)
        - Least negative score → probability 0.0 (legitimate)
        
        Args:
            scores: Raw anomaly scores from score_samples()
            
        Returns:
            Normalized probabilities [0.0-1.0]
            
        Raises:
            RuntimeError: If model hasn't been trained yet
        """
        if self.score_min is None or self.score_max is None:
            raise RuntimeError(
                "Model must be trained before normalizing scores. "
                "Call train() first to compute normalization parameters."
            )
        
        # Flip scores (more negative = higher fraud probability)
        flipped = -scores
        
        # Min-Max normalization using TRAINING statistics
        score_range = self.score_max - self.score_min
        
        if score_range == 0:
            # All training scores were identical (edge case)
            logger.warning("Score range is zero - returning middle probability")
            return np.full_like(scores, 0.5)
        
        normalized = (flipped - self.score_min) / score_range
        
        # Clip to [0, 1] in case of extreme outliers beyond training range
        normalized = np.clip(normalized, 0.0, 1.0)
        
        return normalized
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function (same as score_samples).
        
        Args:
            X: Feature matrix (scaled)
            
        Returns:
            Decision scores (negative = outlier)
        """
        return self.model.decision_function(X)
    
    def save(
        self,
        filepath: str,
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Save model to disk.
        
        CRITICAL: Also saves normalization metadata (score_min, score_max)
        to a separate JSON file for production consistency.
        
        Args:
            filepath: Path to save model (e.g., "isolation_forest.pkl")
            metrics: Optional evaluation metrics
        """
        # Save model with metadata
        model_data = {
            'model': self.model,
            'score_min': self.score_min,
            'score_max': self.score_max,
            'trained': True,
            'metrics': metrics or {}
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        # CRITICAL FIX: Save normalization metadata separately
        # This allows loading just the metadata without unpickling the model
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        metadata = {
            'score_min': float(self.score_min),
            'score_max': float(self.score_max),
            'trained': True,
            'metrics': metrics or {},
            'model_path': filepath,
            'created_at': self._get_timestamp()
        }
        
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Isolation Forest saved to {filepath}")
        logger.info(f"Normalization metadata saved to {metadata_path}")
        logger.info(f"Score range: [{self.score_min:.4f}, {self.score_max:.4f}]")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def load_model(self, model_path: str) -> None:
        """
        Load model from disk (useful if metadata was loaded first).
        
        Args:
            model_path: Path to model pickle file
        """
        logger.info(f"Loading Isolation Forest model from {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        
        # Update normalization params if not already set
        if self.score_min is None:
            self.score_min = model_data.get('score_min')
        if self.score_max is None:
            self.score_max = model_data.get('score_max')
        
        logger.info(f"✓ Model loaded successfully")
    
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
            'n_estimators': self.model.n_estimators,
            'contamination': self.model.contamination,
            'unsupervised': True,
            'n_features': self.model.n_features_in_ if hasattr(self.model, 'n_features_in_') else None,
            'score_min': self.score_min,
            'score_max': self.score_max
        }


def create_isolation_forest_service(
    n_estimators: int = 100,
    contamination: float = 0.001,
    max_samples: str = 'auto'
) -> IsolationForestService:
    """
    Factory function to create Isolation Forest service.
    
    Args:
        n_estimators: Number of isolation trees
        contamination: Expected proportion of outliers (fraud rate)
        max_samples: Number of samples to draw for each tree
        
    Returns:
        Configured IsolationForestService
    """
    return IsolationForestService(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        random_state=42,
        n_jobs=-1
    )






