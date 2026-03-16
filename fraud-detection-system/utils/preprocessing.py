"""
Data preprocessing utilities.

CRITICAL: Feature order and scaling must match training pipeline.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN
import joblib
import logging

logger = logging.getLogger(__name__)


class FeatureScaler:
    """
    Feature scaling wrapper with persistence.
    
    CRITICAL: Maintains feature order from training data.
    """
    
    def __init__(self, scaler_path: Optional[str] = None):
        """
        Initialize scaler.
        
        Args:
            scaler_path: Path to saved scaler (if loading existing)
        """
        if scaler_path:
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scaler from {scaler_path}")
            logger.info(f"Scaler expects {self.scaler.n_features_in_} features")
        else:
            self.scaler = StandardScaler()
            logger.info("Initialized new StandardScaler")
    
    def fit(self, X: pd.DataFrame) -> 'FeatureScaler':
        """
        Fit scaler to training data.
        
        Args:
            X: Training features
            
        Returns:
            Self for chaining
        """
        self.scaler.fit(X)
        logger.info(f"Fitted scaler on {X.shape[0]} samples, {X.shape[1]} features")
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted scaler.
        
        CRITICAL: Input must have same feature order as training data.
        
        Args:
            X: Features to transform
            
        Returns:
            Scaled features as numpy array
        """
        if not hasattr(self.scaler, 'mean_'):
            raise ValueError("Scaler not fitted. Call fit() first or load trained scaler.")
        
        # Verify feature count
        if X.shape[1] != self.scaler.n_features_in_:
            raise ValueError(
                f"Feature count mismatch: expected {self.scaler.n_features_in_}, "
                f"got {X.shape[1]}"
            )
        
        scaled = self.scaler.transform(X)
        return scaled
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit scaler and transform in one step.
        
        Args:
            X: Training features
            
        Returns:
            Scaled features as numpy array
        """
        self.fit(X)
        return self.transform(X)
    
    def save(self, filepath: str) -> None:
        """
        Save scaler to disk.
        
        Args:
            filepath: Path to save scaler
        """
        joblib.dump(self.scaler, filepath)
        logger.info(f"Saved scaler to {filepath}")
    
    def get_feature_stats(self) -> dict:
        """
        Get scaling statistics (mean, std).
        
        Returns:
            Dict with mean and scale (std) arrays
        """
        if not hasattr(self.scaler, 'mean_'):
            return {}
        
        return {
            'mean': self.scaler.mean_.tolist(),
            'scale': self.scaler.scale_.tolist(),
            'n_features': self.scaler.n_features_in_
        }


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset by removing duplicates and handling missing values.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    initial_count = len(df)
    
    # Remove duplicates
    df = df.drop_duplicates()
    duplicates_removed = initial_count - len(df)
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate records")
    
    # Handle missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        logger.warning(f"Found {missing_count} missing values, dropping rows")
        df = df.dropna()
    
    logger.info(f"Cleaned data: {len(df)} records remaining")
    return df


def balance_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = 'smote',
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Balance dataset using oversampling.
    
    Args:
        X: Feature matrix
        y: Target labels
        method: 'smote' or 'adasyn'
        random_state: Random seed
        
    Returns:
        Tuple of (X_resampled, y_resampled)
    """
    logger.info(f"Original class distribution:\n{y.value_counts()}")
    
    if method.lower() == 'smote':
        sampler = SMOTE(random_state=random_state)
    elif method.lower() == 'adasyn':
        sampler = ADASYN(random_state=random_state)
    else:
        raise ValueError(f"Unsupported balancing method: {method}")
    
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    
    logger.info(f"Resampled with {method.upper()}")
    logger.info(f"New class distribution:\n{pd.Series(y_resampled).value_counts()}")
    
    return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer additional features (if needed).
    
    Currently creditcard.csv is already PCA-transformed,
    so no additional engineering needed.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    # For creditcard.csv, features are already PCA-transformed
    # This function is a placeholder for future feature engineering
    logger.info("No additional feature engineering required (PCA already applied)")
    return df


def prepare_inference_data(
    transaction_array: np.ndarray,
    scaler_path: str
) -> np.ndarray:
    """
    Prepare transaction data for model inference.
    
    CRITICAL: This is the production inference pipeline.
    Must match training preprocessing exactly.
    
    Args:
        transaction_array: Raw transaction features (1, 30)
        scaler_path: Path to trained scaler
        
    Returns:
        Scaled features ready for prediction
    """
    # Verify shape
    if transaction_array.shape != (1, 30):
        raise ValueError(
            f"Expected shape (1, 30), got {transaction_array.shape}"
        )
    
    # Load trained scaler
    scaler = FeatureScaler(scaler_path)
    
    # Transform (CRITICAL: uses same scaling parameters as training)
    scaled = scaler.transform(
        pd.DataFrame(transaction_array, columns=get_feature_names())
    )
    
    return scaled


def get_feature_names() -> list:
    """
    Get ordered feature names.
    
    CRITICAL: Must match data_loader.get_feature_names()
    
    Returns:
        List of 30 feature names in correct order
    """
    features = ['Time']
    features.extend([f'V{i}' for i in range(1, 29)])
    features.append('Amount')
    return features


def verify_preprocessing_pipeline(
    sample_data: np.ndarray,
    scaler_path: str
) -> bool:
    """
    Verify preprocessing pipeline works correctly.
    
    Args:
        sample_data: Sample transaction (1, 30)
        scaler_path: Path to scaler
        
    Returns:
        True if pipeline works, False otherwise
    """
    try:
        scaled = prepare_inference_data(sample_data, scaler_path)
        
        # Verify output shape
        assert scaled.shape == (1, 30), f"Output shape mismatch: {scaled.shape}"
        
        # Verify no NaN values
        assert not np.isnan(scaled).any(), "NaN values in scaled output"
        
        logger.info("✅ Preprocessing pipeline verification passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Preprocessing pipeline verification failed: {e}")
        return False


def create_preprocessing_config() -> dict:
    """
    Create configuration dict for preprocessing pipeline.
    
    Returns:
        Dict with preprocessing settings
    """
    return {
        'scaling_method': 'StandardScaler',
        'feature_count': 30,
        'feature_order': get_feature_names(),
        'balancing_methods': ['SMOTE', 'ADASYN'],
        'expected_input_shape': (1, 30),
        'expected_output_shape': (1, 30)
    }
