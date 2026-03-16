"""
Model management utilities.

Functions for saving, loading, and managing trained models.
"""

import os
import joblib
import json
from typing import Any, Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def save_model(
    model: Any,
    filepath: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save trained model to disk.
    
    Args:
        model: Trained model object
        filepath: Path to save model
        metadata: Optional metadata dict
    """
    # Create directory if needed
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save model
    joblib.dump(model, filepath)
    logger.info(f"Saved model to {filepath}")
    
    # Save metadata if provided
    if metadata:
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")


def load_model(filepath: str) -> Any:
    """
    Load trained model from disk.
    
    Args:
        filepath: Path to model file
        
    Returns:
        Loaded model object
        
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model not found: {filepath}")
    
    model = joblib.load(filepath)
    logger.info(f"Loaded model from {filepath}")
    
    return model


def save_scaler(
    scaler: Any,
    filepath: str
) -> None:
    """
    Save feature scaler to disk.
    
    Args:
        scaler: Fitted scaler object
        filepath: Path to save scaler
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(scaler, filepath)
    logger.info(f"Saved scaler to {filepath}")


def load_scaler(filepath: str) -> Any:
    """
    Load feature scaler from disk.
    
    Args:
        filepath: Path to scaler file
        
    Returns:
        Loaded scaler object
        
    Raises:
        FileNotFoundError: If scaler file doesn't exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Scaler not found: {filepath}")
    
    scaler = joblib.load(filepath)
    logger.info(f"Loaded scaler from {filepath}")
    
    return scaler


def get_model_metadata(filepath: str) -> Dict[str, Any]:
    """
    Load model metadata.
    
    Args:
        filepath: Path to model file
        
    Returns:
        Dict with metadata, or empty dict if not found
    """
    metadata_path = filepath.replace('.pkl', '_metadata.json')
    
    if not os.path.exists(metadata_path):
        logger.warning(f"Metadata not found: {metadata_path}")
        return {}
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Loaded metadata from {metadata_path}")
    return metadata


def verify_model_compatibility(
    model_path: str,
    scaler_path: str
) -> bool:
    """
    Verify model and scaler are compatible.
    
    Args:
        model_path: Path to model file
        scaler_path: Path to scaler file
        
    Returns:
        True if compatible, False otherwise
    """
    try:
        # Load model and scaler
        model = load_model(model_path)
        scaler = load_scaler(scaler_path)
        
        # Check if scaler has expected attributes
        if not hasattr(scaler, 'n_features_in_'):
            logger.error("Scaler missing n_features_in_ attribute")
            return False
        
        # Expected 30 features for fraud detection
        if scaler.n_features_in_ != 30:
            logger.error(f"Scaler expects {scaler.n_features_in_} features, expected 30")
            return False
        
        # Check if model can make predictions
        import numpy as np
        sample = np.random.randn(1, 30)
        sample_scaled = scaler.transform(sample)
        
        try:
            _ = model.predict(sample_scaled)
            logger.info("✅ Model and scaler are compatible")
            return True
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Compatibility check failed: {e}")
        return False


def create_model_metadata(
    model_name: str,
    model_type: str,
    metrics: Dict[str, float],
    training_date: Optional[str] = None,
    dataset_size: Optional[int] = None,
    hyperparameters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create standardized model metadata.
    
    Args:
        model_name: Name of the model
        model_type: Type (logistic, rf, xgboost, etc.)
        metrics: Performance metrics dict
        training_date: Training date (defaults to now)
        dataset_size: Size of training dataset
        hyperparameters: Model hyperparameters
        
    Returns:
        Dict with standardized metadata
    """
    metadata = {
        'model_name': model_name,
        'model_type': model_type,
        'training_date': training_date or datetime.now().isoformat(),
        'metrics': metrics,
        'dataset_size': dataset_size,
        'hyperparameters': hyperparameters or {},
        'feature_count': 30,
        'feature_order': get_feature_names(),
        'version': '1.0.0'
    }
    
    return metadata


def get_feature_names() -> list:
    """
    Get ordered feature names.
    
    Returns:
        List of 30 feature names in correct order
    """
    features = ['Time']
    features.extend([f'V{i}' for i in range(1, 29)])
    features.append('Amount')
    return features


def list_available_models(models_dir: str = "trained_models") -> Dict[str, Dict[str, Any]]:
    """
    List all available trained models.
    
    Args:
        models_dir: Directory containing models
        
    Returns:
        Dict mapping model names to their metadata
    """
    if not os.path.exists(models_dir):
        logger.warning(f"Models directory not found: {models_dir}")
        return {}
    
    available_models = {}
    
    for filename in os.listdir(models_dir):
        if filename.endswith('.pkl'):
            model_path = os.path.join(models_dir, filename)
            model_name = filename.replace('.pkl', '')
            
            # Get metadata if available
            metadata = get_model_metadata(model_path)
            
            # Get file size
            file_size = os.path.getsize(model_path)
            
            available_models[model_name] = {
                'path': model_path,
                'size_bytes': file_size,
                'metadata': metadata
            }
    
    logger.info(f"Found {len(available_models)} trained models")
    return available_models


def validate_model_files(models_dir: str = "trained_models") -> Dict[str, bool]:
    """
    Validate all model files can be loaded.
    
    Args:
        models_dir: Directory containing models
        
    Returns:
        Dict mapping model names to validation status
    """
    validation_results = {}
    
    available_models = list_available_models(models_dir)
    
    for model_name, info in available_models.items():
        try:
            model = load_model(info['path'])
            validation_results[model_name] = True
            logger.info(f"✅ {model_name} loaded successfully")
        except Exception as e:
            validation_results[model_name] = False
            logger.error(f"❌ {model_name} failed to load: {e}")
    
    return validation_results


def get_model_size(filepath: str) -> Dict[str, Any]:
    """
    Get model file size information.
    
    Args:
        filepath: Path to model file
        
    Returns:
        Dict with size information
    """
    if not os.path.exists(filepath):
        return {'error': 'File not found'}
    
    size_bytes = os.path.getsize(filepath)
    size_kb = size_bytes / 1024
    size_mb = size_kb / 1024
    
    return {
        'size_bytes': size_bytes,
        'size_kb': round(size_kb, 2),
        'size_mb': round(size_mb, 2)
    }


def cleanup_old_models(
    models_dir: str = "trained_models",
    keep_latest: int = 3
) -> int:
    """
    Clean up old model versions, keeping only latest N.
    
    Args:
        models_dir: Directory containing models
        keep_latest: Number of latest versions to keep
        
    Returns:
        Number of models deleted
    """
    if not os.path.exists(models_dir):
        return 0
    
    # Get all model files with timestamps
    model_files = []
    for filename in os.listdir(models_dir):
        if filename.endswith('.pkl'):
            filepath = os.path.join(models_dir, filename)
            mtime = os.path.getmtime(filepath)
            model_files.append((filepath, mtime))
    
    # Sort by modification time (newest first)
    model_files.sort(key=lambda x: x[1], reverse=True)
    
    # Delete old models
    deleted_count = 0
    for filepath, _ in model_files[keep_latest:]:
        os.remove(filepath)
        logger.info(f"Deleted old model: {filepath}")
        
        # Delete metadata if exists
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        
        deleted_count += 1
    
    logger.info(f"Cleaned up {deleted_count} old models")
    return deleted_count
