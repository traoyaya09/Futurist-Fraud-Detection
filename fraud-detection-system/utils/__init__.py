"""
Utilities for Fraud Detection System.

Core utility functions for data loading, preprocessing, metrics, and model management.
"""

from .data_loader import load_dataset, validate_dataset, split_data
from .preprocessing import (
    FeatureScaler,
    balance_dataset,
    clean_data,
    engineer_features
)
from .metrics import (
    calculate_metrics,
    calculate_roc_auc,
    calculate_precision_recall,
    plot_confusion_matrix,
    evaluate_model
)
from .visualization import (
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_feature_importance,
    plot_fraud_distribution,
    plot_training_history
)
from .model_utils import (
    save_model,
    load_model,
    save_scaler,
    load_scaler,
    get_model_metadata,
    verify_model_compatibility
)

__all__ = [
    # Data loading
    "load_dataset",
    "validate_dataset",
    "split_data",
    # Preprocessing
    "FeatureScaler",
    "balance_dataset",
    "clean_data",
    "engineer_features",
    # Metrics
    "calculate_metrics",
    "calculate_roc_auc",
    "calculate_precision_recall",
    "plot_confusion_matrix",
    "evaluate_model",
    # Visualization
    "plot_roc_curve",
    "plot_precision_recall_curve",
    "plot_feature_importance",
    "plot_fraud_distribution",
    "plot_training_history",
    # Model utilities
    "save_model",
    "load_model",
    "save_scaler",
    "load_scaler",
    "get_model_metadata",
    "verify_model_compatibility",
]
