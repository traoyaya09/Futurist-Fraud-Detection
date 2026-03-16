"""
Model evaluation metrics.

Functions for calculating and reporting model performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    auc
)
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for ROC-AUC)
        
    Returns:
        Dict with all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    # Add ROC-AUC if probabilities provided
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
    
    # Calculate confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update({
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    })
    
    # Calculate additional metrics
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    return metrics


def calculate_roc_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate ROC-AUC and curve data.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        
    Returns:
        Tuple of (roc_auc, fpr, tpr, thresholds)
    """
    roc_auc = roc_auc_score(y_true, y_proba)
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    
    return roc_auc, fpr, tpr, thresholds


def calculate_precision_recall(
    y_true: np.ndarray,
    y_proba: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Precision-Recall AUC and curve data.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        
    Returns:
        Tuple of (pr_auc, precision, recall, thresholds)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    
    return pr_auc, precision, recall, thresholds


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, int]:
    """
    Calculate confusion matrix components.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dict with TN, FP, FN, TP
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    cm_dict = {
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    }
    
    # Log results
    logger.info("Confusion Matrix:")
    logger.info(f"  TN: {tn}  FP: {fp}")
    logger.info(f"  FN: {fn}  TP: {tp}")
    
    return cm_dict


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    predict_method: str = None
) -> Dict[str, float]:
    """
    Evaluate a trained model on test data.
    
    Intelligently detects if `model` is:
    - A Service wrapper (has .model attribute)
    - A raw scikit-learn model
    - A custom object with predict_proba method
    
    Args:
        model: Trained model (Service wrapper or raw sklearn model)
        X_test: Test features
        y_test: Test labels
        model_name: Name for logging
        predict_method: (Optional) Force specific prediction method
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Smart detection: Service wrapper vs raw model
    if hasattr(model, 'model') and hasattr(model, 'predict_proba'):
        # It's a Service wrapper - use the wrapper's predict_proba
        y_pred_proba = model.predict_proba(X_test)
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
            y_pred_proba = y_pred_proba[:, 1]  # Get fraud probability
        y_pred = (y_pred_proba > 0.5).astype(int)
        actual_model = model.model  # For feature importance, etc.
    elif hasattr(model, 'predict_proba') and not hasattr(model, 'model'):
        # It's either a raw sklearn model or a custom predictor (like StackingEnsemble)
        if predict_method == 'predict_proba' or not hasattr(model, 'predict'):
            # Custom predictor that only has predict_proba
            y_pred_proba = model.predict_proba(X_test)
            if isinstance(y_pred_proba, np.ndarray) and len(y_pred_proba.shape) > 1:
                y_pred_proba = y_pred_proba[:, 1] if y_pred_proba.shape[1] == 2 else y_pred_proba
            elif isinstance(y_pred_proba, (list, np.ndarray)) and len(y_pred_proba.shape) == 1:
                pass  # Already 1D array
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            # Raw sklearn model
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
        actual_model = model
    else:
        # Fallback: assume it has predict method
        y_pred = model.predict(X_test)
        # Try to get probabilities for ROC-AUC
        if hasattr(model, 'decision_function'):
            y_pred_proba = model.decision_function(X_test)
        elif hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            # No probabilities available - use predictions
            y_pred_proba = y_pred.astype(float)
        actual_model = model
    
    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
    }
    
    return metrics


def calculate_fraud_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate fraud-specific metrics.
    
    Focuses on fraud detection performance (class 1).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        
    Returns:
        Dict with fraud-focused metrics
    """
    # Get confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Fraud detection rate (recall for fraud class)
    fraud_detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # False alarm rate (FP rate)
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # Precision for fraud class
    fraud_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    metrics = {
        'fraud_detection_rate': fraud_detection_rate,
        'false_alarm_rate': false_alarm_rate,
        'fraud_precision': fraud_precision,
        'frauds_detected': int(tp),
        'frauds_missed': int(fn),
        'false_alarms': int(fp),
        'total_frauds': int(tp + fn)
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
    
    return metrics


def compare_models(
    results_dict: Dict[str, Dict]
) -> pd.DataFrame:
    """
    Compare multiple model results.
    
    Args:
        results_dict: Dict mapping model names to evaluation results
        
    Returns:
        DataFrame with comparison
    """
    comparison_data = []
    
    for model_name, results in results_dict.items():
        metrics = results.get('metrics', {})
        row = {
            'Model': model_name,
            'Accuracy': metrics.get('accuracy', 0),
            'Precision': metrics.get('precision', 0),
            'Recall': metrics.get('recall', 0),
            'F1-Score': metrics.get('f1_score', 0),
            'ROC-AUC': results.get('roc_auc', 0)
        }
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('ROC-AUC', ascending=False)
    
    logger.info("\nModel Comparison:")
    logger.info(df.to_string(index=False))
    
    return df


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Find optimal classification threshold.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'precision', 'recall')
        
    Returns:
        Tuple of (optimal_threshold, best_metric_value)
    """
    thresholds = np.linspace(0, 1, 100)
    best_threshold = 0.5
    best_score = 0.0
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    logger.info(f"Optimal threshold for {metric}: {best_threshold:.3f} (score: {best_score:.4f})")
    
    return best_threshold, best_score

