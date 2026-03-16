"""
Visualization utilities for fraud detection results.

Uses Plotly for interactive visualizations.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    model_name: str = "Model",
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Plot ROC curve.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        roc_auc: ROC-AUC score
        model_name: Model name for title
        save_path: Optional path to save HTML
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'{model_name} (AUC={roc_auc:.4f})',
        line=dict(color='blue', width=2)
    ))
    
    # Random baseline
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random (AUC=0.5)',
        line=dict(color='red', width=1, dash='dash')
    ))
    
    fig.update_layout(
        title=f'ROC Curve - {model_name}',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=800,
        height=600,
        template='plotly_white'
    )
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"Saved ROC curve to {save_path}")
    
    return fig


def plot_precision_recall_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    pr_auc: float,
    model_name: str = "Model",
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Plot Precision-Recall curve.
    
    Args:
        precision: Precision values
        recall: Recall values
        pr_auc: PR-AUC score
        model_name: Model name for title
        save_path: Optional path to save HTML
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode='lines',
        name=f'{model_name} (AUC={pr_auc:.4f})',
        line=dict(color='green', width=2),
        fill='tozeroy'
    ))
    
    fig.update_layout(
        title=f'Precision-Recall Curve - {model_name}',
        xaxis_title='Recall',
        yaxis_title='Precision',
        width=800,
        height=600,
        template='plotly_white'
    )
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"Saved PR curve to {save_path}")
    
    return fig


def plot_confusion_matrix(
    cm_dict: Dict[str, int],
    model_name: str = "Model",
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Plot interactive confusion matrix.
    
    Args:
        cm_dict: Dict with TN, FP, FN, TP
        model_name: Model name for title
        save_path: Optional path to save HTML
        
    Returns:
        Plotly figure
    """
    # Create matrix
    cm = np.array([
        [cm_dict['true_negatives'], cm_dict['false_positives']],
        [cm_dict['false_negatives'], cm_dict['true_positives']]
    ])
    
    # Create heatmap
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['Legitimate', 'Fraud'],
        y=['Legitimate', 'Fraud'],
        text_auto=True,
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        title=f'Confusion Matrix - {model_name}',
        width=600,
        height=600,
        template='plotly_white'
    )
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"Saved confusion matrix to {save_path}")
    
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    model_name: str = "Model",
    top_n: int = 20,
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Plot feature importance.
    
    Args:
        feature_names: List of feature names
        importances: Feature importance values
        model_name: Model name for title
        top_n: Number of top features to show
        save_path: Optional path to save HTML
        
    Returns:
        Plotly figure
    """
    # Create DataFrame and sort
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    df = df.sort_values('Importance', ascending=False).head(top_n)
    
    # Create bar chart
    fig = px.bar(
        df,
        x='Importance',
        y='Feature',
        orientation='h',
        title=f'Top {top_n} Feature Importances - {model_name}'
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        width=800,
        height=600,
        template='plotly_white'
    )
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"Saved feature importance to {save_path}")
    
    return fig


def plot_fraud_distribution(
    df: pd.DataFrame,
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Plot fraud vs legitimate transaction distribution.
    
    Args:
        df: DataFrame with 'Class' column
        save_path: Optional path to save HTML
        
    Returns:
        Plotly figure
    """
    counts = df['Class'].value_counts()
    
    fig = px.bar(
        x=['Legitimate', 'Fraud'],
        y=[counts[0], counts[1]],
        title='Transaction Distribution',
        labels={'x': 'Transaction Type', 'y': 'Count'}
    )
    
    # Add percentage annotations
    total = counts.sum()
    fig.update_traces(
        text=[
            f'{counts[0]} ({counts[0]/total*100:.2f}%)',
            f'{counts[1]} ({counts[1]/total*100:.2f}%)'
        ],
        textposition='outside'
    )
    
    fig.update_layout(
        width=600,
        height=500,
        template='plotly_white'
    )
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"Saved distribution plot to {save_path}")
    
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Plot training history (for neural networks).
    
    Args:
        history: Dict with 'loss', 'val_loss', etc.
        save_path: Optional path to save HTML
        
    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Loss', 'Accuracy')
    )
    
    # Loss plot
    if 'loss' in history:
        fig.add_trace(
            go.Scatter(
                y=history['loss'],
                mode='lines',
                name='Training Loss',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
    
    if 'val_loss' in history:
        fig.add_trace(
            go.Scatter(
                y=history['val_loss'],
                mode='lines',
                name='Validation Loss',
                line=dict(color='red')
            ),
            row=1, col=1
        )
    
    # Accuracy plot
    if 'accuracy' in history:
        fig.add_trace(
            go.Scatter(
                y=history['accuracy'],
                mode='lines',
                name='Training Accuracy',
                line=dict(color='blue')
            ),
            row=1, col=2
        )
    
    if 'val_accuracy' in history:
        fig.add_trace(
            go.Scatter(
                y=history['val_accuracy'],
                mode='lines',
                name='Validation Accuracy',
                line=dict(color='red')
            ),
            row=1, col=2
        )
    
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)
    
    fig.update_layout(
        title='Training History',
        width=1200,
        height=500,
        template='plotly_white'
    )
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"Saved training history to {save_path}")
    
    return fig


def plot_model_comparison(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Plot model comparison chart.
    
    Args:
        results_df: DataFrame with model comparison results
        save_path: Optional path to save HTML
        
    Returns:
        Plotly figure
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    fig = go.Figure()
    
    for _, row in results_df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row.get(m, 0) for m in metrics],
            theta=metrics,
            fill='toself',
            name=row['Model']
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title='Model Performance Comparison',
        width=800,
        height=600,
        template='plotly_white'
    )
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"Saved model comparison to {save_path}")
    
    return fig


def plot_threshold_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Plot precision/recall vs threshold.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        save_path: Optional path to save HTML
        
    Returns:
        Plotly figure
    """
    from sklearn.metrics import precision_recall_curve
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=precision[:-1],
        mode='lines',
        name='Precision',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=recall[:-1],
        mode='lines',
        name='Recall',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title='Precision-Recall vs Threshold',
        xaxis_title='Threshold',
        yaxis_title='Score',
        width=800,
        height=500,
        template='plotly_white'
    )
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"Saved threshold analysis to {save_path}")
    
    return fig
