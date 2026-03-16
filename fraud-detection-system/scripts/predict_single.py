"""
Fraud Detection System - Single Transaction Prediction

Test script for making predictions on a single transaction.
Useful for:
- Quick testing of loaded models
- Debugging prediction pipeline
- Understanding model behavior

Usage:
    # Use sample transaction
    python scripts/predict_single.py
    
    # Use custom transaction (via code modification)
    # Edit the transaction features in this file and run

Output:
    - Fraud probability
    - Risk level
    - Base model contributions (for ensemble)
"""

import sys
import os
from pathlib import Path
import numpy as np
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from utils.data_loader import create_sample_transaction, get_feature_names
from utils.preprocessing import FeatureScaler
from services import load_stacking_ensemble_from_parts
from models.responses import RiskLevel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_ensemble():
    """Load the stacking ensemble model."""
    models_dir = settings.MODEL_PATH
    
    logger.info("Loading Stacking Ensemble model...")
    
    # Load ensemble
    ensemble = load_stacking_ensemble_from_parts(
        str(models_dir / 'logistic_regression.pkl'),
        str(models_dir / 'random_forest.pkl'),
        str(models_dir / 'xgboost_model.pkl'),
        str(models_dir / 'isolation_forest.pkl'),
        str(models_dir / 'stacking_ensemble.pkl')
    )
    
    logger.info("✓ Ensemble loaded successfully")
    
    return ensemble


def load_scaler():
    """Load the feature scaler."""
    scaler_path = settings.MODEL_PATH / "feature_scaler.pkl"
    
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    
    logger.info("Loading feature scaler...")
    scaler = FeatureScaler(str(scaler_path))
    logger.info("✓ Scaler loaded successfully")
    
    return scaler


def get_risk_level(probability: float) -> str:
    """Map probability to risk level."""
    if probability >= 0.9:
        return "CRITICAL"
    elif probability >= 0.7:
        return "HIGH"
    elif probability >= 0.4:
        return "MEDIUM"
    else:
        return "LOW"


def predict_transaction(
    transaction: np.ndarray,
    ensemble,
    scaler,
    show_contributions: bool = True
):
    """
    Predict fraud probability for a transaction.
    
    Args:
        transaction: Raw transaction features (30 features)
        ensemble: Loaded stacking ensemble
        scaler: Loaded feature scaler
        show_contributions: Whether to show base model contributions
        
    Returns:
        Dict with prediction results
    """
    # Scale features
    transaction_scaled = scaler.transform(transaction.reshape(1, -1))
    
    # Predict
    probability = ensemble.predict_proba(transaction_scaled)[0]
    risk_level = get_risk_level(probability)
    
    # Get base model contributions (if available)
    contributions = None
    if show_contributions:
        try:
            contributions = ensemble.get_model_contributions(transaction_scaled)
        except Exception as e:
            logger.warning(f"Could not get contributions: {e}")
    
    return {
        'probability': probability,
        'risk_level': risk_level,
        'contributions': contributions
    }


def print_results(transaction: np.ndarray, results: dict):
    """Print prediction results in a nice format."""
    print("\n" + "=" * 80)
    print("FRAUD DETECTION PREDICTION")
    print("=" * 80)
    
    # Transaction summary
    print("\nTransaction Features:")
    feature_names = get_feature_names()
    print(f"  Time:   {transaction[0]:.2f}")
    print(f"  Amount: ${transaction[-1]:.2f}")
    print(f"  V1-V28: (PCA-transformed features)")
    
    # Prediction
    print("\nPrediction:")
    probability = results['probability']
    risk_level = results['risk_level']
    
    # Color coding based on risk
    risk_colors = {
        'LOW': '🟢',
        'MEDIUM': '🟡',
        'HIGH': '🟠',
        'CRITICAL': '🔴'
    }
    
    print(f"  Fraud Probability: {probability:.2%}")
    print(f"  Risk Level:        {risk_colors[risk_level]} {risk_level}")
    
    # Contributions
    if results['contributions']:
        print("\nBase Model Contributions:")
        contributions = results['contributions']
        for model, prob in contributions.items():
            bar_length = int(prob * 50)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"  {model:20s}: {bar} {prob:.2%}")
    
    # Recommendation
    print("\nRecommendation:")
    if risk_level == 'CRITICAL':
        print("  ⛔ BLOCK TRANSACTION - Extremely high fraud risk")
    elif risk_level == 'HIGH':
        print("  ⚠️  FLAG FOR REVIEW - High fraud risk detected")
    elif risk_level == 'MEDIUM':
        print("  ⚡ ENHANCED VERIFICATION - Moderate fraud risk")
    else:
        print("  ✅ APPROVE - Low fraud risk")
    
    print("=" * 80 + "\n")


def main():
    """Main entry point."""
    try:
        # Load models
        logger.info("Initializing fraud detection system...")
        ensemble = load_ensemble()
        scaler = load_scaler()
        
        # Create sample transaction
        # You can modify this to test different scenarios
        logger.info("\nGenerating sample transaction...")
        transaction = create_sample_transaction(fraud=False)
        
        # Alternative: Create a custom transaction
        # transaction = np.array([
        #     123.45,  # Time
        #     -1.5, 2.3, -0.8, 1.2, 0.5, -0.3, 0.9, -1.1, 0.4, -0.6,  # V1-V10
        #     1.8, -0.2, 0.7, -1.3, 0.1, 0.8, -0.5, 1.1, -0.9, 0.3,   # V11-V20
        #     -0.4, 0.6, -1.2, 0.2, -0.7, 1.4, -0.1, 0.9,             # V21-V28
        #     250.00   # Amount
        # ])
        
        # Make prediction
        logger.info("Making prediction...\n")
        results = predict_transaction(
            transaction,
            ensemble,
            scaler,
            show_contributions=True
        )
        
        # Print results
        print_results(transaction, results)
        
        # Interactive mode (optional)
        print("💡 Tip: Edit this file to test different transactions")
        print("   Set fraud=True in create_sample_transaction() for fraud examples")
        print("   Or create custom transactions with specific features\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Prediction failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
