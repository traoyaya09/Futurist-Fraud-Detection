
"""
Fraud Detection System - Model Evaluation Script

Load saved models and evaluate them on test data.
Useful for:
- Validating loaded models work correctly
- Re-evaluating after deployment
- Comparing model versions

Usage:
    python scripts/evaluate_models.py

Output:
    - Evaluation metrics printed to console
    - Results saved to: results/evaluation_results.json
"""

import sys
import os
from pathlib import Path
import json
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from utils.data_loader import load_dataset, split_data
from utils.preprocessing import FeatureScaler, clean_data
from utils.metrics import evaluate_model, compare_models
from services import (
    LogisticRegressionService,
    RandomForestService,
    XGBoostService,
    IsolationForestService,
    StackingEnsembleService,
    load_stacking_ensemble_from_parts
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates trained fraud detection models."""
    
    def __init__(self):
        """Initialize the model evaluator."""
        self.models_dir = settings.MODEL_PATH
        self.results_dir = Path("results")
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage
        self.services = {}
        self.results = {}
        
        logger.info("ModelEvaluator initialized")
    
    def load_test_data(self):
        """Load and prepare test data."""
        logger.info("Loading test data...")
        
        # Load dataset
        df = load_dataset(settings.DATA_PATH)
        logger.info(f"Dataset loaded: {len(df):,} rows")
        
        # Clean and split
        df_clean = clean_data(df)
        X_train, X_test, y_train, y_test = split_data(df_clean, test_size=0.2)
        
        # Load scaler
        scaler_path = self.models_dir / "feature_scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        
        scaler = FeatureScaler(str(scaler_path))
        X_test_scaled = scaler.transform(X_test)
        
        logger.info(f"Test set: {len(X_test):,} samples")
        logger.info(f"Fraud samples: {y_test.sum():,} ({(y_test.sum()/len(y_test))*100:.2f}%)")
        
        self.X_test = X_test_scaled
        self.y_test = y_test
        
        return X_test_scaled, y_test
    
    def load_models(self):
        """Load all saved models."""
        logger.info("\nLoading saved models...")
        
        # Model paths
        model_files = {
            'logistic_regression': 'logistic_regression.pkl',
            'random_forest': 'random_forest.pkl',
            'xgboost': 'xgboost_model.pkl',
            'isolation_forest': 'isolation_forest.pkl',
            'stacking_ensemble': 'stacking_ensemble.pkl'
        }
        
        # Load each model
        for name, filename in model_files.items():
            model_path = self.models_dir / filename
            
            if not model_path.exists():
                logger.warning(f"  ⚠️  {name}: Not found ({model_path})")
                continue
            
            try:
                if name == 'logistic_regression':
                    service = LogisticRegressionService(model_path=str(model_path))
                elif name == 'random_forest':
                    service = RandomForestService(model_path=str(model_path))
                elif name == 'xgboost':
                    service = XGBoostService(model_path=str(model_path))
                elif name == 'isolation_forest':
                    service = IsolationForestService(model_path=str(model_path))
                elif name == 'stacking_ensemble':
                    # Load ensemble from parts
                    service = load_stacking_ensemble_from_parts(
                        str(self.models_dir / 'logistic_regression.pkl'),
                        str(self.models_dir / 'random_forest.pkl'),
                        str(self.models_dir / 'xgboost_model.pkl'),
                        str(self.models_dir / 'isolation_forest.pkl'),
                        str(model_path)
                    )
                
                self.services[name] = service
                logger.info(f"  ✓ {name}: Loaded successfully")
                
            except Exception as e:
                logger.error(f"  ❌ {name}: Failed to load - {str(e)}")
        
        logger.info(f"\n✓ Loaded {len(self.services)} models")
        return self.services
    
    def evaluate_all(self):
        """Evaluate all loaded models."""
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATING MODELS")
        logger.info("=" * 80)
        
        if not self.services:
            logger.error("No models loaded!")
            return {}
        
        # Evaluate each model
        for name, service in self.services.items():
            logger.info(f"\nEvaluating {name}...")
            
            try:
                # FIXED: Pass service wrapper consistently
                # The evaluate_model function will detect if it's a Service or raw model
                if name == 'stacking_ensemble':
                    # Stacking Ensemble is a special service
                    results = evaluate_model(
                        service,  # Pass the service itself
                        self.X_test,
                        self.y_test,
                        model_name=name.replace('_', ' ').title()
                    )
                else:
                    # Pass the service wrapper, not service.model
                    # This ensures consistent evaluation logic
                    results = evaluate_model(
                        service,  # Pass service wrapper (has .model and .predict_proba)
                        self.X_test,
                        self.y_test,
                        model_name=name.replace('_', ' ').title()
                    )
                
                self.results[name] = results
                
                logger.info(f"  ROC-AUC:   {results['roc_auc']:.4f}")
                logger.info(f"  Precision: {results['precision']:.4f}")
                logger.info(f"  Recall:    {results['recall']:.4f}")
                logger.info(f"  F1 Score:  {results['f1']:.4f}")
                
            except Exception as e:
                logger.error(f"  ❌ Evaluation failed: {str(e)}")
        
        return self.results
    
    def save_results(self):
        """Save evaluation results to JSON."""
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 80)
        
        # Prepare summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'test_samples': len(self.X_test),
            'fraud_samples': int(self.y_test.sum()),
            'models': {}
        }
        
        # Add model results
        for name, results in self.results.items():
            summary['models'][name] = {
                'roc_auc': float(results['roc_auc']),
                'precision': float(results['precision']),
                'recall': float(results['recall']),
                'f1': float(results['f1'])
            }
        
        # Find best model
        if self.results:
            best_model = max(self.results.items(), key=lambda x: x[1]['roc_auc'])
            summary['best_model'] = {
                'name': best_model[0],
                'roc_auc': float(best_model[1]['roc_auc'])
            }
            
            # Print ranking
            logger.info("\nModel Ranking (by ROC-AUC):")
            for i, (name, results) in enumerate(sorted(
                self.results.items(), 
                key=lambda x: x[1]['roc_auc'], 
                reverse=True
            ), 1):
                logger.info(f"  {i}. {name:20s}: {results['roc_auc']:.4f}")
            
            logger.info(f"\n🏆 Best Model: {summary['best_model']['name']}")
        
        # Save to file
        results_path = self.results_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n✓ Results saved to: {results_path}")
        logger.info("=" * 80)
        
        return summary
    
    def run(self):
        """Run full evaluation pipeline."""
        try:
            # Load test data
            self.load_test_data()
            
            # Load models
            self.load_models()
            
            # Evaluate
            self.evaluate_all()
            
            # Save results
            summary = self.save_results()
            
            logger.info("\n✅ Evaluation complete!")
            
            return summary
            
        except Exception as e:
            logger.error(f"\n❌ Evaluation failed: {str(e)}", exc_info=True)
            raise


def main():
    """Main entry point."""
    # Check if models exist
    models_dir = settings.MODEL_PATH
    if not models_dir.exists():
        logger.error(f"❌ Models directory not found: {models_dir}")
        logger.error("Please train models first: python scripts/train_models.py")
        return 1
    
    # Run evaluation
    evaluator = ModelEvaluator()
    evaluator.run()
    
    return 0


if __name__ == "__main__":
    exit(main())

