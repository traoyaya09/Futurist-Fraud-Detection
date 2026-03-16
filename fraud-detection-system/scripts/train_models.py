
"""
Fraud Detection System - Model Training Script

This script trains all 5 fraud detection models and saves them to disk:
1. Logistic Regression
2. Random Forest
3. XGBoost
4. Isolation Forest
5. Stacking Ensemble (combines all 4 base models)

Usage:
    python scripts/train_models.py

Output:
    - Trained models saved to: trained_models/
    - Evaluation results saved to: results/
    - Visualizations saved to: results/plots/

Author: AI-Powered E-Commerce Platform
Date: 2026-03-16
"""

import sys
import os
from pathlib import Path
import time
from datetime import datetime
import json
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from utils.data_loader import (
    load_dataset,
    validate_dataset,
    split_data,
    get_feature_names
)
from utils.preprocessing import (
    FeatureScaler,
    clean_data,
    balance_dataset
)
from utils.metrics import (
    evaluate_model,
    compare_models
)
from utils.visualization import (
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison
)
from services import (
    create_logistic_regression_service,
    create_random_forest_service,
    create_xgboost_service,
    create_isolation_forest_service,
    create_stacking_ensemble_service
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Orchestrates the training of all fraud detection models."""
    
    def __init__(self, data_path: str = None):
        """
        Initialize the model trainer.
        
        Args:
            data_path: Path to creditcard.csv (default: from settings)
        """
        self.data_path = data_path or settings.DATA_PATH
        self.models_dir = settings.MODEL_PATH
        self.results_dir = Path("results")
        self.plots_dir = self.results_dir / "plots"
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for trained models and results
        self.services = {}
        self.results = {}
        self.training_times = {}
        
        logger.info(f"ModelTrainer initialized")
        logger.info(f"Data path: {self.data_path}")
        logger.info(f"Models directory: {self.models_dir}")
        logger.info(f"Results directory: {self.results_dir}")
    
    def load_and_prepare_data(self):
        """
        Load and prepare data for training.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, scaler)
        """
        logger.info("=" * 80)
        logger.info("STEP 1: LOADING AND PREPARING DATA")
        logger.info("=" * 80)
        
        # Load dataset
        logger.info(f"Loading dataset from: {self.data_path}")
        df = load_dataset(self.data_path)
        logger.info(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
        
        # Validate dataset
        logger.info("Validating dataset structure...")
        validate_dataset(df)
        logger.info("✓ Dataset validation passed")
        
        # Check class distribution
        fraud_count = df['Class'].sum()
        legitimate_count = len(df) - fraud_count
        fraud_percentage = (fraud_count / len(df)) * 100
        
        logger.info(f"Class distribution:")
        logger.info(f"  - Legitimate transactions: {legitimate_count:,} ({100-fraud_percentage:.2f}%)")
        logger.info(f"  - Fraudulent transactions: {fraud_count:,} ({fraud_percentage:.2f}%)")
        
        # Clean data
        logger.info("Cleaning data (removing NaN, infinity)...")
        df_clean = clean_data(df)
        logger.info(f"✓ Data cleaned: {len(df_clean):,} rows remaining")
        
        # Split data
        logger.info("Splitting data into train/test sets (80/20)...")
        X_train, X_test, y_train, y_test = split_data(df_clean, test_size=0.2)
        logger.info(f"✓ Train set: {len(X_train):,} samples")
        logger.info(f"✓ Test set: {len(X_test):,} samples")
        
        # Feature scaling
        logger.info("Fitting StandardScaler on training data...")
        scaler = FeatureScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        logger.info("✓ Feature scaling complete")
        
        # Save scaler
        scaler_path = self.models_dir / "feature_scaler.pkl"
        scaler.save(str(scaler_path))
        logger.info(f"✓ Scaler saved to: {scaler_path}")
        
        # Balance training data (for supervised models)
        logger.info("Balancing training data using SMOTE...")
        X_train_balanced, y_train_balanced = balance_dataset(
            X_train_scaled, 
            y_train,
            method='smote'
        )
        logger.info(f"✓ Balanced train set: {len(X_train_balanced):,} samples")
        
        fraud_balanced = y_train_balanced.sum()
        logger.info(f"  - Legitimate: {len(y_train_balanced) - fraud_balanced:,}")
        logger.info(f"  - Fraudulent: {fraud_balanced:,}")
        
        # Store for later use
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        self.X_train_balanced = X_train_balanced
        self.y_train_balanced = y_train_balanced
        self.scaler = scaler
        
        logger.info("✓ Data preparation complete\n")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    
    def train_logistic_regression(self):
        """Train Logistic Regression model."""
        logger.info("=" * 80)
        logger.info("STEP 2: TRAINING LOGISTIC REGRESSION")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Create service
        service = create_logistic_regression_service()
        
        # Train on balanced data
        logger.info("Training Logistic Regression on balanced data...")
        train_results = service.train(
            self.X_train_balanced,
            self.y_train_balanced,
            X_val=self.X_test,
            y_val=self.y_test
        )
        
        training_time = time.time() - start_time
        self.training_times['logistic_regression'] = training_time
        
        logger.info(f"✓ Training complete in {training_time:.2f} seconds")
        
        # Evaluate
        logger.info("Evaluating on test set...")
        results = evaluate_model(
            service.model,
            service,  # FIXED: Pass service wrapper, not raw model
            self.X_test,
            self.y_test,
            model_name="Logistic Regression"
        )
        
        # Add training info
        results['training_time'] = training_time
        results['training_samples'] = len(self.X_train_balanced)
        
        logger.info(f"✓ ROC-AUC: {results['roc_auc']:.4f}")
        logger.info(f"✓ Precision: {results['precision']:.4f}")
        logger.info(f"✓ Recall: {results['recall']:.4f}")
        logger.info(f"✓ F1 Score: {results['f1']:.4f}")
        
        # Save model
        model_path = self.models_dir / "logistic_regression.pkl"
        service.save(str(model_path), metrics=results)
        logger.info(f"✓ Model saved to: {model_path}\n")
        
        # Store
        self.services['logistic_regression'] = service
        self.results['logistic_regression'] = results
        
        return service, results
    
    def train_random_forest(self):
        """Train Random Forest model."""
        logger.info("=" * 80)
        logger.info("STEP 3: TRAINING RANDOM FOREST")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Create service
        service = create_random_forest_service()
        
        # Train on balanced data
        logger.info("Training Random Forest on balanced data...")
        train_results = service.train(
            self.X_train_balanced,
            self.y_train_balanced,
            X_val=self.X_test,
            y_val=self.y_test
        )
        
        training_time = time.time() - start_time
        self.training_times['random_forest'] = training_time
        
        logger.info(f"✓ Training complete in {training_time:.2f} seconds")
        
        # Evaluate
        logger.info("Evaluating on test set...")
        results = evaluate_model(
            service.model,
            service,  # FIXED: Pass service wrapper
            self.X_test,
            self.y_test,
            model_name="Random Forest"
        )
        
        # Add training info
        results['training_time'] = training_time
        results['training_samples'] = len(self.X_train_balanced)
        
        logger.info(f"✓ ROC-AUC: {results['roc_auc']:.4f}")
        logger.info(f"✓ Precision: {results['precision']:.4f}")
        logger.info(f"✓ Recall: {results['recall']:.4f}")
        logger.info(f"✓ F1 Score: {results['f1']:.4f}")
        
        # Save model
        model_path = self.models_dir / "random_forest.pkl"
        service.save(str(model_path), metrics=results)
        logger.info(f"✓ Model saved to: {model_path}\n")
        
        # Store
        self.services['random_forest'] = service
        self.results['random_forest'] = results
        
        return service, results
    
    def train_xgboost(self):
        """Train XGBoost model."""
        logger.info("=" * 80)
        logger.info("STEP 4: TRAINING XGBOOST")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Create service
        service = create_xgboost_service()
        
        # Train on balanced data
        logger.info("Training XGBoost on balanced data...")
        train_results = service.train(
            self.X_train_balanced,
            self.y_train_balanced,
            X_val=self.X_test,
            y_val=self.y_test
        )
        
        training_time = time.time() - start_time
        self.training_times['xgboost'] = training_time
        
        logger.info(f"✓ Training complete in {training_time:.2f} seconds")
        
        # Evaluate
        logger.info("Evaluating on test set...")
        results = evaluate_model(
            service.model,
            service,  # FIXED: Pass service wrapper
            self.X_test,
            self.y_test,
            model_name="XGBoost"
        )
        
        # Add training info
        results['training_time'] = training_time
        results['training_samples'] = len(self.X_train_balanced)
        
        logger.info(f"✓ ROC-AUC: {results['roc_auc']:.4f}")
        logger.info(f"✓ Precision: {results['precision']:.4f}")
        logger.info(f"✓ Recall: {results['recall']:.4f}")
        logger.info(f"✓ F1 Score: {results['f1']:.4f}")
        
        # Save model
        model_path = self.models_dir / "xgboost_model.pkl"
        service.save(str(model_path), metrics=results)
        logger.info(f"✓ Model saved to: {model_path}\n")
        
        # Store
        self.services['xgboost'] = service
        self.results['xgboost'] = results
        
        return service, results
    
    def train_isolation_forest(self):
        """Train Isolation Forest model."""
        logger.info("=" * 80)
        logger.info("STEP 5: TRAINING ISOLATION FOREST")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Create service
        service = create_isolation_forest_service()
        
        # Train on ORIGINAL (unbalanced) data - unsupervised
        logger.info("Training Isolation Forest on original data (unsupervised)...")
        logger.info("Note: Isolation Forest doesn't use labels during training")
        
        train_results = service.train(
            self.X_train,  # Original unbalanced data
            self.y_train,  # Labels only for evaluation
            X_val=self.X_test,
            y_val=self.y_test
        )
        
        training_time = time.time() - start_time
        self.training_times['isolation_forest'] = training_time
        
        logger.info(f"✓ Training complete in {training_time:.2f} seconds")
        logger.info(f"✓ Normalization params: min={service.score_min:.4f}, max={service.score_max:.4f}")
        
        # Evaluate
        logger.info("Evaluating on test set...")
        results = evaluate_model(
            service.model,
            service,  # FIXED: Pass service wrapper
            self.X_test,
            self.y_test,
            model_name="Isolation Forest",
            predict_method='predict_proba'  # Use our custom predict_proba
        )
        
        # Add training info
        results['training_time'] = training_time
        results['training_samples'] = len(self.X_train)
        results['score_min'] = service.score_min
        results['score_max'] = service.score_max
        
        logger.info(f"✓ ROC-AUC: {results['roc_auc']:.4f}")
        logger.info(f"✓ Precision: {results['precision']:.4f}")
        logger.info(f"✓ Recall: {results['recall']:.4f}")
        logger.info(f"✓ F1 Score: {results['f1']:.4f}")
        
        # Save model
        model_path = self.models_dir / "isolation_forest.pkl"
        service.save(str(model_path), metrics=results)
        logger.info(f"✓ Model saved to: {model_path}\n")
        
        # Store
        self.services['isolation_forest'] = service
        self.results['isolation_forest'] = results
        
        return service, results
    
    def train_stacking_ensemble(self):
        """Train Stacking Ensemble model."""
        logger.info("=" * 80)
        logger.info("STEP 6: TRAINING STACKING ENSEMBLE")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Get base models
        base_models = {
            'logistic_regression': self.services['logistic_regression'],
            'random_forest': self.services['random_forest'],
            'xgboost': self.services['xgboost'],
            'isolation_forest': self.services['isolation_forest']
        }
        
        # Create ensemble service
        logger.info("Creating Stacking Ensemble from base models...")
        service = create_stacking_ensemble_service(base_models)
        
        # Train meta-learner
        logger.info("Training meta-learner on balanced data...")
        train_results = service.train(
            self.X_train_balanced,
            self.y_train_balanced,
            X_val=self.X_test,
            y_val=self.y_test
        )
        
        training_time = time.time() - start_time
        self.training_times['stacking_ensemble'] = training_time
        
        logger.info(f"✓ Training complete in {training_time:.2f} seconds")
        
        # Evaluate
        logger.info("Evaluating on test set...")
        results = evaluate_model(
            service,  # Pass service (has predict_proba method)
            service,  # FIXED: Already correct - stacking ensemble service
            self.X_test,
            self.y_test,
            model_name="Stacking Ensemble",
            predict_method='predict_proba'
        )
        
        # Add training info
        results['training_time'] = training_time
        results['training_samples'] = len(self.X_train_balanced)
        results['base_models'] = list(base_models.keys())
        
        logger.info(f"✓ ROC-AUC: {results['roc_auc']:.4f}")
        logger.info(f"✓ Precision: {results['precision']:.4f}")
        logger.info(f"✓ Recall: {results['recall']:.4f}")
        logger.info(f"✓ F1 Score: {results['f1']:.4f}")
        
        # Save ensemble
        ensemble_path = self.models_dir / "stacking_ensemble.pkl"
        service.save(str(ensemble_path), metrics=results)
        logger.info(f"✓ Ensemble saved to: {ensemble_path}\n")
        
        # Store
        self.services['stacking_ensemble'] = service
        self.results['stacking_ensemble'] = results
        
        return service, results
    
    def generate_visualizations(self):
        """Generate all visualization plots."""
        logger.info("=" * 80)
        logger.info("STEP 7: GENERATING VISUALIZATIONS")
        logger.info("=" * 80)
        
        # Get predictions from all models
        predictions = {}
        for name, service in self.services.items():
            if name == 'stacking_ensemble':
                predictions[name] = service.predict_proba(self.X_test)
            else:
                predictions[name] = service.model.predict_proba(self.X_test)[:, 1]
        
        # 1. ROC Curves (all models)
        logger.info("Generating ROC curves...")
        roc_path = self.plots_dir / "roc_curves_all_models.html"
        plot_roc_curve(
            self.y_test,
            predictions,
            save_path=str(roc_path)
        )
        logger.info(f"✓ Saved to: {roc_path}")
        
        # 2. Precision-Recall Curves (all models)
        logger.info("Generating Precision-Recall curves...")
        pr_path = self.plots_dir / "precision_recall_all_models.html"
        plot_precision_recall_curve(
            self.y_test,
            predictions,
            save_path=str(pr_path)
        )
        logger.info(f"✓ Saved to: {pr_path}")
        
        # 3. Confusion Matrices (individual models)
        logger.info("Generating confusion matrices...")
        for name, service in self.services.items():
            if name == 'stacking_ensemble':
                y_pred = (service.predict_proba(self.X_test) > 0.5).astype(int)
            else:
                y_pred = service.model.predict(self.X_test)
            
            cm_path = self.plots_dir / f"confusion_matrix_{name}.html"
            plot_confusion_matrix(
                self.y_test,
                y_pred,
                model_name=name.replace('_', ' ').title(),
                save_path=str(cm_path)
            )
            logger.info(f"  ✓ {name}: {cm_path}")
        
        # 4. Feature Importance (tree-based models)
        logger.info("Generating feature importance plots...")
        feature_names = get_feature_names()
        
        for name in ['random_forest', 'xgboost']:
            service = self.services[name]
            importance = service.get_feature_importance()
            
            fi_path = self.plots_dir / f"feature_importance_{name}.html"
            plot_feature_importance(
                importance,
                feature_names,
                model_name=name.replace('_', ' ').title(),
                save_path=str(fi_path),
                top_n=20
            )
            logger.info(f"  ✓ {name}: {fi_path}")
        
        # 5. Model Comparison
        logger.info("Generating model comparison chart...")
        comparison_path = self.plots_dir / "model_comparison.html"
        plot_model_comparison(
            self.results,
            save_path=str(comparison_path)
        )
        logger.info(f"✓ Saved to: {comparison_path}\n")
    
    def save_results(self):
        """Save training results to JSON."""
        logger.info("=" * 80)
        logger.info("STEP 8: SAVING RESULTS")
        logger.info("=" * 80)
        
        # Prepare results summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'data': {
                'total_samples': len(self.X_train) + len(self.X_test),
                'train_samples': len(self.X_train),
                'test_samples': len(self.X_test),
                'balanced_train_samples': len(self.X_train_balanced),
                'features': 30
            },
            'models': {},
            'training_times': self.training_times,
            'best_model': None
        }
        
        # Add model results
        for name, results in self.results.items():
            summary['models'][name] = {
                'roc_auc': float(results['roc_auc']),
                'precision': float(results['precision']),
                'recall': float(results['recall']),
                'f1': float(results['f1']),
                'training_time': float(results.get('training_time', 0))
            }
        
        # Determine best model (by ROC-AUC)
        best_model = max(self.results.items(), key=lambda x: x[1]['roc_auc'])
        summary['best_model'] = {
            'name': best_model[0],
            'roc_auc': float(best_model[1]['roc_auc'])
        }
        
        # Save to JSON
        results_path = self.results_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✓ Results saved to: {results_path}")
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total training time: {sum(self.training_times.values()):.2f} seconds")
        logger.info(f"\nModel Performance (ROC-AUC):")
        for name, results in sorted(self.results.items(), key=lambda x: x[1]['roc_auc'], reverse=True):
            logger.info(f"  {name:20s}: {results['roc_auc']:.4f}")
        logger.info(f"\nBest Model: {summary['best_model']['name']} (ROC-AUC: {summary['best_model']['roc_auc']:.4f})")
        logger.info("=" * 80 + "\n")
        
        return summary
    
    def train_all(self):
        """
        Train all models in sequence.
        
        This is the main entry point for training.
        """
        logger.info("🚀 Starting Fraud Detection Model Training")
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        try:
            # Step 1: Load and prepare data
            self.load_and_prepare_data()
            
            # Step 2-6: Train all models
            self.train_logistic_regression()
            self.train_random_forest()
            self.train_xgboost()
            self.train_isolation_forest()
            self.train_stacking_ensemble()
            
            # Step 7: Generate visualizations
            self.generate_visualizations()
            
            # Step 8: Save results
            summary = self.save_results()
            
            logger.info("✅ All models trained successfully!")
            logger.info(f"📁 Models saved to: {self.models_dir}")
            logger.info(f"📁 Results saved to: {self.results_dir}")
            logger.info(f"📊 Plots saved to: {self.plots_dir}")
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}", exc_info=True)
            raise


def main():
    """Main entry point."""
    # Check if data file exists
    data_path = settings.DATA_PATH
    if not os.path.exists(data_path):
        logger.error(f"❌ Data file not found: {data_path}")
        logger.error("Please download creditcard.csv and place it in data/raw/")
        logger.error("Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        return 1
    
    # Create trainer and run
    trainer = ModelTrainer(data_path=data_path)
    trainer.train_all()
    
    return 0


if __name__ == "__main__":
    exit(main())

