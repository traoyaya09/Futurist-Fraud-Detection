"""
Data loading and validation utilities.

Functions for loading fraud detection datasets and validating data quality.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


def load_dataset(filepath: str, validate: bool = True) -> pd.DataFrame:
    """
    Load fraud detection dataset from CSV.
    
    Args:
        filepath: Path to CSV file
        validate: Whether to validate dataset after loading
        
    Returns:
        DataFrame with loaded data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If validation fails
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    logger.info(f"Loading dataset from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
    
    if validate:
        validate_dataset(df)
    
    return df


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate fraud detection dataset structure and quality.
    
    Expected structure for creditcard.csv:
    - 30 columns: Time, V1-V28, Amount, Class
    - No missing values in critical columns
    - Class column has values 0 (legitimate) and 1 (fraud)
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Dict with validation results
        
    Raises:
        ValueError: If critical validation fails
    """
    validation_results = {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "missing_values": df.isnull().sum().sum(),
        "duplicate_records": df.duplicated().sum(),
        "issues": []
    }
    
    # Check for required columns
    required_columns = ['Time', 'Amount', 'Class']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        validation_results["issues"].append(f"Missing required columns: {missing_cols}")
    
    # Check for V1-V28 columns
    v_columns = [f'V{i}' for i in range(1, 29)]
    missing_v_cols = [col for col in v_columns if col not in df.columns]
    if missing_v_cols:
        validation_results["issues"].append(f"Missing V columns: {missing_v_cols}")
    
    # Validate Class column
    if 'Class' in df.columns:
        unique_classes = df['Class'].unique()
        if not set(unique_classes).issubset({0, 1}):
            validation_results["issues"].append(
                f"Class column has invalid values: {unique_classes}"
            )
        
        fraud_count = (df['Class'] == 1).sum()
        fraud_rate = (fraud_count / len(df)) * 100
        validation_results["fraud_count"] = int(fraud_count)
        validation_results["fraud_rate"] = float(fraud_rate)
        
        logger.info(f"Dataset has {fraud_count} frauds ({fraud_rate:.2f}%)")
    
    # Check for missing values
    if validation_results["missing_values"] > 0:
        missing_by_col = df.isnull().sum()
        missing_cols = missing_by_col[missing_by_col > 0].to_dict()
        validation_results["missing_by_column"] = missing_cols
        logger.warning(f"Found {validation_results['missing_values']} missing values")
    
    # Check for duplicates
    if validation_results["duplicate_records"] > 0:
        logger.warning(f"Found {validation_results['duplicate_records']} duplicate records")
    
    # Raise error if critical issues found
    if validation_results["issues"]:
        raise ValueError(f"Dataset validation failed: {validation_results['issues']}")
    
    logger.info("Dataset validation passed")
    return validation_results


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    stratify: bool = True,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split dataset into training and testing sets.
    
    Args:
        df: DataFrame with features and Class column
        test_size: Fraction of data to use for testing
        stratify: Whether to stratify split by Class
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    if 'Class' not in df.columns:
        raise ValueError("DataFrame must have 'Class' column")
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=stratify_param,
        random_state=random_state
    )
    
    logger.info(f"Split data: Train={len(X_train)}, Test={len(X_test)}")
    logger.info(f"Train fraud rate: {(y_train == 1).sum() / len(y_train) * 100:.2f}%")
    logger.info(f"Test fraud rate: {(y_test == 1).sum() / len(y_test) * 100:.2f}%")
    
    return X_train, X_test, y_train, y_test


def load_processed_data(
    train_path: str,
    test_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load pre-processed training and test datasets.
    
    Args:
        train_path: Path to processed training data
        test_path: Path to processed test data
        
    Returns:
        Tuple of (train_df, test_df)
    """
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found: {test_path}")
    
    logger.info(f"Loading processed training data from {train_path}")
    train_df = pd.read_csv(train_path)
    
    logger.info(f"Loading processed test data from {test_path}")
    test_df = pd.read_csv(test_path)
    
    logger.info(f"Loaded train: {len(train_df)}, test: {len(test_df)}")
    
    return train_df, test_df


def get_feature_names() -> list:
    """
    Get ordered list of feature names for fraud detection.
    
    CRITICAL: This order must match TransactionRequest.to_array()
    
    Returns:
        List of feature names in correct order: [Time, V1-V28, Amount]
    """
    features = ['Time']
    features.extend([f'V{i}' for i in range(1, 29)])
    features.append('Amount')
    return features


def verify_feature_order(df: pd.DataFrame) -> bool:
    """
    Verify DataFrame columns match expected feature order.
    
    Args:
        df: DataFrame to verify
        
    Returns:
        True if order matches, False otherwise
    """
    expected = get_feature_names()
    
    # Get all columns except Class
    actual = [col for col in df.columns if col != 'Class']
    
    if actual == expected:
        logger.info("✅ Feature order matches expected order")
        return True
    else:
        logger.error(f"❌ Feature order mismatch!")
        logger.error(f"Expected: {expected}")
        logger.error(f"Actual: {actual}")
        return False


def create_sample_transaction() -> Dict[str, float]:
    """
    Create a sample transaction for testing.
    
    Returns:
        Dict with sample transaction features
    """
    sample = {
        'Time': 0.0,
        'Amount': 149.62
    }
    
    # Add V1-V28 with sample values
    v_values = [
        -1.3598071336738, -0.0727811733098497, 2.53634673796914,
        1.37815522427443, -0.338320769942518, 0.462387777762292,
        0.239598554061257, 0.0986979012610507, 0.363786969611213,
        0.0907941719789316, -0.551599533260813, -0.617800855762348,
        -0.991389847235408, -0.311169353699879, 1.46817697209427,
        -0.470400525259478, 0.207971241929242, 0.0257905801985591,
        0.403992960255733, 0.251412098239705, -0.018306777944153,
        0.277837575558899, -0.110473910188767, 0.0669280749146731,
        0.128539358273528, -0.189114843888824, 0.133558376740387,
        -0.0210530534538215
    ]
    
    for i, val in enumerate(v_values, 1):
        sample[f'V{i}'] = val
    
    return sample
