#!/usr/bin/env python3
"""
train_supervised_model.py - Model Training on Pre-Processed Features

This script loads pre-processed, numerical feature matrices (e.g., .npz files)
and trains a classifier. It is designed to be the second step in a two-stage
ML pipeline, following the execution of feature_engineer.py.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import load_npz, issparse, csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

warnings.filterwarnings('ignore')

# --------------------------- Logging ---------------------------
def setup_logging(verbose: bool = False, log_file: str = None) -> logging.Logger:
    logger = logging.getLogger('train_model')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

# --------------------------- Data Loading ---------------------------
def load_preprocessed_data(features_dir: Path, logger: logging.Logger) -> Tuple:
    logger.info(f"Loading pre-processed data from {features_dir}/")

    # Check for required files
    required_files = [
        'train_features.npz', 'val_features.npz', 'test_features.npz',
        'train_labels.csv', 'val_labels.csv', 'test_labels.csv', 'metadata.json'
    ]
    for f in required_files:
        if not (features_dir / f).exists():
            raise FileNotFoundError(f"Missing required file in features directory: {f}")

    # Load sparse matrices
    X_train = load_npz(features_dir / 'train_features.npz')
    X_val   = load_npz(features_dir / 'val_features.npz')
    X_test  = load_npz(features_dir / 'test_features.npz')

    # Load labels
    y_train = pd.read_csv(features_dir / 'train_labels.csv')['label']
    y_val   = pd.read_csv(features_dir / 'val_labels.csv')['label']
    y_test  = pd.read_csv(features_dir / 'test_labels.csv')['label']

    with open(features_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    # Validation checks
    assert X_train.shape[0] == len(y_train), "Train data shape mismatch"
    assert X_val.shape[0]   == len(y_val),   "Validation data shape mismatch"
    assert X_test.shape[0]  == len(y_test),  "Test data shape mismatch"
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], "Feature dimension mismatch"

    logger.info("✓ Data loaded successfully:")
    logger.info(f"  - Train: {X_train.shape[0]:,} samples x {X_train.shape[1]:,} features")
    logger.info(f"  - Val:   {X_val.shape[0]:,} samples")
    logger.info(f"  - Test:  {X_test.shape[0]:,} samples")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, metadata

# --------------------------- Training ---------------------------
def train_model(X_train, y_train, config: Dict, logger: logging.Logger) -> RandomForestClassifier:
    logger.info("Training RandomForestClassifier...")
    
    model = RandomForestClassifier(
        n_estimators=config['n_estimators'],
        max_depth=config['max_depth'],
        min_samples_split=config['min_samples_split'],
        min_samples_leaf=config['min_samples_leaf'],
        max_features=config['max_features'],
        class_weight='balanced' if config['use_class_weights'] else None,
        random_state=config['random_seed'],
        n_jobs=-1,
        verbose=1 if logger.level == logging.DEBUG else 0
    )

    start_time = datetime.now()
    model.fit(X_train, y_train)
    duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"✓ Training completed in {duration:.2f} seconds.")
    return model

# --------------------------- Evaluation ---------------------------
def evaluate_model(model, X, y, split_name: str, logger: logging.Logger) -> Dict:
    logger.info(f"\n{'='*50}\nEvaluating on {split_name.upper()} set...\n{'='*50}")
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info("Classification Report:")
    logger.info(classification_report(y, y_pred, digits=4, zero_division=0))
    return {'accuracy': accuracy, 'report': classification_report(y, y_pred, output_dict=True)}

# --------------------------- Save ---------------------------
def save_artifacts(model, config: Dict, metrics: Dict, fe_metadata: Dict, output_path: Path, logger: logging.Logger):
    logger.info(f"\nSaving model and metrics to {output_path}...")
    
    # We save just the classifier, as preprocessing is handled by transformers.joblib
    artifacts = {
        'model': model,
        'config': config,
        'metrics': metrics,
        'feature_engineering_metadata': fe_metadata
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, output_path)
    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"✓ Model artifact saved ({size_mb:.2f} MB)")
    
    # Save a separate JSON summary for easy inspection
    summary_path = output_path.with_suffix('.json')
    with open(summary_path, 'w') as f:
        json.dump({'config': config, 'metrics': metrics}, f, indent=2)
    logger.info(f"✓ Metrics summary saved to {summary_path}")

# --------------------------- CLI and Main Execution ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Train a model on pre-processed features.")
    
    # Use the arguments that match your original command
    parser.add_argument('--features_dir', type=Path, required=True,
                        help='Directory containing pre-processed .npz and .csv files from feature_engineer.py.')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output path for the trained model artifact (.joblib).')
    
    # Hyperparameters
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=None)
    parser.add_argument('--min_samples_split', type=int, default=2)
    parser.add_argument('--min_samples_leaf', type=int, default=1)
    parser.add_argument('--max_features', default='sqrt')
    parser.add_argument('--use_class_weights', action='store_true')
    
    # Misc
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--log_file', type=str)

    args = parser.parse_args()
    logger = setup_logging(args.verbose, args.log_file)

    logger.info("="*50 + "\nMODEL TRAINING ON PRE-PROCESSED DATA\n" + "="*50)

    try:
        X_train, y_train, X_val, y_val, X_test, y_test, fe_metadata = load_preprocessed_data(args.features_dir, logger)

        config = {k: v for k, v in vars(args).items() if k not in ['features_dir', 'output', 'verbose', 'log_file']}
        config['max_depth'] = None if config.get('max_depth', 0) <= 0 else config['max_depth']
        
        model = train_model(X_train, y_train, config, logger)

        train_metrics = evaluate_model(model, X_train, y_train, 'train', logger)
        val_metrics   = evaluate_model(model, X_val, y_val, 'validation', logger)
        test_metrics  = evaluate_model(model, X_test, y_test, 'test', logger)
        metrics = {'train': train_metrics, 'validation': val_metrics, 'test': test_metrics}
        
        save_artifacts(model, config, metrics, fe_metadata, args.output, logger)
        
        logger.info("\n" + "="*50 + "\nTRAINING COMPLETE\n" + "="*50)
        logger.info(f"Final model artifact saved to: {args.output}")
        logger.info(f"Final Test Set Accuracy: {test_metrics['accuracy']:.4f}")

    except FileNotFoundError as e:
        logger.critical(f"A required file was not found. Please run feature_engineer.py first. Details: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred during training: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
