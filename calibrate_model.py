#!/usr/bin/env python3
"""
calibrate_model.py - Apply Platt Scaling to Trained Model

This script implements Platt scaling (probability calibration) on a trained
RandomForest model. It uses the validation set to fit the calibration and
evaluates the improvement on the test set.

Usage:
    python calibrate_model.py \
        --model_path models/rf_weighted_20251012_131423.joblib \
        --features_dir features_production \
        --output models/rf_weighted_calibrated.joblib

Author: Danilo A. Duran
Institution: Georgia Institute of Technology
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
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score, classification_report
import joblib

warnings.filterwarnings('ignore')

# ============= LOGGING =============

def setup_logging(verbose: bool = False, log_file: str = None) -> logging.Logger:
    """Configure logging with optional file output."""
    logger = logging.getLogger('calibrate_model')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# ============= DATA LOADING =============

def load_model_artifact(model_path: Path, logger: logging.Logger) -> Dict:
    """Load the trained model artifact."""
    logger.info(f"Loading model from {model_path}...")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    artifacts = joblib.load(model_path)
    
    # Validate structure
    required_keys = ['model', 'config', 'metrics']
    missing = [k for k in required_keys if k not in artifacts]
    if missing:
        raise ValueError(f"Model artifact missing keys: {missing}")
    
    model = artifacts['model']
    logger.info(f"✓ Loaded RandomForestClassifier")
    logger.info(f"  - n_estimators: {model.n_estimators}")
    logger.info(f"  - n_classes: {len(model.classes_)}")
    logger.info(f"  - classes: {model.classes_}")
    logger.info(f"  - Original test accuracy: {artifacts['metrics']['test']['accuracy']:.4f}")
    
    return artifacts


def load_dataset(features_dir: Path, split: str, logger: logging.Logger) -> Tuple:
    """Load features and labels for a given split."""
    features_path = features_dir / f'{split}_features.npz'
    labels_path = features_dir / f'{split}_labels.csv'
    
    if not features_path.exists():
        raise FileNotFoundError(f"Features not found: {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")
    
    X = load_npz(features_path)
    y = pd.read_csv(labels_path)['label']
    
    logger.info(f"  - {split}: {X.shape[0]:,} samples x {X.shape[1]:,} features")
    
    return X, y


# ============= CALIBRATION METRICS =============

def evaluate_calibration(model, X, y, split_name: str, logger: logging.Logger) -> Dict:
    """
    Evaluate calibration quality using Brier score and log loss.
    
    Brier score: measures mean squared difference between predicted probabilities
                 and actual outcomes (0 = perfect, 1 = worst)
    Log loss: measures the quality of probability predictions (lower is better)
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"CALIBRATION METRICS - {split_name.upper()}")
    logger.info(f"{'='*70}")
    
    # Get probability predictions
    y_proba = model.predict_proba(X)
    y_pred = model.predict(X)
    
    # Overall accuracy (unchanged by calibration)
    accuracy = accuracy_score(y, y_pred)
    logger.info(f"\nAccuracy: {accuracy:.4f} (unchanged by calibration)")
    
    # Calculate metrics per class
    classes = model.classes_
    brier_scores = {}
    
    logger.info(f"\nPer-Class Calibration Metrics:")
    logger.info(f"{'Class':<15} {'Brier Score':<15} {'Support':<10}")
    logger.info(f"{'-'*40}")
    
    for idx, class_name in enumerate(classes):
        # Create binary labels for this class
        y_binary = (y == class_name).astype(int)
        y_proba_class = y_proba[:, idx]
        
        # Brier score for this class
        brier = brier_score_loss(y_binary, y_proba_class)
        brier_scores[class_name] = brier
        
        support = y_binary.sum()
        logger.info(f"{class_name:<15} {brier:<15.6f} {support:<10,}")
    
    # Overall log loss (multiclass)
    overall_logloss = log_loss(y, y_proba)
    logger.info(f"\nOverall Log Loss: {overall_logloss:.6f}")
    logger.info(f"  (Lower is better, measures probability accuracy)")
    
    # Average Brier score
    avg_brier = np.mean(list(brier_scores.values()))
    logger.info(f"\nAverage Brier Score: {avg_brier:.6f}")
    logger.info(f"  (Lower is better, 0 = perfect, 1 = worst)")
    
    return {
        'accuracy': accuracy,
        'brier_scores': brier_scores,
        'avg_brier': avg_brier,
        'log_loss': overall_logloss,
        'y_proba': y_proba
    }


# ============= PLATT SCALING =============

def apply_platt_scaling(model, X_cal, y_cal, logger: logging.Logger):
    """
    Apply Platt scaling using CalibratedClassifierCV.
    
    Platt scaling fits a logistic regression on top of the classifier's
    outputs to calibrate the probabilities.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"APPLYING PLATT SCALING")
    logger.info(f"{'='*70}")
    
    logger.info(f"\nCalibration set: {X_cal.shape[0]:,} samples")
    logger.info(f"Fitting sigmoid calibration (Platt scaling)...")
    logger.info(f"This may take a few minutes...")
    
    start_time = datetime.now()
    
    # Create calibrated classifier
    # method='sigmoid' is Platt scaling
    # cv='prefit' means we use the already-trained model
    calibrated_model = CalibratedClassifierCV(
        model,
        method='sigmoid',  # Platt scaling
        cv='prefit',       # Model is already trained
        ensemble=False     # Don't create ensemble, just calibrate
    )
    
    # Fit calibration on validation set
    calibrated_model.fit(X_cal, y_cal)
    
    duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"✓ Platt scaling complete in {duration:.2f} seconds!")
    
    return calibrated_model


# ============= VISUALIZATION =============

def plot_calibration_curves(
    y_true, 
    y_proba_before, 
    y_proba_after, 
    classes, 
    output_dir: Path,
    logger: logging.Logger
):
    """
    Generate calibration curves comparing before and after Platt scaling.
    
    A calibration curve shows the relationship between predicted probabilities
    and actual frequencies. Perfect calibration follows the diagonal line.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"GENERATING CALIBRATION CURVES")
    logger.info(f"{'='*70}")
    
    for idx, class_name in enumerate(classes):
        # Create binary labels for this class
        y_binary = (y_true == class_name).astype(int)
        
        # Get probabilities for this class
        y_proba_class_before = y_proba_before[:, idx]
        y_proba_class_after = y_proba_after[:, idx]
        
        # Calculate calibration curves (10 bins)
        fraction_pos_before, mean_pred_before = calibration_curve(
            y_binary, y_proba_class_before, n_bins=10, strategy='uniform'
        )
        
        fraction_pos_after, mean_pred_after = calibration_curve(
            y_binary, y_proba_class_after, n_bins=10, strategy='uniform'
        )
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Perfect calibration line
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
        
        # Before calibration
        plt.plot(
            mean_pred_before, fraction_pos_before,
            's-', label='Before Platt Scaling', linewidth=2, markersize=8,
            color='#e74c3c'  # Red
        )
        
        # After calibration
        plt.plot(
            mean_pred_after, fraction_pos_after,
            'o-', label='After Platt Scaling', linewidth=2, markersize=8,
            color='#2ecc71'  # Green
        )
        
        plt.xlabel('Mean Predicted Probability', fontsize=12)
        plt.ylabel('Fraction of Positives (True Frequency)', fontsize=12)
        plt.title(f'Calibration Curve - Class: {class_name}', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.tight_layout()
        
        # Save
        output_path = output_dir / f'calibration_curve_{class_name}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Saved: {output_path}")
        plt.close()
    
    logger.info(f"\n✓ All calibration curves saved to {output_dir}/")


def create_comparison_summary(
    metrics_before: Dict,
    metrics_after: Dict,
    output_dir: Path,
    logger: logging.Logger
):
    """Create a visual comparison of before/after metrics."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    classes = list(metrics_before['brier_scores'].keys())
    brier_before = [metrics_before['brier_scores'][c] for c in classes]
    brier_after = [metrics_after['brier_scores'][c] for c in classes]
    
    # Brier score comparison
    x = np.arange(len(classes))
    width = 0.35
    
    axes[0].bar(x - width/2, brier_before, width, label='Before', color='#e74c3c', alpha=0.8)
    axes[0].bar(x + width/2, brier_after, width, label='After', color='#2ecc71', alpha=0.8)
    axes[0].set_xlabel('Class', fontsize=11)
    axes[0].set_ylabel('Brier Score', fontsize=11)
    axes[0].set_title('Brier Score by Class\n(Lower is Better)', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(classes, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Overall metrics comparison
    metrics = ['Avg Brier', 'Log Loss']
    before_vals = [metrics_before['avg_brier'], metrics_before['log_loss']]
    after_vals = [metrics_after['avg_brier'], metrics_after['log_loss']]
    
    x = np.arange(len(metrics))
    axes[1].bar(x - width/2, before_vals, width, label='Before', color='#e74c3c', alpha=0.8)
    axes[1].bar(x + width/2, after_vals, width, label='After', color='#2ecc71', alpha=0.8)
    axes[1].set_ylabel('Score', fontsize=11)
    axes[1].set_title('Overall Calibration Metrics\n(Lower is Better)', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = output_dir / 'calibration_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"  ✓ Saved comparison: {output_path}")
    plt.close()


# ============= SAVE ARTIFACTS =============

def save_calibrated_model(
    calibrated_model,
    original_artifacts: Dict,
    metrics_before: Dict,
    metrics_after: Dict,
    output_path: Path,
    logger: logging.Logger
):
    """Save the calibrated model with updated metadata."""
    
    logger.info(f"\n{'='*70}")
    logger.info(f"SAVING CALIBRATED MODEL")
    logger.info(f"{'='*70}")
    
    # Create new artifacts structure
    calibrated_artifacts = {
        'model': calibrated_model,  # The CalibratedClassifierCV wrapper
        'original_model': original_artifacts['model'],  # Original RF model
        'config': original_artifacts['config'],
        'original_metrics': original_artifacts['metrics'],
        'calibration_metrics': {
            'before': metrics_before,
            'after': metrics_after
        },
        'feature_engineering_metadata': original_artifacts.get('feature_engineering_metadata'),
        'calibration_timestamp': datetime.now().isoformat(),
        'calibration_method': 'platt_scaling'
    }
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated_artifacts, output_path)
    
    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"✓ Calibrated model saved to {output_path}")
    logger.info(f"  Size: {size_mb:.2f} MB")
    
    # Save JSON summary
    summary = {
        'calibration_method': 'platt_scaling',
        'calibration_timestamp': calibrated_artifacts['calibration_timestamp'],
        'original_model': str(original_artifacts['model'].__class__.__name__),
        'improvement': {
            'avg_brier_before': metrics_before['avg_brier'],
            'avg_brier_after': metrics_after['avg_brier'],
            'avg_brier_reduction': metrics_before['avg_brier'] - metrics_after['avg_brier'],
            'log_loss_before': metrics_before['log_loss'],
            'log_loss_after': metrics_after['log_loss'],
            'log_loss_reduction': metrics_before['log_loss'] - metrics_after['log_loss']
        }
    }
    
    summary_path = output_path.with_suffix('.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"✓ Summary saved to {summary_path}")


# ============= MAIN EXECUTION =============

def main():
    parser = argparse.ArgumentParser(
        description="Apply Platt scaling to calibrate model probabilities"
    )
    parser.add_argument('--model_path', type=Path, required=True,
                       help='Path to trained model artifact (.joblib)')
    parser.add_argument('--features_dir', type=Path, required=True,
                       help='Directory containing train/val/test features')
    parser.add_argument('--calibration_set', choices=['train', 'val'], default='val',
                       help='Which set to use for calibration (default: val)')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output path for calibrated model')
    parser.add_argument('--plots_dir', type=Path, default=Path('calibration_plots'),
                       help='Directory for calibration plots')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose logging')
    parser.add_argument('--log_file', type=str,
                       help='Log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose, args.log_file)
    
    logger.info("="*70)
    logger.info("MODEL PROBABILITY CALIBRATION WITH PLATT SCALING")
    logger.info("="*70)
    logger.info(f"\nInput model: {args.model_path}")
    logger.info(f"Features directory: {args.features_dir}")
    logger.info(f"Calibration set: {args.calibration_set}")
    logger.info(f"Output: {args.output}")
    
    try:
        # 1. Load model
        artifacts = load_model_artifact(args.model_path, logger)
        model = artifacts['model']
        
        # 2. Load datasets
        logger.info(f"\nLoading datasets from {args.features_dir}...")
        X_cal, y_cal = load_dataset(args.features_dir, args.calibration_set, logger)
        X_test, y_test = load_dataset(args.features_dir, 'test', logger)
        
        # 3. Evaluate calibration BEFORE
        logger.info("\n" + "="*70)
        logger.info("STEP 1: BASELINE CALIBRATION EVALUATION")
        logger.info("="*70)
        
        metrics_before_test = evaluate_calibration(
            model, X_test, y_test, 'Test Set (Before)', logger
        )
        
        # 4. Apply Platt scaling
        logger.info("\n" + "="*70)
        logger.info("STEP 2: APPLYING PLATT SCALING")
        logger.info("="*70)
        
        calibrated_model = apply_platt_scaling(model, X_cal, y_cal, logger)
        
        # 5. Evaluate calibration AFTER
        logger.info("\n" + "="*70)
        logger.info("STEP 3: CALIBRATED MODEL EVALUATION")
        logger.info("="*70)
        
        metrics_after_test = evaluate_calibration(
            calibrated_model, X_test, y_test, 'Test Set (After)', logger
        )
        
        # 6. Compare improvements
        logger.info("\n" + "="*70)
        logger.info("CALIBRATION IMPROVEMENT SUMMARY")
        logger.info("="*70)
        
        brier_improvement = metrics_before_test['avg_brier'] - metrics_after_test['avg_brier']
        logloss_improvement = metrics_before_test['log_loss'] - metrics_after_test['log_loss']
        
        logger.info(f"\nAverage Brier Score:")
        logger.info(f"  Before:  {metrics_before_test['avg_brier']:.6f}")
        logger.info(f"  After:   {metrics_after_test['avg_brier']:.6f}")
        logger.info(f"  Change:  {brier_improvement:+.6f} ({brier_improvement/metrics_before_test['avg_brier']*100:+.2f}%)")
        
        logger.info(f"\nLog Loss:")
        logger.info(f"  Before:  {metrics_before_test['log_loss']:.6f}")
        logger.info(f"  After:   {metrics_after_test['log_loss']:.6f}")
        logger.info(f"  Change:  {logloss_improvement:+.6f} ({logloss_improvement/metrics_before_test['log_loss']*100:+.2f}%)")
        
        # 7. Generate visualizations
        logger.info("\n" + "="*70)
        logger.info("STEP 4: GENERATING VISUALIZATIONS")
        logger.info("="*70)
        
        plot_calibration_curves(
            y_test,
            metrics_before_test['y_proba'],
            metrics_after_test['y_proba'],
            model.classes_,
            args.plots_dir,
            logger
        )
        
        create_comparison_summary(
            metrics_before_test,
            metrics_after_test,
            args.plots_dir,
            logger
        )
        
        # 8. Save calibrated model
        logger.info("\n" + "="*70)
        logger.info("STEP 5: SAVING ARTIFACTS")
        logger.info("="*70)
        
        save_calibrated_model(
            calibrated_model,
            artifacts,
            metrics_before_test,
            metrics_after_test,
            args.output,
            logger
        )
        
        # Final summary
        logger.info("\n" + "="*70)
        logger.info("CALIBRATION COMPLETE!")
        logger.info("="*70)
        logger.info(f"\n✓ Calibrated model: {args.output}")
        logger.info(f"✓ Calibration plots: {args.plots_dir}/")
        logger.info(f"\nNext steps:")
        logger.info(f"  1. Review calibration curves in '{args.plots_dir}/'")
        logger.info(f"  2. Use calibrated model for predictions with reliable probabilities")
        logger.info(f"  3. When model says 90% confidence, it should be correct ~90% of the time")
        
    except Exception as e:
        logger.error(f"\n❌ Error during calibration: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
