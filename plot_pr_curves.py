#!/usr/bin/env python3
"""
plot_pr_curves.py

Loads a saved RandomForest model package + features, 
plots per-class precision-recall curves.

Usage:
  python plot_pr_curves.py --features_dir features_production \
      --model models/rf_weighted_20250930_175219.joblib \
      --output pr_curves.png
"""

import argparse
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import load_npz
from sklearn.metrics import precision_recall_curve, average_precision_score

def load_data(features_dir: Path):
    X_test = load_npz(features_dir / "test_features.npz")
    y_test = pd.read_csv(features_dir / "test_labels.csv")["label"]
    return X_test, y_test

def plot_pr_curves(model, X_test, y_test, output_path: Path):
    # predict_proba required
    y_prob = model.predict_proba(X_test)

    classes = model.classes_
    plt.figure(figsize=(8, 6))

    for i, cls in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_test == cls, y_prob[:, i])
        ap = average_precision_score(y_test == cls, y_prob[:, i])
        lw = 2 if cls == "suspicious" else 1
        plt.plot(recall, precision, lw=lw, label=f"{cls} (AP={ap:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves (One-vs-Rest)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved PR curves to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", type=str, required=True,
                        help="Directory with test_features.npz and test_labels.csv")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to .joblib model file")
    parser.add_argument("--output", type=str, default="pr_curves.png",
                        help="Output image filename")
    args = parser.parse_args()

    # Load model package
    package = joblib.load(args.model)
    model = package["model"]

    # Load test data
    X_test, y_test = load_data(Path(args.features_dir))

    # Plot PR curves
    plot_pr_curves(model, X_test, y_test, Path(args.output))

if __name__ == "__main__":
    main()
