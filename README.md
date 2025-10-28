# ML-Assisted WAF Alert Prioritization (In Progress)

This repository contains the code for a multi-stage machine learning pipeline designed to analyze ModSecurity WAF logs, train a model to classify traffic, and generate calibrated risk scores for alert prioritization.

## Codebase Overview

This project implements a **four-stage pipeline**: Parsing -> Feature Engineering -> Training -> Calibration.

## Scripts

* `parser.py`: Parses raw ModSecurity audit logs, extracts features (including WAF judgments like rule IDs and severities), and applies heuristic rules to programmatically label each log entry as 'normal', 'suspicious', or 'malicious'. Outputs a comprehensive CSV dataset.
* `feature_engineer.py`: Takes the labeled CSV from `parser.py`, performs extensive feature engineering (statistical features, TF-IDF on raw text and WAF judgments), handles scaling/encoding, splits data into train/val/test sets, and saves features as `.npz` files and transformers as `.joblib`.
* `train_supervised_model.py`: Loads the pre-processed features (`.npz` files) and trains a RandomForestClassifier, using **class weights** to handle imbalance. Saves the trained (but uncalibrated) model.
* `calibrate_model.py`: Takes the uncalibrated model from `train_supervised_model.py` and the validation set features. Applies **Platt scaling** (`CalibratedClassifierCV`) to adjust the model's probability outputs, making them reliable. Saves the final, **calibrated** model.
* `ml_test.py`: Evaluates a trained (and ideally calibrated) model against a pre-parsed CSV test set, generating classification reports, accuracy metrics, and optional risk score distribution plots.
* `synthetic_benign_logs.py`: Script to generate synthetic log data representing benign/normal web traffic.
* `synthetic_malicious_logs.py`: Script to generate synthetic log data representing malicious web traffic, likely using SecLists for realistic patterns.
* `synthetic_suspicious_logs.py`: Script to generate synthetic log data representing suspicious web traffic, likely using SecLists randomization to create ambiguous/edge cases.
* `plot_pr_curves.py`: Utility script to generate Precision-Recall curves for evaluating model performance, especially useful for imbalanced datasets.
* `filter_modsecurity.py`: Utility script likely used for filtering or selecting specific ModSecurity rules or logs based on certain criteria.

### **Disclaimer**

This software/information is provided **"As Is"**, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software/information or the use or other dealings in the software/information.