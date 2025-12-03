# ML-Assisted WAF Alert Prioritization

Machine learning pipeline for ModSecurity WAF log analysis and intelligent alert triage. This system trains a Random Forest classifier on ModSecurity audit logs to provide probabilistic risk scores for alert prioritization.

## Prerequisites

- Python 3.10+
- ModSecurity v2.9.3+ with OWASP CRS v4.18.0
- ModSecurity audit logs in JSON format

## Installation

```bash
# Clone repository
git clone https://github.com/Dan-Duran/modsec-ml-helper.git
cd modsec-ml-helper

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Core Dependencies

- **scikit-learn** - Random Forest, feature engineering, calibration
- **pandas/numpy** - Data manipulation and numerical operations
- **scipy** - Sparse matrix handling (CSR format)
- **joblib** - Model serialization
- **matplotlib** - Visualization and plotting

## Pipeline Architecture

```
Raw ModSecurity Logs
    ↓
[1] parser.py → Labeled CSV
    ↓
[2] feature_engineer.py → Feature matrices (.npz) + Transformers
    ↓
[3] train_supervised_model.py → Trained model
    ↓
[4] calibrate_model.py → Calibrated model
    ↓
[5] ml_test.py / ml_test_analysis.py → Evaluation
```

## Stage 1: Parse ModSecurity Logs

Reconstructs HTTP transactions from ModSecurity audit logs and applies heuristic labeling.

**Input:** Raw ModSecurity audit logs

**Output:** CSV with extracted features and labels

```bash
python parser.py \
  --input modsec-logs/modsec_audit.log* \
  --output datasets/modsec_dataset.csv
```

**Heuristic labeling rules:**
- `malicious`: CRITICAL rule severity OR attack tool signatures
- `suspicious`: ERROR rule severity OR elevated anomaly scores
- `normal`: All other traffic

**Output columns:** timestamp, uri, payload, method, user_agent, rule_ids, rule_severities, label, etc.

## Stage 2: Feature Engineering

Transforms labeled CSV into sparse feature matrices with train/validation/test splits.

**Input:** Labeled CSV from parser

**Output:** `.npz` feature matrices + fitted `transformers.joblib`

```bash
# Full production run
python feature_engineer.py \
  --input datasets/modsec_dataset.csv \
  --mode fit \
  --output_dir features_production \
  --train_frac 0.7 \
  --val_frac 0.15 \
  --min_df 3 \
  --uri_max_features 800 \
  --payload_max_features 800 \
  --verbose \
  --log_file logs/feature_eng_$(date +%Y%m%d_%H%M%S).log

# Quick test run (limited samples)
python feature_engineer.py \
  --input datasets/modsec_dataset.csv \
  --mode fit \
  --output_dir features_test \
  --limit 10000 \
  --train_frac 0.7 \
  --val_frac 0.15 \
  --verbose
```

**Feature types:**
- **Statistical**: URI length, payload entropy, query depth, parameter count, temporal features
- **Evidence (TF-IDF)**: Character n-grams from URI/payload, word tokens from User-Agent
- **Judgment (TF-IDF)**: Triggered Rule IDs and Severities from WAF

**Outputs:**
- `X_train.npz`, `X_val.npz`, `X_test.npz` - Feature matrices
- `y_train.npz`, `y_val.npz`, `y_test.npz` - Labels
- `transformers.joblib` - Fitted scalers and vectorizers

## Stage 3: Train Random Forest

Trains weighted Random Forest classifier on engineered features.

**Input:** Feature matrices from Stage 2

**Output:** Trained model (`.joblib`)

```bash
# Baseline model (unweighted)
python train_supervised_model.py \
  --features_dir features_production \
  --output models/rf_baseline_$(date +%Y%m%d_%H%M%S).joblib \
  --n_estimators 200 \
  --max_depth 20 \
  --min_samples_split 10 \
  --min_samples_leaf 5 \
  --max_features log2 \
  --verbose \
  --log_file logs/training_baseline_$(date +%Y%m%d_%H%M%S).log

# Weighted model (recommended for imbalanced data)
python train_supervised_model.py \
  --features_dir features_production \
  --output models/rf_weighted_$(date +%Y%m%d_%H%M%S).joblib \
  --n_estimators 300 \
  --max_depth 22 \
  --min_samples_split 12 \
  --min_samples_leaf 6 \
  --max_features sqrt \
  --use_class_weights \
  --verbose \
  --log_file logs/training_$(date +%Y%m%d_%H%M%S).log
```

**Key parameters:**
- `--use_class_weights`: Applies `class_weight='balanced'` to handle imbalanced classes
- `--n_estimators`: Number of trees in forest
- `--max_depth`: Maximum tree depth
- `--max_features`: Features to consider per split

## Stage 4: Calibrate Model (Platt Scaling)

Applies sigmoid calibration to transform raw scores into reliable probabilities.

**Input:** Trained model + validation set

**Output:** Calibrated model + diagnostic plots

```bash
python calibrate_model.py \
  --model_path models/rf_weighted_20251012_131423.joblib \
  --features_dir features_production \
  --output models/rf_weighted_calibrated.joblib \
  --plots_dir calibration_plots \
  --verbose
```

**Metrics computed:**
- Brier Score (before/after)
- Log Loss (before/after)
- Per-class calibration curves

**Why calibrate:** Raw Random Forest scores are ordinal rankings, not true probabilities. Calibration ensures risk scores reflect actual likelihood.

## Stage 5: Testing and Evaluation

### Quantitative Testing (Batch)

Evaluates model on large test sets with aggregate metrics.

```bash
python ml_test.py \
  --csv-file test-logs/test-datasets/modsec_natural_mixed.csv \
  --model-path models/rf_weighted_calibrated.joblib \
  --transformers-path features_production/transformers.joblib \
  --expected mixed \
  --plot-dist risk_distribution.png
```

**Parameters:**
- `--expected`: Expected class distribution (options: `mixed`, `malicious`, `normal`, `suspicious`)
- `--plot-dist`: Output path for risk score histogram

**Output metrics:** Accuracy, precision, recall, F1-score, throughput (req/sec)

### Qualitative Testing (Decision Analysis)

Generates per-request inference decisions with feature attribution.

```bash
python ml_test_analysis.py \
  --csv-file test-logs/test-datasets/qualitative_test_suite.csv \
  --model-path models/rf_weighted_calibrated.joblib \
  --transformers-path features_production/transformers.joblib \
  --rules-dir modsec-rules/coreruleset-4.18.0 \
  --output-csv analysis_report.csv \
  --filtered
```

**Parameters:**
- `--filtered`: Excludes high-confidence predictions (>95%) to isolate ambiguous traffic
- `--rules-dir`: Path to OWASP CRS rules for metadata lookup

**Output:** CSV with per-request predictions, confidence scores, and top contributing features

## Precision-Recall Curves

Generate PR curves for model evaluation.

```bash
python plot_pr_curves.py \
  --features_dir features_production \
  --model models/rf_weighted_20250930_175219.joblib \
  --output pr_curves.png
```

## Utility Scripts

### Synthetic Log Generation

Generate synthetic ModSecurity logs for testing:

```bash
# Benign traffic
python synthetic_benign_logs.py --output synthetic_benign.log --count 10000

# Malicious traffic (uses SecLists payloads)
python synthetic_malicious_logs.py --output synthetic_malicious.log --count 5000

# Suspicious traffic (ambiguous patterns)
python synthetic_suspicious_logs.py --output synthetic_suspicious.log --count 1000
```

### Filter ModSecurity Logs

Filter parsed datasets by label:

```bash
python filter_modsecurity.py \
  --input datasets/modsec_dataset.csv \
  --filter malicious \
  --output malicious_only.csv
```

Options for `--filter`: `normal`, `suspicious`, `malicious`

### Gather Stratified Samples

Extract stratified random samples for testing:

```bash
python gather_samples.py \
  --input test-logs/test-datasets/modsec_natural_mixed.csv \
  --output sampled_1000.csv \
  --split 400 400 200 \  # Normal, Malicious, Suspicious counts
  --sanitize
```

**Parameters:**
- `--split`: Number of samples per class (normal, malicious, suspicious)
- `--sanitize`: Remove sensitive fields before export

### Remove Labels (Blind Testing)

Strip ground truth labels for blind evaluation:

```bash
python remove_labels.py \
  input_with_labels.csv \
  output_without_labels.csv
```

## Project Structure

```
modsec-ml-helper/
├── parser.py                      # Stage 1: Log parsing
├── feature_engineer.py            # Stage 2: Feature engineering
├── train_supervised_model.py      # Stage 3: Model training
├── calibrate_model.py             # Stage 4: Probability calibration
├── ml_test.py                     # Stage 5: Batch evaluation
├── ml_test_analysis.py            # Stage 5: Decision analysis
├── plot_pr_curves.py              # Precision-recall visualization
├── filter_modsecurity.py          # Filter logs by label
├── gather_samples.py              # Stratified sampling
├── remove_labels.py               # Strip labels for blind tests
├── synthetic_benign_logs.py       # Generate benign synthetic logs
├── synthetic_malicious_logs.py    # Generate attack synthetic logs
├── synthetic_suspicious_logs.py   # Generate ambiguous synthetic logs
└── requirements.txt               # Python dependencies
```

## Important Notes

**Data Leakage Prevention:** The feature engineering explicitly omits `anomaly_score` and other aggregated WAF scores that are direct proxies for labels. Only individual Rule IDs and Severities are included as "judgment features."

**WAF Dependency:** This system is tightly coupled to ModSecurity v2.9.3 with OWASP CRS v4.18.0. Log format changes or different WAF platforms require code modifications.

**Directory Structure:** The pipeline expects certain directories to exist:
- `modsec-logs/` - Raw audit logs
- `datasets/` - Parsed CSVs
- `features_*/` - Feature matrices
- `models/` - Trained models
- `logs/` - Training/evaluation logs

These directories are gitignored. Create them as needed:

```bash
mkdir -p modsec-logs datasets features_production models logs test-logs/test-datasets
```

## Disclaimer

This code is provided **as-is** from a research project. It was developed for a specific environment (ModSecurity v2.9.3 + OWASP CRS v4.18.0) and may require modifications to work with your setup. File paths, log formats, and dependencies are configured for the author's infrastructure. Users should expect to adapt the code to their own systems and data.

## License

MIT License
