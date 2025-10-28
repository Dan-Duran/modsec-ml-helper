
# Interim readme

## Pipeline commands

## PARSER:
python parser.py --input modsec-logs/modsec.log* --output datasets/modsec_dataset.csv

## FEATURE ENGINEERING

# Dry run - just shows stats, no processing
python feature_engineer.py \
    --input datasets/modsec_dataset.csv \
    --mode fit \
    --dry_run \
    --verbose

# Test Run - gives a quick run to make sure every works well
python feature_engineer.py \
    --input datasets/modsec_dataset.csv \
    --mode fit \
    --output_dir features_test \
    --limit 10000 \
    --train_frac 0.7 \
    --val_frac 0.15 \
    --verbose \
    --log_file logs/test_$(date +%Y%m%d_%H%M%S).log

# Production run (all samples)
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
    --log_file logs/feature_eng_production_$(date +%Y%m%d_%H%M%S).log

## TRAINING

# Dry run
python - <<'PY'
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, issparse, csr_matrix

features_dir = Path("features_production")
req = [
  'train_features.npz','val_features.npz','test_features.npz',
  'train_labels.csv','val_labels.csv','test_labels.csv','metadata.json'
]
miss = [f for f in req if not (features_dir/f).exists()]
if miss: raise SystemExit(f"Missing: {miss}")

def check(name):
    X = load_npz(features_dir / f'{name}_features.npz')
    if not issparse(X): raise SystemExit(f"{name}: not sparse")
    if not isinstance(X, csr_matrix): print(f"{name}: converting to CSR for training")
    print(f"{name}: shape={X.shape}, nnz={X.nnz:,}, density={X.nnz/(X.shape[0]*X.shape[1]):.4%}, dtype={X.dtype}")

for split in ["train","val","test"]:
    check(split)
    y = pd.read_csv(features_dir / f"{split}_labels.csv")['label']
    print(f"{split}: labels={y.value_counts().to_dict()}")

with open(features_dir / 'metadata.json','r') as f:
    meta = json.load(f)
print("features total:", meta.get("total_features"), "versions:", meta.get("version"))
PY

# Baseline - Baseline (fast-ish, RAM-friendly)

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

# Class imbalance emphasis (same capacity, weighted)

# Weighted run

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
  --log_file logs/training_weighted_$(date +%Y%m%d_%H%M%S).log



# Recall-push variant

python train_supervised_model.py \
  --features_dir features_production \
  --output models/rf_recallpush_$(date +%Y%m%d_%H%M%S).joblib \
  --n_estimators 400 \
  --max_depth 18 \
  --min_samples_split 12 \
  --min_samples_leaf 10 \
  --max_features log2 \
  --use_class_weights \
  --verbose \
  --log_file logs/training_recallpush_$(date +%Y%m%d_%H%M%S).log


# Paper run (more trees, still safe)

python train_supervised_model.py \
  --features_dir features_production \
  --output models/rf_paper_$(date +%Y%m%d_%H%M%S).joblib \
  --n_estimators 400 \
  --max_depth 25 \
  --min_samples_split 12 \
  --min_samples_leaf 6 \
  --max_features 0.2 \
  --use_class_weights \
  --verbose \
  --log_file logs/training_paper_$(date +%Y%m%d_%H%M%S).log

# paper run (if resolver present)
python train_supervised_model.py \
  --features_dir features_production \
  --output models/rf_paper_$(date +%Y%m%d_%H%M%S).joblib \
  --n_estimators 400 \
  --max_depth 22 \
  --min_samples_split 12 \
  --min_samples_leaf 6 \
  --max_features 0.2 \
  --use_class_weights \
  --verbose \
  --log_file logs/training_paper_$(date +%Y%m%d_%H%M%S).log

# If resolver is not in the script

# “sqrt” variant
--max_features sqrt
# or absolute count (~20% of 2443)
--max_features 489

# precision–recall curves

python plot_pr_curves.py \
  --features_dir features_production \
  --model models/rf_weighted_20250930_175219.joblib \
  --output pr_curves.png

###### CALIBRATE THE MODEL ######
python calibrate_model.py \
    --model_path models/rf_weighted_20251012_131423.joblib \
    --features_dir features_production \
    --output models/rf_weighted_calibrated.joblib \
    --plots_dir calibration_plots \
    --verbose

######## TESTING ###########

python ml_test.py --csv-file test-logs/test-datasets/modsec_natural_sqlmap.csv --model-path models/rf_weighted_calibrated.joblib --transformers-path features_production/transformers.joblib --expected mixed --plot-dist risk_distribution.png 

python ml_test_analysis.py --csv-file test-logs/test-datasets/modsec_natural_sqlmap.csv --model-path models/rf_weighted_calibrated.joblib --transformers-path features_production/transformers.joblib --rules-dir modsec-rules/coreruleset-4.18.0 --output-csv analysis_report.csv --filtered
