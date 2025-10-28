#!/usr/bin/env python3
"""
ml_test.py - ML WAF Testing Script for CSV Input

This script evaluates a trained ML model on pre-parsed CSV data by:
- Loading structured data from a CSV file
- Applying the SAME fitted transformers from feature_engineer.py
- Predicting with the trained RandomForest

Expected CSV columns (minimum required):
    - method, uri, uri_path, uri_query, payload, user_agent, referer
    - Optional: timestamp, source_port, dest_port, content_type, host
    - Optional: anomaly scores if you have them pre-computed

Usage:
    python ml_test.py \
        --csv-file parsed_logs.csv \
        --model-path models/rf_baseline_20250930_183356.joblib \
        --transformers-path features_production/transformers.joblib \
        --expected malicious
"""

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # <-- CHANGE 1: Added import
import warnings
from scipy.sparse import hstack, csr_matrix

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# ================== LOGGING ==================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ================== CONSTANTS ==================
ALLOWED_HTTP_METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS', 'PATCH', 'CONNECT', 'TRACE']
FIXED_TIMESTAMP = pd.Timestamp('1970-01-01', tz='UTC')
DTYPE = np.float32

# ================== FEATURE FUNCTIONS (INFERENCE) ==================
def _safe_str_ops(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].fillna('').astype(str)
        else:
            df[c] = ''
    return df

def _normalize_http_methods(df: pd.DataFrame) -> pd.DataFrame:
    if 'method' not in df.columns:
        df['method'] = 'GET'
    df['method'] = df['method'].astype(str).fillna('GET').str.upper()
    df.loc[~df['method'].isin(ALLOWED_HTTP_METHODS), 'method'] = 'OTHER'
    return df

def _validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all required columns exist with appropriate defaults."""
    required = ['uri', 'uri_path', 'uri_query', 'payload', 'user_agent', 'referer', 'method']
    for c in required:
        if c not in df.columns:
            df[c] = ''
    if 'source_port' not in df.columns: df['source_port'] = 0
    if 'dest_port' not in df.columns:   df['dest_port']   = 0
    if 'timestamp' not in df.columns:   df['timestamp']   = '1970-01-01 00:00:00 +0000'
    return df

def _validate_timestamps(df: pd.DataFrame) -> pd.Series:
    if 'timestamp' not in df.columns:
        return pd.Series([FIXED_TIMESTAMP] * len(df), index=df.index)
    parsed = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    parsed = parsed.fillna(FIXED_TIMESTAMP)
    return parsed

def _create_statistical_features(df: pd.DataFrame, dtype=DTYPE) -> pd.DataFrame:
    df = _safe_str_ops(df, ['uri', 'uri_path', 'uri_query', 'payload', 'user_agent', 'referer'])
    f = pd.DataFrame(index=df.index)
    # Lengths
    f['uri_length'] = df['uri'].str.len()
    f['uri_path_length'] = df['uri_path'].str.len()
    f['uri_query_length'] = df['uri_query'].str.len()
    f['payload_length'] = df['payload'].str.len()
    f['user_agent_length'] = df['user_agent'].str.len()
    f['referer_length'] = df['referer'].str.len()
    # Structure / counts
    f['uri_depth'] = df['uri'].str.count('/')
    f['path_depth'] = df['uri_path'].str.count('/')
    f['uri_num_params'] = df['uri_query'].str.count('&') + (df['uri_query'].str.len() > 0).astype(dtype)
    f['param_key_count'] = df['uri_query'].apply(lambda x: len(x.split('=')) - 1 if x else 0)
    def _avg_len(x):
        if not x: return 0.0
        parts = [p for p in x.split('&') if '=' in p]
        return float(np.mean([len(p.split('=')[-1]) for p in parts])) if parts else 0.0
    def _max_len(x):
        if not x: return 0
        parts = [p for p in x.split('&') if '=' in p]
        return max([len(p.split('=')[-1]) for p in parts]) if parts else 0
    f['avg_param_len'] = df['uri_query'].apply(_avg_len)
    f['max_param_len'] = df['uri_query'].apply(_max_len)
    # Ratios
    f['query_uri_ratio'] = f['uri_query_length'] / (f['uri_length'] + 1)
    f['payload_uri_ratio'] = f['payload_length'] / (f['uri_length'] + 1)
    # Char distributions
    f['uri_num_digits'] = df['uri'].str.count(r'\d')
    f['uri_num_special'] = df['uri'].str.count(r'[^a-zA-Z0-9/.\-_?&=]')
    f['uri_alpha_ratio'] = df['uri'].str.count(r'[a-zA-Z]') / (f['uri_length'] + 1)
    f['uri_digit_ratio'] = f['uri_num_digits'] / (f['uri_length'] + 1)
    f['payload_num_digits'] = df['payload'].str.count(r'\d')
    f['payload_num_special'] = df['payload'].str.count(r'[^a-zA-Z0-9]')
    f['payload_alpha_ratio'] = df['payload'].str.count(r'[a-zA-Z]') / (f['payload_length'] + 1)
    f['uri_percent_encoded_ratio'] = df['uri'].str.count('%') / (f['uri_length'] + 1)
    # Header-ish
    f['has_referer'] = (df['referer'].str.len() > 0).astype(dtype)
    f['source_port_numeric'] = pd.to_numeric(df['source_port'], errors='coerce').fillna(0).astype(dtype)
    f['dest_port_numeric'] = pd.to_numeric(df['dest_port'], errors='coerce').fillna(0).astype(dtype)
    f['is_standard_port'] = f['dest_port_numeric'].isin([80, 443, 8080, 8443]).astype(dtype)
    return f.replace([np.inf, -np.inf], 0).fillna(0).astype(dtype)

def _extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    t = _validate_timestamps(df)
    out = pd.DataFrame(index=df.index)
    out['hour'] = t.dt.hour.fillna(0).astype(np.int8)
    out['day_of_week'] = t.dt.dayofweek.fillna(0).astype(np.int8)
    out['is_weekend'] = (out['day_of_week'] >= 5).astype(np.int8)
    out['is_night'] = ((out['hour'] < 6) | (out['hour'] > 22)).astype(np.int8)
    return out

def _extract_anomaly_features(df: pd.DataFrame, dtype=DTYPE) -> pd.DataFrame:
    """Extract pre-computed anomaly scores if they exist in the CSV."""
    anomaly_cols = [
        'anomaly_score', 'sql_injection_score', 'xss_score', 'rce_score', 
        'lfi_rfi_score', 'session_fixation_score', 'request_anomalies'
    ]
    f = pd.DataFrame(index=df.index)
    for col in anomaly_cols:
        if col in df.columns:
            f[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(dtype)
        else:
            f[col] = 0.0  # Default to 0 if not present
    return f

def transform_with_fitted_transformers(df_raw: pd.DataFrame, transformers) -> csr_matrix:
    """
    Apply the exact fitted transformers (from feature_engineer.py) to raw log rows.
    Return a CSR float32 matrix with identical column layout to training.
    """
    # 1) schema, fill, method normalization
    df = _validate_schema(df_raw.copy())
    df = _normalize_http_methods(df)
    
    # Fill NA for text and rule columns before processing
    df.fillna({
        'payload': '', 'uri': '', 'uri_path': '', 'uri_query': '',
        'user_agent': '', 'method': 'GET', 'referer': '',
        'triggered_rule_ids': '[]', 'rule_severities': '[]'
    }, inplace=True)

    # 2) numerical blocks
    stat = _create_statistical_features(df, dtype=DTYPE)
    timef = _extract_time_features(df)
    anomaly = _extract_anomaly_features(df, dtype=DTYPE)
    numeric = pd.concat([stat, timef, anomaly], axis=1)

    # 3) column alignment + scaling
    num_cols = transformers['numerical_columns']
    numeric = numeric.reindex(columns=num_cols, fill_value=0)
    numeric_sparse = csr_matrix(numeric.values.astype(DTYPE))
    numeric_scaled = transformers['scaler'].transform(numeric_sparse).astype(DTYPE)

    # 4) text TF-IDF, rule vectorization, and method OHE
    # --- FIX: Added rule_ids and severities to the columns being prepared ---
    df = _safe_str_ops(df, [
        'uri', 'uri_path', 'uri_query', 'payload', 'user_agent', 
        'triggered_rule_ids', 'rule_severities'
    ])
    
    uri = transformers['uri_vectorizer'].transform(df['uri'])
    uri_path = transformers['uri_path_vectorizer'].transform(df['uri_path'])
    uri_query = transformers['uri_query_vectorizer'].transform(df['uri_query'])
    payload = transformers['payload_vectorizer'].transform(df['payload'])
    ua = transformers['ua_vectorizer'].transform(df['user_agent'])
    method = transformers['method_encoder'].transform(df[['method']])

    # --- FIX: Vectorize the rule and severity features using the fitted transformers ---
    # These vectorizers must exist in your transformers.joblib file.
    rule_ids = transformers['rule_ids_vectorizer'].transform(df['triggered_rule_ids'])
    severities = transformers['severities_vectorizer'].transform(df['rule_severities'])

    # 5) assemble → CSR
    # --- FIX: Add the new features to the final horizontal stack ---
    X = hstack([
        numeric_scaled, uri, uri_path, uri_query, payload, ua, method,
        rule_ids, severities
    ]).tocsr()
    
    return X


# ================== ARTIFACT LOADING ==================
def load_artifacts(artifact_path: Path) -> Dict[str, Any]:
    logging.info(f"Loading artifacts from '{artifact_path}'...")
    if not artifact_path.exists():
        logging.error(f"Artifact file not found: {artifact_path}")
        sys.exit("Please provide the correct path to a trained .joblib file.")
    artifacts = joblib.load(artifact_path)
    logging.info("Artifacts loaded successfully.")
    return artifacts

# ================== MAIN ==================
def main():
    parser = argparse.ArgumentParser(
        description="Test a trained ML WAF model on a pre-parsed CSV file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--csv-file', required=True, type=Path,
                        help='Path to the pre-parsed CSV log file to test.')
    parser.add_argument('--model-path', required=True, type=Path,
                        help='Path to the trained model artifacts (.joblib) saved by train_supervised_model.py.')
    parser.add_argument('--transformers-path', required=True, type=Path,
                        help='Path to transformers.joblib produced by feature_engineer.py (SAME RUN as training).')
    parser.add_argument('--expected', choices=['normal', 'malicious', 'suspicious', 'mixed'],
                        help='Expected ground truth of the log file for validation.')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit processing to the first N rows for quick testing.')
    parser.add_argument('--top-n-errors', type=int, default=50,
                        help='Number of misclassification examples to display.')
    parser.add_argument('--mal-prob-threshold', type=float, default=None,
                        help='If set, promote any row with P(malicious)+P(suspicious) >= THRESH to malicious.')
    
    # --- CHANGE 2: Added new argument ---
    parser.add_argument('--plot-dist', type=Path, default=None,
                        help='If set, save a risk score distribution plot to this file path (e.g., plot.png).')
    
    args = parser.parse_args()

    if not args.csv_file.exists():
        logging.error(f"CSV file not found: {args.csv_file}")
        sys.exit(1)
    if not args.transformers_path.exists():
        logging.error(f"Transformers file not found: {args.transformers_path}")
        sys.exit(1)

    logging.info("=" * 50)
    logging.info("ML WAF MODEL TESTING (CSV INPUT)")
    logging.info("=" * 50)
    logging.info(f"CSV file: {args.csv_file}")
    logging.info(f"Model artifacts file: {args.model_path}")
    logging.info(f"Transformers: {args.transformers_path}")
    if args.expected:
        logging.info(f"Expected traffic type: {args.expected.upper()}")

    try:
        # Load model package and transformers
        package = load_artifacts(args.model_path)
        if 'model' not in package:
            logging.critical("Artifact missing key 'model'. Available keys: %s", list(package.keys()))
            sys.exit(1)
        model = package['model']  # RandomForestClassifier
        transformers = joblib.load(args.transformers_path)

        # Load CSV data
        logging.info("Loading CSV file...")
        t0 = time.time()
        df = pd.read_csv(args.csv_file, low_memory=False)
        
        if args.limit:
            df = df.head(args.limit)
            logging.warning(f"Processing limited to the first {args.limit} rows.")
        
        total = len(df)
        logging.info(f"Loaded {total:,} rows in {time.time() - t0:.2f}s.")
        
        if total == 0:
            logging.error("No entries found in the CSV file.")
            sys.exit(1)

        # Validate required columns
        required_cols = ['uri', 'uri_path', 'uri_query', 'payload', 'user_agent', 'method']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            logging.warning(f"Missing columns will be filled with defaults: {missing_cols}")

        # Keep reference to URIs for error reporting
        if 'uri' in df.columns:
            raw_lines = df['uri'].fillna('').astype(str).str.slice(0, 120).tolist()
        else:
            raw_lines = [f"Row {i}" for i in range(total)]

        # Transform + predict
        logging.info("Transforming entries and predicting...")
        t1 = time.time()
        X = transform_with_fitted_transformers(df, transformers)

        # --- CHANGE 3: Modified prediction & risk score block ---
        
        # Base predictions
        base_preds = model.predict(X)
        probs = model.predict_proba(X)
        classes = list(model.classes_)

        # --- Calculate Risk Score (P(malicious) + P(suspicious)) ---
        # This is now calculated *always* for plotting and/or thresholding.
        idx_mal = classes.index("malicious") if "malicious" in classes else -1
        idx_susp = classes.index("suspicious") if "suspicious" in classes else -1

        if idx_mal == -1:
            logging.warning("Model class 'malicious' not found. Risk scores will be 0.")
            risk_scores = np.zeros(total)
        else:
            risk_scores = probs[:, idx_mal]
            if idx_susp != -1:
                risk_scores += probs[:, idx_susp] # Add suspicious prob if it exists

        # --- Optional thresholding (uses the risk_scores above) ---
        final_preds = base_preds
        if args.mal_prob_threshold is not None:
            if idx_mal == -1:
                logging.warning("Thresholding requested, but class 'malicious' is not present. Skipping.")
            else:
                promote_mask = risk_scores >= float(args.mal_prob_threshold)
                promoted = int(np.sum(promote_mask & (base_preds != "malicious")))
                final_preds = np.where(promote_mask, "malicious", base_preds)
                logging.info(
                    "Applied thresholding at %.2f → promoted %d/%d (%.2f%%) predictions to 'malicious'.",
                    float(args.mal_prob_threshold), promoted, total, (promoted / total * 100.0)
                )

        elapsed = time.time() - t1
        counts = Counter(final_preds)
        rps = total / elapsed if elapsed > 0 else float('inf')
        logging.info(f"Predicted {total:,} entries in {elapsed:.2f}s ({rps:.0f} req/sec).")

        # --- CHANGE 4: Added new plotting block ---
        
        # --- Optional: Generate and save distribution plot ---
        if args.plot_dist:
            try:
                logging.info(f"Generating risk score distribution plot to {args.plot_dist}...")

                # 1. Calculate Stats and Counts
                mean_score = float(np.mean(risk_scores))
                median_score = float(np.median(risk_scores))
                std_dev = float(np.std(risk_scores))
                
                c_norm = counts.get('normal', 0)
                c_susp = counts.get('suspicious', 0)
                c_mal = counts.get('malicious', 0)

                p_norm = (c_norm / total) * 100.0
                p_susp = (c_susp / total) * 100.0
                p_mal = (c_mal / total) * 100.0

                # 2. Create Figure and Subplots
                # We create a 2-row layout: 1 for plot, 1 for table.
                # 'gridspec_kw' controls the relative height: 3 parts for the plot, 1 part for the table.
                # Figure is taller (10in) to accommodate the table.
                fig, (ax_plot, ax_table) = plt.subplots(
                    nrows=2, 
                    ncols=1, 
                    figsize=(12, 10), 
                    gridspec_kw={'height_ratios': [3, 1]}
                )

                # 3. --- PLOT COMMANDS (on the top axes: 'ax_plot') ---
                ax_plot.hist(risk_scores, bins=100, density=True, alpha=0.75, label='Score Distribution', color='royalblue')
                
                # Add stat lines
                ax_plot.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.4f}')
                ax_plot.axvline(median_score, color='orange', linestyle=':', linewidth=2, label=f'Median: {median_score:.4f}')
                
                # Add labels and title
                ax_plot.set_title(f'Risk Score Distribution (N={total:,})', fontsize=16)
                # We can remove the x-label from the plot, as the table below makes it clear
                # ax_plot.set_xlabel('Predicted Malicious-like Probability', fontsize=12)
                ax_plot.set_ylabel('Density', fontsize=12)
                ax_plot.set_yscale('symlog', linthresh=0.01)
                ax_plot.legend(loc='upper left')
                ax_plot.grid(True, which='both', linestyle=':', linewidth=0.5)

                # 4. --- TABLE COMMANDS (on the bottom axes: 'ax_table') ---
                
                # Define table data
                table_rows = [
                    ["Mean Score", f"{mean_score:.4f}"],
                    ["Median Score", f"{median_score:.4f}"],
                    ["Std Dev", f"{std_dev:.4f}"],
                    ["-", "-"], # Separator
                    ["Normal", f"{c_norm:,} ({p_norm:.2f}%)"],
                    ["Suspicious", f"{c_susp:,} ({p_susp:.2f}%)"],
                    ["Malicious", f"{c_mal:,} ({p_mal:.2f}%)"],
                    ["Total", f"{total:,} (100.00%)"]
                ]
                
                # Add the table to the *bottom* axes
                the_table = ax_table.table(cellText=table_rows,
                                          colLabels=["Metric", "Value"],
                                          loc='center', # Center the table in the axes
                                          colWidths=[0.35, 0.65],
                                          cellLoc='left',
                                          colLoc='left')
                
                the_table.auto_set_font_size(False)
                the_table.set_fontsize(10)
                the_table.scale(1.2, 1.2) # Scale table 
                
                # Turn off the axes (border, ticks) for the table subplot
                ax_table.axis('off')

                # 5. Save and close
                plt.tight_layout() # Adjusts plot to prevent labels from overlapping
                plt.savefig(args.plot_dist, dpi=150)
                plt.close()
                logging.info(f"Plot saved successfully to {args.plot_dist}")
                
            except Exception as e:
                # Don't crash the whole script if plotting fails
                logging.error(f"Failed to generate plot: {e}", exc_info=False)
        
        # Summaries
        logging.info("-" * 50)
        logging.info("RESULTS")
        logging.info("-" * 50)
        # The 'counts = Counter(final_preds)' line is removed from here
        logging.info(f"Prediction Breakdown (Total: {total:,}):")
        for k in sorted(counts.keys()):
            c = counts[k]
            pct = c / total * 100
            logging.info(f"  - {k.capitalize():<12}: {c:>8,} ({pct:>6.2f}%)")

        # Optional quick validation for expected distribution
        if args.expected:
            fp = fn = 0
            if args.expected == 'normal':
                tn = counts.get('normal', 0)
                fp = total - tn
                fpr = fp / total * 100
                logging.info(f"\nValidation (Expected: NORMAL):")
                logging.info(f"  - False Positive Rate: {fpr:.2f}% ({fp}/{total})")
            elif args.expected == 'malicious':
                tp = counts.get('malicious', 0) + counts.get('suspicious', 0)
                fn = counts.get('normal', 0)
                dr = tp / total * 100
                logging.info(f"\nValidation (Expected: MALICIOUS):")
                logging.info(f"  - Detection Rate: {dr:.2f}% ({tp}/{total})")

            if (fp > 0 or fn > 0) and args.top_n_errors > 0:
                logging.info(f"\nTop {args.top_n_errors} Misclassification Examples:")
                shown = 0
                class_names = list(model.classes_)
                for i, (pred, prob, raw) in enumerate(zip(final_preds, probs, raw_lines)):
                    wrong = (args.expected == 'normal' and pred != 'normal') or \
                            (args.expected == 'malicious' and pred == 'normal')
                    if wrong and shown < args.top_n_errors:
                        pred_idx = int(np.argmax(prob))
                        pred_name = class_names[pred_idx]
                        logging.warning(
                            "  - Entry #%d | Predicted: %s (%.1f%%) | URI: %s",
                            i + 1, pred_name, float(np.max(prob)) * 100.0, raw
                        )
                        shown += 1

        logging.info("=" * 50)

    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit("Script terminated due to a critical error.")

if __name__ == "__main__":
    main()