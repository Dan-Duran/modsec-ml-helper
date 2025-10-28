#!/usr/bin/env python3
"""
ml_test_analysis.py — Unified Analyst Script (filtered/unfiltered + CRS messages)

Unfiltered (default): prints every entry. Shows "why" only for split decisions (<95%).
Filtered (--filtered): prints ONLY split decisions (<95%) to console, but still writes ALL to CSV.

Also supports --rules-dir to map Rule_IDs -> short CRS messages
(e.g., 980170 → "Anomaly Scores") while keeping the ruleset untouched.

Usage (unfiltered):
    python ml_test_analysis.py \
        --csv-file test-logs/test-datasets/test.csv \
        --model-path models/rf_weighted_calibrated.joblib \
        --transformers-path features_production/transformers.joblib \
        --rules-dir modsec-rules/coreruleset-4.18.0 \
        --output-csv analysis_report.csv

Usage (filtered console output):
    python ml_test_csv_analyst.py \
        --csv-file test-logs/test-datasets/test.csv \
        --model-path models/rf_weighted_calibrated.joblib \
        --transformers-path features_production/transformers.joblib \
        --rules-dir modsec-rules/coreruleset-4.18.0 \
        --output-csv analysis_report.csv \
        --filtered
"""

import argparse
import logging
import sys
import time
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Tuple

import joblib
import numpy as np
import pandas as pd
import warnings
from scipy.sparse import hstack, csr_matrix

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

# ================== LOGGING ==================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [ANALYST] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger('ANALYST')

# ================== CONSTANTS ==================
ALLOWED_HTTP_METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS', 'PATCH', 'CONNECT', 'TRACE']
FIXED_TIMESTAMP = pd.Timestamp('1970-01-01', tz='UTC')
DTYPE = np.float32
CONFIDENCE_THRESHOLD = 0.95
TRUNCATE_LENGTH = 700
PAYLOAD_TRUNCATE_LENGTH = 300

# Colors
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_ORANGE = "\033[38;5;208m"
COLOR_RESET = "\033[0m"

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
    required = ['uri', 'uri_path', 'uri_query', 'payload', 'user_agent', 'referer', 'method']
    for c in required:
        if c not in df.columns:
            df[c] = ''
    optional = ['source_port', 'dest_port', 'timestamp', 'triggered_rule_ids', 'rule_severities', 'content_type', 'host']
    defaults = {'source_port': 0, 'dest_port': 0, 'timestamp': '1970-01-01 00:00:00 +0000', 'triggered_rule_ids': '[]', 'rule_severities': '[]', 'content_type': '', 'host': ''}
    for c in optional:
        if c not in df.columns:
            df[c] = defaults[c]
    return df

def _validate_timestamps(df: pd.DataFrame) -> pd.Series:
    if 'timestamp' not in df.columns:
        return pd.Series([FIXED_TIMESTAMP] * len(df), index=df.index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    return parsed.fillna(FIXED_TIMESTAMP)

def _create_statistical_features(df: pd.DataFrame, dtype=DTYPE) -> pd.DataFrame:
    df = _safe_str_ops(df, ['uri', 'uri_path', 'uri_query', 'payload', 'user_agent', 'referer'])
    f = pd.DataFrame(index=df.index)
    f['uri_length'] = df['uri'].str.len()
    f['uri_path_length'] = df['uri_path'].str.len()
    f['uri_query_length'] = df['uri_query'].str.len()
    f['payload_length'] = df['payload'].str.len()
    f['user_agent_length'] = df['user_agent'].str.len()
    f['referer_length'] = df['referer'].str.len()
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
    f['query_uri_ratio'] = f['uri_query_length'] / (f['uri_length'].replace(0, 1))
    f['payload_uri_ratio'] = f['payload_length'] / (f['uri_length'].replace(0, 1))
    f['uri_num_digits'] = df['uri'].str.count(r'\d')
    f['uri_num_special'] = df['uri'].str.count(r'[^a-zA-Z0-9/.\-_?&=]')
    f['uri_alpha_ratio'] = df['uri'].str.count(r'[a-zA-Z]') / (f['uri_length'].replace(0, 1))
    f['uri_digit_ratio'] = f['uri_num_digits'] / (f['uri_length'].replace(0, 1))
    f['payload_num_digits'] = df['payload'].str.count(r'\d')
    f['payload_num_special'] = df['payload'].str.count(r'[^a-zA-Z0-9]')
    f['payload_alpha_ratio'] = df['payload'].str.count(r'[a-zA-Z]') / (f['payload_length'].replace(0, 1))
    f['uri_percent_encoded_ratio'] = df['uri'].str.count('%') / (f['uri_length'].replace(0, 1))
    f['has_referer'] = (df['referer'].str.len() > 0).astype(dtype)
    f['source_port_numeric'] = pd.to_numeric(df.get('source_port', 0), errors='coerce').fillna(0).astype(dtype)
    f['dest_port_numeric'] = pd.to_numeric(df.get('dest_port', 0), errors='coerce').fillna(0).astype(dtype)
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
    anomaly_cols = ['anomaly_score', 'sql_injection_score', 'xss_score', 'rce_score', 'lfi_rfi_score', 'session_fixation_score', 'request_anomalies']
    f = pd.DataFrame(index=df.index)
    for col in anomaly_cols:
        if col in df.columns:
            f[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(dtype)
        else:
            f[col] = 0.0
    return f

def transform_with_fitted_transformers(df_raw: pd.DataFrame, transformers) -> csr_matrix:
    df = _validate_schema(df_raw.copy())
    df = _normalize_http_methods(df)
    fill_dict = {'payload': '', 'uri': '', 'uri_path': '', 'uri_query': '', 'user_agent': '', 'method': 'GET', 'referer': '', 'triggered_rule_ids': '[]', 'rule_severities': '[]', 'content_type': '', 'host': ''}
    df.fillna({k: v for k, v in fill_dict.items() if k in df.columns}, inplace=True)

    stat = _create_statistical_features(df, dtype=DTYPE)
    timef = _extract_time_features(df)
    anomaly = _extract_anomaly_features(df, dtype=DTYPE)

    numeric = pd.concat([stat, timef, anomaly], axis=1)
    num_cols = transformers['numerical_columns']
    numeric = numeric.reindex(columns=num_cols, fill_value=0)

    numeric_sparse = csr_matrix(numeric.values.astype(DTYPE))
    numeric_scaled = transformers['scaler'].transform(numeric_sparse).astype(DTYPE)

    df = _safe_str_ops(df, ['uri', 'uri_path', 'uri_query', 'payload', 'user_agent', 'triggered_rule_ids', 'rule_severities'])
    uri = transformers['uri_vectorizer'].transform(df['uri'])
    uri_path = transformers['uri_path_vectorizer'].transform(df['uri_path'])
    uri_query = transformers['uri_query_vectorizer'].transform(df['uri_query'])
    payload = transformers['payload_vectorizer'].transform(df['payload'])
    ua = transformers['ua_vectorizer'].transform(df['user_agent'])
    method = transformers['method_encoder'].transform(df[['method']])
    rule_ids = transformers['rule_ids_vectorizer'].transform(df['triggered_rule_ids'])
    severities = transformers['severities_vectorizer'].transform(df['rule_severities'])

    X = hstack([numeric_scaled, uri, uri_path, uri_query, payload, ua, method, rule_ids, severities]).tocsr()
    return X

# ================== ARTIFACT LOADING ==================
def load_artifacts(artifact_path: Path) -> Dict[str, Any]:
    log.info(f"Loading artifacts from '{artifact_path}'...")
    if not artifact_path.exists():
        log.error(f"Artifact file not found: {artifact_path}")
        sys.exit("Please provide the correct path to a trained .joblib file.")
    artifacts = joblib.load(artifact_path)
    log.info("Artifacts loaded successfully.")
    return artifacts

# ================== RULE PARSING (CRS) ==================
def _squash_line_continuations(text: str) -> str:
    text = text.replace("\r\n", "\n")
    return re.sub(r"(?:\\[ \t]*\n)+", " ", text)

def _short_msg_label(msg: str) -> str:
    s = re.sub(r"\s+", " ", msg).strip().rstrip(",")
    s = re.split(r"[:\-\(]", s, maxsplit=1)[0].strip()
    return s or msg.strip()

def _load_rule_descriptions(rules_dir: Path) -> Dict[str, str]:
    rule_map: Dict[str, str] = {}
    if not rules_dir or not rules_dir.is_dir():
        log.warning(f"Rules directory not found or invalid: {rules_dir}. Cannot load descriptions.")
        return rule_map

    log.info(f"Loading rule descriptions from {rules_dir} (recursing **/*.conf)...")

    block_pat = re.compile(r'^\s*(?:SecRule|SecAction)\b.*?(?=^\s*(?:SecRule|SecAction)\b|\Z)', re.MULTILINE | re.DOTALL)
    id_pat   = re.compile(r'\bid\s*:\s*(\d+)\b', re.IGNORECASE)
    msg_pat  = re.compile(r'\bmsg\s*:\s*(["\'])(?P<msg>(?:\\.|[^\'"])*?)\1', re.IGNORECASE | re.DOTALL)

    files_parsed = 0
    for conf in rules_dir.rglob("*.conf"):
        try:
            content = conf.read_text(encoding="utf-8", errors="ignore")
            content = _squash_line_continuations(content)
            content = re.sub(r'(?m)^\s*#.*$', '', content)

            for block in block_pat.findall(content):
                rid_m = id_pat.search(block)
                msg_m = msg_pat.search(block)
                if not (rid_m and msg_m):
                    continue
                rid = rid_m.group(1)
                raw = msg_m.group('msg').replace(r"\'", "'").replace(r'\"', '"')
                msg = re.sub(r"\s+", " ", raw).strip().rstrip(",")
                rule_map[rid] = msg
            files_parsed += 1
        except Exception as e:
            log.warning(f"Could not parse {conf}: {e}")

    if files_parsed == 0:
        log.error(f"No .conf files found or parsed in rules directory: {rules_dir}")
    elif not rule_map:
        log.warning(f"Parsed {files_parsed} files but found 0 rule descriptions.")
    else:
        log.info(f"Loaded {len(rule_map)} rule descriptions from {files_parsed} files.")
    return rule_map

def _parse_rule_ids(rules_str: str) -> List[str]:
    if not rules_str:
        return []
    ids = re.findall(r'\d{3,}', rules_str)
    seen, out = set(), []
    for rid in ids:
        if rid not in seen:
            seen.add(rid)
            out.append(rid)
    return out

# ================== "WHY" ANALYSIS FUNCTIONS ==================
def _build_feature_map(transformers: Dict) -> Tuple[Dict, List[str]]:
    log.debug("Building feature index map...")
    feature_indices = {}
    numerical_feature_names: List[str] = []
    current_idx = 0

    num_cols = transformers.get('numerical_columns', [])
    numerical_feature_names.extend(num_cols)
    feature_indices['Numerical'] = slice(current_idx, current_idx + len(num_cols))
    current_idx += len(num_cols)

    vec_map = {
        'URI': 'uri_vectorizer', 'URI_Path': 'uri_path_vectorizer',
        'URI_Query': 'uri_query_vectorizer', 'Payload': 'payload_vectorizer',
        'User_Agent': 'ua_vectorizer', 'Method': 'method_encoder',
        'Rule_IDs': 'rule_ids_vectorizer', 'Severities': 'severities_vectorizer'
    }

    for group_name, vec_key in vec_map.items():
        vectorizer = transformers.get(vec_key)
        if vectorizer:
            try:
                if hasattr(vectorizer, 'get_feature_names_out'):
                    num_features = len(vectorizer.get_feature_names_out())
                elif hasattr(vectorizer, 'categories_'):
                    num_features = sum(len(c) for c in vectorizer.categories_)
                else:
                    num_features = vectorizer.transform(pd.DataFrame([''])).shape[1]
                feature_indices[group_name] = slice(current_idx, current_idx + num_features)
                current_idx += num_features
            except Exception as e:
                log.warning(f"Could not get feature count for {vec_key}: {e}.")
    log.debug("Feature index map built.")
    return feature_indices, numerical_feature_names

def _run_local_contribution(X_row: csr_matrix, model_original: Any, baseline_prob_original: float, malicious_idx_original: int, feature_indices: Dict) -> Dict:
    contributions = {}
    if X_row.shape[1] == 0:
        return contributions
    X_dense = X_row.toarray().copy()[0]
    for group_name, idx_slice in feature_indices.items():
        if idx_slice.start >= X_dense.shape[0]:
            continue
        X_muted_dense = X_dense.copy()
        safe_stop = min(idx_slice.stop, X_dense.shape[0])
        safe_slice = slice(idx_slice.start, safe_stop)
        if safe_slice.start < safe_slice.stop:
            X_muted_dense[safe_slice] = 0.0
        X_muted = X_muted_dense.reshape(1, -1)
        try:
            prob_muted = model_original.predict_proba(X_muted)[0]
            if malicious_idx_original < len(prob_muted):
                prob_malicious_muted = prob_muted[malicious_idx_original]
                contribution = baseline_prob_original - prob_malicious_muted
                if abs(contribution) > 0.01:
                    contributions[group_name] = contribution
        except Exception as e:
            log.error(f"Error predicting with muted features for group {group_name}: {e}")
    return contributions

# ================== MAIN ==================
def main():
    parser = argparse.ArgumentParser(
        description="Unified ML WAF Analyst (filtered/unfiltered + CRS rule messages)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--csv-file', required=True, type=Path, help='Path to the small, pre-parsed CSV log file to analyze.')
    parser.add_argument('--model-path', required=True, type=Path, help='Path to the *calibrated* model artifacts (.joblib).')
    parser.add_argument('--transformers-path', required=True, type=Path, help='Path to transformers.joblib produced by feature_engineer.py.')
    parser.add_argument('--rules-dir', type=Path, default=None, help='Optional: ModSecurity CRS directory to map rule ids to messages.')
    parser.add_argument('--output-csv', type=Path, default=None, help='Optional: Path to save a detailed CSV report with probabilities.')
    parser.add_argument('--filtered', action='store_true', help='If set, only print split decisions (<95%%) to console (CSV still contains ALL).')
    args = parser.parse_args()

    if not args.csv_file.exists():
        log.error(f"CSV file not found: {args.csv_file}"); sys.exit(1)
    if not args.transformers_path.exists():
        log.error(f"Transformers file not found: {args.transformers_path}"); sys.exit(1)
    if args.rules_dir and not args.rules_dir.is_dir():
        log.warning(f"Rules directory invalid: {args.rules_dir}. Descriptions disabled.")
        args.rules_dir = None

    log.info("=" * 70)
    log.info("ML WAF ANALYST — Unified")
    log.info("=" * 70)
    log.info(f"CSV file: {args.csv_file}")
    log.info(f"Model artifacts file: {args.model_path}")
    log.info(f"Transformers: {args.transformers_path}")
    log.info(f"Rules Directory: {args.rules_dir if args.rules_dir else 'Not Provided'}")
    log.info(f"Console Mode: {'FILTERED (<95% only)' if args.filtered else 'UNFILTERED (all entries)'}")

    try:
        package = load_artifacts(args.model_path)
        if 'model' not in package or 'original_model' not in package:
            log.critical("Artifact invalid. Missing 'model' or 'original_model'."); sys.exit(1)

        model_calibrated = package['model']
        model_original = package['original_model']
        model_original.verbose = 0
        if hasattr(model_calibrated, 'base_estimator') and hasattr(model_calibrated.base_estimator, 'verbose'):
            model_calibrated.base_estimator.verbose = 0

        transformers = joblib.load(args.transformers_path)
        classes_calibrated = list(model_calibrated.classes_)
        classes_original = list(model_original.classes_)
        malicious_idx_original = classes_original.index('malicious') if 'malicious' in classes_original else -1

        feature_indices, numerical_feature_names = _build_feature_map(transformers)

        rule_id_to_msg_map: Dict[str, str] = {}
        if args.rules_dir:
            rule_id_to_msg_map = _load_rule_descriptions(args.rules_dir)

        log.info("Loading CSV file...")
        t0 = time.time()
        df = pd.read_csv(args.csv_file, low_memory=False)
        total = len(df)
        log.info(f"Loaded {total:,} rows in {time.time() - t0:.2f}s.")
        if total == 0:
            log.error("No entries found."); sys.exit(1)

        df = _validate_schema(df.copy())
        log.info("Transforming entries...")
        t1 = time.time()
        X_all = transform_with_fitted_transformers(df, transformers)
        log.info(f"Transformed {total:,} entries in {time.time() - t1:.2f}s.")

        log.info("Predicting probabilities...")
        all_preds_calibrated = model_calibrated.predict(X_all)
        all_probs_calibrated = model_calibrated.predict_proba(X_all)
        all_probs_original = model_original.predict_proba(X_all)

        log.info("\n" + "=" * 70)
        log.info("DETAILED ANALYSIS RESULTS")
        log.info("=" * 70)

        results_for_csv = []

        for i in range(total):
            pred = all_preds_calibrated[i]
            prob_vector_calibrated = all_probs_calibrated[i]
            max_prob = float(np.max(prob_vector_calibrated))
            X_row = X_all.getrow(i)

# --- Extract More Fields ---
            row_data = df.iloc[i]
            method_str = str(row_data.get('method', 'N/A'))
            uri_str = str(row_data.get('uri', 'N/A')) # Base URI (path only, from parser)
            query_str = str(row_data.get('uri_query', ''))
            # Construct the *actual* full request path + query
            full_request_path = uri_str + ('?' + query_str if query_str else '')

            # Keep uri_path_str separate if needed later for contributions
            uri_path_str = str(row_data.get('uri_path', 'N/A'))

            ua_str = str(row_data.get('user_agent', 'N/A'))
            # Handle potential NaN from pandas for referer before converting to string
            referer_val = row_data.get('referer')
            referer_str = str(referer_val) if pd.notna(referer_val) else 'N/A'

            # Handle potential NaN from pandas for payload before converting to string
            payload_val = row_data.get('payload')
            payload_str = str(payload_val) if pd.notna(payload_val) else 'N/A'

            rules_str = str(row_data.get('triggered_rule_ids', '[]'))
            severity_str = str(row_data.get('rule_severities', '[]'))
            # --- End Extract ---

            prob_dict_float = {c: float(p) for c, p in zip(classes_calibrated, prob_vector_calibrated)}
            prob_dict_str = {c: f"{p:.4f}" for c, p in prob_dict_float.items()}
            prob_log_str = ", ".join([f"P({c})={v}" for c, v in sorted(prob_dict_str.items(), key=lambda x: x[1], reverse=True)])

            action = {
                'malicious': "ACTION: Triage & Investigate. Confirm and block source IP.",
                'suspicious': "ACTION: Monitor. Low-priority investigation. Correlate.",
                'normal': "ACTION: No action required. Observe."
            }.get(pred, "ACTION: N/A")

            verdict_color = ""
            if pred == 'malicious':
                verdict_color = COLOR_RED if max_prob >= CONFIDENCE_THRESHOLD else COLOR_ORANGE
            elif pred == 'suspicious':
                verdict_color = COLOR_ORANGE
            elif pred == 'normal' and max_prob < CONFIDENCE_THRESHOLD:
                verdict_color = COLOR_YELLOW

            # Console filtering behavior
            should_print = True
            if args.filtered and max_prob >= CONFIDENCE_THRESHOLD:
                should_print = False

            insights_log = []
            console_lines: List[str] = []

            if should_print:
                console_lines.append("\n" + "-" * 70)
                # --- NEW Enhanced Output ---
                # ** Display Method + Full Path + Query **
                console_lines.append(f"ENTRY #{i+1} | {method_str} {full_request_path[:TRUNCATE_LENGTH]}")

                # Display Referer only if it exists and isn't empty/NaN
                if referer_str and referer_str != 'N/A' and referer_str.strip():
                    console_lines.append(f"  Referer:     {referer_str[:TRUNCATE_LENGTH]}")
                # Display User-Agent only if it exists and isn't empty/NaN
                if ua_str and ua_str != 'N/A' and ua_str.strip():
                    console_lines.append(f"  User-Agent:  {ua_str[:TRUNCATE_LENGTH]}")
                # Display Payload only if it exists and isn't empty/NaN
                if payload_str and payload_str != 'N/A' and payload_str.strip():
                    # Apply specific payload truncation
                    payload_display = payload_str[:PAYLOAD_TRUNCATE_LENGTH] + ('...' if len(payload_str) > PAYLOAD_TRUNCATE_LENGTH else '')
                    console_lines.append(f"  Payload:     {payload_display}")
                # --- End Enhanced Output ---
                console_lines.append(f"  VERDICT:     {verdict_color}{pred.upper()}{COLOR_RESET}")
                console_lines.append(f"  PROBABILITY: {prob_log_str}")
                console_lines.append(f"  {action}")

            if max_prob < CONFIDENCE_THRESHOLD:
                if should_print:
                    console_lines.append(f"  --- Split Decision ({max_prob*100:.1f}% confidence): ---")

                if malicious_idx_original != -1:
                    baseline_prob_original = float(all_probs_original[i][malicious_idx_original])
                    contributions = _run_local_contribution(X_row, model_original, baseline_prob_original, malicious_idx_original, feature_indices)

                    if contributions:
                        for group, impact in sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True):
                            data_to_show = ""
                            if group == 'URI': data_to_show = uri_str
                            elif group == 'URI_Path': data_to_show = uri_path_str
                            elif group == 'User_Agent': data_to_show = ua_str
                            elif group == 'Payload': data_to_show = payload_str
                            elif group == 'Severities': data_to_show = severity_str
                            elif group == 'Rule_IDs':
                                ids = _parse_rule_ids(rules_str)
                                if ids:
                                    if rule_id_to_msg_map:
                                        parts = []
                                        for rid in ids:
                                            full_msg = rule_id_to_msg_map.get(rid, 'Unknown Rule')
                                            short_msg = _short_msg_label(full_msg) if full_msg != 'Unknown Rule' else full_msg
                                            parts.append(f"{rid}: '{short_msg}'")
                                        data_to_show = "; ".join(parts)
                                    else:
                                        data_to_show = "[" + ", ".join([f'"{rid}"' for rid in ids]) + "]"
                                else:
                                    data_to_show = rules_str if rules_str.strip() else "(No Rules Triggered)"
                            elif group == 'Numerical':
                                num_slice = feature_indices.get('Numerical')
                                if num_slice:
                                    num_values = X_row[0, num_slice.start:min(num_slice.stop, X_row.shape[1])].toarray()[0]
                                    nz = np.where(num_values != 0)[0]
                                    if len(nz) > 0:
                                        feat_vals = {numerical_feature_names[idx]: num_values[idx] for idx in nz if idx < len(numerical_feature_names)}
                                        top_num = sorted(feat_vals.items(), key=lambda item: abs(item[1]), reverse=True)[:3]
                                        data_to_show = ", ".join([f"{n}({v:.2f})" for n, v in top_num])
                                    else:
                                        data_to_show = "(No significant values)"
                                else:
                                    data_to_show = "(Numerical slice error)"
                            else:
                                data_to_show = f"(Contribution: {impact:+.3f})"

                            # Truncate long data
                            if data_to_show and not data_to_show.startswith("(Contribution") and not data_to_show.startswith("(No significant"):
                                data_to_show = data_to_show[:TRUNCATE_LENGTH] + ('...' if len(data_to_show) > TRUNCATE_LENGTH else '')

                            direction = "contributed to" if impact > 0 else "lowered"
                            insight_text = f"'{group}' {direction} Malicious score ({impact:+.3f}): {data_to_show}"
                            insights_log.append(insight_text)
                            if should_print:
                                console_lines.append(f"    - {insight_text}")
                    else:
                        insights_log.append("No single feature group had a dominant impact.")
                        if should_print:
                            console_lines.append("    - No single feature group had a dominant impact.")
                else:
                    insights_log.append("Contribution analysis skipped (no 'malicious' class).")
                    if should_print:
                        console_lines.append("    - Contribution analysis skipped (no 'malicious' class).")
            else:
                insights_log.append(f"Model is {max_prob*100:.1f}% confident.")
                if should_print:
                    console_lines.append(f"  - Model is {max_prob*100:.1f}% confident in this verdict.")

            # Print buffered console lines (respecting filter)
            if should_print:
                for line in console_lines:
                    log.info(line)

            # Always write ALL rows to CSV summary
            csv_row = {
                'entry': i + 1,
                'prediction': pred,
                'uri': uri_str,
                **prob_dict_float,
                'insights': "; ".join(insights_log)
            }
            results_for_csv.append(csv_row)

        # Summary
        log.info("\n" + "=" * 70)
        log.info("OVERALL SUMMARY")
        log.info("=" * 70)
        counts = Counter([p for p in all_preds_calibrated])
        log.info(f"Prediction Breakdown (Total: {total:,}):")
        for k in sorted(counts.keys()):
            c = counts[k]; pct = c / total * 100
            log.info(f"  - {k.capitalize():<12}: {c:>8,} ({pct:>6.2f}%)")

        # CSV export
        if args.output_csv:
            try:
                log.info("\n" + "=" * 70)
                log.info(f"Saving detailed report to {args.output_csv}...")
                results_df = pd.DataFrame(results_for_csv)
                cols = ['entry', 'prediction', 'uri'] + list(classes_calibrated) + ['insights']
                results_df = results_df[results_df.columns.intersection(cols)]
                results_df.to_csv(args.output_csv, index=False, lineterminator='\n')
                log.info("✓ Report saved successfully.")
            except Exception as e:
                log.error(f"Failed to save CSV report: {e}")

        log.info("\n" + "=" * 70)
        log.info("Analysis Complete.")
        log.info("=" * 70)

    except Exception as e:
        log.critical(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit("Script terminated due to a critical error.")

if __name__ == "__main__":
    main()
