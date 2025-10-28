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

import os
import requests
from dotenv import load_dotenv

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

# ================== MS GRAPH EMAIL FUNCTIONS ==================
def get_ms_graph_token(tenant_id: str, client_id: str, client_secret: str) -> str:
    """Get OAuth token for MS Graph API using client credentials."""
    token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    token_data = {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret,
        'scope': 'https://graph.microsoft.com/.default'
    }
    
    try:
        response = requests.post(token_url, data=token_data, timeout=10)
        response.raise_for_status()
        return response.json()['access_token']
    except Exception as e:
        log.error(f"Failed to get MS Graph token: {e}")
        return None

def send_alert_email(access_token: str, from_user: str, entry_data: dict) -> bool:
    """Send alert email via MS Graph API."""
    send_mail_url = f"https://graph.microsoft.com/v1.0/users/{from_user}/sendMail"
    
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    
    # Build email content
    verdict = entry_data['verdict']
    prob_str = entry_data['prob_str']
    method = entry_data['method']
    uri = entry_data['uri']
    entry_num = entry_data['entry_num']
    insights = entry_data.get('insights', [])
    
    # Set priority based on verdict
    importance = 'high' if verdict == 'malicious' else 'normal'
    
    # Build HTML body
    html_body = f"""
    <html>
    <body style="font-family: 'Courier New', monospace;">
        <h2 style="color: {'#d9534f' if verdict == 'malicious' else '#f0ad4e' if verdict == 'suspicious' else '#5cb85c'};">
            ModSec ML Bot Alert: {verdict.upper()}
        </h2>
        <hr>
        <p><strong>Entry:</strong> #{entry_num}</p>
        <p><strong>Method:</strong> {method}</p>
        <p><strong>URI:</strong> {uri}</p>
        <p><strong>Probability:</strong> {prob_str}</p>
        <hr>
        <h3>Analysis Details:</h3>
        <pre style="background-color: #f5f5f5; padding: 10px; border-left: 3px solid #{'d9534f' if verdict == 'malicious' else 'f0ad4e'};">
"""
    
    for insight in insights:
        html_body += f"{insight}\n"
    
    html_body += """
        </pre>
        <hr>
        <p style="color: #666; font-size: 12px;">
            This is an automated alert from ML ModSec Analyst. Investigate and escalate as necessary.
        </p>
    </body>
    </html>
    """
    
    # Build email message
    email_message = {
        'message': {
            'subject': f'[{verdict.upper()}] - Client: DEMO CLIENT -ModSec ML Bot Alert',
            'importance': importance,
            'body': {
                'contentType': 'HTML',
                'content': html_body
            },
            'toRecipients': [
                {
                    'emailAddress': {
                        'address': from_user  # Send to same mailbox for demo
                    }
                }
            ]
        },
        'saveToSentItems': 'true'
    }
    
    try:
        response = requests.post(send_mail_url, headers=headers, json=email_message, timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        log.error(f"Failed to send email for entry #{entry_num}: {e}")
        return False

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
    # added ELK log file argument
    parser.add_argument('--elk-log-file', type=Path, default=None,
                        help='Optional: Path to write clean, multi-line analysis output specifically for ELK/Filebeat ingestion.')
    # added MS Graph email alerting arguments
    parser.add_argument('--send-alert', action='store_true', help='Send email alerts via MS Graph for malicious/suspicious verdicts.')
    
    args = parser.parse_args()

    # --- DEDICATED ELK LOGGER SETUP ---
    elk_log = None
    elk_formatter = None
    continuation_formatter = None
    if args.elk_log_file:
        try:
            args.elk_log_file.parent.mkdir(parents=True, exist_ok=True)
            elk_log = logging.getLogger('elk_logger')
            elk_log.setLevel(logging.INFO) # Match the main logger level or set as needed

            # Formatter for the FIRST line of each event (includes timestamp for Filebeat)
            elk_formatter = logging.Formatter('%(asctime)s [ANALYST] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            # Formatter for subsequent lines (message only)
            continuation_formatter = logging.Formatter('%(message)s')

            file_handler = logging.FileHandler(args.elk_log_file, mode='a') # Use append mode
            # Set initial formatter (will be changed in the loop)
            file_handler.setFormatter(elk_formatter)

            elk_log.addHandler(file_handler)
            elk_log.propagate = False # Prevent duplication to console logger
            log.info(f"Configured separate logging for ELK to: {args.elk_log_file}")
        except Exception as e:
            log.error(f"Failed to configure ELK logger: {e}")
            elk_log = None # Disable ELK logging if setup fails
   
    # --- END ELK LOGGER SETUP ---

    # Load MS Graph credentials if --send-alert is enabled
    ms_graph_token = None
    ms_imap_user = None
    if args.send_alert:
        log.info("Loading MS Graph credentials from .env file...")
        load_dotenv()
        tenant_id = os.getenv('MS_OAUTH_TENANT_ID')
        client_id = os.getenv('MS_OAUTH_CLIENT_ID')
        client_secret = os.getenv('MS_OAUTH_CLIENT_SECRET')
        ms_imap_user = os.getenv('MS_IMAP_USER')
        
        if not all([tenant_id, client_id, client_secret, ms_imap_user]):
            log.error("Missing MS Graph credentials in .env file. Alerts disabled.")
            args.send_alert = False
        else:
            ms_graph_token = get_ms_graph_token(tenant_id, client_id, client_secret)
            if not ms_graph_token:
                log.error("Failed to authenticate with MS Graph. Alerts disabled.")
                args.send_alert = False
            else:
                log.info(f"✓ MS Graph authenticated successfully for {ms_imap_user}")
    # --- END MS GRAPH SETUP ---

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
            # --- Existing data extraction ---
            pred = all_preds_calibrated[i]
            prob_vector_calibrated = all_probs_calibrated[i]
            max_prob = float(np.max(prob_vector_calibrated))
            X_row = X_all.getrow(i)

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
            # --- End Extraction ---

            prob_dict_float = {c: float(p) for c, p in zip(classes_calibrated, prob_vector_calibrated)}
            prob_dict_str = {c: f"{p:.4f}" for c, p in prob_dict_float.items()}
            prob_log_str = ", ".join([f"P({c})={v}" for c, v in sorted(prob_dict_str.items(), key=lambda x: x[1], reverse=True)])

            action = {
                'malicious': "ACTION: Triage & Investigate. Confirm and block source IP.",
                'suspicious': "ACTION: Monitor. Low-priority investigation. Correlate.",
                'normal': "ACTION: No action required. Observe."
            }.get(pred, "ACTION: N/A")

            verdict_color = ""
            verdict_text = pred.upper() # Base text without color
            if pred == 'malicious':
                verdict_color = COLOR_RED if max_prob >= CONFIDENCE_THRESHOLD else COLOR_ORANGE
            elif pred == 'suspicious':
                verdict_color = COLOR_ORANGE
            elif pred == 'normal' and max_prob < CONFIDENCE_THRESHOLD:
                verdict_color = COLOR_YELLOW

            # --- Build the core analysis lines (WITHOUT color codes initially) ---
            analysis_lines: List[str] = [] # Use a new list name
            analysis_lines.append(f"ENTRY #{i+1} | {method_str} {full_request_path[:TRUNCATE_LENGTH]}") # << FIRST LINE
            if referer_str and referer_str != 'N/A' and referer_str.strip():
                analysis_lines.append(f"  Referer:     {referer_str[:TRUNCATE_LENGTH]}")
            if ua_str and ua_str != 'N/A' and ua_str.strip():
                analysis_lines.append(f"  User-Agent:  {ua_str[:TRUNCATE_LENGTH]}")
            if payload_str and payload_str != 'N/A' and payload_str.strip():
                payload_display = payload_str[:PAYLOAD_TRUNCATE_LENGTH] + ('...' if len(payload_str) > PAYLOAD_TRUNCATE_LENGTH else '')
                analysis_lines.append(f"  Payload:     {payload_display}")
            analysis_lines.append(f"  VERDICT:     {verdict_text}") # Log verdict text only
            analysis_lines.append(f"  PROBABILITY: {prob_log_str}")
            analysis_lines.append(f"  {action}")

            # --- Build insights (split decision details) ---
            insights_lines: List[str] = []
            insights_log_for_csv = [] # Separate list for CSV insights
            if max_prob < CONFIDENCE_THRESHOLD:
                insights_lines.append(f"  --- Split Decision ({max_prob*100:.1f}% confidence): ---")
                if malicious_idx_original != -1:
                    baseline_prob_original = float(all_probs_original[i][malicious_idx_original])
                    contributions = _run_local_contribution(X_row, model_original, baseline_prob_original, malicious_idx_original, feature_indices)

                    if contributions:
                        for group, impact in sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True):
                            # ... (Keep your existing logic to get data_to_show for each group) ...
                            data_to_show = "" # Placeholder - ensure this gets populated by your logic
                            if group == 'URI': data_to_show = uri_str # Use base URI here for context
                            elif group == 'URI_Path': data_to_show = uri_path_str
                            elif group == 'URI_Query': data_to_show = query_str # Use query string here
                            elif group == 'User_Agent': data_to_show = ua_str
                            elif group == 'Payload': data_to_show = payload_str # Use full payload_str here for context before truncating
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
                                    else: data_to_show = "(No significant values)"
                                else: data_to_show = "(Numerical slice error)"
                            else: data_to_show = f"(Contribution: {impact:+.3f})"

                            # Truncate long data *for display/logging*
                            display_data = data_to_show
                            if display_data and not display_data.startswith("(Contribution") and not display_data.startswith("(No significant"):
                                display_data = display_data[:TRUNCATE_LENGTH] + ('...' if len(display_data) > TRUNCATE_LENGTH else '')

                            direction = "contributed to" if impact > 0 else "lowered"
                            insight_text = f"'{group}' {direction} Malicious score ({impact:+.3f}): {display_data}"
                            insights_lines.append(f"    - {insight_text}") # Indent insights for readability
                            insights_log_for_csv.append(insight_text) # Store potentially untruncated version for CSV
                    else:
                        no_impact_msg = "    - No single feature group had a dominant impact."
                        insights_lines.append(no_impact_msg)
                        insights_log_for_csv.append("No single feature group had a dominant impact.")
                else:
                    skip_msg = "    - Contribution analysis skipped (no 'malicious' class)."
                    insights_lines.append(skip_msg)
                    insights_log_for_csv.append("Contribution analysis skipped (no 'malicious' class).")
            else:
                 confident_msg = f"  - Model is {max_prob*100:.1f}% confident in this verdict."
                 insights_lines.append(confident_msg)
                 insights_log_for_csv.append(f"Model is {max_prob*100:.1f}% confident.")


            # --- Write to ELK Log (if configured) ---
            if elk_log and elk_formatter and continuation_formatter:
                try:
                    # Log the first line with the main formatter (includes timestamp + prefix)
                    elk_log.handlers[0].setFormatter(elk_formatter)
                    elk_log.info(analysis_lines[0]) # Log ENTRY # line

                    # Log subsequent lines with continuation formatter (message only)
                    elk_log.handlers[0].setFormatter(continuation_formatter)
                    for line in analysis_lines[1:]:
                        line_no_color = re.sub(r'\x1b\[[0-9;]*[mK]', '', line) # Remove colors
                        # Indent subsequent lines for ELK readability
                        elk_log.info(f"  {line_no_color.strip()}")
                    for line in insights_lines:
                        line_no_color = re.sub(r'\x1b\[[0-9;]*[mK]', '', line) # Remove colors
                        elk_log.info(line_no_color) # Insights already have indentation
                except Exception as e_elk:
                    log.error(f"Error writing entry #{i+1} to ELK log: {e_elk}")


            # --- Write to Console Log (respecting filter) ---
            should_print_to_console = True
            if args.filtered and max_prob >= CONFIDENCE_THRESHOLD:
                should_print_to_console = False

            if should_print_to_console:
                log.info("\n" + "-" * 70) # Separator for console
                # Print lines with color codes for console
                for line_idx, line in enumerate(analysis_lines):
                    if "VERDICT:" in line: # Add color back to verdict line
                         log.info(f"  VERDICT:     {verdict_color}{verdict_text}{COLOR_RESET}")
                    else:
                         log.info(line)
                for line in insights_lines:
                    log.info(line)

            # --- Send Email Alert (if enabled) ---
            if args.send_alert and ms_graph_token and pred in ['malicious', 'suspicious']:
                alert_data = {
                    'entry_num': i + 1,
                    'verdict': pred,
                    'method': method_str,
                    'uri': full_request_path,
                    'prob_str': prob_log_str,
                    'insights': [line.strip() for line in insights_lines if line.strip() and not line.strip().startswith('---')]
                }
                if send_alert_email(ms_graph_token, ms_imap_user, alert_data):
                    log.info(f"  ✓ Alert email sent for entry #{i+1}")
                else:
                    log.warning(f"  ✗ Failed to send alert email for entry #{i+1}")

            # --- Always write ALL rows to CSV summary ---
            csv_row = {
                'entry': i + 1,
                'prediction': pred,
                'uri': full_request_path, # Use full path for CSV
                **prob_dict_float,
                'insights': "; ".join(insights_log_for_csv) # Use CSV-specific insights
            }
            results_for_csv.append(csv_row)

        # --- End of loop ---

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
