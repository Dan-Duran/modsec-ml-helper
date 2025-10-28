#!/usr/bin/env python3
"""
feature_engineer.py - Leakage-Free Feature Engineering

Usage:

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

Author: Danilo A. Duran
Institution: Georgia Institute of Technology

"""

import pandas as pd
import numpy as np
import argparse
import json
import logging
import subprocess
from pathlib import Path
from collections import Counter
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix, save_npz, issparse
import joblib
import sklearn
import scipy
import sys

warnings.filterwarnings('ignore')

# ============= CONSTANTS =============

# CRITICAL: Fixed timestamp to prevent "now" leakage
FIXED_TIMESTAMP = pd.Timestamp('1970-01-01', tz='UTC')

# CRITICAL: Locked HTTP method categories to prevent drift
ALLOWED_HTTP_METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS', 'PATCH', 'CONNECT', 'TRACE']

# ============= VERSION COMPATIBILITY =============

def get_ohe_kwargs():
    """Get OneHotEncoder kwargs based on sklearn version."""
    sklearn_version = tuple(map(int, sklearn.__version__.split('.')[:2]))
    if sklearn_version >= (1, 2):
        return {'sparse_output': True}
    else:
        return {'sparse': True}


def get_git_sha() -> Optional[str]:
    """Get current git commit SHA if available."""
    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()
        return sha
    except Exception:
        return None


# ============= CONFIGURATION =============

DEFAULT_CONFIG = {
    'train_frac': 0.7,
    'val_frac': 0.15,
    'test_frac': 0.15,
    'min_df': 3,
    'uri_ngram_range': (3, 5),
    'uri_max_features': 800,
    'uri_path_ngram_range': (2, 4),
    'uri_path_max_features': 300,
    'uri_query_ngram_range': (2, 4),
    'uri_query_max_features': 400,
    'payload_ngram_range': (3, 5),
    'payload_max_features': 800,
    'ua_max_features': 100,
    'random_seed': 42,
    'dtype': np.float32,
    'min_samples_per_class': 2
}

# ============= SCHEMA VALIDATION =============

def validate_schema(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    CRITICAL FIX: Validate schema and create missing columns with defaults.
    
    Fail fast on missing required columns, create optional columns with zeros.
    """
    required_columns = [
        'uri', 'uri_path', 'uri_query', 'payload', 
        'user_agent', 'referer', 'method'
    ]
    
    optional_columns = {
        'source_port': 0,
        'dest_port': 0,
        'timestamp': '1970-01-01 00:00:00 +0000'
    }
    
    # Check required columns
    missing_required = [c for c in required_columns if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")
    
    # Create optional columns if missing
    for col, default in optional_columns.items():
        if col not in df.columns:
            logger.warning(f"Missing optional column '{col}', filling with default: {default}")
            df[col] = default
    
    logger.info("✓ Schema validation passed")
    return df


# ============= HELPER FUNCTIONS =============

def calculate_entropy_fast(text: str) -> float:
    """Fast Shannon entropy calculation using Counter (O(n))."""
    if not text or len(text) == 0:
        return 0.0
    
    char_counts = Counter(text)
    text_len = len(text)
    
    entropy = 0.0
    for count in char_counts.values():
        prob = count / text_len
        entropy -= prob * np.log2(prob)
    
    return entropy


def setup_logging(verbose: bool = False, log_file: str = None) -> logging.Logger:
    """Configure logging with optional file output."""
    logger = logging.getLogger('feature_engineer')
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


def validate_fractions(train_frac: float, val_frac: float, logger: logging.Logger):
    """Validate train/val/test fractions sum to ~1.0."""
    test_frac = 1.0 - train_frac - val_frac
    
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and 0 < test_frac < 1):
        raise ValueError(
            f"Invalid fractions: train={train_frac}, val={val_frac}, test={test_frac}. "
            f"All must be in (0, 1) and sum to 1.0"
        )
    
    if test_frac < 0.05:
        logger.warning(f"Test fraction is very small: {test_frac:.3f}")
    
    logger.info(f"Split fractions: train={train_frac}, val={val_frac}, test={test_frac:.3f}")


def validate_timestamps(df: pd.DataFrame, logger: logging.Logger) -> pd.Series:
    """
    Parse timestamps with multiple format fallbacks.
    
    CRITICAL FIX: Uses fixed timestamp (1970-01-01) instead of "now" to prevent leakage.
    """
    if 'timestamp' not in df.columns:
        logger.warning("No 'timestamp' column found. Using fixed default.")
        return pd.Series([FIXED_TIMESTAMP] * len(df), index=df.index)
    
    formats_to_try = [
        '%d/%b/%Y:%H:%M:%S %z',  # Apache format
        '%Y-%m-%d %H:%M:%S',      # ISO-like
        '%Y-%m-%dT%H:%M:%S%z',    # ISO 8601
    ]
    
    parsed_times = None
    parse_errors = len(df)
    
    for fmt in formats_to_try:
        try:
            parsed_times = pd.to_datetime(df['timestamp'], format=fmt, errors='coerce', utc=True)
            parse_errors = parsed_times.isna().sum()
            if parse_errors < len(df) * 0.5:
                break
        except Exception:
            continue
    
    # CRITICAL FIX: Use fixed timestamp, not "now"
    if parsed_times is None or parse_errors == len(df):
        logger.warning(f"Timestamp parsing failed. Using fixed default: {FIXED_TIMESTAMP}")
        parsed_times = pd.Series([FIXED_TIMESTAMP] * len(df), index=df.index)
    else:
        error_rate = parse_errors / len(df) * 100
        logger.info(f"Timestamp parse error rate: {error_rate:.2f}% ({parse_errors:,}/{len(df):,})")
        # Fill failed parses with fixed timestamp
        parsed_times = parsed_times.fillna(FIXED_TIMESTAMP)
    
    return parsed_times


def check_integrity(matrix, labels, expected_features: Optional[int], logger: logging.Logger):
    """
    Validate feature matrix integrity.
    
    CRITICAL FIX: Checks dimension drift and ensures CSR format.
    Returns the matrix (possibly converted to CSR).
    """
    if issparse(matrix):
        # Force CSR format
        if matrix.format != 'csr':
            logger.warning(f"Matrix format is {matrix.format}, converting to CSR")
            matrix = matrix.tocsr()
        
        # Check for NaN/Inf
        if np.any(~np.isfinite(matrix.data)):
            raise ValueError("Feature matrix contains NaN or Inf values!")
        
        sparsity = 1 - (matrix.nnz / (matrix.shape[0] * matrix.shape[1]))
        memory_mb = (matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes) / 1024 / 1024
        
        logger.info(f"Matrix integrity check:")
        logger.info(f"  Shape: {matrix.shape}")
        logger.info(f"  Format: {matrix.format}")
        logger.info(f"  Dtype: {matrix.dtype}")
        logger.info(f"  Sparsity: {sparsity*100:.2f}%")
        logger.info(f"  Non-zero: {matrix.nnz:,}")
        logger.info(f"  Memory: {memory_mb:.2f} MB (sparse)")
        
        # CRITICAL: Check dimension drift
        if expected_features is not None and matrix.shape[1] != expected_features:
            raise ValueError(
                f"Feature dimension mismatch! Expected {expected_features}, got {matrix.shape[1]}. "
                f"This indicates column alignment drift."
            )
    
    # Check labels
    if labels is not None and not isinstance(labels, int):
        if labels.isna().any():
            raise ValueError(f"Labels contain {labels.isna().sum()} NaN values!")
        logger.info(f"Label distribution: {labels.value_counts().to_dict()}")
    
    return matrix  # CRITICAL FIX: Return (possibly converted) matrix


def safe_str_ops(df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
    """
    CRITICAL FIX: Ensure text columns are strings before .str operations.
    """
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)
    return df


def normalize_http_methods(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    CRITICAL FIX: Normalize HTTP methods to locked categories.
    
    Prevents category explosion from junk/rare methods.
    """
    if 'method' not in df.columns:
        df['method'] = 'GET'
        return df
    
    # CRITICAL FIX: Cast to string before .str operations
    df['method'] = df['method'].astype(str).fillna('GET').str.upper()
    
    original_methods = df['method'].value_counts()
    
    # Map unknowns to 'OTHER'
    df.loc[~df['method'].isin(ALLOWED_HTTP_METHODS), 'method'] = 'OTHER'
    
    normalized_methods = df['method'].value_counts()
    
    if len(original_methods) != len(normalized_methods):
        logger.info(f"HTTP method normalization: {len(original_methods)} → {len(normalized_methods)} categories")
        logger.debug(f"Original methods: {original_methods.to_dict()}")
        logger.debug(f"Normalized methods: {normalized_methods.to_dict()}")
    
    normalized_methods = df['method'].value_counts()
    
    if len(original_methods) != len(normalized_methods):
        logger.info(f"HTTP method normalization: {len(original_methods)} → {len(normalized_methods)} categories")
        logger.debug(f"Original methods: {original_methods.to_dict()}")
        logger.debug(f"Normalized methods: {normalized_methods.to_dict()}")
    
    return df


# ============= FEATURE ENGINEERING FUNCTIONS =============

def create_statistical_features(df: pd.DataFrame, config: Dict, logger: logging.Logger) -> pd.DataFrame:
    """
    Generate inference-time statistical features.
    
    CRITICAL FIXES:
    - Schema validation ensures all columns exist
    - All text columns cast to string before .str operations
    - Excludes server response features
    """
    # CRITICAL FIX: Cast text columns to string FIRST
    text_cols = ['uri', 'uri_path', 'uri_query', 'payload', 'user_agent', 'referer']
    df = safe_str_ops(df, text_cols)
    
    features = pd.DataFrame(index=df.index)
    
    # Length features
    features['uri_length'] = df['uri'].str.len()
    features['uri_path_length'] = df['uri_path'].str.len()
    features['uri_query_length'] = df['uri_query'].str.len()
    features['payload_length'] = df['payload'].str.len()
    features['user_agent_length'] = df['user_agent'].str.len()
    features['referer_length'] = df['referer'].str.len()
    
    # Structural features
    features['uri_depth'] = df['uri'].str.count('/')
    features['path_depth'] = df['uri_path'].str.count('/')
    features['uri_num_params'] = df['uri_query'].str.count('&') + (df['uri_query'].str.len() > 0).astype(config['dtype'])
    
    # Parameter-level features
    features['param_key_count'] = df['uri_query'].apply(lambda x: len(x.split('=')) - 1 if x else 0)
    
    def safe_avg_param_len(x):
        if not x:
            return 0.0
        params = [p for p in x.split('&') if '=' in p]
        if not params:
            return 0.0
        return float(np.mean([len(p.split('=')[-1]) for p in params]))
    
    def safe_max_param_len(x):
        if not x:
            return 0
        params = [p for p in x.split('&') if '=' in p]
        if not params:
            return 0
        return max([len(p.split('=')[-1]) for p in params])
    
    features['avg_param_len'] = df['uri_query'].apply(safe_avg_param_len)
    features['max_param_len'] = df['uri_query'].apply(safe_max_param_len)
    
    # Ratio features
    features['query_uri_ratio'] = features['uri_query_length'] / (features['uri_length'] + 1)
    features['payload_uri_ratio'] = features['payload_length'] / (features['uri_length'] + 1)
    
    # Character distribution
    features['uri_num_digits'] = df['uri'].str.count(r'\d')
    features['uri_num_special'] = df['uri'].str.count(r'[^a-zA-Z0-9/.\-_?&=]')
    features['uri_alpha_ratio'] = df['uri'].str.count(r'[a-zA-Z]') / (features['uri_length'] + 1)
    features['uri_digit_ratio'] = features['uri_num_digits'] / (features['uri_length'] + 1)
    
    features['payload_num_digits'] = df['payload'].str.count(r'\d')
    features['payload_num_special'] = df['payload'].str.count(r'[^a-zA-Z0-9]')
    features['payload_alpha_ratio'] = df['payload'].str.count(r'[a-zA-Z]') / (features['payload_length'] + 1)
    
    # Entropy features
    logger.debug("Computing entropy features...")
    features['uri_entropy'] = df['uri'].apply(calculate_entropy_fast)
    features['uri_query_entropy'] = df['uri_query'].apply(calculate_entropy_fast)
    features['payload_entropy'] = df['payload'].apply(calculate_entropy_fast)
    
    # Encoding features
    features['uri_percent_encoded_ratio'] = df['uri'].str.count('%') / (features['uri_length'] + 1)
    
    # Header features (inference-time only - no status_code/content_length)
    features['has_referer'] = (df['referer'].str.len() > 0).astype(config['dtype'])
    features['source_port_numeric'] = pd.to_numeric(df['source_port'], errors='coerce').fillna(0).astype(config['dtype'])
    features['dest_port_numeric'] = pd.to_numeric(df['dest_port'], errors='coerce').fillna(0).astype(config['dtype'])
    features['is_standard_port'] = features['dest_port_numeric'].isin([80, 443, 8080, 8443]).astype(config['dtype'])
    
    # Replace NaN/Inf with 0
    features = features.replace([np.inf, -np.inf], 0).fillna(0)
    
    return features.astype(config['dtype'])


def extract_time_features(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Extract temporal features with defensive parsing."""
    parsed_times = validate_timestamps(df, logger)
    
    time_features = pd.DataFrame(index=df.index)
    time_features['hour'] = parsed_times.dt.hour.fillna(0).astype(np.int8)
    time_features['day_of_week'] = parsed_times.dt.dayofweek.fillna(0).astype(np.int8)
    time_features['is_weekend'] = (time_features['day_of_week'] >= 5).astype(np.int8)
    time_features['is_night'] = ((time_features['hour'] < 6) | (time_features['hour'] > 22)).astype(np.int8)
    
    return time_features


def check_stratify_validity(labels: pd.Series, min_samples: int, logger: logging.Logger, split_name: str = "") -> bool:
    """
    CRITICAL FIX: Check if stratification is possible for a given label set.
    """
    class_counts = labels.value_counts()
    min_class_count = class_counts.min()
    
    if min_class_count < min_samples:
        logger.warning(
            f"Stratification disabled for {split_name}: smallest class has {min_class_count} samples "
            f"(need ≥{min_samples}). Distribution: {class_counts.to_dict()}"
        )
        return False
    
    return True


# ============= MAIN FUNCTIONS =============

def fit_mode(df: pd.DataFrame, config: Dict, args, logger: logging.Logger):
    """
    FIT mode: Train/val/test split, fit transformers on training data only.
    
    CRITICAL FIXES v2.2:
    - Schema validation upfront
    - Float32 enforcement after scaling
    - CSR format enforcement before save
    - HTTP method normalization
    - Double stratify check
    """
    logger.info(f"Running in FIT mode with {len(df):,} total samples")
    
    # CRITICAL: Validate schema first
    df = validate_schema(df, logger)
    
    # Validate fractions
    validate_fractions(config['train_frac'], config['val_frac'], logger)
    
    # Separate labels
    if 'label' not in df.columns:
        raise ValueError("Input CSV missing required 'label' column")
    
    labels = df['label']
    features_df = df.drop(columns=['label'])
    
    # Normalize HTTP methods
    features_df = normalize_http_methods(features_df, logger)
    
    # Fill missing values
    logger.info("Handling missing values...")
    features_df.fillna({
        'payload': '', 'uri': '', 'uri_path': '', 'uri_query': '',
        'user_agent': '', 'content_type': '', 'host': '', 'referer': '',
        'method': 'GET', 'source_port': 0, 'dest_port': 0,
        'timestamp': '1970-01-01 00:00:00 +0000'
    }, inplace=True)
    
    # CRITICAL: Check stratification for first split
    can_stratify = check_stratify_validity(labels, config['min_samples_per_class'], logger, "train/test split")
    stratify_arg = labels if can_stratify else None
    
    # Train/Test split
    logger.info(f"Splitting: train={config['train_frac']}, val={config['val_frac']}, test={config['test_frac']:.3f}")
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        features_df, labels, 
        test_size=config['test_frac'],
        random_state=config['random_seed'],
        stratify=stratify_arg
    )
    
    # CRITICAL FIX: Check stratification for second split (train/val)
    can_stratify_temp = check_stratify_validity(y_temp, config['min_samples_per_class'], logger, "train/val split")
    stratify_temp = y_temp if can_stratify_temp else None
    
    val_ratio = config['val_frac'] / (config['train_frac'] + config['val_frac'])
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=config['random_seed'],
        stratify=stratify_temp
    )
    
    logger.info(f"Split sizes: train={len(X_train):,}, val={len(X_val):,}, test={len(X_test):,}")
    
    # Generate statistical features
    logger.info("Generating statistical features...")
    stat_features_train = create_statistical_features(X_train, config, logger)
    stat_features_val = create_statistical_features(X_val, config, logger)
    stat_features_test = create_statistical_features(X_test, config, logger)
    
    # Time features
    logger.info("Extracting temporal features...")
    time_features_train = extract_time_features(X_train, logger)
    time_features_val = extract_time_features(X_val, logger)
    time_features_test = extract_time_features(X_test, logger)
    
    # Combine statistical + temporal
    numerical_train = pd.concat([stat_features_train, time_features_train], axis=1)
    numerical_val = pd.concat([stat_features_val, time_features_val], axis=1)
    numerical_test = pd.concat([stat_features_test, time_features_test], axis=1)
    
    # CRITICAL: Save column order for transform mode
    numerical_columns = list(numerical_train.columns)
    logger.info(f"Numerical features: {len(numerical_columns)} columns")
    
    # CRITICAL FIX: Sparse-native scaling with float32 enforcement
    logger.info("Scaling numerical features (sparse-native with float32)...")
    numerical_train_sparse = csr_matrix(numerical_train.values.astype(config['dtype']))
    numerical_val_sparse = csr_matrix(numerical_val.values.astype(config['dtype']))
    numerical_test_sparse = csr_matrix(numerical_test.values.astype(config['dtype']))
    
    scaler = MaxAbsScaler()
    # CRITICAL FIX: Force float32 after scaling
    numerical_train_scaled = scaler.fit_transform(numerical_train_sparse).astype(config['dtype'])
    numerical_val_scaled = scaler.transform(numerical_val_sparse).astype(config['dtype'])
    numerical_test_scaled = scaler.transform(numerical_test_sparse).astype(config['dtype'])
    
    logger.info(f"Scaled dtype: {numerical_train_scaled.dtype}")
    
    # Text vectorization
    logger.info("Vectorizing text features...")
    
    # Cast to string
    X_train = safe_str_ops(X_train, ['uri', 'uri_path', 'uri_query', 'payload', 'user_agent'])
    X_val = safe_str_ops(X_val, ['uri', 'uri_path', 'uri_query', 'payload', 'user_agent'])
    X_test = safe_str_ops(X_test, ['uri', 'uri_path', 'uri_query', 'payload', 'user_agent'])
    
    # URI (full)
    uri_vec = TfidfVectorizer(
        analyzer='char', ngram_range=config['uri_ngram_range'],
        max_features=config['uri_max_features'], min_df=config['min_df']
    )
    uri_train = uri_vec.fit_transform(X_train['uri'].fillna('')).astype(config['dtype'])
    uri_val = uri_vec.transform(X_val['uri'].fillna('')).astype(config['dtype'])
    uri_test = uri_vec.transform(X_test['uri'].fillna('')).astype(config['dtype'])
    
    # URI Path
    uri_path_vec = TfidfVectorizer(
        analyzer='char', ngram_range=config['uri_path_ngram_range'],
        max_features=config['uri_path_max_features'], min_df=config['min_df']
    )
    uri_path_train = uri_path_vec.fit_transform(X_train['uri_path'].fillna('')).astype(config['dtype'])
    uri_path_val = uri_path_vec.transform(X_val['uri_path'].fillna('')).astype(config['dtype'])
    uri_path_test = uri_path_vec.transform(X_test['uri_path'].fillna('')).astype(config['dtype'])
    
    # URI Query
    uri_query_vec = TfidfVectorizer(
        analyzer='char', ngram_range=config['uri_query_ngram_range'],
        max_features=config['uri_query_max_features'], min_df=config['min_df']
    )
    uri_query_train = uri_query_vec.fit_transform(X_train['uri_query'].fillna('')).astype(config['dtype'])
    uri_query_val = uri_query_vec.transform(X_val['uri_query'].fillna('')).astype(config['dtype'])
    uri_query_test = uri_query_vec.transform(X_test['uri_query'].fillna('')).astype(config['dtype'])
    
    # Payload
    payload_vec = TfidfVectorizer(
        analyzer='char', ngram_range=config['payload_ngram_range'],
        max_features=config['payload_max_features'], min_df=config['min_df']
    )
    payload_train = payload_vec.fit_transform(X_train['payload'].fillna('')).astype(config['dtype'])
    payload_val = payload_vec.transform(X_val['payload'].fillna('')).astype(config['dtype'])
    payload_test = payload_vec.transform(X_test['payload'].fillna('')).astype(config['dtype'])
    
    # User Agent
    ua_vec = TfidfVectorizer(
        analyzer='word', max_features=config['ua_max_features'],
        min_df=config['min_df']
    )
    ua_train = ua_vec.fit_transform(X_train['user_agent'].fillna('')).astype(config['dtype'])
    ua_val = ua_vec.transform(X_val['user_agent'].fillna('')).astype(config['dtype'])
    ua_test = ua_vec.transform(X_test['user_agent'].fillna('')).astype(config['dtype'])

    # =================================================================
    # --- FIX: ADD THIS ENTIRE BLOCK FOR THE MISSING FEATURES ---
    # =================================================================
    logger.info("Vectorizing ModSecurity rule and severity features...")

    # Cast rule/severity columns to string
    X_train = safe_str_ops(X_train, ['triggered_rule_ids', 'rule_severities'])
    X_val = safe_str_ops(X_val, ['triggered_rule_ids', 'rule_severities'])
    X_test = safe_str_ops(X_test, ['triggered_rule_ids', 'rule_severities'])

    # Vectorize Triggered Rule IDs
    rule_ids_vec = TfidfVectorizer(
        analyzer='word', # Treat each ID as a "word"
        token_pattern=r'\d{6}', # Match 6-digit rule IDs
        binary=True, # Presence/absence is enough
        min_df=config['min_df']
    )
    rule_ids_train = rule_ids_vec.fit_transform(X_train['triggered_rule_ids'].fillna('[]')).astype(config['dtype'])
    rule_ids_val = rule_ids_vec.transform(X_val['triggered_rule_ids'].fillna('[]')).astype(config['dtype'])
    rule_ids_test = rule_ids_vec.transform(X_test['triggered_rule_ids'].fillna('[]')).astype(config['dtype'])

    # Vectorize Rule Severities
    severities_vec = TfidfVectorizer(
        analyzer='word',
        binary=True
    )
    severities_train = severities_vec.fit_transform(X_train['rule_severities'].fillna('[]')).astype(config['dtype'])
    severities_val = severities_vec.transform(X_val['rule_severities'].fillna('[]')).astype(config['dtype'])
    severities_test = severities_vec.transform(X_test['rule_severities'].fillna('[]')).astype(config['dtype'])
    # =================================================================
    # --- END OF BLOCK TO ADD ---
    # =================================================================
    
    # Categorical encoding
    logger.info("Encoding categorical features...")
    
    ohe_kwargs = get_ohe_kwargs()
    # CRITICAL FIX: Lock categories to prevent drift
    method_enc = OneHotEncoder(
        categories=[ALLOWED_HTTP_METHODS + ['OTHER']],
        handle_unknown='ignore',
        dtype=config['dtype'],
        **ohe_kwargs
    )
    method_train = method_enc.fit_transform(X_train[['method']])
    method_val = method_enc.transform(X_val[['method']])
    method_test = method_enc.transform(X_test[['method']])
    
    # Combine all features
    logger.info("Assembling final feature matrices...")
    
    train_features = hstack([
        numerical_train_scaled, uri_train, uri_path_train, uri_query_train,
        payload_train, ua_train, method_train,
        rule_ids_train, severities_train
    ])
    val_features = hstack([
        numerical_val_scaled, uri_val, uri_path_val, uri_query_val,
        payload_val, ua_val, method_val,
        rule_ids_val, severities_val
    ])
    test_features = hstack([
        numerical_test_scaled, uri_test, uri_path_test, uri_query_test,
        payload_test, ua_test, method_test,
        rule_ids_test, severities_test
    ])
    
    # CRITICAL FIX: Force CSR format before saving
    train_features = train_features.tocsr()
    val_features = val_features.tocsr()
    test_features = test_features.tocsr()
    
    # Integrity checks (returns possibly converted matrix)
    logger.info("Running integrity checks...")
    train_features = check_integrity(train_features, y_train, None, logger)
    val_features = check_integrity(val_features, y_val, train_features.shape[1], logger)
    test_features = check_integrity(test_features, y_test, train_features.shape[1], logger)
    
    # Save everything
    logger.info("Saving outputs...")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_npz(output_dir / 'train_features.npz', train_features)
    save_npz(output_dir / 'val_features.npz', val_features)
    save_npz(output_dir / 'test_features.npz', test_features)
    
    y_train.to_csv(output_dir / 'train_labels.csv', index=False)
    y_val.to_csv(output_dir / 'val_labels.csv', index=False)
    y_test.to_csv(output_dir / 'test_labels.csv', index=False)
    
    # Save transformers
    transformers = {
        'scaler': scaler,
        'uri_vectorizer': uri_vec,
        'uri_path_vectorizer': uri_path_vec,
        'uri_query_vectorizer': uri_query_vec,
        'payload_vectorizer': payload_vec,
        'ua_vectorizer': ua_vec,
        'method_encoder': method_enc,
        'numerical_columns': numerical_columns,
        'allowed_http_methods': ALLOWED_HTTP_METHODS,
        'rule_ids_vectorizer': rule_ids_vec,
        'severities_vectorizer': severities_vec
    }
    joblib.dump(transformers, output_dir / 'transformers.joblib')
    
    # Save comprehensive metadata
    # CRITICAL FIX: Convert dtype to string for JSON serialization
    config_serializable = config.copy()
    config_serializable['dtype'] = str(config['dtype'].__name__)  # np.float32 -> 'float32'
    
    metadata = {
        'version': '2.3.1',
        'timestamp': datetime.now().isoformat(),
        'git_sha': get_git_sha(),
        'system_info': {
            'python_version': sys.version,
            'sklearn_version': sklearn.__version__,
            'pandas_version': pd.__version__,
            'numpy_version': np.__version__,
            'scipy_version': scipy.__version__
        },
        'config': config_serializable,
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'total_features': train_features.shape[1],
        'feature_breakdown': {
            'numerical': len(numerical_columns),
            'uri_tfidf': uri_train.shape[1],
            'uri_path_tfidf': uri_path_train.shape[1],
            'uri_query_tfidf': uri_query_train.shape[1],
            'payload_tfidf': payload_train.shape[1],
            'ua_tfidf': ua_train.shape[1],
            'method_ohe': method_train.shape[1]
        },
        'label_distribution': {
            'train': y_train.value_counts().to_dict(),
            'val': y_val.value_counts().to_dict(),
            'test': y_test.value_counts().to_dict()
        },
        'stratify_enabled': {
            'train_test_split': can_stratify,
            'train_val_split': can_stratify_temp
        }
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✓ All outputs saved to {output_dir}/")
    logger.info(f"✓ Total features: {train_features.shape[1]}")
    logger.info(f"✓ Matrix dtype: {train_features.dtype}")
    logger.info(f"✓ Matrix format: {train_features.format}")
    logger.info("✓ Ready for model training!")


def transform_mode(df: pd.DataFrame, args, logger: logging.Logger):
    """
    TRANSFORM mode: Load fitted transformers and transform new data.
    
    CRITICAL FIXES:
    - Schema validation
    - Column alignment enforcement
    - Float32 enforcement
    - CSR format enforcement
    """
    logger.info(f"Running in TRANSFORM mode with {len(df):,} samples")
    
    # Load transformers
    transformers_path = Path(args.transformers_path)
    if not transformers_path.exists():
        raise FileNotFoundError(f"Transformers not found: {transformers_path}")
    
    transformers = joblib.load(transformers_path)
    logger.info("Loaded fitted transformers")
    
    # Load metadata
    metadata_path = transformers_path.parent / 'metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    config = metadata['config']
    expected_features = metadata['total_features']
    
    # CRITICAL: Validate schema
    df = validate_schema(df, logger)
    
    # Normalize HTTP methods
    df = normalize_http_methods(df, logger)
    
    # Fill missing values
    df.fillna({
        'payload': '', 'uri': '', 'uri_path': '', 'uri_query': '',
        'user_agent': '', 'method': 'GET', 'source_port': 0, 'dest_port': 0,
        'timestamp': '1970-01-01 00:00:00 +0000'
    }, inplace=True)
    
    # Generate features
    logger.info("Generating statistical features...")
    stat_features = create_statistical_features(df, config, logger)
    
    logger.info("Extracting temporal features...")
    time_features = extract_time_features(df, logger)
    
    numerical = pd.concat([stat_features, time_features], axis=1)
    
    # CRITICAL: Enforce column alignment
    numerical_columns = transformers['numerical_columns']
    numerical = numerical.reindex(columns=numerical_columns, fill_value=0)
    logger.info(f"Aligned numerical features to {len(numerical_columns)} columns")
    
    # Cast to string
    df = safe_str_ops(df, ['uri', 'uri_path', 'uri_query', 'payload', 'user_agent'])
    
    # CRITICAL FIX: Sparse-native scaling with float32 enforcement
    logger.info("Applying fitted transformers...")
    numerical_sparse = csr_matrix(numerical.values.astype(config['dtype']))
    numerical_scaled = transformers['scaler'].transform(numerical_sparse).astype(config['dtype'])
    
    # CRITICAL FIX: Cast to float32 after transform for sklearn compatibility
    uri_tfidf = transformers['uri_vectorizer'].transform(df['uri'].fillna('')).astype(config['dtype'])
    uri_path_tfidf = transformers['uri_path_vectorizer'].transform(df['uri_path'].fillna('')).astype(config['dtype'])
    uri_query_tfidf = transformers['uri_query_vectorizer'].transform(df['uri_query'].fillna('')).astype(config['dtype'])
    payload_tfidf = transformers['payload_vectorizer'].transform(df['payload'].fillna('')).astype(config['dtype'])
    ua_tfidf = transformers['ua_vectorizer'].transform(df['user_agent'].fillna('')).astype(config['dtype'])
    method_ohe = transformers['method_encoder'].transform(df[['method']])
    
    # Combine
    features = hstack([
        numerical_scaled, uri_tfidf, uri_path_tfidf, uri_query_tfidf,
        payload_tfidf, ua_tfidf, method_ohe
    ])
    
    # CRITICAL FIX: Force CSR format
    features = features.tocsr()
    
    # Integrity check
    logger.info("Running integrity check...")
    check_integrity(features, 0, expected_features, logger)
    
    # Save
    output_path = Path(args.output_features)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(output_path, features)
    
    logger.info(f"✓ Features saved to {output_path}")
    logger.info(f"✓ Shape: {features.shape}")
    logger.info(f"✓ Dtype: {features.dtype}")
    logger.info(f"✓ Format: {features.format}")
    
    return features


# ============= CLI =============

def main():
    parser = argparse.ArgumentParser(
        description="Production-grade leakage-free feature engineering v2.3.1 (FINAL)"
    )
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--mode', choices=['fit', 'transform'], default='fit',
                       help='fit: train/val/test split; transform: use fitted transformers')
    parser.add_argument('--output_dir', type=str, default='features_output',
                       help='Output directory for fit mode')
    parser.add_argument('--output_features', type=str, default='features.npz',
                       help='Output features for transform mode')
    parser.add_argument('--transformers_path', type=str, default='features_output/transformers.joblib',
                       help='Path to fitted transformers (transform mode)')
    parser.add_argument('--train_frac', type=float, default=0.7, help='Training fraction')
    parser.add_argument('--val_frac', type=float, default=0.15, help='Validation fraction')
    parser.add_argument('--min_df', type=int, default=3, help='Minimum document frequency for TF-IDF')
    parser.add_argument('--uri_max_features', type=int, default=800, help='Max URI TF-IDF features')
    parser.add_argument('--payload_max_features', type=int, default=800, help='Max payload TF-IDF features')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    parser.add_argument('--log_file', type=str, help='Log file path')
    parser.add_argument('--limit', type=int, help='Limit number of rows (for testing)')
    parser.add_argument('--dry_run', action='store_true', help='Compute shapes/stats only')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose, args.log_file)
    
    # Log version info
    logger.info(f"Feature Engineering Pipeline v2.3.1 (FINAL)")
    logger.info(f"Python {sys.version}")
    logger.info(f"sklearn {sklearn.__version__}, pandas {pd.__version__}, numpy {np.__version__}")
    
    # Load config
    config = DEFAULT_CONFIG.copy()
    config.update({
        'train_frac': args.train_frac,
        'val_frac': args.val_frac,
        'test_frac': 1 - args.train_frac - args.val_frac,
        'min_df': args.min_df,
        'uri_max_features': args.uri_max_features,
        'payload_max_features': args.payload_max_features
    })
    
    # Load data
    logger.info(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input, nrows=args.limit if args.limit else None)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    if args.dry_run:
        logger.info("DRY RUN - computing stats only")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        if 'label' in df.columns:
            logger.info(f"Label distribution:\n{df['label'].value_counts()}")
        return
    
    # Run appropriate mode
    if args.mode == 'fit':
        fit_mode(df, config, args, logger)
    else:
        transform_mode(df, args, logger)


if __name__ == "__main__":
    main()
