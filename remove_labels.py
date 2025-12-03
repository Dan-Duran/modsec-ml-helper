#!/usr/bin/env python3
"""
remove_labels.py - Remove ONLY the programmatic label for analyst review

Analysts will see the same WAF metadata (rules, severities, scores) that the model sees.
They'll make their own expert judgment to compare against both the WAF baseline and model prediction.

Author: Danilo A. Duran
Institution: Georgia Institute of Technology
"""
import pandas as pd
import sys

def remove_labels(input_file, output_file, label_column='label'):
    """Remove only the programmatic label column."""
    
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file, low_memory=False)
    
    print(f"Original shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    if label_column not in df.columns:
        print(f"ERROR: Column '{label_column}' not found!")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)
    
    # Show what analysts WILL see (everything except the label)
    print(f"\n=== Analysts Will See These Features ===")
    print("Evidence Features (raw request data):")
    evidence_cols = ['method', 'uri', 'uri_path', 'uri_query', 'payload', 
                     'user_agent', 'referer', 'status_code', 'content_type']
    for col in evidence_cols:
        if col in df.columns:
            print(f"  ✓ {col}")
    
    print("\nWAF Judgment Features (what ModSecurity said):")
    waf_cols = ['triggered_rule_ids', 'rule_severities', 'rule_tags',
                'anomaly_score', 'sql_injection_score', 'xss_score']
    for col in waf_cols:
        if col in df.columns:
            print(f"  ✓ {col}")
    
    print(f"\nRemoving ONLY: {label_column} (programmatic ground truth)")
    
    # Show label distribution before removal
    print(f"\n=== Label Distribution (being removed) ===")
    print(df[label_column].value_counts())
    
    # Drop ONLY the label column
    df_no_labels = df.drop(columns=[label_column])
    
    print(f"\n=== Result ===")
    print(f"New shape: {df_no_labels.shape[0]} rows × {df_no_labels.shape[1]} columns")
    print(f"Removed: {label_column}")
    
    # Save
    df_no_labels.to_csv(output_file, index=False, lineterminator='\n')
    
    # Verify
    verify_df = pd.read_csv(output_file, low_memory=False)
    
    if label_column in verify_df.columns:
        print(f"\n❌ ERROR: Label column still exists in output!")
        sys.exit(1)
    
    # Verify transaction_id exists
    if 'transaction_id' not in verify_df.columns:
        print(f"\n⚠️  WARNING: No transaction_id - can't match back to ground truth!")
    
    print(f"\n✓ Success! Saved to: {output_file}")
    print(f"✓ Verified: {len(verify_df)} rows, label removed")
    
    print(f"\n=== Validation Study Design ===")
    print("Analysts will see what ModSecurity detected (rules, severities)")
    print("They'll make expert judgment: Is this truly malicious/suspicious/normal?")
    print("\nComparison will be:")
    print("  - WAF baseline (from severities/rules)")
    print("  - Model prediction (contextual refinement)")
    print("  - Analyst consensus (expert ground truth)")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python remove_labels.py <input_with_labels.csv> <output_for_analysts.csv>")
        print("\nExample:")
        print("  python remove_labels.py qualitative_test_suite_with_labels.csv qualitative_test_suite_for_analysts.csv")
        sys.exit(1)
    
    remove_labels(sys.argv[1], sys.argv[2])
