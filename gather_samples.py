#!/usr/bin/env python3
"""
gather_samples.py - Extract evenly-spaced samples from a CSV file with optional stratification
This script extracts rows from a CSV file using either:
- Two-stage sampling: frequency filter then even distribution
- Stratified sampling: exact counts per class label

Author: Dan Duran
"""
import argparse
import sys
import pandas as pd
import numpy as np

def sanitize_field(value):
    """Remove newlines and extra whitespace from field values."""
    if pd.isna(value):
        return value
    if isinstance(value, str):
        # Replace newlines with spaces and collapse multiple spaces
        return ' '.join(value.split())
    return value

def extract_samples(input_file, output_file, frequency, total_samples, sanitize=False):
    """
    Extract samples using two-stage sampling: frequency filter then even distribution.

    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
        frequency (int): First-stage filter - extract every Nth row
        total_samples (int): Second-stage - number of samples from filtered set (excluding header)
        sanitize (bool): Remove embedded newlines from fields
    """
    df = pd.read_csv(input_file, low_memory=False)
    total_rows = len(df)

    # Stage 1: Filter by frequency
    freq_indices = list(range(0, total_rows, frequency))
    freq_filtered = df.iloc[freq_indices]
    print(f"Stage 1: Filtered {total_rows} rows down to {len(freq_filtered)} using frequency {frequency}")

    # Stage 2: Extract evenly distributed samples from filtered set
    filtered_count = len(freq_filtered)
    if total_samples >= filtered_count:
        print(f"Warning: Requested {total_samples} samples but filtered set only has {filtered_count} rows")
        freq_filtered.to_csv(output_file, index=False)
        print(f"Extracted all {filtered_count} rows")
        return

    # Calculate interval for even distribution
    interval = max(1, filtered_count // total_samples)

    # Get evenly spaced samples
    sample_indices = list(range(0, filtered_count, interval))[:total_samples]
    final_samples = freq_filtered.iloc[sample_indices].copy()

    if sanitize:
        print("Sanitizing sampled data (removing embedded newlines)...")
        for col in final_samples.columns:
            if final_samples[col].dtype == 'object':
                final_samples[col] = final_samples[col].apply(sanitize_field)

    final_samples.to_csv(output_file, index=False, lineterminator='\n')
    print(f"Stage 2: Extracted {len(final_samples)} rows with interval {interval}")
    print(f"Final output: {len(final_samples)} rows written to {output_file}")

def extract_stratified_samples(input_file, output_file, split_counts, label_column='label', sanitize=False):
    """
    Extract stratified samples with exact counts per class.

    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
        split_counts (list): [normal_count, malicious_count, suspicious_count]
        label_column (str): Name of the label column in CSV
        sanitize (bool): Remove embedded newlines from fields
    """
    print("Loading dataset...")
    df = pd.read_csv(input_file, low_memory=False)

    # Check if label column exists
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found. Available columns: {df.columns.tolist()}")

    normal_count, malicious_count, suspicious_count = split_counts

    # Get label distribution
    label_dist = df[label_column].value_counts()
    print(f"\n=== Dataset Label Distribution ===")
    print(label_dist)
    print(f"Total rows: {len(df)}")
    print(f"\n=== Requested Stratified Sample ===")
    print(f"Normal: {normal_count}")
    print(f"Malicious: {malicious_count}")
    print(f"Suspicious: {suspicious_count}")
    print(f"Total requested: {sum(split_counts)}")

    # Sample from each class
    print("\n=== Sampling ===")
    samples = []
    labels_map = {
        'normal': normal_count,
        'malicious': malicious_count,
        'suspicious': suspicious_count
    }

    for label, count in labels_map.items():
        # Skip if count is 0
        if count == 0:
            print(f"⊘ Skipping '{label}' (count = 0)")
            continue
            
        label_df = df[df[label_column] == label].reset_index(drop=True)
        available = len(label_df)

        if available == 0:
            print(f"⚠️  WARNING: No '{label}' samples in dataset!")
            continue

        if available < count:
            print(f"⚠️  WARNING: Not enough '{label}' samples!")
            print(f"   Requested: {count}, Available: {available}")
            print(f"   Using all {available} available samples")
            sampled = label_df.copy()
        else:
            # Sample evenly across the available data
            step = available / count
            indices = [int(i * step) for i in range(count)]
            sampled = label_df.iloc[indices].copy()

            # Verify count
            assert len(sampled) == count, f"Sampling error: got {len(sampled)} instead of {count}"
            print(f"✓ Sampled {len(sampled)} '{label}' rows from {available} available")

        samples.append(sampled)

    # Check if we got any samples
    if not samples:
        print("\n❌ ERROR: No samples were collected!")
        sys.exit(1)

    # Combine and shuffle
    print("\nCombining and shuffling samples...")
    result = pd.concat(samples, ignore_index=True)

    # Verify total count matches non-zero requested counts
    expected_total = sum(split_counts)
    actual_total = len(result)
    
    if actual_total != expected_total:
        print(f"⚠️  Note: Collected {actual_total} samples (requested {expected_total})")
        print(f"   This may differ if some classes were unavailable or skipped (count=0)")

    result = result.sample(frac=1, random_state=42).reset_index(drop=True)

    # Sanitize ONLY the sampled data (not the original 12M rows!)
    if sanitize:
        print("\n=== Sanitizing Sampled Data ===")
        print(f"Scanning {len(result)} sampled rows for embedded newlines...")
        newline_count = 0
        for col in result.columns:
            if result[col].dtype == 'object':
                # Count newlines in SAMPLE only
                has_newlines = result[col].astype(str).str.contains('\n|\r', regex=True)
                col_newlines = has_newlines.sum()
                if col_newlines > 0:
                    print(f"  Column '{col}': {col_newlines} fields contain newlines")
                    newline_count += col_newlines
                result[col] = result[col].apply(sanitize_field)

        if newline_count > 0:
            print(f"✓ Sanitized {newline_count} fields")
        else:
            print("✓ No embedded newlines found")

    # Save
    print(f"\nWriting to {output_file}...")
    result.to_csv(output_file, index=False, lineterminator='\n')

    # Final verification
    print(f"\n=== Final Output ===")
    print(f"Total rows written: {len(result)}")
    print(f"Label distribution in output:")
    print(result[label_column].value_counts())

    # Verify the file
    verify_df = pd.read_csv(output_file, low_memory=False)

    # Count actual lines
    with open(output_file, 'r') as f:
        line_count = sum(1 for _ in f)

    print(f"\n=== Verification ===")
    print(f"Pandas reads: {len(verify_df)} data rows")
    print(f"File has: {line_count} lines (wc -l count)")

    if len(verify_df) != actual_total:
        print(f"❌ ERROR: Expected {actual_total} rows but pandas reads {len(verify_df)}")
        return False

    if line_count != actual_total + 1:  # +1 for header
        print(f"⚠️  WARNING: wc -l shows {line_count} but expected {actual_total + 1}")
        print(f"   Difference: {line_count - (actual_total + 1)} extra lines")
        if not sanitize:
            print(f"   → Likely embedded newlines. Try running with --sanitize")
        return False

    print(f"✓ Verification passed: {actual_total} data rows + 1 header = {line_count} lines")
    return True

def main():
    parser = argparse.ArgumentParser(
        description='Extract samples from a CSV file using two-stage or stratified sampling.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stratified sampling: Extract 400 normal, 400 malicious, 200 suspicious
  python gather_samples.py --input data.csv --output samples.csv --split 400 400 200

  # Only suspicious samples (skip normal and malicious)
  python gather_samples.py --input data.csv --output samples.csv --split 0 0 1000

  # With sanitization (removes embedded newlines for clean CSV)
  python gather_samples.py --input data.csv --output samples.csv --split 400 400 200 --sanitize

  # Two-stage sampling
  python gather_samples.py --input data.csv --output samples.csv --freq 1000 --total_samples 100
        """
    )

    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output', required=True, help='Output CSV file path')

    # Two-stage sampling arguments
    parser.add_argument('--freq', type=int,
                       help='Stage 1: Extract every Nth row')
    parser.add_argument('--total_samples', type=int,
                       help='Stage 2: Number of samples to extract')

    # Stratified sampling arguments
    parser.add_argument('--split', type=int, nargs=3, metavar=('NORMAL', 'MALICIOUS', 'SUSPICIOUS'),
                       help='Stratified sampling: exact counts per class (e.g., 400 400 200). Use 0 to skip a class.')
    parser.add_argument('--label_col', type=str, default='label',
                       help='Label column name (default: label)')

    # Common arguments
    parser.add_argument('--sanitize', action='store_true',
                       help='Remove embedded newlines (recommended for analyst review)')

    args = parser.parse_args()

    # Validate arguments
    if args.split and (args.freq or args.total_samples):
        print("Error: Cannot use --split with --freq/--total_samples", file=sys.stderr)
        sys.exit(1)

    if not args.split and not (args.freq and args.total_samples):
        print("Error: Must specify either --split OR both --freq and --total_samples", file=sys.stderr)
        sys.exit(1)

    try:
        if args.split:
            success = extract_stratified_samples(args.input, args.output, args.split,
                                                args.label_col, args.sanitize)
            sys.exit(0 if success else 1)
        else:
            extract_samples(args.input, args.output, args.freq, args.total_samples, args.sanitize)

    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
