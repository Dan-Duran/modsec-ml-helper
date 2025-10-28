#!/usr/bin/env python3
"""
filter_modsecurity.py - ModSecurity Log Filter with Deduplication
EXACT MATCH to parser.py for ML testing (including deduplication)

Reads a raw ModSecurity audit log and filters records by classification:
- MALICIOUS: Attack tools in User-Agent OR CRITICAL severity
- SUSPICIOUS: ERROR severity OR (WARNING + anomaly >= 10) OR anomaly >= 20
- NORMAL: Everything else

DEDUPLICATION: Removes duplicate transaction IDs (keeps first occurrence)
"""

import argparse
import logging
import re
import sys
from pathlib import Path

# ================== LOGGING ==================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ================== CORE FUNCTIONS ==================

def record_generator(filepath: Path):
    """
    Reads a ModSecurity log file and yields one full record at a time.
    This function is robust to different transaction ID formats.
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            record_lines = []
            in_record = False
            for line in f:
                # Match the start of a record (e.g., --xxxxxxxx-A--)
                if re.match(r'--[^-]+-A--', line):
                    if in_record and record_lines:
                        yield "".join(record_lines)
                    record_lines = [line]
                    in_record = True
                elif in_record:
                    record_lines.append(line)
                    # Match the end of a record (e.g., --xxxxxxxx-Z--)
                    if re.match(r'--[^-]+-Z--', line):
                        yield "".join(record_lines)
                        record_lines = []
                        in_record = False
            if record_lines:  # Yield the last record if the file doesn't end with -Z--
                yield "".join(record_lines)
    except FileNotFoundError:
        logging.error(f"Error: Log file not found at {filepath}")
        return

def extract_transaction_id(record_text: str) -> str:
    """Extract transaction ID from record (matches parser.py)."""
    tx_match = re.search(r'--([^-]+)-A--', record_text)
    if tx_match:
        return tx_match.group(1)
    return None

def extract_user_agent(record_text: str) -> str:
    """
    Extract User-Agent header EXACTLY as parser.py does.
    """
    part_b_match = re.search(r'--[^-]+-B--\n(.*?)\n--[^-]+-', record_text, re.DOTALL)
    if not part_b_match:
        return ""
    
    part_b_content = part_b_match.group(1).strip().split('\n')
    if not part_b_content:
        return ""
    
    # Skip the request line, process headers
    for header_line in part_b_content[1:]:
        if ':' in header_line:
            key, value = header_line.split(':', 1)
            if key.strip().lower() == 'user-agent':
                return value.strip().lower()
    
    return ""

def classify_record(record_text: str) -> str:
    """
    EXACT MATCH to parser.py classification logic
    
    Classification rules (EXACTLY matching parser.py):
    1. Attack tools in User-Agent ONLY → malicious
    2. CRITICAL severity → malicious
    3. ERROR severity → suspicious
    4. Anomaly score >= 20 → suspicious
    5. WARNING + anomaly score >= 10 → suspicious
    6. Everything else → normal
    """
    # === Step 1: Extract User-Agent EXACTLY as parser.py does ===
    user_agent = extract_user_agent(record_text)
    
    # Attack tool list MUST match parser.py exactly
    attack_tools = [
        'hydra', 'sqlmap', 'nikto', 'gobuster', 'openvas',
        'burp', 'zap', 'ffuf', 'nmap', 'masscan', 'dirbuster',
        'wfuzz', 'metasploit', 'nuclei', 'acunetix'
    ]
    
    # Check ONLY User-Agent for attack tools (matches parser.py)
    if user_agent and any(tool in user_agent for tool in attack_tools):
        return "malicious"
    
    # === Step 2: Find H-block for severity and anomaly score ===
    part_h_match = re.search(r'--[^-]+-H--\n(.*?)(?:\n--[^-]+-|$)', record_text, re.DOTALL)
    if not part_h_match:
        return "normal"  # No H-block means no alerts

    h_block_content = part_h_match.group(1)

    # === Step 3: Check severity levels ===
    has_critical = 'severity "CRITICAL"' in h_block_content
    has_error = 'severity "ERROR"' in h_block_content
    has_warning = 'severity "WARNING"' in h_block_content

    # === Step 4: Extract anomaly score ===
    anomaly_score = 0
    anomaly_match = re.search(r'Inbound Anomaly Score: (\d+)', h_block_content)
    if anomaly_match:
        anomaly_score = int(anomaly_match.group(1))

    # === Classification logic (EXACT match to parser.py) ===
    
    # CRITICAL → malicious
    if has_critical:
        return "malicious"
    
    # ERROR → suspicious
    if has_error:
        return "suspicious"
    
    # Anomaly score >= 20 → suspicious
    if anomaly_score >= 20:
        return "suspicious"
    
    # WARNING + anomaly >= 10 → suspicious
    if has_warning and anomaly_score >= 10:
        return "suspicious"
    
    # Everything else → normal
    return "normal"

# ================== MAIN EXECUTION ==================

def main():
    parser = argparse.ArgumentParser(
        description="Filter a ModSecurity audit log by classification type (EXACT match to parser.py with deduplication).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input', type=Path, required=True,
                        help='Path to the input ModSecurity audit log file.')
    parser.add_argument('--output', type=Path, required=False,
                        help='Path to the output file for filtered logs. Not required for --run-dry mode.')
    parser.add_argument('--filter', type=str, required=False,
                        choices=['suspicious', 'malicious', 'normal', 'all'],
                        help='Type of records to filter: suspicious, malicious, normal, or all (keeps everything).')
    parser.add_argument('--split', action='store_true',
                        help='Split output into three separate files (one for each classification). '
                             'The --output value will be used as a prefix.')
    parser.add_argument('--run-dry', action='store_true',
                        help='Dry run mode: count records by classification without writing any output files.')
    parser.add_argument('--no-dedup', action='store_true',
                        help='Disable deduplication (keep all records including duplicates).')
    args = parser.parse_args()

    # Validation: if not dry-run, require output and filter
    if not args.run_dry:
        if not args.output:
            parser.error("--output is required when not using --run-dry")
        if not args.filter:
            parser.error("--filter is required when not using --run-dry")

    if not args.input.exists():
        logging.error(f"Input file not found: {args.input}")
        sys.exit(1)

    if not args.run_dry:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Reading from: {args.input}")
    logging.info("Classification logic: EXACT MATCH to parser.py for ML testing")
    logging.info("  → Attack tools checked in User-Agent ONLY (not other headers)")
    
    if args.no_dedup:
        logging.info("  → Deduplication: DISABLED (keeping all records)")
    else:
        logging.info("  → Deduplication: ENABLED (matches parser.py)")
    
    if args.run_dry:
        logging.info("DRY RUN MODE: No files will be written. Counting records only.")
    
    # Counter for statistics
    stats = {
        'total': 0,
        'duplicates_removed': 0,
        'malicious': 0,
        'suspicious': 0,
        'normal': 0
    }
    
    # Track seen transaction IDs for deduplication
    seen_tx_ids = set()

    try:
        if args.run_dry:
            # Dry run: just count without writing
            for record_text in record_generator(args.input):
                stats['total'] += 1
                
                # Deduplication check (matches parser.py)
                if not args.no_dedup:
                    tx_id = extract_transaction_id(record_text)
                    if tx_id:
                        if tx_id in seen_tx_ids:
                            stats['duplicates_removed'] += 1
                            continue  # Skip duplicate
                        seen_tx_ids.add(tx_id)
                
                classification = classify_record(record_text)
                stats[classification] += 1
                
                if stats['total'] % 50000 == 0:
                    logging.info(f"Processed {stats['total']:,} records...")
        
        elif args.split:
            # Split mode: create three separate files
            output_files = {
                'malicious': open(args.output.parent / f"{args.output.stem}_malicious{args.output.suffix}", 
                                'w', encoding='utf-8'),
                'suspicious': open(args.output.parent / f"{args.output.stem}_suspicious{args.output.suffix}", 
                                 'w', encoding='utf-8'),
                'normal': open(args.output.parent / f"{args.output.stem}_normal{args.output.suffix}", 
                             'w', encoding='utf-8')
            }
            logging.info(f"Split mode: Writing to {args.output.parent / args.output.stem}_[TYPE]{args.output.suffix}")
            
            for record_text in record_generator(args.input):
                stats['total'] += 1
                
                # Deduplication check
                if not args.no_dedup:
                    tx_id = extract_transaction_id(record_text)
                    if tx_id:
                        if tx_id in seen_tx_ids:
                            stats['duplicates_removed'] += 1
                            continue
                        seen_tx_ids.add(tx_id)
                
                classification = classify_record(record_text)
                stats[classification] += 1
                output_files[classification].write(record_text + '\n')
                
                if stats['total'] % 50000 == 0:
                    logging.info(f"Processed {stats['total']:,} records...")
            
            for f in output_files.values():
                f.close()
                
        else:
            # Single filter mode
            logging.info(f"Filtering for: {args.filter}")
            logging.info(f"Writing to: {args.output}")
            
            with open(args.output, 'w', encoding='utf-8') as outfile:
                for record_text in record_generator(args.input):
                    stats['total'] += 1
                    
                    # Deduplication check
                    if not args.no_dedup:
                        tx_id = extract_transaction_id(record_text)
                        if tx_id:
                            if tx_id in seen_tx_ids:
                                stats['duplicates_removed'] += 1
                                continue
                            seen_tx_ids.add(tx_id)
                    
                    classification = classify_record(record_text)
                    stats[classification] += 1
                    
                    if args.filter == 'all' or classification == args.filter:
                        outfile.write(record_text + '\n')
                    
                    if stats['total'] % 50000 == 0:
                        logging.info(f"Processed {stats['total']:,} records...")

    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)

    # Calculate unique records
    unique_records = stats['total'] - stats['duplicates_removed']

    # Print statistics
    logging.info("=" * 60)
    if args.run_dry:
        logging.info("Dry Run Complete - Record Count Summary")
    else:
        logging.info("Filtering Complete")
    logging.info(f"Total records processed: {stats['total']:,}")
    
    if not args.no_dedup:
        logging.info(f"Duplicates removed: {stats['duplicates_removed']:,}")
        logging.info(f"Unique records: {unique_records:,}")
    
    logging.info(f"\nClassification (after deduplication):")
    logging.info(f"  - Malicious: {stats['malicious']:,} ({stats['malicious']/unique_records*100:.1f}%)")
    logging.info(f"  - Suspicious: {stats['suspicious']:,} ({stats['suspicious']/unique_records*100:.1f}%)")
    logging.info(f"  - Normal: {stats['normal']:,} ({stats['normal']/unique_records*100:.1f}%)")
    
    if not args.run_dry:
        if args.split:
            logging.info(f"\nOutput files created at: {args.output.parent / args.output.stem}_[TYPE]{args.output.suffix}")
        else:
            if args.filter == 'all':
                written = unique_records
            else:
                written = stats[args.filter]
            logging.info(f"\nRecords written to output: {written:,}")
            logging.info(f"Output file is ready at: {args.output}")
    
    logging.info("\n✓ Classification logic EXACTLY matches parser.py:")
    logging.info("  ✓ Attack tools checked in User-Agent header ONLY")
    logging.info("  ✓ CRITICAL severity → malicious")
    logging.info("  ✓ ERROR severity → suspicious")
    logging.info("  ✓ Anomaly score >= 20 → suspicious")
    logging.info("  ✓ WARNING + anomaly >= 10 → suspicious")
    if not args.no_dedup:
        logging.info("  ✓ Deduplication enabled (keeps first occurrence)")
    logging.info("=" * 60)

if __name__ == "__main__":
    main()
