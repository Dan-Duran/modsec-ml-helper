#!/usr/bin/env python3
"""
parser.py - ModSecurity log parser using severity-based labeling strategy.
Extracts request data and labels based on ModSec's own severity ratings.

Usage:
    python parser.py \
        --input modsec-logs/modsec.log* \
        --output datasets/modsec_dataset.csv

Author: Danilo A. Duran
Institution: Georgia Institute of Technology

"""

import argparse
import re
import pandas as pd
import hashlib
import json
from urllib.parse import urlparse

def anonymize_value(value, salt="research2024"):
    """Anonymize sensitive data while maintaining consistency."""
    if not value:
        return None
    return hashlib.md5(f"{value}{salt}".encode()).hexdigest()[:12]

def sanitize_uri(uri):
    """Remove domain information from URIs, keep only path and params."""
    if not uri or not uri.startswith('http'):
        return uri
    try:
        parsed = urlparse(uri)
        return parsed.path + ('?' + parsed.query if parsed.query else '')
    except ValueError:
        return uri

def record_generator(filepath):
    """Reads a ModSecurity log file and yields one full record at a time."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            record_lines = []
            in_record = False
            for line in f:
                if re.match(r'--[^-]+-A--', line):
                    if in_record and record_lines:
                        yield "".join(record_lines)
                    record_lines = [line]
                    in_record = True
                elif in_record:
                    record_lines.append(line)
                    if re.match(r'--[^-]+-Z--', line):
                        yield "".join(record_lines)
                        record_lines = []
                        in_record = False
            if record_lines:
                yield "".join(record_lines)
    except FileNotFoundError:
        print(f"Error: Log file not found at {filepath}")
        return

def parse_record(record_text):
    """Parse ModSec record and extract all relevant data."""
    data = {
        'method': None, 'uri': None, 'uri_path': None, 'uri_query': None,
        'payload': None, 'status_code': None, 'user_agent': None,
        'content_type': None, 'content_length': 0, 'host': None,
        'referer': None, 'timestamp': None, 'transaction_id': None,
        'source_ip': None, 'source_port': None, 'dest_ip': None,
        'dest_port': None, 'triggered_rule_ids': [], 'rule_severities': [],
        'rule_tags': [], 'anomaly_score': 0, 'sql_injection_score': 0,
        'xss_score': 0, 'rce_score': 0, 'lfi_rfi_score': 0,
        'session_fixation_score': 0, 'request_anomalies': 0
    }
    
    # Regex to handle all transaction ID characters ---
    tx_match = re.search(r'--([^-]+)-A--', record_text)
    if tx_match:
        data['transaction_id'] = tx_match.group(1)
    
    part_a_match = re.search(r'--[^-]+-A--\n(.*?)\n--[^-]+-', record_text, re.DOTALL)
    if part_a_match:
        part_a = part_a_match.group(1)
        time_match = re.search(r'\[([^\]]+)\]', part_a)
        if time_match:
            data['timestamp'] = time_match.group(1)
        ip_match = re.search(r'(\d+\.\d+\.\d+\.\d+)\s+(\d+)\s+(\d+\.\d+\.\d+\.\d+)\s+(\d+)', part_a)
        if ip_match:
            data['source_ip'], data['source_port'], data['dest_ip'], data['dest_port'] = ip_match.groups()
            data['source_port'] = int(data['source_port'])
            data['dest_port'] = int(data['dest_port'])
    
    part_b_match = re.search(r'--[^-]+-B--\n(.*?)\n--[^-]+-', record_text, re.DOTALL)
    if part_b_match:
        part_b_content = part_b_match.group(1).strip().split('\n')
        if part_b_content:
            request_line = part_b_content[0]
            parts = request_line.split()
            if len(parts) >= 2:
                data['method'] = parts[0]
                raw_uri = parts[1]
                data['uri'] = sanitize_uri(raw_uri)
                if '?' in data['uri']:
                    data['uri_path'], data['uri_query'] = data['uri'].split('?', 1)
                else:
                    data['uri_path'] = data['uri']
            
            headers = {}
            for header_line in part_b_content[1:]:
                if ':' in header_line:
                    key, value = header_line.split(':', 1)
                    headers[key.strip().lower()] = value.strip()
            
            data['user_agent'] = headers.get('user-agent')
            data['content_type'] = headers.get('content-type')
            data['host'] = headers.get('host')
            data['referer'] = sanitize_uri(headers.get('referer'))
            data['content_length'] = int(headers.get('content-length', 0))
    
    part_c_match = re.search(r'--[^-]+-C--\n(.*?)\n--[^-]+-', record_text, re.DOTALL)
    if part_c_match:
        data['payload'] = part_c_match.group(1).strip()
    
    part_f_match = re.search(r'--[^-]+-F--\nHTTP/[\d\.]+ (\d+)', record_text)
    if part_f_match:
        data['status_code'] = int(part_f_match.group(1))
    
    part_h_match = re.search(r'--[^-]+-H--\n(.*?)(?:\n--[^-]+-|$)', record_text, re.DOTALL)
    if part_h_match:
        part_h = part_h_match.group(1)
        
        for line in part_h.split('\n'):
            if 'Message:' in line:
                rule_match = re.search(r'\[id "(\d+)"\]', line)
                if rule_match:
                    rule_id = rule_match.group(1)
                    if not rule_id.startswith('90'):
                        data['triggered_rule_ids'].append(rule_id)
                
                sev_match = re.search(r'\[severity "([^"]+)"\]', line)
                if sev_match:
                    data['rule_severities'].append(sev_match.group(1).upper()) # Standardize severity
                
                data['rule_tags'].extend(re.findall(r'\[tag "([^"]+)"\]', line))

        # Regex to find the correct anomaly score format ---
        anomaly_match = re.search(r'Inbound Anomaly Score: (\d+)', part_h)
        if anomaly_match:
            data['anomaly_score'] = int(anomaly_match.group(1))

    return data

def assign_label(record):
    """
    Label based on ModSecurity's severity ratings and attack tool signatures.
    """
    attack_tools = [
        'hydra', 'sqlmap', 'nikto', 'gobuster', 'openvas',
        'burp', 'zap', 'ffuf', 'nmap', 'masscan', 'dirbuster',
        'wfuzz', 'metasploit', 'nuclei', 'acunetix'
    ]
    
    user_agent = str(record.get('user_agent', '')).lower()
    if any(tool in user_agent for tool in attack_tools):
        return 'malicious'
    
    rule_severities = record.get('rule_severities', [])
    
    if 'CRITICAL' in rule_severities:
        return 'malicious'
    
    if 'ERROR' in rule_severities:
        return 'suspicious'
    
    if record.get('anomaly_score', 0) >= 20:
        return 'suspicious'
    
    if 'WARNING' in rule_severities and record.get('anomaly_score', 0) >= 10:
        return 'suspicious'
    
    return 'normal'

def main(args):
    """Parse ModSec logs using severity-based labeling."""
    print(f"Reading ModSec records from {len(args.input)} file(s): {', '.join(args.input)}...")
    print("Labeling strategy:")
    print("  - Attack tools → malicious")
    print("  - CRITICAL severity → malicious")
    print("  - ERROR severity → suspicious")
    print("  - High anomaly scores → suspicious")
    print("  - Everything else → normal\n")
    
    parsed_records = []
    record_count = 0
    
    print("Parsing records", end='')
    
    for input_file in args.input:
        print(f"\nProcessing {input_file}...", end='')
        log_gen = record_generator(input_file)
        
        for record_text in log_gen:
            parsed_data = parse_record(record_text)
            
            if parsed_data['source_ip']:
                parsed_data['source_ip'] = anonymize_value(parsed_data['source_ip'])
            if parsed_data['dest_ip']:
                parsed_data['dest_ip'] = anonymize_value(parsed_data['dest_ip'])
            if parsed_data['host']:
                parsed_data['host'] = anonymize_value(parsed_data['host'])
            
            for field in ['triggered_rule_ids', 'rule_severities', 'rule_tags']:
                if parsed_data.get(field):
                    parsed_data[field] = json.dumps(parsed_data[field])
                else:
                    parsed_data[field] = '[]'
            
            parsed_data['label'] = assign_label(parsed_data)
            
            parsed_records.append(parsed_data)
            
            record_count += 1
            if record_count % 10000 == 0:
                print('.', end='', flush=True)
            
            if args.limit and record_count >= args.limit:
                break
        
        if args.limit and record_count >= args.limit:
            break
    
    if not parsed_records:
        print("\nNo records were parsed from the input file.")
        return
    
    print(f"\n\nSuccessfully parsed {len(parsed_records)} records")
    
    df = pd.DataFrame(parsed_records)
    df.drop_duplicates(subset=['transaction_id'], keep='first', inplace=True)
    
    print("\n=== PARSING STATISTICS ===")
    print(f"Total records: {len(df)}")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    print(f"\nRecords with triggered rules: {(df['triggered_rule_ids'] != '[]').sum()}")
    print(f"Records with CRITICAL severity: {df['rule_severities'].str.contains('CRITICAL').sum()}")
    print(f"Records with ERROR severity: {df['rule_severities'].str.contains('ERROR').sum()}")
    
    df.to_csv(args.output, index=False)
    print(f"\nSaved dataset to '{args.output}'")
    
    if args.sample:
        sample_df = df.head(100)
        sample_file = args.output.replace('.csv', '_sample.csv')
        sample_df.to_csv(sample_file, index=False)
        print(f"Saved sample (first 100 records) to '{sample_file}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ModSecurity log parser with severity-based labeling"
    )
    parser.add_argument('--input', type=str, nargs='+', required=True,
                        help='Input ModSecurity audit log files')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV file path')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of records to parse (for testing)')
    parser.add_argument('--sample', action='store_true',
                        help='Also save a sample CSV with first 100 records')
    
    args = parser.parse_args()
    main(args)
