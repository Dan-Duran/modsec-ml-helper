# parser_expert_opinion.py - DEPRECATED

import argparse
import re
import pandas as pd
import json
from urllib.parse import unquote

def record_generator(filepath):
    """Reads a ModSecurity log file and yields one full record at a time."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            record_lines = []
            in_record = False
            for line in f:
                if re.match(r'--[a-f0-9]+-A--', line):
                    if in_record and record_lines:
                        yield "".join(record_lines)
                    record_lines = [line]
                    in_record = True
                elif in_record:
                    record_lines.append(line)
                    if re.match(r'--[a-f0-9]+-Z--', line):
                        yield "".join(record_lines)
                        record_lines = []
                        in_record = False
            if record_lines:
                yield "".join(record_lines)
    except FileNotFoundError:
        print(f"Error: Log file not found at {filepath}")
        return

def parse_record(record_text):
    """
    Parses the request and Part H to calculate scores and extract rule metadata.
    """
    data = {
        'transaction_id': None, 'uri': None, 'payload': None, 'anomaly_score': 0, 
        'sql_injection_score': 0, 'xss_score': 0, 'rce_score': 0, 'lfi_rfi_score': 0, 
        'session_fixation_score': 0, 'triggered_rule_ids': [], 'rule_severities': [], 'rule_tags': []
    }
    
    tx_match = re.search(r'--([a-f0-9]+)-A--', record_text)
    if tx_match:
        data['transaction_id'] = tx_match.group(1)

    all_headers = "" # <-- NEW: Variable to hold all header values

    part_b_match = re.search(r'--[a-f0-9]+-B--\n(.*?)\n--[a-f0-9]+-', record_text, re.DOTALL)
    if part_b_match:
        part_b_content = part_b_match.group(1).strip().split('\n')
        if part_b_content:
            request_line = part_b_content[0]
            parts = request_line.split()
            if len(parts) >= 2:
                data['uri'] = parts[1]
            for header_line in part_b_content[1:]:
                if ':' in header_line:
                    _, value = header_line.split(':', 1)
                    all_headers += " " + value.strip() # <-- NEW: Append header value

    part_c_match = re.search(r'--[a-f0-9]+-C--\n(.*?)\n--[a-f0-9]+-', record_text, re.DOTALL)
    if part_c_match:
        data['payload'] = part_c_match.group(1).strip()

    part_h_match = re.search(r'--[a-f0-9]+-H--\n(.*?)(?:\n--[a-f0-9]+-|$)', record_text, re.DOTALL)
    if part_h_match:
        part_h = part_h_match.group(1)
        for line in part_h.split('\n'):
            if 'Message:' in line:
                rule_match = re.search(r'\[id "(\d+)"\]', line)
                if rule_match and not rule_match.group(1).startswith('90'):
                    data['triggered_rule_ids'].append(rule_match.group(1))
                sev_match = re.search(r'\[severity "([^"]+)"\]', line)
                if sev_match: data['rule_severities'].append(sev_match.group(1))
                data['rule_tags'].extend(re.findall(r'\[tag "([^"]+)"\]', line))

    # --- SCORE CALCULATION LOGIC ---
    # <-- MODIFIED --> Combine URI, payload, AND all headers
    combined_data = unquote(str(data['uri'] or '') + " " + str(data['payload'] or '') + " " + all_headers)
    
    score_increment = 5
    
    sql_patterns = [r"(\b(union|select|insert|update|delete|drop|exec|sleep|waitfor)\b)", r"(\'|\"|;|--|\*|/\*|\*/)", r"(\bor\b|\band\b).?="]
    if any(re.search(p, combined_data, re.IGNORECASE) for p in sql_patterns):
        data['sql_injection_score'] += score_increment

    xss_patterns = [r"<script", r"javascript:", r"on\w+\s*=", r"<iframe", r"<embed", r"<object", r"data:text/html"]
    if any(re.search(p, combined_data, re.IGNORECASE) for p in xss_patterns):
        data['xss_score'] += score_increment

    cmd_patterns = [r"(\||;|&|`|\$\()", r"\b(cat|ls|wget|curl|bash|sh|cmd|whoami|powershell|\?)\b"]
    if any(re.search(p, combined_data, re.IGNORECASE) for p in cmd_patterns):
        data['rce_score'] += score_increment

    traversal_patterns = [r"\.\./", r"\.\.%2[fF]", r"\%2e\%2e", r"/etc/passwd", r"boot\.ini", r"win\.ini", r"%c0%af"]
    if any(re.search(p, combined_data, re.IGNORECASE) for p in traversal_patterns):
        data['lfi_rfi_score'] += score_increment
    
    # <-- NEW --> Add specific patterns for Log4j and SSTI
    if re.search(r'\$\{jndi:(ldap|dns|rmi):', combined_data, re.IGNORECASE):
        data['rce_score'] += score_increment * 2
    if re.search(r'\{\{.*\}\}|__class__|__globals__', combined_data, re.IGNORECASE):
        data['rce_score'] += score_increment
    if 'php://filter' in combined_data:
        data['lfi_rfi_score'] += score_increment

    data['anomaly_score'] = sum([
        data['sql_injection_score'], data['xss_score'],
        data['rce_score'], data['lfi_rfi_score']
    ])
    
    return data

def main(args):
    """Parse ModSec logs to create the expert context file by calculating scores."""
    print(f"Extracting ModSecurity expert context from {len(args.input)} file(s)...")
    print("This parser calculates scores from request content and extracts rule metadata.\n")
    
    parsed_records = []
    record_count = 0
    
    for input_file in args.input:
        print(f"\nProcessing {input_file}...", end='')
        for record_text in record_generator(input_file):
            parsed_data = parse_record(record_text)
            
            # Convert lists to JSON strings for CSV storage
            for field in ['triggered_rule_ids', 'rule_severities', 'rule_tags']:
                if parsed_data[field]:
                    unique_items = list(set(parsed_data[field]))
                    parsed_data[field] = json.dumps(unique_items)
                else:
                    parsed_data[field] = None
            
            parsed_records.append(parsed_data)
            
            record_count += 1
            if record_count % 10000 == 0:
                print('.', end='', flush=True)

            if args.limit and record_count >= args.limit:
                break
        if args.limit and record_count >= args.limit:
            break
    
    print(f"\n\nSuccessfully parsed {len(parsed_records)} records")
    
    df = pd.DataFrame(parsed_records)
    df.drop_duplicates(subset=['transaction_id'], keep='first', inplace=True)

    
    final_columns = [
        'transaction_id', 'anomaly_score', 'sql_injection_score', 'xss_score',
        'rce_score', 'lfi_rfi_score', 'session_fixation_score',
        'triggered_rule_ids', 'rule_severities', 'rule_tags'
    ]
    df = df[final_columns]
    
    print("\n=== MODSECURITY CONTEXT STATISTICS ===")
    print(f"Total records: {len(df)}")
    print(f"Records with triggered rules: {df['triggered_rule_ids'].notna().sum()}")
    print(f"Records with anomaly score > 0: {(df['anomaly_score'] > 0).sum()}")
    print(f"Records with SQL injection score > 0: {(df['sql_injection_score'] > 0).sum()}")
    print(f"Records with XSS score > 0: {(df['xss_score'] > 0).sum()}")
    print(f"Records with RCE score > 0: {(df['rce_score'] > 0).sum()}")
    print(f"Records with LFI/RFI score > 0: {(df['lfi_rfi_score'] > 0).sum()}")
    
    df.to_csv(args.output, index=False)
    print(f"\nSaved ModSecurity context to '{args.output}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Streamlined ModSecurity context extractor."
    )
    parser.add_argument('--input', type=str, nargs='+', required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--limit', type=int)
    
    args = parser.parse_args()
    main(args)
