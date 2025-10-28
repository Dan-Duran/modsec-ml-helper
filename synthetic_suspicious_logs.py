#!/usr/bin/env python3
"""
Synthetic Suspicious Apache Log Generator v3 - Enhanced Variety
Uses SecLists wordlists with improved variety for 1M+ log generation
"""

import argparse
import gzip
import random
import string
import sys
from datetime import datetime, timedelta
from pathlib import Path
from faker import Faker
from ipaddress import IPv4Address
from random import getrandbits

class SuspiciousLogGenerator:
    def __init__(self, seclists_path='/home/dan/Projects/cs6727/NEW/SecLists'):
        self.faker = Faker()
        self.seclists_base = Path(seclists_path)

        # Load all wordlists
        print("Loading wordlists from SecLists...")
        self.load_wordlists()
        print(f"  Loaded {len(self.path_sources)} path source categories")
        print(f"  Loaded {len(self.bot_user_agents)} bot user agents")
        print(f"  Loaded {len(self.parameter_names)} parameter names")
        print(f"  Loaded {len(self.file_extensions)} file extensions")
        print()

        # Initialize variety enhancers
        self.init_variety_patterns()

        # Suspicious referers
        self.suspicious_referers = [
            '-',
            'http://localhost/',
            'http://127.0.0.1/',
            'file://',
            'about:blank',
            'data:text/html',
        ]

        # Methods with unusual distributions for suspicious traffic
        self.method_weights = {
            'GET': 0.60,
            'POST': 0.20,
            'HEAD': 0.10,
            'OPTIONS': 0.05,
            'PUT': 0.03,
            'DELETE': 0.02
        }

        # Response codes weighted for suspicious traffic
        self.response_codes = ['200', '301', '302', '400', '401', '403', '404', '405', '500']
        self.response_weights = [0.20, 0.05, 0.05, 0.15, 0.15, 0.15, 0.15, 0.05, 0.05]

        # IP pool for realistic scanner behavior
        self._recent_ips = []
        self._burst_mode = False
        self._burst_counter = 0

    def init_variety_patterns(self):
        """Initialize patterns for variety enhancement"""
        self.sql_keywords = [
            'SELECT', 'UNION', 'DROP', 'INSERT', 'UPDATE', 'DELETE',
            'TABLE', 'FROM', 'WHERE', 'ORDER', 'BY', 'AND', 'OR'
        ]
        
        self.path_traversal_variants = [
            '../', '..\\', '..../', '..\\..\\',
            '%2e%2e/', '%2e%2e%5c', '..%2f', '..%5c',
            '....%2f', '....%5c', '%252e%252e%252f', '%c0%ae%c0%ae/',
        ]
        
        self.common_files = [
            'etc/passwd', 'windows/system32/config/sam', 'etc/shadow',
            'boot.ini', 'config.php', '.env', '.git/config', 'wp-config.php',
            'web.config', 'database.yml', 'settings.py'
        ]
        
        self.xss_vectors = [
            "<script>alert({})</script>",
            "<img src=x onerror=alert({})>",
            "<svg/onload=alert({})>",
            "javascript:alert({})",
            "<iframe src=javascript:alert({})>",
            "<body onload=alert({})>",
            "'-alert({})-'",
            '"><script>alert({})</script>',
        ]

    def load_wordlist(self, filepath, strip_comments=True):
        """Load a wordlist file and return list of entries"""
        full_path = self.seclists_base / filepath
        if not full_path.exists():
            print(f"  WARNING: {filepath} not found, skipping...")
            return []

        entries = []
        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line or (strip_comments and line.startswith('#')):
                        continue
                    entries.append(line)
        except Exception as e:
            print(f"  ERROR loading {filepath}: {e}")
            return []

        return entries

    def load_wordlists(self):
        """Load all wordlists with weighted categories (NOT duplicated lists)"""
        
        # Store path sources with their weights (don't duplicate!)
        self.path_sources = {}
        
        # Load each category
        paths_raft = self.load_wordlist('Discovery/Web-Content/raft-medium-directories.txt')
        paths_common = self.load_wordlist('Discovery/Web-Content/common.txt')
        paths_wordpress = self.load_wordlist('Discovery/Web-Content/CMS/wordpress.fuzz.txt')
        paths_drupal = self.load_wordlist('Discovery/Web-Content/CMS/Drupal.txt')
        paths_backups = self.load_wordlist('Discovery/Web-Content/Common-DB-Backups.txt')
        paths_dotfiles = self.load_wordlist('Discovery/Web-Content/UnixDotfiles.fuzz.txt')
        paths_api1 = self.load_wordlist('Discovery/Web-Content/api/api-endpoints.txt')
        paths_api2 = self.load_wordlist('Discovery/Web-Content/api/api-seen-in-wild.txt')

        # Store with weights (only once each!)
        if paths_raft:
            self.path_sources['raft'] = (paths_raft, 50)
        if paths_common:
            self.path_sources['common'] = (paths_common, 15)
        if paths_wordpress:
            self.path_sources['wordpress'] = (paths_wordpress, 5)
        if paths_drupal:
            self.path_sources['drupal'] = (paths_drupal, 5)
        if paths_backups:
            self.path_sources['backups'] = (paths_backups, 7)
        if paths_dotfiles:
            self.path_sources['dotfiles'] = (paths_dotfiles, 8)
        if paths_api1:
            self.path_sources['api_endpoints'] = (paths_api1, 5)
        if paths_api2:
            self.path_sources['api_wild'] = (paths_api2, 5)

        # Normalize all paths
        for key in self.path_sources:
            paths, weight = self.path_sources[key]
            normalized = [p if p.startswith('/') else f'/{p}' for p in paths if p and len(p.strip()) > 0]
            self.path_sources[key] = (normalized, weight)

        # USER AGENTS
        self.bot_user_agents = self.load_wordlist('Fuzzing/User-Agents/UserAgents.fuzz.txt')
        if not self.bot_user_agents:
            self.bot_user_agents = [
                'python-requests/2.28.0', 'curl/7.64.0', 'Wget/1.20.3',
                'HTTPie/3.2.1', 'Go-http-client/1.1', 'PostmanRuntime/7.29.0',
            ]

        # PARAMETERS
        self.parameter_names = self.load_wordlist('Discovery/Web-Content/burp-parameter-names.txt')
        if not self.parameter_names:
            self.parameter_names = [
                'id', 'page', 'search', 'q', 'query', 'filter', 'sort',
                'user', 'name', 'file', 'path', 'redirect', 'callback',
            ]

        # FILE EXTENSIONS
        self.file_extensions = self.load_wordlist('Discovery/Web-Content/web-extensions.txt')
        self.file_extensions = [ext.lstrip('.') for ext in self.file_extensions if ext]
        if not self.file_extensions:
            self.file_extensions = [
                'php', 'asp', 'aspx', 'jsp', 'cgi', 'pl', 'py',
                'conf', 'config', 'bak', 'sql', 'zip', 'tar.gz'
            ]

    def select_weighted_path(self):
        """Select a path from weighted sources at generation time"""
        if not self.path_sources:
            return '/index.php'
        
        source_keys = list(self.path_sources.keys())
        weights = [self.path_sources[k][1] for k in source_keys]
        
        selected_source = random.choices(source_keys, weights=weights)[0]
        paths = self.path_sources[selected_source][0]
        
        return random.choice(paths) if paths else '/index.php'

    def generate_random_param_value(self, value_type=None):
        """Generate diverse parameter values"""
        if value_type is None:
            value_type = random.choice([
                'numeric', 'boolean', 'text', 'special_chars',
                'sql_injection', 'xss', 'path_traversal', 
                'command_injection', 'encoded'
            ])
        
        if value_type == 'numeric':
            return str(random.choice([
                random.randint(1, 100),
                random.randint(1000, 9999),
                random.randint(100000, 999999),
                random.randint(10000000, 99999999)
            ]))
        
        elif value_type == 'boolean':
            return random.choice([
                'true', 'false', '1', '0', 'yes', 'no',
                'on', 'off', 'True', 'False', 'TRUE', 'FALSE'
            ])
        
        elif value_type == 'text':
            templates = [
                'test', 'data', 'admin', 'user', 'value', 'name',
                f'test{random.randint(1, 9999)}',
                f'user{random.randint(1, 9999)}',
                f'data{random.randint(1, 9999)}',
                f'value{random.randint(1, 9999)}',
                ''.join(random.choices(string.ascii_lowercase, k=random.randint(4, 12))),
                ''.join(random.choices(string.ascii_lowercase + string.digits, k=random.randint(6, 10))),
            ]
            return random.choice(templates)
        
        elif value_type == 'special_chars':
            patterns = [
                f"user{random.choice(['*', '&', '^', '%', '$', '#', '@'])}",
                f"data(){{{random.choice(['', ';', '|', '&'])}}}",
                f"name<{random.choice(['', '>', 'script', 'img', 'svg'])}>",
                f"test[]{{{random.choice(['', ';', '|'])}}}",
                f"value{random.choice(['@', '#', '$', '%', '&', '*'])}{random.choice(['@', '#', '$', '%'])}",
            ]
            return random.choice(patterns)
        
        elif value_type == 'sql_injection':
            kw1 = random.choice(self.sql_keywords)
            kw2 = random.choice(self.sql_keywords)
            patterns = [
                f"' OR '1'='1",
                f"' OR 1=1--",
                f"' {kw1} {kw2}--",
                f"admin' --",
                f"1' {kw1} {kw2}--",
                f"' OR 'x'='x",
                f"'; {kw1} {kw2} users--",
                f"1 {kw1} {kw2} {random.choice(self.sql_keywords)}--",
                f"admin'{random.choice(['#', '--', '/*'])}",
            ]
            return random.choice(patterns)
        
        elif value_type == 'xss':
            num = random.randint(1, 9999)
            vector = random.choice(self.xss_vectors).format(num)
            return vector
        
        elif value_type == 'path_traversal':
            depth = random.randint(2, 6)
            traversal = random.choice(self.path_traversal_variants)
            target = random.choice(self.common_files)
            return f"{traversal * depth}{target}"
        
        elif value_type == 'command_injection':
            cmd = random.choice(['ls', 'cat', 'whoami', 'id', 'pwd', 'uname', 'netstat'])
            prefix = random.choice([';', '|', '&', '`', '$(',  '&&', '||'])
            suffix = ')' if prefix == '$(' else ''
            return f"{prefix} {cmd}{suffix}"
        
        elif value_type == 'encoded':
            patterns = [
                '%00', '%0A', '%0D', '%20%20', '%2520',
                '%2E%2E%2F', '%252E%252E%252F',
                '%3C', '%3E', '%27', '%22',
                f'%{random.randint(20, 126):02x}',
                f'%{random.randint(20, 126):02x}%{random.randint(20, 126):02x}',
            ]
            return random.choice(patterns)
        
        return 'default'

    def generate_many_params(self):
        """Generate URL with many diverse parameters"""
        param_count = random.randint(8, 18)
        params = []
        
        value_types = ['numeric', 'boolean', 'text', 'special_chars', 'encoded']
        
        for _ in range(param_count):
            param = random.choice(self.parameter_names)
            value_type = random.choice(value_types)
            value = self.generate_random_param_value(value_type)
            params.append(f'{param}={value}')
        
        return '&'.join(params)

    def generate_duplicate_params(self):
        """Generate URL with duplicate parameter names"""
        param_name = random.choice(self.parameter_names)
        count = random.randint(3, 6)
        values = [str(random.randint(1, 100)) for _ in range(count)]
        return '&'.join([f'{param_name}={v}' for v in values])

    def generate_long_param_value(self):
        """Generate long parameter values with variety"""
        patterns = [
            'A' * random.randint(200, 500),
            'x' * random.randint(200, 500),
            '1' * random.randint(200, 500),
            'test' * random.randint(50, 125),
            '../' * random.randint(50, 100),
            '%20' * random.randint(50, 100),
            '<script>' * random.randint(30, 60),
            'UNION SELECT ' * random.randint(20, 40),
            ''.join(random.choices(string.ascii_uppercase, k=random.randint(200, 500))),
        ]
        return random.choice(patterns)

    def apply_mixed_case(self, path):
        """Apply mixed case to evade simple filters"""
        if random.random() < 0.2:  # 20% chance
            return ''.join(
                c.upper() if random.random() < 0.5 else c.lower()
                for c in path
            )
        return path

    def generate_ip(self):
        """Generate IP with realistic scanner behavior"""
        # Enter burst mode occasionally (reduced from 5% to 3%)
        if random.random() < 0.03:
            self._burst_mode = True
            self._burst_counter = random.randint(10, 25)  # Reduced from 30 to 25
        
        # During burst, reuse same IP
        if self._burst_mode:
            if self._burst_counter > 0:
                self._burst_counter -= 1
                if self._recent_ips:
                    return self._recent_ips[-1]
            else:
                self._burst_mode = False
        
        # Regular behavior: 30% reuse rate (reduced from 40%)
        if random.random() < 0.30 and self._recent_ips:
            return random.choice(self._recent_ips)

        # Generate new IP
        ip = str(IPv4Address(getrandbits(32)))
        
        self._recent_ips.append(ip)
        if len(self._recent_ips) > 100:  # Increased pool from 50 to 100
            self._recent_ips.pop(0)
        
        return ip

    def generate_timestamp(self, base_time):
        """Generate timestamp in Apache log format"""
        dt_str = base_time.strftime('%d/%b/%Y:%H:%M:%S')
        offset_hours = random.randint(-11, 12)
        offset_minutes = random.choice([0, 15, 30, 45])
        tz = f"{offset_hours:+03d}{offset_minutes:02d}"
        return f"{dt_str} {tz}"

    def generate_suspicious_uri(self):
        """Generate a suspicious URI pattern with high variety"""
        pattern_type = random.choices(
            ['wordlist_path', 'wordlist_with_params', 'wordlist_with_extension',
             'sql_injection', 'xss_attempt', 'path_traversal', 'command_injection',
             'long_param', 'many_params', 'duplicate_params', 'special_chars',
             'double_encoding', 'empty_param', 'mixed_case'],
            weights=[20, 15, 8, 10, 8, 8, 5, 8, 8, 3, 3, 2, 1, 1]
        )[0]

        if pattern_type == 'wordlist_path':
            return self.select_weighted_path()

        elif pattern_type == 'wordlist_with_params':
            base_path = self.select_weighted_path()
            param = random.choice(self.parameter_names)
            value = self.generate_random_param_value()
            return f"{base_path}?{param}={value}"

        elif pattern_type == 'wordlist_with_extension':
            base_path = self.select_weighted_path().rstrip('/')
            ext = random.choice(self.file_extensions)
            return f"{base_path}.{ext}"

        elif pattern_type == 'sql_injection':
            base_path = random.choice(['/search.php', '/login.php', '/product.php', '/user.php'])
            param = random.choice(['id', 'user', 'query', 'search', 'name'])
            value = self.generate_random_param_value('sql_injection')
            return f"{base_path}?{param}={value}"

        elif pattern_type == 'xss_attempt':
            base_path = random.choice(['/search.php', '/comment.php', '/post.php'])
            param = random.choice(['q', 'search', 'comment', 'msg', 'text'])
            value = self.generate_random_param_value('xss')
            return f"{base_path}?{param}={value}"

        elif pattern_type == 'path_traversal':
            base_path = random.choice(['/page', '/file', '/download', '/view', '/read'])
            param = random.choice(['file', 'path', 'page', 'doc', 'include'])
            value = self.generate_random_param_value('path_traversal')
            return f"{base_path}?{param}={value}"

        elif pattern_type == 'command_injection':
            base_path = random.choice(['/exec', '/cmd', '/ping', '/run', '/system'])
            param = random.choice(['cmd', 'command', 'exec', 'run', 'host'])
            value = self.generate_random_param_value('command_injection')
            return f"{base_path}?{param}={value}"

        elif pattern_type == 'long_param':
            base_path = random.choice(['/search.php', '/index.php', '/page.php'])
            param = random.choice(self.parameter_names)
            long_value = self.generate_long_param_value()
            return f"{base_path}?{param}={long_value}"

        elif pattern_type == 'many_params':
            base_path = random.choice(['/search.php', '/api/data', '/view'])
            return f"{base_path}?{self.generate_many_params()}"

        elif pattern_type == 'duplicate_params':
            base_path = random.choice(['/search.php', '/api/data', '/filter'])
            return f"{base_path}?{self.generate_duplicate_params()}"

        elif pattern_type == 'special_chars':
            base_path = random.choice(['/search.php', '/page.php'])
            param = random.choice(self.parameter_names)
            value = self.generate_random_param_value('special_chars')
            return f"{base_path}?{param}={value}"

        elif pattern_type == 'double_encoding':
            base_path = random.choice(['/page', '/file', '/download'])
            encodings = ['%252E', '%252F', '%2520', '%253C', '%253E']
            return f"{base_path}?path={random.choice(encodings)}"

        elif pattern_type == 'empty_param':
            base_path = random.choice(['/callback', '/redirect', '/view'])
            param = random.choice(['callback', 'redirect', 'return_to'])
            return f"{base_path}?{param}="

        elif pattern_type == 'mixed_case':
            base_path = self.select_weighted_path()
            return self.apply_mixed_case(base_path)

        return '/index.php'

    def generate_time_increment(self):
        """Generate realistic time increment with burst patterns"""
        if self._burst_mode:
            # Fast scanning during burst
            return random.uniform(0.05, 0.5)
        else:
            # Varied timing patterns
            pattern = random.choices(
                ['fast', 'medium', 'slow'],
                weights=[0.6, 0.3, 0.1]
            )[0]
            
            if pattern == 'fast':
                return random.uniform(0.1, 1.0)
            elif pattern == 'medium':
                return random.uniform(1.0, 5.0)
            else:  # slow/stealthy
                return random.uniform(5.0, 30.0)

    def generate_suspicious_log(self, timestamp):
        """Generate a single suspicious log entry"""
        ip = self.generate_ip()
        ts = self.generate_timestamp(timestamp)

        # Choose method
        methods = list(self.method_weights.keys())
        weights = list(self.method_weights.values())
        method = random.choices(methods, weights=weights)[0]

        # Generate suspicious URI
        uri = self.generate_suspicious_uri()

        # Response code
        response = random.choices(self.response_codes, weights=self.response_weights)[0]

        # Response size
        size = random.randint(100, 5000)

        # Suspicious user agent
        user_agent = random.choice(self.bot_user_agents)

        # Suspicious or normal referer
        if random.random() < 0.3:
            referer = random.choice(self.suspicious_referers)
        else:
            referer = f"https://{self.faker.domain_name()}/"

        # Standard Apache Combined Log Format
        log_line = (
            f'{ip} - - [{ts}] '
            f'"{method} {uri} HTTP/1.1" '
            f'{response} {size} '
            f'"{referer}" '
            f'"{user_agent}"'
        )

        return log_line


def main():
    parser = argparse.ArgumentParser(
        description="Generate suspicious Apache access logs using SecLists wordlists"
    )
    parser.add_argument(
        '-n', '--num-lines',
        type=int,
        default=1000,
        help='Number of log lines to generate (default: 1000)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file path'
    )
    parser.add_argument(
        '--gzip',
        action='store_true',
        help='Compress output with gzip'
    )
    parser.add_argument(
        '--start-time',
        type=str,
        help='Start timestamp (format: YYYY-MM-DD HH:MM:SS, default: now)'
    )
    parser.add_argument(
        '--seclists-path',
        type=str,
        default='/home/dan/Projects/cs6727/NEW/SecLists',
        help='Path to SecLists directory'
    )

    args = parser.parse_args()

    # Initialize generator
    generator = SuspiciousLogGenerator(seclists_path=args.seclists_path)

    # Determine start time
    if args.start_time:
        try:
            start_time = datetime.strptime(args.start_time, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            print(f"Error: Invalid start time format. Use: YYYY-MM-DD HH:MM:SS")
            sys.exit(1)
    else:
        start_time = datetime.now()

    # Setup output
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = start_time.strftime('%Y%m%d_%H%M%S')
        ext = '.log.gz' if args.gzip else '.log'
        output_path = Path(f"suspicious_traffic_{timestamp}{ext}")

    # Open output file
    if args.gzip or str(output_path).endswith('.gz'):
        output_file = gzip.open(output_path, 'wt', encoding='utf-8')
    else:
        output_file = open(output_path, 'w', encoding='utf-8')

    print(f"Generating {args.num_lines:,} suspicious log entries to {output_path}...")

    try:
        current_time = start_time
        for i in range(args.num_lines):
            log_line = generator.generate_suspicious_log(current_time)
            output_file.write(log_line + '\n')

            # Increment time with realistic patterns
            increment = generator.generate_time_increment()
            current_time += timedelta(seconds=increment)

            # Progress indicator
            if (i + 1) % max(1, args.num_lines // 10) == 0:
                progress = ((i + 1) / args.num_lines) * 100
                print(f"  Progress: {progress:.0f}% ({i + 1:,}/{args.num_lines:,})")

        print(f"\n✓ Successfully generated {args.num_lines:,} suspicious log entries")
        print(f"  Output: {output_path}")

    finally:
        output_file.close()


if __name__ == "__main__":
    main()
