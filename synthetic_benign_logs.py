#!/usr/bin/env python3
"""
Synthetic Benign Apache Log Generator
Generates realistic benign web traffic logs for ML model testing
"""

import argparse
import gzip
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from faker import Faker
import numpy as np

class BenignLogGenerator:
    def __init__(self):
        self.faker = Faker()
        
        # Realistic benign URL patterns for corporate applications
        # Categorize by resource type for realistic method assignment
        self.static_resources = [
            # Static assets (should only use GET/HEAD)
            "/css/main.css", "/css/bootstrap.min.css", "/css/style.css",
            "/js/app.js", "/js/jquery.min.js", "/js/main.js",
            "/images/logo.png", "/images/icon.svg", "/favicon.ico",
        ]
        
        self.document_resources = [
            # Documents (primarily GET, occasionally POST for upload)
            "/documents/report.pdf", "/files/data.xlsx",
            "/downloads/manual.pdf", "/uploads/attachment.zip",
        ]
        
        self.page_resources = [
            # HTML pages (primarily GET)
            "/", "/index.html", "/about", "/contact", "/login", "/logout",
            "/dashboard", "/profile", "/settings", "/help", "/docs",
            "/search", "/list?page=1", "/list?page=2", "/list?sort=date",
        ]
        
        self.api_resources = [
            # API endpoints (can use all methods)
            "/api/users", "/api/auth", "/api/data", "/api/reports",
            "/api/v1/status", "/api/v1/health",
        ]
        
        self.app_resources = [
            # Application routes (GET for views, POST/PUT for actions)
            "/app/dashboard", "/app/analytics", "/app/reports",
            "/app/settings", "/app/users", "/app/projects",
            "/filter?status=active", "/search?q=project",
        ]
        
        # HTTP methods with realistic distribution for benign traffic
        self.methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"]
        self.method_weights = [0.75, 0.15, 0.05, 0.02, 0.02, 0.01]
        
        # Response codes for benign traffic (mostly success)
        self.responses = ["200", "201", "204", "301", "302", "304", "400", "401", "404"]
        self.response_weights = [0.70, 0.05, 0.02, 0.05, 0.03, 0.05, 0.03, 0.03, 0.04]
        
        # Common legitimate referers
        self.referers = [
            "-",  # Direct navigation
            "https://www.google.com/",
            "https://www.bing.com/",
            "https://mail.google.com/",
            "https://teams.microsoft.com/",
            "https://slack.com/",
        ]
        
        # User agent generators
        self.user_agents = [
            self.faker.chrome,
            self.faker.firefox,
            self.faker.safari,
            self.faker.internet_explorer,
        ]
        self.ua_weights = [0.50, 0.25, 0.20, 0.05]
        
    def generate_ip(self):
        """Generate realistic IP addresses (mix of IPv4 and some IPv6)"""
        if random.random() < 0.95:
            return self.faker.ipv4()
        else:
            return self.faker.ipv6()
    
    def generate_timestamp(self, base_time):
        """Generate timestamp in Apache log format"""
        dt_str = base_time.strftime('%d/%b/%Y:%H:%M:%S')
        tz = base_time.strftime('%z')
        # Ensure timezone is formatted correctly (e.g., -0400)
        if not tz:
            tz = '+0000'
        return f"{dt_str} {tz}"
    
    def generate_method_and_uri(self):
        """Generate HTTP method and URI together for realistic combinations"""
        # Choose resource type with realistic distribution
        resource_type = random.choices(
            ['static', 'document', 'page', 'api', 'app'],
            weights=[0.25, 0.10, 0.35, 0.20, 0.10]
        )[0]
        
        if resource_type == 'static':
            # Static resources: 98% GET, 2% HEAD
            uri = random.choice(self.static_resources)
            method = random.choices(['GET', 'HEAD'], weights=[0.98, 0.02])[0]
            
        elif resource_type == 'document':
            # Documents: 95% GET, 5% POST (uploads)
            uri = random.choice(self.document_resources)
            method = random.choices(['GET', 'POST'], weights=[0.95, 0.05])[0]
            
        elif resource_type == 'page':
            # Pages: 95% GET, 5% POST (form submissions)
            uri = random.choice(self.page_resources)
            method = random.choices(['GET', 'POST'], weights=[0.95, 0.05])[0]
            
        elif resource_type == 'api':
            # API: All methods allowed with realistic distribution
            uri = random.choice(self.api_resources)
            method = np.random.choice(
                ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
                p=[0.60, 0.25, 0.10, 0.03, 0.02]
            )
            
        else:  # app
            # App routes: GET for views, POST/PUT for actions
            uri = random.choice(self.app_resources)
            method = np.random.choice(
                ['GET', 'POST', 'PUT'],
                p=[0.75, 0.20, 0.05]
            )
        
        # Add query parameters to some requests (except static resources)
        if resource_type != 'static' and random.random() < 0.15 and '?' not in uri:
            params = []
            for _ in range(random.randint(1, 2)):
                key = random.choice(['id', 'page', 'sort', 'filter', 'q', 'limit'])
                value = random.choice(['1', '2', 'asc', 'desc', 'active', 'name', '10'])
                params.append(f"{key}={value}")
            uri += '?' + '&'.join(params)
        
        return method, uri
    
    def generate_response(self):
        """Generate response code with realistic distribution"""
        return np.random.choice(self.responses, p=self.response_weights)
    
    def generate_bytes(self):
        """Generate response size in bytes (realistic distribution)"""
        # Most responses are small to medium, ensure always positive
        size = int(random.gauss(5000, 3000))
        return max(100, size)  # Ensure minimum of 100 bytes
    
    def generate_referer(self, base_uri):
        """Generate referer field"""
        if random.random() < 0.3:
            return "-"
        elif random.random() < 0.5:
            # Internal referer - pick from any resource category
            all_resources = (self.static_resources + self.document_resources + 
                           self.page_resources + self.api_resources + self.app_resources)
            return f"https://app.example.com{random.choice(all_resources)}"
        else:
            # External referer
            return random.choice(self.referers)
    
    def generate_user_agent(self):
        """Generate user agent string"""
        ua_func = np.random.choice(self.user_agents, p=self.ua_weights)
        return ua_func()
    
    def generate_log_line(self, timestamp):
        """Generate a single benign log line in Apache Combined format"""
        ip = self.generate_ip()
        ts = self.generate_timestamp(timestamp)
        method, uri = self.generate_method_and_uri()
        response = self.generate_response()
        bytes_sent = self.generate_bytes()
        referer = self.generate_referer(uri)
        user_agent = self.generate_user_agent()
        
        # Apache Combined Log Format
        log_line = (
            f'{ip} - - [{ts}] '
            f'"{method} {uri} HTTP/1.1" '
            f'{response} {bytes_sent} '
            f'"{referer}" '
            f'"{user_agent}"'
        )
        
        return log_line

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic benign Apache access logs for ML testing"
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
        help='Output file path (default: stdout)'
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
        '--time-span',
        type=int,
        default=3600,
        help='Time span in seconds to distribute logs over (default: 3600 = 1 hour)'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='synthetic_benign',
        help='Output file prefix (default: synthetic_benign)'
    )
    parser.add_argument(
        '--sleep',
        type=float,
        default=0.0,
        help='Sleep between lines in seconds (default: 0)'
    )
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = BenignLogGenerator()
    
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
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if args.gzip:
            output_path = Path(f"{args.prefix}_{timestamp}.log.gz")
        else:
            output_path = Path(f"{args.prefix}_{timestamp}.log")
    
    # Open output file
    if args.output is None and not output_path:
        # Write to stdout
        output_file = sys.stdout
        should_close = False
    elif args.gzip or str(output_path).endswith('.gz'):
        output_file = gzip.open(output_path, 'wt')
        should_close = True
        print(f"Generating {args.num_lines:,} benign log entries to {output_path} (gzipped)...")
    else:
        output_file = open(output_path, 'w')
        should_close = True
        print(f"Generating {args.num_lines:,} benign log entries to {output_path}...")
    
    # Generate logs
    time_increment = args.time_span / args.num_lines if args.num_lines > 0 else 1
    current_time = start_time
    
    try:
        for i in range(args.num_lines):
            # Generate log line
            log_line = generator.generate_log_line(current_time)
            output_file.write(log_line + '\n')
            
            # Increment time
            current_time += timedelta(seconds=time_increment)
            
            # Optional: sleep between lines
            if args.sleep > 0:
                time.sleep(args.sleep)
            
            # Progress indicator (every 10%)
            if should_close and (i + 1) % max(1, args.num_lines // 10) == 0:
                progress = ((i + 1) / args.num_lines) * 100
                print(f"  Progress: {progress:.0f}% ({i + 1:,}/{args.num_lines:,})")
        
        if should_close:
            print(f"✓ Successfully generated {args.num_lines:,} benign log entries")
            print(f"  Output: {output_path}")
            
    finally:
        if should_close:
            output_file.close()

if __name__ == "__main__":
    main()
