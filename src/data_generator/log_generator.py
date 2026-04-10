"""
Synthetic Log Data Generator
Generates realistic log files in multiple formats for testing the batch processing pipeline.
"""

import os
import random
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Optional


class LogGenerator:
    """Generates synthetic log data in multiple formats with configurable anomaly injection."""

    # Realistic data pools
    ENDPOINTS = [
        "/api/v1/users", "/api/v1/login", "/api/v1/logout", "/api/v1/products",
        "/api/v1/orders", "/api/v1/payments", "/api/v1/search", "/api/v1/cart",
        "/api/v1/checkout", "/api/v1/profile", "/api/v1/settings", "/api/v1/admin",
        "/api/v1/dashboard", "/api/v1/reports", "/api/v1/notifications",
        "/static/css/main.css", "/static/js/app.js", "/static/images/logo.png",
        "/health", "/metrics", "/favicon.ico"
    ]

    HTTP_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH"]
    METHOD_WEIGHTS = [50, 25, 10, 5, 10]

    STATUS_CODES = {
        "normal": [200, 201, 204, 301, 302, 304],
        "client_error": [400, 401, 403, 404, 405, 429],
        "server_error": [500, 502, 503, 504]
    }

    IPS = [f"192.168.1.{i}" for i in range(1, 50)] + \
          [f"10.0.{i}.{j}" for i in range(1, 5) for j in range(1, 20)] + \
          ["203.0.113.42", "198.51.100.77", "172.16.0.101"]

    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
        "curl/8.4.0", "python-requests/2.31.0", "PostmanRuntime/7.35.0",
    ]

    LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    LOG_LEVEL_WEIGHTS = [10, 50, 20, 15, 5]

    SERVICES = ["auth-service", "api-gateway", "user-service", "payment-service",
                "order-service", "notification-service", "search-service", "cache-service"]

    NORMAL_MESSAGES = [
        "Request processed successfully",
        "User authentication successful",
        "Database query completed in {ms}ms",
        "Cache hit for key: {key}",
        "Session created for user {user_id}",
        "API rate limit check passed",
        "Health check: all systems operational",
        "Configuration reloaded successfully",
        "Background job completed: {job}",
        "Connection pool stats: active={active}, idle={idle}",
    ]

    WARNING_MESSAGES = [
        "High memory usage detected: {pct}%",
        "Slow query detected: {ms}ms for table {table}",
        "API rate limit approaching: {count}/1000 requests",
        "Disk space below threshold: {pct}% remaining",
        "Connection pool nearing capacity: {active}/{max}",
        "Deprecated API endpoint called: {endpoint}",
        "SSL certificate expires in {days} days",
        "Response time exceeded SLA: {ms}ms > 500ms",
    ]

    ERROR_MESSAGES = [
        "Database connection failed: timeout after 30s",
        "Authentication failed for user: {user_id}",
        "NullPointerException in {class}.{method}",
        "Payment gateway returned error: insufficient_funds",
        "File not found: {path}",
        "Memory allocation failed: OutOfMemoryError",
        "Service {service} is unreachable",
        "Invalid JSON payload in request body",
        "SSL handshake failed with remote host",
        "Deadlock detected in transaction {tx_id}",
    ]

    CRITICAL_MESSAGES = [
        "SYSTEM CRASH: Kernel panic - not syncing",
        "DATA CORRUPTION detected in table {table}",
        "SECURITY BREACH: Unauthorized access from {ip}",
        "DISK FAILURE: I/O error on /dev/sda1",
        "DATABASE DOWN: All replicas unreachable",
        "MEMORY EXHAUSTED: Process killed by OOM killer",
    ]

    ANOMALY_PATTERNS = [
        "brute_force",    # Rapid login attempts from single IP
        "ddos",           # Massive request spike
        "error_burst",    # Sudden error rate increase
        "data_exfil",     # Large data transfers
        "privilege_esc",  # Admin endpoint access attempts
    ]

    def __init__(self, config: dict):
        self.config = config
        self.num_lines = config.get("num_lines", 50000)
        self.anomaly_rate = config.get("anomaly_rate", 0.05)
        self.date_range_days = config.get("date_range_days", 30)
        self.output_dir = config.get("output_dir", "data/raw")

        self.base_date = datetime.now() - timedelta(days=self.date_range_days)
        random.seed(42)

    def _random_timestamp(self, base: Optional[datetime] = None) -> datetime:
        """Generate a random timestamp within the configured date range."""
        if base is None:
            base = self.base_date
        offset = random.randint(0, self.date_range_days * 86400)
        return base + timedelta(seconds=offset)

    def _fill_template(self, template: str) -> str:
        """Fill message template placeholders with realistic values."""
        replacements = {
            "{ms}": str(random.randint(1, 5000)),
            "{key}": f"cache:{random.choice(['user', 'session', 'product'])}:{random.randint(1, 10000)}",
            "{user_id}": f"usr_{random.randint(10000, 99999)}",
            "{job}": random.choice(["email_send", "report_gen", "data_sync", "cleanup"]),
            "{active}": str(random.randint(1, 50)),
            "{idle}": str(random.randint(0, 20)),
            "{max}": str(random.randint(50, 100)),
            "{pct}": str(random.randint(1, 99)),
            "{count}": str(random.randint(500, 999)),
            "{table}": random.choice(["users", "orders", "products", "sessions"]),
            "{endpoint}": random.choice(self.ENDPOINTS),
            "{days}": str(random.randint(1, 30)),
            "{class}": random.choice(["UserController", "PaymentService", "OrderProcessor"]),
            "{method}": random.choice(["process", "validate", "execute", "handle"]),
            "{service}": random.choice(self.SERVICES),
            "{path}": f"/var/log/{random.choice(['app', 'system', 'error'])}.log",
            "{tx_id}": f"tx_{hashlib.md5(str(random.randint(0, 99999)).encode()).hexdigest()[:8]}",
            "{ip}": random.choice(self.IPS),
        }
        result = template
        for k, v in replacements.items():
            result = result.replace(k, v)
        return result

    def _get_message(self, level: str) -> str:
        """Get a realistic log message based on log level."""
        if level == "CRITICAL":
            return self._fill_template(random.choice(self.CRITICAL_MESSAGES))
        elif level == "ERROR":
            return self._fill_template(random.choice(self.ERROR_MESSAGES))
        elif level == "WARNING":
            return self._fill_template(random.choice(self.WARNING_MESSAGES))
        else:
            return self._fill_template(random.choice(self.NORMAL_MESSAGES))

    def generate_apache_log(self, timestamp: datetime, is_anomaly: bool = False) -> str:
        """Generate an Apache-style access log line."""
        ip = random.choice(self.IPS)
        method = random.choices(self.HTTP_METHODS, weights=self.METHOD_WEIGHTS, k=1)[0]
        endpoint = random.choice(self.ENDPOINTS)

        if is_anomaly:
            status = random.choice(self.STATUS_CODES["server_error"] + self.STATUS_CODES["client_error"])
            size = random.randint(0, 100) if random.random() > 0.5 else random.randint(50000, 5000000)
            ip = "203.0.113.42"  # Suspicious IP
            endpoint = random.choice(["/api/v1/admin", "/api/v1/login", "/etc/passwd"])
        else:
            status = random.choices(
                self.STATUS_CODES["normal"] + self.STATUS_CODES["client_error"],
                weights=[15, 15, 10, 5, 5, 5, 2, 1, 1, 1, 1, 1], k=1
            )[0]
            size = random.randint(200, 50000)

        ts = timestamp.strftime("%d/%b/%Y:%H:%M:%S +0000")
        ua = random.choice(self.USER_AGENTS)
        return f'{ip} - - [{ts}] "{method} {endpoint} HTTP/1.1" {status} {size} "-" "{ua}"'

    def generate_syslog(self, timestamp: datetime, is_anomaly: bool = False) -> str:
        """Generate a syslog-style log line."""
        level = random.choices(self.LOG_LEVELS, weights=self.LOG_LEVEL_WEIGHTS, k=1)[0]
        if is_anomaly:
            level = random.choice(["ERROR", "CRITICAL"])

        service = random.choice(self.SERVICES)
        pid = random.randint(1000, 65535)
        message = self._get_message(level)
        ts = timestamp.strftime("%b %d %H:%M:%S")
        hostname = "prod-server-01"

        return f"{ts} {hostname} {service}[{pid}]: [{level}] {message}"

    def generate_json_log(self, timestamp: datetime, is_anomaly: bool = False) -> str:
        """Generate a JSON-formatted log entry."""
        level = random.choices(self.LOG_LEVELS, weights=self.LOG_LEVEL_WEIGHTS, k=1)[0]
        if is_anomaly:
            level = random.choice(["ERROR", "CRITICAL"])

        entry = {
            "timestamp": timestamp.isoformat(),
            "level": level,
            "service": random.choice(self.SERVICES),
            "host": f"prod-server-{random.randint(1, 5):02d}",
            "message": self._get_message(level),
            "request_id": f"req_{hashlib.md5(str(random.randint(0, 999999)).encode()).hexdigest()[:12]}",
            "duration_ms": random.randint(1, 500) if not is_anomaly else random.randint(2000, 30000),
            "status_code": random.choice(self.STATUS_CODES["normal"]) if not is_anomaly else random.choice(self.STATUS_CODES["server_error"]),
        }

        if is_anomaly:
            entry["anomaly_hint"] = True
            entry["error_trace"] = f"at com.app.{random.choice(['Service', 'Controller', 'Handler'])}.process(line:{random.randint(1, 500)})"

        return json.dumps(entry)

    def generate_application_log(self, timestamp: datetime, is_anomaly: bool = False) -> str:
        """Generate an application-style log line."""
        level = random.choices(self.LOG_LEVELS, weights=self.LOG_LEVEL_WEIGHTS, k=1)[0]
        if is_anomaly:
            level = random.choice(["ERROR", "CRITICAL"])

        service = random.choice(self.SERVICES)
        thread = f"thread-{random.randint(1, 50)}"
        message = self._get_message(level)
        ts = timestamp.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]

        return f"{ts} [{thread}] {level:8s} {service} - {message}"

    def _inject_anomaly_burst(self, lines: list, fmt: str, timestamp: datetime):
        """Inject a burst of anomalous log entries to simulate attack patterns."""
        pattern = random.choice(self.ANOMALY_PATTERNS)
        burst_size = random.randint(20, 100)

        generators = {
            "apache": self.generate_apache_log,
            "syslog": self.generate_syslog,
            "json": self.generate_json_log,
            "application": self.generate_application_log,
        }
        gen = generators[fmt]

        for i in range(burst_size):
            burst_ts = timestamp + timedelta(seconds=i)
            lines.append(gen(burst_ts, is_anomaly=True))

    def generate_logs(self, fmt: str = "apache", num_lines: Optional[int] = None) -> List[str]:
        """Generate a list of log lines in the specified format."""
        if num_lines is None:
            num_lines = self.num_lines

        generators = {
            "apache": self.generate_apache_log,
            "syslog": self.generate_syslog,
            "json": self.generate_json_log,
            "application": self.generate_application_log,
        }

        if fmt not in generators:
            raise ValueError(f"Unknown log format: {fmt}. Supported: {list(generators.keys())}")

        gen = generators[fmt]
        lines = []
        anomaly_count = 0

        # Generate sorted timestamps
        timestamps = sorted([self._random_timestamp() for _ in range(num_lines)])

        for i, ts in enumerate(timestamps):
            is_anomaly = random.random() < self.anomaly_rate
            if is_anomaly:
                anomaly_count += 1

            lines.append(gen(ts, is_anomaly=is_anomaly))

            # Occasionally inject anomaly bursts
            if is_anomaly and random.random() < 0.1:
                self._inject_anomaly_burst(lines, fmt, ts)

        # Add some duplicate lines for testing dedup
        num_dupes = int(num_lines * 0.02)
        for _ in range(num_dupes):
            idx = random.randint(0, len(lines) - 1)
            lines.insert(random.randint(0, len(lines)), lines[idx])

        print(f"  Generated {len(lines)} {fmt} log lines ({anomaly_count} anomalous, {num_dupes} duplicates)")
        return lines

    def save_logs(self, lines: List[str], filename: str):
        """Save log lines to a file."""
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"  Saved to {filepath} ({os.path.getsize(filepath) / 1024:.1f} KB)")
        return filepath

    def generate_all(self) -> List[str]:
        """Generate log files in all configured formats."""
        formats = self.config.get("formats", ["apache", "syslog", "json", "application"])
        generated_files = []

        print("=" * 60)
        print("LOG DATA GENERATION")
        print("=" * 60)

        for fmt in formats:
            print(f"\n[{fmt.upper()}] Generating logs...")
            lines = self.generate_logs(fmt, self.num_lines // len(formats))
            filename = f"{fmt}_logs_{datetime.now().strftime('%Y%m%d')}.log"
            filepath = self.save_logs(lines, filename)
            generated_files.append(filepath)

        print(f"\n[OK] Generated {len(generated_files)} log files in {self.output_dir}")
        return generated_files


def generate_sample_data(config: dict) -> List[str]:
    """Convenience function to generate sample log data."""
    generator = LogGenerator(config)
    return generator.generate_all()
