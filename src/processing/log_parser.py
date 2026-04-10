"""
Log Parser - Multi-format log parsing utilities.
Parses Apache, syslog, JSON, and application log formats into structured records.
"""

import re
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class LogParser:
    """Parses multiple log formats into structured dictionaries."""

    # Apache Combined Log Format regex
    APACHE_PATTERN = re.compile(
        r'(?P<ip>\S+) \S+ \S+ \[(?P<timestamp>[^\]]+)\] '
        r'"(?P<method>\S+) (?P<endpoint>\S+) (?P<protocol>\S+)" '
        r'(?P<status>\d+) (?P<size>\d+) '
        r'"(?P<referrer>[^"]*)" "(?P<user_agent>[^"]*)"'
    )

    # Syslog format regex
    SYSLOG_PATTERN = re.compile(
        r'(?P<timestamp>\w+ \d+ \d+:\d+:\d+) '
        r'(?P<hostname>\S+) '
        r'(?P<service>\S+)\[(?P<pid>\d+)\]: '
        r'\[(?P<level>\w+)\] '
        r'(?P<message>.*)'
    )

    # Application log format regex
    APP_PATTERN = re.compile(
        r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) '
        r'\[(?P<thread>[^\]]+)\] '
        r'(?P<level>\S+)\s+'
        r'(?P<service>\S+) - '
        r'(?P<message>.*)'
    )

    LOG_LEVEL_SEVERITY = {
        "DEBUG": 0,
        "INFO": 1,
        "WARNING": 2,
        "WARN": 2,
        "ERROR": 3,
        "CRITICAL": 4,
        "FATAL": 4
    }

    def __init__(self):
        self.parse_errors = 0
        self.total_parsed = 0

    def detect_format(self, line: str) -> str:
        """Auto-detect the log format of a line."""
        line = line.strip()
        if not line:
            return "unknown"

        # Try JSON first
        if line.startswith("{"):
            try:
                json.loads(line)
                return "json"
            except json.JSONDecodeError:
                pass

        # Try Apache
        if self.APACHE_PATTERN.match(line):
            return "apache"

        # Try syslog
        if self.SYSLOG_PATTERN.match(line):
            return "syslog"

        # Try application
        if self.APP_PATTERN.match(line):
            return "application"

        return "unknown"

    def parse_line(self, line: str, fmt: Optional[str] = None) -> Optional[Dict]:
        """
        Parse a single log line into a structured dictionary.

        Args:
            line: Raw log line
            fmt: Log format (auto-detected if None)

        Returns:
            Parsed dictionary or None if parsing fails
        """
        line = line.strip()
        if not line:
            return None

        if fmt is None:
            fmt = self.detect_format(line)

        self.total_parsed += 1

        parsers = {
            "apache": self._parse_apache,
            "syslog": self._parse_syslog,
            "json": self._parse_json,
            "application": self._parse_application,
        }

        parser = parsers.get(fmt)
        if parser is None:
            self.parse_errors += 1
            return self._parse_generic(line)

        try:
            result = parser(line)
            if result:
                result["_format"] = fmt
                result["_raw"] = line
            return result
        except Exception:
            self.parse_errors += 1
            return self._parse_generic(line)

    def _parse_apache(self, line: str) -> Optional[Dict]:
        """Parse an Apache combined log format line."""
        match = self.APACHE_PATTERN.match(line)
        if not match:
            self.parse_errors += 1
            return None

        d = match.groupdict()
        try:
            ts = datetime.strptime(d["timestamp"], "%d/%b/%Y:%H:%M:%S %z")
        except ValueError:
            ts = datetime.strptime(d["timestamp"].split(" ")[0], "%d/%b/%Y:%H:%M:%S")

        status = int(d["status"])
        level = "INFO"
        if status >= 500:
            level = "ERROR"
        elif status >= 400:
            level = "WARNING"

        return {
            "timestamp": ts.isoformat(),
            "ip": d["ip"],
            "method": d["method"],
            "endpoint": d["endpoint"],
            "protocol": d["protocol"],
            "status_code": status,
            "response_size": int(d["size"]),
            "referrer": d["referrer"],
            "user_agent": d["user_agent"],
            "level": level,
            "message": f"{d['method']} {d['endpoint']} {status}",
        }

    def _parse_syslog(self, line: str) -> Optional[Dict]:
        """Parse a syslog format line."""
        match = self.SYSLOG_PATTERN.match(line)
        if not match:
            self.parse_errors += 1
            return None

        d = match.groupdict()
        # Parse syslog timestamp (add current year)
        try:
            ts_str = f"{datetime.now().year} {d['timestamp']}"
            ts = datetime.strptime(ts_str, "%Y %b %d %H:%M:%S")
        except ValueError:
            ts = datetime.now()

        return {
            "timestamp": ts.isoformat(),
            "hostname": d["hostname"],
            "service": d["service"],
            "pid": int(d["pid"]),
            "level": d["level"].upper(),
            "severity": self.LOG_LEVEL_SEVERITY.get(d["level"].upper(), 1),
            "message": d["message"],
        }

    def _parse_json(self, line: str) -> Optional[Dict]:
        """Parse a JSON format log line."""
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            self.parse_errors += 1
            return None

        # Normalize common fields
        result = {
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
            "level": data.get("level", "INFO").upper(),
            "service": data.get("service", "unknown"),
            "message": data.get("message", ""),
            "host": data.get("host", "unknown"),
        }

        # Copy additional fields
        for key in ["request_id", "duration_ms", "status_code", "error_trace", "anomaly_hint"]:
            if key in data:
                result[key] = data[key]

        result["severity"] = self.LOG_LEVEL_SEVERITY.get(result["level"], 1)
        return result

    def _parse_application(self, line: str) -> Optional[Dict]:
        """Parse an application log format line."""
        match = self.APP_PATTERN.match(line)
        if not match:
            self.parse_errors += 1
            return None

        d = match.groupdict()
        try:
            ts = datetime.strptime(d["timestamp"], "%Y-%m-%d %H:%M:%S,%f")
        except ValueError:
            ts = datetime.now()

        return {
            "timestamp": ts.isoformat(),
            "thread": d["thread"],
            "level": d["level"].upper().strip(),
            "severity": self.LOG_LEVEL_SEVERITY.get(d["level"].upper().strip(), 1),
            "service": d["service"],
            "message": d["message"],
        }

    def _parse_generic(self, line: str) -> Dict:
        """Fallback parser for unrecognized formats."""
        # Try to extract a timestamp
        ts_patterns = [
            (r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}', "%Y-%m-%dT%H:%M:%S"),
            (r'\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}', "%d/%b/%Y:%H:%M:%S"),
        ]

        timestamp = datetime.now().isoformat()
        for pattern, fmt in ts_patterns:
            match = re.search(pattern, line)
            if match:
                try:
                    timestamp = datetime.strptime(match.group(), fmt).isoformat()
                except ValueError:
                    pass
                break

        # Try to extract log level
        level_match = re.search(r'\b(DEBUG|INFO|WARN(?:ING)?|ERROR|CRITICAL|FATAL)\b', line, re.IGNORECASE)
        level = level_match.group().upper() if level_match else "INFO"

        return {
            "timestamp": timestamp,
            "level": level,
            "severity": self.LOG_LEVEL_SEVERITY.get(level, 1),
            "message": line,
            "_format": "generic",
            "_raw": line,
        }

    def parse_file(self, filepath: str, fmt: Optional[str] = None) -> List[Dict]:
        """
        Parse an entire log file into structured records.

        Args:
            filepath: Path to the log file
            fmt: Log format (auto-detected from first line if None)

        Returns:
            List of parsed log records
        """
        records = []
        self.parse_errors = 0
        self.total_parsed = 0

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            # Auto-detect format from first non-empty line
            if fmt is None:
                for line in f:
                    if line.strip():
                        fmt = self.detect_format(line)
                        record = self.parse_line(line, fmt)
                        if record:
                            records.append(record)
                        break
                # Parse remaining lines
                for line in f:
                    record = self.parse_line(line, fmt)
                    if record:
                        records.append(record)
            else:
                for line in f:
                    record = self.parse_line(line, fmt)
                    if record:
                        records.append(record)

        return records

    def get_stats(self) -> Dict:
        """Get parsing statistics."""
        return {
            "total_lines": self.total_parsed,
            "parse_errors": self.parse_errors,
            "success_rate": (
                (self.total_parsed - self.parse_errors) / self.total_parsed * 100
                if self.total_parsed > 0 else 0
            ),
        }
