"""Tests for LogParser"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processing.log_parser import LogParser


def test_detect_apache():
    parser = LogParser()
    line = '192.168.1.1 - - [09/Mar/2026:10:15:30 +0000] "GET /api/v1/users HTTP/1.1" 200 1234 "-" "Mozilla/5.0"'
    assert parser.detect_format(line) == "apache"


def test_detect_syslog():
    parser = LogParser()
    line = 'Mar 09 10:15:30 prod-server-01 auth-service[12345]: [ERROR] Authentication failed for user: usr_12345'
    assert parser.detect_format(line) == "syslog"


def test_detect_json():
    parser = LogParser()
    line = '{"timestamp": "2026-03-09T10:15:30", "level": "INFO", "message": "test"}'
    assert parser.detect_format(line) == "json"


def test_detect_application():
    parser = LogParser()
    line = '2026-03-09 10:15:30,123 [thread-1] INFO     auth-service - Request processed successfully'
    assert parser.detect_format(line) == "application"


def test_parse_apache():
    parser = LogParser()
    line = '192.168.1.1 - - [09/Mar/2026:10:15:30 +0000] "GET /api/v1/users HTTP/1.1" 200 1234 "-" "Mozilla/5.0"'
    result = parser.parse_line(line, "apache")
    assert result is not None
    assert result["ip"] == "192.168.1.1"
    assert result["method"] == "GET"
    assert result["endpoint"] == "/api/v1/users"
    assert result["status_code"] == 200
    assert result["response_size"] == 1234


def test_parse_json():
    parser = LogParser()
    line = '{"timestamp": "2026-03-09T10:15:30", "level": "ERROR", "service": "api", "message": "test error"}'
    result = parser.parse_line(line, "json")
    assert result is not None
    assert result["level"] == "ERROR"
    assert result["message"] == "test error"
    assert result["service"] == "api"


def test_parse_syslog():
    parser = LogParser()
    line = 'Mar 09 10:15:30 prod-server-01 auth-service[12345]: [ERROR] Auth failed'
    result = parser.parse_line(line, "syslog")
    assert result is not None
    assert result["level"] == "ERROR"
    assert result["service"] == "auth-service"
    assert result["hostname"] == "prod-server-01"


def test_parse_empty_line():
    parser = LogParser()
    assert parser.parse_line("") is None
    assert parser.parse_line("   ") is None


def test_parse_malformed():
    parser = LogParser()
    result = parser.parse_line("this is a random malformed line with no structure")
    # Should fall through to generic parser
    assert result is not None
    assert "message" in result


def test_stats():
    parser = LogParser()
    parser.parse_line('{"timestamp": "2026-01-01", "level": "INFO", "message": "ok"}', "json")
    parser.parse_line("malformed line")
    stats = parser.get_stats()
    assert stats["total_lines"] == 2
    assert stats["success_rate"] > 0
