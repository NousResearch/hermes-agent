"""Tests for tools/file_type_registry.py — format detection and summarization."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.file_type_registry import (
    _csv_handler,
    _har_handler,
    _json_structure_handler,
    _log_file_handler,
    _format_size,
    _detect_csv_delimiter,
    detect_and_summarize,
    detect_format,
    known_format_extensions,
    register_format,
)


# =========================================================================
# Format size helper
# =========================================================================

class TestFormatSize:
    def test_bytes(self):
        assert _format_size(500) == "500 B"

    def test_kilobytes(self):
        assert _format_size(2048) == "2.0 KB"

    def test_megabytes(self):
        assert _format_size(5_242_880) == "5.0 MB"

    def test_gigabytes(self):
        assert _format_size(2_147_483_648) == "2.0 GB"

    def test_edge_zero(self):
        assert _format_size(0) == "0 B"


# =========================================================================
# CSV delimiter detection
# =========================================================================

class TestDetectCsvDelimiter:
    def test_comma(self):
        assert _detect_csv_delimiter("a,b,c") == ","

    def test_tab(self):
        assert _detect_csv_delimiter("a\tb\tc") == "\t"

    def test_semicolon(self):
        assert _detect_csv_delimiter("a;b;c") == ";"

    def test_pipe(self):
        assert _detect_csv_delimiter("a|b|c") == "|"

    def test_empty_defaults_to_comma(self):
        assert _detect_csv_delimiter("") == ","


# =========================================================================
# HAR handler
# =========================================================================

class TestHarHandler:
    def test_valid_har_with_entries(self):
        har_content = json.dumps({
            "log": {
                "entries": [
                    {
                        "request": {
                            "url": "https://api.example.com/users",
                            "method": "GET",
                        },
                        "response": {
                            "status": 200,
                            "_transferSize": 1200,
                            "content": {"size": 5000, "mimeType": "application/json"},
                        },
                        "startedDateTime": "2026-07-08T10:00:00Z",
                    },
                    {
                        "request": {
                            "url": "https://api.example.com/data",
                            "method": "POST",
                        },
                        "response": {
                            "status": 201,
                            "_transferSize": 350,
                            "content": {"size": 200, "mimeType": "text/plain"},
                        },
                        "startedDateTime": "2026-07-08T10:00:01Z",
                    },
                    {
                        "request": {
                            "url": "https://cdn.example.com/image.png",
                            "method": "GET",
                        },
                        "response": {
                            "status": 404,
                            "_transferSize": 0,
                            "content": {"size": 0, "mimeType": "text/html"},
                        },
                        "startedDateTime": "2026-07-08T10:00:02Z",
                    },
                ]
            }
        })
        path = "/tmp/test.har"
        result = _har_handler(path, har_content, 2048)
        assert result is not None
        assert "HAR Network Trace" in result
        assert "3 requests" in result
        assert "api.example.com" in result
        assert "cdn.example.com" in result
        assert "200" in result
        assert "404" in result
        assert "GET" in result
        assert "POST" in result
        # Error count: 1 (the 404)
        assert "Error responses: 1" in result

    def test_empty_har(self):
        har_content = json.dumps({"log": {"entries": []}})
        result = _har_handler("/tmp/empty.har", har_content, 100)
        assert result is not None
        assert "empty" in result.lower()

    def test_invalid_json_har(self):
        result = _har_handler("/tmp/bad.har", "not json", 100)
        assert result is not None
        assert "not valid JSON" in result

    def test_har_no_log_key(self):
        # Some HAR files use the root as the log object
        har_content = json.dumps({
            "entries": [{
                "request": {"url": "https://example.com/", "method": "GET"},
                "response": {"status": 200, "_transferSize": 100, "content": {"size": 500}},
            }]
        })
        result = _har_handler("/tmp/alt.har", har_content, 500)
        assert result is not None
        assert "HAR Network Trace" in result
        assert "1 requests" in result


# =========================================================================
# JSON handler
# =========================================================================

class TestJsonHandler:
    def test_json_object(self):
        content = json.dumps({"name": "Alice", "age": 30, "items": [1, 2, 3], "meta": {"key": "val"}})
        result = _json_structure_handler("/tmp/data.json", content, 200)
        assert result is not None
        assert "JSON Object" in result
        assert "4 top-level key" in result
        assert "name" in result
        assert "items" in result

    def test_json_array(self):
        content = json.dumps([{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}, {"id": 5}, {"id": 6}])
        result = _json_structure_handler("/tmp/list.json", content, 300)
        assert result is not None
        assert "JSON Array" in result
        assert "6 items" in result

    def test_invalid_json(self):
        result = _json_structure_handler("/tmp/bad.json", "{invalid", 100)
        assert result is not None
        assert "not valid" in result

    def test_primitive_json(self):
        result = _json_structure_handler("/tmp/num.json", "42", 10)
        assert result is not None
        assert "JSON value" in result


# =========================================================================
# CSV handler
# =========================================================================

class TestCsvHandler:
    def test_basic_csv(self):
        content = "name,age,city\nAlice,30,NYC\nBob,25,LA\nCharlie,35,SF\n"
        result = _csv_handler("/tmp/data.csv", content, 200)
        assert result is not None
        assert "CSV File" in result
        assert "3 data rows" in result
        assert "3 columns" in result
        assert "name" in result
        assert "age" in result
        assert "city" in result
        assert "Alice" in result

    def test_empty_csv(self):
        result = _csv_handler("/tmp/empty.csv", "", 0)
        assert result is not None
        assert "empty" in result.lower()

    def test_tsv_detection(self):
        content = "name\tage\tcity\nAlice\t30\tNYC\n"
        result = _csv_handler("/tmp/data.tsv", content, 100)
        assert result is not None
        assert "CSV File" in result
        assert "1 data rows" in result


# =========================================================================
# Log file handler
# =========================================================================

class TestLogFileHandler:
    def test_with_severity_levels(self):
        content = (
            "2026-07-08 INFO  Server started\n"
            "2026-07-08 ERROR Connection refused\n"
            "2026-07-08 WARN  Disk space low\n"
            "2026-07-08 INFO  Request received\n"
            "2026-07-08 ERROR Timeout exceeded\n"
        )
        result = _log_file_handler("/tmp/app.log", content, 300)
        assert result is not None
        assert "Log File" in result
        assert "5 lines" in result
        assert "INFO: 2" in result
        assert "ERROR: 2" in result
        assert "WARN: 1" in result

    def test_no_severity(self):
        content = "line one\nline two\nline three\n"
        result = _log_file_handler("/tmp/plain.log", content, 100)
        assert result is not None
        assert "Log File" in result
        assert "3 lines" in result


# =========================================================================
# Registry and detection
# =========================================================================

class TestRegistry:
    def test_detect_known_format(self):
        assert detect_format("/path/file.har") == ".har"
        assert detect_format("/path/file.json") == ".json"
        assert detect_format("/path/file.csv") == ".csv"
        assert detect_format("/path/file.log") == ".log"

    def test_detect_unknown_format(self):
        assert detect_format("/path/file.md") is None
        assert detect_format("/path/file.py") is None

    def test_known_extensions(self):
        exts = known_format_extensions()
        assert ".har" in exts
        assert ".json" in exts
        assert ".csv" in exts
        assert ".log" in exts

    def test_detect_and_summarize_happy_path(self):
        content = json.dumps({"log": {"entries": [{
            "request": {"url": "https://example.com/", "method": "GET"},
            "response": {"status": 200, "_transferSize": 100, "content": {"size": 500}},
        }]}})
        result = detect_and_summarize("/tmp/test.har", content, 500)
        assert result is not None
        assert "HAR Network Trace" in result

    def test_detect_and_summarize_unknown(self):
        result = detect_and_summarize("/tmp/unknown.xyz", "some content", 100)
        assert result is None

    def test_detect_and_summarize_handler_failure_graceful(self):
        # Handler that raises
        def _broken(_p, _c, _s):
            raise RuntimeError("oops")
        register_format(".broken", _broken, "test")
        result = detect_and_summarize("/tmp/test.broken", "content", 100)
        assert result is None


# =========================================================================
# Integration test (requires ShellFileOperations)
# =========================================================================

class TestReadFileIntegration:
    def test_har_file_reads_as_summary(self):
        """When read_file encounters a .har file, it should return a summary."""
        from tools.file_operations import ShellFileOperations
        from unittest.mock import MagicMock

        har_data = json.dumps({
            "log": {
                "entries": [
                    {
                        "request": {"url": "https://example.com/api", "method": "GET"},
                        "response": {"status": 200, "_transferSize": 500, "content": {"size": 2000, "mimeType": "application/json"}},
                        "startedDateTime": "2026-07-08T10:00:00Z",
                    }
                ]
            }
        })

        mock_env = MagicMock()
        file_size = len(har_data.encode("utf-8"))

        # Mock executor for wc -c, head -c, cat
        def mock_execute(cmd: str, cwd=None, **kwargs):
            if "wc -c" in cmd:
                return {"output": str(file_size), "returncode": 0}
            if "head -c 1000" in cmd:
                return {"output": har_data[:1000], "returncode": 0}
            if cmd.startswith("cat "):
                return {"output": har_data, "returncode": 0}
            if cmd.startswith("command -v"):
                return {"output": "yes", "returncode": 0}
            if "wc -l" in cmd:
                line_count = har_data.count("\n") + 1
                return {"output": str(line_count), "returncode": 0}
            if "echo $HOME" in cmd:
                return {"output": "/tmp", "returncode": 0}
            return {"output": "", "returncode": 0}

        mock_env.execute = mock_execute
        mock_env.cwd = "/tmp"

        ops = ShellFileOperations(mock_env, cwd="/tmp")
        result = ops.read_file("/tmp/test.har")

        assert result.format_type == "har"
        assert result.error is None
        assert "HAR Network Trace" in result.content
        assert "1 requests" in result.content

    def test_json_file_reads_as_structure(self):
        """read_file should return a JSON structure summary for .json files."""
        from tools.file_operations import ShellFileOperations
        from unittest.mock import MagicMock

        json_data = json.dumps({"users": [1, 2, 3], "config": {"debug": True}, "name": "test"})

        mock_env = MagicMock()
        file_size = len(json_data.encode("utf-8"))

        def mock_execute(cmd: str, cwd=None, **kwargs):
            if "wc -c" in cmd:
                return {"output": str(file_size), "returncode": 0}
            if "head -c 1000" in cmd:
                return {"output": json_data[:1000], "returncode": 0}
            if cmd.startswith("cat "):
                return {"output": json_data, "returncode": 0}
            if cmd.startswith("command -v"):
                return {"output": "yes", "returncode": 0}
            if "wc -l" in cmd:
                line_count = json_data.count("\n") + 1
                return {"output": str(line_count), "returncode": 0}
            if "echo $HOME" in cmd:
                return {"output": "/tmp", "returncode": 0}
            return {"output": "", "returncode": 0}

        mock_env.execute = mock_execute
        mock_env.cwd = "/tmp"

        ops = ShellFileOperations(mock_env, cwd="/tmp")
        result = ops.read_file("/tmp/data.json")

        assert result.format_type == "json"
        assert result.error is None
        assert "JSON Object" in result.content
        assert "3 top-level key" in result.content
