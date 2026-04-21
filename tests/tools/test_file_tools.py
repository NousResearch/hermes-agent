"""Tests for the file tools module (schema, handler wiring, error paths).

Tests verify tool schemas, handler dispatch, validation logic, and error
handling without requiring a running terminal environment.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from tools.file_tools import (
    READ_FILE_SCHEMA,
    WRITE_FILE_SCHEMA,
    PATCH_SCHEMA,
    SEARCH_FILES_SCHEMA,
)


class TestReadFileHandler:
    @patch("tools.file_tools._get_file_ops")
    def test_returns_file_content(self, mock_get):
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.content = "line1\nline2"
        result_obj.to_dict.return_value = {"content": "line1\nline2", "total_lines": 2}
        mock_ops.read_file.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import read_file_tool
        result = json.loads(read_file_tool("/tmp/test.txt"))
        assert result["content"] == "line1\nline2"
        assert result["total_lines"] == 2
        mock_ops.read_file.assert_called_once_with("/tmp/test.txt", 1, 500)

    @patch("tools.file_tools._get_file_ops")
    def test_custom_offset_and_limit(self, mock_get):
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.content = "line10"
        result_obj.to_dict.return_value = {"content": "line10", "total_lines": 50}
        mock_ops.read_file.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import read_file_tool
        read_file_tool("/tmp/big.txt", offset=10, limit=20)
        mock_ops.read_file.assert_called_once_with("/tmp/big.txt", 10, 20)

    @patch("tools.file_tools._get_file_ops")
    def test_exception_returns_error_json(self, mock_get):
        mock_get.side_effect = RuntimeError("terminal not available")

        from tools.file_tools import read_file_tool
        result = json.loads(read_file_tool("/tmp/test.txt"))
        assert "error" in result
        assert "terminal not available" in result["error"]


class TestWriteFileHandler:
    @patch("tools.file_tools._get_file_ops")
    def test_writes_content(self, mock_get):
        mock_ops = MagicMock()
        mock_ops.cwd = "/tmp"
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"status": "ok", "path": "/tmp/out.txt", "bytes": 13}
        mock_ops.write_file.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import write_file_tool
        result = json.loads(write_file_tool("/tmp/out.txt", "hello world!\n"))
        assert result["status"] == "ok"
        mock_ops.write_file.assert_called_once_with("/tmp/out.txt", "hello world!\n")

    @patch("tools.file_tools._validate_case_json_targets")
    @patch("tools.file_tools._get_file_ops")
    def test_non_case_write_keeps_success_behavior(self, mock_get, mock_validate):
        mock_ops = MagicMock()
        mock_ops.cwd = "/tmp"
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"status": "ok", "path": "/tmp/out.txt", "bytes": 13}
        mock_ops.write_file.return_value = result_obj
        mock_get.return_value = mock_ops
        mock_validate.return_value = None

        from tools.file_tools import write_file_tool
        result = json.loads(write_file_tool("/tmp/out.txt", "hello world!\n"))

        assert result["status"] == "ok"
        mock_validate.assert_called_once_with(["/tmp/out.txt"], task_cwd="/tmp")

    @patch("tools.file_tools._validate_case_json_targets")
    @patch("tools.file_tools._get_file_ops")
    def test_case_write_runs_validator_and_returns_success(self, mock_get, mock_validate):
        case_path = "/Users/maxmcair/.hermes/learning/cases/2026-04-10/example.json"
        mock_ops = MagicMock()
        mock_ops.cwd = "/tmp"
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"status": "ok", "path": case_path, "bytes": 42}
        mock_ops.write_file.return_value = result_obj
        mock_get.return_value = mock_ops
        mock_validate.return_value = None

        from tools.file_tools import write_file_tool
        result = json.loads(write_file_tool(case_path, "{\"id\": 1}\n"))

        assert result["status"] == "ok"
        mock_validate.assert_called_once_with([case_path], task_cwd="/tmp")

    @patch("tools.file_tools._update_read_timestamp")
    @patch("tools.file_tools._validate_case_json_targets")
    @patch("tools.file_tools._get_file_ops")
    def test_case_write_validation_failure_returns_error(self, mock_get, mock_validate, mock_update_timestamp):
        case_path = "/Users/maxmcair/.hermes/learning/cases/2026-04-10/example.json"
        mock_ops = MagicMock()
        mock_ops.cwd = "/tmp"
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"status": "ok", "path": case_path, "bytes": 42}
        mock_ops.write_file.return_value = result_obj
        mock_get.return_value = mock_ops
        mock_validate.return_value = "Case JSON validation failed for example.json: missing field"

        from tools.file_tools import write_file_tool
        result = json.loads(write_file_tool(case_path, "{\"id\": 1}\n"))

        assert result["error"] == "Case JSON validation failed for example.json: missing field"
        mock_validate.assert_called_once_with([case_path], task_cwd="/tmp")
        mock_update_timestamp.assert_not_called()

    @patch("tools.file_tools._validate_case_json_targets")
    @patch("tools.file_tools._get_file_ops")
    def test_write_error_skips_case_validator(self, mock_get, mock_validate):
        case_path = "/Users/maxmcair/.hermes/learning/cases/2026-04-10/example.json"
        mock_ops = MagicMock()
        mock_ops.cwd = "/tmp"
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"error": "disk full"}
        mock_ops.write_file.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import write_file_tool
        result = json.loads(write_file_tool(case_path, "{\"id\": 1}\n"))

        assert result["error"] == "disk full"
        mock_validate.assert_not_called()

    @patch("tools.file_tools._get_file_ops")
    def test_permission_error_returns_error_json_without_error_log(self, mock_get, caplog):
        mock_get.side_effect = PermissionError("read-only filesystem")

        from tools.file_tools import write_file_tool
        with caplog.at_level(logging.DEBUG, logger="tools.file_tools"):
            result = json.loads(write_file_tool("/tmp/out.txt", "data"))
        assert "error" in result
        assert "read-only" in result["error"]
        assert any("write_file expected denial" in r.getMessage() for r in caplog.records)
        assert not any(r.levelno >= logging.ERROR for r in caplog.records)

    @patch("tools.file_tools._get_file_ops")
    def test_unexpected_exception_still_logs_error(self, mock_get, caplog):
        mock_get.side_effect = RuntimeError("boom")

        from tools.file_tools import write_file_tool
        with caplog.at_level(logging.ERROR, logger="tools.file_tools"):
            result = json.loads(write_file_tool("/tmp/out.txt", "data"))
        assert result["error"] == "boom"
        assert any("write_file error" in r.getMessage() for r in caplog.records)


class TestPatchHandler:
    @patch("tools.file_tools._validate_case_json_targets")
    @patch("tools.file_tools._get_file_ops")
    def test_replace_mode_calls_patch_replace(self, mock_get, mock_validate):
        mock_ops = MagicMock()
        mock_ops.cwd = "/tmp"
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"status": "ok", "replacements": 1}
        mock_ops.patch_replace.return_value = result_obj
        mock_get.return_value = mock_ops
        mock_validate.return_value = None

        from tools.file_tools import patch_tool
        result = json.loads(patch_tool(
            mode="replace", path="/tmp/f.py",
            old_string="foo", new_string="bar"
        ))
        assert result["status"] == "ok"
        mock_ops.patch_replace.assert_called_once_with("/tmp/f.py", "foo", "bar", False)
        mock_validate.assert_called_once_with(["/tmp/f.py"], task_cwd="/tmp")

    @patch("tools.file_tools._update_read_timestamp")
    @patch("tools.file_tools._validate_case_json_targets")
    @patch("tools.file_tools._get_file_ops")
    def test_replace_mode_case_validation_failure_returns_error(self, mock_get, mock_validate, mock_update_timestamp):
        case_path = "/Users/maxmcair/.hermes/learning/cases/2026-04-10/example.json"
        mock_ops = MagicMock()
        mock_ops.cwd = "/tmp"
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"status": "ok", "replacements": 1}
        mock_ops.patch_replace.return_value = result_obj
        mock_get.return_value = mock_ops
        mock_validate.return_value = "Case JSON validation failed for example.json: missing field"

        from tools.file_tools import patch_tool
        result = json.loads(patch_tool(
            mode="replace", path=case_path,
            old_string="foo", new_string="bar"
        ))

        assert result["error"] == "Case JSON validation failed for example.json: missing field"
        mock_validate.assert_called_once_with([case_path], task_cwd="/tmp")
        mock_update_timestamp.assert_not_called()

    @patch("tools.file_tools._get_file_ops")
    def test_replace_mode_replace_all_flag(self, mock_get):
        mock_ops = MagicMock()
        mock_ops.cwd = "/tmp"
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"status": "ok", "replacements": 5}
        mock_ops.patch_replace.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import patch_tool
        patch_tool(mode="replace", path="/tmp/f.py",
                   old_string="x", new_string="y", replace_all=True)
        mock_ops.patch_replace.assert_called_once_with("/tmp/f.py", "x", "y", True)

    @patch("tools.file_tools._get_file_ops")
    def test_replace_mode_missing_path_errors(self, mock_get):
        from tools.file_tools import patch_tool
        result = json.loads(patch_tool(mode="replace", path=None, old_string="a", new_string="b"))
        assert "error" in result

    @patch("tools.file_tools._get_file_ops")
    def test_replace_mode_missing_strings_errors(self, mock_get):
        from tools.file_tools import patch_tool
        result = json.loads(patch_tool(mode="replace", path="/tmp/f.py", old_string=None, new_string="b"))
        assert "error" in result

    @patch("tools.file_tools._validate_case_json_targets")
    @patch("tools.file_tools._get_file_ops")
    def test_patch_mode_calls_patch_v4a(self, mock_get, mock_validate):
        mock_ops = MagicMock()
        mock_ops.cwd = "/tmp"
        result_obj = MagicMock()
        case_path = "/Users/maxmcair/.hermes/learning/cases/2026-04-10/example.json"
        result_obj.to_dict.return_value = {"status": "ok", "operations": 1, "files_modified": [case_path]}
        mock_ops.patch_v4a.return_value = result_obj
        mock_get.return_value = mock_ops
        mock_validate.return_value = None

        from tools.file_tools import patch_tool
        result = json.loads(patch_tool(mode="patch", patch="*** Begin Patch\n..."))
        assert result["status"] == "ok"
        mock_ops.patch_v4a.assert_called_once()
        mock_validate.assert_called_once_with([case_path], task_cwd="/tmp")

    @patch("tools.file_tools._update_read_timestamp")
    @patch("tools.file_tools._validate_case_json_targets")
    @patch("tools.file_tools._get_file_ops")
    def test_patch_mode_case_validation_failure_returns_error(self, mock_get, mock_validate, mock_update_timestamp):
        mock_ops = MagicMock()
        mock_ops.cwd = "/tmp"
        case_path = "/Users/maxmcair/.hermes/learning/cases/2026-04-10/example.json"
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"status": "ok", "operations": 1, "files_modified": [case_path]}
        mock_ops.patch_v4a.return_value = result_obj
        mock_get.return_value = mock_ops
        mock_validate.return_value = "Case JSON validation failed for example.json: missing field"

        from tools.file_tools import patch_tool
        result = json.loads(patch_tool(mode="patch", patch="*** Begin Patch\n..."))

        assert result["error"] == "Case JSON validation failed for example.json: missing field"
        mock_validate.assert_called_once_with([case_path], task_cwd="/tmp")
        mock_update_timestamp.assert_not_called()

    @patch("tools.file_tools._get_file_ops")
    def test_patch_mode_missing_content_errors(self, mock_get):
        from tools.file_tools import patch_tool
        result = json.loads(patch_tool(mode="patch", patch=None))
        assert "error" in result

    @patch("tools.file_tools._get_file_ops")
    def test_unknown_mode_errors(self, mock_get):
        from tools.file_tools import patch_tool
        result = json.loads(patch_tool(mode="invalid_mode"))
        assert "error" in result
        assert "Unknown mode" in result["error"]


class TestSearchHandler:
    @patch("tools.file_tools._get_file_ops")
    def test_search_calls_file_ops(self, mock_get):
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"matches": ["file1.py:3:match"]}
        mock_ops.search.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import search_tool
        result = json.loads(search_tool(pattern="TODO", target="content", path="."))
        assert "matches" in result
        mock_ops.search.assert_called_once()

    @patch("tools.file_tools._get_file_ops")
    def test_search_passes_all_params(self, mock_get):
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"matches": []}
        mock_ops.search.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import search_tool
        search_tool(pattern="class", target="files", path="/src",
                    file_glob="*.py", limit=10, offset=5, output_mode="count", context=2)
        mock_ops.search.assert_called_once_with(
            pattern="class", path="/src", target="files", file_glob="*.py",
            limit=10, offset=5, output_mode="count", context=2,
        )

    @patch("tools.file_tools._get_file_ops")
    def test_search_exception_returns_error(self, mock_get):
        mock_get.side_effect = RuntimeError("no terminal")

        from tools.file_tools import search_tool
        result = json.loads(search_tool(pattern="x"))
        assert "error" in result


class TestCaseValidationHelpers:
    def test_collect_targets_resolves_relative_paths_from_task_cwd(self, monkeypatch):
        temp_root = tempfile.mkdtemp()
        hermes_home = os.path.join(temp_root, ".hermes")
        case_root = os.path.join(hermes_home, "learning", "cases", "2026-04-10")
        os.makedirs(case_root, exist_ok=True)

        from tools.file_tools import _collect_case_json_validation_targets

        monkeypatch.setenv("HERMES_HOME", hermes_home)
        targets = _collect_case_json_validation_targets(
            ["example.json"],
            task_cwd=case_root,
        )

        assert len(targets) == 1
        assert targets[0] == Path(case_root, "example.json").resolve(strict=False)

    def test_validate_targets_reports_missing_node(self, monkeypatch):
        temp_root = tempfile.mkdtemp()
        hermes_home = os.path.join(temp_root, ".hermes")
        case_root = os.path.join(hermes_home, "learning", "cases", "2026-04-10")
        script_root = os.path.join(hermes_home, "learning", "scripts")
        os.makedirs(case_root, exist_ok=True)
        os.makedirs(script_root, exist_ok=True)
        case_path = os.path.join(case_root, "example.json")
        validator_path = os.path.join(script_root, "validate-case.mjs")

        with open(case_path, "w", encoding="utf-8") as f:
            f.write("{}\n")
        with open(validator_path, "w", encoding="utf-8") as f:
            f.write("console.log('ok')\n")

        from tools.file_tools import _validate_case_json_targets

        monkeypatch.setenv("HERMES_HOME", hermes_home)
        with patch("tools.file_tools.subprocess.run", side_effect=FileNotFoundError):
            error = _validate_case_json_targets([case_path], task_cwd=case_root)

        assert error == "Case JSON validation failed: 'node' executable not found"


# ---------------------------------------------------------------------------
# Tool result hint tests (#722)
# ---------------------------------------------------------------------------

class TestPatchHints:
    """Patch tool should hint when old_string is not found."""

    @patch("tools.file_tools._get_file_ops")
    def test_no_match_includes_hint(self, mock_get):
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {
            "error": "Could not find match for old_string in foo.py"
        }
        mock_ops.patch_replace.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import patch_tool
        raw = patch_tool(mode="replace", path="foo.py", old_string="x", new_string="y")
        assert "[Hint:" in raw
        assert "read_file" in raw

    @patch("tools.file_tools._get_file_ops")
    def test_success_no_hint(self, mock_get):
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"success": True, "diff": "--- a\n+++ b"}
        mock_ops.patch_replace.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import patch_tool
        raw = patch_tool(mode="replace", path="foo.py", old_string="x", new_string="y")
        assert "[Hint:" not in raw


class TestSearchHints:
    """Search tool should hint when results are truncated."""

    def setup_method(self):
        """Clear read/search tracker between tests to avoid cross-test state."""
        from tools.file_tools import _read_tracker
        _read_tracker.clear()

    @patch("tools.file_tools._get_file_ops")
    def test_truncated_results_hint(self, mock_get):
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {
            "total_count": 100,
            "matches": [{"path": "a.py", "line": 1, "content": "x"}] * 50,
            "truncated": True,
        }
        mock_ops.search.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import search_tool
        raw = search_tool(pattern="foo", offset=0, limit=50)
        assert "[Hint:" in raw
        assert "offset=50" in raw

    @patch("tools.file_tools._get_file_ops")
    def test_non_truncated_no_hint(self, mock_get):
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {
            "total_count": 3,
            "matches": [{"path": "a.py", "line": 1, "content": "x"}] * 3,
        }
        mock_ops.search.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import search_tool
        raw = search_tool(pattern="foo")
        assert "[Hint:" not in raw

    @patch("tools.file_tools._get_file_ops")
    def test_truncated_hint_with_nonzero_offset(self, mock_get):
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {
            "total_count": 150,
            "matches": [{"path": "a.py", "line": 1, "content": "x"}] * 50,
            "truncated": True,
        }
        mock_ops.search.return_value = result_obj
        mock_get.return_value = mock_ops

        from tools.file_tools import search_tool
        raw = search_tool(pattern="foo", offset=50, limit=50)
        assert "[Hint:" in raw
        assert "offset=100" in raw
