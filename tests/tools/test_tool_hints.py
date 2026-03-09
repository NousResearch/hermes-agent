"""Tests for structured tool result hints (CTAs) — issue #722."""
import json
import pytest
from unittest.mock import MagicMock, patch


class TestTerminalToolHints:
    def test_hint_injected_for_nonzero_exit(self):
        returncode = 1
        output = "bash: command not found"
        result_json = json.dumps({"output": output, "exit_code": returncode, "error": None})
        if "OUTPUT TRUNCATED" in output:
            result_json += "\n\n[Hint: Output was truncated.]"
        elif returncode != 0:
            result_json += f"\n\n[Hint: Exit code {returncode}. Check the error output above.]"
        assert "[Hint:" in result_json
        assert "1" in result_json

    def test_hint_injected_for_truncated_output(self):
        returncode = 0
        output = "line1\n... [OUTPUT TRUNCATED - 1000 chars omitted out of 2000 total] ...\nline2"
        result_json = json.dumps({"output": output, "exit_code": returncode, "error": None})
        if "OUTPUT TRUNCATED" in output:
            result_json += "\n\n[Hint: Output was truncated. Pipe to head/tail for specific sections.]"
        elif returncode != 0:
            result_json += f"\n\n[Hint: Exit code {returncode}.]"
        assert "[Hint:" in result_json
        assert "truncated" in result_json.lower()

    def test_no_hint_for_success(self):
        returncode = 0
        output = "all good"
        result_json = json.dumps({"output": output, "exit_code": returncode, "error": None})
        if "OUTPUT TRUNCATED" in output:
            result_json += "\n\n[Hint: truncated]"
        elif returncode != 0:
            result_json += "\n\n[Hint: nonzero]"
        assert "[Hint:" not in result_json


class TestWebSearchToolHints:
    @patch("tools.web_tools._get_firecrawl_client")
    @patch("tools.web_tools._debug")
    def test_results_hint_includes_top_url(self, mock_debug, mock_client):
        mock_response = MagicMock()
        mock_response.web = [
            MagicMock(model_dump=lambda: {"url": "https://example.com", "title": "Example"})
        ]
        mock_client.return_value.search.return_value = mock_response
        mock_debug.log_call = MagicMock()
        mock_debug.save = MagicMock()
        from tools.web_tools import web_search_tool
        result = web_search_tool(query="test query")
        assert "[Hint:" in result
        assert "web_extract" in result

    @patch("tools.web_tools._get_firecrawl_client")
    @patch("tools.web_tools._debug")
    def test_no_results_hint(self, mock_debug, mock_client):
        mock_response = MagicMock()
        mock_response.web = []
        mock_client.return_value.search.return_value = mock_response
        mock_debug.log_call = MagicMock()
        mock_debug.save = MagicMock()
        from tools.web_tools import web_search_tool
        result = web_search_tool(query="xyzzy123")
        assert "[Hint:" in result
        assert "No results" in result


class TestBrowserSnapshotHints:
    @patch("tools.browser_tool._run_browser_command")
    def test_no_hint_when_short(self, mock_run):
        mock_run.return_value = {
            "success": True,
            "data": {"snapshot": "short content", "refs": {}}
        }
        from tools.browser_tool import browser_snapshot
        result = browser_snapshot()
        assert "[Hint:" not in result

    @patch("tools.browser_tool._run_browser_command")
    def test_hint_when_truncated(self, mock_run):
        mock_run.return_value = {
            "success": True,
            "data": {"snapshot": "x" * 100, "refs": {}}
        }
        with patch("tools.browser_tool.SNAPSHOT_SUMMARIZE_THRESHOLD", 10):
            from tools.browser_tool import browser_snapshot
            result = browser_snapshot()
        assert "[Hint:" in result
        assert "browser_scroll" in result
