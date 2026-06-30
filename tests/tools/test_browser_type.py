"""Tests for browser_type wrapper behaviour."""

import json
import os
import sys
from unittest.mock import patch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TestBrowserType:
    def test_browser_type_clears_with_keystrokes_then_types(self, monkeypatch):
        from tools.browser_tool import browser_type

        monkeypatch.setattr("tools.browser_tool.sys.platform", "linux")
        side_effect = [
            {"success": True},
            {"success": True},
            {"success": True},
            {"success": True},
        ]

        with (
            patch("tools.browser_tool._last_session_key", return_value="sess"),
            patch("tools.browser_tool._run_browser_command", side_effect=side_effect) as mock_cmd,
        ):
            result = json.loads(browser_type("e3", "hello", task_id="task"))

        assert result["success"] is True
        assert result["typed"] == "hello"
        assert result["element"] == "@e3"
        assert mock_cmd.call_args_list[0][0] == ("sess", "click", ["@e3"])
        assert mock_cmd.call_args_list[1][0] == ("sess", "press", ["Control+a"])
        assert mock_cmd.call_args_list[2][0] == ("sess", "press", ["Backspace"])
        assert mock_cmd.call_args_list[3][0] == ("sess", "type", ["@e3", "hello"])

    def test_browser_type_empty_text_only_clears(self, monkeypatch):
        from tools.browser_tool import browser_type

        monkeypatch.setattr("tools.browser_tool.sys.platform", "darwin")
        side_effect = [
            {"success": True},
            {"success": True},
            {"success": True},
        ]

        with (
            patch("tools.browser_tool._last_session_key", return_value="sess"),
            patch("tools.browser_tool._run_browser_command", side_effect=side_effect) as mock_cmd,
        ):
            result = json.loads(browser_type("@e7", "", task_id="task"))

        assert result["success"] is True
        assert result["typed"] == ""
        assert result["element"] == "@e7"
        assert len(mock_cmd.call_args_list) == 3
        assert mock_cmd.call_args_list[1][0] == ("sess", "press", ["Meta+a"])
        assert mock_cmd.call_args_list[2][0] == ("sess", "press", ["Backspace"])

    def test_browser_type_reports_type_failure(self):
        from tools.browser_tool import browser_type

        side_effect = [
            {"success": True},
            {"success": True},
            {"success": True},
            {"success": False, "error": "type failed"},
        ]

        with (
            patch("tools.browser_tool._last_session_key", return_value="sess"),
            patch("tools.browser_tool._run_browser_command", side_effect=side_effect),
        ):
            result = json.loads(browser_type("@e1", "hello", task_id="task"))

        assert result == {"success": False, "error": "type failed"}
