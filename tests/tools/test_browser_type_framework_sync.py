"""Tests for React/framework input sync after browser_type."""

import json
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TestControlledInputSyncExpression:
    def test_expression_uses_native_setter_and_input_event(self):
        from tools.browser_input_sync import build_controlled_input_sync_expression

        js = build_controlled_input_sync_expression("Jane Doe")
        assert "HTMLInputElement.prototype" in js
        assert "insertText" in js
        assert '"Jane Doe"' in js
        assert "change" in js


class TestBrowserTypeFrameworkSync:
    def test_successful_type_runs_fill_focus_and_sync_eval(self):
        from tools.browser_tool import browser_type

        fill_ok = {"success": True}
        focus_ok = {"success": True}
        eval_ok = json.dumps({"success": True, "result": {"ok": True, "value": "Jane"}})

        with (
            patch("tools.browser_tool._run_browser_command") as mock_cmd,
            patch("tools.browser_tool._browser_eval", return_value=eval_ok) as mock_eval,
        ):
            mock_cmd.side_effect = [fill_ok, focus_ok]
            result = json.loads(browser_type("@e3", "Jane", task_id="sess"))

        assert result["success"] is True
        assert result["framework_synced"] is True
        assert "framework_sync_warning" not in result

        assert mock_cmd.call_args_list[0][0][1] == "fill"
        assert mock_cmd.call_args_list[0][0][2] == ["@e3", "Jane"]
        assert mock_cmd.call_args_list[1][0][1] == "focus"
        assert mock_cmd.call_args_list[1][0][2] == ["@e3"]
        mock_eval.assert_called_once()
        assert "insertText" in mock_eval.call_args[0][0]

    def test_sync_failure_surfaces_warning_but_keeps_success(self):
        from tools.browser_tool import browser_type

        with (
            patch("tools.browser_tool._run_browser_command") as mock_cmd,
            patch(
                "tools.browser_tool._browser_eval",
                return_value=json.dumps(
                    {"success": True, "result": {"ok": False, "reason": "no active element"}}
                ),
            ),
        ):
            mock_cmd.side_effect = [{"success": True}, {"success": True}]
            result = json.loads(browser_type("@e3", "Jane", task_id="sess"))

        assert result["success"] is True
        assert result["framework_synced"] is False
        assert "no active element" in result["framework_sync_warning"]

    def test_fill_failure_skips_sync(self):
        from tools.browser_tool import browser_type

        with (
            patch(
                "tools.browser_tool._run_browser_command",
                return_value={"success": False, "error": "element not found"},
            ) as mock_cmd,
            patch("tools.browser_tool._browser_eval") as mock_eval,
        ):
            result = json.loads(browser_type("@e9", "Jane", task_id="sess"))

        assert result["success"] is False
        assert mock_cmd.call_count == 1
        mock_eval.assert_not_called()
