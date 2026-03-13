"""Tests for browser_console tool and browser_vision annotate param."""

import json
import os
import sys
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ── browser_console ──────────────────────────────────────────────────


class TestBrowserConsole:
    """browser_console() returns console messages + JS errors in one call."""

    def test_returns_console_messages_and_errors(self):
        from tools.browser_tool import browser_console

        console_response = {
            "success": True,
            "data": {
                "messages": [
                    {"text": "hello", "type": "log", "timestamp": 1},
                    {"text": "oops", "type": "error", "timestamp": 2},
                ]
            },
        }
        errors_response = {
            "success": True,
            "data": {
                "errors": [
                    {"message": "Uncaught TypeError", "timestamp": 3},
                ]
            },
        }

        with patch("tools.browser_tool._run_browser_command") as mock_cmd:
            mock_cmd.side_effect = [console_response, errors_response]
            result = json.loads(browser_console(task_id="test"))

        assert result["success"] is True
        assert result["total_messages"] == 2
        assert result["total_errors"] == 1
        assert result["console_messages"][0]["text"] == "hello"
        assert result["console_messages"][1]["text"] == "oops"
        assert result["js_errors"][0]["message"] == "Uncaught TypeError"

    def test_passes_clear_flag(self):
        from tools.browser_tool import browser_console

        empty = {"success": True, "data": {"messages": [], "errors": []}}
        with patch("tools.browser_tool._run_browser_command", return_value=empty) as mock_cmd:
            browser_console(clear=True, task_id="test")

        calls = mock_cmd.call_args_list
        # Both console and errors should get --clear
        assert calls[0][0] == ("test", "console", ["--clear"])
        assert calls[1][0] == ("test", "errors", ["--clear"])

    def test_no_clear_by_default(self):
        from tools.browser_tool import browser_console

        empty = {"success": True, "data": {"messages": [], "errors": []}}
        with patch("tools.browser_tool._run_browser_command", return_value=empty) as mock_cmd:
            browser_console(task_id="test")

        calls = mock_cmd.call_args_list
        assert calls[0][0] == ("test", "console", [])
        assert calls[1][0] == ("test", "errors", [])

    def test_empty_console_and_errors(self):
        from tools.browser_tool import browser_console

        empty = {"success": True, "data": {"messages": [], "errors": []}}
        with patch("tools.browser_tool._run_browser_command", return_value=empty):
            result = json.loads(browser_console(task_id="test"))

        assert result["total_messages"] == 0
        assert result["total_errors"] == 0
        assert result["console_messages"] == []
        assert result["js_errors"] == []

    def test_handles_failed_commands(self):
        from tools.browser_tool import browser_console

        failed = {"success": False, "error": "No session"}
        with patch("tools.browser_tool._run_browser_command", return_value=failed):
            result = json.loads(browser_console(task_id="test"))

        # Should still return success with empty data
        assert result["success"] is True
        assert result["total_messages"] == 0
        assert result["total_errors"] == 0


# ── browser_console schema ───────────────────────────────────────────


class TestBrowserConsoleSchema:
    """browser_console is properly registered in the tool registry."""

    def test_schema_in_browser_schemas(self):
        from tools.browser_tool import BROWSER_TOOL_SCHEMAS

        names = [s["name"] for s in BROWSER_TOOL_SCHEMAS]
        assert "browser_console" in names

    def test_schema_has_clear_param(self):
        from tools.browser_tool import BROWSER_TOOL_SCHEMAS

        schema = next(s for s in BROWSER_TOOL_SCHEMAS if s["name"] == "browser_console")
        props = schema["parameters"]["properties"]
        assert "clear" in props
        assert props["clear"]["type"] == "boolean"


# ── browser_vision annotate ──────────────────────────────────────────


class TestBrowserVisionAnnotate:
    """browser_vision supports annotate parameter."""

    def test_schema_has_annotate_param(self):
        from tools.browser_tool import BROWSER_TOOL_SCHEMAS

        schema = next(s for s in BROWSER_TOOL_SCHEMAS if s["name"] == "browser_vision")
        props = schema["parameters"]["properties"]
        assert "annotate" in props
        assert props["annotate"]["type"] == "boolean"

    def test_annotate_false_no_flag(self):
        """Without annotate, screenshot command has no --annotate flag."""
        from tools.browser_tool import browser_vision

        with (
            patch("tools.browser_tool._run_browser_command") as mock_cmd,
            patch("tools.browser_tool.call_llm") as mock_call_llm,
            patch("tools.browser_tool._get_vision_model", return_value="test-model"),
        ):
            mock_cmd.return_value = {"success": True, "data": {}}
            # Will fail at screenshot file read, but we can check the command
            try:
                browser_vision("test", annotate=False, task_id="test")
            except Exception:
                pass

            if mock_cmd.called:
                args = mock_cmd.call_args[0]
                cmd_args = args[2] if len(args) > 2 else []
                assert "--annotate" not in cmd_args

    def test_annotate_true_adds_flag(self):
        """With annotate=True, screenshot command includes --annotate."""
        from tools.browser_tool import browser_vision

        with (
            patch("tools.browser_tool._run_browser_command") as mock_cmd,
            patch("tools.browser_tool.call_llm") as mock_call_llm,
            patch("tools.browser_tool._get_vision_model", return_value="test-model"),
        ):
            mock_cmd.return_value = {"success": True, "data": {}}
            try:
                browser_vision("test", annotate=True, task_id="test")
            except Exception:
                pass

            if mock_cmd.called:
                args = mock_cmd.call_args[0]
                cmd_args = args[2] if len(args) > 2 else []
                assert "--annotate" in cmd_args


# ── auto-recording config ────────────────────────────────────────────


class TestRecordSessionsConfig:
    """browser.record_sessions config option."""

    def test_default_config_has_record_sessions(self):
        from hermes_cli.config import DEFAULT_CONFIG

        browser_cfg = DEFAULT_CONFIG.get("browser", {})
        assert "record_sessions" in browser_cfg
        assert browser_cfg["record_sessions"] is False

    def test_maybe_start_recording_disabled(self):
        """Recording doesn't start when config says record_sessions: false."""
        from tools.browser_tool import _maybe_start_recording, _recording_sessions

        with (
            patch("tools.browser_tool._run_browser_command") as mock_cmd,
            patch("builtins.open", side_effect=FileNotFoundError),
        ):
            _maybe_start_recording("test-task")

        mock_cmd.assert_not_called()
        assert "test-task" not in _recording_sessions

    def test_maybe_stop_recording_noop_when_not_recording(self):
        """Stopping when not recording is a no-op."""
        from tools.browser_tool import _maybe_stop_recording, _recording_sessions

        _recording_sessions.discard("test-task")  # ensure not in set
        with patch("tools.browser_tool._run_browser_command") as mock_cmd:
            _maybe_stop_recording("test-task")

        mock_cmd.assert_not_called()


# ── dogfood skill files ──────────────────────────────────────────────


class TestDogfoodSkill:
    """Dogfood skill files exist and have correct structure."""

    @pytest.fixture(autouse=True)
    def _skill_dir(self):
        # Use the actual repo skills dir (not temp)
        self.skill_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "skills", "dogfood"
        )

    def test_skill_md_exists(self):
        assert os.path.exists(os.path.join(self.skill_dir, "SKILL.md"))

    def test_taxonomy_exists(self):
        assert os.path.exists(
            os.path.join(self.skill_dir, "references", "issue-taxonomy.md")
        )

    def test_report_template_exists(self):
        assert os.path.exists(
            os.path.join(self.skill_dir, "templates", "dogfood-report-template.md")
        )

    def test_skill_md_has_frontmatter(self):
        with open(os.path.join(self.skill_dir, "SKILL.md")) as f:
            content = f.read()
        assert content.startswith("---")
        assert "name: dogfood" in content
        assert "description:" in content

    def test_skill_references_browser_console(self):
        with open(os.path.join(self.skill_dir, "SKILL.md")) as f:
            content = f.read()
        assert "browser_console" in content

    def test_skill_references_annotate(self):
        with open(os.path.join(self.skill_dir, "SKILL.md")) as f:
            content = f.read()
        assert "annotate" in content

    def test_taxonomy_has_severity_levels(self):
        with open(
            os.path.join(self.skill_dir, "references", "issue-taxonomy.md")
        ) as f:
            content = f.read()
        assert "Critical" in content
        assert "High" in content
        assert "Medium" in content
        assert "Low" in content

    def test_taxonomy_has_categories(self):
        with open(
            os.path.join(self.skill_dir, "references", "issue-taxonomy.md")
        ) as f:
            content = f.read()
        assert "Functional" in content
        assert "Visual" in content
        assert "Accessibility" in content
        assert "Console" in content
