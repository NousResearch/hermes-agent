"""Tests for browser_console tool and browser_vision annotate param."""

import json
import os
import sys
from pathlib import Path
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


class TestBrowserConsoleToolsetWiring:
    """browser_console must be reachable via toolset resolution."""

    def test_in_browser_toolset(self):
        from toolsets import TOOLSETS
        assert "browser_console" in TOOLSETS["browser"]["tools"]

    def test_in_hermes_core_tools(self):
        from toolsets import _HERMES_CORE_TOOLS
        assert "browser_console" in _HERMES_CORE_TOOLS

    def test_in_legacy_toolset_map(self):
        from model_tools import _LEGACY_TOOLSET_MAP
        assert "browser_console" in _LEGACY_TOOLSET_MAP["browser_tools"]

    def test_in_registry(self):
        from tools.registry import registry
        from tools import browser_tool  # noqa: F401
        assert "browser_console" in registry._tools


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


class TestBrowserVisionConfig:
    def _setup_screenshot(self, tmp_path):
        shots_dir = tmp_path / "browser_screenshots"
        shots_dir.mkdir()
        screenshot = shots_dir / "shot.png"
        screenshot.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
        return shots_dir, screenshot

    def test_browser_vision_uses_configured_temperature_and_timeout(self, tmp_path):
        from tools.browser_tool import browser_vision

        shots_dir, screenshot = self._setup_screenshot(tmp_path)
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Annotated screenshot analysis"
        mock_response.choices = [mock_choice]

        with (
            patch("hermes_constants.get_hermes_dir", return_value=shots_dir),
            patch("tools.browser_tool._cleanup_old_screenshots"),
            patch("tools.browser_tool._run_browser_command", return_value={"success": True, "data": {"path": str(screenshot)}}),
            patch("tools.browser_tool._get_vision_model", return_value="test-model"),
            patch("hermes_cli.config.load_config", return_value={"auxiliary": {"vision": {"temperature": 1, "timeout": 45}}}),
            patch("tools.browser_tool.call_llm", return_value=mock_response) as mock_llm,
        ):
            result = json.loads(browser_vision("what is on the page?", task_id="test"))

        assert result["success"] is True
        assert result["analysis"] == "Annotated screenshot analysis"
        assert mock_llm.call_args.kwargs["temperature"] == 1.0
        assert mock_llm.call_args.kwargs["timeout"] == 45.0

    def test_browser_vision_defaults_temperature_when_config_omits_it(self, tmp_path):
        from tools.browser_tool import browser_vision

        shots_dir, screenshot = self._setup_screenshot(tmp_path)
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Default screenshot analysis"
        mock_response.choices = [mock_choice]

        with (
            patch("hermes_constants.get_hermes_dir", return_value=shots_dir),
            patch("tools.browser_tool._cleanup_old_screenshots"),
            patch("tools.browser_tool._run_browser_command", return_value={"success": True, "data": {"path": str(screenshot)}}),
            patch("tools.browser_tool._get_vision_model", return_value="test-model"),
            patch("hermes_cli.config.load_config", return_value={"auxiliary": {"vision": {}}}),
            patch("tools.browser_tool.call_llm", return_value=mock_response) as mock_llm,
        ):
            result = json.loads(browser_vision("what is on the page?", task_id="test"))

        assert result["success"] is True
        assert result["analysis"] == "Default screenshot analysis"
        assert mock_llm.call_args.kwargs["temperature"] == 0.1
        assert mock_llm.call_args.kwargs["timeout"] == 120.0


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


class TestBrowserConsoleEvalSafety:
    """Security checks for browser_console(expression=...)."""

    def test_cloud_eval_blocks_network_primitive(self, monkeypatch):
        from tools import browser_tool as bt

        monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(bt, "_is_local_backend", lambda: False)
        monkeypatch.setattr(bt, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(bt, "_is_always_blocked_url", lambda url: False)
        monkeypatch.setattr(bt, "_is_safe_url", lambda url: True)
        monkeypatch.setattr(bt, "check_website_access", lambda url: None)
        monkeypatch.setattr(
            bt,
            "_run_browser_command",
            lambda *args, **kwargs: pytest.fail("unsafe eval must be blocked before backend execution"),
        )

        result = json.loads(bt.browser_console(expression="fetch('https://example.com/')"))

        assert result["success"] is False
        assert "cannot perform navigation or network requests" in result["error"]

    def test_cloud_eval_blocks_metadata_url_literal(self, monkeypatch):
        from tools import browser_tool as bt

        monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(bt, "_is_local_backend", lambda: False)
        monkeypatch.setattr(bt, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(bt, "_is_always_blocked_url", lambda url: "169.254.169.254" in url)
        monkeypatch.setattr(
            bt,
            "_run_browser_command",
            lambda *args, **kwargs: pytest.fail("metadata URL must be blocked before backend execution"),
        )

        result = json.loads(bt.browser_console(expression="'http://169.254.169.254/latest/meta-data/'"))

        assert result["success"] is False
        assert "cloud metadata" in result["error"]

    def test_cloud_eval_blocks_token_exfiltration_literal(self, monkeypatch):
        from tools import browser_tool as bt

        monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(bt, "_is_local_backend", lambda: False)
        monkeypatch.setattr(bt, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(bt, "_is_always_blocked_url", lambda url: False)
        monkeypatch.setattr(
            bt,
            "_run_browser_command",
            lambda *args, **kwargs: pytest.fail("token-like URL must be blocked before backend execution"),
        )

        result = json.loads(bt.browser_console(expression="'https://evil.test/?key=sk-ant-api03-abc'"))

        assert result["success"] is False
        assert "API key or token" in result["error"]

    def test_local_eval_preserves_dom_inspection(self, monkeypatch):
        from tools import browser_tool as bt

        monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(bt, "_is_local_backend", lambda: True)
        monkeypatch.setattr(bt, "_last_session_key", lambda task_id: "test")
        monkeypatch.setattr(
            bt,
            "_run_browser_command",
            lambda *args, **kwargs: {"success": True, "data": {"result": "Example"}},
        )

        result = json.loads(bt.browser_console(expression="document.title", task_id="test"))

        assert result["success"] is True
        assert result["result"] == "Example"

    def test_eval_blanks_page_after_unsafe_final_url(self, monkeypatch):
        from tools import browser_tool as bt

        calls = []

        def fake_run(task_id, command, args, **kwargs):
            calls.append((command, args))
            if command == "eval" and args == ["document.title"]:
                return {"success": True, "data": {"result": "ok"}}
            if command == "eval" and args == ["location.href"]:
                return {"success": True, "data": {"result": "http://169.254.169.254/latest/meta-data/"}}
            if command == "open" and args == ["about:blank"]:
                return {"success": True, "data": {}}
            pytest.fail(f"unexpected browser command: {command} {args}")

        monkeypatch.setattr(bt, "_browser_eval_security_error", lambda expression: None)
        monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(bt, "_is_local_backend", lambda: False)
        monkeypatch.setattr(bt, "_last_session_key", lambda task_id: "cloud-task")
        monkeypatch.setattr(bt, "_is_always_blocked_url", lambda url: "169.254.169.254" in url)
        monkeypatch.setattr(bt, "_run_browser_command", fake_run)

        result = json.loads(bt.browser_console(expression="document.title", task_id="cloud-task"))

        assert result["success"] is False
        assert "cloud metadata" in result["error"]
        assert ("open", ["about:blank"]) in calls
