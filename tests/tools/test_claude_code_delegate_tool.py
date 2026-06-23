"""Tests for tools/claude_code_delegate_tool.py."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

from tools.claude_code_delegate_tool import (
    _as_float,
    _as_int,
    _command_for,
    _existing_add_dirs,
    _parse_stream_json,
    _redacted_command,
    _resolve_cwd,
    _wrap_prompt,
    check_claude_code_available,
    delegate_to_claude_code,
    DELEGATE_TO_CLAUDE_CODE_SCHEMA,
)


# ---------------------------------------------------------------------------
# _as_int / _as_float
# ---------------------------------------------------------------------------


class TestAsInt:
    def test_valid_value(self):
        assert _as_int(42, 10, 0, 100) == 42

    def test_string_coercion(self):
        assert _as_int("42", 10, 0, 100) == 42

    def test_below_minimum(self):
        assert _as_int(-5, 10, 0, 100) == 0

    def test_above_maximum(self):
        assert _as_int(200, 10, 0, 100) == 100

    def test_invalid_fallback(self):
        assert _as_int("abc", 10, 0, 100) == 10

    def test_none_fallback(self):
        assert _as_int(None, 10, 0, 100) == 10


class TestAsFloat:
    def test_valid_value(self):
        assert _as_float(1.5, 1.0, 0.0, 10.0) == 1.5

    def test_string_coercion(self):
        assert _as_float("2.5", 1.0, 0.0, 10.0) == 2.5

    def test_below_minimum(self):
        assert _as_float(-1.0, 1.0, 0.0, 10.0) == 0.0

    def test_above_maximum(self):
        assert _as_float(20.0, 1.0, 0.0, 10.0) == 10.0

    def test_invalid_fallback(self):
        assert _as_float("nope", 1.0, 0.0, 10.0) == 1.0


# ---------------------------------------------------------------------------
# _resolve_cwd
# ---------------------------------------------------------------------------


class TestResolveCwd:
    def test_empty_string_falls_back_to_home(self):
        result = _resolve_cwd("")
        assert result == Path.home()

    def test_none_falls_back_to_home(self):
        result = _resolve_cwd(None)
        assert result == Path.home()

    def test_nonexistent_path_falls_back_to_home(self):
        result = _resolve_cwd("/nonexistent/path/that/should/not/exist")
        assert result == Path.home()

    def test_valid_directory(self, tmp_path):
        result = _resolve_cwd(str(tmp_path))
        assert result == tmp_path.resolve()


# ---------------------------------------------------------------------------
# _parse_stream_json
# ---------------------------------------------------------------------------


class TestParseStreamJson:
    def test_empty_input(self):
        result = _parse_stream_json("")
        assert result["init"] == {}
        assert result["final"] == {}
        assert result["tool_calls"] == []
        assert result["text"] == ""
        assert result["parse_errors"] == 0

    def test_init_event(self):
        line = json.dumps({"type": "system", "subtype": "init", "model": "claude-sonnet"})
        result = _parse_stream_json(line)
        assert result["init"]["model"] == "claude-sonnet"

    def test_text_extraction(self):
        line = json.dumps({
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": "Hello world"}],
            },
        })
        result = _parse_stream_json(line)
        assert result["text"] == "Hello world"

    def test_tool_call_extraction(self):
        line = json.dumps({
            "type": "assistant",
            "message": {
                "content": [{
                    "type": "tool_use",
                    "name": "read_file",
                    "input": {"path": "/tmp/test.txt"},
                }],
            },
        })
        result = _parse_stream_json(line)
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "read_file"

    def test_result_event(self):
        line = json.dumps({
            "type": "result",
            "result": "Task completed",
            "session_id": "abc123",
        })
        result = _parse_stream_json(line)
        assert result["final"]["result"] == "Task completed"

    def test_parse_errors_counted(self):
        lines = "not-json\nalso-not-json\n"
        result = _parse_stream_json(lines)
        assert result["parse_errors"] == 2
        assert len(result["bad_lines"]) == 2

    def test_bad_lines_capped_at_five(self):
        lines = "\n".join(f"bad-line-{i}" for i in range(10))
        result = _parse_stream_json(lines)
        assert result["parse_errors"] == 10
        assert len(result["bad_lines"]) == 5


# ---------------------------------------------------------------------------
# _redacted_command
# ---------------------------------------------------------------------------


class TestRedactedCommand:
    def test_prompt_redacted(self):
        cmd = ["claude", "-p", "secret prompt", "--model", "sonnet"]
        result = _redacted_command(cmd)
        assert result[2] == "<prompt>"
        assert result[4] == "sonnet"

    def test_no_prompt_flag(self):
        cmd = ["claude", "--model", "sonnet"]
        result = _redacted_command(cmd)
        assert result == cmd

    def test_original_not_mutated(self):
        cmd = ["claude", "-p", "secret"]
        _redacted_command(cmd)
        assert cmd[2] == "secret"


# ---------------------------------------------------------------------------
# _wrap_prompt
# ---------------------------------------------------------------------------


class TestWrapPrompt:
    def test_prompt_included(self):
        result = _wrap_prompt("do something")
        assert "do something" in result

    def test_runtime_context_present(self):
        result = _wrap_prompt("test")
        assert "Hermes Runtime Context" in result
        assert "Response Contract" in result


# ---------------------------------------------------------------------------
# _command_for (security-gating)
# ---------------------------------------------------------------------------


class TestCommandFor:
    def test_skip_permissions_default_off(self):
        """Without env var, --dangerously-skip-permissions must NOT appear."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_DELEGATE_SKIP_PERMISSIONS", None)
            cmd = _command_for("test prompt", 1.0, "sonnet")
        assert "--dangerously-skip-permissions" not in cmd

    def test_skip_permissions_enabled(self):
        """With env var set to '1', flag must appear."""
        with patch.dict(os.environ, {"HERMES_DELEGATE_SKIP_PERMISSIONS": "1"}):
            cmd = _command_for("test prompt", 1.0, "sonnet")
        assert "--dangerously-skip-permissions" in cmd

    def test_skip_permissions_explicit_off(self):
        """With env var set to '0', flag must NOT appear."""
        with patch.dict(os.environ, {"HERMES_DELEGATE_SKIP_PERMISSIONS": "0"}):
            cmd = _command_for("test prompt", 1.0, "sonnet")
        assert "--dangerously-skip-permissions" not in cmd

    def test_model_in_command(self):
        cmd = _command_for("prompt", 2.0, "opus")
        assert "--model" in cmd
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "opus"

    def test_budget_in_command(self):
        cmd = _command_for("prompt", 5.0, "sonnet")
        assert "--max-budget-usd" in cmd
        idx = cmd.index("--max-budget-usd")
        assert cmd[idx + 1] == "5.0000"


# ---------------------------------------------------------------------------
# _existing_add_dirs
# ---------------------------------------------------------------------------


class TestExistingAddDirs:
    def test_nonexistent_dirs_excluded(self):
        """Dirs that don't exist on disk must not appear."""
        with patch("tools.claude_code_delegate_tool.get_skills_dir",
                    return_value=Path("/nonexistent/skills/path")):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("HERMES_DELEGATE_ADD_DIRS", None)
                dirs = _existing_add_dirs()
        assert "/nonexistent/skills/path" not in dirs

    def test_extra_dirs_from_env(self, tmp_path):
        """HERMES_DELEGATE_ADD_DIRS should be respected."""
        extra_dir = tmp_path / "extra"
        extra_dir.mkdir()
        with patch("tools.claude_code_delegate_tool.get_skills_dir",
                    return_value=Path("/nonexistent")):
            with patch.dict(os.environ, {"HERMES_DELEGATE_ADD_DIRS": str(extra_dir)}):
                dirs = _existing_add_dirs()
        assert str(extra_dir.resolve()) in dirs

    def test_hermes_home_not_in_add_dirs(self, tmp_path):
        """get_hermes_home() must NOT appear — it contains secrets."""
        skills = tmp_path / "skills"
        skills.mkdir()
        with patch("tools.claude_code_delegate_tool.get_skills_dir",
                    return_value=skills):
            with patch("tools.claude_code_delegate_tool.get_hermes_home",
                        return_value=tmp_path):
                with patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("HERMES_DELEGATE_ADD_DIRS", None)
                    dirs = _existing_add_dirs()
        # Only skills dir should be present, not hermes home root
        assert str(skills.resolve()) in dirs
        # hermes home itself should NOT be there (unless it equals skills)
        resolved_home = str(tmp_path.resolve())
        resolved_skills = str(skills.resolve())
        if resolved_home != resolved_skills:
            assert resolved_home not in dirs


# ---------------------------------------------------------------------------
# check_claude_code_available
# ---------------------------------------------------------------------------


class TestCheckClaudeCodeAvailable:
    @patch("shutil.which", return_value="/usr/local/bin/claude")
    def test_available(self, mock_which):
        assert check_claude_code_available() is True

    @patch("shutil.which", return_value=None)
    def test_not_available(self, mock_which):
        assert check_claude_code_available() is False


# ---------------------------------------------------------------------------
# delegate_to_claude_code (integration with mock subprocess)
# ---------------------------------------------------------------------------


class TestDelegateToClaudeCode:
    def test_empty_prompt_returns_error(self):
        result = json.loads(delegate_to_claude_code({"prompt": ""}))
        assert "error" in str(result).lower() or result.get("success") is False

    @patch("subprocess.Popen")
    def test_successful_delegation(self, mock_popen_cls, tmp_path):
        stream_output = json.dumps({
            "type": "result",
            "result": "Task done",
            "session_id": "test-session",
            "num_turns": 2,
            "total_cost_usd": 0.05,
        })
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (stream_output, "")
        mock_proc.returncode = 0
        mock_popen_cls.return_value = mock_proc

        with patch("tools.claude_code_delegate_tool._get_log_root", return_value=tmp_path):
            result = json.loads(delegate_to_claude_code({
                "prompt": "test task",
                "timeout_s": 30,
            }))

        assert result["success"] is True
        assert result["status"] == "completed"
        assert "Task done" in result["result"]

    @patch("subprocess.Popen", side_effect=FileNotFoundError)
    def test_missing_cli(self, mock_popen_cls, tmp_path):
        with patch("tools.claude_code_delegate_tool._get_log_root", return_value=tmp_path):
            result = json.loads(delegate_to_claude_code({"prompt": "test"}))

        assert result["success"] is False
        assert result["status"] == "missing_claude_cli"

    def test_schema_has_required_fields(self):
        schema = DELEGATE_TO_CLAUDE_CODE_SCHEMA
        assert schema["name"] == "delegate_to_claude_code"
        assert "prompt" in schema["parameters"]["required"]
        props = schema["parameters"]["properties"]
        assert "prompt" in props
        assert "timeout_s" in props
        assert "max_budget_usd" in props
        assert "model" in props
