"""Tests for the Claude Code CLI delegation tool."""

import json
import subprocess
from unittest.mock import patch, MagicMock

import pytest

from tools.claude_code_tool import (
    claude_code,
    claude_code_streaming,
    _handle_claude_code_dispatch,
    check_claude_code_available,
    _build_claude_command,
    _parse_json_output,
    get_claude_session,
    set_claude_session,
    clear_claude_session,
    DEFAULT_TIMEOUT,
    MAX_PROMPT_LENGTH,
)


class TestCheckAvailability:
    def test_available_when_claude_on_path(self):
        with patch("shutil.which", return_value="/usr/local/bin/claude"):
            assert check_claude_code_available() is True

    def test_unavailable_when_not_on_path(self):
        with patch("shutil.which", return_value=None):
            assert check_claude_code_available() is False


class TestBuildCommand:
    def test_basic_command(self):
        cmd = _build_claude_command("hello")
        assert cmd == ["claude", "-p", "--output-format", "json", "hello"]

    def test_with_model(self):
        cmd = _build_claude_command("hello", model="sonnet")
        assert "--model" in cmd
        assert "sonnet" in cmd

    def test_with_max_turns(self):
        cmd = _build_claude_command("hello", max_turns=5)
        assert "--max-turns" in cmd
        assert "5" in cmd

    def test_with_session_resume(self):
        cmd = _build_claude_command("hello", session_id="abc-123")
        assert "--resume" in cmd
        assert "abc-123" in cmd

    def test_zero_max_turns_ignored(self):
        cmd = _build_claude_command("hello", max_turns=0)
        assert "--max-turns" not in cmd

    def test_stream_format(self):
        cmd = _build_claude_command("hello", output_format="stream-json")
        assert "--output-format" in cmd
        assert "stream-json" in cmd


class TestSessionStore:
    def test_set_and_get(self):
        set_claude_session("test-key", "session-123")
        assert get_claude_session("test-key") == "session-123"
        clear_claude_session("test-key")

    def test_get_nonexistent(self):
        assert get_claude_session("nonexistent") is None

    def test_clear(self):
        set_claude_session("test-clear", "session-456")
        clear_claude_session("test-clear")
        assert get_claude_session("test-clear") is None

    def test_overwrite(self):
        set_claude_session("test-overwrite", "old")
        set_claude_session("test-overwrite", "new")
        assert get_claude_session("test-overwrite") == "new"
        clear_claude_session("test-overwrite")


class TestParseJsonOutput:
    def test_parse_full_json_output(self):
        data = json.dumps([
            {"type": "system", "subtype": "init", "session_id": "sess-abc"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "Hello world"}]}},
            {"type": "result", "result": "Hello world", "session_id": "sess-abc",
             "total_cost_usd": 0.01, "duration_ms": 1500},
        ])
        parsed = _parse_json_output(data)
        assert parsed["result_text"] == "Hello world"
        assert parsed["session_id"] == "sess-abc"
        assert parsed["cost_usd"] == 0.01
        assert parsed["duration_ms"] == 1500

    def test_parse_plain_text_fallback(self):
        parsed = _parse_json_output("just plain text")
        assert parsed["result_text"] == "just plain text"
        assert parsed["session_id"] is None

    def test_parse_empty_content(self):
        data = json.dumps([
            {"type": "system", "subtype": "init", "session_id": "sess-empty"},
            {"type": "result", "result": "", "session_id": "sess-empty"},
        ])
        parsed = _parse_json_output(data)
        assert parsed["result_text"] == ""
        assert parsed["session_id"] == "sess-empty"


class TestClaudeCode:
    def test_empty_prompt(self):
        result = claude_code(prompt="")
        assert result["success"] is False
        assert "empty" in result["error"].lower()

    def test_prompt_too_long(self):
        result = claude_code(prompt="x" * (MAX_PROMPT_LENGTH + 1))
        assert result["success"] is False
        assert "too long" in result["error"].lower()

    def test_cli_not_found(self):
        with patch("tools.claude_code_tool.check_claude_code_available", return_value=False):
            result = claude_code(prompt="hello")
            assert result["success"] is False
            assert "not found" in result["error"].lower()

    def test_successful_execution_with_json(self):
        json_output = json.dumps([
            {"type": "system", "subtype": "init", "session_id": "sess-ok"},
            {"type": "result", "result": "Done!", "session_id": "sess-ok",
             "total_cost_usd": 0.005, "duration_ms": 800},
        ])
        mock_result = MagicMock()
        mock_result.stdout = json_output
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("tools.claude_code_tool.check_claude_code_available", return_value=True), \
             patch("subprocess.run", return_value=mock_result):
            result = claude_code(prompt="do something")
            assert result["success"] is True
            assert result["output"] == "Done!"
            assert result["session_id"] == "sess-ok"
            assert result["cost_usd"] == 0.005

    def test_session_persistence(self):
        json_output = json.dumps([
            {"type": "system", "subtype": "init", "session_id": "sess-persist"},
            {"type": "result", "result": "ok", "session_id": "sess-persist"},
        ])
        mock_result = MagicMock()
        mock_result.stdout = json_output
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("tools.claude_code_tool.check_claude_code_available", return_value=True), \
             patch("subprocess.run", return_value=mock_result):
            claude_code(prompt="first call", session_key="hermes-test")
            assert get_claude_session("hermes-test") == "sess-persist"
            clear_claude_session("hermes-test")

    def test_session_resume_in_command(self):
        set_claude_session("hermes-resume", "old-session-id")
        json_output = json.dumps([
            {"type": "system", "subtype": "init", "session_id": "old-session-id"},
            {"type": "result", "result": "resumed", "session_id": "old-session-id"},
        ])
        mock_result = MagicMock()
        mock_result.stdout = json_output
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("tools.claude_code_tool.check_claude_code_available", return_value=True), \
             patch("subprocess.run", return_value=mock_result) as mock_run:
            claude_code(prompt="continue", session_key="hermes-resume")
            cmd = mock_run.call_args[0][0]
            assert "--resume" in cmd
            assert "old-session-id" in cmd
            clear_claude_session("hermes-resume")

    def test_nonzero_exit_code(self):
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = "Authentication failed"
        mock_result.returncode = 1

        with patch("tools.claude_code_tool.check_claude_code_available", return_value=True), \
             patch("subprocess.run", return_value=mock_result):
            result = claude_code(prompt="do something")
            assert result["success"] is False
            assert "Authentication failed" in result["error"]

    def test_timeout(self):
        with patch("tools.claude_code_tool.check_claude_code_available", return_value=True), \
             patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=10)):
            result = claude_code(prompt="long task", timeout=10)
            assert result["success"] is False
            assert "timed out" in result["error"].lower()

    def test_invalid_cwd(self):
        with patch("tools.claude_code_tool.check_claude_code_available", return_value=True):
            result = claude_code(prompt="test", cwd="/nonexistent/path/xyz")
            assert result["success"] is False
            assert "does not exist" in result["error"]

    def test_timeout_capped_at_600(self):
        json_output = json.dumps([{"type": "result", "result": "ok"}])
        mock_result = MagicMock()
        mock_result.stdout = json_output
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("tools.claude_code_tool.check_claude_code_available", return_value=True), \
             patch("subprocess.run", return_value=mock_result) as mock_run:
            claude_code(prompt="test", timeout=9999)
            call_kwargs = mock_run.call_args
            assert call_kwargs.kwargs["timeout"] == 600


class TestDispatchHandler:
    def test_handler_returns_json(self):
        json_output = json.dumps([
            {"type": "result", "result": "dispatch result", "session_id": "sess-d"},
        ])
        mock_result = MagicMock()
        mock_result.stdout = json_output
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("tools.claude_code_tool.check_claude_code_available", return_value=True), \
             patch("subprocess.run", return_value=mock_result):
            output = _handle_claude_code_dispatch({"prompt": "test"})
            parsed = json.loads(output)
            assert parsed["success"] is True
            assert parsed["output"] == "dispatch result"

    def test_handler_receives_args_dict(self):
        with patch("tools.claude_code_tool.check_claude_code_available", return_value=True), \
             patch("tools.claude_code_tool.claude_code", return_value={"success": True, "output": "ok"}) as mock_cc:
            _handle_claude_code_dispatch({"prompt": "test", "model": "", "max_turns": 0, "cwd": "", "timeout": 0})
            mock_cc.assert_called_once_with(
                prompt="test", model=None, max_turns=None, cwd=None, timeout=None, session_key=None,
            )
