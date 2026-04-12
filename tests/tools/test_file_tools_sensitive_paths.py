#!/usr/bin/env python3
"""Tests for sensitive-path read blocking in read_file_tool.

Verifies that auth.json, mcp-tokens/, and .env inside HERMES_HOME
are blocked from being read, while normal files and identically-named
files outside HERMES_HOME remain accessible.

Run with:  python -m pytest tests/tools/test_file_tools_sensitive_paths.py -v
"""

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from tools.file_tools import read_file_tool, clear_read_tracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeReadResult:
    """Minimal stand-in for FileOperations.read_file return value."""

    def __init__(self, content="line1\nline2\n", total_lines=2, file_size=100):
        self.content = content
        self._total_lines = total_lines
        self._file_size = file_size

    def to_dict(self):
        return {
            "content": self.content,
            "total_lines": self._total_lines,
            "file_size": self._file_size,
        }


def _make_fake_ops(content="hello\n", total_lines=1, file_size=6):
    fake = MagicMock()
    fake.read_file = lambda path, offset=1, limit=500: _FakeReadResult(
        content=content, total_lines=total_lines, file_size=file_size,
    )
    return fake


@pytest.fixture(autouse=True)
def _clean_tracker():
    """Reset read tracker between tests."""
    clear_read_tracker()
    yield
    clear_read_tracker()


# ---------------------------------------------------------------------------
# Sensitive path blocking inside HERMES_HOME
# ---------------------------------------------------------------------------

class TestSensitivePathBlocking:
    """auth.json, mcp-tokens/, and .env must be blocked inside HERMES_HOME."""

    def test_auth_json_blocked(self, tmp_path):
        """Reading auth.json inside HERMES_HOME is blocked."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        auth_file = hermes_home / "auth.json"
        auth_file.write_text('{"api_key": "sk-secret"}')

        with patch("hermes_constants.get_hermes_home", return_value=hermes_home):
            result = json.loads(read_file_tool(str(auth_file), task_id="auth_test"))

        assert "error" in result
        assert "credentials" in result["error"].lower() or "denied" in result["error"].lower()

    def test_mcp_tokens_json_blocked(self, tmp_path):
        """Reading a token file inside mcp-tokens/ is blocked."""
        hermes_home = tmp_path / ".hermes"
        tokens_dir = hermes_home / "mcp-tokens"
        tokens_dir.mkdir(parents=True)
        token_file = tokens_dir / "github.json"
        token_file.write_text('{"access_token": "ghp_secret"}')

        with patch("hermes_constants.get_hermes_home", return_value=hermes_home):
            result = json.loads(read_file_tool(str(token_file), task_id="mcp_test"))

        assert "error" in result
        assert "denied" in result["error"].lower()

    def test_mcp_tokens_nested_blocked(self, tmp_path):
        """Reading nested files inside mcp-tokens/ is also blocked."""
        hermes_home = tmp_path / ".hermes"
        nested_dir = hermes_home / "mcp-tokens" / "providers"
        nested_dir.mkdir(parents=True)
        token_file = nested_dir / "oauth.json"
        token_file.write_text('{"refresh_token": "secret"}')

        with patch("hermes_constants.get_hermes_home", return_value=hermes_home):
            result = json.loads(read_file_tool(str(token_file), task_id="nested_mcp"))

        assert "error" in result
        assert "denied" in result["error"].lower()

    def test_dotenv_blocked(self, tmp_path):
        """Reading .env inside HERMES_HOME is blocked."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        env_file = hermes_home / ".env"
        env_file.write_text("API_KEY=sk-secret\nDATABASE_URL=postgres://...")

        with patch("hermes_constants.get_hermes_home", return_value=hermes_home):
            result = json.loads(read_file_tool(str(env_file), task_id="env_test"))

        assert "error" in result
        assert "credentials" in result["error"].lower() or "denied" in result["error"].lower()

    def test_error_message_informative(self, tmp_path):
        """Blocked reads should include an informative error message."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        auth_file = hermes_home / "auth.json"
        auth_file.write_text('{}')

        with patch("hermes_constants.get_hermes_home", return_value=hermes_home):
            result = json.loads(read_file_tool(str(auth_file), task_id="msg_test"))

        assert "error" in result
        # Should mention credentials and suggest an alternative
        assert "credentials" in result["error"].lower()
        assert "auth" in result["error"].lower() or "settings" in result["error"].lower()


# ---------------------------------------------------------------------------
# Normal files inside HERMES_HOME should still be readable
# ---------------------------------------------------------------------------

class TestNonSensitivePathsAllowed:
    """Normal files inside HERMES_HOME must NOT be blocked."""

    @patch("tools.file_tools._get_file_ops")
    def test_session_log_readable(self, mock_ops, tmp_path):
        """Session log files inside HERMES_HOME are not blocked."""
        hermes_home = tmp_path / ".hermes"
        logs_dir = hermes_home / "sessions"
        logs_dir.mkdir(parents=True)
        log_file = logs_dir / "session_abc.log"
        log_file.write_text("session started")

        mock_ops.return_value = _make_fake_ops(
            content="session started", file_size=15,
        )

        with patch("hermes_constants.get_hermes_home", return_value=hermes_home):
            result = json.loads(read_file_tool(str(log_file), task_id="log_test"))

        assert "error" not in result
        assert "content" in result

    @patch("tools.file_tools._get_file_ops")
    def test_config_yaml_readable(self, mock_ops, tmp_path):
        """config.yaml inside HERMES_HOME is not blocked."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_file = hermes_home / "config.yaml"
        config_file.write_text("model: default")

        mock_ops.return_value = _make_fake_ops(
            content="model: default", file_size=14,
        )

        with patch("hermes_constants.get_hermes_home", return_value=hermes_home):
            result = json.loads(read_file_tool(str(config_file), task_id="cfg_test"))

        assert "error" not in result
        assert "content" in result


# ---------------------------------------------------------------------------
# Files with the same name outside HERMES_HOME should NOT be blocked
# ---------------------------------------------------------------------------

class TestOutsideHermesHomeNotBlocked:
    """Identically-named files outside HERMES_HOME must remain accessible."""

    @patch("tools.file_tools._get_file_ops")
    def test_auth_json_outside_hermes_readable(self, mock_ops, tmp_path):
        """auth.json in a project directory (not HERMES_HOME) is allowed."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()

        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        auth_file = project_dir / "auth.json"
        auth_file.write_text('{"type": "service_account"}')

        mock_ops.return_value = _make_fake_ops(
            content='{"type": "service_account"}', file_size=28,
        )

        with patch("hermes_constants.get_hermes_home", return_value=hermes_home):
            result = json.loads(read_file_tool(str(auth_file), task_id="ext_auth"))

        assert "error" not in result
        assert "content" in result

    @patch("tools.file_tools._get_file_ops")
    def test_dotenv_outside_hermes_readable(self, mock_ops, tmp_path):
        """.env in a project directory (not HERMES_HOME) is allowed."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()

        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        env_file = project_dir / ".env"
        env_file.write_text("PORT=3000")

        mock_ops.return_value = _make_fake_ops(
            content="PORT=3000", file_size=10,
        )

        with patch("hermes_constants.get_hermes_home", return_value=hermes_home):
            result = json.loads(read_file_tool(str(env_file), task_id="ext_env"))

        assert "error" not in result
        assert "content" in result

    @patch("tools.file_tools._get_file_ops")
    def test_mcp_tokens_dir_outside_hermes_readable(self, mock_ops, tmp_path):
        """mcp-tokens/ dir outside HERMES_HOME is allowed."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()

        other_dir = tmp_path / "other" / "mcp-tokens"
        other_dir.mkdir(parents=True)
        token_file = other_dir / "token.json"
        token_file.write_text('{"token": "test"}')

        mock_ops.return_value = _make_fake_ops(
            content='{"token": "test"}', file_size=17,
        )

        with patch("hermes_constants.get_hermes_home", return_value=hermes_home):
            result = json.loads(read_file_tool(str(token_file), task_id="ext_mcp"))

        assert "error" not in result
        assert "content" in result
