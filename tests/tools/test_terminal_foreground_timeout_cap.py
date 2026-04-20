"""Tests for foreground timeout cap in terminal_tool.

Ensures that foreground commands with timeout > FOREGROUND_MAX_TIMEOUT
are rejected with an error suggesting background=true.
"""
import json
import os
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Shared test config dict — mirrors _get_env_config() return shape.
# ---------------------------------------------------------------------------
def _make_env_config(**overrides):
    """Return a minimal _get_env_config()-shaped dict with optional overrides."""
    config = {
        "env_type": "local",
        "timeout": 180,
        "cwd": "/tmp",
        "host_cwd": None,
        "modal_mode": "auto",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
    }
    config.update(overrides)
    return config


class TestForegroundTimeoutCap:
    """FOREGROUND_MAX_TIMEOUT rejects foreground commands that exceed it."""

    def test_foreground_timeout_rejected_above_max(self):
        """When model requests timeout > FOREGROUND_MAX_TIMEOUT, return error."""
        from tools.terminal_tool import terminal_tool, FOREGROUND_MAX_TIMEOUT

        with patch("tools.terminal_tool._get_env_config", return_value=_make_env_config()), \
             patch("tools.terminal_tool._start_cleanup_thread"):

            result = json.loads(terminal_tool(
                command="echo hello",
                timeout=9999,  # Way above max
            ))

        assert "error" in result
        assert "9999" in result["error"]
        assert str(FOREGROUND_MAX_TIMEOUT) in result["error"]
        assert "background=true" in result["error"]

    def test_foreground_rejects_shell_level_background_wrappers(self):
        """Foreground nohup/disown/setsid commands should be redirected to background mode."""
        from tools.terminal_tool import terminal_tool

        with patch("tools.terminal_tool._get_env_config", return_value=_make_env_config()), \
             patch("tools.terminal_tool._start_cleanup_thread"):

            result = json.loads(terminal_tool(
                command="nohup pnpm dev > /tmp/sg-server.log 2>&1 &",
            ))

        assert result["exit_code"] == -1
        assert "background=true" in result["error"]
        assert "nohup" in result["error"].lower()

    def test_foreground_rejects_long_lived_server_command(self):
        """Foreground dev server commands should be redirected to background mode."""
        from tools.terminal_tool import terminal_tool

        with patch("tools.terminal_tool._get_env_config", return_value=_make_env_config()), \
             patch("tools.terminal_tool._start_cleanup_thread"):

            result = json.loads(terminal_tool(command="pnpm dev"))

        assert result["exit_code"] == -1
        assert "long-lived" in result["error"].lower()
        assert "background=true" in result["error"]

    def test_foreground_allows_help_variant_for_server_command(self):
        """Informational variants like '--help' should not be blocked."""
        from tools.terminal_tool import terminal_tool

        with patch("tools.terminal_tool._get_env_config", return_value=_make_env_config()), \
             patch("tools.terminal_tool._start_cleanup_thread"):

            mock_env = MagicMock()
            mock_env.execute.return_value = {"output": "usage", "returncode": 0}

            with patch("tools.terminal_tool._active_environments", {"default": mock_env}), \
                 patch("tools.terminal_tool._last_activity", {"default": 0}), \
                 patch("tools.terminal_tool._check_all_guards", return_value={"approved": True}):
                result = json.loads(terminal_tool(command="pnpm dev --help"))

        assert result["error"] is None
        call_kwargs = mock_env.execute.call_args
        assert call_kwargs[0][0] == "pnpm dev --help"

    def test_foreground_timeout_within_max_executes(self):
        """When model requests timeout <= FOREGROUND_MAX_TIMEOUT, execute normally."""
        from tools.terminal_tool import terminal_tool

        with patch("tools.terminal_tool._get_env_config", return_value=_make_env_config()), \
             patch("tools.terminal_tool._start_cleanup_thread"):

            mock_env = MagicMock()
            mock_env.execute.return_value = {"output": "done", "returncode": 0}

            with patch("tools.terminal_tool._active_environments", {"default": mock_env}), \
                 patch("tools.terminal_tool._last_activity", {"default": 0}), \
                 patch("tools.terminal_tool._check_all_guards", return_value={"approved": True}):
                result = json.loads(terminal_tool(
                    command="echo hello",
                    timeout=300,  # Within max
                ))

        call_kwargs = mock_env.execute.call_args
        assert call_kwargs[1]["timeout"] == 300
        assert "error" not in result or result["error"] is None

    def test_config_default_above_cap_not_rejected(self):
        """When config default timeout > cap but model passes no timeout, execute normally.

        Only the model's explicit timeout parameter triggers rejection,
        not the user's configured default.
        """
        from tools.terminal_tool import terminal_tool, FOREGROUND_MAX_TIMEOUT

        # User configured TERMINAL_TIMEOUT=900 in their env
        with patch("tools.terminal_tool._get_env_config",
                    return_value=_make_env_config(timeout=900)), \
             patch("tools.terminal_tool._start_cleanup_thread"):

            mock_env = MagicMock()
            mock_env.execute.return_value = {"output": "done", "returncode": 0}

            with patch("tools.terminal_tool._active_environments", {"default": mock_env}), \
                 patch("tools.terminal_tool._last_activity", {"default": 0}), \
                 patch("tools.terminal_tool._check_all_guards", return_value={"approved": True}):
                result = json.loads(terminal_tool(command="make build"))

        # Should execute with the config default, NOT be rejected
        call_kwargs = mock_env.execute.call_args
        assert call_kwargs[1]["timeout"] == 900
        assert "error" not in result or result["error"] is None

    def test_background_not_rejected(self):
        """Background commands should NOT be subject to foreground timeout cap."""
        from tools.terminal_tool import terminal_tool

        with patch("tools.terminal_tool._get_env_config", return_value=_make_env_config()), \
             patch("tools.terminal_tool._start_cleanup_thread"):

            mock_env = MagicMock()
            mock_env.env = {}
            mock_proc_session = MagicMock()
            mock_proc_session.id = "test-123"
            mock_proc_session.pid = 1234

            mock_registry = MagicMock()
            mock_registry.spawn_local.return_value = mock_proc_session

            with patch("tools.terminal_tool._active_environments", {"default": mock_env}), \
                 patch("tools.terminal_tool._last_activity", {"default": 0}), \
                 patch("tools.terminal_tool._check_all_guards", return_value={"approved": True}), \
                 patch("tools.process_registry.process_registry", mock_registry), \
                 patch("tools.approval.get_current_session_key", return_value=""):
                result = json.loads(terminal_tool(
                    command="python server.py",
                    background=True,
                    timeout=9999,
                ))

        # Background should NOT be rejected
        assert "error" not in result or result["error"] is None

    def test_default_timeout_not_rejected(self):
        """Default timeout (180s) should not trigger rejection."""
        from tools.terminal_tool import terminal_tool, FOREGROUND_MAX_TIMEOUT

        # 180 < 600, so no rejection
        assert 180 < FOREGROUND_MAX_TIMEOUT

        with patch("tools.terminal_tool._get_env_config", return_value=_make_env_config()), \
             patch("tools.terminal_tool._start_cleanup_thread"):

            mock_env = MagicMock()
            mock_env.execute.return_value = {"output": "done", "returncode": 0}

            with patch("tools.terminal_tool._active_environments", {"default": mock_env}), \
                 patch("tools.terminal_tool._last_activity", {"default": 0}), \
                 patch("tools.terminal_tool._check_all_guards", return_value={"approved": True}):
                result = json.loads(terminal_tool(command="echo hello"))

        call_kwargs = mock_env.execute.call_args
        assert call_kwargs[1]["timeout"] == 180
        assert "error" not in result or result["error"] is None

    def test_exactly_at_max_not_rejected(self):
        """Timeout exactly at FOREGROUND_MAX_TIMEOUT should execute normally."""
        from tools.terminal_tool import terminal_tool, FOREGROUND_MAX_TIMEOUT

        with patch("tools.terminal_tool._get_env_config", return_value=_make_env_config()), \
             patch("tools.terminal_tool._start_cleanup_thread"):

            mock_env = MagicMock()
            mock_env.execute.return_value = {"output": "done", "returncode": 0}

            with patch("tools.terminal_tool._active_environments", {"default": mock_env}), \
                 patch("tools.terminal_tool._last_activity", {"default": 0}), \
                 patch("tools.terminal_tool._check_all_guards", return_value={"approved": True}):
                result = json.loads(terminal_tool(
                    command="echo hello",
                    timeout=FOREGROUND_MAX_TIMEOUT,  # Exactly at limit
                ))

        call_kwargs = mock_env.execute.call_args
        assert call_kwargs[1]["timeout"] == FOREGROUND_MAX_TIMEOUT
        assert "error" not in result or result["error"] is None


class TestForegroundMaxTimeoutConstant:
    """Verify the FOREGROUND_MAX_TIMEOUT constant and schema."""

    def test_default_value_is_600(self):
        """Default FOREGROUND_MAX_TIMEOUT is 600 when env var is not set."""
        from tools.terminal_tool import FOREGROUND_MAX_TIMEOUT
        assert FOREGROUND_MAX_TIMEOUT == 600

    def test_schema_mentions_max(self):
        """Tool schema description should mention the max timeout."""
        from tools.terminal_tool import TERMINAL_SCHEMA, FOREGROUND_MAX_TIMEOUT
        timeout_desc = TERMINAL_SCHEMA["parameters"]["properties"]["timeout"]["description"]
        assert str(FOREGROUND_MAX_TIMEOUT) in timeout_desc
        assert "background=true" in timeout_desc
