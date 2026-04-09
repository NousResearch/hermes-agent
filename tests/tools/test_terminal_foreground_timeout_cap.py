"""Tests for foreground timeout clamping in terminal_tool.

Ensures that foreground commands have a hard timeout cap to prevent
a single tool call from blocking the entire agent session.
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
    """FOREGROUND_MAX_TIMEOUT prevents foreground commands from blocking too long."""

    def test_foreground_timeout_clamped_to_max(self):
        """When model requests timeout > FOREGROUND_MAX_TIMEOUT, it's clamped."""
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
                    timeout=9999,  # Way above max
                ))

            # Verify the timeout was clamped
            call_kwargs = mock_env.execute.call_args
            assert call_kwargs[1]["timeout"] == FOREGROUND_MAX_TIMEOUT
            assert result.get("timeout_note") is not None
            assert "clamped" in result["timeout_note"]
            assert "9999" in result["timeout_note"]
            assert "background=true" in result["timeout_note"]

    def test_foreground_timeout_within_max_not_clamped(self):
        """When model requests timeout <= FOREGROUND_MAX_TIMEOUT, no clamping."""
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
            assert "timeout_note" not in result

    def test_config_default_exceeds_cap_no_model_timeout(self):
        """When config default timeout > cap and model passes no timeout, clamping fires."""
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

            # Should be clamped
            call_kwargs = mock_env.execute.call_args
            assert call_kwargs[1]["timeout"] == FOREGROUND_MAX_TIMEOUT
            # Note should reference the original 900s, NOT "None"
            note = result.get("timeout_note", "")
            assert "900" in note, f"Expected '900' in timeout_note but got: {note!r}"
            assert "None" not in note, f"timeout_note contains 'None': {note!r}"
            assert "clamped" in note

    def test_background_not_clamped(self):
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

            # Background should NOT be clamped
            assert result.get("timeout_note") is None

    def test_default_timeout_not_clamped(self):
        """Default timeout (180s) should not trigger clamping."""
        from tools.terminal_tool import terminal_tool, FOREGROUND_MAX_TIMEOUT

        # 180 < 600, so no clamping
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
            assert "timeout_note" not in result


class TestForegroundMaxTimeoutConstant:
    """Verify the FOREGROUND_MAX_TIMEOUT constant and schema."""

    def test_default_value_is_600(self):
        """Default FOREGROUND_MAX_TIMEOUT is 600 when env var is not set."""
        from tools.terminal_tool import FOREGROUND_MAX_TIMEOUT
        # Module-level constant should be 600 in a clean test environment.
        # If TERMINAL_MAX_FOREGROUND_TIMEOUT is set, it may differ — but the
        # conftest _isolate_hermes_home fixture ensures a clean env for tests.
        assert FOREGROUND_MAX_TIMEOUT == 600

    def test_schema_mentions_max(self):
        """Tool schema description should mention the max timeout."""
        from tools.terminal_tool import TERMINAL_SCHEMA, FOREGROUND_MAX_TIMEOUT
        timeout_desc = TERMINAL_SCHEMA["parameters"]["properties"]["timeout"]["description"]
        assert str(FOREGROUND_MAX_TIMEOUT) in timeout_desc
        assert "max" in timeout_desc.lower()
