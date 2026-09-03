"""Tests for foreground timeout cap in terminal_tool.

Ensures that foreground commands with timeout > FOREGROUND_MAX_TIMEOUT
are rejected with an error suggesting background=true.
"""
import json
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

    def test_zero_timeout_rejected(self):
        """timeout=0 must be rejected, not silently coerced to the default."""
        from tools.terminal_tool import terminal_tool

        with patch("tools.terminal_tool._get_env_config", return_value=_make_env_config()), \
             patch("tools.terminal_tool._start_cleanup_thread"):
            result = json.loads(terminal_tool(command="echo hi", timeout=0))

        assert result.get("error")
        assert "positive" in result["error"]

    def test_negative_timeout_rejected(self):
        """timeout=-1 must be rejected, not fire an immediate '-1s' timeout."""
        from tools.terminal_tool import terminal_tool

        with patch("tools.terminal_tool._get_env_config", return_value=_make_env_config()), \
             patch("tools.terminal_tool._start_cleanup_thread"):
            result = json.loads(terminal_tool(command="echo hi", timeout=-1))

        assert result.get("error")
        assert "positive" in result["error"]


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


    def test_config_default_above_cap_clamped_for_foreground(self):
        """When config default timeout > cap and model passes no timeout, clamp.

        A raised TERMINAL_TIMEOUT default serves dispatcher-owned background
        workflows (#100451); the model expressed no timeout intent, so the
        call is not rejected, but the foreground wait is still bounded by
        FOREGROUND_MAX_TIMEOUT instead of inheriting the raised default.
        """
        from tools.terminal_tool import terminal_tool, FOREGROUND_MAX_TIMEOUT

        # Profile raised TERMINAL_TIMEOUT=900 for background workflows
        with patch("tools.terminal_tool._get_env_config",
                    return_value=_make_env_config(timeout=900)), \
             patch("tools.terminal_tool._start_cleanup_thread"):

            mock_env = MagicMock()
            mock_env.execute.return_value = {"output": "done", "returncode": 0}

            with patch("tools.terminal_tool._active_environments", {"default": mock_env}), \
                 patch("tools.terminal_tool._last_activity", {"default": 0}), \
                 patch("tools.terminal_tool._check_all_guards", return_value={"approved": True}):
                result = json.loads(terminal_tool(command="make build"))

        # Executes (not rejected) with the wait clamped to the cap
        call_kwargs = mock_env.execute.call_args
        assert call_kwargs[1]["timeout"] == FOREGROUND_MAX_TIMEOUT
        assert "error" not in result or result["error"] is None


    def test_config_default_at_or_below_cap_untouched(self):
        """A config default at or below the cap must pass through unchanged."""
        from tools.terminal_tool import terminal_tool

        with patch("tools.terminal_tool._get_env_config",
                    return_value=_make_env_config(timeout=300)), \
             patch("tools.terminal_tool._start_cleanup_thread"):

            mock_env = MagicMock()
            mock_env.execute.return_value = {"output": "done", "returncode": 0}

            with patch("tools.terminal_tool._active_environments", {"default": mock_env}), \
                 patch("tools.terminal_tool._last_activity", {"default": 0}), \
                 patch("tools.terminal_tool._check_all_guards", return_value={"approved": True}):
                result = json.loads(terminal_tool(command="make build"))

        call_kwargs = mock_env.execute.call_args
        assert call_kwargs[1]["timeout"] == 300
        assert "error" not in result or result["error"] is None


    def test_background_spawn_keeps_raised_default(self):
        """Background spawns are not clamped by the foreground cap.

        The raised TERMINAL_TIMEOUT default exists for dispatcher-owned
        background workflows; a background spawn must keep using it.
        """
        from types import SimpleNamespace

        from tools.terminal_tool import terminal_tool
        import tools.process_registry as process_registry_mod

        spawn_calls = []
        create_env_calls = []

        class FakeRegistry:
            pending_watchers = []

            def spawn_local(self, **kwargs):
                spawn_calls.append(kwargs)
                return SimpleNamespace(id="sess-1", pid=4321)

        mock_env = SimpleNamespace(env={})

        def _fake_create_environment(**kwargs):
            create_env_calls.append(kwargs)
            return mock_env

        # Empty environment cache so the spawn runs through _create_environment,
        # the consumer of the raised default on the background path.
        with patch("tools.terminal_tool._get_env_config",
                    return_value=_make_env_config(timeout=900)), \
             patch("tools.terminal_tool._start_cleanup_thread"), \
             patch("tools.terminal_tool._active_environments", {}), \
             patch("tools.terminal_tool._last_activity", {}), \
             patch("tools.terminal_tool._task_env_overrides", {}), \
             patch("tools.terminal_tool._resolve_container_task_id", lambda value: value or "default"), \
             patch("tools.terminal_tool._create_environment", side_effect=_fake_create_environment), \
             patch("tools.terminal_tool._check_all_guards", return_value={"approved": True}), \
             patch.object(process_registry_mod, "process_registry", FakeRegistry()):
            result = json.loads(terminal_tool(command="long-runner", background=True))

        assert result["exit_code"] == 0
        assert len(spawn_calls) == 1
        # The raised default survives into the environment the background
        # process runs in — the cap only bounds foreground waits.
        assert create_env_calls[0]["timeout"] == 900


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
