"""Tests for foreground timeout cap in terminal_tool.

A foreground command with timeout > FOREGROUND_MAX_TIMEOUT is promoted to a tracked background
process with notify_on_complete (never refused: in one 1,393-agent run 454 refusals were every one
re-sent lower/split/background, 251 of them test suites).
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

    def test_foreground_timeout_above_max_is_promoted_to_tracked_background(self, tmp_path, monkeypatch):
        """Real local backend, real registry: the command runs (once), the result is a background
        session with notify_on_complete and a note naming the requested and cap seconds."""
        import time
        from tools.terminal_tool import terminal_tool, FOREGROUND_MAX_TIMEOUT

        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hh"))
        marker = tmp_path / "ran"
        with patch("tools.terminal_tool._get_env_config", return_value=_make_env_config(cwd=str(tmp_path))), \
             patch("tools.terminal_tool._start_cleanup_thread"), \
             patch("tools.terminal_tool._check_all_guards", return_value={"approved": True}):
            result = json.loads(terminal_tool(command=f"echo x >> {marker}", timeout=9999))

        assert result.get("error") is None
        assert result["output"] == "Background process started" and result["session_id"].startswith("proc_")
        assert result["notify_on_complete"] is True
        assert "9999" in result["promoted_from_foreground"]
        assert str(FOREGROUND_MAX_TIMEOUT) in result["promoted_from_foreground"]
        deadline = time.time() + 10
        while not marker.exists() and time.time() < deadline:
            time.sleep(0.05)
        assert marker.read_text().count("x") == 1  # ran exactly once, in the background

    def test_shell_backgrounding_is_still_refused(self):
        """`&`/nohup need the command rewritten; the tool cannot do that safely, so it still refuses."""
        from tools.terminal_tool import terminal_tool

        with patch("tools.terminal_tool._get_env_config", return_value=_make_env_config()), \
             patch("tools.terminal_tool._start_cleanup_thread"):
            result = json.loads(terminal_tool(command="sleep 5 &"))
        assert "'&' backgrounding" in result["error"]

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


    def test_config_default_above_cap_not_rejected(self):
        """When config default timeout > cap but model passes no timeout, execute normally.

        Only the model's explicit timeout parameter triggers rejection,
        not the user's configured default.
        """
        from tools.terminal_tool import terminal_tool

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
        assert "background process" in timeout_desc


class TestPromotionKeepsTheDetachmentGuard:
    def test_over_cap_timeout_with_shell_backgrounding_is_still_refused(self):
        """Independent-review witness: a promoted `cmd &` started a tracked shell that exited at once
        while the payload ran untracked, defeating the guidance the refusal exists for."""
        from tools.terminal_tool import terminal_tool

        with patch("tools.terminal_tool._get_env_config", return_value=_make_env_config()), \
             patch("tools.terminal_tool._start_cleanup_thread"):
            result = json.loads(terminal_tool(command="sleep 5 &", timeout=9999))
            result2 = json.loads(terminal_tool(command="nohup make test", timeout=9999))
        assert "'&' backgrounding" in result["error"]
        assert "nohup" in result2["error"]

    def test_note_does_not_promise_a_notification_the_session_cannot_receive(self):
        from tools.terminal_tool import _with_promoted_note

        kept = json.loads(_with_promoted_note(json.dumps({"session_id": "proc_x", "error": None, "notify_on_complete": True}), 900))
        assert "arrives as a notification" in kept["promoted_from_foreground"]
        dropped = json.loads(_with_promoted_note(json.dumps({"session_id": "proc_x", "error": None, "notify_on_complete": False}), 900))
        assert "cannot receive completion notifications" in dropped["promoted_from_foreground"]
        assert "poll" in dropped["promoted_from_foreground"]
