"""Tests for the foreground hard-timeout watchdog in terminal_tool.

The tool's own timeout machinery only covers the phase while the command is
still running; finalization can still block without a bound on a stdout pipe a
spawned GUI child keeps open (e.g. Windows ``cmd start "" "app.exe"``), wedging
the turn. The hard watchdog returns exit 124 past
timeout + HARD_TIMEOUT_GRACE instead of blocking forever.
"""
import json
import time
from unittest.mock import patch, MagicMock


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


def _run_terminal(mock_env, tmp_path, command, timeout=None):
    """Run terminal_tool through its real entry point with a mock environment."""
    from tools.terminal_tool import terminal_tool

    kwargs = {"timeout": timeout} if timeout is not None else {}
    with patch("tools.terminal_tool._get_env_config", return_value=_make_env_config(cwd=str(tmp_path))), \
         patch("tools.terminal_tool._start_cleanup_thread"), \
         patch("tools.terminal_tool._active_environments", {"default": mock_env}), \
         patch("tools.terminal_tool._last_activity", {"default": 0}), \
         patch("tools.terminal_tool._check_all_guards", return_value={"approved": True}):
        return json.loads(terminal_tool(command=command, **kwargs))


class TestHardTimeoutWatchdog:
    """The foreground hard watchdog bounds the execute call even when its own
    timeout machinery can no longer fire."""

    def test_fast_execute_unaffected(self, tmp_path):
        """A normal, fast execute call is returned as-is (no behavior change)."""
        mock_env = MagicMock()
        mock_env.execute.return_value = {"output": "ok", "returncode": 0}

        result = _run_terminal(mock_env, tmp_path, "echo ok")

        assert result.get("error") is None

    def test_wedged_execute_returns_124_past_grace(self, tmp_path, monkeypatch):
        """An execute call that never returns is hard-failed at
        timeout + HARD_TIMEOUT_GRACE with exit 124 instead of blocking."""
        import tools.terminal_tool as tt

        monkeypatch.setattr(tt, "HARD_TIMEOUT_GRACE", 1)
        mock_env = MagicMock()

        def _wedge(*_args, **_kwargs):
            time.sleep(6)
            return {"output": "late", "returncode": 0}

        mock_env.execute.side_effect = _wedge
        started = time.monotonic()
        result = _run_terminal(mock_env, tmp_path, "echo hi", timeout=1)
        elapsed = time.monotonic() - started

        assert result["exit_code"] == 124
        assert "hard watchdog" in result["error"]
        # Returned at timeout + grace, not when the wedged call finally finished.
        assert elapsed < 4

    def test_execute_exception_still_propagates(self, tmp_path, monkeypatch):
        """Exceptions from execute keep flowing through the retry path (the
        watchdog must not swallow them)."""
        monkeypatch.setattr(time, "sleep", lambda *_args, **_kwargs: None)  # skip retry backoff
        mock_env = MagicMock()
        mock_env.execute.side_effect = RuntimeError("backend exploded")

        result = _run_terminal(mock_env, tmp_path, "echo hi", timeout=5)

        assert result["exit_code"] != 124
        assert "backend exploded" in result["error"]
