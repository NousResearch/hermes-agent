"""Remote terminal backend graceful degradation (terminal.degraded_mode).

Connection-class infrastructure failures (SSH unreachable, Docker daemon
down) must come back to the model as a structured ``status: "degraded"``
tool result with a reason and a retry hint — not as a raised traceback
blob.  Command failures (nonzero exit codes) are NOT infrastructure
failures and must stay untouched.  ``terminal.degraded_mode: fail``
preserves the historical raise/traceback behavior for anyone relying
on it.

Inspired by: Claude Cowork degraded-backend behavior (idea-level,
docs-only evidence).
"""

import json
import subprocess
from unittest.mock import MagicMock

import pytest

from tools.environments.base import EnvironmentConnectionError


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """Isolated HERMES_HOME + a clean environment cache for terminal_tool."""
    import tools.terminal_tool as tt

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    # The one-shot config bridge would overwrite our TERMINAL_* test vars
    # from the developer's real config.yaml; mark it as already attempted.
    monkeypatch.setattr(tt, "_terminal_config_bridge_attempted", True)

    def _clear():
        with tt._env_lock:
            tt._active_environments.clear()
            tt._last_activity.clear()

    _clear()
    yield tt
    _clear()


def _mock_ssh_unreachable(monkeypatch, stderr="ssh: connect to host unreachable.invalid port 22: Connection refused"):
    """Make every ssh subprocess in the ssh backend fail like a dead host."""
    monkeypatch.setattr("tools.environments.ssh.shutil.which", lambda _x: "/usr/bin/ssh")
    monkeypatch.setattr(
        "tools.environments.ssh.subprocess.run",
        lambda *a, **k: subprocess.CompletedProcess([], 255, stdout="", stderr=stderr),
    )


def _ssh_backend_env(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_SSH_HOST", "unreachable.invalid")
    monkeypatch.setenv("TERMINAL_SSH_USER", "nobody")
    monkeypatch.delenv("TERMINAL_SSH_PORT", raising=False)
    monkeypatch.delenv("TERMINAL_SSH_KEY", raising=False)


class TestExceptionClassification:
    """Backends raise EnvironmentConnectionError for connection-class failures."""

    def test_ssh_connect_refused_raises_connection_error(self, monkeypatch):
        from tools.environments.ssh import SSHEnvironment

        _mock_ssh_unreachable(monkeypatch)
        with pytest.raises(EnvironmentConnectionError):
            SSHEnvironment(host="unreachable.invalid", user="nobody")

    def test_ssh_connect_timeout_raises_connection_error(self, monkeypatch):
        from tools.environments.ssh import SSHEnvironment

        monkeypatch.setattr("tools.environments.ssh.shutil.which", lambda _x: "/usr/bin/ssh")

        def _timeout(*a, **k):
            raise subprocess.TimeoutExpired(cmd="ssh", timeout=15)

        monkeypatch.setattr("tools.environments.ssh.subprocess.run", _timeout)
        with pytest.raises(EnvironmentConnectionError):
            SSHEnvironment(host="unreachable.invalid", user="nobody")

    def test_docker_missing_executable_raises_connection_error(self, monkeypatch):
        from tools.environments import docker as docker_env

        monkeypatch.setattr(docker_env, "find_docker", lambda: None)
        with pytest.raises(EnvironmentConnectionError):
            docker_env._ensure_docker_available()

    def test_docker_daemon_timeout_raises_connection_error(self, monkeypatch):
        from tools.environments import docker as docker_env

        monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")

        def _timeout(*a, **k):
            raise subprocess.TimeoutExpired(cmd="docker version", timeout=5)

        monkeypatch.setattr(docker_env.subprocess, "run", _timeout)
        with pytest.raises(EnvironmentConnectionError):
            docker_env._ensure_docker_available()

    def test_connection_error_is_a_runtime_error(self):
        # Existing catchers of RuntimeError must keep working unchanged.
        assert issubclass(EnvironmentConnectionError, RuntimeError)
        err = EnvironmentConnectionError("boom")
        assert err.reason == "boom"
        assert err.retry_hint  # non-empty default hint


class TestDegradedToolResult:
    """terminal_tool returns structured degraded results in warn mode."""

    def test_ssh_unreachable_returns_degraded_result(self, isolated_env, monkeypatch):
        _ssh_backend_env(monkeypatch)
        _mock_ssh_unreachable(monkeypatch)
        monkeypatch.delenv("TERMINAL_DEGRADED_MODE", raising=False)

        r = json.loads(isolated_env.terminal_tool("echo hi", task_id="t-degraded-ssh"))
        assert r["status"] == "degraded"
        assert r["exit_code"] == -1
        assert "reason" in r and r["reason"]
        assert "retry_hint" in r and r["retry_hint"]
        assert "traceback" not in r

    def test_docker_daemon_down_returns_degraded_result(self, isolated_env, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.delenv("TERMINAL_DEGRADED_MODE", raising=False)
        monkeypatch.setattr(isolated_env, "_maybe_reap_docker_orphans", lambda _cc: None)
        monkeypatch.setattr("tools.environments.docker.find_docker", lambda: None)

        r = json.loads(isolated_env.terminal_tool("echo hi", task_id="t-degraded-docker"))
        assert r["status"] == "degraded"
        assert "reason" in r and r["reason"]
        assert "retry_hint" in r and r["retry_hint"]
        assert "traceback" not in r

    def test_degraded_env_is_not_cached(self, isolated_env, monkeypatch):
        """A degraded backend must not be cached — a later call must retry."""
        _ssh_backend_env(monkeypatch)
        _mock_ssh_unreachable(monkeypatch)

        r = json.loads(isolated_env.terminal_tool("echo hi", task_id="t-degraded-cache"))
        assert r["status"] == "degraded"
        with isolated_env._env_lock:
            assert not isolated_env._active_environments

    def test_recovery_after_degraded(self, isolated_env, monkeypatch):
        """When the backend comes back, the next call just works."""
        import shutil as real_shutil

        real_run = subprocess.run
        real_which = real_shutil.which
        _ssh_backend_env(monkeypatch)
        _mock_ssh_unreachable(monkeypatch)
        r1 = json.loads(isolated_env.terminal_tool("echo hi", task_id="t-degraded-recover"))
        assert r1["status"] == "degraded"

        # Backend "recovers" — restore the real subprocess machinery (the
        # ssh-module patch hits the shared subprocess/shutil modules) and
        # switch to a reachable backend; the tool path must not be poisoned.
        monkeypatch.setattr(subprocess, "run", real_run)
        monkeypatch.setattr(real_shutil, "which", real_which)
        monkeypatch.setenv("TERMINAL_ENV", "local")
        r2 = json.loads(isolated_env.terminal_tool("echo back", task_id="t-degraded-recover"))
        assert r2["exit_code"] == 0
        assert "back" in r2["output"]


class TestNonInfrastructureFailuresUntouched:
    def test_nonzero_exit_is_not_degraded(self, isolated_env, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "local")
        r = json.loads(isolated_env.terminal_tool("exit 3", task_id="t-degraded-exit3"))
        assert r["exit_code"] == 3
        assert r.get("status") != "degraded"

    def test_command_not_found_is_not_degraded(self, isolated_env, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "local")
        r = json.loads(isolated_env.terminal_tool(
            "definitely_not_a_real_command_zzz_42", task_id="t-degraded-notfound"))
        assert r["exit_code"] != 0
        assert r.get("status") != "degraded"


class TestFailModePreservesRaiseBehavior:
    def test_fail_mode_returns_error_with_traceback(self, isolated_env, monkeypatch):
        _ssh_backend_env(monkeypatch)
        _mock_ssh_unreachable(monkeypatch)
        monkeypatch.setenv("TERMINAL_DEGRADED_MODE", "fail")

        r = json.loads(isolated_env.terminal_tool("echo hi", task_id="t-degraded-fail"))
        assert r["status"] == "error"
        assert "traceback" in r
        assert "SSH connection failed" in r["error"]

    def test_invalid_mode_falls_back_to_warn(self, isolated_env, monkeypatch):
        _ssh_backend_env(monkeypatch)
        _mock_ssh_unreachable(monkeypatch)
        monkeypatch.setenv("TERMINAL_DEGRADED_MODE", "bogus-value")

        r = json.loads(isolated_env.terminal_tool("echo hi", task_id="t-degraded-bogus"))
        assert r["status"] == "degraded"


class TestConfigBridging:
    def test_degraded_mode_is_bridged_everywhere(self):
        """terminal.degraded_mode must ride every config->env bridge path,
        same four-site invariant as the docker_* keys."""
        from tests.tools.test_terminal_config_env_sync import (
            _cli_env_map_keys,
            _gateway_env_map_keys,
            _save_config_env_sync_keys,
            _terminal_tool_env_var_names,
        )

        assert "degraded_mode" in _cli_env_map_keys()
        assert "degraded_mode" in _gateway_env_map_keys()
        assert "degraded_mode" in _save_config_env_sync_keys()
        assert "TERMINAL_DEGRADED_MODE" in _terminal_tool_env_var_names()

    def test_default_config_carries_degraded_mode(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["terminal"].get("degraded_mode") == "warn"
