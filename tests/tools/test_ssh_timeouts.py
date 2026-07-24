"""Tests for configurable SSH environment timeouts (TERMINAL_SSH_*_TIMEOUT)."""

import pytest

from tools.environments import ssh as ssh_env
from tools.environments.ssh import SSHEnvironment, _timeout_env


@pytest.fixture
def make_env(monkeypatch):
    """Factory building an SSHEnvironment with mocked connection/sync."""
    monkeypatch.setattr(ssh_env.shutil, "which", lambda _name: "/usr/bin/ssh")
    monkeypatch.setattr(ssh_env.SSHEnvironment, "_establish_connection", lambda self: None)
    monkeypatch.setattr(ssh_env.SSHEnvironment, "_detect_remote_home", lambda self: "/home/testuser")
    monkeypatch.setattr(ssh_env.SSHEnvironment, "_ensure_remote_dirs", lambda self: None)
    monkeypatch.setattr(ssh_env.SSHEnvironment, "init_session", lambda self: None)
    monkeypatch.setattr(
        ssh_env, "FileSyncManager",
        lambda **kw: type("M", (), {"sync": lambda self, **k: None})(),
    )

    def _make():
        return SSHEnvironment(host="example.com", user="testuser")

    return _make


class TestTimeoutEnvHelper:
    """Unit tests for the _timeout_env parser."""

    def test_unset_returns_default(self, monkeypatch):
        monkeypatch.delenv("TERMINAL_SSH_CONNECT_TIMEOUT", raising=False)
        assert _timeout_env("TERMINAL_SSH_CONNECT_TIMEOUT", 10) == 10

    def test_valid_override(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_SSH_CONNECT_TIMEOUT", "45")
        assert _timeout_env("TERMINAL_SSH_CONNECT_TIMEOUT", 10) == 45

    @pytest.mark.parametrize("raw", ["abc", "", "12.5", "10s"])
    def test_non_integer_falls_back(self, monkeypatch, raw):
        monkeypatch.setenv("TERMINAL_SSH_CONNECT_TIMEOUT", raw)
        assert _timeout_env("TERMINAL_SSH_CONNECT_TIMEOUT", 10) == 10

    @pytest.mark.parametrize("raw", ["0", "-5"])
    def test_non_positive_falls_back(self, monkeypatch, raw):
        monkeypatch.setenv("TERMINAL_SSH_CONNECT_TIMEOUT", raw)
        assert _timeout_env("TERMINAL_SSH_CONNECT_TIMEOUT", 10) == 10


class TestSSHEnvironmentTimeouts:
    """Timeout wiring on SSHEnvironment instances."""

    def test_defaults_match_previous_hardcoded_values(self, monkeypatch, make_env):
        for name in ("TERMINAL_SSH_CONNECT_TIMEOUT",
                     "TERMINAL_SSH_ESTABLISH_TIMEOUT",
                     "TERMINAL_SSH_FILE_SYNC_TIMEOUT"):
            monkeypatch.delenv(name, raising=False)
        env = make_env()
        assert env.connect_timeout == 10
        assert env.establish_timeout == 15
        assert env.file_sync_timeout == 120
        assert "ConnectTimeout=10" in " ".join(env._build_ssh_command())

    def test_env_overrides_applied(self, monkeypatch, make_env):
        monkeypatch.setenv("TERMINAL_SSH_CONNECT_TIMEOUT", "60")
        monkeypatch.setenv("TERMINAL_SSH_ESTABLISH_TIMEOUT", "90")
        monkeypatch.setenv("TERMINAL_SSH_FILE_SYNC_TIMEOUT", "600")
        env = make_env()
        assert env.connect_timeout == 60
        assert env.establish_timeout == 90
        assert env.file_sync_timeout == 600
        assert "ConnectTimeout=60" in " ".join(env._build_ssh_command())

    def test_invalid_overrides_fall_back_to_defaults(self, monkeypatch, make_env):
        monkeypatch.setenv("TERMINAL_SSH_CONNECT_TIMEOUT", "not-a-number")
        monkeypatch.setenv("TERMINAL_SSH_ESTABLISH_TIMEOUT", "-1")
        monkeypatch.setenv("TERMINAL_SSH_FILE_SYNC_TIMEOUT", "0")
        env = make_env()
        assert env.connect_timeout == 10
        assert env.establish_timeout == 15
        assert env.file_sync_timeout == 120
