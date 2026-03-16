import pytest

from tools.environments import ssh as ssh_env


def test_ensure_ssh_available_raises_clear_error_when_missing(monkeypatch):
    monkeypatch.setattr(ssh_env.shutil, "which", lambda _name: None)

    with pytest.raises(RuntimeError, match="SSH is not installed or not in PATH"):
        ssh_env._ensure_ssh_available()


def test_ssh_environment_checks_availability_before_connect(monkeypatch):
    monkeypatch.setattr(ssh_env.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        ssh_env.SSHEnvironment,
        "_establish_connection",
        lambda self: pytest.fail("_establish_connection should not run when ssh is missing"),
    )

    with pytest.raises(RuntimeError, match="openssh-client"):
        ssh_env.SSHEnvironment(host="example.com", user="alice")


def test_ssh_environment_connects_when_ssh_exists(monkeypatch):
    called = {"count": 0}

    monkeypatch.setattr(ssh_env.shutil, "which", lambda _name: "/usr/bin/ssh")

    def _fake_establish(self):
        called["count"] += 1

    monkeypatch.setattr(ssh_env.SSHEnvironment, "_establish_connection", _fake_establish)

    env = ssh_env.SSHEnvironment(host="example.com", user="alice")

    assert called["count"] == 1
    assert env.host == "example.com"
    assert env.user == "alice"
