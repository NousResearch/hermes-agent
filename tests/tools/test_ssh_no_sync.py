import pytest

from tools.environments import ssh as ssh_env


def test_ssh_environment_can_skip_remote_profile_sync(monkeypatch):
    calls = {"dirs": 0, "manager": 0}

    monkeypatch.setattr(ssh_env, "_ensure_ssh_available", lambda **kwargs: None)
    monkeypatch.setattr(
        ssh_env.SSHEnvironment, "_establish_connection", lambda self: None
    )
    monkeypatch.setattr(
        ssh_env.SSHEnvironment, "_detect_remote_home", lambda self: "/home/developer"
    )
    monkeypatch.setattr(
        ssh_env.SSHEnvironment,
        "_ensure_remote_dirs",
        lambda self: calls.__setitem__("dirs", calls["dirs"] + 1),
    )
    monkeypatch.setattr(ssh_env.SSHEnvironment, "init_session", lambda self: None)
    monkeypatch.setattr(
        ssh_env,
        "FileSyncManager",
        lambda **kwargs: calls.__setitem__("manager", calls["manager"] + 1),
    )

    env = ssh_env.SSHEnvironment(
        host="workspace.example",
        user="developer",
        cwd="/workspace/project",
        sync=False,
    )

    assert calls == {"dirs": 0, "manager": 0}
    assert env._sync_manager is None
    env._before_execute()


def test_no_sync_mode_does_not_require_scp(monkeypatch):
    monkeypatch.setattr(
        ssh_env.shutil,
        "which",
        lambda command: "/usr/bin/ssh" if command == "ssh" else None,
    )

    ssh_env._ensure_ssh_available(require_scp=False)

    with pytest.raises(RuntimeError, match="SCP is not installed"):
        ssh_env._ensure_ssh_available(require_scp=True)
