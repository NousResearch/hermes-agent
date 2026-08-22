"""Regression tests for remote SSH command-group cleanup."""

from types import SimpleNamespace

from tools.environments import ssh as ssh_mod
from tools.environments.ssh import SSHEnvironment


def _env(monkeypatch):
    env = object.__new__(SSHEnvironment)
    monkeypatch.setattr(env, "_build_ssh_command", lambda: ["ssh", "mini2"])
    return env


def test_ssh_spawn_wraps_command_in_remote_process_group(monkeypatch):
    env = _env(monkeypatch)
    captured = {}

    def fake_popen(command, stdin_data=None):
        captured["command"] = command
        captured["stdin_data"] = stdin_data
        return SimpleNamespace(pid=123, kill=lambda: None)

    monkeypatch.setattr(ssh_mod, "_popen_bash", fake_popen)
    proc = env._run_bash("sleep 30")

    rendered = " ".join(captured["command"])
    assert "os.setsid()" in rendered
    assert "os.fork()" in rendered
    assert "os.execvp" in rendered and "bash" in rendered
    assert "hermes-ssh-" in rendered
    assert getattr(proc, "_hermes_remote_pid_file").startswith("/tmp/hermes-ssh-")


def test_ssh_kill_terminates_remote_group_before_local_client(monkeypatch):
    env = _env(monkeypatch)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ssh_mod.subprocess, "run", fake_run)
    killed = []
    proc = SimpleNamespace(
        pid=123,
        _hermes_remote_pid_file="/tmp/hermes-ssh-test.pid",
        kill=lambda: killed.append(True),
    )

    env._kill_process(proc)  # type: ignore[arg-type]

    assert len(calls) == 1
    remote_kill = " ".join(calls[0][0])
    assert "os.killpg(pgid, signal.SIGTERM)" in remote_kill
    assert "os.killpg(pgid, signal.SIGKILL)" in remote_kill
    assert "/tmp/hermes-ssh-test.pid" in remote_kill
    assert killed == [True]