"""Per-call SSH sudo uses the same masked prompt without changing local terminal state."""
import json
import os
import subprocess

import tools.terminal_tool as terminal
from tools.environments import ssh


def test_remote_ssh_sudo_prompt_and_local_isolation(monkeypatch, tmp_path):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.setattr(terminal, "_get_env_config", lambda: {
        "env_type": "local", "cwd": str(tmp_path), "timeout": 30,
        "ssh_host": "test-host", "ssh_user": "tester", "ssh_port": 22,
        "ssh_key": "", "ssh_persistent": False,
    })
    monkeypatch.setattr(terminal, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal, "_pre_exec_block", lambda *a, **kw: None)
    monkeypatch.setattr(terminal, "_run_approval_guards", lambda *a, **kw: terminal._ApprovalVerdict())
    monkeypatch.setattr(ssh.SSHEnvironment, "_establish_connection", lambda self: None)
    monkeypatch.setattr(ssh.SSHEnvironment, "_detect_remote_home", lambda self: str(tmp_path))
    monkeypatch.setattr(ssh.SSHEnvironment, "_ensure_remote_dirs", lambda self: None)
    monkeypatch.setattr(ssh.SSHEnvironment, "init_session", lambda self: None)
    monkeypatch.setattr(ssh.FileSyncManager, "sync", lambda self, **kw: None)
    monkeypatch.setattr(ssh.SSHEnvironment, "_sudo_nopasswd_works", lambda self: False)
    monkeypatch.setattr(ssh.SSHEnvironment, "_wrap_command", lambda self, command, cwd: command)
    captured = []

    def remote_shell(self, command, *, login=False, timeout=120, stdin_data=None):
        captured.append((command, stdin_data))
        # Simulate the SSH process: the -S password is carried by stdin, not argv.
        return ssh._popen_bash(["bash", "-c", command], stdin_data)

    monkeypatch.setattr(ssh.SSHEnvironment, "_run_bash", remote_shell)
    sudo = tmp_path / "sudo"
    sudo.write_text("#!/bin/bash\nread -r secret\n[ \"$secret\" = sample ] && printf 'remote-root\\n'\n")
    sudo.chmod(0o700)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    prompts = []
    terminal.set_sudo_password_callback(lambda: prompts.append("prompt") or "sample")
    try:
        before = terminal.get_session_cwd("remote-ssh-default")
        result = json.loads(terminal._handle_terminal({"command": "sudo id", "remote": "ssh"}, task_id="remote-test"))
        assert result["exit_code"] == 0, result
        assert "remote-root" in result["output"]
        assert prompts == ["prompt"]
        assert captured[-1][1] == "sample\n"
        assert "sample" not in captured[-1][0] and "sample" not in json.dumps(result)
        assert terminal.get_session_cwd("remote-test") is None
        assert terminal.get_session_cwd("remote-ssh-default") == before
        assert terminal._active_environments["remote-ssh-default"].host == "test-host"
        local = json.loads(terminal._handle_terminal({"command": "pwd"}, task_id="remote-test"))
        assert local["exit_code"] == 0, local
        assert str(tmp_path) in local["output"]
        assert terminal._active_environments["default"].is_local
    finally:
        terminal.set_sudo_password_callback(None)
        terminal._active_environments.pop("remote-ssh-default", None)
        terminal._active_environments.pop("default", None)
        from tools.terminal_tool_sudo import _reset_cached_sudo_passwords
        _reset_cached_sudo_passwords()


def test_remote_rejects_unconfigured_or_background_and_sudo_stdin_guard(monkeypatch):
    monkeypatch.setattr(terminal, "_get_env_config", lambda: {"env_type": "local", "cwd": "/", "timeout": 30,
                                                               "ssh_host": "", "ssh_user": ""})
    missing = json.loads(terminal._handle_terminal({"command": "sudo id", "remote": "ssh"}))
    assert missing["status"] == "error" and "ssh_host" in missing["error"]
    background = json.loads(terminal._handle_terminal({"command": "sudo id", "remote": "ssh", "background": True}))
    assert background["status"] == "error"
    from tools.approval_detection import _check_sudo_stdin_guard
    from tools.approval import check_all_command_guards
    nested = "bash -c 'sudo -S id'"
    assert _check_sudo_stdin_guard(nested)[0]
    assert not check_all_command_guards(nested, "ssh")["approved"]
    assert not _check_sudo_stdin_guard("env -u X sudo id")[0]
    assert not _check_sudo_stdin_guard("echo 'sudo -S id'")[0]


def test_ssh_transport_sends_password_only_on_stdin(monkeypatch, tmp_path):
    env = ssh.SSHEnvironment.__new__(ssh.SSHEnvironment)
    env.host, env.user, env.port, env.key_path = "target", "operator", 22, ""
    env.control_socket = tmp_path / "control.sock"
    monkeypatch.setattr(ssh, "resolve_passthrough_env", lambda **kw: ({}, ()))
    sent = []
    monkeypatch.setattr(ssh, "_popen_bash", lambda *a, **kw: sent.append((a, kw)))
    env._run_bash("sudo -S -p '' id", stdin_data="opaque\n")
    args, kwargs = sent[0]
    assert args[1] == "opaque\n"
    assert "opaque" not in repr(args[0]) and "opaque" not in repr(kwargs)
    assert "operator@target" in args[0]


def test_remote_password_cache_never_reuses_local_or_other_host(monkeypatch):
    from tools import terminal_tool_sudo as sudo
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    sudo._reset_cached_sudo_passwords()
    terminal.set_sudo_password_callback(lambda: "prompted")
    try:
        sudo._set_cached_sudo_password("local-secret")
        one = ssh.SSHEnvironment.__new__(ssh.SSHEnvironment)
        one.user, one.host, one.port = "operator", "one", 22
        one._sudo_nopasswd_works = lambda: False
        two = ssh.SSHEnvironment.__new__(ssh.SSHEnvironment)
        two.user, two.host, two.port = "operator", "two", 22
        two._sudo_nopasswd_works = lambda: False
        _, first_stdin = one._prepare_command("sudo id")
        _, second_stdin = two._prepare_command("sudo id")
        assert first_stdin == second_stdin == "prompted\n"
        assert sudo._get_cached_sudo_password() == "local-secret"
        assert sudo._get_cached_sudo_password(one._sudo_cache_target) == "prompted"
        assert sudo._invalidate_cached_sudo_on_auth_failure(
            "sudo id", "sudo: authentication failed", one._sudo_cache_target)
        assert sudo._get_cached_sudo_password(one._sudo_cache_target) == ""
        assert sudo._get_cached_sudo_password(two._sudo_cache_target) == "prompted"
    finally:
        terminal.set_sudo_password_callback(None)
        sudo._reset_cached_sudo_passwords()
