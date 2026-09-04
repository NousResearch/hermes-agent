"""Focused coverage for opt-in foreground local command memory limits."""

from __future__ import annotations

from unittest.mock import MagicMock

from hermes_cli.config_defaults import DEFAULT_CONFIG
from tools.environments import local as local_env
from tools import process_registry


def _bare_environment(tmp_path):
    env = object.__new__(local_env.LocalEnvironment)
    env.cwd = str(tmp_path)
    env.env = {}
    return env


def test_local_memory_guard_default_is_unset(monkeypatch):
    monkeypatch.delenv("TERMINAL_LOCAL_MEMORY_MAX_MB", raising=False)

    assert DEFAULT_CONFIG["terminal"]["local_memory_max_mb"] == ""
    assert local_env._local_memory_max_bytes() is None


def test_local_memory_guard_requires_positive_integer(monkeypatch, caplog):
    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "0")
    assert local_env._local_memory_max_bytes() is None

    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "invalid")
    assert local_env._local_memory_max_bytes() is None
    assert "expected a positive integer" in caplog.text


def test_foreground_guard_reuses_systemd_scope_builder(monkeypatch):
    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "256")
    monkeypatch.setattr(
        process_registry, "_systemd_run_user_scope_available", lambda: True
    )
    captured = {}

    def fake_builder(argv, unit_suffix, **kwargs):
        captured.update(argv=argv, unit_suffix=unit_suffix, **kwargs)
        return ["systemd-run", *argv]

    monkeypatch.setattr(process_registry, "_build_systemd_scope_argv", fake_builder)

    argv, enabled = local_env._maybe_guard_foreground_argv(["/bin/bash", "-c", "true"])

    assert enabled is True
    assert argv[:2] == ["systemd-run", "/bin/bash"]
    assert captured["memory_max_bytes"] == 256 * 1024 * 1024
    assert captured["unit_prefix"] == "hermes-terminal"


def test_foreground_guard_falls_back_when_user_scopes_unavailable(monkeypatch):
    direct = ["/bin/bash", "-c", "true"]
    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "256")
    monkeypatch.setattr(
        process_registry, "_systemd_run_user_scope_available", lambda: False
    )

    assert local_env._maybe_guard_foreground_argv(direct) == (direct, False)


def test_systemd_scope_builder_accepts_explicit_foreground_limit(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/systemd-run")

    argv = process_registry._build_systemd_scope_argv(
        ["/bin/bash", "-c", "true"],
        "abc123",
        memory_max_bytes=256 * 1024 * 1024,
        unit_prefix="hermes-terminal",
    )

    assert argv[argv.index("--unit") + 1] == "hermes-terminal-abc123"
    assert "MemoryMax=268435456" in argv
    assert argv[-3:] == ["/bin/bash", "-c", "true"]


def test_guard_spawn_failure_falls_back_directly_once(monkeypatch, tmp_path):
    env = _bare_environment(tmp_path)
    monkeypatch.setattr(local_env, "_find_bash", lambda: "/bin/bash")
    monkeypatch.setattr(
        local_env,
        "_maybe_guard_foreground_argv",
        lambda args: (["systemd-run", *args], True),
    )
    monkeypatch.setattr(local_env.os, "getpgid", lambda _pid: 4321)

    direct_proc = MagicMock(pid=1234)
    calls = []

    def fake_popen(argv, **kwargs):
        calls.append(argv)
        if len(calls) == 1:
            raise OSError("systemd-run disappeared")
        return direct_proc

    monkeypatch.setattr(local_env.subprocess, "Popen", fake_popen)

    result = local_env.LocalEnvironment._run_bash(env, "echo ok")

    assert result is direct_proc
    assert calls == [
        ["systemd-run", "/bin/bash", "-c", "echo ok"],
        ["/bin/bash", "-c", "echo ok"],
    ]


def test_launched_guard_is_never_retried(monkeypatch, tmp_path):
    env = _bare_environment(tmp_path)
    monkeypatch.setattr(local_env, "_find_bash", lambda: "/bin/bash")
    monkeypatch.setattr(
        local_env,
        "_maybe_guard_foreground_argv",
        lambda args: (["systemd-run", *args], True),
    )
    monkeypatch.setattr(local_env.os, "getpgid", lambda _pid: 4321)
    guarded_proc = MagicMock(pid=1234)
    guarded_proc.poll.return_value = 1
    popen = MagicMock(return_value=guarded_proc)
    monkeypatch.setattr(local_env.subprocess, "Popen", popen)

    result = local_env.LocalEnvironment._run_bash(env, "exit 1")

    assert result is guarded_proc
    popen.assert_called_once()
