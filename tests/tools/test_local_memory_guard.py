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


import json
import os
import shlex
import subprocess
import sys
import time

import pytest


@pytest.mark.parametrize("value", ["", "0", "-1", "invalid"])
def test_disabled_guard_does_not_probe_or_wrap(monkeypatch, value):
    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", value)
    monkeypatch.setattr(
        process_registry,
        "_systemd_run_user_scope_available",
        lambda: pytest.fail("disabled guard probed systemd"),
    )
    args = ["/bin/bash", "-c", "true"]
    assert local_env._maybe_guard_foreground_argv(args) == (args, False)


def test_foreground_never_widens_native_bound(monkeypatch):
    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "8192")
    monkeypatch.setattr(
        process_registry, "_worker_memory_max_bytes", lambda: 512 * 1024 * 1024
    )
    monkeypatch.setattr(
        process_registry, "_systemd_run_user_scope_available", lambda: True
    )
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/systemd-run")
    argv, enabled = local_env._maybe_guard_foreground_argv(["/bin/true"])
    assert enabled
    assert "MemoryMax=536870912" in argv
    assert not any("MemorySwapMax" in arg for arg in argv)


@pytest.mark.parametrize(
    "raw,expected",
    [("", None), ("0", 0), ("32", 33554432), ("-1", None), ("invalid", None)],
)
def test_foreground_swap_tristate(monkeypatch, raw, expected):
    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "128")
    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_SWAP_MAX_MB", raw)
    monkeypatch.setattr(
        process_registry, "_systemd_run_user_scope_available", lambda: True
    )
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/systemd-run")
    argv, _ = local_env._maybe_guard_foreground_argv(["/bin/true"])
    swaps = [a for a in argv if a.startswith("MemorySwapMax=")]
    assert swaps == ([] if expected is None else [f"MemorySwapMax={expected}"])


def test_background_swap_policy_stays_native(monkeypatch):
    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_SWAP_MAX_MB", "0")
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/systemd-run")
    argv = process_registry._build_systemd_scope_argv(["/bin/true"], "background")
    assert not any("MemorySwapMax" in a for a in argv)
    assert argv[argv.index("--unit") + 1] == "hermes-worker-background"


def test_platform_negative_control(monkeypatch):
    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "128")
    monkeypatch.setattr(local_env, "_IS_WINDOWS", True)
    args = ["bash", "-c", "true"]
    assert local_env._maybe_guard_foreground_argv(args) == (args, False)


def test_disappeared_binary_is_not_reported_guarded(monkeypatch):
    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "128")
    monkeypatch.setattr(
        process_registry, "_systemd_run_user_scope_available", lambda: True
    )
    monkeypatch.setattr("shutil.which", lambda _: None)
    args = ["/bin/true"]
    assert local_env._maybe_guard_foreground_argv(args) == (args, False)


def _require_real_scopes():
    if not process_registry._systemd_run_user_scope_available():
        pytest.skip("systemd user scopes unavailable")


@pytest.mark.parametrize("limit,scoped", [("", False), ("128", True), ("8192", True)])
def test_real_foreground_scope_and_default_off(monkeypatch, tmp_path, limit, scoped):
    if scoped:
        _require_real_scopes()
    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", limit)
    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_SWAP_MAX_MB", "0")
    env = _bare_environment(tmp_path)
    probe = """import json,pathlib
cg=pathlib.Path('/proc/self/cgroup').read_text().strip().partition('0::')[2]
p=pathlib.Path('/sys/fs/cgroup') / cg.lstrip('/')
print(json.dumps({'cgroup':cg,'memory':(p/'memory.max').read_text().strip(), 'swap':(p/'memory.swap.max').read_text().strip()}))
"""
    proc = env._run_bash(f"{shlex.quote(sys.executable)} -c {shlex.quote(probe)}")
    output, _ = proc.communicate(timeout=15)
    assert proc.returncode == 0, output
    data = json.loads(output.strip())
    assert ("hermes-terminal-" in data["cgroup"]) is scoped
    if scoped:
        assert int(data["memory"]) == min(
            int(limit) * 1024 * 1024, process_registry._worker_memory_max_bytes()
        )
        assert data["swap"] == "0"
        unit = data["cgroup"].rsplit("/", 1)[-1]
        result = subprocess.run(
            ["systemctl", "--user", "is-active", unit],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, result.stdout


def test_real_foreground_cleanup_kills_child(monkeypatch, tmp_path):
    _require_real_scopes()
    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "128")
    env = _bare_environment(tmp_path)
    pidfile = tmp_path / "child.pid"
    proc = env._run_bash(f"sleep 60 & echo $! > {shlex.quote(str(pidfile))}; wait")
    deadline = time.monotonic() + 10
    while not pidfile.exists() and proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    try:
        assert pidfile.exists()
        child = int(pidfile.read_text())
        env._kill_process(proc)
        proc.wait(timeout=10)
        import psutil

        assert (
            not psutil.pid_exists(child)
            or psutil.Process(child).status() == psutil.STATUS_ZOMBIE
        )
    finally:
        if proc.poll() is None:
            env._kill_process(proc)
            proc.wait(timeout=10)


def test_real_guard_bounds_allocation_and_negative_control(monkeypatch, tmp_path):
    _require_real_scopes()
    env = _bare_environment(tmp_path)
    command = f"{shlex.quote(sys.executable)} -c " + shlex.quote(
        "x=bytearray(192*1024*1024); print('allocation-succeeded')"
    )
    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_SWAP_MAX_MB", "0")
    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "128")
    guarded = env._run_bash(command)
    output, _ = guarded.communicate(timeout=20)
    assert guarded.returncode != 0, output
    assert "allocation-succeeded" not in output
    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "")
    direct = env._run_bash(command)
    output, _ = direct.communicate(timeout=20)
    assert direct.returncode == 0, output
    assert "allocation-succeeded" in output


def test_real_failing_command_runs_once_with_stdin_and_env(monkeypatch, tmp_path):
    _require_real_scopes()
    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "128")
    env = _bare_environment(tmp_path)
    env.env = {"MEMORY_AUDIT_VALUE": "literal $VALUE secret"}
    marker = tmp_path / "runs"
    command = f'echo run >> {shlex.quote(str(marker))}; read line; printf \'%s|%s\' "$line" "$MEMORY_AUDIT_VALUE"; exit 7'
    proc = env._run_bash(command, stdin_data="hello\n")
    output, _ = proc.communicate(timeout=15)
    assert proc.returncode == 7
    assert output == "hello|literal $VALUE secret"
    assert marker.read_text().splitlines() == ["run"]
