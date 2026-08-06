"""Unit coverage for local terminal memory-guard wrapping."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

from tools.environments import local as local_env
from tools.process_registry import ProcessRegistry, ProcessSession


def test_memory_guard_disabled_without_limit(monkeypatch):
    monkeypatch.delenv("TERMINAL_LOCAL_MEMORY_MAX_MB", raising=False)
    monkeypatch.setattr(local_env, "_systemd_run_available", lambda: True)

    assert local_env._maybe_wrap_with_systemd_memory_guard(
        cmd_string="echo hi",
        login=False,
        run_env={},
        cwd="/tmp",
    ) == (None, None, None, None, None)


def test_memory_guard_disabled_when_systemd_run_unavailable(monkeypatch):
    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "256")
    monkeypatch.setattr(local_env, "_systemd_run_available", lambda: False)

    assert local_env._maybe_wrap_with_systemd_memory_guard(
        cmd_string="echo hi",
        login=False,
        run_env={},
        cwd="/tmp",
    ) == (None, None, None, None, None)


def test_systemd_run_available_rejects_bus_failure(monkeypatch):
    local_env._reset_systemd_run_available_cache()
    monkeypatch.setattr(local_env.shutil, "which", lambda _: "/usr/bin/systemd-run")
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/tmp")
    monkeypatch.setattr(local_env.os.path, "isdir", lambda _: True)
    monkeypatch.setattr(local_env.uuid, "uuid4", lambda: Mock(hex="beef"))

    def fake_run(*args, **kwargs):
        return local_env.subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="",
            stderr="Failed to connect to bus: Operation not permitted",
        )

    monkeypatch.setattr(local_env.subprocess, "run", fake_run)
    assert local_env._systemd_run_available() is False


def test_systemd_run_available_runs_transient_unit_preflight(monkeypatch):
    local_env._reset_systemd_run_available_cache()
    monkeypatch.setattr(local_env.shutil, "which", lambda _: "/usr/bin/systemd-run")
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/tmp")
    monkeypatch.setattr(local_env.os.path, "isdir", lambda _: True)
    monkeypatch.setattr(local_env.uuid, "uuid4", lambda: Mock(hex="beef"))

    def fake_run(*args, **kwargs):
        command = args[0]
        assert command[0] == "/usr/bin/systemd-run"
        assert "--user" in command
        assert "--wait" in command
        assert "--unit" in command
        assert command[-1] == "/bin/true"
        assert "list-units" not in command
        return local_env.subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(local_env.subprocess, "run", fake_run)
    assert local_env._systemd_run_available() is True


def test_systemd_run_available_reuses_cache_until_positive_ttl(monkeypatch):
    local_env._reset_systemd_run_available_cache()
    monkeypatch.setattr(local_env.shutil, "which", lambda _: "/usr/bin/systemd-run")
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/tmp")
    monkeypatch.setattr(local_env.os.path, "isdir", lambda _: True)
    monkeypatch.setattr(local_env.uuid, "uuid4", lambda: Mock(hex="beef"))

    times = iter([100.0, 100.0, 161.0])
    monkeypatch.setattr(local_env.time, "monotonic", lambda: next(times))

    run_calls = []

    def fake_run(*args, **kwargs):
        run_calls.append(args[0])
        return local_env.subprocess.CompletedProcess(
            args=run_calls[-1],
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(local_env.subprocess, "run", fake_run)

    assert local_env._systemd_run_available() is True
    assert local_env._systemd_run_available() is True
    assert local_env._systemd_run_available() is True
    assert len(run_calls) == 2


def test_systemd_run_available_rechecks_after_negative_ttl(monkeypatch):
    local_env._reset_systemd_run_available_cache()
    monkeypatch.setattr(local_env.shutil, "which", lambda _: "/usr/bin/systemd-run")
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/tmp")
    monkeypatch.setattr(local_env.os.path, "isdir", lambda _: True)
    monkeypatch.setattr(local_env.uuid, "uuid4", lambda: Mock(hex="beef"))

    times = iter([200.0, 200.0, 211.0])
    monkeypatch.setattr(local_env.time, "monotonic", lambda: next(times))

    run_calls = []

    def fake_run(*args, **kwargs):
        run_calls.append(args[0])
        return local_env.subprocess.CompletedProcess(
            args=run_calls[-1],
            returncode=1,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(local_env.subprocess, "run", fake_run)
    assert local_env._systemd_run_available() is False
    assert local_env._systemd_run_available() is False
    assert local_env._systemd_run_available() is False
    assert len(run_calls) == 2


def test_systemd_run_available_absent_binary_does_not_cache(monkeypatch):
    local_env._reset_systemd_run_available_cache()
    which_calls = []

    def fake_which(_):
        which_calls.append(1)
        return None

    monkeypatch.setattr(local_env.shutil, "which", fake_which)
    assert local_env._systemd_run_available() is False
    assert local_env._systemd_run_available() is False
    assert which_calls == [1, 1]


def test_memory_guard_builds_transient_unit_without_secret_argv_leak(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "256")
    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_SWAP_MAX_MB", "64")
    monkeypatch.setattr(local_env, "_systemd_run_available", lambda: True)
    monkeypatch.setattr(local_env, "_find_bash", lambda: "/bin/bash")
    monkeypatch.setattr(local_env.tempfile, "gettempdir", lambda: str(tmp_path))

    argv, env, cwd, unit, guard_files = local_env._maybe_wrap_with_systemd_memory_guard(
        cmd_string="echo $SECRET_VALUE",
        login=True,
        run_env={"SECRET_VALUE": "super secret", "OK_NAME": "ok", "BAD-NAME": "drop"},
        cwd="/work dir",
    )

    assert argv is not None
    assert env == os.environ.copy()
    assert cwd == "/"
    assert unit.startswith("hermes-terminal-")
    assert argv[:6] == [
        "systemd-run",
        "--user",
        "--collect",
        "--pipe",
        "--quiet",
        "--unit",
    ]
    assert "MemoryMax=256M" in argv
    assert "MemorySwapMax=64M" in argv
    assert "OOMPolicy=stop" in argv
    assert "KillMode=control-group" in argv
    script_path = next(
        str(p)
        for p in argv
        if str(p).startswith(str(tmp_path / "hermes-memguard-"))
        and str(p).endswith(".sh")
    )
    assert argv[-2:] == ["-l", script_path]

    rendered_argv = "\n".join(map(str, argv))
    assert "super secret" not in rendered_argv
    assert "SECRET_VALUE" not in rendered_argv

    assert guard_files is not None
    assert len(guard_files) == 4
    scripts = [
        tmp_path / script
        for script in guard_files
        if "hermes-memguard-" in script
        and "hermes-memguard-env-" not in script
        and script.endswith(".sh")
    ]
    env_files = list(tmp_path.glob("hermes-memguard-env-*.sh"))
    assert len(scripts) == 1
    assert len(env_files) == 1
    script_path = scripts[0]
    env_text = env_files[0].read_text()
    assert "export SECRET_VALUE='super secret'" in env_text
    assert "export OK_NAME=ok" in env_text
    assert "BAD-NAME" not in env_text
    script_text = script_path.read_text()
    assert "source " in script_text
    assert "builtin cd -- '/work dir'" in script_text
    assert " -lic " in script_text


def test_memory_guard_executes_user_shell_in_background_wrapper(monkeypatch, tmp_path):
    user_shell = "/usr/bin/zsh"
    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "256")
    monkeypatch.setattr(local_env, "_systemd_run_available", lambda: True)
    monkeypatch.setattr(local_env, "_find_bash", lambda: "/bin/bash")
    monkeypatch.setattr(local_env.tempfile, "gettempdir", lambda: str(tmp_path))

    argv, _env, _cwd, unit, guard_files = (
        local_env._maybe_wrap_with_systemd_memory_guard(
            cmd_string="set +m; echo hi",
            login=True,
            run_env={},
            cwd="/tmp",
            shell=user_shell,
        )
    )

    assert unit.startswith("hermes-terminal-")
    assert guard_files is not None
    script_text = Path(guard_files[0]).read_text()
    assert f"{shlex.quote(user_shell)} -lic" in script_text


def test_memory_guard_uses_systemd_pty_for_background_pty(monkeypatch, tmp_path):
    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "256")
    monkeypatch.setattr(local_env, "_systemd_run_available", lambda: True)
    monkeypatch.setattr(local_env, "_find_bash", lambda: "/bin/bash")
    monkeypatch.setattr(local_env.tempfile, "gettempdir", lambda: str(tmp_path))

    argv, _env, _cwd, _unit, guard_files = (
        local_env._maybe_wrap_with_systemd_memory_guard(
            cmd_string="python -i",
            login=True,
            run_env={},
            cwd="/tmp",
            pty=True,
        )
    )

    assert argv is not None
    assert "--pty" in argv
    assert "--pipe" not in argv
    local_env._cleanup_systemd_memory_guard_files(guard_files)


def test_memory_guard_wrapper_cleanup_temp_files_after_success(monkeypatch, tmp_path):
    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "256")
    monkeypatch.setattr(local_env, "_systemd_run_available", lambda: True)
    monkeypatch.setattr(local_env, "_find_bash", lambda: "/bin/bash")
    monkeypatch.setattr(local_env.tempfile, "gettempdir", lambda: str(tmp_path))

    argv, _env, _cwd, _unit, guard_files = (
        local_env._maybe_wrap_with_systemd_memory_guard(
            cmd_string="printf 'memguard-ok\\n'",
            login=False,
            run_env={"MEM_GUARD_VALUE": "visible"},
            cwd="/tmp",
        )
    )

    assert argv is not None
    assert guard_files is not None
    script_path = Path(guard_files[0])
    env_path = Path(guard_files[1])
    assert script_path.exists()
    assert env_path.exists()

    result = subprocess.run(
        [str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "memguard-ok" in result.stdout
    assert not script_path.exists()
    assert not env_path.exists()


def test_memory_guard_runs_non_login_shell_with_c(monkeypatch, tmp_path):
    user_shell = "/usr/bin/zsh"
    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "256")
    monkeypatch.setattr(local_env, "_systemd_run_available", lambda: True)
    monkeypatch.setattr(local_env, "_find_bash", lambda: "/bin/bash")
    monkeypatch.setattr(local_env.tempfile, "gettempdir", lambda: str(tmp_path))

    argv, _env, _cwd, _unit, guard_files = (
        local_env._maybe_wrap_with_systemd_memory_guard(
            cmd_string="echo hi",
            login=False,
            run_env={},
            cwd="/tmp",
            shell=user_shell,
        )
    )

    assert guard_files is not None
    script_path = Path(guard_files[0])
    assert argv[-1] == str(script_path)
    assert f"{shlex.quote(user_shell)} -c" in script_path.read_text()
    assert f"exec {shlex.quote(user_shell)} -lc" not in script_path.read_text()


def test_run_bash_falls_back_to_unguarded_when_systemd_wrapper_fails(
    monkeypatch, tmp_path
):
    env = object.__new__(local_env.LocalEnvironment)
    env.cwd = str(tmp_path)
    env.env = {}
    env.timeout = 10

    guard_script = tmp_path / "hermes-memguard-script.sh"
    guard_env = tmp_path / "hermes-memguard-env-script.sh"
    guard_script.write_text("script")
    guard_env.write_text("env")

    popen_calls = []

    proc = MagicMock()
    proc.pid = 1001
    proc.stdout = iter([])
    proc.stdin = MagicMock()
    proc.poll.return_value = None

    def fake_popen(*args, **kwargs):
        popen_calls.append(args[0])
        if len(popen_calls) == 1:
            raise RuntimeError("systemd-run unavailable")
        return proc

    monkeypatch.setattr(local_env, "_find_bash", lambda: "/bin/bash")
    monkeypatch.setattr(
        local_env,
        "_maybe_wrap_with_systemd_memory_guard",
        lambda **kwargs: (
            ["systemd-run", "--user", str(guard_script)],
            {"USER_ENV": "1"},
            "/",
            "hermes-terminal-test.service",
            (str(guard_script), str(guard_env)),
        ),
    )
    monkeypatch.setattr(local_env.os, "getpgid", lambda _: 2002)
    monkeypatch.setattr(local_env.subprocess, "Popen", fake_popen)

    result = local_env.LocalEnvironment._run_bash(env, "echo hi", login=False)

    assert len(popen_calls) == 2
    assert popen_calls[0][0] == "systemd-run"
    assert popen_calls[1][0] == "/bin/bash"
    assert not guard_script.exists()
    assert not guard_env.exists()
    assert proc is result


def test_run_bash_falls_back_when_systemd_run_exits_before_command_starts(
    monkeypatch, tmp_path
):
    env = object.__new__(local_env.LocalEnvironment)
    env.cwd = str(tmp_path)
    env.env = {}
    env.timeout = 10

    guard_script = tmp_path / "hermes-memguard-script.sh"
    guard_env = tmp_path / "hermes-memguard-env-script.sh"
    start_pending = tmp_path / "hermes-memguard-start.pending"
    start_ready = tmp_path / "hermes-memguard-start.ready"
    for path in (guard_script, guard_env, start_pending):
        path.write_text("")

    failed_launcher = MagicMock(pid=1001)
    failed_launcher.poll.return_value = 1
    direct_proc = MagicMock(pid=1002)
    direct_proc.poll.return_value = None
    popen_calls = []

    def fake_popen(*args, **kwargs):
        popen_calls.append(args[0])
        return failed_launcher if len(popen_calls) == 1 else direct_proc

    monkeypatch.setattr(local_env, "_find_bash", lambda: "/bin/bash")
    monkeypatch.setattr(
        local_env,
        "_maybe_wrap_with_systemd_memory_guard",
        lambda **kwargs: (
            ["systemd-run", "--user", str(guard_script)],
            {"USER_ENV": "1"},
            "/",
            "hermes-terminal-test.service",
            (
                str(guard_script),
                str(guard_env),
                str(start_pending),
                str(start_ready),
            ),
        ),
    )
    monkeypatch.setattr(local_env.os, "getpgid", lambda _: 2002)
    monkeypatch.setattr(local_env.subprocess, "Popen", fake_popen)

    result = local_env.LocalEnvironment._run_bash(env, "echo hi", login=False)

    assert result is direct_proc
    assert len(popen_calls) == 2
    assert popen_calls[0][0] == "systemd-run"
    assert popen_calls[1][0] == "/bin/bash"
    assert not start_pending.exists()
    assert not start_ready.exists()


def test_run_bash_does_not_duplicate_command_after_start_marker(monkeypatch, tmp_path):
    env = object.__new__(local_env.LocalEnvironment)
    env.cwd = str(tmp_path)
    env.env = {}
    env.timeout = 10

    guard_script = tmp_path / "hermes-memguard-script.sh"
    guard_env = tmp_path / "hermes-memguard-env-script.sh"
    start_pending = tmp_path / "hermes-memguard-start.pending"
    start_ready = tmp_path / "hermes-memguard-start.ready"
    for path in (guard_script, guard_env, start_ready):
        path.write_text("")

    command_proc = MagicMock(pid=1001)
    command_proc.poll.return_value = 23
    popen = MagicMock(return_value=command_proc)

    monkeypatch.setattr(local_env, "_find_bash", lambda: "/bin/bash")
    monkeypatch.setattr(
        local_env,
        "_maybe_wrap_with_systemd_memory_guard",
        lambda **kwargs: (
            ["systemd-run", "--user", str(guard_script)],
            {"USER_ENV": "1"},
            "/",
            "hermes-terminal-test.service",
            (
                str(guard_script),
                str(guard_env),
                str(start_pending),
                str(start_ready),
            ),
        ),
    )
    monkeypatch.setattr(local_env.os, "getpgid", lambda _: 2002)
    monkeypatch.setattr(local_env.subprocess, "Popen", popen)

    result = local_env.LocalEnvironment._run_bash(env, "exit 23", login=False)

    assert result is command_proc
    popen.assert_called_once()


def test_memory_guard_invalid_limit_disables(monkeypatch, caplog):
    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "not-an-int")
    monkeypatch.setattr(local_env, "_systemd_run_available", lambda: True)

    assert local_env._maybe_wrap_with_systemd_memory_guard(
        cmd_string="echo hi",
        login=False,
        run_env={},
        cwd="/tmp",
    ) == (None, None, None, None, None)
    assert "Invalid TERMINAL_LOCAL_MEMORY_MAX_MB" in caplog.text


def test_kill_process_stops_systemd_guard_before_host_tree(monkeypatch):
    registry = ProcessRegistry()
    stopped = []
    terminated = []
    moved = []

    def fake_run(args, **kwargs):
        stopped.append(args)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("tools.process_registry.subprocess.run", fake_run)
    monkeypatch.setattr(
        registry,
        "_terminate_host_pid",
        lambda pid, expected: terminated.append((pid, expected)),
    )
    monkeypatch.setattr(
        registry, "_move_to_finished", lambda session: moved.append(session.id)
    )
    monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)

    proc = Mock()
    proc.pid = 4321
    session = ProcessSession(
        id="proc_memguard",
        command="sleep 60",
        pid=4321,
        process=proc,
        host_start_time=999,
        systemd_unit="hermes-terminal-test.service",
    )
    registry._running[session.id] = session

    result = registry.kill_process(session.id)

    assert result["status"] == "killed"
    assert stopped == [["systemctl", "--user", "stop", "hermes-terminal-test.service"]]
    assert terminated == [(4321, 999)]
    assert moved == ["proc_memguard"]
    assert session.completion_reason == "killed"
    assert session.termination_source == "process.kill"
