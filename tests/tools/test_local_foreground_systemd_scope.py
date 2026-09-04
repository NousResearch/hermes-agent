"""Foreground local commands must not share the gateway cgroup."""

import signal
from unittest.mock import MagicMock

from tools.environments.local import LocalEnvironment


def _bare_local_environment(tmp_path):
    env = object.__new__(LocalEnvironment)
    env.env = {}
    env.cwd = str(tmp_path)
    return env


def test_gateway_foreground_command_uses_bounded_systemd_scope(monkeypatch, tmp_path):
    env = _bare_local_environment(tmp_path)
    proc = MagicMock(pid=12345)
    popen = MagicMock(return_value=proc)

    monkeypatch.setattr("tools.environments.local._IS_WINDOWS", False)
    monkeypatch.setattr("tools.environments.local._find_bash", lambda: "/bin/bash")
    monkeypatch.setattr("tools.environments.local._make_run_env", lambda value: value)
    monkeypatch.setattr("tools.environments.local.os.getpgid", lambda pid: pid)
    monkeypatch.setattr("tools.environments.local.subprocess.Popen", popen)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")
    monkeypatch.setattr(
        "tools.process_registry._is_supervised_gateway_process", lambda: True
    )
    monkeypatch.setattr(
        "tools.process_registry._systemd_run_user_scope_available", lambda: True
    )
    monkeypatch.setattr(
        "tools.process_registry._worker_memory_max_bytes", lambda: 4 * 1024**3
    )

    result = env._run_bash("printf ready")

    argv = popen.call_args.args[0]
    assert argv[:5] == [
        "/usr/bin/systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "--unit",
    ]
    assert argv[5].startswith("hermes-worker-foreground-")
    assert "MemoryMax=4294967296" in argv
    assert "OOMPolicy=kill" in argv
    assert argv[-4:] == ["--", "/bin/bash", "-c", "printf ready"]
    assert getattr(result, "_hermes_systemd_unit") == f"{argv[5]}.scope"
    assert popen.call_args.kwargs["start_new_session"] is True


def test_kill_foreground_scope_stops_cgroup_before_process_group(monkeypatch, tmp_path):
    env = _bare_local_environment(tmp_path)
    proc = MagicMock(pid=12345)
    setattr(proc, "_hermes_systemd_unit", "hermes-worker-foreground-test.scope")
    stopped = []
    killpg = MagicMock()

    monkeypatch.setattr("tools.environments.local._IS_WINDOWS", False)
    monkeypatch.setattr(
        "tools.process_registry._stop_systemd_unit",
        lambda unit: stopped.append(unit) or True,
    )
    monkeypatch.setattr("tools.environments.local.os.killpg", killpg)

    env._kill_process(proc)

    assert stopped == ["hermes-worker-foreground-test.scope"]
    killpg.assert_not_called()


def test_non_gateway_foreground_command_keeps_direct_shell_spawn(monkeypatch, tmp_path):
    env = _bare_local_environment(tmp_path)
    proc = MagicMock(pid=12345)
    popen = MagicMock(return_value=proc)

    monkeypatch.setattr("tools.environments.local._IS_WINDOWS", False)
    monkeypatch.setattr("tools.environments.local._find_bash", lambda: "/bin/bash")
    monkeypatch.setattr("tools.environments.local._make_run_env", lambda value: value)
    monkeypatch.setattr("tools.environments.local.os.getpgid", lambda pid: pid)
    monkeypatch.setattr("tools.environments.local.subprocess.Popen", popen)
    monkeypatch.setattr(
        "tools.process_registry._is_supervised_gateway_process", lambda: False
    )

    result = env._run_bash("printf ready")

    assert popen.call_args.args[0] == ["/bin/bash", "-c", "printf ready"]
    assert "_hermes_systemd_unit" not in vars(result)


def test_scope_builder_fallback_does_not_record_nonexistent_unit(monkeypatch, tmp_path):
    env = _bare_local_environment(tmp_path)
    proc = MagicMock(pid=12345)
    popen = MagicMock(return_value=proc)

    monkeypatch.setattr("tools.environments.local._IS_WINDOWS", False)
    monkeypatch.setattr("tools.environments.local._find_bash", lambda: "/bin/bash")
    monkeypatch.setattr("tools.environments.local._make_run_env", lambda value: value)
    monkeypatch.setattr("tools.environments.local.os.getpgid", lambda pid: pid)
    monkeypatch.setattr("tools.environments.local.subprocess.Popen", popen)
    monkeypatch.setattr(
        "tools.process_registry._is_supervised_gateway_process", lambda: True
    )
    monkeypatch.setattr(
        "tools.process_registry._systemd_run_user_scope_available", lambda: True
    )
    monkeypatch.setattr(
        "tools.process_registry._build_systemd_scope_argv",
        lambda argv, *, unit_suffix: argv,
    )

    result = env._run_bash("printf ready")

    assert popen.call_args.args[0] == ["/bin/bash", "-c", "printf ready"]
    assert "_hermes_systemd_unit" not in vars(result)


def test_failed_scope_stop_falls_back_to_process_group(monkeypatch, tmp_path):
    env = _bare_local_environment(tmp_path)
    proc = MagicMock(pid=12345)
    setattr(proc, "_hermes_systemd_unit", "hermes-worker-foreground-test.scope")
    stopped = []
    killpg_calls = []

    def fake_killpg(pgid, sig):
        killpg_calls.append((pgid, sig))
        if sig == 0:
            raise ProcessLookupError

    monkeypatch.setattr("tools.environments.local._IS_WINDOWS", False)
    monkeypatch.setattr(
        "tools.process_registry._stop_systemd_unit",
        lambda unit: stopped.append(unit) or False,
    )
    monkeypatch.setattr("tools.environments.local.os.getpgid", lambda pid: 67890)
    monkeypatch.setattr("tools.environments.local.os.killpg", fake_killpg)

    env._kill_process(proc)

    assert stopped == ["hermes-worker-foreground-test.scope"]
    assert killpg_calls == [(67890, signal.SIGTERM), (67890, 0)]
