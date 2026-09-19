"""Execution tools must report active security-config mutations safely."""

import json
import shlex
import sys

import pytest

import hermes_cli.config as hermes_config
from tools.code_execution_tool import execute_code
from tools.security_config_guard import ActiveConfigSnapshot


def _config_home(tmp_path, monkeypatch, content=b"approvals:\n  mode: smart\n"):
    home = tmp_path / "hermes"
    home.mkdir()
    config_path = home / "config.yaml"
    config_path.write_bytes(content)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    hermes_config._LOAD_CONFIG_CACHE.clear()
    return home, config_path


def _symlink_config(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    target_a = home / "config-a.yaml"
    target_b = home / "config-b.yaml"
    target_a.write_text("approvals:\n  mode: smart\n", encoding="utf-8")
    target_b.write_text("approvals:\n  mode: off\n", encoding="utf-8")
    config_path = home / "config.yaml"
    try:
        config_path.symlink_to(target_a)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable on this host: {exc}")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    hermes_config._LOAD_CONFIG_CACHE.clear()
    return config_path, target_b


def test_execute_code_reports_direct_config_write_without_stale_rollback(tmp_path, monkeypatch):
    _, config_path = _config_home(tmp_path, monkeypatch)
    changed = b"approvals:\n  mode: off\n"

    result = json.loads(execute_code(
        code=f"open({str(config_path)!r}, 'wb').write({changed!r})",
        task_id="config-write-guard",
        reset=True,
    ))

    assert result["status"] == "error"
    assert "modified the active Hermes config.yaml" in result["error"]
    assert "was not rolled back" in result["error"]
    assert config_path.read_bytes() == changed


def test_terminal_reports_direct_config_write_without_stale_rollback(tmp_path, monkeypatch):
    _, config_path = _config_home(
        tmp_path, monkeypatch, b"hooks:\n  pre_tool_call: []\n"
    )
    changed = b"hooks: {}\n"

    from tools.terminal_tool import terminal_tool

    script = f"open({config_path.as_posix()!r}, 'wb').write({changed!r})"
    command = f"{shlex.quote(sys.executable.replace(chr(92), '/'))} -c {shlex.quote(script)}"
    result = json.loads(terminal_tool(command, task_id="terminal-config-write-guard"))

    assert result["exit_code"] == 126, result
    assert "modified the active Hermes config.yaml" in result["error"]
    assert "was not rolled back" in result["error"]
    assert config_path.read_bytes() == changed


def test_snapshot_detects_config_symlink_retarget(tmp_path, monkeypatch):
    config_path, target_b = _symlink_config(tmp_path, monkeypatch)
    snapshot, error = ActiveConfigSnapshot.capture()
    assert error is None
    assert snapshot is not None

    config_path.unlink()
    config_path.symlink_to(target_b)

    assert snapshot.mutation_error() is not None


def test_execute_code_detects_config_symlink_retarget(tmp_path, monkeypatch):
    config_path, target_b = _symlink_config(tmp_path, monkeypatch)
    code = (
        "import os\n"
        f"os.unlink({str(config_path)!r})\n"
        f"os.symlink({str(target_b)!r}, {str(config_path)!r})\n"
    )

    result = json.loads(execute_code(
        code=code,
        task_id="config-symlink-retarget-guard",
        reset=True,
    ))

    assert result["status"] == "error"
    assert config_path.resolve() == target_b.resolve()


def test_terminal_detects_config_symlink_retarget(tmp_path, monkeypatch):
    config_path, target_b = _symlink_config(tmp_path, monkeypatch)

    from tools.terminal_tool import terminal_tool

    script = (
        f"import os; os.unlink({config_path.as_posix()!r}); "
        f"os.symlink({target_b.as_posix()!r}, {config_path.as_posix()!r})"
    )
    command = f"{shlex.quote(sys.executable.replace(chr(92), '/'))} -c {shlex.quote(script)}"
    result = json.loads(terminal_tool(command, task_id="terminal-symlink-retarget-guard"))

    assert result["exit_code"] == 126, result
    assert config_path.resolve() == target_b.resolve()


def test_background_terminal_does_not_capture_a_stale_config_generation(
    tmp_path, monkeypatch
):
    _, config_path = _config_home(tmp_path, monkeypatch)

    class FakeEnv:
        env = {}
        cwd = str(tmp_path)

    class FakeRegistry:
        pending_watchers = []

        def spawn_local(self, **kwargs):
            config_path.write_text("approvals:\n  mode: manual\n", encoding="utf-8")
            return type("Session", (), {"id": "proc_safe", "pid": 1234})()

    import tools.process_registry as process_registry_module
    import tools.terminal_tool as terminal_module

    task_id = "background-config-generation"
    monkeypatch.setattr(terminal_module, "_active_environments", {task_id: FakeEnv()})
    monkeypatch.setattr(terminal_module, "_last_activity", {})
    monkeypatch.setattr(terminal_module, "_session_cwd", {})
    monkeypatch.setattr(terminal_module, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_module, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_module, "_resolve_container_task_id", lambda value: value)
    monkeypatch.setattr(
        terminal_module,
        "_get_env_config",
        lambda: {
            "env_type": "local",
            "cwd": str(tmp_path),
            "timeout": 60,
            "lifetime_seconds": 3600,
        },
    )
    monkeypatch.setattr(
        terminal_module,
        "_check_all_guards",
        lambda command, env_type, **kwargs: {"approved": True},
    )
    monkeypatch.setattr(process_registry_module, "process_registry", FakeRegistry())
    monkeypatch.setattr(
        ActiveConfigSnapshot,
        "capture",
        classmethod(lambda cls: pytest.fail("background command captured config state")),
    )

    result = json.loads(terminal_module.terminal_tool(
        command="trusted-background-process",
        task_id=task_id,
        background=True,
    ))

    assert result["session_id"] == "proc_safe"
    assert config_path.read_text(encoding="utf-8") == "approvals:\n  mode: manual\n"
