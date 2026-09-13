"""Registry file calls must honor local execution routing, not just backend type."""
import json
import shlex
import shutil
import sys

import pytest

from hermes_cli.session_execution import (
    SessionExecutionContext,
    register_session_execution_context,
    remove_session_execution_context,
)
from tools import file_tools, terminal_tool
from tools.registry import registry


def _dispatch(tool, args, **kwargs):
    result = registry.dispatch(tool, args, **kwargs)
    assert isinstance(result, str), result
    return json.loads(result)


@pytest.fixture
def local_files(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "1")
    monkeypatch.setattr(file_tools, "_file_ops_cache", {})
    monkeypatch.setattr(terminal_tool, "_active_environments", {})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    witness = tmp_path / "witness.txt"
    witness.write_text("routing-needle\n", encoding="utf-8")
    yield witness
    for env in terminal_tool._active_environments.values():
        env.cleanup()


@pytest.mark.linux_only
@pytest.mark.parametrize("tool", ["search_files", "read_file"])
def test_registered_file_calls_run_through_real_prefix(local_files, tmp_path, monkeypatch, tool):
    rg = shutil.which("rg")
    assert rg is not None, "integration test requires real ripgrep"
    log = tmp_path / "launch.jsonl"
    launcher = tmp_path / "launch.py"
    launcher.write_text(
        "import json, os, sys\n"
        f"with open({str(log)!r}, 'a', encoding='utf-8') as f:\n"
        "    f.write(json.dumps({'argv': sys.argv[1:], 'owner': os.getenv('FILE_ROUTE_OWNER'), "
        "'removed': os.getenv('FILE_ROUTE_HOST')}) + '\\n')\n"
        "os.execvpe(sys.argv[1], sys.argv[1:], os.environ)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FILE_ROUTE_HOST", "host-only")
    valid = [True]
    sid, task = "file-routing-owner", "file-routing-task"
    register_session_execution_context(sid, SessionExecutionContext(
        env_set={"FILE_ROUTE_OWNER": sid}, env_unset=frozenset({"FILE_ROUTE_HOST"}),
        command_prefix=(sys.executable, str(launcher)), validate=lambda: valid[0],
    ), task_ids=(task,))
    try:
        ops = file_tools._get_file_ops(task)
        assert ops.env.execution_context is not None
        # Warm command discovery separately: its prefix alone cannot prove the
        # actual search/read was routed (the native optimization skips only that).
        assert ops._resolve_command("rg") == rg
        log.write_text("", encoding="utf-8")
        args = {"path": str(local_files)}
        if tool == "search_files":
            args["pattern"] = "routing-needle"
        result = _dispatch(tool, args, task_id=task, session_id=sid)
        assert not result.get("error"), result
        assert "routing-needle" in json.dumps(result)
        launches = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        commands = [shlex.join(row["argv"]) for row in launches]
        assert any(str(local_files) in command for command in commands), launches
        if tool == "search_files":
            assert result["total_count"] == 1
            assert any(rg in command and "routing-needle" in command for command in commands), launches
        assert all(row["owner"] == sid and row["removed"] is None for row in launches)
        # A cached backend must still reject a stale lease before any new launch.
        valid[0] = False
        before = log.read_text(encoding="utf-8")
        denied = _dispatch(tool, args, task_id=task, session_id=sid)
        assert "validation failed" in denied.get("error", ""), denied
        assert log.read_text(encoding="utf-8") == before
    finally:
        remove_session_execution_context(sid)


@pytest.mark.linux_only
@pytest.mark.parametrize("tool", ["search_files", "read_file"])
def test_unregistered_file_calls_keep_native_fast_path(local_files, monkeypatch, tool):
    task = "ordinary-file-task"
    ops = file_tools._get_file_ops(task)
    assert ops.env.execution_context is None
    assert ops._resolve_command("rg") == shutil.which("rg")
    native_calls, shell_calls = [], []
    method = "_run_rg_native" if tool == "search_files" else "_read_file_native"
    real_native, real_execute = getattr(ops, method), ops.env.execute

    def native(*args, **kwargs):
        native_calls.append(args)
        return real_native(*args, **kwargs)

    def execute(*args, **kwargs):
        shell_calls.append(args)
        return real_execute(*args, **kwargs)

    monkeypatch.setattr(ops, method, native)
    monkeypatch.setattr(ops.env, "execute", execute)
    args = {"path": str(local_files)}
    if tool == "search_files":
        args["pattern"] = "routing-needle"
    result = _dispatch(tool, args, task_id=task)
    assert not result.get("error"), result
    assert "routing-needle" in json.dumps(result)
    assert native_calls
    assert not shell_calls
