"""The public command context a terminal backend reads in ``execute()``.

A plugin environment receives commands from many Hermes call sites and must be able to tell
Hermes's own work from the model's commands (and to know whether the dangerous-command gate
already approved one) without walking the Python call stack. Every site stamps
``tools.environments.command_context``; these tests drive each site through a recording
environment and assert what ``execute()`` sees.
"""

import asyncio
import base64
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.environments.command_context import (
    CommandApproval, CommandContext, command_context, current_command_context,
)
from tools.process_registry import ProcessRegistry, ProcessSession

PNG = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


class RecordingEnv:
    """BaseEnvironment duck type that records the command context of every ``execute()``."""

    def __init__(self, output="", responder=None):
        self.commands = []
        self.contexts = []
        self.cwd = "~"
        self.output = output
        self.responder = responder

    def execute(self, command, cwd="", *, timeout=None, stdin_data=None, **kwargs):
        self.commands.append(command)
        self.contexts.append(current_command_context())
        output = self.responder(command) if self.responder else self.output
        return {"output": output, "returncode": 0}

    def cleanup(self):
        pass

    def get_temp_dir(self):
        return "/tmp"

    def origins(self):
        return [c.origin if c is not None else None for c in self.contexts]


def _terminal_config(tmp_path):
    """The ``_get_env_config()`` shape ``terminal_tool`` reads (as tests do elsewhere)."""
    return {
        "env_type": "local", "timeout": 180, "cwd": str(tmp_path), "host_cwd": None,
        "modal_mode": "auto", "docker_image": "", "modal_image": "", "daytona_image": "",
    }


def _run_terminal_tool(env, command, guard_result=None, **kwargs):
    """Drive ``terminal_tool`` against *env* with the guard stubbed to *guard_result*."""
    from tools import terminal_tool
    with patch("tools.terminal_tool._get_env_config",
               return_value=_terminal_config(Path("/tmp"))), \
            patch("tools.terminal_tool._start_cleanup_thread"), \
            patch("tools.terminal_tool._acquire_env", return_value=env), \
            patch("tools.terminal_tool._check_all_guards",
                  return_value=guard_result if guard_result is not None else {"approved": True}):
        return json.loads(terminal_tool.terminal_tool(command=command, **kwargs))


def test_no_context_outside_a_stamped_site(tmp_path, monkeypatch):
    """The default for anything that has not opted in is "unknown", never a stale stamp."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    assert current_command_context() is None
    with command_context("terminal", task_id="t1"):
        assert current_command_context() == CommandContext(origin="terminal", task_id="t1")
    assert current_command_context() is None


def test_terminal_call_stamps_the_command_the_gate_saw(tmp_path, monkeypatch):
    """``origin="terminal"`` carries the exact command the gate approved, so a backend can trust it."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    env = RecordingEnv()
    result = _run_terminal_tool(
        env, "rm -rf build/",
        {"approved": True, "user_approved": True, "description": "recursive delete"},
    )

    assert result["exit_code"] == 0
    assert len(env.contexts) == 1
    ctx = env.contexts[0]
    assert ctx.origin == "terminal"
    assert ctx.approval == CommandApproval(command="rm -rf build/", kind="user")


@pytest.mark.parametrize("guard,kind", [
    ({"approved": True, "user_approved": True}, "user"),
    ({"approved": True, "smart_approved": True}, "smart"),
    ({"approved": True, "pre_approved": [{"description": "recursive delete", "scope": "session"}]},
     "pre_approved"),
    ({"approved": True}, "no_findings"),
    ({"approved": True, "gate_skipped": True}, "gate_skipped"),
    ({"approved": True, "batch": True, "user_approved": True}, "batch"),
])
def test_terminal_call_reports_how_the_gate_let_it_through(tmp_path, monkeypatch, guard, kind):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    env = RecordingEnv()
    _run_terminal_tool(env, "ls", guard)

    assert env.contexts[0].approval == CommandApproval(command="ls", kind=kind)


def test_forced_run_is_stamped_as_forced(tmp_path, monkeypatch):
    """``force`` skips the gate entirely; the backend is told that instead of a fake approval."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    env = RecordingEnv()
    _run_terminal_tool(env, "ls", force=True)

    assert env.contexts[0].approval == CommandApproval(command="ls", kind="forced")


def test_background_spawn_stamps_its_job_and_the_gated_command(monkeypatch):
    env = RecordingEnv(output="4242\n")
    registry = ProcessRegistry()
    monkeypatch.setattr(registry, "_track_started", lambda *a, **k: None)
    approval = CommandApproval(command="python job.py", kind="user")

    session = registry.spawn_via_env(env, "python job.py", task_id="t1", session_key="s1",
                                     approval=approval)

    ctx = env.contexts[0]
    assert ctx.origin == "background_spawn"
    assert ctx.job_id == session.id
    assert ctx.inner_command == "python job.py"
    assert ctx.approval == approval


def test_poller_and_stop_stamp_the_job_without_an_approval(monkeypatch):
    """Poller threads start under the spawn's copied context, so they set their own value."""
    env = RecordingEnv(output="0 0 exit:0\n")
    registry = ProcessRegistry()
    session = ProcessSession(id="proc_ctx", command="sleep 5", task_id="t1",
                             started_at=time.time(), output_buffer="")
    session.exited = False
    with patch("tools.process_registry_env_scope.time.sleep", return_value=None), \
            patch.object(registry, "_move_to_finished"):
        registry._env_poller_loop(session, env, "/tmp/bg.log", "/tmp/bg.pid", "/tmp/bg.exit")

    assert env.origins() == ["background_poll"] * len(env.contexts)
    assert all(c.job_id == session.id for c in env.contexts)
    assert all(c.approval is None for c in env.contexts)

    killed = RecordingEnv()
    session.env_ref, session.pid, session.pid_scope = killed, 4242, "sandbox"
    registry._signal_kill(session, consume_output=False)

    assert killed.commands == ["kill 4242 2>/dev/null"]
    assert killed.contexts[0].origin == "background_kill"
    assert killed.contexts[0].job_id == session.id


def _run_execute_code(env, code, guard):
    """Drive the real ``execute_code`` entry point against a remote *env*."""
    from tools import code_execution_tool
    with patch.object(code_execution_tool, "_get_or_create_env", return_value=(env, "docker")), \
            patch.object(code_execution_tool, "_load_config", return_value={"timeout": 30, "max_tool_calls": 5}), \
            patch.object(code_execution_tool, "_sandbox_tools_for",
                         return_value=frozenset({"terminal"})), \
            patch("tools.approval.check_execute_code_guard", return_value=guard), \
            patch("tools.terminal_tool._get_env_config", return_value={"env_type": "docker"}), \
            patch("tools.terminal_tool._docker_has_host_access", return_value=False), \
            patch("tools.terminal_tool._resolve_container_task_id", side_effect=lambda value: value), \
            patch("tools.code_kernel_remote.execute_in_remote_kernel", return_value=None), \
            patch("tools.process_registry._is_supervised_gateway_process", return_value=False), \
            patch("hermes_cli.observability.shared_metrics_loop.record_execution_backend",
                  side_effect=lambda kind, backend, result: result):
        return json.loads(code_execution_tool.execute_code(code, task_id="t1"))


def test_execute_code_separates_the_script_launch_from_its_own_traffic(monkeypatch, tmp_path):
    """The script the gate approved as a whole is ``execute_code``; the files and probes
    around it are ``execute_code_rpc`` — a backend must not mistake a script approval for a
    terminal-command one."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    env = RecordingEnv(output="OK")
    code = "print('hi')"

    _run_execute_code(env, code, {"approved": True, "user_approved": True})

    launches = [c for c in env.contexts if c is not None and c.origin == "execute_code"]
    assert len(launches) == 1
    assert launches[0].approval == CommandApproval(command=code, kind="execute_code:user")
    rpc = [c for c in env.contexts if c is not None and c.origin == "execute_code_rpc"]
    assert rpc, "the ship/probe/cleanup commands must announce themselves"
    assert all(c.approval is None for c in rpc)


def test_nested_terminal_call_inside_execute_code_is_a_terminal_call(tmp_path, monkeypatch):
    """execute_code dispatches nested tool calls in its own copied context; the terminal tool
    stamps its own and the reset on return restores execute_code's, so nothing leaks."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    env = RecordingEnv()
    with command_context("execute_code", approval=CommandApproval("print(1)", "execute_code:user")):
        _run_terminal_tool(env, "echo hi")
        assert current_command_context().origin == "execute_code"

    assert env.contexts[0].origin == "terminal"
    assert env.contexts[0].approval == CommandApproval("echo hi", "no_findings")


def test_file_operations_announce_file_ops():
    from tools.file_operations import ShellFileOperations
    env = RecordingEnv(output="")

    ShellFileOperations(env)._exec("ls -1")

    assert env.contexts[0].origin == "file_op"


def test_media_read_announces_itself(tmp_path, monkeypatch):
    """Both media readers — the sandbox image read and the attachment fetch — are ``media_read``."""
    from tools.environments.base import BaseEnvironment

    payload = base64.b64encode(PNG).decode()

    def marker_fenced(command):
        marker = re.search(r"__HERMES_FETCH_[0-9a-f]+__", command).group(0)
        return f"{marker}\n{payload}\n{marker}"

    env = RecordingEnv(responder=marker_fenced)
    BaseEnvironment.fetch_file(env, "/tmp/x.png", tmp_path / "x.png", max_bytes=1024)

    assert env.contexts[0].origin == "media_read"
    assert (tmp_path / "x.png").read_bytes() == PNG

    image_env = RecordingEnv(output=base64.b64encode(PNG).decode())
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    from tools import image_source as isrc
    with patch("tools.image_source._get_active_env", return_value=image_env), \
            patch("tools.image_source._ensure_container_env", lambda task_id: None):
        resolved = asyncio.run(isrc._resolve_container_fallback(
            Path("/sandbox/shot.png"), isrc.ResolveContext(task_id="t1"), "shot.png"))

    assert resolved.origin == "container"
    assert image_env.origins() == ["media_read"]


def test_spill_write_announces_itself(tmp_path, monkeypatch):
    from tools.tool_result_storage import _write_to_sandbox
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    env = RecordingEnv(output="11")  # the wc -c round-trip verification

    assert _write_to_sandbox("hello world", "/tmp/hermes-results/r1.txt", env) is True
    assert env.origins() == ["spill", "spill"]
    assert all(c.approval is None for c in env.contexts)


def test_guard_script_read_announces_itself():
    from tools.terminal_tool_guards import _read_script_for_guard
    env = RecordingEnv(output="echo hi")

    assert _read_script_for_guard(env, "/", "relative/script.sh", 4096) == "echo hi"
    assert env.contexts[0].origin == "guard_probe"


def test_prompt_builder_backend_probe_announces_itself():
    from agent import prompt_builder
    env = RecordingEnv(output="os=Linux\nkernel=6.1\n")
    fake_terminal_tool = SimpleNamespace(_get_env_config=lambda: {
        "cwd": "/tmp", "timeout": 10, "host_cwd": None,
    }, _is_container_backend=lambda env_type: False)

    with patch("tools.terminal_tool_backends._create_environment", return_value=env), \
            patch("tools.terminal_tool_lifecycle._cleanup_env", lambda env, **kw: None):
        output = prompt_builder._run_backend_probe("ssh", fake_terminal_tool)

    assert "os=Linux" in output
    assert env.contexts[0].origin == "internal"
