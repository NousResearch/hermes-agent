"""The public command context a terminal backend reads in ``execute()``.

A plugin environment receives commands from many Hermes call sites and must be able to tell
Hermes's own work from the model's commands (and to know whether the dangerous-command gate
already approved one) without walking the Python call stack. Every site stamps
``tools.environments.command_context``; these tests drive each site through a recording
environment and assert what ``execute()`` sees.
"""

import asyncio
import base64
import contextlib
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
        self.stdins = []
        self.contexts = []
        self.cwd = "~"
        self.output = output
        self.responder = responder

    def execute(self, command, cwd="", *, timeout=None, stdin_data=None, **kwargs):
        self.commands.append(command)
        self.stdins.append(stdin_data)
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
    """A poller thread copies its caller's context (which may carry the spawn's stamp), so it sets its own value."""
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


def _run_execute_code(env, code, guard, *, kernel_result=None, use_kernel=False):
    """Drive the real ``execute_code`` entry point against a remote *env*; ``use_kernel`` leaves
    the real remote-kernel path in place instead of patching it out to the per-call fallback."""
    from tools import code_execution_tool
    with patch.object(code_execution_tool, "_get_or_create_env", return_value=(env, "docker")), \
            patch.object(code_execution_tool, "_load_config", return_value={"timeout": 30, "max_tool_calls": 5}), \
            patch.object(code_execution_tool, "_sandbox_tools_for",
                         return_value=frozenset({"terminal"})), \
            patch("tools.approval.check_execute_code_guard", return_value=guard), \
            patch("tools.terminal_tool._get_env_config", return_value={"env_type": "docker"}), \
            patch("tools.terminal_tool._docker_has_host_access", return_value=False), \
            patch("tools.terminal_tool._resolve_container_task_id", side_effect=lambda value: value), \
            patch("tools.code_kernel_remote.execute_in_remote_kernel", return_value=kernel_result) \
            if not use_kernel else contextlib.nullcontext(), \
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


def test_spill_readability_probe_announces_itself(monkeypatch):
    """The ``test -r`` probe that decides whether a spill is visible in the sandbox is Hermes's
    own traffic, not an unknown caller's."""
    from tools.tool_result_storage import _sandbox_visible_spillover_path
    env = RecordingEnv()
    monkeypatch.setattr("tools.credential_files.to_agent_visible_cache_path", lambda path: path)

    assert _sandbox_visible_spillover_path("/host/cache/spill.txt", env) == "/host/cache/spill.txt"
    assert env.commands == ["test -r /host/cache/spill.txt"]
    assert env.origins() == ["spill"]


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


def _kernel_env():
    """A remote that behaves like a live kernel host: a runner pid, liveness, one finished cell."""
    def respond(command):
        if "command -v python3" in command:
            return "OK"
        if "kill -0" in command:
            return "ALIVE"
        if "cat " in command and "cell_res_" in command:
            return json.dumps({"status": "ok", "stdout": "hi\n", "stderr": "", "execution_count": 1})
        if "nohup python3" in command:
            return "PID:4242"
        return ""
    return RecordingEnv(responder=respond)


def test_remote_kernel_ships_the_model_script_as_execute_code(monkeypatch, tmp_path):
    """The kernel path ships the model's code in the cell request. That one ``execute()`` carries
    ``origin="execute_code"`` and the gate's verdict; the mkdir, liveness, result read and cleanup
    around it stay ``execute_code_rpc``."""
    from tools import code_kernel_remote
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    env = _kernel_env()
    code = "print('model script')"

    try:
        result = _run_execute_code(env, code, {"approved": True, "user_approved": True}, use_kernel=True)
    finally:
        code_kernel_remote.shutdown_all_remote_kernels()

    assert result["status"] == "success", result
    ships = [ctx for ctx, stdin in zip(env.contexts, env.stdins)
             if stdin and code in base64.b64decode(stdin).decode("utf-8", "replace")]
    assert [ctx.origin for ctx in ships] == ["execute_code"]
    assert ships[0].approval == CommandApproval(command=code, kind="execute_code:user")
    assert ships[0].task_id == "t1"
    others = [ctx for ctx in env.contexts if ctx not in ships]
    assert others and all(ctx is not None and ctx.approval is None for ctx in others)
    assert {ctx.origin for ctx in others} <= {"execute_code_rpc", "internal"}


def test_a_plugin_can_read_the_context_but_not_set_it():
    """The reader and its value types are public; the setter is Hermes's, so a plugin cannot forge an
    approval stamp."""
    import agent.terminal_env_provider as provider

    assert {"CommandApproval", "CommandContext", "current_command_context"} <= set(provider.__all__)
    assert "command_context" not in provider.__all__
    assert not hasattr(provider, "command_context")



# ---- the exact command Hermes's own code sends (a backend's second check on Hermes-owned work) ----

def _exact(env):
    return [(c.command if c is not None else None) for c in env.contexts]


def test_background_spawn_names_the_exact_wrapper_it_sends(monkeypatch):
    env = RecordingEnv(output="4242\n")
    registry = ProcessRegistry()
    monkeypatch.setattr(registry, "_track_started", lambda *a, **k: None)
    registry.spawn_via_env(env, "python job.py", task_id="t1")
    assert _exact(env) == env.commands


def test_every_poll_and_status_read_names_its_own_exact_text():
    """The poller's combined read, and the two separate reads it sends when a reply is unreadable, each carry
    exactly the text sent: a command run inside the poller's block with other text would not match."""
    replies = iter(["garbled", "1", "0"])
    env = RecordingEnv(responder=lambda command: next(replies, "0 0 exit:0\n"))
    registry = ProcessRegistry()
    session = ProcessSession(id="proc_exact", command="sleep 5", task_id="t1",
                             started_at=time.time(), output_buffer="")
    session.exited = False
    with patch("tools.process_registry_env_scope.time.sleep", return_value=None), \
            patch.object(registry, "_move_to_finished"):
        registry._env_poller_loop(session, env, "/tmp/bg.log", "/tmp/bg.pid", "/tmp/bg.exit")
    assert len(env.commands) >= 3                       # the combined read, then kill -0 and the exit-file read
    assert _exact(env) == env.commands
    assert env.origins() == ["background_poll"] * len(env.commands)
    assert all(c.job_id == session.id for c in env.contexts)


def test_a_stop_names_its_exact_kill():
    registry = ProcessRegistry()
    session = ProcessSession(id="proc_kill", command="sleep 5", task_id="t1",
                             started_at=time.time(), output_buffer="")
    killed = RecordingEnv()
    session.env_ref, session.pid, session.pid_scope = killed, 4242, "sandbox"
    registry._signal_kill(session, consume_output=False)
    assert _exact(killed) == ["kill 4242 2>/dev/null"]


def test_execute_code_cleanup_names_only_its_own_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    env = RecordingEnv(output="OK")
    _run_execute_code(env, "print('hi')", {"approved": True, "user_approved": True})
    cleanups = [(cmd, ctx) for cmd, ctx in zip(env.commands, env.contexts) if cmd.startswith("rm -rf ")]
    assert len(cleanups) == 1
    command, ctx = cleanups[0]
    assert ctx.origin == "execute_code_rpc" and ctx.command == command


def test_a_command_without_a_fixed_text_names_none():
    """A terminal call's text is the model's: the stamp names no Hermes-owned command for it."""
    with command_context("terminal", approval=CommandApproval("ls", "user")):
        assert current_command_context().command is None
