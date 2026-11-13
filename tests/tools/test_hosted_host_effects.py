"""Effects an allowed hosted tool must never leave on this host, and the host features a hosted
turn must never reach (skills, shell hooks). Each check runs the real code path under a hosted
scope and asserts no host artifact; the local counterpart asserts behaviour there is unchanged."""

from __future__ import annotations

import json
import shlex
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

from hermes_agent.agent.runtime_scope import bind_state
from tests.tools.test_sandbox_scope_conformance import A
from tests.tools.test_sandbox_scope_conformance import host_dir as _conformance_host_dir
from tests.tools.test_sandbox_scope_conformance import provider as _conformance_provider
from hermes_agent.tools.environments import sandbox_conformance as c

provider = _conformance_provider  # the recording-sandbox fixture, by its name
host_dir = _conformance_host_dir


# --------------------------------------------------------------------------------------------------
# execute_code stdout spill


def _remote_execute_code(provider, monkeypatch, payload):
    from hermes_agent.tools import approval
    import hermes_agent.tools.code_kernel_remote as remote
    import hermes_agent.tools.process_registry as processes
    from hermes_agent.tools.registry import registry

    original = c.RecordingSandboxEnvironment.execute

    def execute(self, command, **kwargs):
        result = original(self, command, **kwargs)
        return {"returncode": 0, "output": "OK"} if "command -v python3" in command else result

    monkeypatch.setattr(c.RecordingSandboxEnvironment, "execute", execute)
    monkeypatch.setattr(processes, "_is_supervised_gateway_process", lambda: False)
    monkeypatch.setattr(approval, "check_execute_code_guard", lambda *a, **k: {"approved": True})
    seen = []

    def cell(*args, **kwargs):
        seen.append(kwargs["env"].sandbox)
        return {"status": "success", "stdout": payload, "kernel": {"remote": True}}

    monkeypatch.setattr(remote, "execute_in_remote_kernel", cell)
    with bind_state(A):
        result = json.loads(registry.dispatch("execute_code", {"code": "print('x')"}, task_id="t"))
    assert seen and result["status"] == "success", result
    return result


def test_execute_code_remote_stdout_never_spills_to_host(provider, monkeypatch):
    from hermes_constants import get_hermes_home
    result = _remote_execute_code(provider, monkeypatch, "TENANT-STDOUT-PRIVATE\n" * 4000)
    exec_cache = get_hermes_home() / "cache" / "exec"
    assert not (exec_cache.exists() and any("TENANT-STDOUT-PRIVATE" in p.read_text() for p in exec_cache.iterdir()))
    path = result.get("stdout_spill_path")
    assert path and not Path(path).exists(), result  # a path in the sandbox, not on this host
    writes = [call for call in provider.calls if call.op == "execute" and "TENANT-STDOUT-PRIVATE" in call.stdin]
    assert writes and all(call.sandbox == c._identity(A) for call in writes)
    assert path in writes[-1].command and result["stdout_truncated"] is True


def test_local_execute_code_stdout_still_spills_to_cache():
    from hermes_constants import get_hermes_home
    from hermes_agent.tools import code_execution_tool
    _text, meta = code_execution_tool._truncate_stdout_text("LOCAL-STDOUT\n" * 8000)
    path = Path(meta["stdout_spill_path"])
    assert path.parent == get_hermes_home() / "cache" / "exec" and "LOCAL-STDOUT" in path.read_text()


# --------------------------------------------------------------------------------------------------
# Skills


def _plant_skill(monkeypatch):
    from hermes_constants import get_hermes_home
    import hermes_agent.tools.skills_tool as skills
    root = get_hermes_home() / "skills"
    skill = root / "private_probe"
    skill.mkdir(parents=True, exist_ok=True)
    (skill / "SKILL.md").write_text("---\nname: private_probe\ndescription: Private review probe.\n---\nHOST-SKILL-PRIVATE\n")
    monkeypatch.setattr(skills, "SKILLS_DIR", root)
    return root


def test_preloaded_skills_never_read_host(monkeypatch):
    from hermes_agent.agent import skill_commands
    from hermes_agent.agent.prompt_builder import build_skills_system_prompt
    from hermes_agent.tools.skills_tool import skill_view, skills_list
    root = _plant_skill(monkeypatch)
    read = []
    real_read = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: read.append(self) or real_read(self, *a, **k))
    with bind_state(A):
        preloaded = skill_commands.build_preloaded_skills_prompt(["private_probe"], task_id="t")
        automatic = skill_commands.build_auto_load_prompt(
            task_id="t", user_config={"skills": {"auto_load": ["private_probe"]}})
        assert skill_commands._load_skill_payload("private_probe") is None
        assert skill_commands.get_skill_commands() == {} and skill_commands.get_interactive_skill_commands() == {}
        assert skill_commands.build_skill_invocation_message("/private-probe", task_id="t") is None
        assert build_skills_system_prompt(skills_dir_override=root) == ""
        viewed, listed = json.loads(skill_view("private_probe")), json.loads(skills_list())
    assert preloaded == ("", [], ["private_probe"]) and automatic == ("", [], [])
    assert not viewed["success"] and "not available to a hosted agent" in viewed["error"]
    assert not listed["success"] and "HOST-SKILL-PRIVATE" not in json.dumps(listed)
    assert not [p for p in read if str(p).startswith(str(root))], read


def test_local_skills_still_preload(monkeypatch):
    from hermes_agent.agent.skill_commands import build_preloaded_skills_prompt
    _plant_skill(monkeypatch)
    prompt, loaded, missing = build_preloaded_skills_prompt(["private_probe"], task_id="t")
    assert "HOST-SKILL-PRIVATE" in prompt and loaded == ["private_probe"] and not missing


def test_a_hosted_write_is_not_judged_by_this_hosts_skill_locks(provider, host_dir):
    """A skill locked on this host's profile changes nothing for a hosted path that names it, through a
    host symlink or literally: the path is the sandbox's, so it is neither resolved on this host nor
    refused by the host skill's name, and the write reaches the sandbox."""
    from hermes_constants import get_hermes_home
    from hermes_agent.tools import skill_usage
    from hermes_agent.tools.skill_lock import skill_refusal
    skill = get_hermes_home() / "skills" / "locked_probe"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: locked_probe\ndescription: x\n---\nbody\n", encoding="utf-8")
    assert skill_usage.set_locked("locked_probe", True)
    (host_dir / "workspace-link").symlink_to(skill, target_is_directory=True)
    paths = [str(host_dir / "workspace-link" / "notes.txt"), str(skill / "notes.txt")]
    with c._watch_host(host_dir) as touched:
        outputs = [out for path in paths for out in c._drive_file_tools(A, host_dir, path)]
        touched = list(touched)
    assert not touched, f"a hosted tool touched the host filesystem: {touched}"
    assert not [o for o in outputs if skill_refusal("locked_probe") in str(o.get("error", ""))], outputs
    writes = [call.command for call in provider.calls if call.op == "execute" and call.sandbox == c._identity(A)
              and ("sandbox copy" in call.stdin or "c2FuZGJveCBjb3B5" in call.stdin)]
    for path in paths:
        assert any(path in command for command in writes), f"write_file never reached the sandbox at {path}"
    assert not (skill / "notes.txt").exists()


# --------------------------------------------------------------------------------------------------
# Shell hooks and other host-side callbacks around a refused call


def _capture_hook(tmp_path, monkeypatch, event="post_tool_call", matcher="memory"):
    from hermes_agent.agent import shell_hooks
    from hermes_cli import plugins
    captured = tmp_path / f"{event}.json"
    script = tmp_path / "capture.py"
    script.write_text("import pathlib,sys\npathlib.Path(sys.argv[1]).write_text(sys.stdin.read())\n")
    manager = plugins.PluginManager()
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    monkeypatch.setattr(plugins, "_delivery_manager", lambda: manager)
    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(script))} {shlex.quote(str(captured))}"
    entry = {"command": command, **({"matcher": matcher} if matcher else {})}
    assert shell_hooks.register_from_config({"hooks": {event: [entry]}}, accept_hooks=True)
    return captured


def _memory_call(tmp_path):
    from hermes_agent.agent import tool_executor as te
    from tests.tools.test_hosted_deny_by_default import _call, _make_agent
    agent, messages = _make_agent(tmp_path), []
    te.execute_tool_calls_sequential(agent, SimpleNamespace(tool_calls=[_call("c1", "memory", {
        "target": "memory", "operations": [{"action": "add", "content": "TENANT-HOOK-PRIVATE"}]})]), messages, "task")
    return messages


def test_shell_hook_cannot_write_refused_tool_payload_to_host(tmp_path, monkeypatch):
    captured = _capture_hook(tmp_path, monkeypatch)
    with bind_state(A):
        messages = _memory_call(tmp_path)
    assert "not available to a hosted agent" in messages[0]["content"], messages
    assert not captured.exists(), f"host hook captured a hosted call: {captured.read_text()}"


def test_shell_hook_never_spawns_in_hosted_scope(tmp_path, monkeypatch):
    """The subprocess site itself refuses, whichever event or caller reaches it."""
    from hermes_agent.agent import shell_hooks
    from hermes_cli.plugins import invoke_hook
    captured = _capture_hook(tmp_path, monkeypatch, event="on_session_start", matcher=None)
    spec = shell_hooks.ShellHookSpec(event="on_session_start", command=f"{shlex.quote(sys.executable)} -c pass")
    with bind_state(A):
        invoke_hook("on_session_start", session_id="s")
        result = shell_hooks.run_once(spec, {"session_id": "s"})
    assert not captured.exists() and result["returncode"] is None and "hosted agent" in result["error"]
    invoke_hook("on_session_start", session_id="s")
    assert captured.exists()  # the same hook still runs locally


def test_middleware_never_sees_a_refused_hosted_call(tmp_path, monkeypatch):
    from hermes_cli import middleware
    seen = []
    monkeypatch.setattr(middleware, "apply_tool_request_middleware", lambda *a, **k: seen.append("request"))
    monkeypatch.setattr(middleware, "run_tool_execution_middleware", lambda *a, **k: seen.append("execution"))
    with bind_state(A):
        messages = _memory_call(tmp_path)
    assert "not available to a hosted agent" in messages[0]["content"] and seen == []


def test_a_fail_closed_host_hook_neither_runs_nor_blocks_a_hosted_tool(tmp_path):
    """A hosted turn is outside the host's hooks: a fail-closed pre_tool_call hook is skipped, not
    turned into a block on the agent's own allowed tool."""
    from hermes_agent.agent import shell_hooks
    marker = tmp_path / "ran"
    spec = shell_hooks.ShellHookSpec(event="pre_tool_call", matcher="terminal", fail_closed=True,
                                     command=f"{shlex.quote(sys.executable)} -c 'open({str(marker)!r}, \"w\")'")
    callback = shell_hooks._make_callback(spec)
    with bind_state(A):
        assert callback(tool_name="terminal", args={"command": "ls"}) is None
    assert not marker.exists()


def test_local_shell_hook_still_runs(tmp_path, monkeypatch):
    captured = _capture_hook(tmp_path, monkeypatch)
    _memory_call(tmp_path)
    assert captured.exists() and "TENANT-HOOK-PRIVATE" in captured.read_text()


# --------------------------------------------------------------------------------------------------
# Background processes: no host checkpoint, no host receipt


def _hosted_background(monkeypatch, command="echo TENANT-PROCESS-PRIVATE"):
    import hermes_agent.tools.process_registry as processes
    from hermes_agent.tools.registry import registry
    r = processes.process_registry
    original = c.RecordingSandboxEnvironment.execute

    def execute(self, command, **kwargs):
        result = original(self, command, **kwargs)
        return {"returncode": 0, "output": "12345\n"} if "nohup bash -lc" in command else result

    monkeypatch.setattr(c.RecordingSandboxEnvironment, "execute", execute)
    release = threading.Event()  # holds the poller until the test delivers the simulated exit
    monkeypatch.setattr(r, "_env_poller_loop", lambda *a: release.wait(10))
    with bind_state(A):
        result = json.loads(registry.dispatch("terminal", {"command": command, "background": True}, task_id="t"))
    assert result.get("session_id"), result
    return r, r._running[result["session_id"]], release


def test_background_process_metadata_and_result_stay_off_host(provider, monkeypatch):
    from hermes_constants import get_hermes_home
    r, session, release = _hosted_background(monkeypatch)
    try:
        checkpoint = get_hermes_home() / "processes.json"
        running_leak = checkpoint.exists() and "TENANT-PROCESS-PRIVATE" in checkpoint.read_text()
        r._write_checkpoint()  # an unscoped writer (a local process's launch) iterates hosted records too
        unscoped_leak = checkpoint.exists() and "TENANT-PROCESS-PRIVATE" in checkpoint.read_text()
        with bind_state(A):
            session.append_output("TENANT-PROCESS-OUTPUT-PRIVATE")
            r._finish_exited(session, 0)
        receipt = get_hermes_home() / "logs" / "process-results" / f"{session.id}.json"
        assert not running_leak and not unscoped_leak and not receipt.exists()
        with bind_state(A):  # live state is still the agent's, in memory
            polled = r.poll(session.id)
        assert polled["status"] == "exited" and "TENANT-PROCESS-OUTPUT-PRIVATE" in polled["output_preview"]
    finally:
        release.set()


def test_local_process_checkpoint_and_receipt_unchanged():
    from hermes_constants import get_hermes_home
    from hermes_agent.tools import process_registry as processes
    from hermes_agent.tools.process_registry_results import save_completed_result
    r = processes.ProcessRegistry()
    session = processes.ProcessSession(id="proc_localcheck", command="echo LOCAL-PROCESS")
    with r._lock:
        r._running[session.id] = session
    r._write_checkpoint()
    assert "LOCAL-PROCESS" in (get_hermes_home() / "processes.json").read_text()
    session.output_buffer = "LOCAL-OUTPUT"
    save_completed_result(session)
    assert "LOCAL-OUTPUT" in (get_hermes_home() / "logs" / "process-results" / "proc_localcheck.json").read_text()


# --------------------------------------------------------------------------------------------------
# Other artifacts an allowed tool keeps about a call


def test_terminal_output_spill_and_blocked_scripts_stay_off_host():
    from hermes_agent.tools.approval_floors import _save_blocked_payload
    from hermes_agent.tools.environments.base_output import _new_output_collector
    with bind_state(A):
        assert _new_output_collector(SimpleNamespace(), True)._spill_path is None
        assert _save_blocked_payload("echo TENANT-BLOCKED") is None
    assert _new_output_collector(SimpleNamespace(), True)._spill_path is not None
    saved = _save_blocked_payload("echo LOCAL-BLOCKED")
    assert saved and "LOCAL-BLOCKED" in Path(saved).read_text()


def test_verification_ledger_records_nothing_for_a_hosted_agent(monkeypatch):
    from hermes_agent.agent import verification_evidence as ve
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "1")
    with bind_state(A):
        assert ve._ledger_enabled() is False
    assert ve._ledger_enabled() is True

