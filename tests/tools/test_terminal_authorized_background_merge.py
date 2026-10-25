"""Behavior regressions for authorized terminal background dispatch.

Native: terminal_tool, planning/conversion, pre-exec orchestration, approval
orchestration, profile plugin discovery, trusted context, authorized middleware,
and promotion/result/error handling. Doubled: environment acquisition/execution,
guard probes/verdicts, deadline scheduling, background spawn/result construction,
and metering. The spawn's conversion text is an explicit fixture sentinel, not
evidence of native process-registry or background-result behavior.
"""
import json
import os
import socket
import subprocess
from dataclasses import dataclass
from types import SimpleNamespace

import pytest


_PLUGIN = '''
from hermes_cli.authorized_tool_execution import hold_foreground_execution
EVENTS = []
MODE = "allow"

def execute(next_call, args, **context):
    EVENTS.append(("admission", dict(args), dict(context)))
    if not context["lineage_valid"]:
        raise RuntimeError("fixture requires trusted lineage")
    if MODE == "foreground_only":
        if args.get("background"):
            raise RuntimeError("fixture policy requires foreground execution")
        with hold_foreground_execution():
            return next_call()
    result = next_call()
    EVENTS.append(("admission_result", result))
    if MODE == "cleanup_error":
        raise RuntimeError("fixture cleanup failed after completed dispatch")
    return result

def register(ctx):
    ctx.register_middleware("authorized_tool_execution", execute)
'''
_AUTO_NOTE = "effect double: trailing & was converted to background=true"
_COMMAND = "echo admission-ready"
_ROUTES = ("explicit", "trailing_amp", "timeout")


@dataclass
class _Agent:
    session_id: str = "pr135418-test-root"
    _delegate_depth: int = 0


@pytest.fixture
def case(tmp_path, monkeypatch):
    # These refusals precede native imports. PermissionError permits native
    # optional version probes to use their existing OSError fallback.
    def refuse_effect(*args, **kwargs):
        raise PermissionError("process/network effects are disabled in this fixture")

    monkeypatch.setattr(subprocess, "Popen", refuse_effect)
    for name in ("system", "fork", "forkpty", "posix_spawn", "posix_spawnp"):
        if hasattr(os, name):
            monkeypatch.setattr(os, name, refuse_effect)
    for name in ("connect", "connect_ex", "bind", "listen", "sendto"):
        monkeypatch.setattr(socket.socket, name, refuse_effect)
    monkeypatch.setattr(socket, "create_connection", refuse_effect)
    monkeypatch.setattr(socket, "getaddrinfo", refuse_effect)

    home = tmp_path / "hermes-home"
    plugin_dir = home / "plugins" / "merge_admission"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        "name: merge_admission\nversion: 0.1.0\n", encoding="utf-8"
    )
    (plugin_dir / "__init__.py").write_text(_PLUGIN, encoding="utf-8")
    (home / "config.yaml").write_text(
        "plugins:\n  enabled: [merge_admission]\n", encoding="utf-8"
    )
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_MAX_FOREGROUND_TIMEOUT", "600")

    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(home)
    plugins = None
    interrupt = None
    try:
        from agent import deadline
        from agent.tool_execution_context import bind_tool_execution_context
        from hermes_cli import plugins
        from hermes_cli.observability import shared_metrics_loop
        from tools import approval, interrupt, terminal_tool as terminal

        plugins._reset_plugin_managers_for_tests()
        interrupt.clear_current_thread_interrupt()
        manager = plugins._delivery_manager()
        module = manager._plugins["merge_admission"].module
        state = SimpleNamespace(
            terminal=terminal, bind=bind_tool_execution_context, module=module,
            home=home, workdir=str(workdir), task="pr135418-" + tmp_path.name,
            events=[], plans=[], spawn_calls=[], spawn_results=[], meters=[],
            approved=True, block_preexec=False,
        )
        module.EVENTS = state.events

        def unexpected_execute(*args, **kwargs):
            pytest.fail("a background scenario reached foreground execution")

        env = SimpleNamespace(host_cwd=str(tmp_path / "mount"), execute=unexpected_execute)
        state.env = env
        config = {
            "env_type": "local", "timeout": 30, "cwd": str(workdir),
            "host_cwd": None, "modal_mode": "auto", "docker_image": "",
            "singularity_image": "", "modal_image": "", "daytona_image": "",
        }

        def acquire(plan, task_id):
            state.plans.append(plan)
            state.events.append(("environment", task_id))
            return env

        def bounded(call, timeout, *, label, **kwargs):
            state.events.append(("bounded_guard", label, timeout))
            value = call()
            return SimpleNamespace(timed_out=False, value=value)

        def lifecycle_probe(**kwargs):
            state.events.append(("lifecycle_guard", kwargs["command"]))
            if state.block_preexec:
                return terminal._error_json("fixture pre-exec refusal", status="blocked")
            return None

        def self_repo_probe(**kwargs):
            state.events.append(("self_repo_guard", kwargs["command"]))

        def approval_probe(command, env_type, **kwargs):
            state.events.append(("approval", command))
            return {
                "approved": state.approved, "user_approved": state.approved,
                "description": "fixture command",
            }

        def spawn(**kwargs):
            state.events.append(("spawn", kwargs["command"]))
            state.spawn_calls.append(dict(kwargs))
            payload = {
                "output": "Background process started", "error": None,
                "session_id": "proc_effect_double",
                "notify_on_complete": kwargs["notify_on_complete"],
            }
            if kwargs["auto_backgrounded"]:
                payload["auto_backgrounded"] = _AUTO_NOTE
            result = json.dumps(payload)
            state.spawn_results.append(result)
            return result

        def meter(plan, result, **kwargs):
            state.meters.append((plan, result, kwargs))
            return result

        monkeypatch.setattr(terminal, "FOREGROUND_MAX_TIMEOUT", 600)
        monkeypatch.setattr(terminal, "_get_env_config", lambda: dict(config))
        monkeypatch.setattr(terminal, "_acquire_env", acquire)
        monkeypatch.setattr(deadline, "run_bounded_sync", bounded)
        monkeypatch.setattr(terminal, "gateway_lifecycle_block", lifecycle_probe)
        monkeypatch.setattr(terminal, "self_repo_block", self_repo_probe)
        monkeypatch.setattr(terminal, "_check_all_guards", approval_probe)
        monkeypatch.setattr(approval, "get_current_session_key", lambda default="": state.task)
        monkeypatch.setattr(terminal, "spawn_background_process", spawn)
        monkeypatch.setattr(shared_metrics_loop, "record_terminal_backend", meter)
        yield state
    finally:
        try:
            if interrupt is not None:
                interrupt.clear_current_thread_interrupt()
            if plugins is not None:
                plugins._reset_plugin_managers_for_tests()
        finally:
            reset_hermes_home_override(token)


def _request(case, route):
    explicit = route == "explicit"
    return {
        "command": _COMMAND + (" &" if route == "trailing_amp" else ""),
        "background": explicit,
        "timeout": case.terminal.FOREGROUND_MAX_TIMEOUT + 1 if route == "timeout" else 30,
        "task_id": case.task, "session_id": _Agent().session_id,
        "workdir": case.workdir, "pty": explicit,
        "notify_on_complete": False,
        # Direct native callers can supply this; timeout promotion must drop it.
        "watch_patterns": ["READY"],
        "_completion_output_chars": 173,
        "heartbeat": 75 if explicit else 0,
        "persist_on_release": explicit,
    }


def _invoke(case, request):
    with case.bind(_Agent(), task_id=case.task, tool_call_id="merge-regression"):
        return json.loads(case.terminal.terminal_tool(**request))


def _admissions(case):
    return [event for event in case.events if event[0] == "admission"]


@pytest.mark.parametrize("route", _ROUTES)
def test_admission_preserves_background_keywords_and_result(case, route):
    request = _request(case, route)
    result = _invoke(case, request)
    assert result.get("error") is None, result
    assert [event[0] for event in case.events] == [
        "environment", "bounded_guard", "lifecycle_guard", "self_repo_guard",
        "approval", "admission", "spawn", "admission_result",
    ]
    assert [event[1] for event in case.events if event[0] in (
        "lifecycle_guard", "self_repo_guard", "approval", "spawn"
    )] == [_COMMAND] * 4
    _, args, context = _admissions(case)[0]
    assert args == {
        "command": _COMMAND, "background": True, "workdir": case.workdir,
        "timeout": request["timeout"], "pty": request["pty"],
    }
    assert context["tool_name"] == "terminal" and context["env_type"] == "local"
    assert context["lineage_valid"] is True
    assert context["root_session_id"] == _Agent().session_id
    assert context["profile_home"] == str(case.home)
    assert context["task_id"] == case.task
    assert len(case.spawn_calls) == len(case.spawn_results) == 1
    plan = case.plans[0]
    assert case.spawn_calls[0] == {
        "command": _COMMAND, "env": case.env, "env_type": "local",
        "effective_task_id": plan.effective_task_id, "task_id": case.task,
        "session_key": case.task, "workdir": case.workdir, "cwd": case.workdir,
        "mounted_host": case.env.host_cwd, "effective_pty": request["pty"],
        "notify_on_complete": route == "timeout",
        "watch_patterns": None if route == "timeout" else ["READY"],
        "approval_note": "Command required approval (fixture command) and was approved by the user.",
        "pty_disabled_reason": None, "completion_output_chars": 173,
        "heartbeat_seconds": request["heartbeat"],
        "persist_on_release": request["persist_on_release"],
        "auto_backgrounded": route == "trailing_amp",
    }
    raw = case.spawn_results[0]
    assert case.events[-1] == ("admission_result", raw)
    if route == "timeout":
        assert result["notify_on_complete"] is True
        assert str(request["timeout"]) in result["promoted_from_foreground"]
        assert str(case.terminal.FOREGROUND_MAX_TIMEOUT) in result["promoted_from_foreground"]
        assert "Do NOT re-run it" in result.pop("promoted_from_foreground")
    assert result == json.loads(raw)
    assert result.get("auto_backgrounded") == (_AUTO_NOTE if route == "trailing_amp" else None)
    assert "hint" not in result  # The doubled result has no re-launch advice to inject.
    assert len(case.meters) == 1 and case.meters[0][0] is plan


@pytest.mark.parametrize("route", _ROUTES)
def test_foreground_only_admission_rejects_detached_work_without_spawn(case, route):
    case.module.MODE = "foreground_only"
    result = _invoke(case, _request(case, route))
    assert "fixture policy requires foreground execution" in result["error"]
    assert len(_admissions(case)) == 1
    assert _admissions(case)[0][1]["background"] is True
    assert _admissions(case)[0][1]["command"] == _COMMAND
    assert case.spawn_calls == case.spawn_results == []
    assert "auto_backgrounded" not in result and "promoted_from_foreground" not in result
    assert len(case.meters) == 1 and case.meters[0][2]["error_class"] == "exception"


@pytest.mark.parametrize("guard", ("preexec", "approval"))
def test_guard_refusal_prevents_background_admission_and_spawn(case, guard):
    case.block_preexec = guard == "preexec"
    case.approved = guard != "approval"
    result = _invoke(case, _request(case, "trailing_amp"))
    assert result.get("error") and result["status"] == "blocked"
    assert _admissions(case) == []
    assert case.spawn_calls == case.spawn_results == []
    assert ("lifecycle_guard", _COMMAND) in case.events
    assert (("approval", _COMMAND) in case.events) is (guard == "approval")


def test_completed_auto_background_result_survives_middleware_cleanup_error(case):
    case.module.MODE = "cleanup_error"
    result = _invoke(case, _request(case, "trailing_amp"))
    assert result.get("error") is None, result
    assert len(_admissions(case)) == 1
    assert len(case.spawn_calls) == len(case.spawn_results) == 1
    assert result == json.loads(case.spawn_results[0])
    assert result["auto_backgrounded"] == _AUTO_NOTE
    assert len(case.meters) == 1 and case.meters[0][1] == case.spawn_results[0]
