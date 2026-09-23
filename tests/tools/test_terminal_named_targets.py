"""Named terminal targets use real shells without changing the parent route."""
import json
import shlex

import pytest

from tests.tools.target_selection_fixtures import lease_selection

from tools import terminal_tool as tt
from tools.registry import registry
from tools.terminal_scope import reset_terminal_scope, set_terminal_scope

pytestmark = pytest.mark.linux_only


@pytest.fixture
def isolated_terminal(tmp_path, monkeypatch):
    token = set_terminal_scope({"TERMINAL_ENV": "local", "TERMINAL_CWD": str(tmp_path)})
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("BASH_ENV", raising=False)
    for name in ("_active_environments", "_last_activity", "_task_env_overrides",
                 "_session_cwd", "_session_cwd_observed", "_container_aliases"):
        monkeypatch.setattr(tt, name, {})
    yield tmp_path
    for env in tt._active_environments.values():
        env.cleanup()
    reset_terminal_scope(token)


def dispatch(command, **args):
    return json.loads(registry.dispatch("terminal", {"command": command, **args},
                                        task_id="parent-task", session_id="parent-session"))


def test_unknown_target_is_refused_before_command_runs(isolated_terminal):
    marker = isolated_terminal / "must-not-run"
    result = dispatch(f"touch {shlex.quote(str(marker))}", target="missing")
    assert result.get("error"), result
    assert not marker.exists()
    assert not tt._active_environments


def test_named_target_keeps_parent_cwd_environment_and_cache(isolated_terminal, monkeypatch):
    import importlib.util
    import shutil
    from hermes_cli import session_execution as execution

    assert importlib.util.find_spec("tools.terminal_targets"), "named target registration is missing"
    from tools.terminal_targets import register_terminal_target_resolver

    root = isolated_terminal
    parent = root / "parent"
    target = root / "target"
    parent.mkdir()
    target.mkdir()
    (target / "nested").mkdir()
    monkeypatch.setenv("ROUTING_VALUE", "parent")
    tt.register_task_env_overrides("parent-task", {"cwd": str(parent)})
    before = dispatch("export PARENT_ONLY=parent; printf '%s|%s' \"$ROUTING_VALUE\" \"$PWD\"")
    assert before["output"] == f"parent|{parent}", before
    parent_env = dict(tt._active_environments)
    calls = []
    execution.register_session_execution_context("synthetic-target", execution.SessionExecutionContext(
        command_prefix=(shutil.which("env"),), backend_cwd=str(target),
        env_set={"ROUTING_VALUE": "target"}))
    lease = execution.resolve_session_execution_context(session_id="synthetic-target")

    def resolve(**kw):
        calls.append(kw)
        return lease

    dispose = register_terminal_target_resolver("fixture", resolve, selector=lambda **kw: lease_selection(resolve(**kw)))
    try:
        result = dispatch("cd nested; export TARGET_ONLY=target; printf '%s|%s|%s' \"$ROUTING_VALUE\" \"$PWD\" \"${PARENT_ONLY-unset}\"", target="fixture")
        assert result["exit_code"] == 0, result
        assert result["output"] == f"target|{target / 'nested'}|unset"
        assert result["target_cwd"] == str(target / "nested")
        assert result["execution_target"] == "fixture"
        assert "cwd" not in result
        again = dispatch("printf '%s|%s' \"$TARGET_ONLY\" \"$PWD\"", target="fixture")
        assert again["output"] == f"target|{target / 'nested'}", again
        after = dispatch("printf '%s|%s|%s|%s' \"$ROUTING_VALUE\" \"$PWD\" \"$PARENT_ONLY\" \"${TARGET_ONLY-unset}\"")
        assert after["output"] == f"parent|{parent}|parent|unset", after
        assert tt.get_session_cwd("parent-task") == str(parent)
        assert all(tt._active_environments[key] is env for key, env in parent_env.items())
        assert execution.resolve_session_execution_context(task_id="parent-task", session_id="parent-session") is None
        assert all(c["session_id"] == "parent-session" and c["task_id"] == "parent-task" for c in calls)
        assert len(calls) == 2
    finally:
        dispose()
        execution.remove_session_execution_context("synthetic-target")


@pytest.fixture
def target_route(isolated_terminal):
    import shutil
    from hermes_cli import session_execution as execution
    from tools.terminal_targets import register_terminal_target_resolver

    target = isolated_terminal / "target"
    target.mkdir()
    (target / "nested").mkdir()
    execution.register_session_execution_context("synthetic-target", execution.SessionExecutionContext(
        # A transport enters guest HOME independently of its host launcher cwd.
        command_prefix=("/bin/bash", "-c", 'cd "$1" && shift && exec "$@"', "guest", str(target)),
        backend_cwd=str(target), env_set={"ROUTING_VALUE": "target"}))
    lease = execution.resolve_session_execution_context(session_id="synthetic-target")
    dispose = register_terminal_target_resolver("fixture", lambda **kw: lease, selector=lambda **kw: lease_selection(lease))
    try:
        yield target, lease
    finally:
        dispose()
        execution.remove_session_execution_context("synthetic-target")


@pytest.mark.parametrize("mode", ["pipe", "pty", "promoted"])
@pytest.mark.parametrize("explicit", [False, True])
def test_background_target_cwd_and_process_ownership(target_route, mode, explicit, monkeypatch):
    from tools.process_registry import process_registry
    target, lease = target_route
    tt.record_session_cwd("parent-task", str(target.parent), observed=True)
    assert dispatch("cd nested", target="fixture")["exit_code"] == 0
    args = {"background": True, "pty": mode == "pty"}
    if mode == "promoted":
        args = {"timeout": tt.FOREGROUND_MAX_TIMEOUT + 1}
    quoted = target / "space and ' quote"
    quoted.mkdir()
    assert dispatch("cd " + shlex.quote(str(quoted)), target="fixture")["exit_code"] == 0
    if explicit:
        args["workdir"] = str(target / "nested")
    expected = target / "nested" if explicit else quoted
    # Guest paths need not pass host isdir checks. Child bash sees real paths.
    import os
    isdir = os.path.isdir
    monkeypatch.setattr(os.path, "isdir", lambda p: False if str(p) == str(expected) else isdir(p))
    result = dispatch("printf '%s|%s' \"$ROUTING_VALUE\" \"$PWD\"", target="fixture", **args)
    assert not result.get("error"), result
    completed = process_registry.wait(result["session_id"], timeout=15)
    assert completed["exit_code"] == 0, completed
    # Interactive bash can emit a job-control warning; assert the shell's
    # actual routing record, not the absence of unrelated startup diagnostics.
    assert f"target|{expected}" in completed["output"].splitlines(), completed
    assert result["target_cwd"] == str(expected)
    assert tt.get_session_cwd(lease.cache_key) == str(quoted)
    proc = process_registry.get(result["session_id"])
    assert proc.cwd == str(expected)
    assert proc.owner_task_id == "parent-task"
    assert proc.session_key == "parent-task"
    assert proc.task_id == lease.cache_key
    assert result["execution_target"] == "fixture"
    assert "cwd" not in result
    assert tt.get_session_cwd("parent-task") == str(target.parent)


def test_ambiguous_target_execution_is_never_replayed(target_route, monkeypatch):
    from tools.environments.local import LocalEnvironment
    target, _ = target_route
    assert dispatch("true", target="fixture")["exit_code"] == 0
    execute = LocalEnvironment.execute
    marker = target / "executions"

    def lost_reply(self, command, **kw):
        result = execute(self, command, **kw)
        if self.execution_context is not None:
            raise RuntimeError("transport lost after command executed")
        return result

    monkeypatch.setattr(LocalEnvironment, "execute", lost_reply)
    monkeypatch.setattr(tt.time, "sleep", lambda _: None)
    result = dispatch(f"printf x >> {shlex.quote(str(marker))}", target="fixture")
    assert result.get("error"), result
    assert marker.read_text() == "x", "an ambiguous shell command must not be replayed"
    assert dispatch("printf parent-alive")["output"] == "parent-alive"


def test_target_startup_failure_does_not_evict_parent(target_route, monkeypatch):
    target, _ = target_route
    assert dispatch("export PARENT_STATE=retained")["exit_code"] == 0
    before = dict(tt._active_environments)
    create = tt._create_configured_env

    def unavailable(*args, **kw):
        if kw.get("local_config", {}).get("execution_context") is not None:
            raise tt.EnvironmentConnectionError("target transport unavailable")
        return create(*args, **kw)

    monkeypatch.setattr(tt, "_create_configured_env", unavailable)
    result = dispatch("touch must-not-run", target="fixture")
    assert result.get("error"), result
    assert all(tt._active_environments.get(key) is env for key, env in before.items())
    assert dispatch("printf '%s' \"$PARENT_STATE\"")["output"] == "retained"
    assert not (target / "must-not-run").exists()


def test_named_target_is_an_optional_parameter_on_canonical_terminal():
    schema = tt.TERMINAL_SCHEMA
    assert schema["name"] == "terminal"
    assert schema["parameters"]["properties"].get("target", {}).get("type") == "string"
    assert "target" not in schema["parameters"]["required"]


@pytest.mark.parametrize("name", [None, "", " fixture", "fixture ", "fixture\0", 23, {}, "host"])
def test_invalid_or_unknown_name_never_executes(target_route, name):
    target, _ = target_route
    marker = target / "must-not-run"
    result = dispatch(f"touch {shlex.quote(str(marker))}", target=name, _host_local=True)
    assert result.get("error"), result
    assert not marker.exists()
    assert not tt._active_environments


@pytest.mark.parametrize("failure", ["missing", "foreign-owner", "foreign-profile", "replaced", "invalid", "exception"])
@pytest.mark.parametrize("background", [False, True])
def test_provider_refusal_has_no_execution_fallback(target_route, failure, background):
    from dataclasses import replace
    from hermes_cli import session_execution as execution
    from tools.terminal_targets import register_terminal_target_resolver

    target, lease = target_route
    marker = target / "must-not-run"
    calls = []

    def resolve(command, session_id, task_id):
        calls.append((command, session_id, task_id))
        if failure == "exception":
            raise TypeError("provider failed after admission work; do not retry")
        if failure == "foreign-owner":
            raise execution.SessionExecutionError("foreign target owner")
        if failure == "foreign-profile":
            return replace(lease, home=str(target / "foreign-profile"))
        if failure == "missing":
            return None
        return lease

    if failure in {"replaced", "invalid"}:
        execution.register_session_execution_context("synthetic-target", execution.SessionExecutionContext())
        if failure == "invalid":
            valid = [True]
            execution.register_session_execution_context("synthetic-target", execution.SessionExecutionContext(validate=lambda: valid[0]))
            lease = execution.resolve_session_execution_context(session_id="synthetic-target")
            valid[0] = False
    dispose = register_terminal_target_resolver("fixture", resolve, selector=lambda **kw: lease_selection(resolve(**kw)))
    command = f"touch {shlex.quote(str(marker))}"
    try:
        result = dispatch(command, target="fixture", background=background)
        assert result.get("error"), result
        assert calls == [(command, "parent-session", "parent-task")]
        assert not marker.exists()
        assert not tt._active_environments
        assert dispatch("printf parent-alive")["output"] == "parent-alive"
    finally:
        dispose()


def test_target_guard_denial_keeps_caller_approval_identity(target_route, monkeypatch):
    from tools.approval_context import get_current_session_key, set_current_session_key, reset_current_session_key
    target, _ = target_route
    marker = target / "must-not-run"
    calls = []
    preflight = tt._pre_exec_block

    def pre_exec(command, **kw):
        calls.append(("preflight", kw["session_key"]))
        return preflight(command, **kw)

    def deny(command, env_type, **kw):
        calls.append(("approval", get_current_session_key(), command, env_type))
        return {"approved": False, "description": "fixture user denied"}

    monkeypatch.setattr(tt, "_pre_exec_block", pre_exec)
    monkeypatch.setattr(tt, "_check_all_guards", deny)
    token = set_current_session_key("caller-approval-key")
    command = f"touch {shlex.quote(str(marker))}"
    try:
        result = dispatch(command, target="fixture", force=True, _host_local=True)
        assert result["status"] == "blocked", result
        # Denial precedes target/shell realization. Resource-dependent preflight
        # still runs before dispatch, but is unnecessary when consent is denied.
        assert calls == [("approval", "caller-approval-key", command, "local")]
        assert not marker.exists()
        assert not tt._active_environments
    finally:
        reset_current_session_key(token)


def test_target_transport_does_not_change_parent_backend_or_bypass_refusal(target_route):
    from tools.terminal_scope import TerminalPolicyRefusal
    from tools.terminal_targets import register_terminal_target_resolver
    target, lease = target_route
    calls = []
    dispose = register_terminal_target_resolver("fixture", lambda **kw: calls.append(kw) or lease, selector=lambda **kw: lease_selection(calls.append(kw) or lease))
    token = set_terminal_scope({"TERMINAL_ENV": "ssh", "TERMINAL_CWD": "/remote-parent"})
    try:
        result = dispatch("printf '%s|%s' \"$ROUTING_VALUE\" \"$PWD\"", target="fixture")
        assert result["output"] == f"target|{target}", result
        assert tt._get_env_config()["env_type"] == "ssh"
        assert tt._get_env_config()["cwd"] == "/remote-parent"
        ordinary = tt._plan_execution("true", task_id="parent-task", session_id="parent-session",
                                       background=False, timeout=10, _host_local=False)
        assert ordinary.env_type == "ssh" and ordinary.execution_context is None
        refusal = set_terminal_scope(TerminalPolicyRefusal("fixture policy unreadable"))
        try:
            result = dispatch("touch must-not-run", target="fixture")
            assert "policy unavailable" in result["error"], result
            assert len(calls) == 1
            assert not (target / "must-not-run").exists()
        finally:
            reset_terminal_scope(refusal)
    finally:
        reset_terminal_scope(token)
        dispose()


def test_resolver_registration_is_profile_scoped_and_disposal_is_generation_safe(target_route):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from tools.terminal_targets import register_terminal_target_resolver
    target, lease = target_route
    calls = []
    old = register_terminal_target_resolver("fixture", lambda **kw: calls.append("old") or lease, selector=lambda **kw: lease_selection(calls.append("old") or lease))
    current = register_terminal_target_resolver("fixture", lambda **kw: calls.append("current") or lease, selector=lambda **kw: lease_selection(calls.append("current") or lease))
    try:
        old()
        old()
        assert calls == []
        assert dispatch("printf current", target="fixture")["output"] == "current"
        token = set_hermes_home_override(target / "other-profile")
        try:
            result = dispatch("printf forbidden", target="fixture")
            assert result.get("error"), result
            assert calls == ["current"]
        finally:
            reset_hermes_home_override(token)
        current()
        assert dispatch("printf forbidden", target="fixture").get("error")
        assert calls == ["current"]
    finally:
        current()


def test_yielded_target_retains_parent_notification_ownership(target_route):
    import threading
    from tools import interrupt
    from tools.process_registry import process_registry
    target, lease = target_route
    assert dispatch("cd nested", target="fixture")["exit_code"] == 0
    release = target / "release"
    command = f"printf '%s|%s' \"$ROUTING_VALUE\" \"$PWD\"; while [ ! -e {shlex.quote(str(release))} ]; do sleep 0.05; done; printf done"
    proc_id = None
    try:
        interrupt.request_yield(threading.current_thread().ident)
        result = dispatch(command, target="fixture", timeout=15)
        proc_id = result.get("session_id")
        assert result.get("status") == "yielded_to_background", result
        assert result["execution_target"] == "fixture" and "cwd" not in result
        proc = process_registry.get(proc_id)
        assert proc.owner_task_id == "parent-task" and proc.session_key == "parent-task"
        assert proc.task_id == lease.cache_key
        release.touch()
        completed = process_registry.wait(proc_id, timeout=15)
        assert completed["exit_code"] == 0, completed
        assert f"target|{target / 'nested'}" in completed["output"]
        assert "done" in completed["output"]
    finally:
        interrupt.consume_yield(threading.current_thread().ident)
        release.touch()
        if proc_id is not None:
            process_registry.kill_process(proc_id)


def test_routed_cwd_observation_does_not_require_host_path_visibility(target_route, monkeypatch):
    """A real shell reports cwd; the transport host cannot stat the guest path."""
    import os.path
    target, _ = target_route
    assert dispatch("true", target="fixture")["exit_code"] == 0
    isdir = os.path.isdir
    guest_cwd = str(target / "nested")
    monkeypatch.setattr(os.path, "isdir", lambda path: False if str(path) == guest_cwd else isdir(path))
    result = dispatch("cd nested; printf '%s' \"$PWD\"", target="fixture")
    assert result["output"] == guest_cwd, result
    assert result["target_cwd"] == guest_cwd, result
    assert dispatch("printf '%s' \"$PWD\"", target="fixture")["output"] == guest_cwd


@pytest.mark.parametrize("mode", ["foreground", "pipe", "pty"])
def test_target_shell_init_restores_guest_values_after_host_scrubbing(target_route, monkeypatch, mode):
    from dataclasses import replace
    from hermes_cli import session_execution as execution
    from tools.terminal_targets import register_terminal_target_resolver
    from tools.process_registry import process_registry
    import subprocess

    target, lease = target_route
    monkeypatch.setenv("ROUTING_VALUE", "host")
    context = replace(lease.context, env_set={}, env_unset={"ROUTING_VALUE"},
                      backend_shell_init="export ROUTING_VALUE=guest")
    execution.register_session_execution_context("synthetic-target", context)
    lease = execution.resolve_session_execution_context(session_id="synthetic-target")
    dispose = register_terminal_target_resolver("fixture", lambda **kw: lease, selector=lambda **kw: lease_selection(lease))
    popen = subprocess.Popen
    launch_values = []

    def observe(*args, **kw):
        launch_values.append(kw.get("env", {}).get("ROUTING_VALUE"))
        return popen(*args, **kw)

    monkeypatch.setattr(subprocess, "Popen", observe)
    try:
        result = dispatch("printf '%s' \"$ROUTING_VALUE\"", target="fixture",
                          background=mode != "foreground", pty=mode == "pty")
        if mode != "foreground" and result.get("session_id"):
            result = process_registry.wait(result["session_id"], timeout=15)
        assert result["exit_code"] == 0, result
        assert result["output"].strip() == "guest", result
        assert launch_values and all(value != "guest" for value in launch_values)
        assert dispatch("printf '%s' \"$ROUTING_VALUE\"")["output"] == "host"
    finally:
        dispose()


def test_session_only_target_background_is_owned_by_caller(target_route):
    from tools.process_registry import process_registry
    result = json.loads(registry.dispatch("terminal", {
        "command": "printf target", "target": "fixture", "background": True,
    }, session_id="parent-session"))
    assert not result.get("error"), result
    completed = process_registry.wait(result["session_id"], timeout=15)
    assert completed["exit_code"] == 0, completed
    proc = process_registry.get(result["session_id"])
    assert proc.owner_task_id == "parent-session"
    assert proc.session_key == "parent-session"


@pytest.mark.parametrize("mode", ["foreground", "pipe", "pty", "promoted", "fallback"])
@pytest.mark.parametrize("transition", ["none", "takeover", "handback"])
def test_target_authority_reaches_actual_launch(isolated_terminal, monkeypatch, mode, transition):
    from hermes_cli import session_execution as execution
    from tools.terminal_targets import register_terminal_target_resolver
    from tools.process_registry import process_registry
    from tools.environments.local import LocalEnvironment

    epoch, held = [0], [False]
    def access():
        if held[0]:
            raise execution.SessionExecutionError("human control held")
        return epoch[0]
    execution.register_session_execution_context("authority-target", execution.SessionExecutionContext(
        command_prefix=("/usr/bin/env",), backend_cwd=str(isolated_terminal),
        terminal_access_epoch=access))
    lease = execution.resolve_session_execution_context(session_id="authority-target")
    dispose = register_terminal_target_resolver("authority", lambda **kw: lease, selector=lambda **kw: lease_selection(lease))
    def change():
        if transition != "none":
            epoch[0] += 1
            held[0] = True
            if transition == "handback":
                epoch[0] += 1
                held[0] = False
    marker = isolated_terminal / "effect"
    try:
        assert dispatch("true", target="authority")["exit_code"] == 0
        cached = dict(tt._active_environments)
        with monkeypatch.context() as late:
            if mode == "foreground":
                recover = LocalEnvironment._recover_cwd
                def scheduled(self):
                    recover(self)
                    change()
                late.setattr(LocalEnvironment, "_recover_cwd", scheduled)
            else:
                spawn_env = process_registry._spawn_env
                def scheduled(*args, **kw):
                    result = spawn_env(*args, **kw)
                    change()
                    return result
                late.setattr(process_registry, "_spawn_env", scheduled)
                if mode == "fallback":
                    def unavailable(*args, **kw):
                        change()
                        raise ImportError("inert PTY failure")
                    late.setattr(process_registry, "_spawn_local_pty", unavailable)
            args = {} if mode == "foreground" else {"background": True, "pty": mode in ("pty", "fallback")}
            if mode == "promoted":
                args = {"timeout": tt.FOREGROUND_MAX_TIMEOUT + 1}
            result = dispatch("touch " + shlex.quote(str(marker)), target="authority", **args)
            if result.get("session_id"):
                process_registry.wait(result["session_id"], timeout=10)
        assert marker.exists() == (transition == "none"), result
        lease.check()  # Revocation belongs to the operation, not the resource.
        held[0] = False
        assert dispatch("printf fresh", target="authority")["output"] == "fresh"
        assert all(tt._active_environments[k] is v for k, v in cached.items())
        assert dispatch("printf parent")["output"] == "parent"
    finally:
        dispose()
        execution.remove_session_execution_context("authority-target")


@pytest.mark.parametrize("transition", ["none", "denied", "handback", "provider"])
def test_target_selection_precedes_approval_not_provisioning(target_route, monkeypatch, transition):
    from hermes_cli import session_execution as execution
    from tools import terminal_targets
    target, lease = target_route
    realized = []
    epoch = [0]
    def select(**kw):
        captured = epoch[0]
        def check():
            if captured != epoch[0]:
                raise execution.SessionExecutionError("selected authority changed")
        def realize(*, before_start=None):
            if before_start is not None:
                before_start()
            realized.append(True)
            return lease
        return execution.TargetSelection(realize, check)
    dispose = terminal_targets.register_terminal_target_resolver(
        "fixture", lambda **kw: pytest.fail("allocating resolver called"), selector=select)
    def approval(*args, **kw):
        assert realized == [] and not tt._active_environments
        if transition == "handback":
            epoch[0] += 2
        if transition == "provider":
            terminal_targets.register_terminal_target_resolver("fixture", lambda **kw: lease, selector=select)
        return {"approved": transition != "denied", "description": "private approval"}
    monkeypatch.setattr(tt, "_check_all_guards", approval)
    marker = target / "effect"
    try:
        result = dispatch("touch " + shlex.quote(str(marker)), target="fixture")
        assert marker.exists() == (transition == "none"), result
        assert realized == ([True] if transition == "none" else [])
    finally:
        dispose()
        terminal_targets.register_terminal_target_resolver("fixture", lambda **kw: lease, selector=lambda **kw: lease_selection(lease))()
