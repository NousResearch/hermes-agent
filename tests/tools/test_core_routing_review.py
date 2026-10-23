"""Read-only review probes: inert CUA transport and task-private real shells only."""
import json
import shlex
import sys
from pathlib import Path

import pytest

from tests.tools.target_selection_fixtures import lease_selection


def api():
    return pytest.importorskip("hermes_cli.session_execution", reason="upstream has no session-execution routing API")


@pytest.fixture
def shell_route(tmp_path, monkeypatch):
    execution = api()
    from tools import terminal_tool as tt
    from tools.terminal_targets import register_terminal_target_resolver
    from tools.terminal_scope import set_terminal_scope, reset_terminal_scope
    home = tmp_path / "guest-home"
    nested = home / "nested"
    nested.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("BASH_ENV", raising=False)
    for name in ("_active_environments", "_last_activity", "_task_env_overrides", "_session_cwd", "_session_cwd_observed", "_container_aliases"):
        monkeypatch.setattr(tt, name, {})
    scope = set_terminal_scope({"TERMINAL_ENV": "local", "TERMINAL_CWD": str(tmp_path)})
    epoch = [0]
    paused = [False]
    def access():
        if paused[0]:
            raise execution.SessionExecutionError("human control held")
        return epoch[0]
    # Like an SSH command transport: child starts at guest HOME independently of host cwd.
    prefix = ("/bin/bash", "-c", 'cd "$1" && shift && exec "$@"', "fixture", str(home))
    execution.register_session_execution_context("review-target", execution.SessionExecutionContext(
        command_prefix=prefix, backend_cwd=str(home), env_set={"HOME": str(home)}, terminal_access_epoch=access))
    lease = execution.resolve_session_execution_context(session_id="review-target")
    dispose = register_terminal_target_resolver("review", lambda **kw: lease, selector=lambda **kw: lease_selection(lease))
    try:
        yield tt, home, nested, lease, epoch, paused
    finally:
        for env in tt._active_environments.values():
            env.cleanup()
        dispose()
        execution.remove_session_execution_context("review-target")
        reset_terminal_scope(scope)


def dispatch(command, **kwargs):
    from tools.registry import registry
    return json.loads(registry.dispatch("terminal", {"command": command, **kwargs}, task_id="review-parent", session_id="review-session"))


@pytest.mark.parametrize("transition", ["none", "takeover", "handback"])
@pytest.mark.parametrize("background", [False, True])
def test_terminal_effect_boundary_keeps_captured_control(shell_route, monkeypatch, transition, background):
    tt, home, nested, lease, epoch, paused = shell_route
    from tools.process_registry import process_registry
    assert dispatch("true", target="review")["exit_code"] == 0
    marker = home / "effect"
    original = tt._resolve_command_cwd
    def scheduled(**kw):
        value = original(**kw)
        if transition != "none":
            epoch[0] += 1
            paused[0] = True
            if transition == "handback":
                epoch[0] += 1
                paused[0] = False
        return value
    monkeypatch.setattr(tt, "_resolve_command_cwd", scheduled)
    result = dispatch("touch " + shlex.quote(str(marker)), target="review", background=background)
    if result.get("session_id"):
        process_registry.wait(result["session_id"], timeout=10)
    assert marker.exists() == (transition == "none"), "user command ran despite control transition after approval and before actual spawn"


@pytest.mark.parametrize("mode", ["foreground", "pipe", "pty", "promoted"])
def test_routed_background_applies_guest_cwd(shell_route, mode):
    tt, home, nested, lease, epoch, paused = shell_route
    from tools.process_registry import process_registry
    assert dispatch("cd nested", target="review")["exit_code"] == 0
    assert tt.get_session_cwd(lease.cache_key) == str(nested)
    args = {}
    if mode in ("pipe", "pty"):
        args = {"background": True, "pty": mode == "pty"}
    elif mode == "promoted":
        args = {"timeout": tt.FOREGROUND_MAX_TIMEOUT + 1}
    result = dispatch("printf 'cwd=%s\\n' \"$PWD\"", target="review", **args)
    assert not result.get("error"), result
    if result.get("session_id"):
        completed = process_registry.wait(result["session_id"], timeout=10)
    else:
        completed = result
    assert completed["exit_code"] == 0, completed
    assert "cwd=" + str(nested) in completed["output"].splitlines(), "background shell started in guest home rather than the reported target_cwd"


@pytest.mark.parametrize("transition", ["none", "takeover", "handback"])
def test_cua_approval_wait_keeps_original_control_epoch(tmp_path, monkeypatch, transition):
    execution = api()
    import tools.computer_use_tool
    from tools.registry import registry
    from tools.computer_use import tool, targets
    from tools.computer_use.cua_backend import CuaDriverBackend
    from tools.computer_use.session_context import check_desktop_call
    from tools.computer_use.cua_backend_parse import _tool_envelope
    epoch, paused, effects = [0], [False], []
    def access():
        if paused[0]:
            raise execution.SessionExecutionError("human control held")
        return epoch[0]
    execution.register_session_execution_context("review-cua", execution.SessionExecutionContext(
        computer_use=execution.ComputerUseLaunchContext(private_daemon=True, desktop_only=True,
        driver_command=sys.executable, desktop_attestor=lambda: ("inert-review",), access_epoch=access)))
    lease = execution.resolve_session_execution_context(session_id="review-cua")
    class InertWire:
        def call_tool(self, name, args, **kw):
            check_desktop_call(lease, name, args)
            if name == "click":
                effects.append(name)
            return _tool_envelope(None, [], {"ok": True}, False)
    def inert_start(self):
        self._session = InertWire()
        self._desktop_target_ready = True
        self._last_app = "screen"
    monkeypatch.setattr(CuaDriverBackend, "start", inert_start)
    monkeypatch.setattr(CuaDriverBackend, "stop", lambda self: None)
    monkeypatch.setattr(tool, "_cua_permission_mode", lambda sid: "standard")
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    prompts = []
    def approve(*args, **kwargs):
        prompts.append(True)
        if transition != "none":
            epoch[0] += 1
            paused[0] = True
            if transition == "handback":
                epoch[0] += 1
                paused[0] = False
        return "once"
    tool.set_approval_callback(approve)
    dispose = targets.register_target_resolver("review", lambda **kw: lease, selector=lambda **kw: lease_selection(lease))
    try:
        tool._get_backend("review-caller")  # previously captured private target is warm
        result = registry.dispatch("computer_use", {"action": "click", "coordinate": [1, 2], "app": "screen", "capture_after": False}, session_id="review-caller", task_id="review-task")
        assert prompts == [True], result
        assert bool(effects) == (transition == "none"), "pending approved click survived takeover plus handback"
    finally:
        dispose()
        execution.remove_session_execution_context("review-cua")
        tool.set_approval_callback(None)
