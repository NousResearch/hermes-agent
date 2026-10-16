"""CUA-only routing: real registries, caller policy, and control fences."""
import importlib
import json
from contextlib import contextmanager

import pytest

from tests.tools.target_selection_fixtures import lease_selection

from hermes_cli import session_execution as execution
from hermes_constants import get_hermes_home, set_hermes_home_override, reset_hermes_home_override
from tools.computer_use import tool


@contextmanager
def hermes_home_scope(home):
    token = set_hermes_home_override(home)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


class RecordingBackend(tool._NoopBackend):
    def __init__(self, permission_mode, *, execution_context=None):
        super().__init__()
        self.execution_context = execution_context
        self.permission_mode = permission_mode
        self.stopped = False

    def stop(self):
        self.stopped = True


def test_selected_target_wins_with_caller_owned_cache_policy_and_retirement(monkeypatch):
    from tools import approval
    from tools.computer_use import targets
    from tools.computer_use import cua_backend_driver
    seen_modes = []
    monkeypatch.setattr(tool, "_new_backend", RecordingBackend)
    monkeypatch.setattr(tool, "_cua_permission_mode", lambda sid: seen_modes.append(sid) or "bounded")
    monkeypatch.setattr(cua_backend_driver, "cua_driver_binary_available", lambda: False)
    assert not tool.check_computer_use_requirements()
    execution.register_session_execution_context("parent", execution.SessionExecutionContext())
    parent = execution.resolve_session_execution_context(session_id="parent")
    leases = []
    for name in ("target-1", "target-2"):
        execution.register_session_execution_context(name, execution.SessionExecutionContext())
        leases.append(execution.resolve_session_execution_context(session_id=name))
    selected = [leases[0]]
    calls = []
    def resolve(session_id, task_id=None):
        calls.append((session_id, task_id))
        return selected[0]
    dispose = targets.register_target_resolver("fixture", resolve, selector=lambda **kw: lease_selection(resolve(**kw), current=lambda: selected[0]))
    try:
        assert tool.check_computer_use_requirements()
        assert calls == []  # availability is not provisioning
        a = tool._get_backend("parent", task_id="actual-task")
        assert a.execution_context is leases[0]
        assert calls == [("parent", "actual-task")]
        assert execution.resolve_session_execution_context(session_id="parent") is parent
        b = tool._get_backend("other")
        assert b is not a
        selected[0] = leases[1]
        c = tool._get_backend("parent")
        assert c is not a
        selected[0] = leases[0]
        assert tool._get_backend("parent") is a
        assert not a.stopped
        approval.approve_session("parent", "cua:click:background")
        tool.release_computer_use_execution_context(leases[1])
        assert c.stopped and not a.stopped and not b.stopped
        assert approval.is_approved("parent", "cua:click:background")
        assert tool.release_computer_use_session("parent")
        assert a.stopped and not b.stopped
        # Backend retirement no longer owns grants; the shared session lifecycle does.
        assert approval.is_approved("parent", "cua:click:background")
        approval.clear_session("parent")
        assert not approval.is_approved("parent", "cua:click:background")
        assert set(seen_modes) == {"parent", "other"}
        selected[0] = None
        result = json.loads(tool.handle_computer_use({"action": "capture"}, session_id="parent"))
        assert result["code"] == "policy_denied"
    finally:
        dispose()
        for name in ("parent", "target-1", "target-2"):
            execution.remove_session_execution_context(name)
        approval.clear_session("parent")


@pytest.mark.parametrize("capture_after", [False, True])
def test_takeover_handback_fences_capture_before_persistence(monkeypatch, capture_after, grant_computer_use_approvals):
    from tools.computer_use import targets
    from tools.computer_use.backend import CaptureResult
    epoch = [0]
    paused = [False]
    def access():
        if paused[0]:
            raise execution.SessionExecutionError("human control paused")
        return epoch[0]
    launch = execution.ComputerUseLaunchContext(private_daemon=True, access_epoch=access)
    execution.register_session_execution_context("target", execution.SessionExecutionContext(computer_use=launch))
    lease = execution.resolve_session_execution_context(session_id="target")
    class CaptureBackend(RecordingBackend):
        def capture(self, **kw):
            # Full takeover + handback: checking only the final boolean would leak.
            epoch[0] += 2
            return CaptureResult(mode="vision", width=20, height=20, png_b64="cHJpdmF0ZQ==")
    monkeypatch.setattr(tool, "_new_backend", CaptureBackend)
    shaped = []
    monkeypatch.setattr(tool, "_capture_response", lambda cap: shaped.append(cap) or {})
    dispose = targets.register_target_resolver("fixture", lambda **kw: lease, selector=lambda **kw: lease_selection(lease))
    try:
        args = {"action": "click", "coordinate": [1, 2], "capture_after": True} if capture_after else {"action": "capture"}
        result = json.loads(tool.handle_computer_use(args, session_id="caller"))
        assert result["code"] == "policy_denied"
        assert shaped == []
        paused[0] = True
        backend = next(iter(tool._backends.values()))
        before = list(backend.calls)
        result = json.loads(tool.handle_computer_use({"action": "click", "coordinate": [1, 2]}, session_id="caller"))
        assert result["code"] == "policy_denied"
        assert backend.calls == before
    finally:
        dispose()
        execution.remove_session_execution_context("target")


@pytest.mark.parametrize("name", ["get_desktop_state", "click"])
def test_async_send_rechecks_control_epoch_after_bridge_scheduling(name):
    import threading
    from types import SimpleNamespace
    from concurrent.futures import ThreadPoolExecutor
    from tools.computer_use.cua_backend_session import _AsyncBridge, _CuaDriverSession
    from tools.computer_use.session_context import desktop_access
    epoch = [0]
    launch = execution.ComputerUseLaunchContext(private_daemon=True, access_epoch=lambda: epoch[0])
    execution.register_session_execution_context("async-target", execution.SessionExecutionContext(computer_use=launch))
    lease = execution.resolve_session_execution_context(session_id="async-target")
    bridge = _AsyncBridge()
    bridge.start()
    session = _CuaDriverSession(bridge, execution_context=lease)
    entered, proceed = threading.Event(), threading.Event()
    sent = []
    class Transport:
        async def call_tool(self, name, args):
            sent.append(name)
            return SimpleNamespace(content=[], isError=False, structuredContent={})
    session._session = Transport()
    session._started = True
    original = session._call_tool_async
    async def delayed(name, args):
        entered.set()
        assert proceed.wait(5)
        return await original(name, args)
    session._call_tool_async = delayed
    def invoke():
        with desktop_access(lease):
            return session.call_tool(name, {}, timeout=10)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(invoke)
            try:
                assert entered.wait(5)
                epoch[0] += 2
            finally:
                proceed.set()
            with pytest.raises(execution.SessionExecutionError, match="epoch"):
                future.result(timeout=10)
        assert sent == []
    finally:
        proceed.set()
        bridge.stop()
        execution.remove_session_execution_context("async-target")


def test_remote_capture_never_reads_guest_path_or_sends_host_output_path(tmp_path, monkeypatch):
    import sys
    from tools.computer_use import cua_backend_session as transport
    secret = tmp_path / "host-only.png"
    secret.write_bytes(b"not guest pixels")
    launch = execution.ComputerUseLaunchContext(private_daemon=True, desktop_only=True,
        driver_command=sys.executable, desktop_attestor=lambda: ("fixture-guest", "generation"))
    execution.register_session_execution_context("remote-target", execution.SessionExecutionContext(computer_use=launch))
    lease = execution.resolve_session_execution_context(session_id="remote-target")
    session = transport._CuaDriverSession(transport._AsyncBridge(), execution_context=lease)
    calls = []
    payload = {"screenshot_file_path": str(secret)}
    def cli(cmd, env, name, timeout, **kw):
        calls.append(json.loads(cmd[3]))
        return dict(payload)
    monkeypatch.setattr(transport, "_cli_run_json", cli)
    try:
        out = session._call_tool_via_cli("get_desktop_state", {}, 5)
        assert out["images"] == []
        assert "screenshot_file_path" not in out["structuredContent"]
        with pytest.raises(execution.SessionExecutionError, match="inline"):
            session._call_tool_via_cli("get_desktop_state", {"screenshot_out_file": str(secret)}, 5)
        assert len(calls) == 1
        assert all("screenshot_out_file" not in args for args in calls)
        payload["screenshot_png_b64"] = "Z3Vlc3Q="
        out = session._call_tool_via_cli("get_desktop_state", {}, 5)
        assert out["images"] == ["Z3Vlc3Q="]
        # Legacy local screenshot-file parsing is unchanged.
        assert transport._cli_result({"screenshot_file_path": str(secret)}, None)["images"]
    finally:
        execution.remove_session_execution_context("remote-target")


def test_cli_retry_rechecks_access_before_each_subprocess(tmp_path, monkeypatch):
    import sys
    from tools.computer_use import cua_backend_session as transport
    from tools.computer_use import cua_backend
    paused = [False]
    def access():
        if paused[0]:
            raise execution.SessionExecutionError("human paused")
        return 0
    launch = execution.ComputerUseLaunchContext(private_daemon=True, desktop_only=True,
        driver_command=sys.executable, desktop_attestor=lambda: ("guest",), access_epoch=access)
    execution.register_session_execution_context("cli-target", execution.SessionExecutionContext(computer_use=launch))
    lease = execution.resolve_session_execution_context(session_id="cli-target")
    session = transport._CuaDriverSession(transport._AsyncBridge(), execution_context=lease)
    spawned = []
    from types import SimpleNamespace
    # External process boundary only: first attempt emits no JSON, then takeover.
    def run(*args, **kw):
        spawned.append(args)
        paused[0] = True
        return SimpleNamespace(stdout="", stderr="", returncode=0)
    monkeypatch.setattr("subprocess.run", run)
    monkeypatch.setattr("time.sleep", lambda _: None)
    monkeypatch.setattr(cua_backend, "sanitize_cua_child_env", lambda env, execution: env)
    monkeypatch.setattr(cua_backend, "cua_driver_child_env", lambda **kw: {})
    try:
        with pytest.raises(execution.SessionExecutionError, match="paused"):
            session._call_tool_via_cli("get_desktop_state", {}, 5)
        assert len(spawned) == 1
    finally:
        execution.remove_session_execution_context("cli-target")


def test_registry_approvals_and_yolo_retire_all_caller_targets(monkeypatch):
    import tools.computer_use_tool  # public registry registration
    from tools.registry import registry
    from tools import approval
    from tools.approval_context import set_current_session_key, reset_current_session_key
    from tools.computer_use import targets
    monkeypatch.setattr(tool, "_new_backend", RecordingBackend)
    selected = []
    leases = []
    for name in ("policy-target-1", "policy-target-2"):
        execution.register_session_execution_context(name, execution.SessionExecutionContext())
        leases.append(execution.resolve_session_execution_context(session_id=name))
    selected[:] = leases[:1]
    dispose = targets.register_target_resolver("policy", lambda **kw: selected[0], selector=lambda **kw: lease_selection(selected[0], current=lambda: selected[0]))
    prompts = []
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    tool.set_approval_callback(lambda command, description, **kw: prompts.append(command) or "session")
    token = set_current_session_key("caller")
    try:
        args = {"action": "click", "coordinate": [1, 2]}
        assert json.loads(registry.dispatch("computer_use", args, session_id="caller"))["ok"]
        a = tool._get_backend("caller")
        selected[0] = leases[1]
        assert json.loads(registry.dispatch("computer_use", args, session_id="caller"))["ok"]
        b = tool._get_backend("caller")
        assert len(prompts) == 1  # approval belongs to caller, not synthetic target
        assert approval.is_approved("caller", "cua:click:background")
        assert all(not approval.is_approved(name, "cua:click:background")
                   for name in ("policy-target-1", "policy-target-2", "other"))
        other = tool._get_backend("other")
        # The host binds approval context; registry kwargs only route the backend.
        other_token = set_current_session_key("other")
        try:
            assert json.loads(registry.dispatch("computer_use", args, session_id="other"))["ok"]
        finally:
            reset_current_session_key(other_token)
        assert len(prompts) == 2
        assert not approval.is_approved("caller", "cua:click:foreground")
        assert json.loads(registry.dispatch("computer_use", {**args, "delivery_mode": "foreground"}, session_id="caller"))["ok"]
        assert len(prompts) == 3  # no background-to-foreground grant widening
        assert "blocked" in registry.dispatch("computer_use", {"action": "key", "keys": "win+l"}, session_id="caller")
        approval.enable_session_yolo("caller")
        assert a.stopped and b.stopped and not other.stopped
        unrestricted = tool._get_backend("caller")
        assert unrestricted.permission_mode == "unrestricted"
        approval.disable_session_yolo("caller")
        assert unrestricted.stopped and not other.stopped
    finally:
        tool.set_approval_callback(None)
        reset_current_session_key(token)
        approval.clear_session("caller")
        approval.clear_session("other")
        approval.disable_session_yolo("caller")
        dispose()
        for name in ("policy-target-1", "policy-target-2"):
            execution.remove_session_execution_context(name)


@pytest.fixture(autouse=True)
def clean_backends():
    tool.reset_backend_for_tests()
    yield
    tool.reset_backend_for_tests()


def test_resolver_is_profile_scoped_and_never_turns_denial_into_unbound(tmp_path):
    targets = importlib.import_module("tools.computer_use.targets")
    home = get_hermes_home()
    seen = []
    assert targets.resolve_target_context("caller") is targets.UNBOUND
    execution.register_session_execution_context("target-only", execution.SessionExecutionContext())
    lease = execution.resolve_session_execution_context(session_id="target-only")
    def resolve(session_id, task_id=None):
        seen.append((session_id, task_id))
        return lease
    dispose = targets.register_target_resolver("fixture", resolve, hermes_home=home, selector=lambda **kw: lease_selection(resolve(**kw)))
    try:
        assert targets.has_target_resolver()
        assert targets.resolve_target_context("caller", "task") is lease
        assert seen == [("caller", "task")]
        assert execution.resolve_session_execution_context(session_id="caller", task_id="task") is None
        with hermes_home_scope(tmp_path / "other"):
            assert not targets.has_target_resolver()
            assert targets.resolve_target_context("caller") is targets.UNBOUND
        # Re-registration must not be undone by a stale disposer.
        replacement = targets.register_target_resolver("fixture", lambda **kw: None, selector=lambda **kw: lease_selection(None))
        dispose()
        assert targets.has_target_resolver()
        with pytest.raises(execution.SessionExecutionError):
            targets.resolve_target_context("caller")
        replacement()
        assert targets.resolve_target_context("caller") is targets.UNBOUND
        def broken(**kw):
            raise RuntimeError("unavailable")
        broken_dispose = targets.register_target_resolver("fixture", broken, selector=lambda **kw: lease_selection(broken(**kw)))
        try:
            with pytest.raises(execution.SessionExecutionError):
                targets.resolve_target_context("caller")
        finally:
            broken_dispose()
    finally:
        dispose()
        execution.remove_session_execution_context("target-only")


@pytest.mark.parametrize("transition", ["none", "denied", "handback", "provider", "lease", "cancel-start"])
def test_approval_binds_selection_without_starting_target(monkeypatch, transition):
    from tools.computer_use import targets
    effects, starts, resolutions = [], [], []
    epoch = [0]
    cancelled = [False]
    execution.register_session_execution_context("selected", execution.SessionExecutionContext(
        computer_use=execution.ComputerUseLaunchContext(private_daemon=True, access_epoch=lambda: epoch[0])))
    lease = execution.resolve_session_execution_context(session_id="selected")
    class Backend(RecordingBackend):
        def start(self):
            starts.append(True)
            if transition == "cancel-start" and not cancelled[0]:
                cancelled[0] = True
                raise KeyboardInterrupt()
        def click(self, **kw):
            effects.append(True)
            return super().click(**kw)
    monkeypatch.setattr(tool, "_new_backend", Backend)
    def select(**kw):
        assert kw == {"session_id": "caller", "task_id": "task"}
        captured = epoch[0]
        def check():
            lease.check()
            if epoch[0] != captured:
                raise execution.SessionExecutionError("selection authority changed")
        def realize(*, before_start=None):
            if before_start is not None:
                before_start()
            resolutions.append(True)
            return lease
        return execution.TargetSelection(realize, check)
    def approval(*args):
        assert starts == resolutions == []
        if transition == "handback":
            epoch[0] += 2
        elif transition == "provider":
            targets.register_target_resolver("fixture", lambda **kw: lease, selector=select)
        elif transition == "lease":
            execution.register_session_execution_context("selected", execution.SessionExecutionContext())
        return json.dumps({"error": "denied"}) if transition == "denied" else None
    monkeypatch.setattr(tool, "_request_approval", approval)
    dispose = targets.register_target_resolver("fixture", lambda **kw: pytest.fail("allocating resolver called"), selector=select)
    try:
        args = {"action": "click", "coordinate": [1, 2], "capture_after": False}
        if transition == "cancel-start":
            with pytest.raises(KeyboardInterrupt):
                tool.handle_computer_use(args, session_id="caller", task_id="task")
            epoch[0] += 2
            starts.clear()
            resolutions.clear()
        result = tool.handle_computer_use(args, session_id="caller", task_id="task")
        expected = transition in ("none", "cancel-start")
        assert bool(effects) == expected, result
        assert starts == resolutions == ([True] if expected else [])
    finally:
        dispose()
        # Dispose any replacement without affecting another provider/profile.
        targets.register_target_resolver("fixture", lambda **kw: lease, selector=lambda **kw: lease_selection(lease))()
        execution.remove_session_execution_context("selected")
