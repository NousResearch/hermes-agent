"""Registry-level controller-dispatch regression: the guarded-eval policy must hold
on the extension-controller lane, not only on the legacy fallback.

An enabled extension controller advertising ``browser_cdp`` is authoritative for the
call once ``routed_browser_handler`` selects it, so the registry handler must apply
the same guarded-evaluation policy BEFORE controller selection: the controller
receives ``throwOnSideEffect`` parameters, and persistent/async evaluation entry
points are rejected without any dispatch. This mirrors the reviewer reproduction
from PR #127908 — the controller-dispatch case fails when ``throwOnSideEffect`` is
absent from the dispatched arguments.
"""
import json

import pytest

from gateway.session_context import clear_session_vars, set_session_vars
from tools import browser_cdp_tool  # noqa: F401  # imported for registry.register side effect
from tools import browser_tool_eval_policy as policy
from tools.registry import registry


class RecordingBroker:
    """Minimal broker contract: one selected controller, records dispatch arguments."""

    def __init__(self, result='{"ok": true, "source": "browser-extension"}'):
        self.result = result
        self.dispatched = []

    def scope_for_session(self, **identity):
        return "scope-fixture"

    def select(self, scope, action):
        return "controller-fixture"

    def dispatch(self, scope, *, action, arguments, tool_call_id=""):
        self.dispatched.append({"action": action, "arguments": arguments,
                                "tool_call_id": tool_call_id})
        return self.result


@pytest.fixture
def controller_lane(monkeypatch):
    """Guarded browser mode with one bound, capable extension controller."""
    monkeypatch.setattr(policy, "_eval_ssrf_guard_active", lambda task: True)
    monkeypatch.setattr(policy, "_allow_unsafe_browser_evaluate", lambda: False)
    monkeypatch.setattr(policy, "_current_page_private_url", lambda task: None)
    from gateway import browser_control_broker
    broker = RecordingBroker()
    monkeypatch.setattr(browser_control_broker, "browser_control_enabled", lambda: True)
    monkeypatch.setattr(browser_control_broker, "get_browser_control_broker", lambda: broker)
    tokens = set_session_vars(
        session_id="session-fixture",
        browser_control_principal="principal-fixture",
        browser_control_transport_family="local-api",
    )
    try:
        yield broker
    finally:
        clear_session_vars(tokens)


def _cdp_handler():
    return registry.get_entry("browser_cdp").handler


def test_controller_receives_guarded_runtime_evaluate_params(controller_lane):
    out = json.loads(_cdp_handler()(
        {"method": "Runtime.evaluate",
         "params": {"expression": "document.title", "returnByValue": True}},
        task_id="task-fixture", session_id="session-fixture"))
    assert out["ok"] is True
    assert len(controller_lane.dispatched) == 1
    dispatched = controller_lane.dispatched[0]
    assert dispatched["action"] == "browser_cdp"
    sent = dispatched["arguments"]["params"]
    assert sent["throwOnSideEffect"] is True
    assert sent["awaitPromise"] is False
    assert sent["expression"] == "document.title"
    assert sent["returnByValue"] is True


def test_controller_receives_guarded_evaluate_on_call_frame(controller_lane):
    out = json.loads(_cdp_handler()(
        {"method": "Debugger.evaluateOnCallFrame",
         "params": {"expression": "document.title", "callFrameId": "frame-fixture"}},
        task_id="task-fixture", session_id="session-fixture"))
    assert out["ok"] is True
    sent = controller_lane.dispatched[0]["arguments"]["params"]
    assert sent["throwOnSideEffect"] is True
    assert "awaitPromise" not in sent  # this method has no awaitPromise parameter


@pytest.mark.parametrize("method,params", [
    ("Runtime.evaluate", {"expression": "await 3", "replMode": True}),
    ("Runtime.callFunctionOn",
     {"functionDeclaration": "async function(){}", "awaitPromise": True, "objectId": "o1"}),
    ("Page.addScriptToEvaluateOnNewDocument", {"source": "fetch('http://127.0.0.1')"}),
    ("Runtime.runScript", {"scriptId": "cached"}),
    ("Debugger.setScriptSource", {"scriptId": "existing", "scriptSource": "1"}),
])
def test_persistent_or_async_entry_points_rejected_before_controller_dispatch(
        controller_lane, method, params):
    out = json.loads(_cdp_handler()(
        {"method": method, "params": params},
        task_id="task-fixture", session_id="session-fixture"))
    assert "Blocked:" in out["error"], out
    assert "unsupported" in out["error"] or "read-only" in out["error"]
    assert controller_lane.dispatched == []


def test_opt_out_leaves_controller_arguments_unchanged(monkeypatch, controller_lane):
    """Outside guarded mode the controller lane keeps the original request bytes."""
    monkeypatch.setattr(policy, "_eval_ssrf_guard_active", lambda task: False)
    original_params = {"expression": "document.title", "awaitPromise": True}
    out = json.loads(_cdp_handler()(
        {"method": "Runtime.evaluate", "params": dict(original_params)},
        task_id="task-fixture", session_id="session-fixture"))
    assert out["ok"] is True
    assert controller_lane.dispatched[0]["arguments"]["params"] == original_params
