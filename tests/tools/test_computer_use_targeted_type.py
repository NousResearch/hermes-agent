"""Public dispatch through real capture/token/input code; only the driver transport is fake."""

import json
from pathlib import Path

import pytest

import tools.computer_use_tool  # noqa: F401 — register the public tool
from tools.computer_use import backend as interface, cua_backend, cua_backend_input, tool
from tools.computer_use.backend import ActionResult
from tools.computer_use.cua_backend_session import _CuaDriverSession
from tools.registry import registry


class Driver(_CuaDriverSession):
    def __init__(self):
        self._capabilities = {"type_text": set(), "get_window_state": set()}
        self._tool_schemas = {"type_text": {"properties": {
            name: {} for name in ("element_index", "element_token", "delivery_mode")
        }}}
        self.elements = [{"element_index": i, "role": "text", "element_token": f"opaque/first/{i}"}
                         for i in (0, 3)]
        self.calls = []
        self.stale = False

    def call_tool(self, name, args, timeout=30):
        self.calls.append((name, dict(args)))
        if name == "get_window_state":
            return {"isError": False, "data": {}, "structuredContent": {"elements": self.elements}}
        if self.stale:
            return {"isError": True, "data": {}, "structuredContent": {"code": "stale_element"}}
        return {"isError": False, "data": {}, "structuredContent": {"effect": "confirmed"}}


@pytest.fixture
def harness(monkeypatch, grant_computer_use_approvals):
    root = Path(__file__).resolve().parents[2]
    for module in (interface, cua_backend, cua_backend_input, tool):
        assert Path(module.__file__).resolve().is_relative_to(root)
    tool.reset_backend_for_tests()
    backends = []

    def new_backend(mode):
        backend = cua_backend.CuaDriverBackend.__new__(cua_backend.CuaDriverBackend)
        backend._clear_active_target()
        backend._session = Driver()
        backend._session_id = f"driver-owner-{len(backends)}"
        backend.permission_mode = mode
        backend.start = backend.stop = lambda: None
        backends.append(backend)
        return backend

    monkeypatch.setattr(tool, "_new_backend", new_backend)
    monkeypatch.setattr(cua_backend, "_computer_use_cfg", lambda: {})

    def invoke(action, sid="owner", **args):
        return json.loads(registry.dispatch("computer_use", {"action": action, **args}, session_id=sid))

    invoke("capture", mode="ax", app="Editor", pid=42, window_id=7)
    backends[0]._session.calls.clear()
    yield invoke, backends
    tool.reset_backend_for_tests()


@pytest.mark.parametrize("element", [None, 0, 3])
@pytest.mark.parametrize("negotiation", ["schema", "capability", "old"])
@pytest.mark.parametrize("delivery_mode", [None, "foreground"])
def test_public_type_preserves_selected_snapshot_or_refuses(harness, element, negotiation, delivery_mode):
    invoke, backends = harness
    backend = backends[0]
    driver = backend._session
    props = driver._tool_schemas["type_text"]["properties"]
    if negotiation != "schema":
        props.pop("element_token")
    if negotiation == "capability":
        driver._capabilities["type_text"].add("accessibility.element_tokens")
    if negotiation == "old":
        props.pop("element_index")
    # Re-capture replaces all old tokens via production capture and parsing.
    driver.elements = [{**e, "element_token": f"opaque/fresh/{e['element_index']}"} for e in driver.elements]
    invoke("capture", mode="ax", app="Editor", pid=42, window_id=7)
    driver.calls.clear()
    result = invoke("type", text="Hello Привет", element=element, delivery_mode=delivery_mode,
                    pid=42, window_id=7)
    if element is not None and negotiation == "old":
        assert result["code"] == "targeted_type_unsupported"
        assert result["ok"] is False
        assert driver.calls == []
        return
    assert result["ok"] is True
    expected = {"text": "Hello Привет", "pid": 42, "window_id": 7, "session": backend._session_id}
    if element is not None:
        expected.update(element_index=element, element_token=f"opaque/fresh/{element}")
    if delivery_mode:
        expected["delivery_mode"] = delivery_mode
    assert driver.calls == [("type_text", expected)]


def test_targeted_type_uses_exact_capture_after_transport_restart(harness):
    invoke, backends = harness
    backend = backends[0]
    driver = backend._session
    driver.elements = [{"element_index": 1, "role": "text", "element_token": "opaque/restarted/1"}]
    original_call = driver.call_tool
    restarted = False

    def call_after_restart(name, args, timeout=30):
        nonlocal restarted
        if name == "get_window_state" and not restarted:
            restarted = True
            backend._handle_transport_reset()
        return original_call(name, args, timeout)

    driver.call_tool = call_after_restart
    captured = invoke("capture", mode="ax", pid=1488337, window_id=94140107057968)
    assert captured["elements"][0]["index"] == 1
    driver.calls.clear()

    result = invoke("type", text="live target", element=1, pid=1488337, window_id=94140107057968)

    assert result["ok"] is True
    assert driver.calls == [("type_text", {
        "text": "live target", "pid": 1488337, "window_id": 94140107057968,
        "element_index": 1, "element_token": "opaque/restarted/1", "session": backend._session_id,
    })]


@pytest.mark.parametrize(("key", "value"), [("pid", 1488338), ("window_id", 94140107057969)])
def test_targeted_type_after_transport_restart_refuses_mismatched_target(harness, key, value):
    invoke, backends = harness
    backend = backends[0]
    driver = backend._session
    driver.elements = [{"element_index": 1, "role": "text", "element_token": "opaque/restarted/1"}]
    original_call = driver.call_tool
    restarted = False

    def call_after_restart(name, args, timeout=30):
        nonlocal restarted
        if name == "get_window_state" and not restarted:
            restarted = True
            backend._handle_transport_reset()
        return original_call(name, args, timeout)

    driver.call_tool = call_after_restart
    invoke("capture", mode="ax", pid=1488337, window_id=94140107057968)
    driver.calls.clear()

    target = {"pid": 1488337, "window_id": 94140107057968}
    target[key] = value
    result = invoke("type", text="wrong target", element=1, **target)

    assert result["ok"] is False
    assert result["code"] == "input_target_mismatch"
    assert driver.calls == []


@pytest.mark.parametrize("case", [
    "missing_index", "missing_token", "empty_token", "invalidated", "other_session",
    "negative", "boolean", "string", "pid", "window_id", "app", "coordinate",
    "from_element", "element_token", "snapshot_id", "old_backend", "kwargs_backend",
    "blocked", "denied", "driver_stale", "index_only_schema", "token_only_schema",
])
def test_targeted_type_fails_closed_without_global_fallback(harness, case):
    invoke, backends = harness
    backend = backends[0]
    driver = backend._session
    args = {"text": "hello", "element": 0}
    sid = "owner"
    expected = "targeted_type_stale"
    if case == "missing_index":
        args["element"] = 99
    elif case in {"missing_token", "empty_token"}:
        driver.elements = [{"element_index": 0, "role": "text"}]
        if case == "empty_token":
            driver.elements[0]["element_token"] = ""
        invoke("capture", mode="ax", pid=42, window_id=7)
        driver.calls.clear()
    elif case == "invalidated":
        backend._handle_transport_reset()
        # A target restored without a fresh snapshot must not restore old tokens.
        backend._set_active_target({"pid": 42, "window_id": 7})
    elif case == "other_session":
        sid = "other-owner"
        invoke("capture", sid=sid, mode="ax", pid=99, window_id=8)
        other = backends[1]._session
        other.elements = [{"element_index": 3, "role": "text", "element_token": "other/3"}]
        invoke("capture", sid=sid, mode="ax", pid=99, window_id=8)
        other.calls.clear()
    elif case in {"negative", "boolean", "string"}:
        args["element"] = {"negative": -1, "boolean": True, "string": "0"}[case]
        expected = "invalid_element"
    elif case in {"pid", "window_id", "app"}:
        args[case] = "Other" if case == "app" else 99
        expected = "input_target_mismatch"
    elif case in {"coordinate", "from_element", "element_token", "snapshot_id"}:
        args[case] = [1, 2] if case == "coordinate" else "other-selector"
        expected = "incompatible_type_selector"
    elif case in {"old_backend", "kwargs_backend"}:
        def legacy(text, *, delivery_mode=None, bring_to_front=False):
            driver.calls.append(("legacy", text))
            return ActionResult(ok=True, action="type")

        def kwargs_only(text, **kwargs):
            return legacy(text)

        backend.type_text = legacy if case == "old_backend" else kwargs_only
        expected = "targeted_type_unsupported"
    elif case == "blocked":
        args["text"] = "sudo rm -rf /"
    elif case == "denied":
        tool.set_approval_callback(lambda *args, **kwargs: "deny")
    elif case == "driver_stale":
        driver.stale = True
        expected = "stale_element"
    elif case in {"index_only_schema", "token_only_schema"}:
        prop = "element_token" if case == "index_only_schema" else "element_index"
        driver._tool_schemas["type_text"]["properties"].pop(prop)
        expected = "targeted_type_unsupported"

    if case not in {"blocked", "denied", "driver_stale"}:
        args.update(delivery_mode="foreground", bring_to_front=True)
    result = invoke("type", sid=sid, **args)
    if case in {"blocked", "denied"}:
        assert ("blocked pattern" if case == "blocked" else "User denied") in result["error"]
    else:
        assert result.get("code") == expected, result
        assert result["ok"] is False
    if case == "driver_stale":
        assert len(driver.calls) == 1
        assert driver.calls[0][1]["element_token"] == "opaque/first/0"
    else:
        assert all(b._session.calls == [] for b in backends)
    if case in {"old_backend", "kwargs_backend"}:
        assert invoke("type", text="legacy")["ok"] is True
        assert driver.calls == [("legacy", "legacy")]
