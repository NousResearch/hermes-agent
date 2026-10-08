"""Public dispatch through real capture/token/input code; only the driver transport is fake."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

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
        self.strict = False

    def call_tool(self, name, args, timeout=30):
        self.calls.append((name, dict(args)))
        if name == "get_window_state":
            return {"isError": False, "data": {}, "structuredContent": {"elements": self.elements}}
        if self.strict:
            schema = self._tool_schemas[name]
            if set(args) - schema["properties"].keys() or set(schema.get("required", [])) - args.keys():
                return {"isError": True, "data": {}, "structuredContent": {"code": "invalid_arguments"}}
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
@pytest.mark.parametrize("negotiation", ["schema", "capability", "old", "token_only"])
@pytest.mark.parametrize("delivery_mode", [None, "foreground"])
def test_public_type_preserves_selected_snapshot_or_refuses(harness, element, negotiation, delivery_mode):
    invoke, backends = harness
    backend = backends[0]
    driver = backend._session
    props = driver._tool_schemas["type_text"]["properties"]
    if negotiation in {"capability", "old"}:
        props.pop("element_token")
    if negotiation == "capability":
        driver._capabilities["type_text"].add("accessibility.element_tokens")
    if negotiation in {"old", "token_only"}:
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
        expected["element_token"] = f"opaque/fresh/{element}"
        if negotiation != "token_only":
            expected["element_index"] = element
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
    "blocked", "denied", "driver_stale", "index_only_schema",
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
    elif case == "index_only_schema":
        driver._tool_schemas["type_text"]["properties"].pop("element_token")
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


# Selected input-property/required-field contract from the official Linux 0.31.0 reference:
# https://github.com/trycua/cua/blob/cua-driver-rs-v0.31.0/docs/content/docs/reference/cua-driver/mcp-tools-linux.mdx
# Output indices remain capture selectors; the strict input surface accepts tokens only.
_ELEMENT_INPUTS = {
    "type_text": ("coordinate_frame delivery_mode element_token pid scope session target text window_id x y", ["text"]),
    "click": ("button capture_id coordinate_frame count cursor_id delivery_mode element_token from_zoom modifier "
              "pid scope session target window_id x y", []),
    "set_value": ("delivery_mode element_token pid session value window_id", ["pid", "value"]),
    "scroll": ("amount by coordinate_frame cursor_id delivery_mode direction element_token pid scope session "
               "target window_id x y", ["direction"]),
}
_PUBLIC_ELEMENT_ACTIONS = [
    ("type", "type_text", {"text": "Hello Привет"}),
    ("click", "click", {}),
    ("set_value", "set_value", {"value": "selected field"}),
    ("scroll", "scroll", {"direction": "down", "amount": 3}),
]


def _discover_strict_contract(driver, contract):
    schemas = {
        name: {"type": "object", "properties": dict.fromkeys(properties.split(), {}),
               "required": required, "additionalProperties": False}
        for name, (properties, required) in _ELEMENT_INPUTS.items()
    }
    schemas["bring_to_front"] = {"properties": dict.fromkeys(("pid", "window_id"), {}),
                                 "additionalProperties": False}
    for schema in schemas.values():
        if contract == "legacy":
            schema["properties"]["element_index"] = {}
        if contract in {"unsupported", "index_only", "false_capability"}:
            schema["properties"].pop("element_token", None)
        if contract in {"index_only", "false_capability"}:
            schema["properties"]["element_index"] = {}

    class Transport:
        async def list_tools(self):
            return SimpleNamespace(tools=[SimpleNamespace(
                name=name, inputSchema=schema,
                capabilities=["accessibility.element_tokens"] if contract == "false_capability" else [],
            ) for name, schema in schemas.items()])

    asyncio.run(driver._populate_capabilities(Transport()))
    driver.strict = True


@pytest.mark.parametrize("action, native, payload", _PUBLIC_ELEMENT_ACTIONS)
@pytest.mark.parametrize("contract", ["token_only", "legacy"])
@pytest.mark.parametrize("element", [0, 3])
def test_public_element_actions_normalize_to_strict_schema(harness, action, native, payload, contract, element):
    invoke, backends = harness
    backend = backends[0]
    driver = backend._session
    _discover_strict_contract(driver, contract)
    driver.elements = [{**e, "element_token": f"opaque/fresh /Привет:{e['element_index']}"} for e in driver.elements]
    invoke("capture", mode="ax", pid=42, window_id=7)
    driver.calls.clear()

    result = invoke(action, element=element, **payload)

    assert result["ok"] is True, result
    expected = {"pid": 42, "window_id": 7, "session": backend._session_id,
                "element_token": f"opaque/fresh /Привет:{element}", **payload}
    if native == "click":
        expected["button"] = "left"
    if contract == "legacy":
        expected["element_index"] = element
    assert driver.calls == [(native, expected)]


@pytest.mark.parametrize("action, native, payload", _PUBLIC_ELEMENT_ACTIONS)
@pytest.mark.parametrize("case", [
    "missing_index", "missing_token", "empty_token", "recaptured", "reset", "no_window",
    "negative", "boolean", "string", "unsupported", "index_only", "false_capability", "driver_stale",
])
def test_public_element_actions_refuse_without_unintended_send(harness, action, native, payload, case):
    invoke, backends = harness
    backend = backends[0]
    driver = backend._session
    _discover_strict_contract(driver, case)
    element = {"missing_index": 99, "negative": -1, "boolean": True, "string": "0"}.get(case, 0)
    if case in {"missing_token", "empty_token", "recaptured"}:
        driver.elements = [{"element_index": 3, "element_token": "other/3"}] if case == "recaptured" else [
            {"element_index": 0, **({"element_token": ""} if case == "empty_token" else {})}]
        invoke("capture", mode="ax", pid=42, window_id=7)
        driver.calls.clear()
    if case == "reset":
        backend._handle_transport_reset()
        backend._set_active_target({"pid": 42, "window_id": 7})
    if case == "no_window":
        backend._active_window_id = None
    driver.stale = case == "driver_stale"
    # If an element was explicitly selected, neither coordinates nor focus may replace it.
    extras = {"coordinate": [10, 20]} if action in {"click", "scroll"} and case != "no_window" else {}
    if action != "set_value" and case not in {"driver_stale", "no_window"}:
        extras.update(delivery_mode="foreground", bring_to_front=True)

    result = invoke(action, element=element, **payload, **extras)

    assert result["ok"] is False, result
    if case == "driver_stale":
        assert result["code"] == "stale_element"
        assert driver.calls == [(native, {"pid": 42, "window_id": 7, "session": backend._session_id,
            "element_token": "opaque/first/0", **payload, **({"button": "left"} if native == "click" else {})})]
    else:
        assert result.get("code") != "invalid_arguments", result
        assert driver.calls == []
