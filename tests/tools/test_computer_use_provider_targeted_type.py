"""PR #120139 integration with provider selection; recording transport, no native input."""

import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

import tools.computer_use_tool  # noqa: F401 -- register the public dispatch entry point
from tools.bot_desktop import lease
from tools.computer_use import cua_backend, tool
from tools.computer_use.cua_backend_session import _CuaDriverSession
from tools.registry import registry


class RecordingSession(_CuaDriverSession):
    """Negotiate real adapter schemas and reject any unexpected wire properties."""

    def __init__(self, bridge, daemon):
        self.calls = []
        self.elements = [{"element_index": 0, "role": "text", "element_token": "opaque/fresh/0"}]
        self.stale = False
        self.reset_on_capture = False
        self.negotiate("token_only")

    def negotiate(self, contract):
        properties = dict.fromkeys(("text", "pid", "window_id", "session", "delivery_mode"), {})
        if contract != "unsupported":
            properties["element_token"] = {}
        if contract == "legacy":
            properties["element_index"] = {}
        schemas = {
            "type_text": {"properties": properties, "required": ["text", "pid", "window_id", "session"],
                          "additionalProperties": False},
            "bring_to_front": {"properties": dict.fromkeys(("pid", "window_id"), {}),
                               "additionalProperties": False},
            "get_window_state": {"properties": dict.fromkeys(("pid", "window_id", "session", "max_elements"), {})},
        }

        class Transport:
            async def list_tools(self):
                return SimpleNamespace(tools=[SimpleNamespace(name=name, inputSchema=schema)
                                               for name, schema in schemas.items()])

        asyncio.run(self._populate_capabilities(Transport()))

    def call_tool(self, name, args, timeout=30):
        self.calls.append((name, dict(args)))
        schema = self._tool_schemas[name]
        assert not (args.keys() - schema["properties"].keys()), (name, args)
        assert not (set(schema.get("required", [])) - args.keys()), (name, args)
        if name == "get_window_state":
            if self.reset_on_capture:
                self.reset_on_capture = False
                self._notify_transport_reset()
            return {"isError": False, "data": {}, "structuredContent": {"elements": self.elements}}
        if self.stale:
            return {"isError": True, "data": {}, "structuredContent": {"code": "stale_element"}}
        return {"isError": False, "data": {}, "structuredContent": {"effect": "confirmed"}}


@pytest.fixture
def selected_cua(monkeypatch, grant_computer_use_approvals):
    # Exercise config resolution, provider import, factory and the actual backend constructor.
    # Only transport and native lifecycle are replaced; approval and lease gates remain real.
    from hermes_constants import get_hermes_home

    (get_hermes_home() / "config.yaml").write_text("computer_use:\n  backend: cua\n")
    tool.reset_backend_for_tests()
    lease._reset_for_tests()
    sessions = []

    def session_factory(bridge, daemon):
        session = RecordingSession(bridge, daemon)
        sessions.append(session)
        return session

    monkeypatch.setattr(cua_backend, "_CuaDriverSession", session_factory)
    monkeypatch.setattr(cua_backend.CuaDriverBackend, "start", lambda self: None)
    monkeypatch.setattr(cua_backend.CuaDriverBackend, "stop", lambda self: None)

    def invoke(action, **args):
        return json.loads(registry.dispatch("computer_use", {"action": action, **args}, session_id="provider-owner"))

    try:
        captured = invoke("capture", mode="ax", app="Editor", pid=42, window_id=7)
        assert captured["elements"][0]["index"] == 0, captured
        assert len(sessions) == 1
        backend = tool._get_backend("provider-owner")
        assert isinstance(backend, cua_backend.CuaDriverBackend)
        assert backend.permission_mode == "standard"
        sessions[0].calls.clear()
        yield invoke, backend, sessions[0]
    finally:
        tool.reset_backend_for_tests()
        lease._reset_for_tests()


@pytest.mark.parametrize("contract", ["token_only", "legacy"])
def test_selected_cua_provider_types_fresh_index_zero(selected_cua, contract):
    invoke, backend, driver = selected_cua
    driver.negotiate(contract)
    # A new capture must replace the constructor's initial snapshot token.
    driver.elements[0]["element_token"] = "opaque/recaptured/0"
    invoke("capture", mode="ax", pid=42, window_id=7)
    driver.calls.clear()

    result = invoke("type", text="Hello Привет", element=0, pid=42, window_id=7)

    assert result["ok"] is True, result
    expected = {"text": "Hello Привет", "pid": 42, "window_id": 7, "session": backend._session_id,
                "element_token": "opaque/recaptured/0"}
    if contract == "legacy":
        expected["element_index"] = 0
    assert driver.calls == [("type_text", expected)]


@pytest.mark.parametrize("case, code", [
    ("missing_token", "targeted_type_stale"), ("missing_index", "targeted_type_stale"),
    ("reset", "targeted_type_stale"), ("pid", "input_target_mismatch"),
    ("window_id", "input_target_mismatch"), ("unsupported", "targeted_type_unsupported"),
])
def test_selected_cua_provider_refuses_before_focus_or_type(selected_cua, case, code):
    invoke, backend, driver = selected_cua
    args = {"element": 0, "text": "hello", "delivery_mode": "foreground", "bring_to_front": True}
    if case == "missing_token":
        driver.elements[0].pop("element_token")
        invoke("capture", mode="ax", pid=42, window_id=7)
        driver.calls.clear()
    elif case == "missing_index":
        args["element"] = 99
    elif case == "reset":
        driver._notify_transport_reset()
        backend._set_active_target({"pid": 42, "window_id": 7})
    elif case in {"pid", "window_id"}:
        args[case] = 99
    elif case == "unsupported":
        driver.negotiate("unsupported")

    result = invoke("type", **args)

    assert result["ok"] is False, result
    assert result["code"] == code
    assert driver.calls == []


def test_selected_cua_provider_preserves_driver_stale_refusal(selected_cua):
    invoke, backend, driver = selected_cua
    driver.stale = True

    result = invoke("type", text="hello", element=0)

    assert result["ok"] is False
    assert result["code"] == "stale_element"
    assert driver.calls == [("type_text", {"text": "hello", "pid": 42, "window_id": 7,
                                         "session": backend._session_id, "element_token": "opaque/fresh/0"})]


def test_selected_cua_provider_rearms_only_fresh_capture_after_reset(selected_cua):
    invoke, backend, driver = selected_cua
    driver.reset_on_capture = True
    driver.elements[0]["element_token"] = "opaque/reconnected/0"
    captured = invoke("capture", mode="ax", pid=84, window_id=14)
    assert captured["elements"][0]["index"] == 0
    driver.calls.clear()

    result = invoke("type", text="hello", element=0, pid=84, window_id=14)

    assert result["ok"] is True, result
    assert driver.calls == [("type_text", {"text": "hello", "pid": 84, "window_id": 14,
                                         "session": backend._session_id, "element_token": "opaque/reconnected/0"})]


@pytest.mark.parametrize("gate", ["approval", "lease", "fence"])
def test_selected_cua_provider_retains_input_gates(selected_cua, gate):
    invoke, backend, driver = selected_cua
    if gate == "lease":
        lease.acquire("human")
    elif gate == "approval":
        tool.set_approval_callback(lambda *args, **kwargs: "deny")
    else:
        def approve_after_takeover(*args, **kwargs):
            lease.acquire("human")
            lease.release("human")
            return "once"

        tool.set_approval_callback(approve_after_takeover)

    result = invoke("type", text="hello", element=0)

    if gate == "approval":
        assert "User denied" in result["error"]
    else:
        assert result["code"] == "human_has_control"
    assert driver.calls == []


def test_selected_legacy_provider_refuses_targeted_type_without_cua_fallback(selected_cua):
    from hermes_constants import get_hermes_home

    invoke, backend, driver = selected_cua
    tool.reset_backend_for_tests()
    plugin = get_hermes_home() / "plugins" / "cu-legacy-targeted-test"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: cu-legacy-targeted-test\nkind: exclusive\n")
    (plugin / "__init__.py").write_text('''
from tools.computer_use.backend import ComputerUseProvider
from tools.computer_use.tool import _NoopBackend

BACKENDS = []

class LegacyProvider(ComputerUseProvider):
    name = "cu-legacy-targeted-test"
    def create_backend(self, *, permission_mode):
        backend = _NoopBackend()
        BACKENDS.append(backend)
        return backend

def register(ctx):
    ctx.register_computer_use_provider(LegacyProvider())
''')
    (get_hermes_home() / "config.yaml").write_text("computer_use:\n  backend: cu-legacy-targeted-test\n")
    namespace = "_hermes_user_computer_use.cu-legacy-targeted-test"
    try:
        result = invoke("type", text="targeted", element=0)
        assert result["code"] == "targeted_type_unsupported", result
        selected = sys.modules[namespace].BACKENDS
        assert len(selected) == 1 and selected[0].calls == []
        assert invoke("type", text="focused")["ok"] is True
        assert selected[0].calls == [("type", {"text": "focused", "delivery_mode": None, "bring_to_front": False})]
        assert driver.calls == []
    finally:
        sys.modules.pop(namespace, None)
