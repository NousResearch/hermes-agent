"""Behavior contracts for cua-driver's verify/escalate and typed-browser ladder.

The fixture used here is a deliberately selected and normalized ``tools/list``
capture.  It contains schemas, not machine/user state, and records the 0.9-era
contract where input properties are the discovery surface.
"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional
from unittest.mock import Mock, patch

import pytest


FIXTURE = Path(__file__).parents[1] / "fixtures" / "cua_driver_0_9_tools_list.json"


@pytest.fixture(autouse=True)
def _reset_computer_use_state():
    from tools.computer_use.tool import reset_backend_for_tests

    reset_backend_for_tests()
    yield
    reset_backend_for_tests()


class _FakeSession:
    def __init__(
        self,
        out: Optional[Dict[str, Any]] = None,
        *,
        input_properties: Optional[Dict[str, set[str]]] = None,
        tools: Optional[set[str]] = None,
    ) -> None:
        self.out = out or {
            "isError": False,
            "data": {},
            "structuredContent": {"effect": "confirmed"},
        }
        self.input_properties = input_properties or {}
        self.tools = tools or {"bring_to_front", *self.input_properties}
        self.calls: list[tuple[str, Dict[str, Any]]] = []

    def call_tool(self, name: str, args: Dict[str, Any], timeout: float = 30.0):
        self.calls.append((name, dict(args)))
        return self.out

    def supports_capability(self, capability: str, tool: Optional[str] = None) -> bool:
        return False

    def supports_input_property(self, tool: str, prop: str) -> bool:
        return prop in self.input_properties.get(tool, set())

    def _has_tool(self, name: str) -> bool:
        return name in self.tools


def _make_backend(session: _FakeSession):
    from tools.computer_use.cua_backend import CuaDriverBackend

    backend = CuaDriverBackend.__new__(CuaDriverBackend)
    backend._session = session
    backend._session_id = "hermes-session"
    backend._snapshot_tokens = {}
    backend._active_pid = 42
    backend._active_window_id = 7
    return backend


def _driver_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"isError": False, "data": {}, "structuredContent": payload}


# ---------------------------------------------------------------------------
# Selected live schema and foreground delivery
# ---------------------------------------------------------------------------


def test_normalized_fixture_is_sanitized_and_records_the_selected_contract():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    tools = {tool["name"]: tool for tool in fixture["tools"]}

    assert fixture["contract_epoch"] == "cua-driver-0.9"
    assert fixture["observed_reported_version"] == "0.8.3"
    assert fixture["capability_version"] == "1"
    assert fixture["observed_tool_count"] == 49
    assert "delivery_mode" in tools["click"]["inputSchema"]["properties"]
    assert "delivery_mode" in tools["type_text"]["inputSchema"]["properties"]
    assert all(
        "input.delivery_mode" not in tool["capabilities"] for tool in tools.values()
    )
    assert "bring_to_front" in tools
    assert "bring_to_front" not in tools["click"]["inputSchema"]["properties"]
    assert {
        "get_browser_state",
        "browser_prepare",
        "browser_navigate",
        "browser_click",
        "browser_type",
        "browser_pointer",
    }.issubset(tools)

    serialized = json.dumps(fixture)
    for forbidden in (
        "/Users/",
        "\\Users\\",
        "localhost",
        "http://",
        "https://",
        "token-",
    ):
        assert forbidden not in serialized


def test_foreground_support_is_discovered_from_tool_input_schema():
    from tools.computer_use.cua_backend import _CuaDriverSession

    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    listed = []
    for item in fixture["tools"]:
        listed.append(
            SimpleNamespace(
                name=item["name"],
                capabilities=item["capabilities"],
                inputSchema=item["inputSchema"],
                model_extra={},
            )
        )

    class _McpSession:
        async def list_tools(self):
            return SimpleNamespace(tools=listed, model_extra={})

    session = _CuaDriverSession.__new__(_CuaDriverSession)
    session._capabilities = {}
    session._input_properties = {}
    session._capability_version = ""
    asyncio.run(session._populate_capabilities(_McpSession()))

    assert session.supports_input_property("click", "delivery_mode") is True
    assert session.supports_input_property("type_text", "delivery_mode") is True
    assert session.supports_input_property("bring_to_front", "delivery_mode") is False
    assert session.supports_capability("input.delivery_mode", tool="click") is False


def test_foreground_focus_is_a_separate_call_before_action():
    session = _FakeSession(input_properties={"click": {"delivery_mode"}})
    backend = _make_backend(session)

    result = backend.click(
        element=3,
        delivery_mode="foreground",
        bring_to_front=True,
    )

    assert result.ok is True
    assert [name for name, _ in session.calls] == ["bring_to_front", "click"]
    focus_args = session.calls[0][1]
    action_args = session.calls[1][1]
    assert focus_args == {"pid": 42, "window_id": 7}
    assert action_args["delivery_mode"] == "foreground"
    assert "bring_to_front" not in action_args


def test_foreground_refuses_only_when_schema_lacks_delivery_property():
    backend = _make_backend(_FakeSession())

    result = backend.click(element=3, delivery_mode="foreground")

    assert result.ok is False
    assert result.code == "foreground_unsupported"
    assert "update" not in result.message.lower()
    assert backend._session.calls == []


def test_invalid_delivery_mode_is_rejected_before_driver_call():
    session = _FakeSession(input_properties={"type_text": {"delivery_mode"}})
    backend = _make_backend(session)

    result = backend.type_text("hello", delivery_mode="sideways")

    assert result.ok is False
    assert result.code == "bad_delivery_mode"
    assert session.calls == []


# ---------------------------------------------------------------------------
# Deterministic verdict precedence and backend isolation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("result_kwargs", "decision"),
    [
        ({"ok": True, "effect": "confirmed", "verified": True}, "done"),
        (
            {
                "ok": True,
                "effect": "unverifiable",
                "verified": False,
                "escalation": {"recommended": "foreground"},
            },
            "verify_fresh_state",
        ),
        ({"ok": True, "effect": "suspected_noop"}, "escalate"),
        ({"ok": False, "code": "browser_input_trust_unavailable"}, "escalate"),
    ],
)
def test_action_verdict_precedence(result_kwargs, decision):
    from tools.computer_use.backend import ActionResult
    from tools.computer_use.tool import _classify_action_result

    result = ActionResult(action="click", **result_kwargs)
    assert _classify_action_result(result)["decision"] == decision


def test_backends_are_isolated_by_hermes_session_and_reused_within_it():
    from tools.computer_use import tool as computer_use

    created = []

    class _Backend:
        def __init__(self):
            created.append(self)

        def start(self):
            pass

        def stop(self):
            pass

    with patch("tools.computer_use.cua_backend.CuaDriverBackend", _Backend):
        first = computer_use._get_backend(session_id="conversation-a")
        first_again = computer_use._get_backend(session_id="conversation-a")
        second = computer_use._get_backend(session_id="conversation-b")

    assert first is first_again
    assert first is not second
    assert created == [first, second]


def test_concurrent_hermes_sessions_do_not_share_backend_state():
    from tools.computer_use import tool as computer_use

    created = []

    class _Backend:
        def __init__(self):
            self.marker = len(created)
            created.append(self)

        def start(self):
            pass

        def stop(self):
            pass

        def typed_browser_state(self, **kwargs):
            return {"marker": self.marker, "pid": kwargs.get("pid")}

    def invoke(session_id):
        return json.loads(
            computer_use.handle_computer_use(
                {"action": "cua_browser_state", "pid": 101, "window_id": 202},
                session_id=session_id,
            )
        )["marker"]

    with patch("tools.computer_use.cua_backend.CuaDriverBackend", _Backend):
        with ThreadPoolExecutor(max_workers=4) as executor:
            markers = list(
                executor.map(invoke, ["conversation-a", "conversation-b"] * 4)
            )

    assert set(markers[0::2]).isdisjoint(set(markers[1::2]))
    assert len(set(markers[0::2])) == 1
    assert len(set(markers[1::2])) == 1
    assert len(created) == 2


def test_persistent_focus_has_a_separate_approval_scope():
    from tools.computer_use import tool as computer_use

    seen = []

    def approve(action, args, summary):
        seen.append(action)
        return "approve_once" if action == "click" else "deny"

    computer_use.set_approval_callback(approve)
    try:
        result = json.loads(
            computer_use.handle_computer_use(
                {
                    "action": "click",
                    "element": 1,
                    "delivery_mode": "foreground",
                    "bring_to_front": True,
                },
                session_id="approval-session",
            )
        )
    finally:
        computer_use.set_approval_callback(None)

    assert seen == ["click", "bring_to_front"]
    assert result["error"] == "denied by user"
    assert result["action"] == "bring_to_front"


# ---------------------------------------------------------------------------
# Session-scoped typed browser routing
# ---------------------------------------------------------------------------


class _BrowserDriver:
    def __init__(self, *, mutation_allowed: bool = True) -> None:
        self.calls: list[tuple[str, Dict[str, Any]]] = []
        self.mutation_allowed = mutation_allowed
        self.snapshot = 0
        self.responses: Dict[str, Dict[str, Any]] = {}

    def has_tool(self, name: str) -> bool:
        return name in {
            "get_browser_state",
            "browser_prepare",
            "browser_navigate",
            "browser_click",
            "browser_type",
            "browser_pointer",
            "browser_dialog",
            "browser_set_input_files",
            "browser_download",
        }

    def call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append((name, dict(args)))
        if name in self.responses:
            return _driver_result(self.responses[name])
        if name == "get_browser_state" and "pid" in args:
            return _driver_result({
                "status": "ok",
                "binding_quality": "exact",
                "mutation_allowed": self.mutation_allowed,
                "target_id": "opaque-target",
                "tabs": [{"tab_id": "opaque-tab"}],
            })
        if name == "get_browser_state":
            self.snapshot += 1
            return _driver_result({
                "status": "ok",
                "refs": {
                    f"p{self.snapshot}:1": {
                        "actions": ["click", "type", "pointer", "scroll"]
                    }
                },
                "continuation": f"continuation-{self.snapshot}",
            })
        return _driver_result({"status": "ok", "effect": "confirmed"})


def _browser_route(driver: _BrowserDriver, session_id: str = "hermes-a"):
    from tools.computer_use.browser_route import CuaTypedBrowserRoute

    return CuaTypedBrowserRoute(
        session_id=session_id,
        call_tool=driver.call,
        has_tool=driver.has_tool,
    )


def _bind_and_snapshot(route) -> str:
    bound = route.observe(pid=101, window_id=202)
    assert bound["exact_binding"] is True
    snapshot = route.observe(tab_id="opaque-tab")
    assert snapshot["fresh_state"] is True
    return next(iter(route.state.refs))


def test_exact_browser_binding_injects_hermes_session_capability():
    driver = _BrowserDriver()
    route = _browser_route(driver, session_id="hermes-owned-session")

    payload = route.observe(pid=101, window_id=202)

    assert payload["exact_binding"] is True
    assert payload["mutation_allowed"] is True
    assert driver.calls == [
        (
            "get_browser_state",
            {"pid": 101, "window_id": 202, "session": "hermes-owned-session"},
        )
    ]


def test_browser_mutation_requires_driver_granted_mutation_capability():
    driver = _BrowserDriver(mutation_allowed=False)
    route = _browser_route(driver)
    route.observe(pid=101, window_id=202)

    result = route.mutate(
        "browser_navigate",
        tab_id="opaque-tab",
        args={"url": "about:blank"},
    )

    assert result["code"] == "browser_mutation_unproven"
    assert result["native_fallback_required"] is True
    assert [name for name, _ in driver.calls] == ["get_browser_state"]


def test_browser_bind_requires_fresh_tab_state_before_first_mutation():
    driver = _BrowserDriver()
    route = _browser_route(driver)
    route.observe(pid=101, window_id=202)

    result = route.mutate(
        "browser_navigate",
        tab_id="opaque-tab",
        args={"url": "about:blank"},
    )

    assert result["code"] == "browser_verification_required"
    assert [name for name, _ in driver.calls] == ["get_browser_state"]


def test_browser_mutation_enforces_current_ref_and_fresh_verification():
    driver = _BrowserDriver()
    route = _browser_route(driver)
    current_ref = _bind_and_snapshot(route)

    stale = route.mutate(
        "browser_click",
        tab_id="opaque-tab",
        args={"ref": "p0:stale"},
    )
    assert stale["code"] == "browser_ref_stale"

    first = route.mutate(
        "browser_click",
        tab_id="opaque-tab",
        args={"ref": current_ref},
    )
    assert first["next_step"] == "fresh_browser_state"
    assert first["verification_required"] is True

    chained = route.mutate(
        "browser_navigate",
        tab_id="opaque-tab",
        args={"url": "about:blank"},
    )
    assert chained["code"] == "browser_verification_required"

    fresh_ref = next(iter(route.observe(tab_id="opaque-tab")["refs"]))
    second = route.mutate(
        "browser_type",
        tab_id="opaque-tab",
        args={"ref": fresh_ref, "text": "hello"},
    )
    assert second["verification_required"] is True


def test_live_semantic_v2_content_refs_are_the_action_capabilities():
    from tools.computer_use.browser_route import _ref_map

    refs = _ref_map({
        "status": "ok",
        "refs": [],
        "content_refs": [
            {
                "ref": "p7:3",
                "role": "button",
                "actions": ["click", "pointer"],
            }
        ],
    })

    assert refs == {"p7:3": {"click", "pointer"}}


def test_dom_event_is_forwarded_only_when_explicitly_requested():
    driver = _BrowserDriver()
    route = _browser_route(driver)
    current_ref = _bind_and_snapshot(route)

    result = route.mutate(
        "browser_pointer",
        tab_id="opaque-tab",
        args={
            "action": "right_click",
            "ref": current_ref,
            "input_route": "dom_event",
        },
    )

    name, sent = driver.calls[-1]
    assert name == "browser_pointer"
    assert sent["input_route"] == "dom_event"
    assert result["input_trust"] == "dom_event"
    assert result["trust_downgrade_explicit"] is True


def test_trust_route_is_rejected_for_tools_without_a_live_route_property():
    driver = _BrowserDriver()
    route = _browser_route(driver)
    current_ref = _bind_and_snapshot(route)

    result = route.mutate(
        "browser_type",
        tab_id="opaque-tab",
        args={"ref": current_ref, "text": "hello", "input_route": "dom_event"},
    )

    assert result["code"] == "browser_input_route_unsupported"
    assert [name for name, _ in driver.calls].count("browser_type") == 0


def test_scope_ref_must_come_from_this_routes_latest_snapshot():
    driver = _BrowserDriver()
    route = _browser_route(driver)
    _bind_and_snapshot(route)

    result = route.observe(tab_id="opaque-tab", scope_ref="other-session:1")

    assert result["code"] == "browser_ref_stale"
    assert len(driver.calls) == 2


def test_typed_browser_refs_do_not_cross_route_sessions():
    driver = _BrowserDriver()
    first = _browser_route(driver, session_id="hermes-a")
    second = _browser_route(driver, session_id="hermes-b")
    first_ref = _bind_and_snapshot(first)
    _bind_and_snapshot(second)

    result = second.mutate(
        "browser_click",
        tab_id="opaque-tab",
        args={"ref": first_ref},
    )

    assert result["code"] == "browser_ref_stale"


def test_trusted_browser_refusal_does_not_silently_change_route():
    driver = _BrowserDriver()
    driver.responses["browser_click"] = {
        "status": "refused",
        "code": "browser_input_trust_unavailable",
    }
    route = _browser_route(driver)
    current_ref = _bind_and_snapshot(route)

    result = route.mutate(
        "browser_click",
        tab_id="opaque-tab",
        args={"ref": current_ref},
    )

    browser_click_calls = [
        args for name, args in driver.calls if name == "browser_click"
    ]
    assert len(browser_click_calls) == 1
    assert browser_click_calls[0].get("input_route") is None
    assert result["trust_change_requires_explicit_choice"] is True
    assert result["native_fallback_available"] is True
    assert route.state.refs == {}
    assert route.state.verification_required is True


def test_typed_mutation_disarms_refs_before_transport_failure():
    driver = _BrowserDriver()
    route = _browser_route(driver)
    current_ref = _bind_and_snapshot(route)

    def fail_transport(name, args):
        raise RuntimeError("connection lost after dispatch")

    route._call_tool = fail_transport
    with pytest.raises(RuntimeError, match="connection lost"):
        route.mutate(
            "browser_click",
            tab_id="opaque-tab",
            args={"ref": current_ref},
        )

    assert route.state.refs == {}
    assert route.state.verification_required is True


def test_read_only_dialog_inspection_does_not_invalidate_page_state():
    driver = _BrowserDriver()
    route = _browser_route(driver)
    current_ref = _bind_and_snapshot(route)

    inspected = route.mutate(
        "browser_dialog",
        tab_id="opaque-tab",
        args={"action": "inspect"},
    )

    assert inspected["fresh_dialog_state"] is True
    assert current_ref in route.state.refs
    assert route.state.verification_required is False


def test_missing_typed_browser_tool_returns_native_fallback_refusal():
    from tools.computer_use.browser_route import CuaTypedBrowserRoute

    call = Mock()
    route = CuaTypedBrowserRoute(
        session_id="hermes-a",
        call_tool=call,
        has_tool=lambda name: False,
    )

    result = route.observe(pid=101, window_id=202)

    assert result["code"] == "typed_browser_unavailable"
    assert result["native_fallback_required"] is True
    call.assert_not_called()


def test_existing_profile_prepare_requires_interactive_driver_grant():
    driver = _BrowserDriver()
    route = _browser_route(driver)

    result = route.prepare(
        pid=101,
        window_id=202,
        profile_mode="existing_profile",
        allow_launch=True,
    )

    assert result["code"] == "browser_consent_required"
    assert result["interactive_grant_required"] is True
    assert driver.calls == []


def test_namespaced_state_and_prepare_actions_use_typed_backend_wrappers():
    from tools.computer_use.tool import _dispatch

    backend = Mock()
    backend.typed_browser_state.return_value = {"status": "ok"}
    backend.typed_browser_prepare.return_value = {"status": "ok"}

    _dispatch(
        backend,
        "cua_browser_state",
        {"pid": 101, "window_id": 202},
    )
    _dispatch(
        backend,
        "cua_browser_prepare",
        {
            "pid": 101,
            "window_id": 202,
            "profile_mode": "isolated_new",
            "allow_launch": True,
        },
    )

    backend.typed_browser_state.assert_called_once_with(pid=101, window_id=202)
    backend.typed_browser_prepare.assert_called_once_with(
        pid=101,
        window_id=202,
        profile_mode="isolated_new",
        profile_name=None,
        allow_launch=True,
    )


def test_public_schema_exposes_only_namespaced_typed_browser_actions():
    from tools.computer_use.schema import COMPUTER_USE_SCHEMA

    action_enum = COMPUTER_USE_SCHEMA["parameters"]["properties"]["action"]["enum"]
    assert "cua_browser_state" in action_enum
    assert "cua_browser_click" in action_enum
    assert "get_browser_state" not in action_enum
    assert "browser_click" not in action_enum
    assert "browser_type_mode" in COMPUTER_USE_SCHEMA["parameters"]["properties"]


@pytest.mark.parametrize(
    ("outer_action", "driver_tool", "args"),
    [
        ("cua_browser_navigate", "browser_navigate", {"url": "about:blank"}),
        ("cua_browser_click", "browser_click", {"ref": "p1:1"}),
        ("cua_browser_type", "browser_type", {"ref": "p1:1", "text": "hello"}),
        (
            "cua_browser_pointer",
            "browser_pointer",
            {"action": "hover", "ref": "p1:1"},
        ),
    ],
)
def test_namespaced_outer_browser_actions_map_to_exact_driver_tools(
    outer_action, driver_tool, args
):
    from tools.computer_use.tool import _dispatch

    backend = Mock()
    backend.typed_browser_action.return_value = {"status": "ok"}

    _dispatch(
        backend,
        outer_action,
        {"tab_id": "opaque-tab", **args},
    )

    backend.typed_browser_action.assert_called_once_with(
        driver_tool,
        tab_id="opaque-tab",
        args=args,
    )


# ---------------------------------------------------------------------------
# Existing additive result and reconnect contracts
# ---------------------------------------------------------------------------


def test_driver_verdict_fields_are_preserved_and_surfaced_additively():
    from tools.computer_use.backend import ActionResult
    from tools.computer_use.tool import _text_response

    result = ActionResult(
        ok=True,
        action="click",
        effect="suspected_noop",
        escalation={"recommended": "foreground"},
        code="background_unavailable",
        path="ax",
        verified=False,
    )
    payload = json.loads(_text_response(result))
    assert payload["effect"] == "suspected_noop"
    assert payload["escalation"] == {"recommended": "foreground"}
    assert payload["code"] == "background_unavailable"
    assert payload["verified"] is False

    bare = json.loads(_text_response(ActionResult(ok=True, action="click")))
    assert bare == {
        "ok": True,
        "action": "click",
        "verdict": {"decision": "verify_fresh_state"},
    }


def test_call_tool_restarts_a_dead_session():
    from tools.computer_use.cua_backend import _CuaDriverSession

    session = _CuaDriverSession.__new__(_CuaDriverSession)
    session._started = False
    starts = []

    def start():
        starts.append(True)
        session._started = True
        session._session = object()

    session.start = start
    session._require_started = lambda: None
    session._is_transient_daemon_error = lambda exc: False
    session._is_closed_session_error = lambda exc: False

    class _Bridge:
        def run(self, coro, timeout=None):
            coro.close()
            return _driver_result({})

    async def call(name, args):
        return {}

    session._bridge = _Bridge()
    session._call_tool_async = call
    session.call_tool("click", {"pid": 1})
    assert starts == [True]
