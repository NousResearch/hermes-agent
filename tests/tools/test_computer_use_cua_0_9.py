"""Behavior contracts for cua-driver's verify/escalate and typed-browser ladder.

The fixture used here is a deliberately selected and normalized ``tools/list``
capture.  It contains schemas, not machine/user state, and records the 0.9-era
contract where input properties are the discovery surface.
"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, Mock, patch

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
        def __init__(self, permission_mode="standard"):
            self.permission_mode = permission_mode
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


def test_release_seam_stops_exact_backend_and_clears_session_state():
    from tools.computer_use import tool as computer_use

    first = MagicMock()
    second = MagicMock()
    computer_use._backends.update({
        "conversation-a": first,
        "conversation-b": second,
    })
    computer_use._backend_call_locks.update({
        "conversation-a": computer_use.threading.RLock(),
        "conversation-b": computer_use.threading.RLock(),
    })
    computer_use._session_auto_approve["conversation-a"] = True
    computer_use._always_allow["conversation-a"] = {
        ("click", "background"),
    }

    assert computer_use.release_computer_use_session("conversation-a") is True
    assert computer_use.release_computer_use_session("conversation-a") is False

    first.stop.assert_called_once_with()
    second.stop.assert_not_called()
    assert "conversation-a" not in computer_use._backends
    assert "conversation-a" not in computer_use._backend_call_locks
    assert "conversation-a" not in computer_use._session_auto_approve
    assert "conversation-a" not in computer_use._always_allow
    assert computer_use._backends["conversation-b"] is second


def test_release_seam_evicts_state_even_when_backend_stop_fails():
    from tools.computer_use import tool as computer_use

    backend = MagicMock()
    backend.stop.side_effect = RuntimeError("driver teardown failed")
    computer_use._backends["failed-run"] = backend
    computer_use._backend_call_locks["failed-run"] = computer_use.threading.RLock()
    computer_use._session_auto_approve["failed-run"] = True

    assert computer_use.release_computer_use_session("failed-run") is True
    assert "failed-run" not in computer_use._backends
    assert "failed-run" not in computer_use._backend_call_locks
    assert "failed-run" not in computer_use._session_auto_approve


def test_release_seam_waits_for_in_flight_action_before_stopping_backend():
    from tools.computer_use import tool as computer_use

    backend = MagicMock()
    call_lock = computer_use.threading.RLock()
    computer_use._backends["cancelled-run"] = backend
    computer_use._backend_call_locks["cancelled-run"] = call_lock

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        call_lock.acquire()
        try:
            released = pool.submit(
                computer_use.release_computer_use_session,
                "cancelled-run",
            )
            with pytest.raises(FutureTimeoutError):
                released.result(timeout=0.05)
            backend.stop.assert_not_called()
        finally:
            call_lock.release()

        assert released.result(timeout=1) is True
    finally:
        pool.shutdown(wait=True)
    backend.stop.assert_called_once_with()


def test_concurrent_hermes_sessions_do_not_share_backend_state():
    from tools.computer_use import tool as computer_use

    created = []

    class _Backend:
        def __init__(self, permission_mode="standard"):
            self.permission_mode = permission_mode
            self.marker = len(created)
            created.append(self)

        def start(self):
            pass

        def stop(self):
            pass

        def list_apps(self):
            return [{"marker": self.marker}]

    def invoke(session_id):
        return json.loads(
            computer_use.handle_computer_use(
                {"action": "list_apps"},
                session_id=session_id,
            )
        )["apps"][0]["marker"]

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

