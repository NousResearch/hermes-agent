"""Element addressing against the live cua-driver action contract.

cua-driver 0.32.0 (trycua/cua#3873, "accept element_token as the only element target") removed
``element_index`` and ``snapshot_id`` from every action schema and refuses unknown argument names at
dispatch. 0.21.0 still advertises ``element_index`` next to ``element_token``. These tests drive the
backend against a strict fake driver built from the frozen live ``tools/list`` schemas of both versions
(tests/fixtures/cua_driver_element_addressing_schemas.json):

* token-only driver: an element action carries the current snapshot's ``element_token`` and never
  ``element_index``; with no token for that index the backend refuses locally — it never falls back to
  coordinates and never sends a bare index;
* legacy driver: unchanged (``element_index`` plus the token).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "cua_driver_element_addressing_schemas.json"
TOKEN_RE = re.compile(r"^s[0-9a-f]{8}:[0-9]+$")
PID, WINDOW = 4242, 77
ELEMENTS = [("button", "Increment"), ("button", "Reset"), ("check box", "I agree")]


class StrictFakeDriver:
    """A cua-driver stand-in that enforces the advertised schema like the real dispatch: unknown argument
    names are refused, element tokens must name the CURRENT snapshot of the window (a newer
    get_window_state supersedes older tokens)."""

    def __init__(self, schemas: Dict[str, Dict[str, Any]]) -> None:
        self.schemas = schemas
        self.snapshot = 0
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    @staticmethod
    def _ok(structured: Any = None, data: Any = "ok") -> Dict[str, Any]:
        return {"data": data, "images": [], "image_mime_types": [], "structuredContent": structured, "isError": False}

    @staticmethod
    def _err(message: str) -> Dict[str, Any]:
        return {"data": message, "images": [], "image_mime_types": [],
                "structuredContent": {"status": "refused", "refusal": {"code": "invalid_arguments", "message": message}},
                "isError": True}

    def call_tool(self, name: str, args: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
        self.calls.append((name, dict(args)))
        if name == "list_windows":
            return self._ok({"windows": [{"app_name": "Main.py", "pid": PID, "window_id": WINDOW,
                                          "is_on_screen": True, "title": "Tasks", "z_index": 0}]})
        if name == "get_window_state":
            self.snapshot += 1
            return self._ok({"elements": [
                {"element_index": i, "role": role, "label": label,
                 "element_token": f"s{self.snapshot:08x}:{i}", "frame": {"x": 10 * i, "y": 5, "w": 8, "h": 8}}
                for i, (role, label) in enumerate(ELEMENTS)]}, data="Tasks - 3 elements\n")
        schema = self.schemas.get(name)
        if schema is None:
            return self._ok()
        unknown = sorted(set(args) - set(schema["properties"]))
        if unknown:
            return self._err(f"{name}: unknown argument {unknown[0]}")
        token = args.get("element_token")
        if token is not None:
            if not TOKEN_RE.match(token):
                return self._err("element_token has invalid format")
            if int(token[1:9], 16) != self.snapshot:
                return self._err("element_token is stale; call get_window_state again to refresh")
        return self._ok({"verified": False})

    def action_calls(self) -> List[Tuple[str, Dict[str, Any]]]:
        return [(n, a) for n, a in self.calls if n not in ("list_windows", "get_window_state")]


def _backend(version: str):
    from tools.computer_use.cua_backend import CuaDriverBackend
    from tools.computer_use.cua_backend_session import _CuaDriverSession

    schemas = json.loads(FIXTURE.read_text())[version]
    driver = StrictFakeDriver(schemas)
    session = _CuaDriverSession.__new__(_CuaDriverSession)  # real capability/schema lookups, fake transport
    session._tool_schemas = schemas
    session._capabilities = {name: set() for name in [*schemas, "list_windows", "get_window_state"]}
    session._started = True
    session.call_tool = driver.call_tool
    backend = CuaDriverBackend()
    backend._session = session
    return backend, driver


TOKEN_ONLY = "cua_driver_0_32_0"
LEGACY = "cua_driver_0_21_0"


def _captured(version: str):
    backend, driver = _backend(version)
    cap = backend.capture(mode="ax", app="Main.py")
    assert [e.index for e in cap.elements] == [0, 1, 2]
    return backend, driver


class TestTokenOnlyDriver:
    """cua-driver 0.32.0 / trycua main: element_token is the only element target."""

    def test_click_by_element_sends_current_token_not_index(self):
        backend, driver = _captured(TOKEN_ONLY)
        result = backend.click(element=2)
        assert result.ok, result.message
        name, args = driver.action_calls()[-1]
        assert name == "click"
        assert "element_index" not in args
        assert args["element_token"] == "s00000001:2"
        assert args["pid"] == PID

    def test_set_value_by_element_sends_current_token_not_index(self):
        backend, driver = _captured(TOKEN_ONLY)
        result = backend.set_value("abc", element=1)
        assert result.ok, result.message
        name, args = driver.action_calls()[-1]
        assert (name, args.get("element_token"), "element_index" in args) == ("set_value", "s00000001:1", False)

    def test_scroll_by_element_sends_current_token_not_index(self):
        backend, driver = _captured(TOKEN_ONLY)
        result = backend.scroll(direction="down", element=0)
        assert result.ok, result.message
        name, args = driver.action_calls()[-1]
        assert (name, args.get("element_token"), "element_index" in args) == ("scroll", "s00000001:0", False)

    def test_element_wins_over_coordinates_and_coordinates_are_not_sent(self):
        backend, driver = _captured(TOKEN_ONLY)
        assert backend.click(element=2, x=500, y=400).ok
        _, args = driver.action_calls()[-1]
        assert "x" not in args and "y" not in args and args["element_token"] == "s00000001:2"

    def test_recapture_uses_the_new_snapshot_token(self):
        backend, driver = _captured(TOKEN_ONLY)
        backend.capture(mode="ax", app="Main.py")
        assert backend.click(element=2).ok
        assert driver.action_calls()[-1][1]["element_token"] == "s00000002:2"

    def test_unknown_index_is_refused_locally_without_coordinate_fallback(self):
        backend, driver = _captured(TOKEN_ONLY)
        result = backend.click(element=9, x=500, y=400)
        assert not result.ok
        assert result.code == "element_token_unavailable"
        assert driver.action_calls() == []  # nothing reached the driver: no bare index, no pixel click

    def test_no_snapshot_means_no_element_action(self):
        backend, driver = _captured(TOKEN_ONLY)
        backend._set_active_target({"pid": PID, "window_id": WINDOW})  # e.g. focus_app: retargets, drops tokens
        result = backend.set_value("abc", element=1)
        assert not result.ok and result.code == "element_token_unavailable"
        assert driver.action_calls() == []

    def test_coordinate_click_is_unchanged(self):
        backend, driver = _captured(TOKEN_ONLY)
        assert backend.click(x=12, y=34).ok
        _, args = driver.action_calls()[-1]
        assert (args["x"], args["y"], args["window_id"]) == (12, 34, WINDOW)
        assert "element_token" not in args and "element_index" not in args


class TestLegacyDriver:
    """cua-driver 0.21.0 (Hermes' pinned release) keeps its current wire shape."""

    def test_click_by_element_keeps_index_and_attaches_token(self):
        backend, driver = _captured(LEGACY)
        assert backend.click(element=2).ok
        _, args = driver.action_calls()[-1]
        assert args["element_index"] == 2 and args["element_token"] == "s00000001:2"

    def test_set_value_by_element_keeps_index(self):
        backend, driver = _captured(LEGACY)
        assert backend.set_value("abc", element=1).ok
        _, args = driver.action_calls()[-1]
        assert args["element_index"] == 1 and args["element_token"] == "s00000001:1"


@pytest.mark.parametrize("version", [TOKEN_ONLY, LEGACY])
def test_fixture_matches_the_contract_under_test(version):
    """Guard the frozen schemas: 0.32.0 advertises element_token only, 0.21.0 both."""
    schemas = json.loads(FIXTURE.read_text())[version]
    for tool in ("click", "double_click", "scroll", "set_value"):
        props = schemas[tool]["properties"]
        assert "element_token" in props
        assert ("element_index" in props) is (version == LEGACY)
        assert schemas[tool]["additionalProperties"] is False


class TestEndedSessionRevival:
    """The Driver retires an ended session's snapshots, so a token minted under it is dead. Hermes revives the
    session and replays a rejected call once; that replay must never carry an element_token from the ended
    session (observed full-stack: the replayed click came back 'element_token is stale ... no current
    snapshot'). Token-free calls keep the existing revive-and-replay behaviour."""

    ENDED = {"data": "session hermes-label has ended; call start_session to begin a new session", "images": [],
             "structuredContent": None, "isError": True}

    def _session(self, effects):
        import threading
        from unittest.mock import MagicMock
        from tools.computer_use.cua_backend_session import _CuaDriverSession

        calls = []

        class Bridge:
            def run(self, value, timeout=None):
                calls.append(value)
                return effects.pop(0)

        s = _CuaDriverSession.__new__(_CuaDriverSession)
        s._bridge, s._session, s._lock, s._started = Bridge(), object(), threading.Lock(), True
        s._capabilities, s._capability_version, s._declared_session_id = {}, "", "hermes-label"
        s._call_tool_async = lambda name, args: ("call", name, args)
        s._transport_reset_callback = MagicMock()
        return s, calls

    def test_token_call_is_not_replayed_after_revival(self):
        ok = {"data": "ok", "images": [], "structuredContent": None, "isError": False}
        s, calls = self._session([dict(self.ENDED), dict(ok), dict(ok)])
        args = {"pid": 1, "element_token": "s00000001:2", "session": "hermes-label"}
        result = s.call_tool("click", args)
        assert calls == [("call", "click", args), ("call", "start_session", {"session": "hermes-label"})]
        assert result["isError"] is True
        assert result["structuredContent"]["code"] == "element_token_session_ended"
        s._transport_reset_callback.assert_called_once_with()  # backend drops the ended session's tokens

    def test_token_free_call_is_still_replayed_after_revival(self):
        ok = {"data": "ok", "images": [], "structuredContent": None, "isError": False}
        s, calls = self._session([dict(self.ENDED), dict(ok), dict(ok)])
        args = {"pid": 1, "window_id": 2, "x": 3, "y": 4, "session": "hermes-label"}
        result = s.call_tool("click", args)
        assert [c[1] for c in calls] == ["click", "start_session", "click"]
        assert result["isError"] is False
        s._transport_reset_callback.assert_called_once_with()  # the ended session's tokens are dropped anyway

    def test_token_free_revival_then_element_action_never_sends_a_dead_token(self):
        """The revive may be triggered by ANY call (type_text, key, a coordinate click). Every token cached from
        the ended session is dead afterwards, so the next element action must not carry one."""
        backend, driver = _captured(TOKEN_ONLY)  # snapshot 1 -> tokens s00000001:*
        fake_call, state = driver.call_tool, {"ended": False}

        class Bridge:
            def run(self, value, timeout=None):
                _, name, args = value
                if name == "start_session":
                    state["ended"] = False
                    return {"data": "ok", "images": [], "structuredContent": None, "isError": False}
                return dict(self.ended) if state["ended"] else fake_call(name, args)

        Bridge.ended = self.ENDED
        s, _ = self._session([])
        s._bridge, s._tool_schemas, s._capabilities = Bridge(), backend._session._tool_schemas, backend._session._capabilities
        s._timeout_suspect = False
        s.set_transport_reset_callback(backend._handle_transport_reset)
        backend._session = s
        state["ended"] = True  # the Driver session ends while the model is thinking
        assert backend.type_text("hello").ok  # token-free call: revive + replay once
        result = backend.click(element=2)
        assert not result.ok and "capture" in result.message  # refused locally: capture again first
        assert [a for n, a in driver.calls if n == "click"] == []  # no stale token reached the Driver

    def test_backend_requires_a_fresh_capture_after_the_ended_session(self):
        backend, driver = _captured(TOKEN_ONLY)
        backend._handle_transport_reset()  # what the session's reset callback runs
        result = backend.click(element=2)
        assert not result.ok and driver.action_calls() == []
