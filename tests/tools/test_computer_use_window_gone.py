"""Closed-window recovery for #133006; prior art: #85627. Fakes only."""

import base64
import json
from unittest.mock import patch

import pytest

from tools.computer_use.backend import ActionResult
from tools.computer_use.cua_backend import CuaDriverBackend
from tools.computer_use.tool import _classify_action_result, _dispatch, _maybe_follow_capture, _text_response


PID, OLD, NEW = 4242, 7, 8
PNG = base64.b64encode(
    b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + (32).to_bytes(4, "big") * 2
).decode()
DRIVER_MESSAGES = ["Window 7 IS STALE OR NO LONGER RUNNING", "No window with window_id 7"]


def _window(pid=PID, window_id=NEW, app_name="Fixture"):
    return {"pid": pid, "window_id": window_id, "app_name": app_name, "title": "",
            "off_screen": False, "z_index": 0}


class _FakeSession:
    capabilities_discovered = True

    def __init__(self, *, code="window_not_found", windows=None, lookup_error=False,
                 click_ok=False, message="", capture_message=""):
        self.code, self.message = code, message
        self.windows = [_window()] if windows is None else windows
        self.lookup_error, self.click_ok = lookup_error, click_ok
        self.capture_message = capture_message
        self.calls = []
        self.cli_calls = []

    def call_tool(self, name, args, timeout=30.0):
        self.calls.append((name, dict(args)))
        if name == "list_windows":
            if self.lookup_error:
                raise RuntimeError("enumeration unavailable")
            return {"isError": False, "data": "windows",
                    "structuredContent": {"windows": self.windows}}
        if name == "get_window_state":
            if args["window_id"] == OLD:
                return {"isError": True,
                        "data": self.capture_message or
                        f"window_id {args['window_id']} is not a live window (closed, or the id is stale)"}
            return {"isError": False, "data": "",
                    "images": [PNG], "structuredContent": {"elements": []}}
        assert name in ("click", "bring_to_front", "type_text")
        return {"isError": not self.click_ok, "data": self.message,
                "structuredContent": {"code": self.code} if self.code and not self.click_ok else {}}

    def supports_input_property(self, *args):
        return False

    def supports_capability(self, *args, **kwargs):
        return False

    def _has_tool(self, name):
        return False

    def _call_tool_via_cli(self, name, args, timeout):
        self.cli_calls.append((name, dict(args)))
        return {"isError": False, "data": "", "structuredContent": {"elements": []}}


def _backend(**kwargs):
    backend = CuaDriverBackend.__new__(CuaDriverBackend)
    backend._session = _FakeSession(**kwargs)
    backend._session_id = "window-gone-fixture"
    backend._set_active_target(_window(window_id=OLD))
    backend._last_app = "Fixture"
    backend._snapshot_tokens = {1: "old-snapshot"}
    return backend


@pytest.mark.parametrize("code", ["window_not_found", "window_id_not_found", "window_target_not_found",
                                  "px_window_not_found", "bring_to_front_window_not_found"])
def test_window_gone_verdict_requests_capture(code):
    verdict = _classify_action_result(ActionResult(ok=False, action="click", code=code))
    assert verdict["decision"] == "recapture"
    assert "capture" in verdict["hint"]
    assert "delivery_mode" not in verdict["hint"]


@pytest.mark.parametrize("kwargs,candidate", [
    ({}, NEW),
    ({"windows": [_window(pid=9999)]}, None),
    ({"windows": [_window(), _window(window_id=9)]}, None),
    ({"lookup_error": True}, None),
    ({"code": None, "message": f"WindowServer has no record of window {OLD} (closed or stale)"}, NEW),
    *[({"code": None, "message": message}, NEW) for message in DRIVER_MESSAGES],
])
def test_dead_target_is_disarmed_without_replay(kwargs, candidate):
    backend = _backend(**kwargs)
    result = backend.click(x=10, y=20)
    assert not result.ok
    assert backend._active_pid is None
    assert backend._active_window_id is None
    assert backend._last_target is None
    assert backend._snapshot_tokens == {}
    assert result.meta["window_changed"] == {"previous_window_id": OLD, "candidate_window_id": candidate}
    assert [name for name, _ in backend._session.calls] == ["click", "list_windows"]
    payload = json.loads(_text_response(result))
    assert payload["meta"]["window_changed"] == result.meta["window_changed"]
    assert payload["verdict"]["decision"] == "recapture"
    if candidate:
        assert f"pid={PID}" in payload["verdict"]["hint"]
        assert f"window_id={NEW}" in payload["verdict"]["hint"]
    calls = list(backend._session.calls)
    refused = backend.click(x=10, y=20)
    assert not refused.ok and "No active window" in refused.message
    assert backend._session.calls == calls


def test_explicit_other_window_failure_keeps_sticky_target():
    backend = _backend()
    result = backend.bring_to_front(pid=PID, window_id=99)
    assert not result.ok
    assert (backend._active_pid, backend._active_window_id) == (PID, OLD)
    assert "window_changed" not in result.meta
    assert [name for name, _ in backend._session.calls] == ["bring_to_front"]


def test_pid_only_failure_disarms_sticky_target():
    backend = _backend(code="window_target_not_found")
    result = backend.type_text("fixture")
    assert not result.ok
    assert backend._active_pid is None
    assert result.meta["window_changed"]["candidate_window_id"] == NEW
    assert [name for name, _ in backend._session.calls] == ["type_text", "list_windows"]


@pytest.mark.parametrize("message", ["", *DRIVER_MESSAGES])
def test_follow_capture_names_candidate_without_capturing_or_arming_it(message):
    backend = _backend(click_ok=True, capture_message=message,
                       windows=[_window(window_id=OLD), _window(), _window(pid=9999)])
    with patch("tools.computer_use.tool._capture_after_mode", return_value="som"), \
         patch("tools.computer_use.tool._capture_response", return_value='{"elements": [{"index": 1}]}') as render:
        response = json.loads(_dispatch(backend, "click", {"coordinate": [10, 20], "capture_after": True}))
    assert response["ok"] is True
    assert response["follow_up_capture"] == {
        "ok": False, "reason": "window_changed", "candidate_window_id": NEW,
        "hint": f"the window closed; take a new capture(pid={PID}, window_id={NEW})"}
    assert backend._active_pid is None
    assert backend._active_window_id is None
    assert backend._last_target is None
    assert backend._last_app is None
    assert backend._snapshot_tokens == {}
    assert [name for name, _ in backend._session.calls] == ["click", "get_window_state", "list_windows"]
    assert backend._session.calls[1][1]["window_id"] == OLD
    assert backend._session.cli_calls == []
    render.assert_not_called()
    calls = list(backend._session.calls)
    refused = json.loads(_dispatch(backend, "click", {"coordinate": [10, 20]}))
    assert not refused["ok"] and "No active window" in refused["message"]
    assert backend._session.calls == calls


@pytest.mark.parametrize("kwargs", [
    {"windows": []},
    {"windows": [_window(window_id=OLD), _window(pid=9999)]},
    {"windows": [_window(), _window(window_id=9)]},
    {"lookup_error": True},
])
def test_follow_capture_without_sole_candidate_clears_target(kwargs):
    backend = _backend(click_ok=True, **kwargs)
    result = backend.click(x=10, y=20)
    with patch("tools.computer_use.tool._capture_after_mode", return_value="som"):
        response = json.loads(_maybe_follow_capture(backend, result, True))
    assert response["ok"] is True
    assert response["follow_up_capture"] == {
        "ok": False, "reason": "window_changed", "candidate_window_id": None,
        "hint": "the window closed; take a new capture"}
    assert backend._active_pid is None
    assert backend._active_window_id is None
    assert backend._last_target is None
    assert backend._last_app is None
    assert backend._snapshot_tokens == {}
    assert [name for name, _ in backend._session.calls] == ["click", "get_window_state", "list_windows"]
    assert backend._session.calls[1][1]["window_id"] == OLD
    assert all(name == "list_windows" for name, _ in backend._session.cli_calls)
    calls = list(backend._session.calls)
    refused = backend.click(x=10, y=20)
    assert not refused.ok and "No active window" in refused.message
    assert backend._session.calls == calls


def test_other_capture_failure_returns_bare_action_result():
    backend = _backend(click_ok=True, capture_message="capture permission denied")
    result = backend.click(x=10, y=20)
    with patch("tools.computer_use.tool._capture_after_mode", return_value="som"):
        response = _maybe_follow_capture(backend, result, True)
    assert response == _text_response(result)
    assert [name for name, _ in backend._session.calls] == ["click", "get_window_state"]


@pytest.mark.parametrize("code", ["stale", "off_space_or_ax_unresolved", None])
def test_other_failures_keep_escalation(code):
    assert _classify_action_result(ActionResult(ok=False, action="click", code=code))["decision"] == "escalate"
