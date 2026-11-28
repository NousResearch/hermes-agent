"""Capture safety diagnostics must survive every presentation path (no real GUI)."""
from __future__ import annotations

import copy
import json
from contextlib import nullcontext
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from tools.computer_use import cua_backend, tool
from tools.computer_use.backend import ActionResult, CaptureResult

# Deliberately a fixture, not a real screenshot or live daemon result.
PNG = "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAADUlEQVR4nGNgGAUgAAABCAABgukLHQAAAABJRU5ErkJggg=="
LOCKED = {
    "degraded": True,
    "degraded_reason": "screen_locked: the screen is locked; screenshots still work",
    "background_input": {"exact_window": {"status": "screen_locked"}},
    "escalation": {"recommended": "wait_for_unlock", "reason": "Wake and unlock, then re-capture."},
}


def backend_with_result(monkeypatch, status, *, cli=False):
    backend = cast(Any, cua_backend.CuaDriverBackend())
    result = {"isError": False, "data": "", "images": [PNG],
              "structuredContent": {"elements": [], **copy.deepcopy(status), "unreviewed_extra": "DO NOT FORWARD"}}
    backend._session = MagicMock()
    backend._session._has_tool.return_value = False
    backend._session.capabilities_discovered = True
    backend._session.supports_input_value.return_value = False
    backend._session.call_tool.return_value = result if not cli else {
        "isError": False, "data": "", "images": [], "structuredContent": {}}
    backend._session._call_tool_via_cli.return_value = result
    monkeypatch.setattr(cua_backend, "_cua_configured_ax_max_elements", lambda: 0)
    return backend


@pytest.mark.parametrize("mode", ["som", "vision", "ax"])
@pytest.mark.parametrize("cli", [False, True])
def test_backend_preserves_capture_status(monkeypatch, mode, cli):
    backend = backend_with_result(monkeypatch, LOCKED, cli=cli)
    cap = backend.capture(mode, pid=1234, window_id=5678)
    for key, value in LOCKED.items():
        assert getattr(cap, key, None) == value
    assert cap.input_blocked is True
    assert not hasattr(cap, "unreviewed_extra")


def test_standalone_vision_screenshot_status_survives(monkeypatch):
    backend = backend_with_result(monkeypatch, LOCKED)
    backend._session._has_tool.return_value = True
    cap = backend.capture("vision", pid=1234, window_id=5678)
    assert getattr(cap, "degraded_reason", None) == LOCKED["degraded_reason"]
    backend._session.call_tool.assert_called_once()


@pytest.fixture
def no_image_io(monkeypatch):
    monkeypatch.setattr(tool, "_persist_capture_image", lambda cap: "/fixture/capture.png")
    monkeypatch.setattr(tool, "_spill_elements_to_file", lambda cap: "/fixture/elements.json")
    monkeypatch.setattr(tool, "_MIN_PROVIDER_IMAGE_DIMENSION", 1)
    tool._reset_screenshot_dedup()


def capture(status=LOCKED, mode="som"):
    return CaptureResult(mode=mode, width=8, height=8, png_b64=PNG, **copy.deepcopy(status))


@pytest.mark.parametrize("branch", ["text", "native", "aux", "aux_failed", "dedup"])
def test_status_is_model_visible_in_all_branches(monkeypatch, no_image_io, branch):
    cap = capture(mode="ax" if branch == "text" else "som")
    monkeypatch.setattr(tool, "_should_route_through_aux_vision", lambda: branch in {"aux", "aux_failed"})
    if branch == "aux":
        monkeypatch.setattr("model_tools._run_async", lambda coro: json.dumps({"analysis": "fixture app view"}))
        monkeypatch.setattr("tools.vision_tools.vision_analyze_tool", lambda *a: "fixture")
        monkeypatch.setattr(tool, "_write_private_bytes", lambda *a: None)
        monkeypatch.setattr(tool, "_shrink_capture_for_vision", lambda raw, ext: (raw, None))
    if branch == "aux_failed":
        monkeypatch.setattr(tool, "_route_capture_through_aux_vision", lambda *a, **kw: None)
    if branch == "dedup":
        tool._capture_response(cap, session_id="status-test")
    response = tool._capture_response(cap, session_id="status-test")
    body = json.loads(response) if isinstance(response, str) else response
    for key, value in LOCKED.items():
        assert body[key] == value
        if branch == "native":
            assert body["meta"][key] == value
    summary = body.get("summary", body.get("text_summary", ""))
    assert "screen_locked" in summary
    assert "wait_for_unlock" in summary
    assert "observation-only" in summary
    assert "Element indices below are fresh and remain the preferred way to act" not in summary
    assert "Element-index actions still work" not in summary


@pytest.mark.parametrize("args", [
    {"action": "click", "coordinate": [1, 1]}, {"action": "key", "keys": "return"},
    {"action": "type", "text": "x"}, {"action": "drag", "from_coordinate": [1, 1], "to_coordinate": [2, 2]},
    {"action": "scroll", "direction": "down"}, {"action": "set_value", "element": 1, "value": "x"},
    {"action": "focus_app", "app": "Chrome", "raise_window": True},
])
def test_locked_capture_refuses_mutations_before_backend(monkeypatch, no_image_io, args):
    backend = MagicMock()
    backend.capture.return_value = capture()
    monkeypatch.setattr(tool, "_should_route_through_aux_vision", lambda: False)
    tool._dispatch(backend, "capture", {})
    backend.reset_mock()
    response = json.loads(tool._dispatch(backend, args["action"], args))
    assert response["ok"] is False
    assert response["code"] == "screen_locked"
    assert response["escalation"]["recommended"] == "wait_for_unlock"
    assert response["verdict"]["decision"] == "wait_for_unlock"
    assert backend.mock_calls == []


def test_clean_capture_clears_lock_latch(monkeypatch, no_image_io):
    backend = MagicMock()
    backend.capture.side_effect = [capture(), capture({"background_input": {"exact_window": {"status": "matched"}}})]
    backend.click.return_value = ActionResult(ok=True, action="click")
    monkeypatch.setattr(tool, "_should_route_through_aux_vision", lambda: False)
    tool._dispatch(backend, "capture", {})
    tool._dispatch(backend, "capture", {})
    assert json.loads(tool._dispatch(backend, "click", {"coordinate": [1, 1]}))["ok"] is True
    backend.click.assert_called_once()


@pytest.mark.parametrize("native", [True, False])
def test_capture_after_keeps_action_escalation_separate(monkeypatch, no_image_io, native):
    monkeypatch.setattr(tool, "_should_route_through_aux_vision", lambda: False)
    backend = MagicMock()
    backend._last_target = {"pid": 1234, "window_id": 5678}
    backend.capture.return_value = capture(mode="som" if native else "ax")
    result = ActionResult(ok=True, action="click", effect="unverifiable",
                          escalation={"recommended": "foreground", "reason": "original action"})
    response = tool._maybe_follow_capture(backend, result, True)
    body = json.loads(response) if isinstance(response, str) else response
    action = body.get("action_result", body)
    assert action["escalation"] == result.escalation
    assert body["capture_status"]["escalation"]["recommended"] == "wait_for_unlock"


def test_normal_capture_has_no_diagnostic_noise(monkeypatch, no_image_io):
    monkeypatch.setattr(tool, "_should_route_through_aux_vision", lambda: False)
    response = tool._capture_response(capture({}))
    assert not set(LOCKED) & response.keys()
    assert not set(LOCKED) & response["meta"].keys()
    assert "observation-only" not in response["text_summary"]


def test_non_locked_degradation_remains_actionable(monkeypatch, no_image_io):
    cap = capture({"degraded": True, "degraded_reason": "ax_tree_empty: canvas",
                   "escalation": {"recommended": "px"}})
    assert cap.input_blocked is False
    monkeypatch.setattr(tool, "_should_route_through_aux_vision", lambda: False)
    response = tool._capture_response(cap)
    assert response["degraded_reason"] == "ax_tree_empty: canvas"
    assert "px" in response["text_summary"]


def test_healthy_background_report_is_retained_without_summary_noise(monkeypatch, no_image_io):
    status = {"background_input": {"exact_window": {"status": "matched"},
                                   "routes": [{"route": "pointer", "status": "available"}]}}
    monkeypatch.setattr(tool, "_should_route_through_aux_vision", lambda: False)
    response = tool._capture_response(capture(status))
    assert response["background_input"] == status["background_input"]
    assert response["meta"]["background_input"] == status["background_input"]
    assert "background_input=" not in response["text_summary"]


@pytest.mark.parametrize("mode", ["som", "ax", "vision"])
def test_fallback_cannot_erase_lock_evidence(monkeypatch, mode):
    backend = backend_with_result(monkeypatch, {})
    backend._session.call_tool.return_value = {
        "isError": False, "data": "", "images": [], "structuredContent": copy.deepcopy(LOCKED)}
    cap = backend.capture(mode, pid=1234, window_id=5678)
    assert cap.input_blocked
    assert cap.degraded_reason == LOCKED["degraded_reason"]


def test_screenshot_fallback_cannot_erase_lock_evidence(monkeypatch):
    backend = backend_with_result(monkeypatch, {})
    backend._session._has_tool.return_value = True
    backend._session.call_tool.side_effect = [
        {"isError": False, "data": "", "images": [], "structuredContent": copy.deepcopy(LOCKED)},
        {"isError": False, "data": "", "images": [PNG], "structuredContent": {}},
    ]
    assert backend.capture("vision", pid=1234, window_id=5678).input_blocked


def test_fullscreen_preserves_lock_diagnostics(monkeypatch):
    backend = backend_with_result(monkeypatch, LOCKED)
    assert backend.capture("vision", app="screen").input_blocked


def test_first_fullscreen_without_image_keeps_lock_diagnostics(monkeypatch, no_image_io):
    backend = backend_with_result(monkeypatch, LOCKED)
    backend._session.call_tool.return_value["images"] = []
    cap = backend.capture("vision", app="screen")
    assert (cap.width, cap.height) == (0, 0)
    assert cap.input_blocked
    assert cap.degraded_reason == LOCKED["degraded_reason"]
    backend.capture = MagicMock(return_value=cap)
    tool._dispatch(backend, "capture", {})
    backend.focus_app = MagicMock()
    response = json.loads(tool._dispatch(backend, "focus_app", {"app": "Chrome", "raise_window": True}))
    assert response["code"] == "screen_locked"
    backend.focus_app.assert_not_called()


@pytest.mark.parametrize("bad_cap", [
    CaptureResult(mode="som", width=0, height=0),
    CaptureResult(mode="vision", width=8, height=8, png_b64=PNG, app="screen"),
])
def test_failed_or_diagnostic_free_capture_cannot_clear_lock(monkeypatch, no_image_io, bad_cap):
    backend = MagicMock()
    backend.capture.side_effect = [capture(), bad_cap]
    monkeypatch.setattr(tool, "_should_route_through_aux_vision", lambda: False)
    tool._dispatch(backend, "capture", {})
    response = tool._dispatch(backend, "capture", {})
    body = json.loads(response) if isinstance(response, str) else response
    assert body["escalation"]["recommended"] == "wait_for_unlock"
    backend.reset_mock()
    refused = json.loads(tool._dispatch(backend, "focus_app", {"app": "Chrome", "raise_window": True}))
    assert refused["code"] == "screen_locked"
    assert backend.mock_calls == []


@pytest.mark.parametrize("mode", ["som", "ax", "vision"])
def test_successful_capture_rearms_only_fresh_target_after_transport_reset(monkeypatch, mode):
    backend = backend_with_result(monkeypatch, {"background_input": {"exact_window": {"status": "matched"}}})
    backend._session._has_tool.return_value = True
    result = backend._session.call_tool.return_value
    result["structuredContent"]["elements"] = [
        {"element_index": 1, "role": "AXButton", "label": "Fresh", "element_token": "fresh-token"}]

    def read_after_reset(name, args):
        assert args["window_id"] == 5678
        backend._handle_transport_reset()
        return result

    backend._session.call_tool.side_effect = read_after_reset
    backend._snapshot_tokens = {99: "old-token"}
    backend.capture(mode, app="Fixture", pid=1234, window_id=5678)
    assert backend._active_pid == 1234
    assert backend._active_window_id == 5678
    assert backend._last_app == "Fixture"
    assert backend._snapshot_tokens == ({} if mode == "vision" else {1: "fresh-token"})


@pytest.mark.parametrize("mode, app", [("som", None), ("ax", None), ("vision", None), ("vision", "screen")])
def test_error_capture_retains_explicit_lock_and_disarms(monkeypatch, no_image_io, mode, app):
    backend = backend_with_result(monkeypatch, LOCKED)
    backend._session.call_tool.return_value["isError"] = True
    backend._session.call_tool.return_value["data"] = "Screen is locked"
    monkeypatch.setattr(tool, "_should_route_through_aux_vision", lambda: False)
    args = {"mode": mode, "app": app} if app else {"mode": mode, "pid": 1234, "window_id": 5678}
    response = tool._dispatch(backend, "capture", args)
    body = json.loads(response) if isinstance(response, str) else response
    assert body["width"] == body["height"] == 0
    assert body["escalation"]["recommended"] == "wait_for_unlock"
    assert backend._active_pid is None
    assert backend._snapshot_tokens == {}
    refused = json.loads(tool._dispatch(backend, "key", {"keys": "return"}))
    assert refused["code"] == "screen_locked"


def test_registry_capture_diagnostics_and_input_refusal(monkeypatch, no_image_io):
    from tools import computer_use_tool
    from tools.registry import registry

    backend = MagicMock()
    backend.capture.return_value = capture()
    monkeypatch.setattr(tool, "_backend_for_call", lambda sid: nullcontext(backend))
    monkeypatch.setattr("tools.bot_desktop.runtime.ensure_started_for_tool", lambda: None)
    monkeypatch.setattr(tool, "_should_route_through_aux_vision", lambda: False)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setattr("tools.terminal_tool._get_approval_callback", lambda: lambda *a, **kw: "once")
    response = registry.dispatch("computer_use", {"action": "capture"}, session_id="locked-fixture")
    assert isinstance(response, dict)
    assert response["meta"]["escalation"]["recommended"] == "wait_for_unlock"
    assert "observation-only" in response["text_summary"]
    backend.reset_mock()
    refused = registry.dispatch("computer_use", {"action": "key", "keys": "return"}, session_id="locked-fixture")
    assert isinstance(refused, str)
    assert json.loads(refused)["code"] == "screen_locked"
    assert backend.mock_calls == []


@pytest.mark.parametrize("mode", ["som", "ax", "vision"])
def test_failed_cli_fallback_does_not_rearm_target(monkeypatch, mode):
    backend = backend_with_result(monkeypatch, {})
    backend._session.call_tool.return_value["images"] = []
    backend._session._call_tool_via_cli.return_value = {"isError": True, "data": "read failed"}
    cap = backend.capture(mode, pid=1234, window_id=5678)
    assert cap.png_b64 is None and cap.elements == []
    assert backend._active_pid is None
    assert backend._active_window_id is None
    assert backend._snapshot_tokens == {}
