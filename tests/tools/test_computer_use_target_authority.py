"""What input may target once the sticky target is gone or replaced (#111622 review F1, F2).

Each test drives the real CuaDriverBackend (input mixin, launch_app, _action) and the real follow-up
capture against a recording driver session; nothing reaches an OS.
"""

import json
from unittest.mock import MagicMock

import pytest

from tools.computer_use.cua_backend import CuaDriverBackend
from tools.computer_use.tool import _dispatch, _maybe_follow_capture

_INPUT_TOOLS = {"click", "double_click", "press_key", "hotkey", "type_text", "scroll", "drag", "set_value"}


def _backend(*, launch: dict | None = None, scope_in_schema: bool = True) -> CuaDriverBackend:
    """A backend that captured App A (pid 101, window 1001, element 7 -> token-from-A)."""
    backend = CuaDriverBackend()
    session = backend._session = MagicMock()

    def call_tool(name, args):
        payload = launch if name == "launch_app" else None
        return {"data": payload or "ok", "images": [], "structuredContent": payload, "isError": False}

    session.call_tool.side_effect = call_tool
    session.supports_capability.return_value = False
    session.supports_input_property.side_effect = lambda tool, prop: prop != "scope" or scope_in_schema
    backend._set_active_target({"pid": 101, "window_id": 1001})
    backend._last_app, backend._snapshot_tokens = "App A", {7: "token-from-A"}
    return backend


def _input_calls(backend: CuaDriverBackend) -> list:
    return [c.args for c in backend._session.call_tool.call_args_list if c.args[0] in _INPUT_TOOLS]


def _fail_capture(backend: CuaDriverBackend) -> None:
    backend.capture(mode="som", app="Ghost")  # the driver lists no such window: the lookup clears the target


@pytest.mark.parametrize("lose_target", [
    pytest.param(lambda b: b._clear_active_target(), id="never-captured"),
    pytest.param(_fail_capture, id="failed-capture"),
    pytest.param(lambda b: b._handle_transport_reset(), id="transport-reset"),
])
@pytest.mark.parametrize("keys", ["escape", "win+r"])
def test_key_without_a_target_refuses_instead_of_widening_to_the_desktop(lose_target, keys):
    backend = _backend()
    lose_target(backend)

    result = backend.key(keys)

    assert not result.ok
    assert _input_calls(backend) == []


def test_desktop_scope_is_explicit_and_needs_the_live_schema():
    backend = _backend()
    backend.key("ctrl+c")
    assert _input_calls(backend)[-1] == ("hotkey", {"pid": 101, "window_id": 1001, "keys": ["ctrl", "c"],
                                                    "session": backend._session_id})

    backend._clear_active_target()
    out = json.loads(_dispatch(backend, "key", {"keys": "win+r", "scope": "desktop"}))
    assert out["ok"] is True
    assert _input_calls(backend)[-1] == ("hotkey", {"scope": "desktop", "keys": ["win", "r"],
                                                    "session": backend._session_id})

    old_driver = _backend(scope_in_schema=False)
    old_driver._clear_active_target()
    refused = old_driver.key("escape", scope="desktop")
    assert (refused.ok, refused.code) == (False, "desktop_scope_unsupported")
    assert _input_calls(old_driver) == []


@pytest.mark.parametrize("windows", [
    pytest.param({"windows": []}, id="empty"),
    pytest.param({}, id="missing"),
    pytest.param({"windows": [{}]}, id="no-window-id"),
    pytest.param({"windows": [{"window_id": 0}]}, id="placeholder-window-id"),
])
def test_launch_without_a_usable_window_drops_the_whole_previous_target(windows):
    backend = _backend(launch={"pid": 202, "name": "App B", **windows})
    launched = backend.launch_app("App B")
    assert launched.ok

    clicked = backend.click(element=7)
    assert not clicked.ok
    assert _input_calls(backend) == []

    backend.capture = MagicMock(side_effect=RuntimeError("recorded"))
    _maybe_follow_capture(backend, launched, True)
    assert backend.capture.call_args.kwargs == {"mode": "som", "app": "App B"}


def test_launch_with_an_exact_window_retargets_input_and_follow_up_capture():
    backend = _backend(launch={"pid": 202, "name": "App B", "windows": [{"window_id": 2002}]})
    launched = backend.launch_app("App B")
    assert launched.ok

    backend.click(element=7)
    assert _input_calls(backend)[-1] == ("click", {"pid": 202, "window_id": 2002, "element_index": 7,
                                                   "button": "left", "session": backend._session_id})

    backend.capture = MagicMock(side_effect=RuntimeError("recorded"))
    _maybe_follow_capture(backend, launched, True)
    assert backend.capture.call_args.kwargs == {"mode": "som", "pid": 202, "window_id": 2002}
