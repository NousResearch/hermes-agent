"""What input may target once launch replaces the sticky target (#111622 review F2).

Each test drives the real CuaDriverBackend (input mixin, launch_app, _action) and the real follow-up
capture against a recording driver session; nothing reaches an OS.
"""

from unittest.mock import MagicMock

import pytest

from tools.computer_use.cua_backend import CuaDriverBackend
from tools.computer_use.tool import _maybe_follow_capture

_INPUT_TOOLS = {"click", "double_click", "press_key", "hotkey", "type_text", "scroll", "drag", "set_value"}


def _backend(*, launch: dict | None = None) -> CuaDriverBackend:
    """A backend that captured App A (pid 101, window 1001, element 7 -> token-from-A)."""
    backend = CuaDriverBackend()
    session = backend._session = MagicMock()

    def call_tool(name, args):
        payload = launch if name == "launch_app" else None
        return {"data": payload or "ok", "images": [], "structuredContent": payload, "isError": False}

    session.call_tool.side_effect = call_tool
    session.supports_capability.return_value = False
    session.supports_input_property.return_value = True
    backend._set_active_target({"pid": 101, "window_id": 1001})
    backend._last_app, backend._snapshot_tokens = "App A", {7: "token-from-A"}
    return backend


def _input_calls(backend: CuaDriverBackend) -> list:
    return [c.args for c in backend._session.call_tool.call_args_list if c.args[0] in _INPUT_TOOLS]


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
