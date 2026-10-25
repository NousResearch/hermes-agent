import base64
import json
from io import BytesIO
from unittest.mock import Mock

import pytest
from PIL import Image

from tools.computer_use.cua_backend import CuaDriverBackend


@pytest.fixture
def backend(monkeypatch):
    backend = CuaDriverBackend()
    backend._session._capabilities = {
        "get_window_state": set(),
        "click": {"accessibility.element_tokens"},
    }
    backend._session._tool_schemas = {"click": {"properties": {"element_token": {}}}}
    monkeypatch.setattr(backend._session, "call_tool", Mock())
    monkeypatch.setattr(backend._session, "_call_tool_via_cli", Mock())
    return backend


@pytest.fixture
def dispatch(backend, monkeypatch):
    import tools.computer_use_tool  # noqa: F401 -- registers the production handler
    from tools.computer_use import tool
    from tools.registry import registry

    monkeypatch.setattr(tool, "_get_backend", lambda **kwargs: backend)

    def call(**args):
        result = registry.dispatch("computer_use", args)
        assert isinstance(result, str)
        return json.loads(result)

    return call


def _snapshot(token=None, *, image=False):
    element = {"element_index": 0, "role": "button", "label": "Counter"}
    if token:
        element["element_token"] = token
    images = []
    if image:
        buffer = BytesIO()
        Image.new("RGB", (2, 2)).save(buffer, format="PNG")
        images.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
    return {"data": "", "images": images, "structuredContent": {"elements": [element]}, "isError": False}


@pytest.mark.parametrize("mode,read_tool", [
    ("ax", "get_window_state"),
    ("som", "get_window_state"),
    ("vision", "get_window_state"),
    ("vision", "screenshot"),
])
def test_capture_adopts_target_and_snapshot_after_transport_reset(backend, dispatch, mode, read_tool):
    transport = backend._session.call_tool
    transport.return_value = _snapshot("old:0")
    assert backend.capture(mode="som", pid=101, window_id=202).elements
    assert backend.click(element=0).ok
    assert transport.call_args.args[1]["element_token"] == "old:0"

    if read_tool == "screenshot":
        backend._session._capabilities["screenshot"] = set()

    def recovered_read(name, args):
        assert name == read_tool
        assert args["window_id"] == 456
        backend._session._notify_transport_reset()
        return _snapshot("fresh:0", image=mode == "vision")

    transport.side_effect = recovered_read
    capture = backend.capture(mode=mode, app="Counter", pid=123, window_id=456)
    if mode == "vision":
        assert capture.png_b64 and not capture.elements
    else:
        assert capture.elements[0].element_token == "fresh:0"

    transport.side_effect = None
    transport.return_value = {"data": "", "structuredContent": {"ok": True}, "isError": False}
    assert dispatch(action="click", element=0, app="Counter")["ok"]
    args = transport.call_args.args[1]
    assert (args["pid"], args["window_id"]) == (123, 456)
    assert args.get("element_token") == (None if mode == "vision" else "fresh:0")
    assert backend.type_text("fresh target").ok
    assert backend.key("return").ok
    backend._session._call_tool_via_cli.assert_not_called()

    backend._session._notify_transport_reset()
    transport.reset_mock()
    assert not backend.click(element=0).ok
    assert not backend.type_text("must not land").ok
    assert not backend.key("return").ok
    transport.assert_not_called()

    transport.return_value = _snapshot()
    assert backend.capture(mode="som", pid=123, window_id=456).elements
    assert backend.click(element=0).ok
    assert "element_token" not in transport.call_args.args[1]


@pytest.mark.parametrize("screenshot_available,use_cli", [(True, False), (True, True), (False, True)])
def test_vision_fallback_preserves_target_after_imageless_reconnect(backend, screenshot_available, use_cli):
    transport = backend._session.call_tool
    if screenshot_available:
        backend._session._capabilities["screenshot"] = set()
    empty = {"data": "", "images": [], "structuredContent": {}, "isError": False}
    reads = []

    def recovered_read(name, args):
        reads.append(name)
        assert args["window_id"] == 456
        if name == "get_window_state":
            assert args["pid"] == 123
        if len(reads) == 1:
            backend._session._notify_transport_reset()
        return empty if name == "screenshot" or use_cli else _snapshot(image=True)

    def cli_read(name, args, timeout):
        assert name == "get_window_state"
        assert (args["pid"], args["window_id"]) == (123, 456)
        return _snapshot(image=True)

    transport.side_effect = recovered_read
    fallback = backend._session._call_tool_via_cli
    fallback.side_effect = cli_read
    capture = backend.capture(mode="vision", pid=123, window_id=456)

    assert capture.png_b64 and not capture.elements
    assert reads == (["screenshot", "get_window_state"] if screenshot_available else ["get_window_state"])
    assert fallback.call_count == int(use_cli)
    transport.side_effect = None
    transport.return_value = {"data": "", "structuredContent": {"ok": True}, "isError": False}
    assert backend.click(x=1, y=1).ok
    args = transport.call_args.args[1]
    assert (args["pid"], args["window_id"]) == (123, 456)


@pytest.mark.parametrize("mode", ["som", "vision"])
@pytest.mark.parametrize("fallback_failure", ["error", "exception", "empty"])
def test_failed_capture_fallback_keeps_input_disarmed(backend, dispatch, mode, fallback_failure):
    transport = backend._session.call_tool
    transport.return_value = _snapshot("old:0")
    assert backend.capture(mode="som", pid=101, window_id=202).elements
    assert backend.click(element=0).ok

    empty = {"data": "", "images": [], "structuredContent": {}, "isError": False}
    transport.return_value = empty
    fallback = backend._session._call_tool_via_cli
    fallback.return_value = empty
    if fallback_failure == "error":
        fallback.return_value = {**empty, "data": "window gone", "isError": True}
    elif fallback_failure == "exception":
        fallback.side_effect = RuntimeError("driver unavailable")

    capture = backend.capture(mode=mode, pid=123, window_id=456)

    assert not capture.elements and not capture.png_b64
    fallback.assert_called_once()
    name, args, _ = fallback.call_args.args
    assert name == "get_window_state"
    assert (args["pid"], args["window_id"]) == (123, 456)
    transport.reset_mock()
    assert not dispatch(action="click", element=0)["ok"]
    assert not backend.click(x=10, y=10).ok
    assert not backend.type_text("must not land").ok
    assert not backend.key("return").ok
    transport.assert_not_called()
