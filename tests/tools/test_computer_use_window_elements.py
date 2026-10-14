"""Negotiate Cua 0.35's structured window view (Refs #47072, Surface 2)."""
import json
from pathlib import Path

import pytest

from tools.computer_use import cua_backend, cua_backend_session


_FORMAT = {"type": "string", "enum": ["markdown", "elements", "both"]}
_WINDOW = {"element_index": 0, "role": "AXWindow", "label": "Fixture",
           "element_token": "s1234abcd:0", "frame": {"x": 0, "y": 0, "w": 800, "h": 600}}
_BUTTON = {"element_index": 7, "role": "AXButton", "label": "Increment",
           "element_token": "s1234abcd:7", "frame": {"x": 10, "y": 20, "w": 30, "h": 40}}
_LEGACY = {"pid": 101, "window_id": 202, "session": "test-session", "max_elements": 200}


def _backend(monkeypatch):
    backend = cua_backend.CuaDriverBackend()
    backend._active_pid, backend._active_window_id = 101, 202
    backend._session_id = "test-session"
    backend._session._tool_schemas = {"get_window_state": {"properties": {"tree_format": _FORMAT}}}
    backend._session._capabilities = {"get_window_state": set()}
    monkeypatch.setattr(cua_backend, "_cua_configured_ax_max_elements", lambda: 200)
    return backend


def _elements_result():
    # Cua 0.35.3 window_state_view::apply removes tree_markdown AND the text tree in elements mode.
    return {"data": 'window_id=202 pid=101 elements=2 snapshot_id=s1234abcd\n\n'
                    '(tree is in structuredContent.elements; pass tree_format "markdown" for the compact text rendering)',
            "structuredContent": {"tree_format": "elements", "snapshot_id": "s1234abcd",
                                  "elements": [_WINDOW, _BUTTON]}, "images": ["cG5n"]}


@pytest.mark.parametrize(("schema", "supported"), [
    (None, False),
    ({}, False),
    ({"properties": []}, False),
    ({"properties": {"tree_format": _FORMAT}}, True),
    ({"properties": {"tree_format": {"type": "string"}}}, True),
    ({"properties": {"tree_format": {"enum": ["markdown", "both"]}}}, False),
    ({"properties": {"tree_format": {"type": "integer"}}}, False),
    ({"properties": {"tree_format": {"enum": "elements"}}}, False),
    ({"properties": {"tree_format": None}}, False),
    ({"properties": {"tree_format": {"$ref": "https://invalid.example/schema"}}}, False),
])
def test_window_args_negotiate_only_an_accepted_value(monkeypatch, schema, supported):
    backend = _backend(monkeypatch)
    remote_reads = []
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: remote_reads.append(args))
    backend._session._tool_schemas = {"get_window_state": schema}
    expected = dict(_LEGACY)
    if supported:
        expected["tree_format"] = "elements"
    # In particular, old/malformed/rejecting schemas keep the serialized legacy payload unchanged.
    assert json.dumps(backend._gws_args()) == json.dumps(expected)
    assert remote_reads == []


@pytest.mark.parametrize(("mode", "metadata"), [("ax", {"window_title": "WindowServer title"}), ("som", {})])
def test_elements_only_capture_keeps_title_frames_and_driver_tokens(monkeypatch, mode, metadata):
    backend = _backend(monkeypatch)
    response = _elements_result()
    response["structuredContent"].update(metadata)
    monkeypatch.setattr(backend._session, "call_tool", lambda name, args: response)
    cap = backend.capture(mode, pid=101, window_id=202)
    assert cap.window_title == metadata.get("window_title", _WINDOW["label"])
    assert [(e.index, e.bounds, e.element_token) for e in cap.elements] == [
        (0, (0, 0, 800, 600), _WINDOW["element_token"]),
        (7, (10, 20, 30, 40), _BUTTON["element_token"])]
    assert backend._snapshot_tokens == {0: _WINDOW["element_token"], 7: _BUTTON["element_token"]}
    assert cap.png_b64 == response["images"][0]


@pytest.mark.parametrize("mode", ["ax", "som", "vision"])
def test_mcp_and_cli_refetch_choose_the_same_view(monkeypatch, mode):
    backend = _backend(monkeypatch)
    backend._session._embedded_daemon = None
    requests, shot_files = [], []
    monkeypatch.setattr(backend._session, "call_tool", lambda name, args: requests.append(dict(args)) or {})
    monkeypatch.setattr("tools.computer_use.cua_backend_driver.resolve_cua_driver_cmd", lambda: "unused-driver")

    def cli_json(cmd, env, name, timeout):
        assert cmd[:3] == ["unused-driver", "call", "get_window_state"]
        args = json.loads(cmd[3])
        shot = Path(args.pop("screenshot_out_file"))
        assert shot.is_file()
        shot_files.append(shot)
        requests.append(args)
        return {**_elements_result()["structuredContent"], "screenshot_png_b64": "cG5n"}

    monkeypatch.setattr(cua_backend_session, "_cli_run_json", cli_json)
    cap = backend.capture(mode, pid=101, window_id=202)
    expected = dict(_LEGACY)
    if mode != "vision":
        expected["tree_format"] = "elements"
        assert cap.elements[1].element_token == _BUTTON["element_token"]
    else:
        assert cap.elements == [] and backend._snapshot_tokens == {}
    assert requests == [expected, expected]
    assert cap.png_b64 == "cG5n"
    assert all(not shot.exists() for shot in shot_files)


def test_vision_ignores_elements_without_cli_refetch(monkeypatch):
    backend = _backend(monkeypatch)
    requests = []
    monkeypatch.setattr(backend._session, "call_tool", lambda name, args: requests.append(dict(args)) or _elements_result())
    cap = backend.capture("vision", pid=101, window_id=202)
    assert requests == [_LEGACY]
    assert cap.png_b64 == "cG5n" and cap.elements == [] and backend._snapshot_tokens == {}
