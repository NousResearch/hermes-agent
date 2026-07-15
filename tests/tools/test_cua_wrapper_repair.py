"""Focused regressions for the HARNESS-CUA1 wrapper repair.

These tests mock only the MCP client shim.  They never start or modify the
privileged ``cua-driver serve`` daemon.
"""

from unittest.mock import MagicMock

import pytest

from tools.computer_use.cua_backend import (
    CuaDriverBackend,
    _CuaDriverSession,
    _ingest_windows,
    _select_app_windows,
    _window_rank,
)


def test_list_apps_prefers_structured_content():
    backend = CuaDriverBackend()
    backend._session = MagicMock()
    backend._session.call_tool.return_value = {
        "structuredContent": {"apps": [{"name": "Safari", "pid": 7}]},
        "data": "- Wrong (pid 9)",
    }
    assert backend.list_apps() == [{"name": "Safari", "pid": 7}]


@pytest.mark.parametrize("bullet", ["- ", "* ", "+ ", "1. "])
def test_list_apps_text_fallback_strips_markdown_bullets(bullet):
    backend = CuaDriverBackend()
    backend._session = MagicMock()
    backend._session.call_tool.return_value = {
        "structuredContent": None, "data": f"{bullet}Safari (pid 7)"
    }
    assert backend.list_apps() == [{"name": "Safari", "pid": 7}]


def _windows():
    return _ingest_windows([
        {"app_name": "Safari", "bundle_id": "com.apple.Safari", "pid": 7,
         "window_id": 1, "width": 800, "height": 600},
        {"app_name": "Brave Browser", "bundle_id": "com.brave.Browser", "pid": 9,
         "window_id": 2, "width": 800, "height": 600},
    ])


@pytest.mark.parametrize(("selector", "pid"), [
    ("Safari", 7), ("Brave Browser", 9), ("com.brave.Browser", 9), ("7", 7),
])
def test_selector_accepts_exact_name_bundle_and_pid(selector, pid):
    assert {w["pid"] for w in _select_app_windows(selector, _windows())} == {pid}


def test_selector_rejects_substrings_and_ambiguous_pid():
    assert _select_app_windows("Brave", _windows()) == []
    windows = _windows() + _ingest_windows([
        {"app_name": "Safari Technology Preview", "pid": 7, "window_id": 3}
    ])
    with pytest.raises(ValueError, match="Ambiguous"):
        _select_app_windows("7", windows)


def test_window_rank_prefers_contentful_visible_non_minimized_window():
    candidates = _ingest_windows([
        {"app_name": "Safari", "pid": 7, "window_id": 1, "z_index": 0,
         "width": 1200, "height": 0},
        {"app_name": "Safari", "pid": 7, "window_id": 2, "z_index": 1,
         "width": 1200, "height": 800},
        {"app_name": "Safari", "pid": 7, "window_id": 3, "z_index": 2,
         "width": 1400, "height": 900, "is_minimized": True},
    ])
    assert min(candidates, key=_window_rank)["window_id"] == 2


def test_reconnect_redeclares_session_before_retry():
    session = object.__new__(_CuaDriverSession)
    session._lock = __import__("threading").Lock()
    session._declared_session_id = "hermes-test"
    session._started = True
    session._capabilities = {}
    session._capability_version = ""
    session._bridge = MagicMock()
    session._stop_lifecycle_locked = MagicMock()
    session._start_lifecycle_locked = MagicMock()
    session._require_started = lambda: None
    session._call_tool_async = MagicMock(side_effect=lambda name, args: (name, args))
    session._bridge.run.side_effect = [BrokenPipeError(), {}, {"data": "ok"}]
    assert session.call_tool("list_apps", {}) == {"data": "ok"}
    calls = session._call_tool_async.call_args_list
    assert calls[1].args == ("start_session", {"session": "hermes-test"})
    assert calls[2].args == ("list_apps", {})


def test_tombstoned_session_is_revived_and_retried_once():
    session = object.__new__(_CuaDriverSession)
    session._lock = __import__("threading").Lock()
    session._declared_session_id = "hermes-test"
    session._started = True
    session._bridge = MagicMock()
    session._require_started = lambda: None
    session._call_tool_async = MagicMock(side_effect=lambda name, args: (name, args))
    session._bridge.run.side_effect = [RuntimeError("session tombstoned"), {}, {"ok": True}]
    assert session.call_tool("click", {}) == {"ok": True}
    assert session._bridge.run.call_count == 3


def test_empty_inventory_is_transport_error_but_no_match_is_typed_result():
    backend = CuaDriverBackend()
    backend._session = MagicMock()
    # Successful empty inventory over both transports = legitimate empty desktop.
    backend._session.call_tool.return_value = {"structuredContent": {"windows": []}}
    backend._session._call_tool_via_cli.return_value = {"structuredContent": {"windows": []}}
    empty = backend.capture(app="Safari")
    assert empty.width == 0
    assert "empty window inventory" in empty.window_title

    # Empty MCP + CLI transport failure = unhealthy.
    backend._session._call_tool_via_cli.side_effect = RuntimeError("cli down")
    with pytest.raises(RuntimeError, match="unhealthy"):
        backend.capture(app="Safari")
    backend._session._call_tool_via_cli.side_effect = None
    backend._session._call_tool_via_cli.return_value = {"structuredContent": {"windows": []}}

    backend._session.call_tool.return_value = {"structuredContent": {"windows": [
        {"app_name": "Safari", "pid": 7, "window_id": 1}
    ]}}
    result = backend.capture(app="Brave Browser")
    assert result.width == 0
    assert "no on-screen window matched" in result.window_title
