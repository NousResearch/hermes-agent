"""Regression tests for #32766 — cua_backend capture() fallback when on_screen_only returns empty."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_window(app_name: str, pid: int, window_id: int,
                 is_on_screen: bool = True, title: str = "",
                 z_index: int = 0) -> dict[str, Any]:
    return {
        "app_name": app_name,
        "pid": pid,
        "window_id": window_id,
        "is_on_screen": is_on_screen,
        "title": title,
        "z_index": z_index,
    }


def _make_list_windows_response(windows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a structuredContent response as returned by cua-driver."""
    return {
        "structuredContent": {"windows": windows},
        "data": "",
    }


@pytest.fixture
def mock_session():
    """Return a MagicMock that behaves like _CuaDriverSession."""
    return MagicMock()


@pytest.fixture
def backend(mock_session):
    """Create a CuaBackend with a mocked session."""
    from tools.computer_use.cua_backend import CuaDriverBackend
    b = CuaDriverBackend.__new__(CuaDriverBackend)
    b._session = mock_session
    b._active_pid = None
    b._active_window_id = None
    b._last_app = None
    return b


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCaptureFallbackWhenOnScreenOnlyEmpty:
    """When list_windows(on_screen_only=True) returns zero windows, capture()
    must retry without the filter rather than giving up immediately."""

    def test_fallback_to_unfiltered_when_on_screen_only_empty(self, backend, mock_session):
        """Primary path empty → fallback succeeds → capture uses fallback windows."""
        # First call (on_screen_only=True) → empty
        # Second call (on_screen_only=False) → has windows
        def call_tool(tool_name: str, args: dict):
            if tool_name == "list_windows":
                if args.get("on_screen_only") is True:
                    return _make_list_windows_response([])
                else:
                    return _make_list_windows_response([
                        _make_window("Terminal", 100, 1, is_on_screen=True,
                                     title="bash", z_index=0),
                    ])
            if tool_name == "screenshot":
                return {"images": []}
            if tool_name == "get_window_state":
                return {"data": "No elements", "images": []}
            return {}

        mock_session.call_tool.side_effect = call_tool

        result = backend.capture(mode="vision")

        # Should have called list_windows twice
        assert mock_session.call_tool.call_count >= 2
        calls = mock_session.call_tool.call_args_list
        first_call = calls[0]
        assert first_call[0] == ("list_windows", {"on_screen_only": True})
        second_call = calls[1]
        assert second_call[0] == ("list_windows", {"on_screen_only": False})

        # Should have found the fallback window
        assert result.app == "Terminal"

    def test_fallback_also_empty_returns_empty_result(self, backend, mock_session):
        """Both on_screen_only=True and fallback return empty → graceful empty result."""
        mock_session.call_tool.return_value = _make_list_windows_response([])

        result = backend.capture(mode="vision")

        assert result.width == 0
        assert result.height == 0
        assert result.png_b64 is None
        assert result.app == ""

    def test_fallback_respects_app_filter(self, backend, mock_session):
        """Fallback windows are still filtered by app name."""
        def call_tool(tool_name: str, args: dict):
            if tool_name == "list_windows":
                if args.get("on_screen_only") is True:
                    return _make_list_windows_response([])
                else:
                    return _make_list_windows_response([
                        _make_window("Terminal", 100, 1, is_on_screen=True,
                                     title="bash", z_index=0),
                        _make_window("Chrome", 200, 2, is_on_screen=True,
                                     title="Google", z_index=1),
                    ])
            if tool_name == "screenshot":
                return {"images": []}
            if tool_name == "get_window_state":
                return {"data": "No elements", "images": []}
            return {}

        mock_session.call_tool.side_effect = call_tool

        result = backend.capture(mode="vision", app="Chrome")

        assert result.app == "Chrome"

    def test_fallback_applies_client_side_on_screen_filter(self, backend, mock_session):
        """Fallback windows respect is_on_screen for picking the target."""
        def call_tool(tool_name: str, args: dict):
            if tool_name == "list_windows":
                if args.get("on_screen_only") is True:
                    return _make_list_windows_response([])
                else:
                    return _make_list_windows_response([
                        _make_window("Hidden", 100, 1, is_on_screen=False,
                                     title="bg", z_index=0),
                        _make_window("Visible", 200, 2, is_on_screen=True,
                                     title="fg", z_index=1),
                    ])
            if tool_name == "screenshot":
                return {"images": []}
            if tool_name == "get_window_state":
                return {"data": "No elements", "images": []}
            return {}

        mock_session.call_tool.side_effect = call_tool

        result = backend.capture(mode="vision")

        # Should prefer the on-screen window from the fallback
        assert result.app == "Visible"
