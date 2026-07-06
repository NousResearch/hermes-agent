"""Regression for the X11 null-PID `list_windows` crash.

On X11 a window's PID comes from the *optional* ``_NET_WM_PID`` property, so
the cua-driver legitimately reports ``pid: null`` for windows that don't set
it (the desktop root, panels, override-redirect popups, …). Both
``capture()`` and ``focus_app()`` previously coerced *every* window's pid via
``int(w["pid"])`` inside a list comprehension, so a single null-pid window
raised::

    TypeError: int() argument must be a string, a bytes-like object or a
    real number, not 'NoneType'

…aborting the whole enumeration before any screenshot was taken — i.e. the
agent could never capture the screen at all on an X11 desktop that had even
one such window.

The fix routes both ingestion sites through ``_ingest_windows``, which keeps
windows that have a usable ``window_id`` even when their PID is null. The
backend uses cua-driver's ``pid=0`` fallback when a later call requires an
integer pid.
"""

from __future__ import annotations

import base64
from unittest.mock import MagicMock

# 8×8 transparent PNG — decodes cleanly so capture() can size it.
_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAADUlEQVR4nG"
    "NgGAUgAAABCAABgukLHQAAAABJRU5ErkJggg=="
)


# ---------------------------------------------------------------------------
# _ingest_windows: the fix locus (pure function, no session needed)
# ---------------------------------------------------------------------------

class TestIngestWindows:
    def test_keeps_window_with_null_pid_when_window_id_is_present(self):
        from tools.computer_use.cua_backend import _ingest_windows

        raw = [
            {"app_name": "Desktop", "pid": None, "window_id": 1, "z_index": 0},
            {"app_name": "Firefox", "pid": 4321, "window_id": 77, "z_index": 1},
        ]

        out = _ingest_windows(raw)

        assert [w["app_name"] for w in out] == ["Desktop", "Firefox"]
        assert out[0]["pid"] is None
        assert out[0]["window_id"] == 1
        assert out[1]["pid"] == 4321
        assert out[1]["window_id"] == 77

    def test_skips_window_with_null_window_id(self):
        from tools.computer_use.cua_backend import _ingest_windows

        raw = [
            {"app_name": "Panel", "pid": 10, "window_id": None, "z_index": 0},
            {"app_name": "Firefox", "pid": 4321, "window_id": 77, "z_index": 1},
        ]

        out = _ingest_windows(raw)

        assert [w["app_name"] for w in out] == ["Firefox"]

    def test_coerces_numeric_strings_like_the_original_int_call(self):
        # The original `int(w["pid"])` accepted numeric strings; preserve that.
        from tools.computer_use.cua_backend import _ingest_windows

        out = _ingest_windows(
            [{"app_name": "Term", "pid": "200", "window_id": "9", "z_index": 0}]
        )

        assert out[0]["pid"] == 200
        assert out[0]["window_id"] == 9
        assert isinstance(out[0]["pid"], int)

    def test_preserves_fields_capture_relies_on(self):
        from tools.computer_use.cua_backend import _ingest_windows

        out = _ingest_windows([
            {
                "app_name": "Firefox",
                "pid": 1,
                "window_id": 2,
                "is_on_screen": False,
                "title": "Mozilla Firefox",
                "z_index": 3,
            }
        ])

        w = out[0]
        assert w["off_screen"] is True          # derived from is_on_screen
        assert w["title"] == "Mozilla Firefox"
        assert w["z_index"] == 3


class TestSelectWindow:
    def test_matches_app_or_window_title_substring(self):
        from tools.computer_use.cua_backend import _select_window

        windows = _ingest_fixture_windows()

        assert _select_window(windows, app="chromium")["pid"] == 10
        assert _select_window(windows, window_title="Crash")["pid"] == 10
        assert _select_window(windows, app="python3", window_title="Hard Case")["pid"] == 20

    def test_exact_pid_and_window_id_target_wins(self):
        from tools.computer_use.cua_backend import _select_window

        windows = _ingest_fixture_windows()

        assert _select_window(windows, pid=20)["app_name"] == "python3"
        assert _select_window(windows, window_id=100)["app_name"] == "Chromium"
        assert _select_window(windows, pid=20, window_id=200)["title"] == "CUA Hard Case Test"
        assert _select_window(windows, pid=20, window_id=999) is None

    def test_skips_known_zero_size_shell_frames_when_no_filter(self):
        from tools.computer_use.cua_backend import _select_window

        windows = _ingest_fixture_windows()

        assert _select_window(windows)["app_name"] == "Chromium"


def _ingest_fixture_windows():
    from tools.computer_use.cua_backend import _ingest_windows

    return _ingest_windows([
        {
            "app_name": "mutter-x11-frames",
            "pid": 1,
            "window_id": 1,
            "is_on_screen": True,
            "title": "",
            "z_index": 0,
            "bounds": {"x": 0, "y": 0, "w": 0, "h": 0},
        },
        {
            "app_name": "Chromium",
            "pid": 10,
            "window_id": 100,
            "is_on_screen": True,
            "title": "Chromium Crash Dialog",
            "z_index": 1,
            "bounds": {"x": 10, "y": 10, "w": 900, "h": 700},
        },
        {
            "app_name": "python3",
            "pid": 20,
            "window_id": 200,
            "is_on_screen": True,
            "title": "CUA Hard Case Test",
            "z_index": 2,
            "bounds": {"x": 20, "y": 20, "w": 400, "h": 300},
        },
    ])


# ---------------------------------------------------------------------------
# capture(): end-to-end proof the null-pid window no longer crashes capture
# ---------------------------------------------------------------------------

def _backend_with_windows(raw_windows):
    """A CuaDriverBackend whose session returns `raw_windows` from
    list_windows and a valid PNG from screenshot."""
    from tools.computer_use.cua_backend import CuaDriverBackend

    backend = CuaDriverBackend()
    session = MagicMock()
    session.capabilities_discovered = True
    session._has_tool.return_value = True

    def _call_tool(name, args, *a, **k):
        if name == "list_windows":
            return {"structuredContent": {"windows": raw_windows}}
        if name == "screenshot":
            return {
                "structuredContent": {
                    "screenshot_png_b64": _PNG_B64,
                    "screenshot_mime_type": "image/png",
                }
            }
        return {}

    session.call_tool.side_effect = _call_tool
    backend._session = session
    return backend


def test_capture_vision_survives_null_pid_window():
    raw = [
        {"app_name": "Desktop", "pid": None, "window_id": 1, "z_index": 0},
        {"app_name": "Firefox", "pid": 4321, "window_id": 77,
         "is_on_screen": True, "title": "Mozilla Firefox", "z_index": 1},
    ]
    backend = _backend_with_windows(raw)

    cap = backend.capture(mode="vision", app="Firefox")

    # The real named window is selected rather than the whole capture crashing
    # on the null-pid desktop window.
    assert cap.app == "Firefox"
    assert cap.png_b64 == _PNG_B64
    assert backend._active_pid == 4321
    assert backend._active_window_id == 77
    assert base64.b64decode(cap.png_b64)  # decodes cleanly


def test_capture_uses_pid_zero_driver_fallback_for_null_pid_target():
    raw = [
        {"app_name": "", "pid": None, "window_id": 77,
         "is_on_screen": True, "title": "CUA Hard Case Test", "z_index": 1},
    ]
    backend = _backend_with_windows(raw)

    cap = backend.capture(mode="vision", window_title="Hard Case")

    assert cap.png_b64 == _PNG_B64
    assert backend._active_pid == 0
    assert backend._active_window_id == 77
    screenshot_call = backend._session.call_tool.call_args_list[-1]  # type: ignore[attr-defined]
    assert screenshot_call.args[0] == "screenshot"
    assert screenshot_call.args[1]["window_id"] == 77


def test_action_preserves_structured_outcome_fields_in_meta():
    from tools.computer_use.cua_backend import CuaDriverBackend

    backend = CuaDriverBackend()
    session = MagicMock()
    session.supports_capability.return_value = False
    session.call_tool.return_value = {
        "isError": False,
        "data": "typed text",
        "structuredContent": {
            "verified": False,
            "effect": "unverifiable",
            "escalation": {"next": "page"},
            "path": "ax",
        },
    }
    backend._session = session

    result = backend._action("type_text", {"pid": 123, "text": "hello"})

    assert result.ok is True
    assert result.message == "typed text"
    assert result.meta["verified"] is False
    assert result.meta["effect"] == "unverifiable"
    assert result.meta["escalation"] == {"next": "page"}
    assert result.meta["path"] == "ax"


def test_linux_foreground_pixel_click_marks_xtest_as_unverifiable_with_driver_hint(monkeypatch):
    from tools.computer_use import cua_backend
    from tools.computer_use.cua_backend import CuaDriverBackend

    backend = CuaDriverBackend()
    backend._active_pid = 0
    backend._active_window_id = 77
    backend._active_window_bounds = {"x": 120, "y": 157, "width": 520, "height": 260}
    session = MagicMock()
    session.supports_capability.return_value = False
    session.call_tool.return_value = {
        "isError": False,
        "data": "",
        "structuredContent": {
            "verified": False,
            "effect": "unverifiable",
            "path": "x11_xtest_fg",
        },
    }
    backend._session = session

    monkeypatch.setattr(cua_backend.sys, "platform", "linux")

    result = backend.click(x=260, y=183, button="left", delivery_mode="foreground")

    assert result.ok is True
    assert result.meta["verified"] is False
    assert result.meta["effect"] == "unverifiable"
    assert "fallback" not in result.meta
    assert result.meta["escalation"]["recommended"] == "verify_or_driver_input_backend"
    assert "portal-libei" in " ".join(result.meta["escalation"]["next"])
    assert "does not require ydotool" in result.meta["foreground_note"]
    driver_call = session.call_tool.call_args
    assert driver_call.args[0] == "click"
    assert driver_call.args[1]["x"] == 260
    assert driver_call.args[1]["y"] == 183
    assert driver_call.args[1]["window_id"] == 77


def test_linux_pixel_click_annotation_is_limited_to_unverifiable_pixel_paths(monkeypatch):
    from tools.computer_use import cua_backend
    from tools.computer_use.cua_backend import CuaDriverBackend

    backend = CuaDriverBackend()
    backend._active_pid = 0
    backend._active_window_id = 77
    session = MagicMock()
    session.supports_capability.return_value = False
    session.call_tool.return_value = {
        "isError": False,
        "data": "",
        "structuredContent": {
            "verified": True,
            "effect": "confirmed",
            "path": "x11_atspi",
        },
    }
    backend._session = session

    monkeypatch.setattr(cua_backend.sys, "platform", "linux")

    result = backend.click(x=260, y=183, button="left", delivery_mode="background")

    assert result.ok is True
    assert result.meta["effect"] == "confirmed"
    assert "escalation" not in result.meta
