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

The fix routes both ingestion sites through ``_ingest_windows``, which skips
windows lacking a usable pid/window_id (uncapturable anyway) and coerces the
rest, so real targetable windows survive.
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
    def test_skips_window_with_null_pid(self):
        from tools.computer_use.cua_backend import _ingest_windows

        raw = [
            {"app_name": "Desktop", "pid": None, "window_id": 1, "z_index": 0},
            {"app_name": "Firefox", "pid": 4321, "window_id": 77, "z_index": 1},
        ]

        out = _ingest_windows(raw)

        assert [w["app_name"] for w in out] == ["Firefox"]
        assert out[0]["pid"] == 4321
        assert out[0]["window_id"] == 77

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

    cap = backend.capture(mode="vision")

    # The real, targetable window is selected rather than the whole capture
    # crashing on the null-pid desktop window.
    assert cap.app == "Firefox"
    assert cap.png_b64 == _PNG_B64
    assert backend._active_pid == 4321
    assert backend._active_window_id == 77
    assert base64.b64decode(cap.png_b64)  # decodes cleanly
