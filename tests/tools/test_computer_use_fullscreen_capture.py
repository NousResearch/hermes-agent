"""Full-screen vs desktop-shell capture routing (#60081, wallpaper-only bug).

`capture(app='screen')` previously resolved to the OS shell/desktop window
(Progman/WorkerW on Windows) via list_windows — the wallpaper + icons layer —
so "screenshot my screen" always showed a bare desktop no matter what was
actually displayed. cua-driver's `get_desktop_state` does a real composited
full-screen grab; the `screen`/`fullscreen`/`all` sentinels now route there,
while `desktop` keeps the shell-window lane (with clickable elements).

Salvaged from @2ndNatureAI's PR #60081 (enumeration-hang bypass) and extended
with the sentinel split + no-elements guidance note.
"""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

# 8×8 transparent PNG — decodes cleanly for dimension sniffing.
_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAADUlEQVR4nG"
    "NgGAUgAAABCAABgukLHQAAAABJRU5ErkJggg=="
)


class _FakeSession:
    """Records tool calls; serves get_config / set_config / get_desktop_state
    and (for the desktop-shell path) list_windows / screenshot."""

    def __init__(
        self,
        windows: Optional[List[Dict[str, Any]]] = None,
        desktop_image: Optional[str] = _PNG_B64,
        capture_scope: str = "window",
    ):
        self.calls: List[tuple] = []
        self._windows = windows or []
        self._desktop_image = desktop_image
        self._scope = capture_scope
        self.capabilities_discovered = True

    def _has_tool(self, name: str) -> bool:
        return name in {"get_desktop_state", "screenshot", "list_windows"}

    def call_tool(self, name: str, args: Dict[str, Any], timeout: float = 30.0):
        self.calls.append((name, dict(args or {})))
        if name == "get_config":
            return {
                "data": "",
                "images": [],
                "structuredContent": {"capture_scope": self._scope},
                "isError": False,
            }
        if name == "set_config":
            self._scope = args["value"]
            return {"data": "ok", "images": [], "structuredContent": None,
                    "isError": False}
        if name == "get_desktop_state":
            images = [self._desktop_image] if self._desktop_image else []
            return {
                "data": "desktop state",
                "images": images,
                "image_mime_types": ["image/png"] if images else [],
                "structuredContent": {"screen_width": 8, "screen_height": 8},
                "isError": False,
            }
        if name == "list_windows":
            return {
                "data": "",
                "images": [],
                "structuredContent": {"windows": self._windows},
                "isError": False,
            }
        if name == "screenshot":
            return {
                "data": "",
                "images": [_PNG_B64],
                "image_mime_types": ["image/png"],
                "structuredContent": None,
                "isError": False,
            }
        raise AssertionError(f"unexpected tool call: {name}")

    def called(self, name: str) -> List[Dict[str, Any]]:
        return [a for (n, a) in self.calls if n == name]


def _make_backend(session: _FakeSession):
    from tools.computer_use.cua_backend import CuaDriverBackend

    backend = CuaDriverBackend()
    backend._session = session
    backend._session_id = "test-session"
    return backend


class TestFullScreenLane:
    @pytest.mark.parametrize("sentinel", ["screen", "Screen", "fullscreen",
                                          "full screen", "all"])
    def test_screen_sentinels_route_to_get_desktop_state(self, sentinel):
        session = _FakeSession()
        backend = _make_backend(session)

        cap = backend.capture(mode="som", app=sentinel)

        assert session.called("get_desktop_state"), (
            f"app={sentinel!r} must use the composited desktop lane"
        )
        assert not session.called("list_windows"), (
            "full-screen capture must not enumerate windows (enumeration "
            "can hang on Windows — trycua/cua#2110)"
        )
        assert cap.png_b64 == _PNG_B64
        assert cap.elements == []
        assert cap.app == "screen"

    def test_full_screen_result_carries_interactive_lane_note(self):
        session = _FakeSession()
        backend = _make_backend(session)

        cap = backend.capture(mode="vision", app="screen")

        assert "no interactable elements" in cap.note
        assert "capture(app='desktop')" in cap.note
        assert "capture(app='<AppName>')" in cap.note

    def test_capture_scope_switched_and_restored(self):
        session = _FakeSession(capture_scope="window")
        backend = _make_backend(session)

        backend.capture(mode="vision", app="screen")

        set_calls = session.called("set_config")
        assert {"key": "capture_scope", "value": "desktop",
                "session": "test-session"} in set_calls
        assert {"key": "capture_scope", "value": "window",
                "session": "test-session"} in set_calls
        assert session._scope == "window", "prior scope must be restored"

    def test_scope_untouched_when_already_desktop(self):
        session = _FakeSession(capture_scope="desktop")
        backend = _make_backend(session)

        backend.capture(mode="vision", app="screen")

        assert not session.called("set_config")

    def test_imageless_desktop_state_fails_closed_with_guidance(self):
        session = _FakeSession(desktop_image=None)
        backend = _make_backend(session)

        cap = backend.capture(mode="vision", app="screen")

        assert cap.png_b64 is None
        assert "get_desktop_state returned no image" in cap.window_title

    def test_dimensions_come_from_decoded_image(self):
        session = _FakeSession()
        backend = _make_backend(session)

        cap = backend.capture(mode="vision", app="screen")

        # The 8×8 PNG's real dimensions win over structuredContent.
        assert (cap.width, cap.height) == (8, 8)
        assert cap.png_bytes_len == len(base64.b64decode(_PNG_B64))

    def test_exact_pid_window_target_bypasses_full_screen_lane(self):
        session = _FakeSession()
        backend = _make_backend(session)

        backend.capture(mode="vision", app="screen", pid=123, window_id=456)

        assert not session.called("get_desktop_state"), (
            "an exact pid/window target must win over the app sentinel"
        )


class TestDesktopShellLane:
    _PROGMAN = {
        "app_name": "Progman",
        "title": "Program Manager",
        "pid": 100,
        "window_id": 1,
        "off_screen": False,
        "z_index": 0,
    }

    def test_desktop_sentinel_keeps_shell_window_lane(self):
        session = _FakeSession(windows=[self._PROGMAN])
        backend = _make_backend(session)

        cap = backend.capture(mode="vision", app="desktop")

        assert session.called("list_windows"), (
            "app='desktop' must still resolve the shell window so desktop "
            "icons stay clickable"
        )
        assert not session.called("get_desktop_state")
        assert backend._active_pid == 100
        assert cap.note == ""

    def test_desktop_sentinel_without_shell_window_fails_with_guidance(self):
        session = _FakeSession(windows=[{
            "app_name": "Notepad", "title": "Untitled", "pid": 7,
            "window_id": 9, "off_screen": False, "z_index": 1,
        }])
        backend = _make_backend(session)

        cap = backend.capture(mode="vision", app="desktop")

        assert cap.png_b64 is None
        assert "no desktop/shell window found" in cap.window_title


class TestNoteInSummary:
    def test_capture_response_appends_note_line(self, tmp_path, monkeypatch):
        import hermes_constants
        from tools.computer_use.backend import CaptureResult
        from tools.computer_use import tool as cu_tool

        monkeypatch.setattr(hermes_constants, "get_hermes_dir",
                            lambda *a, **k: tmp_path)
        monkeypatch.setattr(cu_tool, "_should_route_through_aux_vision",
                            lambda: False)

        cap = CaptureResult(
            mode="vision", width=8, height=8, png_b64=_PNG_B64,
            elements=[], app="screen",
            window_title="Full screen (composited)",
            png_bytes_len=len(base64.b64decode(_PNG_B64)),
            note="full-screen capture has no interactable elements; "
                 "call capture(app='desktop') for the desktop shell",
        )
        result = cu_tool._capture_response(cap)

        text = str(result)
        assert "full-screen capture has no interactable elements" in text
