"""Regression tests for Windows desktop/screen capture fallback.

When cua-driver's Windows enumeration tools hang (`list_windows`,
`list_apps`, UIA tree walks), the driver can still take a full desktop
screenshot via `get_desktop_state`. Explicit `app="screen"` captures must use
that path directly instead of getting stuck on window enumeration first.
"""

from __future__ import annotations


_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAADUlEQVR4nG"
    "NgGAUgAAABCAABgukLHQAAAABJRU5ErkJggg=="
)


class FakeSession:
    def __init__(self):
        self.calls = []

    def call_tool(self, name, args, timeout=30.0):
        self.calls.append((name, args, timeout))
        if name == "list_windows":
            raise AssertionError("screen capture must not enumerate windows")
        if name == "get_config":
            return {
                "structuredContent": {"capture_scope": "window"},
                "data": {},
                "images": [],
                "image_mime_types": [],
                "isError": False,
            }
        if name == "set_config":
            return {
                "structuredContent": {"capture_scope": args.get("value")},
                "data": {},
                "images": [],
                "image_mime_types": [],
                "isError": False,
            }
        if name == "get_desktop_state":
            return {
                "structuredContent": {
                    "screen_width": 1920,
                    "screen_height": 1080,
                    "screenshot_width": 8,
                    "screenshot_height": 8,
                    "screenshot_mime_type": "image/png",
                },
                "data": "desktop screenshot 8x8 px",
                "images": [_PNG_B64],
                "image_mime_types": ["image/png"],
                "isError": False,
            }
        raise AssertionError(f"unexpected tool {name}")


def test_screen_capture_bypasses_list_windows_and_uses_desktop_state():
    from tools.computer_use.cua_backend import CuaDriverBackend

    backend = CuaDriverBackend()
    fake = FakeSession()
    backend._session = fake

    cap = backend.capture(mode="vision", app="screen")

    assert cap.app == "screen"
    assert cap.width == 8
    assert cap.height == 8
    assert cap.png_b64 == _PNG_B64
    assert cap.image_mime_type == "image/png"
    assert cap.elements == []
    assert [c[0] for c in fake.calls] == [
        "get_config",
        "set_config",
        "get_desktop_state",
        "set_config",
    ]
    assert fake.calls[1][1] == {"key": "capture_scope", "value": "desktop"}
    assert fake.calls[-1][1] == {"key": "capture_scope", "value": "window"}
