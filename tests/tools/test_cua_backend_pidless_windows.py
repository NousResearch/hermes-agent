from __future__ import annotations

from tools.computer_use.cua_backend import CuaDriverBackend


class _FakeSession:
    def __init__(self, responses):
        self._responses = responses
        self.calls = []
        self.capabilities_discovered = True

    def call_tool(self, name, args, timeout=30.0):
        self.calls.append((name, dict(args)))
        resp = self._responses.get(name)
        if callable(resp):
            return resp(args)
        if resp is None:
            raise AssertionError(f"Unexpected tool call: {name}")
        return resp

    def _has_tool(self, _name):
        return False


def _backend_with_responses(responses):
    backend = CuaDriverBackend()
    backend._session = _FakeSession(responses)
    backend._session_id = "test-session"
    return backend


def test_capture_reports_explicit_pidless_window_diagnostic(monkeypatch):
    monkeypatch.setattr(
        "tools.computer_use.cua_backend_capture._resolve_linux_x11_pid_from_window_id",
        lambda _wid: None,
    )
    backend = _backend_with_responses(
        {
            "list_windows": {
                "structuredContent": {
                    "windows": [
                        {
                            "app_name": "Tk",
                            "title": "Hermes Probe",
                            "pid": None,
                            "window_id": 123,
                            "is_on_screen": True,
                            "z_index": 0,
                        }
                    ]
                }
            }
        }
    )

    cap = backend.capture(mode="som")

    assert cap.width == 0
    assert cap.height == 0
    assert "no pid" in cap.window_title.lower()
    assert "Hermes Probe" in cap.window_title
    assert "get_window_state requires pid" in cap.window_title


def test_capture_uses_linux_x11_pid_fallback_before_get_window_state(monkeypatch):
    monkeypatch.setattr(
        "tools.computer_use.cua_backend_capture._resolve_linux_x11_pid_from_window_id",
        lambda wid: 4321 if wid == 456 else None,
    )
    backend = _backend_with_responses(
        {
            "list_windows": {
                "structuredContent": {
                    "windows": [
                        {
                            "app_name": "Tk",
                            "title": "Hermes Probe",
                            "pid": None,
                            "window_id": 456,
                            "is_on_screen": True,
                            "z_index": 0,
                        }
                    ]
                }
            },
            "get_window_state": {
                "data": '✅ Tk — 0 elements\nAXWindow "Hermes Probe"',
                "structuredContent": {},
                "images": [],
                "image_mime_types": [],
            },
        }
    )

    cap = backend.capture(mode="som")

    assert backend._active_pid == 4321
    assert backend._active_window_id == 456
    assert cap.app == "Tk"
    assert cap.window_title == "Hermes Probe"
    gws_call = next(call for call in backend._session.calls if call[0] == "get_window_state")
    assert gws_call[1]["pid"] == 4321
    assert gws_call[1]["window_id"] == 456


def test_focus_app_reports_pidless_match_instead_of_false_not_found(monkeypatch):
    monkeypatch.setattr(
        "tools.computer_use.cua_backend_capture._resolve_linux_x11_pid_from_window_id",
        lambda _wid: None,
    )
    backend = _backend_with_responses(
        {
            "list_windows": {
                "structuredContent": {
                    "windows": [
                        {
                            "app_name": "Tk",
                            "title": "Hermes Probe",
                            "pid": None,
                            "window_id": 789,
                            "is_on_screen": True,
                            "z_index": 0,
                        }
                    ]
                }
            }
        }
    )

    result = backend.focus_app("Tk")

    assert result.ok is False
    assert "no usable pid" in result.message
    assert "Hermes Probe" in result.message
    assert result.meta["pidless_windows"][0]["window_id"] == 789
