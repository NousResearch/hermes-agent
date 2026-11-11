"""computer_use zoom + invoke_menu (port of code-yeongyu/oh-my-openagent#10007 and its menus cluster).

cua-driver 0.21+ ships ``zoom`` and ``invoke_menu``; hermes' action enum never exposed them.
"""

import json

from tools.computer_use.backend import ActionResult, CaptureResult
from tools.computer_use.cua_backend_zoom_menu import _ZoomMenuMixin
from tools.computer_use.schema import COMPUTER_USE_SCHEMA
from tools.computer_use.tool import _ACTIONS, _dispatch


class _Session:
    def __init__(self, out):
        self.out, self.calls = out, []

    def call_tool(self, name, args):
        self.calls.append((name, args))
        return self.out


class _Backend(_ZoomMenuMixin):
    def __init__(self, out):
        self._session = _Session(out)
        self._session_id, self._active_pid, self._active_window_id, self._last_app = "s", 42, 7, "App"

    def _action(self, name, args):
        out = self._session.call_tool(name, args)
        return ActionResult(ok=not out["isError"], action=name, message=out["data"], meta=out["structuredContent"])


def test_zoom_is_free_and_menu_is_gated_and_targets_the_sticky_window():
    enum = COMPUTER_USE_SCHEMA["parameters"]["properties"]["action"]["enum"]
    assert {"zoom", "invoke_menu"} <= set(enum)
    assert not _ACTIONS["zoom"].destructive  # a read like capture
    assert _ACTIONS["invoke_menu"].destructive  # runs an app command -> approval prompt
    backend = _Backend({"isError": False, "data": "Zoom", "images": ["/9j/AAAA"], "image_mime_types": ["image/jpeg"],
                        "structuredContent": {"width": 480, "height": 360}})
    cap = backend.zoom((0, 0, 400, 300))
    assert isinstance(cap, CaptureResult) and cap.png_b64 == "/9j/AAAA" and (cap.width, cap.height) == (480, 360)
    assert backend._session.calls[-1] == ("zoom", {"window_id": 7, "x1": 0, "y1": 0, "x2": 400, "y2": 300,
                                                   "session": "s", "pid": 42})


def test_refused_menu_path_surfaces_its_code_without_a_pixel_escalation_verdict():
    # The driver fails closed on a missing/disabled segment; the tool must not recommend the pixel rungs.
    backend = _Backend({"isError": True, "data": "invoke_menu: no item 'Nope'", "structuredContent": {
        "status": "refused", "refusal": {"code": "menu_path_unavailable", "message": "no item"}}})
    out = json.loads(_dispatch(backend, "invoke_menu", {"menu_path": ["File", "Nope"]}))
    assert out["code"] == "menu_path_unavailable" and out["ok"] is False and "verdict" not in out
    assert backend._session.calls[-1][1]["path"] == ["File", "Nope"]
