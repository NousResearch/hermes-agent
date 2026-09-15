"""The desktop-only tools cross the typed event boundary as the models their events declare.

``desktop_ui.emit_or_error`` hands the tool's payload to the sink ``session_notifications`` installs,
which is ``server._emit`` — the typed frame builder. A producer still passing a dict got ``TypeError``
back as tool_error text and the renderer event never fired; the tools' own tests use permissive
capture callbacks and could not see it, so this one goes through the real sink and the real frame.
"""

from __future__ import annotations

import json

import pytest

from tools import apply_layout_tool, close_preview_tool, desktop_ui, focus_pane_tool, open_preview_tool, preview_tool
from tui_gateway import server, session_notifications

_TOOL_CALLS = [
    ("pane.reveal", lambda: focus_pane_tool.focus_pane_tool("terminal"), {"pane": "terminal"}),
    ("layout.apply", lambda: apply_layout_tool.apply_layout_tool("focus"), {"preset": "focus"}),
    ("preview.open", lambda: open_preview_tool.open_preview_tool("localhost:3000", label="dev"),
     {"url": "http://localhost:3000", "label": "dev"}),
    ("preview.close", lambda: close_preview_tool.close_preview_tool("localhost:3000"), {"url": "http://localhost:3000"}),
    ("preview.close", lambda: preview_tool.preview_close(""), {"url": ""}),
]


@pytest.fixture
def desktop_sink(monkeypatch):
    frames = []
    monkeypatch.setattr(server, "write_json", lambda frame: frames.append(frame) or True)
    monkeypatch.setattr(desktop_ui, "get_session_env",
                        lambda name, default="": "win-7" if name == "HERMES_UI_SESSION_ID" else default)
    monkeypatch.setattr(server, "_desktop_ui_wired", False)
    session_notifications._wire_desktop_sinks()
    yield frames
    desktop_ui.set_emitter(None)


@pytest.mark.parametrize("event,call,payload", _TOOL_CALLS, ids=[f"{e}:{i}" for i, (e, _c, _p) in enumerate(_TOOL_CALLS)])
def test_desktop_tool_event_reaches_the_renderer_through_the_typed_frame(desktop_sink, event, call, payload):
    result = json.loads(call())

    assert result["success"] is True, result
    frame = desktop_sink[-1]["params"]
    assert (frame["type"], frame["session_id"], frame["payload"]) == (event, "win-7", payload)
