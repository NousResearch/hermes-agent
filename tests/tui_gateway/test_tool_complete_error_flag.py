"""A recovered tool failure must reach clients on the ``tool.complete`` frame itself.

Regression: ``_on_tool_complete`` built its payload without the failure verdict, so the
``error`` flag the renderer flashes a failed reaction from was never set — only
turn-ending errors could ever reach that path.
"""

import json

import tui_gateway.server as server
from tui_gateway.contracts.events import ToolCompletePayload


def _capture(monkeypatch, sid="tool-error-flag"):
    events = []
    monkeypatch.setitem(
        server._sessions,
        sid,
        {"agent": None, "edit_snapshots": {}, "tool_started_at": {}},
    )
    monkeypatch.setattr(server, "_tool_progress_enabled", lambda _sid: True)
    monkeypatch.setattr(server, "_session_verbose", lambda _sid: False)
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, event_sid, payload=None: events.append((event, event_sid, payload)),
    )
    return sid, events


def test_failed_tool_result_is_flagged_on_the_complete_frame(monkeypatch):
    sid, events = _capture(monkeypatch)

    server._on_tool_complete(
        sid, "call-1", "terminal", {"command": "false"}, json.dumps({"exit_code": 1, "error": "boom"})
    )

    payload = events[-1][2]
    assert payload["error"] is True
    # The frame must stay inside its declared contract.
    ToolCompletePayload.model_validate(payload)


def test_successful_tool_result_is_not_flagged(monkeypatch):
    sid, events = _capture(monkeypatch)

    server._on_tool_complete(
        sid, "call-2", "terminal", {"command": "true"}, json.dumps({"exit_code": 0, "stdout": "ok"})
    )

    assert "error" not in events[-1][2]
