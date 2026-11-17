"""``tool.complete`` carries a hosted artifact's viewer link for the Ink TUI (and the gateway's local viewers)."""

from __future__ import annotations

import json

import pytest

from tui_gateway import server
from tui_gateway.contracts.registry import EVENTS

VIEWER = "https://portal.example.com/a/01hx"
TICKET = "https://01hx.frame.example.com/__av?t=secret-ticket"
CREATE = "mcp__hermes_artifacts__artifact_create"
OK = json.dumps({"result": json.dumps({"id": "01hx", "version": 1, "title": "Plan",
                                       "viewer_url": VIEWER, "content_url": TICKET})})
STALE = json.dumps({"error": "stale_base: base_version 1 is not the head (2)"})


def _completes(monkeypatch, mode, calls):
    events = []
    monkeypatch.setattr(server, "_emit", lambda event_type, sid, payload=None: events.append((event_type, payload)))
    monkeypatch.setitem(server._sessions, "art-sid", {"tool_progress_mode": mode, "tool_started_at": {},
                                                      "edit_snapshots": {}})
    for call_id, name, result in calls:
        server._on_tool_complete("art-sid", call_id, name, {}, result)
    return {payload["tool_id"]: payload for event, payload in events if event == "tool.complete"}


@pytest.mark.parametrize("mode", ["all", "off"])
def test_tool_complete_names_the_viewer_link(monkeypatch, mode):
    completes = _completes(monkeypatch, mode, [("t1", CREATE, OK), ("t2", CREATE, STALE), ("t3", "terminal", OK)])
    assert completes["t1"]["artifact_link"] == {"url": VIEWER, "title": "Plan"}
    EVENTS["tool.complete"].payload.model_validate(completes["t1"])
    # With tool progress off an ordinary call stays quiet; the one carrying a link still reaches the client.
    assert ("t3" in completes) is (mode == "all")
    assert all("artifact_link" not in completes[t] for t in ("t2", "t3") if t in completes)


def test_local_viewer_frames_carry_the_same_link():
    from types import SimpleNamespace

    from gateway.local_viewer_events import complete_fields

    turn = SimpleNamespace()
    assert complete_fields(turn, "c1", CREATE, {}, OK)["artifact_link"] == {"url": VIEWER, "title": "Plan"}
    assert complete_fields(turn, "c2", CREATE, {}, STALE, is_error=True)["artifact_link"] is None
    assert complete_fields(turn, "c3", "terminal", {}, OK)["artifact_link"] is None
