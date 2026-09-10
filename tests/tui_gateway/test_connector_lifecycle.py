"""Connection actions remain visible with optional tool chrome off."""

import json

import pytest


@pytest.mark.parametrize("name,args,result", [
    ("manage_connections", {"action": "connect", "connectors": ["gmail"]},
     {"results": [{"connector": "gmail", "status": "initiated", "connect_url": "https://connect.example/?token=keep"}]}),
    ("manage_connections", {"action": "wait", "connectors": ["gmail"]},
     {"status": "timeout", "pending": ["gmail"]}),
    ("tool_call", {"name": "manage_connections", "arguments": {"action": "reconnect", "connectors": ["gmail"]}},
     {"results": [{"connector": "gmail", "status": "active"}]}),
    ("tool_call", {"calls": [{"name": "connectors__gmail__READ", "arguments": {}}]},
     {"results": [{"name": "connectors__gmail__READ", "error": {"code": "CONNECTION_REQUIRED", "connector": "gmail"}}]}),
])
def test_only_connector_lifecycle_survives_progress_off(monkeypatch, name, args, result):
    from tui_gateway import server
    events = []
    monkeypatch.setattr(server, "_sessions", {"owner": {"tool_progress_mode": "off"}})
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload: events.append((event, sid, payload)))
    server._on_tool_start("owner", "call", name, args)
    server._on_tool_complete("owner", "call", name, args, json.dumps(result))
    assert [event for event, _, _ in events] == ["tool.start", "tool.complete"]
    assert events[-1][2]["result"] == result
    assert all(sid == "owner" for _, sid, _ in events)
    events.clear()
    for other, other_args in (("terminal", {"command": "whoami"}),
                              ("tool_call", {"name": "web_search", "arguments": {}})):
        server._on_tool_start("owner", "other", other, other_args)
        server._on_tool_complete("owner", "other", other, other_args, "{}")
    assert events == []


def test_connector_lifecycle_drops_retired_generation_and_redacts_secrets(monkeypatch):
    from tui_gateway import server
    owner = {"tool_progress_mode": "off"}
    monkeypatch.setattr(server, "_sessions", {"owner": owner})
    events = []
    monkeypatch.setattr(server, "_emit", lambda *event: events.append(event))
    args = {"action": "connect", "connectors": ["gmail"]}
    result = {"results": [{"connector": "gmail", "status": "initiated",
                           "connect_url": "https://connect.example/?token=keep",
                           "access_token": "private-access", "Authorization": "Bearer private-auth",
                           "cookie": "private-cookie"}]}
    token = server._current_runtime_session_record.set(owner)
    try:
        server._on_tool_start("owner", "call", "manage_connections", args)
        server._on_tool_complete("owner", "call", "manage_connections", args, json.dumps(result))
        assert "private-" not in json.dumps(events)
        assert events[-1][2]["result"]["results"][0]["connect_url"].endswith("token=keep")
        events.clear()
        server._sessions["owner"] = {"tool_progress_mode": "off"}
        server._on_tool_start("owner", "old", "manage_connections", args)
        server._on_tool_complete("owner", "old", "manage_connections", args, json.dumps(result))
        assert events == []
    finally:
        server._current_runtime_session_record.reset(token)


def test_connector_generation_is_rechecked_after_payload_projection(monkeypatch):
    from tui_gateway import server, connector_payload
    owner = {"tool_progress_mode": "off"}
    monkeypatch.setattr(server, "_sessions", {"owner": owner})
    events = []
    monkeypatch.setattr(server, "_emit", lambda *event: events.append(event))
    project = connector_payload.connector_ui_payload

    def retire(value):
        projected = project(value)
        server._sessions["owner"] = {"tool_progress_mode": "off"}
        return projected

    monkeypatch.setattr(connector_payload, "connector_ui_payload", retire)
    token = server._current_runtime_session_record.set(owner)
    try:
        server._on_tool_complete("owner", "late", "manage_connections", {"action": "wait"}, "{}")
        assert not events
    finally:
        server._current_runtime_session_record.reset(token)
