"""``trace.events`` serves the Relay events recorded for a session and subscribes it, so every
further recorded event reaches that session's own transport as ``trace.event``."""

from __future__ import annotations

import json
from types import SimpleNamespace


def test_trace_events_reads_the_session_log_then_forwards_live_events(tmp_path, monkeypatch):
    import tui_gateway.server as server
    from agent import relay_runtime
    from hermes_cli.observability import relay_traces

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(server, "_hermes_home", home)
    recorded = {"kind": "scope", "scope_category": "start", "uuid": "u1", "name": "hermes.turn",
                "category": "function", "timestamp": "2026-09-30T12:00:00+00:00"}
    relay_traces.traces_dir(home).mkdir()
    relay_traces.trace_path(home, "stored-1").write_text(json.dumps(recorded) + "\n", encoding="utf-8")

    frames: list[dict] = []
    watched = {"session_key": "stored-1", "agent": SimpleNamespace(session_id="stored-1"),
               "history": [], "transport": SimpleNamespace(write=lambda frame: frames.append(frame) or True)}
    bystander = {"session_key": "stored-2", "agent": SimpleNamespace(session_id="stored-2"), "history": [],
                 "transport": SimpleNamespace(write=lambda frame: frames.append(frame) or True)}
    monkeypatch.setitem(server._sessions, "ui-1", watched)
    monkeypatch.setitem(server._sessions, "ui-2", bystander)

    reply = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "trace.events",
                                   "params": {"session_id": "ui-1"}})["result"]
    assert reply["session_id"] == "stored-1"
    assert reply["events"] == [recorded]

    live = {**recorded, "uuid": "u2"}
    profile = relay_runtime.current_profile_key()
    server._forward_trace_event(profile, "stored-1", "stored-1", live)
    server._forward_trace_event(profile, "stored-2", "stored-2", live)  # nobody traces ui-2
    server._forward_trace_event("/another/profile", "stored-1", "stored-1", live)

    events = [f["params"] for f in frames if f.get("method") == "event" and f["params"]["type"] == "trace.event"]
    assert [(e["session_id"], e["payload"]) for e in events] == [
        ("ui-1", {"trace_session_id": "stored-1", "event": live})
    ]
