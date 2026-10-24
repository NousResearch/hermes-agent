"""Desktop player requests require the exact live turn, record, durable session, and transport."""

import json
import threading

import pytest

from gateway.session_context import clear_session_vars, set_session_vars
from tui_gateway import server
from tui_gateway.transport import StdioTransport


class _Agent:
    def __init__(self, session_id, callback):
        self.session_id = session_id
        self.player_agent_callback = callback


def _bind(monkeypatch, records, runtime_id, transport):
    monkeypatch.setattr(server, "_sessions", records)
    for record in records.values():
        record["transport"] = transport
    vars_token = set_session_vars(ui_session_id=runtime_id)
    transport_token = server.bind_transport(transport)
    runtime_token = server._current_runtime_session_record.set(records.get(runtime_id))
    return vars_token, transport_token, runtime_token


def _reset(records, tokens):
    server._current_runtime_session_record.reset(tokens[2])
    server.reset_transport(tokens[1])
    clear_session_vars(tokens[0])


def test_player_request_uses_active_runtime_record_with_durable_session_mapping(monkeypatch):
    calls = []
    records = {
        "runtime-a": {"agent": _Agent("stored-a", lambda op: calls.append(("a", op)) or '{"ok":true}')},
        "runtime-b": {"agent": _Agent("stored-b", lambda op: calls.append(("b", op)) or '{"ok":true}')},
    }
    transport = StdioTransport(lambda: None, threading.Lock())
    for runtime_id, durable_id in (("runtime-a", "stored-a"), ("runtime-b", "stored-b"), ("runtime-a", "stored-a")):
        tokens = _bind(monkeypatch, records, runtime_id, transport)
        try:
            result = server.request_player_agent_for_session(durable_id, {"type": "context"})
            assert json.loads(result) == {"ok": True}
        finally:
            _reset(records, tokens)
    assert calls == [("a", {"type": "context"}), ("b", {"type": "context"}),
                     ("a", {"type": "context"})]


@pytest.mark.parametrize("operation", [
    {"type": "context"},
    {"type": "control", "context_token": "x" * 24, "navigation_generation": 2, "action": "seek",
     "position_seconds": 12.5},
    {"type": "transcript.start", "context_token": "x" * 24, "language": "en-US",
     "range": {"start_seconds": 2, "end_seconds": 8}},
    {"type": "download.start", "context_token": "x" * 24, "quality": "1080p"},
    {"type": "jobs.list"},
    {"type": "job.status", "job_id": "job_12"},
    {"type": "job.cancel", "job_id": "job_12"},
])
def test_each_closed_player_operation_reaches_the_linked_agent(monkeypatch, operation):
    received = []
    record = {"agent": _Agent("stored-a", lambda payload: received.append(payload) or "{}")}
    transport = StdioTransport(lambda: None, threading.Lock())
    tokens = _bind(monkeypatch, {"runtime-a": record}, "runtime-a", transport)
    try:
        assert json.loads(server.request_player_agent_for_session("stored-a", operation)) == {}
    finally:
        _reset({"runtime-a": record}, tokens)
    assert received == [operation]


@pytest.mark.parametrize("operation", [
    {"type": "context", "session_id": "forged"},
    {"type": "context", "url": "https://example.invalid"},
    {"type": []},
    {"type": "control", "context_token": "short", "navigation_generation": 0, "action": "play"},
    {"type": "control", "context_token": "x" * 24, "navigation_generation": 0, "action": "seek"},
    {"type": "transcript.start", "context_token": "x" * 24, "language": "x"},
    {"type": "download.start", "context_token": "x" * 24, "quality": "2160p"},
    {"type": "job.status", "job_id": "../../state.db"},
])
def test_player_request_rejects_malformed_closed_operations(operation):
    assert json.loads(server.request_player_agent_for_session("stored-a", operation)) == {
        "error": "invalid-operation"
    }


def test_player_request_fails_closed_for_wrong_identity_transport_or_stale_record(monkeypatch):
    called = []
    record = {"agent": _Agent("stored-a", lambda op: called.append(op) or "{}")}
    transport = StdioTransport(lambda: None, threading.Lock())
    tokens = _bind(monkeypatch, {"runtime-a": record}, "runtime-a", transport)
    try:
        assert json.loads(server.request_player_agent_for_session("stored-b", {"type": "context"})) == {
            "error": "desktop-player-unavailable"
        }
        other_transport = StdioTransport(lambda: None, threading.Lock())
        other_token = server.bind_transport(other_transport)
        try:
            assert json.loads(server.request_player_agent_for_session("stored-a", {"type": "context"})) == {
                "error": "desktop-player-unavailable"
            }
        finally:
            server.reset_transport(other_token)
    finally:
        _reset({"runtime-a": record}, tokens)

    tokens = _bind(monkeypatch, {"runtime-a": record}, "runtime-a", transport)
    replacement = dict(record)
    replacement_token = server._current_runtime_session_record.set(record)
    monkeypatch.setattr(server, "_sessions", {"runtime-a": replacement})
    try:
        assert json.loads(server.request_player_agent_for_session("stored-a", {"type": "context"})) == {
            "error": "desktop-player-unavailable"
        }
    finally:
        server._current_runtime_session_record.reset(replacement_token)
        _reset({"runtime-a": record}, tokens)
    assert called == []
