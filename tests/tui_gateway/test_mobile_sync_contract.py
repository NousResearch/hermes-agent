"""Public JSON-RPC conformance tests for revisioned conversation sync."""

import threading
import time

import pytest

from tui_gateway import mobile_contract
from tui_gateway import server
from tui_gateway.mobile_sync import SessionEventStream


class RecordingTransport:
    def __init__(self):
        self.authorization = {
            "subject": "mobile-user",
            "provider": "stub",
            "audience": "hermes.mobile",
            "scopes": (
                "conversation.read",
                "conversation.write",
                "conversation.control",
            ),
        }
        self.frames = []

    def write(self, obj):
        self.frames.append(obj)
        return True

    def close(self):
        pass


class SessionDBStub:
    def __init__(self, session_id, messages=None):
        self.session_id = session_id
        self.messages = list(messages or [])

    def get_session(self, session_id):
        if session_id == self.session_id:
            return {"id": self.session_id, "cwd": server.os.getcwd()}
        return None

    def get_session_by_title(self, _title):
        return None

    def resolve_resume_session_id(self, session_id):
        return session_id

    def reopen_session(self, _session_id):
        return None

    def get_messages_as_conversation(self, _session_id, include_ancestors=False):
        return list(self.messages)


@pytest.fixture(autouse=True)
def isolated_gateway(monkeypatch):
    server._sessions.clear()
    monkeypatch.setattr(
        server, "_claim_active_session_slot", lambda *_a, **_k: (None, None)
    )
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_profile_home", lambda *_a, **_k: None)
    monkeypatch.setattr(
        server, "_completion_cwd", lambda *_a, **_k: server.os.getcwd()
    )
    monkeypatch.setattr(server, "_git_branch_for_cwd", lambda *_a, **_k: "")
    monkeypatch.setattr(server, "_resolve_model", lambda: "test/model")
    monkeypatch.setattr(server, "_current_profile_name", lambda: "default")
    yield
    server._sessions.clear()


def create_session(transport):
    response = server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": "create",
            "method": "session.create",
            "params": {"cols": 100},
        },
        transport,
    )
    assert response is not None
    return response["result"]


def resume_session(transport, stored_session_id, cursor=None):
    params = {"session_id": stored_session_id, "cols": 100}
    if cursor is not None:
        params["cursor"] = cursor
    request_id = f"resume-{len(transport.frames)}"
    response = server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "session.resume",
            "params": params,
        },
        transport,
    )
    if response is not None:
        return response["result"]
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        responses = [
            frame for frame in transport.frames if frame.get("id") == request_id
        ]
        if responses:
            assert "error" not in responses[-1]
            return responses[-1]["result"]
        time.sleep(0.01)
    raise AssertionError("session.resume response was not delivered")


def test_create_returns_revisioned_authoritative_sync_snapshot():
    transport = RecordingTransport()

    result = create_session(transport)

    synchronization = result["synchronization"]
    snapshot = synchronization["snapshot"]
    assert synchronization["schema_major"] == 1
    assert snapshot == {
        "schema_major": 1,
        "server_instance_id": mobile_contract.SERVER_INSTANCE_ID,
        "stream_id": snapshot["stream_id"],
        "revision": 1,
        "watermark": 0,
        "conversation_id": result["stored_session_id"],
        "stored_session_id": result["stored_session_id"],
        "live_session_id": result["session_id"],
        "messages": [],
        "inflight_turn": None,
        "active_tools": [],
        "pending_interactions": [],
        "status": "idle",
    }
    assert synchronization["recovery"] == {
        "outcome": "reset",
        "reason": "cursor_missing",
        "snapshot_required": True,
        "events": [],
        "cursor": {
            "server_instance_id": mobile_contract.SERVER_INSTANCE_ID,
            "stream_id": snapshot["stream_id"],
            "sequence": 0,
        },
    }


def test_live_resume_replays_every_event_after_a_matching_cursor(monkeypatch):
    transport = RecordingTransport()
    created = create_session(transport)
    cursor = created["synchronization"]["recovery"]["cursor"]
    sid = created["session_id"]
    monkeypatch.setattr(
        server,
        "_get_db",
        lambda: SessionDBStub(created["stored_session_id"]),
    )

    server._emit("status.update", sid, {"kind": "working", "text": "Working"})
    server._emit("session.info", sid, {"running": True})
    resumed = resume_session(transport, created["stored_session_id"], cursor)

    synchronization = resumed["synchronization"]
    assert resumed["session_id"] == sid
    assert synchronization["snapshot"]["stream_id"] == cursor["stream_id"]
    assert synchronization["snapshot"]["watermark"] == 2
    assert synchronization["snapshot"]["revision"] == 3
    assert synchronization["recovery"]["outcome"] == "complete"
    assert synchronization["recovery"]["snapshot_required"] is False
    assert [
        event["params"]["type"] for event in synchronization["recovery"]["events"]
    ] == ["status.update", "session.info"]
    assert [
        event["params"]["sequence"] for event in synchronization["recovery"]["events"]
    ] == [1, 2]


def test_cold_resume_has_the_same_shape_and_requires_stream_reset(monkeypatch):
    stored_id = "stored-cold"
    transport = RecordingTransport()
    monkeypatch.setattr(
        server,
        "_get_db",
        lambda: SessionDBStub(
            stored_id,
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
        ),
    )
    prior_cursor = {
        "server_instance_id": mobile_contract.SERVER_INSTANCE_ID,
        "stream_id": "retired-stream",
        "sequence": 12,
    }

    resumed = resume_session(transport, stored_id, prior_cursor)

    synchronization = resumed["synchronization"]
    snapshot = synchronization["snapshot"]
    assert set(snapshot) == {
        "schema_major",
        "server_instance_id",
        "stream_id",
        "revision",
        "watermark",
        "conversation_id",
        "stored_session_id",
        "live_session_id",
        "messages",
        "inflight_turn",
        "active_tools",
        "pending_interactions",
        "status",
    }
    assert snapshot["conversation_id"] == stored_id
    assert snapshot["messages"] == [
        {"role": "user", "text": "hello"},
        {"role": "assistant", "text": "hi"},
    ]
    assert synchronization["recovery"]["outcome"] == "reset"
    assert synchronization["recovery"]["reason"] == "stream_changed"


def test_resume_preserves_conversation_lineage_while_exposing_current_ids(
    monkeypatch,
):
    class LineageDB(SessionDBStub):
        def get_compression_lineage(self, _session_id):
            return ["conversation-root", self.session_id]

    stored_id = "compression-tip"
    transport = RecordingTransport()
    monkeypatch.setattr(server, "_get_db", lambda: LineageDB(stored_id))

    resumed = resume_session(transport, stored_id)

    snapshot = resumed["synchronization"]["snapshot"]
    assert snapshot["conversation_id"] == "conversation-root"
    assert snapshot["stored_session_id"] == stored_id
    assert snapshot["live_session_id"] == resumed["session_id"]


def test_replay_overflow_is_an_explicit_gap(monkeypatch):
    transport = RecordingTransport()
    created = create_session(transport)
    sid = created["session_id"]
    session = server._sessions[sid]
    session["mobile_sync"] = SessionEventStream(
        mobile_contract.SERVER_INSTANCE_ID,
        max_events=2,
        max_bytes=1024 * 1024,
    )
    cursor = session["mobile_sync"].cursor()
    monkeypatch.setattr(
        server,
        "_get_db",
        lambda: SessionDBStub(created["stored_session_id"]),
    )

    for index in range(3):
        server._emit("status.update", sid, {"kind": "step", "text": str(index)})
    resumed = resume_session(transport, created["stored_session_id"], cursor)

    recovery = resumed["synchronization"]["recovery"]
    assert recovery["outcome"] == "gap"
    assert recovery["reason"] == "replay_evicted"
    assert recovery["snapshot_required"] is True
    assert recovery["events"] == []


def test_oversized_event_is_delivered_live_but_not_claimed_as_replayable(
    monkeypatch,
):
    transport = RecordingTransport()
    created = create_session(transport)
    sid = created["session_id"]
    session = server._sessions[sid]
    session["mobile_sync"] = SessionEventStream(
        mobile_contract.SERVER_INSTANCE_ID,
        max_events=20,
        max_bytes=128,
    )
    cursor = session["mobile_sync"].cursor()
    monkeypatch.setattr(
        server,
        "_get_db",
        lambda: SessionDBStub(created["stored_session_id"]),
    )

    server._emit("status.update", sid, {"kind": "large", "text": "x" * 256})
    assert transport.frames[-1]["params"]["sequence"] == 1
    resumed = resume_session(transport, created["stored_session_id"], cursor)

    recovery = resumed["synchronization"]["recovery"]
    assert recovery["outcome"] == "gap"
    assert recovery["snapshot_required"] is True
    assert recovery["events"] == []


def test_concurrent_session_events_are_monotonic_in_wire_order():
    transport = RecordingTransport()
    created = create_session(transport)
    sid = created["session_id"]
    start = threading.Barrier(9)

    def publish(index):
        start.wait()
        server._emit("status.update", sid, {"kind": "worker", "text": str(index)})

    threads = [threading.Thread(target=publish, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(timeout=1)
        assert not thread.is_alive()

    events = [frame for frame in transport.frames if frame.get("method") == "event"]
    assert [event["params"]["sequence"] for event in events] == list(range(1, 9))
    assert all(event["params"]["schema_major"] == 1 for event in events)
    assert len({event["params"]["stream_id"] for event in events}) == 1


def test_coalesced_deltas_carry_one_turn_identity_and_absolute_utf8_offsets():
    transport = RecordingTransport()
    created = create_session(transport)
    sid = created["session_id"]
    session = server._sessions[sid]
    with session["history_lock"]:
        server._start_inflight_turn(session, "question")
        turn_id = session["inflight_turn"]["turn_id"]
    server._emit_inflight_delta(sid, session, "A💡")
    server._emit_inflight_delta(sid, session, "B")

    deltas = [
        frame["params"]["payload"]
        for frame in transport.frames
        if frame.get("params", {}).get("type") == "message.delta"
    ]
    assert deltas == [
        {"text": "A💡", "turn_id": turn_id, "offset": 0},
        {"text": "B", "turn_id": turn_id, "offset": 5},
    ]


def test_snapshot_tracks_only_sanitized_active_tool_descriptors(monkeypatch):
    transport = RecordingTransport()
    created = create_session(transport)
    sid = created["session_id"]
    stored_id = created["stored_session_id"]
    monkeypatch.setattr(server, "_get_db", lambda: SessionDBStub(stored_id))
    monkeypatch.setattr(server, "_tool_progress_enabled", lambda _sid: True)
    monkeypatch.setattr(server, "_tool_ctx", lambda _name, _args: "workspace file")

    server._on_tool_start(
        sid,
        "tool-1",
        "terminal",
        {"command": "printf super-secret"},
    )
    running = resume_session(transport, stored_id)

    descriptor = running["synchronization"]["snapshot"]["active_tools"][0]
    assert descriptor["tool_id"] == "tool-1"
    assert descriptor["name"] == "terminal"
    assert isinstance(descriptor["started_at"], float)
    assert "context" not in descriptor
    assert "args" not in descriptor
    assert "command" not in descriptor

    server._on_tool_complete(sid, "tool-1", "terminal", {}, "{}")
    finished = resume_session(transport, stored_id)
    assert finished["synchronization"]["snapshot"]["active_tools"] == []


def test_snapshot_recovers_and_then_clears_a_pending_interaction(monkeypatch):
    transport = RecordingTransport()
    created = create_session(transport)
    sid = created["session_id"]
    stored_id = created["stored_session_id"]
    monkeypatch.setattr(server, "_get_db", lambda: SessionDBStub(stored_id))
    answer = {}

    def ask():
        answer["value"] = server._block(
            "clarify.request",
            sid,
            {"question": "Pick one", "choices": ["A", "B"]},
            timeout=2,
        )

    thread = threading.Thread(target=ask)
    thread.start()
    deadline = time.monotonic() + 1
    request = None
    while time.monotonic() < deadline:
        request = next(
            (
                frame
                for frame in transport.frames
                if frame.get("params", {}).get("type") == "clarify.request"
            ),
            None,
        )
        if request is not None:
            break
        time.sleep(0.01)
    assert request is not None
    request_id = request["params"]["payload"]["request_id"]

    waiting = resume_session(transport, stored_id)
    snapshot = waiting["synchronization"]["snapshot"]
    assert snapshot["status"] == "waiting"
    assert snapshot["pending_interactions"] == [
        {
            "request_id": request_id,
            "kind": "clarify",
            "payload": {
                "request_id": request_id,
                "question": "Pick one",
                "choices": ["A", "B"],
            },
        }
    ]

    response = server._methods["clarify.respond"](
        "answer",
        {"request_id": request_id, "answer": "A"},
    )
    assert response["result"] == {"status": "ok"}
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert answer == {"value": "A"}

    settled = resume_session(transport, stored_id)
    assert settled["synchronization"]["snapshot"]["pending_interactions"] == []
