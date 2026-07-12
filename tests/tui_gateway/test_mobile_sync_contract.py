"""Public JSON-RPC conformance tests for revisioned conversation sync."""

import json
import threading
import time

import pytest

from tui_gateway import mobile_contract
from tui_gateway import server
from tui_gateway.mobile_mutations import MobileMutationStore
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
def isolated_gateway(monkeypatch, tmp_path):
    server._sessions.clear()
    mutation_store = MobileMutationStore(tmp_path / "mobile-mutations.sqlite3")
    monkeypatch.setattr(server, "_mobile_mutation_store", lambda: mutation_store)
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
    mutation_store.close()


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


def test_snapshot_recovers_exact_redacted_approval_until_terminal_event(monkeypatch):
    transport = RecordingTransport()
    created = create_session(transport)
    sid = created["session_id"]
    stored_id = created["stored_session_id"]
    monkeypatch.setattr(server, "_get_db", lambda: SessionDBStub(stored_id))
    approval_id = "a" * 32
    raw_secret = "sk-proj-super-secret-mobile-contract"

    server._emit_approval_request(
        sid,
        {
            "approval_id": approval_id,
            "command": f"curl -H 'Authorization: Bearer {raw_secret}' example.test",
            "description": "network request",
            "created_at": 10.0,
            "expires_at": 310.0,
            "state": "pending",
            "resolution": None,
        },
    )

    waiting = resume_session(transport, stored_id)
    snapshot = waiting["synchronization"]["snapshot"]
    assert snapshot["status"] == "waiting"
    assert len(snapshot["pending_interactions"]) == 1
    descriptor = snapshot["pending_interactions"][0]
    assert set(descriptor) == {
        "approval_id",
        "command",
        "description",
        "created_at",
        "expires_at",
        "state",
        "resolution",
        "kind",
    }
    assert descriptor["approval_id"] == approval_id
    assert descriptor["description"] == "network request"
    assert descriptor["created_at"] == 10.0
    assert descriptor["expires_at"] == 310.0
    assert descriptor["state"] == "pending"
    assert descriptor["resolution"] is None
    assert descriptor["kind"] == "approval"
    serialized = json.dumps(snapshot["pending_interactions"])
    assert raw_secret not in serialized
    assert "sk-proj-" not in serialized

    server._emit_approval_terminal(
        sid,
        {
            "approval_id": approval_id,
            "state": "resolved",
            "resolution": {"choice": "once", "resolved_at": 11.0},
        },
    )

    terminal = next(
        frame
        for frame in reversed(transport.frames)
        if frame.get("params", {}).get("type") == "approval.resolved"
    )
    assert terminal["params"]["payload"]["approval_id"] == approval_id
    assert terminal["params"]["payload"]["state"] == "resolved"
    settled = resume_session(transport, stored_id)
    assert settled["synchronization"]["snapshot"]["pending_interactions"] == []


def test_approval_response_targets_exact_identity_without_changing_legacy_fifo(
    monkeypatch,
):
    from tools import approval

    transport = RecordingTransport()
    created = create_session(transport)
    sid = created["session_id"]
    session = server._sessions[sid]
    session["agent_ready"] = None
    calls = []

    def resolve_exact(session_key, approval_id, choice, **kwargs):
        calls.append((session_key, approval_id, choice, kwargs))
        return {
            "outcome": "resolved",
            "approval": {
                "approval_id": approval_id,
                "state": "resolved",
                "resolution": {"choice": choice},
            },
        }

    monkeypatch.setattr(approval, "resolve_gateway_approval_by_id", resolve_exact)
    monkeypatch.setattr(approval, "resolve_gateway_approval", lambda *_a, **_k: 2)

    exact = server._methods["approval.respond"](
        "exact",
        {
            "session_id": sid,
            "approval_id": "b" * 32,
            "choice": "once",
            "reason": "approved from phone",
        },
    )
    assert exact["result"]["outcome"] == "resolved"
    assert calls == [
        (
            session["session_key"],
            "b" * 32,
            "once",
            {
                "reason": "approved from phone",
                "resolution_metadata": {
                    "source": "tui_gateway",
                    "live_session_id": sid,
                },
            },
        )
    ]

    legacy = server._methods["approval.respond"](
        "legacy",
        {"session_id": sid, "choice": "deny", "all": True},
    )
    assert legacy["result"] == {"resolved": 2}


def test_disconnect_resume_recovers_and_resolves_same_approval_once(monkeypatch):
    from tools import approval

    transport = RecordingTransport()
    created = create_session(transport)
    sid = created["session_id"]
    stored_id = created["stored_session_id"]
    session = server._sessions[sid]
    session["agent_ready"] = None
    monkeypatch.setattr(server, "_get_db", lambda: SessionDBStub(stored_id))
    monkeypatch.setattr(
        approval,
        "_get_approval_config",
        lambda: {"gateway_timeout": 2},
    )
    server._register_gateway_approval_callbacks(session["session_key"], sid)
    decision = {}

    def wait_for_decision():
        decision["value"] = approval._await_gateway_decision(
            session["session_key"],
            approval._gateway_notify_cbs[session["session_key"]],
            {
                "command": "rm -rf /tmp/example",
                "description": "recursive delete",
                "pattern_key": "rm_recursive",
                "pattern_keys": ["rm_recursive"],
            },
        )

    thread = threading.Thread(target=wait_for_decision)
    thread.start()
    try:
        deadline = time.monotonic() + 1
        approval_id = ""
        while time.monotonic() < deadline:
            requests = [
                frame
                for frame in transport.frames
                if frame.get("params", {}).get("type") == "approval.request"
            ]
            if requests:
                approval_id = requests[-1]["params"]["payload"]["approval_id"]
                break
            time.sleep(0.01)
        assert approval_id

        reconnected = RecordingTransport()
        recovered = resume_session(reconnected, stored_id)
        pending = recovered["synchronization"]["snapshot"]["pending_interactions"]
        assert [item["approval_id"] for item in pending] == [approval_id]
        active_sid = recovered["session_id"]
        response_params = {
            "approval_id": approval_id,
            "choice": "once",
            "client_request_id": "resume-approval-response-1",
            "expected_stored_session_id": stored_id,
            "session_id": active_sid,
        }

        first = server.dispatch(
            {
                "jsonrpc": "2.0",
                "id": "resolve-first",
                "method": "approval.respond",
                "params": response_params,
            },
            reconnected,
        )
        duplicate = server.dispatch(
            {
                "jsonrpc": "2.0",
                "id": "resolve-duplicate",
                "method": "approval.respond",
                "params": response_params,
            },
            reconnected,
        )
        assert first["result"]["outcome"] == "resolved"
        assert first["result"]["mutation"]["deduplicated"] is False
        assert duplicate["result"]["outcome"] == "resolved"
        assert duplicate["result"]["mutation"]["deduplicated"] is True

        thread.join(timeout=1)
        assert not thread.is_alive()
        assert decision == {
            "value": {"resolved": True, "choice": "once", "reason": None}
        }
        terminal_events = [
            frame
            for frame in reconnected.frames
            if frame.get("params", {}).get("type") == "approval.resolved"
        ]
        assert len(terminal_events) == 1
        assert terminal_events[0]["params"]["payload"]["approval_id"] == approval_id
        settled = resume_session(reconnected, stored_id)
        assert settled["synchronization"]["snapshot"]["pending_interactions"] == []
    finally:
        approval.unregister_gateway_notify(session["session_key"])
        approval._gateway_tombstones.pop(session["session_key"], None)
        thread.join(timeout=1)


def test_mobile_resolves_second_approval_before_legacy_fifo_first(monkeypatch):
    from tools import approval

    transport = RecordingTransport()
    created = create_session(transport)
    sid = created["session_id"]
    stored_id = created["stored_session_id"]
    session = server._sessions[sid]
    session["agent_ready"] = None
    monkeypatch.setattr(server, "_get_db", lambda: SessionDBStub(stored_id))
    monkeypatch.setattr(
        approval,
        "_get_approval_config",
        lambda: {"gateway_timeout": 2},
    )
    server._register_gateway_approval_callbacks(session["session_key"], sid)
    decisions = {}

    def wait_for_decision(name):
        decisions[name] = approval._await_gateway_decision(
            session["session_key"],
            approval._gateway_notify_cbs[session["session_key"]],
            {
                "command": f"dangerous {name}",
                "description": name,
                "pattern_key": name,
                "pattern_keys": [name],
            },
        )

    def wait_for_request(description):
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            for frame in transport.frames:
                params = frame.get("params", {})
                payload = params.get("payload", {})
                if (
                    params.get("type") == "approval.request"
                    and payload.get("description") == description
                ):
                    return payload["approval_id"]
            time.sleep(0.01)
        raise AssertionError(f"approval request {description!r} not emitted")

    first_thread = threading.Thread(target=wait_for_decision, args=("first",))
    second_thread = threading.Thread(target=wait_for_decision, args=("second",))
    first_thread.start()
    first_id = wait_for_request("first")
    second_thread.start()
    second_id = wait_for_request("second")
    try:
        second = server.dispatch(
            {
                "jsonrpc": "2.0",
                "id": "resolve-second",
                "method": "approval.respond",
                "params": {
                    "approval_id": second_id,
                    "choice": "once",
                    "client_request_id": "resolve-second-1",
                    "expected_stored_session_id": stored_id,
                    "session_id": sid,
                },
            },
            transport,
        )
        assert second["result"]["outcome"] == "resolved"
        assert second["result"]["approval"]["approval_id"] == second_id
        second_thread.join(timeout=1)
        assert not second_thread.is_alive()
        assert first_thread.is_alive()

        pending = resume_session(transport, stored_id)["synchronization"][
            "snapshot"
        ]["pending_interactions"]
        assert [item["approval_id"] for item in pending] == [first_id]

        legacy = server._methods["approval.respond"](
            "legacy-first",
            {"session_id": sid, "choice": "deny"},
        )
        assert legacy["result"] == {"resolved": 1}
        first_thread.join(timeout=1)
        assert not first_thread.is_alive()
        assert decisions == {
            "first": {"resolved": True, "choice": "deny", "reason": None},
            "second": {"resolved": True, "choice": "once", "reason": None},
        }
    finally:
        approval.unregister_gateway_notify(session["session_key"])
        approval._gateway_tombstones.pop(session["session_key"], None)
        first_thread.join(timeout=1)
        second_thread.join(timeout=1)


def test_legacy_session_keeps_original_response_and_event_shape():
    class LegacyTransport(RecordingTransport):
        def __init__(self):
            super().__init__()
            self.authorization = {
                "subject": "dashboard-user",
                "provider": "password",
                "audience": "dashboard",
                "scopes": ("*",),
            }

    transport = LegacyTransport()
    created = create_session(transport)
    sid = created["session_id"]

    assert "synchronization" not in created
    assert not server._sessions[sid].get("mobile_sync_retention")

    server._emit("status.update", sid, {"kind": "idle", "text": "Ready"})

    event = transport.frames[-1]
    assert event == {
        "jsonrpc": "2.0",
        "method": "event",
        "params": {
            "type": "status.update",
            "session_id": sid,
            "payload": {"kind": "idle", "text": "Ready"},
        },
    }
    assert not server._sessions[sid].get("mobile_sync_retention")
    assert server._sessions[sid]["mobile_sync"].cursor()["sequence"] == 0


def test_mobile_replay_stays_retained_across_a_legacy_handoff(monkeypatch):
    class LegacyTransport(RecordingTransport):
        def __init__(self):
            super().__init__()
            self.authorization = {
                "subject": "dashboard-user",
                "provider": "password",
                "audience": "dashboard",
                "scopes": ("*",),
            }

    mobile = RecordingTransport()
    legacy = LegacyTransport()
    created = create_session(mobile)
    sid = created["session_id"]
    stored_id = created["stored_session_id"]
    cursor = created["synchronization"]["recovery"]["cursor"]
    session = server._sessions[sid]
    monkeypatch.setattr(server, "_get_db", lambda: SessionDBStub(stored_id))

    legacy_payload = server._live_session_payload(
        sid,
        session,
        transport=legacy,
        cursor=cursor,
    )
    assert "synchronization" not in legacy_payload

    server._emit("status.update", sid, {"kind": "legacy", "text": "Still here"})
    legacy_event = legacy.frames[-1]
    assert "sequence" not in legacy_event["params"]
    assert "stream_id" not in legacy_event["params"]

    reconnected = resume_session(mobile, stored_id, cursor)
    recovery = reconnected["synchronization"]["recovery"]
    assert recovery["outcome"] == "complete"
    assert [event["params"]["type"] for event in recovery["events"]] == [
        "status.update"
    ]
    assert recovery["events"][0]["params"]["sequence"] == 1


def test_transport_handoffs_serialize_audience_snapshot_and_delivery(monkeypatch):
    class LegacyTransport(RecordingTransport):
        def __init__(self):
            super().__init__()
            self.authorization = {
                "subject": "dashboard-user",
                "provider": "password",
                "audience": "dashboard",
                "scopes": ("*",),
            }

    class BlockingWriteMobile(RecordingTransport):
        def __init__(self, entered, release):
            super().__init__()
            self._entered = entered
            self._release = release

        def write(self, obj):
            self._entered.set()
            assert self._release.wait(timeout=2)
            return super().write(obj)

    mobile = RecordingTransport()
    legacy = LegacyTransport()
    created = create_session(mobile)
    sid = created["session_id"]
    stored_id = created["stored_session_id"]
    session = server._sessions[sid]
    monkeypatch.setattr(server, "_get_db", lambda: SessionDBStub(stored_id))

    write_entered = threading.Event()
    release_write = threading.Event()
    blocking_mobile = BlockingWriteMobile(write_entered, release_write)
    with session["history_lock"]:
        server._set_session_transport_locked(session, blocking_mobile)

    emitted = threading.Thread(
        target=server._emit,
        args=("status.update", sid, {"kind": "race", "text": "mobile"}),
    )
    legacy_result = {}

    def attach_legacy():
        legacy_result["payload"] = server._live_session_payload(
            sid,
            session,
            transport=legacy,
        )

    emitted.start()
    assert write_entered.wait(timeout=1)
    legacy_handoff = threading.Thread(target=attach_legacy)
    legacy_handoff.start()
    assert legacy_handoff.is_alive()
    release_write.set()
    emitted.join(timeout=1)
    legacy_handoff.join(timeout=1)
    assert not emitted.is_alive()
    assert not legacy_handoff.is_alive()
    mobile_event = blocking_mobile.frames[-1]
    assert mobile_event["params"]["sequence"] == 1
    assert "synchronization" not in legacy_result["payload"]

    auth_entered = threading.Event()
    release_auth = threading.Event()

    class BlockingAuthMobile(RecordingTransport):
        def __init__(self):
            self._authorization = dict(RecordingTransport().authorization)
            self.frames = []

        @property
        def authorization(self):
            auth_entered.set()
            assert release_auth.wait(timeout=2)
            return self._authorization

    blocking_auth_mobile = BlockingAuthMobile()
    mobile_result = {}
    final_legacy_result = {}

    def attach_mobile():
        mobile_result["payload"] = server._live_session_payload(
            sid,
            session,
            transport=blocking_auth_mobile,
        )

    def attach_final_legacy():
        final_legacy_result["payload"] = server._live_session_payload(
            sid,
            session,
            transport=legacy,
        )

    mobile_handoff = threading.Thread(target=attach_mobile)
    mobile_handoff.start()
    assert auth_entered.wait(timeout=1)
    final_legacy_handoff = threading.Thread(target=attach_final_legacy)
    final_legacy_handoff.start()
    assert final_legacy_handoff.is_alive()
    release_auth.set()
    mobile_handoff.join(timeout=1)
    final_legacy_handoff.join(timeout=1)
    assert not mobile_handoff.is_alive()
    assert not final_legacy_handoff.is_alive()
    assert "synchronization" in mobile_result["payload"]
    assert "synchronization" not in final_legacy_result["payload"]


def test_concurrent_first_mobile_snapshot_and_publish_share_one_stream():
    transport = RecordingTransport()
    sid = "first-attach"
    session = {
        "agent": None,
        "conversation_id": "first-conversation",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "inflight_turn": None,
        "running": False,
        "session_key": "first-stored",
        "transport": transport,
    }
    server._sessions[sid] = session
    start = threading.Barrier(3)
    attached = {}

    def attach():
        start.wait()
        attached["payload"] = server._attach_synchronization({}, sid, session)

    def publish():
        start.wait()
        server._emit("status.update", sid, {"kind": "race", "text": "Ready"})

    attach_thread = threading.Thread(target=attach)
    publish_thread = threading.Thread(target=publish)
    attach_thread.start()
    publish_thread.start()
    start.wait()
    attach_thread.join(timeout=1)
    publish_thread.join(timeout=1)
    assert not attach_thread.is_alive()
    assert not publish_thread.is_alive()

    snapshot = attached["payload"]["synchronization"]["snapshot"]
    event = next(
        frame
        for frame in transport.frames
        if frame.get("params", {}).get("type") == "status.update"
    )
    assert snapshot["stream_id"] == event["params"]["stream_id"]
    assert session["mobile_sync"].stream_id == snapshot["stream_id"]


def test_completion_history_and_event_share_one_snapshot_barrier(monkeypatch):
    transport = RecordingTransport()
    sid = "barrier-live"
    stream = SessionEventStream(mobile_contract.SERVER_INSTANCE_ID)
    session = {
        "agent": None,
        "conversation_id": "barrier-conversation",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "inflight_turn": None,
        "mobile_sync": stream,
        "mobile_sync_retention": True,
        "running": True,
        "session_key": "barrier-stored",
        "transport": transport,
    }
    server._sessions[sid] = session
    with session["history_lock"]:
        server._start_inflight_turn(session, "question")
    cursor = stream.cursor()

    entered_completion = threading.Event()
    release_completion = threading.Event()
    original_clear = server._clear_inflight_turn

    def blocking_clear(target):
        entered_completion.set()
        assert release_completion.wait(timeout=2)
        original_clear(target)

    monkeypatch.setattr(server, "_clear_inflight_turn", blocking_clear)
    completion = threading.Thread(
        target=server._commit_prompt_completion,
        kwargs={
            "sid": sid,
            "session": session,
            "expected_history_version": 0,
            "result_messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
            "payload": {"text": "answer", "status": "complete", "usage": {}},
        },
    )
    completion.start()
    assert entered_completion.wait(timeout=1)

    captured = {}
    snapshot_started = threading.Event()
    snapshot_done = threading.Event()

    def capture_snapshot():
        snapshot_started.set()
        captured["value"] = server._session_synchronization(sid, session, cursor)
        snapshot_done.set()

    snapshot_thread = threading.Thread(target=capture_snapshot)
    snapshot_thread.start()
    assert snapshot_started.wait(timeout=1)
    assert not snapshot_done.wait(timeout=0.05)

    release_completion.set()
    completion.join(timeout=1)
    snapshot_thread.join(timeout=1)
    assert not completion.is_alive()
    assert not snapshot_thread.is_alive()

    synchronization = captured["value"]
    assert synchronization["snapshot"]["messages"] == [
        {"role": "user", "text": "question"},
        {"role": "assistant", "text": "answer"},
    ]
    assert synchronization["snapshot"]["inflight_turn"] is None
    assert synchronization["snapshot"]["watermark"] == 1
    assert [
        event["params"]["type"]
        for event in synchronization["recovery"]["events"]
    ] == ["message.complete"]


def test_deferred_prompt_start_advances_revision_before_agent_ready(monkeypatch):
    transport = RecordingTransport()
    created = create_session(transport)
    sid = created["session_id"]
    stored_id = created["stored_session_id"]
    cursor = created["synchronization"]["recovery"]["cursor"]
    initial_revision = created["synchronization"]["snapshot"]["revision"]
    entered_wait = threading.Event()
    release_wait = threading.Event()
    monkeypatch.setattr(server, "_get_db", lambda: SessionDBStub(stored_id))
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: None)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda _session: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_args: None)

    def blocked_wait(_session, rid):
        entered_wait.set()
        assert release_wait.wait(timeout=2)
        return server._err(rid, 5000, "test build stopped")

    monkeypatch.setattr(server, "_wait_agent", blocked_wait)
    try:
        submitted = server.dispatch(
            {
                "jsonrpc": "2.0",
                "id": "deferred-submit",
                "method": "prompt.submit",
                "params": {
                    "client_request_id": "deferred-submit-1",
                    "expected_stored_session_id": stored_id,
                    "session_id": sid,
                    "text": "question",
                },
            },
            transport,
        )
        assert submitted["result"] == {
            "mutation": {
                "client_request_id": "deferred-submit-1",
                "deduplicated": False,
                "state": "in_progress",
            },
            "status": "streaming",
        }
        assert entered_wait.wait(timeout=1)

        resumed = resume_session(transport, stored_id, cursor)
        synchronization = resumed["synchronization"]
        assert synchronization["snapshot"]["revision"] == initial_revision + 1
        assert synchronization["snapshot"]["watermark"] == cursor["sequence"]
        assert synchronization["snapshot"]["inflight_turn"]["user"] == "question"
        assert synchronization["recovery"]["outcome"] == "complete"
        assert synchronization["recovery"]["events"] == []
    finally:
        release_wait.set()
        run_thread = server._sessions.get(sid, {}).get("_run_thread")
        if run_thread is not None:
            run_thread.join(timeout=1)


def test_undo_advances_snapshot_revision_without_a_wire_event(monkeypatch):
    transport = RecordingTransport()
    created = create_session(transport)
    sid = created["session_id"]
    stored_id = created["stored_session_id"]
    session = server._sessions[sid]
    session["agent"] = object()
    session["history"] = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    initial_revision = created["synchronization"]["snapshot"]["revision"]
    initial_watermark = created["synchronization"]["snapshot"]["watermark"]
    monkeypatch.setattr(server, "_get_db", lambda: SessionDBStub(stored_id))
    monkeypatch.setattr(server, "_start_agent_build", lambda *_args: None)
    monkeypatch.setattr(server, "_wait_agent", lambda *_args: None)

    result = server._methods["session.undo"]("undo", {"session_id": sid})
    resumed = resume_session(transport, stored_id)

    assert result["result"] == {"removed": 2}
    snapshot = resumed["synchronization"]["snapshot"]
    assert snapshot["messages"] == []
    assert snapshot["revision"] == initial_revision + 1
    assert snapshot["watermark"] == initial_watermark


def test_child_mirror_sync_recovers_inflight_text_and_active_tool(monkeypatch):
    transport = RecordingTransport()
    created = create_session(transport)
    sid = created["session_id"]
    stored_id = created["stored_session_id"]
    monkeypatch.setattr(server, "_get_db", lambda: SessionDBStub(stored_id))
    monkeypatch.setattr(server, "_tool_progress_enabled", lambda _sid: True)

    server._mirror_subagent_to_child(
        "subagent.start",
        {"child_session_id": stored_id, "text": "goal"},
    )
    server._mirror_subagent_to_child(
        "subagent.text",
        {"child_session_id": stored_id, "text": "answer"},
    )
    server._mirror_subagent_to_child(
        "subagent.tool",
        {
            "child_session_id": stored_id,
            "tool_name": "terminal",
            "tool_preview": "workspace command",
        },
    )

    running = resume_session(transport, stored_id)
    snapshot = running["synchronization"]["snapshot"]
    assert snapshot["status"] == "working"
    assert snapshot["inflight_turn"]["assistant"] == "goal\nanswer"
    assert snapshot["inflight_turn"]["streaming"] is True
    assert snapshot["active_tools"] == [
        {
            "tool_id": snapshot["active_tools"][0]["tool_id"],
            "name": "terminal",
            "started_at": snapshot["active_tools"][0]["started_at"],
        }
    ]
    assert isinstance(snapshot["active_tools"][0]["started_at"], float)
    assert "preview" not in snapshot["active_tools"][0]

    deltas = [
        frame["params"]["payload"]
        for frame in transport.frames
        if frame.get("params", {}).get("type") == "message.delta"
    ]
    assert [delta["offset"] for delta in deltas] == [0, 5]
    assert len({delta["turn_id"] for delta in deltas}) == 1

    server._mirror_subagent_to_child(
        "subagent.complete",
        {
            "child_session_id": stored_id,
            "summary": "done",
        },
    )
    completed = resume_session(transport, stored_id)
    settled = completed["synchronization"]["snapshot"]
    assert settled["status"] == "idle"
    assert settled["inflight_turn"] is None
    assert settled["active_tools"] == []
    assert settled["messages"][-1] == {
        "role": "assistant",
        "text": "answer",
    }
    complete = next(
        frame
        for frame in reversed(transport.frames)
        if frame.get("params", {}).get("type") == "message.complete"
    )
    assert complete["params"]["payload"]["text"] == "answer"


def test_conversation_root_does_not_hide_callable_attribute_errors():
    class BrokenLineageDB:
        @staticmethod
        def get_compression_lineage(_session_id):
            raise AttributeError("lineage implementation bug")

    with pytest.raises(AttributeError, match="lineage implementation bug"):
        server._conversation_root(BrokenLineageDB(), "stored")
