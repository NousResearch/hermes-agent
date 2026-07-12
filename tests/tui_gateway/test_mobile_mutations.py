"""Durable receipt behavior for consequential mobile JSON-RPC mutations."""

from __future__ import annotations

import contextlib
import threading

import pytest

from tui_gateway.mobile_mutations import (
    MobileMutationStore,
    MutationConflict,
    MutationDisposition,
)


class _MobileTransport:
    def __init__(self, *scopes: str, subject: str = "mobile-user"):
        self.authorization = {
            "audience": "hermes.mobile",
            "provider": "password",
            "scopes": scopes,
            "subject": subject,
        }

    def write(self, _obj):
        return True


def _reserve(
    store: MobileMutationStore,
    *,
    request_id: str = "request-1",
    text: str = "hello",
):
    return store.reserve(
        provider="password",
        subject="mobile-user",
        client_request_id=request_id,
        method="prompt.submit",
        resource_id="conversation-root",
        semantic_parameters={"text": text},
    )


def test_completed_mutation_replays_exact_outcome_after_database_reopen(tmp_path):
    path = tmp_path / "mobile-mutations.sqlite3"
    first = MobileMutationStore(path, owner_instance_id="process-a")
    claim = _reserve(first)
    assert claim.disposition is MutationDisposition.EXECUTE
    outcome = {
        "jsonrpc": "2.0",
        "id": "original-correlation",
        "result": {"status": "streaming"},
    }
    assert first.complete(claim, outcome) is True
    first.close()

    reopened = MobileMutationStore(path, owner_instance_id="process-b")
    replay = _reserve(reopened)

    assert replay.disposition is MutationDisposition.REPLAY
    assert replay.outcome == outcome
    assert reopened.status(
        provider="password",
        subject="mobile-user",
        client_request_id="request-1",
    )["state"] == "completed"


def test_same_request_identity_with_different_semantics_conflicts(tmp_path):
    store = MobileMutationStore(tmp_path / "mobile-mutations.sqlite3")
    claim = _reserve(store, text="first")
    store.complete(claim, {"result": {"status": "streaming"}})

    with pytest.raises(MutationConflict):
        _reserve(store, text="different")


def test_abandoned_in_progress_mutation_becomes_outcome_unknown_after_reopen(
    tmp_path,
):
    path = tmp_path / "mobile-mutations.sqlite3"
    first = MobileMutationStore(path, owner_instance_id="process-a")
    claim = _reserve(first)
    assert claim.disposition is MutationDisposition.EXECUTE
    first.close()

    reopened = MobileMutationStore(path, owner_instance_id="process-b")
    uncertain = _reserve(reopened)

    assert uncertain.disposition is MutationDisposition.OUTCOME_UNKNOWN
    status = reopened.status(
        provider="password",
        subject="mobile-user",
        client_request_id="request-1",
    )
    assert status["state"] == "outcome_unknown"
    assert status["outcome"] is None


def test_principals_do_not_share_request_identity_namespaces(tmp_path):
    store = MobileMutationStore(tmp_path / "mobile-mutations.sqlite3")
    first = _reserve(store)
    store.complete(first, {"result": {"status": "streaming"}})

    other = store.reserve(
        provider="password",
        subject="different-user",
        client_request_id="request-1",
        method="prompt.submit",
        resource_id="conversation-root",
        semantic_parameters={"text": "hello"},
    )

    assert other.disposition is MutationDisposition.EXECUTE


def test_same_process_duplicate_waits_for_one_authoritative_outcome(tmp_path):
    store = MobileMutationStore(
        tmp_path / "mobile-mutations.sqlite3",
        owner_instance_id="process-a",
    )
    original = _reserve(store)
    duplicate = _reserve(store)
    assert duplicate.disposition is MutationDisposition.IN_PROGRESS

    completed = threading.Event()

    def finish_original():
        store.complete(original, {"result": {"status": "interrupted"}})
        completed.set()

    thread = threading.Thread(target=finish_original)
    thread.start()
    replay = store.wait_for_outcome(duplicate, timeout=2)
    thread.join(timeout=2)

    assert completed.is_set()
    assert replay.disposition is MutationDisposition.REPLAY
    assert replay.outcome == {"result": {"status": "interrupted"}}


def test_status_lookup_does_not_require_a_live_conversation(tmp_path):
    store = MobileMutationStore(tmp_path / "mobile-mutations.sqlite3")
    claim = _reserve(store)
    store.complete(claim, {"result": {"deleted": "conversation-root"}})

    status = store.status(
        provider="password",
        subject="mobile-user",
        client_request_id="request-1",
    )

    assert status == {
        "client_request_id": "request-1",
        "method": "prompt.submit",
        "outcome": {"result": {"deleted": "conversation-root"}},
        "state": "completed",
    }


def test_mobile_interrupt_requires_a_client_request_identity(monkeypatch):
    from tui_gateway import server

    response = server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": "missing-id",
            "method": "session.interrupt",
            "params": {
                "session_id": "live-1",
                "expected_stored_session_id": "conversation-root",
            },
        },
        _MobileTransport("conversation.read", "conversation.control"),
    )

    assert response["error"] == {
        "code": -32602,
        "message": "client_request_id is required for mobile mutation",
        "data": {
            "method": "session.interrupt",
            "reason": "client_request_id_required",
        },
    }


def test_mobile_mutation_rejects_an_oversized_request_identity():
    from tui_gateway import server

    response = server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": "invalid-id",
            "method": "session.interrupt",
            "params": {
                "client_request_id": "x" * 257,
                "expected_stored_session_id": "conversation-root",
                "session_id": "live-1",
            },
        },
        _MobileTransport("conversation.read", "conversation.control"),
    )

    assert response["error"]["code"] == -32602
    assert response["error"]["data"] == {
        "reason": "invalid_client_request_id"
    }



def test_mobile_interrupt_retry_executes_once_and_status_survives_session_removal(
    tmp_path,
    monkeypatch,
):
    from tui_gateway import server

    class Agent:
        interrupt_calls = 0

        def interrupt(self):
            self.interrupt_calls += 1

    class RunThread:
        @staticmethod
        def is_alive():
            return True

    store = MobileMutationStore(
        tmp_path / "mobile-mutations.sqlite3",
        owner_instance_id="process-a",
    )
    monkeypatch.setattr(server, "_mobile_mutation_store", lambda: store)
    monkeypatch.setattr(server, "_clear_pending", lambda _sid: None)
    agent = Agent()
    session = {
        "_run_thread": RunThread(),
        "agent": agent,
        "history_lock": threading.Lock(),
        "profile_home": str(tmp_path),
        "queued_prompt": None,
        "running": True,
        "session_key": "conversation-root",
    }
    server._sessions.clear()
    server._sessions["live-1"] = session
    transport = _MobileTransport("conversation.read", "conversation.control")
    params = {
        "session_id": "live-1",
        "expected_stored_session_id": "conversation-root",
        "client_request_id": "interrupt-1",
    }

    first = server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": "first-correlation",
            "method": "session.interrupt",
            "params": params,
        },
        transport,
    )
    retry = server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": "retry-correlation",
            "method": "session.interrupt",
            "params": params,
        },
        transport,
    )

    assert agent.interrupt_calls == 1
    assert first == {
        "jsonrpc": "2.0",
        "id": "first-correlation",
        "result": {
            "mutation": {
                "client_request_id": "interrupt-1",
                "deduplicated": False,
                "state": "completed",
            },
            "status": "interrupted",
        },
    }
    assert retry == {
        "jsonrpc": "2.0",
        "id": "retry-correlation",
        "result": {
            "mutation": {
                "client_request_id": "interrupt-1",
                "deduplicated": True,
                "state": "completed",
            },
            "status": "interrupted",
        },
    }

    server._sessions.clear()
    replay_without_live_session = server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": "post-removal-correlation",
            "method": "session.interrupt",
            "params": params,
        },
        transport,
    )
    assert replay_without_live_session["result"]["status"] == "interrupted"
    assert (
        replay_without_live_session["result"]["mutation"]["deduplicated"] is True
    )

    status = server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": "status-correlation",
            "method": "mutation.status",
            "params": {"client_request_id": "interrupt-1"},
        },
        transport,
    )

    assert status["result"]["client_request_id"] == "interrupt-1"
    assert status["result"]["method"] == "session.interrupt"
    assert status["result"]["outcome"] == {
        "result": {"status": "interrupted"}
    }
    assert status["result"]["state"] == "completed"


def test_simultaneous_mobile_prompt_submissions_create_one_turn(tmp_path, monkeypatch):
    from tui_gateway import server

    class Db:
        @staticmethod
        def get_compression_lineage(_session_id):
            return ["conversation-root"]

    store = MobileMutationStore(
        tmp_path / "mobile-mutations.sqlite3",
        owner_instance_id="process-a",
    )
    monkeypatch.setattr(server, "_mobile_mutation_store", lambda: store)
    monkeypatch.setattr(
        server,
        "_session_db",
        lambda _session: contextlib.nullcontext(Db()),
    )
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: None)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda _session: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_args: None)
    monkeypatch.setattr(server, "_wait_agent", lambda *_args: None)
    counts = {"inflight": 0, "turn": 0}
    monkeypatch.setattr(
        server,
        "_start_inflight_turn",
        lambda *_args: counts.__setitem__("inflight", counts["inflight"] + 1),
    )

    def run_turn(_rid, _sid, session, _text):
        counts["turn"] += 1
        with session["history_lock"]:
            session["running"] = False

    monkeypatch.setattr(server, "_run_prompt_submit", run_turn)
    session = {
        "agent": object(),
        "history": [],
        "history_lock": threading.Lock(),
        "profile_home": str(tmp_path),
        "queued_prompt": None,
        "running": False,
        "session_key": "conversation-root",
        "transport": None,
    }
    server._sessions.clear()
    server._sessions["live-1"] = session
    transport = _MobileTransport(
        "conversation.read",
        "conversation.write",
        "conversation.control",
    )
    params = {
        "client_request_id": "prompt-1",
        "expected_stored_session_id": "conversation-root",
        "session_id": "live-1",
        "text": "run exactly once",
    }

    original_complete = store.complete
    first_at_completion = threading.Event()
    release_completion = threading.Event()

    def blocking_complete(claim, outcome):
        first_at_completion.set()
        assert release_completion.wait(timeout=2)
        return original_complete(claim, outcome)

    monkeypatch.setattr(store, "complete", blocking_complete)
    responses = {}

    def dispatch(name, correlation):
        responses[name] = server.dispatch(
            {
                "jsonrpc": "2.0",
                "id": correlation,
                "method": "prompt.submit",
                "params": params,
            },
            transport,
        )

    first = threading.Thread(target=dispatch, args=("first", "correlation-1"))
    duplicate = threading.Thread(
        target=dispatch,
        args=("duplicate", "correlation-2"),
    )
    first.start()
    assert first_at_completion.wait(timeout=2)
    duplicate.start()
    release_completion.set()
    first.join(timeout=2)
    duplicate.join(timeout=2)

    assert counts == {"inflight": 1, "turn": 1}
    assert responses["first"]["result"]["mutation"]["deduplicated"] is False
    assert responses["duplicate"]["result"]["mutation"]["deduplicated"] is True
    assert responses["first"]["result"]["status"] == "streaming"
    assert responses["duplicate"]["result"]["status"] == "streaming"


def test_mobile_delete_retry_replays_after_conversation_row_is_gone(
    tmp_path,
    monkeypatch,
):
    from hermes_state import SessionDB
    from tui_gateway import server

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("conversation-delete", source="tui")
    store = MobileMutationStore(tmp_path / "mobile-mutations.sqlite3")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(server, "_mobile_mutation_store", lambda: store)
    server._sessions.clear()
    calls = 0
    original_delete = db.delete_session

    def counted_delete(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_delete(*args, **kwargs)

    monkeypatch.setattr(db, "delete_session", counted_delete)
    transport = _MobileTransport("conversation.read", "conversation.delete")
    params = {
        "client_request_id": "delete-1",
        "session_id": "conversation-delete",
    }

    first = server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": "delete-correlation-1",
            "method": "session.delete",
            "params": params,
        },
        transport,
    )
    retry = server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": "delete-correlation-2",
            "method": "session.delete",
            "params": params,
        },
        transport,
    )

    assert calls == 1
    assert db.get_session("conversation-delete") is None
    assert first["result"]["deleted"] == "conversation-delete"
    assert first["result"]["mutation"]["deduplicated"] is False
    assert retry["result"]["deleted"] == "conversation-delete"
    assert retry["result"]["mutation"]["deduplicated"] is True
    db.close()


def test_mobile_prompt_retry_uses_durable_identity_after_live_id_rotation(
    tmp_path,
    monkeypatch,
):
    from tui_gateway import server

    class Db:
        @staticmethod
        def resolve_resume_session_id(session_id):
            return "conversation-tip" if session_id == "conversation-root" else session_id

        @staticmethod
        def get_compression_lineage(_session_id):
            return ["conversation-root", "conversation-tip"]

    store = MobileMutationStore(tmp_path / "mobile-mutations.sqlite3")
    monkeypatch.setattr(server, "_mobile_mutation_store", lambda: store)
    monkeypatch.setattr(
        server,
        "_session_db",
        lambda _session: contextlib.nullcontext(Db()),
    )
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: None)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda _session: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_args: None)
    monkeypatch.setattr(server, "_wait_agent", lambda *_args: None)
    turns = 0
    turn_done = threading.Event()

    def run_turn(_rid, _sid, session, _text):
        nonlocal turns
        turns += 1
        with session["history_lock"]:
            session["running"] = False
        turn_done.set()

    monkeypatch.setattr(server, "_run_prompt_submit", run_turn)

    def live_session(stored_id):
        return {
            "agent": type("Agent", (), {"session_id": stored_id})(),
            "history": [],
            "history_lock": threading.Lock(),
            "profile_home": str(tmp_path),
            "queued_prompt": None,
            "running": False,
            "session_key": stored_id,
            "transport": None,
        }

    server._sessions.clear()
    server._sessions["live-before"] = live_session("conversation-root")
    transport = _MobileTransport(
        "conversation.read",
        "conversation.write",
        "conversation.control",
    )
    base_params = {
        "client_request_id": "prompt-rotation-1",
        "expected_stored_session_id": "conversation-root",
        "text": "continue exactly once",
    }

    first = server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": "before-rotation",
            "method": "prompt.submit",
            "params": {**base_params, "session_id": "live-before"},
        },
        transport,
    )
    assert turn_done.wait(timeout=2)
    server._sessions.clear()
    server._sessions["live-after"] = live_session("conversation-tip")
    retry = server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": "after-rotation",
            "method": "prompt.submit",
            "params": {**base_params, "session_id": "live-after"},
        },
        transport,
    )

    assert turns == 1
    assert first["result"]["mutation"]["deduplicated"] is False
    assert retry["result"]["mutation"]["deduplicated"] is True


def test_mobile_abandoned_interrupt_is_outcome_unknown_and_never_reexecuted(
    tmp_path,
    monkeypatch,
):
    from tui_gateway import server

    path = tmp_path / "mobile-mutations.sqlite3"
    before_restart = MobileMutationStore(path, owner_instance_id="process-a")
    claim = before_restart.reserve(
        provider="password",
        subject="mobile-user",
        client_request_id="interrupt-unknown",
        method="session.interrupt",
        resource_id="conversation-root",
        semantic_parameters={},
    )
    assert claim.disposition is MutationDisposition.EXECUTE
    before_restart.close()
    after_restart = MobileMutationStore(path, owner_instance_id="process-b")
    monkeypatch.setattr(server, "_mobile_mutation_store", lambda: after_restart)

    class Db:
        @staticmethod
        def get_compression_lineage(_session_id):
            return ["conversation-root"]

    monkeypatch.setattr(
        server,
        "_session_db",
        lambda _session: contextlib.nullcontext(Db()),
    )

    class Agent:
        interrupt_calls = 0

        def interrupt(self):
            self.interrupt_calls += 1

    agent = Agent()
    server._sessions.clear()
    server._sessions["live-unknown"] = {
        "agent": agent,
        "history_lock": threading.Lock(),
        "running": True,
        "session_key": "conversation-root",
    }
    response = server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": "unknown-correlation",
            "method": "session.interrupt",
            "params": {
                "client_request_id": "interrupt-unknown",
                "expected_stored_session_id": "conversation-root",
                "session_id": "live-unknown",
            },
        },
        _MobileTransport("conversation.read", "conversation.control"),
    )

    assert agent.interrupt_calls == 0
    assert response["error"] == {
        "code": 4091,
        "message": "the prior mutation outcome is unknown and will not be re-executed",
        "data": {
            "client_request_id": "interrupt-unknown",
            "reason": "mutation_outcome_unknown",
            "state": "outcome_unknown",
        },
    }
