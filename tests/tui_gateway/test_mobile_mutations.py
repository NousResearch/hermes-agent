"""Durable receipt behavior for consequential mobile JSON-RPC mutations."""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import threading
import time

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


@pytest.fixture(autouse=True)
def _isolated_gateway_mutation_store(tmp_path, monkeypatch):
    """Keep every dispatch in this module off the user's Hermes home."""
    from tui_gateway import server

    store = MobileMutationStore(tmp_path / "gateway-mobile-mutations.sqlite3")
    monkeypatch.setattr(server, "_mobile_mutation_store", lambda: store)
    try:
        yield store
    finally:
        server._sessions.clear()
        store.close()
        assert store._closed is True


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


def test_owned_pre_execution_reservation_can_be_released_and_retried(tmp_path):
    store = MobileMutationStore(tmp_path / "mobile-mutations.sqlite3")
    first = _reserve(store)
    duplicate = _reserve(store)

    assert duplicate.disposition is MutationDisposition.IN_PROGRESS
    assert store.release_before_execution(duplicate) is False
    assert store.release_before_execution(first) is True
    assert store.status(
        provider="password",
        subject="mobile-user",
        client_request_id="request-1",
    ) is None
    assert _reserve(store).disposition is MutationDisposition.EXECUTE
    store.close()


def test_advertised_mutations_share_one_policy_and_resource_authority():
    from tui_gateway import mobile_contract

    payload = mobile_contract.gateway_ready_payload(skin="default")
    advertised = set(
        payload["capabilities"]["mutation.idempotency"]["methods"]
    )
    described = {
        method
        for method, policy in mobile_contract.MOBILE_METHOD_POLICIES.items()
        if policy.mutation is not None
    }

    assert advertised == set(mobile_contract.MOBILE_MUTATION_METHODS)
    assert advertised == described
    for method in advertised:
        policy = mobile_contract.MOBILE_METHOD_POLICIES[method]
        descriptor = policy.mutation
        assert descriptor is not None
        assert descriptor.resource_parameter in policy.allowed_parameters
        assert set(descriptor.semantic_parameters) <= policy.allowed_parameters


def test_transient_lineage_failure_releases_receipt_for_identical_retry(
    _isolated_gateway_mutation_store,
    monkeypatch,
):
    from tui_gateway import server

    outage = {"active": True}

    class Db:
        @staticmethod
        def get_compression_lineage(_session_id):
            return ["conversation-root", "conversation-tip"]

    @contextlib.contextmanager
    def session_db(_session):
        if outage["active"]:
            raise sqlite3.OperationalError("lineage store unavailable")
        yield Db()

    class Agent:
        session_id = "conversation-tip"
        interrupt_calls = 0

        def interrupt(self):
            self.interrupt_calls += 1

    agent = Agent()
    monkeypatch.setattr(server, "_session_db", session_db)
    monkeypatch.setattr(server, "_clear_pending", lambda _sid: None)
    server._sessions["live-1"] = {
        "agent": agent,
        "history_lock": threading.Lock(),
        "queued_prompt": None,
        "running": True,
        "session_key": "conversation-tip",
    }
    request = {
        "jsonrpc": "2.0",
        "id": "first-correlation",
        "method": "session.interrupt",
        "params": {
            "client_request_id": "retry-after-lineage-outage",
            "expected_stored_session_id": "conversation-root",
            "session_id": "live-1",
        },
    }
    transport = _MobileTransport("conversation.read", "conversation.control")

    unavailable = server.dispatch(request, transport)

    assert unavailable["error"] == {
        "code": 5037,
        "message": "mobile mutation preflight is temporarily unavailable",
        "data": {
            "client_request_id": "retry-after-lineage-outage",
            "reason": "mutation_preflight_unavailable",
        },
    }
    assert _isolated_gateway_mutation_store.status(
        provider="password",
        subject="mobile-user",
        client_request_id="retry-after-lineage-outage",
    ) is None

    outage["active"] = False
    request["id"] = "retry-correlation"
    retry = server.dispatch(request, transport)

    assert agent.interrupt_calls == 1
    assert retry["result"]["status"] == "interrupted"
    assert retry["result"]["mutation"] == {
        "client_request_id": "retry-after-lineage-outage",
        "deduplicated": False,
        "state": "completed",
    }


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


@pytest.mark.parametrize(
    "failure",
    [sqlite3.OperationalError("database unavailable"), RuntimeError("unexpected")],
)
def test_receipt_reservation_failures_return_structured_unavailable_errors(
    _isolated_gateway_mutation_store,
    monkeypatch,
    caplog,
    failure,
):
    from tui_gateway import server

    def fail_reservation(**_kwargs):
        raise failure

    monkeypatch.setattr(
        _isolated_gateway_mutation_store,
        "reserve",
        fail_reservation,
    )
    with caplog.at_level(logging.WARNING, logger="tui_gateway.server"):
        response = server.dispatch(
            {
                "jsonrpc": "2.0",
                "id": "store-outage",
                "method": "session.interrupt",
                "params": {
                    "client_request_id": "store-outage-1",
                    "expected_stored_session_id": "conversation-root",
                    "session_id": "live-1",
                },
            },
            _MobileTransport("conversation.read", "conversation.control"),
        )

    assert response["error"] == {
        "code": 5037,
        "message": "durable mutation receipt is temporarily unavailable",
        "data": {"reason": "mutation_store_unavailable"},
    }
    assert any(record.exc_info is not None for record in caplog.records)


@pytest.mark.parametrize(
    "failure",
    [sqlite3.OperationalError("database unavailable"), RuntimeError("unexpected")],
)
def test_receipt_wait_failures_return_structured_unavailable_errors(
    _isolated_gateway_mutation_store,
    monkeypatch,
    caplog,
    failure,
):
    from tui_gateway import server

    original = _isolated_gateway_mutation_store.reserve(
        provider="password",
        subject="mobile-user",
        client_request_id="wait-outage-1",
        method="session.interrupt",
        resource_id="conversation-root",
        semantic_parameters={},
    )
    assert original.disposition is MutationDisposition.EXECUTE

    def fail_wait(_claim, *, timeout):
        assert timeout == 30.0
        raise failure

    monkeypatch.setattr(
        _isolated_gateway_mutation_store,
        "wait_for_outcome",
        fail_wait,
    )
    with caplog.at_level(logging.WARNING, logger="tui_gateway.server"):
        response = server.dispatch(
            {
                "jsonrpc": "2.0",
                "id": "wait-outage",
                "method": "session.interrupt",
                "params": {
                    "client_request_id": "wait-outage-1",
                    "expected_stored_session_id": "conversation-root",
                    "session_id": "live-1",
                },
            },
            _MobileTransport("conversation.read", "conversation.control"),
        )

    assert response["error"] == {
        "code": 5037,
        "message": "durable mutation receipt is temporarily unavailable",
        "data": {"reason": "mutation_store_unavailable"},
    }
    assert any(record.exc_info is not None for record in caplog.records)


@pytest.mark.parametrize(
    "failure",
    [sqlite3.OperationalError("database unavailable"), RuntimeError("unexpected")],
)
def test_receipt_status_failures_return_structured_unavailable_errors(
    _isolated_gateway_mutation_store,
    monkeypatch,
    caplog,
    failure,
):
    from tui_gateway import server

    def fail_status(**_kwargs):
        raise failure

    monkeypatch.setattr(_isolated_gateway_mutation_store, "status", fail_status)
    with caplog.at_level(logging.WARNING, logger="tui_gateway.server"):
        response = server.dispatch(
            {
                "jsonrpc": "2.0",
                "id": "status-outage",
                "method": "mutation.status",
                "params": {"client_request_id": "status-outage-1"},
            },
            _MobileTransport("conversation.read"),
        )

    assert response["error"] == {
        "code": 5037,
        "message": "durable mutation receipt is temporarily unavailable",
        "data": {"reason": "mutation_store_unavailable"},
    }
    assert any(record.exc_info is not None for record in caplog.records)


@pytest.mark.parametrize(
    "failure",
    [sqlite3.OperationalError("database unavailable"), RuntimeError("unexpected")],
)
def test_receipt_completion_failures_become_structured_unknown_outcomes(
    _isolated_gateway_mutation_store,
    monkeypatch,
    caplog,
    failure,
):
    from tui_gateway import server

    class Agent:
        interrupt_calls = 0

        def interrupt(self):
            self.interrupt_calls += 1

    agent = Agent()
    monkeypatch.setattr(server, "_clear_pending", lambda _sid: None)
    server._sessions["live-1"] = {
        "agent": agent,
        "history_lock": threading.Lock(),
        "queued_prompt": None,
        "running": True,
        "session_key": "conversation-root",
    }
    original_complete = _isolated_gateway_mutation_store.complete

    def fail_completion(_claim, _outcome):
        raise failure

    monkeypatch.setattr(
        _isolated_gateway_mutation_store,
        "complete",
        fail_completion,
    )
    request = {
        "jsonrpc": "2.0",
        "id": "completion-outage",
        "method": "session.interrupt",
        "params": {
            "client_request_id": "completion-outage-1",
            "expected_stored_session_id": "conversation-root",
            "session_id": "live-1",
        },
    }
    transport = _MobileTransport("conversation.read", "conversation.control")

    with caplog.at_level(logging.WARNING, logger="tui_gateway.server"):
        response = server.dispatch(request, transport)

    assert response["error"] == {
        "code": 5037,
        "message": "mutation outcome could not be recorded durably",
        "data": {"reason": "mutation_outcome_unknown"},
    }
    assert _isolated_gateway_mutation_store.status(
        provider="password",
        subject="mobile-user",
        client_request_id="completion-outage-1",
    )["state"] == "outcome_unknown"
    assert agent.interrupt_calls == 1
    assert any(record.exc_info is not None for record in caplog.records)

    monkeypatch.setattr(
        _isolated_gateway_mutation_store,
        "complete",
        original_complete,
    )
    request["id"] = "completion-outage-retry"
    retry = server.dispatch(request, transport)

    assert retry["error"]["code"] == 4091
    assert agent.interrupt_calls == 1


def test_unexpected_handler_failure_returns_structured_unknown_outcome(
    _isolated_gateway_mutation_store,
    monkeypatch,
    caplog,
):
    from tui_gateway import server

    class Agent:
        interrupt_calls = 0

        def interrupt(self):
            self.interrupt_calls += 1
            raise RuntimeError("interrupt transport failed")

    agent = Agent()
    server._sessions["live-1"] = {
        "agent": agent,
        "history_lock": threading.Lock(),
        "queued_prompt": None,
        "running": True,
        "session_key": "conversation-root",
    }
    request = {
        "jsonrpc": "2.0",
        "id": "handler-failure",
        "method": "session.interrupt",
        "params": {
            "client_request_id": "handler-failure-1",
            "expected_stored_session_id": "conversation-root",
            "session_id": "live-1",
        },
    }
    transport = _MobileTransport("conversation.read", "conversation.control")

    with caplog.at_level(logging.ERROR, logger="tui_gateway.server"):
        response = server.dispatch(request, transport)

    assert response["error"] == {
        "code": 5037,
        "message": "mutation handler failed before a durable outcome was available",
        "data": {"reason": "mutation_outcome_unknown"},
    }
    assert _isolated_gateway_mutation_store.status(
        provider="password",
        subject="mobile-user",
        client_request_id="handler-failure-1",
    )["state"] == "outcome_unknown"
    assert agent.interrupt_calls == 1
    assert any(record.exc_info is not None for record in caplog.records)

    request["id"] = "handler-failure-retry"
    retry = server.dispatch(request, transport)

    assert retry["error"]["code"] == 4091
    assert agent.interrupt_calls == 1



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
    worker_started = threading.Event()
    release_worker = threading.Event()
    inner_workers = []
    monkeypatch.setattr(
        server,
        "_start_inflight_turn",
        lambda *_args: counts.__setitem__("inflight", counts["inflight"] + 1),
    )

    def run_turn(_rid, _sid, session, text):
        counts["turn"] += 1

        def persist_turn():
            worker_started.set()
            assert release_worker.wait(timeout=2)
            with session["history_lock"]:
                receipt_tags = list(
                    session.get("_pending_mobile_mutation_receipt_tags") or ()
                )
                assert len(receipt_tags) == 1
                session["history"].append(
                    {
                        "role": "user",
                        "content": text,
                        "_db_persisted": True,
                        "_mobile_mutation_receipt_tags": receipt_tags,
                    }
                )
                session["running"] = False

        # Match the production topology: prompt.submit first runs an agent-ready
        # wrapper, then _run_prompt_submit replaces its handle with the real turn
        # worker. Receipt monitoring must follow that live handle replacement.
        worker = threading.Thread(target=persist_turn)
        inner_workers.append(worker)
        session["_run_thread"] = worker
        worker.start()

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
    assert worker_started.wait(timeout=2)
    first.join(timeout=2)
    assert not first.is_alive()
    assert store.status(
        provider="password",
        subject="mobile-user",
        client_request_id="prompt-1",
    )["state"] == "in_progress"

    release_worker.set()
    assert first_at_completion.wait(timeout=2)
    duplicate.start()
    release_completion.set()
    duplicate.join(timeout=2)
    inner_workers[0].join(timeout=2)

    assert counts == {"inflight": 1, "turn": 1}
    assert responses["first"]["result"]["mutation"]["deduplicated"] is False
    assert responses["first"]["result"]["mutation"]["state"] == "in_progress"
    assert responses["duplicate"]["result"]["mutation"]["deduplicated"] is True
    assert responses["duplicate"]["result"]["mutation"]["state"] == "completed"
    assert responses["first"]["result"]["status"] == "streaming"
    assert responses["duplicate"]["result"]["status"] == "streaming"


def test_mobile_prompt_proof_tag_is_scoped_to_authenticated_principal(tmp_path):
    from tui_gateway import server

    store = MobileMutationStore(tmp_path / "mobile-mutations.sqlite3")
    shared = {
        "client_request_id": "same-request-id",
        "method": "prompt.submit",
        "resource_id": "conversation-root",
        "semantic_parameters": {"text": "same text"},
    }
    alice = store.reserve(provider="password", subject="alice", **shared)
    bob = store.reserve(provider="password", subject="bob", **shared)
    session = {
        "agent": object(),
        "history": [
            {
                "role": "user",
                "content": "same text",
                "_db_persisted": True,
                "_mobile_mutation_receipt_tags": [alice.proof_tag],
            }
        ],
        "history_lock": threading.Lock(),
    }

    assert alice.proof_tag != bob.proof_tag
    assert server._mobile_prompt_turn_is_durable(
        ("live-1", session, alice.proof_tag)
    )
    assert not server._mobile_prompt_turn_is_durable(
        ("live-1", session, bob.proof_tag)
    )
    store.close()


def test_mobile_prompt_durable_proof_survives_user_sequence_repair():
    from run_agent import AIAgent
    from tui_gateway import server

    proof_tag = "principal-scoped-proof"
    messages = [
        {"role": "user", "content": "interrupted", "_db_persisted": True},
        {
            "role": "user",
            "content": "mobile send",
            "_db_persisted": True,
            "_mobile_mutation_receipt_tags": [proof_tag],
        },
    ]
    session = {
        "agent": object(),
        "history": messages,
        "history_lock": threading.Lock(),
    }
    context = ("live-1", session, proof_tag)

    assert server._mobile_prompt_turn_is_durable(context)
    AIAgent._repair_message_sequence(AIAgent.__new__(AIAgent), messages)
    assert len(messages) == 1
    assert server._mobile_prompt_turn_is_durable(context)


def test_mobile_prompt_receipt_rechecks_proof_after_worker_exit_race(
    _isolated_gateway_mutation_store,
    monkeypatch,
):
    from tui_gateway import server

    claim = _isolated_gateway_mutation_store.reserve(
        provider="password",
        subject="mobile-user",
        client_request_id="persist-at-exit-1",
        method="prompt.submit",
        resource_id="conversation-root",
        semantic_parameters={"text": "persist at exit"},
    )
    session = {
        "agent": object(),
        "history": [],
        "history_lock": threading.Lock(),
        "running": True,
    }
    context = ("live-1", session, claim.proof_tag)

    def persist_while_observing_worker_exit(_context):
        with session["history_lock"]:
            session["history"].append(
                {
                    "role": "user",
                    "content": "persist at exit",
                    "_db_persisted": True,
                    "_mobile_mutation_receipt_tags": [claim.proof_tag],
                }
            )
            session["running"] = False
        return False

    monkeypatch.setattr(
        server,
        "_mobile_prompt_turn_is_pending",
        persist_while_observing_worker_exit,
    )
    assert server._defer_mobile_prompt_receipt(
        store=_isolated_gateway_mutation_store,
        claim=claim,
        outcome={"result": {"status": "streaming"}},
        context=context,
    )

    deadline = time.monotonic() + 1
    status = None
    while time.monotonic() < deadline:
        status = _isolated_gateway_mutation_store.status(
            provider="password",
            subject="mobile-user",
            client_request_id="persist-at-exit-1",
        )
        if status and status["state"] == "completed":
            break
        time.sleep(0.01)
    assert status is not None and status["state"] == "completed"


def test_mobile_prompt_receipts_share_one_constant_time_session_monitor(
    _isolated_gateway_mutation_store,
    monkeypatch,
):
    from tui_gateway import server

    claims = [
        _isolated_gateway_mutation_store.reserve(
            provider="password",
            subject="mobile-user",
            client_request_id=f"shared-monitor-{index}",
            method="prompt.submit",
            resource_id="conversation-root",
            semantic_parameters={"text": f"prompt {index}"},
        )
        for index in (1, 2)
    ]

    class WorkThread:
        alive = True

        def is_alive(self):
            return self.alive

    work_thread = WorkThread()
    proof_tags = [claim.proof_tag for claim in claims]
    session = {
        "agent": object(),
        "history": [],
        "history_lock": threading.Lock(),
        "running": True,
        "_pending_mobile_mutation_receipt_tags": proof_tags,
        "_live_mobile_mutation_receipt_tags": set(proof_tags),
        "_run_thread": work_thread,
    }
    contexts = [("live-1", session, tag) for tag in proof_tags]
    original_durable_check = server._mobile_prompt_turn_is_durable
    hot_path_scans = 0

    def count_hot_path_scan(context):
        nonlocal hot_path_scans
        hot_path_scans += 1
        return original_durable_check(context)

    monkeypatch.setattr(
        server,
        "_mobile_prompt_turn_is_durable",
        count_hot_path_scan,
    )
    for claim, context in zip(claims, contexts):
        assert server._defer_mobile_prompt_receipt(
            store=_isolated_gateway_mutation_store,
            claim=claim,
            outcome={"result": {"status": "queued"}},
            context=context,
        )

    monitor = session["_mobile_prompt_receipt_monitor_thread"]
    time.sleep(0.3)
    assert session["_mobile_prompt_receipt_monitor_thread"] is monitor
    assert len(session["_mobile_prompt_receipt_entries"]) == 2
    assert hot_path_scans == 0

    with session["history_lock"]:
        work_thread.alive = False
        session["running"] = False
        session.pop("_pending_mobile_mutation_receipt_tags", None)
    server._notify_mobile_prompt_receipt_change(session)

    deadline = time.monotonic() + 1
    statuses = []
    while time.monotonic() < deadline:
        statuses = [
            _isolated_gateway_mutation_store.status(
                provider="password",
                subject="mobile-user",
                client_request_id=f"shared-monitor-{index}",
            )
            for index in (1, 2)
        ]
        if all(status and status["state"] == "outcome_unknown" for status in statuses):
            break
        time.sleep(0.01)
    assert all(
        status is not None and status["state"] == "outcome_unknown"
        for status in statuses
    )
    assert hot_path_scans == 2


def test_mobile_prompt_receipt_follows_real_turn_context_and_session_db(
    _isolated_gateway_mutation_store,
    tmp_path,
    monkeypatch,
):
    from agent.turn_context import build_turn_context
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tui_gateway import server

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    db = SessionDB(db_path=tmp_path / "state.db")
    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        session_db=db,
        session_id="conversation-root",
        skip_context_files=True,
        skip_memory=True,
    )
    agent._cached_system_prompt = "SYSTEM"
    agent.compression_enabled = False
    agent._skip_mcp_refresh = True
    turn_done = threading.Event()
    turns = 0

    class RuntimeAdapter:
        @staticmethod
        def _set_interrupt(_value, _thread_id):
            return None

    def run_turn(_rid, _sid, session, text):
        nonlocal turns
        turns += 1
        with session["history_lock"]:
            receipt_tags = list(
                session.pop("_pending_mobile_mutation_receipt_tags", ()) or ()
            )
            session["_active_mobile_mutation_receipt_tags"] = receipt_tags
            history = list(session["history"])
        agent._pending_mobile_mutation_receipt_tags = receipt_tags
        context = build_turn_context(
            agent=agent,
            user_message=text,
            system_message=None,
            conversation_history=history,
            task_id=None,
            stream_callback=None,
            persist_user_message=None,
            restore_or_build_system_prompt=lambda *_args, **_kwargs: None,
            install_safe_stdio=lambda: None,
            sanitize_surrogates=lambda value: value,
            summarize_user_message_for_log=lambda value: value,
            set_session_context=lambda _session_id: None,
            set_current_write_origin=lambda _origin: None,
            ra=lambda: RuntimeAdapter,
        )
        agent._session_messages = context.messages
        with session["history_lock"]:
            session["history"] = context.messages
            session.pop("_active_mobile_mutation_receipt_tags", None)
            session["running"] = False
        turn_done.set()

    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: None)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda _session: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_args: None)
    monkeypatch.setattr(server, "_wait_agent", lambda *_args: None)
    monkeypatch.setattr(server, "_run_prompt_submit", run_turn)
    session = {
        "agent": agent,
        "history": [],
        "history_lock": threading.Lock(),
        "profile_home": str(tmp_path),
        "queued_prompt": None,
        "running": False,
        "session_key": "conversation-root",
        "transport": None,
    }
    server._sessions["live-1"] = session
    transport = _MobileTransport("conversation.read", "conversation.write")
    request = {
        "jsonrpc": "2.0",
        "id": "real-path-first",
        "method": "prompt.submit",
        "params": {
            "client_request_id": "real-path-1",
            "expected_stored_session_id": "conversation-root",
            "session_id": "live-1",
            "text": "persist through the real path",
        },
    }

    first = server.dispatch(request, transport)
    assert first["result"]["mutation"]["state"] == "in_progress"
    assert turn_done.wait(timeout=2)
    deadline = time.monotonic() + 2
    status = None
    while time.monotonic() < deadline:
        status = _isolated_gateway_mutation_store.status(
            provider="password",
            subject="mobile-user",
            client_request_id="real-path-1",
        )
        if status and status["state"] == "completed":
            break
        time.sleep(0.01)

    assert status is not None and status["state"] == "completed"
    rows = db.get_messages("conversation-root")
    assert [(row["role"], row["content"]) for row in rows] == [
        ("user", "persist through the real path")
    ]
    assert session["history"][-1]["_db_persisted"] is True
    assert session["history"][-1]["_mobile_mutation_receipt_tags"]

    request["id"] = "real-path-retry"
    retry = server.dispatch(request, transport)
    assert retry["result"]["mutation"]["deduplicated"] is True
    assert turns == 1
    db.close()


def test_mobile_prompt_receipt_completes_only_after_durable_turn(
    _isolated_gateway_mutation_store,
    monkeypatch,
):
    from tui_gateway import server

    class Db:
        @staticmethod
        def get_compression_lineage(_session_id):
            return ["conversation-root"]

    class Agent:
        @staticmethod
        def steer(_text):
            raise AssertionError("durable mobile prompts must queue, not steer")

    monkeypatch.setattr(
        server,
        "_session_db",
        lambda _session: contextlib.nullcontext(Db()),
    )
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "steer")
    entered = threading.Event()
    release = threading.Event()
    delivered = 0
    session = {
        "agent": Agent(),
        "history": [],
        "history_lock": threading.Lock(),
        "profile_home": None,
        "queued_prompt": None,
        "running": True,
        "session_key": "conversation-root",
        "transport": None,
    }

    def finish_active_turn():
        nonlocal delivered
        entered.set()
        assert release.wait(timeout=2)
        with session["history_lock"]:
            queued = session["queued_prompt"]
            assert queued is not None
            delivered += 1
            session["history"].append(
                {
                    "role": "user",
                    "content": queued["text"],
                    "_db_persisted": True,
                    "_mobile_mutation_receipt_tags": list(
                        queued["mobile_mutation_receipt_tags"]
                    ),
                }
            )
            session["queued_prompt"] = None
            session["running"] = False

    active_turn = threading.Thread(target=finish_active_turn)
    session["_run_thread"] = active_turn
    server._sessions["live-1"] = session
    active_turn.start()
    assert entered.wait(timeout=1)
    transport = _MobileTransport(
        "conversation.read",
        "conversation.write",
        "conversation.control",
    )
    request = {
        "jsonrpc": "2.0",
        "id": "prompt-first",
        "method": "prompt.submit",
        "params": {
            "client_request_id": "prompt-durable-1",
            "expected_stored_session_id": "conversation-root",
            "session_id": "live-1",
            "text": "persist this once",
        },
    }

    first = server.dispatch(request, transport)
    assert first["result"]["status"] == "queued"
    assert first["result"]["mutation"] == {
        "client_request_id": "prompt-durable-1",
        "deduplicated": False,
        "state": "in_progress",
    }
    assert _isolated_gateway_mutation_store.status(
        provider="password",
        subject="mobile-user",
        client_request_id="prompt-durable-1",
    )["state"] == "in_progress"

    release.set()
    deadline = time.monotonic() + 1
    status = None
    while time.monotonic() < deadline:
        status = _isolated_gateway_mutation_store.status(
            provider="password",
            subject="mobile-user",
            client_request_id="prompt-durable-1",
        )
        if status and status["state"] == "completed":
            break
        time.sleep(0.01)
    assert status is not None and status["state"] == "completed"

    request["id"] = "prompt-retry"
    retry = server.dispatch(request, transport)
    assert delivered == 1
    assert retry["result"]["status"] == "queued"
    assert retry["result"]["mutation"]["deduplicated"] is True


def test_interrupting_queued_mobile_prompt_terminalizes_its_receipt(
    _isolated_gateway_mutation_store,
    monkeypatch,
):
    from tui_gateway import server

    class Db:
        @staticmethod
        def get_compression_lineage(_session_id):
            return ["conversation-root"]

    class Agent:
        interrupted = 0

        def interrupt(self):
            self.interrupted += 1

    class LiveThread:
        @staticmethod
        def is_alive():
            return True

    monkeypatch.setattr(
        server,
        "_session_db",
        lambda _session: contextlib.nullcontext(Db()),
    )
    agent = Agent()
    session = {
        "agent": agent,
        "history": [],
        "history_lock": threading.Lock(),
        "profile_home": None,
        "queued_prompt": None,
        "running": True,
        "session_key": "conversation-root",
        "transport": None,
        "_run_thread": LiveThread(),
    }
    server._sessions["live-1"] = session
    transport = _MobileTransport(
        "conversation.read",
        "conversation.write",
        "conversation.control",
    )
    prompt = server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": "queued-prompt",
            "method": "prompt.submit",
            "params": {
                "client_request_id": "queued-prompt-1",
                "expected_stored_session_id": "conversation-root",
                "session_id": "live-1",
                "text": "cancel before delivery",
            },
        },
        transport,
    )
    assert prompt["result"]["status"] == "queued"
    assert prompt["result"]["mutation"]["state"] == "in_progress"
    assert session["queued_prompt"] is not None

    interrupted = server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": "interrupt-queued-prompt",
            "method": "session.interrupt",
            "params": {
                "client_request_id": "interrupt-queued-prompt-1",
                "expected_stored_session_id": "conversation-root",
                "session_id": "live-1",
            },
        },
        transport,
    )
    assert interrupted["result"]["status"] == "interrupted"
    assert interrupted["result"]["mutation"]["state"] == "completed"
    assert agent.interrupted == 1
    assert session["queued_prompt"] is None

    deadline = time.monotonic() + 1
    status = None
    while time.monotonic() < deadline:
        status = _isolated_gateway_mutation_store.status(
            provider="password",
            subject="mobile-user",
            client_request_id="queued-prompt-1",
        )
        if status and status["state"] == "outcome_unknown":
            break
        time.sleep(0.01)
    assert status is not None and status["state"] == "outcome_unknown"
    assert "_live_mobile_mutation_receipt_tags" not in session


def test_mobile_prompt_without_durable_history_becomes_outcome_unknown(
    _isolated_gateway_mutation_store,
    monkeypatch,
):
    from tui_gateway import server

    class Db:
        @staticmethod
        def get_compression_lineage(_session_id):
            return ["conversation-root"]

    monkeypatch.setattr(
        server,
        "_session_db",
        lambda _session: contextlib.nullcontext(Db()),
    )
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: None)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda _session: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_args: None)
    monkeypatch.setattr(server, "_wait_agent", lambda *_args: None)
    turn_done = threading.Event()
    turns = 0

    def run_turn(_rid, _sid, session, _text):
        nonlocal turns
        turns += 1
        with session["history_lock"]:
            session.pop("_pending_mobile_mutation_receipt_tags", None)
            session["history"].append(
                {
                    "role": "user",
                    "content": "banana",
                    "_db_persisted": True,
                    "_mobile_mutation_receipt_tags": ["different-proof"],
                }
            )
            session["running"] = False
        turn_done.set()

    monkeypatch.setattr(server, "_run_prompt_submit", run_turn)
    server._sessions["live-1"] = {
        "agent": object(),
        "history": [],
        "history_lock": threading.Lock(),
        "profile_home": None,
        "queued_prompt": None,
        "running": False,
        "session_key": "conversation-root",
        "transport": None,
    }
    transport = _MobileTransport(
        "conversation.read",
        "conversation.write",
        "conversation.control",
    )
    request = {
        "jsonrpc": "2.0",
        "id": "prompt-undurable",
        "method": "prompt.submit",
        "params": {
            "client_request_id": "prompt-undurable-1",
            "expected_stored_session_id": "conversation-root",
            "session_id": "live-1",
            "text": "a",
        },
    }

    first = server.dispatch(request, transport)
    assert turn_done.wait(timeout=1)
    assert first["result"]["mutation"]["state"] == "in_progress"
    deadline = time.monotonic() + 1
    status = None
    while time.monotonic() < deadline:
        status = _isolated_gateway_mutation_store.status(
            provider="password",
            subject="mobile-user",
            client_request_id="prompt-undurable-1",
        )
        if status and status["state"] == "outcome_unknown":
            break
        time.sleep(0.01)
    assert status is not None and status["state"] == "outcome_unknown"

    request["id"] = "prompt-undurable-retry"
    retry = server.dispatch(request, transport)
    assert retry["error"]["code"] == 4091
    assert turns == 1


@pytest.mark.parametrize(
    "exit_mode",
    ["agent_init_failure", "pre_worker_cancel", "worker_setup_failure"],
)
def test_mobile_prompt_pre_worker_exit_terminalizes_receipt(
    _isolated_gateway_mutation_store,
    monkeypatch,
    exit_mode,
):
    from tui_gateway import server

    class Db:
        @staticmethod
        def get_compression_lineage(_session_id):
            return ["conversation-root"]

    monkeypatch.setattr(
        server,
        "_session_db",
        lambda _session: contextlib.nullcontext(Db()),
    )
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: None)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda _session: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_args: None)
    def wait_agent(session, *_args):
        if exit_mode == "pre_worker_cancel":
            session["_turn_cancel_requested"] = True
            return None
        if exit_mode == "agent_init_failure":
            return {"error": {"message": "agent initialization failed"}}
        return None

    monkeypatch.setattr(server, "_wait_agent", wait_agent)
    if exit_mode == "worker_setup_failure":
        monkeypatch.setattr(
            server,
            "_emit",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("transport closed before worker start")
            ),
        )
    else:
        monkeypatch.setattr(
            server,
            "_run_prompt_submit",
            lambda *_args: pytest.fail("turn worker must not start"),
        )
    session = {
        "agent": object(),
        "history": [],
        "history_lock": threading.Lock(),
        "profile_home": None,
        "queued_prompt": None,
        "running": False,
        "session_key": "conversation-root",
        "transport": None,
    }
    server._sessions["live-1"] = session
    transport = _MobileTransport("conversation.read", "conversation.write")
    client_request_id = f"prompt-{exit_mode}-1"
    request = {
        "jsonrpc": "2.0",
        "id": f"prompt-{exit_mode}",
        "method": "prompt.submit",
        "params": {
            "client_request_id": client_request_id,
            "expected_stored_session_id": "conversation-root",
            "session_id": "live-1",
            "text": "never started",
        },
    }

    first = server.dispatch(request, transport)
    assert first["result"]["mutation"]["state"] == "in_progress"
    deadline = time.monotonic() + 1
    status = None
    while time.monotonic() < deadline:
        status = _isolated_gateway_mutation_store.status(
            provider="password",
            subject="mobile-user",
            client_request_id=client_request_id,
        )
        if status and status["state"] == "outcome_unknown":
            break
        time.sleep(0.01)

    assert status is not None and status["state"] == "outcome_unknown"
    assert "_pending_mobile_mutation_receipt_tags" not in session
    request["id"] = f"prompt-{exit_mode}-retry"
    retry = server.dispatch(request, transport)
    assert retry["error"]["code"] == 4091


def test_blocked_context_clears_unconsumed_mobile_receipt_tag(
    tmp_path,
    monkeypatch,
):
    from agent import context_references, model_metadata
    from tui_gateway import server

    class Agent:
        api_key = "test-key"
        api_mode = "openai_chat"
        base_url = ""
        model = "test/model"
        provider = "test"
        _config_context_length = None
        _pending_mobile_mutation_receipt_tags = []

        @staticmethod
        def run_conversation(*_args, **_kwargs):
            raise AssertionError("blocked context must not reach the agent")

    class ImmediateThread:
        def __init__(self, target=None, daemon=None):
            self._target = target

        def start(self):
            self._target()

        @staticmethod
        def is_alive():
            return False

    monkeypatch.setattr(server.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(
        server,
        "_sync_agent_model_with_config",
        lambda _sid, _session: None,
    )
    monkeypatch.setattr(server, "_session_cwd", lambda _session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    monkeypatch.setattr(server, "make_stream_renderer", lambda _cols: None)
    monkeypatch.setattr(server, "_session_info", lambda *_args: {})
    monkeypatch.setattr(
        model_metadata,
        "get_model_context_length",
        lambda *_args, **_kwargs: 100_000,
    )
    monkeypatch.setattr(
        context_references,
        "preprocess_context_references",
        lambda *_args, **_kwargs: type(
            "BlockedContext",
            (),
            {
                "blocked": True,
                "message": "",
                "warnings": ["Context injection refused."],
            },
        )(),
    )
    agent = Agent()
    proof_tag = "principal-scoped-proof"
    session = {
        "agent": agent,
        "attached_images": [],
        "cols": 80,
        "history": [],
        "history_lock": threading.Lock(),
        "profile_home": None,
        "queued_prompt": None,
        "running": True,
        "session_key": "conversation-root",
        "_pending_mobile_mutation_receipt_tags": [proof_tag],
    }

    server._run_prompt_submit("rid", "live-1", session, "@blocked")

    assert session["running"] is False
    assert "_pending_mobile_mutation_receipt_tags" not in session
    assert "_active_mobile_mutation_receipt_tags" not in session
    assert agent._pending_mobile_mutation_receipt_tags == []
    assert not server._mobile_prompt_turn_is_pending(
        ("live-1", session, proof_tag)
    )


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

    def run_turn(_rid, _sid, session, text):
        nonlocal turns
        turns += 1
        with session["history_lock"]:
            receipt_tags = list(
                session.get("_pending_mobile_mutation_receipt_tags") or ()
            )
            assert len(receipt_tags) == 1
            session["history"].append(
                {
                    "role": "user",
                    "content": text,
                    "_db_persisted": True,
                    "_mobile_mutation_receipt_tags": receipt_tags,
                }
            )
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
