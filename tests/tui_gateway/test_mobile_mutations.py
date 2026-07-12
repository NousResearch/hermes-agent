"""Durable receipt behavior for consequential mobile JSON-RPC mutations."""

from __future__ import annotations

import contextlib
import logging
import sqlite3
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


def test_waiting_duplicate_re_reserves_after_preflight_release(
    _isolated_gateway_mutation_store,
    monkeypatch,
):
    from tui_gateway import server

    first_preflight = threading.Event()
    release_first_preflight = threading.Event()
    duplicate_waiting = threading.Event()
    preflight_calls = 0
    preflight_lock = threading.Lock()

    class Db:
        @staticmethod
        def get_compression_lineage(_session_id):
            return ["conversation-root", "conversation-tip"]

    @contextlib.contextmanager
    def session_db(_session):
        nonlocal preflight_calls
        with preflight_lock:
            preflight_calls += 1
            call = preflight_calls
        if call == 1:
            first_preflight.set()
            assert release_first_preflight.wait(timeout=2)
            raise sqlite3.OperationalError("lineage store unavailable")
        yield Db()

    original_wait = _isolated_gateway_mutation_store.wait_for_outcome

    def observed_wait(claim, *, timeout):
        duplicate_waiting.set()
        return original_wait(claim, timeout=timeout)

    class Agent:
        session_id = "conversation-tip"
        interrupt_calls = 0

        def interrupt(self):
            self.interrupt_calls += 1

    agent = Agent()
    monkeypatch.setattr(server, "_session_db", session_db)
    monkeypatch.setattr(server, "_clear_pending", lambda _sid: None)
    monkeypatch.setattr(
        _isolated_gateway_mutation_store,
        "wait_for_outcome",
        observed_wait,
    )
    server._sessions["live-1"] = {
        "agent": agent,
        "history_lock": threading.Lock(),
        "queued_prompt": None,
        "running": True,
        "session_key": "conversation-tip",
    }
    request = {
        "jsonrpc": "2.0",
        "method": "session.interrupt",
        "params": {
            "client_request_id": "waiter-after-release",
            "expected_stored_session_id": "conversation-root",
            "session_id": "live-1",
        },
    }
    transport = _MobileTransport("conversation.read", "conversation.control")
    responses = {}

    def dispatch(name, correlation):
        responses[name] = server.dispatch(
            {**request, "id": correlation},
            transport,
        )

    owner = threading.Thread(target=dispatch, args=("owner", "owner-request"))
    duplicate = threading.Thread(
        target=dispatch,
        args=("duplicate", "duplicate-request"),
    )
    owner.start()
    assert first_preflight.wait(timeout=2)
    duplicate.start()
    assert duplicate_waiting.wait(timeout=2)
    release_first_preflight.set()
    owner.join(timeout=2)
    duplicate.join(timeout=2)

    assert not owner.is_alive()
    assert not duplicate.is_alive()
    assert responses["owner"]["error"]["code"] == 5037
    assert responses["duplicate"]["result"]["status"] == "interrupted"
    assert responses["duplicate"]["result"]["mutation"]["deduplicated"] is False
    assert agent.interrupt_calls == 1


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


def test_corrupt_completed_outcome_is_store_unavailable_not_invalid_request(
    tmp_path,
    monkeypatch,
    caplog,
):
    from tui_gateway import server

    path = tmp_path / "corrupt-mobile-mutations.sqlite3"
    first = MobileMutationStore(path, owner_instance_id="process-a")
    claim = first.reserve(
        provider="password",
        subject="mobile-user",
        client_request_id="corrupt-outcome-1",
        method="session.interrupt",
        resource_id="conversation-root",
        semantic_parameters={},
    )
    assert first.complete(claim, {"result": {"status": "interrupted"}}) is True
    first.close()
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE mobile_mutations SET outcome_json = ?",
            ("{not-json",),
        )

    reopened = MobileMutationStore(path, owner_instance_id="process-b")
    monkeypatch.setattr(server, "_mobile_mutation_store", lambda: reopened)
    with caplog.at_level(logging.WARNING, logger="tui_gateway.server"):
        response = server.dispatch(
            {
                "jsonrpc": "2.0",
                "id": "corrupt-outcome-retry",
                "method": "session.interrupt",
                "params": {
                    "client_request_id": "corrupt-outcome-1",
                    "expected_stored_session_id": "conversation-root",
                    "session_id": "no-live-session-needed-for-replay",
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
    reopened.close()


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


def test_completion_and_terminalization_failures_recycle_to_unknown(
    _isolated_gateway_mutation_store,
    monkeypatch,
    caplog,
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

    def fail_write(*_args, **_kwargs):
        raise sqlite3.OperationalError("receipt connection unhealthy")

    monkeypatch.setattr(_isolated_gateway_mutation_store, "complete", fail_write)
    monkeypatch.setattr(
        _isolated_gateway_mutation_store,
        "mark_outcome_unknown",
        fail_write,
    )
    request = {
        "jsonrpc": "2.0",
        "id": "double-write-failure",
        "method": "session.interrupt",
        "params": {
            "client_request_id": "double-write-failure-1",
            "expected_stored_session_id": "conversation-root",
            "session_id": "live-1",
        },
    }
    transport = _MobileTransport("conversation.read", "conversation.control")

    with caplog.at_level(logging.WARNING, logger="tui_gateway.server"):
        response = server.dispatch(request, transport)

    assert response["error"]["code"] == 5037
    assert _isolated_gateway_mutation_store.status(
        provider="password",
        subject="mobile-user",
        client_request_id="double-write-failure-1",
    )["state"] == "outcome_unknown"
    assert agent.interrupt_calls == 1
    assert any(record.exc_info is not None for record in caplog.records)

    request["id"] = "double-write-failure-retry"
    retry = server.dispatch(request, transport)

    assert retry["error"]["code"] == 4091
    assert agent.interrupt_calls == 1


def test_failed_recycle_is_retried_after_storage_recovers(
    _isolated_gateway_mutation_store,
    monkeypatch,
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

    def fail_write(*_args, **_kwargs):
        raise sqlite3.OperationalError("receipt connection unhealthy")

    original_open = _isolated_gateway_mutation_store._open_connection
    recycle_attempts = 0

    def fail_first_recycle(owner_instance_id):
        nonlocal recycle_attempts
        recycle_attempts += 1
        if recycle_attempts == 1:
            raise sqlite3.OperationalError("storage still unavailable")
        return original_open(owner_instance_id)

    original_wait = _isolated_gateway_mutation_store.wait_for_outcome

    def no_slow_wait(claim, *, timeout):
        assert timeout == 30.0
        return original_wait(claim, timeout=0.0)

    monkeypatch.setattr(_isolated_gateway_mutation_store, "complete", fail_write)
    monkeypatch.setattr(
        _isolated_gateway_mutation_store,
        "mark_outcome_unknown",
        fail_write,
    )
    monkeypatch.setattr(
        _isolated_gateway_mutation_store,
        "_open_connection",
        fail_first_recycle,
    )
    monkeypatch.setattr(
        _isolated_gateway_mutation_store,
        "wait_for_outcome",
        no_slow_wait,
    )
    request = {
        "jsonrpc": "2.0",
        "id": "recycle-still-down",
        "method": "session.interrupt",
        "params": {
            "client_request_id": "recycle-recovery-1",
            "expected_stored_session_id": "conversation-root",
            "session_id": "live-1",
        },
    }
    transport = _MobileTransport("conversation.read", "conversation.control")

    unavailable = server.dispatch(request, transport)

    assert unavailable["error"]["code"] == 5037
    assert recycle_attempts == 1
    assert agent.interrupt_calls == 1

    request["id"] = "recycle-storage-recovered"
    retry = server.dispatch(request, transport)

    assert retry["error"]["code"] == 4091
    assert recycle_attempts == 2
    assert agent.interrupt_calls == 1
    assert _isolated_gateway_mutation_store.status(
        provider="password",
        subject="mobile-user",
        client_request_id="recycle-recovery-1",
    )["state"] == "outcome_unknown"


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
