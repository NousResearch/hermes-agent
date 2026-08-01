"""Durable child-event ledger contracts for async delegation batches."""

from __future__ import annotations

import time
import uuid

import pytest

from tools import async_delegation as ad
from tools.process_registry import process_registry


@pytest.fixture(autouse=True)
def _clean_async_state():
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield
    deadline = time.monotonic() + 2
    while ad.active_count() and time.monotonic() < deadline:
        time.sleep(0.01)
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


def _record(
    *,
    goals=("audit",),
    turn_id="turn-current",
    delivery="inject",
    parent_session_id="parent-session",
):
    delegation_id = f"deleg_test_{uuid.uuid4().hex}"
    record = {
        "delegation_id": delegation_id,
        "goal": goals[0] if len(goals) == 1 else f"{len(goals)} tasks",
        "goals": list(goals),
        "context": "parent context",
        "toolsets": ["file"],
        "role": "leaf",
        "model": "child-model",
        "session_key": "agent:main:cli:dm:local",
        "origin_ui_session_id": "",
        "origin_session_id": "",
        "parent_session_id": parent_session_id,
        "parent_turn_id": turn_id,
        "status": "running",
        "dispatched_at": time.time(),
        "completed_at": None,
        "is_batch": True,
        "result_delivery": delivery,
    }
    with ad._records_lock:
        ad._records[delegation_id] = record
    ad._persist_dispatch(record)
    return delegation_id


def _child(index: int, summary: str, *, status="completed", error=None):
    return {
        "task_index": index,
        "status": status,
        "summary": summary,
        "error": error,
        "api_calls": 2,
        "duration_seconds": 0.25,
    }


def _queue_contents():
    items = []
    while not process_registry.completion_queue.empty():
        items.append(process_registry.completion_queue.get_nowait())
    for item in items:
        process_registry.completion_queue.put(item)
    return items


def _event_state(delegation_id: str, event_key: str):
    with ad._DB_LOCK, ad._transaction() as conn:
        return conn.execute(
            "SELECT delivery_state, delivery_attempts FROM async_delegation_events "
            "WHERE delegation_id=? AND event_key=?",
            (delegation_id, event_key),
        ).fetchone()


def _parent_state(delegation_id: str):
    with ad._DB_LOCK, ad._transaction() as conn:
        return conn.execute(
            "SELECT state, delivery_state FROM async_delegations "
            "WHERE delegation_id=?",
            (delegation_id,),
        ).fetchone()


def _durable_event_keys(delegation_id: str):
    with ad._DB_LOCK, ad._transaction() as conn:
        return [
            row[0]
            for row in conn.execute(
                "SELECT event_key FROM async_delegation_events "
                "WHERE delegation_id=? ORDER BY event_key",
                (delegation_id,),
            ).fetchall()
        ]

def test_after_turn_coalesced_claim_is_atomic_across_ready_children():
    delegation_id = _record(goals=("A", "B"), delivery="after_turn")
    assert ad.publish_batch_child_completion(delegation_id, 0, _child(0, "A"))
    assert ad.publish_batch_child_completion(delegation_id, 1, _child(1, "B"))
    children = [
        process_registry.completion_queue.get_nowait(),
        process_registry.completion_queue.get_nowait(),
    ]
    grouped = ad.coalesce_ready_after_turn_events(children)[0]

    competing_claim = ad.claim_event_delivery(children[1], "competing-consumer")
    assert competing_claim
    assert ad.claim_event_delivery(grouped, "group-consumer") is None
    # task:0's attempted group claim rolled back with the task:1 conflict.
    assert _event_state(delegation_id, "task:0") == ("pending", 0)
    assert _event_state(delegation_id, "task:1") == ("pending", 1)

    assert ad.release_event_delivery(children[1], competing_claim)
    group_claim = ad.claim_event_delivery(grouped, "group-consumer")
    assert group_claim
    assert ad.complete_event_delivery(grouped, group_claim)
    assert _event_state(delegation_id, "task:0") == ("delivered", 1)
    assert _event_state(delegation_id, "task:1") == ("delivered", 2)


def test_group_release_prunes_attempt_capped_child_without_blocking_sibling():
    delegation_id = _record(goals=("fresh", "near-cap"), delivery="after_turn")
    for index, summary in enumerate(("fresh", "near-cap")):
        assert ad.publish_batch_child_completion(
            delegation_id, index, _child(index, summary)
        )
    children = [
        process_registry.completion_queue.get_nowait(),
        process_registry.completion_queue.get_nowait(),
    ]
    near_cap = children[1]
    for _ in range(ad._MAX_DELIVERY_ATTEMPTS - 1):
        claim = ad.claim_event_delivery(near_cap, "failing-consumer")
        assert claim
        assert ad.release_event_delivery(near_cap, claim)

    grouped = ad.coalesce_ready_after_turn_events(children)[0]
    group_claim = ad.claim_event_delivery(grouped, "group-consumer")
    assert group_claim
    assert ad.release_event_delivery(grouped, group_claim)

    assert grouped["delivery_event_keys"] == ["task:0"]
    assert [result["summary"] for result in grouped["results"]] == ["fresh"]
    assert _event_state(delegation_id, "task:0") == ("pending", 1)
    assert _event_state(delegation_id, "task:1") == (
        "dropped",
        ad._MAX_DELIVERY_ATTEMPTS,
    )
    retry_claim = ad.claim_event_delivery(grouped, "retry-consumer")
    assert retry_claim
    assert ad.complete_event_delivery(grouped, retry_claim)
    assert _event_state(delegation_id, "task:0") == ("delivered", 2)

def test_inject_batch_finalization_rolls_back_children_if_parent_update_fails(
    monkeypatch,
):
    delegation_id = _record(goals=("A", "B"))
    with ad._records_lock:
        event_record = dict(ad._records[delegation_id])
    combined = {"results": [_child(0, "A"), _child(1, "B")]}
    parent_event = {
        **event_record,
        "type": "async_delegation",
        "status": "completed",
        "completed_at": time.time(),
    }

    def crash_before_parent_update(*_args, **_kwargs):
        raise RuntimeError("simulated crash before parent terminal update")

    monkeypatch.setattr(ad, "_update_completion_row", crash_before_parent_update)
    with pytest.raises(RuntimeError, match="simulated crash"):
        ad._persist_batch_child_finalization(
            event_record, parent_event, combined, "completed"
        )

    assert _durable_event_keys(delegation_id) == []
    assert _parent_state(delegation_id) == ("running", "pending")


def test_after_turn_finalization_keeps_one_legacy_queue_envelope():
    delegation_id = _record(goals=("A", "B"), delivery="after_turn")
    with ad._records_lock:
        event_record = dict(ad._records[delegation_id])
    children = [_child(0, "A"), _child(1, "B")]
    combined = {"results": children, "total_duration_seconds": 0.1}
    parent_event = {
        **event_record,
        "type": "async_delegation",
        "status": "completed",
        "is_batch": True,
        "results": children,
        "completed_at": time.time(),
    }

    ad._persist_batch_child_finalization(
        event_record, parent_event, combined, "completed"
    )

    grouped = process_registry.completion_queue.get_nowait()
    assert grouped["delivery_event_keys"] == ["task:0", "task:1"]
    assert [result["summary"] for result in grouped["results"]] == ["A", "B"]
    assert process_registry.completion_queue.empty()


def test_persisted_children_wait_for_legacy_batch_finalization():
    delegation_id = _record(goals=("A", "B"), delivery="after_turn")
    children = [_child(0, "A"), _child(1, "B")]

    assert ad.persist_batch_child_completion(delegation_id, 0, children[0])
    assert ad.persist_batch_child_completion(delegation_id, 1, children[1])
    assert _durable_event_keys(delegation_id) == ["task:0", "task:1"]
    assert process_registry.completion_queue.empty()

    with ad._records_lock:
        event_record = dict(ad._records[delegation_id])
    combined = {"results": children, "total_duration_seconds": 0.1}
    parent_event = {
        **event_record,
        "type": "async_delegation",
        "status": "completed",
        "is_batch": True,
        "results": children,
        "completed_at": time.time(),
    }

    ad._persist_batch_child_finalization(
        event_record, parent_event, combined, "completed"
    )

    grouped = process_registry.completion_queue.get_nowait()
    assert grouped["delivery_event_keys"] == ["task:0", "task:1"]
    assert [result["summary"] for result in grouped["results"]] == ["A", "B"]
    assert process_registry.completion_queue.empty()


@pytest.mark.parametrize("delivery", ["inject", "after_turn"])
def test_published_child_is_not_requeued_by_batch_finalization(delivery):
    delegation_id = _record(goals=("A",), delivery=delivery)
    child = _child(0, "A")
    assert ad.publish_batch_child_completion(delegation_id, 0, child)
    assert process_registry.completion_queue.qsize() == 1

    with ad._records_lock:
        event_record = dict(ad._records[delegation_id])
    combined = {"results": [child], "total_duration_seconds": 0.1}
    parent_event = {
        **event_record,
        "type": "async_delegation",
        "status": "completed",
        "is_batch": True,
        "results": [child],
        "completed_at": time.time(),
    }

    ad._persist_batch_child_finalization(
        event_record, parent_event, combined, "completed"
    )

    assert process_registry.completion_queue.qsize() == 1
    event = process_registry.completion_queue.get_nowait()
    assert event["delivery_event_key"] == "task:0"
    assert event["result_delivery"] == delivery


@pytest.mark.parametrize("delivery", ["inject", "after_turn"])
def test_recovery_with_complete_children_suppresses_parent_aggregate(delivery):
    delegation_id = _record(goals=("A", "B"), delivery=delivery)
    assert ad.publish_batch_child_completion(delegation_id, 0, _child(0, "A"))
    assert ad.publish_batch_child_completion(delegation_id, 1, _child(1, "B"))
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    with ad._DB_LOCK, ad._transaction() as conn:
        conn.execute(
            "UPDATE async_delegations SET owner_pid=99999999, owner_started_at=0 "
            "WHERE delegation_id=?",
            (delegation_id,),
        )

    assert ad.recover_abandoned_delegations() == 1

    assert _parent_state(delegation_id) == ("completed", "delivered")
    assert _durable_event_keys(delegation_id) == ["task:0", "task:1"]
    restored = __import__("queue").Queue()
    assert ad.restore_undelivered_completions(restored) == 2
    if delivery == "after_turn":
        grouped = restored.get_nowait()
        assert grouped["delivery_event_keys"] == ["task:0", "task:1"]
        assert [result["summary"] for result in grouped["results"]] == ["A", "B"]
        assert restored.empty()
    else:
        assert {restored.get_nowait()["delivery_event_key"] for _ in range(2)} == {
            "task:0",
            "task:1",
        }


@pytest.mark.parametrize("delivery", ["inject", "after_turn"])
def test_recovery_with_partial_children_emits_only_terminal_gap_event(delivery):
    delegation_id = _record(goals=("A", "B"), delivery=delivery)
    assert ad.publish_batch_child_completion(delegation_id, 0, _child(0, "A"))
    process_registry.completion_queue.get_nowait()
    with ad._DB_LOCK, ad._transaction() as conn:
        conn.execute(
            "UPDATE async_delegations SET owner_pid=99999999, owner_started_at=0 "
            "WHERE delegation_id=?",
            (delegation_id,),
        )

    assert ad.recover_abandoned_delegations() == 1

    assert _parent_state(delegation_id) == ("unknown", "delivered")
    assert _durable_event_keys(delegation_id) == ["task:0", "terminal"]
    restored = __import__("queue").Queue()
    assert ad.restore_undelivered_completions(restored) == 2
    events = [restored.get_nowait() for _ in range(2)]
    assert {
        key
        for event in events
        for key in event.get("delivery_event_keys")
        or [event.get("delivery_event_key")]
    } == {
        "task:0",
        "terminal",
    }
    terminal = next(
        event for event in events if event.get("delivery_event_key") == "terminal"
    )
    assert terminal["status"] == "unknown"
    assert terminal["result_delivery"] == delivery
    assert "1/2 batch child results" in terminal["error"]


@pytest.mark.parametrize("delivery", ["inject", "after_turn"])
def test_recovery_requires_exact_expected_batch_child_keys(delivery):
    delegation_id = _record(goals=("A", "B"), delivery=delivery)
    assert ad.publish_batch_child_completion(delegation_id, 0, _child(0, "A"))
    process_registry.completion_queue.get_nowait()
    with ad._records_lock:
        event_record = dict(ad._records[delegation_id])
    rogue = ad._build_batch_child_event(event_record, 99, _child(99, "rogue"))
    with ad._DB_LOCK, ad._transaction() as conn:
        assert ad._insert_batch_event(conn, rogue, now=time.time())
        conn.execute(
            "UPDATE async_delegations SET owner_pid=99999999, owner_started_at=0 "
            "WHERE delegation_id=?",
            (delegation_id,),
        )

    assert ad.recover_abandoned_delegations() == 1

    assert _parent_state(delegation_id) == ("unknown", "delivered")
    assert _durable_event_keys(delegation_id) == ["task:0", "task:99", "terminal"]

def test_durable_claim_renewal_prevents_expiry_steal():
    delegation_id = _record()
    assert ad.publish_batch_child_completion(
        delegation_id, 0, _child(0, "renew durable lease")
    )
    event = process_registry.completion_queue.get_nowait()
    claim = ad.claim_event_delivery(event, "slow-main")
    assert claim
    with ad._DB_LOCK, ad._transaction() as conn:
        conn.execute(
            "UPDATE async_delegation_events SET delivery_claimed_at=0 "
            "WHERE delegation_id=? AND event_key='task:0'",
            (delegation_id,),
        )

    assert ad.renew_event_delivery(event, claim) is True
    assert ad.claim_event_delivery(event, "competing-main") is None
    assert ad.complete_event_delivery(event, claim) is True

def test_pending_child_event_restores_with_same_durable_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    delegation_id = _record(goals=("restore me",))
    assert ad.publish_batch_child_completion(
        delegation_id, 0, _child(0, "restored result")
    )
    process_registry.completion_queue.get_nowait()

    restored_queue = __import__("queue").Queue()
    assert ad.restore_undelivered_completions(restored_queue) == 1
    event = restored_queue.get_nowait()
    assert event["restored"] is True
    assert event["delegation_id"] == delegation_id
    assert event["delivery_event_key"] == "task:0"

    claim = ad.claim_event_delivery(event, "restart-consumer")
    assert claim
    ad.complete_event_delivery(event, claim)
    assert ad.restore_undelivered_completions(restored_queue) == 0
