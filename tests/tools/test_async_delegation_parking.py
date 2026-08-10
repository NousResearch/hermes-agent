"""An undeliverable delegation completion must stop replaying on every boot.

Restart survival for background `delegate_task` works by persisting completions
and re-enqueuing the pending ones at registry startup
(`restore_undelivered_completions`). `_MAX_DELIVERY_ATTEMPTS` is meant to stop a
completion that can never be delivered from replaying forever — but the cap is
only consulted by `release_completion_delivery`, i.e. only on the path where a
consumer successfully *claimed* the row and then failed.

A completion whose origin session is permanently gone (dead owner pid) never
gets that far: the consumer's ownership gate drops it BEFORE claiming, so
`delivery_attempts` never advances and `delivery_state` stays `'pending'`.
`restore_undelivered_completions` re-enqueues it on the next boot, it is dropped
again, and the cycle repeats for the life of the install — re-logging a WARNING
each time.
"""

import json
import time

import pytest


@pytest.fixture
def delegation_db(tmp_path, monkeypatch):
    """Point the durable delegation store at a temp HERMES_HOME."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import tools.async_delegation as ad

    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    return ad


def _insert_pending(ad, delegation_id: str, *, attempts: int) -> None:
    """Insert a durable completion in the pending, undelivered state."""
    now = time.time()
    event = {
        "type": "async_delegation",
        "delegation_id": delegation_id,
        "session_key": "telegram:dead-session",
        "origin_ui_session_id": "",
        "origin_session_id": "",
        "parent_session_id": None,
        "goal": "a task whose origin session is gone",
        "status": "success",
        "summary": "done",
    }
    with ad._DB_LOCK, ad._transaction() as conn:
        conn.execute(
            """INSERT INTO async_delegations
               (delegation_id, origin_session, origin_ui_session_id, state,
                dispatched_at, completed_at, updated_at, event_json,
                delivery_state, delivery_attempts)
               VALUES (?, ?, '', 'success', ?, ?, ?, ?, 'pending', ?)""",
            (delegation_id, "telegram:dead-session", now, now, now,
             json.dumps(event), attempts),
        )


def _delivery_row(ad, delegation_id: str):
    with ad._DB_LOCK, ad._transaction() as conn:
        return conn.execute(
            "SELECT delivery_state, delivery_attempts FROM async_delegations "
            "WHERE delegation_id=?",
            (delegation_id,),
        ).fetchone()


class _Queue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def test_exhausted_completion_is_parked_instead_of_replayed(delegation_db):
    """A row past the attempt budget must not be re-enqueued again."""
    ad = delegation_db
    _insert_pending(ad, "d-exhausted", attempts=ad._MAX_DELIVERY_ATTEMPTS)

    queue = _Queue()
    restored = ad.restore_undelivered_completions(queue)

    assert queue.items == [], "an exhausted completion was replayed to the queue"
    assert restored == 0, "an exhausted completion was counted as restored"


def test_exhausted_completion_reaches_a_terminal_state(delegation_db):
    """Parking must be durable, or the next boot repeats the same work."""
    ad = delegation_db
    _insert_pending(ad, "d-exhausted", attempts=ad._MAX_DELIVERY_ATTEMPTS)

    ad.restore_undelivered_completions(_Queue())

    state, _attempts = _delivery_row(ad, "d-exhausted")
    assert state != "pending", (
        "row left 'pending' — it will be re-enqueued and re-dropped every boot"
    )
    # 'parked' (not 'delivered'): the result was never handed to anyone, so
    # claiming delivery would be dishonest. The row stays queryable for audit.
    assert state == "parked"


def test_parking_is_idempotent_across_repeated_restarts(delegation_db):
    """The second boot must be a no-op, proving the cycle is actually broken."""
    ad = delegation_db
    _insert_pending(ad, "d-exhausted", attempts=ad._MAX_DELIVERY_ATTEMPTS)

    first = _Queue()
    ad.restore_undelivered_completions(first)
    second = _Queue()
    ad.restore_undelivered_completions(second)

    assert first.items == []
    assert second.items == []


def test_healthy_completion_is_still_restored(delegation_db):
    """Control: restart survival itself must not regress.

    A completion below the budget is exactly what the persist + auto-redispatch
    machinery exists to recover, so it must still be re-enqueued and stamped
    ``restored=True``.
    """
    ad = delegation_db
    _insert_pending(ad, "d-healthy", attempts=0)

    queue = _Queue()
    restored = ad.restore_undelivered_completions(queue)

    assert restored == 1
    assert len(queue.items) == 1
    assert queue.items[0]["delegation_id"] == "d-healthy"
    assert queue.items[0]["restored"] is True
    state, _ = _delivery_row(ad, "d-healthy")
    assert state == "pending"


def test_boundary_one_attempt_below_budget_is_still_restored(delegation_db):
    """Off-by-one guard: the cap must not retire a row that has budget left."""
    ad = delegation_db
    _insert_pending(ad, "d-nearly", attempts=ad._MAX_DELIVERY_ATTEMPTS - 1)

    queue = _Queue()
    restored = ad.restore_undelivered_completions(queue)

    assert restored == 1
    assert queue.items[0]["delegation_id"] == "d-nearly"


def test_mixed_batch_parks_only_the_exhausted_row(delegation_db):
    """One poisoned row must not suppress recovery of healthy siblings."""
    ad = delegation_db
    _insert_pending(ad, "d-exhausted", attempts=ad._MAX_DELIVERY_ATTEMPTS)
    _insert_pending(ad, "d-healthy", attempts=1)

    queue = _Queue()
    restored = ad.restore_undelivered_completions(queue)

    assert restored == 1
    assert [item["delegation_id"] for item in queue.items] == ["d-healthy"]
    assert _delivery_row(ad, "d-exhausted")[0] == "parked"
    assert _delivery_row(ad, "d-healthy")[0] == "pending"


def test_ownership_drop_advances_the_delivery_attempt_counter(delegation_db):
    """The counter must advance on the pre-claim drop path too.

    `claim_completion_delivery` is the only site that bumps `delivery_attempts`,
    but an unownable completion is dropped BEFORE it is claimed. Without a
    counter bump on that path the budget is never consumed and the parking
    threshold above is unreachable — the fix would be dead code.
    """
    ad = delegation_db
    _insert_pending(ad, "d-unowned", attempts=0)

    ad._note_delivery_attempt("d-unowned")

    _state, attempts = _delivery_row(ad, "d-unowned")
    assert attempts == 1


def test_repeated_ownership_drops_eventually_reach_the_parking_threshold(
    delegation_db,
):
    """End-to-end: the drop path alone must be able to retire a poisoned row."""
    ad = delegation_db
    _insert_pending(ad, "d-unowned", attempts=0)

    for _ in range(ad._MAX_DELIVERY_ATTEMPTS):
        ad._note_delivery_attempt("d-unowned")

    queue = _Queue()
    ad.restore_undelivered_completions(queue)

    assert queue.items == []
    assert _delivery_row(ad, "d-unowned")[0] == "parked"
