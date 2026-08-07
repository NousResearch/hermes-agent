"""Durable pending completions must not replay forever.

`restore_undelivered_completions()` re-enqueues every row still marked
`delivery_state='pending'` on process start. Delivery is deliberately
fail-closed — a restored event is never adopted by an unrelated session — so a
completion whose owning session never comes back stays `pending` and is
re-enqueued on *every* start, forever, and `_prune_durable_records()` never
removes it because it only deletes `delivered` rows.

`expire_stale_pending_completions()` bounds that replay window by moving aged
rows to the terminal `orphaned` state: unlike `delivered` it does not claim the
user saw the result, and unlike `pending` it is not replayed again.
"""

from __future__ import annotations

import json
import time

import pytest

import tools.async_delegation as ad


@pytest.fixture(autouse=True)
def _clean_table():
    with ad._DB_LOCK, ad._transaction() as conn:
        conn.execute("DELETE FROM async_delegations")
    yield
    with ad._DB_LOCK, ad._transaction() as conn:
        conn.execute("DELETE FROM async_delegations")


def _insert(
    delegation_id: str,
    *,
    state: str,
    delivery_state: str,
    completed_at: float | None,
    updated_at: float,
) -> None:
    event = json.dumps({"delegation_id": delegation_id, "summary": "done"})
    with ad._DB_LOCK, ad._transaction() as conn:
        conn.execute(
            """INSERT INTO async_delegations
                   (delegation_id, origin_session, state, delivery_state,
                    event_json, completed_at, updated_at, dispatched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                delegation_id,
                "session-A",
                state,
                delivery_state,
                event,
                completed_at,
                updated_at,
                updated_at,
            ),
        )


def _row(delegation_id: str) -> tuple[str, str] | None:
    with ad._DB_LOCK, ad._transaction() as conn:
        cur = conn.execute(
            "SELECT state, delivery_state FROM async_delegations WHERE delegation_id=?",
            (delegation_id,),
        )
        row = cur.fetchone()
    return (row[0], row[1]) if row else None


def test_aged_pending_completion_becomes_orphaned():
    old = time.time() - 10 * 24 * 60 * 60
    _insert("aged", state="completed", delivery_state="pending",
            completed_at=old, updated_at=old)

    assert ad.expire_stale_pending_completions() == 1
    assert _row("aged") == ("completed", "orphaned")


def test_recent_pending_completion_is_left_alone():
    recent = time.time() - 60
    _insert("recent", state="completed", delivery_state="pending",
            completed_at=recent, updated_at=recent)

    assert ad.expire_stale_pending_completions() == 0
    assert _row("recent") == ("completed", "pending")


@pytest.mark.parametrize("live_state", ["running", "finalizing"])
def test_live_work_is_never_expired(live_state):
    """Age alone must not orphan work that has not finished yet."""
    old = time.time() - 10 * 24 * 60 * 60
    _insert(live_state, state=live_state, delivery_state="pending",
            completed_at=None, updated_at=old)

    assert ad.expire_stale_pending_completions() == 0
    assert _row(live_state) == (live_state, "pending")


def test_legacy_row_without_completed_at_uses_updated_at():
    old = time.time() - 10 * 24 * 60 * 60
    _insert("legacy", state="completed", delivery_state="pending",
            completed_at=None, updated_at=old)

    assert ad.expire_stale_pending_completions() == 1
    assert _row("legacy") == ("completed", "orphaned")


def test_orphaned_rows_are_not_replayed():
    """The whole point: an orphaned row must not be re-enqueued on start."""
    old = time.time() - 10 * 24 * 60 * 60
    _insert("aged", state="completed", delivery_state="pending",
            completed_at=old, updated_at=old)
    _insert("fresh", state="completed", delivery_state="pending",
            completed_at=time.time(), updated_at=time.time())

    enqueued: list[object] = []

    class _Queue:
        def put(self, item):
            enqueued.append(item)

        def put_nowait(self, item):
            enqueued.append(item)

    restored = ad.restore_undelivered_completions(_Queue())

    assert restored == 1, "only the fresh completion should be replayed"
    assert _row("aged")[1] == "orphaned"
    assert _row("fresh")[1] == "pending"


def test_orphaned_rows_age_out_of_retention():
    """`orphaned` must share the retention window with `delivered`.

    Otherwise expiry just swaps an infinite replay for an infinite row.
    """
    ancient = time.time() - 2 * ad._DURABLE_RETENTION_SECONDS
    _insert("ancient", state="completed", delivery_state="orphaned",
            completed_at=ancient, updated_at=ancient)

    ad._prune_durable_records()

    assert _row("ancient") is None
