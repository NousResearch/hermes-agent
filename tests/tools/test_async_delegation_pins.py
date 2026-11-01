"""Behaviour pins for async-delegation completion delivery (tools/async_delegation.py).

Freezes the delivery-claim lease, the attempt budget enforced at release, and the replay age
boundary. Rows live in a real temp ``state.db``; elapsed time is written into the row
(``delivery_claimed_at``) or injected as ``now`` — never raced against the wall clock.
"""

from __future__ import annotations

import json
import queue
import sqlite3
import time

import pytest

from tools import async_delegation as ad


@pytest.fixture()
def ledger(tmp_path, monkeypatch):
    ad._reset_for_tests()
    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    yield tmp_path / "state.db"
    ad._reset_for_tests()


def _insert_pending(delegation_id: str, *, completed_at: float) -> None:
    evt = {"type": "async_delegation", "delegation_id": delegation_id, "session_key": "s", "status": "completed"}
    with ad._DB_LOCK, ad._transaction() as conn:
        conn.execute(
            """INSERT INTO async_delegations
               (delegation_id, origin_session, origin_ui_session_id, parent_session_id, state,
                dispatched_at, completed_at, delivery_state, event_json, updated_at)
               VALUES (?, 's', '', 'p', 'completed', ?, ?, 'pending', ?, ?)""",
            (delegation_id, completed_at, completed_at, json.dumps(evt), completed_at))


def _row(path, delegation_id: str) -> dict:
    conn = sqlite3.connect(path)
    try:
        conn.row_factory = sqlite3.Row
        return dict(conn.execute("SELECT * FROM async_delegations WHERE delegation_id=?", (delegation_id,)).fetchone())
    finally:
        conn.close()


def _set(path, delegation_id: str, **cols) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(f"UPDATE async_delegations SET {', '.join(f'{k}=?' for k in cols)} WHERE delegation_id=?",
                     (*cols.values(), delegation_id))
        conn.commit()
    finally:
        conn.close()


def test_expired_delivery_claim_lease_lets_a_second_claimer_take_it(ledger):
    """A held claim blocks rivals until ``delivery_claimed_at`` is older than ``_CLAIM_LEASE_S``; then a second claimer wins and fences the first."""
    _insert_pending("deleg_lease", completed_at=time.time())
    assert ad.claim_completion_delivery("deleg_lease", "first")
    assert not ad.claim_completion_delivery("deleg_lease", "second")

    # Just inside the lease: still held.
    _set(ledger, "deleg_lease", delivery_claimed_at=time.time() - ad._CLAIM_LEASE_S + 30)
    assert not ad.claim_completion_delivery("deleg_lease", "second")

    _set(ledger, "deleg_lease", delivery_claimed_at=time.time() - ad._CLAIM_LEASE_S - 1)
    assert ad.claim_completion_delivery("deleg_lease", "second")
    row = _row(ledger, "deleg_lease")
    assert (row["delivery_claim"], row["delivery_attempts"]) == ("second", 2)
    # The expired holder is fenced: its ack and release are no-ops; the new holder settles.
    assert not ad.complete_completion_delivery("deleg_lease", "first")
    assert not ad.release_completion_delivery("deleg_lease", "first")
    assert ad.complete_completion_delivery("deleg_lease", "second")
    assert _row(ledger, "deleg_lease")["delivery_state"] == "delivered"


def test_release_drops_the_row_once_attempts_reach_the_cap(ledger):
    """Each claim spends an attempt; the release of the ``_MAX_DELIVERY_ATTEMPTS``-th claim converges the row to ``dropped`` and nothing can claim it again."""
    _insert_pending("deleg_cap", completed_at=time.time())
    for attempt in range(1, ad._MAX_DELIVERY_ATTEMPTS):
        claim = f"c{attempt}"
        assert ad.claim_completion_delivery("deleg_cap", claim)
        assert ad.release_completion_delivery("deleg_cap", claim)
        row = _row(ledger, "deleg_cap")
        assert (row["delivery_state"], row["delivery_attempts"], row["delivery_claim"]) == ("pending", attempt, None)

    assert ad.claim_completion_delivery("deleg_cap", "last")
    assert ad.release_completion_delivery("deleg_cap", "last")
    row = _row(ledger, "deleg_cap")
    assert (row["delivery_state"], row["delivery_attempts"]) == ("dropped", ad._MAX_DELIVERY_ATTEMPTS)
    assert not ad.claim_completion_delivery("deleg_cap", "after")


def test_defer_refunds_the_attempt_so_busy_targets_never_exhaust_the_budget(ledger):
    """``defer_completion_delivery`` returns the row to pending and refunds the claim's attempt."""
    _insert_pending("deleg_defer", completed_at=time.time())
    for i in range(ad._MAX_DELIVERY_ATTEMPTS + 2):
        assert ad.claim_completion_delivery("deleg_defer", f"d{i}")
        assert ad.defer_completion_delivery("deleg_defer", f"d{i}")
    row = _row(ledger, "deleg_defer")
    assert (row["delivery_state"], row["delivery_attempts"]) == ("pending", 0)


def test_replay_age_boundary_drops_strictly_older_rows_only(ledger):
    """``_replay_pending`` offers a row exactly ``_MAX_COMPLETION_REPLAY_AGE_S`` old and terminally drops one a second older."""
    now = 2_000_000_000.0
    _insert_pending("deleg_edge", completed_at=now - ad._MAX_COMPLETION_REPLAY_AGE_S)
    _insert_pending("deleg_old", completed_at=now - ad._MAX_COMPLETION_REPLAY_AGE_S - 1)
    q: queue.Queue = queue.Queue()
    with ad._DB_LOCK, ad._transaction() as conn:
        rows = conn.execute("""SELECT delegation_id, event_json, completed_at, dispatched_at
                               FROM async_delegations ORDER BY delegation_id""").fetchall()
        assert ad._replay_pending(conn, rows, q, now) == 1

    offered = [q.get_nowait() for _ in range(q.qsize())]
    assert [(e["delegation_id"], e["restored"]) for e in offered] == [("deleg_edge", True)]
    assert _row(ledger, "deleg_old")["delivery_state"] == "dropped"
    assert _row(ledger, "deleg_edge")["delivery_state"] == "pending"
