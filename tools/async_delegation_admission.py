"""Durable ownership of async completions admitted to a gateway's pending queue."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence

Receipt = tuple[str, str]


def _owner_stamp() -> tuple[int, int]:
    from gateway.status import get_process_start_time

    pid = os.getpid()
    started = get_process_start_time(pid)
    if started is None or started <= 0:
        raise RuntimeError("Cannot admit async completion without a process start time")
    return pid, started


def queue_completion_deliveries(receipts: Sequence[Receipt]) -> tuple[Receipt, ...]:
    """Transfer claimed rows to this process before publishing the event in memory.

    Legacy events with no ledger row need no durable receipt. Every existing row
    must still belong to the supplied pending claim, or the entire batch is refused.
    """
    from tools.async_delegation import _DB_LOCK, _transaction

    receipts = tuple(dict.fromkeys(receipts))
    if not receipts:
        return ()
    owned: list[Receipt] = []
    with _DB_LOCK, _transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        for delegation_id, claim_id in receipts:
            row = conn.execute(
                "SELECT delivery_state, delivery_claim FROM async_delegations WHERE delegation_id=?",
                (delegation_id,),
            ).fetchone()
            if row is None:
                continue
            if not claim_id or row != ("pending", claim_id):
                raise RuntimeError(f"Async completion claim no longer owned: {delegation_id}")
            owned.append((delegation_id, claim_id))
        if owned:
            pid, started = _owner_stamp()
            now = time.time()
            for delegation_id, claim_id in owned:
                conn.execute(
                    """UPDATE async_delegations SET delivery_state='queued',
                       delivery_owner_pid=?, delivery_owner_started_at=?, updated_at=?
                       WHERE delegation_id=? AND delivery_state='pending' AND delivery_claim=?""",
                    (pid, started, now, delegation_id, claim_id),
                )
    return tuple(owned)


def begin_queued_completion_deliveries(receipts: Sequence[Receipt]) -> bool:
    """Claim the whole queued event for execution; stale or duplicate events cannot run."""
    from tools.async_delegation import _DB_LOCK, _transaction

    receipts = tuple(dict.fromkeys(receipts))
    if not receipts:
        return True
    pid, started = _owner_stamp()
    with _DB_LOCK, _transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        for delegation_id, claim_id in receipts:
            row = conn.execute(
                """SELECT delivery_state, delivery_claim, delivery_owner_pid, delivery_owner_started_at
                   FROM async_delegations WHERE delegation_id=?""", (delegation_id,),
            ).fetchone()
            if not claim_id or row != ("queued", claim_id, pid, started):
                return False
        now = time.time()
        for delegation_id, claim_id in receipts:
            conn.execute(
                """UPDATE async_delegations SET delivery_state='processing', updated_at=?
                   WHERE delegation_id=? AND delivery_state='queued' AND delivery_claim=?
                     AND delivery_owner_pid=? AND delivery_owner_started_at=?""",
                (now, delegation_id, claim_id, pid, started),
            )
    return True


def settle_queued_completion_deliveries(receipts: Sequence[Receipt], outcome: str) -> int:
    """Settle exact receipts; preparation failures spend attempts, refusals refund."""
    from tools.async_delegation import _DB_LOCK, _MAX_DELIVERY_ATTEMPTS, _transaction

    expected = {
        "delivered": "processing", "unknown": "processing",
        "pending": "queued", "dropped": "queued", "retry": "queued",
    }
    if outcome not in expected:
        raise ValueError(f"Invalid async completion outcome: {outcome}")
    receipts = tuple(dict.fromkeys(receipts))
    if not receipts:
        return 0
    pid, started = _owner_stamp()
    now, changed = time.time(), 0
    with _DB_LOCK, _transaction() as conn:
        for delegation_id, claim_id in receipts:
            changed += conn.execute(
                """UPDATE async_delegations SET
                   delivery_state=CASE WHEN ?='retry' THEN
                       CASE WHEN delivery_attempts>=? THEN 'dropped' ELSE 'pending' END
                       ELSE ? END, updated_at=?,
                   delivered_at=CASE WHEN ?='delivered' THEN ? ELSE delivered_at END,
                   delivery_attempts=MAX(0, delivery_attempts - ?),
                   delivery_claim=NULL, delivery_claimed_at=NULL,
                   delivery_owner_pid=NULL, delivery_owner_started_at=NULL
                   WHERE delegation_id=? AND delivery_state=? AND delivery_claim=?
                     AND delivery_owner_pid=? AND delivery_owner_started_at=?""",
                (outcome, _MAX_DELIVERY_ATTEMPTS, outcome, now, outcome, now, int(outcome == "pending"),
                 delegation_id, expected[outcome], claim_id, pid, started),
            ).rowcount
    return changed


def recover_queued_completion_deliveries() -> int:
    """Recover dead owners' queued work, but never rerun a possibly executed turn."""
    from gateway.status import _pid_exists, get_process_start_time
    from tools.async_delegation import _DB_LOCK, _transaction

    now, recovered = time.time(), 0
    with _DB_LOCK, _transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute(
            """SELECT delegation_id, delivery_state, delivery_claim,
                      delivery_owner_pid, delivery_owner_started_at
               FROM async_delegations WHERE delivery_state IN ('queued','processing')""",
        ).fetchall()
        for delegation_id, state, claim_id, pid, started in rows:
            # Missing/unreadable identity is not proof that an owner disappeared.
            if not pid or started is None:
                continue
            if _pid_exists(pid):
                current_start = get_process_start_time(pid)
                if current_start is None or current_start == started:
                    continue
            outcome = "pending" if state == "queued" else "unknown"
            recovered += conn.execute(
                """UPDATE async_delegations SET delivery_state=?, updated_at=?,
                   delivery_attempts=MAX(0, delivery_attempts - ?),
                   delivery_claim=NULL, delivery_claimed_at=NULL,
                   delivery_owner_pid=NULL, delivery_owner_started_at=NULL
                   WHERE delegation_id=? AND delivery_state=? AND delivery_claim IS ?
                     AND delivery_owner_pid=? AND delivery_owner_started_at=?""",
                (outcome, now, int(state == "queued"), delegation_id, state, claim_id, pid, started),
            ).rowcount
    return recovered
