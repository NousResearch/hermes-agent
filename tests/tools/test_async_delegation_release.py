"""Event-level retryable release preserves the durable delivery and orphan-sweep contracts."""

import queue
import time

import pytest

from tools import async_delegation as ad


@pytest.fixture(autouse=True)
def _clean_state():
    ad._reset_for_tests()
    yield
    ad._reset_for_tests()


def _pending_completion(delegation_id):
    evt = {
        "type": "async_delegation", "session_key": "s", "delegation_id": delegation_id,
        "summary": delegation_id, "status": "completed", "dispatched_at": time.time(),
    }
    ad._persist_dispatch(evt)
    ad._persist_completion(evt, {"status": "completed", "summary": delegation_id})
    # The ledger outlives its owner, just as after a consumer/runtime restart.
    with ad._transaction() as conn:
        conn.execute("UPDATE async_delegations SET owner_pid=NULL WHERE delegation_id=?", (delegation_id,))
    return evt


def _row(delegation_id):
    with ad._transaction() as conn:
        return conn.execute(
            "SELECT delivery_state, delivery_attempts, delivery_claim, delivery_claimed_at, "
            "updated_at, delivered_at FROM async_delegations WHERE delegation_id=?",
            (delegation_id,),
        ).fetchone()


@pytest.mark.parametrize("retryable", [False, True])
def test_release_budget_and_returned_orphan_offer(retryable):
    """Transient refusals refund claims; ordinary failures still exhaust the bounded budget."""
    evt = _pending_completion("release-budget")
    q = queue.Queue()
    assert ad.restore_undelivered_completions(q) == 1
    evt = q.get_nowait()
    cycles = ad._MAX_DELIVERY_ATTEMPTS + 2 if retryable else ad._MAX_DELIVERY_ATTEMPTS
    for attempt in range(1, cycles + 1):
        before = _row(evt["delegation_id"])
        assert ad.sweep_orphaned_completions(q, now=before[4] + ad._ORPHAN_STALE_S + 1) == 0
        assert q.empty()  # the current in-memory copy suppresses duplicate offers
        claim = ad.claim_event_delivery(evt, "consumer")
        assert claim
        assert ad.claim_event_delivery(evt, "competitor") is None
        if retryable:
            ad.release_event_delivery(evt, claim, retryable=True)
        else:
            ad.release_event_delivery(evt, claim)  # legacy two-argument API
        row = _row(evt["delegation_id"])
        terminal = not retryable and attempt == ad._MAX_DELIVERY_ATTEMPTS
        assert row[:4] == ("dropped" if terminal else "pending", 0 if retryable else attempt, None, None)
        swept = ad.sweep_orphaned_completions(q, now=row[4] + ad._ORPHAN_STALE_S + 1)
        if terminal:
            assert swept == 0 and q.empty()
            assert ad.claim_event_delivery(evt, "consumer") is None
        else:
            assert swept == 1  # release returned the discarded in-memory offer
            evt = q.get_nowait()
            assert evt["delegation_id"] == "release-budget" and evt["restored"] is True
    if retryable:
        claim = ad.claim_event_delivery(evt, "consumer")
        assert claim
        ad.complete_event_delivery(evt, claim)
        row = _row(evt["delegation_id"])
        assert row[:4] == ("delivered", 1, None, None)
        assert ad.sweep_orphaned_completions(q, now=row[4] + ad._ORPHAN_STALE_S + 1) == 0
        assert q.empty()


@pytest.mark.parametrize("terminal", ["delivered", "dropped"])
def test_retryable_release_fences_foreign_stale_and_terminal_claims(terminal):
    evt = _pending_completion("release-fenced")
    old_claim = ad.claim_event_delivery(evt, "consumer")
    assert old_claim
    before = _row(evt["delegation_id"])
    ad.release_event_delivery(evt, "foreign-claim", retryable=True)
    assert _row(evt["delegation_id"]) == before
    ad.release_event_delivery(evt, old_claim, retryable=True)
    assert _row(evt["delegation_id"])[:4] == ("pending", 0, None, None)

    claim = ad.claim_event_delivery(evt, "consumer")
    assert claim and claim != old_claim
    before = _row(evt["delegation_id"])
    ad.release_event_delivery(evt, old_claim, retryable=True)
    assert _row(evt["delegation_id"]) == before
    if terminal == "delivered":
        # The legacy acknowledgement keeps the claim: only the state fence prevents a refund.
        assert ad.mark_completion_delivered(evt["delegation_id"])
        before = _row(evt["delegation_id"])
        assert before[:3] == (terminal, 1, claim)
        assert before[3] is not None
    else:
        assert ad.drop_completion_delivery(evt["delegation_id"], claim)
        before = _row(evt["delegation_id"])
        assert before[:4] == (terminal, 1, None, None)
    ad.release_event_delivery(evt, claim, retryable=True)
    assert _row(evt["delegation_id"]) == before  # neither terminal state can be refunded/resurrected
