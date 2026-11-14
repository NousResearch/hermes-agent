"""A refunded (deferred) completion converges to ``dropped`` once it outlives the replay cap."""

import time

import pytest

from gateway import session_worker_delegation as worker
from tools import async_delegation as ad


@pytest.fixture(autouse=True)
def _clean_state():
    ad._reset_for_tests()
    yield
    ad._reset_for_tests()


def _claimed(delegation_id: str, *, completed_at: float) -> None:
    event = {
        "type": "async_delegation", "delegation_id": delegation_id, "session_key": "agent:main:discord:thread:1:1",
        "parent_session_id": "parent", "dispatched_at": completed_at, "status": "completed", "summary": "done",
    }
    ad._persist_dispatch(event)
    ad._persist_completion(event, {"status": "completed", "summary": "done"})
    with ad._DB_LOCK, ad._transaction() as conn:
        conn.execute("UPDATE async_delegations SET completed_at=?, dispatched_at=? WHERE delegation_id=?",
                     (completed_at, completed_at, delegation_id))
    assert ad.claim_completion_delivery(delegation_id, "claim")


def _row(delegation_id: str):
    with ad._DB_LOCK, ad._transaction() as conn:
        return conn.execute("SELECT delivery_state, delivery_attempts, delivery_claim FROM async_delegations "
                            "WHERE delegation_id=?", (delegation_id,)).fetchone()


def test_defer_refunds_a_fresh_completion():
    _claimed("fresh", completed_at=time.time())

    assert ad.defer_completion_delivery("fresh", "claim") is True

    assert _row("fresh") == ("pending", 0, None)


def test_defer_drops_a_completion_past_the_replay_cap():
    _claimed("stale", completed_at=time.time() - ad._MAX_COMPLETION_REPLAY_AGE_S - 60)

    assert ad.defer_completion_delivery("stale", "claim") is True

    state, _attempts, claim = _row("stale")
    assert (state, claim) == ("dropped", None)
    assert ad.claim_completion_delivery("stale", "again") is False


def test_defer_ignores_a_foreign_claim():
    _claimed("owned", completed_at=time.time() - ad._MAX_COMPLETION_REPLAY_AGE_S - 60)

    assert ad.defer_completion_delivery("owned", "other") is False

    assert _row("owned")[0] == "pending"


@pytest.mark.parametrize(("age", "state"), [(0, "pending"), (ad._MAX_COMPLETION_REPLAY_AGE_S + 60, "dropped")])
def test_worker_defer_applies_the_same_replay_cap(monkeypatch, age, state):
    _claimed("remote", completed_at=time.time() - age)
    monkeypatch.setattr(worker, "_owned", lambda *_args: None)

    with ad._DB_LOCK, ad._transaction() as conn:
        result = worker._delivery(None, conn, "parent", {"delegation_id": "remote", "claim_id": "claim"},
                                  execution_id="exec", action="defer")

    assert result == {"value": True}
    assert _row("remote") == (state, 0, None)
