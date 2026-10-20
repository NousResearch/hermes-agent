"""An acknowledged completion goes back to the delivery queue when its turn could not confirm the owner."""

import queue

import pytest

from hermes_constants import get_hermes_home, hermes_home_key
from tools import async_delegation as ad


@pytest.fixture(autouse=True)
def _clean_state():
    ad._reset_for_tests()
    yield
    ad._reset_for_tests()


def _persisted(delegation_id: str, *, delivered: bool = True) -> None:
    event = {
        "type": "async_delegation", "delegation_id": delegation_id, "session_key": "agent:main:telegram:dm:1",
        "parent_session_id": "parent", "dispatched_at": 1.0, "status": "completed", "summary": "done",
    }
    ad._persist_dispatch(event)
    ad._persist_completion(event, {"status": "completed", "summary": "done"})
    if delivered:
        assert ad.claim_completion_delivery(delegation_id, "claim")
        assert ad.complete_completion_delivery(delegation_id, "claim")


def _column(delegation_id: str, column: str):
    with ad._DB_LOCK, ad._transaction() as conn:
        return conn.execute(f"SELECT {column} FROM async_delegations WHERE delegation_id=?", (delegation_id,)).fetchone()[0]


def _set(delegation_id: str, column: str, value) -> None:
    with ad._DB_LOCK, ad._transaction() as conn:
        conn.execute(f"UPDATE async_delegations SET {column}=? WHERE delegation_id=?", (value, delegation_id))


def test_requeue_returns_an_acknowledged_completion_to_pending_and_offers_it():
    _persisted("requeue-ok")
    offered: queue.Queue = queue.Queue()

    assert ad.requeue_admitted_completion("requeue-ok", offered) is True

    assert _column("requeue-ok", "delivery_state") == "pending"
    assert _column("requeue-ok", "delivered_at") is None
    evt = offered.get_nowait()
    assert evt["delegation_id"] == "requeue-ok" and evt["restored"] is True
    assert (hermes_home_key(get_hermes_home()), "requeue-ok") in ad._offered


def test_requeue_past_the_delivery_budget_drops_the_completion():
    _persisted("requeue-spent")
    _set("requeue-spent", "delivery_attempts", ad._MAX_DELIVERY_ATTEMPTS)
    offered: queue.Queue = queue.Queue()

    assert ad.requeue_admitted_completion("requeue-spent", offered) is False

    assert _column("requeue-spent", "delivery_state") == "dropped"
    assert offered.empty()


@pytest.mark.parametrize("case", ["not_delivered", "no_stored_event"])
def test_requeue_leaves_rows_it_cannot_offer(case):
    _persisted(case, delivered=case != "not_delivered")
    if case == "no_stored_event":
        _set(case, "event_json", None)
    before = _column(case, "delivery_state")
    offered: queue.Queue = queue.Queue()

    assert ad.requeue_admitted_completion(case, offered) is False

    assert _column(case, "delivery_state") == before
    assert offered.empty()
