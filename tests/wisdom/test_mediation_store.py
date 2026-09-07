from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context

import pytest

from hermes_wisdom.mediation_store import LEASE_SECONDS, MediationStore
from hermes_wisdom.store import WisdomStore


@pytest.fixture
def state(tmp_path):
    store = WisdomStore(tmp_path / "wisdom")
    store.activate_installation_identity("installation", "org")
    now = [1000.0]
    return MediationStore(store, clock=lambda: now[0]), now


def register(queue, key="session", *, busy=False, activity=True):
    queue.register_session(
        "org",
        session_key=key,
        session_id=f"id-{key}",
        platform="telegram",
        actor_id="user",
        private=True,
        available=not busy,
        user_activity=activity,
    )


def test_duplicate_events_and_cross_connection_claim(state):
    queue, now = state
    register(queue)
    first = queue.enqueue("org", "feed:1", {"skill_id": "skill"})
    assert queue.enqueue("org", "feed:1", {"skill_id": "changed"}) == first

    def claim(_):
        other = MediationStore(WisdomStore(queue.store.root), clock=lambda: now[0])
        return other.claim("org", "session")

    with ThreadPoolExecutor(max_workers=4) as pool:
        claims = list(pool.map(claim, range(4)))
    assert sum(len(value) for value in claims) == 1
    assert queue.assessments("org")[0]["reference"] == {"skill_id": "skill"}


def test_recent_busy_session_wins_over_idle_older_session(state):
    queue, now = state
    register(queue, "older")
    now[0] += 1
    register(queue, "newer", busy=True)
    queue.enqueue("org", "feed:1", {})
    assert not queue.claim("org", "older")
    assert not queue.claim("org", "newer")
    register(queue, "older", activity=False)
    assert not queue.claim("org", "older")
    register(queue, "newer", activity=False)
    assert len(queue.claim("org", "newer")) == 1


def test_qualification_stays_with_origin_and_no_inactive_wake(state):
    queue, now = state
    register(queue, "origin")
    now[0] += 1
    register(queue, "other")
    queue.enqueue("org", "candidate:1", {}, origin_session="id-origin")
    assert not queue.claim("org", "other")
    now[0] += 121
    assert not queue.claim("org", "origin")
    register(queue, "origin")
    assert len(queue.claim("org", "origin")) == 1


def test_lease_fencing_and_bounded_attempts(state):
    queue, now = state
    register(queue)
    identity = queue.enqueue("org", "feed:1", {})
    old = queue.claim("org", "session")[0]
    for _ in range(3):
        now[0] += LEASE_SECONDS + 1
        register(queue)
        current = queue.claim("org", "session")[0]
    assert current["state"] == "fallback"
    assert current["attempts"] == 3
    assert not queue.save_advice("org", identity, old["lease_token"], {})
    assert not queue.fail("org", identity, old["lease_token"], "old worker")
    assert queue.begin_delivery("org", identity, current["lease_token"])
    assert queue.complete_delivery(
        "org", identity, current["lease_token"], introduced=True
    )
    assert queue.introduced("org")
    assert not queue.claim("org", "session")


def test_advice_persisted_without_consuming_reads_and_ambiguous_send_not_replayed(
    state,
):
    queue, now = state
    register(queue)
    identity = queue.enqueue("org", "feed:1", {})
    job = queue.claim("org", "session")[0]
    assert queue.save_advice("org", identity, job["lease_token"], {"summary": "Useful"})
    assert not queue.claim("org", "session")
    assert queue.begin_delivery("org", identity, job["lease_token"])
    now[0] += LEASE_SECONDS + 1
    register(queue)
    assert not queue.claim("org", "session")
    assert queue.assessments("org")[0]["state"] == "delivery_uncertain"
    assert not queue.introduced("org")


def test_org_switch_fences_previous_worker_and_preserves_other_org(state):
    queue, _ = state
    register(queue)
    queue.enqueue("org", "feed:1", {})
    job = queue.claim("org", "session")[0]
    queue.store.activate_installation_identity("installation-2", "other-org")
    with pytest.raises(ValueError, match="no longer active"):
        queue.save_advice("org", job["id"], job["lease_token"], {})
    assert not queue.assessments("other-org")
    queue.store.activate_installation_identity("installation", "org")
    assert len(queue.assessments("org")) == 1


def test_schema_upgrade_and_reopen_preserve_pending_work(state):
    queue, _ = state
    queue.enqueue("org", "feed:1", {})
    with queue.store.transaction() as db:
        db.execute("UPDATE schema_meta SET value='9' WHERE key='schema_version'")
    reopened = MediationStore(WisdomStore(queue.store.root))
    assert len(reopened.assessments("org")) == 1


def _process_claim(root):
    queue = MediationStore(WisdomStore(root), clock=lambda: 1000.0)
    return len(queue.claim("org", "session"))


def test_independent_processes_elect_one_worker(state):
    queue, _ = state
    register(queue)
    queue.enqueue("org", "feed:processes", {})
    with get_context("spawn").Pool(2) as pool:
        assert sum(pool.map(_process_claim, [queue.store.root] * 2)) == 1


def test_saved_advice_can_move_after_owner_disconnect_without_reassessment(state):
    queue, now = state
    register(queue, "old")
    identity = queue.enqueue("org", "feed:1", {})
    claimed = queue.claim("org", "old")[0]
    queue.save_advice(
        "org", identity, claimed["lease_token"], {"title": "Saved advice"}
    )
    now[0] += LEASE_SECONDS + 1
    register(queue, "new")
    new = queue.claim("org", "new")[0]
    assert new["state"] == "ready" and new["advice"]["title"] == "Saved advice"
    assert new["attempts"] == 1 and new["owner_session"] == "new"
