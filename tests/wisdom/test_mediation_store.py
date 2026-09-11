from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from multiprocessing import get_context

import pytest

from hermes_wisdom.mediation_store import LEASE_SECONDS, MediationStore
from hermes_wisdom.store import WisdomStore
from hermes_wisdom.delivery import DeliveryReceipt

RECEIPT = DeliveryReceipt(
    platform="telegram",
    destination="chat",
    message_id="1",
    acknowledgement="provider_accepted",
)


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
        address={"chat_id": "chat"},
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


@pytest.mark.parametrize("background", ["delegate", "descendant", "cron", "background_review", "side_question"])
def test_background_cannot_register_or_claim_parent_delivery(state, monkeypatch, background):
    from agent.delegation_context import delegated_child_context, DELEGATED_CHILD_ENV_MARKER
    from gateway.session_context import _VAR_MAP
    from tools.skill_provenance import set_current_write_origin, reset_current_write_origin

    queue, now = state
    register(queue)
    queue.enqueue("org", "feed:1", {})
    with queue.store.transaction() as db:
        before = dict(db.execute("SELECT * FROM wisdom_agent_session").fetchone())
    now[0] += 1
    with ExitStack() as stack:
        env = stack.enter_context(monkeypatch.context())
        if background == "delegate":
            stack.enter_context(delegated_child_context("child-id"))
        elif background == "descendant":
            env.setenv(DELEGATED_CHILD_ENV_MARKER, "1")
        elif background == "cron":
            var = _VAR_MAP["HERMES_CRON_SESSION"]
            stack.callback(var.reset, var.set("1"))
        else:
            stack.callback(reset_current_write_origin, set_current_write_origin(background))
        register(queue, busy=True)
        register(queue, "child")
        assert queue.claim("org", "session") == []
        with queue.store.transaction() as db:
            assert [dict(row) for row in db.execute("SELECT * FROM wisdom_agent_session")] == [before]
        assert queue.assessments("org")[0]["state"] == "pending"
    assert len(queue.claim("org", "session")) == 1


@pytest.mark.parametrize("platform", ["subagent", "cron", "kanban", "api_server", "unknown"])
def test_noninteractive_surface_cannot_register(state, platform):
    queue, _ = state
    queue.register_session("org", session_key="child", session_id="child", platform=platform,
                           actor_id="user", private=True, available=True, user_activity=True)
    with queue.store.transaction() as db:
        assert db.execute("SELECT COUNT(*) FROM wisdom_agent_session").fetchone()[0] == 0


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


@pytest.mark.parametrize("model_available", [False, True])
def test_requested_claim_skips_proactive_backlog_without_spending_attempts(state, model_available):
    from hermes_wisdom.mediation_store import BATCH_SIZE

    queue, _ = state
    register(queue)
    for index in range(BATCH_SIZE + 1):
        queue.enqueue("org", f"feed:{index}", {"user_requested": "true"})
    requested = queue.enqueue("org", "manual:1", {"kind": "setup_handoff", "user_requested": True}, origin_session="session")
    claimed = queue.claim("org", "session", requested_only=True, allow_model_work=model_available)
    assert [item["id"] for item in claimed] == [requested]
    assert all(item["attempts"] == 0 for item in queue.assessments("org") if item["id"] != requested)
    # A second worker cannot bypass an in-flight requested job's lease.
    assert queue.claim("org", "session") == []
    assert queue.save_advice("org", requested, claimed[0]["lease_token"], {"title": "Requested review"})
    assert queue.begin_delivery("org", requested, claimed[0]["lease_token"])
    assert queue.complete_delivery("org", requested, claimed[0]["lease_token"], receipt=RECEIPT)
    assert queue.claim("org", "session", requested_only=True) == []
    assert queue.claim("org", "session", allow_model_work=False) == []
    assert len(queue.claim("org", "session")) == BATCH_SIZE


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
        "org", identity, current["lease_token"], introduced=True, receipt=RECEIPT
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
