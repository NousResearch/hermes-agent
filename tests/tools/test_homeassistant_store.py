import os
import threading
from datetime import datetime, timedelta, timezone

import pytest

from tools.homeassistant_store import (
    HomeAssistantChangeStore,
    ProposalExpired,
    ProposalStale,
    ProposalUnavailable,
    canonical_fingerprint,
    structured_diff,
)


NOW = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)


def _store(tmp_path, *, history_limit=500):
    return HomeAssistantChangeStore(
        tmp_path / "state" / "homeassistant_changes.sqlite3",
        history_limit=history_limit,
    )


def test_fingerprint_is_stable_across_mapping_key_order():
    assert canonical_fingerprint({"b": 2, "a": 1}) == canonical_fingerprint(
        {"a": 1, "b": 2}
    )
    assert canonical_fingerprint(None) != canonical_fingerprint({})


def test_structured_diff_reports_nested_add_change_and_remove():
    before = {"name": "Evening", "trigger": {"at": "18:00"}, "old": True}
    after = {"name": "Night", "trigger": {"at": "19:00"}, "enabled": True}

    assert structured_diff(before, after) == [
        {"path": "/enabled", "before": None, "after": True, "change": "added"},
        {"path": "/name", "before": "Evening", "after": "Night", "change": "changed"},
        {"path": "/old", "before": True, "after": None, "change": "removed"},
        {"path": "/trigger/at", "before": "18:00", "after": "19:00", "change": "changed"},
    ]


def test_create_and_get_proposal_uses_profile_local_private_database(tmp_path):
    store = _store(tmp_path)
    proposal = store.create_proposal(
        resource_type="automation",
        resource_id="evening",
        operation="update",
        before={"alias": "Evening"},
        desired={"alias": "Night"},
        ttl_seconds=900,
        now=NOW,
    )

    loaded = store.get_proposal(proposal["id"])
    assert loaded == proposal
    assert loaded["status"] == "pending"
    assert loaded["expires_at"] == NOW + timedelta(seconds=900)
    assert store.path == tmp_path / "state" / "homeassistant_changes.sqlite3"
    assert os.stat(store.path).st_mode & 0o777 == 0o600


def test_claim_rejects_expired_proposal(tmp_path):
    store = _store(tmp_path)
    proposal = store.create_proposal(
        resource_type="script",
        resource_id="welcome",
        operation="update",
        before={"sequence": []},
        desired={"sequence": [{"service": "light.turn_on"}]},
        ttl_seconds=30,
        now=NOW,
    )

    with pytest.raises(ProposalExpired):
        store.claim_proposal(
            proposal["id"],
            current_fingerprint=proposal["before_fingerprint"],
            now=NOW + timedelta(seconds=31),
        )
    assert store.get_proposal(proposal["id"])["status"] == "expired"


def test_claim_rejects_stale_resource_snapshot(tmp_path):
    store = _store(tmp_path)
    proposal = store.create_proposal(
        resource_type="scene",
        resource_id="movie",
        operation="update",
        before={"name": "Movie"},
        desired={"name": "Cinema"},
        now=NOW,
    )

    with pytest.raises(ProposalStale):
        store.claim_proposal(
            proposal["id"],
            current_fingerprint=canonical_fingerprint({"name": "Changed elsewhere"}),
            now=NOW,
        )
    assert store.get_proposal(proposal["id"])["status"] == "stale"


def test_proposal_can_only_be_claimed_once(tmp_path):
    store = _store(tmp_path)
    proposal = store.create_proposal(
        resource_type="timer",
        resource_id="tea",
        operation="create",
        before=None,
        desired={"name": "Tea", "duration": "00:05:00"},
        now=NOW,
    )
    store.claim_proposal(
        proposal["id"], proposal["before_fingerprint"], now=NOW
    )

    with pytest.raises(ProposalUnavailable):
        store.claim_proposal(
            proposal["id"], proposal["before_fingerprint"], now=NOW
        )


def test_record_applied_creates_auditable_change_and_prunes_history(tmp_path):
    store = _store(tmp_path, history_limit=2)
    change_ids = []
    for i in range(3):
        before = {"value": i}
        desired = {"value": i + 1}
        proposal = store.create_proposal(
            resource_type="counter",
            resource_id=f"counter_{i}",
            operation="update",
            before=before,
            desired=desired,
            now=NOW + timedelta(seconds=i),
        )
        store.claim_proposal(
            proposal["id"], proposal["before_fingerprint"], now=NOW + timedelta(seconds=i)
        )
        change = store.record_applied(
            proposal["id"],
            after=desired,
            created_by_hermes=False,
            now=NOW + timedelta(seconds=i),
        )
        change_ids.append(change["id"])

    history = store.list_history()
    assert [item["id"] for item in history] == list(reversed(change_ids[-2:]))
    assert store.get_proposal(proposal["id"])["status"] == "applied"
    assert history[0]["before"] == {"value": 2}
    assert history[0]["after"] == {"value": 3}


def test_rollback_claim_requires_current_state_to_match_applied_state(tmp_path):
    store = _store(tmp_path)
    proposal = store.create_proposal(
        resource_type="input_boolean",
        resource_id="guest_mode",
        operation="create",
        before=None,
        desired={"name": "Guest mode"},
        now=NOW,
    )
    store.claim_proposal(proposal["id"], proposal["before_fingerprint"], now=NOW)
    change = store.record_applied(
        proposal["id"], after=proposal["desired"], created_by_hermes=True, now=NOW
    )

    with pytest.raises(ProposalStale):
        store.claim_rollback(
            change["id"],
            canonical_fingerprint({"name": "Edited manually"}),
            now=NOW,
        )

    claimed = store.claim_rollback(
        change["id"], change["after_fingerprint"], now=NOW
    )
    assert claimed["status"] == "rolling_back"
    rolled_back = store.record_rolled_back(change["id"], now=NOW)
    assert rolled_back["status"] == "rolled_back"


def test_concurrent_claim_has_exactly_one_winner(tmp_path):
    store = _store(tmp_path)
    proposal = store.create_proposal(
        resource_type="automation",
        resource_id="morning",
        operation="update",
        before={"alias": "Morning"},
        desired={"alias": "Early morning"},
        now=NOW,
    )
    outcomes = []

    def claim():
        try:
            HomeAssistantChangeStore(store.path).claim_proposal(
                proposal["id"], proposal["before_fingerprint"], now=NOW
            )
            outcomes.append("claimed")
        except ProposalUnavailable:
            outcomes.append("unavailable")

    threads = [threading.Thread(target=claim) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == ["claimed", "unavailable"]
