import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_wisdom.client import WisdomMuteResponse, WisdomSuppressionResponse
from hermes_wisdom.preferences import WisdomPreferences, suppression_key
from hermes_wisdom.store import WisdomStore


def iso(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


@pytest.fixture
def preferences(tmp_path):
    store = WisdomStore(tmp_path / "wisdom")
    store.activate_installation_identity("installation", "org")
    now = [1000.0]
    client = Mock(identity={"owner": "account"}, display_org_id="org")
    client.recommendation_mute.return_value = WisdomMuteResponse(
        org_id="org", muted=False, duration=None, muted_until=None, forever=False
    )
    client.recommendation_suppressions.return_value = []
    client.suppress_recommendation.side_effect = lambda key: WisdomSuppressionResponse(
        org_id="org", key=key, suppress_until=iso(now[0] + 30 * 86400)
    )
    service = SimpleNamespace(store=store, client=client)
    return WisdomPreferences(service, clock=lambda: now[0]), service, now


def stage(p, reference, days=30):
    user = p.identity("org")
    with p.store.transaction() as db:
        return p.stage_suppression(
            db, org="org", user=user, reference=reference, days=days
        )


REF = {"kind": "candidate", "content_hash": "sha256:" + "a" * 64}


def test_keys_hide_private_evidence_and_change_with_exact_reference():
    assert suppression_key({
        **REF,
        "skill_name": "private-name",
        "usage": "private-history",
    }) == suppression_key(REF)
    assert len(suppression_key(REF)) == 71
    assert suppression_key(REF) != suppression_key({**REF, "content_hash": "changed"})
    assert suppression_key({
        "kind": "skill",
        "skill_id": "s",
        "version": 1,
    }) != suppression_key({"kind": "skill", "skill_id": "s", "version": 2})
    for ref in [
        {"kind": "candidate"},
        {"kind": "skill", "skill_id": "s", "version": True},
        {"kind": "notice"},
    ]:
        with pytest.raises(ValueError):
            suppression_key(ref)


def test_idempotent_stage_and_independent_connection_claim(preferences):
    p, service, now = preferences
    first = stage(p, REF)
    now[0] += 5
    assert stage(p, REF)["suppressed_until"] == first["suppressed_until"]
    other_store = WisdomStore(p.store.path.parent)
    other = WisdomPreferences(
        SimpleNamespace(store=other_store, client=service.client), clock=lambda: now[0]
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda instance: instance.flush("org"), [p, other]))
    service.client.suppress_recommendation.assert_called_once_with(suppression_key(REF))
    with p.store.transaction() as db:
        row = dict(db.execute("SELECT * FROM wisdom_preference_outbox").fetchone())
    assert row["state"] == "synced"
    assert "private-name" not in json.dumps(row)


def test_network_failure_retains_local_suppression_and_retries_are_bounded(preferences):
    p, service, now = preferences
    stage(p, REF)
    service.client.suppress_recommendation.side_effect = TimeoutError
    for delta in [0, 61, 121, 241]:
        now[0] += delta
        result = p.check("org", [REF])
        assert result["available"] and suppression_key(REF) in result["suppressed"]
    assert service.client.suppress_recommendation.call_count == 3
    with p.store.transaction() as db:
        assert (
            db.execute("SELECT state FROM wisdom_preference_outbox").fetchone()[0]
            == "failed"
        )


def test_gateway_failure_is_not_permission_to_notify(preferences):
    p, service, _ = preferences
    service.client.recommendation_mute.side_effect = TimeoutError
    assert p.check("org", [REF])["available"] is False
    assert p.check("org", [REF])["muted"] is True


def test_remote_suppression_and_mute_are_shared_but_exact_hash_changes_are_not(
    preferences,
):
    p, service, now = preferences
    service.client.recommendation_suppressions.return_value = [
        WisdomSuppressionResponse(
            org_id="org", key=suppression_key(REF), suppress_until=iso(now[0] + 300)
        )
    ]
    result = p.check("org", [REF])
    assert result["suppressed"] == {suppression_key(REF): now[0] + 300}
    service.client.recommendation_suppressions.return_value = []
    assert p.check("org", [{**REF, "content_hash": "changed"}])["suppressed"] == {}
    service.client.recommendation_mute.return_value = WisdomMuteResponse(
        org_id="org", muted=True, duration="forever", muted_until=None, forever=True
    )
    assert p.check("org", [REF])["muted"] is True


def test_account_or_org_switch_never_flushes_previous_users_requests(preferences):
    p, service, _ = preferences
    stage(p, REF)
    service.client.identity = {"owner": "other-account"}
    p.flush("org")
    service.client.suppress_recommendation.assert_not_called()
    p.store.activate_installation_identity("installation", "other-org")
    with pytest.raises(ValueError):
        p.flush("org")


def test_expired_pending_preference_is_not_replayed_after_long_offline_period(
    preferences,
):
    p, service, now = preferences
    stage(p, REF, days=1)
    now[0] += 86401
    p.flush("org")
    service.client.suppress_recommendation.assert_not_called()


def test_expired_lease_recovers(preferences):
    p, service, now = preferences
    stage(p, REF)
    with p.store.transaction() as db:
        db.execute(
            "UPDATE wisdom_preference_outbox SET state='syncing',lease_token='abandoned',lease_until=?,attempts=1",
            (now[0] - 1,),
        )
    p.flush("org")
    with p.store.transaction() as db:
        row = db.execute(
            "SELECT state,attempts,lease_token FROM wisdom_preference_outbox"
        ).fetchone()
    assert tuple(row) == ("synced", 2, None)


def test_invalid_remote_expiry_fails_closed(preferences):
    p, service, _ = preferences
    service.client.recommendation_suppressions.return_value = [
        SimpleNamespace(key=suppression_key(REF), suppress_until="not-a-date")
    ]
    assert p.check("org", [REF])["available"] is False


def test_expired_inflight_writer_cannot_overwrite_new_claim(preferences):
    p, service, now = preferences
    stage(p, REF)
    entered, release = threading.Event(), threading.Event()
    first_expiry = now[0] + 100
    second_expiry = now[0] + 200

    def send(key):
        if not entered.is_set():
            entered.set()
            assert release.wait(5)
            until = first_expiry
        else:
            until = second_expiry
        return WisdomSuppressionResponse(
            org_id="org", key=key, suppress_until=iso(until)
        )

    service.client.suppress_recommendation.side_effect = send
    other = WisdomPreferences(
        SimpleNamespace(store=WisdomStore(p.store.path.parent), client=service.client),
        clock=lambda: now[0],
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        stale = pool.submit(p.flush, "org")
        try:
            assert entered.wait(5)
            now[0] += 61
            other.flush("org")
        finally:
            release.set()
        stale.result(timeout=5)
    with p.store.transaction() as db:
        row = db.execute(
            "SELECT state,suppress_until,attempts FROM wisdom_preference_outbox"
        ).fetchone()
    assert tuple(row) == ("synced", second_expiry, 2)
