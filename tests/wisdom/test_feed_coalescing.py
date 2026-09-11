from copy import deepcopy
from datetime import datetime, timezone

import pytest

from hermes_wisdom.delivery import DeliveryReceipt
from hermes_wisdom.client_delivery import ClientDeliveryResponse
from hermes_wisdom.delivery_outbox import DeliveryOutbox
from hermes_wisdom.mediation_store import MediationStore
from hermes_wisdom.store import WisdomStore
from tests.wisdom.test_feed_mediation import event, mediation  # noqa: F401
from tests.wisdom.test_mediation import consent  # noqa: F401


@pytest.mark.parametrize("versions", [(1, 2, 3, 3), (3, 1, 3, 2)])
@pytest.mark.parametrize("legacy_queue", [False, True])
@pytest.mark.parametrize("manual_key", ["requested:skill-v1", "feed:manual-request"])
def test_only_latest_automatic_arrival_is_assessed_after_restart(
    mediation,
    versions,
    legacy_queue,
    manual_key,
):
    instance, actor, now, _ = mediation
    events = [
        dict(event(version=version), event_id=f"event-{index}")
        for index, version in enumerate(versions)
    ]
    events.append(dict(event(version=2), skill_id="other-skill", event_id="other"))
    instance.service.notifications.return_value = {"events": events}
    if legacy_queue:
        from hermes_wisdom.mediation import _feed_reference

        for arrival in events:
            instance.queue.enqueue(
                "org", f"feed:{arrival['event_id']}", _feed_reference(arrival)
            )
    else:
        instance.ingest()
    manual = instance.queue.enqueue(
        "org",
        manual_key,
        {
            "kind": "skill",
            "skill_id": "skill",
            "version": 1,
            "user_requested": True,
        },
        origin_session=actor.session_key,
    )
    notice = instance.queue.enqueue(
        "org",
        "feed:decision",
        {
            "kind": "notice",
            "skill_id": "skill",
            "version": 999,
            "notification": {"category": "publication_decision"},
        },
    )
    instance.queue = MediationStore(
        WisdomStore(instance.service.store.path.parent),
        clock=lambda: now[0],
    )
    template = deepcopy(instance.service.version_detail.return_value)

    def detail(skill_id, version):
        result = deepcopy(template)
        result["skill"]["id"] = skill_id
        result["version"]["version"] = version
        return result

    instance.service.version_detail.side_effect = detail
    assessed = []

    def assess(evidence, **kwargs):
        assessed.extend(evidence)
        return {
            item["assessment_id"]: {
                "title": "Arrival",
                "explanation": "Existing workflow.",
                "relevance": "digest",
            }
            for item in evidence
        }

    result = instance.prepare(
        "org",
        actor,
        runtime={"model": "test", "provider": "test"},
        history=[],
        assessor=assess,
    )
    assert {
        (
            item["assessment"]["reference"]["kind"],
            item["assessment"]["reference"]["skill_id"],
            item["assessment"]["reference"]["version"],
        )
        for item in result
    } == {
        ("skill", "skill", max(versions)),
        ("skill", "other-skill", 2),
        ("skill", "skill", 1),
        ("notice", "skill", 999),
    }
    assert len(result) == len(assessed) == 4
    rows = instance.queue.assessments("org")
    assert len(rows) == len(events) + 2
    assert sum(row["state"] == "retired" for row in rows) == len(events) - 2
    assert next(row for row in rows if row["id"] == manual)["state"] == "ready"
    assert next(row for row in rows if row["id"] == notice)["state"] == "ready"
    assert instance.service.notifications(mark_seen=False)["events"] == events
    instance.service.install_apply.assert_not_called()


@pytest.mark.parametrize(
    "state",
    ["ready", "assessing", "fallback", "delivering", "delivered", "delivery_uncertain"],
)
@pytest.mark.parametrize("new_version", [1, 2])
def test_coalescing_fences_unsent_work_without_replaying_or_losing_receipts(
    consent,
    state,
    new_version,
):
    instance, actor, identity, now = consent
    card = instance.present("org", identity, actor)
    job = instance.queue.assessments("org")[0]
    token = job["lease_token"]
    if state in {"delivering", "delivered", "delivery_uncertain"}:
        outbox = DeliveryOutbox(instance.service, clock=lambda: now[0])

        def response(request_id, reference, *, state="claimed"):
            return ClientDeliveryResponse(
                org_id="org",
                recipient_user_id="account-user",
                event_id="remote-event",
                request_id=request_id,
                reference=reference,
                state=state,
                reason=None,
                lease_until=datetime
                .fromtimestamp(now[0] + 120, timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
                if state == "claimed"
                else None,
            )

        instance.service.client.claim_notification_delivery.side_effect = response
        instance.service.client.settle_notification_delivery.side_effect = (
            lambda event_id, request_id, reference, *, outcome, receipt: response(
                request_id, reference, state=outcome
            )
        )
        request_id = outbox.reserve("org", job)
        assert request_id
        assert instance.queue.begin_delivery(
            "org", identity, token, request_id=request_id
        )
        if state == "delivered":
            assert instance.queue.complete_delivery(
                "org",
                identity,
                token,
                receipt=DeliveryReceipt(
                    platform=actor.platform,
                    destination=actor.chat_id,
                    thread_id=actor.thread_id,
                    message_id="message",
                    acknowledgement="provider_accepted",
                ),
            )
        elif state == "delivery_uncertain":
            instance.queue.uncertain_delivery("org", identity, token)
    else:
        with instance.service.store.transaction() as db:
            db.execute(
                "UPDATE wisdom_assessment SET state=? WHERE id=?", (state, identity)
            )
    before = instance.queue.assessments("org")[0]
    reference = {
        "kind": "skill",
        "skill_id": "skill",
        "version": new_version,
        "event_id": "later",
    }
    later = instance.queue.reconcile_feed("org", "later", reference)
    instance.queue.reconcile_feed("org", "later", reference)
    rows = {row["id"]: row for row in instance.queue.assessments("org")}
    replaced = new_version > 1 and state in {"ready", "assessing", "fallback"}
    assert rows[identity]["state"] == ("retired" if replaced else state)
    assert rows[later]["state"] == ("retired" if new_version == 1 else "pending")
    assert rows[identity].get("advice") == before.get("advice")
    assert rows[identity]["delivered_at"] == before["delivered_at"]
    with instance.service.store.transaction() as db:
        assert db.execute(
            "SELECT state FROM wisdom_consent WHERE id=?", (card["id"],)
        ).fetchone()[0] == ("stale" if replaced else "pending")
        assert db.execute(
            "SELECT count(*) FROM wisdom_delivery_receipt WHERE assessment_id=?",
            (identity,),
        ).fetchone()[0] == int(state == "delivered")
    if replaced:
        assert not instance.queue.save_advice("org", identity, token, {"title": "late"})
        assert not instance.queue.begin_delivery("org", identity, token)
    if state == "delivering":
        assert instance.queue.delivery_ready("org", identity, token) is (
            new_version == 1
        )
        # A newer arrival cannot invalidate a receipt for a send already dispatched.
        assert instance.queue.complete_delivery(
            "org",
            identity,
            token,
            receipt=DeliveryReceipt(
                platform=actor.platform,
                destination=actor.chat_id,
                thread_id=actor.thread_id,
                message_id="late-message",
                acknowledgement="provider_accepted",
            ),
        )
    if state in {"delivering", "delivered"}:
        outbox.flush("org")
        assert (
            instance.service.client.settle_notification_delivery.call_args.kwargs[
                "outcome"
            ]
            == "acknowledged"
        )
        with instance.service.store.transaction() as db:
            assert db.execute(
                "SELECT state,outcome FROM wisdom_remote_delivery WHERE assessment_id=?",
                (identity,),
            ).fetchone()[:] == ("settled", "acknowledged")
