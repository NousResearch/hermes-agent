import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_wisdom.client import WisdomError
from hermes_wisdom.client_delivery import ClientDeliveryResponse
from hermes_wisdom.delivery import DeliveryReceipt
from hermes_wisdom.delivery_outbox import DeliveryOutbox, MAX_SYNC_ATTEMPTS
from hermes_wisdom.mediation_store import LEASE_SECONDS, MediationStore
from hermes_wisdom.store import WisdomStore


@pytest.fixture
def delivery(tmp_path):
    store = WisdomStore(tmp_path / "wisdom")
    store.activate_installation_identity("installation", "org")
    now = [1000.0]
    queue = MediationStore(store, clock=lambda: now[0])
    queue.register_session(
        "org",
        session_key="session",
        session_id="session",
        platform="telegram",
        actor_id="actor",
        private=True,
        available=True,
        user_activity=True,
        address={"chat_id": "42", "thread_id": "7"},
    )
    identity = queue.enqueue(
        "org", "feed:event", {"kind": "skill", "skill_id": "skill", "version": 1}
    )
    job = queue.claim("org", "session")[0]
    assert queue.save_advice("org", identity, job["lease_token"], {"title": "Advice"})
    client = Mock(identity={"owner": "user"}, display_org_id="org")
    service = SimpleNamespace(store=store, client=client)
    outbox = DeliveryOutbox(service, clock=lambda: now[0])

    def response(request_id, reference, *, state="claimed", reason=None):
        return ClientDeliveryResponse(
            org_id="org",
            recipient_user_id="user",
            event_id="event",
            request_id=request_id,
            reference=reference,
            state=state,
            reason=reason,
            lease_until=datetime
            .fromtimestamp(now[0] + 120, timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
            if state == "claimed"
            else None,
        )

    client.claim_notification_delivery.side_effect = response
    client.settle_notification_delivery.side_effect = (
        lambda event_id, request_id, reference, *, outcome, receipt: response(
            request_id,
            reference,
            state=outcome,
        )
    )
    return SimpleNamespace(
        store=store,
        queue=queue,
        job=job,
        now=now,
        client=client,
        outbox=outbox,
        response=response,
        service=service,
    )


def receipt(**overrides):
    return DeliveryReceipt(**{
        "platform": "telegram",
        "destination": "42",
        "thread_id": "7",
        "message_id": "19",
        "acknowledgement": "provider_accepted",
        **overrides,
    })


def row(d):
    with d.store.transaction() as db:
        return dict(db.execute("SELECT * FROM wisdom_remote_delivery").fetchone())


def local(d):
    return d.queue.assessments("org")[0]


def start(d):
    request_id = d.outbox.reserve("org", d.job)
    assert request_id
    assert d.queue.begin_delivery(
        "org", d.job["id"], d.job["lease_token"], request_id=request_id
    )
    assert d.queue.delivery_ready("org", d.job["id"], d.job["lease_token"])
    return request_id


def complete(d, **kwargs):
    return d.queue.complete_delivery(
        "org", d.job["id"], d.job["lease_token"], receipt=receipt(), **kwargs
    )


def test_intent_precedes_network_and_lost_claim_reuses_exact_request(delivery):
    d = delivery

    def lost(request_id, reference):
        assert row(d)["request_id"] == request_id
        assert json.loads(row(d)["reference_json"]) == reference
        raise TimeoutError("lost response after remote commit")

    d.client.claim_notification_delivery.side_effect = lost
    with pytest.raises(TimeoutError):
        d.outbox.reserve("org", d.job)
    first = row(d)["request_id"]
    assert row(d)["state"] == "claiming" and local(d)["state"] == "ready"
    d.client.claim_notification_delivery.side_effect = d.response
    assert d.outbox.reserve("org", d.job) == first
    assert (
        d.client.claim_notification_delivery.call_args_list[0]
        == d.client.claim_notification_delivery.call_args_list[1]
    )


def test_local_and_remote_send_ownership_change_together(delivery):
    d = delivery
    request_id = start(d)
    assert local(d)["state"] == "delivering" and row(d)["state"] == "sending"
    assert not d.queue.begin_delivery(
        "org", d.job["id"], d.job["lease_token"], request_id=request_id
    )
    assert not d.queue.begin_delivery(
        "org", d.job["id"], "other", request_id=request_id
    )
    assert not d.queue.delivery_ready("org", d.job["id"], "other")


def test_begin_rolls_back_remote_transition_if_local_update_fails(delivery):
    d = delivery
    request_id = d.outbox.reserve("org", d.job)
    with d.store.transaction() as db:
        db.execute(
            "CREATE TRIGGER reject_send BEFORE UPDATE ON wisdom_assessment WHEN NEW.state='delivering' BEGIN SELECT RAISE(ABORT,'injected'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        d.queue.begin_delivery(
            "org", d.job["id"], d.job["lease_token"], request_id=request_id
        )
    assert row(d)["state"] == "claimed" and local(d)["state"] == "ready"


def test_cancel_before_transport_releases_claim_without_consuming_advice(delivery):
    d = delivery
    first = start(d)
    d.queue.cancel_delivery("org", d.job["id"], d.job["lease_token"])
    assert local(d)["state"] == "ready" and local(d)["advice"] == {"title": "Advice"}
    assert row(d)["outcome"] == "not_sent"
    d.outbox.flush("org")
    assert row(d)["state"] == "settled"
    assert d.client.settle_notification_delivery.call_args.kwargs == {
        "outcome": "not_sent",
        "receipt": None,
    }
    d.now[0] += 61
    d.job = d.queue.claim("org", "session")[0]
    second = d.outbox.reserve("org", d.job)
    assert second and second != first
    assert not d.queue.introduced("org")


def test_lost_ack_retries_only_settlement_after_restart(delivery):
    d = delivery
    request_id = start(d)
    assert complete(d, introduced=True)
    assert row(d)["outcome"] == "acknowledged" and row(d)["receipt_json"]
    d.client.settle_notification_delivery.side_effect = TimeoutError(
        "remote ack committed"
    )
    d.outbox.flush("org")
    assert row(d)["state"] == "outcome" and row(d)["attempts"] == 1
    assert local(d)["state"] == "delivered" and d.queue.introduced("org")
    d.now[0] += 61
    d.client.settle_notification_delivery.side_effect = (
        lambda event_id, request_id, reference, **kwargs: d.response(
            request_id, reference, state="acknowledged"
        )
    )
    reopened = DeliveryOutbox(
        SimpleNamespace(store=WisdomStore(d.store.root), client=d.client),
        clock=lambda: d.now[0],
    )
    reopened.flush("org")
    assert row(d)["state"] == "settled" and row(d)["request_id"] == request_id
    assert (
        d.client.settle_notification_delivery.call_args_list[0]
        == d.client.settle_notification_delivery.call_args_list[1]
    )
    assert d.client.claim_notification_delivery.call_count == 1
    with d.store.transaction() as db:
        assert (
            db.execute("SELECT COUNT(*) FROM wisdom_delivery_receipt").fetchone()[0]
            == 1
        )


@pytest.mark.parametrize("stage", ["claiming", "claimed", "sending"])
def test_abandoned_owner_recovery_never_sends(delivery, stage):
    d = delivery
    if stage == "claiming":
        d.client.claim_notification_delivery.side_effect = TimeoutError
        with pytest.raises(TimeoutError):
            d.outbox.reserve("org", d.job)
        d.client.claim_notification_delivery.side_effect = d.response
    elif stage == "claimed":
        d.outbox.reserve("org", d.job)
    else:
        start(d)
    d.now[0] += LEASE_SECONDS + 1
    d.outbox.flush("org")
    expected = "uncertain" if stage == "sending" else "not_sent"
    assert row(d)["state"] == "settled" and row(d)["outcome"] == expected
    assert d.client.settle_notification_delivery.call_args.kwargs == {
        "outcome": expected,
        "receipt": None,
    }
    assert not d.queue.introduced("org")


def test_late_valid_receipt_supersedes_uncertainty_but_never_repeats_send(delivery):
    d = delivery
    start(d)
    d.now[0] += LEASE_SECONDS + 1
    d.outbox.flush("org")
    assert row(d)["outcome"] == "uncertain"
    assert not complete(d, introduced=True)
    assert row(d)["outcome"] == "acknowledged"
    d.outbox.flush("org")
    assert local(d)["state"] == "delivered" and d.queue.introduced("org")
    assert d.client.claim_notification_delivery.call_count == 1


def test_receipt_recovers_against_original_address_if_session_moves(delivery):
    d = delivery
    start(d)
    with d.store.transaction() as db:
        db.execute(
            "UPDATE wisdom_agent_session SET address_json=?",
            (json.dumps({"chat_id": "other"}),),
        )
    assert not complete(d)
    assert row(d)["outcome"] == "acknowledged"
    d.outbox.flush("org")
    assert local(d)["state"] == "delivered"


def test_wrong_destination_and_failed_local_transaction_cannot_stage_ack(delivery):
    d = delivery
    start(d)
    with pytest.raises(ValueError):
        d.queue.complete_delivery(
            "org",
            d.job["id"],
            d.job["lease_token"],
            receipt=receipt(destination="other"),
        )
    assert row(d)["state"] == "sending" and row(d)["receipt_json"] is None
    with d.store.transaction() as db:
        db.execute(
            "CREATE TRIGGER reject_receipt BEFORE INSERT ON wisdom_delivery_receipt BEGIN SELECT RAISE(ABORT,'injected'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        complete(d, introduced=True)
    assert row(d)["state"] == "sending" and row(d)["receipt_json"] is None
    assert local(d)["state"] == "delivering" and not d.queue.introduced("org")


def test_recovery_workers_share_a_sqlite_lease(delivery):
    d = delivery
    start(d)
    complete(d)
    entered, release = Event(), Event()

    def settle(event_id, request_id, reference, **kwargs):
        entered.set()
        assert release.wait(5)
        return d.response(request_id, reference, state="acknowledged")

    d.client.settle_notification_delivery.side_effect = settle
    other = DeliveryOutbox(
        SimpleNamespace(store=WisdomStore(d.store.root), client=d.client),
        clock=lambda: d.now[0],
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        task = pool.submit(d.outbox.flush, "org")
        try:
            assert entered.wait(5)
            other.flush("org")
            assert d.client.settle_notification_delivery.call_count == 1
        finally:
            release.set()
        task.result(timeout=5)
    assert row(d)["state"] == "settled"


def test_late_receipt_fences_an_in_flight_uncertainty_settlement(delivery):
    d = delivery
    start(d)
    d.now[0] += LEASE_SECONDS + 1
    entered, release = Event(), Event()

    def settle(event_id, request_id, reference, **kwargs):
        entered.set()
        assert release.wait(5)
        return d.response(request_id, reference, state=kwargs["outcome"])

    d.client.settle_notification_delivery.side_effect = settle
    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(d.outbox.flush, "org")
        try:
            assert entered.wait(5)
            assert not complete(d, introduced=True)
            assert row(d)["outcome"] == "acknowledged"
        finally:
            release.set()
        task.result(timeout=5)
    d.outbox.flush("org")
    assert row(d)["state"] == "settled" and row(d)["outcome"] == "acknowledged"
    assert local(d)["state"] == "delivered" and d.queue.introduced("org")
    assert d.client.claim_notification_delivery.call_count == 1
    assert [
        call.kwargs["outcome"]
        for call in d.client.settle_notification_delivery.call_args_list
    ] == ["uncertain", "acknowledged"]


def test_identity_switch_does_not_sync_other_accounts_receipts(delivery):
    d = delivery
    start(d)
    complete(d)
    d.client.identity = {"owner": "other"}
    d.outbox.flush("org")
    d.client.settle_notification_delivery.assert_not_called()
    d.store.activate_installation_identity("installation", "other-org")
    with pytest.raises(ValueError, match="organization"):
        d.outbox.flush("org")
    d.client.settle_notification_delivery.assert_not_called()


def test_retry_exhaustion_keeps_receipt_without_resending(delivery):
    d = delivery
    start(d)
    complete(d)
    d.client.settle_notification_delivery.side_effect = TimeoutError
    for _ in range(MAX_SYNC_ATTEMPTS + 2):
        d.outbox.flush("org")
        d.now[0] += 3601
    assert row(d)["attempts"] == MAX_SYNC_ATTEMPTS
    assert row(d)["state"] == "outcome" and row(d)["receipt_json"]
    assert d.client.settle_notification_delivery.call_count == MAX_SYNC_ATTEMPTS
    assert d.client.claim_notification_delivery.call_count == 1


def test_remote_acknowledgement_keeps_passive_advice_without_introduction_or_read(
    delivery,
):
    d = delivery
    d.client.claim_notification_delivery.side_effect = lambda request_id, reference: (
        d.response(request_id, reference, state="acknowledged")
    )
    assert d.outbox.reserve("org", d.job) is None
    assert local(d)["state"] == "delivered" and local(d)["advice"] == {
        "title": "Advice"
    }
    assert local(d)["delivered_at"] is None and not d.queue.introduced("org")
    with d.store.transaction() as db:
        assert (
            db.execute("SELECT COUNT(*) FROM wisdom_delivery_receipt").fetchone()[0]
            == 0
        )


def test_candidate_reservation_only_uploads_opaque_reference(delivery):
    d = delivery
    d.job["reference"] = {
        "kind": "candidate",
        "content_hash": "private-hash",
        "name": "Private",
        "path": "/private/skill",
    }
    assert d.outbox.reserve("org", d.job)
    reference = d.client.claim_notification_delivery.call_args.args[1]
    assert set(reference) == {"kind", "key"} and reference["key"].startswith("sha256:")
    assert "private" not in json.dumps(reference)


def test_expired_remote_lease_cannot_start_a_send(delivery):
    d = delivery
    request_id = d.outbox.reserve("org", d.job)
    d.now[0] += 111
    assert not d.queue.begin_delivery(
        "org", d.job["id"], d.job["lease_token"], request_id=request_id
    )
    assert row(d)["state"] == "claimed" and local(d)["state"] == "ready"


@pytest.mark.parametrize(
    "changed", ["account", "address", "remote_lease", "local_lease"]
)
def test_final_send_check_revalidates_identity_destination_and_both_leases(
    delivery, changed
):
    d = delivery
    start(d)
    user = "user"
    if changed == "account":
        user = "other"
    elif changed == "address":
        with d.store.transaction() as db:
            db.execute(
                "UPDATE wisdom_agent_session SET address_json=?",
                (json.dumps({"chat_id": "other"}),),
            )
    elif changed == "remote_lease":
        d.now[0] += 121
    else:
        d.now[0] += LEASE_SECONDS + 1
    assert not d.queue.delivery_ready(
        "org", d.job["id"], d.job["lease_token"], user_id=user
    )


def test_not_sent_conflict_is_retired_but_uncertain_conflict_is_not(delivery):
    d = delivery
    start(d)
    d.queue.cancel_delivery("org", d.job["id"], d.job["lease_token"])
    d.client.settle_notification_delivery.side_effect = WisdomError(
        "different owner", status=409
    )
    d.outbox.flush("org")
    assert row(d)["state"] == "settled" and row(d)["outcome"] == "not_sent"
    d.now[0] += 61
    d.job = d.queue.claim("org", "session")[0]
    start(d)
    d.queue.uncertain_delivery("org", d.job["id"], d.job["lease_token"])
    d.outbox.flush("org")
    assert row(d)["state"] == "outcome" and row(d)["outcome"] == "uncertain"
    assert row(d)["last_error"] == "settlement_conflict"


def mediation_for(d, monkeypatch):
    from hermes_wisdom.mediation import WisdomMediation

    d.service.require_setup = Mock()
    instance = WisdomMediation(d.service, clock=lambda: d.now[0])
    # Isolate delivery from the already-covered preference and canonical feed checks.
    monkeypatch.setattr("hermes_wisdom.mediation.delivery_mode", lambda: "agent")
    monkeypatch.setattr(instance, "_eligible_jobs", lambda org, jobs: jobs)
    monkeypatch.setattr(instance, "_current_feed_jobs", lambda org, jobs: jobs)
    return instance


@pytest.mark.parametrize("failure", [None, "timeout", "malformed", "deferred"])
def test_mediation_requires_valid_reservation_before_local_send(
    delivery, monkeypatch, failure
):
    d = delivery
    instance = mediation_for(d, monkeypatch)
    if failure == "timeout":
        d.client.claim_notification_delivery.side_effect = TimeoutError
    elif failure == "malformed":
        d.client.claim_notification_delivery.side_effect = lambda *args: {
            "state": "claimed"
        }
    elif failure == "deferred":
        d.client.claim_notification_delivery.side_effect = (
            lambda request_id, reference: d.response(
                request_id, reference, state="deferred", reason="recipient_muted"
            )
        )
    item = {"assessment": d.job, "advice": {"title": "Advice"}, "interaction": None}
    assert instance.begin_delivery("org", [item]) == ([] if failure else [item])
    assert local(d)["state"] == ("ready" if failure else "delivering")
    assert local(d)["advice"] == {"title": "Advice"}


@pytest.mark.parametrize("kind", ["manual", "outcome", "fixed"])
def test_explicit_and_fixed_work_does_not_reserve_automatic_notifications(
    delivery, monkeypatch, kind
):
    d = delivery
    instance = mediation_for(d, monkeypatch)
    if kind == "manual":
        d.job["reference"]["user_requested"] = True
    elif kind == "outcome":
        d.job["reference"] = {"kind": "outcome"}
    else:
        monkeypatch.setattr("hermes_wisdom.mediation.delivery_mode", lambda: "fixed")
    item = {"assessment": d.job, "advice": {"title": "Advice"}, "interaction": None}
    assert instance.begin_delivery("org", [item]) == ([] if kind == "fixed" else [item])
    d.client.claim_notification_delivery.assert_not_called()


@pytest.mark.parametrize(
    "outcome", ["accepted", "disconnected", "new_turn", "approval", "render_error"]
)
def test_embedded_and_native_tui_delivery_uses_same_reservation_and_outbox(
    delivery, monkeypatch, outcome
):
    import threading
    import time
    from contextlib import nullcontext

    from tui_gateway.wisdom_mediation import poll

    d = delivery
    instance = mediation_for(d, monkeypatch)
    monkeypatch.setattr(
        "tui_gateway.wisdom_mediation.WisdomMediation", lambda _: instance
    )
    monkeypatch.setattr("hermes_wisdom.service.WisdomService", lambda: d.service)
    monkeypatch.setattr(instance.queue, "claim_refresh", lambda _: False)
    item = {
        "assessment": d.job,
        "advice": {
            "title": "Advice",
            "explanation": "For review",
            "relevance": "digest",
        },
        "interaction": None,
    }
    monkeypatch.setattr(instance, "prepare", lambda *args, **kwargs: [item])
    session = {
        "session_key": "session",
        "_wisdom_user_activity": time.time(),
        "agent": Mock(),
        "history_lock": threading.Lock(),
        "history": [],
        "running": False,
    }
    live = {"connected": True, "approval": False}
    monkeypatch.setattr(
        "tools.approval.get_pending_gateway_approval", lambda _: live["approval"]
    )
    monkeypatch.setattr("tools.clarify_gateway.has_pending", lambda _: False)

    def claim(request_id, reference):
        if outcome == "disconnected":
            live["connected"] = False
        elif outcome == "new_turn":
            session["_queued_prompt_generation"] = 1
        elif outcome == "approval":
            live["approval"] = True
        return d.response(request_id, reference)

    d.client.claim_notification_delivery.side_effect = claim
    if outcome == "render_error":
        monkeypatch.setattr(
            "tui_gateway.wisdom_mediation.advice_view",
            Mock(side_effect=ValueError("invalid view")),
        )
    emit = Mock(return_value=True)
    if outcome == "render_error":
        with pytest.raises(ValueError, match="invalid view"):
            poll(
                session,
                emit=emit,
                profile_scope=lambda _: nullcontext(),
                connected=lambda: live["connected"],
            )
    else:
        poll(
            session,
            emit=emit,
            profile_scope=lambda _: nullcontext(),
            connected=lambda: live["connected"],
        )
    d.outbox.flush("org")
    assert session["history"] == []
    d.client.claim_notification_delivery.assert_called_once()
    if outcome == "accepted":
        emit.assert_called_once()
        assert local(d)["state"] == "delivered"
        assert row(d)["outcome"] == "acknowledged"
        ack = d.client.settle_notification_delivery.call_args.kwargs["receipt"]
        assert ack.platform == "local" and ack.acknowledgement == "transport_accepted"
        assert ack.destination == "local:session"
    else:
        emit.assert_not_called()
        assert local(d)["state"] == "ready"
        assert row(d)["outcome"] == "not_sent"
