import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest

from hermes_wisdom.client import WisdomError
from hermes_wisdom.client_delivery import ClientDeliveryResponse
from hermes_wisdom.client_outcome import ClientOperationResponse
from hermes_wisdom.delivery import DeliveryReceipt
from hermes_wisdom.delivery_outbox import DeliveryOutbox
from hermes_wisdom.mediation import WisdomMediation
from hermes_wisdom.operation_outbox import MAX_ATTEMPTS, SYNC_LEASE, OperationOutbox
from hermes_wisdom.preferences import suppression_key
from tests.wisdom.test_mediation import consent  # noqa: F401
from tests.wisdom.test_share_queue import sharing  # noqa: F401
from tests.wisdom.test_share_staging import staged  # noqa: F401


def reserve(service, queue, job, now, actor):
    client = service.client
    client.identity = {"owner": "account-user"}
    client.display_org_id = "org"

    def response(request_id, reference, *, state="claimed"):
        return ClientDeliveryResponse(
            org_id="org",
            recipient_user_id="account-user",
            event_id="gateway-event",
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

    client.claim_notification_delivery = Mock(side_effect=response)
    client.settle_notification_delivery = Mock(
        side_effect=lambda event_id, request_id, reference, *, outcome, receipt: (
            response(request_id, reference, state=outcome)
        )
    )
    client.report_operation_outcome = Mock(
        side_effect=lambda event_id, outcome: ClientOperationResponse(
            org_id="org",
            recipient_user_id="account-user",
            event_id=event_id,
            outcome=outcome,
            attestation="client_reported",
            duplicate=False,
        )
    )
    delivery = DeliveryOutbox(service, clock=lambda: now[0])
    request_id = delivery.reserve("org", job)
    assert queue.begin_delivery(
        "org", job["id"], job["lease_token"], request_id=request_id
    )

    def acknowledge():
        assert queue.complete_delivery(
            "org",
            job["id"],
            job["lease_token"],
            receipt=DeliveryReceipt(
                platform=actor.platform,
                destination=actor.chat_id,
                thread_id=actor.thread_id,
                scope_id=actor.scope_id,
                message_id="private-message",
                acknowledgement="provider_accepted",
            ),
        )

    return delivery, acknowledge


@pytest.fixture
def outcome(consent):
    instance, actor, identity, now = consent
    shown = instance.present("org", identity, actor)
    delivery, ack = reserve(
        instance.service,
        instance.queue,
        instance.queue.assessments("org")[0],
        now,
        actor,
    )
    return SimpleNamespace(
        instance=instance,
        service=instance.service,
        actor=actor,
        shown=shown,
        now=now,
        delivery=delivery,
        ack=ack,
        outbox=OperationOutbox(instance.service, clock=lambda: now[0]),
    )


def rows(o):
    with o.service.store.transaction() as db:
        return [dict(r) for r in db.execute("SELECT * FROM wisdom_operation_outbox")]


def finish(o):
    return o.instance.resolve("org", o.shown["id"], o.actor, "confirm")


def ready(o):
    assert finish(o)["state"] == "completed"
    o.ack()
    o.delivery.flush("org")


def test_operation_is_staged_atomically_and_waits_for_delivery_ack(outcome):
    o = outcome
    assert finish(o)["state"] == "completed"
    assert len(rows(o)) == 1
    assert rows(o)[0]["state"] == "pending"
    o.service.install_apply.assert_called_once()
    o.outbox.flush("org")
    o.service.client.report_operation_outcome.assert_not_called()
    o.ack()
    o.outbox.flush("org")
    o.service.client.report_operation_outcome.assert_not_called()
    WisdomMediation(o.service, clock=lambda: o.now[0]).flush_delivery("org")
    o.service.client.report_operation_outcome.assert_called_once()
    assert rows(o)[0]["state"] == "settled"
    sent = o.service.client.report_operation_outcome.call_args.args[1]
    assert set(sent) == {"request_id", "operation_key", "operation", "state"}
    assert sent["operation"] == "install" and sent["state"] == "completed"
    for secret in [
        o.shown["id"],
        "skill",
        "wip_one",
        "private-message",
        "account-user",
    ]:
        assert secret not in json.dumps(sent)
    finish(o)
    o.outbox.flush("org")
    o.service.install_apply.assert_called_once()
    o.service.client.report_operation_outcome.assert_called_once()


def test_lost_reply_retries_same_report_without_reapplying(outcome):
    o = outcome
    ready(o)
    success = o.service.client.report_operation_outcome.side_effect
    received = []

    def lost_reply(event, report):
        received.append(report)
        raise TimeoutError("secret provider output")

    o.service.client.report_operation_outcome.side_effect = lost_reply
    o.outbox.flush("org")
    assert rows(o)[0]["last_error"] == "outcome_unavailable"
    assert finish(o)["state"] == "completed"
    o.now[0] += 60
    o.service.client.report_operation_outcome.side_effect = success
    o.outbox.flush("org")
    assert o.service.client.report_operation_outcome.call_args.args[1] == received[0]
    assert rows(o)[0]["state"] == "settled"
    o.service.install_apply.assert_called_once()


def test_concurrent_pollers_claim_one_report_and_stale_worker_cannot_overwrite(outcome):
    o = outcome
    ready(o)
    entered, release = Event(), Event()
    success = o.service.client.report_operation_outcome.side_effect

    def slow(event, report):
        entered.set()
        assert release.wait(5)
        raise TimeoutError()

    o.service.client.report_operation_outcome.side_effect = slow
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(o.outbox.flush, "org")
        assert entered.wait(5)
        OperationOutbox(o.service, clock=lambda: o.now[0]).flush("org")
        assert o.service.client.report_operation_outcome.call_count == 1
        o.now[0] += SYNC_LEASE + 1
        o.service.client.report_operation_outcome.side_effect = success
        OperationOutbox(o.service, clock=lambda: o.now[0]).flush("org")
        assert rows(o)[0]["state"] == "settled"
        release.set()
        first.result()
    assert rows(o)[0]["state"] == "settled" and rows(o)[0]["last_error"] is None
    o.service.install_apply.assert_called_once()


@pytest.mark.parametrize("kind", ["org", "user", "during_request"])
def test_account_isolation_keeps_report_for_original_identity(outcome, kind):
    o = outcome
    ready(o)
    if kind == "org":
        o.service.store.activate_installation_identity("other-installation", "other")
        with pytest.raises(ValueError, match="no longer active"):
            o.outbox.flush("org")
    elif kind == "user":
        o.service.client.identity["owner"] = "other"
        o.outbox.flush("org")
    else:
        success = o.service.client.report_operation_outcome.side_effect

        def switch(event, report):
            o.service.client.identity["owner"] = "other"
            return success(event, report)

        o.service.client.report_operation_outcome.side_effect = switch
        o.outbox.flush("org")
    assert rows(o)[0]["state"] == "pending"
    if kind != "during_request":
        o.service.client.report_operation_outcome.assert_not_called()


@pytest.mark.parametrize("conflict", [False, True])
def test_bounded_failures_preserve_native_result_without_retrying_work(
    outcome, conflict
):
    o = outcome
    ready(o)
    o.service.client.report_operation_outcome.side_effect = WisdomError(
        "untrusted", status=409 if conflict else 503
    )
    for _ in range(MAX_ATTEMPTS + 2):
        o.outbox.flush("org")
        o.now[0] += 4000
    assert rows(o)[0]["state"] == "failed"
    assert o.service.client.report_operation_outcome.call_count == (
        1 if conflict else MAX_ATTEMPTS
    )
    assert finish(o)["state"] == "completed"
    o.service.install_apply.assert_called_once()


def test_last_crashed_attempt_records_failure_after_lease(outcome):
    o = outcome
    ready(o)
    with o.service.store.transaction() as db:
        db.execute(
            "UPDATE wisdom_operation_outbox SET attempts=?,sync_token='dead',sync_until=?",
            (MAX_ATTEMPTS, o.now[0] + SYNC_LEASE),
        )
    o.outbox.flush("org")
    assert rows(o)[0]["state"] == "pending"
    o.now[0] += SYNC_LEASE
    o.outbox.flush("org")
    assert rows(o)[0]["state"] == "failed"
    o.service.client.report_operation_outcome.assert_not_called()


def test_bad_response_cannot_acknowledge_local_outcome(outcome):
    o = outcome
    ready(o)
    o.service.client.report_operation_outcome.side_effect = lambda event, report: (
        ClientOperationResponse(
            org_id="other",
            recipient_user_id="account-user",
            event_id=event,
            outcome=report,
            attestation="client_reported",
            duplicate=False,
        )
    )
    o.outbox.flush("org")
    assert rows(o)[0]["state"] == "pending"


def test_failed_atomic_staging_leaves_applying_for_journal_recovery(outcome):
    o = outcome
    with o.service.store.transaction() as db:
        db.execute(
            "CREATE TRIGGER reject_outcome BEFORE INSERT ON wisdom_operation_outbox BEGIN SELECT RAISE(ABORT,'injected'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="injected"):
        finish(o)
    with o.service.store.transaction() as db:
        assert (
            db.execute(
                "SELECT state FROM wisdom_consent WHERE id=?", (o.shown["id"],)
            ).fetchone()[0]
            == "applying"
        )
        assert (
            db.execute("SELECT COUNT(*) FROM wisdom_consent_outcome").fetchone()[0] == 0
        )
    assert rows(o) == []
    o.service.install_apply.assert_called_once()
    # The actual apply's exact receipt is the authority for recovery, not a
    # second call to install_apply or just finding a skill with the same name.
    journal = o.service.store.journal(
        "install", "skill", "done", {"receipt": "wip_one"}
    )
    o.service.store.advance(journal, "done", done=True)
    with o.service.store.transaction() as db:
        db.execute("DROP TRIGGER reject_outcome")
    o.now[0] += 901
    o.instance.recover("org")
    assert rows(o)[0]["state"] == "pending"
    assert json.loads(rows(o)[0]["outcome_json"])["state"] == "completed"
    o.instance.recover("org")
    assert len(rows(o)) == 1
    o.service.install_apply.assert_called_once()


@pytest.mark.parametrize("change", ["hash", "version", "no_delivery"])
def test_unrelated_or_manual_consent_does_not_fabricate_remote_evidence(
    outcome, change
):
    o = outcome
    with o.service.store.transaction() as db:
        if change == "no_delivery":
            db.execute("DELETE FROM wisdom_remote_delivery")
        else:
            ref = {
                "kind": "skill",
                "skill_id": "skill" if change == "version" else "other",
                "version": 2 if change == "version" else 1,
                "event_type": "teammate_published",
            }
            db.execute(
                "UPDATE wisdom_remote_delivery SET reference_json=?", (json.dumps(ref),)
            )
    assert finish(o)["state"] == "completed"
    assert rows(o) == []


@pytest.mark.parametrize("result", ["stale", "needs_review", "expired"])
def test_incomplete_native_results_are_not_reported_as_success(outcome, result):
    o = outcome
    if result == "stale":
        o.service.install_plan.return_value["content_hash"] = "changed"
    elif result == "needs_review":
        o.service.install_apply.side_effect = RuntimeError("secret")
    else:
        o.now[0] = o.shown["expires_at"] + 1
    assert finish(o)["state"] == result
    assert json.loads(rows(o)[0]["outcome_json"])["state"] == result
    assert "secret" not in rows(o)[0]["outcome_json"]


def attach_shared_delivery(sharing):
    service, mediation, actor, shown, _, _, now = sharing
    service.client.identity = {"owner": "account-user"}
    service.client.display_org_id = "org"
    service.client.report_operation_outcome = Mock(
        side_effect=lambda event_id, outcome: ClientOperationResponse(
            org_id="org",
            recipient_user_id="account-user",
            event_id=event_id,
            outcome=outcome,
            attestation="client_reported",
            duplicate=False,
        )
    )
    job = mediation.queue.assessments("org")[0]
    request_id = str(uuid4())
    # The sharing fixture already has a validated local delivered receipt.
    # Pair that fixture with its acknowledged Gateway reservation.
    with service.store.transaction() as db:
        receipt = db.execute(
            "SELECT receipt_json FROM wisdom_delivery_receipt WHERE assessment_id=?",
            (job["id"],),
        ).fetchone()[0]
        db.execute(
            """INSERT INTO wisdom_remote_delivery(request_id,organization_id,user_id,
            assessment_id,reference_json,local_token,owner_session,platform,address_json,state,event_id,
            outcome,receipt_json,available_at) VALUES(?,'org','account-user',?,?,'delivered',?,?,?,'settled',
            'gateway-candidate','acknowledged',?,?)""",
            (
                request_id,
                job["id"],
                json.dumps({
                    "kind": "candidate",
                    "key": suppression_key(job["reference"]),
                }),
                actor.session_key,
                actor.platform,
                json.dumps(actor.address),
                receipt,
                now[0],
            ),
        )
    return request_id


def test_share_and_separate_publication_keep_the_same_delivery_but_distinct_outcomes(
    sharing,
):
    request = attach_shared_delivery(sharing)
    service, mediation, actor, shown, _, source, now = sharing
    first = mediation.consent.resolve("org", shown["id"], actor, "confirm")
    assert (
        first["result"]["packaging_state"] == "queued"
        and service.client.publications == 0
    )
    final = mediation.prepare("org", actor, runtime={}, history=[], assessor=Mock())[0][
        "interaction"
    ]
    assert service.client.report_operation_outcome.call_count == 1
    queued = service.client.report_operation_outcome.call_args.args[1]
    assert queued["operation"] == "share" and queued["state"] == "queued"
    assert queued["request_id"] == request
    result = mediation.consent.resolve("org", final["id"], actor, "confirm")
    assert result["state"] == "completed" and service.client.publications == 1
    OperationOutbox(service, clock=lambda: now[0]).flush("org")
    published = service.client.report_operation_outcome.call_args.args[1]
    assert published["operation"] == "publish" and published["state"] == "completed"
    assert published["request_id"] == request
    assert published["operation_key"] != queued["operation_key"]
    assert service.client.report_operation_outcome.call_count == 2
    for secret in [
        str(source),
        "ORIGINAL_PRIVATE_SETUP",
        shown["id"],
        final["id"],
        "notes",
    ]:
        assert secret not in json.dumps(
            service.client.report_operation_outcome.call_args_list, default=str
        )
    mediation.consent.resolve("org", final["id"], actor, "confirm")
    mediation.flush_delivery("org")
    assert (
        service.client.report_operation_outcome.call_count == 2
        and service.client.publications == 1
    )


def test_share_queue_and_report_roll_back_with_native_acceptance(sharing):
    attach_shared_delivery(sharing)
    service, mediation, actor, shown, model, _, _ = sharing
    with service.store.transaction() as db:
        db.execute(
            "CREATE TRIGGER reject_outcome BEFORE INSERT ON wisdom_operation_outbox BEGIN SELECT RAISE(ABORT,'injected'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="injected"):
        mediation.consent.resolve("org", shown["id"], actor, "confirm")
    assert len(mediation.queue.assessments("org")) == 1
    with service.store.transaction() as db:
        assert (
            db.execute(
                "SELECT state FROM wisdom_consent WHERE id=?", (shown["id"],)
            ).fetchone()[0]
            == "pending"
        )
        assert (
            db.execute("SELECT COUNT(*) FROM wisdom_operation_outbox").fetchone()[0]
            == 0
        )
    model.assert_not_called()
    assert service.client.publications == 0


@pytest.mark.parametrize("mismatch", ["parent_job", "candidate_hash"])
def test_publication_cannot_attach_to_an_unrelated_parent_delivery(sharing, mismatch):
    attach_shared_delivery(sharing)
    service, mediation, actor, shown, _, _, _ = sharing
    mediation.consent.resolve("org", shown["id"], actor, "confirm")
    final = mediation.prepare("org", actor, runtime={}, history=[], assessor=Mock())[0][
        "interaction"
    ]
    with service.store.transaction() as db:
        if mismatch == "parent_job":
            db.execute(
                "UPDATE wisdom_consent SET result_json=json_set(result_json,'$.assessment_id','other') WHERE id=?",
                (shown["id"],),
            )
        else:
            db.execute(
                "UPDATE wisdom_consent SET plan_json=json_set(plan_json,'$.source_hash','other') WHERE id=?",
                (shown["id"],),
            )
    assert (
        mediation.consent.resolve("org", final["id"], actor, "confirm")["state"]
        == "completed"
    )
    with service.store.transaction() as db:
        assert (
            db.execute("SELECT COUNT(*) FROM wisdom_operation_outbox").fetchone()[0]
            == 1
        )
