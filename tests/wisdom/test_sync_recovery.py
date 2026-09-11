import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from hermes_wisdom.client import WisdomError
from hermes_wisdom.delivery_outbox import MAX_SYNC_ATTEMPTS
from hermes_wisdom.operation_outbox import MAX_ATTEMPTS, SYNC_LEASE
from hermes_wisdom.sync_recovery import WisdomSyncRecovery
from tests.wisdom.test_delivery_outbox import (  # noqa: F401
    complete,
    delivery,
    local,
    row,
    start,
)
from tests.wisdom.test_operation_outbox import (  # noqa: F401
    consent,
    finish,
    outcome,
    ready,
    rows,
)


def recovery(o):
    return WisdomSyncRecovery(o.service, clock=lambda: o.now[0])


def fail_report(o):
    ready(o)
    success = o.service.client.report_operation_outcome.side_effect
    o.service.client.report_operation_outcome.side_effect = TimeoutError(
        "private output"
    )
    for _ in range(MAX_ATTEMPTS):
        o.outbox.flush("org")
        o.now[0] += 4000
    o.service.client.report_operation_outcome.side_effect = success


def test_status_is_private_metadata_only_and_read_only(outcome):
    o = outcome
    fail_report(o)
    before = rows(o)
    result = recovery(o).status()
    assert result["operation"]["retryable"] == 1 and result["can_retry"]
    assert result["delivery"]["retryable"] == 0
    assert rows(o) == before
    serialized = json.dumps(result)
    for private in (
        "account-user",
        "private output",
        "gateway-event",
        "sha256:",
        "skill",
        o.shown["id"],
    ):
        assert private not in serialized


def test_retry_keeps_exact_report_and_never_reapplies(outcome):
    o = outcome
    fail_report(o)
    before = rows(o)[0]
    result = recovery(o).retry()
    assert not result["can_retry"]
    assert rows(o)[0]["state"] == "settled"
    assert rows(o)[0]["outcome_json"] == before["outcome_json"]
    assert rows(o)[0]["request_id"] == before["request_id"]
    count = o.service.client.report_operation_outcome.call_count
    recovery(o).retry()
    assert o.service.client.report_operation_outcome.call_count == count
    o.service.install_apply.assert_called_once()


def test_retry_recovers_saved_receipt_without_another_send(delivery):
    d = delivery
    request = start(d)
    complete(d)
    accepted = row(d)["receipt_json"]
    success = d.client.settle_notification_delivery.side_effect
    d.client.settle_notification_delivery.side_effect = TimeoutError()
    for _ in range(MAX_SYNC_ATTEMPTS):
        d.outbox.flush("org")
        d.now[0] += 4000
    assert recovery(d).status()["delivery"]["retryable"] == 1
    d.client.settle_notification_delivery.side_effect = success
    result = recovery(d).retry()
    assert not result["can_retry"]
    assert row(d)["request_id"] == request and row(d)["receipt_json"] == accepted
    assert row(d)["outcome"] == "acknowledged" and row(d)["state"] == "settled"
    assert local(d)["state"] == "delivered"
    d.client.claim_notification_delivery.assert_called_once()


def test_uncertain_delivery_is_not_a_resend_control(delivery):
    d = delivery
    start(d)
    d.now[0] += 4000
    d.outbox.flush("org")
    result = recovery(d).status()
    assert result["delivery"]["uncertain"] == 1 and not result["can_retry"]
    count = d.client.settle_notification_delivery.call_count
    recovery(d).retry()
    assert row(d)["outcome"] == "uncertain"
    assert d.client.settle_notification_delivery.call_count == count
    d.client.claim_notification_delivery.assert_called_once()


@pytest.mark.parametrize("kind", ["delivery", "operation"])
def test_conflicting_records_are_not_retried(outcome, kind):
    o = outcome
    if kind == "delivery":
        finish(o)
        o.ack()
        client_call = o.service.client.settle_notification_delivery
        client_call.side_effect = WisdomError("private", status=409)
        o.delivery.flush("org")
    else:
        ready(o)
        client_call = o.service.client.report_operation_outcome
        client_call.side_effect = WisdomError("private", status=409)
        o.outbox.flush("org")
    assert recovery(o).status()[kind]["conflict"] == 1
    count = client_call.call_count
    o.now[0] += 4000
    result = recovery(o).retry()
    assert result[kind]["conflict"] == 1 and not result["can_retry"]
    assert client_call.call_count == count


def test_pending_report_waits_for_receipt(outcome):
    o = outcome
    finish(o)
    assert recovery(o).status()["operation"]["waiting_for_receipt"] == 1
    recovery(o).retry()
    o.service.client.report_operation_outcome.assert_not_called()


def test_last_crashed_attempt_requires_expired_lease_before_retry(outcome):
    o = outcome
    ready(o)
    with o.service.store.transaction() as db:
        db.execute(
            "UPDATE wisdom_operation_outbox SET attempts=?,sync_token=?,sync_until=?",
            (MAX_ATTEMPTS, "worker", o.now[0] + SYNC_LEASE),
        )
    assert recovery(o).status()["operation"]["syncing"] == 1
    recovery(o).retry()
    o.service.client.report_operation_outcome.assert_not_called()
    o.now[0] += SYNC_LEASE
    assert recovery(o).status()["operation"]["retryable"] == 1
    recovery(o).retry()
    assert rows(o)[0]["state"] == "settled"


def test_concurrent_explicit_retries_do_not_steal_live_worker(outcome):
    o = outcome
    fail_report(o)
    success = o.service.client.report_operation_outcome.side_effect
    entered, release = Event(), Event()

    def slow(event, payload):
        entered.set()
        assert release.wait(5)
        return success(event, payload)

    o.service.client.report_operation_outcome.side_effect = slow
    count = o.service.client.report_operation_outcome.call_count
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(recovery(o).retry)
        assert entered.wait(5)
        assert recovery(o).retry()["operation"]["syncing"] == 1
        release.set()
        first.result()
    assert o.service.client.report_operation_outcome.call_count == count + 1


def test_recovery_is_scoped_to_authenticated_account(outcome):
    o = outcome
    fail_report(o)
    before = rows(o)
    o.service.client.identity["owner"] = "other"
    assert not recovery(o).status()["can_retry"]
    recovery(o).retry()
    assert rows(o) == before
    o.service.store.activate_installation_identity("other-installation", "other-org")
    with pytest.raises(WisdomError):
        recovery(o).status()
