import json
import sqlite3
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from hermes_wisdom.delivery import DeliveryReceipt, slack_receipt, telegram_receipt
from hermes_wisdom.mediation_store import LEASE_SECONDS, MediationStore
from hermes_wisdom.store import WisdomStore


def receipt(**changes):
    return DeliveryReceipt(**{
        "platform": "telegram",
        "destination": "42",
        "message_id": "19",
        "acknowledgement": "provider_accepted",
        **changes,
    })


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
        actor_id="user",
        private=True,
        available=True,
        user_activity=True,
        address={"chat_id": "42"},
    )
    identity = queue.enqueue("org", "feed:event", {"kind": "notice"})
    job = queue.claim("org", "session")[0]
    assert queue.save_advice("org", identity, job["lease_token"], {"title": "Advice"})
    assert queue.begin_delivery("org", identity, job["lease_token"])
    return queue, job, now


@pytest.mark.parametrize("response_kind", ["raw", "envelope", "sdk"])
def test_telegram_accepts_raw_envelope_and_sdk_message_without_persisting_body(response_kind):
    data = {
        "message_id": 19,
        "date": 0,
        "chat": {"id": 42, "type": "private"},
        "text": "PRIVATE BODY",
    }
    expected = receipt()
    response = data
    if response_kind == "envelope":
        response = {"ok": True, "result": data}
    elif response_kind == "sdk":
        telegram = pytest.importorskip("telegram", reason="optional messaging SDK not installed")
        response = telegram.Message.de_json(data, None)
    assert telegram_receipt(response, chat_id="42") == expected
    assert "PRIVATE BODY" not in expected.model_dump_json()


@pytest.mark.parametrize(
    "response",
    [
        None,
        True,
        {},
        {"ok": False, "result": {"message_id": 19, "chat": {"id": 42}}},
        {"message_id": True, "chat": {"id": 42}},
        {"message_id": 0, "chat": {"id": 42}},
        {"message_id": 19, "chat": {"id": 43}},
        {"message_id": 19, "chat": {"id": 42}, "message_thread_id": 3},
    ],
)
def test_telegram_missing_or_wrong_acknowledgement_is_not_success(response):
    with pytest.raises(ValueError):
        telegram_receipt(response, chat_id="42")


def test_slack_receipt_uses_acknowledged_message_not_content():
    data = {
        "ok": True,
        "channel": "D1",
        "ts": "123.456",
        "message": {"thread_ts": "123", "text": "PRIVATE"},
    }
    result = slack_receipt(
        SimpleNamespace(data=data), channel_id="D1", thread_id="123", scope_id="T1"
    )
    assert result.message_id == "123.456" and result.scope_id == "T1"
    assert "PRIVATE" not in result.model_dump_json()


@pytest.mark.parametrize(
    "data",
    [
        None,
        {},
        {"ok": False},
        {"ok": True, "channel": "D1"},
        {"ok": True, "channel": "D2", "ts": "123.456"},
        {
            "ok": True,
            "channel": "D1",
            "ts": "123.456",
            "message": {"thread_ts": "other"},
        },
    ],
)
def test_slack_missing_or_wrong_acknowledgement_is_not_success(data):
    with pytest.raises(ValueError):
        slack_receipt(data, channel_id="D1", thread_id="123")


@pytest.mark.parametrize(
    "changes",
    [
        {"acknowledgement": "read"},
        {"platform": "local"},
        {"message_id": ""},
        {"message_id": "x" * 129},
        {"destination": "bad\nchannel"},
        {"text": "untrusted"},
    ],
)
def test_receipts_are_bounded_and_do_not_claim_read_status(changes):
    with pytest.raises(ValidationError):
        receipt(**changes)


def test_receipt_delivery_and_introduction_commit_together_and_survive_restart(
    delivery,
):
    queue, job, _ = delivery
    assert queue.complete_delivery(
        "org", job["id"], job["lease_token"], receipt=receipt(), introduced=True
    )
    assert not queue.complete_delivery(
        "org", job["id"], job["lease_token"], receipt=receipt(), introduced=True
    )
    reopened = MediationStore(WisdomStore(queue.store.root))
    assert reopened.introduced("org")
    assert reopened.assessments("org")[0]["state"] == "delivered"
    with reopened.store.transaction() as db:
        rows = db.execute("SELECT * FROM wisdom_delivery_receipt").fetchall()
    assert len(rows) == 1 and rows[0]["owner_session"] == "session"
    assert rows[0]["organization_id"] == "org"
    assert json.loads(rows[0]["receipt_json"]) == receipt().model_dump()


def test_database_failure_rolls_back_delivery_and_introduction(delivery):
    queue, job, _ = delivery
    with queue.store.transaction() as db:
        db.execute(
            "CREATE TRIGGER reject_receipt BEFORE INSERT ON wisdom_delivery_receipt BEGIN SELECT RAISE(ABORT,'injected'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        queue.complete_delivery(
            "org", job["id"], job["lease_token"], receipt=receipt(), introduced=True
        )
    assert queue.assessments("org")[0]["state"] == "delivering"
    assert not queue.introduced("org")
    with queue.store.transaction() as db:
        assert (
            db.execute("SELECT COUNT(*) FROM wisdom_delivery_receipt").fetchone()[0]
            == 0
        )
        db.execute("DROP TRIGGER reject_receipt")
    assert queue.complete_delivery(
        "org", job["id"], job["lease_token"], receipt=receipt(), introduced=True
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"destination": "other"},
        {"thread_id": "other"},
        {"scope_id": "other"},
        {"platform": "slack"},
    ],
)
def test_receipt_must_match_owning_session(delivery, changes):
    queue, job, _ = delivery
    with pytest.raises(ValueError):
        queue.complete_delivery(
            "org",
            job["id"],
            job["lease_token"],
            receipt=receipt(**changes),
            introduced=True,
        )
    assert not queue.introduced("org")
    assert queue.assessments("org")[0]["state"] == "delivering"


def test_missing_receipt_cannot_complete_delivery(delivery):
    queue, job, _ = delivery
    with pytest.raises(ValueError):
        queue.complete_delivery("org", job["id"], job["lease_token"], receipt=None)
    assert queue.assessments("org")[0]["state"] == "delivering"


def test_expired_or_uncertain_delivery_cannot_commit_late_receipt(delivery):
    queue, job, now = delivery
    now[0] += LEASE_SECONDS + 1
    assert not queue.complete_delivery(
        "org", job["id"], job["lease_token"], receipt=receipt()
    )
    assert not queue.uncertain_delivery("org", job["id"], "wrong-token")
    assert queue.uncertain_delivery("org", job["id"], job["lease_token"])
    assert queue.assessments("org")[0]["state"] == "delivery_uncertain"
    assert not queue.complete_delivery(
        "org", job["id"], job["lease_token"], receipt=receipt()
    )
    assert not queue.introduced("org")


def test_existing_database_gains_receipt_table_without_changing_assessments(delivery):
    queue, _, _ = delivery
    before = queue.assessments("org")
    with queue.store.transaction() as db:
        db.execute("DROP TABLE wisdom_delivery_receipt")
    reopened = MediationStore(WisdomStore(queue.store.root))
    assert reopened.assessments("org") == before
    with reopened.store.transaction() as db:
        assert (
            db.execute("SELECT COUNT(*) FROM wisdom_delivery_receipt").fetchone()[0]
            == 0
        )
