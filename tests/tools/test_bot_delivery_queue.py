"""Tests for tools/bot_delivery_queue.py (spec §2 queue + §1 envelope)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from tools import bot_delivery_queue as q

DEFAULTS = dict(q._DEFAULTS)


@pytest.fixture(autouse=True)
def _pin_config(monkeypatch):
    """Pin bot_mode config to the spec defaults so tests are deterministic."""
    monkeypatch.setattr(q, "_cfg", lambda key, **kw: DEFAULTS[key])
    yield


def _did(n: int) -> str:
    return "%032x" % n


def _admit(home, n=1, *, sender="alpha", target="bravo", profile_hint=None, now_ns=None):
    return q.admit(
        home,
        sender_profile=sender,
        target_profile=target,
        target_session_id="sess-1",
        idempotency_key=f"peer-{_did(n)}",
        fingerprint=f"fp-{_did(n)}",
        delivery_id=_did(n),
        message=f"hello {n}",
        now_ns=now_ns,
    )


def _queued_path(home, n):
    return Path(home) / "runtime" / q.DELIVERY_DIR_NAME / q.QUEUE_DIR / f"{_did(n)}.json"


def _claimed_path(home, n):
    return Path(home) / "runtime" / q.DELIVERY_DIR_NAME / q.CLAIMED_DIR / f"{_did(n)}.json"


def _settled_path(home, n):
    return Path(home) / "runtime" / q.DELIVERY_DIR_NAME / q.SETTLED_DIR / f"{_did(n)}.json"


# ---------------------------------------------------------------- vocabulary
def test_no_target_busy_in_vocabulary():
    assert len(q.STATUSES) == 7
    assert "target_busy" not in q.STATUSES
    assert set(q.STATUSES) == {
        "queued",
        "running",
        "delivered",
        "failed",
        "expired",
        "cancelled",
        "ambiguous",
    }
    assert set(q.RESULTS) == {"delivered", "receipt", "failed", "refused", "unknown"}


def test_failure_reason_vocabulary_adds_only_queue_full():
    from tools import bot_failure_reasons as reasons

    assert "queue_full" in reasons.ALL_REASONS
    assert "target_busy" not in reasons.ALL_REASONS
    assert "queue_full" not in reasons.AUTO_RETRYABLE


def test_field_counts():
    assert len(q.ENVELOPE_FIELDS) == 23
    assert q.ENVELOPE_FIELDS[0] == "object"
    assert len(q.RECORD_FIELDS) == 20


# ---------------------------------------------------------------- enqueue
def test_defaults_match_spec():
    assert DEFAULTS["receipt_after_seconds"] == 15
    assert DEFAULTS["delivery_queue_ttl_seconds"] == 1800
    assert DEFAULTS["delivery_queue_max_per_profile"] == 32
    assert DEFAULTS["delivery_queue_max_per_sender"] == 8
    assert DEFAULTS["delivery_retry_base_seconds"] == 2
    assert DEFAULTS["delivery_sweep_seconds"] == 30
    assert DEFAULTS["delivery_max_turn_attempts"] == 3
    assert DEFAULTS["delivery_probe_base_seconds"] == 0.5
    assert DEFAULTS["delivery_probe_max_seconds"] == 2.0
    assert DEFAULTS["lease_probe_seconds"] == 2


def test_admit_writes_twenty_field_record(tmp_path):
    record = _admit(tmp_path, 1)
    assert list(record) == list(q.RECORD_FIELDS)
    assert record["status"] == "queued"
    assert record["attempts"] == 0
    assert record["reoffer_count"] == 0
    assert record["queue_position"] == 1
    assert record["sequence"] == 1
    assert record["claimed_at"] is None
    assert record["attempts_log"] == []
    assert record["reply"] is None and record["error"] is None
    assert record["reason"] is None
    assert record["status_detail"].startswith("target_busy: turn slot held")
    assert record["created_at"] == record["updated_at"]
    assert record["created_at"] > 10 ** 18  # ns epoch

    on_disk = json.loads(_queued_path(tmp_path, 1).read_text())
    assert on_disk == record


def test_admit_is_idempotent_on_delivery_id(tmp_path):
    first = _admit(tmp_path, 1)
    second = _admit(tmp_path, 1)
    assert second == first
    files = list(
        (Path(tmp_path) / "runtime" / q.DELIVERY_DIR_NAME / q.QUEUE_DIR).glob("*.json")
    )
    assert len(files) == 1
    assert q.queue_depth(tmp_path, target_profile="bravo") == 1


def test_admit_assigns_fifo_sequence_and_position(tmp_path):
    a = _admit(tmp_path, 1)
    b = _admit(tmp_path, 2)
    c = _admit(tmp_path, 3)
    assert [a["sequence"], b["sequence"], c["sequence"]] == [1, 2, 3]
    assert [a["queue_position"], b["queue_position"], c["queue_position"]] == [1, 2, 3]


def test_capacity_per_profile_is_32(tmp_path):
    for n in range(1, 33):
        _admit(tmp_path, n, sender=f"sender-{n}")
    assert q.queue_depth(tmp_path, target_profile="bravo") == 32
    with pytest.raises(q.QueueFullError) as excinfo:
        _admit(tmp_path, 99, sender="sender-99")
    assert excinfo.value.reason == "queue_full"
    assert "queue full (32 per profile)" in str(excinfo.value)
    assert excinfo.value.retry_after_seconds == 60.0
    # no reservation was taken: the 33rd delivery left nothing behind
    assert q.queue_depth(tmp_path, target_profile="bravo") == 32


def test_capacity_per_sender_is_8(tmp_path):
    for n in range(1, 9):
        _admit(tmp_path, n, sender="alpha")
    with pytest.raises(q.QueueFullError) as excinfo:
        _admit(tmp_path, 9, sender="alpha")
    assert excinfo.value.reason == "queue_full"
    # a different sender still fits under the per-profile cap
    _admit(tmp_path, 10, sender="charlie")
    assert q.queue_depth(tmp_path, target_profile="bravo") == 9


def test_admit_rejects_bad_keys_and_ids(tmp_path):
    with pytest.raises(ValueError):
        q.validate_idempotency_key("")
    with pytest.raises(ValueError):
        q.validate_idempotency_key("bad\nkey")
    with pytest.raises(ValueError):
        q.validate_idempotency_key("x" * 256)
    assert q.validate_idempotency_key("  peer-abc  ") == "peer-abc"
    with pytest.raises(ValueError):
        q.validate_delivery_id("NOTHEX")


# ---------------------------------------------------------------- claim
def test_claim_next_is_fifo(tmp_path):
    _admit(tmp_path, 1)
    _admit(tmp_path, 2)
    _admit(tmp_path, 3)
    claimed = [q.claim_next(tmp_path, target_profile="bravo", lease_ok=True) for _ in range(3)]
    assert [r["delivery_id"] for r in claimed] == [_did(1), _did(2), _did(3)]
    assert all(r["status"] == "running" for r in claimed)
    assert [r["attempts"] for r in claimed] == [1, 1, 1]
    assert claimed[0]["claimed_at"] is not None
    assert claimed[0]["attempts_log"][0]["status"] == "running"
    assert claimed[0]["attempts_log"][0]["ended_at"] is None
    assert _queued_path(tmp_path, 1).exists() is False
    assert _claimed_path(tmp_path, 1).exists() is True
    assert q.claim_next(tmp_path, target_profile="bravo", lease_ok=True) is None


def test_claim_contended_lease_keeps_attempts_and_reoffers(tmp_path):
    _admit(tmp_path, 1)
    assert q.claim_next(tmp_path, target_profile="bravo", lease_ok=False) is None
    record = q.read_record(tmp_path, _did(1))
    assert record["status"] == "queued"
    assert record["attempts"] == 0  # contention never consumes an attempt
    assert record["reoffer_count"] == 1
    assert record["status_detail"] == q.STATUS_DETAIL_LEASE_CONTENDED
    assert record["claimed_at"] is None
    assert _claimed_path(tmp_path, 1).exists() is False
    q.claim_next(tmp_path, target_profile="bravo", lease_ok=False)
    assert q.read_record(tmp_path, _did(1))["reoffer_count"] == 2


def test_claim_ignores_other_profiles(tmp_path):
    _admit(tmp_path, 1, target="bravo")
    _admit(tmp_path, 2, target="delta")
    claimed = q.claim_next(tmp_path, target_profile="delta", lease_ok=True)
    assert claimed["delivery_id"] == _did(2)


# ---------------------------------------------------------------- requeue
def test_requeue_leaves_attempts_unchanged(tmp_path):
    _admit(tmp_path, 1)
    claimed = q.claim_next(tmp_path, target_profile="bravo", lease_ok=True)
    assert claimed["attempts"] == 1
    requeued = q.requeue(tmp_path, _did(1))
    assert requeued["status"] == "queued"
    assert requeued["attempts"] == 1  # unchanged by requeue
    assert requeued["reoffer_count"] == 1
    assert requeued["claimed_at"] is None
    assert requeued["attempts_log"][0]["ended_at"] is not None
    assert requeued["attempts_log"][0]["reason"] == "requeued"
    assert _queued_path(tmp_path, 1).exists()
    assert not _claimed_path(tmp_path, 1).exists()
    # idempotent
    assert q.requeue(tmp_path, _did(1))["reoffer_count"] == 1


def test_requeue_missing_record(tmp_path):
    with pytest.raises(FileNotFoundError):
        q.requeue(tmp_path, _did(7))


# ---------------------------------------------------------------- settle
def test_settle_terminal_delivered(tmp_path):
    _admit(tmp_path, 1)
    q.claim_next(tmp_path, target_profile="bravo", lease_ok=True)
    settled = q.settle(tmp_path, _did(1), status="delivered", reply="pong")
    assert settled["status"] == "delivered"
    assert settled["reply"] == "pong"
    assert settled["attempts"] == 1
    assert settled["attempts_log"][0]["ended_at"] is not None
    assert _settled_path(tmp_path, 1).exists()
    assert not _claimed_path(tmp_path, 1).exists()
    assert q.queue_depth(tmp_path, target_profile="bravo") == 0
    # idempotent replay of the same terminal outcome
    assert q.settle(tmp_path, _did(1), status="delivered")["status"] == "delivered"


def test_settle_rejects_non_terminal_and_conflicting_replay(tmp_path):
    _admit(tmp_path, 1)
    q.claim_next(tmp_path, target_profile="bravo", lease_ok=True)
    with pytest.raises(ValueError):
        q.settle(tmp_path, _did(1), status="queued")
    q.settle(tmp_path, _did(1), status="delivered")
    with pytest.raises(ValueError):
        q.settle(tmp_path, _did(1), status="failed", error="boom")


def test_settle_failed_carries_error_and_reason(tmp_path):
    _admit(tmp_path, 1)
    q.claim_next(tmp_path, target_profile="bravo", lease_ok=True)
    settled = q.settle(
        tmp_path, _did(1), status="failed", error="turn failed", reason="runtime_offline"
    )
    env = q.build_envelope(settled)
    assert env["result"] == "failed"
    assert env["error"] == "turn failed"
    assert env["reason"] == "runtime_offline"
    assert env["retryable"] is True


# ---------------------------------------------------------------- expiry + sweep
def test_sweep_expires_over_age_records(tmp_path):
    old = time.time_ns() - int(2 * 3600 * 1e9)
    _admit(tmp_path, 1, now_ns=old)
    assert q.sweep_delivery_queue(tmp_path) == 1
    record = q.read_record(tmp_path, _did(1))
    assert record["status"] == "expired"
    assert record["reason"] == q.REASON_QUEUED_EXPIRED
    assert record["status_detail"] == "queued 1800s without a free turn slot; not delivered"
    assert _settled_path(tmp_path, 1).exists()
    assert q.sweep_delivery_queue(tmp_path) == 0


def test_sweep_leaves_fresh_records_alone(tmp_path):
    _admit(tmp_path, 1)
    assert q.sweep_delivery_queue(tmp_path) == 0
    assert q.read_record(tmp_path, _did(1))["status"] == "queued"


def test_sweep_recovers_orphaned_claim(tmp_path):
    record = _admit(tmp_path, 1)
    claimed_dir = Path(tmp_path) / "runtime" / q.DELIVERY_DIR_NAME / q.CLAIMED_DIR
    claimed_dir.mkdir(parents=True, exist_ok=True)
    (claimed_dir / f"{_did(1)}.json").write_text(json.dumps(record))
    assert q.sweep_delivery_queue(tmp_path) == 1
    assert _queued_path(tmp_path, 1).exists()
    assert not _claimed_path(tmp_path, 1).exists()


# ---------------------------------------------------------------- derivation
def test_queued_seconds_counts_waiting_only(tmp_path):
    day_ago = time.time_ns() - int(86400 * 1e9)
    record = _admit(tmp_path, 1, now_ns=day_ago)
    waiting = q.queued_seconds(record)
    assert 86390 < waiting < 86410
    q.claim_next(tmp_path, target_profile="bravo", lease_ok=True)
    running = q.read_record(tmp_path, _did(1))
    # the 24h wait before the turn started is still counted; no new queued time
    assert q.queued_seconds(running) == pytest.approx(86400.0, abs=1.0)

    fresh = _admit(tmp_path, 2, now_ns=time.time_ns())
    assert q.queued_seconds(fresh) < 1.0
    q.claim_next(tmp_path, target_profile="bravo", lease_ok=True)
    assert q.queued_seconds(q.read_record(tmp_path, _did(2))) < 1.0


def test_backoff_curves(tmp_path):
    class _Mid:
        def uniform(self, a, b):
            return (a + b) / 2

    mid = _Mid()
    assert [round(q.probe_delay(n, rng=mid), 3) for n in range(4)] == [0.5, 1.0, 2.0, 2.0]
    assert [round(q.drainer_delay(n, rng=mid), 3) for n in range(7)] == [
        1.5,
        3.0,
        6.0,
        12.0,
        22.5,
        22.5,
        22.5,
    ]


def test_no_send_id_or_reoffer_in_record(tmp_path):
    record = _admit(tmp_path, 1)
    assert "send_id" not in record
    assert "reoffer_count" in record
    envelope = q.build_envelope(record)
    assert "reoffer_count" not in envelope
    assert len(envelope["send_id"]) == 32
    assert q.ENVELOPE_FIELDS.count("send_id") == 1
    assert "send_id" not in q.RECORD_FIELDS


# ---------------------------------------------------------------- envelope
def test_build_envelope_field_order(tmp_path):
    record = _admit(tmp_path, 1)
    env = q.build_envelope(record, send_id=_did(5), busy=True, waited_seconds=0.2)
    assert list(env) == list(q.ENVELOPE_FIELDS)
    assert env["object"] == "hermes.peer.send_result"
    assert env["result"] == "receipt"
    assert env["status"] == "queued"
    assert env["status_detail"].startswith("target_busy")
    assert env["send_id"] == _did(5)
    assert env["delivery_id"] == _did(1)
    assert env["idempotency_key"] == f"peer-{_did(1)}"
    assert env["replayed"] is False
    assert env["profile"] == "bravo"
    assert env["session_id"] == "sess-1"
    assert env["queue_position"] == 1
    assert env["attempts"] == 0
    assert env["waited_seconds"] == 0.2
    assert env["busy"] is True
    assert env["retryable"] is False
    assert env["retry_after_seconds"] is None
    assert env["reply"] is None
    assert env["error"] is None
    assert env["reason"] is None
    assert env["detail"] == q.DETAIL_RECEIPT
    assert env["at"] > 10 ** 9


def test_build_receipt_is_busy_zero_attempts(tmp_path):
    record = _admit(tmp_path, 1)
    env = q.build_receipt(record)
    assert env["result"] == "receipt"
    assert env["busy"] is True
    assert env["attempts"] == 0
    assert env["status"] == "queued"
    assert env["detail"] == q.DETAIL_RECEIPT
    assert env["waited_seconds"] == 15


def test_build_envelope_delivered(tmp_path):
    _admit(tmp_path, 1)
    q.claim_next(tmp_path, target_profile="bravo", lease_ok=True)
    record = q.settle(tmp_path, _did(1), status="delivered", reply="pong")
    env = q.build_envelope(record)
    assert env["result"] == "delivered"
    assert env["status"] == "delivered"
    assert env["attempts"] == 1
    assert env["reply"] == "pong"
    assert env["queue_position"] is None
    assert env["detail"] is None
    assert env["busy"] is False


def test_build_envelope_expired_and_unknown(tmp_path):
    old = time.time_ns() - int(2 * 3600 * 1e9)
    _admit(tmp_path, 1, now_ns=old)
    q.sweep_delivery_queue(tmp_path)
    env = q.build_envelope(q.read_record(tmp_path, _did(1)))
    assert env["result"] == "failed"
    assert env["status"] == "expired"
    assert env["reason"] == q.REASON_QUEUED_EXPIRED
    assert env["detail"] == "queued 1800s without a free turn slot; not delivered"
    assert env["retryable"] is False


def test_retryable_is_reason_only():
    assert q.retryable({"result": "failed", "reason": "queue_full"}) is True
    assert q.retryable({"result": "failed", "reason": "runtime_offline"}) is True
    assert q.retryable({"result": "failed", "reason": "delivery_failed"}) is False
    assert q.retryable({"result": "receipt", "reason": "queue_full"}) is False
    assert q.retryable({"result": "delivered", "reason": None}) is False


def test_notification_text_strings(tmp_path):
    record = _admit(tmp_path, 1)
    assert q.build_notification_text(record) == q.DETAIL_RECEIPT
    live = dict(record, status_detail=q.STATUS_DETAIL_LIVE_OWNER)
    assert q.build_notification_text(live) == (
        "Delivery remains pending or its outcome is unknown. "
        "Do not resend; receipt is retained."
    )
    assert "do NOT resend" in q.DETAIL_RECEIPT


# --------------------------------------------------- admission-path helpers
def test_check_capacity_matches_admit_caps(tmp_path):
    """The pre-reserve capacity probe must agree with admit()'s own gate (§4.6 step 3)."""
    q.check_capacity(tmp_path, target_profile="bravo", sender_profile="alpha")  # empty: fine
    for n in range(1, q.max_per_sender() + 1):
        _admit(tmp_path, n)
    with pytest.raises(q.QueueFullError) as exc:
        q.check_capacity(tmp_path, target_profile="bravo", sender_profile="alpha")
    assert exc.value.reason == "queue_full"
    assert exc.value.retry_after_seconds == 60.0
    # A different sender is blocked by the same per-sender cap only for itself.
    q.check_capacity(tmp_path, target_profile="bravo", sender_profile="charlie")


def test_check_capacity_per_profile_cap(tmp_path):
    for n in range(1, q.max_per_profile() + 1):
        _admit(tmp_path, n, sender=f"s{n}")
    with pytest.raises(q.QueueFullError):
        q.check_capacity(tmp_path, target_profile="bravo", sender_profile="zeta")


def test_check_capacity_takes_no_reservation(tmp_path):
    for n in range(1, q.max_per_profile() + 1):
        _admit(tmp_path, n, sender=f"s{n}")
    with pytest.raises(q.QueueFullError):
        q.check_capacity(tmp_path, target_profile="bravo", sender_profile="alpha")
    assert not _queued_path(tmp_path, q.max_per_profile() + 1).exists()
    assert len(list((Path(tmp_path) / "runtime" / q.DELIVERY_DIR_NAME / q.QUEUE_DIR).glob("*.json"))) == q.max_per_profile()


def test_mark_contended_stamps_detail_without_consuming_attempts(tmp_path):
    _admit(tmp_path, 1)
    record = q.mark_contended(tmp_path, _did(1), now_ns=123)
    assert record["status"] == "queued"
    assert record["status_detail"] == "target_busy: turn slot held by another turn; delivery queued"
    assert record["attempts"] == 0
    assert record["reoffer_count"] == 0
    assert record["updated_at"] == 123
    assert q.read_record(tmp_path, _did(1))["status_detail"] == record["status_detail"]


def test_mark_contended_returns_none_for_unknown_delivery(tmp_path):
    assert q.mark_contended(tmp_path, _did(9)) is None


def test_mark_contended_leaves_a_claimed_record_alone(tmp_path):
    _admit(tmp_path, 1)
    claimed = q.claim_next(tmp_path, target_profile="bravo", lease_ok=True)
    assert q.mark_contended(tmp_path, _did(1)) is None
    stored = q.read_record(tmp_path, _did(1))
    assert stored["status"] == "running"
    assert stored["attempts"] == 1
    assert stored["status_detail"] == claimed["status_detail"]


# ------------------------------------------------- crash recovery (one replay)
def test_reoffer_ambiguous_requeues_exactly_once(tmp_path):
    _admit(tmp_path, 1)
    q.claim_next(tmp_path, target_profile="bravo", lease_ok=True)
    q.settle(tmp_path, _did(1), status="ambiguous", error="outcome unknown")
    assert _settled_path(tmp_path, 1).exists()

    record = q.reoffer_ambiguous(tmp_path, _did(1))
    assert record["status"] == "queued"
    assert record["reoffer_count"] == 1
    assert record["attempts"] == 1  # the re-run's attempt is counted at claim time
    assert record["status_detail"] == q.STATUS_DETAIL_TARGET_BUSY
    assert _queued_path(tmp_path, 1).exists()
    assert not _settled_path(tmp_path, 1).exists()

    # At most one recovery replay: the second same-key attempt gets nothing back.
    q.claim_next(tmp_path, target_profile="bravo", lease_ok=True)
    q.settle(tmp_path, _did(1), status="ambiguous")
    assert q.reoffer_ambiguous(tmp_path, _did(1)) is None


def test_reoffer_ambiguous_refuses_non_ambiguous_and_unknown(tmp_path):
    _admit(tmp_path, 1)
    q.claim_next(tmp_path, target_profile="bravo", lease_ok=True)
    q.settle(tmp_path, _did(1), status="delivered", reply="pong")
    assert q.reoffer_ambiguous(tmp_path, _did(1)) is None  # a delivered record never re-runs
    assert q.reoffer_ambiguous(tmp_path, _did(1)) is None  # unknown id
    assert q.read_record(tmp_path, _did(1))["status"] == "delivered"


def test_requeue_unstarted_rolls_back_an_uncharged_attempt(tmp_path):
    """Contention must never consume an attempt (§2.7 step 2, §4.9)."""
    _admit(tmp_path, 1)
    claimed = q.claim_next(tmp_path, target_profile="bravo", lease_ok=True)
    assert claimed["attempts"] == 1 and len(claimed["attempts_log"]) == 1

    back = q.requeue_unstarted(tmp_path, _did(1))
    assert back["status"] == "queued"
    assert back["attempts"] == 0
    assert back["attempts_log"] == []
    assert back["reoffer_count"] == 1
    assert back["claimed_at"] is None
    assert back["status_detail"] == q.STATUS_DETAIL_LEASE_CONTENDED
    assert _queued_path(tmp_path, 1).exists()
    assert not _claimed_path(tmp_path, 1).exists()

    # The rollback is not a free pass: the next claim charges its own attempt.
    again = q.claim_next(tmp_path, target_profile="bravo", lease_ok=True)
    assert again["delivery_id"] == _did(1) and again["attempts"] == 1

    # A second contention rolls that claim back too; once queued the call is a no-op.
    rolled_back = q.requeue_unstarted(tmp_path, _did(1))
    assert rolled_back["attempts"] == 0
    untouched = q.requeue_unstarted(tmp_path, _did(1))
    assert untouched["status"] == "queued" and untouched["attempts"] == 0
