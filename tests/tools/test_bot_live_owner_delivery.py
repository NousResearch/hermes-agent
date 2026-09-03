"""Reliable Bot Chat delivery through an existing live session owner."""

from __future__ import annotations

import threading
import time

from hermes_cli.active_sessions import try_acquire_active_session
from hermes_state import SessionDB
from tools import bot_live_delivery


def test_find_canonical_live_owner_uses_profile_registry(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("canonical", source="desktop")
    db.set_session_title("canonical", "Bot Chat")
    db.set_session_hidden("canonical", True)
    lease, refusal = try_acquire_active_session(
        session_id="canonical",
        surface="desktop",
        config={},
        metadata={"live_session_id": "live-target"},
        registry_home=tmp_path,
    )
    assert lease is not None and refusal is None

    assert bot_live_delivery.find_canonical_live_owner(tmp_path) == "canonical"


def test_find_canonical_live_owner_ignores_cli_owner(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("canonical", source="cli")
    db.set_session_title("canonical", "Bot Chat")
    db.set_session_hidden("canonical", True)
    lease, refusal = try_acquire_active_session(
        session_id="canonical",
        surface="cli",
        config={},
        metadata={"live_session_id": "classic-cli"},
        registry_home=tmp_path,
    )
    assert lease is not None and refusal is None

    assert bot_live_delivery.find_canonical_live_owner(tmp_path) is None


def test_find_canonical_live_owner_returns_none_when_session_is_idle(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("canonical", source="desktop")
    db.set_session_title("canonical", "Bot Chat")
    db.set_session_hidden("canonical", True)

    assert bot_live_delivery.find_canonical_live_owner(tmp_path) is None


def test_live_owner_claim_and_terminal_receipt_round_trip(tmp_path):
    result = {}

    def sender():
        result.update(
            bot_live_delivery.deliver_to_live_owner(
                tmp_path,
                "canonical",
                "Message from agent: hello",
                owner_wait_seconds=1,
                receipt_wait_seconds=1,
            )
        )

    thread = threading.Thread(target=sender)
    thread.start()

    claimed = None
    deadline = time.monotonic() + 1
    while claimed is None and time.monotonic() < deadline:
        claimed = bot_live_delivery.claim_pending_delivery(tmp_path, "canonical")
        if claimed is None:
            time.sleep(0.01)

    assert claimed is not None
    assert claimed["message"] == "Message from agent: hello"
    bot_live_delivery.complete_delivery(
        tmp_path,
        claimed["id"],
        status="settled",
        reply="received",
    )
    thread.join(timeout=2)

    assert result == {
        "status": "delivered",
        "delivery_id": claimed["id"],
        "reply": "received",
    }


def test_live_owner_busy_timeout_is_definitively_not_delivered(tmp_path):
    result = bot_live_delivery.deliver_to_live_owner(
        tmp_path,
        "canonical",
        "wait for owner",
        owner_wait_seconds=0.02,
        receipt_wait_seconds=1,
        poll_seconds=0.005,
    )

    assert result["status"] == "not_delivered"
    assert result["reason"] == "target_busy"
    assert bot_live_delivery.claim_pending_delivery(tmp_path, "canonical") is None


def test_expired_pending_claim_makes_sender_result_not_delivered(tmp_path):
    result = {}

    def sender():
        result.update(
            bot_live_delivery.deliver_to_live_owner(
                tmp_path,
                "canonical",
                "expired before owner could claim",
                owner_wait_seconds=0.05,
                receipt_wait_seconds=1,
                poll_seconds=0.001,
            )
        )

    thread = threading.Thread(target=sender)
    thread.start()
    pending_dir, _claimed_dir, _replies_dir = bot_live_delivery._paths(
        tmp_path, "canonical"
    )
    deadline = time.monotonic() + 1
    pending = next(pending_dir.glob("*.json"), None)
    while pending is None and time.monotonic() < deadline:
        time.sleep(0.001)
        pending = next(pending_dir.glob("*.json"), None)
    assert pending is not None
    payload = bot_live_delivery._read_json(pending)
    assert payload is not None
    payload["owner_deadline"] = time.time() - 1
    bot_live_delivery._atomic_json(pending, payload)

    assert bot_live_delivery.claim_pending_delivery(tmp_path, "canonical") is None
    thread.join(timeout=2)

    assert result["status"] == "not_delivered"
    assert result["reason"] == "target_busy"


def test_claimed_delivery_timeout_is_transport_ambiguous(tmp_path):
    result = {}

    def sender():
        result.update(
            bot_live_delivery.deliver_to_live_owner(
                tmp_path,
                "canonical",
                "claimed but receipt lost",
                owner_wait_seconds=1,
                receipt_wait_seconds=0.02,
                poll_seconds=0.005,
            )
        )

    thread = threading.Thread(target=sender)
    thread.start()
    deadline = time.monotonic() + 1
    claimed = None
    while claimed is None and time.monotonic() < deadline:
        claimed = bot_live_delivery.claim_pending_delivery(tmp_path, "canonical")
        if claimed is None:
            time.sleep(0.005)
    assert claimed is not None
    thread.join(timeout=2)

    assert result["status"] == "ambiguous"
    assert result["reason"] == "delivery_timeout"


def test_claim_pending_delivery_uses_created_at_fifo_not_uuid_order(tmp_path):
    pending_dir, _claimed_dir, _replies_dir = bot_live_delivery._paths(
        tmp_path, "canonical"
    )
    now = time.time()
    requests = [
        ("f" * 32, now - 2, "older"),
        ("0" * 32, now - 1, "newer"),
    ]
    for delivery_id, created_at, message in requests:
        bot_live_delivery._atomic_json(
            pending_dir / f"{delivery_id}.json",
            {
                "id": delivery_id,
                "session_id": "canonical",
                "message": message,
                "created_at": created_at,
                "owner_deadline": now + 30,
            },
        )

    first = bot_live_delivery.claim_pending_delivery(tmp_path, "canonical")
    second = bot_live_delivery.claim_pending_delivery(tmp_path, "canonical")

    assert first is not None and first["message"] == "older"
    assert second is not None and second["message"] == "newer"
