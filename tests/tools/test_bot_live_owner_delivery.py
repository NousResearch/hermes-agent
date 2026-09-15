"""Durable mailbox invariants, using real disk and exec boundaries."""
import json
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("terminal_status", ["settled", "failed", "cancelled"])
def test_delivery_is_idempotent_fenced_and_permanent(tmp_path, terminal_status):
    from tools import bot_live_delivery as mailbox

    owner = dict(profile_home=str(tmp_path.resolve()), session_id="chat",
                 lease_id="lease", live_session_id="live")
    delivery_id = "a" * 32
    queued = mailbox.deliver_to_live_owner(tmp_path, owner, "hello", delivery_id=delivery_id)
    assert queued["status"] == "queued"
    assert mailbox.deliver_to_live_owner(tmp_path, owner, "hello", delivery_id=delivery_id) == queued
    with pytest.raises(ValueError):
        mailbox.deliver_to_live_owner(tmp_path, owner, "different", delivery_id=delivery_id)
    assert mailbox.claim_pending_delivery(tmp_path, dict(owner, lease_id="other")) is None
    assert mailbox.claim_pending_delivery(tmp_path, dict(owner, live_session_id="other")) is None
    script = (
        "import json,sys; from tools.bot_live_delivery import claim_pending_delivery; "
        "print(json.dumps(claim_pending_delivery(sys.argv[1],json.loads(sys.argv[2]))))"
    )
    children = [subprocess.Popen([sys.executable, "-c", script, str(tmp_path), json.dumps(owner)],
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True) for _ in range(2)]
    results = []
    for child in children:
        out, err = child.communicate(timeout=30)
        assert child.returncode == 0, err
        results.append(json.loads(out))
    claims = [r for r in results if r is not None]
    assert len(claims) == 1 and claims[0]["message"] == "hello"
    assert mailbox.read_delivery_result(tmp_path, delivery_id)["status"] == "claimed"
    assert mailbox.claim_pending_delivery(tmp_path, owner) is None
    receipt = mailbox.complete_delivery(tmp_path, delivery_id, status=terminal_status, reply="answer")
    assert mailbox.read_delivery_result(tmp_path, delivery_id) == receipt
    assert mailbox.complete_delivery(tmp_path, delivery_id, status=terminal_status, reply="answer") == receipt
    with pytest.raises(ValueError):
        mailbox.complete_delivery(tmp_path, delivery_id, status=terminal_status, reply="rewrite")
    assert mailbox.deliver_to_live_owner(tmp_path, owner, "hello", delivery_id=delivery_id) == receipt
    assert mailbox.claim_pending_delivery(tmp_path, owner) is None
    if os.name != "nt":
        for path in (tmp_path / "runtime" / mailbox.DELIVERY_DIR_NAME).iterdir():
            assert path.stat().st_mode & 0o077 == 0


def test_corrupt_receipt_does_not_block_other_pending_delivery(tmp_path):
    from tools import bot_live_delivery as mailbox

    owner = dict(profile_home=str(tmp_path.resolve()), session_id="chat",
                 lease_id="lease", live_session_id="live")
    queued = mailbox.deliver_to_live_owner(tmp_path, owner, "healthy")
    root = tmp_path / "runtime" / mailbox.DELIVERY_DIR_NAME
    damaged = {
        root / ("d" * 32 + ".json"): "{truncated",
        root / ("e" * 32 + ".json"): "[]",
        root / ("f" * 32 + ".json"): "{}",
        root / ("a" * 32 + ".json"): json.dumps(dict(
            delivery_id="a" * 32, id="a" * 32, status="queued", created_at=1,
            sequence=1, owner=None, message="damaged-owner")),
        root / ("b" * 32 + ".json"): json.dumps(dict(
            delivery_id="b" * 32, id="b" * 32, status="queued", created_at=2,
            sequence="old", owner=owner, message="damaged-sequence", **owner)),
        root / ("c" * 32 + ".json"): json.dumps(dict(
            delivery_id="../wrong", id="../wrong", status="queued", created_at=3,
            sequence=3, owner=owner, message="damaged-id", **owner)),
        root / ("1" * 32 + ".json"): json.dumps(dict(
            delivery_id="1" * 32, id="1" * 32, status=[], created_at=4,
            sequence=4, owner=owner, message="damaged-array-status", **owner)),
        root / ("2" * 32 + ".json"): json.dumps(dict(
            delivery_id="2" * 32, id="2" * 32, status={}, created_at=5,
            sequence=5, owner=owner, message="damaged-object-status", **owner)),
    }
    for path, contents in damaged.items():
        path.write_text(contents, encoding="utf-8")

    admitted = mailbox.deliver_to_live_owner(tmp_path, owner, "also healthy")
    claimed = mailbox.claim_pending_delivery(tmp_path, owner)

    assert admitted["sequence"] == queued["sequence"] + 1
    assert claimed is not None
    assert claimed["delivery_id"] == queued["delivery_id"]
    assert {path: path.read_text(encoding="utf-8") for path in damaged} == damaged


def test_fifo_survives_clock_rollback(tmp_path, monkeypatch):
    from tools import bot_live_delivery as mailbox

    owner = dict(profile_home=str(tmp_path.resolve()), session_id="chat",
                 lease_id="lease", live_session_id="live")
    for timestamp, message in ((100, "first"), (90, "second")):
        monkeypatch.setattr(mailbox.time, "time_ns", lambda: timestamp)
        mailbox.deliver_to_live_owner(tmp_path, owner, message)
    assert mailbox.claim_pending_delivery(tmp_path, owner)["message"] == "first"
    assert mailbox.claim_pending_delivery(tmp_path, owner)["message"] == "second"


@pytest.mark.parametrize("capable", [True, False])
def test_only_canonical_capable_owner_receives_across_compression(tmp_path, capable):
    from hermes_state import SessionDB
    from hermes_cli.active_sessions import try_acquire_active_session, transfer_active_session
    from tools import bot_live_delivery as mailbox

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="chat", source="cli")
    db.set_session_title("chat", "Bot Chat")
    meta = dict(live_session_id="live", bot_live_delivery_consumer=capable)
    lease, refusal = try_acquire_active_session(session_id="chat", surface="desktop", config={},
                                               registry_home=tmp_path, metadata=meta)
    assert refusal is None
    try:
        owner = mailbox.find_canonical_live_owner(tmp_path)
        if not capable:
            assert owner is None
            return
        assert owner["lease_id"] == lease.lease_id
        queued = mailbox.deliver_to_live_owner(tmp_path, owner, "before compression")
        db.end_session("chat", "compression")
        db.create_session(session_id="tip", source="cli", parent_session_id="chat")
        assert transfer_active_session(lease, session_id="tip", metadata=meta)
        current = mailbox.find_canonical_live_owner(tmp_path)
        assert current["session_id"] == "tip"
        claim = mailbox.claim_pending_delivery(tmp_path, current)
        assert claim["delivery_id"] == queued["delivery_id"]
        assert claim["session_id"] == "chat"
        assert mailbox.claim_pending_delivery(tmp_path, current) is None
    finally:
        lease.release()
        db.close()


def test_delivery_keeps_the_sender_and_refuses_a_different_one_under_the_same_id(tmp_path):
    from tools import bot_live_delivery as mailbox

    owner = dict(profile_home=str(tmp_path.resolve()), session_id="chat", lease_id="lease", live_session_id="live")
    author = {"id": "bot:coder", "name": "coder", "is_bot": True}
    queued = mailbox.deliver_to_live_owner(tmp_path, owner, "hello", delivery_id="b" * 32, author=author)
    assert queued["author"] == author
    assert mailbox.deliver_to_live_owner(tmp_path, owner, "hello", delivery_id="b" * 32, author=author) == queued
    with pytest.raises(ValueError):
        mailbox.deliver_to_live_owner(tmp_path, owner, "hello", delivery_id="b" * 32, author={**author, "id": "bot:other"})
    assert "author" not in mailbox.deliver_to_live_owner(tmp_path, owner, "no sender", delivery_id="c" * 32)
