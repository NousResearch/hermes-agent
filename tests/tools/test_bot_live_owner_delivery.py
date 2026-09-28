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
    queued = mailbox.deliver_to_live_owner(tmp_path, owner, "héllo 世界", delivery_id=delivery_id)
    assert queued["status"] == "queued"
    path = tmp_path / "runtime" / mailbox.DELIVERY_DIR_NAME / f"{delivery_id}.json"
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    path.write_bytes(b"\xef\xbb\xbf" + raw)
    assert mailbox.deliver_to_live_owner(tmp_path, owner, "héllo 世界", delivery_id=delivery_id) == queued
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
    assert len(claims) == 1 and claims[0]["message"] == "héllo 世界"
    assert mailbox.read_delivery_result(tmp_path, delivery_id)["status"] == "claimed"
    assert mailbox.claim_pending_delivery(tmp_path, owner) is None
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    receipt = mailbox.complete_delivery(tmp_path, delivery_id, status=terminal_status, reply="réponse 世界")
    assert not path.read_bytes().startswith(b"\xef\xbb\xbf")
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    assert mailbox.read_delivery_result(tmp_path, delivery_id) == receipt
    assert mailbox.complete_delivery(tmp_path, delivery_id, status=terminal_status, reply="réponse 世界") == receipt
    with pytest.raises(ValueError):
        mailbox.complete_delivery(tmp_path, delivery_id, status=terminal_status, reply="rewrite")
    assert mailbox.deliver_to_live_owner(tmp_path, owner, "héllo 世界", delivery_id=delivery_id) == receipt
    assert mailbox.claim_pending_delivery(tmp_path, owner) is None
    if os.name != "nt":
        for path in (tmp_path / "runtime" / mailbox.DELIVERY_DIR_NAME).iterdir():
            assert path.stat().st_mode & 0o077 == 0

@pytest.mark.parametrize("intent_state", ["new", "existing", "raced"])
def test_live_dm_bom_readers_preserve_pinned_intent(tmp_path, monkeypatch, intent_state):
    from pathlib import Path

    from hermes_cli.active_sessions import try_acquire_active_session
    from hermes_state import SessionDB
    from tools import bot_live_delivery as mailbox, bot_mode_dm

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="chat", source="cli")
    db.set_session_title("chat", "Bot Chat")
    lease, refusal = try_acquire_active_session(
        session_id="chat", surface="desktop", config={}, registry_home=tmp_path,
        metadata=dict(live_session_id="live", bot_live_delivery_consumer=True))
    assert refusal is None and lease is not None
    try:
        owner = mailbox.find_canonical_live_owner(tmp_path)
        assert owner is not None
        payload = tmp_path / "message.txt"
        payload.write_bytes("héllo 世界".encode("utf-8-sig"))
        intent_path = Path(str(payload) + ".live.json")
        author = {"id": "bot:coder", "name": "Renée", "is_bot": True}
        intent = dict(owner=owner, message="pinned 世界", delivery_id="d" * 32, author=author)
        encoded = json.dumps(intent, ensure_ascii=False).encode("utf-8-sig")
        if intent_state == "existing":
            intent_path.write_bytes(encoded)
        elif intent_state == "raced":
            real_open = os.open

            def competing_intent(path, flags, *args, **kwargs):
                if Path(path) == intent_path:
                    intent_path.write_bytes(encoded)
                return real_open(path, flags, *args, **kwargs)

            monkeypatch.setattr(os, "open", competing_intent)
        record = bot_mode_dm._admit_live_dm(tmp_path, str(payload), author)
        assert record is not None
        assert record["message"] == ("héllo 世界" if intent_state == "new" else "pinned 世界")
        assert record["owner"] == owner and record["author"] == author
        if intent_state == "new":
            assert not intent_path.read_bytes().startswith(b"\xef\xbb\xbf")
            intent_path.write_bytes(b"\xef\xbb\xbf" + intent_path.read_bytes())
        payload.write_text("must not replace pinned payload", encoding="utf-8")
        assert bot_mode_dm._admit_live_dm(None, str(payload)) == record
        claim = mailbox.claim_pending_delivery(tmp_path, owner)
        assert claim is not None and claim["delivery_id"] == record["delivery_id"]
        assert mailbox.claim_pending_delivery(tmp_path, owner) is None
        intent_path.write_bytes(b"\xef\xbb\xbf{broken")
        assert bot_mode_dm._run_delivery([], str(payload), stdin_file=False, profile_home=tmp_path) == 1
        assert payload.exists()  # Ambiguous admission must retain its evidence, not retry a transport.
    finally:
        lease.release()
        db.close()

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
    queued = mailbox.deliver_to_live_owner(tmp_path, owner, "héllo 世界", delivery_id="b" * 32, author=author)
    assert queued["author"] == author
    assert mailbox.deliver_to_live_owner(tmp_path, owner, "héllo 世界", delivery_id="b" * 32, author=author) == queued
    with pytest.raises(ValueError):
        mailbox.deliver_to_live_owner(tmp_path, owner, "héllo 世界", delivery_id="b" * 32, author={**author, "id": "bot:other"})
    assert "author" not in mailbox.deliver_to_live_owner(tmp_path, owner, "no sender", delivery_id="c" * 32)

@pytest.mark.platforms("posix")
@pytest.mark.skipif(getattr(os, "geteuid", lambda: 1)() == 0, reason='needs POSIX file permissions for an unreadable ticket')
def test_unreadable_ticket_does_not_wedge_bulk_scans(tmp_path, caplog):
    import logging

    from tools import bot_live_delivery as mailbox

    owner = dict(profile_home=str(tmp_path.resolve()), session_id="chat",
                 lease_id="lease", live_session_id="live")
    queued = mailbox.deliver_to_live_owner(tmp_path, owner, "readable", delivery_id="d" * 32)
    root = tmp_path / "runtime" / mailbox.DELIVERY_DIR_NAME
    # A real admission that later turns unreadable: its sequence must survive the skip.
    hidden = mailbox.deliver_to_live_owner(tmp_path, owner, "hidden", delivery_id="e" * 32)
    (root / f"{'e' * 32}.json").chmod(0)
    corrupt = root / f"{'1' * 32}.json"
    corrupt.write_text("{not json", encoding="utf-8")
    (root / f"{'2' * 32}.json").write_bytes(b"\xff\xfe\x00garbage")  # invalid UTF-8, not just bad JSON
    with caplog.at_level(logging.WARNING, logger="tools.bot_live_delivery"):
        # Sender side: admission of a fresh id must survive the sequence sweep.
        admitted = mailbox.deliver_to_live_owner(tmp_path, owner, "second", delivery_id="f" * 32)
        # Receiver side: every readable queued ticket must still be claimed, in order.
        assert mailbox.claim_pending_delivery(tmp_path, owner)["delivery_id"] == queued["delivery_id"]
        assert mailbox.claim_pending_delivery(tmp_path, owner)["delivery_id"] == admitted["delivery_id"]
        for _ in range(10):  # the idle poller rescans twice a second
            assert mailbox.claim_pending_delivery(tmp_path, owner) is None
    assert admitted["status"] == "queued"
    assert admitted["sequence"] > hidden["sequence"] > queued["sequence"]
    denied = [record for record in caplog.records
              if record.message.startswith(f"bot_live_delivery: skipping unreadable ticket {'e' * 32}.json")
              and "Permission denied" in record.message]
    assert len(denied) == 1, "one persistent bad ticket must warn once per process, not per scan"

@pytest.mark.platforms("posix")
@pytest.mark.skipif(getattr(os, "geteuid", lambda: 1)() == 0, reason='needs POSIX file permissions for an unreadable ticket')
def test_unreadable_ticket_keeps_exact_id_reads_fail_closed(tmp_path):
    from tools import bot_live_delivery as mailbox

    owner = dict(profile_home=str(tmp_path.resolve()), session_id="chat",
                 lease_id="lease", live_session_id="live")
    unreadable = tmp_path / "runtime" / mailbox.DELIVERY_DIR_NAME / f"{'e' * 32}.json"
    unreadable.parent.mkdir(parents=True, exist_ok=True)
    unreadable.write_text('{"status": "queued"}', encoding="utf-8")
    unreadable.chmod(0)
    # Uninspectable is not absent: an exact-id retry must fail closed instead of
    # minting a fresh receipt that overwrites the possibly-live one (#109820).
    with pytest.raises(PermissionError):
        mailbox.deliver_to_live_owner(tmp_path, owner, "same id", delivery_id="e" * 32)
    with pytest.raises(PermissionError):
        mailbox.read_delivery_result(tmp_path, "e" * 32)

def test_non_dict_ticket_is_skipped_by_scans_and_fails_exact_id_reads_closed(tmp_path, caplog):
    import logging

    from tools import bot_live_delivery as mailbox

    owner = dict(profile_home=str(tmp_path.resolve()), session_id="chat",
                 lease_id="lease", live_session_id="live")
    queued = mailbox.deliver_to_live_owner(tmp_path, owner, "readable", delivery_id="d" * 32)
    bad = tmp_path / "runtime" / mailbox.DELIVERY_DIR_NAME / f"{'e' * 32}.json"
    bad.write_text('"oops"', encoding="utf-8")  # parses, but is not a record
    with caplog.at_level(logging.WARNING, logger="tools.bot_live_delivery"):
        admitted = mailbox.deliver_to_live_owner(tmp_path, owner, "second", delivery_id="f" * 32)
        assert mailbox.claim_pending_delivery(tmp_path, owner)["delivery_id"] == queued["delivery_id"]
        assert mailbox.claim_pending_delivery(tmp_path, owner)["delivery_id"] == admitted["delivery_id"]
        assert mailbox.claim_pending_delivery(tmp_path, owner) is None
    assert sum(r.message.startswith(f"bot_live_delivery: skipping unreadable ticket {'e' * 32}.json")
               for r in caplog.records) == 1
    # Malformed is not absent: exact-id reads fail closed rather than overwrite the receipt.
    with pytest.raises(ValueError):
        mailbox.deliver_to_live_owner(tmp_path, owner, "same id", delivery_id="e" * 32)
    with pytest.raises(ValueError):
        mailbox.read_delivery_result(tmp_path, "e" * 32)
    assert bad.read_text(encoding="utf-8") == '"oops"'

def test_schema_damaged_ticket_does_not_wedge_bulk_scans(tmp_path, caplog):
    """Valid JSON that lost a field must degrade like corrupt JSON: skipped, warned once, never raised."""
    import logging

    from tools import bot_live_delivery as mailbox

    owner = dict(profile_home=str(tmp_path.resolve()), session_id="chat",
                 lease_id="lease", live_session_id="live")
    queued = mailbox.deliver_to_live_owner(tmp_path, owner, "healthy", delivery_id="d" * 32)
    root = tmp_path / "runtime" / mailbox.DELIVERY_DIR_NAME
    damaged = {
        root / f"{'a' * 32}.json": "{}",
        root / f"{'b' * 32}.json": json.dumps(dict(
            delivery_id="b" * 32, id="b" * 32, status="queued", created_at=1,
            sequence=1, owner=None, message="owner lost")),
        root / f"{'c' * 32}.json": json.dumps(dict(
            delivery_id="c" * 32, id="c" * 32, status="queued", created_at=2,
            sequence="old", owner=owner, message="sequence lost", **owner)),
        root / f"{'e' * 32}.json": json.dumps(dict(
            delivery_id="../wrong", id="../wrong", status="queued", created_at=3,
            sequence=3, owner=owner, message="id lost", **owner)),
        root / f"{'1' * 32}.json": json.dumps(dict(
            delivery_id="1" * 32, id="1" * 32, status=[], created_at=4,
            sequence=4, owner=owner, message="status lost", **owner)),
    }
    for path, contents in damaged.items():
        path.write_text(contents, encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="tools.bot_live_delivery"):
        admitted = mailbox.deliver_to_live_owner(tmp_path, owner, "also healthy", delivery_id="f" * 32)
        assert mailbox.claim_pending_delivery(tmp_path, owner)["delivery_id"] == queued["delivery_id"]
        assert mailbox.claim_pending_delivery(tmp_path, owner)["delivery_id"] == admitted["delivery_id"]
        for _ in range(3):
            assert mailbox.claim_pending_delivery(tmp_path, owner) is None
    assert admitted["sequence"] == queued["sequence"] + 1
    assert {path: path.read_text(encoding="utf-8") for path in damaged} == damaged
    skipped = [r.message for r in caplog.records if r.message.startswith("bot_live_delivery: skipping unreadable ticket")]
    assert len(skipped) == len(damaged), "each damaged ticket warns once per process, not per scan"
