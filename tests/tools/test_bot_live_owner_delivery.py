"""Durable mailbox invariants, using real disk and exec boundaries."""
import json
import logging
import os
import subprocess
import sys
import time

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
        assert owner is not None
        assert owner["lease_id"] == lease.lease_id
        queued = mailbox.deliver_to_live_owner(tmp_path, owner, "before compression")
        db.end_session("chat", "compression")
        db.create_session(session_id="tip", source="cli", parent_session_id="chat")
        assert transfer_active_session(lease, session_id="tip", metadata=meta)
        current = mailbox.find_canonical_live_owner(tmp_path)
        assert current is not None
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


@pytest.mark.skipif(os.name == "nt" or getattr(os, "geteuid", lambda: 1)() == 0,
                    reason="needs POSIX file permissions for an unreadable ticket")
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


@pytest.mark.skipif(os.name == "nt" or getattr(os, "geteuid", lambda: 1)() == 0,
                    reason="needs POSIX file permissions for an unreadable ticket")
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


def test_non_dict_ticket_is_quarantined_by_scans_and_fails_exact_id_reads_closed(tmp_path, caplog):
    import logging

    from tools import bot_live_delivery as mailbox

    owner = dict(profile_home=str(tmp_path.resolve()), session_id="chat",
                 lease_id="lease", live_session_id="live")
    queued = mailbox.deliver_to_live_owner(tmp_path, owner, "readable", delivery_id="d" * 32)
    bad = tmp_path / "runtime" / mailbox.DELIVERY_DIR_NAME / f"{'e' * 32}.json"
    bad.write_text('"oops"', encoding="utf-8")  # parses, but is not a record
    # Malformed is not absent: while the ticket is still in place an exact-id read fails
    # closed rather than minting a fresh receipt over a possibly-live one.
    with pytest.raises(ValueError):
        mailbox.deliver_to_live_owner(tmp_path, owner, "same id", delivery_id="e" * 32)
    with pytest.raises(ValueError):
        mailbox.read_delivery_result(tmp_path, "e" * 32)
    with caplog.at_level(logging.WARNING, logger="tools.bot_live_delivery"):
        admitted = mailbox.deliver_to_live_owner(tmp_path, owner, "second", delivery_id="f" * 32)
        assert mailbox.claim_pending_delivery(tmp_path, owner)["delivery_id"] == queued["delivery_id"]
        assert mailbox.claim_pending_delivery(tmp_path, owner)["delivery_id"] == admitted["delivery_id"]
        assert mailbox.claim_pending_delivery(tmp_path, owner) is None
    assert sum(r.message.startswith(f"Quarantined unreadable delivery record {'e' * 32}.json")
               for r in caplog.records) == 1
    # Provably-dead content is preserved, never deleted, and never re-read on the next pass.
    quarantine = bad.parent / mailbox.QUARANTINE_DIR_NAME
    assert (quarantine / bad.name).read_text(encoding="utf-8") == '"oops"'
    assert not bad.exists()


def _live_owner(profile_home):
    """A real registered Bot Chat consumer — the only thing that counts as a live owner."""
    from hermes_cli.active_sessions import try_acquire_active_session
    from hermes_state import SessionDB

    db = SessionDB(db_path=profile_home / "state.db")
    db.create_session(session_id="chat", source="cli")
    db.set_session_title("chat", "Bot Chat")
    lease, refusal = try_acquire_active_session(
        session_id="chat", surface="desktop", config={}, registry_home=profile_home,
        metadata=dict(live_session_id="live", bot_live_delivery_consumer=True))
    assert refusal is None
    assert lease is not None
    return db, lease


def test_stopped_poll_loop_is_not_a_live_owner(tmp_path):
    from tools import bot_live_delivery as mailbox

    db, lease = _live_owner(tmp_path)
    try:
        owner = mailbox.find_canonical_live_owner(tmp_path)
        assert owner is not None
        assert owner["lease_id"] == lease.lease_id
        # The process is still alive; its poll loop is not. Past the TTL an entry that
        # only proves "the process exists" must stop being a destination at all.
        stale_at = time.time() + mailbox.LIVE_CONSUMER_TTL_SECONDS + 1
        assert mailbox.find_canonical_live_owner(tmp_path, now=stale_at) is None
        assert mailbox.find_canonical_live_owner(tmp_path, ttl_seconds=0.0) is None
        assert mailbox._owner_is_live_consumer(tmp_path, owner, now=stale_at) is False
    finally:
        lease.release()
        db.close()


def test_corrupt_ticket_is_quarantined_once_and_preserved(tmp_path, caplog):
    import logging

    from tools import bot_live_delivery as mailbox

    owner = dict(profile_home=str(tmp_path.resolve()), session_id="chat",
                 lease_id="lease", live_session_id="live")
    readable = mailbox.deliver_to_live_owner(tmp_path, owner, "readable", delivery_id="a" * 32)
    root = tmp_path / "runtime" / mailbox.DELIVERY_DIR_NAME
    (root / f"{'b' * 32}.json").write_text("{not json", encoding="utf-8")
    (root / f"{'c' * 32}.json").write_bytes(b"\xff\xfe\x00garbage")  # invalid UTF-8, not just bad JSON
    (root / f"{'d' * 32}.json").write_text("[1, 2, 3]", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="tools.bot_live_delivery"):
        claim = mailbox.claim_pending_delivery(tmp_path, owner)
        quarantined = [record for record in caplog.records
                       if record.message.startswith("Quarantined unreadable delivery record")]
        for _ in range(10):  # the idle poller rescans twice a second
            assert mailbox.claim_pending_delivery(tmp_path, owner) is None
    assert claim["delivery_id"] == readable["delivery_id"]
    quarantine = root / mailbox.QUARANTINE_DIR_NAME
    assert sorted(path.name for path in quarantine.iterdir()) == [
        f"{'b' * 32}.json", f"{'c' * 32}.json", f"{'d' * 32}.json"]
    assert (quarantine / f"{'b' * 32}.json").read_text(encoding="utf-8") == "{not json"
    assert len(quarantined) == 3, "dead content is quarantined once, not re-read on every pass"
    # The sequence high-water mark outlives the tickets it was built from.
    later = mailbox.deliver_to_live_owner(tmp_path, owner, "later", delivery_id="f" * 32)
    assert later["sequence"] > readable["sequence"]


@pytest.mark.skipif(os.name == "nt" or getattr(os, "geteuid", lambda: 1)() == 0,
                    reason="needs POSIX file permissions for an unreadable ticket")
def test_unreadable_ticket_does_not_wedge_a_live_claimant(tmp_path):
    from tools import bot_live_delivery as mailbox

    db, lease = _live_owner(tmp_path)
    try:
        owner = mailbox.find_canonical_live_owner(tmp_path)
        assert owner is not None
        queued = mailbox.deliver_to_live_owner(tmp_path, owner, "readable", delivery_id="a" * 32)
        root = tmp_path / "runtime" / mailbox.DELIVERY_DIR_NAME
        denied = root / f"{'b' * 32}.json"
        denied.write_text('{"status": "queued"}', encoding="utf-8")
        denied.chmod(0)
        # A live consumer is the one caller whose sweep reaches the reclaim lanes: an
        # uninspectable ticket must be skipped there too, not subscripted.
        assert mailbox.claim_pending_delivery(tmp_path, owner)["delivery_id"] == queued["delivery_id"]
        for _ in range(3):
            assert mailbox.claim_pending_delivery(tmp_path, owner) is None
    finally:
        lease.release()
        db.close()


def test_foreign_json_in_the_spool_does_not_wedge_a_live_sweep(tmp_path):
    from tools import bot_live_delivery as mailbox

    db, lease = _live_owner(tmp_path)
    try:
        owner = mailbox.find_canonical_live_owner(tmp_path)
        assert owner is not None
        queued = mailbox.deliver_to_live_owner(tmp_path, owner, "readable", delivery_id="a" * 32)
        root = tmp_path / "runtime" / mailbox.DELIVERY_DIR_NAME
        foreign = root / "hand-written.json"
        foreign.write_text('{"note": "not a delivery record"}', encoding="utf-8")
        assert mailbox.claim_pending_delivery(tmp_path, owner)["delivery_id"] == queued["delivery_id"]
        assert mailbox.claim_pending_delivery(tmp_path, owner) is None
        assert foreign.read_text(encoding="utf-8") == '{"note": "not a delivery record"}'
        assert not (root / mailbox.QUARANTINE_DIR_NAME).exists()
        later = mailbox.deliver_to_live_owner(tmp_path, owner, "later", delivery_id="b" * 32)
        assert later["sequence"] > queued["sequence"]
    finally:
        lease.release()
        db.close()


def test_reclaim_settles_a_body_the_store_already_has(tmp_path):
    from tools import bot_live_delivery as mailbox

    db, lease = _live_owner(tmp_path)
    try:
        current = mailbox.find_canonical_live_owner(tmp_path)
        assert current is not None
        ghost = dict(current, lease_id="ghost-lease", live_session_id="ghost-live")
        already_ran = "an ask already answered by the lease this chat replaced, verbatim"
        db.append_message("chat", "user", already_ran)
        settled = mailbox.deliver_to_live_owner(tmp_path, ghost, already_ran, delivery_id="a" * 32)
        fresh = mailbox.deliver_to_live_owner(tmp_path, ghost, "an ask the store does not have",
                                              delivery_id="b" * 32)
        claim = mailbox.claim_pending_delivery(tmp_path, current)
        assert claim["delivery_id"] == fresh["delivery_id"], "a body the store has must not run twice"
        receipt = mailbox.read_delivery_result(tmp_path, settled["delivery_id"])
        assert (receipt["status"], receipt["reason"]) == ("settled", "already_present")
        assert receipt["message"] == already_ran
    finally:
        lease.release()
        db.close()


def test_unreachable_queued_delivery_expires_with_a_reason(tmp_path, monkeypatch):
    from tools import bot_live_delivery as mailbox

    db, lease = _live_owner(tmp_path)
    try:
        current = mailbox.find_canonical_live_owner(tmp_path)
        assert current is not None
        other = dict(current, session_id="other-chat", lease_id="ghost-lease",
                     live_session_id="ghost-live")
        monkeypatch.setattr(mailbox.time, "time_ns",
                            lambda: int((time.time() - 2 * mailbox.UNREACHABLE_AFTER_SECONDS) * 1e9))
        expired = mailbox.deliver_to_live_owner(tmp_path, other, "an ask this chat cannot reach",
                                                delivery_id="a" * 32)
        monkeypatch.undo()
        young = mailbox.deliver_to_live_owner(tmp_path, other, "an ask its owner may still claim",
                                              delivery_id="b" * 32)
        assert mailbox.claim_pending_delivery(tmp_path, current) is None
        receipt = mailbox.read_delivery_result(tmp_path, expired["delivery_id"])
        assert (receipt["status"], receipt["reason"]) == ("failed", "no_live_consumer")
        assert receipt["message"] == expired["message"]
        assert mailbox.read_delivery_result(tmp_path, young["delivery_id"])["status"] == "queued"
    finally:
        lease.release()
        db.close()


_AGE_UNREADABLE = object()


@pytest.mark.parametrize("stamp", [_AGE_UNREADABLE, 0, "yesterday"],
                         ids=["no-created-at", "zero", "not-a-number"])
def test_unreachable_delivery_without_a_measurable_age_is_kept(tmp_path, monkeypatch, stamp):
    """A ticket whose age cannot be read stays queued, never failed as infinitely old.

    ``created_at`` is not a required field of a delivery record, so a foreign or half-written ticket
    reaches the expiry arm with no usable stamp. Expiry is terminal — the record never reaches a turn
    afterwards — so an unreadable age must not be read as infinitely old (nor raise out of the sweep).
    """
    from tools import bot_live_delivery as mailbox

    db, lease = _live_owner(tmp_path)
    try:
        current = mailbox.find_canonical_live_owner(tmp_path)
        assert current is not None
        other = dict(current, session_id="other-chat", lease_id="ghost-lease",
                     live_session_id="ghost-live")
        monkeypatch.setattr(mailbox.time, "time_ns",
                            lambda: int((time.time() - 2 * mailbox.UNREACHABLE_AFTER_SECONDS) * 1e9))
        aged = mailbox.deliver_to_live_owner(tmp_path, other, "an ask this chat cannot reach",
                                             delivery_id="a" * 32)
        monkeypatch.undo()
        root = tmp_path / "runtime" / mailbox.DELIVERY_DIR_NAME
        unreadable_id = "b" * 32
        record = dict(aged, delivery_id=unreadable_id, id=unreadable_id, message="ask b")
        if stamp is _AGE_UNREADABLE:
            record.pop("created_at", None)
        else:
            record["created_at"] = stamp
        (root / f"{unreadable_id}.json").write_text(json.dumps(record), encoding="utf-8")
        assert mailbox.claim_pending_delivery(tmp_path, current) is None
        # Control: the same aged ticket with its stamp intact is still expired by this same sweep,
        # so the assertion below cannot pass on a sweep that never reached the expiry arm.
        receipt = mailbox.read_delivery_result(tmp_path, aged["delivery_id"])
        assert (receipt["status"], receipt["reason"]) == ("failed", "no_live_consumer")
        kept = mailbox.read_delivery_result(tmp_path, unreadable_id)
        assert kept["status"] == "queued", "an unreadable age must not expire a record"
        assert kept["message"] == "ask b"
    finally:
        lease.release()
        db.close()


def test_reclaim_logs_the_matching_arm_and_the_lengths(tmp_path, caplog):
    """The audit trail names the arm that matched, so a hash match never hides behind a substring one."""
    from tools import bot_live_delivery as mailbox

    db, lease = _live_owner(tmp_path)
    try:
        current = mailbox.find_canonical_live_owner(tmp_path)
        assert current is not None
        ghost = dict(current, lease_id="ghost-lease", live_session_id="ghost-live")
        exact = "an ask already answered by the lease this chat replaced, verbatim"
        inner = ("an ask this chat already ran inside a longer stored body that the substring arm "
                 "still finds, which is the arm that can match a body it never saw")
        assert len(inner) >= mailbox._SHORTEST_SUBSTRING_BODY, "the substring arm needs a long body"
        stored = f"{inner} and then more of the stored body around it"
        db.append_message("chat", "user", exact)
        db.append_message("chat", "user", stored)
        mailbox.deliver_to_live_owner(tmp_path, ghost, exact, delivery_id="a" * 32)
        mailbox.deliver_to_live_owner(tmp_path, ghost, inner, delivery_id="b" * 32)
        with caplog.at_level(logging.INFO, logger="tools.bot_live_delivery"):
            assert mailbox.claim_pending_delivery(tmp_path, current) is None
        logged = "\n".join(record.getMessage() for record in caplog.records
                           if record.name == "tools.bot_live_delivery")
        assert "sha256" in logged and str(len(exact)) in logged
        assert "substring" in logged and str(len(inner)) in logged and str(len(stored)) in logged
        assert exact not in logged and inner not in logged and stored not in logged, \
            "log the lengths, not the bodies"
        for delivery_id in ("a" * 32, "b" * 32):
            receipt = mailbox.read_delivery_result(tmp_path, delivery_id)
            assert (receipt["status"], receipt["reason"]) == ("settled", "already_present")
    finally:
        lease.release()
        db.close()


def test_exact_id_reads_fail_closed_on_a_corrupt_ticket(tmp_path):
    from tools import bot_live_delivery as mailbox

    owner = dict(profile_home=str(tmp_path.resolve()), session_id="chat",
                 lease_id="lease", live_session_id="live")
    root = tmp_path / "runtime" / mailbox.DELIVERY_DIR_NAME
    root.mkdir(parents=True, exist_ok=True)
    corrupt = root / f"{'e' * 32}.json"
    corrupt.write_text("{not json", encoding="utf-8")
    # Uninspectable is not absent: an exact-id read must fail closed instead of
    # minting a fresh receipt over a ticket whose bytes it could not read.
    with pytest.raises(ValueError):
        mailbox.deliver_to_live_owner(tmp_path, owner, "same id", delivery_id="e" * 32)
    with pytest.raises(ValueError):
        mailbox.read_delivery_result(tmp_path, "e" * 32)
    assert corrupt.read_text(encoding="utf-8") == "{not json"


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


