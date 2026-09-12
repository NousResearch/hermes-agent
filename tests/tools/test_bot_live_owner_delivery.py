"""Canonical delivery invariants: stable envelope IDs, authority-only admission,
immutable receipts. The mailbox is receipt storage, never an execution queue."""
import os
from pathlib import Path

import pytest

from tools import bot_live_delivery as mailbox


class _FakeAuthority:
    """Stands in for the profile authority's ``bot_relay.deliver`` RPC."""

    def __init__(self):
        self.calls = []
        self.records = {}

    def __call__(self, home, params):
        self.calls.append((Path(home).resolve(), dict(params)))
        record = self.records.get(params["id"])
        if record is None:
            record = self.records[params["id"]] = dict(
                status="queued", delivery_id=params["id"], profile_home=str(Path(home).resolve()),
                session_id="local-bot", message=params["message"], admission_id="adm-" + params["id"][:6],
                reply="", **({"author": params["author"]} if "author" in params else {}))
            with mailbox._locked(home) as root:
                mailbox._write(root / f"{params['id']}.json", record)
        elif (record["message"] != params["message"]
              or record.get("author") != params.get("author")):
            raise ValueError("admission_conflict")
        return {k: record[k] for k in ("status", "delivery_id", "profile_home", "session_id", "reply")}


def _owner(home):
    return dict(profile_home=str(Path(home).resolve()), session_id="local-bot",
                lease_id="authority-1", live_session_id="local-bot")


def test_same_envelope_id_admits_once_and_conflicts_on_changed_payload(tmp_path, monkeypatch):
    authority = _FakeAuthority()
    monkeypatch.setattr(mailbox, "authority_delivery", authority)
    delivery_id = "a" * 32
    queued = mailbox.deliver_to_live_owner(tmp_path, _owner(tmp_path), "hello", delivery_id=delivery_id)
    assert queued["status"] == "queued" and queued["delivery_id"] == delivery_id
    # Exact retry inspects the same admission; it never mints a second envelope.
    assert mailbox.deliver_to_live_owner(tmp_path, _owner(tmp_path), "hello", delivery_id=delivery_id) == queued
    assert [params["id"] for _home, params in authority.calls] == [delivery_id, delivery_id]
    assert len(authority.records) == 1
    with pytest.raises(ValueError):
        mailbox.deliver_to_live_owner(tmp_path, _owner(tmp_path), "different", delivery_id=delivery_id)
    # Reading the receipt resolves through the same admission, not a local guess.
    assert mailbox.read_delivery_result(tmp_path, delivery_id)["status"] == "queued"
    # The retired UI claim consumer never hands out work.
    assert mailbox.claim_pending_delivery(tmp_path, _owner(tmp_path)) is None
    if os.name != "nt":
        for path in (tmp_path / "runtime" / mailbox.DELIVERY_DIR_NAME).iterdir():
            assert path.stat().st_mode & 0o077 == 0


def test_owner_from_another_home_or_malformed_id_is_refused_before_admission(tmp_path, monkeypatch):
    authority = _FakeAuthority()
    monkeypatch.setattr(mailbox, "authority_delivery", authority)
    foreign = dict(_owner(tmp_path), profile_home=str(tmp_path / "elsewhere"))
    with pytest.raises(ValueError, match="different profile home"):
        mailbox.deliver_to_live_owner(tmp_path, foreign, "hello", delivery_id="b" * 32)
    with pytest.raises(ValueError, match="delivery id"):
        mailbox.deliver_to_live_owner(tmp_path, _owner(tmp_path), "hello", delivery_id="../evil")
    assert authority.calls == []


def test_receipt_read_forwards_the_immutable_author(tmp_path, monkeypatch):
    """F10: an authored admission is re-read with its stored author, so the authority sees the same payload."""
    authority = _FakeAuthority()
    monkeypatch.setattr(mailbox, "authority_delivery", authority)
    delivery_id = "d" * 32
    author = {"kind": "user", "id": "u-1", "display": "Ann"}
    queued = mailbox.deliver_to_live_owner(tmp_path, _owner(tmp_path), "hello", delivery_id=delivery_id, author=author)
    assert mailbox.read_delivery_result(tmp_path, delivery_id) == queued
    assert [params.get("author") for _home, params in authority.calls] == [author, author]


@pytest.mark.parametrize("terminal_status", ["settled", "failed", "cancelled"])
def test_terminal_receipt_is_immutable(tmp_path, terminal_status):
    delivery_id = "c" * 32
    with mailbox._locked(tmp_path) as root:
        mailbox._write(root / f"{delivery_id}.json", dict(delivery_id=delivery_id, status="claimed",
                                                          message="hello", created_at=1))
    receipt = mailbox.complete_delivery(tmp_path, delivery_id, status=terminal_status, reply="answer")
    assert mailbox.read_delivery_result(tmp_path, delivery_id) == receipt
    assert mailbox.complete_delivery(tmp_path, delivery_id, status=terminal_status, reply="answer") == receipt
    with pytest.raises(ValueError):
        mailbox.complete_delivery(tmp_path, delivery_id, status=terminal_status, reply="rewrite")
    with pytest.raises(ValueError):
        mailbox.complete_delivery(tmp_path, "d" * 32, status="not-terminal")


def _ready(instance_id):
    from types import SimpleNamespace

    return lambda home, timeout: SimpleNamespace(state="ready", endpoint=SimpleNamespace(instance_id=instance_id))


def test_canonical_owner_is_the_authority_and_follows_compression(tmp_path, monkeypatch):
    from hermes_state import SessionDB
    import hermes_cli.gateway_runtime as runtime

    from types import SimpleNamespace

    monkeypatch.setattr(runtime, "discover_gateway_endpoint",
                        lambda home, timeout: SimpleNamespace(state="absent", endpoint=None))
    with pytest.raises(ValueError, match="authority"):
        mailbox.find_canonical_live_owner(tmp_path)

    monkeypatch.setattr(runtime, "discover_gateway_endpoint", _ready("authority-1"))
    assert mailbox.find_canonical_live_owner(tmp_path) is None  # no state.db yet
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(session_id="scratch", source="cli")
        db.set_session_title("scratch", "Scratch")
        assert mailbox.find_canonical_live_owner(tmp_path) is None  # no Bot Chat
        db.create_session(session_id="chat", source="cli")
        db.set_session_title("chat", "Bot Chat")
        owner = mailbox.find_canonical_live_owner(tmp_path)
        assert owner["canonical"] is True and owner["lease_id"] == "authority-1"
        assert owner["session_id"] == owner["live_session_id"] == "chat"
        db.end_session("chat", "compression")
        db.create_session(session_id="tip", source="cli", parent_session_id="chat")
        assert mailbox.find_canonical_live_owner(tmp_path)["session_id"] == "tip"
    finally:
        db.close()


def test_delivery_keeps_the_sender_and_refuses_a_different_one_under_the_same_id(tmp_path, monkeypatch):
    from tools import bot_live_delivery as mailbox

    authority = _FakeAuthority()
    monkeypatch.setattr(mailbox, "authority_delivery", authority)
    owner = dict(profile_home=str(tmp_path.resolve()), session_id="chat", lease_id="lease", live_session_id="live")
    author = {"id": "bot:coder", "name": "coder", "is_bot": True}
    queued = mailbox.deliver_to_live_owner(tmp_path, owner, "hello", delivery_id="b" * 32, author=author)
    assert authority.calls[-1][1]["author"] == author
    assert authority.records["b" * 32]["author"] == author
    assert mailbox.deliver_to_live_owner(tmp_path, owner, "hello", delivery_id="b" * 32, author=author) == queued
    with pytest.raises(ValueError):
        mailbox.deliver_to_live_owner(tmp_path, owner, "hello", delivery_id="b" * 32, author={**author, "id": "bot:other"})
    mailbox.deliver_to_live_owner(tmp_path, owner, "no sender", delivery_id="c" * 32)
    assert "author" not in authority.calls[-1][1]
