"""P11 (§4.8): ENUM B is a BOUNDARY translation, never a storage rename.

The mailbox keeps queued -> claimed -> settled on disk; anything surfaced to a
sender says queued -> running -> delivered.

Admission and the claim belong to the profile's canonical authority
(``bot_relay.deliver``), so the fixture below stands in for it: one admission per
envelope id, and a settle republishes the receipt file the sender polls.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import bot_live_delivery as mailbox
from tools import bot_mode_dm


@pytest.fixture()
def owner(tmp_path):
    return dict(profile_home=str(tmp_path.resolve()), session_id="chat",
                lease_id="lease", live_session_id="live")


class _FakeAuthority:
    """The profile's canonical delivery door: admit once per id, settle on the owner's word."""

    def __init__(self):
        self.records = {}

    def __call__(self, home, params):
        record = self.records.get(params["id"])
        if record is None:
            record = self.records[params["id"]] = dict(
                status="queued", delivery_id=params["id"],
                profile_home=str(Path(home).resolve()), session_id="chat",
                message=params["message"], admission_id="adm-" + params["id"][:8], reply="")
            with mailbox._locked(home) as root:
                mailbox._write(root / f"{params['id']}.json", record)
        return {key: record[key] for key in ("status", "delivery_id", "profile_home", "session_id", "reply")}

    def settle(self, delivery_id, **outcome):
        record = self.records[delivery_id]
        record.update(outcome)
        with mailbox._locked(record["profile_home"]) as root:
            mailbox._write(root / f"{delivery_id}.json", record)


@pytest.fixture()
def authority(tmp_path, owner, monkeypatch):
    fake = _FakeAuthority()
    monkeypatch.setattr(mailbox, "authority_delivery", fake)
    monkeypatch.setattr(mailbox, "find_canonical_live_owner", lambda home: owner)
    return fake


def test_public_status_maps_only_the_storage_words():
    assert mailbox.public_delivery_status("claimed") == "running"
    assert mailbox.public_delivery_status("settled") == "delivered"
    assert mailbox.public_delivery_status("queued") == "queued"
    assert mailbox.public_delivery_status("ambiguous") == "ambiguous"
    assert mailbox.public_delivery_status("") == ""


def test_public_read_translates_without_touching_storage(tmp_path, owner, authority):
    delivery_id = "b" * 32
    mailbox.deliver_to_live_owner(tmp_path, owner, "hello", delivery_id=delivery_id)
    assert mailbox.read_public_delivery_result(tmp_path, delivery_id)["status"] == "queued"

    authority.settle(delivery_id, status="claimed")
    assert mailbox.read_delivery_result(tmp_path, delivery_id)["status"] == "claimed"
    public = mailbox.read_public_delivery_result(tmp_path, delivery_id)
    assert public["status"] == "running" and public["message"] == "hello"

    mailbox.complete_delivery(tmp_path, delivery_id, status="settled", reply="answer")
    assert mailbox.read_delivery_result(tmp_path, delivery_id)["status"] == "settled"
    public = mailbox.read_public_delivery_result(tmp_path, delivery_id)
    assert public["status"] == "delivered" and public["reply"] == "answer"
    # storage keeps its own word, and the caller's copy is a copy
    assert mailbox.read_delivery_result(tmp_path, delivery_id)["status"] == "settled"
    assert mailbox.read_public_delivery_result(tmp_path, "c" * 32) is None


def test_wait_live_dm_prints_enum_b_while_disk_keeps_the_storage_word(
    tmp_path, owner, authority, monkeypatch, capsys
):
    delivery_id = "d" * 32
    monkeypatch.setattr(bot_mode_dm, "_LIVE_WAIT_SECONDS", 0.01)
    monkeypatch.setattr(bot_mode_dm, "_LIVE_WAIT_MAX_SECONDS", 0.05)
    mailbox.deliver_to_live_owner(tmp_path, owner, "hello", delivery_id=delivery_id)
    authority.settle(delivery_id, status="claimed")

    assert bot_mode_dm._wait_live_dm(str(tmp_path), delivery_id) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "running", "a sender never sees the word 'claimed'"
    assert payload["detail"].startswith("Delivery remains pending")
    assert mailbox.read_delivery_result(tmp_path, delivery_id)["status"] == "claimed"

    mailbox.complete_delivery(tmp_path, delivery_id, status="settled", reply="answer")
    assert bot_mode_dm._wait_live_dm(str(tmp_path), delivery_id) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "delivered" and payload["reply"] == "answer"
    assert mailbox.read_delivery_result(tmp_path, delivery_id)["status"] == "settled"


def test_failed_delivery_keeps_its_own_word(tmp_path, owner, authority, monkeypatch, capsys):
    delivery_id = "e" * 32
    monkeypatch.setattr(bot_mode_dm, "_LIVE_WAIT_SECONDS", 0)
    mailbox.deliver_to_live_owner(tmp_path, owner, "hello", delivery_id=delivery_id)
    authority.settle(delivery_id, status="claimed")
    mailbox.complete_delivery(tmp_path, delivery_id, status="failed", error="boom")
    assert bot_mode_dm._wait_live_dm(str(tmp_path), delivery_id) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed" and payload["error"] == "boom"


def test_ambiguous_without_a_record_is_reported_honestly(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bot_mode_dm, "_LIVE_WAIT_SECONDS", 0)
    assert bot_mode_dm._wait_live_dm(str(tmp_path), "f" * 32) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ambiguous"
    assert "Do not resend" in payload["detail"], "an unknown outcome is never a licence to resend"
