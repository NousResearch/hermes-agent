"""P11 (§4.8): ENUM B is a BOUNDARY translation, never a storage rename.

The mailbox keeps queued -> claimed -> settled on disk; anything surfaced to a
sender says queued -> running -> delivered.
"""
from __future__ import annotations

import json

import pytest

from tools import bot_live_delivery as mailbox
from tools import bot_mode_dm


@pytest.fixture()
def owner(tmp_path):
    return dict(profile_home=str(tmp_path.resolve()), session_id="chat",
                lease_id="lease", live_session_id="live")


def test_public_status_maps_only_the_storage_words():
    assert mailbox.public_delivery_status("claimed") == "running"
    assert mailbox.public_delivery_status("settled") == "delivered"
    assert mailbox.public_delivery_status("queued") == "queued"
    assert mailbox.public_delivery_status("ambiguous") == "ambiguous"
    assert mailbox.public_delivery_status("") == ""


def test_public_read_translates_without_touching_storage(tmp_path, owner):
    delivery_id = "b" * 32
    mailbox.deliver_to_live_owner(tmp_path, owner, "hello", delivery_id=delivery_id)
    assert mailbox.read_public_delivery_result(tmp_path, delivery_id)["status"] == "queued"

    assert mailbox.claim_pending_delivery(tmp_path, owner) is not None
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
    tmp_path, owner, monkeypatch, capsys
):
    delivery_id = "d" * 32
    monkeypatch.setattr(bot_mode_dm, "_LIVE_WAIT_SECONDS", 0)
    mailbox.deliver_to_live_owner(tmp_path, owner, "hello", delivery_id=delivery_id)
    mailbox.claim_pending_delivery(tmp_path, owner)

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


def test_failed_delivery_keeps_its_own_word(tmp_path, owner, monkeypatch, capsys):
    delivery_id = "e" * 32
    monkeypatch.setattr(bot_mode_dm, "_LIVE_WAIT_SECONDS", 0)
    mailbox.deliver_to_live_owner(tmp_path, owner, "hello", delivery_id=delivery_id)
    mailbox.claim_pending_delivery(tmp_path, owner)
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
