"""P11 (§4.8): ENUM B is a BOUNDARY translation, never a storage rename.

The transport keeps queued -> claimed -> settled; anything surfaced to a
sender says queued -> running -> delivered. These tests pin the boundary
functions and the sender-facing wait output directly, with the delivery store
stubbed: the translation is what is under test here, not the read path.
"""
from __future__ import annotations

import json

import pytest

from tools import bot_live_delivery as mailbox
from tools import bot_mode_dm


def _stored_await(record):
    """An ``await_delivery`` stand-in: hands back the stored record, never dials."""

    def _await(home, delivery_id, timeout=0.0):
        return dict(record)

    return _await


def test_public_status_maps_only_the_storage_words():
    assert mailbox.public_delivery_status("claimed") == "running"
    assert mailbox.public_delivery_status("settled") == "delivered"
    assert mailbox.public_delivery_status("queued") == "queued"
    assert mailbox.public_delivery_status("ambiguous") == "ambiguous"
    assert mailbox.public_delivery_status("") == ""


def test_public_read_translates_without_touching_storage(tmp_path, monkeypatch):
    delivery_id = "b" * 32
    stored = dict(delivery_id=delivery_id, status="queued", message="hello")
    monkeypatch.setattr(
        mailbox, "read_delivery_result",
        lambda home, did: dict(stored) if did == delivery_id else None)
    assert mailbox.read_public_delivery_result(tmp_path, delivery_id)["status"] == "queued"

    stored["status"] = "claimed"
    assert mailbox.read_delivery_result(tmp_path, delivery_id)["status"] == "claimed"
    public = mailbox.read_public_delivery_result(tmp_path, delivery_id)
    assert public["status"] == "running" and public["message"] == "hello"
    assert public is not stored and stored["status"] == "claimed"

    stored.update(status="settled", reply="answer")
    assert mailbox.read_delivery_result(tmp_path, delivery_id)["status"] == "settled"
    public = mailbox.read_public_delivery_result(tmp_path, delivery_id)
    assert public["status"] == "delivered" and public["reply"] == "answer"
    # storage keeps its own word, and the caller's copy is a copy
    assert stored["status"] == "settled"
    assert mailbox.read_public_delivery_result(tmp_path, "c" * 32) is None


def test_wait_live_dm_prints_enum_b_while_disk_keeps_the_storage_word(
    tmp_path, monkeypatch, capsys
):
    delivery_id = "d" * 32
    stored = dict(delivery_id=delivery_id, status="claimed", message="hello")
    monkeypatch.setattr(bot_mode_dm, "_LIVE_WAIT_SECONDS", 0)
    monkeypatch.setattr(mailbox, "await_delivery", _stored_await(stored))

    assert bot_mode_dm._wait_live_dm(str(tmp_path), delivery_id) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "running", "a sender never sees the word 'claimed'"
    assert payload["detail"].startswith("Delivery remains pending")

    stored.update(status="settled", reply="answer")
    assert bot_mode_dm._wait_live_dm(str(tmp_path), delivery_id) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "delivered" and payload["reply"] == "answer"
    assert stored["status"] == "settled", "the store keeps queued/claimed/settled"


def test_failed_delivery_keeps_its_own_word(tmp_path, monkeypatch, capsys):
    delivery_id = "e" * 32
    stored = dict(delivery_id=delivery_id, status="failed", message="hello", error="boom")
    monkeypatch.setattr(bot_mode_dm, "_LIVE_WAIT_SECONDS", 0)
    monkeypatch.setattr(mailbox, "await_delivery", _stored_await(stored))

    assert bot_mode_dm._wait_live_dm(str(tmp_path), delivery_id) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed" and payload["error"] == "boom"


def test_ambiguous_without_a_record_is_reported_honestly(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bot_mode_dm, "_LIVE_WAIT_SECONDS", 0)
    assert bot_mode_dm._wait_live_dm(str(tmp_path), "f" * 32) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ambiguous"
    assert "Do not resend" in payload["detail"], "an unknown outcome is never a licence to resend"
