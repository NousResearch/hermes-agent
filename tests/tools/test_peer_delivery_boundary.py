"""P11 (§4.8): ENUM B is a BOUNDARY translation, never a storage rename.

The mailbox keeps queued -> claimed -> settled on disk; anything surfaced to a
sender says queued -> running -> delivered.

The records here are written the way the profile authority's consumer mirrors an
admission into the local store (``gateway/session_bot_mailbox.py``): the live-owner
mailbox is no longer a producer of its own, so the boundary contract is pinned
against the store's own shape instead of a local admission call.
"""
from __future__ import annotations

import json
import time

import pytest

from tools import bot_live_delivery as mailbox
from tools import bot_mode_dm


@pytest.fixture()
def owner(tmp_path):
    return dict(profile_home=str(tmp_path.resolve()), session_id="chat",
                lease_id="lease", live_session_id="live")


@pytest.fixture()
def record(tmp_path, owner):
    """Write one mailbox record at ``status`` and return it."""
    def _make(delivery_id, status="queued", message="hello", **extra):
        pinned = mailbox._owner(tmp_path, owner)
        root = mailbox._root(tmp_path)
        root.mkdir(parents=True, exist_ok=True)
        written = dict(delivery_id=delivery_id, id=delivery_id, owner=pinned, **pinned,
                       message=message, status=status, created_at=time.time_ns(),
                       sequence=1, **extra)
        mailbox._write(root / f"{delivery_id}.json", written)
        return written
    return _make


def test_public_status_maps_only_the_storage_words():
    assert mailbox.public_delivery_status("claimed") == "running"
    assert mailbox.public_delivery_status("settled") == "delivered"
    assert mailbox.public_delivery_status("queued") == "queued"
    assert mailbox.public_delivery_status("ambiguous") == "ambiguous"
    assert mailbox.public_delivery_status("") == ""


def test_public_read_translates_without_touching_storage(tmp_path, record):
    delivery_id = "b" * 32
    record(delivery_id, status="claimed")

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
    tmp_path, record, monkeypatch, capsys
):
    delivery_id = "d" * 32
    monkeypatch.setattr(bot_mode_dm, "_LIVE_WAIT_SECONDS", 0)
    record(delivery_id, status="claimed")

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


def test_failed_delivery_keeps_its_own_word(tmp_path, record, monkeypatch, capsys):
    delivery_id = "e" * 32
    monkeypatch.setattr(bot_mode_dm, "_LIVE_WAIT_SECONDS", 0)
    record(delivery_id, status="claimed")
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
