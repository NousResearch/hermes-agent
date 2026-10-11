"""#128996: a DM behind an unadvertised Bot Chat owner is parked, then drained — never cloned.

``_run_delivery`` used to fall through to ``_run_local_turn`` whenever the
recipient's Bot Chat had no advertised live consumer, even though another
surface (a one-shot CLI turn, a window that raced registration) held the
lease: the spawned CLI copy took the lease and collided with the human
surface. The fix parks the DM under the recipient's home and delivers it via
the cron ``bot_chat_pending`` contract:

- live conversion: a consumer that advertises takes the parked record as a
  mailbox ticket (sender runner or live-owner poller — idempotent either way);
- lease release: with no surface holding the chat, the record is claimed and
  the ONE sanctioned CLI turn runs (at-most-once: the claim never expires);
- lapsed budget: the runner exits ``status=queued`` (receipt retained, do
  not resend).

These tests lock the drain's state machine on the real record store; the
no-clone guard itself lives in ``tests/tools/test_bot_mode_dm.py``.
"""
import json
import subprocess

import pytest

from tools import bot_dm_pending as pending
from tools import bot_mode_dm


def _owner(home, live=True):
    entry = {"profile_home": str(home), "session_id": "s", "lease_id": "l"}
    if live:
        entry["live_session_id"] = "live"
    return entry


def _parked(tmp_path, monkeypatch, *, argv=("hermes", "-p", "researcher")):
    dm_file = tmp_path / "message.txt"
    dm_file.write_text("hello", encoding="utf-8")
    home = tmp_path / "profile"
    home.mkdir()
    delivery_id = bot_mode_dm._dm_delivery_id(dm_file)
    record = pending.park(delivery_id, home, argv=list(argv), dm_file=str(dm_file),
                          message="hello", author={"id": "bot:default", "name": "hermes", "is_bot": True})
    return home, dm_file, delivery_id, record


def test_park_is_idempotent_per_payload(tmp_path):
    home, dm_file, delivery_id, record = _parked(tmp_path, None)
    again = pending.park(delivery_id, home, argv=["hermes", "-p", "researcher"], dm_file=str(dm_file),
                         message="hello", author={"id": "bot:default", "name": "hermes", "is_bot": True})
    assert again["status"] == "queued" and again["id"] == delivery_id
    with pytest.raises(ValueError, match="different payload"):
        pending.park(delivery_id, home, argv=["hermes", "-p", "researcher"], dm_file=str(dm_file),
                     message="tampered", author=None)


def test_claim_settles_once_and_never_reclaims(tmp_path):
    home, dm_file, delivery_id, _ = _parked(tmp_path, None)
    assert pending.claim(home, delivery_id, reason="test")["status"] == "claimed"
    assert pending.claim(home, delivery_id, reason="test") is None
    settled = pending.settle(home, delivery_id, status="settled")
    assert settled["status"] == "settled"
    # The receipt is permanent; a second settle does not resurrect or change it.
    assert pending.settle(home, delivery_id, status="failed", error="x")["status"] == "settled"


def test_convert_to_live_owner_writes_ticket_and_fences_the_cli_drain(tmp_path):
    home, dm_file, delivery_id, _ = _parked(tmp_path, None)
    record = pending.convert_to_live_owner(home, delivery_id, _owner(home))
    assert record["status"] == "transferred"

    from tools import bot_live_delivery as mailbox

    ticket = mailbox.read_delivery_result(home, delivery_id)
    assert ticket["status"] == "queued"
    assert ticket["message"] == "hello"
    assert ticket["author"] == {"id": "bot:default", "name": "hermes", "is_bot": True}
    assert ticket["owner"]["lease_id"] == "l"
    # The pending record no longer executes on any CLI drain.
    assert pending.claim(home, delivery_id, reason="lease released") is None
    assert [r["id"] for r in pending.pending_records_for_home(home)] == []


def test_damaged_pending_receipt_is_skipped_not_wedge(tmp_path, caplog):
    import logging

    home, _, delivery_id, _ = _parked(tmp_path, None)
    bad = pending._root(home) / f"{'e' * 64}.json"
    bad.write_text("[1, 2, 3]", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger=pending.logger.name):
        records = pending.pending_records_for_home(home)
    assert [r["id"] for r in records] == [delivery_id]
    assert sum("Unreadable pending DM receipt" in r.message for r in caplog.records) == 1


def test_deferred_runner_spawns_the_cli_turn_only_after_release(tmp_path, monkeypatch, capsys):
    """The lapsed-owner case end to end: unadvertised hold → park → release → one CLI turn."""
    home, dm_file, argv = tmp_path / "profile", tmp_path / "message.txt", ["hermes", "-p", "researcher"]
    home.mkdir()
    dm_file.write_text("hello", encoding="utf-8")
    monkeypatch.setattr(bot_mode_dm, "_local_delivery_home", lambda a: home)
    monkeypatch.setattr(bot_mode_dm, "_DEFER_POLL_SECONDS", 0)

    holds = {"owner": True}
    monkeypatch.setattr(
        "tools.bot_live_delivery.find_canonical_owner",
        lambda profile_home: _owner(home) if holds["owner"] else None,
    )
    monkeypatch.setattr("tools.bot_live_delivery.find_canonical_live_owner", lambda profile_home: None)
    turns = []
    monkeypatch.setattr(bot_mode_dm, "_run_local_turn",
                        lambda a, f, env=None: turns.append(a) or 0)
    monkeypatch.setattr(bot_mode_dm, "_delivery_lock", lambda *a, **k: __import__("contextlib").nullcontext())

    import threading

    def release():
        import time as _t

        _t.sleep(0.05)
        holds["owner"] = False

    threading.Thread(target=release, daemon=True).start()
    assert bot_mode_dm._run_delivery(list(argv), str(dm_file), stdin_file=False) == 0
    assert turns == [argv]  # exactly one CLI turn, only after the release
    assert not dm_file.exists()  # the drain owns the file cleanup
    delivery_id = bot_mode_dm._dm_delivery_id(dm_file)
    assert pending.read_pending(home, delivery_id)["status"] == "settled"
    # A second delivery of the same dm file id cannot re-run: the record is terminal.
    assert pending.claim(home, delivery_id, reason="lease released") is None


def test_deferred_runner_converts_when_the_owner_advertises(tmp_path, monkeypatch):
    home, dm_file, argv = tmp_path / "profile", tmp_path / "message.txt", ["hermes", "-p", "researcher"]
    home.mkdir()
    dm_file.write_text("hello", encoding="utf-8")
    monkeypatch.setattr(bot_mode_dm, "_local_delivery_home", lambda a: home)
    monkeypatch.setattr(bot_mode_dm, "_DEFER_POLL_SECONDS", 0)
    monkeypatch.setattr(bot_mode_dm, "_run_local_turn", lambda *a, **k: pytest.fail("CLI clone spawned"))

    advertised = {"on": False}
    monkeypatch.setattr(
        "tools.bot_live_delivery.find_canonical_owner",
        lambda profile_home: _owner(home),
    )
    monkeypatch.setattr(
        "tools.bot_live_delivery.find_canonical_live_owner",
        lambda profile_home: _owner(home) if advertised["on"] else None,
    )
    monkeypatch.setattr(bot_mode_dm, "_wait_live_dm", lambda *a, **k: 0)
    monkeypatch.setattr(bot_mode_dm, "_admit_live_dm",
                        lambda *a, **k: pytest.fail("direct admission must not race the parked record"))

    import threading

    def advertise():
        import time as _t

        _t.sleep(0.05)
        advertised["on"] = True

    threading.Thread(target=advertise, daemon=True).start()
    assert bot_mode_dm._run_delivery(list(argv), str(dm_file), stdin_file=False) == 0
    delivery_id = bot_mode_dm._dm_delivery_id(dm_file)
    from tools import bot_live_delivery as mailbox

    assert mailbox.read_delivery_result(home, delivery_id)["status"] == "queued"
    assert pending.read_pending(home, delivery_id)["status"] == "transferred"
