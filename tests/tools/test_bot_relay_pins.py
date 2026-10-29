"""Behaviour pins for the cross-machine relay queue (tools/bot_relay.py).

Freezes the stale-file sweep edge so a later move onto shared cleanup can delete this code without
changing what is kept. Replay-until-reply, delivery_timeout and expiry are pinned in
tests/tools/test_bot_relay.py. Time is injected via ``now=`` and file mtimes.
"""

from __future__ import annotations

import os

import pytest

from tools import bot_relay

T0 = 1_000_000.0  # fixed epoch for created_at / claim mtimes


@pytest.fixture()
def base(tmp_path):
    return bot_relay._ensure_dirs(tmp_path)


def test_sweep_stale_removes_only_files_older_than_stale_after(tmp_path, base):
    """``_sweep_stale`` unlinks ``*.json`` in outbox/claimed/replies whose mtime is strictly before ``now - STALE_AFTER_SECONDS``."""
    now = T0 + bot_relay.STALE_AFTER_SECONDS + 100
    cutoff = now - bot_relay.STALE_AFTER_SECONDS
    cases = {}
    for sub in (bot_relay.OUTBOX_DIR, bot_relay.CLAIMED_DIR, bot_relay.REPLIES_DIR):
        for label, mtime in (("old", cutoff - 1), ("edge", cutoff), ("fresh", cutoff + 1)):
            path = base / sub / f"{sub}-{label}.json"
            path.write_text("{}", encoding="utf-8")
            os.utime(path, (mtime, mtime))
            cases[path] = label
    other = base / bot_relay.CLAIMED_DIR / "notes.txt"
    other.write_text("x", encoding="utf-8")
    os.utime(other, (0, 0))

    assert bot_relay._sweep_stale(base, now=now) == 3
    for path, label in cases.items():
        assert path.exists() is (label != "old"), path
    assert other.exists()  # non-JSON files are never swept
