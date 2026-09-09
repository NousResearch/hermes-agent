"""DM payload files must be reaped by gateway housekeeping, not only in-band.

`message_agent` writes the message body to a file and hands the path to a
*background* delivery, so it cannot be removed at the call site. The runner
owns per-delivery cleanup, and `_write_dm_file` sweeps opportunistically —
but a gateway that never sends another DM would still keep orphans forever.
`cleanup_bot_dm_cache` follows the same contract as the other
``cleanup_*_cache`` helpers (returns the number of files removed) so the
gateway housekeeping loop in ``gateway/run.py`` prunes this cache on the
same hourly cadence as the media caches.
"""

import os
import time
from pathlib import Path

import pytest

from tools import bot_mode_dm


@pytest.fixture()
def temp_root(tmp_path, monkeypatch):
    monkeypatch.setattr(bot_mode_dm.tempfile, "gettempdir", lambda: str(tmp_path))
    return tmp_path


def _age(path: Path, seconds: float) -> None:
    past = time.time() - seconds
    os.utime(path, (past, past))


class TestCleanupContract:
    def test_expired_payloads_are_removed_and_counted(self, temp_root):
        old = Path(bot_mode_dm._write_dm_file("stale"))
        fresh = Path(bot_mode_dm._write_dm_file("recent"))
        _age(old, bot_mode_dm._DM_STALE_SECONDS + 1)

        removed = bot_mode_dm.cleanup_bot_dm_cache()

        assert removed == 1
        assert not old.exists()
        assert fresh.exists(), "a payload still in flight was reaped"

    def test_cleanup_reports_how_many_it_removed(self, temp_root):
        # write all files first: _write_dm_file itself sweeps opportunistically
        paths = [Path(bot_mode_dm._write_dm_file("x")) for _ in range(3)]
        for p in paths:
            _age(p, bot_mode_dm._DM_STALE_SECONDS + 1)
        assert bot_mode_dm.cleanup_bot_dm_cache() == 3

    def test_a_missing_dm_dir_is_not_an_error(self, temp_root):
        assert bot_mode_dm.cleanup_bot_dm_cache() == 0

    def test_legacy_and_relay_prefixed_orphans_are_swept(self, temp_root):
        legacy = temp_root / "hermes-dm-legacy.txt"
        relay = temp_root / "hermes-relay-dm-orphan.txt"
        unrelated = temp_root / "other.txt"
        for f in (legacy, relay, unrelated):
            f.write_text("secret", encoding="utf-8")
            _age(f, bot_mode_dm._DM_STALE_SECONDS + 1)

        removed = bot_mode_dm.cleanup_bot_dm_cache()

        assert removed == 2
        assert not legacy.exists()
        assert not relay.exists()
        assert unrelated.exists()

    def test_shorter_max_age_hours_is_honored(self, temp_root):
        recent = Path(bot_mode_dm._write_dm_file("an hour old"))
        _age(recent, 2 * 3600)
        assert bot_mode_dm.cleanup_bot_dm_cache(max_age_hours=1) == 1
        assert not recent.exists()
