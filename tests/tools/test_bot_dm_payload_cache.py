"""DM payload files must be sweepable, not left in the OS temp dir.

`message_agent` writes the message body to a file and hands the path to a
*background* delivery, so it cannot be removed at the call site. Nothing
reaped it either, so every agent-to-agent DM leaked one file for the life
of the machine. `tools/tool_result_storage.py` states the convention this
now follows: Hermes-owned payloads live under `$HERMES_HOME/cache/`
"instead of littering the OS temp dir", where the housekeeping loop can
prune them.
"""

import os
import time
from pathlib import Path

import pytest

from tools import bot_mode_dm


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


class TestLocation:
    def test_payloads_land_in_the_sweepable_cache_dir(self, home):
        path = Path(bot_mode_dm._write_dm_file("hello teammate"))

        assert path.parent == home / "cache" / "bot-dm", (
            f"payload written to {path.parent} — outside the swept cache dir, "
            "so nothing will ever reap it"
        )
        assert path.read_text(encoding="utf-8") == "hello teammate"

    def test_the_body_is_written_verbatim(self, home):
        body = 'he said "ship it"; $(id) `id`\nsecond line\n'
        path = Path(bot_mode_dm._write_dm_file(body))
        assert path.read_text(encoding="utf-8") == body

    def test_permissions_stay_owner_only(self, home):
        path = Path(bot_mode_dm._write_dm_file("private"))
        assert oct(path.stat().st_mode & 0o777) == "0o600"

    def test_an_unusable_cache_dir_falls_back_instead_of_failing(self, home, monkeypatch):
        """A read-only HERMES_HOME must degrade, never drop the DM."""
        def _boom(*a, **k):
            raise OSError("read-only")

        monkeypatch.setattr(Path, "mkdir", _boom)
        path = Path(bot_mode_dm._write_dm_file("still delivered"))
        assert path.read_text(encoding="utf-8") == "still delivered"


class TestCleanup:
    def test_expired_payloads_are_removed(self, home):
        old = Path(bot_mode_dm._write_dm_file("stale"))
        fresh = Path(bot_mode_dm._write_dm_file("recent"))
        past = time.time() - (bot_mode_dm.BOT_DM_MAX_AGE_HOURS + 1) * 3600
        os.utime(old, (past, past))

        removed = bot_mode_dm.cleanup_bot_dm_cache()

        assert removed == 1
        assert not old.exists()
        assert fresh.exists(), "a payload still in flight was reaped"

    def test_a_missing_dir_is_not_an_error(self, home):
        assert bot_mode_dm.cleanup_bot_dm_cache() == 0

    def test_cleanup_reports_how_many_it_removed(self, home):
        past = time.time() - (bot_mode_dm.BOT_DM_MAX_AGE_HOURS + 1) * 3600
        for _ in range(3):
            p = Path(bot_mode_dm._write_dm_file("x"))
            os.utime(p, (past, past))
        assert bot_mode_dm.cleanup_bot_dm_cache() == 3

    def test_the_once_per_process_prune_runs_at_most_once(self, home, monkeypatch):
        """CLI-only installs never run gateway housekeeping."""
        monkeypatch.setattr(bot_mode_dm, "_dm_pruned_once", False)
        calls = []
        monkeypatch.setattr(
            bot_mode_dm, "cleanup_bot_dm_cache", lambda *a, **k: calls.append(1) or 0
        )
        bot_mode_dm._prune_bot_dm_once()
        bot_mode_dm._prune_bot_dm_once()
        assert len(calls) == 1

