"""Tests for terminal truncation spill + metadata (deferred retrieval)."""

import json
import os
from pathlib import Path

import pytest

from tools.terminal_tool import terminal_tool


@pytest.fixture
def small_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_constants import hermes_home_key
    import tools.tool_output_limits as lim
    monkeypatch.setattr(lim, "_cached_limits", {hermes_home_key(): {
        "max_bytes": 2000, "max_lines": 2000, "max_line_length": 2000,
    }})
    return tmp_path


class TestTruncationSpill:
    @pytest.mark.platforms("linux")
    def test_truncated_output_has_metadata_and_spill(self, small_cap):
        r = json.loads(terminal_tool(
            "python3 -c \"print('marker_head'); [print(f'row_{i}', 'x'*80) for i in range(200)]; print('marker_tail')\"",
            task_id="t-spill-1"))
        assert r["exit_code"] == 0
        assert "OUTPUT TRUNCATED" in r["output"]
        assert r["output_total_chars"] > 2000
        p = Path(r["full_output_path"])
        assert p.exists()
        full = p.read_text()
        assert "marker_head" in full and "marker_tail" in full
        # The spill contains rows that were cut from the visible window.
        assert "row_100 " in full
        assert "read_file" in r["truncation_note"]

    def test_small_output_has_no_metadata(self, small_cap):
        r = json.loads(terminal_tool("echo tiny", task_id="t-spill-2"))
        assert r["exit_code"] == 0
        assert "full_output_path" not in r
        assert "output_total_chars" not in r

    @pytest.mark.platforms("linux")
    def test_spill_is_redacted(self, small_cap):
        r = json.loads(terminal_tool(
            "python3 -c \"print('sk-proj-' + 'a1B2c3D4e5F6g7H8i9J0' * 3); [print('pad', 'y'*90) for i in range(200)]\"",
            task_id="t-spill-3"))
        p = Path(r["full_output_path"])
        full = p.read_text()
        assert "a1B2c3D4e5F6g7H8i9J0a1B2c3D4e5F6g7H8i9J0" not in full

    def test_old_spills_cleaned(self, small_cap, tmp_path):
        spill_dir = tmp_path / ".hermes" / "cache" / "terminal-output"
        spill_dir.mkdir(parents=True, exist_ok=True)
        stale = spill_dir / "out-1-2-dead.log"
        stale.write_text("old")
        os.utime(stale, (1, 1))
        json.loads(terminal_tool(
            "python3 -c \"[print('z'*90) for i in range(200)]\"", task_id="t-spill-4"))
        assert not stale.exists()

    @pytest.mark.platforms("linux")
    def test_failed_command_still_gets_spill(self, small_cap):
        r = json.loads(terminal_tool(
            "python3 -c \"[print('e'*90) for i in range(200)]; import sys; sys.exit(3)\"",
            task_id="t-spill-5"))
        assert r["exit_code"] == 3
        assert Path(r["full_output_path"]).exists()


class TestSpillRetention:
    """#126441: spill files are archives the transcript points at ("page it with
    read_file instead of re-running the command"), so their lifetime is the session
    retention window, not a 7-day cache TTL."""

    def test_retention_seconds_reads_session_retention_days(self, monkeypatch):
        import hermes_cli.config
        from tools.environments import base_output as bo
        monkeypatch.setattr(
            hermes_cli.config, "load_config_readonly",
            lambda: {"sessions": {"retention_days": 7}})
        assert bo._spill_retention_seconds() == 7 * 86400

    def test_retention_seconds_defaults_floors_and_falls_back(self, monkeypatch):
        import hermes_cli.config
        from tools.environments import base_output as bo
        monkeypatch.setattr(hermes_cli.config, "load_config_readonly", lambda: {})
        assert bo._spill_retention_seconds() == 90 * 86400

        def _boom():
            raise RuntimeError("config unreadable")

        monkeypatch.setattr(hermes_cli.config, "load_config_readonly", _boom)
        assert bo._spill_retention_seconds() == bo._SPILL_MAX_AGE_S

        # 0 is a legal prune setting (ended sessions go at once) but open sessions
        # survive it; a 0h spill window would delete the file the transcript points
        # at. Floor at 24h, matching the spillover-archive window (#126351).
        monkeypatch.setattr(
            hermes_cli.config, "load_config_readonly",
            lambda: {"sessions": {"retention_days": 0}})
        assert bo._spill_retention_seconds() == 24 * 3600

    def test_prune_keeps_spill_beyond_legacy_7_day_cutoff(self, small_cap, monkeypatch):
        """A 10-day-old spill survives the opportunistic prune: the session that
        references it is still inside its (default 90-day) retention window."""
        import time as _time
        from tools.environments import base_output as bo
        spill_dir = small_cap / ".hermes" / "cache" / "terminal-output"
        spill_dir.mkdir(parents=True, exist_ok=True)
        old = spill_dir / "out-1-2-dead.log"
        old.write_text("old")
        stale = _time.time() - (10 * 86400)
        os.utime(old, (stale, stale))

        monkeypatch.setattr(bo, "_spill_retention_seconds", lambda: 90 * 86400)
        bo._new_output_collector(proc=object(), bounded_capture=True)

        assert old.exists()
