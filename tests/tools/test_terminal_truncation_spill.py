"""Tests for terminal truncation spill + metadata (deferred retrieval)."""

import json
import os
import stat
from pathlib import Path

import pytest

from tools.terminal_tool import terminal_tool
import tools.terminal_tool as terminal_tool_module


@pytest.fixture
def small_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_constants import hermes_home_key
    import tools.tool_output_limits as lim
    monkeypatch.setattr(lim, "_cached_limits", {hermes_home_key(): {
        "max_bytes": 2000, "max_lines": 2000, "max_line_length": 2000,
    }})
    return tmp_path


@pytest.fixture
def deterministic_spill_env(monkeypatch):
    from tools.environments.base_output import _finalize_wait_result, _new_output_collector

    class DeterministicOutputEnv:
        env = {}
        cwd = "/workspace"

        def execute(self, command, **kwargs):
            collector = _new_output_collector(self, bounded_capture=kwargs["bounded_capture"])
            output = (
                "marker_head\n"
                + "".join(f"row_{i} {'x' * 80}\n" for i in range(200))
                + "marker_tail\n"
                if command == "produce"
                else "next\n"
            )
            collector.append(output)
            return _finalize_wait_result(collector, collector.render(), 0)

    monkeypatch.setattr(terminal_tool_module, "_active_environments", {"default": DeterministicOutputEnv()})
    monkeypatch.setattr(terminal_tool_module, "_last_activity", {})
    monkeypatch.setattr(terminal_tool_module, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_tool_module, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool_module, "_get_env_config", lambda: {
        "env_type": "local", "cwd": "/workspace", "timeout": 60, "lifetime_seconds": 3600,
    })
    monkeypatch.setattr(
        terminal_tool_module, "_check_all_guards",
        lambda command, env_type, **kwargs: {"approved": True},
    )


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

    def test_small_output_has_no_metadata(self, small_cap, deterministic_spill_env):
        r = json.loads(terminal_tool("next", task_id="t-spill-2"))
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

    @pytest.mark.platforms("linux")
    def test_failed_command_still_gets_spill(self, small_cap):
        r = json.loads(terminal_tool(
            "python3 -c \"[print('e'*90) for i in range(200)]; import sys; sys.exit(3)\"",
            task_id="t-spill-5"))
        assert r["exit_code"] == 3
        assert Path(r["full_output_path"]).exists()

    @pytest.mark.platforms("posix")
    def test_persisted_spill_follows_owning_session_lifetime(self, small_cap, deterministic_spill_env):
        """An old spill stays readable while its transcript exists, then leaves with it."""
        from hermes_state import SessionDB

        home = small_cap / ".hermes"
        db = SessionDB(home / "state.db")
        try:
            db.create_session("spill-owner", source="cli")
            raw_result = terminal_tool(
                "produce",
                task_id="t-spill-owned",
                session_id="spill-owner",
            )
            result = json.loads(raw_result)
            spill = Path(result["full_output_path"])
            spill_dir = home / "cache" / "terminal-output"
            assert spill.parent.resolve() == spill_dir.resolve()
            assert stat.S_IMODE(spill_dir.stat().st_mode) == 0o700
            assert stat.S_IMODE(spill.stat().st_mode) == 0o600

            db.append_message(
                "spill-owner", role="tool", content=raw_result, tool_name="terminal",
            )
            os.utime(spill, (1, 1))

            # The old implementation pruned every >7-day spill on the next command.
            terminal_tool(
                "next",
                task_id="t-spill-next",
                session_id="spill-owner",
            )
            assert spill.exists()

            assert db.delete_session("spill-owner") is True
            assert not spill.exists()
        finally:
            db.close()

    @pytest.mark.platforms("posix")
    def test_shared_spill_survives_until_final_owner_is_pruned(
        self, small_cap, deterministic_spill_env,
    ):
        """One session cannot unlink an artifact still referenced by another session."""
        from hermes_state import SessionDB

        home = small_cap / ".hermes"
        db = SessionDB(home / "state.db")
        try:
            db.create_session("spill-owner-a", source="owner-a")
            db.create_session("spill-owner-b", source="owner-b")
            raw_result = terminal_tool(
                "produce", task_id="t-spill-shared", session_id="spill-owner-a",
            )
            spill = Path(json.loads(raw_result)["full_output_path"])
            for session_id in ("spill-owner-a", "spill-owner-b"):
                db.append_message(
                    session_id, role="tool", content=raw_result, tool_name="terminal",
                )
                db.end_session(session_id, "completed")

            assert db.delete_session("spill-owner-a") is True
            assert spill.exists()

            assert db.prune_sessions(older_than_days=None, source="owner-b") == 1
            assert not spill.exists()
        finally:
            db.close()
