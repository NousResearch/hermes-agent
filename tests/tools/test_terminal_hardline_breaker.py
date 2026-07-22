"""Regression tests for the terminal-tool hardline-block circuit breaker (#69256).

A hardline block is deterministic — the identical command is blocked on every
retry, regardless of --yolo / approvals.mode=off / cron approve mode.  Without
a breaker, a weak tool-follower can re-emit the identical blocked call dozens
of times in a row, burning the turn budget.  These tests verify the breaker:

1. Warns (appends guidance) on the 3rd consecutive identical hardline block.
2. Hard-stops with a "do not retry" message on the 4th+.
3. A *different* command resets the streak.
4. A command that later executes (approved) resets the streak.
5. Non-hardline blocks (pending_approval / plain dangerous) do NOT trip it.
6. Different task_ids are tracked independently.

Run with:  python -m pytest tests/tools/test_terminal_hardline_breaker.py -v
"""

import json
from unittest.mock import patch, MagicMock

import pytest

from tools.terminal_tool import (
    terminal_tool,
    _record_hardline_block,
    _reset_hardline_block_tracker,
    _hardline_block_tracker,
    _HARDBREAK_WARN_AT,
    _HARDBREAK_BLOCK_AT,
)


# ---------------------------------------------------------------------------
# Shared test config dict — mirrors _get_env_config() return shape.
# ---------------------------------------------------------------------------
def _make_env_config(**overrides):
    config = {
        "env_type": "local",
        "timeout": 180,
        "cwd": "/tmp",
        "host_cwd": None,
        "modal_mode": "auto",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
    }
    config.update(overrides)
    return config


def _hardline_approval(desc="recursive delete of root filesystem"):
    """A _check_all_guards()-shaped hardline-block result."""
    return {
        "approved": False,
        "hardline": True,
        "description": desc,
        "message": (
            f"BLOCKED (hardline): {desc}. This command is on the unconditional "
            "blocklist and cannot be executed via the agent."
        ),
    }


@pytest.fixture(autouse=True)
def _clear_tracker():
    """Reset the global tracker before each test so they're independent."""
    _hardline_block_tracker.clear()
    yield
    _hardline_block_tracker.clear()


def _run_with_hardline_block(command, task_id=None):
    """Invoke terminal_tool with _check_all_guards hardline-blocking everything."""
    with patch("tools.terminal_tool._get_env_config",
               return_value=_make_env_config()), \
         patch("tools.terminal_tool._start_cleanup_thread"), \
         patch("tools.terminal_tool._check_all_guards",
               return_value=_hardline_approval()):
        return json.loads(terminal_tool(command=command, task_id=task_id))


class TestHardlineBreakerUnit:
    """Direct tests of the _record_hardline_block / _reset helpers."""

    def test_thresholds_match_file_tools(self):
        assert _HARDBREAK_WARN_AT == 3
        assert _HARDBREAK_BLOCK_AT == 4

    def test_consecutive_identical_increments(self):
        assert _record_hardline_block("t1", "rm -rf /") == 1
        assert _record_hardline_block("t1", "rm -rf /") == 2
        assert _record_hardline_block("t1", "rm -rf /") == 3
        assert _record_hardline_block("t1", "rm -rf /") == 4

    def test_different_command_resets_streak(self):
        _record_hardline_block("t1", "rm -rf /")
        _record_hardline_block("t1", "rm -rf /")
        # A different command resets to 1.
        assert _record_hardline_block("t1", "rm -rf /home") == 1
        # Going back to the original also starts fresh.
        assert _record_hardline_block("t1", "rm -rf /") == 1

    def test_reset_clears_streak(self):
        _record_hardline_block("t1", "rm -rf /")
        _record_hardline_block("t1", "rm -rf /")
        _reset_hardline_block_tracker("t1")
        assert _record_hardline_block("t1", "rm -rf /") == 1

    def test_tasks_are_independent(self):
        assert _record_hardline_block("t1", "rm -rf /") == 1
        assert _record_hardline_block("t2", "rm -rf /") == 1
        assert _record_hardline_block("t1", "rm -rf /") == 2
        assert _record_hardline_block("t2", "rm -rf /") == 2

    def test_default_task_id(self):
        assert _record_hardline_block(None, "rm -rf /") == 1
        assert _record_hardline_block(None, "rm -rf /") == 2
        _reset_hardline_block_tracker(None)
        assert _record_hardline_block(None, "rm -rf /") == 1


class TestHardlineBreakerIntegration:
    """End-to-end through terminal_tool with _check_all_guards mocked."""

    def test_first_two_blocks_return_normal_hardline_message(self):
        r1 = _run_with_hardline_block("rm -rf /", task_id="t1")
        r2 = _run_with_hardline_block("rm -rf /", task_id="t1")
        for r in (r1, r2):
            assert r["status"] == "blocked"
            assert r["exit_code"] == -1
            assert "BLOCKED (hardline)" in r["error"]
            # No circuit-breaker escalation yet.
            assert "blocked_repeat_count" not in r
            assert "in a row" not in r["error"]

    def test_third_block_appends_warning(self):
        _run_with_hardline_block("rm -rf /", task_id="t1")
        _run_with_hardline_block("rm -rf /", task_id="t1")
        r3 = _run_with_hardline_block("rm -rf /", task_id="t1")
        assert r3["status"] == "blocked"
        assert "BLOCKED (hardline)" in r3["error"]
        # Warning escalation appended.
        assert "in a row" in r3["error"]
        assert "blocked_repeat_count" not in r3

    def test_fourth_block_hard_stops_with_do_not_retry(self):
        for _ in range(3):
            _run_with_hardline_block("rm -rf /", task_id="t1")
        r4 = _run_with_hardline_block("rm -rf /", task_id="t1")
        assert r4["status"] == "blocked"
        assert r4["exit_code"] == -1
        assert r4.get("hardline") is True
        assert r4["blocked_repeat_count"] == 4
        assert "STOP retrying" in r4["error"]
        assert "NEVER" in r4["error"]
        # The original hardline message is superseded by the breaker message.
        assert "BLOCKED (hardline)" not in r4["error"]

    def test_fifth_block_continues_hard_stop(self):
        for _ in range(4):
            _run_with_hardline_block("rm -rf /", task_id="t1")
        r5 = _run_with_hardline_block("rm -rf /", task_id="t1")
        assert r5["blocked_repeat_count"] == 5
        assert "STOP retrying" in r5["error"]

    def test_different_command_resets_streak_in_tool(self):
        # Build a streak of 3 on one command.
        for _ in range(3):
            _run_with_hardline_block("rm -rf /", task_id="t1")
        # A different hardline-blocked command starts fresh — no hard-stop.
        r = _run_with_hardline_block("rm -rf /home", task_id="t1")
        assert r["status"] == "blocked"
        assert "blocked_repeat_count" not in r
        # And the original command now also starts fresh.
        r_orig = _run_with_hardline_block("rm -rf /", task_id="t1")
        assert "blocked_repeat_count" not in r_orig

    def test_approved_command_resets_streak(self):
        """Once a command executes, the streak clears — a later retry is fresh."""
        # Two hardline blocks on command A.
        _run_with_hardline_block("rm -rf /", task_id="t1")
        _run_with_hardline_block("rm -rf /", task_id="t1")
        # Now run a *different* command B that gets approved and executes,
        # which should reset the tracker.
        mock_env = MagicMock()
        mock_env.execute.return_value = {"output": "done", "returncode": 0}
        with patch("tools.terminal_tool._get_env_config",
                   return_value=_make_env_config()), \
             patch("tools.terminal_tool._start_cleanup_thread"), \
             patch("tools.terminal_tool._active_environments", {"default": mock_env}), \
             patch("tools.terminal_tool._last_activity", {"default": 0}), \
             patch("tools.terminal_tool._check_all_guards",
                   return_value={"approved": True}):
            r = json.loads(terminal_tool(command="echo hello", task_id="t1"))
        assert "error" not in r or r["error"] is None
        # The streak for A is cleared — the next hardline block on A is count 1.
        r_a = _run_with_hardline_block("rm -rf /", task_id="t1")
        assert "blocked_repeat_count" not in r_a
        assert "in a row" not in r_a["error"]

    def test_non_hardline_block_does_not_trip_breaker(self):
        """A plain dangerous (approvable) block must not increment the streak."""
        plain_block = {
            "approved": False,
            "description": "recursive delete",
            "message": "BLOCKED: recursive delete. Use the approval prompt to allow it.",
        }
        with patch("tools.terminal_tool._get_env_config",
                   return_value=_make_env_config()), \
             patch("tools.terminal_tool._start_cleanup_thread"), \
             patch("tools.terminal_tool._check_all_guards",
                   return_value=plain_block):
            for _ in range(6):
                r = json.loads(terminal_tool(command="rm -rf build", task_id="t1"))
        # Every call returns the plain block — never escalates.
        assert r["status"] == "blocked"
        assert "blocked_repeat_count" not in r
        assert "in a row" not in r["error"]

    def test_pending_approval_does_not_trip_breaker(self):
        """pending_approval returns early and must not touch the streak."""
        pending = {
            "approved": False,
            "status": "pending_approval",
            "command": "rm -rf build",
            "description": "command flagged",
            "pattern_key": "",
            "smart_denied": False,
            "allow_permanent": True,
        }
        with patch("tools.terminal_tool._get_env_config",
                   return_value=_make_env_config()), \
             patch("tools.terminal_tool._start_cleanup_thread"), \
             patch("tools.terminal_tool._check_all_guards",
                   return_value=pending):
            r = json.loads(terminal_tool(command="rm -rf build", task_id="t1"))
        assert r["status"] == "pending_approval"
        # Tracker untouched.
        assert _hardline_block_tracker.get("t1") is None
