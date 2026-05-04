"""Tests for delegate_tool Phase C — worktree, transcript, coordinator/swarm."""

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.delegate_tool import (
    _resolve_isolation,
    _create_worktree,
    _cleanup_worktree,
    _get_worktree_base_dir,
    _write_transcript_event,
    _get_transcript_dir,
    _apply_coordinator_mode,
    claim_task,
    COORDINATOR_ALLOWED_TOOLSETS,
    COORDINATOR_BLOCKED_TOOLS,
)


# ── Worktree Resolution ─────────────────────────────────────────────────

class TestWorktreeResolution:

    def test_worktree_isolation_now_valid(self):
        """Phase C: worktree isolation is implemented."""
        iso, warns, err = _resolve_isolation(
            requested_isolation="worktree", profile_isolation=None, permission_mode=None,
        )
        assert iso == "worktree"
        assert err is None

    def test_worktree_base_dir_in_hermes(self):
        """Worktree base directory is under ~/.hermes/worktrees/."""
        base = _get_worktree_base_dir()
        assert "hermes" in str(base)
        assert "worktrees" in str(base)


# ── Worktree Creation (unit-level, no actual git) ───────────────────────

class TestWorktreeCreation:

    def test_create_worktree_in_non_git_dir(self):
        """Creating a worktree in a non-git directory returns error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("os.getcwd", return_value=tmpdir):
                with patch.dict(os.environ, {"TERMINAL_CWD": tmpdir}):
                    result = _create_worktree("test-sa-0")
                    assert result.get("error") is not None


# ── Cleanup Safety ──────────────────────────────────────────────────────

class TestWorktreeCleanup:

    def test_cleanup_rejects_path_outside_managed_dir(self):
        """Cleanup refuses to delete paths outside Hermes worktree dir."""
        # Use a path that exists but is outside the managed dir, e.g. /tmp
        result = _cleanup_worktree(
            worktree_path="/tmp",
            original_head="abc123",
        )
        assert result["kept"] is True
        assert "not under Hermes" in result.get("reason", "")

    def test_cleanup_nonexistent_path(self):
        result = _cleanup_worktree(
            worktree_path="/tmp/nonexistent-path-xyz-12345",
            original_head="abc123",
        )
        # Nonexistent path returns kept=False (nothing to keep)
        assert result["removed"] is False
        assert result["kept"] is False
        assert "does not exist" in result.get("reason", "")


# ── Transcript ──────────────────────────────────────────────────────────

class TestTranscript:

    def setup_method(self):
        self._tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir_obj.name)

    def teardown_method(self):
        self._tmpdir_obj.cleanup()

    def test_transcript_dir_under_hermes(self):
        d = _get_transcript_dir()
        assert "hermes" in str(d)
        assert "subagents" in str(d)

    def test_write_transcript_event_creates_file(self):
        with patch("tools.delegate_tool._get_transcript_dir", return_value=self.tmpdir):
            path = _write_transcript_event(
                session_id="test-session",
                subagent_id="sa-test-0",
                event_type="final",
                tool=None,
                preview="Task completed successfully",
                usage={"input_tokens": 100, "output_tokens": 50, "api_calls": 2},
                agent_id="kimi",
            )
            assert path is not None
            assert path.exists()

            # Verify JSONL content
            lines = path.read_text().strip().split("\n")
            assert len(lines) == 1
            entry = json.loads(lines[0])
            assert entry["event_type"] == "final"
            assert entry["subagent_id"] == "sa-test-0"
            assert entry["agent_id"] == "kimi"


# ── Coordinator Mode ────────────────────────────────────────────────────

class TestCoordinatorMode:

    def test_coordinator_mode_restricts_toolsets(self):
        agent = {"coordinator_mode": True}
        toolsets, blocked, warnings = _apply_coordinator_mode(
            agent,
            toolsets=["file", "terminal", "delegation", "web"],
            blocked_tools=set(),
            warnings=[],
        )
        assert "terminal" not in toolsets
        assert "web" not in toolsets
        assert "file" in toolsets
        assert "delegation" in toolsets
        assert len(warnings) >= 1
        assert "coordinator_mode" in warnings[0]

    def test_no_coordinator_mode_no_restriction(self):
        agent = {"coordinator_mode": False}
        toolsets, blocked, warnings = _apply_coordinator_mode(
            agent,
            toolsets=["file", "terminal", "delegation", "web"],
            blocked_tools={"send_message"},
            warnings=[],
        )
        assert "terminal" in toolsets
        assert "web" in toolsets
        assert blocked == {"send_message"}  # unchanged

    def test_coordinator_mode_blocks_write_tools(self):
        agent = {"coordinator_mode": True}
        _, blocked, _ = _apply_coordinator_mode(
            agent,
            toolsets=["file"],
            blocked_tools=set(),
            warnings=[],
        )
        assert "write_file" in blocked
        assert "patch" in blocked
        assert "terminal" in blocked

    def test_coordinator_blocked_tools_constant(self):
        assert "terminal" in COORDINATOR_BLOCKED_TOOLS
        assert "write_file" in COORDINATOR_BLOCKED_TOOLS
        assert "send_message" in COORDINATOR_BLOCKED_TOOLS


# ── Claim Task ──────────────────────────────────────────────────────────

class TestClaimTask:

    def test_claim_task_succeeds(self):
        result = claim_task("test-task-1", "agent-kimi")
        assert result is True

    def test_claim_task_second_claim_fails(self):
        """Second claim for the same task_id should fail."""
        task_id = f"test-task-{time.time()}"
        first = claim_task(task_id, "agent-claude")
        assert first is True
        second = claim_task(task_id, "agent-kimi")
        assert second is False

    def test_claim_task_different_tasks_both_succeed(self):
        first = claim_task("test-task-a", "agent-claude")
        second = claim_task("test-task-b", "agent-kimi")
        assert first is True
        assert second is True

    def test_claim_task_concurrent(self):
        """Concurrent claims: only one succeeds."""
        task_id = f"test-concurrent-{time.time()}"
        results = {"claimed": 0, "rejected": 0}
        lock = threading.Lock()

        def try_claim(agent):
            if claim_task(task_id, agent):
                with lock:
                    results["claimed"] += 1
            else:
                with lock:
                    results["rejected"] += 1

        threads = [
            threading.Thread(target=try_claim, args=(f"agent-{i}",))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results["claimed"] == 1
        assert results["rejected"] == 4
