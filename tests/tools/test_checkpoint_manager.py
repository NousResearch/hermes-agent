"""Tests for tools/checkpoint_manager.py — CheckpointManager."""

import logging
import os
import json
import shutil
import subprocess
import pytest
from pathlib import Path
from unittest.mock import patch

from tools.checkpoint_manager import (
    CheckpointManager,
    _shadow_repo_path,
    _init_shadow_repo,
    _run_git,
    _git_env,
    _dir_file_count,
    format_checkpoint_list,
    DEFAULT_EXCLUDES,
    CHECKPOINT_BASE,
)


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture()
def work_dir(tmp_path):
    """Temporary working directory."""
    d = tmp_path / "project"
    d.mkdir()
    (d / "main.py").write_text("print('hello')\\n")
    (d / "README.md").write_text("# Project\\n")
    return d


@pytest.fixture()
def checkpoint_base(tmp_path):
    """Isolated checkpoint base — never writes to ~/.hermes/."""
    return tmp_path / "checkpoints"


@pytest.fixture()
def mgr(work_dir, checkpoint_base, monkeypatch):
    """CheckpointManager with redirected checkpoint base."""
    monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", checkpoint_base)
    return CheckpointManager(enabled=True, max_snapshots=50)


@pytest.fixture()
def disabled_mgr(checkpoint_base, monkeypatch):
    """Disabled CheckpointManager."""
    monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", checkpoint_base)
    return CheckpointManager(enabled=False)


# =========================================================================
# Shadow repo path
# =========================================================================

class TestShadowRepoPath:
    def test_deterministic(self, work_dir, checkpoint_base, monkeypatch):
        monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", checkpoint_base)
        p1 = _shadow_repo_path(str(work_dir))
        p2 = _shadow_repo_path(str(work_dir))
        assert p1 == p2

    def test_different_dirs_different_paths(self, tmp_path, checkpoint_base, monkeypatch):
        monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", checkpoint_base)
        p1 = _shadow_repo_path(str(tmp_path / "a"))
        p2 = _shadow_repo_path(str(tmp_path / "b"))
        assert p1 != p2

    def test_under_checkpoint_base(self, work_dir, checkpoint_base, monkeypatch):
        monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", checkpoint_base)
        p = _shadow_repo_path(str(work_dir))
        assert str(p).startswith(str(checkpoint_base))


# =========================================================================
# Shadow repo init
# =========================================================================

class TestShadowRepoInit:
    def test_creates_git_repo(self, work_dir, checkpoint_base, monkeypatch):
        monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", checkpoint_base)
        shadow = _shadow_repo_path(str(work_dir))
        err = _init_shadow_repo(shadow, str(work_dir))
        assert err is None
        assert (shadow / "HEAD").exists()

    def test_no_git_in_project_dir(self, work_dir, checkpoint_base, monkeypatch):
        monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", checkpoint_base)
        shadow = _shadow_repo_path(str(work_dir))
        _init_shadow_repo(shadow, str(work_dir))
        assert not (work_dir / ".git").exists()

    def test_has_exclude_file(self, work_dir, checkpoint_base, monkeypatch):
        monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", checkpoint_base)
        shadow = _shadow_repo_path(str(work_dir))
        _init_shadow_repo(shadow, str(work_dir))
        exclude = shadow / "info" / "exclude"
        assert exclude.exists()
        content = exclude.read_text()
        assert "node_modules/" in content
        assert ".env" in content

    def test_has_workdir_file(self, work_dir, checkpoint_base, monkeypatch):
        monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", checkpoint_base)
        shadow = _shadow_repo_path(str(work_dir))
        _init_shadow_repo(shadow, str(work_dir))
        workdir_file = shadow / "HERMES_WORKDIR"
        assert workdir_file.exists()
        assert str(work_dir.resolve()) in workdir_file.read_text()

    def test_idempotent(self, work_dir, checkpoint_base, monkeypatch):
        monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", checkpoint_base)
        shadow = _shadow_repo_path(str(work_dir))
        err1 = _init_shadow_repo(shadow, str(work_dir))
        err2 = _init_shadow_repo(shadow, str(work_dir))
        assert err1 is None
        assert err2 is None


# =========================================================================
# CheckpointManager — disabled
# =========================================================================

class TestDisabledManager:
    def test_ensure_checkpoint_returns_false(self, disabled_mgr, work_dir):
        assert disabled_mgr.ensure_checkpoint(str(work_dir)) is False

    def test_new_turn_works(self, disabled_mgr):
        disabled_mgr.new_turn()  # should not raise


# =========================================================================
# CheckpointManager — taking checkpoints
# =========================================================================

class TestTakeCheckpoint:
    def test_first_checkpoint(self, mgr, work_dir):
        result = mgr.ensure_checkpoint(str(work_dir), "initial")
        assert result is True

    def test_successful_checkpoint_does_not_log_expected_diff_exit(self, mgr, work_dir, caplog):
        with caplog.at_level(logging.ERROR, logger="tools.checkpoint_manager"):
            result = mgr.ensure_checkpoint(str(work_dir), "initial")
        assert result is True
        assert not any("diff --cached --quiet" in r.getMessage() for r in caplog.records)

    def test_dedup_same_turn(self, mgr, work_dir):
        r1 = mgr.ensure_checkpoint(str(work_dir), "first")
        r2 = mgr.ensure_checkpoint(str(work_dir), "second")
        assert r1 is True
        assert r2 is False  # dedup'd

    def test_new_turn_resets_dedup(self, mgr, work_dir):
        r1 = mgr.ensure_checkpoint(str(work_dir), "turn 1")
        assert r1 is True

        mgr.new_turn()

        # Modify a file so there's something to commit
        (work_dir / "main.py").write_text("print('modified')\\n")
        r2 = mgr.ensure_checkpoint(str(work_dir), "turn 2")
        assert r2 is True

    def test_no_changes_skips_commit(self, mgr, work_dir):
        # First checkpoint
        mgr.ensure_checkpoint(str(work_dir), "initial")
        mgr.new_turn()

        # No file changes — should return False (nothing to commit)
        r = mgr.ensure_checkpoint(str(work_dir), "no changes")
        assert r is False

    def test_skip_root_dir(self, mgr):
        r = mgr.ensure_checkpoint("/", "root")
        assert r is False

    def test_skip_home_dir(self, mgr):
        r = mgr.ensure_checkpoint(str(Path.home()), "home")
        assert r is False


# =========================================================================
# CheckpointManager — listing checkpoints
# =========================================================================

class TestListCheckpoints:
    def test_empty_when_no_checkpoints(self, mgr, work_dir):
        result = mgr.list_checkpoints(str(work_dir))
        assert result == []

    def test_list_after_take(self, mgr, work_dir):
        mgr.ensure_checkpoint(str(work_dir), "test checkpoint")
        result = mgr.list_checkpoints(str(work_dir))
        assert len(result) == 1
        assert result[0]["reason"] == "test checkpoint"
        assert "hash" in result[0]
        assert "short_hash" in result[0]
        assert "timestamp" in result[0]

    def test_multiple_checkpoints_ordered(self, mgr, work_dir):
        mgr.ensure_checkpoint(str(work_dir), "first")
        mgr.new_turn()

        (work_dir / "main.py").write_text("v2\\n")
        mgr.ensure_checkpoint(str(work_dir), "second")
        mgr.new_turn()

        (work_dir / "main.py").write_text("v3\\n")
        mgr.ensure_checkpoint(str(work_dir), "third")

        result = mgr.list_checkpoints(str(work_dir))
        assert len(result) == 3
        # Most recent first
        assert result[0]["reason"] == "third"
        assert result[2]["reason"] == "first"


# =========================================================================
# CheckpointManager — restoring
# =========================================================================

class TestRestore:
    def test_restore_to_previous(self, mgr, work_dir):
        # Write original content
        (work_dir / "main.py").write_text("original\\n")
        mgr.ensure_checkpoint(str(work_dir), "original state")
        mgr.new_turn()

        # Modify the file
        (work_dir / "main.py").write_text("modified\\n")

        # Get the checkpoint hash
        checkpoints = mgr.list_checkpoints(str(work_dir))
        assert len(checkpoints) == 1

        # Restore
        result = mgr.restore(str(work_dir), checkpoints[0]["hash"])
        assert result["success"] is True

        # File should be back to original
        assert (work_dir / "main.py").read_text() == "original\\n"

    def test_restore_invalid_hash(self, mgr, work_dir):
        mgr.ensure_checkpoint(str(work_dir), "initial")
        result = mgr.restore(str(work_dir), "deadbeef1234")
        assert result["success"] is False

    def test_restore_no_checkpoints(self, mgr, work_dir):
        result = mgr.restore(str(work_dir), "abc123")
        assert result["success"] is False

    def test_restore_creates_pre_rollback_snapshot(self, mgr, work_dir):
        (work_dir / "main.py").write_text("v1\\n")
        mgr.ensure_checkpoint(str(work_dir), "v1")
        mgr.new_turn()

        (work_dir / "main.py").write_text("v2\\n")

        checkpoints = mgr.list_checkpoints(str(work_dir))
        mgr.restore(str(work_dir), checkpoints[0]["hash"])

        # Should now have 2 checkpoints: original + pre-rollback
        all_cps = mgr.list_checkpoints(str(work_dir))
        assert len(all_cps) >= 2
        assert "pre-rollback" in all_cps[0]["reason"]


# =========================================================================
# CheckpointManager — working dir resolution
# =========================================================================

class TestWorkingDirResolution:
    def test_resolves_git_project_root(self, tmp_path):
        mgr = CheckpointManager(enabled=True)
        project = tmp_path / "myproject"
        project.mkdir()
        (project / ".git").mkdir()
        subdir = project / "src"
        subdir.mkdir()
        filepath = subdir / "main.py"
        filepath.write_text("x\\n")

        result = mgr.get_working_dir_for_path(str(filepath))
        assert result == str(project)

    def test_resolves_pyproject_root(self, tmp_path):
        mgr = CheckpointManager(enabled=True)
        project = tmp_path / "pyproj"
        project.mkdir()
        (project / "pyproject.toml").write_text("[project]\\n")
        subdir = project / "src"
        subdir.mkdir()

        result = mgr.get_working_dir_for_path(str(subdir / "file.py"))
        assert result == str(project)

    def test_falls_back_to_parent(self, tmp_path):
        mgr = CheckpointManager(enabled=True)
        filepath = tmp_path / "random" / "file.py"
        filepath.parent.mkdir(parents=True)
        filepath.write_text("x\\n")

        result = mgr.get_working_dir_for_path(str(filepath))
        assert result == str(filepath.parent)


# =========================================================================
# Git env isolation
# =========================================================================

class TestGitEnvIsolation:
    def test_sets_git_dir(self, tmp_path):
        shadow = tmp_path / "shadow"
        env = _git_env(shadow, str(tmp_path / "work"))
        assert env["GIT_DIR"] == str(shadow)

    def test_sets_work_tree(self, tmp_path):
        shadow = tmp_path / "shadow"
        work = tmp_path / "work"
        env = _git_env(shadow, str(work))
        assert env["GIT_WORK_TREE"] == str(work.resolve())

    def test_clears_index_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GIT_INDEX_FILE", "/some/index")
        shadow = tmp_path / "shadow"
        env = _git_env(shadow, str(tmp_path))
        assert "GIT_INDEX_FILE" not in env


# =========================================================================
# format_checkpoint_list
# =========================================================================

class TestFormatCheckpointList:
    def test_empty_list(self):
        result = format_checkpoint_list([], "/some/dir")
        assert "No checkpoints" in result

    def test_formats_entries(self):
        cps = [
            {"hash": "abc123", "short_hash": "abc1", "timestamp": "2026-03-09T21:15:00-07:00", "reason": "before write_file"},
            {"hash": "def456", "short_hash": "def4", "timestamp": "2026-03-09T21:10:00-07:00", "reason": "before patch"},
        ]
        result = format_checkpoint_list(cps, "/home/user/project")
        assert "abc1" in result
        assert "def4" in result
        assert "before write_file" in result
        assert "/rollback" in result


# =========================================================================
# File count guard
# =========================================================================

class TestDirFileCount:
    def test_counts_files(self, work_dir):
        count = _dir_file_count(str(work_dir))
        assert count >= 2  # main.py + README.md

    def test_nonexistent_dir(self, tmp_path):
        count = _dir_file_count(str(tmp_path / "nonexistent"))
        assert count == 0


# =========================================================================
# Error resilience
# =========================================================================

class TestErrorResilience:
    def test_no_git_installed(self, work_dir, checkpoint_base, monkeypatch):
        monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", checkpoint_base)
        mgr = CheckpointManager(enabled=True)
        # Mock git not found
        monkeypatch.setattr("shutil.which", lambda x: None)
        mgr._git_available = None  # reset lazy probe
        result = mgr.ensure_checkpoint(str(work_dir), "test")
        assert result is False

    def test_run_git_allows_expected_nonzero_without_error_log(self, tmp_path, caplog):
        completed = subprocess.CompletedProcess(
            args=["git", "diff", "--cached", "--quiet"],
            returncode=1,
            stdout="",
            stderr="",
        )
        with patch("tools.checkpoint_manager.subprocess.run", return_value=completed):
            with caplog.at_level(logging.ERROR, logger="tools.checkpoint_manager"):
                ok, stdout, stderr = _run_git(
                    ["diff", "--cached", "--quiet"],
                    tmp_path / "shadow",
                    str(tmp_path / "work"),
                    allowed_returncodes={1},
                )
        assert ok is False
        assert stdout == ""
        assert stderr == ""
        assert not caplog.records

    def test_checkpoint_failure_does_not_raise(self, mgr, work_dir, monkeypatch):
        """Checkpoint failures should never raise — they're silently logged."""
        def broken_run_git(*args, **kwargs):
            raise OSError("git exploded")
        monkeypatch.setattr("tools.checkpoint_manager._run_git", broken_run_git)
        # Should not raise
        result = mgr.ensure_checkpoint(str(work_dir), "test")
        assert result is False


# =========================================================================
# Security / Input validation
# =========================================================================

class TestSecurity:
    def test_restore_rejects_argument_injection(self, mgr, work_dir):
        mgr.ensure_checkpoint(str(work_dir), "initial")
        # Try to pass a git flag as a commit hash
        result = mgr.restore(str(work_dir), "--patch")
        assert result["success"] is False
        assert "Invalid commit hash" in result["error"]
        assert "must not start with '-'" in result["error"]
        
        result = mgr.restore(str(work_dir), "-p")
        assert result["success"] is False
        assert "Invalid commit hash" in result["error"]
        
    def test_restore_rejects_invalid_hex_chars(self, mgr, work_dir):
        mgr.ensure_checkpoint(str(work_dir), "initial")
        # Git hashes should not contain characters like ;, &, |
        result = mgr.restore(str(work_dir), "abc; rm -rf /")
        assert result["success"] is False
        assert "expected 4-64 hex characters" in result["error"]
        
        result = mgr.diff(str(work_dir), "abc&def")
        assert result["success"] is False
        assert "expected 4-64 hex characters" in result["error"]

    def test_restore_rejects_path_traversal(self, mgr, work_dir):
        mgr.ensure_checkpoint(str(work_dir), "initial")
        # Real commit hash but malicious path
        checkpoints = mgr.list_checkpoints(str(work_dir))
        target_hash = checkpoints[0]["hash"]
        
        # Absolute path outside
        result = mgr.restore(str(work_dir), target_hash, file_path="/etc/passwd")
        assert result["success"] is False
        assert "got absolute path" in result["error"]
        
        # Relative traversal outside path
        result = mgr.restore(str(work_dir), target_hash, file_path="../outside_file.txt")
        assert result["success"] is False
        assert "escapes the working directory" in result["error"]

    def test_restore_accepts_valid_file_path(self, mgr, work_dir):
        mgr.ensure_checkpoint(str(work_dir), "initial")
        checkpoints = mgr.list_checkpoints(str(work_dir))
        target_hash = checkpoints[0]["hash"]
        
        # Valid path inside directory
        result = mgr.restore(str(work_dir), target_hash, file_path="main.py")
        assert result["success"] is True
        
        # Another valid path with subdirectories
        (work_dir / "subdir").mkdir()
        (work_dir / "subdir" / "test.txt").write_text("hello")
        mgr.new_turn()
        mgr.ensure_checkpoint(str(work_dir), "second")
        checkpoints = mgr.list_checkpoints(str(work_dir))
        target_hash = checkpoints[0]["hash"]
        
        result = mgr.restore(str(work_dir), target_hash, file_path="subdir/test.txt")
        assert result["success"] is True
