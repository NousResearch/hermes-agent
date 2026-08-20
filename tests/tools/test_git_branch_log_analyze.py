"""Tests for git_branch and log_analyze tools."""

import json
import os
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# git_branch tests
# ---------------------------------------------------------------------------


class TestGitBranchToolsetNotInCore:
    """git_branch must not be in _HERMES_CORE_TOOLS — it's opt-in via the git toolset."""

    def test_git_branch_not_in_core_tools(self):
        from toolsets import _HERMES_CORE_TOOLS
        assert "git_branch" not in _HERMES_CORE_TOOLS

    def test_git_branch_in_git_toolset(self):
        from toolsets import TOOLSETS
        assert "git" in TOOLSETS
        assert "git_branch" in TOOLSETS["git"]["tools"]

    def test_log_analyze_not_in_core_tools(self):
        from toolsets import _HERMES_CORE_TOOLS
        assert "log_analyze" not in _HERMES_CORE_TOOLS

    def test_log_analyze_in_monitoring_toolset(self):
        from toolsets import TOOLSETS
        assert "monitoring" in TOOLSETS
        assert "log_analyze" in TOOLSETS["monitoring"]["tools"]


class TestGitBranchDangerousDelete:
    """git branch -D must be blocked by the dangerous-command detector."""

    def test_force_delete_blocked(self):
        from tools.git_branch import git_branch
        result = json.loads(git_branch(operation="delete", branch_name="feature", force=True))
        assert result["success"] is False
        assert result.get("dangerous") is True
        assert "Blocked" in result["error"]

    def test_non_force_delete_allowed(self):
        """git branch -d (non-force) is safe — should not be blocked."""
        from tools.git_branch import git_branch
        # This will fail because no git repo, but it shouldn't be blocked by approval
        result = json.loads(git_branch(operation="delete", branch_name="feature", force=False))
        # Should NOT have dangerous=True — the failure is from git, not approval
        assert result.get("dangerous") is not True


class TestGitBranchOperations:
    """Basic git_branch operation tests (mocked git)."""

    def test_list_branches(self):
        from tools.git_branch import _run_git
        with patch("tools.git_branch.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="* main\n  feature\n  remotes/origin/main",
                stderr="",
            )
            result = _run_git(["branch", "-a"], "/tmp")
            assert result["success"] is True
            assert "main" in result["stdout"]

    def test_unknown_operation(self):
        from tools.git_branch import git_branch
        result = json.loads(git_branch(operation="bogus"))
        assert result["success"] is False
        assert "Unknown operation" in result["error"]

    def test_create_requires_branch_name(self):
        from tools.git_branch import git_branch
        result = json.loads(git_branch(operation="create"))
        assert result["success"] is False
        assert "branch_name required" in result["error"]

    def test_delete_requires_branch_name(self):
        from tools.git_branch import git_branch
        result = json.loads(git_branch(operation="delete"))
        assert result["success"] is False
        assert "branch_name required" in result["error"]

    def test_switch_requires_branch_name(self):
        from tools.git_branch import git_branch
        result = json.loads(git_branch(operation="switch"))
        assert result["success"] is False
        assert "branch_name required" in result["error"]


# ---------------------------------------------------------------------------
# log_analyze tests
# ---------------------------------------------------------------------------


class TestLogAnalyze:
    """log_analyze tool tests."""

    def test_file_not_found(self):
        from tools.log_analyze import log_analyze
        result = json.loads(log_analyze(file_path="/nonexistent/file.log"))
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_not_a_file(self):
        from tools.log_analyze import log_analyze
        with tempfile.TemporaryDirectory() as tmpdir:
            result = json.loads(log_analyze(file_path=tmpdir))
            assert result["success"] is False
            assert "not a file" in result["error"].lower()

    def test_analyze_log_file(self):
        from tools.log_analyze import log_analyze
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("2024-01-01 INFO Starting app\n")
            f.write("2024-01-01 ERROR Something failed\n")
            f.write("2024-01-01 INFO Done\n")
            tmpfile = f.name
        try:
            result = json.loads(log_analyze(file_path=tmpfile))
            assert result["success"] is True
            assert result["total_lines"] == 3
        finally:
            os.unlink(tmpfile)

    def test_filter_by_level(self):
        from tools.log_analyze import log_analyze
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("2024-01-01 INFO Starting app\n")
            f.write("2024-01-01 ERROR Something failed\n")
            f.write("2024-01-01 INFO Done\n")
            tmpfile = f.name
        try:
            result = json.loads(log_analyze(file_path=tmpfile, level="ERROR"))
            assert result["success"] is True
            assert result["filtered_count"] == 1
        finally:
            os.unlink(tmpfile)

    def test_search_pattern(self):
        from tools.log_analyze import log_analyze
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("2024-01-01 INFO Starting app\n")
            f.write("2024-01-01 ERROR DB connection failed\n")
            f.write("2024-01-01 INFO Done\n")
            tmpfile = f.name
        try:
            result = json.loads(log_analyze(file_path=tmpfile, search_pattern="DB"))
            assert result["success"] is True
            assert result["filtered_count"] == 1
        finally:
            os.unlink(tmpfile)

    def test_statistics(self):
        from tools.log_analyze import log_analyze
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("2024-01-01 INFO Starting app\n")
            f.write("2024-01-01 ERROR Something failed\n")
            f.write("2024-01-01 INFO Done\n")
            tmpfile = f.name
        try:
            result = json.loads(log_analyze(file_path=tmpfile, statistics=True))
            assert result["success"] is True
            assert "statistics" in result
            assert result["statistics"]["total"] == 3
        finally:
            os.unlink(tmpfile)

    def test_invalid_regex(self):
        from tools.log_analyze import log_analyze
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("test line\n")
            tmpfile = f.name
        try:
            result = json.loads(log_analyze(file_path=tmpfile, search_pattern="[invalid"))
            assert result["success"] is False
            assert "regex" in result["error"].lower()
        finally:
            os.unlink(tmpfile)
