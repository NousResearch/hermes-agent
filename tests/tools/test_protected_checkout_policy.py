"""Tests for tools/protected_checkout_policy.py.

Covers:
- Protected canonical path → BLOCKED
- Allowed .worktrees path → ALLOWED
- Non-protected repo → ALLOWED
- Branch-lookup-unavailable under protected root → BLOCKED
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from unittest.mock import patch

import pytest

from tools.protected_checkout_policy import (
    ALLOWED_WORKTREE_PREFIXES,
    PROTECTED_CANONICAL_ROOTS,
    ProtectedCheckoutDecision,
    check_path_mutation,
    effective_protected_checkout_registry,
)


class TestCheckPathMutation:
    """Core decision logic."""

    def test_non_protected_repo_allowed(self, tmp_path):
        """A path outside any protected root returns ALLOWED_NON_PROTECTED."""
        path = str(tmp_path / "some_file.txt")
        result = check_path_mutation(path)
        assert result.allowed is True
        assert result.reason_code == "ALLOWED_NON_PROTECTED"

    def test_allowed_worktree_prefix_allowed(self):
        """A path inside an allowed worktree prefix returns ALLOWED_WORKTREE."""
        # /home/ubuntu/.hermes/hermes-agent/.worktrees is in the default list
        path = "/home/ubuntu/.hermes/hermes-agent/.worktrees/bo-236-anything/foo.txt"
        result = check_path_mutation(path)
        assert result.allowed is True
        assert result.reason_code == "ALLOWED_WORKTREE"

    def test_protected_canonical_task_branch_allowed(self, tmp_path):
        """Protected root + autopilot/ branch → ALLOWED_TASK_WORKTREE."""
        with patch.object(
            subprocess, "run", autospec=True
        ) as mock_run, patch(
            "tools.protected_checkout_policy.PROTECTED_CANONICAL_ROOTS",
            [str(tmp_path)],
        ):
            mock_result = subprocess.CompletedProcess(
                args=["git", "branch", "--show-current"],
                returncode=0,
                stdout="autopilot/bo-236-something\n",
                stderr="",
            )
            mock_run.return_value = mock_result
            result = check_path_mutation(str(tmp_path / "src/app.py"))
            assert result.allowed is True
            assert result.reason_code == "ALLOWED_TASK_WORKTREE"

    def test_protected_canonical_wt_branch_allowed(self, tmp_path):
        """Protected root + wt/ branch → ALLOWED_TASK_WORKTREE."""
        with patch.object(
            subprocess, "run", autospec=True
        ) as mock_run, patch(
            "tools.protected_checkout_policy.PROTECTED_CANONICAL_ROOTS",
            [str(tmp_path)],
        ):
            mock_result = subprocess.CompletedProcess(
                args=["git", "branch", "--show-current"],
                returncode=0,
                stdout="wt/some-task\n",
                stderr="",
            )
            mock_run.return_value = mock_result
            result = check_path_mutation(str(tmp_path / "file.py"))
            assert result.allowed is True
            assert result.reason_code == "ALLOWED_TASK_WORKTREE"

    def test_protected_canonical_non_task_branch_blocked(self, tmp_path):
        """Protected root + main/master/dev/feature branch → BLOCKED."""
        with patch.object(
            subprocess, "run", autospec=True
        ) as mock_run, patch(
            "tools.protected_checkout_policy.PROTECTED_CANONICAL_ROOTS",
            [str(tmp_path)],
        ):
            mock_result = subprocess.CompletedProcess(
                args=["git", "branch", "--show-current"],
                returncode=0,
                stdout="main\n",
                stderr="",
            )
            mock_run.return_value = mock_result
            result = check_path_mutation(str(tmp_path / "src/index.js"))
            assert result.allowed is False
            assert result.reason_code == "BLOCKED_PROTECTED_CANONICAL"

    def test_protected_canonical_feature_branch_blocked(self, tmp_path):
        """Protected root + feature/ branch (not autopilot/ not wt/) → BLOCKED."""
        with patch.object(
            subprocess, "run", autospec=True
        ) as mock_run, patch(
            "tools.protected_checkout_policy.PROTECTED_CANONICAL_ROOTS",
            [str(tmp_path)],
        ):
            mock_result = subprocess.CompletedProcess(
                args=["git", "branch", "--show-current"],
                returncode=0,
                stdout="feature/new-ui\n",
                stderr="",
            )
            mock_run.return_value = mock_result
            result = check_path_mutation(str(tmp_path / "file.ts"))
            assert result.allowed is False
            assert result.reason_code == "BLOCKED_PROTECTED_CANONICAL"

    def test_branch_lookup_git_returns_nonzero_blocked(self, tmp_path):
        """git returns non-zero → BLOCKED_BRANCH_LOOKUP_FAILED."""
        with patch.object(
            subprocess, "run", autospec=True
        ) as mock_run, patch(
            "tools.protected_checkout_policy.PROTECTED_CANONICAL_ROOTS",
            [str(tmp_path)],
        ):
            mock_result = subprocess.CompletedProcess(
                args=["git", "branch", "--show-current"],
                returncode=128,
                stdout="",
                stderr="fatal: not a git repository\n",
            )
            mock_run.return_value = mock_result
            result = check_path_mutation(str(tmp_path / "config.yaml"))
            assert result.allowed is False
            assert result.reason_code == "BLOCKED_BRANCH_LOOKUP_FAILED"
            assert "fatal:" in result.reason_detail

    def test_branch_lookup_empty_output_blocked(self, tmp_path):
        """git returns empty output → BLOCKED_BRANCH_LOOKUP_FAILED."""
        with patch.object(
            subprocess, "run", autospec=True
        ) as mock_run, patch(
            "tools.protected_checkout_policy.PROTECTED_CANONICAL_ROOTS",
            [str(tmp_path)],
        ):
            mock_result = subprocess.CompletedProcess(
                args=["git", "branch", "--show-current"],
                returncode=0,
                stdout="",
                stderr="",
            )
            mock_run.return_value = mock_result
            result = check_path_mutation(str(tmp_path / "anything.txt"))
            assert result.allowed is False
            assert result.reason_code == "BLOCKED_BRANCH_LOOKUP_FAILED"

    def test_branch_lookup_timeout_blocked(self, tmp_path):
        """git times out → BLOCKED_BRANCH_LOOKUP_FAILED."""
        with patch.object(
            subprocess, "run", autospec=True
        ) as mock_run, patch(
            "tools.protected_checkout_policy.PROTECTED_CANONICAL_ROOTS",
            [str(tmp_path)],
        ):
            mock_run.side_effect = subprocess.TimeoutExpired(
                cmd=["git", "branch", "--show-current"], timeout=5
            )
            result = check_path_mutation(str(tmp_path / "slow.txt"))
            assert result.allowed is False
            assert result.reason_code == "BLOCKED_BRANCH_LOOKUP_FAILED"

    def test_branch_lookup_oserror_blocked(self, tmp_path):
        """git is not available → BLOCKED_BRANCH_LOOKUP_FAILED."""
        with patch.object(
            subprocess, "run", autospec=True
        ) as mock_run, patch(
            "tools.protected_checkout_policy.PROTECTED_CANONICAL_ROOTS",
            [str(tmp_path)],
        ):
            mock_run.side_effect = OSError("No such file or directory")
            result = check_path_mutation(str(tmp_path / "no-git.txt"))
            assert result.allowed is False
            assert result.reason_code == "BLOCKED_BRANCH_LOOKUP_FAILED"

    def test_branch_lookup_filenotfounderror_blocked(self, tmp_path):
        """git binary not found → BLOCKED_BRANCH_LOOKUP_FAILED."""
        with patch.object(
            subprocess, "run", autospec=True
        ) as mock_run, patch(
            "tools.protected_checkout_policy.PROTECTED_CANONICAL_ROOTS",
            [str(tmp_path)],
        ):
            mock_run.side_effect = FileNotFoundError("git")
            result = check_path_mutation(str(tmp_path / "missing-git.py"))
            assert result.allowed is False
            assert result.reason_code == "BLOCKED_BRANCH_LOOKUP_FAILED"

    def test_protected_root_exact_match_blocked(self, tmp_path):
        """The exact protected root itself is blocked unless on task branch."""
        with patch.object(
            subprocess, "run", autospec=True
        ) as mock_run, patch(
            "tools.protected_checkout_policy.PROTECTED_CANONICAL_ROOTS",
            [str(tmp_path)],
        ):
            mock_result = subprocess.CompletedProcess(
                args=["git", "branch", "--show-current"],
                returncode=0,
                stdout="main\n",
                stderr="",
            )
            mock_run.return_value = mock_result
            result = check_path_mutation(str(tmp_path))
            assert result.allowed is False
            assert result.reason_code == "BLOCKED_PROTECTED_CANONICAL"

    def test_protected_root_exact_match_task_branch_allowed(self, tmp_path):
        """The exact protected root with task branch → ALLOWED."""
        with patch.object(
            subprocess, "run", autospec=True
        ) as mock_run, patch(
            "tools.protected_checkout_policy.PROTECTED_CANONICAL_ROOTS",
            [str(tmp_path)],
        ):
            mock_result = subprocess.CompletedProcess(
                args=["git", "branch", "--show-current"],
                returncode=0,
                stdout="autopilot/bo-236-module\n",
                stderr="",
            )
            mock_run.return_value = mock_result
            result = check_path_mutation(str(tmp_path))
            assert result.allowed is True
            assert result.reason_code == "ALLOWED_TASK_WORKTREE"

    def test_path_resolved_detects_protected_via_symlink(self, tmp_path):
        """Symlink that resolves into protected root is blocked."""
        with patch.object(
            subprocess, "run", autospec=True
        ) as mock_run, patch(
            "tools.protected_checkout_policy.PROTECTED_CANONICAL_ROOTS",
            [str(tmp_path / "real-target")],
        ):
            real_target = tmp_path / "real-target"
            real_target.mkdir()
            symlink = tmp_path / "link-to-target"
            symlink.symlink_to(real_target)
            mock_result = subprocess.CompletedProcess(
                args=["git", "branch", "--show-current"],
                returncode=0,
                stdout="main\n",
                stderr="",
            )
            mock_run.return_value = mock_result
            result = check_path_mutation(str(symlink / "config.toml"))
            assert result.allowed is False
            assert result.reason_code == "BLOCKED_PROTECTED_CANONICAL"

    def test_subpath_not_protected_when_root_is_prefix_of_other(self, tmp_path):
        """Path starting with a protected root string but not actually under it is allowed."""
        # e.g. protected root is /home/ubuntu/repos/dailychingu, but path is
        # /home/ubuntu/repos/dailychingu-other
        with patch(
            "tools.protected_checkout_policy.PROTECTED_CANONICAL_ROOTS",
            [str(tmp_path / "dailychingu")],
        ):
            other = tmp_path / "dailychingu-other"
            other.mkdir()
            result = check_path_mutation(str(other / "file.txt"))
            assert result.allowed is True
            assert result.reason_code == "ALLOWED_NON_PROTECTED"

    def test_configured_registry_overrides_defaults(self, tmp_path):
        """protected_checkouts config is the executable SSOT when present."""
        protected = tmp_path / "canonical"
        allowed = tmp_path / "allowed-worktrees"
        protected.mkdir()
        allowed.mkdir()
        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value={
                "protected_checkouts": {
                    "canonical_roots": [str(protected)],
                    "allowed_worktree_prefixes": [str(allowed)],
                }
            },
        ):
            registry = effective_protected_checkout_registry()
            assert registry["canonical_roots"] == [str(protected)]
            assert registry["allowed_worktree_prefixes"] == [str(allowed)]
            result = check_path_mutation(str(allowed / "bo-239" / "file.py"))
            assert result.allowed is True
            assert result.reason_code == "ALLOWED_WORKTREE"

    def test_malformed_config_falls_back_to_defaults(self, tmp_path):
        """Malformed config must not globally break the guard."""
        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"protected_checkouts": "bad-shape"},
        ):
            registry = effective_protected_checkout_registry()
            assert registry["canonical_roots"] == PROTECTED_CANONICAL_ROOTS
            assert registry["allowed_worktree_prefixes"] == ALLOWED_WORKTREE_PREFIXES
            result = check_path_mutation(str(tmp_path / "ordinary" / "file.py"))
            assert result.allowed is True
            assert result.reason_code == "ALLOWED_NON_PROTECTED"

    def test_decision_type(self, tmp_path):
        """Return type is a frozen ProtectedCheckoutDecision dataclass."""
        result = check_path_mutation(str(tmp_path / "x"))
        assert isinstance(result, ProtectedCheckoutDecision)
        assert isinstance(result.allowed, bool)
        assert isinstance(result.reason_code, str)
        assert isinstance(result.reason_detail, str)
