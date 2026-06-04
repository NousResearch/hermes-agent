"""Tests for protected cwd command guard in terminal tools.

Covers BO-238 acceptance criteria:
- DC1: Terminal guard blocks obvious mutators and ambiguous interpreter/package-manager
        commands from protected canonical cwd.
- DC2: Terminal guard allows known read-only git inspection commands from protected
        canonical cwd.
- DC3: Yolo/approval mode does not bypass protected canonical policy unless a separate
        override design is approved.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch, Mock

import pytest

from tools.approval import (
    _check_protected_cwd_command,
    check_all_command_guards,
    _split_tokens,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def non_protected_cwd() -> str:
    """A cwd that is NOT under any protected root."""
    return "/tmp"


@pytest.fixture
def protected_cwd() -> str:
    """A cwd that IS under a protected root (DailyChingu canonical)."""
    return "/home/ubuntu/repos/dailychingu"


# ---------------------------------------------------------------------------
# _split_tokens unit tests
# ---------------------------------------------------------------------------


class TestSplitTokens:
    def test_simple(self):
        assert _split_tokens("rm -rf /tmp/x") == ["rm", "-rf", "/tmp/x"]

    def test_multiline(self):
        assert _split_tokens("echo hello\\\n world") == ["echo", "hello", "world"]

    def test_semicolon_split(self):
        """Semicolons are token boundaries to stop 'safe_cmd; rm -rf /' bypasses."""
        assert _split_tokens("true; rm -rf /") == ["true"]

    def test_pipe_split(self):
        assert _split_tokens("cat f | rm -rf /") == ["cat", "f"]

    def test_and_split(self):
        assert _split_tokens("true && rm -rf /") == ["true"]

    def test_or_split(self):
        assert _split_tokens("false || rm -rf /") == ["false"]

    def test_empty(self):
        assert _split_tokens("") == []
        assert _split_tokens("   ") == []


# ---------------------------------------------------------------------------
# _check_protected_cwd_command unit tests
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_protected_root_blocked():
    """Mock _check_protected_root to return BLOCKED_PROTECTED_CANONICAL."""
    from tools.protected_checkout_policy import ProtectedCheckoutDecision

    decision = ProtectedCheckoutDecision(
        allowed=False,
        reason_code="BLOCKED_PROTECTED_CANONICAL",
        reason_detail="Protected canonical checkout /home/ubuntu/repos/dailychingu on non-task branch 'develop'",
    )
    with patch(
        "tools.protected_checkout_policy._check_protected_root", return_value=decision
    ) as mock_fn:
        yield mock_fn


@pytest.fixture
def mock_is_under_or_equal_true():
    """Mock _is_under_or_equal to always return True (cwd IS protected)."""
    with patch(
        "tools.protected_checkout_policy._is_under_or_equal", return_value=True
    ) as mock_fn:
        yield mock_fn


class TestProtectedCwdCommand:
    """DC1: mutator/ambiguous commands blocked from protected cwd."""

    def test_configured_protected_root_blocks_terminal_mutator(self, tmp_path):
        """Terminal guard uses protected_checkouts config, not only defaults."""
        configured_root = tmp_path / "configured-canonical"
        configured_root.mkdir()
        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value={
                "protected_checkouts": {
                    "canonical_roots": [str(configured_root)],
                    "allowed_worktree_prefixes": [str(tmp_path / "worktrees")],
                }
            },
        ):
            result = _check_protected_cwd_command("touch file.txt", str(configured_root))
        assert result is not None
        assert result["approved"] is False
        assert "[PROTECTED-CWD]" in result["message"]

    def test_non_protected_cwd_allows_all(
        self, non_protected_cwd, mock_protected_root_blocked
    ):
        """Non-protected cwd passes everything through."""
        # rm from /tmp should pass (cwd not protected)
        result = _check_protected_cwd_command("rm -rf test.txt", non_protected_cwd)
        assert result is None

    def test_rm_blocked(
        self, protected_cwd, mock_protected_root_blocked, mock_is_under_or_equal_true
    ):
        result = _check_protected_cwd_command("rm -rf test.txt", protected_cwd)
        assert result is not None
        assert result["approved"] is False
        assert result["status"] == "blocked"
        assert "[PROTECTED-CWD]" in result["message"]
        assert "'rm'" in result["message"]

    def test_mv_blocked(
        self, protected_cwd, mock_protected_root_blocked, mock_is_under_or_equal_true
    ):
        result = _check_protected_cwd_command("mv a b", protected_cwd)
        assert result is not None
        assert result["approved"] is False
        assert "[PROTECTED-CWD]" in result["message"]
        assert "'mv'" in result["message"]

    def test_cp_blocked(
        self, protected_cwd, mock_protected_root_blocked, mock_is_under_or_equal_true
    ):
        result = _check_protected_cwd_command("cp a b", protected_cwd)
        assert result is not None
        assert result["approved"] is False

    def test_touch_blocked(
        self, protected_cwd, mock_protected_root_blocked, mock_is_under_or_equal_true
    ):
        result = _check_protected_cwd_command("touch newfile.txt", protected_cwd)
        assert result is not None
        assert result["approved"] is False

    def test_truncate_blocked(
        self, protected_cwd, mock_protected_root_blocked, mock_is_under_or_equal_true
    ):
        result = _check_protected_cwd_command("truncate -s 0 file.txt", protected_cwd)
        assert result is not None
        assert result["approved"] is False

    def test_tee_blocked(
        self, protected_cwd, mock_protected_root_blocked, mock_is_under_or_equal_true
    ):
        # Pipe-split keeps only 'tee' since it's the first segment.
        result = _check_protected_cwd_command("tee file.txt < input.txt", protected_cwd)
        assert result is not None
        assert result["approved"] is False

    def test_sed_inline_blocked(
        self, protected_cwd, mock_protected_root_blocked, mock_is_under_or_equal_true
    ):
        result = _check_protected_cwd_command(
            "sed -i 's/foo/bar/' file.txt", protected_cwd
        )
        assert result is not None
        assert result["approved"] is False
        assert "sed -i" in result["message"]

    def test_sed_inplace_long_flag(
        self, protected_cwd, mock_protected_root_blocked, mock_is_under_or_equal_true
    ):
        result = _check_protected_cwd_command(
            "sed --in-place 's/foo/bar/' file.txt", protected_cwd
        )
        assert result is not None
        assert result["approved"] is False

    def test_perl_pi_blocked(
        self, protected_cwd, mock_protected_root_blocked, mock_is_under_or_equal_true
    ):
        result = _check_protected_cwd_command(
            "perl -pi -e 's/foo/bar/' file.txt", protected_cwd
        )
        assert result is not None
        assert result["approved"] is False

    def test_python_blocked(
        self, protected_cwd, mock_protected_root_blocked, mock_is_under_or_equal_true
    ):
        result = _check_protected_cwd_command("python script.py", protected_cwd)
        assert result is not None
        assert result["approved"] is False
        assert "'python'" in result["message"]

    def test_node_blocked(
        self, protected_cwd, mock_protected_root_blocked, mock_is_under_or_equal_true
    ):
        result = _check_protected_cwd_command("node script.js", protected_cwd)
        assert result is not None
        assert result["approved"] is False

    def test_bash_blocked(
        self, protected_cwd, mock_protected_root_blocked, mock_is_under_or_equal_true
    ):
        result = _check_protected_cwd_command("bash script.sh", protected_cwd)
        assert result is not None
        assert result["approved"] is False

    def test_sh_blocked(
        self, protected_cwd, mock_protected_root_blocked, mock_is_under_or_equal_true
    ):
        result = _check_protected_cwd_command("sh script.sh", protected_cwd)
        assert result is not None
        assert result["approved"] is False

    def test_npm_blocked(
        self, protected_cwd, mock_protected_root_blocked, mock_is_under_or_equal_true
    ):
        result = _check_protected_cwd_command("npm install", protected_cwd)
        assert result is not None
        assert result["approved"] is False

    def test_pip_blocked(
        self, protected_cwd, mock_protected_root_blocked, mock_is_under_or_equal_true
    ):
        result = _check_protected_cwd_command("pip install requests", protected_cwd)
        assert result is not None
        assert result["approved"] is False

    def test_git_restore_blocked(
        self, protected_cwd, mock_protected_root_blocked, mock_is_under_or_equal_true
    ):
        result = _check_protected_cwd_command("git restore file.txt", protected_cwd)
        assert result is not None
        assert result["approved"] is False
        assert "git restore" in result["message"]

    def test_git_reset_blocked(
        self, protected_cwd, mock_protected_root_blocked, mock_is_under_or_equal_true
    ):
        result = _check_protected_cwd_command("git reset --hard", protected_cwd)
        assert result is not None
        assert result["approved"] is False

    def test_git_clean_blocked(
        self, protected_cwd, mock_protected_root_blocked, mock_is_under_or_equal_true
    ):
        result = _check_protected_cwd_command("git clean -fd", protected_cwd)
        assert result is not None
        assert result["approved"] is False

    def test_git_add_blocked(
        self, protected_cwd, mock_protected_root_blocked, mock_is_under_or_equal_true
    ):
        result = _check_protected_cwd_command("git add file.txt", protected_cwd)
        assert result is not None
        assert result["approved"] is False

    def test_git_commit_blocked(
        self, protected_cwd, mock_protected_root_blocked, mock_is_under_or_equal_true
    ):
        result = _check_protected_cwd_command('git commit -m "msg"', protected_cwd)
        assert result is not None
        assert result["approved"] is False

    def test_git_checkout_flag_blocked(
        self, protected_cwd, mock_protected_root_blocked, mock_is_under_or_equal_true
    ):
        """git checkout as a subcommand is not in the known lists —
        an unknown git subcommand from protected cwd is blocked."""
        result = _check_protected_cwd_command("git checkout -- file.txt", protected_cwd)
        assert result is not None
        assert result["approved"] is False


class TestProtectedCwdReadOnly:
    """DC2: read-only git commands allowed from protected cwd."""

    def test_git_status_allowed(
        self, protected_cwd, mock_protected_root_blocked, mock_is_under_or_equal_true
    ):
        result = _check_protected_cwd_command("git status", protected_cwd)
        assert result is None  # allowed

    def test_git_diff_allowed(
        self, protected_cwd, mock_protected_root_blocked, mock_is_under_or_equal_true
    ):
        result = _check_protected_cwd_command("git diff", protected_cwd)
        assert result is None

    def test_git_log_allowed(
        self, protected_cwd, mock_protected_root_blocked, mock_is_under_or_equal_true
    ):
        result = _check_protected_cwd_command("git log --oneline", protected_cwd)
        assert result is None

    def test_git_rev_parse_allowed(
        self, protected_cwd, mock_protected_root_blocked, mock_is_under_or_equal_true
    ):
        result = _check_protected_cwd_command("git rev-parse HEAD", protected_cwd)
        assert result is None

    def test_git_worktree_list_allowed(
        self, protected_cwd, mock_protected_root_blocked, mock_is_under_or_equal_true
    ):
        result = _check_protected_cwd_command("git worktree list", protected_cwd)
        assert result is None

    def test_git_branch_list_allowed(
        self, protected_cwd, mock_protected_root_blocked, mock_is_under_or_equal_true
    ):
        result = _check_protected_cwd_command("git branch -a", protected_cwd)
        assert result is None

    def test_git_show_allowed(
        self, protected_cwd, mock_protected_root_blocked, mock_is_under_or_equal_true
    ):
        result = _check_protected_cwd_command("git show HEAD", protected_cwd)
        assert result is None

    def test_git_ls_files_allowed(
        self, protected_cwd, mock_protected_root_blocked, mock_is_under_or_equal_true
    ):
        result = _check_protected_cwd_command("git ls-files", protected_cwd)
        assert result is None

    def test_git_stash_list_allowed(
        self, protected_cwd, mock_protected_root_blocked, mock_is_under_or_equal_true
    ):
        """git stash list is read-only, allowed."""
        result = _check_protected_cwd_command("git stash list", protected_cwd)
        assert result is None

    def test_git_stash_show_allowed(
        self, protected_cwd, mock_protected_root_blocked, mock_is_under_or_equal_true
    ):
        """git stash show is read-only, allowed."""
        result = _check_protected_cwd_command("git stash show", protected_cwd)
        assert result is None

    def test_git_stash_push_blocked(
        self, protected_cwd, mock_protected_root_blocked, mock_is_under_or_equal_true
    ):
        """git stash push is mutation, blocked."""
        result = _check_protected_cwd_command("git stash push", protected_cwd)
        assert result is not None
        assert result["approved"] is False
        assert "stash push" in result["message"]

    def test_git_stash_pop_blocked(
        self, protected_cwd, mock_protected_root_blocked, mock_is_under_or_equal_true
    ):
        """git stash pop is mutation, blocked."""
        result = _check_protected_cwd_command("git stash pop", protected_cwd)
        assert result is not None
        assert result["approved"] is False

    def test_cat_allowed(
        self, protected_cwd, mock_protected_root_blocked, mock_is_under_or_equal_true
    ):
        """cat is read-only inspection, allowed."""
        result = _check_protected_cwd_command("cat file.txt", protected_cwd)
        assert result is None

    def test_ls_allowed(
        self, protected_cwd, mock_protected_root_blocked, mock_is_under_or_equal_true
    ):
        """ls is read-only, allowed."""
        result = _check_protected_cwd_command("ls -la", protected_cwd)
        assert result is None

    def test_echo_allowed(
        self, protected_cwd, mock_protected_root_blocked, mock_is_under_or_equal_true
    ):
        """echo to stdout is allowed (no file redirect in this test)."""
        result = _check_protected_cwd_command("echo hello", protected_cwd)
        assert result is None


# ---------------------------------------------------------------------------
# DC3: YOLO does NOT bypass protected cwd guard
# ---------------------------------------------------------------------------


class TestProtectedCwdYoloNoBypass:
    """DC3: Yolo mode does not bypass protected cwd guard."""

    def test_yolo_does_not_bypass_protected_cwd(self, protected_cwd):
        """When cwd is protected canonical and yolo is on,
        the protected-cwd guard should still block (it fires before yolo)."""
        from tools.protected_checkout_policy import ProtectedCheckoutDecision

        block_decision = ProtectedCheckoutDecision(
            allowed=False,
            reason_code="BLOCKED_PROTECTED_CANONICAL",
            reason_detail="Protected canonical checkout on 'develop'",
        )

        with patch(
            "tools.protected_checkout_policy._is_under_or_equal", return_value=True
        ), patch(
            "tools.protected_checkout_policy._check_protected_root", return_value=block_decision
        ), patch(
            "tools.approval._YOLO_MODE_FROZEN", True
        ), patch(
            "tools.approval.is_current_session_yolo_enabled", return_value=False
        ):
            result = check_all_command_guards(
                "rm -rf test.txt", "local", workdir=protected_cwd
            )
            assert result["approved"] is False
            assert "[PROTECTED-CWD]" in result["message"]

    def test_mode_off_does_not_bypass_protected_cwd(self, protected_cwd):
        """approvals.mode=off should still not bypass protected cwd guard."""
        from tools.protected_checkout_policy import ProtectedCheckoutDecision

        block_decision = ProtectedCheckoutDecision(
            allowed=False,
            reason_code="BLOCKED_PROTECTED_CANONICAL",
            reason_detail="Protected canonical checkout on 'develop'",
        )

        with patch(
            "tools.protected_checkout_policy._is_under_or_equal", return_value=True
        ), patch(
            "tools.protected_checkout_policy._check_protected_root", return_value=block_decision
        ), patch(
            "tools.approval._get_approval_mode", return_value="off"
        ), patch(
            "tools.approval._YOLO_MODE_FROZEN", False
        ), patch(
            "tools.approval.is_current_session_yolo_enabled", return_value=False
        ):
            result = check_all_command_guards(
                "rm -rf test.txt", "local", workdir=protected_cwd
            )
            assert result["approved"] is False
            assert "[PROTECTED-CWD]" in result["message"]


# ---------------------------------------------------------------------------
# Integration: check_all_command_guards with workdir
# ---------------------------------------------------------------------------


class TestCheckAllCommandGuardsIntegration:
    """Integration tests for the full guard pipeline with workdir."""

    def test_non_protected_cwd_passes_through(self, non_protected_cwd):
        """When cwd is not protected, the guard pipeline works normally."""
        with patch(
            "tools.approval.detect_hardline_command", return_value=(False, "")
        ), patch(
            "tools.approval._check_sudo_stdin_guard", return_value=(False, "")
        ):
            result = check_all_command_guards(
                "ls -la", "local", workdir=non_protected_cwd
            )
            assert result["approved"] is True

    def test_workdir_none_skips_protected_check(self):
        """When workdir is None, protected cwd guard is skipped."""
        with patch(
            "tools.approval.detect_hardline_command", return_value=(False, "")
        ), patch(
            "tools.approval._check_sudo_stdin_guard", return_value=(False, "")
        ):
            result = check_all_command_guards(
                "rm -rf test.txt", "local", workdir=None
            )
            assert result["approved"] is True

    def test_container_env_skips_all_guards(self):
        """Container envs skip all checks (including protected cwd)."""
        result = check_all_command_guards(
            "rm -rf /", "docker", workdir="/home/ubuntu/repos/dailychingu"
        )
        assert result["approved"] is True

    def test_task_worktree_branch_allows_all(
        self, protected_cwd, mock_is_under_or_equal_true
    ):
        """Autopilot/wt branches on a protected root allow all commands."""
        from tools.protected_checkout_policy import ProtectedCheckoutDecision

        allow_decision = ProtectedCheckoutDecision(
            allowed=True,
            reason_code="ALLOWED_TASK_WORKTREE",
            reason_detail="Protected root worktree with task branch 'autopilot/bo-238'",
        )

        with patch(
            "tools.protected_checkout_policy._check_protected_root", return_value=allow_decision
        ), patch(
            "tools.approval.detect_hardline_command", return_value=(False, "")
        ), patch(
            "tools.approval._check_sudo_stdin_guard", return_value=(False, "")
        ):
            result = check_all_command_guards(
                "rm -rf test.txt", "local", workdir=protected_cwd
            )
            assert result["approved"] is True
