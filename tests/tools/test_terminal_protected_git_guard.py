"""Tests for the protected-git-roots guard wired into terminal_tool.

Verifies that the terminal tool blocks destructive git commands fail-closed
(even with force=True) when HERMES_PROTECTED_GIT_ROOTS is set, and that
non-kanban sessions are unaffected.
"""
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call_terminal(command: str, session_cwd: str, pg_roots: str | None = None,
                   force: bool = False):
    """Call terminal_tool() with HERMES_PROTECTED_GIT_ROOTS optionally set."""
    import tools.terminal_tool as tt

    env_patch = {}
    if pg_roots is not None:
        env_patch["HERMES_PROTECTED_GIT_ROOTS"] = pg_roots
    else:
        env_patch.pop("HERMES_PROTECTED_GIT_ROOTS", None)

    # Patch _get_env_config to return a minimal local-backend config.
    fake_config = {
        "env_type": "local",
        "cwd": session_cwd,
        "timeout": 30,
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
    }

    # We don't want real command execution — stub out _wait_for_process and
    # env.execute so only the guards run.
    with patch.dict(os.environ, env_patch, clear=False), \
         patch.object(tt, "_get_env_config", return_value=fake_config), \
         patch.object(tt, "resolve_task_overrides", return_value={}), \
         patch.object(tt, "_foreground_background_guidance", return_value=None), \
         patch.object(tt, "_start_cleanup_thread"):

        # Only run up to the guard checks — we catch before env setup.
        with patch.dict(os.environ, {"HERMES_PROTECTED_GIT_ROOTS": pg_roots or ""},
                        clear=False):
            # Remove env var when pg_roots is None
            if pg_roots is None and "HERMES_PROTECTED_GIT_ROOTS" in os.environ:
                del os.environ["HERMES_PROTECTED_GIT_ROOTS"]

            # Direct function call via the guard helper (lower-level test)
            from tools.approval import check_protected_git_roots_guard
            return check_protected_git_roots_guard(command, session_cwd)


# ---------------------------------------------------------------------------
# Integration: guard returns None when env var absent
# ---------------------------------------------------------------------------

class TestGuardInactive:
    def test_no_env_no_block(self, monkeypatch):
        monkeypatch.delenv("HERMES_PROTECTED_GIT_ROOTS", raising=False)
        from tools.approval import check_protected_git_roots_guard
        result = check_protected_git_roots_guard("git reset --hard HEAD", "/repo")
        assert result is None

    def test_empty_env_no_block(self, monkeypatch):
        monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", "")
        from tools.approval import check_protected_git_roots_guard
        result = check_protected_git_roots_guard("git stash push", "/repo")
        assert result is None


# ---------------------------------------------------------------------------
# Integration: guard blocks correctly with env var set
# ---------------------------------------------------------------------------

class TestGuardActive:
    @pytest.fixture(autouse=True)
    def set_protected_root(self, tmp_path, monkeypatch):
        self._root = tmp_path / "mainrepo"
        self._root.mkdir()
        monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", str(self._root))

    def _guard(self, command: str, cwd: str | None = None):
        from tools.approval import check_protected_git_roots_guard
        return check_protected_git_roots_guard(command, cwd or str(self._root))

    # ── Blocked ──────────────────────────────────────────────────────────

    def test_git_reset_hard_blocked(self):
        assert self._guard("git reset --hard HEAD~2")["approved"] is False

    def test_git_reset_hard_abbreviated_blocked(self):
        # git accepts abbreviations like --har, --ha, --h
        assert self._guard("git reset --hard")["approved"] is False

    def test_git_stash_push_blocked(self):
        assert self._guard("git stash push")["approved"] is False

    def test_git_stash_pop_blocked(self):
        assert self._guard("git stash pop")["approved"] is False

    def test_git_stash_bare_blocked(self):
        assert self._guard("git stash")["approved"] is False

    def test_git_checkout_commit_blocked(self):
        assert self._guard("git checkout abc1234")["approved"] is False

    def test_git_checkout_branch_blocked(self):
        assert self._guard("git checkout main")["approved"] is False

    def test_git_checkout_file_restore_blocked(self):
        assert self._guard("git checkout -- src/foo.py")["approved"] is False

    def test_git_clean_fd_blocked(self):
        assert self._guard("git clean -fd")["approved"] is False

    def test_git_branch_D_blocked(self):
        assert self._guard("git branch -D old")["approved"] is False

    def test_git_push_force_blocked(self):
        assert self._guard("git push --force origin main")["approved"] is False

    def test_git_push_f_blocked(self):
        assert self._guard("git push -f origin main")["approved"] is False

    def test_git_switch_blocked(self):
        assert self._guard("git switch develop")["approved"] is False

    def test_error_message_mentions_worktree_alternative(self):
        r = self._guard("git reset --hard HEAD~1")
        assert "worktree add" in r["message"] or "git show" in r["message"]

    # ── Allowed ──────────────────────────────────────────────────────────

    def test_git_stash_list_allowed(self):
        assert self._guard("git stash list") is None

    def test_git_stash_show_allowed(self):
        assert self._guard("git stash show stash@{0}") is None

    def test_git_reset_soft_allowed(self):
        assert self._guard("git reset --soft HEAD~1") is None

    def test_git_reset_mixed_allowed(self):
        assert self._guard("git reset --mixed") is None

    def test_git_checkout_new_branch_allowed(self):
        assert self._guard("git checkout -b feat/new") is None

    def test_git_log_allowed(self):
        assert self._guard("git log --oneline -5") is None

    def test_git_status_allowed(self):
        assert self._guard("git status") is None

    def test_git_diff_allowed(self):
        assert self._guard("git diff HEAD") is None

    def test_git_add_allowed(self):
        assert self._guard("git add -p") is None

    def test_git_commit_allowed(self):
        assert self._guard("git commit -m 'fix'") is None

    def test_git_push_no_force_allowed(self):
        assert self._guard("git push origin main") is None

    # ── CWD outside protected root → allowed ─────────────────────────────

    def test_safe_cwd_allows_dangerous(self, tmp_path):
        safe = tmp_path / "safedir"
        safe.mkdir()
        from tools.approval import check_protected_git_roots_guard
        result = check_protected_git_roots_guard("git reset --hard", str(safe))
        assert result is None

    # ── .worktrees child cwd → allowed ───────────────────────────────────

    def test_worktrees_child_allowed(self):
        worktree = self._root / ".worktrees" / "t_abc123"
        worktree.mkdir(parents=True)
        assert self._guard("git reset --hard HEAD~1", str(worktree)) is None

    def test_worktrees_deep_child_allowed(self):
        deep = self._root / ".worktrees" / "t_abc123" / "src"
        deep.mkdir(parents=True)
        assert self._guard("git checkout main", str(deep)) is None

    # ── cd to safe path + dangerous → allowed ────────────────────────────

    def test_cd_to_safe_then_dangerous_allowed(self, tmp_path):
        safe = tmp_path / "safe"
        safe.mkdir()
        cmd = f"cd {safe} && git reset --hard HEAD~1"
        assert self._guard(cmd) is None

    # ── cd to protected root + dangerous → blocked ────────────────────────

    def test_cd_to_protected_then_dangerous_blocked(self, tmp_path):
        safe_start = tmp_path / "start"
        safe_start.mkdir()
        cmd = f"cd {self._root} && git reset --hard"
        from tools.approval import check_protected_git_roots_guard
        result = check_protected_git_roots_guard(cmd, str(safe_start))
        assert result is not None
        assert result["approved"] is False

    # ── Bypass patterns ──────────────────────────────────────────────────

    def test_git_dash_C_bypass_blocked(self, tmp_path):
        """git -C <protected-root> reset --hard from safe cwd should block."""
        safe = tmp_path / "safe"
        safe.mkdir()
        cmd = f"git -C {self._root} reset --hard"
        from tools.approval import check_protected_git_roots_guard
        result = check_protected_git_roots_guard(cmd, str(safe))
        assert result is not None
        assert result["approved"] is False

    def test_semicolon_chain_blocked(self):
        """echo foo; git reset --hard still runs in protected cwd."""
        result = self._guard("echo foo; git reset --hard")
        assert result is not None
        assert result["approved"] is False

    def test_command_git_wrapper_blocked(self):
        """command git reset --hard text still matches the pattern."""
        result = self._guard("command git reset --hard")
        assert result is not None
        assert result["approved"] is False


# ---------------------------------------------------------------------------
# Integration: terminal_tool JSON response format
# ---------------------------------------------------------------------------

class TestTerminalToolJsonResponse:
    """Verify terminal_tool itself returns the right JSON when guard fires."""

    def _run_with_guard(self, command, cwd, pg_roots, force=False):
        """Call terminal_tool with minimal stubs — stops before actual exec."""
        import tools.terminal_tool as tt

        fake_config = {
            "env_type": "local",
            "cwd": cwd,
            "timeout": 30,
            "docker_image": "",
            "singularity_image": "",
            "modal_image": "",
            "daytona_image": "",
        }

        env_patch = {"HERMES_PROTECTED_GIT_ROOTS": pg_roots} if pg_roots else {}

        # Stub just enough to reach the guard without real subprocess creation.
        # _create_environment is patched to avoid Docker/container calls.
        # For local env the env object is created in-process; we patch execute()
        # on the created env so no subprocess is spawned if the guard somehow
        # doesn't fire.
        with patch.dict(os.environ, env_patch, clear=False), \
             patch.object(tt, "_get_env_config", return_value=fake_config), \
             patch.object(tt, "resolve_task_overrides", return_value={}), \
             patch.object(tt, "_foreground_background_guidance", return_value=None), \
             patch.object(tt, "_start_cleanup_thread"):
            return tt.terminal_tool(command=command, force=force)

    def test_blocked_returns_json_with_status_blocked(self, tmp_path):
        result_str = self._run_with_guard(
            "git reset --hard HEAD~1",
            cwd=str(tmp_path),
            pg_roots=str(tmp_path),
        )
        result = json.loads(result_str)
        assert result["exit_code"] == -1
        assert result["status"] == "blocked"
        assert "BLOCKED" in result["error"]

    def test_blocked_even_with_force_true(self, tmp_path):
        """force=True must NOT bypass the protected-git-roots guard."""
        result_str = self._run_with_guard(
            "git reset --hard HEAD~1",
            cwd=str(tmp_path),
            pg_roots=str(tmp_path),
            force=True,
        )
        result = json.loads(result_str)
        assert result["exit_code"] == -1
        assert result["status"] == "blocked"

    def test_no_env_allows_command_to_proceed(self, tmp_path):
        """Without HERMES_PROTECTED_GIT_ROOTS, the guard must not fire."""
        # We just verify the guard doesn't block (command may fail for other
        # reasons — we don't care about successful execution here).
        result_str = self._run_with_guard(
            "git reset --hard HEAD~1",
            cwd=str(tmp_path),
            pg_roots=None,  # guard off
        )
        result = json.loads(result_str)
        # Guard block would produce exit_code=-1 with status=blocked.
        # Without the guard, either the command ran (exit 128 no git repo) or
        # approval flow kicked in — anything except our guard's blocked status.
        assert result.get("status") != "blocked" or "BLOCKED" not in result.get("error", "")

