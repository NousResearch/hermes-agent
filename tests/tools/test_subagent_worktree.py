"""Tests for opt-in subagent worktree isolation (tools/subagent_worktree.py).

Inspired by Muse Code's --subagent-worktree-isolation (clean-room
implementation from documented behavior).
"""

import os
import subprocess
import sys
import tempfile
import shutil
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tools import subagent_worktree as sw  # noqa: E402


def _git(args, cwd, check=True):
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=check
    )


def _make_repo(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir()
    _git(["init", "-q"], repo)
    _git(["config", "user.email", "test@test"], repo)
    _git(["config", "user.name", "Test"], repo)
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "seed"], repo)
    return repo


class SubagentWorktreeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hermes-sw-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    # ── resolve_repo_root ──────────────────────────────────────────────

    def test_resolve_repo_root_in_repo(self):
        repo = _make_repo(self.tmp)
        sub = repo / "src"
        sub.mkdir()
        root = sw.resolve_repo_root(str(sub))
        assert root is not None
        self.assertEqual(Path(root).resolve(), repo.resolve())

    def test_resolve_repo_root_non_git(self):
        plain = self.tmp / "plain"
        plain.mkdir()
        self.assertIsNone(sw.resolve_repo_root(str(plain)))

    def test_resolve_repo_root_none_and_missing(self):
        self.assertIsNone(sw.resolve_repo_root(None))
        self.assertIsNone(sw.resolve_repo_root(str(self.tmp / "nope")))

    # ── create_subagent_worktree ───────────────────────────────────────

    def test_create_in_non_git_returns_none(self):
        plain = self.tmp / "plain"
        plain.mkdir()
        self.assertIsNone(sw.create_subagent_worktree(str(plain), "abc"))

    def test_create_makes_isolated_worktree(self):
        repo = _make_repo(self.tmp)
        info = sw.create_subagent_worktree(str(repo), "abc123")
        self.assertIsNotNone(info)
        assert info is not None
        self.assertTrue(os.path.isdir(info["path"]))
        self.assertIn(".worktrees", info["path"])
        self.assertEqual(info["branch"], "hermes-subagent/subagent-abc123")
        self.assertTrue(info["base_commit"])
        # Worktree carries the committed file
        self.assertTrue((Path(info["path"]) / "README.md").exists())
        # .gitignore gained the .worktrees/ entry
        self.assertIn(
            ".worktrees/", (repo / ".gitignore").read_text(encoding="utf-8").splitlines()
        )
        # A write in the worktree does not touch the parent checkout
        (Path(info["path"]) / "child.txt").write_text("x", encoding="utf-8")
        self.assertFalse((repo / "child.txt").exists())

    def test_create_unborn_head_returns_none(self):
        repo = self.tmp / "empty"
        repo.mkdir()
        _git(["init", "-q"], repo)
        self.assertIsNone(sw.create_subagent_worktree(str(repo), "abc"))

    # ── finalize_subagent_worktree ─────────────────────────────────────

    def test_finalize_prunes_clean_worktree(self):
        repo = _make_repo(self.tmp)
        info = sw.create_subagent_worktree(str(repo), "clean1")
        assert info is not None
        payload = sw.finalize_subagent_worktree(info)
        self.assertTrue(payload["pruned"])
        self.assertEqual(payload["commits"], 0)
        self.assertFalse(payload["dirty"])
        self.assertFalse(os.path.isdir(info["path"]))
        # branch deleted too
        branches = _git(["branch", "--list", info["branch"]], repo).stdout
        self.assertEqual(branches.strip(), "")

    def test_finalize_keeps_worktree_with_commits(self):
        repo = _make_repo(self.tmp)
        info = sw.create_subagent_worktree(str(repo), "work1")
        assert info is not None
        wt = Path(info["path"])
        (wt / "feature.py").write_text("print('hi')\n", encoding="utf-8")
        _git(["add", "-A"], wt)
        _git(["config", "user.email", "child@test"], wt)
        _git(["config", "user.name", "Child"], wt)
        _git(["commit", "-q", "-m", "child work"], wt)
        payload = sw.finalize_subagent_worktree(info)
        self.assertFalse(payload["pruned"])
        self.assertEqual(payload["commits"], 1)
        self.assertTrue(os.path.isdir(info["path"]))

    def test_finalize_keeps_dirty_worktree(self):
        repo = _make_repo(self.tmp)
        info = sw.create_subagent_worktree(str(repo), "dirty1")
        assert info is not None
        (Path(info["path"]) / "wip.txt").write_text("uncommitted\n", encoding="utf-8")
        payload = sw.finalize_subagent_worktree(info)
        self.assertFalse(payload["pruned"])
        self.assertTrue(payload["dirty"])
        self.assertTrue(os.path.isdir(info["path"]))

    def test_finalize_keeps_worktree_when_git_inspection_fails(self):
        """#88113: a non-zero git status exit must not be read as "clean".

        Corrupting the index makes the real `git status --porcelain` probe
        exit 128. The old code kept the payload defaults (commits=0,
        dirty=False) and pruned on them — permanently deleting the child's
        uncommitted work. A destructive cleanup requires affirmative proof
        of a clean tree."""
        repo = _make_repo(self.tmp)
        info = sw.create_subagent_worktree(str(repo), "inspect-fail1")
        assert info is not None
        wt = Path(info["path"])
        (wt / "UNCOMMITTED-WORK.txt").write_text(
            "irreplaceable\n", encoding="utf-8"
        )

        git_dir = Path(_git(["rev-parse", "--git-dir"], wt).stdout.strip())
        if not git_dir.is_absolute():
            git_dir = (wt / git_dir).resolve()
        (git_dir / "index").write_bytes(b"not-a-valid-git-index\n")
        # Sanity: the probe really fails now.
        broken = _git(["status", "--porcelain"], wt, check=False)
        self.assertNotEqual(broken.returncode, 0)

        payload = sw.finalize_subagent_worktree(info)

        self.assertFalse(payload["pruned"])
        self.assertTrue(os.path.isdir(info["path"]))
        self.assertTrue((wt / "UNCOMMITTED-WORK.txt").exists())
        branches = _git(["branch", "--list", info["branch"]], repo).stdout
        self.assertNotEqual(branches.strip(), "")

    def test_finalize_missing_path_reports_pruned(self):
        payload = sw.finalize_subagent_worktree(
            {"path": str(self.tmp / "gone"), "branch": "b", "repo_root": "",
             "base_commit": ""}
        )
        self.assertTrue(payload["pruned"])

    # ── local_backend_active ───────────────────────────────────────────

    def test_local_backend_active_local(self):
        with mock.patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"terminal": {"backend": "local"}},
        ):
            self.assertTrue(sw.local_backend_active())

    def test_local_backend_active_docker(self):
        with mock.patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"terminal": {"backend": "docker"}},
        ):
            self.assertFalse(sw.local_backend_active())

    # ── context note ───────────────────────────────────────────────────

    def test_context_note_names_path_and_branch(self):
        note = sw.build_worktree_context_note(
            {"path": "/x/wt", "branch": "hermes-subagent/subagent-1"}
        )
        self.assertIn("/x/wt", note)
        self.assertIn("hermes-subagent/subagent-1", note)
        self.assertIn("WORKTREE ISOLATION", note)


class DelegationConfigGateTests(unittest.TestCase):
    def test_worktree_isolation_default_off(self):
        from tools import delegate_tool

        with mock.patch.object(delegate_tool, "_load_config", return_value={}):
            self.assertFalse(delegate_tool._get_worktree_isolation())

    def test_worktree_isolation_enabled(self):
        from tools import delegate_tool

        with mock.patch.object(
            delegate_tool, "_load_config",
            return_value={"worktree_isolation": True},
        ):
            self.assertTrue(delegate_tool._get_worktree_isolation())


if __name__ == "__main__":
    unittest.main()
