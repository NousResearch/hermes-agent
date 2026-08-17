"""Tests for opt-in subagent worktree isolation (tools/subagent_worktree.py).

Inspired by Muse Code's --subagent-worktree-isolation (clean-room
implementation from documented behavior).
"""

import ast
import inspect
import os
import subprocess
import sys
import tempfile
import textwrap
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


def _break_git_index(wt: Path) -> None:
    """Corrupt a worktree's index so the real git probes exit non-zero."""
    git_dir = Path(_git(["rev-parse", "--git-dir"], wt).stdout.strip())
    if not git_dir.is_absolute():
        git_dir = (wt / git_dir).resolve()
    (git_dir / "index").write_bytes(b"not-a-valid-git-index\n")


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
        # The parent agent only ever sees this payload (it cannot read logs),
        # so the uncertainty must travel in the dict — otherwise "0 commits,
        # clean" reads as "the child produced nothing" and the work we just
        # preserved never gets looked at. Assert the actionable invariants
        # (flag set; note names the worktree + branch), not the prose.
        self.assertTrue(payload["inspection_failed"])
        self.assertIn(info["path"], payload["note"])
        self.assertIn(info["branch"], payload["note"])

    def test_finalize_flags_unproven_state_distinguishably(self):
        """#88113 follow-up: a failed inspection must not look like "no work".

        Without an explicit flag, "inspection failed, uncommitted work
        preserved" and "inspected fine, child left nothing" are
        indistinguishable to the parent agent — so its rational reading of the
        failure case is the exact wrong conclusion. The contract asserted here
        is *distinguishability*, not the specific field values (a future
        change emitting ``commits: None`` for "unknown" would be strictly
        better and must not break this test).
        """
        repo = _make_repo(self.tmp)

        # Case 1: inspection SUCCEEDED, tree genuinely clean, prune disabled.
        ok_info = sw.create_subagent_worktree(str(repo), "proven-clean")
        assert ok_info is not None
        ok_payload = sw.finalize_subagent_worktree(ok_info, prune=False)

        # Case 2: inspection FAILED with real uncommitted work on disk.
        bad_info = sw.create_subagent_worktree(str(repo), "unproven")
        assert bad_info is not None
        bad_wt = Path(bad_info["path"])
        (bad_wt / "WIP.txt").write_text("real work\n", encoding="utf-8")
        _break_git_index(bad_wt)
        bad_payload = sw.finalize_subagent_worktree(bad_info)

        # Both keep the worktree, so "pruned" alone cannot separate them...
        self.assertFalse(ok_payload["pruned"])
        self.assertFalse(bad_payload["pruned"])
        self.assertTrue(os.path.isdir(bad_info["path"]))
        self.assertTrue((bad_wt / "WIP.txt").exists())
        # ...the flag must. Tolerant of an always-present-but-False refactor.
        self.assertFalse(ok_payload.get("inspection_failed", False))
        self.assertTrue(bad_payload["inspection_failed"])
        self.assertNotEqual(
            bool(ok_payload.get("inspection_failed")),
            bool(bad_payload.get("inspection_failed")),
        )
        self.assertFalse((ok_payload.get("note") or "").strip())

    def test_finalize_flags_unproven_state_when_inspection_raises(self):
        """A raising probe is the same unknown state as a non-zero exit."""
        repo = _make_repo(self.tmp)
        info = sw.create_subagent_worktree(str(repo), "raises")
        assert info is not None

        def _boom(*_a, **_k):
            raise subprocess.TimeoutExpired(cmd="git", timeout=30)

        with mock.patch.object(sw, "_run_git", side_effect=_boom) as m:
            payload = sw.finalize_subagent_worktree(info)

        # Prove the patched seam was actually exercised.
        self.assertGreaterEqual(m.call_count, 1)
        self.assertFalse(payload["pruned"])
        self.assertTrue(payload["inspection_failed"])
        self.assertIn(info["path"], payload["note"])
        self.assertIn(info["branch"], payload["note"])
        self.assertTrue(os.path.isdir(info["path"]))
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


class WorktreePayloadSchemaTests(unittest.TestCase):
    """The parent agent reads ONE schema under ``entry["worktree"]``.

    Two producers write it: ``finalize_subagent_worktree`` and
    ``delegate_tool``'s fallback for when finalize itself raises. Compare the
    fallback against the REAL finalize output (not a hardcoded key list) so
    the two can never drift apart.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hermes-sw-schema-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_fallback_matches_finalize_schema(self):
        from tools import delegate_tool

        repo = _make_repo(self.tmp)
        info = sw.create_subagent_worktree(str(repo), "schema")
        assert info is not None
        happy = sw.finalize_subagent_worktree(info, prune=False)

        # Read the fallback's ACTUAL keys out of the source (not a hardcoded
        # copy), so drift in delegate_tool is what fails this test.
        tree = ast.parse(
            textwrap.dedent(inspect.getsource(delegate_tool._run_single_child))
        )
        fallback_keys = None
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(
                node.value, ast.Dict
            ):
                continue
            target = node.targets[0]
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "worktree"
            ):
                fallback_keys = {
                    k.value
                    for k in node.value.keys
                    if isinstance(k, ast.Constant)
                }
                break

        self.assertIsNotNone(
            fallback_keys,
            "delegate_tool no longer assigns a dict literal to "
            'entry_dict["worktree"] — update this schema guard',
        )
        assert fallback_keys is not None  # narrow for type checkers
        # Fallback must be a superset: every happy-path key plus the two
        # unproven-state keys, and nothing invented. The pre-fix version
        # emitted repo_root/base_commit and omitted commits/dirty/pruned.
        self.assertTrue(set(happy).issubset(fallback_keys))
        self.assertEqual(
            fallback_keys - set(happy), {"inspection_failed", "note"}
        )
        for leaked in ("repo_root", "base_commit"):
            self.assertNotIn(leaked, fallback_keys)


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
