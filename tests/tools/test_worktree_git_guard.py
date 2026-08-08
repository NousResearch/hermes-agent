"""Tests for the git worktree ``.git`` write guard in write_file/patch (#78565).

In a git worktree, ``.git`` is a FILE containing a single line like
``gitdir: /path/to/bare`` — not a directory.  ``write_file``/``patch``
auto-create parent directories and atomically move content into place, so a
write whose path touches that file (``<worktree>/.git`` itself, or anything
under it) replaces the link file and destroys the worktree checkout.

These tests pin the guard:
  * writes into ``.git`` paths inside a worktree are refused with a clear
    error and the link file survives untouched;
  * normal writes keep working (including ``.gitignore`` / ``.github`` paths
    that merely contain a ``.git``-prefixed component);
  * read-only access (read_file / search_files) to ``.git`` paths stays
    allowed;
  * a real git worktree (bare repo + worktree in tmp) is used to prove the
    ``.git`` link file survives a refused write.
"""

import json
import os
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tools.file_tools import patch_tool, read_file_tool, search_tool, write_file_tool


def _make_fake_worktree(tmp_path, name="wt"):
    """Create a directory that looks like a git worktree: a ``.git`` FILE
    containing a ``gitdir:`` pointer (the worktree link) plus a normal file."""
    wt = tmp_path / name
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {tmp_path}/bare.git/worktrees/{name}\n", encoding="utf-8")
    (wt / "hello.txt").write_text("hello\n", encoding="utf-8")
    return wt


def _make_real_worktree(tmp_path):
    """Create a REAL git worktree (bare repo + linked worktree) under tmp.

    Returns the worktree directory; ``<wt>/.git`` is the ``gitdir:`` link file.
    """
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "readme.md").write_text("seed\n", encoding="utf-8")
    env = {
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    subprocess.run(["git", "init", "-q", str(seed)], check=True)
    subprocess.run(["git", "-C", str(seed), "add", "readme.md"], check=True)
    subprocess.run(["git", "-C", str(seed), "commit", "-q", "-m", "seed"], check=True, env=env)
    bare = tmp_path / "bare.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(seed), str(bare)], check=True)
    wt = tmp_path / "worktree"
    subprocess.run(["git", "worktree", "add", "-q", str(wt)], check=True, cwd=str(bare))
    assert (wt / ".git").is_file()
    return wt


class TestWriteFileWorktreeGitGuard:
    def test_write_into_git_config_refused(self, tmp_path):
        wt = _make_fake_worktree(tmp_path)
        result = json.loads(write_file_tool(str(wt / ".git" / "config"), "evil"))
        assert "error" in result
        assert "worktree" in result["error"].lower()
        assert ".git" in result["error"]
        # The link file is untouched.
        assert (wt / ".git").read_text(encoding="utf-8").startswith("gitdir: ")

    def test_write_to_git_file_itself_refused(self, tmp_path):
        wt = _make_fake_worktree(tmp_path)
        result = json.loads(write_file_tool(str(wt / ".git"), "broken"))
        assert "error" in result
        assert "worktree" in result["error"].lower()
        assert (wt / ".git").read_text(encoding="utf-8").startswith("gitdir: ")

    def test_write_deep_under_git_refused(self, tmp_path):
        wt = _make_fake_worktree(tmp_path)
        result = json.loads(write_file_tool(str(wt / ".git" / "objects" / "ab" / "cdef"), "x"))
        assert "error" in result
        assert "worktree" in result["error"].lower()

    @patch("tools.file_tools._get_file_ops")
    def test_write_normal_path_still_works(self, mock_get, tmp_path):
        wt = _make_fake_worktree(tmp_path)
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"status": "ok", "path": str(wt / "out.txt"), "bytes": 2}
        mock_ops.write_file.return_value = result_obj
        mock_get.return_value = mock_ops
        result = json.loads(write_file_tool(str(wt / "out.txt"), "ok"))
        assert result["status"] == "ok"
        mock_ops.write_file.assert_called_once()

    @patch("tools.file_tools._get_file_ops")
    def test_gitignore_and_github_components_not_blocked(self, mock_get, tmp_path):
        # ".gitignore" / ".github" contain a ".git"-prefixed component but are
        # NOT the ".git" component — they must keep working.
        wt = _make_fake_worktree(tmp_path)
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"status": "ok"}
        mock_ops.write_file.return_value = result_obj
        mock_get.return_value = mock_ops
        for rel in (".gitignore", ".github/workflows/ci.yml"):
            res = json.loads(write_file_tool(str(wt / rel), "data"))
            assert res.get("status") == "ok", rel
        assert mock_ops.write_file.call_count == 2

    @patch("tools.file_tools._get_file_ops")
    def test_plain_git_directory_repo_not_blocked(self, mock_get, tmp_path):
        # Normal repo: .git is a DIRECTORY, not a gitdir: link file — outside
        # the worktree guard's scope (issue #78565 targets worktrees only).
        repo = tmp_path / "repo"
        (repo / ".git").mkdir(parents=True)
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"status": "ok"}
        mock_ops.write_file.return_value = result_obj
        mock_get.return_value = mock_ops
        result = json.loads(write_file_tool(str(repo / ".git" / "config"), "x"))
        assert result["status"] == "ok"


class TestPatchWorktreeGitGuard:
    @patch("tools.file_tools._get_file_ops")
    def test_replace_mode_into_git_refused(self, mock_get, tmp_path):
        wt = _make_fake_worktree(tmp_path)
        result = json.loads(
            patch_tool(
                mode="replace",
                path=str(wt / ".git" / "config"),
                old_string="a",
                new_string="b",
            )
        )
        assert "error" in result
        assert "worktree" in result["error"].lower()
        mock_get.assert_not_called()

    @patch("tools.file_tools._get_file_ops")
    def test_v4a_mode_update_git_refused(self, mock_get, tmp_path):
        wt = _make_fake_worktree(tmp_path)
        v4a = (
            "*** Begin Patch\n"
            f"*** Update File: {wt}/.git/config\n"
            "@@\n"
            "-old\n"
            "+new\n"
            "*** End Patch\n"
        )
        result = json.loads(patch_tool(mode="patch", patch=v4a))
        assert "error" in result
        assert "worktree" in result["error"].lower()
        mock_get.assert_not_called()

    @patch("tools.file_tools._get_file_ops")
    def test_v4a_mode_add_under_git_refused(self, mock_get, tmp_path):
        wt = _make_fake_worktree(tmp_path)
        v4a = (
            "*** Begin Patch\n"
            f"*** Add File: {wt}/.git/hooks/pre-commit\n"
            "+#!/bin/sh\n"
            "*** End Patch\n"
        )
        result = json.loads(patch_tool(mode="patch", patch=v4a))
        assert "error" in result
        assert "worktree" in result["error"].lower()
        mock_get.assert_not_called()

    @patch("tools.file_tools._get_file_ops")
    def test_replace_mode_normal_file_works(self, mock_get, tmp_path):
        wt = _make_fake_worktree(tmp_path)
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"status": "ok"}
        mock_ops.patch_replace.return_value = result_obj
        mock_get.return_value = mock_ops
        result = json.loads(
            patch_tool(
                mode="replace",
                path=str(wt / "hello.txt"),
                old_string="hello",
                new_string="bye",
            )
        )
        assert result["status"] == "ok"
        mock_ops.patch_replace.assert_called_once()


class TestReadToolsUnaffected:
    @patch("tools.file_tools._get_file_ops")
    def test_read_file_can_read_git_link(self, mock_get, tmp_path):
        wt = _make_fake_worktree(tmp_path)
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.content = "gitdir: /tmp/bare\n"
        result_obj.total_lines = 1
        result_obj.to_dict.return_value = {"content": "gitdir: /tmp/bare\n", "total_lines": 1}
        mock_ops.read_file.return_value = result_obj
        mock_get.return_value = mock_ops
        result = json.loads(read_file_tool(str(wt / ".git")))
        assert "error" not in result
        assert "gitdir" in result.get("content", "")
        mock_ops.read_file.assert_called_once()

    @patch("tools.file_tools._get_file_ops")
    def test_search_files_can_search_git_dir(self, mock_get, tmp_path):
        wt = _make_fake_worktree(tmp_path)
        fake_result = SimpleNamespace(
            matches=[],
            to_dict=lambda densify=False: {"matches": [], "total": 0},
        )
        mock_ops = MagicMock()
        mock_ops.search.return_value = fake_result
        mock_get.return_value = mock_ops
        result = json.loads(search_tool(pattern="*", target="files", path=str(wt / ".git")))
        assert "error" not in result
        mock_ops.search.assert_called_once()


class TestRealWorktree:
    def test_real_worktree_git_file_survives_refused_write(self, tmp_path):
        wt = _make_real_worktree(tmp_path)
        before = (wt / ".git").read_text(encoding="utf-8")
        assert before.lstrip().startswith("gitdir: ")

        with patch("tools.file_tools._get_file_ops") as mock_get:
            result = json.loads(write_file_tool(str(wt / ".git" / "config"), "clobber"))
        assert "error" in result
        assert "worktree" in result["error"].lower()
        mock_get.assert_not_called()
        # The worktree link survived the refused write.
        assert (wt / ".git").is_file()
        assert (wt / ".git").read_text(encoding="utf-8") == before
        # The worktree is still a usable git checkout.
        subprocess.run(["git", "status", "--porcelain"], check=True, cwd=str(wt))

    @patch("tools.file_tools._get_file_ops")
    def test_real_worktree_normal_write_still_works(self, mock_get, tmp_path):
        wt = _make_real_worktree(tmp_path)
        mock_ops = MagicMock()
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {"status": "ok"}
        mock_ops.write_file.return_value = result_obj
        mock_get.return_value = mock_ops
        result = json.loads(write_file_tool(str(wt / "notes.md"), "note"))
        assert result["status"] == "ok"
        mock_ops.write_file.assert_called_once()
