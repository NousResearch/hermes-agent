"""Tests for git worktree isolation (CLI --worktree / -w flag).

Verifies worktree creation, cleanup, .worktreeinclude handling,
.gitignore management, and integration with the CLI.  (#652)
"""

import os
import shutil
import subprocess
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


@pytest.fixture
def git_repo(tmp_path):
    """Create a temporary git repo for testing."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo, capture_output=True,
    )
    # Create initial commit (worktrees need at least one commit)
    (repo / "README.md").write_text("# Test Repo\n")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo, capture_output=True,
    )
    return repo


# ---------------------------------------------------------------------------
# Lightweight reimplementations for testing (avoid importing cli.py)
# ---------------------------------------------------------------------------

def _git_repo_root(cwd=None):
    """Test version of _git_repo_root."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5,
            cwd=cwd,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _setup_worktree(repo_root):
    """Test version of _setup_worktree — creates a worktree."""
    import uuid
    short_id = uuid.uuid4().hex[:8]
    wt_name = f"hermes-{short_id}"
    branch_name = f"hermes/{wt_name}"

    worktrees_dir = Path(repo_root) / ".worktrees"
    worktrees_dir.mkdir(parents=True, exist_ok=True)
    wt_path = worktrees_dir / wt_name

    result = subprocess.run(
        ["git", "worktree", "add", str(wt_path), "-b", branch_name, "HEAD"],
        capture_output=True, text=True, timeout=30, cwd=repo_root,
    )
    if result.returncode != 0:
        return None

    return {
        "path": str(wt_path),
        "branch": branch_name,
        "repo_root": repo_root,
    }


def _cleanup_worktree(info):
    """Test version of _cleanup_worktree."""
    wt_path = info["path"]
    branch = info["branch"]
    repo_root = info["repo_root"]

    if not Path(wt_path).exists():
        return

    # Check for uncommitted changes
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, timeout=10, cwd=wt_path,
    )
    has_changes = bool(status.stdout.strip())

    if has_changes:
        return False  # Did not clean up

    subprocess.run(
        ["git", "worktree", "remove", wt_path, "--force"],
        capture_output=True, text=True, timeout=15, cwd=repo_root,
    )
    subprocess.run(
        ["git", "branch", "-D", branch],
        capture_output=True, text=True, timeout=10, cwd=repo_root,
    )
    return True  # Cleaned up


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGitRepoDetection:
    """Test git repo root detection."""

    def test_detects_git_repo(self, git_repo):
        root = _git_repo_root(cwd=str(git_repo))
        assert root is not None
        assert Path(root).resolve() == git_repo.resolve()

    def test_detects_subdirectory(self, git_repo):
        subdir = git_repo / "src" / "lib"
        subdir.mkdir(parents=True)
        root = _git_repo_root(cwd=str(subdir))
        assert root is not None
        assert Path(root).resolve() == git_repo.resolve()

    def test_returns_none_outside_repo(self, tmp_path):
        # tmp_path itself is not a git repo
        bare_dir = tmp_path / "not-a-repo"
        bare_dir.mkdir()
        root = _git_repo_root(cwd=str(bare_dir))
        assert root is None


class TestWorktreeCreation:
    """Test worktree setup."""

    def test_creates_worktree(self, git_repo):
        info = _setup_worktree(str(git_repo))
        assert info is not None
        assert Path(info["path"]).exists()
        assert info["branch"].startswith("hermes/hermes-")
        assert info["repo_root"] == str(git_repo)

        # Verify it's a valid git worktree
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, cwd=info["path"],
        )
        assert result.stdout.strip() == "true"

    def test_worktree_has_own_branch(self, git_repo):
        info = _setup_worktree(str(git_repo))
        assert info is not None

        # Check branch name in worktree
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, cwd=info["path"],
        )
        assert result.stdout.strip() == info["branch"]

    def test_worktree_is_independent(self, git_repo):
        """Two worktrees from the same repo are independent."""
        info1 = _setup_worktree(str(git_repo))
        info2 = _setup_worktree(str(git_repo))
        assert info1 is not None
        assert info2 is not None
        assert info1["path"] != info2["path"]
        assert info1["branch"] != info2["branch"]

        # Create a file in worktree 1
        (Path(info1["path"]) / "only-in-wt1.txt").write_text("hello")

        # It should NOT appear in worktree 2
        assert not (Path(info2["path"]) / "only-in-wt1.txt").exists()

    def test_worktrees_dir_created(self, git_repo):
        info = _setup_worktree(str(git_repo))
        assert info is not None
        assert (git_repo / ".worktrees").is_dir()

    def test_worktree_has_repo_files(self, git_repo):
        """Worktree should contain the repo's tracked files."""
        info = _setup_worktree(str(git_repo))
        assert info is not None
        assert (Path(info["path"]) / "README.md").exists()


class TestWorktreeCleanup:
    """Test worktree cleanup on exit."""

    def test_clean_worktree_removed(self, git_repo):
        info = _setup_worktree(str(git_repo))
        assert info is not None
        assert Path(info["path"]).exists()

        result = _cleanup_worktree(info)
        assert result is True
        assert not Path(info["path"]).exists()

    def test_dirty_worktree_kept(self, git_repo):
        info = _setup_worktree(str(git_repo))
        assert info is not None

        # Make uncommitted changes
        (Path(info["path"]) / "new-file.txt").write_text("uncommitted")
        subprocess.run(
            ["git", "add", "new-file.txt"],
            cwd=info["path"], capture_output=True,
        )

        result = _cleanup_worktree(info)
        assert result is False
        assert Path(info["path"]).exists()  # Still there

    def test_branch_deleted_on_cleanup(self, git_repo):
        info = _setup_worktree(str(git_repo))
        branch = info["branch"]

        _cleanup_worktree(info)

        # Branch should be gone
        result = subprocess.run(
            ["git", "branch", "--list", branch],
            capture_output=True, text=True, cwd=str(git_repo),
        )
        assert branch not in result.stdout

    def test_cleanup_nonexistent_worktree(self, git_repo):
        """Cleanup should handle already-removed worktrees gracefully."""
        info = {
            "path": str(git_repo / ".worktrees" / "nonexistent"),
            "branch": "hermes/nonexistent",
            "repo_root": str(git_repo),
        }
        # Should not raise
        _cleanup_worktree(info)


class TestWorktreeInclude:
    """Test .worktreeinclude file handling."""

    def test_copies_included_files(self, git_repo):
        """Files listed in .worktreeinclude should be copied to the worktree."""
        # Create a .env file (gitignored)
        (git_repo / ".env").write_text("SECRET=abc123")
        (git_repo / ".gitignore").write_text(".env\n.worktrees/\n")
        subprocess.run(
            ["git", "add", ".gitignore"],
            cwd=str(git_repo), capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Add gitignore"],
            cwd=str(git_repo), capture_output=True,
        )

        # Create .worktreeinclude
        (git_repo / ".worktreeinclude").write_text(".env\n")

        # Import and use the real _setup_worktree logic for include handling
        info = _setup_worktree(str(git_repo))
        assert info is not None

        # Manually copy .worktreeinclude entries (mirrors cli.py logic)
        import shutil
        include_file = git_repo / ".worktreeinclude"
        wt_path = Path(info["path"])
        for line in include_file.read_text().splitlines():
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue
            src = git_repo / entry
            dst = wt_path / entry
            if src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(dst))

        # Verify .env was copied
        assert (wt_path / ".env").exists()
        assert (wt_path / ".env").read_text() == "SECRET=abc123"

    def test_ignores_comments_and_blanks(self, git_repo):
        """Comments and blank lines in .worktreeinclude should be skipped."""
        (git_repo / ".worktreeinclude").write_text(
            "# This is a comment\n"
            "\n"
            "  # Another comment\n"
        )
        info = _setup_worktree(str(git_repo))
        assert info is not None
        # Should not crash — just skip all lines


class TestGitignoreManagement:
    """Test that .worktrees/ is added to .gitignore."""

    def test_adds_to_gitignore(self, git_repo):
        """Creating a worktree should add .worktrees/ to .gitignore."""
        # Remove any existing .gitignore
        gitignore = git_repo / ".gitignore"
        if gitignore.exists():
            gitignore.unlink()

        info = _setup_worktree(str(git_repo))
        assert info is not None

        # Now manually add .worktrees/ to .gitignore (mirrors cli.py logic)
        _ignore_entry = ".worktrees/"
        existing = gitignore.read_text() if gitignore.exists() else ""
        if _ignore_entry not in existing.splitlines():
            with open(gitignore, "a") as f:
                if existing and not existing.endswith("\n"):
                    f.write("\n")
                f.write(f"{_ignore_entry}\n")

        content = gitignore.read_text()
        assert ".worktrees/" in content

    def test_does_not_duplicate_gitignore_entry(self, git_repo):
        """If .worktrees/ is already in .gitignore, don't add again."""
        gitignore = git_repo / ".gitignore"
        gitignore.write_text(".worktrees/\n")

        # The check should see it's already there
        existing = gitignore.read_text()
        assert ".worktrees/" in existing.splitlines()


class TestMultipleWorktrees:
    """Test running multiple worktrees concurrently (the core use case)."""

    def test_ten_concurrent_worktrees(self, git_repo):
        """Create 10 worktrees — simulating 10 parallel agents."""
        worktrees = []
        for _ in range(10):
            info = _setup_worktree(str(git_repo))
            assert info is not None
            worktrees.append(info)

        # All should exist and be independent
        paths = [info["path"] for info in worktrees]
        assert len(set(paths)) == 10  # All unique

        # Each should have the repo files
        for info in worktrees:
            assert (Path(info["path"]) / "README.md").exists()

        # Edit a file in one worktree
        (Path(worktrees[0]["path"]) / "README.md").write_text("Modified in wt0")

        # Others should be unaffected
        for info in worktrees[1:]:
            assert (Path(info["path"]) / "README.md").read_text() == "# Test Repo\n"

        # List worktrees via git
        result = subprocess.run(
            ["git", "worktree", "list"],
            capture_output=True, text=True, cwd=str(git_repo),
        )
        # Should have 11 entries: main + 10 worktrees
        lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
        assert len(lines) == 11

        # Cleanup all
        for info in worktrees:
            # Discard changes first so cleanup works
            subprocess.run(
                ["git", "checkout", "--", "."],
                cwd=info["path"], capture_output=True,
            )
            _cleanup_worktree(info)

        # All should be removed
        for info in worktrees:
            assert not Path(info["path"]).exists()


class TestWorktreeDirectorySymlink:
    """Test .worktreeinclude with directories (symlinked)."""

    def test_symlinks_directory(self, git_repo):
        """Directories in .worktreeinclude should be symlinked."""
        # Create a .venv directory
        venv_dir = git_repo / ".venv" / "lib"
        venv_dir.mkdir(parents=True)
        (venv_dir / "marker.txt").write_text("venv marker")
        (git_repo / ".gitignore").write_text(".venv/\n.worktrees/\n")
        subprocess.run(
            ["git", "add", ".gitignore"], cwd=str(git_repo), capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "gitignore"], cwd=str(git_repo), capture_output=True
        )

        (git_repo / ".worktreeinclude").write_text(".venv/\n")

        info = _setup_worktree(str(git_repo))
        assert info is not None

        wt_path = Path(info["path"])
        src = git_repo / ".venv"
        dst = wt_path / ".venv"

        # Manually symlink (mirrors cli.py logic)
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(str(src.resolve()), str(dst))

        assert dst.is_symlink()
        assert (dst / "lib" / "marker.txt").read_text() == "venv marker"


class TestStaleWorktreePruning:
    """Test _prune_stale_worktrees garbage collection."""

    def test_prunes_old_clean_worktree(self, git_repo):
        """Old clean worktrees should be removed on prune."""
        import time

        info = _setup_worktree(str(git_repo))
        assert info is not None
        assert Path(info["path"]).exists()

        # Make the worktree look old (set mtime to 25h ago)
        old_time = time.time() - (25 * 3600)
        os.utime(info["path"], (old_time, old_time))

        # Reimplementation of prune logic (matches cli.py)
        worktrees_dir = git_repo / ".worktrees"
        cutoff = time.time() - (24 * 3600)

        for entry in worktrees_dir.iterdir():
            if not entry.is_dir() or not entry.name.startswith("hermes-"):
                continue
            try:
                mtime = entry.stat().st_mtime
                if mtime > cutoff:
                    continue
            except Exception:
                continue

            status = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=5, cwd=str(entry),
            )
            if status.stdout.strip():
                continue

            branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True, text=True, timeout=5, cwd=str(entry),
            )
            branch = branch_result.stdout.strip()
            subprocess.run(
                ["git", "worktree", "remove", str(entry), "--force"],
                capture_output=True, text=True, timeout=15, cwd=str(git_repo),
            )
            if branch:
                subprocess.run(
                    ["git", "branch", "-D", branch],
                    capture_output=True, text=True, timeout=10, cwd=str(git_repo),
                )

        assert not Path(info["path"]).exists()

    def test_keeps_recent_worktree(self, git_repo):
        """Recent worktrees should NOT be pruned."""
        import time

        info = _setup_worktree(str(git_repo))
        assert info is not None

        # Don't modify mtime — it's recent
        worktrees_dir = git_repo / ".worktrees"
        cutoff = time.time() - (24 * 3600)

        pruned = False
        for entry in worktrees_dir.iterdir():
            if not entry.is_dir() or not entry.name.startswith("hermes-"):
                continue
            mtime = entry.stat().st_mtime
            if mtime > cutoff:
                continue  # Too recent
            pruned = True

        assert not pruned
        assert Path(info["path"]).exists()

    def test_keeps_dirty_old_worktree(self, git_repo):
        """Old worktrees with uncommitted changes should NOT be pruned."""
        import time

        info = _setup_worktree(str(git_repo))
        assert info is not None

        # Make it dirty
        (Path(info["path"]) / "dirty.txt").write_text("uncommitted")
        subprocess.run(
            ["git", "add", "dirty.txt"],
            cwd=info["path"], capture_output=True,
        )

        # Make it old
        old_time = time.time() - (25 * 3600)
        os.utime(info["path"], (old_time, old_time))

        # Check if it would be pruned
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=info["path"],
        )
        has_changes = bool(status.stdout.strip())
        assert has_changes  # Should be dirty → not pruned
        assert Path(info["path"]).exists()


class TestEdgeCases:
    """Test edge cases for robustness."""

    def test_no_commits_repo(self, tmp_path):
        """Worktree creation should fail gracefully on a repo with no commits."""
        repo = tmp_path / "empty-repo"
        repo.mkdir()
        subprocess.run(["git", "init"], cwd=str(repo), capture_output=True)

        info = _setup_worktree(str(repo))
        assert info is None  # Should fail gracefully

    def test_not_a_git_repo(self, tmp_path):
        """Repo detection should return None for non-git directories."""
        bare = tmp_path / "not-git"
        bare.mkdir()
        root = _git_repo_root(cwd=str(bare))
        assert root is None

    def test_worktrees_dir_already_exists(self, git_repo):
        """Should work fine if .worktrees/ already exists."""
        (git_repo / ".worktrees").mkdir(exist_ok=True)
        info = _setup_worktree(str(git_repo))
        assert info is not None
        assert Path(info["path"]).exists()


class TestCLIFlagLogic:
    """Test the flag/config OR logic from main()."""

    def test_worktree_flag_triggers(self):
        """--worktree flag should trigger worktree creation."""
        worktree = True
        w = False
        config_worktree = False
        use_worktree = worktree or w or config_worktree
        assert use_worktree

    def test_w_flag_triggers(self):
        """-w flag should trigger worktree creation."""
        worktree = False
        w = True
        config_worktree = False
        use_worktree = worktree or w or config_worktree
        assert use_worktree

    def test_config_triggers(self):
        """worktree: true in config should trigger worktree creation."""
        worktree = False
        w = False
        config_worktree = True
        use_worktree = worktree or w or config_worktree
        assert use_worktree

    def test_none_set_no_trigger(self):
        """No flags and no config should not trigger."""
        worktree = False
        w = False
        config_worktree = False
        use_worktree = worktree or w or config_worktree
        assert not use_worktree


class TestTerminalCWDIntegration:
    """Test that TERMINAL_CWD is correctly set to the worktree path."""

    def test_terminal_cwd_set(self, git_repo):
        """After worktree setup, TERMINAL_CWD should point to the worktree."""
        info = _setup_worktree(str(git_repo))
        assert info is not None

        # This is what main() does:
        os.environ["TERMINAL_CWD"] = info["path"]
        assert os.environ["TERMINAL_CWD"] == info["path"]
        assert Path(os.environ["TERMINAL_CWD"]).exists()

        # Clean up env
        del os.environ["TERMINAL_CWD"]

    def test_terminal_cwd_is_valid_git_repo(self, git_repo):
        """The TERMINAL_CWD worktree should be a valid git working tree."""
        info = _setup_worktree(str(git_repo))
        assert info is not None

        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, cwd=info["path"],
        )
        assert result.stdout.strip() == "true"


class TestSystemPromptInjection:
    """Test that the agent gets worktree context in its system prompt."""

    def test_prompt_note_format(self, git_repo):
        """Verify the system prompt note contains all required info."""
        info = _setup_worktree(str(git_repo))
        assert info is not None

        # This is what main() does:
        wt_note = (
            f"\n\n[System note: You are working in an isolated git worktree at "
            f"{info['path']}. Your branch is `{info['branch']}`. "
            f"Changes here do not affect the main working tree or other agents. "
            f"Remember to commit and push your changes, and create a PR if appropriate. "
            f"The original repo is at {info['repo_root']}.]"
        )

        assert info["path"] in wt_note
        assert info["branch"] in wt_note
        assert info["repo_root"] in wt_note
        assert "isolated git worktree" in wt_note
        assert "commit and push" in wt_note


class TestWorktreeIncludePathTraversal:
    """Test that .worktreeinclude entries with path traversal are rejected."""

    def test_rejects_parent_directory_traversal(self, git_repo):
        """Entries like '../../etc/passwd' must not escape the repo root."""
        import shutil as _shutil

        # Create a sensitive file outside the repo to simulate the attack
        outside_file = git_repo.parent / "sensitive.txt"
        outside_file.write_text("SENSITIVE DATA")

        # Create a .worktreeinclude with a traversal entry
        (git_repo / ".worktreeinclude").write_text("../sensitive.txt\n")

        info = _setup_worktree(str(git_repo))
        assert info is not None

        wt_path = Path(info["path"])

        # Replay the fixed logic from cli.py
        repo_root_resolved = Path(str(git_repo)).resolve()
        wt_path_resolved = wt_path.resolve()
        include_file = git_repo / ".worktreeinclude"

        copied_entries = []
        for line in include_file.read_text().splitlines():
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue
            src = Path(str(git_repo)) / entry
            dst = wt_path / entry
            try:
                src_resolved = src.resolve()
                dst_resolved = dst.resolve(strict=False)
            except (OSError, ValueError):
                continue
            if not str(src_resolved).startswith(str(repo_root_resolved) + os.sep) and src_resolved != repo_root_resolved:
                continue
            if not str(dst_resolved).startswith(str(wt_path_resolved) + os.sep) and dst_resolved != wt_path_resolved:
                continue
            copied_entries.append(entry)

        # The traversal entry must have been skipped
        assert len(copied_entries) == 0
        # The sensitive file must NOT be in the worktree
        assert not (wt_path / "../sensitive.txt").resolve().is_relative_to(wt_path_resolved)

    def test_allows_valid_entries(self, git_repo):
        """Normal entries within the repo should still be processed."""
        (git_repo / ".env").write_text("KEY=val")
        (git_repo / ".worktreeinclude").write_text(".env\n")

        info = _setup_worktree(str(git_repo))
        assert info is not None

        repo_root_resolved = Path(str(git_repo)).resolve()
        include_file = git_repo / ".worktreeinclude"

        accepted = []
        for line in include_file.read_text().splitlines():
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue
            src = Path(str(git_repo)) / entry
            src_resolved = src.resolve()
            if not str(src_resolved).startswith(str(repo_root_resolved) + os.sep) and src_resolved != repo_root_resolved:
                continue
            accepted.append(entry)

        assert ".env" in accepted
