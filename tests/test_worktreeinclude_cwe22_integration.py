"""Integration tests: call the REAL cli._setup_worktree and verify CWE-22 fix.

These tests import the actual _setup_worktree from cli.py (not a local
re-implementation) and verify that path-traversal entries in .worktreeinclude
are rejected while legitimate entries still work.
"""

import os
import subprocess
import pytest
from pathlib import Path


@pytest.fixture
def git_repo(tmp_path):
    """Create a temporary git repo with an initial commit."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo, capture_output=True,
    )
    (repo / "README.md").write_text("# Test\n")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=repo, capture_output=True, check=True
    )
    return repo


class TestRealSetupWorktreePathTraversal:
    """Exercise the real _setup_worktree from cli.py to verify CWE-22 fix."""

    def test_traversal_entry_rejected(self, git_repo, tmp_path):
        """A ../secret.txt entry must NOT be copied into the worktree."""
        from cli import _setup_worktree

        # Create a sensitive file OUTSIDE the repo
        secret = tmp_path / "secret.txt"
        secret.write_text("TOP SECRET DATA")

        # .worktreeinclude references it via path traversal
        rel = os.path.relpath(str(secret), str(git_repo))
        assert ".." in rel  # sanity: it really escapes the repo
        (git_repo / ".worktreeinclude").write_text(rel + "\n")

        info = _setup_worktree(str(git_repo))
        assert info is not None

        wt = Path(info["path"])
        # The secret must NOT appear anywhere in the worktree
        for root, _dirs, files in os.walk(str(wt)):
            for f in files:
                content = Path(root, f).read_text(errors="ignore")
                assert "TOP SECRET DATA" not in content, (
                    f"Sensitive data leaked into worktree file {Path(root, f)}"
                )

    def test_absolute_path_entry_rejected(self, git_repo, tmp_path):
        """/etc/passwd-style absolute paths must be rejected."""
        from cli import _setup_worktree

        # Create a file to simulate an absolute path target
        target = tmp_path / "abs_target.txt"
        target.write_text("ABSOLUTE LEAK")

        (git_repo / ".worktreeinclude").write_text(str(target) + "\n")

        info = _setup_worktree(str(git_repo))
        assert info is not None

        wt = Path(info["path"])
        for root, _dirs, files in os.walk(str(wt)):
            for f in files:
                content = Path(root, f).read_text(errors="ignore")
                assert "ABSOLUTE LEAK" not in content

    def test_double_dot_chain_rejected(self, git_repo, tmp_path):
        """subdir/../../outside should be rejected."""
        from cli import _setup_worktree

        outside = tmp_path / "chain_target.txt"
        outside.write_text("CHAIN ESCAPED")

        # Craft entry: subdir/../<relative-to-repo>
        rel = os.path.relpath(str(outside), str(git_repo))
        entry = f"subdir/{rel}"
        (git_repo / ".worktreeinclude").write_text(entry + "\n")

        info = _setup_worktree(str(git_repo))
        assert info is not None

        wt = Path(info["path"])
        for root, _dirs, files in os.walk(str(wt)):
            for f in files:
                content = Path(root, f).read_text(errors="ignore")
                assert "CHAIN ESCAPED" not in content

    def test_valid_file_still_copied(self, git_repo):
        """Legitimate entries within the repo must still be copied."""
        from cli import _setup_worktree

        (git_repo / ".env").write_text("KEY=safe_value")
        (git_repo / ".worktreeinclude").write_text(".env\n")

        info = _setup_worktree(str(git_repo))
        assert info is not None

        wt = Path(info["path"])
        env_file = wt / ".env"
        assert env_file.exists(), ".env should be copied to worktree"
        assert env_file.read_text() == "KEY=safe_value"

    def test_valid_directory_symlinked(self, git_repo):
        """Valid directories should still be symlinked."""
        from cli import _setup_worktree

        venv = git_repo / ".venv" / "lib"
        venv.mkdir(parents=True)
        (venv / "marker.txt").write_text("ok")
        (git_repo / ".worktreeinclude").write_text(".venv\n")

        info = _setup_worktree(str(git_repo))
        assert info is not None

        wt = Path(info["path"])
        assert (wt / ".venv").exists()
        assert (wt / ".venv").is_symlink()
        assert (wt / ".venv" / "lib" / "marker.txt").read_text() == "ok"

    def test_symlink_based_traversal_rejected(self, git_repo, tmp_path):
        """A symlink inside the repo pointing outside should be rejected.

        When we resolve() the path, the symlink is followed, and the resolved
        path is outside the repo root → should be skipped.
        """
        from cli import _setup_worktree

        # Place a sensitive file outside
        sensitive = tmp_path / "sensitive_via_symlink.txt"
        sensitive.write_text("SYMLINK LEAK")

        # Create a symlink inside the repo pointing to the sensitive file
        link = git_repo / "evil_link"
        os.symlink(str(sensitive), str(link))
        assert link.resolve() == sensitive.resolve()

        (git_repo / ".worktreeinclude").write_text("evil_link\n")

        info = _setup_worktree(str(git_repo))
        assert info is not None

        wt = Path(info["path"])
        # The symlink target (sensitive file) should NOT be copied
        for root, _dirs, files in os.walk(str(wt)):
            for f in files:
                fpath = Path(root, f)
                if fpath.is_symlink():
                    # If a symlink was created, it shouldn't point outside
                    target = fpath.resolve()
                    wt_resolved = wt.resolve()
                    assert str(target).startswith(str(wt_resolved) + os.sep) or target == wt_resolved, (
                        f"Symlink {fpath} points outside worktree to {target}"
                    )
                else:
                    content = fpath.read_text(errors="ignore")
                    assert "SYMLINK LEAK" not in content

    def test_mixed_valid_and_malicious(self, git_repo, tmp_path):
        """Valid entries should be processed; malicious ones skipped."""
        from cli import _setup_worktree

        # Valid file
        (git_repo / ".env").write_text("GOOD=data")
        # Malicious traversal
        outside = tmp_path / "stolen.txt"
        outside.write_text("STOLEN")
        rel = os.path.relpath(str(outside), str(git_repo))

        (git_repo / ".worktreeinclude").write_text(
            f".env\n{rel}\n"
        )

        info = _setup_worktree(str(git_repo))
        assert info is not None

        wt = Path(info["path"])
        # Valid file should be copied
        assert (wt / ".env").read_text() == "GOOD=data"
        # Stolen file should NOT be anywhere
        for root, _dirs, files in os.walk(str(wt)):
            for f in files:
                content = Path(root, f).read_text(errors="ignore")
                assert "STOLEN" not in content
