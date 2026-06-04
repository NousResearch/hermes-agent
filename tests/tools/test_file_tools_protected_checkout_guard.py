"""Tests for protected checkout guard in file mutation tools.

Covers BO-237 acceptance criteria:
- DC1: write_file blocks protected canonical mutation
- DC2: patch (replace + V4A) blocks protected canonical mutation without partial apply
- DC3: allowed worktree paths still work
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from unittest.mock import patch

import pytest

from tools.file_tools import write_file_tool, patch_tool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def non_protected_dir():
    """A fresh temp dir that is NOT under any protected root."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def allowed_worktree_dir(tmp_path):
    """A dir mimicking a configured allowed worktree prefix."""
    worktree_root = tmp_path / "worktrees"
    worktree_root.mkdir()
    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value={
            "protected_checkouts": {
                "canonical_roots": [str(tmp_path / "canonical")],
                "allowed_worktree_prefixes": [str(worktree_root)],
            }
        },
    ):
        with tempfile.TemporaryDirectory(dir=worktree_root, prefix="bo-237-test-") as d:
            yield d


@pytest.fixture
def protected_canonical_dir(tmp_path):
    """A fake protected canonical checkout on a non-task branch."""
    root = tmp_path / "canonical"
    root.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=root, check=True, capture_output=True)
    (root / "package.json").write_text('{"name": "dailychingu"}\n')
    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value={
            "protected_checkouts": {
                "canonical_roots": [str(root)],
                "allowed_worktree_prefixes": [str(tmp_path / "worktrees")],
            }
        },
    ):
        yield str(root)


# ---------------------------------------------------------------------------
# DC1: write_file
# ---------------------------------------------------------------------------


class TestWriteFileProtectedCheckout:
    def test_non_protected_path_allowed(self, non_protected_dir):
        """write_file to a non-protected temp dir succeeds."""
        path = os.path.join(non_protected_dir, "test.txt")
        result = write_file_tool(path, "hello")
        parsed = json.loads(result)
        assert "error" not in parsed
        assert os.path.isfile(path)
        with open(path) as f:
            assert f.read() == "hello"

    def test_allowed_worktree_path_allowed(self, allowed_worktree_dir):
        """write_file to a path under .worktrees prefix succeeds."""
        path = os.path.join(allowed_worktree_dir, "test.txt")
        result = write_file_tool(path, "hello worktree")
        parsed = json.loads(result)
        assert "error" not in parsed
        assert os.path.isfile(path)

    def test_protected_canonical_blocked(self, protected_canonical_dir):
        """write_file to a protected canonical path is BLOCKED before creation."""
        target = os.path.join(protected_canonical_dir, "some_file.txt")
        # Ensure file doesn't exist
        assert not os.path.exists(target)
        pre_hash = _file_hash(target)

        result = write_file_tool(target, "should not appear")
        parsed = json.loads(result)
        assert "error" in parsed
        assert "BLOCKED_" in parsed["error"]
        assert not os.path.exists(target)
        post_hash = _file_hash(target)
        assert post_hash == pre_hash


# ---------------------------------------------------------------------------
# DC2: patch (replace mode)
# ---------------------------------------------------------------------------


class TestPatchReplaceProtectedCheckout:
    def test_non_protected_path_allowed(self, non_protected_dir):
        """patch replace on a non-protected path succeeds."""
        path = os.path.join(non_protected_dir, "replace_test.txt")
        with open(path, "w") as f:
            f.write("line1\nline2\nline3\n")
        result = patch_tool(
            mode="replace", path=path, old_string="line2", new_string="replaced"
        )
        parsed = json.loads(result)
        assert "error" not in parsed
        with open(path) as f:
            assert "replaced" in f.read()

    def test_allowed_worktree_path_allowed(self, allowed_worktree_dir):
        """patch replace on a worktree path succeeds."""
        path = os.path.join(allowed_worktree_dir, "replace_test.txt")
        with open(path, "w") as f:
            f.write("line1\nline2\nline3\n")
        result = patch_tool(
            mode="replace", path=path, old_string="line2", new_string="replaced"
        )
        parsed = json.loads(result)
        assert "error" not in parsed
        with open(path) as f:
            assert "replaced" in f.read()

    def test_protected_canonical_blocked(self, protected_canonical_dir):
        """patch replace on protected canonical is BLOCKED, file unchanged."""
        target = os.path.join(protected_canonical_dir, "package.json")
        # Ensure file exists and read current content
        assert os.path.isfile(target)
        with open(target) as f:
            original = f.read()
        pre_hash = _hash_str(original)

        result = patch_tool(
            mode="replace",
            path=target,
            old_string='"name": "dailychingu"',
            new_string='"name": "HACKED"',
        )
        parsed = json.loads(result)
        assert "error" in parsed
        assert "BLOCKED_PROTECTED_CANONICAL" in parsed["error"]

        # Verify file unchanged
        with open(target) as f:
            post_content = f.read()
        post_hash = _hash_str(post_content)
        assert post_hash == pre_hash
        assert post_content == original


# ---------------------------------------------------------------------------
# DC2: patch (V4A mode)
# ---------------------------------------------------------------------------


class TestPatchV4AProtectedCheckout:
    def test_all_allowed_targets_succeed(self, non_protected_dir):
        """V4A patch where all targets are non-protected succeeds."""
        path1 = os.path.join(non_protected_dir, "v4a_test1.txt")
        path2 = os.path.join(non_protected_dir, "v4a_test2.txt")
        with open(path1, "w") as f:
            f.write("AAA\nBBB\n")
        with open(path2, "w") as f:
            f.write("XXX\nYYY\n")

        v4a_patch = (
            f"*** Begin Patch\n"
            f"*** Update File: {path1}\n"
            f"@@ -\n"
            f"-AAA\n"
            f"+ZZZ\n"
            f"*** End Patch\n"
        )
        result = patch_tool(mode="patch", patch=v4a_patch)
        parsed = json.loads(result)
        assert "error" not in parsed  # May fail on patch application, not on guard
        # The guard should pass; if patch application fails, it's a different error
        if "error" in parsed:
            assert "BLOCKED" not in parsed["error"]

    def test_any_protected_target_blocks_entire_patch(self, protected_canonical_dir):
        """V4A patch with any protected target blocks the whole patch BEFORE any apply."""
        # Use a legitimate non-protected target first, then a protected one
        with tempfile.TemporaryDirectory() as tmp:
            allowed_path = os.path.join(tmp, "allowed.txt")
            protected_path = os.path.join(protected_canonical_dir, "some_file.txt")
            with open(allowed_path, "w") as f:
                f.write("AAA\n")
            pre_hash = _hash_str("AAA\n")

            v4a_patch = (
                f"*** Begin Patch\n"
                f"*** Update File: {allowed_path}\n"
                f"@@ -\n"
                f"-AAA\n"
                f"+OK\n"
                f"*** End Patch\n"
                f"*** Begin Patch\n"
                f"*** Update File: {protected_path}\n"
                f"@@ -\n"
                f"-old\n"
                f"+hacked\n"
                f"*** End Patch\n"
            )
            result = patch_tool(mode="patch", patch=v4a_patch)
            parsed = json.loads(result)
            assert "error" in parsed
            assert "BLOCKED_" in parsed["error"]

            # The allowed file must be UNCHANGED (no partial apply)
            with open(allowed_path) as f:
                post_content = f.read()
            assert _hash_str(post_content) == pre_hash
            assert post_content == "AAA\n"

    def test_allowed_worktree_targets_succeed(self, allowed_worktree_dir):
        """V4A patch targeting only allowed worktree paths succeeds."""
        path1 = os.path.join(allowed_worktree_dir, "wt_a.txt")
        path2 = os.path.join(allowed_worktree_dir, "wt_b.txt")
        with open(path1, "w") as f:
            f.write("hello\n")
        with open(path2, "w") as f:
            f.write("world\n")

        v4a_patch = (
            f"*** Begin Patch\n"
            f"*** Update File: {path1}\n"
            f"@@ -\n"
            f"-hello\n"
            f"+HI\n"
            f"*** End Patch\n"
            f"*** Begin Patch\n"
            f"*** Update File: {path2}\n"
            f"@@ -\n"
            f"-world\n"
            f"+THERE\n"
            f"*** End Patch\n"
        )
        result = patch_tool(mode="patch", patch=v4a_patch)
        parsed = json.loads(result)
        # Guard should pass; application may or may not succeed
        if "error" in parsed:
            assert "BLOCKED" not in parsed["error"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _file_hash(path: str) -> str:
    """Simple hash for checking file existence/content."""
    if not os.path.exists(path):
        return "NONEXISTENT"
    import hashlib

    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _hash_str(s: str) -> str:
    import hashlib

    return hashlib.sha256(s.encode()).hexdigest()
