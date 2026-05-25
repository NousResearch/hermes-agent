"""Tests for tools.path_security — path validation helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.path_security import has_traversal_component, validate_within_dir


# ============================================================================
# has_traversal_component
# ============================================================================
class TestHasTraversalComponent:
    def test_no_traversal(self):
        assert has_traversal_component("foo/bar/baz.txt") is False

    def test_single_dot_dot(self):
        assert has_traversal_component("../escape") is True

    def test_mid_path_traversal(self):
        assert has_traversal_component("foo/../bar") is True

    def test_trailing_dot_dot(self):
        assert has_traversal_component("foo/..") is True

    def test_multiple_dot_dots(self):
        assert has_traversal_component("../../../../etc/passwd") is True

    def test_plain_filename(self):
        assert has_traversal_component("file.txt") is False

    def test_absolute_path_no_traversal(self):
        assert has_traversal_component("/etc/passwd") is False

    def test_empty_string(self):
        assert has_traversal_component("") is False

    def test_dot_only(self):
        """Single '.' is not '..'."""
        assert has_traversal_component(".") is False

    def test_double_dot_in_filename(self):
        """'file..txt' has 'file..txt' as one part, not '..'."""
        assert has_traversal_component("file..txt") is False

    def test_double_dot_with_extra_dots(self):
        """'...' is a part [\"...\"], not '..'."""
        assert has_traversal_component("...") is False


# ============================================================================
# validate_within_dir
# ============================================================================
class TestValidateWithinDir:
    def test_path_inside_root(self, tmp_path: Path):
        root = tmp_path / "root"
        root.mkdir()
        child = root / "child.txt"
        child.write_text("hello")

        assert validate_within_dir(child, root) is None

    def test_path_equals_root(self, tmp_path: Path):
        root = tmp_path / "root"
        root.mkdir()
        assert validate_within_dir(root, root) is None

    def test_nested_inside_root(self, tmp_path: Path):
        root = tmp_path / "root"
        root.mkdir()
        nested = root / "a" / "b" / "c"
        nested.mkdir(parents=True)

        assert validate_within_dir(nested, root) is None

    def test_path_outside_root(self, tmp_path: Path):
        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("escape")

        error = validate_within_dir(outside, root)
        assert error is not None
        assert "escapes" in error

    def test_symlink_inside_root(self, tmp_path: Path):
        root = tmp_path / "root"
        root.mkdir()
        target = root / "target.txt"
        target.write_text("data")
        link = root / "link.txt"
        link.symlink_to(target)

        assert validate_within_dir(link, root) is None

    def test_symlink_outside_root(self, tmp_path: Path):
        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("escape")
        link = root / "evil_link.txt"
        link.symlink_to(outside)

        error = validate_within_dir(link, root)
        assert error is not None
        assert "escapes" in error

    def test_nonexistent_path_outside_root(self, tmp_path: Path):
        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "nonexistent_dir" / "file.txt"

        error = validate_within_dir(outside, root)
        # resolve() fails with OSError for nonexistent paths
        assert error is not None

    def test_nonexistent_path_inside_root(self, tmp_path: Path):
        root = tmp_path / "root"
        root.mkdir()
        inside = root / "does_not_exist.txt"

        # Path.resolve() on a non-existent file under an existing dir works fine
        assert validate_within_dir(inside, root) is None

    def test_dot_dot_traversal(self, tmp_path: Path):
        root = tmp_path / "root"
        root.mkdir()
        traversal = root / ".." / "outside.txt"

        error = validate_within_dir(traversal, root)
        assert error is not None

    def test_same_file_different_path(self, tmp_path: Path):
        """resolve() normalizes, so different representations of same path pass."""
        root = tmp_path / "root"
        root.mkdir()
        (root / "sub").mkdir()
        via_dot = root / "sub" / "." / "file.txt"

        assert validate_within_dir(via_dot, root) is None
