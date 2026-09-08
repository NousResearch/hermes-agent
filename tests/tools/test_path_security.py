"""Unit tests for the shared path-traversal guards in ``tools.path_security``.

These two helpers protect seven tool surfaces (skills, cron job args, credential
files, file tools, tts, skill manager) from symlink / ``..`` escapes.  They had
no direct unit coverage — a regression that swapped ``resolve()`` for a
non-following check, or dropped the ``ValueError``/``OSError`` net, would let a
crafted path escape its allowed directory across all seven callers and only
surface via an integration test (if at all).  Pin each guard here.
"""

import os
import sys
from pathlib import Path

import pytest

# Make ``tools`` importable when run from the repo root without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.path_security import has_traversal_component, validate_within_dir  # noqa: E402


# ---------------------------------------------------------------------------
# has_traversal_component
# ---------------------------------------------------------------------------

def test_traversal_detects_dotdot_in_middle():
    assert has_traversal_component("foo/../bar") is True


def test_traversal_detects_leading_dotdot():
    assert has_traversal_component("../foo") is True


def test_traversal_detects_sole_dotdot():
    assert has_traversal_component("..") is True


def test_traversal_detects_stacked_dotdot():
    assert has_traversal_component("a/../../b") is True


def test_traversal_false_for_plain_path():
    assert has_traversal_component("foo/bar") is False


def test_traversal_false_for_filename_ending_in_dotdot():
    """``bar..`` is a legal filename, not a parent-dir component."""
    assert has_traversal_component("foo/bar..") is False


def test_traversal_false_for_triple_dot():
    """``...`` is a legal filename, not ``..``."""
    assert has_traversal_component("foo/.../bar") is False


def test_traversal_false_for_single_dot():
    assert has_traversal_component("foo/./bar") is False


def test_traversal_false_for_empty():
    assert has_traversal_component("") is False


def test_traversal_false_for_sole_dot():
    assert has_traversal_component(".") is False


# ---------------------------------------------------------------------------
# validate_within_dir
# ---------------------------------------------------------------------------

def test_validate_root_itself_is_ok(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    assert validate_within_dir(root, root) is None


def test_validate_nested_path_is_ok(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    assert validate_within_dir(root / "sub" / "deep" / "file.txt", root) is None


def test_validate_dotdot_that_stays_inside_is_ok(tmp_path):
    """``a/../b`` resolves back inside root — valid, not an escape."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "b").mkdir()
    assert validate_within_dir(root / "a" / ".." / "b", root) is None


def test_validate_dotdot_escape_returns_error(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    err = validate_within_dir(root / ".." / "secret", root)
    assert err is not None
    assert "escapes" in err


def test_validate_sibling_outside_root_returns_error(tmp_path):
    """A path that is a sibling of root (not under it) is rejected."""
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    err = validate_within_dir(outside, root)
    assert err is not None
    assert "escapes" in err


def test_validate_absolute_outside_root_returns_error(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    err = validate_within_dir(Path("/etc/passwd"), root)
    assert err is not None


def test_validate_symlink_escape_returns_error(tmp_path):
    """A symlink inside root pointing outside root is rejected after resolve()
    follows the link — the core symlink-traversal guard."""
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    link = root / "link"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unsupported on this platform")
    err = validate_within_dir(link, root)
    assert err is not None
    assert "escapes" in err


def test_validate_symlink_to_inside_root_is_ok(tmp_path):
    """A symlink inside root pointing to another path inside root is allowed."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "real.txt").write_text("ok")
    link = root / "link"
    try:
        os.symlink(root / "real.txt", link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unsupported on this platform")
    assert validate_within_dir(link, root) is None


def test_validate_error_message_is_human_readable(tmp_path):
    """The returned string carries the underlying resolution failure so callers
    can surface it without a second lookup."""
    root = tmp_path / "root"
    root.mkdir()
    err = validate_within_dir(root / ".." / "escape", root)
    assert err is not None
    assert err.startswith("Path escapes allowed directory")
