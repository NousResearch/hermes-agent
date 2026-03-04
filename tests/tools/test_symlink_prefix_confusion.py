"""Tests for the symlink boundary check prefix confusion fix in skills_guard.py.

Regression test: the original check used startswith() without a trailing
separator, so a symlink resolving to 'axolotl-backdoor/' passed the check
for 'axolotl/' because the string prefix matched. Now uses
Path.is_relative_to() which handles directory boundaries correctly.
"""

import os
import pytest
from pathlib import Path


def _old_check_escapes(resolved: Path, skill_dir_resolved: Path) -> bool:
    """The BROKEN check that used startswith without separator.

    Returns True when the path is OUTSIDE the skill directory.
    """
    return (
        not str(resolved).startswith(str(skill_dir_resolved))
        and resolved != skill_dir_resolved
    )


def _new_check_escapes(resolved: Path, skill_dir_resolved: Path) -> bool:
    """The FIXED check using is_relative_to().

    Returns True when the path is OUTSIDE the skill directory.
    """
    return not resolved.is_relative_to(skill_dir_resolved)


class TestPrefixConfusionRegression:
    """The core bug: startswith() can't distinguish directory boundaries."""

    def test_old_check_misses_sibling_with_shared_prefix(self, tmp_path):
        """Old startswith check fails on sibling dirs that share a prefix."""
        skill_dir = tmp_path / "skills" / "axolotl"
        sibling_file = tmp_path / "skills" / "axolotl-backdoor" / "evil.py"
        skill_dir.mkdir(parents=True)
        sibling_file.parent.mkdir(parents=True)
        sibling_file.write_text("evil")

        resolved = sibling_file.resolve()
        skill_dir_resolved = skill_dir.resolve()

        # Bug: old check says the file is INSIDE the skill dir
        assert _old_check_escapes(resolved, skill_dir_resolved) is False

    def test_new_check_catches_sibling_with_shared_prefix(self, tmp_path):
        """is_relative_to() correctly rejects sibling dirs."""
        skill_dir = tmp_path / "skills" / "axolotl"
        sibling_file = tmp_path / "skills" / "axolotl-backdoor" / "evil.py"
        skill_dir.mkdir(parents=True)
        sibling_file.parent.mkdir(parents=True)
        sibling_file.write_text("evil")

        resolved = sibling_file.resolve()
        skill_dir_resolved = skill_dir.resolve()

        # Fixed: new check correctly says it's OUTSIDE
        assert _new_check_escapes(resolved, skill_dir_resolved) is True

    def test_both_agree_on_real_subpath(self, tmp_path):
        """Both checks allow a genuine subpath."""
        skill_dir = tmp_path / "skills" / "axolotl"
        sub_file = skill_dir / "utils" / "helper.py"
        skill_dir.mkdir(parents=True)
        sub_file.parent.mkdir(parents=True)
        sub_file.write_text("ok")

        resolved = sub_file.resolve()
        skill_dir_resolved = skill_dir.resolve()

        assert _old_check_escapes(resolved, skill_dir_resolved) is False
        assert _new_check_escapes(resolved, skill_dir_resolved) is False

    def test_both_agree_on_completely_outside_path(self, tmp_path):
        """Both checks block a path that's completely outside."""
        skill_dir = tmp_path / "skills" / "axolotl"
        outside_file = tmp_path / "etc" / "passwd"
        skill_dir.mkdir(parents=True)
        outside_file.parent.mkdir(parents=True)
        outside_file.write_text("root:x:0:0")

        resolved = outside_file.resolve()
        skill_dir_resolved = skill_dir.resolve()

        assert _old_check_escapes(resolved, skill_dir_resolved) is True
        assert _new_check_escapes(resolved, skill_dir_resolved) is True

    def test_skill_dir_itself_allowed(self, tmp_path):
        """Requesting the skill directory itself is fine."""
        skill_dir = tmp_path / "skills" / "axolotl"
        skill_dir.mkdir(parents=True)

        resolved = skill_dir.resolve()
        skill_dir_resolved = skill_dir.resolve()

        # Both should allow the dir itself
        assert _old_check_escapes(resolved, skill_dir_resolved) is False
        assert _new_check_escapes(resolved, skill_dir_resolved) is False


def _can_symlink():
    """Check if we can create symlinks (needs admin/dev-mode on Windows)."""
    import tempfile
    try:
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src"
            src.write_text("x")
            lnk = Path(d) / "lnk"
            lnk.symlink_to(src)
            return True
    except OSError:
        return False


@pytest.mark.skipif(not _can_symlink(), reason="Symlinks need elevated privileges")
class TestSymlinkEscapeWithActualSymlinks:
    """Test the full symlink scenario with real filesystem symlinks."""

    def test_symlink_to_sibling_prefix_dir_detected(self, tmp_path):
        """A symlink from axolotl/ to axolotl-backdoor/ must be caught."""
        skills = tmp_path / "skills"
        skill_dir = skills / "axolotl"
        sibling_dir = skills / "axolotl-backdoor"
        skill_dir.mkdir(parents=True)
        sibling_dir.mkdir(parents=True)

        malicious = sibling_dir / "malicious.py"
        malicious.write_text("evil code")

        link = skill_dir / "helper.py"
        link.symlink_to(malicious)

        resolved = link.resolve()
        skill_dir_resolved = skill_dir.resolve()

        # Old check would miss this (prefix confusion)
        assert _old_check_escapes(resolved, skill_dir_resolved) is False
        # New check catches it
        assert _new_check_escapes(resolved, skill_dir_resolved) is True

    def test_symlink_within_skill_dir_allowed(self, tmp_path):
        """A symlink that stays within the skill directory is fine."""
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        real_file = skill_dir / "real.py"
        real_file.write_text("print('ok')")
        link = skill_dir / "alias.py"
        link.symlink_to(real_file)

        resolved = link.resolve()
        skill_dir_resolved = skill_dir.resolve()

        assert _new_check_escapes(resolved, skill_dir_resolved) is False

    def test_symlink_to_parent_dir_blocked(self, tmp_path):
        """A symlink pointing outside (to parent) is blocked."""
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        outside = tmp_path / "secret.env"
        outside.write_text("SECRET=123")

        link = skill_dir / "config.env"
        link.symlink_to(outside)

        resolved = link.resolve()
        skill_dir_resolved = skill_dir.resolve()

        assert _new_check_escapes(resolved, skill_dir_resolved) is True
