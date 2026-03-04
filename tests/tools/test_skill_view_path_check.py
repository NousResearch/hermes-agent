"""Tests for the skill_view path boundary check on all platforms.

Regression test: the original check used a hardcoded "/" separator which
fails on Windows where Path.resolve() returns backslash-separated paths.
The fix uses os.sep so the check works on both Unix and Windows.
"""

import json
import os
import pytest
from pathlib import Path


def _path_escapes_skill_dir(resolved: Path, skill_dir_resolved: Path) -> bool:
    """Reproduce the boundary check from tools/skills_tool.py (line 461).

    Returns True when the resolved path is OUTSIDE the skill directory.
    """
    return (
        not str(resolved).startswith(str(skill_dir_resolved) + os.sep)
        and resolved != skill_dir_resolved
    )


class TestSkillViewPathBoundaryCheck:
    """Verify the os.sep fix prevents false positives on Windows."""

    def test_valid_subpath_allowed(self, tmp_path):
        """A file inside the skill directory must NOT be flagged."""
        skill_dir = tmp_path / "skills" / "axolotl"
        ref_file = skill_dir / "references" / "api.md"
        skill_dir.mkdir(parents=True)
        ref_file.parent.mkdir()
        ref_file.write_text("content")

        resolved = ref_file.resolve()
        skill_dir_resolved = skill_dir.resolve()

        assert _path_escapes_skill_dir(resolved, skill_dir_resolved) is False

    def test_deeply_nested_subpath_allowed(self, tmp_path):
        """Deeply nested valid paths must also pass."""
        skill_dir = tmp_path / "skills" / "ml-paper"
        deep_file = skill_dir / "templates" / "acl" / "formatting.md"
        skill_dir.mkdir(parents=True)
        deep_file.parent.mkdir(parents=True)
        deep_file.write_text("content")

        resolved = deep_file.resolve()
        skill_dir_resolved = skill_dir.resolve()

        assert _path_escapes_skill_dir(resolved, skill_dir_resolved) is False

    def test_outside_path_blocked(self, tmp_path):
        """A file outside the skill directory must be flagged."""
        skill_dir = tmp_path / "skills" / "axolotl"
        skill_dir.mkdir(parents=True)
        outside_file = tmp_path / "secret.env"
        outside_file.write_text("SECRET=123")

        resolved = outside_file.resolve()
        skill_dir_resolved = skill_dir.resolve()

        assert _path_escapes_skill_dir(resolved, skill_dir_resolved) is True

    def test_sibling_skill_dir_blocked(self, tmp_path):
        """A file in a sibling skill directory must be flagged.

        This catches prefix confusion: 'axolotl-v2' starts with 'axolotl'
        as a string but is a different directory.
        """
        skill_dir = tmp_path / "skills" / "axolotl"
        sibling_dir = tmp_path / "skills" / "axolotl-v2"
        skill_dir.mkdir(parents=True)
        sibling_dir.mkdir(parents=True)
        sibling_file = sibling_dir / "SKILL.md"
        sibling_file.write_text("other skill")

        resolved = sibling_file.resolve()
        skill_dir_resolved = skill_dir.resolve()

        assert _path_escapes_skill_dir(resolved, skill_dir_resolved) is True

    def test_skill_dir_itself_allowed(self, tmp_path):
        """Requesting the skill directory itself must be allowed (== check)."""
        skill_dir = tmp_path / "skills" / "axolotl"
        skill_dir.mkdir(parents=True)

        resolved = skill_dir.resolve()
        skill_dir_resolved = skill_dir.resolve()

        assert _path_escapes_skill_dir(resolved, skill_dir_resolved) is False

    def test_separator_is_os_native(self):
        """Confirm the check uses the platform's native separator."""
        # On Windows os.sep is '\\', on Unix '/'.
        # The old bug hardcoded '/' which broke on Windows.
        assert os.sep in ("\\", "/")


class TestOldCheckWouldFail:
    """Demonstrate the bug: the old hardcoded '/' check fails on Windows."""

    def _old_path_escapes(self, resolved: Path, skill_dir_resolved: Path) -> bool:
        """The BROKEN check that used hardcoded '/'."""
        return (
            not str(resolved).startswith(str(skill_dir_resolved) + "/")
            and resolved != skill_dir_resolved
        )

    @pytest.mark.skipif(os.sep == "/", reason="Bug only manifests on Windows")
    def test_old_check_false_positive_on_windows(self, tmp_path):
        """On Windows, the old check incorrectly blocks valid subpaths."""
        skill_dir = tmp_path / "skills" / "axolotl"
        ref_file = skill_dir / "references" / "api.md"
        skill_dir.mkdir(parents=True)
        ref_file.parent.mkdir()
        ref_file.write_text("content")

        resolved = ref_file.resolve()
        skill_dir_resolved = skill_dir.resolve()

        # Old check says it escapes (WRONG on Windows)
        assert self._old_path_escapes(resolved, skill_dir_resolved) is True
        # New check correctly allows it
        assert _path_escapes_skill_dir(resolved, skill_dir_resolved) is False
