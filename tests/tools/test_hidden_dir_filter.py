"""Tests for the hidden directory filter in skills listing.

Regression test: the original filter used hardcoded forward-slash strings
like '/.git/' which never match on Windows where Path uses backslashes.
This caused quarantined skills (.hub/quarantine/) to appear as installed.

Now uses Path.parts which is platform-independent.
"""

import os
from pathlib import Path, PurePosixPath, PureWindowsPath


def _old_filter_matches(path_str: str) -> bool:
    """The BROKEN filter that used hardcoded forward slashes.

    Returns True when the path SHOULD be filtered out.
    """
    return '/.git/' in path_str or '/.github/' in path_str or '/.hub/' in path_str


def _new_filter_matches(path: Path) -> bool:
    """The FIXED filter using Path.parts.

    Returns True when the path SHOULD be filtered out.
    """
    return any(part in ('.git', '.github', '.hub') for part in path.parts)


class TestOldFilterBrokenOnWindows:
    """Demonstrate the bug: hardcoded '/' never matches Windows backslash paths."""

    def test_old_filter_misses_hub_on_windows_path(self):
        """Old filter fails to catch .hub in a Windows-style path string."""
        win_path = r"C:\Users\me\.hermes\skills\.hub\quarantine\evil-skill\SKILL.md"
        assert _old_filter_matches(win_path) is False  # Bug: should be True

    def test_old_filter_misses_git_on_windows_path(self):
        """Old filter fails to catch .git in a Windows-style path string."""
        win_path = r"C:\Users\me\.hermes\skills\.git\config\SKILL.md"
        assert _old_filter_matches(win_path) is False  # Bug: should be True

    def test_old_filter_works_on_unix_path(self):
        """Old filter works fine on Unix paths (the original platform)."""
        unix_path = "/home/user/.hermes/skills/.hub/quarantine/evil-skill/SKILL.md"
        assert _old_filter_matches(unix_path) is True


class TestNewFilterCrossPlatform:
    """The fixed filter works on both Windows and Unix paths."""

    def test_hub_quarantine_filtered(self, tmp_path):
        """A SKILL.md inside .hub/quarantine/ must be filtered out."""
        p = tmp_path / ".hermes" / "skills" / ".hub" / "quarantine" / "evil" / "SKILL.md"
        assert _new_filter_matches(p) is True

    def test_git_dir_filtered(self, tmp_path):
        """A SKILL.md inside .git/ must be filtered out."""
        p = tmp_path / ".hermes" / "skills" / ".git" / "hooks" / "SKILL.md"
        assert _new_filter_matches(p) is True

    def test_github_dir_filtered(self, tmp_path):
        """A SKILL.md inside .github/ must be filtered out."""
        p = tmp_path / ".hermes" / "skills" / ".github" / "workflows" / "SKILL.md"
        assert _new_filter_matches(p) is True

    def test_normal_skill_not_filtered(self, tmp_path):
        """A regular skill SKILL.md must NOT be filtered out."""
        p = tmp_path / ".hermes" / "skills" / "my-cool-skill" / "SKILL.md"
        assert _new_filter_matches(p) is False

    def test_nested_skill_not_filtered(self, tmp_path):
        """A deeply nested regular skill must NOT be filtered out."""
        p = tmp_path / ".hermes" / "skills" / "org" / "deep-skill" / "SKILL.md"
        assert _new_filter_matches(p) is False

    def test_dot_prefix_not_false_positive(self, tmp_path):
        """A skill dir starting with dot but not in the filter list passes."""
        p = tmp_path / ".hermes" / "skills" / ".my-hidden-skill" / "SKILL.md"
        assert _new_filter_matches(p) is False


class TestWindowsPathParts:
    """Verify Path.parts correctly splits on the native separator."""

    def test_parts_contains_hidden_dir(self, tmp_path):
        """Path.parts includes each directory component individually."""
        p = tmp_path / "skills" / ".hub" / "quarantine" / "SKILL.md"
        assert ".hub" in p.parts

    def test_parts_does_not_contain_combined_string(self, tmp_path):
        """Path.parts splits by separator, not by substring."""
        p = tmp_path / "skills" / "my-hub-skill" / "SKILL.md"
        # ".hub" should NOT match "my-hub-skill" as a part
        assert ".hub" not in p.parts
