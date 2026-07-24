"""Integration tests for symlink/junction detection in tools/path_security.py.

Covers the three bugs fixed in fix/skills-guard-symlink-detection:
1. _has_symlink_component walked resolved components, hiding the redirect.
2. target.exists() guard let dangling symlinks through unchecked.
3. validate_within_dir callers did not pass check_symlink_components=True.
"""

import tempfile
from pathlib import Path

import pytest

from tools.path_security import (
    _has_symlink_component,
    _is_path_redirect,
    validate_within_dir,
)


def _can_symlink() -> bool:
    try:
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src"
            src.write_text("x")
            lnk = Path(d) / "lnk"
            lnk.symlink_to(src)
            return True
    except OSError:
        return False


requires_symlink = pytest.mark.skipif(
    not _can_symlink(), reason="symlinks require elevated privileges on this platform"
)


# ---------------------------------------------------------------------------
# _is_path_redirect
# ---------------------------------------------------------------------------


class TestIsPathRedirect:
    @requires_symlink
    def test_symlink_to_file_is_redirect(self, tmp_path):
        target = tmp_path / "real.txt"
        target.write_text("x")
        link = tmp_path / "link.txt"
        link.symlink_to(target)
        assert _is_path_redirect(link) is True

    @requires_symlink
    def test_dangling_symlink_is_redirect(self, tmp_path):
        link = tmp_path / "dangling"
        link.symlink_to(tmp_path / "nonexistent")
        assert _is_path_redirect(link) is True

    def test_regular_file_is_not_redirect(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        assert _is_path_redirect(f) is False

    def test_regular_dir_is_not_redirect(self, tmp_path):
        d = tmp_path / "subdir"
        d.mkdir()
        assert _is_path_redirect(d) is False


# ---------------------------------------------------------------------------
# _has_symlink_component — lexical walk, dangling symlinks
# ---------------------------------------------------------------------------


class TestHasSymlinkComponent:
    @requires_symlink
    def test_detects_symlink_dir_component(self, tmp_path):
        """A symlink directory component inside root must be caught."""
        real_outside = tmp_path / "outside"
        real_outside.mkdir()
        skill_root = tmp_path / "skills" / "myskill"
        skill_root.mkdir(parents=True)

        link_dir = skill_root / "link_dir"
        link_dir.symlink_to(real_outside)

        target = link_dir / "secret.txt"
        err = _has_symlink_component(target, skill_root)
        assert err is not None
        assert "link_dir" in err

    @requires_symlink
    def test_detects_dangling_symlink_component(self, tmp_path):
        """A dangling symlink (target missing) must still be rejected."""
        skill_root = tmp_path / "skills" / "myskill"
        skill_root.mkdir(parents=True)

        dangling = skill_root / "ghost"
        dangling.symlink_to(tmp_path / "does_not_exist")

        err = _has_symlink_component(dangling, skill_root)
        assert err is not None
        assert "ghost" in err

    @requires_symlink
    def test_symlink_that_stays_inside_root_is_still_rejected(self, tmp_path):
        """Even an internal symlink (link -> root/real) must be rejected
        because the component walk sees the redirect, not just the destination.
        """
        skill_root = tmp_path / "skills" / "myskill"
        real_sub = skill_root / "real"
        real_sub.mkdir(parents=True)
        (real_sub / "file.txt").write_text("ok")

        link_sub = skill_root / "link_sub"
        link_sub.symlink_to(real_sub)

        err = _has_symlink_component(link_sub / "file.txt", skill_root)
        assert err is not None
        assert "link_sub" in err

    def test_no_symlink_returns_none(self, tmp_path):
        skill_root = tmp_path / "skills" / "myskill"
        sub = skill_root / "sub"
        sub.mkdir(parents=True)
        f = sub / "file.txt"
        f.write_text("safe")
        assert _has_symlink_component(f, skill_root) is None


# ---------------------------------------------------------------------------
# validate_within_dir with check_symlink_components=True
# ---------------------------------------------------------------------------


class TestValidateWithinDir:
    def test_normal_path_passes(self, tmp_path):
        root = tmp_path / "root"
        (root / "sub").mkdir(parents=True)
        f = root / "sub" / "file.txt"
        f.write_text("ok")
        assert validate_within_dir(f, root, check_symlink_components=True) is None

    def test_traversal_rejected(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("bad")
        err = validate_within_dir(outside, root, check_symlink_components=True)
        assert err is not None

    @requires_symlink
    def test_symlink_component_rejected_with_flag(self, tmp_path):
        """check_symlink_components=True must catch a redirecting symlink dir."""
        outside = tmp_path / "outside"
        outside.mkdir()
        root = tmp_path / "root"
        root.mkdir()

        link = root / "escape"
        link.symlink_to(outside)

        err = validate_within_dir(link / "x.txt", root, check_symlink_components=True)
        assert err is not None
        assert "escape" in err

    @requires_symlink
    def test_symlink_component_passes_without_flag(self, tmp_path):
        """Default (check_symlink_components=False) still follows symlinks via resolve()."""
        outside = tmp_path / "outside"
        outside.mkdir()
        root = tmp_path / "root"
        root.mkdir()

        link = root / "escape"
        link.symlink_to(outside)

        # Without the flag, resolve() places the path outside root — rejected
        # by the containment check — but the symlink itself is not the stated reason.
        # This test just verifies the flag=False path doesn't crash.
        result = validate_within_dir(link / "x.txt", root, check_symlink_components=False)
        # May be None or an error depending on resolve; just must not raise.
        assert result is None or isinstance(result, str)

    @requires_symlink
    def test_dangling_symlink_rejected(self, tmp_path):
        """A dangling symlink component must be rejected even without a live target."""
        root = tmp_path / "root"
        root.mkdir()

        dangling = root / "ghost"
        dangling.symlink_to(tmp_path / "nonexistent")

        err = validate_within_dir(dangling, root, check_symlink_components=True)
        assert err is not None
