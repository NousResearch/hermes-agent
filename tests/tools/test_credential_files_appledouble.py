"""Tests for AppleDouble filtering in the remote-sync file enumeration.

macOS writes ``._name`` sidecars next to real files on non-HFS volumes.  They
carry no content Hermes needs, cannot always be stat'd by the time tar reads
them, and on this workstation were 6,620 of 13,241 enumerated sync files --
half of every remote sync was pure waste.
"""

from tools.credential_files import _is_syncable_sync_file


class TestAppleDoubleFiltering:
    def test_appledouble_is_skipped(self, tmp_path):
        f = tmp_path / "._SKILL.md"
        f.write_text("resource fork junk")
        assert _is_syncable_sync_file(f) is False

    def test_normal_file_is_kept(self, tmp_path):
        f = tmp_path / "SKILL.md"
        f.write_text("real content")
        assert _is_syncable_sync_file(f) is True

    def test_leading_dot_file_is_kept(self, tmp_path):
        """Only the ``._`` prefix is AppleDouble -- plain dotfiles are real."""
        f = tmp_path / ".env"
        f.write_text("KEY=value")
        assert _is_syncable_sync_file(f) is True

    def test_directory_is_skipped(self, tmp_path):
        d = tmp_path / "subdir"
        d.mkdir()
        assert _is_syncable_sync_file(d) is False

    def test_symlink_is_skipped(self, tmp_path):
        target = tmp_path / "real.md"
        target.write_text("x")
        link = tmp_path / "link.md"
        link.symlink_to(target)
        assert _is_syncable_sync_file(link) is False
