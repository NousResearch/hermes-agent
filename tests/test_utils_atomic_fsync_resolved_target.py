"""``fsync_dir=True`` must sync the directory that actually received the rename.

``_atomic_write`` publishes the new entry via ``atomic_replace``, which resolves a
symlinked destination and renames onto the real file in its own directory. The
directory fsync used to target the parent of the (unresolved) input path, so a
write through a cross-directory symlink synced the link's directory while the
rename's new directory entry lived elsewhere — the requested durability never
covered it (issue #115030).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from utils import atomic_write_bytes, atomic_write_text


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable in test environment: {exc}")


class TestFsyncDirFollowsResolvedTarget:
    def test_symlinked_destination_syncs_resolved_directory(self, tmp_path):
        real = tmp_path / "real"
        links = tmp_path / "links"
        real.mkdir()
        links.mkdir()
        target = real / "value"
        target.write_text("old", encoding="utf-8")
        link = links / "value"
        _symlink_or_skip(link, target)

        with patch("utils.fsync_directory") as sync:
            atomic_write_text(link, "new", mode=0o600, fsync_dir=True)

        assert target.read_text(encoding="utf-8") == "new"
        assert link.is_symlink()
        sync.assert_called_once_with(real)

    def test_plain_destination_still_syncs_its_own_parent(self, tmp_path):
        target = tmp_path / "plain"

        with patch("utils.fsync_directory") as sync:
            atomic_write_text(target, "data", fsync_dir=True)

        assert target.read_text(encoding="utf-8") == "data"
        sync.assert_called_once_with(tmp_path)

    def test_symlinked_destination_bytes_variant(self, tmp_path):
        real = tmp_path / "real"
        links = tmp_path / "links"
        real.mkdir()
        links.mkdir()
        target = real / "blob"
        target.write_bytes(b"old")
        link = links / "blob"
        _symlink_or_skip(link, target)

        with patch("utils.fsync_directory") as sync:
            atomic_write_bytes(link, b"new", mode=0o600, fsync_dir=True)

        assert target.read_bytes() == b"new"
        assert link.is_symlink()
        sync.assert_called_once_with(real)
