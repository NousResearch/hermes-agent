"""Tests for tools.dir_sync.sync_dir_in_place.

The core contract: mirror *src* into *dst* without ever removing or
recreating *dst* itself, so directory inodes stay stable for long-lived
bind mounts (Docker persistent containers — hermes-agent#53630, #73842).
"""

import os
from pathlib import Path

import pytest

from tools.dir_sync import sync_dir_in_place


def _stat_inode(p: Path) -> int:
    return p.stat().st_ino


class TestSyncDirInPlace:
    def test_dst_inode_stable_across_syncs(self, tmp_path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()
        (src / "a.txt").write_text("a")

        sync_dir_in_place(src, dst)
        first_inode = _stat_inode(dst)

        (src / "b.txt").write_text("b")
        sync_dir_in_place(src, dst)

        assert _stat_inode(dst) == first_inode
        assert (dst / "a.txt").read_text() == "a"
        assert (dst / "b.txt").read_text() == "b"

    def test_creates_dst_when_missing(self, tmp_path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        (src / "a.txt").write_text("a")

        sync_dir_in_place(src, dst)

        assert (dst / "a.txt").read_text() == "a"

    def test_removes_deleted_files(self, tmp_path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()
        (src / "keep.txt").write_text("keep")
        (src / "gone.txt").write_text("gone")
        sync_dir_in_place(src, dst)

        (src / "gone.txt").unlink()
        sync_dir_in_place(src, dst)

        assert (dst / "keep.txt").exists()
        assert not (dst / "gone.txt").exists()

    def test_removes_stale_directories(self, tmp_path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()
        (src / "keep").mkdir()
        (src / "keep" / "k.txt").write_text("k")
        (src / "stale").mkdir()
        (src / "stale" / "s.txt").write_text("s")
        sync_dir_in_place(src, dst)

        (src / "stale" / "s.txt").unlink()
        (src / "stale").rmdir()
        sync_dir_in_place(src, dst)

        assert (dst / "keep" / "k.txt").exists()
        assert not (dst / "stale").exists()

    def test_file_to_dir_transition(self, tmp_path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()
        (src / "thing").write_text("flat")
        sync_dir_in_place(src, dst)
        assert (dst / "thing").is_file()

        (src / "thing").unlink()
        (src / "thing").mkdir()
        (src / "thing" / "inner.md").write_text("inner")
        sync_dir_in_place(src, dst)

        assert (dst / "thing").is_dir()
        assert (dst / "thing" / "inner.md").read_text() == "inner"

    def test_dir_to_file_transition(self, tmp_path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()
        (src / "thing").mkdir()
        (src / "thing" / "inner.md").write_text("inner")
        sync_dir_in_place(src, dst)
        assert (dst / "thing").is_dir()

        import shutil

        shutil.rmtree(src / "thing")
        (src / "thing").write_text("flat again")
        sync_dir_in_place(src, dst)

        assert (dst / "thing").is_file()
        assert (dst / "thing").read_text() == "flat again"

    def test_file_symlinks_in_src_are_dereferenced(self, tmp_path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()
        target_file = tmp_path / "shared.txt"
        target_file.write_text("shared content")
        try:
            (src / "link.txt").symlink_to(target_file)
        except OSError:
            pytest.skip("symlinks unavailable in test environment")

        sync_dir_in_place(src, dst)

        # Historical copytree behavior: the symlink is dereferenced into a
        # regular file with the target's content — never a symlink in dst.
        assert (dst / "link.txt").is_file()
        assert not (dst / "link.txt").is_symlink()
        assert (dst / "link.txt").read_text() == "shared content"

    def test_dir_symlinks_in_src_are_skipped(self, tmp_path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()
        (src / "real.txt").write_text("real")
        target_dir = tmp_path / "outside"
        target_dir.mkdir()
        (target_dir / "secret.md").write_text("TOP SECRET")
        try:
            (src / "linkdir").symlink_to(target_dir, target_is_directory=True)
        except OSError:
            pytest.skip("symlinks unavailable in test environment")

        sync_dir_in_place(src, dst)

        assert (dst / "real.txt").exists()
        assert not (dst / "linkdir").exists()
        assert not (dst / "secret.md").exists()

    def test_removes_symlinks_in_dst(self, tmp_path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()
        (src / "real.txt").write_text("real")
        secret = tmp_path / "secret.txt"
        secret.write_text("TOP SECRET")
        try:
            (dst / "evil_link").symlink_to(secret)
        except OSError:
            pytest.skip("symlinks unavailable in test environment")

        sync_dir_in_place(src, dst)

        assert not (dst / "evil_link").exists()
        assert not (dst / "secret.txt").exists()
        assert (dst / "real.txt").exists()

    def test_skip_names_excludes_top_level_entries(self, tmp_path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()
        (src / "keep.txt").write_text("keep")
        (src / "memories").mkdir()
        (src / "memories" / "MEMORY.md").write_text("private")

        sync_dir_in_place(src, dst, skip_names={"memories"})

        assert (dst / "keep.txt").exists()
        assert not (dst / "memories").exists()

    def test_skip_names_preserves_existing_dst_entries(self, tmp_path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()
        (src / "keep.txt").write_text("keep")
        # Pre-existing user-owned data in dst must survive the sync —
        # skip_names shields it from the stale-entry removal pass.
        (dst / "memories").mkdir()
        (dst / "memories" / "user.txt").write_text("user data")

        sync_dir_in_place(src, dst, skip_names={"memories"})

        assert (dst / "keep.txt").exists()
        assert (dst / "memories" / "user.txt").read_text() == "user data"

    def test_unchanged_file_mtime_preserved(self, tmp_path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()
        (src / "a.txt").write_text("content")
        sync_dir_in_place(src, dst)

        # Simulate an unchanged file: the source already matches the mirror
        # (same mtime + size), so the sync must not rewrite the destination.
        dst_mtime = (dst / "a.txt").stat().st_mtime
        os.utime(src / "a.txt", (dst_mtime, dst_mtime))
        sync_dir_in_place(src, dst)

        # The destination copy keeps its original mtime (not rewritten).
        assert (dst / "a.txt").stat().st_mtime == dst_mtime

    def test_file_mode_preserved(self, tmp_path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()
        script = src / "run.sh"
        script.write_text("#!/bin/sh\necho hi\n")
        script.chmod(0o755)

        sync_dir_in_place(src, dst)

        assert (dst / "run.sh").stat().st_mode & 0o777 == 0o755

    def test_no_temp_file_litter_after_sync(self, tmp_path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()
        (src / "a.txt").write_text("content")
        sync_dir_in_place(src, dst)

        leftovers = [p for p in dst.rglob("*") if ".hermes-sync-" in p.name]
        assert leftovers == []

    def test_empty_dir_in_src_is_preserved(self, tmp_path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()
        (src / "empty").mkdir()
        sync_dir_in_place(src, dst)

        assert (dst / "empty").is_dir()

    def test_nested_changes_propagate(self, tmp_path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()
        (src / "skills" / "demo" / "scripts").mkdir(parents=True)
        (src / "skills" / "demo" / "scripts" / "run.sh").write_text("#!/bin/sh\n")
        (src / "skills" / "demo" / "SKILL.md").write_text("# demo")

        sync_dir_in_place(src, dst)

        # Modify nested content in src and re-sync.
        (src / "skills" / "demo" / "SKILL.md").write_text("# demo v2")
        (src / "skills" / "demo" / "scripts" / "run.sh").unlink()
        sync_dir_in_place(src, dst)

        assert (dst / "skills" / "demo" / "SKILL.md").read_text() == "# demo v2"
        assert not (dst / "skills" / "demo" / "scripts" / "run.sh").exists()
