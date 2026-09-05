"""Symlink-refusal and permission tests for tools.spill_safety.

The bug class: spill/cache writers used ``open(path, "w")`` /
``Path.write_text`` in predictable directories, which follows a pre-planted
symlink and redirects the write onto an arbitrary user-owned file. Every
helper must refuse the link (never write through it) while keeping normal
writes byte-identical.
"""

import os
import stat
import sys

import pytest

from tools.spill_safety import (
    ensure_spill_dir,
    open_exclusive,
    write_text_exclusive,
)

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX perms/symlinks")


def test_write_creates_file_with_content(tmp_path):
    target = tmp_path / "spill.txt"
    write_text_exclusive(target, "hello\n")
    assert target.read_text(encoding="utf-8") == "hello\n"


@posix_only
def test_private_file_is_0600(tmp_path):
    target = tmp_path / "spill.txt"
    write_text_exclusive(target, "secret", private=True)
    assert stat.S_IMODE(os.lstat(target).st_mode) == 0o600


@posix_only
def test_private_dir_is_0700_and_tightened(tmp_path):
    d = tmp_path / "spills"
    d.mkdir(mode=0o755)
    ensure_spill_dir(d, private=True)
    assert stat.S_IMODE(os.lstat(d).st_mode) == 0o700


def test_ensure_spill_dir_refuses_symlinked_leaf(tmp_path):
    victim = tmp_path / "victim-dir"
    victim.mkdir()
    link = tmp_path / "spills"
    link.symlink_to(victim)
    with pytest.raises(OSError):
        ensure_spill_dir(link)


def test_refuses_planted_symlink(tmp_path):
    """The core attack: symlink at the spill path must fail, not redirect."""
    victim = tmp_path / "victim.txt"
    victim.write_text("original")
    target = tmp_path / "spill.txt"
    target.symlink_to(victim)
    with pytest.raises(OSError):
        write_text_exclusive(target, "attacker-controlled")
    assert victim.read_text() == "original"


def test_refuses_dangling_symlink(tmp_path):
    target = tmp_path / "spill.txt"
    target.symlink_to(tmp_path / "does-not-exist.txt")
    with pytest.raises(OSError):
        write_text_exclusive(target, "x")
    assert not (tmp_path / "does-not-exist.txt").exists()


def test_overwrite_removes_symlink_not_its_target(tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("original")
    target = tmp_path / "spill.txt"
    target.symlink_to(victim)
    write_text_exclusive(target, "redacted copy", overwrite=True)
    # Link replaced by a real file; the link's target untouched.
    assert not target.is_symlink()
    assert target.read_text(encoding="utf-8") == "redacted copy"
    assert victim.read_text() == "original"


def test_overwrite_replaces_regular_file(tmp_path):
    target = tmp_path / "spill.txt"
    target.write_text("raw")
    write_text_exclusive(target, "redacted", overwrite=True)
    assert target.read_text(encoding="utf-8") == "redacted"


def test_overwrite_refuses_directory(tmp_path):
    target = tmp_path / "spill.txt"
    target.mkdir()
    with pytest.raises(OSError):
        write_text_exclusive(target, "x", overwrite=True)
    assert target.is_dir()


def test_exclusive_create_fails_on_existing_without_overwrite(tmp_path):
    target = tmp_path / "spill.txt"
    target.write_text("first")
    with pytest.raises(OSError):
        write_text_exclusive(target, "second")
    assert target.read_text() == "first"


def test_open_exclusive_streaming_write(tmp_path):
    target = tmp_path / "spill.log"
    with open_exclusive(target, errors="replace") as fh:
        fh.write("chunk1")
        fh.write("chunk2")
    assert target.read_text(encoding="utf-8") == "chunk1chunk2"
