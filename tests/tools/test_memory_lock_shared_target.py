"""One publish target, one writer lock (#154699).

A profile that shares ``USER.md`` through a symlink publishes onto the *shared real file*
(``utils.atomic_replace`` renames onto the resolved target), but the store took its lock on
the profile-local pathname — so two profiles writing the same file serialized against two
different lock files, and a read-modify-write of the whole file could drop the other
profile's entry.

The lock identity is therefore pinned to the file the write lands on: both spellings must
leave exactly one ``*.lock`` beside the resolved target, and holding it must exclude any
other writer of that target.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path

import pytest

from tools.memory_tool import MemoryStore


def _shared_pair(tmp_path: Path) -> tuple[Path, list[Path]]:
    """A shared real ``USER.md`` plus two profile-local symlinks to it."""
    shared = tmp_path / "shared"
    shared.mkdir()
    real = shared / "USER.md"
    real.write_text("§\nshared fact\n", encoding="utf-8")
    links = []
    for profile in ("a", "b"):
        memdir = tmp_path / profile / "memories"
        memdir.mkdir(parents=True)
        link = memdir / "USER.md"
        link.symlink_to(real)
        links.append(link)
    return real, links


def _lock_files(root: Path) -> list[Path]:
    return sorted(path.relative_to(root) for path in root.rglob("*.lock"))


@pytest.mark.platforms("posix")  # flock is fcntl's
def test_two_symlinked_profiles_share_one_lock(tmp_path):
    real, (link_a, link_b) = _shared_pair(tmp_path)

    with MemoryStore._file_lock(link_a):
        assert _lock_files(tmp_path) == [Path("shared") / "USER.md.lock"]

    with MemoryStore._file_lock(link_b):
        # The second spelling must reuse the shared lock instead of minting its own: a
        # per-pathname lock is what let two profiles rewrite the same file concurrently.
        assert _lock_files(tmp_path) == [Path("shared") / "USER.md.lock"]
        fd = os.open(str(tmp_path / "shared" / "USER.md.lock"), os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)

    assert real.read_text(encoding="utf-8") == "§\nshared fact\n"


@pytest.mark.platforms("posix")  # flock is fcntl's
def test_a_symlinked_profile_still_writes_through_its_link(tmp_path, monkeypatch):
    real, (_link_a, link_b) = _shared_pair(tmp_path)
    monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: link_b.parent)
    store = MemoryStore(memory_char_limit=500, user_char_limit=300)
    store.load_from_disk()

    assert store.add("user", "second fact")["success"] is True

    assert link_b.is_symlink()
    written = real.read_text(encoding="utf-8")
    assert "shared fact" in written
    assert "second fact" in written
