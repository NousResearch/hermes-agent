"""Regression tests for memory directory permissions (#66183).

In Docker and other restricted-umask environments (e.g. UGREEN NAS, common
base images with ``umask 0777``), ``Path.mkdir(parents=True, exist_ok=True)``
creates directories with ``000`` permissions when called without an explicit
mode, leaving MEMORY.md / USER.md / the .lock file inaccessible.

The fix replaces all three ``.mkdir()`` calls in ``memory_tool.py`` with a
``_mkdir_p()`` helper that calls ``chmod(0o755)`` after every ``mkdir``,
regardless of umask.
"""

import os
import stat
import tempfile
from pathlib import Path

import pytest

from tools.memory_tool import _mkdir_p, _MEMORY_DIR_MODE


# ---------------------------------------------------------------------------
# _mkdir_p unit tests
# ---------------------------------------------------------------------------

def test_mkdir_p_creates_with_correct_permissions(tmp_path):
    """Freshly created directory gets 0o755 regardless of umask."""
    target = tmp_path / "a" / "b" / "c"
    _mkdir_p(target)
    assert target.is_dir()
    mode = target.stat().st_mode & 0o777
    assert mode == 0o755, f"expected 0o755, got {oct(mode)}"


def test_mkdir_p_repairs_existing_000_permissions(tmp_path):
    """If the directory already exists with 000 perms, chmod repairs it."""
    target = tmp_path / "broken"
    target.mkdir(parents=True, exist_ok=True)
    target.chmod(0o000)
    _mkdir_p(target)
    mode = target.stat().st_mode & 0o777
    assert mode == 0o755, f"expected 0o755 after repair, got {oct(mode)}"


def test_mkdir_p_creates_parents(tmp_path):
    """Parents are created with 0o755 too (via mkdir's parents=True)."""
    target = tmp_path / "a" / "b" / "c"
    _mkdir_p(target)
    assert target.is_dir()
    # Check each parent
    for p in [target, target.parent, target.parent.parent]:
        mode = p.stat().st_mode & 0o777
        assert mode == 0o755, f"{p} has {oct(mode)}, expected 0o755"


def test_mkdir_p_idempotent_on_existing(tmp_path):
    """Calling _mkdir_p on an already-0o755 dir is a no-op."""
    _mkdir_p(tmp_path)
    mode = tmp_path.stat().st_mode & 0o777
    assert mode == 0o755


def test_mkdir_p_works_under_restrictive_umask(tmp_path):
    """Even with umask 0777, _mkdir_p produces 0o755 directories."""
    old_umask = os.umask(0o777)
    try:
        target = tmp_path / "a" / "b" / "c"
        _mkdir_p(target)
        assert target.is_dir()
        mode = target.stat().st_mode & 0o777
        assert mode == 0o755, f"expected 0o755 under umask 0777, got {oct(mode)}"
    finally:
        os.umask(old_umask)


def test_mkdir_p_repairs_existing_000_under_umask(tmp_path):
    """Even if the dir was created with 000 perms, _mkdir_p repairs it."""
    target = tmp_path / "broken"
    target.mkdir(parents=True, exist_ok=True)
    target.chmod(0o000)
    _mkdir_p(target)
    mode = target.stat().st_mode & 0o777
    assert mode == 0o755, f"expected 0o755 after repair, got {oct(mode)}"


def test_memory_tool_creates_dir_with_accessible_perms(tmp_path, monkeypatch):
    """End-to-end: MemoryStore creates the memories dir with 0o755."""
    from tools.memory_tool import MemoryStore, get_memory_dir
    monkeypatch.setattr("tools.memory_tool.get_hermes_home", lambda: tmp_path)
    mt = MemoryStore()
    mt.load_from_disk()
    mem_dir = get_memory_dir()
    assert mem_dir.is_dir()
    mode = mem_dir.stat().st_mode & 0o777
    assert mode == 0o755, f"expected 0o755, got {oct(mode)}"


def test_mkdir_p_repairs_existing_000(tmp_path):
    """If the dir was created with 000 perms (e.g. by old Hermes in Docker), _mkdir_p repairs it."""
    target = tmp_path / "memories"
    target.mkdir(parents=True, exist_ok=True)
    target.chmod(0o000)
    _mkdir_p(target)
    mode = target.stat().st_mode & 0o777
    assert mode == 0o755, f"expected 0o755 after repair, got {oct(mode)}"
