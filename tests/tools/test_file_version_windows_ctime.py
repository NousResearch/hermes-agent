"""A file the task fully read stays writable on Windows.

Regression for #146022: on Windows ``os.fstat(fd)`` and ``os.stat(path)`` can report
different ``st_ctime_ns`` for the same just-written file (~1 in 4 fresh files on the
reporter's host), so ``_file_version()`` returned None for every version call, no write
baseline was ever recorded (reads actively purge it), and ``write_file`` refused forever
after any full read.
"""

import json
import os

import pytest

from tools.file_tools import clear_file_ops_cache
from tools.file_tools_read_tracking import _file_metadata, _file_version
from tools.registry import registry

pytestmark = pytest.mark.platforms("windows")

CONTENT = "".join(f"line {n}\n" for n in range(1, 61))


def _dispatch(name, task_id, **arguments):
    result = registry.dispatch(name, arguments, task_id=task_id)
    assert isinstance(result, str)
    return json.loads(result)


def _read(path, task_id, **paging):
    result = _dispatch("read_file", task_id, path=str(path), **paging)
    assert "error" not in result, result
    return result


def _write(path, task_id, content):
    return _dispatch("write_file", task_id, path=str(path), content=content)


def _fresh_diverged_file(tmp_path, prefix):
    """A freshly written file whose fstat/stat ctime views disagree, or None.

    The divergence is a host race (~1 in 4 fresh files here), so keep writing until
    one diverges; comparing the raw views keeps this independent of the fix.
    """
    for i in range(40):
        path = tmp_path / f"{prefix}-{i}.md"
        path.write_text(CONTENT, encoding="utf-8")
        with os.fdopen(os.open(str(path), os.O_RDONLY | getattr(os, "O_BINARY", 0)), "rb") as stream:
            before = os.fstat(stream.fileno())
        after = os.stat(str(path))
        if before.st_ctime_ns != after.st_ctime_ns:
            return path
    return None


def _diverged_fstat(monkeypatch):
    """Stub only the fd view's ctime, leaving every other field (and stat) real."""
    import types

    real_fstat = os.fstat

    def _fstat(fd):
        st = real_fstat(fd)
        # A namespace is enough: _file_version only reads attributes off the view.
        return types.SimpleNamespace(**{
            name: getattr(st, name) for name in (
                "st_mode", "st_ino", "st_dev", "st_nlink", "st_uid", "st_gid",
                "st_size", "st_atime", "st_mtime", "st_ctime",
                "st_atime_ns", "st_mtime_ns",
            )
        }, st_ctime_ns=st.st_ctime_ns + 1_000_000)

    monkeypatch.setattr(os, "fstat", _fstat)


def test_diverged_ctime_views_still_version(tmp_path, monkeypatch):
    """Deterministic core: disagreeing ctime views must not make the file unversionable."""
    _diverged_fstat(monkeypatch)
    path = tmp_path / "stubbed.md"
    path.write_text(CONTENT, encoding="utf-8")
    version = _file_version(str(path))
    assert version is not None
    # The stored version is the path view, so it equals _file_metadata (the contract
    # tools/file_tools.py:551 and _note_read_coverage's caller rely on).
    assert version[:-1] == _file_metadata(str(path))


def test_paged_full_read_of_diverged_file_allows_write(tmp_path):
    """Repro A from the issue, end to end: two pages to EOF, then the write lands."""
    path = _fresh_diverged_file(tmp_path, "paged")
    if path is None:
        pytest.skip("host did not produce divergent ctime views in 40 fresh files")
    assert path is not None
    task = "ctime-paged-external"
    try:
        first = _read(path, task, offset=1, limit=30)
        assert first["truncated"] is True
        second = _read(path, task, offset=31, limit=30)
        assert second["truncated"] is False
        written = _write(path, task, "replacement\n")
        assert "error" not in written, written
        assert path.read_text(encoding="utf-8") == "replacement\n"
    finally:
        clear_file_ops_cache(task)


def test_single_full_read_of_diverged_file_allows_write(tmp_path):
    """Repro A step 6, end to end: one unpaginated full read unblocks the write."""
    path = _fresh_diverged_file(tmp_path, "single")
    if path is None:
        pytest.skip("host did not produce divergent ctime views in 40 fresh files")
    assert path is not None
    task = "ctime-single-external"
    try:
        _read(path, task)
        written = _write(path, task, "replacement\n")
        assert "error" not in written, written
        assert path.read_text(encoding="utf-8") == "replacement\n"
    finally:
        clear_file_ops_cache(task)


def test_changed_bytes_after_baseline_still_refuse_write(tmp_path):
    """Weakening the comparison must not open the gate: new bytes stay refused."""
    task = "ctime-tamper"
    try:
        path = tmp_path / "tamper.txt"
        path.write_text("one\ntwo\nthree\n", encoding="utf-8")
        _read(path, task)
        path.write_text("other\nsecond\nthird\n", encoding="utf-8")
        refused = _write(path, task, "replacement\n")
        assert refused.get("stale_write_blocked"), refused
        assert path.read_text(encoding="utf-8") == "other\nsecond\nthird\n"
    finally:
        clear_file_ops_cache(task)
