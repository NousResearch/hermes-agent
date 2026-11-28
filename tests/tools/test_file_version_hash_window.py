"""Regression for #128639: a Windows path ctime cannot protect an in-flight digest."""

import hashlib
import os
from types import SimpleNamespace

from tools import file_tools_read_tracking as tracking


def test_rewrite_during_hash_with_restored_mtime_rejects_stale_digest(tmp_path, monkeypatch):
    """A stable creation-time path stat must not validate bytes changed mid-read."""
    path = tmp_path / "snapshot.txt"
    first = b"first-v1\n"
    second = b"other-v2\n"
    assert len(first) == len(second)
    path.write_bytes(first)
    original = os.stat(path)
    real_stat = os.stat
    real_fstat = os.fstat
    real_digest = hashlib.file_digest
    edited = False

    class CtimeView:
        def __init__(self, stat_value, ctime):
            self.stat_value = stat_value
            self.ctime = ctime

        def __getattr__(self, name):
            return self.ctime if name == "st_ctime_ns" else getattr(self.stat_value, name)

    # Mirror Windows' stable path creation time vs handle change time without
    # altering the interpreter's OS or testing an alternate implementation.
    def path_stat(target, *args, **kwargs):
        result = real_stat(target, *args, **kwargs)
        return CtimeView(result, original.st_ctime_ns) if os.fspath(target) == str(path) else result

    def handle_stat(fd):
        result = real_fstat(fd)
        return CtimeView(result, original.st_ctime_ns + (1_000_000 if edited else 0))

    def mutate_after_hash(stream, algorithm):
        nonlocal edited
        digest = real_digest(stream, algorithm)
        path.write_bytes(second)
        os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
        edited = True
        return digest

    filesystem_view = SimpleNamespace(**vars(os))
    filesystem_view.stat = path_stat
    filesystem_view.fstat = handle_stat
    monkeypatch.setattr(tracking, "os", filesystem_view)
    monkeypatch.setattr(tracking.hashlib, "file_digest", mutate_after_hash)

    assert tracking._file_version(str(path)) is None
    assert edited
    assert path.read_bytes() == second
    assert path.stat().st_mtime_ns == original.st_mtime_ns
