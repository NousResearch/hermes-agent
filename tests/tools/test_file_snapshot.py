"""File snapshots tolerate stat/fstat ctime differences, not concurrent changes."""

import hashlib
import json
import os
import stat
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools import file_tools_read_tracking as tracking


_FIELDS = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")


@pytest.mark.parametrize("ctime_offset", [0, 100])
@pytest.mark.parametrize("change", [None, "identity", *(
    f"{api}:{field}" for api in ("stat", "fstat") for field in _FIELDS
)])
def test_snapshot_requires_stable_path_and_descriptor(tmp_path, monkeypatch, ctime_offset, change):
    path = tmp_path / "snapshot.txt"
    content = b"raw bytes\r\nincluding \x1a\n"
    path.write_bytes(content)
    metadata = dict(st_mode=stat.S_IFREG, st_dev=1, st_ino=2,
                    st_size=len(content), st_mtime_ns=3, st_ctime_ns=4)
    path_before = SimpleNamespace(**metadata)
    path_after = SimpleNamespace(**metadata)
    fd_before = SimpleNamespace(**{**metadata, "st_ctime_ns": metadata["st_ctime_ns"] + ctime_offset})
    fd_after = SimpleNamespace(**vars(fd_before))
    if change == "identity":
        # Replacement between path stat and open, stable thereafter on the fd.
        fd_before.st_ino += 1
        fd_after.st_ino += 1
    elif change is not None:
        api, field = change.split(":")
        snapshot = path_after if api == "stat" else fd_after
        setattr(snapshot, field, getattr(snapshot, field) + 1)

    # Only this module sees synthetic metadata; open/read/hash use real bytes.
    fake_os = SimpleNamespace(**vars(os))
    fake_os.stat = Mock(side_effect=[path_before, path_after])
    fake_os.fstat = Mock(side_effect=[fd_before, fd_after])
    monkeypatch.setattr(tracking, "os", fake_os)
    version = tracking._file_version(str(path))
    if change is None:
        assert version == (*tuple(metadata[field] for field in _FIELDS), hashlib.sha256(content).digest())
    else:
        assert version is None


@pytest.mark.platforms("any")
@pytest.mark.parametrize("external_change", [None, "edit", "replace"])
def test_real_read_write_baseline_preserves_freshness(tmp_path, external_change):
    from tools.file_tools import clear_file_ops_cache
    from tools.registry import registry

    path = tmp_path / "notes.txt"
    path.write_bytes(b"original\n")
    task = f"snapshot-{external_change}"

    def call(name, **args):
        return json.loads(registry.dispatch(name, {"path": str(path), **args}, task_id=task))

    try:
        read = call("read_file")
        assert "error" not in read, read
        assert "original" in read["content"]
        tracking.reset_file_dedup(task)
        assert tracking._has_full_write_baseline(str(path), task)
        written = call("write_file", content="own edit\n")
        assert "error" not in written, written
        assert written.get("verified") is True
        assert path.read_bytes() == b"own edit\n"
        assert tracking._has_full_write_baseline(str(path), task)

        if external_change is not None:
            stamp = path.stat()
            target = path if external_change == "edit" else tmp_path / "replacement.txt"
            target.write_bytes(b"external\n")
            os.utime(target, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            if external_change == "replace":
                os.replace(target, path)
            # Neither size nor mtime alone can spot this edit/replacement.
            assert path.stat().st_size == stamp.st_size
            assert path.stat().st_mtime_ns == stamp.st_mtime_ns
            refused = call("write_file", content="clobber\n")
            assert refused.get("stale_write_blocked"), refused
            assert path.read_bytes() == b"external\n"
        else:
            written = call("write_file", content="next edit\n")
            assert "error" not in written, written
            assert path.read_bytes() == b"next edit\n"
    finally:
        clear_file_ops_cache(task)
        tracking._read_tracker.pop(task, None)
