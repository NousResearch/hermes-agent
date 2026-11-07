"""Real filesystem read-dedup metadata boundaries; no model or live home."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

from tools import file_tools as file_tools
from tools import file_tools_read_tracking as tracking
from tests.tools.test_file_read_guards import _make_fake_ops

FIELDS = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
CONTENT = "line one\nline two\n"


def sample_boundaries(count=64):
    rows = []
    for index in range(count):
        with tempfile.TemporaryDirectory(prefix="dedup-boundary-") as directory:
            path = Path(directory) / "loop_test.txt"
            path.write_text(CONTENT, encoding="utf-8")
            before = os.stat(path)
            with path.open("rb") as stream:
                descriptor = os.fstat(stream.fileno())
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            after = os.stat(path)
            metadata = tracking._file_metadata(str(path))
            version = tracking._file_version(str(path))
            tracking._read_tracker.clear()
            fake = _make_fake_ops(content=CONTENT, file_size=20)
            with patch.object(file_tools, "_get_file_ops", return_value=fake):
                results = [json.loads(file_tools.read_file_tool(str(path), task_id="loop"))
                           for _ in range(3)]
            rows.append({"index": index, "path": str(path),
                         "stat_before": [getattr(before, field) for field in FIELDS],
                         "fstat": [getattr(descriptor, field) for field in FIELDS],
                         "stat_after": [getattr(after, field) for field in FIELDS],
                         "metadata": metadata, "version": str(version), "sha256": digest,
                         "results": results,
                         "tracker": str(tracking._read_tracker.get("loop"))})
    tracking._read_tracker.clear()
    return rows


def test_real_unchanged_files_dedup_then_third_read_blocks():
    rows = sample_boundaries()
    failed = [row for row in rows if not row["results"][1].get("dedup")
              or "BLOCKED" not in row["results"][2].get("error", "")]
    print(json.dumps({"python": sys.executable, "module": file_tools.__file__,
                      "tracking": tracking.__file__, "home": os.environ.get("HERMES_HOME"),
                      "fields": FIELDS, "sample_count": len(rows), "failed": failed}))
    assert not failed, failed


def test_rewritten_then_stable_file_has_byte_snapshot(tmp_path):
    import time

    path = tmp_path / "rewritten.txt"
    path.write_bytes(b"old bytes")
    time.sleep(0.03)
    path.write_bytes(b"new bytes")
    before = tracking._file_metadata(str(path))
    version = tracking._file_version(str(path))
    assert version is not None, {"metadata": before, "version": version}
    assert version[:-1] == before
    assert version[-1] == hashlib.sha256(b"new bytes").digest()


def test_path_rebound_before_open_refuses_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "rebound.txt"
    replacement = tmp_path / "replacement.txt"
    path.write_bytes(b"old bytes")
    replacement.write_bytes(b"new bytes")
    before = os.stat(path)
    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
    real_open = os.open
    rebound = False

    def open_rebound(filename, flags, *args, **kwargs):
        nonlocal rebound
        if str(filename) == str(path) and not rebound:
            rebound = True
            os.replace(replacement, path)
        return real_open(filename, flags, *args, **kwargs)

    monkeypatch.setattr(tracking.os, "open", open_rebound)
    assert tracking._file_version(str(path)) is None
    assert rebound
    assert path.read_bytes() == b"new bytes"


def test_ctime_drift_during_hash_refuses_with_restored_mtime(tmp_path, monkeypatch):
    import time

    path = tmp_path / "changed.txt"
    path.write_bytes(b"old bytes")
    before = os.stat(path)
    real_digest = hashlib.file_digest

    def change_after_hash(stream, algorithm):
        digest = real_digest(stream, algorithm)
        time.sleep(0.03)
        path.write_bytes(b"new bytes")
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        return digest

    monkeypatch.setattr(tracking.hashlib, "file_digest", change_after_hash)
    assert tracking._file_version(str(path)) is None
    assert os.stat(path).st_size == before.st_size
    assert os.stat(path).st_mtime_ns == before.st_mtime_ns


if __name__ == "__main__":
    test_real_unchanged_files_dedup_then_third_read_blocks()
