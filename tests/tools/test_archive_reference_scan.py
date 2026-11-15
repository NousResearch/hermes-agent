"""Large transcript reference scans stay complete, and failed scans keep archives."""

import os
import sqlite3
import time
from unittest.mock import Mock

import pytest

from hermes_state import SessionDB
from tools.tool_result_storage import _sweep_archive_dir, referenced_archive_refs


@pytest.fixture
def busy_home(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s1", "cli")
        db._write_sql(
            "INSERT INTO messages(session_id, role, content, api_content, timestamp) VALUES (?, ?, ?, ?, ?)",
            [("s1", "tool", f"cache/pruned/{i}.json" if i % 2 else "display",
              f"cache/pruned/{i}.json" if not i % 2 else None, 1.0) for i in range(20_001)],
            many=True,
        )
        yield tmp_path
    finally:
        db.close()


def test_reference_scan_has_no_row_cap(busy_home):
    assert referenced_archive_refs(busy_home) == {f"cache/pruned/{i}.json" for i in range(20_001)}


def _old_archives(home):
    archive_dir = home / "cache" / "pruned"
    archive_dir.mkdir(parents=True)
    for name in ("0.json", "20000.json", "orphan.json"):
        path = archive_dir / name
        path.write_text("original")
        os.utime(path, (1, 1))
    return archive_dir


def test_sweep_removes_only_unreferenced_files_after_large_scan(busy_home):
    archive_dir = _old_archives(busy_home)
    assert _sweep_archive_dir(archive_dir, time.time() - 3600) == 1
    assert {p.name for p in archive_dir.iterdir()} == {"0.json", "20000.json"}


@pytest.mark.parametrize("error", [sqlite3.OperationalError, OSError])
def test_mid_iteration_error_keeps_every_archive(tmp_path, monkeypatch, error):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.close()
    archive_dir = _old_archives(tmp_path)
    yielded = []

    def failing_rows():
        yielded.append(True)
        yield ("cache/pruned/0.json", None)
        raise error("synthetic read failure")

    cursor = Mock()
    cursor.__iter__ = lambda _: failing_rows()
    cursor.fetchall.side_effect = lambda: list(failing_rows())
    connection = Mock()
    connection.execute.return_value = cursor
    monkeypatch.setattr("hermes_cli.sqlite_safe_read.connect_tracked", lambda *a, **kw: connection)

    assert referenced_archive_refs(tmp_path) is None
    connection.close.assert_called_once_with()
    assert _sweep_archive_dir(archive_dir, time.time() - 3600) == 0
    assert len(yielded) == 2
    assert {p.name for p in archive_dir.iterdir()} == {"0.json", "20000.json", "orphan.json"}


def test_missing_database_has_no_references(tmp_path):
    assert referenced_archive_refs(tmp_path) == set()
