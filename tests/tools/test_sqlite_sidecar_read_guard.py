"""SQLite sidecars are binary: reading them hands the model lossy text.

A WAL-mode database's ``-wal``/``-shm`` pass a dot-suffix binary check, so
read_file returned mojibake that write_file could then write back, destroying
the file for the process that still held the WAL generation.
"""
from tools.file_tools import read_file_tool, write_file_tool


def test_read_file_refuses_sqlite_sidecars(tmp_path):
    for name in ("state.db-wal", "state.db-shm", "state.db-journal", "app.sqlite3-wal"):
        path = tmp_path / name
        path.write_bytes(b"\x37\x7f\x06\x82" * 8)
        out = read_file_tool(str(path))
        assert "Cannot read binary file" in out, (name, out[:200])


def test_a_normal_text_file_still_reads(tmp_path):
    path = tmp_path / "notes.sql"
    path.write_text("select 1;\n", encoding="utf-8")
    assert "select 1;" in read_file_tool(str(path))


def test_refused_sidecar_read_keeps_the_write_off_disk(tmp_path):
    path = tmp_path / "state.db-wal"
    path.write_bytes(b"\x37\x7f\x06\x82" * 8)
    assert "Cannot read binary file" in read_file_tool(str(path))
    before = path.read_bytes()
    write_file_tool(str(path), "PLAIN TEXT OVERWRITE")
    assert path.read_bytes() == before
