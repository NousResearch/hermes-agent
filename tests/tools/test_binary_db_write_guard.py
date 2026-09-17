"""Tests for the database/binary write guard extension.

A plain-text write can never produce a valid SQLite database, and the text
tools' read->modify->write round-trip re-encodes binary bytes lossily (the
terminal transport decodes stdout with errors="replace"). write_file/patch
must therefore refuse db-family targets (.db/.sqlite/.sqlite3 plus the
-wal/-shm/-journal sidecars, whose suffixed extension defeats a plain
BINARY_EXTENSIONS lookup) and refuse OVERWRITING other existing binary
files — while new-file creation with those extensions stays allowed.
"""

import json
import sqlite3
from pathlib import Path

from tools.file_tools import patch_tool, write_file_tool
from tools.file_tools_write_guards import (
    _SQLITE_DB_FAMILY_RE,
    _check_binary_document_write,
)


def _make_minimal_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO t VALUES (1, 'alpha')")
    conn.commit()
    conn.close()


class TestRegex:
    def test_main_db_names(self):
        for name in ("kanban.db", "app.sqlite", "db.sqlite3", "DATA.DB"):
            assert _SQLITE_DB_FAMILY_RE.search(name), name

    def test_sidecar_names(self):
        # .db-wal splits to ".db-wal" which is NOT in BINARY_EXTENSIONS —
        # the hole this regex closes.
        for name in ("kanban.db-wal", "kanban.db-shm", "kanban.db-journal",
                     "app.sqlite-wal", "x.sqlite3-shm"):
            assert _SQLITE_DB_FAMILY_RE.search(name), name

    def test_non_db_names(self):
        for name in ("notes.txt", "main.py", "hermes.dbk", "mongodb.txt",
                     "nodotfile", "a.dbx"):
            assert not _SQLITE_DB_FAMILY_RE.search(name), name


class TestCheckBinaryDocumentWrite:
    def test_db_always_rejected(self, tmp_path: Path):
        # Even a NON-existing .db is rejected — text can never be a database.
        err = _check_binary_document_write(str(tmp_path / "fresh.db"))
        assert err is not None
        assert "database" in err.lower()

    def test_wal_sidecar_rejected(self, tmp_path: Path):
        err = _check_binary_document_write(str(tmp_path / "kanban.db-wal"))
        assert err is not None
        assert "database" in err.lower()

    def test_plain_text_allowed(self, tmp_path: Path):
        assert _check_binary_document_write(str(tmp_path / "notes.txt")) is None

    def test_new_binary_file_allowed(self, tmp_path: Path):
        # New-file creation with a non-db binary extension stays allowed
        # (mirrors the new-.pdf rule): a text file named .dat is odd but
        # harmless; the corruption argument only applies to overwrites.
        assert _check_binary_document_write(str(tmp_path / "new.png")) is None


class TestWriteFileTool:
    def test_write_file_rejects_existing_db(self, tmp_path: Path):
        db = tmp_path / "kanban.db"
        _make_minimal_db(db)
        original = db.read_bytes()

        result = json.loads(write_file_tool(str(db), "CREATE TABLE x(y);"))

        assert result.get("error"), "text write into .db must be refused"
        assert db.read_bytes() == original, "database bytes must be untouched"
        # And still a valid database.
        conn = sqlite3.connect(db)
        assert conn.execute("PRAGMA integrity_check;").fetchone()[0] == "ok"
        conn.close()

    def test_write_file_rejects_new_db_creation(self, tmp_path: Path):
        fresh = tmp_path / "fresh.db"
        result = json.loads(write_file_tool(str(fresh), "CREATE TABLE x(y);"))
        assert result.get("error"), "new .db via text write must be refused"
        assert not fresh.exists(), "no file may be created"

    def test_write_file_rejects_existing_binary_overwrite(self, tmp_path: Path):
        img = tmp_path / "logo.png"
        img.write_bytes(b"\\x89PNG\\r\\n\\x1a\\n" + b"\\x00" * 64)
        original = img.read_bytes()
        result = json.loads(write_file_tool(str(img), "not an image"))
        assert result.get("error"), "binary overwrite must be refused"
        assert img.read_bytes() == original


class TestPatchTool:
    def test_patch_rejects_existing_db(self, tmp_path: Path):
        db = tmp_path / "kanban.db"
        _make_minimal_db(db)
        original = db.read_bytes()

        result = json.loads(
            patch_tool(mode="replace", path=str(db),
                       old_string="alpha", new_string="beta"))

        assert result.get("error"), "patch into .db must be refused"
        assert db.read_bytes() == original, "database bytes must be untouched"
        conn = sqlite3.connect(db)
        assert conn.execute("PRAGMA integrity_check;").fetchone()[0] == "ok"
        conn.close()

    def test_patch_rejects_wal_sidecar(self, tmp_path: Path):
        wal = tmp_path / "kanban.db-wal"
        wal.write_bytes(b"\\x00\\x01\\x02\\x00data")
        original = wal.read_bytes()
        result = json.loads(
            patch_tool(mode="replace", path=str(wal),
                       old_string="data", new_string="DATA"))
        assert result.get("error"), "patch into .db-wal must be refused"
        assert wal.read_bytes() == original
