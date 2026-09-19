"""Data-only restored-store fixture; not an unmigrated cold service startup.

The service is constructed inertly against current schema, publishes genuine data,
then its closed SQLite pathname receives a valid historical-layout reconstruction.
Production service/store handles retain paths, not connections, so this models a
restored historical DB behind an existing handle. No service worker is started.
DDL is retained verbatim from f086299b3406815115a3765fd7257998727cffee;
rows are projected from current publication APIs onto those historical columns.
No table/trigger is dropped and no production initializer or authorizer is mocked.
"""
from contextlib import closing
from pathlib import Path
import os
import sqlite3


def restore_historical_layout(db_path):
    rebuilt = db_path.with_name("historical-restored.db")
    ddl = Path(__file__).with_name("fixtures").joinpath("files_viewer_historical.sql").read_text()
    with closing(sqlite3.connect(db_path)) as source, closing(sqlite3.connect(rebuilt)) as dest:
        dest.executescript(ddl)
        tables = [row[0] for row in dest.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        for table in tables:
            columns = [row[1] for row in dest.execute(f"PRAGMA table_info({table})")]
            fields = ",".join(columns)
            rows = source.execute(f"SELECT {fields} FROM {table}").fetchall()
            dest.executemany(f"INSERT INTO {table} ({fields}) VALUES ({','.join('?' for _ in columns)})", rows)
        dest.commit()
        assert dest.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert dest.execute("PRAGMA foreign_key_check").fetchall() == []
        assert not dest.execute("SELECT 1 FROM sqlite_master WHERE type='trigger'").fetchall()
        assert not {"hosted_room_quarantine", "hosted_room_disband_fences"}.intersection(tables)
        assert "catalog_name" not in {r[1] for r in dest.execute("PRAGMA table_info(hosted_room_attachments)")}
    os.replace(rebuilt, db_path)
