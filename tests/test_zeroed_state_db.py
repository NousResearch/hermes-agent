"""#68474 hardening: zeroed state.db detection + quarantine."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_is_zeroed_state_db_and_quarantine(tmp_path):
    import hermes_state as hs

    db = tmp_path / "state.db"
    db.write_bytes(bytes(1024))
    assert hs.is_zeroed_state_db(db) is True

    q = hs.quarantine_zeroed_state_db(db)
    assert q is not None
    assert q.exists()
    assert not db.exists()
    assert q.read_bytes() == bytes(1024)


def test_sessiondb_opens_fresh_after_zeroed_quarantine(tmp_path, monkeypatch):
    import hermes_state as hs

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = tmp_path / "state.db"
    db.write_bytes(bytes(4096))

    sdb = hs.SessionDB(db_path=db)
    try:
        # Fresh DB should open and accept schema
        assert db.exists()
        assert not hs.is_zeroed_state_db(db)
        # Quarantine retained
        backups = list(tmp_path.glob("state.db.zeroed-*.bak"))
        assert len(backups) == 1
        assert backups[0].stat().st_size == 4096
    finally:
        sdb.close()
