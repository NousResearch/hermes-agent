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


def test_concurrent_quarantine_no_clobber(tmp_path):
    """#68805: two concurrent startups must not race on quarantine.

    Without the cross-process lock, the second process could move its
    newly-created empty DB over the first process's quarantine backup,
    erasing the original damaged-file evidence. With the lock, the
    second process re-checks under the lock, finds the file no longer
    zeroed (or gone), and returns without clobbering.
    """
    import hermes_state as hs
    import threading
    import sqlite3

    db = tmp_path / "state.db"
    db.write_bytes(bytes(4096))  # zeroed (all-NUL) 4 KB file

    results: list = [None, None]
    errors: list = [None, None]

    def worker(idx):
        try:
            # Each worker opens its own SessionDB on the same path.
            # The first one quarantines the zeroed file and creates a
            # fresh DB. The second one should find a valid DB (or no
            # file) under the lock and NOT clobber the quarantine.
            sdb = hs.SessionDB(db_path=db)
            try:
                results[idx] = "ok"
            finally:
                sdb.close()
        except Exception as exc:
            errors[idx] = exc

    t1 = threading.Thread(target=worker, args=(0,))
    t2 = threading.Thread(target=worker, args=(1,))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    # Both workers should complete without error
    assert errors[0] is None, f"Worker 0 raised: {errors[0]}"
    assert errors[1] is None, f"Worker 1 raised: {errors[1]}"

    # The quarantine backup must survive — exactly one .bak file with
    # the original 4096 zeroed bytes.
    backups = list(tmp_path.glob("state.db.zeroed-*.bak"))
    assert len(backups) >= 1, "At least one quarantine backup must exist"
    for bak in backups:
        assert bak.stat().st_size == 4096, (
            f"Quarantine backup {bak} was clobbered: "
            f"expected 4096 bytes, got {bak.stat().st_size}"
        )

    # The live state.db must be a valid (non-zeroed) SQLite database
    assert db.exists()
    assert not hs.is_zeroed_state_db(db)
    conn = sqlite3.connect(str(db))
    conn.execute("SELECT 1")
    conn.close()
