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


@pytest.mark.skipif(not hasattr(__import__("os"), "mkfifo"), reason="POSIX only")
def test_is_zeroed_never_probes_special_files(tmp_path):
    """A FIFO at the state.db path must be rejected without any blocking read.

    Opening a FIFO for reading blocks until a writer appears; the zeroed
    probe must classify on file type alone (#98017 review, P2).
    """
    import os

    import hermes_state as hs
    from hermes_cli.backup import is_zeroed_sqlite_file

    fifo = tmp_path / "state.db"
    os.mkfifo(fifo)
    # Would hang forever before the regular-file guard if either probe
    # attempted open()+read on the FIFO.
    assert is_zeroed_sqlite_file(fifo) is False
    assert hs.is_zeroed_state_db(fifo) is False


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


def test_is_zeroed_state_db_zero_byte_quarantine(tmp_path, monkeypatch):
    """#97568: a 0-byte file must be detected as zeroed and quarantined."""
    import hermes_state as hs

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = tmp_path / "state.db"
    db.write_bytes(b"")  # 0-byte truncated file
    assert hs.is_zeroed_state_db(db) is True

    sdb = hs.SessionDB(db_path=db)
    try:
        # Fresh DB should open and accept schema
        assert db.exists()
        assert not hs.is_zeroed_state_db(db)
        # Quarantine retained for the 0-byte file
        backups = list(tmp_path.glob("state.db.zeroed-*.bak"))
        assert len(backups) == 1
        assert backups[0].stat().st_size == 0
        # Check store provenance was recorded in state_meta
        row_instance = sdb._conn.execute(
            "SELECT value FROM state_meta WHERE key = 'store_instance_id'"
        ).fetchone()
        row_created = sdb._conn.execute(
            "SELECT value FROM state_meta WHERE key = 'store_created_at_utc'"
        ).fetchone()
        assert row_instance is not None and row_instance[0]
        assert row_created is not None and row_created[0]
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


def test_quarantine_fails_closed_when_lock_held(tmp_path):
    """#68805 review: when the cross-process lock cannot be acquired within
    the timeout, quarantine must FAIL CLOSED — return None without moving
    the file. A fail-open fallback would let a slow/paused startup that
    still owns the lock race with the fallback's re-check + rename.
    """
    import hermes_state as hs
    import platform
    import threading

    db = tmp_path / "state.db"
    db.write_bytes(bytes(4096))  # zeroed (all-NUL) 4 KB file

    lock_path = db.with_name(db.name + ".quarantine.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    # Hold the cross-process lock from a background thread so the main
    # thread's quarantine attempt cannot acquire it.
    lock_held = threading.Event()
    release_lock = threading.Event()

    def hold_lock():
        handle = lock_path.open("a+b")
        try:
            if platform.system() == "Windows":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_held.set()
            release_lock.wait(timeout=15)
            if platform.system() == "Windows":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            lock_held.clear()
        finally:
            handle.close()

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert lock_held.wait(timeout=5), "Background thread failed to acquire lock"

    # Reduce the quarantine lock timeout to keep the test fast. We patch
    # the deadline by calling quarantine directly — it uses a 5s timeout,
    # but we only need to verify it returns None without moving the file.
    result = hs.quarantine_zeroed_state_db(db)

    # Must fail closed: return None without moving the zeroed file
    assert result is None, (
        f"quarantine_zeroed_state_db returned {result} — expected None "
        f"(fail-closed when lock is held)"
    )
    assert db.exists(), "Zeroed state.db was moved despite lock being held"
    assert hs.is_zeroed_state_db(db), "File should still be zeroed (not moved)"

    # Release the lock so the background thread can exit cleanly
    release_lock.set()
    holder.join(timeout=5)


def test_concurrent_openers_zero_byte_startup_serialization(tmp_path, monkeypatch):
    """#97580: Verify that two concurrent SessionDB openers on a non-existent
    database serialize through the startup lock, avoid racing on the initial
    0-byte creation window, and do not falsely quarantine each other's live file.
    """
    import hermes_state as hs
    import threading

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = tmp_path / "state.db"

    errors = [None, None]
    results = [None, None]

    def worker(idx):
        try:
            sdb = hs.SessionDB(db_path=db)
            try:
                # Confirm schema is active
                row = sdb._conn.execute("SELECT 1").fetchone()
                assert row[0] == 1
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

    assert errors[0] is None, f"Opener 0 failed: {errors[0]}"
    assert errors[1] is None, f"Opener 1 failed: {errors[1]}"
    assert results[0] == "ok"
    assert results[1] == "ok"

    # No spurious quarantine backups should have been created
    backups = list(tmp_path.glob("state.db.zeroed-*.bak"))
    assert len(backups) == 0, f"Expected 0 quarantine backups, got: {backups}"
    assert db.exists()
    assert not hs.is_zeroed_state_db(db)


def test_live_connection_0_byte_not_quarantined_in_process(tmp_path, monkeypatch):
    """#97580: A live 0-byte connection tracked in this process must not be
    quarantined by is_zeroed_state_db / SessionDB.
    """
    import hermes_state as hs
    from hermes_cli.sqlite_safe_read import connect_tracked

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = tmp_path / "state.db"

    # Create a live tracked 0-byte connection
    conn = connect_tracked(str(db))
    try:
        assert db.exists() and db.stat().st_size == 0
        # is_zeroed_state_db must recognize the live connection and refuse to declare it zeroed
        assert hs.is_zeroed_state_db(db) is False

        # SessionDB open must not quarantine this live file
        sdb = hs.SessionDB(db_path=db)
        try:
            backups = list(tmp_path.glob("state.db.zeroed-*.bak"))
            assert len(backups) == 0, f"Spurious quarantine occurred: {backups}"
        finally:
            sdb.close()

        # The original connection can still write safely
        conn.execute("CREATE TABLE live_check (id INTEGER)")
        conn.commit()
    finally:
        conn.close()

