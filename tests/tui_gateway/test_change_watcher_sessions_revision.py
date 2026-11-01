"""sessions.changed on a store that maintains ``sessions_revision``: the watcher reads one
``state_meta`` row instead of hashing every sessions row, and falls back to the exact digest
whenever the store cannot vouch for the revision (legacy, missing or foreign triggers)."""
import os
import sqlite3

import pytest

import hermes_state
from hermes_state import SessionDB
from tui_gateway import server


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_watcher_home", lambda: tmp_path)
    monkeypatch.setattr(server, "_served_profile_homes", [])
    monkeypatch.setattr(server, "_sessions_db_sig_cache", {})
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", "cli")
    db.set_session_title("s1", "Original")
    yield tmp_path, db
    db.close()


def _tables_read_by_watcher(monkeypatch):
    reads = []
    real = hermes_state._connect_tracked_db

    def traced(*args, **kwargs):
        conn = real(*args, **kwargs)

        def authorizer(action, arg1, *_rest):
            if action == sqlite3.SQLITE_READ:
                reads.append(arg1)
            return sqlite3.SQLITE_OK
        conn.set_authorizer(authorizer)
        return conn
    monkeypatch.setattr(hermes_state, "_connect_tracked_db", traced)
    return reads


def test_revision_store_signature_never_reads_session_rows(home, monkeypatch):
    reads = _tables_read_by_watcher(monkeypatch)

    server._sessions_sig()

    assert reads and "sessions" not in reads


def test_heartbeat_keeps_and_session_change_moves_revision_signature(home):
    _path, db = home
    before = server._sessions_sig()

    db.touch_session_activity("s1", description="thinking")
    assert server._sessions_sig() == before

    db.set_session_archived("s1", True)
    assert server._sessions_sig() != before


def test_missing_trigger_falls_back_to_the_exact_digest(home, monkeypatch):
    path, db = home
    conn = sqlite3.connect(path / "state.db")
    conn.execute("DROP TRIGGER sessions_revision_update")
    conn.commit()
    conn.close()
    reads = _tables_read_by_watcher(monkeypatch)
    before = server._sessions_sig()
    assert "sessions" in reads

    db.touch_session_activity("s1", description="thinking")
    assert server._sessions_sig() == before
    db.set_session_title("s1", "Renamed")
    assert server._sessions_sig() != before


def _same_shape_store(path, title):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, title TEXT)")
    conn.execute("INSERT INTO sessions VALUES ('s1', ?)", (title,))
    conn.commit()
    conn.close()


@pytest.mark.parametrize("kind", ["legacy", "revision"])
def test_replaced_store_with_same_size_and_mtime_moves_signature(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(server, "_served_profile_homes", [])
    monkeypatch.setattr(server, "_sessions_db_sig_cache", {})
    live, other = tmp_path / "live", tmp_path / "other"
    live.mkdir()
    other.mkdir()
    for home_dir, title in ((live, "AAAA"), (other, "BBBB")):
        if kind == "legacy":
            _same_shape_store(home_dir / "state.db", title)
        else:
            db = SessionDB(home_dir / "state.db")
            db.create_session("s1", "cli")
            db.set_session_title("s1", title)
            db.close()
            for suffix in ("-wal", "-shm"):
                (home_dir / f"state.db{suffix}").unlink(missing_ok=True)
    if kind == "revision":
        # Same store identity and revision: only the file itself tells the copies apart.
        conn = sqlite3.connect(other / "state.db")
        reader = sqlite3.connect(live / "state.db")
        try:
            for key in ("store_instance_id", "sessions_revision"):
                value = reader.execute("SELECT value FROM state_meta WHERE key = ?", (key,)).fetchone()[0]
                conn.execute("UPDATE state_meta SET value = ? WHERE key = ?", (value, key))
            conn.commit()
        finally:
            reader.close()
            conn.close()
    stamp = 1_700_000_000_000_000_000
    for home_dir in (live, other):
        os.utime(home_dir / "state.db", ns=(stamp, stamp))
    assert (live / "state.db").stat().st_size == (other / "state.db").stat().st_size
    monkeypatch.setattr(server, "_watcher_home", lambda: live)

    before = server._sessions_sig()
    os.replace(other / "state.db", live / "state.db")
    assert (live / "state.db").stat().st_mtime_ns == stamp

    assert server._sessions_sig() != before


def test_busy_read_after_a_good_revision_keeps_the_signature(home, monkeypatch):
    _path, db = home
    before = server._sessions_sig()

    def locked(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(hermes_state, "_connect_tracked_db", locked)
    db.touch_session_activity("s1", description="moves the mtime")

    assert server._sessions_sig() == before


def test_commit_inside_the_same_mtime_tick_moves_revision_signature(home, monkeypatch):
    import time

    _path, db = home
    frozen = time.time_ns()
    monkeypatch.setattr(server, "_newest_mtime_ns", lambda _paths: frozen)

    before = server._sessions_sig()
    db.set_session_title("s1", "Same tick")

    assert server._sessions_sig() != before


def test_served_profile_stores_are_tracked_separately(home, monkeypatch, tmp_path_factory):
    _path, db = home
    sibling_home = tmp_path_factory.mktemp("sibling")
    sibling = SessionDB(sibling_home / "state.db")
    sibling.create_session("s1", "cli")
    monkeypatch.setattr(server, "_served_profile_homes", [sibling_home])
    try:
        first = server._sessions_sig()
        sibling.set_session_title("s1", "Sibling only")
        second = server._sessions_sig()
        assert second[0] == first[0] and second[1] != first[1]

        sibling.touch_session_activity("s1", description="beat")
        db.touch_session_activity("s1", description="beat")
        assert server._sessions_sig() == second
    finally:
        sibling.close()


def test_digest_reader_on_a_revision_store_moves_exactly_when_the_revision_does(home, monkeypatch):
    """A reader without revision support (an older build, or the digest fallback) on the same
    triggered store agrees with the revision on which writes are session changes."""
    import hermes_state_common

    path, db = home
    real = hermes_state_common.read_session_revision

    def both():
        sigs = []
        for reader in (real, lambda _conn: None):
            monkeypatch.setattr(hermes_state_common, "read_session_revision", reader)
            server._sessions_db_sig_cache.clear()
            sigs.append(server._sessions_sig())
        return sigs

    raw = sqlite3.connect(path / "state.db", isolation_level=None)
    writes = {
        "heartbeat": lambda: db.touch_session_activity("s1", description="beat"),
        "noop": lambda: raw.execute("UPDATE sessions SET title = title WHERE id = 's1'"),
        "untracked": lambda: raw.execute("UPDATE sessions SET input_tokens = input_tokens + 1"),
        "title": lambda: db.set_session_title("s1", "Renamed"),
        "null_to_value": lambda: raw.execute("UPDATE sessions SET last_read_at = 7 WHERE id = 's1'"),
        "value_to_null": lambda: raw.execute("UPDATE sessions SET last_read_at = NULL WHERE id = 's1'"),
        "insert": lambda: db.create_session("s2", "cli"),
        "delete": lambda: raw.execute("DELETE FROM sessions WHERE id = 's2'"),
    }
    moved = {}
    before = both()
    assert before[0][0][0] == "revision" and before[1][0][0] != "revision"
    for name, write in writes.items():
        write()
        after = both()
        moved[name] = (after[0] != before[0], after[1] != before[1])
        before = after
    raw.close()

    assert all(revision == digest for revision, digest in moved.values()), moved
    assert [n for n, (rev, _d) in moved.items() if rev] == [
        "title", "null_to_value", "value_to_null", "insert", "delete"]
