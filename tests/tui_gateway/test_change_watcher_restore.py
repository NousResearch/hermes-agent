"""The fast session-list path must also notice native in-place restores."""

import errno
import sqlite3
from contextlib import closing

import pytest

from persistence.sqlite.admin import _backup
from persistence_assembly import open_persistence, open_persistence_reader, open_store_admin, profile_scope
from tests.fixtures.storage_holder import omit_session_list_revision
from tui_gateway import server


def _store(home, title=None):
    home.mkdir()
    path = home / 'state.db'
    with open_persistence(profile_scope(home=home), eager=True):
        if title is not None:
            with closing(sqlite3.connect(path)) as conn, conn:
                conn.execute(
                    "INSERT INTO sessions (id, source, started_at, title) VALUES ('sample', 'cli', 1, ?)",
                    (title,),
                )
    return path


def _stamp(path):
    with open_persistence_reader(path) as store:
        return store.maintenance.session_projection_revision()


@pytest.mark.parametrize('legacy', [False, True], ids=['equal-revision', 'legacy-reopened-before-poll'])
def test_in_place_restore_invalidates_the_session_list(tmp_path, monkeypatch, legacy):
    live = _store(tmp_path / 'live', None if legacy else 'Original')
    snapshot = _store(tmp_path / 'snapshot', 'Restored')
    tracked = _stamp(live)
    assert tracked is not None
    if legacy:
        with closing(sqlite3.connect(snapshot)) as conn, conn:
            omit_session_list_revision(conn)
        assert _stamp(snapshot) is None
    else:
        assert _stamp(snapshot) == tracked

    def no_space_for_staging(*_args, **_kwargs):
        raise OSError(errno.ENOSPC, 'No room for a second full SQLite image')

    monkeypatch.setattr(_backup, '_safe_copy_db', no_space_for_staging)
    monkeypatch.setattr(server, '_watcher_home', lambda: live.parent)
    monkeypatch.setattr(server, '_served_profile_homes', [])
    monkeypatch.setattr(server, '_sessions_db_sig_cache', {})
    before = server._sessions_sig()
    original = snapshot.read_bytes()
    inode = live.stat().st_ino
    assert open_store_admin(live, read_only=False).restore_from(snapshot)
    assert live.stat().st_ino == inode
    assert snapshot.read_bytes() == original
    with closing(sqlite3.connect(live)) as conn:
        assert conn.execute('SELECT title FROM sessions').fetchall() == [('Restored',)]
    if legacy:
        # The reader never sees the restored database without its revision triggers.
        with open_persistence(profile_scope(home=live.parent), eager=True):
            assert _stamp(live) is not None
            after = server._sessions_sig()
    else:
        assert _stamp(live)[0] == tracked[0]
        after = server._sessions_sig()
    assert after != before
