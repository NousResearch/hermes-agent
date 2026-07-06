"""Session persistence must never write into the wrong profile's ``state.db``.

Regression tests for issue #59566. A Desktop/Dashboard ``session.create``
under the launch/default profile was persisting its row + messages into a
different profile's ``state.db`` whenever the gateway's cached
``_get_db()`` handle had been initialised under that other profile
(app-global remote mode). The fix forces every session-owned DB read/write
to resolve the DB path from the live session/profile instead of relying
on the process-global handle.

Coverage:

1. ``_ensure_session_db_row`` writes to the launch profile's ``state.db``,
   even when ``_get_db()`` was previously initialised to another profile.
2. ``_session_db`` (the context manager) yields the launch profile's db.
3. ``_session_db_path`` returns the explicit launch profile path for
   default sessions and the named-profile path for non-default sessions.
4. ``_default_profile_state_db_path`` always tracks ``_hermes_home``.
5. Migration seam: legacy rows in the wrong profile's db are NOT touched
   by a new default-session write (so the fix doesn't try to "rewrite
   history", it just stops the bleed).
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_state import SessionDB


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_profile_homes(tmp_path, monkeypatch):
    """Two distinct HERMES_HOME directories plus a clean module import.

    The gateway reads ``get_hermes_home()`` at import time and caches the
    result in ``_hermes_home``. To exercise the bug reliably we need that
    module-level cache to point at a launch/default profile while we also
    construct an isolated ``SessionDB`` under a separate profile. We do
    that by patching ``tui_gateway.server._hermes_home`` to the launch
    home before exercising the helpers.
    """
    launch_home = tmp_path / "default-home"
    other_home = tmp_path / "other-profile"
    for p in (launch_home, other_home):
        (p / "skills").mkdir(parents=True, exist_ok=True)
        (p / "cache").mkdir(parents=True, exist_ok=True)

    monkeypatch.delenv("HERMES_HOME", raising=False)

    import tui_gateway.server as server

    monkeypatch.setattr(server, "_hermes_home", str(launch_home))

    return server, launch_home, other_home


def _make_session(*, profile_home: str | None = None, key: str = "test-key") -> dict:
    """Build a minimal session dict for the helpers under test."""
    return {
        "session_key": key,
        "profile_home": profile_home,
        "agent": None,
        "history": [],
        "history_lock": threading.Lock(),
    }


def _seed_row(db_path: Path, session_key: str) -> None:
    """Insert a bare row so we can probe cross-profile leakage."""
    db = SessionDB(db_path=db_path)
    try:
        db.create_session(session_id=session_key, source="tui", session_key=session_key)
    finally:
        db.close()


def _rows(db_path: Path) -> list[dict]:
    """Read every session row from ``db_path`` (raw SQL).

    Returns rows keyed by ``id`` (the primary key the gateway passes as
    ``session_id`` / the row's session_key). The gateway's
    ``_ensure_session_db_row`` calls ``db.create_session(key, ...)`` so the
    session identity is on ``id`` — ``session_key`` is null on first write.
    """
    db = SessionDB(db_path=db_path)
    try:
        cur = db._conn.cursor()
        try:
            cur.execute("SELECT id, session_key, title FROM sessions")
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
        finally:
            cur.close()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Helpers — direct path resolution
# ---------------------------------------------------------------------------


class TestSessionDbPathResolution:
    """``_session_db_path`` and ``_default_profile_state_db_path`` route correctly."""

    def test_default_session_returns_launch_home_db(self, two_profile_homes):
        server, launch_home, _other = two_profile_homes

        path = server._session_db_path(_make_session())

        assert path == launch_home / "state.db"

    def test_named_profile_session_returns_profile_db(self, two_profile_homes):
        server, _launch, other = two_profile_homes

        path = server._session_db_path(_make_session(profile_home=str(other)))

        assert path == other / "state.db"

    def test_default_state_db_path_tracks_hermes_home(self, two_profile_homes):
        server, launch_home, _other = two_profile_homes

        assert server._default_profile_state_db_path() == launch_home / "state.db"

    def test_default_state_db_path_is_cwd_safe(self, two_profile_homes, monkeypatch):
        """Even after a stray os.chdir, the resolved path is the launch home."""
        server, launch_home, other = two_profile_homes

        monkeypatch.chdir(str(other))
        resolved = Path(server._default_profile_state_db_path()).resolve()
        launch_resolved = (launch_home / "state.db").resolve()
        assert resolved == launch_resolved


# ---------------------------------------------------------------------------
# Helpers — _ensure_session_db_row must NOT follow a stale cached DB
# ---------------------------------------------------------------------------


class TestEnsureSessionDbRowIsolation:
    """``_ensure_session_db_row`` must use the launch-profile DB, not the cached one."""

    def test_default_session_writes_to_launch_home_not_stale_db(
        self, two_profile_homes
    ):
        """The repro from the issue: stale ``_get_db`` handle points at another profile.

        We simulate this by patching ``server._db`` (the cached handle) to
        point at ``other_home/state.db``. With the bug, the default session
        row lands in the other profile's db. With the fix, it lands in
        the launch profile's db.
        """
        server, launch_home, other_home = two_profile_homes

        # Seed a row in the OTHER profile's db to mimic "the cache was
        # initialised against this profile".
        _seed_row(other_home / "state.db", "stale-key")

        # Now swap the cached handle so _get_db() returns the "other" db.
        stale_handle = SessionDB(db_path=other_home / "state.db")
        try:
            with patch.object(server, "_db", stale_handle):
                session = _make_session(key="desktop-default-key")
                server._ensure_session_db_row(session)
        finally:
            stale_handle.close()

        launch_keys = {row["id"] for row in _rows(launch_home / "state.db")}
        other_keys = {row["id"] for row in _rows(other_home / "state.db")}

        assert "desktop-default-key" in launch_keys, (
            f"row landed in wrong db. launch={launch_keys}, other={other_keys}"
        )
        assert "desktop-default-key" not in other_keys, (
            "row leaked into the stale (other) profile db"
        )

    def test_explicit_profile_session_writes_to_that_profile(
        self, two_profile_homes
    ):
        """A session created under profile X must persist to X's state.db."""
        server, launch_home, other_home = two_profile_homes

        session = _make_session(
            profile_home=str(other_home), key="other-profile-key"
        )
        server._ensure_session_db_row(session)

        other_keys = {row["id"] for row in _rows(other_home / "state.db")}
        launch_keys = {row["id"] for row in _rows(launch_home / "state.db")}

        assert "other-profile-key" in other_keys
        assert "other-profile-key" not in launch_keys


# ---------------------------------------------------------------------------
# Helpers — _session_db context manager
# ---------------------------------------------------------------------------


class TestSessionDbContextManager:
    """``_session_db`` yields the launch-profile db, not the cached one."""

    def test_default_session_yields_launch_db(self, two_profile_homes):
        server, launch_home, other_home = two_profile_homes

        stale_handle = SessionDB(db_path=other_home / "state.db")
        try:
            with patch.object(server, "_db", stale_handle):
                session = _make_session()
                with server._session_db(session) as db:
                    assert db is not None
                    resolved = Path(db.db_path).resolve()
                    assert resolved == (launch_home / "state.db").resolve()
        finally:
            stale_handle.close()

    def test_named_profile_session_yields_profile_db(self, two_profile_homes):
        server, _launch, other_home = two_profile_homes

        session = _make_session(profile_home=str(other_home))
        with server._session_db(session) as db:
            assert db is not None
            resolved = Path(db.db_path).resolve()
            assert resolved == (other_home / "state.db").resolve()


# ---------------------------------------------------------------------------
# Migration seam
# ---------------------------------------------------------------------------


class TestMigrationLeavesLegacyRowsAlone:
    """Pre-existing rows in a stale profile db must NOT be re-touched for default sessions.

    The ``INSERT OR IGNORE`` semantics of ``create_session`` mean that if
    a row already exists in the stale (other) profile's db, the fix
    shouldn't try to "rewrite history" — it just needs to ensure that
    *new* default-session writes go to the launch profile's db.
    """

    def test_default_session_creates_row_only_in_launch_db(
        self, two_profile_homes
    ):
        server, launch_home, other_home = two_profile_homes

        # Simulate the buggy pre-fix state: a row "ghost-key" already
        # lives in the OTHER profile's db (it leaked there under the
        # cached-handle bug).
        _seed_row(other_home / "state.db", "ghost-key")

        # After the fix, a default-profile rebuild with a NEW key must
        # write to the launch db, not the stale one.
        session = _make_session(key="fresh-default-key")
        server._ensure_session_db_row(session)

        launch_keys = {row["id"] for row in _rows(launch_home / "state.db")}
        other_keys = {row["id"] for row in _rows(other_home / "state.db")}

        assert "fresh-default-key" in launch_keys
        assert "fresh-default-key" not in other_keys
        # The pre-existing ghost row stays in the other db — that's fine,
        # the migration is "new writes go to the right place", not
        # "rewrite history".
        assert "ghost-key" in other_keys