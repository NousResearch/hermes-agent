"""``sessions.show_oneshot`` reaches the gateway's ``session.list`` (#160464).

This listing filters AFTER the SQL read, so the opt-in cannot travel as a query field the way
``sessions.show_subagents`` does: the run is already denied by ``_LISTING_DENY_SOURCES``. The
opt-in therefore arrives as a per-listing deny-list exemption (``admit_sources``), read from the
store's own profile config. Off by default, one-shot runs stay hidden — the pre-existing contract.
"""

import pytest

import tui_gateway.server as srv
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_session


def _make_db(home, monkeypatch):
    database = open_persistence(profile_scope(home=home))
    monkeypatch.setattr(srv, "_get_db", lambda: database)
    return database


@pytest.fixture
def db(tmp_path, monkeypatch):
    database = _make_db(tmp_path, monkeypatch)
    try:
        yield database
    finally:
        database.close()


def _call(method: str, params: dict) -> dict:
    return srv._methods[method](1, params)


def _seed(store, sid: str, source: str) -> None:
    seed_session(store, sid, source)
    raw_holder(store)._execute_write(
        lambda conn: conn.execute("UPDATE sessions SET message_count = 1 WHERE id = ?", (sid,)))


def _ids(params: dict | None = None) -> set[str]:
    envelope = _call("session.list", params or {})
    assert "error" not in envelope, envelope
    return {row["id"] for row in envelope["result"]["sessions"]}


def test_the_opt_in_admits_the_oneshot_run_into_session_list(db):
    """With ``sessions.show_oneshot`` on the store's home, a finite run survives the deny-list."""
    _seed(db, "plain-chat", "desktop")
    _seed(db, "oneshot-run", "oneshot")
    assert _ids() == {"plain-chat"}

    (db.scope.profile_dir / "config.yaml").write_text("sessions:\n  show_oneshot: true\n")

    assert _ids() == {"plain-chat", "oneshot-run"}


def test_the_opt_in_admits_only_the_oneshot_source(db):
    """Admission is per source: kanban/tool rows keep their exclusion even with the flag on."""
    _seed(db, "plain-chat", "desktop")
    _seed(db, "oneshot-run", "oneshot")
    _seed(db, "kanban-run", "kanban")
    (db.scope.profile_dir / "config.yaml").write_text("sessions:\n  show_oneshot: true\n")

    assert _ids() == {"plain-chat", "oneshot-run"}


def test_without_the_flag_the_deny_list_is_unchanged(db):
    """Default (no config, or the flag explicitly off): the pre-#160464 behavior."""
    _seed(db, "plain-chat", "desktop")
    _seed(db, "oneshot-run", "oneshot")

    (db.scope.profile_dir / "config.yaml").write_text("sessions:\n  show_oneshot: false\n")
    assert _ids() == {"plain-chat"}


def test_the_flag_is_read_from_the_stores_home_not_the_opt_in_of_another(tmp_path, monkeypatch):
    """One gateway serves many profiles: the store's home decides, never the launching profile."""
    opted_in = tmp_path / "opted-in"
    opted_out = tmp_path / "opted-out"
    for home in (opted_in, opted_out):
        home.mkdir(parents=True, exist_ok=True)
    (opted_in / "config.yaml").write_text("sessions:\n  show_oneshot: true\n")
    (opted_out / "config.yaml").write_text("sessions:\n  show_oneshot: false\n")

    for home in (opted_in, opted_out):
        database = _make_db(home, monkeypatch)
        try:
            _seed(database, "oneshot-run", "oneshot")
        finally:
            database.close()

    database = _make_db(opted_in, monkeypatch)
    try:
        assert _ids() == {"oneshot-run"}
    finally:
        database.close()

    database = _make_db(opted_out, monkeypatch)
    try:
        assert _ids() == set()
    finally:
        database.close()
