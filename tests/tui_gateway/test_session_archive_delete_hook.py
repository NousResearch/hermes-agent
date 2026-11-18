"""``session.archive`` fires ``on_session_delete`` (``reason="archive"``) for the rows it newly hides.

Both tiers are covered: a stored id archived through the profile store and a live runtime session whose row
exists. Unarchiving, or archiving an already archived session, hides nothing and fires nothing.
"""

import pytest

import tui_gateway.server as srv
import tui_gateway.methods_session
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_seed import seed_session


@pytest.fixture
def db(tmp_path, monkeypatch):
    database = open_persistence(profile_scope(home=tmp_path))
    monkeypatch.setattr(srv, "_get_db", lambda: database)
    try:
        yield database
    finally:
        database.close()


@pytest.fixture
def hooks(monkeypatch):
    calls = []
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda name, **kw: calls.append((name, kw)) or [])
    return calls


def _archive(target: str, archived: bool) -> dict:
    envelope = srv._methods["session.archive"](1, {"session_id": target, "archived": archived})
    assert "error" not in envelope, envelope
    return envelope


def _delete_hooks(calls) -> list[tuple[str, str]]:
    return [(kw["session_id"], kw["reason"]) for name, kw in calls if name == "on_session_delete"]


def test_archiving_a_stored_session_fires_the_archive_hook(db, hooks):
    seed_session(db, "stored-chat", "desktop")

    _archive("stored-chat", True)

    assert _delete_hooks(hooks) == [("stored-chat", "archive")]


def test_archiving_a_live_session_with_a_row_fires_the_archive_hook(db, hooks):
    seed_session(db, "live-key", "desktop")
    srv._sessions["live-sid"] = {"session_key": "live-key", "history": [], "running": False}
    try:
        _archive("live-sid", True)
    finally:
        srv._sessions.pop("live-sid", None)

    assert _delete_hooks(hooks) == [("live-key", "archive")]


def test_unarchive_and_repeat_archive_fire_nothing(db, hooks):
    seed_session(db, "stored-chat", "desktop")
    _archive("stored-chat", True)
    hooks.clear()

    _archive("stored-chat", True)
    _archive("stored-chat", False)

    assert _delete_hooks(hooks) == []
