"""``profiles.list`` reuses a profile's session fields only while its session store has not moved.

The Bots roster polls this every 5s per connection and the session fields are a pure function of
the profile's ``state.db``, so an idle fleet re-opened every bot's store every five seconds. The
memo must be invisible: a write anywhere in the store has to show up on the very next poll, or the
roster silently paints a stale preview.
"""
from __future__ import annotations

import pytest

import tui_gateway.server as srv
from hermes_state import SessionDB
from tui_gateway import profile_roster_cache as cache


@pytest.fixture(autouse=True)
def _clean_memo():
    cache.invalidate()
    yield
    cache.invalidate()


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "profiles" / "bob").mkdir(parents=True)
    return tmp_path


def _seed(profile_dir, session_id: str, title: str, text: str = "hello") -> None:
    db = SessionDB(db_path=profile_dir / "state.db")
    try:
        db.create_session(session_id=session_id, source="cli")
        db.set_session_title(session_id, title)
        db.append_message(session_id=session_id, role="user", content=text)
    finally:
        db.close()


def _rows(params=None):
    return srv._methods["profiles.list"](1, params or {})["result"]["profiles"]


def _row(name):
    return next(p for p in _rows() if p["name"] == name)


def test_a_new_session_shows_up_on_the_very_next_poll(home):
    bob = home / "profiles" / "bob"
    _seed(bob, "20260920_000001_a", "first chat")
    assert _row("bob")["last_session"]["title"] == "first chat"

    _seed(bob, "20260920_000002_b", "second chat")

    # The memo keys on the store's signature, so this must not be the cached answer.
    assert _row("bob")["last_session"]["title"] == "second chat"


def test_a_message_appended_to_an_existing_session_is_not_masked(home):
    bob = home / "profiles" / "bob"
    _seed(bob, "20260920_000001_a", "Bot Chat", text="first")
    before = _row("bob")["canonical_session"]

    db = SessionDB(db_path=bob / "state.db")
    try:
        db.append_message(session_id="20260920_000001_a", role="assistant", content="a later reply")
    finally:
        db.close()

    after = _row("bob")["canonical_session"]
    assert after != before, "an append inside the store was served from the memo"


def test_a_profile_without_a_store_is_not_memoised_and_picks_one_up(home):
    bare = home / "profiles" / "bare"
    bare.mkdir()
    (bare / "config.yaml").write_text("model:\n  provider: openai\n", encoding="utf-8")
    assert _row("bare")["last_session"] is None
    assert cache.store_signature(bare) is None  # nothing to key on, so nothing is cached

    _seed(bare, "20260920_000003_c", "now it exists")

    assert _row("bare")["last_session"]["title"] == "now it exists"


def test_one_profile_s_rows_never_answer_for_another(home):
    bob, ann = home / "profiles" / "bob", home / "profiles" / "ann"
    ann.mkdir()
    _seed(bob, "20260920_000001_a", "bob's chat")
    _seed(ann, "20260920_000002_b", "ann's chat")

    rows = {p["name"]: p for p in _rows()}

    assert rows["bob"]["last_session"]["title"] == "bob's chat"
    assert rows["ann"]["last_session"]["title"] == "ann's chat"


def test_an_unchanged_store_is_read_once_across_repeated_polls(home, monkeypatch):
    """The saving itself: the 5s poll stops reopening a store that has not moved."""
    bob = home / "profiles" / "bob"
    _seed(bob, "20260920_000001_a", "steady")

    opens: list = []
    real = SessionDB

    def _counting(*args, **kwargs):
        opens.append(kwargs.get("db_path"))
        return real(*args, **kwargs)

    monkeypatch.setattr("hermes_state.SessionDB", _counting)

    _rows()
    after_first = len(opens)
    _rows()
    _rows()

    assert after_first >= 1, "the first poll must actually read the store"
    assert len(opens) == after_first, "an unchanged store was reopened by a later poll"
