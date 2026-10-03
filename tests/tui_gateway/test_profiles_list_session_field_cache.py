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


def test_cached_session_fields_never_outlive_profile_retirement_or_recreation(home):
    import os
    import shutil

    from hermes_cli import profile_lifecycle
    from hermes_cli.profile_incarnation import ensure_profile_incarnation, write_fresh_profile_incarnation
    from hermes_constants import clear_named_profile_deleted, mark_named_profile_deleted

    bob = home / "profiles" / "bob"
    _seed(bob, "20260920_000001_a", "old chat")
    ensure_profile_incarnation(bob)
    original = {}
    srv._profile_session_fields(original, bob)
    assert original["last_session"]["title"] == "old chat"
    mark_named_profile_deleted(bob)
    try:
        retired = {}
        srv._profile_session_fields(retired, bob)
        assert retired == dict(last_session=None, worker_session=None, canonical_session=None)
    finally:
        clear_named_profile_deleted(bob)

    # Same pathname, size and timestamps must not preserve the old generation's
    # memo. Keep a backup alive so an allocator cannot reuse its file identities.
    srv._profile_session_fields({}, bob)
    old = home / "old-bob"
    shutil.copytree(bob, old)
    signature = cache.store_signature(bob)
    with profile_lifecycle.profile_lifecycle_lease(bob):
        write_fresh_profile_incarnation(bob)
    for path in old.iterdir():
        target = bob / path.name
        if target.exists():
            stat = path.stat()
            os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert cache.store_signature(bob) != signature
