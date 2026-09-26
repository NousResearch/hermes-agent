"""Roster memos must not turn transient admission failures into durable state."""
import threading

import pytest

from hermes_cli import profile_lifecycle
from hermes_state import SessionDB
import tui_gateway.server as srv
from tui_gateway import profile_roster_cache as cache


@pytest.fixture
def profile(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    bob = tmp_path / "profiles" / "bob"
    bob.mkdir(parents=True)
    db = SessionDB(db_path=bob / "state.db")
    try:
        db.create_session(session_id="20260920_000001_a", source="cli")
        db.set_session_title("20260920_000001_a", "steady chat")
    finally:
        db.close()
    cache.invalidate()
    yield bob
    cache.invalidate()


def test_roster_retries_after_real_profile_lease_timeout(profile, monkeypatch):
    acquired, release = threading.Event(), threading.Event()

    def hold():
        with profile_lifecycle.profile_lifecycle_lease(profile):
            acquired.set()
            assert release.wait(10)

    thread = threading.Thread(target=hold)
    thread.start()
    monkeypatch.setattr(profile_lifecycle, "_PROFILE_LIFECYCLE_LOCK_TIMEOUT_SECONDS", 0.05)
    try:
        assert acquired.wait(5)
        during = {}
        srv._profile_session_fields(during, profile)
        assert during == dict(last_session=None, worker_session=None, canonical_session=None)
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    after = {}
    srv._profile_session_fields(after, profile)
    assert after["last_session"]["title"] == "steady chat"


def test_warm_hit_rechecks_retirement_before_publication(profile, monkeypatch):
    import hermes_constants

    warm = {}
    srv._profile_session_fields(warm, profile)
    assert warm["last_session"]["title"] == "steady chat"
    real = cache.cached_session_fields

    def retire_after_lookup(path, compute):
        fields = real(path, compute)
        with profile_lifecycle.profile_lifecycle_lease(profile):
            profile_lifecycle.mark_profile_deleting(profile)
        return fields

    monkeypatch.setattr(cache, "cached_session_fields", retire_after_lookup)
    row = {}
    try:
        srv._profile_session_fields(row, profile)
        assert row == dict(last_session=None, worker_session=None, canonical_session=None)
    finally:
        hermes_constants.clear_named_profile_deleted(profile)


def test_warm_hit_rechecks_replacement_before_publication(profile, monkeypatch):
    from hermes_cli.profile_incarnation import write_fresh_profile_incarnation

    srv._profile_session_fields({}, profile)
    real = cache.cached_session_fields

    def replace_after_lookup(path, compute):
        fields = real(path, compute)
        with profile_lifecycle.profile_lifecycle_lease(profile):
            write_fresh_profile_incarnation(profile)
        return fields

    monkeypatch.setattr(cache, "cached_session_fields", replace_after_lookup)
    row = {}
    srv._profile_session_fields(row, profile)
    assert row == dict(last_session=None, worker_session=None, canonical_session=None)
