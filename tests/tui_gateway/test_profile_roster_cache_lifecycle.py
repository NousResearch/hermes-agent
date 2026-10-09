"""Roster memos must not turn transient admission failures into durable state."""

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


def test_readonly_roster_remains_available_during_mutation_lease(profile, busy_profile_lease):
    end_hold = busy_profile_lease(profile)
    during = {}
    srv._profile_session_fields(during, profile)
    # Read-only admission uses an incarnation/directory snapshot, so a
    # writer's lease alone must not hide an otherwise live profile.
    assert during["last_session"]["title"] == "steady chat"
    assert end_hold()
    after = {}
    srv._profile_session_fields(after, profile)
    assert after["last_session"]["title"] == "steady chat"


def test_warm_hit_rechecks_retirement_before_publication(profile, monkeypatch):
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
        profile_lifecycle.clear_profile_deletion_marker(profile)
    monkeypatch.setattr(cache, "cached_session_fields", real)
    recovered = {}
    srv._profile_session_fields(recovered, profile)
    assert recovered["last_session"]["title"] == "steady chat"


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
