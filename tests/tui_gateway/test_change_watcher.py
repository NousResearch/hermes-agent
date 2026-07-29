"""The generalized change watcher (#73618): cheap on-disk signatures →
``pet.changed`` / ``cron.changed`` / ``sessions.changed`` global broadcasts.

Behavior contracts, exercised against a real temp HERMES_HOME (no mocks on the
filesystem path): first sighting seeds silently, a moved signature broadcasts
once, the sessions floor coalesces a write burst but keeps its trailing edge,
and the pet signature only moves for a *renderable* pet.
"""

import time

import pytest

from tui_gateway import server


@pytest.fixture()
def watcher_home(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text("display: {}\n")
    (tmp_path / "cron").mkdir()

    monkeypatch.setattr(server, "_hermes_home", str(tmp_path))
    monkeypatch.setattr(server, "_cfg_cache", None)
    monkeypatch.setattr(server, "_change_sigs", {})
    monkeypatch.setattr(server, "_change_checked_at", {})
    monkeypatch.setattr(server, "_change_broadcast_at", {})

    events = []
    monkeypatch.setattr(
        server, "_broadcast_global_event", lambda ev, payload=None: events.append((ev, payload))
    )
    return tmp_path, events


def test_first_sighting_seeds_without_broadcasting(watcher_home):
    home, events = watcher_home
    (home / "cron" / "jobs.json").write_text("[]")
    (home / "state.db").write_text("x")

    server._broadcast_watched_changes(now=0.0)

    assert events == []


def test_cron_jobs_file_move_broadcasts_cron_changed(watcher_home):
    home, events = watcher_home
    server._broadcast_watched_changes(now=0.0)

    (home / "cron" / "jobs.json").write_text("[]")
    server._broadcast_watched_changes(now=10.0)

    assert ("cron.changed", {}) in events


def test_state_db_move_broadcasts_sessions_changed(watcher_home):
    home, events = watcher_home
    server._broadcast_watched_changes(now=0.0)

    (home / "state.db").write_text("x")
    server._broadcast_watched_changes(now=10.0)

    assert ("sessions.changed", {}) in events


def test_gateway_state_move_broadcasts_platforms_changed(watcher_home):
    home, events = watcher_home
    server._broadcast_watched_changes(now=0.0)

    (home / "gateway_state.json").write_text('{"platforms": {}}')
    server._broadcast_watched_changes(now=10.0)

    assert ("platforms.changed", {}) in events


def test_sessions_floor_coalesces_burst_but_keeps_trailing_edge(watcher_home):
    home, events = watcher_home
    server._broadcast_watched_changes(now=0.0)

    (home / "state.db").write_text("x")
    server._broadcast_watched_changes(now=10.0)
    events.clear()

    # A second write lands inside the 2s floor: no broadcast yet…
    time.sleep(0.02)
    (home / "state.db").write_text("xy")
    server._broadcast_watched_changes(now=11.0)
    assert events == []

    # …but the change is not lost — it fires once the window opens.
    server._broadcast_watched_changes(now=13.0)
    assert ("sessions.changed", {}) in events


def test_pet_sig_stays_off_without_a_renderable_pet(watcher_home):
    home, events = watcher_home
    server._broadcast_watched_changes(now=0.0)

    # Config flips enabled but no pet exists on disk → signature stays ("off",).
    (home / "config.yaml").write_text("display:\n  pet:\n    enabled: true\n    slug: boba\n")
    server._cfg_cache = None
    server._broadcast_watched_changes(now=10.0)

    assert not [e for e in events if e[0] == "pet.changed"]


def test_renderable_pet_broadcasts_meta_payload(watcher_home, monkeypatch):
    home, events = watcher_home
    (home / "config.yaml").write_text("display:\n  pet:\n    enabled: true\n    slug: boba\n")
    server._cfg_cache = None
    server._broadcast_watched_changes(now=0.0)

    sheet = home / "sheet.png"
    sheet.write_text("png")

    class FakePet:
        slug = "boba"
        display_name = "Boba"
        exists = True
        spritesheet = sheet

    monkeypatch.setattr(server, "_pet_active_selection", lambda: (True, FakePet(), 0.33))
    server._broadcast_watched_changes(now=10.0)

    pet_events = [e for e in events if e[0] == "pet.changed"]
    assert pet_events
    payload = pet_events[0][1]
    assert payload["enabled"] is True
    assert payload["slug"] == "boba"
    assert payload["spritesheetRevision"]


def test_broken_probe_never_kills_the_pass(watcher_home, monkeypatch):
    home, events = watcher_home
    server._broadcast_watched_changes(now=0.0)

    monkeypatch.setitem(
        server._CHANGE_WATCHES,
        "cron.changed",
        (1.0, lambda: (_ for _ in ()).throw(RuntimeError("boom")), lambda: {}),
    )
    (home / "state.db").write_text("x")
    server._broadcast_watched_changes(now=10.0)

    # The broken cron probe is skipped; sessions still broadcasts.
    assert ("sessions.changed", {}) in events
