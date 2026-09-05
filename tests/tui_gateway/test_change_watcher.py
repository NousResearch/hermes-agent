"""The generalized change watcher (#73618): cheap on-disk signatures →
``pet.changed`` / ``cron.changed`` / ``sessions.changed`` global broadcasts.

Behavior contracts, exercised against a real temp HERMES_HOME (no mocks on the
filesystem path): first sighting seeds silently, a moved signature broadcasts
once, the sessions floor coalesces a write burst but keeps its trailing edge,
and the pet signature only moves for a *renderable* pet.
"""

import os
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
    monkeypatch.setattr(server, "_bot_relay_outbox_seen", 0)

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


def test_served_profile_store_move_broadcasts_sessions_changed(watcher_home, monkeypatch):
    """A backend serving a sibling profile must see that profile's state.db
    move too — otherwise a routed profile's Bot Chat never refreshes (#99333)."""
    home, events = watcher_home
    bot_home = home / "profiles" / "bot"
    bot_home.mkdir(parents=True)
    monkeypatch.setattr(server, "_served_profile_homes", set())
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: home / "profiles" / name)
    assert server._profile_home("bot") == bot_home
    server._broadcast_watched_changes(now=0.0)

    (bot_home / "state.db").write_text("x")
    server._broadcast_watched_changes(now=10.0)

    assert ("sessions.changed", {}) in events


def test_gateway_state_move_broadcasts_platforms_changed(watcher_home):
    home, events = watcher_home
    server._broadcast_watched_changes(now=0.0)

    (home / "gateway_state.json").write_text('{"platforms": {}}')
    server._broadcast_watched_changes(now=10.0)

    assert ("platforms.changed", {}) in events


def test_pending_pairing_request_broadcasts_pairing_changed(watcher_home):
    """A new pending request must reach the Messaging page on its own signal.

    The messaging gateway writes the pending code from a different process, and
    it moves nothing in gateway_state.json — so platforms.changed cannot stand
    in for this. Without a dedicated signal the badge stays invisible until an
    unrelated connect/disconnect happens to fire.
    """
    home, events = watcher_home
    store = home / "platforms" / "pairing"
    store.mkdir(parents=True)
    server._broadcast_watched_changes(now=0.0)

    (store / "telegram-pending.json").write_text('{"abc": {"user_id": "1"}}')
    server._broadcast_watched_changes(now=10.0)

    assert ("pairing.changed", {}) in events
    assert ("platforms.changed", {}) not in events


def test_pairing_signal_follows_a_profile_store(watcher_home):
    """Each profile keeps its own whitelist, and the page can be scoped to any."""
    home, events = watcher_home
    store = home / "profiles" / "work" / "platforms" / "pairing"
    store.mkdir(parents=True)
    server._broadcast_watched_changes(now=0.0)

    (store / "telegram-approved.json").write_text('{"u1": {"user_id": "u1"}}')
    server._broadcast_watched_changes(now=10.0)

    assert ("pairing.changed", {}) in events


def test_rate_limit_churn_does_not_broadcast_pairing_changed(watcher_home):
    """_rate_limits.json moves on every unauthorized DM, including ones that
    produce no new row — signalling on it would refetch for nothing."""
    home, events = watcher_home
    store = home / "platforms" / "pairing"
    store.mkdir(parents=True)
    (store / "telegram-pending.json").write_text("{}")
    server._broadcast_watched_changes(now=0.0)

    (store / "_rate_limits.json").write_text('{"telegram:1": 123}')
    server._broadcast_watched_changes(now=10.0)

    assert ("pairing.changed", {}) not in events


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


def test_enqueued_envelope_broadcasts_outbox_pending(watcher_home):
    """A cross-connection envelope written by the agent process must reach the
    Desktop's push-triggered drain on its own signal (#93091) — the drain poll
    is the backstop, not the transport."""
    home, events = watcher_home
    outbox = home / "bot_relay" / "outbox"
    outbox.mkdir(parents=True)
    server._broadcast_watched_changes(now=0.0)

    (outbox / ("a" * 32 + ".json")).write_text('{"id": "' + "a" * 32 + '"}')
    server._broadcast_watched_changes(now=10.0)

    assert ("bot_relay.outbox.pending", {}) in events


def test_drained_outbox_does_not_rebroadcast_pending(watcher_home):
    """Signature is monotone: a drain empties outbox/ (rename → claimed/), and
    that emptying must NOT look like a change — only new envelopes fire."""
    home, events = watcher_home
    outbox = home / "bot_relay" / "outbox"
    outbox.mkdir(parents=True)
    envelope = outbox / ("b" * 32 + ".json")
    envelope.write_text("{}")
    server._broadcast_watched_changes(now=0.0)

    envelope.unlink()  # the Desktop drained it
    server._broadcast_watched_changes(now=10.0)
    server._broadcast_watched_changes(now=20.0)

    assert not [e for e in events if e[0] == "bot_relay.outbox.pending"]


def test_new_envelope_after_drain_fires_pending_again(watcher_home):
    """The other half of the monotone contract: the watermark must not eat
    GENUINELY new envelopes. write → drain → write-newer fires twice."""
    home, events = watcher_home
    outbox = home / "bot_relay" / "outbox"
    outbox.mkdir(parents=True)
    first = outbox / ("c" * 32 + ".json")
    first.write_text("{}")
    server._broadcast_watched_changes(now=0.0)
    first.write_text("{}")  # make the first sighting a change, not a seed
    bump_ns = first.stat().st_mtime_ns + 1_000_000
    os.utime(first, ns=(bump_ns, bump_ns))  # strictly newer, FS-independent
    server._broadcast_watched_changes(now=10.0)

    first.unlink()  # the Desktop drained it
    server._broadcast_watched_changes(now=20.0)

    second = outbox / ("d" * 32 + ".json")
    second.write_text("{}")
    newer_ns = bump_ns + 1_000_000  # strictly beyond the watermark
    os.utime(second, ns=(newer_ns, newer_ns))
    server._broadcast_watched_changes(now=30.0)

    assert [e for e in events if e[0] == "bot_relay.outbox.pending"] == [
        ("bot_relay.outbox.pending", {}),
        ("bot_relay.outbox.pending", {}),
    ]


def test_no_outbox_dir_never_fires_pending(watcher_home):
    home, events = watcher_home
    server._broadcast_watched_changes(now=0.0)
    server._broadcast_watched_changes(now=10.0)

    assert not [e for e in events if e[0] == "bot_relay.outbox.pending"]


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


def test_desktop_room_command_signal_broadcasts_pending(watcher_home):
    home, events = watcher_home
    signal = home / "desktop_room_mailbox.pending"
    server._broadcast_watched_changes(now=0.0)

    signal.write_text("1")
    server._broadcast_watched_changes(now=10.0)

    assert ("desktop_rooms.commands.pending", {}) in events
