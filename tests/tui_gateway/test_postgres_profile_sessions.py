"""The desktop profile roster reads PostgreSQL conversation metadata."""

from tui_gateway import server


def test_profile_roster_reads_remote_session_without_local_state_file(postgres_db, postgres_home):
    postgres_db.create_session("postgres-roster", source="cli")
    postgres_db.append_message("postgres-roster", "user", "Remote profile conversation")
    postgres_db.append_message("postgres-roster", "assistant", "The latest remote response")
    postgres_db.set_session_title("postgres-roster", "Bot Chat")
    row = {}

    server._profile_session_fields(row, postgres_home)

    assert row["last_session"]["id"] == "postgres-roster"
    assert row["last_session"]["title"] == "Bot Chat"
    assert row["last_session"]["preview"] == "The latest remote response"
    assert row["canonical_session"]["id"] == "postgres-roster"
    assert row["canonical_session"]["preview"] == "The latest remote response"
    assert row["worker_session"] is None
    assert not (postgres_home / "state.db").exists()


def test_remote_message_write_broadcasts_sessions_changed(postgres_db, postgres_home, monkeypatch):
    monkeypatch.setattr(server, "_hermes_home", str(postgres_home))
    monkeypatch.setattr(server, "_served_profile_homes", set())
    monkeypatch.setattr(server, "_change_sigs", {})
    monkeypatch.setattr(server, "_change_checked_at", {})
    monkeypatch.setattr(server, "_change_broadcast_at", {})
    events = []
    monkeypatch.setattr(server, "_broadcast_global_event", lambda event, payload: events.append(event))
    server._broadcast_watched_changes(now=0.0)

    postgres_db.create_session("remote-write", source="cli")
    postgres_db.append_message("remote-write", "user", "Written outside the TUI")
    server._broadcast_watched_changes(now=10.0)

    assert "sessions.changed" in events
    assert not (postgres_home / "state.db").exists()
