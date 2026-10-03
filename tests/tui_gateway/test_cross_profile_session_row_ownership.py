"""Cross-profile session ids must never be materialized in the launch store."""

from __future__ import annotations

import importlib
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB

SESSION_KEY = "20260101_000000_deadbe"


@pytest.fixture()
def server():
    with patch.dict(
        "sys.modules",
        {
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
        },
    ):
        mod = importlib.import_module("tui_gateway.server")
    yield mod
    mod._sessions.clear()
    mod._served_profile_homes.clear()
    mod._db = None


@pytest.fixture()
def stores(tmp_path, monkeypatch, server):
    launch_home = tmp_path / "launch"
    profile_home = tmp_path / "profiles" / "bot"
    launch_home.mkdir(parents=True)
    profile_home.mkdir(parents=True)

    launch_db = SessionDB(db_path=launch_home / "state.db")
    profile_db = SessionDB(db_path=profile_home / "state.db")
    profile_db.create_session(SESSION_KEY, source="desktop")

    monkeypatch.setattr(server, "_get_db", lambda: launch_db)
    monkeypatch.setattr(server, "_db_error", None)
    monkeypatch.setattr(server, "_served_profile_homes", {profile_home})
    monkeypatch.setattr(server, "_current_profile_name", lambda: "default")

    yield launch_db, profile_home, profile_db

    profile_db.close()
    launch_db.close()


def test_launch_write_is_refused_when_a_served_profile_owns_the_id(server, stores):
    launch_db, _, profile_db = stores

    with pytest.raises(server.SessionProfileOwnershipError):
        server._ensure_session_db_row({"session_key": SESSION_KEY, "source": "desktop"})

    assert launch_db.get_session(SESSION_KEY) is None
    assert profile_db.get_session(SESSION_KEY) is not None


def test_launch_write_still_lands_for_a_genuinely_local_session(server, stores):
    launch_db, _, _ = stores

    ok = server._ensure_session_db_row({"session_key": "local-session-1", "source": "desktop"})

    assert ok is True
    assert launch_db.get_session("local-session-1") is not None


def test_profile_scoped_session_never_writes_or_probes_the_launch_store(server, stores, tmp_path, monkeypatch):
    launch_db, profile_home, profile_db = stores
    unreadable_home = tmp_path / "profiles" / "unreadable"
    unreadable_home.mkdir(parents=True)
    (unreadable_home / "state.db").write_bytes(b"not sqlite")
    monkeypatch.setattr(server, "_served_profile_homes", {profile_home, unreadable_home})

    ok = server._ensure_session_db_row(
        {"session_key": SESSION_KEY, "profile_home": str(profile_home), "source": "desktop"}
    )

    assert ok is True
    assert profile_db.get_session(SESSION_KEY) is not None
    assert launch_db.get_session(SESSION_KEY) is None


def test_single_profile_install_keeps_the_launch_store_path(server, stores, monkeypatch):
    launch_db, _, _ = stores
    monkeypatch.setattr(server, "_served_profile_homes", set())
    import hermes_state
    monkeypatch.setattr(hermes_state, "SessionDB", MagicMock(side_effect=AssertionError("sibling probe")))

    ok = server._ensure_session_db_row({"session_key": SESSION_KEY, "source": "desktop"})

    assert ok is True
    assert launch_db.get_session(SESSION_KEY) is not None


def test_unstatable_served_store_fails_closed(server, stores, tmp_path, monkeypatch):
    launch_db, _, _ = stores
    blocked_home = tmp_path / "profiles" / "blocked"
    blocked_home.mkdir(parents=True)
    blocked_db = blocked_home / "state.db"
    blocked_db.write_bytes(b"sqlite placeholder")
    monkeypatch.setattr(server, "_served_profile_homes", {blocked_home})

    original_stat = Path.stat

    def guarded_stat(path, *args, **kwargs):
        if path == blocked_db:
            raise PermissionError("profile store metadata denied")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", guarded_stat)

    with pytest.raises(server.SessionProfileOwnershipError) as raised:
        server._ensure_session_db_row({"session_key": "unverified-session", "source": "desktop"})

    assert raised.value.probe_failed is True
    assert launch_db.get_session("unverified-session") is None


def test_unreadable_served_store_fails_closed(server, stores, tmp_path, monkeypatch):
    launch_db, profile_home, _ = stores
    unreadable_home = tmp_path / "profiles" / "unreadable"
    unreadable_home.mkdir(parents=True)
    (unreadable_home / "state.db").write_bytes(b"not sqlite")
    monkeypatch.setattr(server, "_served_profile_homes", {profile_home, unreadable_home})

    with pytest.raises(server.SessionProfileOwnershipError) as raised:
        server._ensure_session_db_row({"session_key": "unverified-session", "source": "desktop"})

    assert raised.value.probe_failed is True
    assert launch_db.get_session("unverified-session") is None


def test_profile_scope_alternation_keeps_each_store_isolated(server, stores):
    launch_db, profile_home, profile_db = stores

    assert server._ensure_session_db_row({"session_key": "launch-a", "source": "desktop"}) is True
    assert server._ensure_session_db_row(
        {"session_key": "profile-b", "profile_home": str(profile_home), "source": "desktop"}
    ) is True
    assert server._ensure_session_db_row({"session_key": "launch-c", "source": "desktop"}) is True

    assert launch_db.get_session("launch-a") is not None
    assert launch_db.get_session("launch-c") is not None
    assert launch_db.get_session("profile-b") is None
    assert profile_db.get_session("profile-b") is not None
    assert profile_db.get_session("launch-a") is None
    assert profile_db.get_session("launch-c") is None


def test_foreign_owner_blocks_synthesized_turn_before_admission_and_releases_claim(
    server, stores, monkeypatch
):
    launch_db, _, _ = stores
    admitted = MagicMock(return_value=None)
    monkeypatch.setattr(server, "_admit_prompt_turn", admitted)

    class Lease:
        track_liveness = False
        enabled = True
        released = False

        def release(self):
            self.released = True

    lease = Lease()
    session = {
        "session_key": SESSION_KEY,
        "source": "desktop",
        "running": True,
        "history_lock": threading.RLock(),
        "inflight_turn": {"user": "private prompt", "status": "streaming"},
        "_submit_user_row": 42,
        "_hosted_room_task": {"id": "task-1"},
        "_auto_continue_scheduled": True,
        "_auto_continue_attempt": 2,
        "_auto_continue_prompt": "private prompt",
        "active_session_lease": lease,
    }

    started = server._run_prompt_submit(
        "request-1", "ui-session", session, "private prompt"
    )

    assert started is False
    admitted.assert_not_called()
    assert session["running"] is False
    assert session["inflight_turn"] is None
    assert "_submit_user_row" not in session
    assert "_hosted_room_task" not in session
    assert "_auto_continue_scheduled" not in session
    assert "_auto_continue_attempt" not in session
    assert "_auto_continue_prompt" not in session
    assert "active_session_lease" not in session
    assert lease.released is True
    assert session["last_active"] > 0
    assert launch_db.get_session(SESSION_KEY) is None


def test_foreign_owner_rejects_busy_queue_without_retaining_prompt(server, stores, monkeypatch):
    launch_db, _, _ = stores
    session = {
        "session_key": SESSION_KEY,
        "source": "desktop",
        "running": True,
        "history_lock": threading.RLock(),
        "attached_images": [],
        "agent": None,
    }
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "queue")
    monkeypatch.setattr(server, "_session_compression_in_flight", lambda _session: False)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _session: False)

    response = server._handle_busy_submit(
        "request-2", "ui-session", session, "private queued prompt", transport=None
    )

    assert response.get("error")
    assert not session.get("queued_prompt")
    assert launch_db.get_session(SESSION_KEY) is None
