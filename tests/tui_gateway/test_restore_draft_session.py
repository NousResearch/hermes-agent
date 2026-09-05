"""Regression coverage for restoring never-sent desktop tab drafts.

A fresh Desktop tab is intentionally lazy: ``session.create`` returns a stored
key but writes no database row until the first prompt. After a process restart
there is therefore nothing for ``session.resume`` to load, while Desktop still
has the tab and its composer text in localStorage. ``restore_session_id``
atomically reserves a zero-message row, then recreates the runtime without
changing the draft's local key.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

import tui_gateway.server as srv
from hermes_state import SessionDB


DRAFT_ID = "20260825_032826_012255"


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    monkeypatch.setattr(srv, "_get_db", lambda: db)
    monkeypatch.setattr(srv, "_completion_cwd", lambda params=None: str(tmp_path))
    monkeypatch.setattr(srv, "_schedule_agent_build", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        srv, "_schedule_session_cap_enforcement", lambda *args, **kwargs: None
    )
    srv._sessions.clear()

    try:
        yield db
    finally:
        srv._sessions.clear()
        db.close()


def test_session_create_restores_an_unpersisted_draft_under_its_original_key(gateway):
    out = srv._methods["session.create"](
        "r1",
        {
            "cols": 96,
            "restore_session_id": DRAFT_ID,
            "source": "desktop",
        },
    )

    assert "error" not in out, out
    result = out["result"]
    assert result["stored_session_id"] == DRAFT_ID
    assert srv._sessions[result["session_id"]]["session_key"] == DRAFT_ID
    reserved = gateway.get_session(DRAFT_ID)
    assert reserved is not None
    assert reserved["message_count"] == 0


def test_session_create_reserves_the_key_only_in_the_requested_profile(
    tmp_path, monkeypatch
):
    launch_db = SessionDB(tmp_path / "launch-state.db")
    profile_home = tmp_path / "profiles" / "ops"
    profile_home.mkdir(parents=True)

    monkeypatch.setattr(srv, "_get_db", lambda: launch_db)
    monkeypatch.setattr(
        srv, "_profile_home", lambda profile: profile_home if profile == "ops" else None
    )
    monkeypatch.setattr(srv, "_completion_cwd", lambda params=None: str(tmp_path))
    monkeypatch.setattr(srv, "_schedule_agent_build", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        srv, "_schedule_session_cap_enforcement", lambda *args, **kwargs: None
    )
    srv._sessions.clear()

    try:
        out = srv._methods["session.create"](
            "profile-restore",
            {
                "profile": "ops",
                "restore_session_id": DRAFT_ID,
                "source": "desktop",
            },
        )

        assert "error" not in out, out
        runtime_id = out["result"]["session_id"]
        assert srv._sessions[runtime_id]["profile_home"] == str(profile_home)
        assert launch_db.get_session(DRAFT_ID) is None

        profile_db = SessionDB(profile_home / "state.db")
        try:
            row = profile_db.get_session(DRAFT_ID)
            assert row is not None
            assert row["profile_name"] == "ops"
            assert row["message_count"] == 0
        finally:
            profile_db.close()
    finally:
        srv._sessions.clear()
        launch_db.close()


def test_restored_draft_reservation_derives_the_default_store_profile(
    tmp_path, monkeypatch
):
    hermes_root = tmp_path / "hermes"
    hermes_root.mkdir()
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root", lambda: hermes_root
    )
    db = SessionDB(hermes_root / "state.db")

    try:
        assert db.try_reserve_restored_draft_session(
            DRAFT_ID,
            source="desktop",
            profile_name=None,
        )
        row = db.get_session(DRAFT_ID)
        assert row is not None
        assert row["profile_name"] == "default"
    finally:
        db.close()


def test_session_create_rejects_a_malformed_restore_key(gateway):
    out = srv._methods["session.create"](
        "r2",
        {
            "restore_session_id": "../../not-a-session",
            "source": "desktop",
        },
    )

    assert out.get("error", {}).get("code") == 4006
    assert srv._sessions == {}


def test_session_create_will_not_reuse_a_durable_session_key(gateway):
    gateway.create_session(session_id=DRAFT_ID, source="desktop", model="test-model")

    out = srv._methods["session.create"](
        "r3",
        {
            "restore_session_id": DRAFT_ID,
            "source": "desktop",
        },
    )

    assert out.get("error", {}).get("code") == 4090
    assert srv._sessions == {}


def test_session_create_will_not_duplicate_a_live_lazy_key(gateway):
    live = {"profile_home": None, "session_key": DRAFT_ID}
    srv._sessions["already-live"] = live

    out = srv._methods["session.create"](
        "r4",
        {
            "restore_session_id": DRAFT_ID,
            "source": "desktop",
        },
    )

    assert out.get("error", {}).get("code") == 4090
    assert srv._sessions == {"already-live": live}


def test_restored_draft_reservation_is_atomic_across_database_handles(tmp_path):
    path = tmp_path / "shared-state.db"
    first = SessionDB(path)
    second = SessionDB(path)
    barrier = threading.Barrier(2)

    def reserve(db: SessionDB) -> bool:
        barrier.wait()
        return db.try_reserve_restored_draft_session(
            DRAFT_ID,
            source="desktop",
            model="gpt-test",
            cwd=str(tmp_path),
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(reserve, (first, second)))

        assert sorted(results) == [False, True]
        row = first.get_session(DRAFT_ID)
        assert row is not None
        assert row["message_count"] == 0
        assert row["model"] == "gpt-test"
        assert row["cwd"] == str(tmp_path)
    finally:
        first.close()
        second.close()
