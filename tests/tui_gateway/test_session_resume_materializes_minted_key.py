"""``session.resume`` materializes a row for a minted-but-never-persisted key.

Bug class: the backend dies between ``session.create`` (which intentionally
writes no state.db row until the first prompt — no "Untitled" litter) and that
first ``prompt.submit``. The stored key exists client-side (pinned tile,
sidebar) but has no row anywhere, and after a restart the in-memory live-lazy
lookup can't find it either — so resume used to 4007 forever and the desktop
surfaced "no route to the backend holding this session".

The fix: when the resume target is a well-formed server-minted key
(``%Y%m%d_%H%M%S_`` + 6 hex, the exact shape ``_new_session_key()`` produces),
materialize the row and continue the normal empty resume instead of failing.
Abandoned drafts still leave no row — nothing is written until a client
explicitly resumes the exact key. Fail-closed: garbage, 8-hex runtime ids,
and titles still 4007, and lazy subagent watch windows are excluded.
"""

from __future__ import annotations

import pytest

from hermes_state import SessionDB
from tui_gateway import server

MINTED_KEY = "20260828_053121_3427a9"  # the morning's lost session shape
EXISTING_KEY = "20260827_224618_956bf8"


@pytest.fixture()
def real_db(monkeypatch, tmp_path):
    """Real SessionDB behind ``_get_db``, with the heavy resume machinery off."""
    db = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_find_live_session_by_key", lambda _key: None)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a, **k: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda *a, **k: None)
    monkeypatch.setattr(server, "_maybe_schedule_auto_continue", lambda *a, **k: None)
    monkeypatch.setattr(server, "_default_session_cwd", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(server, "_profile_configured_cwd", lambda _home: str(tmp_path))
    monkeypatch.setattr(server, "_child_run_active", lambda _key: False)
    known = set(server._sessions)
    yield db
    with server._sessions_lock:
        for sid in [s for s in server._sessions if s not in known]:
            server._sessions.pop(sid, None)
    db.close()


def _resume(**params):
    return server.handle_request({"id": "1", "method": "session.resume", "params": params})


def _assert_resumed_ok(resp, target):
    assert "error" not in resp, resp
    assert resp["result"]["resumed"] == target
    assert resp["result"]["session_id"], "a live runtime must be registered"


def test_resume_materializes_row_for_minted_key(real_db):
    """The core fix: a minted-but-never-persisted key resumes, row now exists."""
    resp = _resume(session_id=MINTED_KEY)

    _assert_resumed_ok(resp, MINTED_KEY)
    row = real_db.get_session(MINTED_KEY)
    assert row is not None, "resume must materialize the DB row"
    assert row["source"] == "tui"
    assert row["model"] == "test-model"
    assert row.get("profile_name") is None  # launch/default profile


def test_resume_existing_row_unaffected(real_db):
    """A persisted session resumes normally and its row is not duplicated."""
    real_db.create_session(EXISTING_KEY, source="tui", model="test-model")
    real_db.append_message(EXISTING_KEY, "user", "hello")

    resp = _resume(session_id=EXISTING_KEY)

    _assert_resumed_ok(resp, EXISTING_KEY)
    # message_count stays 1 — a second create would have been an INSERT-OR-IGNORE
    # no-op, but the row must remain the same single conversation.
    assert real_db.get_session(EXISTING_KEY)["message_count"] == 1


def test_resume_garbage_id_still_4007(real_db):
    """Non-key garbage stays fail-closed: same error, no write."""
    resp = _resume(session_id="not-a-session")

    assert resp["error"]["code"] == 4007
    # nothing materialized for arbitrary strings
    assert len(real_db.list_sessions_rich(source="tui")) == 0


def test_resume_8hex_runtime_id_still_4007(real_db):
    """8-hex runtime session ids (``uuid4().hex[:8]``) are NOT stored keys."""
    resp = _resume(session_id="5a81f231")

    assert resp["error"]["code"] == 4007
    assert len(real_db.list_sessions_rich(source="tui")) == 0


def test_resume_lazy_watch_missing_key_still_4007(real_db):
    """Lazy subagent watch windows must not mint phantom rows for dead children."""
    resp = _resume(session_id=MINTED_KEY, lazy=True)

    assert resp["error"]["code"] == 4007
    assert real_db.get_session(MINTED_KEY) is None


def test_resume_minted_key_claimed_by_other_profile_fails_closed(real_db):
    """A key claimed by a live session under ANOTHER profile must not mint here.

    Mirrors test_resume_live_lazy_session's unscoped-resume fail-closed rule
    (#93296): the materialize gate must not turn a cross-profile leak into a
    phantom launch-store row. The key looks server-minted, but a live record
    owns it — routing guesses are forbidden.
    """
    record = {
        "history": [],
        "last_active": 0.0,
        "pending_title": "Bot Chat",
        "pending_hidden": True,
        "profile_home": "/tmp/other-profile",
        "running": False,
        "session_key": MINTED_KEY,
        "source": "desktop",
    }
    server._sessions["live-other-profile"] = record
    try:
        resp = _resume(session_id=MINTED_KEY)
        assert resp["error"]["code"] == 4007
        assert real_db.get_session(MINTED_KEY) is None
    finally:
        with server._sessions_lock:
            server._sessions.pop("live-other-profile", None)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("20260828_053121_3427a9", True),
        ("20260827_224618_956bf8", True),
        ("5a81f231", False),  # 8-hex runtime sid
        ("fcd97d0c", False),  # 8-hex runtime sid
        ("not-a-session", False),
        ("", False),
        ("20260828_053121_3427", False),  # too-short hex tail
        ("20260828_053121_3427a9ZZ", False),  # trailing garbage
        ("Bot Chat", False),  # titles are not keys
    ],
)
def test_is_server_minted_key_shape(value, expected):
    assert server._is_server_minted_key(value) is expected