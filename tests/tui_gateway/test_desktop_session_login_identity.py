"""A Desktop session's state.db row must record the login that created it.

The dashboard/serve backend resolves the human at WS-upgrade auth (it writes
``login_success`` with the right ``user_id`` to logs/dashboard-auth.log) and the
runtime already carries that identity on the session record as ``auth_user_id``
(``_transport_auth_user_id``) — the same value the agent is built with. The
row-creating write just never passed it on, so every Desktop session read back
from ``state.db`` with an empty ``user_id`` while Discord/Telegram sessions
carried theirs (their adapters put the sender on the row at creation).

Anonymous sessions (no password provider, legacy token, stdio) stay
identity-less: nothing is inferred and nothing is stamped.
"""

from __future__ import annotations

from hermes_state import SessionDB
from tui_gateway import server
from tui_gateway.transport import bind_transport, reset_transport

_LOADED_USER = "alice"
_LOGIN = f"basic:{_LOADED_USER}"


class _LoginSocket:
    """A live Desktop WS peer carrying the identity the upgrade auth minted."""

    def __init__(self, user_id=_LOADED_USER, provider="basic"):
        self.auth_identity = {"provider": provider, "user_id": user_id}

    def write(self, frame):
        return True


class _AnonymousSocket:
    """A peer with no login behind it (stdio, legacy token, auth disabled)."""

    def write(self, frame):
        return True


def _real_db(monkeypatch, tmp_path) -> SessionDB:
    db = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
    monkeypatch.setattr(server, "_start_agent_build", lambda *a, **k: None)
    return db


def _create_desktop_session(transport, tmp_path) -> tuple[dict, str, str]:
    """``session.create`` as the Desktop client sends it; returns (record, sid, stored key)."""
    token = bind_transport(transport)
    try:
        resp = server.handle_request({
            "id": "1", "method": "session.create",
            "params": {"cols": 80, "source": "desktop", "cwd": str(tmp_path)},
        })
        sid = resp["result"]["session_id"]
        return dict(server._sessions[sid]), sid, resp["result"]["stored_session_id"]
    finally:
        reset_transport(token)


def test_desktop_session_row_records_the_logged_in_user(monkeypatch, tmp_path):
    """Login succeeded, then a conversation: the row must name the human."""
    db = _real_db(monkeypatch, tmp_path)
    record, sid, key = _create_desktop_session(_LoginSocket(), tmp_path)
    try:
        assert record["auth_user_id"] == _LOGIN
        # What prompt.submit does before the agent exists (the first row write).
        assert server._ensure_session_db_row(record) is True
        # A branch is a Desktop session too — the child row names the same human.
        server._seed_branch_row(record, "branch-key", key, [{"role": "user", "content": "hi"}],
                                "desktop", None)
    finally:
        server._sessions.pop(sid, None)

    assert db.get_session(key)["user_id"] == _LOGIN
    assert db.get_session("branch-key")["user_id"] == _LOGIN


def test_anonymous_desktop_session_row_stays_identity_less(monkeypatch, tmp_path):
    """No login on the transport => the row keeps its empty user_id."""
    db = _real_db(monkeypatch, tmp_path)
    record, sid, key = _create_desktop_session(_AnonymousSocket(), tmp_path)
    try:
        assert record["auth_user_id"] is None
        assert server._ensure_session_db_row(record) is True
    finally:
        server._sessions.pop(sid, None)

    assert not (db.get_session(key)["user_id"] or "").strip()
