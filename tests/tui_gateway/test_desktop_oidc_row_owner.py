"""A Desktop chat created by an issuer-bearing (OIDC) dashboard login belongs to that login.

Desktop rows record their creator in ``user_id``. For an issuer-less login that is the
``<provider>:<user>`` label, and ``_login_label`` maps the login's subject back to it. The label
drops the issuer, so ``_login_label`` deliberately refuses issuer-bearing subjects (#140183 review
F1). Rows created *by* an issuer-bearing login were stamped with that same issuer-less label, so
no subject could ever match them: the login that created the chat got ``permission_denied`` on
every sidebar edit (delete, rename, archive, pin) of its own chats.

The row now records the full authenticated subject for an issuer-bearing login, which only that
exact (provider, issuer, user) can spell. The agent keeps the ``<provider>:<user>`` label, so
memory providers see the same user id as before.
"""

from __future__ import annotations

import pytest

from gateway.session_contract import Principal
from gateway.session_identity import authenticated_subject
from hermes_state import SessionDB
from hermes_state_local_migration import validate_legacy_row
from hermes_state_mutation_binding import authorize_history
from hermes_state_runtime import RuntimeStoreError
from tui_gateway import server
from tui_gateway.transport import bind_transport, reset_transport

_ISSUER = "https://idp.example/application/o/hermes/"
_OIDC = {"provider": "self-hosted", "issuer": _ISSUER, "user_id": "4f1c2e"}
_CAPS = frozenset({"session:read", "session:control", "session:create"})


class _Socket:
    """A live Desktop WS peer carrying the identity the upgrade auth minted (``consume_ticket``)."""

    def __init__(self, identity):
        self.auth_identity = dict(identity)

    def write(self, frame):
        return True


def _real_db(monkeypatch, tmp_path) -> SessionDB:
    db = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
    monkeypatch.setattr(server, "_start_agent_build", lambda *a, **k: None)
    return db


def _create_row(transport, tmp_path) -> tuple[dict, str]:
    """``session.create`` as Desktop sends it, then the first row write; returns (record, stored key)."""
    token = bind_transport(transport)
    try:
        resp = server.handle_request({
            "id": "1", "method": "session.create",
            "params": {"cols": 80, "source": "desktop", "cwd": str(tmp_path)},
        })
        sid = resp["result"]["session_id"]
        record = server._sessions[sid]
        try:
            assert server._ensure_session_db_row(record) is True
            return dict(record), resp["result"]["stored_session_id"]
        finally:
            server._sessions.pop(sid, None)
    finally:
        reset_transport(token)


def _principal(identity, tmp_path) -> Principal:
    return Principal(authenticated_subject(identity), str(tmp_path), _CAPS, "http")


def test_oidc_login_owns_the_desktop_chats_it_created(monkeypatch, tmp_path):
    db = _real_db(monkeypatch, tmp_path)
    record, key = _create_row(_Socket(_OIDC), tmp_path)
    try:
        # The agent's user id is unchanged: memory providers keep keying on the label.
        assert record["auth_user_id"] == "self-hosted:4f1c2e"
        owner = _principal(_OIDC, tmp_path)
        row = db.get_session(key)
        assert row["user_id"] == owner.subject
        validate_legacy_row(db, row, subject=owner.subject)
        with db._read_ctx() as conn:
            assert authorize_history(conn, owner, key) is True
    finally:
        db.close()


def test_same_user_id_from_another_issuer_stays_foreign(monkeypatch, tmp_path):
    """The reason ``_login_label`` refuses issuer-bearing subjects still holds: an IdP re-pointed under
    the same provider name (or a second IdP with a colliding ``sub``) must not reach these chats."""
    db = _real_db(monkeypatch, tmp_path)
    _, key = _create_row(_Socket(_OIDC), tmp_path)
    try:
        others = (
            {**_OIDC, "issuer": "https://other-idp.example/"},
            {"provider": "self-hosted", "user_id": "4f1c2e"},  # issuer-less login spelling the old label
            {**_OIDC, "user_id": "someone-else"},
        )
        for identity in others:
            stranger = _principal(identity, tmp_path)
            with pytest.raises(RuntimeStoreError, match="not_found"):
                validate_legacy_row(db, db.get_session(key), subject=stranger.subject)
            with db._read_ctx() as conn, pytest.raises(RuntimeStoreError, match="permission_denied"):
                authorize_history(conn, stranger, key, claim=True)
    finally:
        db.close()


def test_issuer_less_login_keeps_the_label_stamp(monkeypatch, tmp_path):
    db = _real_db(monkeypatch, tmp_path)
    identity = {"provider": "basic", "user_id": "alice"}
    _, key = _create_row(_Socket(identity), tmp_path)
    try:
        assert db.get_session(key)["user_id"] == "basic:alice"
        with db._read_ctx() as conn:
            assert authorize_history(conn, _principal(identity, tmp_path), key) is True
    finally:
        db.close()
