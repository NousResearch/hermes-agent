"""Regression guard for cross-tenant SESSION leak (leak #2).

The in-app Super Agent multiplexes EVERY Avocado end user through ONE Hermes
process, pinning identity via the ``x-avocado-user-id`` header (surfaced to the
agent as ``gateway_session_key = "avocado:<user_id>"``). The session store
(``state.db``) used to be a single process-wide file read/written with no
per-user filter, so one tenant's ``session_search`` browse/discover/read — and
the ``X-Hermes-Session-Id`` continuation path — returned ANOTHER tenant's full
transcripts (the confirmed prod incident).

The fix mirrors the already-shipped MEMORY fix: a per-user scope, sanitized for
traversal, routes ALL session reads+writes to a physically separate
``sessions/<scope>/state.db``. No scope (single-tenant CLI / Telegram fleet)
keeps the shared ``state.db`` byte-for-byte unchanged.

These tests prove:
  * two scopes can't see each other's sessions via the session_search tool,
  * the scope is path-traversal sanitized,
  * no scope uses the shared default path.
"""
import json
import time

import pytest

from hermes_state import (
    DEFAULT_DB_PATH,
    SessionDB,
    sanitize_session_scope,
    scoped_session_db_path,
)
from tools.session_search_tool import session_search


def _seed_session(db, sid, title, lines):
    db.create_session(sid, source="api_server")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, title = ? WHERE id = ?",
        (int(time.time()), title, sid),
    )
    for role, content in lines:
        db.append_message(sid, role=role, content=content)
    db._conn.commit()


class TestSessionScopeIsolation:
    def test_scoped_dbs_are_physically_separate(self, tmp_path, monkeypatch):
        """A scoped SessionDB lands at sessions/<scope>/state.db, isolated."""
        monkeypatch.setattr("hermes_state.get_hermes_home", lambda: tmp_path)

        alice = SessionDB(user_scope="user_alice")
        bob = SessionDB(user_scope="user_bob")

        assert alice.db_path == tmp_path / "sessions" / "user_alice" / "state.db"
        assert bob.db_path == tmp_path / "sessions" / "user_bob" / "state.db"
        assert alice.db_path != bob.db_path

    def test_browse_cannot_see_other_tenant(self, tmp_path, monkeypatch):
        """Tenant B's 'list my recent sessions' (browse) must not return A's."""
        monkeypatch.setattr("hermes_state.get_hermes_home", lambda: tmp_path)

        alice = SessionDB(user_scope="user_alice")
        bob = SessionDB(user_scope="user_bob")

        _seed_session(
            alice, "a_secret", "Alice IG plan",
            [("user", "My Instagram handle is @alice_secret"),
             ("assistant", "Noted, @alice_secret.")],
        )
        _seed_session(
            bob, "b_topic", "Bob video",
            [("user", "Make me a product video"),
             ("assistant", "On it.")],
        )

        # Bob browses his own scoped DB — must see ONLY his session.
        out = json.loads(session_search(db=bob))
        ids = {r["session_id"] for r in out["results"]}
        assert "b_topic" in ids
        assert "a_secret" not in ids

        # And Alice's transcript content is not reachable from Bob's DB.
        dump = json.loads(session_search(db=bob, session_id="a_secret"))
        assert dump.get("success") is False

    def test_discover_cannot_see_other_tenant(self, tmp_path, monkeypatch):
        """FTS5 discovery in B's DB must not surface A's matching messages."""
        monkeypatch.setattr("hermes_state.get_hermes_home", lambda: tmp_path)

        alice = SessionDB(user_scope="user_alice")
        bob = SessionDB(user_scope="user_bob")

        _seed_session(
            alice, "a_kraken", "Alice kraken",
            [("user", "Tell me about the kraken project"),
             ("assistant", "The kraken project is Alice-only.")],
        )
        _seed_session(
            bob, "b_other", "Bob other",
            [("user", "Unrelated chat"), ("assistant", "Sure.")],
        )

        out = json.loads(session_search(query="kraken", db=bob))
        assert out["success"] is True
        assert out["count"] == 0  # Alice's "kraken" session is in a different DB

        # Sanity: Alice CAN find it in her own DB (scoping isn't over-broad).
        out_a = json.loads(session_search(query="kraken", db=alice))
        assert out_a["count"] == 1

    def test_continuation_session_id_isolated(self, tmp_path, monkeypatch):
        """The X-Hermes-Session-Id continuation reads only the caller's DB.

        Simulates the prod incident shape: tenant B passes tenant A's
        session_id. Resolving against B's scoped DB returns no history rather
        than A's transcript.
        """
        monkeypatch.setattr("hermes_state.get_hermes_home", lambda: tmp_path)

        alice = SessionDB(user_scope="user_alice")
        bob = SessionDB(user_scope="user_bob")

        _seed_session(
            alice, "a_session_id", "Alice private",
            [("user", "secret alpha"), ("assistant", "secret beta")],
        )

        # Bob's scoped DB has never heard of Alice's session id.
        history = bob.get_messages_as_conversation("a_session_id")
        assert history == []

        # Alice's own DB still resolves it (no over-scoping).
        history_a = alice.get_messages_as_conversation("a_session_id")
        assert any("secret" in (m.get("content") or "") for m in history_a)

    def test_create_session_stamps_user_id_on_scoped_db(self, tmp_path, monkeypatch):
        """Defense in depth: scoped DB rows carry the owning user_id."""
        monkeypatch.setattr("hermes_state.get_hermes_home", lambda: tmp_path)

        alice = SessionDB(user_scope="user_alice")
        alice.create_session("a_row", source="api_server")
        row = alice.get_session("a_row")
        assert row["user_id"] == "user_alice"

    def test_unscoped_db_uses_shared_default_path(self, tmp_path, monkeypatch):
        """No scope (single-tenant / Telegram fleet) keeps the shared path."""
        monkeypatch.setattr("hermes_state.get_hermes_home", lambda: tmp_path)

        # Explicit db_path still wins (cross-profile reads / tests).
        explicit = SessionDB(tmp_path / "explicit.db")
        assert explicit.db_path == tmp_path / "explicit.db"

        # No scope, no db_path => the module-level DEFAULT_DB_PATH, unchanged.
        shared = SessionDB()
        assert shared.db_path == DEFAULT_DB_PATH

        # create_session must NOT stamp a user_id on the shared/legacy path.
        shared.create_session("legacy_row", source="cli")
        assert shared.get_session("legacy_row")["user_id"] is None

    def test_sanitize_blocks_path_traversal(self):
        assert "/" not in (sanitize_session_scope("../../etc/passwd") or "")
        assert "\\" not in (sanitize_session_scope("..\\..\\win") or "")
        assert sanitize_session_scope("user_2abcDEF-09") == "user_2abcDEF-09"
        assert sanitize_session_scope("") is None
        assert sanitize_session_scope(None) is None
        assert sanitize_session_scope("...") is None

    def test_scoped_path_traversal_cannot_escape(self, tmp_path, monkeypatch):
        """A malicious scope can't write outside the sessions/ dir."""
        monkeypatch.setattr("hermes_state.get_hermes_home", lambda: tmp_path)
        p = scoped_session_db_path("../../escape")
        # Sanitized to a flat name under sessions/, no parent escape.
        assert (tmp_path / "sessions") in p.parents
        assert ".." not in p.parts

    def test_no_scope_path_helper_returns_default(self):
        assert scoped_session_db_path(None) == DEFAULT_DB_PATH
        assert scoped_session_db_path("") == DEFAULT_DB_PATH
