"""Cross-user isolation tests for the session_search tool (P1, gateway).

In a multi-user gateway (Telegram / Discord / Slack) every user's agent runs a
separate session that carries a ``user_id``. The session_search tool must scope
its queries to the caller's own sessions, otherwise one user can read another
user's conversation history — the same security boundary as CVE-2026-11461
(/resume cross-user access).

These tests prove that:
  * FTS message search (``search_messages``) cannot return another user's
    message content when scoped by ``user_id``;
  * the title-match path (``resolve_session_by_title``) cannot resolve another
    user's session by title;
  * the browse / recent-sessions path (``list_sessions_rich``) cannot surface
    another user's session titles.

They mirror the regression style of test_hermes_state.py / test_resume_command.py.

Note: the tool intentionally hides the caller's *current* session (its content
is already in context), so "owner can see own" assertions use a *different*
session owned by the same user_id — not the current_session_id.
"""

import json
import time

import pytest

from hermes_state import SessionDB
from tools.session_search_tool import session_search


@pytest.fixture
def gateway_db(tmp_path):
    """A shared state.db with two distinct gateway users' sessions."""
    db = SessionDB(tmp_path / "state.db")
    now = int(time.time())

    # User A — telegram (current session + a separate owned session)
    db.create_session("s_userA", source="telegram", user_id="tg-userA")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, title = ? WHERE id = ?",
        (now - 5000, "User A private project", "s_userA"),
    )
    db.append_message("s_userA", role="user",
                      content="A is working on the current thing")

    db.create_session("s_userA_secret", source="telegram", user_id="tg-userA")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, title = ? WHERE id = ?",
        (now - 4500, "User A secret project", "s_userA_secret"),
    )
    db.append_message("s_userA_secret", role="user",
                      content="SECRET_A_TERM planning the alpha launch")

    # User B — telegram, with TWO sessions:
    #   s_userB_current — B's "current" session (used as current_session_id)
    #   s_userB_secret  — a different B-owned session that must be visible to B
    #                     but hidden from A.
    db.create_session("s_userB_current", source="telegram", user_id="tg-userB")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, title = ? WHERE id = ?",
        (now - 4000, "User B current project", "s_userB_current"),
    )
    db.append_message("s_userB_current", role="user",
                      content="B is working on the current thing")

    db.create_session("s_userB_secret", source="telegram", user_id="tg-userB")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, title = ? WHERE id = ?",
        (now - 3000, "User B private project", "s_userB_secret"),
    )
    db.append_message("s_userB_secret", role="user",
                      content="SECRET_B_TERM my bank login credentials discussion")
    db.append_message("s_userB_secret", role="assistant",
                      content="SECRET_B_TERM noted, here is the plan.")
    db._conn.commit()
    return db


class TestCrossUserDiscoveryLeak:
    def test_userA_cannot_search_userB_messages(self, gateway_db):
        """User A's agent searching for B's unique term returns nothing."""
        result = json.loads(session_search(
            query="SECRET_B_TERM", db=gateway_db, current_session_id="s_userA"
        ))
        assert result["success"] is True
        assert result["count"] == 0, "user A leaked user B's message content"
        assert result["results"] == []

    def test_userB_can_search_own_other_session(self, gateway_db):
        """User B (current=s_userB_current) can find a *different* B-owned
        session that contains the term."""
        result = json.loads(session_search(
            query="SECRET_B_TERM", db=gateway_db, current_session_id="s_userB_current"
        ))
        assert result["success"] is True
        assert result["count"] == 1
        assert result["results"][0]["session_id"] == "s_userB_secret"

    def test_userA_searching_own_term_only_returns_own(self, gateway_db):
        # Sanity: A (current=s_userA) searching a *different* A-owned session's
        # term only returns A's own session, never B's.
        result = json.loads(session_search(
            query="SECRET_A_TERM", db=gateway_db, current_session_id="s_userA"
        ))
        assert result["count"] == 1
        assert result["results"][0]["session_id"] == "s_userA_secret"


class TestCrossUserTitleLeak:
    def test_userA_cannot_resolve_userB_title(self, gateway_db):
        """Title-match must scope by user_id — A cannot open B's session by title."""
        result = json.loads(session_search(
            query="User B private project", db=gateway_db,
            current_session_id="s_userA",
        ))
        assert result["success"] is True
        for hit in result["results"]:
            assert hit["session_id"] != "s_userB_secret"

    def test_userB_can_resolve_own_title(self, gateway_db):
        # B (current=s_userB_current) resolving B's *other* session title.
        result = json.loads(session_search(
            query="User B private project", db=gateway_db,
            current_session_id="s_userB_current",
        ))
        assert result["count"] == 1
        assert result["results"][0]["session_id"] == "s_userB_secret"


class TestCrossUserBrowseLeak:
    def test_userA_browse_excludes_userB_session(self, gateway_db):
        result = json.loads(session_search(
            db=gateway_db, current_session_id="s_userA"
        ))
        sids = [r["session_id"] for r in result["results"]]
        assert "s_userB_secret" not in sids, "user A's browse leaked user B's session"
        assert "s_userB_current" not in sids

    def test_userB_browse_includes_own_other_session(self, gateway_db):
        result = json.loads(session_search(
            db=gateway_db, current_session_id="s_userB_current"
        ))
        sids = [r["session_id"] for r in result["results"]]
        assert "s_userB_secret" in sids, "user B should see own other session"
        assert "s_userA" not in sids


class TestUnscopedSingleUserStillWorks:
    """Backward compatibility: single-user (CLI) sessions have user_id=None and
    must remain fully searchable when no caller identity is supplied."""

    def test_no_caller_identity_returns_all(self, gateway_db):
        result = json.loads(session_search(query="SECRET_B_TERM", db=gateway_db))
        # No current_session_id → no scoping → both users' content visible.
        assert result["count"] == 1
        assert result["results"][0]["session_id"] == "s_userB_secret"
