"""Tests: profiles.list ``canonical_session`` registry summaries.

Why: a bot's canonical forever-chat has exactly ONE identity — the session
titled "Bot Chat" on that bot's profile (core UNIQUE(title) makes it a
registry of at most one row). The desktop BOTS roster previews it and clicks
open it, so the gateway resolves the registry row server-side on every
``profiles.list`` and reports it per profile as ``canonical_session``. No
client ever passes a session pointer: the previous ``preferred_session_ids``
pin-verification contract is REMOVED (pointers dangle; names cannot).

Contract under test:
- Every profile row (with include_sessions on) carries ``canonical_session``:
  a summary dict when a "Bot Chat" row exists, ``None`` when it does not
  (no row, denied internal source, archived).
- Summary keys: ``id`` (the durable registry row), ``resolved_id`` (live
  compression tip; equal to ``id`` when uncompressed), ``root_title``,
  ``title``, ``preview`` (newest user/assistant text at the tip),
  ``started_at``, ``last_active``, ``message_count``.
- Hidden rows resolve (canonical chats are always hidden).
- ``last_session`` behaviour is unchanged in every case.
- ``include_sessions: false`` skips resolution entirely.
- Resolution reads each profile's OWN state.db (strict per-profile scoping).
"""

from __future__ import annotations

import pytest

import tui_gateway.server as srv


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Temp HERMES_HOME with the default profile plus one named profile."""
    h = tmp_path / ".hermes"
    (h / "profiles" / "ops").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


def _db(profile_dir):
    from hermes_state import SessionDB

    return SessionDB(db_path=profile_dir / "state.db")


def _add_session(db, sid, *, source="cli", title="", ts, text, hidden=False,
                 parent=None, end_reason=None):
    """Create one session with a single user message at an exact timestamp."""
    db.create_session(sid, source, parent_session_id=parent)
    db.append_message(sid, "user", text, timestamp=ts)
    with db._lock:
        db._conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, sid))
        if end_reason:
            # Mark ended AFTER appending: the DB (correctly) refuses writes
            # to a compression-closed session.
            db._conn.execute(
                "UPDATE sessions SET ended_at = ?, end_reason = ? WHERE id = ?",
                (ts + 1, end_reason, sid),
            )
    if hidden:
        db.set_session_hidden(sid, True)


def _profiles(params):
    envelope = srv._methods["profiles.list"](1, params)
    return envelope["result"]["profiles"]


def _row(profiles, name):
    return next(p for p in profiles if p["name"] == name)


# ---------------------------------------------------------------------------
# canonical_session resolution
# ---------------------------------------------------------------------------


def test_canonical_session_is_the_bot_chat_row_not_latest(home):
    db = _db(home)
    _add_session(db, "forever1", title="Bot Chat", ts=1000, text="forever chat content")
    _add_session(db, "other1", title="Scratch", ts=2000, text="scratch pad content")
    db.close()

    row = _row(_profiles({}), "default")

    canonical = row["canonical_session"]
    assert canonical["id"] == "forever1"
    assert canonical["resolved_id"] == "forever1"
    assert canonical["root_title"] == "Bot Chat"
    assert canonical["title"] == "Bot Chat"
    assert "forever chat content" in canonical["preview"]
    # last_session keeps its own contract: the most recently active session.
    assert row["last_session"]["id"] == "other1"


def test_canonical_session_resolves_hidden_row(home):
    db = _db(home)
    _add_session(db, "hiddenchat", title="Bot Chat", ts=1000,
                 text="hidden bot chat content", hidden=True)
    _add_session(db, "visible1", title="Visible", ts=2000, text="visible content")
    db.close()

    row = _row(_profiles({}), "default")

    # Canonical chats are always hidden — the registry lookup must see them.
    assert row["canonical_session"] is not None
    assert row["canonical_session"]["id"] == "hiddenchat"
    assert "hidden bot chat content" in row["canonical_session"]["preview"]
    # …while the generic latest-session listing still excludes hidden rows.
    assert row["last_session"]["id"] == "visible1"


def test_canonical_session_none_when_no_bot_chat_row(home):
    db = _db(home)
    _add_session(db, "real1", title="Real", ts=1000, text="real content")
    db.close()

    row = _row(_profiles({}), "default")

    assert row["canonical_session"] is None
    assert row["last_session"]["id"] == "real1"


def test_canonical_session_denied_internal_source_returns_none(home):
    db = _db(home)
    _add_session(db, "toolrun", source="tool", title="Bot Chat", ts=1000, text="tool output")
    _add_session(db, "human1", title="Human", ts=2000, text="human content")
    db.close()

    row = _row(_profiles({}), "default")

    # Internal sources (tool sub-agent runs, kanban workers) are not
    # conversations — a registry row minted by one resolves as absent.
    assert row["canonical_session"] is None


def test_canonical_session_resolves_compression_tip(home):
    db = _db(home)
    _add_session(db, "root1", title="Bot Chat", ts=1000,
                 text="pre-compression content", end_reason="compression")
    _add_session(db, "tip1", title="Bot Chat (continued)", ts=3000,
                 text="post-compression content", parent="root1")
    _add_session(db, "other1", title="Other", ts=4000, text="other content")
    db.close()

    row = _row(_profiles({}), "default")

    canonical = row["canonical_session"]
    # The registry row keeps its durable identity; the summary comes from the
    # live tip.
    assert canonical["id"] == "root1"
    assert canonical["resolved_id"] == "tip1"
    assert canonical["root_title"] == "Bot Chat"
    assert canonical["title"] == "Bot Chat (continued)"
    assert "post-compression content" in canonical["preview"]


# ---------------------------------------------------------------------------
# Contract guards
# ---------------------------------------------------------------------------


def test_include_sessions_false_skips_canonical(home):
    db = _db(home)
    _add_session(db, "s1", title="Bot Chat", ts=1000, text="content")
    db.close()

    row = _row(_profiles({"include_sessions": False}), "default")
    assert "last_session" not in row
    assert "canonical_session" not in row


def test_canonical_session_scoped_per_profile_db(home):
    # A "Bot Chat" row in BOTH profiles' state.db files, different content —
    # each roster row must summarize its own profile's database.
    default_db = _db(home)
    _add_session(default_db, "chat-default", title="Bot Chat", ts=1000,
                 text="default profile content")
    default_db.close()

    ops_db = _db(home / "profiles" / "ops")
    _add_session(ops_db, "chat-ops", title="Bot Chat", ts=1000,
                 text="ops profile content")
    ops_db.close()

    rows = _profiles({})
    assert "default profile content" in _row(rows, "default")["canonical_session"]["preview"]
    assert "ops profile content" in _row(rows, "ops")["canonical_session"]["preview"]
