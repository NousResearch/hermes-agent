"""Tests: profiles.list ``preferred_session_ids`` precise session summaries.

Why: the desktop BOTS roster previews ``last_session`` (the profile's most
recently active session) but clicking a bot row opens the PINNED canonical
chat — two different session identities, so the preview shows one
conversation and the click lands in another (NousResearch/hermes-agent#88200).
The generic fix at the RPC layer: callers that know which session they care
about pass ``preferred_session_ids={profile: session_id}`` and receive a
precise ``preferred_session`` summary per profile — hidden sessions included,
compression lineages resolved to the live tip, no pagination window — while
``last_session`` keeps its existing "most recent" contract.

Contract under test:
- A profile named in the map gets ``preferred_session``: a summary dict when
  the id resolves, ``None`` when it definitively does not (missing row,
  denied internal source).
- Summary keys: ``id`` (the requested pin, durable identity), ``resolved_id``
  (live compression tip; equal to ``id`` when uncompressed), ``title``,
  ``preview`` (newest user/assistant text at the tip), ``started_at``,
  ``last_active``, ``message_count``.
- Profiles not named in the map carry no ``preferred_session`` key at all.
- ``last_session`` behaviour is unchanged in every case.
- ``include_sessions: false`` skips preferred resolution entirely.
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
# preferred_session resolution
# ---------------------------------------------------------------------------


def test_preferred_session_summarizes_pin_not_latest(home):
    db = _db(home)
    _add_session(db, "pinned1", title="Bot Chat", ts=1000, text="pinned chat content")
    _add_session(db, "other1", title="Scratch", ts=2000, text="scratch pad content")
    db.close()

    rows = _profiles({"preferred_session_ids": {"default": "pinned1"}})
    row = _row(rows, "default")

    pref = row["preferred_session"]
    assert pref["id"] == "pinned1"
    assert pref["resolved_id"] == "pinned1"
    assert pref["root_title"] == "Bot Chat"
    assert pref["title"] == "Bot Chat"
    assert "pinned chat content" in pref["preview"]
    # last_session keeps its own contract: the most recently active session.
    assert row["last_session"]["id"] == "other1"


def test_preferred_session_resolves_hidden_pin(home):
    db = _db(home)
    _add_session(db, "hiddenpin", title="Bot Chat", ts=1000,
                 text="hidden bot chat content", hidden=True)
    _add_session(db, "visible1", title="Visible", ts=2000, text="visible content")
    db.close()

    rows = _profiles({"preferred_session_ids": {"default": "hiddenpin"}})
    row = _row(rows, "default")

    # The pin is precise: hidden from listings must not mean "does not exist".
    assert row["preferred_session"] is not None
    assert row["preferred_session"]["id"] == "hiddenpin"
    assert "hidden bot chat content" in row["preferred_session"]["preview"]
    # …while the generic latest-session listing still excludes hidden rows.
    assert row["last_session"]["id"] == "visible1"


def test_preferred_session_missing_returns_none_and_keeps_last_session(home):
    db = _db(home)
    _add_session(db, "real1", title="Real", ts=1000, text="real content")
    db.close()

    rows = _profiles({"preferred_session_ids": {"default": "does-not-exist"}})
    row = _row(rows, "default")

    assert row["preferred_session"] is None
    assert row["last_session"]["id"] == "real1"


def test_preferred_session_denied_internal_source_returns_none(home):
    db = _db(home)
    _add_session(db, "toolrun", source="tool", title="", ts=1000, text="tool output")
    _add_session(db, "human1", title="Human", ts=2000, text="human content")
    db.close()

    rows = _profiles({"preferred_session_ids": {"default": "toolrun"}})
    row = _row(rows, "default")

    # Internal sources (tool sub-agent runs, kanban workers) are not
    # conversations — a pin pointing at one resolves as absent.
    assert row["preferred_session"] is None


def test_preferred_session_resolves_compression_tip(home):
    db = _db(home)
    _add_session(db, "root1", title="Bot Chat", ts=1000,
                 text="pre-compression content", end_reason="compression")
    _add_session(db, "tip1", title="Bot Chat (continued)", ts=3000,
                 text="post-compression content", parent="root1")
    _add_session(db, "other1", title="Other", ts=4000, text="other content")
    db.close()

    rows = _profiles({"preferred_session_ids": {"default": "root1"}})
    row = _row(rows, "default")

    pref = row["preferred_session"]
    # The pin keeps its durable identity; the summary comes from the live tip.
    assert pref["id"] == "root1"
    assert pref["resolved_id"] == "tip1"
    assert pref["root_title"] == "Bot Chat"
    assert pref["title"] == "Bot Chat (continued)"
    assert "post-compression content" in pref["preview"]


# ---------------------------------------------------------------------------
# Contract guards
# ---------------------------------------------------------------------------


def test_no_param_omits_preferred_key(home):
    db = _db(home)
    _add_session(db, "s1", title="S", ts=1000, text="content")
    db.close()

    row = _row(_profiles({}), "default")
    assert "preferred_session" not in row
    assert row["last_session"]["id"] == "s1"


def test_include_sessions_false_skips_preferred(home):
    db = _db(home)
    _add_session(db, "s1", title="S", ts=1000, text="content")
    db.close()

    row = _row(
        _profiles({"include_sessions": False,
                   "preferred_session_ids": {"default": "s1"}}),
        "default",
    )
    assert "last_session" not in row
    assert "preferred_session" not in row


def test_preferred_ids_scoped_per_profile_db(home):
    # Same session id in BOTH profiles' state.db files, different content —
    # each row must summarize its own profile's database.
    default_db = _db(home)
    _add_session(default_db, "shared1", title="Default Bot", ts=1000,
                 text="default profile content")
    default_db.close()

    ops_db = _db(home / "profiles" / "ops")
    _add_session(ops_db, "shared1", title="Ops Bot", ts=1000,
                 text="ops profile content")
    ops_db.close()

    rows = _profiles(
        {"preferred_session_ids": {"default": "shared1", "ops": "shared1"}}
    )
    assert "default profile content" in _row(rows, "default")["preferred_session"]["preview"]
    assert "ops profile content" in _row(rows, "ops")["preferred_session"]["preview"]
