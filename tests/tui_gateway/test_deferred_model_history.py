"""Bounded Desktop hydration salvaged from Benjamin Brumbaugh's PR #106838."""

import threading

import pytest

from agent.replay_cleanup import canonicalize_replay_history
from persistence.conversation.messages import ConversationRequest, NewMessage
from persistence.conversation.sessions import NewSession
from persistence_assembly import close_stores_under, open_persistence, profile_scope, store_location
from tests.fixtures.storage_holder import raw_holder
from tui_gateway import server


def _repository_type(home, name):
    """The class of the storage interface repository ``name`` bound on ``home``'s store."""
    with open_persistence(profile_scope(home=home)) as persistence:
        return type(getattr(persistence, name))


@pytest.mark.parametrize("source,omit_messages", [("desktop", True), ("desktop", False), ("tui", True)])
@pytest.mark.parametrize("profile", [None, "work"])
def test_deferred_resume_preserves_model_history_and_db_ownership(tmp_path, monkeypatch, source, omit_messages, profile):
    home = tmp_path / "work"
    home.mkdir()
    db = open_persistence(profile_scope(home=home), eager=True)
    db.sessions.create(NewSession(session_id="parent", source=source))
    db.messages.append("parent", NewMessage(role="user", content="ancestor-only display", timestamp=100.0))
    db.sessions.end("parent", "compression")
    db.sessions.create(NewSession(session_id="tip", source=source, parent_session_id="parent"))
    db.messages.append("tip", NewMessage(role="user", content="archived display", timestamp=101.0))
    db.messages.compact("tip", [
        NewMessage(role="assistant", content="summary", compressed_summary=True),
        NewMessage(role="user", content="current ask"),
        NewMessage(role="assistant", content="", tool_calls=(
            {"id": "dangling", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},)),
    ])
    expected = canonicalize_replay_history([m.as_openai_message() for m in db.messages.read_conversation(
        "tip", ConversationRequest(repair_alternation=True, include_row_ids=True))])
    stored = db.sessions.get("tip")
    assert stored is not None
    stored_count = stored.message_count
    display = db.messages.read_resume_conversations("tip").display
    prefix = [m.as_openai_message() for m in db.lineage.ancestor_display_prefix("tip")]
    display_reads = []
    messages_type, lineage_type = _repository_type(home, "messages"), _repository_type(home, "lineage")
    original_display = messages_type.read_resume_conversations
    original_prefix = lineage_type.ancestor_display_prefix
    closed = threading.Event()
    built = threading.Event()
    events = []
    original_close = db.close
    dedicated_closes = []
    real_profile_session_db = server._profile_session_db

    def close():
        original_close()
        closed.set()

    def profile_session_db(profile_home):
        opened, owns = real_profile_session_db(profile_home)
        if owns:
            assert store_location(opened.scope) == (home / "state.db").resolve()
            release = opened.close

            def close_dedicated():
                release()
                dedicated_closes.append(opened)
                closed.set()

            opened.close = close_dedicated
        return opened, owns

    def read_display(repository, sid):
        display_reads.append("display")
        return original_display(repository, sid)

    def read_prefix(repository, sid):
        display_reads.append("prefix")
        return original_prefix(repository, sid)

    monkeypatch.setattr(db, "close", close)
    monkeypatch.setattr(messages_type, "read_resume_conversations", read_display)
    monkeypatch.setattr(lineage_type, "ancestor_display_prefix", read_prefix)
    monkeypatch.setattr(server, "_profile_session_db", profile_session_db)
    monkeypatch.setattr(server, "_profile_home", lambda p: home if p else None)
    monkeypatch.setattr(server, "_profile_configured_cwd", lambda _: str(tmp_path))
    monkeypatch.setattr(server, "_default_session_cwd", lambda: str(tmp_path))
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_maybe_schedule_auto_continue", lambda *args: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda *args: built.set())
    monkeypatch.setattr(server, "_emit", lambda kind, sid, payload: events.append((kind, payload)))
    sid = None
    try:
        response = server.handle_request({"id": "resume", "method": "session.resume", "params": {
            "session_id": "tip", "source": source, "defer_history": True,
            "omit_messages": omit_messages, **({"profile": profile} if profile else {}),
        }})
        assert response is not None and "error" not in response, response
        sid = response["result"]["session_id"]
        session = server._sessions[sid]
        assert session["resume_history_ready"].wait(5)
        assert built.wait(5)
        model_only = source == "desktop" and omit_messages
        assert display_reads == ([] if model_only else ["display", "prefix"])
        assert session["history"] == expected
        assert session["display_history_prefix"] == ([] if model_only else prefix)
        count = stored_count if model_only else len(display)
        assert session["resume_message_count"] == count
        assert ("session.resume_progress", {"message_count": count, "phase": "history", "status": "complete"}) in events
        if profile:
            assert closed.wait(5)
            assert len(dedicated_closes) == 1
        else:
            assert not closed.is_set()
            assert db.sessions.get("tip") is not None
    finally:
        if sid is not None:
            server._sessions.pop(sid, None)
        db.close()


@pytest.mark.parametrize("outcome", ["replaced", "failed"])
def test_model_hydration_discards_stale_results_and_closes_owned_db(tmp_path, monkeypatch, outcome):
    db = open_persistence(profile_scope(home=tmp_path), eager=True)
    db.sessions.create(NewSession(session_id="stored", source="desktop"))
    db.messages.append("stored", NewMessage(role="user", content="model context"))
    started, release, closed = threading.Event(), threading.Event(), threading.Event()
    messages_type = _repository_type(tmp_path, "messages")
    original_read, original_close = messages_type.read_conversation, db.close
    events, builds = [], []
    old = {"history": [], "history_lock": threading.RLock(), "resume_hydrating": True,
           "resume_history_ready": threading.Event(), "agent_ready": threading.Event(),
           "resume_message_count": 1}
    replacement = {"history": [{"role": "user", "content": "replacement"}]}
    monkeypatch.setitem(server._sessions, "hydrating", old)

    def read(repository, *args, **kwargs):
        started.set()
        assert release.wait(5)
        if outcome == "failed":
            with raw_holder(db)._read_ctx() as conn:
                conn.execute("SELECT * FROM deliberately_missing_table")
        return original_read(repository, *args, **kwargs)

    def close():
        original_close()
        closed.set()

    monkeypatch.setattr(messages_type, "read_conversation", read)
    monkeypatch.setattr(db, "close", close)
    monkeypatch.setattr(server, "_emit", lambda *args: events.append(args))
    monkeypatch.setattr(server, "_start_agent_build", lambda *args: builds.append(args))
    try:
        server._schedule_resume_hydration("hydrating", "stored", db, close_db=True, model_history_only=True)
        assert started.wait(5)
        assert not closed.is_set()
        if outcome == "replaced":
            server._sessions["hydrating"] = replacement
        release.set()
        assert closed.wait(5)
        assert close_stores_under(tmp_path) == 0
        assert builds == []
        if outcome == "replaced":
            assert server._sessions["hydrating"] is replacement
            assert replacement == {"history": [{"role": "user", "content": "replacement"}]}
            assert old["history"] == []
            assert not any(payload.get("status") == "complete" for _, _, payload in events)
        else:
            assert "hydrating" not in server._sessions
            assert old["resume_history_ready"].is_set()
            assert "deliberately_missing_table" in old["resume_history_error"]
    finally:
        release.set()
        assert closed.wait(5)
        server._sessions.pop("hydrating", None)
        original_close()
