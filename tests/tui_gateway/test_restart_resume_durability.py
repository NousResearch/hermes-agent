"""Regression coverage for Desktop restart/reconnect session durability."""

from __future__ import annotations

import threading

from hermes_state import SessionDB


def _reset_live_sessions(server) -> None:
    """Drop process-local sessions to model a backend process restart."""
    with server._sessions_lock:
        server._sessions.clear()


def _role_content(messages: list[dict]) -> list[dict[str, str]]:
    """Compare transcript identity without generated SQLite timestamps."""
    return [
        {"role": message["role"], "content": message["content"]} for message in messages
    ]


def test_restart_resume_rejects_empty_history_and_preserves_durable_transcript(
    tmp_path, monkeypatch
):
    """A reconnect must not attach an empty agent to an existing durable id."""
    import tui_gateway.server as server

    db_path = tmp_path / "state.db"
    session_id = "20260719_142032_0e82bb"
    expected = [
        {"role": "user", "content": "FitMass durable prompt"},
        {"role": "assistant", "content": "FitMass durable answer"},
    ]

    seed_db = SessionDB(db_path=db_path)
    seed_db.create_session(session_id, source="desktop")
    for message in expected:
        seed_db.append_message(
            session_id,
            role=message["role"],
            content=message["content"],
        )
    seed_db.close()

    opened_dbs: list[SessionDB] = []

    def restart_db() -> SessionDB:
        db = SessionDB(db_path=db_path)
        opened_dbs.append(db)
        monkeypatch.setattr(server, "_get_db", lambda: db)
        return db

    monkeypatch.setattr(server, "_schedule_agent_build", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server,
        "_wait_agent",
        lambda *_args, **_kwargs: {
            "error": {"message": "agent build disabled by durability test"}
        },
    )
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    monkeypatch.setattr(server, "_default_session_cwd", lambda: str(tmp_path))
    monkeypatch.setattr(
        server,
        "_claim_active_session_slot",
        lambda *_args, **_kwargs: (None, None),
    )

    try:
        _reset_live_sessions(server)
        first_db = restart_db()
        first = server.handle_request({
            "id": "resume-after-reconnect",
            "method": "session.resume",
            "params": {"session_id": session_id, "source": "desktop"},
        })
        assert first is not None
        assert "error" not in first
        assert first["result"]["session_key"] == session_id
        assert [
            {"role": message["role"], "content": message["text"]}
            for message in first["result"]["messages"]
        ] == expected
        assert (
            _role_content(first_db.get_messages_as_conversation(session_id)) == expected
        )

        # A second process-local reset models the backend restart that followed
        # `hermes update`. Reopening the same durable conversation must hydrate
        # it again rather than manufacturing an empty runtime under the same id.
        _reset_live_sessions(server)
        first_db.close()
        second_db = restart_db()
        second = server.handle_request({
            "id": "resume-after-restart",
            "method": "session.resume",
            "params": {"session_id": session_id, "source": "desktop"},
        })
        assert second is not None
        assert "error" not in second
        assert second["result"]["message_count"] == len(expected)
        assert (
            _role_content(second_db.get_messages_as_conversation(session_id))
            == expected
        )

        # Reproduce the incident's post-restart shape without touching the
        # durable rows: the loader unexpectedly returns history=0 for the same
        # id even though its canonical transcript still exists in SQLite.
        _reset_live_sessions(server)
        real_get_resume_conversations = second_db.get_resume_conversations
        monkeypatch.setattr(
            second_db,
            "get_resume_conversations",
            lambda target: (
                ([], [])
                if target == session_id
                else real_get_resume_conversations(target)
            ),
        )
        damaged = server.handle_request({
            "id": "resume-empty-after-restart",
            "method": "session.resume",
            "params": {"session_id": session_id, "source": "desktop"},
        })

        assert damaged is not None
        assert damaged["error"]["code"] == 4091
        assert "empty transcript" in damaged["error"]["message"]
        assert not any(
            session.get("session_key") == session_id
            for session in server._sessions.values()
        )

        # Reconnect has a distinct fast path: a process-local session can still
        # exist after its WebSocket detached. It must enforce the same invariant
        # rather than returning that empty record merely because the id matches.
        with server._sessions_lock:
            server._sessions["empty-runtime"] = {
                "history": [],
                "history_lock": threading.Lock(),
                "inflight_turn": None,
                "queued_prompt": None,
                "resumed_session": True,
                "running": False,
                "session_key": session_id,
            }
        reconnected = server.handle_request({
            "id": "reconnect-empty-live-session",
            "method": "session.activate",
            "params": {"session_id": "empty-runtime"},
        })
        assert reconnected is not None
        assert reconnected["error"]["code"] == 4091

        submitted = server.handle_request({
            "id": "submit-to-empty-live-session",
            "method": "prompt.submit",
            "params": {"session_id": "empty-runtime", "text": "new conversation"},
        })
        assert submitted is not None
        assert submitted["error"]["code"] == 4091
        assert (
            _role_content(second_db.get_messages_as_conversation(session_id))
            == expected
        )
    finally:
        _reset_live_sessions(server)
        for db in opened_dbs:
            try:
                db.close()
            except Exception:
                pass
