"""Real, value-asserting integration tests for the session-ID resume round-trip.

Why: PR #64664 adds PTY reconnect-replay + session-ID resume. The resume path is
    server emits ``session.info`` carrying ``session_id``  →  client persists it
    →  client sends it back as ``?resume=``  →  server maps it to
    ``HERMES_TUI_RESUME``  →  ``session.resume`` RPC  →  SQLite history reattach.
The earlier e2e spec ran code without asserting the id actually flows through.
These tests capture the concrete id at each hop and assert equality — the RPC,
the SQLite session store, and the emit wiring are ALL real (no mocked DB, no
mocked RPC). The only boundary stubbed is the LLM agent build (``_make_agent`` /
the deferred pre-warm timer), which is the true system boundary the task allows.

What: drives ``tui_gateway.server`` — the real ``session.resume`` handler, the
    real ``_emit``/``write_json`` event wiring, and a real temp ``SessionDB`` —
    and asserts the session id and the prior conversation history round-trip.

Test: run ``pytest tests/tui_gateway/test_session_resume_roundtrip.py -v``; every
    assertion checks a concrete captured value (id equality, message content),
    not merely that code executed.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Real server module (NOT the mocked-hermes_state ``server`` fixture used by the
# protocol tests). We need the genuine SessionDB + resume handler here, exactly
# like tests/tui_gateway/test_finalize_session_persist.py does.
# ---------------------------------------------------------------------------


@pytest.fixture()
def srv():
    """Why: the resume round-trip must exercise the REAL SQLite store + RPC, so
    we import the production module unmocked and only scrub its module-level
    session dict between tests.
    What: yields ``tui_gateway.server``; clears live-session state on teardown.
    Test: each test seeds its own temp DB and asserts against real reads.
    """
    import tui_gateway.server as server

    yield server
    server._sessions.clear()
    server._pending.clear()
    server._answers.clear()


@pytest.fixture()
def db(tmp_path):
    """Why: an isolated real SQLite SessionDB so tests never share state.
    What: creates a SessionDB at a temp path, yields it, closes on teardown.
    Test: seeded rows are read back via get_messages_as_conversation.
    """
    from hermes_state import SessionDB

    database = SessionDB(db_path=tmp_path / "state.db")
    yield database
    database.close()


def _use_db(srv, monkeypatch, db):
    """Why: point the resume handler at our temp DB and neutralise the deferred
    LLM agent pre-warm (the system boundary) so the round-trip is deterministic.
    What: monkeypatches ``_get_db`` → our DB and ``_schedule_agent_build`` → no-op.
    Test: callers assert the DB-sourced payload returned by session.resume.
    """
    monkeypatch.setattr(srv, "_get_db", lambda: db)
    # The cold-resume path fires a daemon Timer that calls _make_agent (the LLM
    # boundary). Its result never feeds the RPC response — the response is built
    # from the DB read below — so silence it to keep the test free of background
    # agent-build noise. This is the LLM boundary the methodology permits mocking.
    monkeypatch.setattr(srv, "_schedule_agent_build", lambda *a, **k: None)


def _resume(srv, session_id, **params):
    """Why: invoke the REAL session.resume RPC synchronously through the same
    dispatch entrypoint the WS transport uses.
    What: calls handle_request with a JSON-RPC session.resume envelope.
    Test: returns the raw response dict for the caller to assert on.
    """
    req = {
        "jsonrpc": "2.0",
        "id": "resume-1",
        "method": "session.resume",
        "params": {"session_id": session_id, **params},
    }
    return srv.handle_request(req)


# ===========================================================================
# Hop A — session.info emit carries a CONCRETE session_id (the value the
# client reads from frame.params.session_id and persists to localStorage).
# ===========================================================================


def test_emit_session_info_frame_carries_session_id(srv):
    """Why: the client persists ``frame.params.session_id`` from the
    ``session.info`` event; if the emit wiring dropped the id, resume would
    never be primed. This locks the exact field/name contract.
    What: drives the REAL ``_emit`` → ``write_json`` path with a concrete sid and
    captures the serialized frame via a bound capture transport.
    Test: the captured event frame's ``params.session_id`` equals the emitted id
    and ``params.type`` == ``session.info``.
    """
    captured: list[dict] = []

    class _CaptureTransport:
        def write(self, obj: dict) -> bool:
            captured.append(obj)
            return True

    sid = "live-abc123"
    token = srv.bind_transport(_CaptureTransport())
    try:
        srv._emit("session.info", sid, {"model": "test-model", "cwd": "/tmp"})
    finally:
        srv.reset_transport(token)

    info_frames = [
        f
        for f in captured
        if f.get("method") == "event"
        and (f.get("params") or {}).get("type") == "session.info"
    ]
    assert info_frames, f"no session.info event frame captured; got: {captured}"
    frame = info_frames[0]
    # The concrete id must ride at params.session_id — the exact path
    # ChatSidebar.tsx reads and ChatPage.tsx persists.
    assert frame["params"]["session_id"] == sid
    assert frame["params"]["payload"]["model"] == "test-model"


# ===========================================================================
# Hop B — session.resume reattaches from the SQLite store on the dead-PTY path
# and REBUILDS the prior conversation history (not a fresh/empty session).
# ===========================================================================


def test_resume_reattaches_dead_pty_session_with_prior_history(srv, monkeypatch, db):
    """Why: the whole point of session-ID resume is that reconnecting to a dead
    PTY restores the earlier conversation from state.db. A resume that returned
    an empty/fresh session would silently lose history — the bug this guards.
    What: seeds a real multi-turn conversation, then calls the REAL
    ``session.resume`` RPC (unmocked DB) and asserts the returned payload maps to
    the SAME persisted id and carries the exact prior turns.
    Test: result ``resumed``/``session_key`` == seeded id; message_count == 4;
    the returned messages contain the exact seeded user/assistant content.
    """
    _use_db(srv, monkeypatch, db)

    target = "20260715_090000_resume_me"
    db.create_session(target, source="tui")
    turns = [
        ("user", "remember: the deploy key is in vault path kv/prod"),
        ("assistant", "Noted — kv/prod."),
        ("user", "what was the vault path?"),
        ("assistant", "kv/prod."),
    ]
    for role, content in turns:
        db.append_message(session_id=target, role=role, content=content)

    # Sanity: the history is genuinely durable before resume.
    assert len(db.get_messages_as_conversation(target)) == 4

    resp = _resume(srv, target)

    assert "error" not in resp, f"resume errored: {resp.get('error')}"
    result = resp["result"]

    # The reattached session corresponds to the SAME persisted id …
    assert result["resumed"] == target
    assert result["session_key"] == target
    # … the live handle is a concrete, non-empty session id (what the client
    # then re-persists) and is distinct from the persisted conversation key.
    assert isinstance(result["session_id"], str) and result["session_id"]

    # … and the PRIOR conversation is present, not a fresh/empty session.
    # _history_to_messages renders user/assistant text under the "text" key.
    assert result["message_count"] == 4, result
    contents = [m.get("text") for m in result["messages"] if isinstance(m, dict)]
    joined = "\n".join(c for c in contents if c)
    assert "kv/prod" in joined, result["messages"]
    assert "what was the vault path?" in joined, result["messages"]


def test_resume_missing_id_reports_not_found_without_crash(srv, monkeypatch, db):
    """Why: a stale/unknown resume id must fail cleanly (structured RPC error),
    never crash the gateway or fabricate a session.
    What: resumes an id that was never created in the real DB.
    Test: the RPC returns error code 4007 (session not found), no exception.
    """
    _use_db(srv, monkeypatch, db)

    resp = _resume(srv, "nonexistent-session-id-xyz")

    assert "result" not in resp, resp
    assert resp["error"]["code"] == 4007, resp


def test_resume_follows_compression_chain_to_live_tip(srv, monkeypatch, db):
    """Why: auto-compression ends a session and forks a continuation child, so a
    resume on the rotated-out parent id must reattach to the descendant that
    holds the post-compression turns — otherwise the reply generated after
    compression is missing (the "I came back and the reply isn't there" bug).
    What: seeds parent → child(continuation), puts the newest turns on the child,
    then resumes the PARENT id.
    Test: the resume resolves to the child tip (``resumed`` == child id) and the
    returned history contains the child's post-compression turn.
    """
    _use_db(srv, monkeypatch, db)

    parent = "20260715_100000_parent"
    child = "20260715_110000_child"
    db.create_session(parent, source="tui")
    db.append_message(session_id=parent, role="user", content="original question")
    db.append_message(session_id=parent, role="assistant", content="pre-compression answer")
    db.create_session(child, source="tui", parent_session_id=parent)
    db.append_message(session_id=child, role="user", content="follow-up after compression")
    db.append_message(session_id=child, role="assistant", content="post-compression answer")

    # Resolver must agree the parent's live tip is the child before we assert the
    # RPC follows it (guards against a schema change silently breaking the chain).
    assert db.resolve_resume_session_id(parent) == child

    resp = _resume(srv, parent)

    assert "error" not in resp, f"resume errored: {resp.get('error')}"
    result = resp["result"]
    # Resume on the parent id reattaches to the live continuation tip …
    assert result["resumed"] == child, result
    assert result["session_key"] == child, result
    # … and surfaces the post-compression turn that lives on the child.
    contents = [m.get("text") for m in result["messages"] if isinstance(m, dict)]
    joined = "\n".join(c for c in contents if c)
    assert "post-compression answer" in joined, result["messages"]
