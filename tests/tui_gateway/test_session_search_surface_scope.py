"""Persisted messaging source must not turn TUI/Desktop into a live gateway."""

from __future__ import annotations

import json
import time

import pytest

from agent.agent_runtime_helpers import invoke_tool
from gateway.session_context import gateway_context_active, reset_session_vars
from hermes_state import SessionDB
from tui_gateway import server


@pytest.mark.parametrize(
    ("persisted_source", "desktop_mode"),
    [("telegram", False), ("discord", True)],
)
def test_resume_messaging_row_keeps_agent_executor_global(
    tmp_path, monkeypatch, persisted_source, desktop_mode
):
    if desktop_mode:
        monkeypatch.setenv("HERMES_DESKTOP", "1")
    else:
        monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    reset_session_vars()

    db = SessionDB(tmp_path / "state.db")
    db.create_session("resumed", source=persisted_source)
    db.append_message("resumed", role="user", content="active turn")
    for session_id, source in (("telegram-peer", "telegram"), ("discord-peer", "discord")):
        db.create_session(session_id, source=source)
        db.append_message(session_id, role="user", content="resume surface needle")

    known_sessions = set(server._sessions)
    captured = {}

    class _Agent:
        def __init__(self, *, session_id, session_db, platform):
            self.session_id = session_id
            self.platform = platform
            self._session_db = session_db

        def _get_session_db_for_recall(self):
            return self._session_db

    def _make_agent(_sid, _key, *, session_id, session_db, platform_override, **_kwargs):
        captured["platform_override"] = platform_override
        return _Agent(
            session_id=session_id,
            session_db=session_db,
            platform=platform_override,
        )

    def _init_session(sid, key, agent, _history, *, source, **_kwargs):
        server._sessions[sid] = {
            "agent": agent,
            "session_key": key,
            "source": source,
            "cwd": str(tmp_path),
            "created_at": time.time(),
            "last_active": time.time(),
            "running": False,
        }

    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_find_live_session_by_key", lambda _key, _profile_home: None)
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_make_agent", _make_agent)
    monkeypatch.setattr(server, "_init_session", _init_session)
    monkeypatch.setattr(server, "_stored_session_runtime_overrides", lambda _row: {})
    monkeypatch.setattr(server, "_session_info", lambda agent, _session: {"model": "test"})
    monkeypatch.setattr(server, "_maybe_schedule_auto_continue", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_default_session_cwd", lambda *_a, **_k: str(tmp_path))

    try:
        response = server._methods["session.resume"](
            "resume-1",
            {
                "session_id": "resumed",
                "source": persisted_source,
                "eager_build": True,
                "omit_messages": True,
            },
        )
        assert "error" not in response, response
        live_id = response["result"]["session_id"]
        agent = server._sessions[live_id]["agent"]

        tokens = server._set_session_context("resumed", ui_session_id=live_id)
        try:
            assert captured["platform_override"] == persisted_source
            assert agent.platform == persisted_source
            assert gateway_context_active() is False
            result = json.loads(
                invoke_tool(
                    agent,
                    "session_search",
                    {"query": "resume surface needle", "limit": 10},
                    "resume-task",
                    pre_tool_block_checked=True,
                    skip_tool_request_middleware=True,
                    skip_tool_execution_middleware=True,
                )
            )
        finally:
            server._clear_session_context(tokens)

        assert result["success"] is True
        assert result["scope"] == "all"
        assert {row["session_id"] for row in result["results"]} == {
            "telegram-peer",
            "discord-peer",
        }
    finally:
        with server._sessions_lock:
            for session_id in set(server._sessions).difference(known_sessions):
                server._sessions.pop(session_id, None)
        db.close()
        reset_session_vars()
