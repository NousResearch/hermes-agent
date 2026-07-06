"""RPC contract tests for session.context_full."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

from tui_gateway import server


SECRET = "SECRET_TOKEN=rpc123"


class _MemoryStore:
    def format_for_system_prompt(self, namespace: str) -> str:
        return f"<{namespace}>remembered {namespace}</{namespace}>"


def _agent():
    return SimpleNamespace(
        model="openai/gpt-5.4",
        tools=[{"type": "function", "function": {"name": "terminal", "description": "run"}}],
        _cached_system_prompt="CACHED SYS BYTES",
        ephemeral_system_prompt=None,
        _memory_store=_MemoryStore(),
        _memory_enabled=True,
        _user_profile_enabled=True,
        context_compressor=SimpleNamespace(context_length=120_000, last_prompt_tokens=42_000),
    )


def _put_session(sid: str, *, agent, history=None):
    server._sessions[sid] = {
        "agent": agent,
        "history": list(history or []),
        "history_lock": threading.RLock(),
    }


def test_context_full_rpc_live_agent_returns_full_context_shape():
    sid = "ctx-full-live"
    _put_session(sid, agent=_agent(), history=[{"role": "user", "content": f"hello {SECRET}"}])
    try:
        resp = server._methods["session.context_full"]("rid-live", {"session_id": sid})
    finally:
        server._sessions.pop(sid, None)

    assert "error" not in resp
    result = resp["result"]
    assert result["available"] is True
    assert result["state"] == "ready"
    assert result["source"] == "reconstructed_base"
    assert result["raw_unredacted"] is True
    assert result["context_max"] == 120_000
    assert result["context_used"] == 42_000
    assert result["messages"][0]["role"] == "system"
    assert result["messages"][0]["content_text"] == "CACHED SYS BYTES"
    assert SECRET in json.dumps(result)
    assert {"slices", "messages", "exact_capture_available"} <= result.keys()


def test_context_full_rpc_no_agent_returns_available_false_context_full_not_categories():
    sid = "ctx-full-no-agent"
    _put_session(sid, agent=None)
    try:
        resp = server._methods["session.context_full"]("rid-empty", {"session_id": sid})
    finally:
        server._sessions.pop(sid, None)

    assert "error" not in resp
    result = resp["result"]
    assert result["available"] is False
    assert result["state"] == "agent_not_built"
    assert result["source"] == "reconstructed_base"
    assert result["raw_unredacted"] is True
    assert result["slices"] == []
    assert result["messages"] == []
    assert result["exact_capture_available"] is False
    assert result["context_max"] == 0
    assert result["context_used"] == 0
    assert "categories" not in result


def test_context_full_rpc_compute_raises_static_error_without_secret(monkeypatch):
    sid = "ctx-full-raises"

    def boom(agent, history):
        raise RuntimeError(f"leaky exception {SECRET} {history!r}")

    monkeypatch.setattr("agent.context_breakdown.compute_session_context_full", boom)
    _put_session(sid, agent=_agent(), history=[{"role": "user", "content": SECRET}])
    try:
        resp = server._methods["session.context_full"]("rid-err", {"session_id": sid})
    finally:
        server._sessions.pop(sid, None)

    assert resp["error"] == {"code": 5000, "message": "Could not compute full context"}
    assert SECRET not in json.dumps(resp)
