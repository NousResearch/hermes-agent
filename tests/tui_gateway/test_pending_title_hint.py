"""Regression tests for the ``pending_title`` -> ``session_title_hint``
threading in ``tui_gateway.server``.

A lazy session's intended title (``pending_title``) is known BEFORE the
session-DB row exists. ``_start_agent_build`` must thread it to
``_make_agent`` as ``session_title_hint`` (-> ``AIAgent.__init__`` ->
``init_agent``) so it reaches an external memory provider's
``initialize(session_title=...)`` without depending on session-DB write
ordering. The post-build assignment of ``agent._session_title_hint`` (the
pre-existing "Bot Chat" gate hint) must keep working unchanged.
"""

import threading
import types
import uuid

import pytest

from tui_gateway import server


def _isolate_build(monkeypatch, captured_kwargs, built):
    def fake_make_agent(_sid, _key, **kwargs):
        captured_kwargs.update(kwargs)
        built.set()
        # Real _make_agent returns an AIAgent whose __init__ (via
        # agent.agent_init.init_agent) sets ._session_title_hint = None as
        # a baseline before the tui_gateway post-build override below.
        return types.SimpleNamespace(model="test", _session_title_hint=None)

    monkeypatch.setattr(server, "_make_agent", fake_make_agent)
    monkeypatch.setattr("tui_gateway.entry.ensure_mcp_discovery_started", lambda: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_SlashWorker", lambda *args: None)
    monkeypatch.setattr(server, "_attach_worker", lambda *args: None)
    monkeypatch.setattr(server, "_config_model_target", lambda: ("", ""))
    monkeypatch.setattr(server, "_start_notification_poller", lambda *a, **k: None)
    monkeypatch.setattr(server, "_schedule_mcp_late_refresh", lambda *a, **k: None)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)


def test_pending_title_is_threaded_as_session_title_hint(monkeypatch):
    """``session.pending_title`` must reach ``_make_agent`` as
    ``session_title_hint`` and must also be re-applied to the built agent
    as ``agent._session_title_hint`` (the pre-existing Bot Chat gate hint).
    """
    captured_kwargs = {}
    built = threading.Event()
    _isolate_build(monkeypatch, captured_kwargs, built)

    ready = threading.Event()
    sid = f"test-sid-{uuid.uuid4().hex[:8]}"
    session = {
        "agent_ready": ready,
        "session_key": f"test-key-{uuid.uuid4().hex[:8]}",
        "pending_title": "Group: room-42",
    }

    server._sessions[sid] = session
    try:
        server._start_agent_build(sid, session)
        assert built.wait(timeout=15), "agent build thread never called _make_agent"
        assert ready.wait(timeout=5), "agent_ready never set after build"
    finally:
        server._sessions.pop(sid, None)

    assert captured_kwargs.get("session_title_hint") == "Group: room-42"
    built_agent = session.get("agent")
    assert built_agent is not None
    assert built_agent._session_title_hint == "Group: room-42"


def test_missing_pending_title_omits_session_title_hint_kwarg(monkeypatch):
    """No ``pending_title`` -> no ``session_title_hint`` kwarg at all (not
    an empty string), so ``init_agent`` falls back to the session-DB title
    lookup path unchanged.
    """
    captured_kwargs = {}
    built = threading.Event()
    _isolate_build(monkeypatch, captured_kwargs, built)

    ready = threading.Event()
    sid = f"test-sid-{uuid.uuid4().hex[:8]}"
    session = {
        "agent_ready": ready,
        "session_key": f"test-key-{uuid.uuid4().hex[:8]}",
    }

    server._sessions[sid] = session
    try:
        server._start_agent_build(sid, session)
        assert built.wait(timeout=15), "agent build thread never called _make_agent"
        assert ready.wait(timeout=5), "agent_ready never set after build"
    finally:
        server._sessions.pop(sid, None)

    assert "session_title_hint" not in captured_kwargs
    built_agent = session.get("agent")
    assert built_agent is not None
    assert built_agent._session_title_hint is None
