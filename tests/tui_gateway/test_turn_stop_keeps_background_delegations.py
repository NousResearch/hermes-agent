"""A turn stop or WS-orphan reap ends the turn only; background delegations keep running (3ae67dc3d6)."""
import threading
import types

import pytest

from agent import interrupt_compat
from tools import async_delegation
from tui_gateway import server


@pytest.mark.parametrize("kwargs", [{}, {"orphan": True, "request_id": "client-gone-sid"}])
def test_turn_stop_leaves_background_delegations(monkeypatch, kwargs):
    calls = []
    monkeypatch.setattr(async_delegation, "interrupt_for_session", lambda **kw: calls.append(kw) or 0)
    monkeypatch.setattr(interrupt_compat, "request_hard_interrupt", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *_a, **_k: False)
    monkeypatch.setattr(server, "_clear_pending", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_broadcast_global_event", lambda *_a, **_k: None)
    session = {"_sid": "sid", "session_key": "k", "history": [], "history_lock": threading.Lock(),
               "running": True, "queued_prompt": None,
               "agent": types.SimpleNamespace(session_id="parent", interrupt=lambda *_a, **_k: None)}

    server._interrupt_session_turn("sid", session, **kwargs)

    assert calls == []
