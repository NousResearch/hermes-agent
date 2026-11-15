"""Every surface that rewinds a session's transcript notifies memory providers ``rewound=True``.

``/undo`` has always fired ``memory_manager.on_session_switch(..., rewound=True)`` so providers drop
per-turn caches (#6672, #21910). The three other surfaces that truncate the same transcript — the
``session.undo`` RPC the Desktop undo button calls, ``/retry`` (undo + resend) and a whole-tree
``rollback.restore`` — reached ``_rewind_active_session_history`` without that hook, so a provider that
keeps per-turn state answered from turns the user had just removed.

The hook lives in the rewind helper now, so the tests below drive it through the RPC surfaces (and once
directly, which is the lane ``rollback.restore`` takes) instead of asserting on a call site.
"""

from __future__ import annotations

import importlib
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home


@pytest.fixture()
def server(hermes_home):
    # Same import isolation as tests/tui_gateway/test_undo_command.py, and the same in-place restore.
    with patch.dict(
        "sys.modules",
        {
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
        },
    ):
        mod = importlib.import_module("tui_gateway.server")

    methods = dict(mod._methods)
    yield mod
    mod._methods.clear()
    mod._methods.update(methods)
    mod._sessions.clear()
    __import__("tui_gateway.server_requests", fromlist=["x"]).reset_for_tests()
    mod._db = None


@pytest.fixture()
def db(hermes_home):
    return SessionDB(db_path=hermes_home / "state.db")


def _seed(db: SessionDB, session_key: str, turns: int = 3) -> list[dict]:
    db.create_session(session_key, source="tui")
    for i in range(1, turns + 1):
        db.append_message(session_key, "user", f"question {i}")
        db.append_message(session_key, "assistant", f"answer {i}")
    return db.get_messages_as_conversation(session_key)


@pytest.fixture()
def live_session(server, db):
    """A live TUI session with three persisted turns and a recording memory manager."""
    sid, session_key = "sid-rewind", "tui-rewind-1"
    history = _seed(db, session_key)
    agent = MagicMock()
    agent._memory_manager = MagicMock()
    agent._last_flushed_db_idx = len(history)
    session = {
        "session_key": session_key,
        "history": list(history),
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "agent": agent,
        "attached_images": [],
        "cols": 120,
    }
    server._sessions[sid] = session
    server._db = db
    return sid, session_key, session, agent


def _rewind_calls(agent) -> list[tuple]:
    """``on_session_switch`` calls that asked for the rewound variant, as ``(args, kwargs)``."""
    return [(call.args, call.kwargs) for call in agent._memory_manager.on_session_switch.call_args_list
            if call.kwargs.get("rewound") is True]


def test_session_undo_rpc_notifies_memory_providers(server, live_session):
    """The Desktop/TUI undo button rewinds through the RPC, not ``/undo``."""
    sid, session_key, _session, agent = live_session
    resp = server._methods["session.undo"](1, {"session_id": sid})
    assert resp["result"]["removed"], resp

    calls = _rewind_calls(agent)
    assert len(calls) == 1, calls
    args, kwargs = calls[0]
    assert args == (session_key,)
    assert kwargs["reset"] is False and kwargs["parent_session_id"] == ""


def test_retry_notifies_memory_providers(server, live_session):
    """``/retry`` is undo + resend: the transcript loses a turn, so the caches must drop it too."""
    sid, session_key, _session, agent = live_session
    resp = server._methods["command.dispatch"](1, {"session_id": sid, "name": "retry", "arg": ""})
    assert resp["result"]["type"] == "send", resp

    calls = _rewind_calls(agent)
    assert len(calls) == 1, calls
    assert calls[0][0] == (session_key,)


def test_a_bare_rewind_notifies_memory_providers(server, db):
    """``rollback.restore`` rewinds by calling the helper directly: the hook rides the rewind itself."""
    session_key = "tui-rewind-2"
    history = _seed(db, session_key)
    agent = MagicMock()
    agent._memory_manager = MagicMock()
    session = {
        "session_key": session_key,
        "history": list(history),
        "history_lock": threading.Lock(),
        "history_version": 0,
        "agent": agent,
    }
    with session["history_lock"]:
        from tui_gateway.server import _rewind_active_session_history
        removed = _rewind_active_session_history(session, 2)[2]
    assert removed

    calls = _rewind_calls(agent)
    assert len(calls) == 1, calls
    assert calls[0][0] == (session_key,)


def test_a_session_without_a_key_or_agent_is_not_notified(server, db):
    """The warm-history-only lane (no durable row) and a not-yet-built agent stay silent, never raising."""
    history = _seed(db, "tui-rewind-3")
    from tui_gateway.server import _rewind_active_session_history

    without_agent = {"history": list(history), "history_lock": threading.Lock(), "history_version": 0}
    assert _rewind_active_session_history(without_agent, 2)[2]

    agent = MagicMock()
    agent._memory_manager = MagicMock()
    without_key = {"history": list(history), "history_lock": threading.Lock(), "history_version": 0, "agent": agent}
    assert _rewind_active_session_history(without_key, 2)[2]
    assert agent._memory_manager.on_session_switch.call_args_list == []
