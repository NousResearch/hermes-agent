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

from tests.fixtures.storage_read import conversation
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.gateway.fixtures.storage_handles import storage_handle


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
    return storage_handle(hermes_home)


def _seed(db, session_key: str, turns: int = 3) -> list[dict]:
    seed_session(db, session_key, "tui")
    for i in range(1, turns + 1):
        seed_messages(
            db,
            session_key,
            {"role": "user", "content": f"question {i}"},
            {"role": "assistant", "content": f"answer {i}"},
        )
    return conversation(db, session_key)


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


def test_a_session_without_a_memory_manager_does_not_enter_provider_scope(server, live_session):
    _sid, _key, _session, agent = live_session
    agent._memory_manager = None
    with patch.object(server, "_session_profile_runtime_scope", wraps=server._session_profile_runtime_scope) as scope:
        server._notify_memory_providers_rewound(_session, _key)
    scope.assert_not_called()


@pytest.mark.parametrize("command", ["session.undo", "undo", "retry"])
def test_rewind_hook_failure_keeps_the_committed_result(server, live_session, db, command):
    sid, session_key, session, agent = live_session
    agent._memory_manager.on_session_switch.side_effect = RuntimeError("provider unavailable")
    if command == "session.undo":
        resp = server._methods[command](1, {"session_id": sid})
    else:
        resp = server._methods["command.dispatch"](1, {"session_id": sid, "name": command, "arg": ""})
    assert "result" in resp, resp
    assert len(session["history"]) == 4
    assert len(conversation(db, session_key)) == 4
    if command == "undo":
        agent._invalidate_system_prompt.assert_called_once_with()
        assert agent._last_flushed_db_idx == 4
    assert len(_rewind_calls(agent)) == 1


def test_rewind_notifies_only_the_owning_profiles_real_provider(server, live_session, hermes_home, tmp_path):
    from agent.memory_manager import MemoryManager
    from agent.memory_provider import MemoryProvider
    from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override

    class RecordingProvider(MemoryProvider):
        name = "recording"

        def __init__(self):
            self.calls = []

        def is_available(self):
            return True

        def initialize(self, session_id, **kwargs):
            pass

        def get_tool_schemas(self):
            return []

        def on_session_switch(self, new_session_id, **kwargs):
            self.calls.append((new_session_id, str(get_hermes_home()), kwargs))

    sid, session_key, session, agent = live_session
    session["profile_home"] = str(hermes_home)
    foreign_home = tmp_path / "foreign"
    foreign_home.mkdir()
    own, foreign = MemoryManager(), MemoryManager()
    own_provider, foreign_provider = RecordingProvider(), RecordingProvider()
    for manager, provider, home in ((own, own_provider, hermes_home), (foreign, foreign_provider, foreign_home)):
        with server._session_profile_runtime_scope({"profile_home": str(home)}):
            manager.add_provider(provider)
            manager.initialize_all(session_id=session_key, hermes_home=str(home))
    agent._memory_manager = own
    token = set_hermes_home_override(str(foreign_home))
    try:
        resp = server._methods["session.undo"](1, {"session_id": sid})
        assert str(get_hermes_home()) == str(foreign_home)
    finally:
        reset_hermes_home_override(token)
    assert "result" in resp, resp
    assert own_provider.calls == [(session_key, str(hermes_home), {
        "parent_session_id": "", "reset": False, "rewound": True})]
    assert foreign_provider.calls == []
