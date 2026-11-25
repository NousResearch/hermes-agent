"""/refine from a completed Desktop/TUI session. The in-memory agent cache may be absent
(or its history empty) on a session that was reattached from disk — refine must replay the
PERSISTED transcript and spawn the background review on the live agent, instead of falling
back to the isolated slash worker and answering "Nothing to refine yet" (#93918, #83455).

Covers tui_gateway/methods_slash.py `_format_live_refine_output`.
"""

from __future__ import annotations

import contextlib
import threading
from types import SimpleNamespace
from unittest.mock import patch

from persistence.conversation.messages import ConversationMessage
from tui_gateway import server
from tui_gateway.transport import StdioTransport


def _live_session(agent, *, session_key: str, history: list):
    return {
        "agent": agent, "session_key": session_key, "history": list(history),
        "history_lock": threading.Lock(), "running": False,
        "transport": StdioTransport(lambda: None, threading.Lock()),
        "cwd": "", "source": "desktop",
    }


class _Agent:
    valid_tool_names = {"memory", "skill_manage"}

    def __init__(self):
        self.spawned: list[dict] = []

    def _spawn_background_review(self, **kwargs):
        self.spawned.append(kwargs)


class _Messages:
    def read_conversation(self, key, request):
        assert key == "refine-key"
        assert request.include_lineage is True
        return (
            ConversationMessage(role="user", content="persisted question", db_persisted=False),
            ConversationMessage(role="assistant", content="persisted answer", db_persisted=False),
        )


@contextlib.contextmanager
def _fake_session_db(_session):
    yield object()


def _persistence_with(messages):
    """A stand-in for ``server._persistence_for`` yielding a handle whose transcript is ``messages``."""
    return lambda _db: contextlib.nullcontext(SimpleNamespace(messages=messages))


def test_refine_spawns_review_from_persisted_transcript():
    """A reattached session with an empty in-memory history still refines, from state.db."""
    sid = "refine-sid"
    agent = _Agent()
    session = _live_session(agent, session_key="refine-key", history=[])
    server._sessions[sid] = session

    try:
        with (
            patch.object(server, "_session_uses_compute_host", return_value=False),
            patch.object(server, "_session_db", _fake_session_db),
            patch.object(server, "_persistence_for", _persistence_with(_Messages())),
        ):
            out = server._live_slash_command_output(sid, session, "refine", "save the workflow")
    finally:
        server._sessions.pop(sid, None)

    assert "Reviewing this conversation" in out
    assert agent.spawned == [{
        "messages_snapshot": [
            {"role": "user", "content": "persisted question"},
            {"role": "assistant", "content": "persisted answer"},
        ],
        "review_memory": True,
        "review_skills": True,
        "focus": "save the workflow",
        # /refine is explicit: it must not be swallowed by the unattended-review gates.
        "explicit": True,
    }]


class _MessagesEmpty:
    def read_conversation(self, key, request):
        return ()


def test_refine_spawns_review_from_in_memory_history_mid_conversation():
    """A live mid-conversation session with no persisted rows yet falls back to the
    locked in-memory history (covers the fallback branch, not just the reattach path)."""
    sid = "refine-mid"
    agent = _Agent()
    history = [
        {"role": "user", "content": "in-flight question"},
        {"role": "assistant", "content": "in-flight answer"},
    ]
    session = _live_session(agent, session_key="mid-key", history=history)
    server._sessions[sid] = session

    try:
        with (
            patch.object(server, "_session_uses_compute_host", return_value=False),
            patch.object(server, "_session_db", _fake_session_db),
            patch.object(server, "_persistence_for", _persistence_with(_MessagesEmpty())),
        ):
            out = server._live_slash_command_output(sid, session, "refine", "")
    finally:
        server._sessions.pop(sid, None)

    assert "Reviewing this conversation" in out
    assert agent.spawned == [{
        "messages_snapshot": [
            {"role": "user", "content": "in-flight question"},
            {"role": "assistant", "content": "in-flight answer"},
        ],
        "review_memory": True,
        "review_skills": True,
        "focus": None,
        # /refine is explicit: it must not be swallowed by the unattended-review gates.
        "explicit": True,
    }]
