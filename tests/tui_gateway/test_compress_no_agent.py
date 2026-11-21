"""Manual /compress on a session with no agent handle yet must refuse cleanly (#145137).

A session restored from state.db has ``session["agent"] is None`` until its first turn. The compress
routes used to crash with ``setattr(None, ...)`` from the context-engine notification helpers, whose
error path then replaced the original failure, or return an empty "(no output)" string.
"""
import threading

import pytest

NO_AGENT = "No active agent -- send a message first."


def _agentless_session():
    return {
        "agent": None,
        "history_lock": threading.Lock(),
        "history": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
        "history_version": 1,
        "running": False,
        "session_key": "sess-no-agent",
    }


@pytest.mark.parametrize("committed", [True, False])
def test_finalize_notification_is_a_noop_without_an_agent(committed):
    from agent.conversation_compression import finalize_context_engine_compression_notification

    assert finalize_context_engine_compression_notification(None, committed=committed) is False


def test_compress_session_history_refuses_without_an_agent():
    from tui_gateway.server import _compress_session_history

    with pytest.raises(ValueError, match=NO_AGENT):
        _compress_session_history(_agentless_session())


def test_compress_slash_mirror_says_why_instead_of_returning_nothing():
    from tui_gateway import server

    session = _agentless_session()
    sid = "sid-no-agent-mirror"
    server._sessions[sid] = session
    try:
        output = server._mirror_slash_side_effects(sid, session, "/compress")
    finally:
        server._sessions.pop(sid, None)

    assert NO_AGENT in output
