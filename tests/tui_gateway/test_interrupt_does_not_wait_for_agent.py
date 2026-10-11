"""session.interrupt must not wait on, or start, the deferred agent build (#133262).

Stop needs the session RECORD (running flag, prompt queue, cancel flag), never the agent: a turn
parked on the build already honors ``_turn_cancel_requested``. Waiting on the build made Stop fail
with 5032 once the build had failed (a removed local-model provider) or outlived the 30s wait. The
Desktop deletes a conversation with session.interrupt -> session.close -> session.delete and gives
up on any interrupt error other than "session not found", so such a session could not be deleted.
"""

from __future__ import annotations

import threading
import types

import pytest

from tui_gateway import server

REMOVED_PROVIDER = "Unknown provider 'custom:llama-local'."


def record(sid: str, **state) -> dict:
    session = {
        "agent": None,
        "agent_ready": threading.Event(),
        "agent_error": None,
        "history": [],
        "history_lock": threading.RLock(),
        "running": False,
        "session_key": sid,
        **state,
    }
    server._sessions[sid] = session
    return session


@pytest.fixture
def sid(request):
    sid = f"stop-{request.node.name}"
    yield sid
    server._sessions.pop(sid, None)


def interrupt(sid: str) -> dict:
    return server._methods["session.interrupt"](1, {"session_id": sid})


def test_stop_succeeds_after_the_agent_build_failed(sid):
    """The reported case: the build failed, and Stop must still succeed so the session can be deleted."""
    session = record(sid, agent_build_started=True, agent_error=REMOVED_PROVIDER)
    session["agent_ready"].set()

    response = interrupt(sid)

    assert response.get("result", {}).get("status") == "interrupted", response


def test_stop_cancels_a_turn_parked_on_an_unfinished_build(sid):
    """The handler returns while the build event is still unset, and the parked turn sees the cancel."""
    session = record(sid, agent_build_started=True, running=True,
                     _run_thread=types.SimpleNamespace(is_alive=lambda: True))

    response = interrupt(sid)

    assert response.get("result", {}).get("status") == "interrupted", response
    assert session["_turn_cancel_requested"] is True
    assert not session["agent_ready"].is_set()


def test_stop_does_not_build_an_agent_for_a_resumed_session(sid, monkeypatch):
    """A resumed session has no agent yet; Stop must not build one (provider resolution, MCP servers)."""
    builds = []

    def failing_build(build_sid, session):
        builds.append(build_sid)
        session["agent_error"] = REMOVED_PROVIDER
        session["agent_ready"].set()

    monkeypatch.setattr(server, "_start_agent_build", failing_build)
    record(sid)

    response = interrupt(sid)

    assert response.get("result", {}).get("status") == "interrupted", response
    assert builds == []
