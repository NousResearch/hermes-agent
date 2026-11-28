"""A model pick that arrives while a fresh session's agent is still building must not be lost.

``session.create`` schedules the agent build ~50 ms later and returns at once. A picker click inside
that window used to reach ``config.set model`` with ``session["agent"] is None``: because the pick named
a provider (the Desktop always sends ``--provider``), the handler skipped waiting for the build and only
wrote ``session["model_override"]``. The build had already captured its kwargs, so it finished on the old
model, announced it in ``session.info``, and the first turn ran on it too - the pill never moved.

The handler must wait for an in-flight build, then switch the built agent like any live pick.
"""

from __future__ import annotations

import threading
import time

from tui_gateway import server


def _building_session():
    return dict(
        agent=None, agent_error=None, agent_ready=threading.Event(), agent_build_started=True,
        session_key="build-race-key", history=[], history_version=0, history_lock=threading.Lock(),
        running=False, attached_images=[], image_counter=0, cols=80, slash_worker=None,
        show_reasoning=False, tool_progress_mode="all", inflight_turn=None,
        created_at=time.time(), last_active=time.time())


def test_pick_with_provider_waits_for_inflight_agent_build(monkeypatch):
    session = _building_session()
    server._sessions["sid-build-race"] = session
    order = []

    def _await(sess, rid):
        order.append("await")
        sess["agent"] = object()  # the build lands while the handler waits
        sess["agent_ready"].set()
        return None

    def _apply(sid, sess, raw, **kw):
        order.append(("apply", sess["agent"] is not None))
        return {"value": "haiku", "warning": "", "confirm_required": False, "scope": "session"}

    monkeypatch.setattr(server, "_start_agent_build", lambda *a, **k: None)
    # Handlers are rebound onto the server module's globals, so patch the names there.
    monkeypatch.setattr(server, "_cfgset_await_agent", _await)
    monkeypatch.setattr(server, "_apply_model_switch", _apply)
    try:
        resp = server.handle_request(
            {"id": "1", "method": "config.set",
             "params": {"session_id": "sid-build-race", "key": "model",
                        "value": "claude-haiku-4-5 --provider anthropic --session"}})
        assert resp["result"]["value"] == "haiku"
        # The switch must run against the BUILT agent, never against agent=None (override-only).
        assert order == ["await", ("apply", True)]
    finally:
        server._sessions.pop("sid-build-race", None)
