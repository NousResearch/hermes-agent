import threading
import types

import pytest

from tui_gateway import server


@pytest.fixture(autouse=True)
def _neuter_agent_prewarm_timer(request, monkeypatch):
    """Stub the deferred agent pre-warm timer for every test in this module.

    ``session.create`` and non-eager ``session.resume`` fire a 50 ms
    background ``threading.Timer`` (``_schedule_agent_build``) that calls
    whatever ``server._make_agent`` is patched in AT FIRE TIME. Left live,
    a timer armed by one test outlives it and lands in the NEXT test's
    ``_make_agent`` mock, racily corrupting its captured state (the
    ``'tip' == 'cont_tip'`` flakes in the session_resume tests). Tests that
    exercise the deferred build itself opt back in with
    ``@pytest.mark.real_agent_prewarm``.
    """
    if request.node.get_closest_marker("real_agent_prewarm"):
        yield
        return
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a, **k: None)
    yield


def _session(agent=None, **extra):
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        **extra,
    }


def test_config_set_model_defers_while_running(monkeypatch):
    """/model via config.set queues the pick during an in-flight turn instead
    of rejecting or racing the worker thread."""
    seen = {"called": False}

    def _fake_apply(sid, session, raw, **_kwargs):
        seen["called"] = True
        return {"value": raw, "warning": ""}

    monkeypatch.setattr(server, "_apply_model_switch", _fake_apply)

    server._sessions["sid"] = _session(running=True)
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {
                    "session_id": "sid",
                    "key": "model",
                    "value": "anthropic/claude-sonnet-4.6",
                },
            }
        )
        assert not resp.get("error")
        result = resp["result"]
        assert result["deferred"] is True
        assert result["value"] == "anthropic/claude-sonnet-4.6"
        assert not seen["called"], (
            "_apply_model_switch ran mid-turn — would race the worker thread "
            "reading agent.model / agent.client; it must defer to turn start"
        )
        pending = server._sessions["sid"].get("pending_model_switch")
        assert pending and pending["raw"] == "anthropic/claude-sonnet-4.6"
    finally:
        server._sessions.pop("sid", None)


def test_apply_pending_model_switch_runs_queued_pick(monkeypatch):
    """The queued pick is consumed once, on the turn thread, via
    _apply_model_switch — and cleared so it can't re-fire next turn."""
    calls = []

    def _fake_apply(sid, session, raw, **kwargs):
        calls.append(raw)
        return {"value": raw, "warning": "", "confirm_required": False}

    monkeypatch.setattr(server, "_apply_model_switch", _fake_apply)

    session = _session(running=False)
    session["agent"] = object()
    session["pending_model_switch"] = {
        "raw": "anthropic/claude-sonnet-4.6",
        "confirm_expensive_model": False,
    }

    server._apply_pending_model_switch("sid", session)
    assert calls == ["anthropic/claude-sonnet-4.6"]
    assert "pending_model_switch" not in session

    # Idempotent: a second turn start with nothing queued is a no-op.
    server._apply_pending_model_switch("sid", session)
    assert calls == ["anthropic/claude-sonnet-4.6"]


def test_config_set_model_allowed_when_idle(monkeypatch):
    """Regression guard: idle sessions can still switch models."""
    seen = {"called": False}

    def _fake_apply(sid, session, raw, **_kwargs):
        seen["called"] = True
        return {"value": "newmodel", "warning": ""}

    monkeypatch.setattr(server, "_apply_model_switch", _fake_apply)

    server._sessions["sid"] = _session(running=False)
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "config.set",
                "params": {"session_id": "sid", "key": "model", "value": "newmodel"},
            }
        )
        assert resp.get("result")
        assert resp["result"]["value"] == "newmodel"
        assert seen["called"]
    finally:
        server._sessions.pop("sid", None)
