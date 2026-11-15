"""TUI / Desktop owner of cache-expiry compaction: the ticker compacts an idle session whose warm
cache window is about to close through the manual-compress choke point (automatic trigger), and
never touches a session whose turn is running."""

import threading

import pytest


@pytest.fixture
def server(monkeypatch):
    from tui_gateway import server as srv

    monkeypatch.setattr(srv, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(srv, "_get_usage", lambda agent: {})
    monkeypatch.setattr(srv, "_session_uses_compute_host", lambda session: False)
    monkeypatch.setattr(srv, "_require_transcript_owner", lambda session: None)
    monkeypatch.setattr(srv, "_drain_queued_prompt", lambda *a, **k: False)
    return srv


class _Agent:
    session_id = "sid-1"

    def __init__(self):
        self.calls = []

    def _compress_context(self, messages, system_message, **kwargs):
        self.calls.append(kwargs)
        return [messages[0], messages[-1]], "prompt"


def _session(agent, *, running=False):
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(8)]
    return {"agent": agent, "history_lock": threading.Lock(), "history": history, "history_version": 3,
            "running": running, "session_key": "sid-1", "cwd": "."}


def test_idle_due_session_compacts_with_the_cache_expiry_trigger_and_running_one_is_skipped(server, monkeypatch):
    import agent.cache_expiry_compaction as decision

    monkeypatch.setattr(decision, "claim_cache_expiry_compaction", lambda agent, **k: True)
    idle_agent, busy_agent = _Agent(), _Agent()
    idle, busy = _session(idle_agent), _session(busy_agent, running=True)
    monkeypatch.setattr(server, "_sessions", {"idle": idle, "busy": busy})

    server._cache_expiry_tick()

    assert busy_agent.calls == [] and len(busy["history"]) == 8
    assert [(c["trigger"], c["force"]) for c in idle_agent.calls] == [("cache_expiry", False)]
    assert len(idle["history"]) == 2 and idle["history_version"] == 4
    assert idle["running"] is False  # the busy claim is released for the next prompt
