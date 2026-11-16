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
    monkeypatch.setattr(decision, "window_still_warm", lambda agent, **k: True)
    idle_agent, busy_agent = _Agent(), _Agent()
    idle, busy = _session(idle_agent), _session(busy_agent, running=True)
    monkeypatch.setattr(server, "_sessions", {"idle": idle, "busy": busy})

    for thread in server._cache_expiry_tick() or ():
        thread.join(timeout=10)

    assert busy_agent.calls == [] and len(busy["history"]) == 8
    assert [(c["trigger"], c["force"]) for c in idle_agent.calls] == [("cache_expiry", False)]
    assert len(idle["history"]) == 2 and idle["history_version"] == 4
    assert idle["running"] is False  # the busy claim is released for the next prompt


def test_a_slow_summary_does_not_delay_another_due_session_and_is_never_run_twice(server, monkeypatch):
    """Each due session compacts in its own job: a summary that runs for minutes must not hold the
    next chat past its window, and a tick while it runs never claims that session again."""
    import agent.cache_expiry_compaction as decision

    monkeypatch.setattr(decision, "claim_cache_expiry_compaction", lambda agent, **k: True)
    monkeypatch.setattr(decision, "window_still_warm", lambda agent, **k: True)
    release, slow_started, fast_done = threading.Event(), threading.Event(), threading.Event()
    slow_returned = threading.Event()

    class _Slow(_Agent):
        def _compress_context(self, messages, system_message, **kwargs):
            slow_started.set()
            try:
                assert release.wait(timeout=5)
                return super()._compress_context(messages, system_message, **kwargs)
            finally:
                slow_returned.set()

    class _Fast(_Agent):
        def _compress_context(self, messages, system_message, **kwargs):
            fast_done.set()
            return super()._compress_context(messages, system_message, **kwargs)

    slow_agent, fast_agent = _Slow(), _Fast()
    slow, fast = _session(slow_agent), _session(fast_agent)
    monkeypatch.setattr(server, "_sessions", {"slow": slow, "fast": fast})
    threads = list(server._cache_expiry_tick() or ())
    try:
        assert not slow_returned.is_set()  # the tick returned while the slow summary still runs
        assert slow_started.wait(timeout=10)
        assert fast_done.wait(timeout=10)  # started while the slow summary is still running
        slow["running"] = False  # even if the busy flag were missed, the in-flight sid is skipped
        threads += list(server._cache_expiry_tick() or ())
    finally:
        release.set()
        for thread in threads:
            thread.join(timeout=10)
    assert len(slow_agent.calls) == 1
    assert len(slow["history"]) == 2 and len(fast["history"]) == 2


def test_a_job_whose_window_closed_before_it_started_does_not_compact(server, monkeypatch):
    import agent.cache_expiry_compaction as decision

    monkeypatch.setattr(decision, "claim_cache_expiry_compaction", lambda agent, **k: True)
    monkeypatch.setattr(decision, "window_still_warm", lambda agent, **k: False)
    agent = _Agent()
    session = _session(agent)
    monkeypatch.setattr(server, "_sessions", {"s": session})
    for thread in server._cache_expiry_tick() or ():
        thread.join(timeout=10)
    assert agent.calls == [] and len(session["history"]) == 8 and session["running"] is False
