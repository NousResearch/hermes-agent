"""TUI / Desktop owner of cache-edge compaction and offload: the ticker compacts an idle session at the end
of its keep-alive budget through the manual-compress choke point (automatic trigger), or offloads its old
tool output once the cache is over, and never touches a session whose turn is running."""

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


class _Edge:
    """The warmer surface the owners read: ``claim_edge`` answers *action* and records each claim."""

    def __init__(self, action="compact", on_claim=None):
        self.action, self.claims, self.on_claim = action, [], on_claim

    def claim_edge(self, *, releasing=False, now=None):
        self.claims.append(releasing)
        if self.on_claim is not None:
            self.on_claim()
        return self.action


class _Agent:
    session_id = "sid-1"

    def __init__(self, action="compact"):
        self.calls = []
        self._prompt_cache_warmer = _Edge(action)

    def _compress_context(self, messages, system_message, **kwargs):
        self.calls.append(kwargs)
        return [messages[0], messages[-1]], "prompt"


def _session(agent, *, running=False):
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(8)]
    return {"agent": agent, "history_lock": threading.Lock(), "history": history, "history_version": 3,
            "running": running, "session_key": "sid-1", "cwd": "."}


def test_idle_due_session_compacts_with_the_cache_expiry_trigger_and_running_one_is_skipped(server, monkeypatch):
    idle_agent, busy_agent = _Agent(), _Agent()
    idle, busy = _session(idle_agent), _session(busy_agent, running=True)
    monkeypatch.setattr(server, "_sessions", {"idle": idle, "busy": busy})

    for job in server._cache_expiry_tick():
        job.result(timeout=10)

    assert busy_agent.calls == [] and len(busy["history"]) == 8
    assert [(c["trigger"], c["force"]) for c in idle_agent.calls] == [("cache_expiry", False)]
    assert len(idle["history"]) == 2 and idle["history_version"] == 4
    assert idle["running"] is False  # the busy claim is released for the next prompt


def test_a_slow_summary_does_not_delay_another_due_session_and_is_never_run_twice(server, monkeypatch):
    """Each due session compacts in its own job: a summary that runs for minutes must not hold the
    next chat past its window, and a tick while it runs never claims that session again."""
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
    jobs = list(server._cache_expiry_tick())
    try:
        assert not slow_returned.is_set()  # the tick returned while the slow summary still runs
        assert slow_started.wait(timeout=10)
        assert fast_done.wait(timeout=10)  # started while the slow summary is still running
        slow["running"] = False  # even if the busy flag were missed, the in-flight sid is skipped
        jobs += list(server._cache_expiry_tick())
    finally:
        release.set()
        for job in jobs:
            job.result(timeout=10)
    assert len(slow_agent.calls) == 1
    assert len(slow["history"]) == 2 and len(fast["history"]) == 2


def test_a_job_queued_behind_busy_slots_never_compacts_a_chat_closed_meanwhile(server, monkeypatch):
    """The job holds the session dict while it waits for a summary slot; the user closes the chat
    (``_pop_session_by_id`` sets ``_closing``). The busy claim refuses it: nothing is summarized."""
    agent = _Agent()
    session = _session(agent)
    monkeypatch.setattr(server, "_sessions", {"s": session})
    gate = threading.Event()
    held = [server._cache_expiry_submit(gate.wait, 10) for _ in range(server._CACHE_EXPIRY_CONCURRENCY)]
    try:
        (job,) = server._cache_expiry_tick()
        with server._sessions_lock:  # what a close does as it unregisters the chat
            server._sessions.pop("s")
            session["_closing"] = True
    finally:
        gate.set()
    for blocker in held:
        blocker.result(timeout=10)
    job.result(timeout=10)
    assert agent.calls == [] and len(session["history"]) == 8 and session["running"] is False


def test_due_sessions_past_the_workers_wait_in_the_queue_not_on_threads(server, monkeypatch):
    """Kshitij #6: the ticker never starts a thread per due chat; at most the worker bound exist."""
    gate = threading.Event()

    class _Blocked(_Agent):
        def _compress_context(self, messages, system_message, **kwargs):
            assert gate.wait(timeout=10)
            return super()._compress_context(messages, system_message, **kwargs)

    monkeypatch.setattr(server, "_sessions", {f"s{i}": _session(_Blocked()) for i in range(8)})
    before = {t.ident for t in threading.enumerate()}
    try:
        jobs = server._cache_expiry_tick()
        assert len(jobs) == 8
        started = [t for t in threading.enumerate() if t.ident not in before]
        assert len(started) <= server._CACHE_EXPIRY_CONCURRENCY
    finally:
        gate.set()
    for job in jobs:
        job.result(timeout=10)


def test_a_chat_past_its_window_is_offloaded_by_the_ticker(server, monkeypatch):
    """Ruling F: no compaction due, the cache is over: the owner offloads old tool output right then."""
    import agent.turn_proactive_prune as prune

    monkeypatch.setattr(prune, "offload_cold_history",
                        lambda agent, history, task_id=None: [*history[:-1], {"role": "assistant", "content": "stub"}])
    agent = _Agent("offload")
    session = _session(agent)
    monkeypatch.setattr(server, "_sessions", {"s": session})
    for job in server._cache_expiry_tick():
        job.result(timeout=10)
    assert agent.calls == []  # no summary
    assert session["history"][-1]["content"] == "stub" and session["history_version"] == 4
    assert session["running"] is False


def test_the_idle_reaper_compacts_a_due_chat_before_it_closes_it(server, monkeypatch):
    """A budget ends at its keep-alives or at the release: the TTL reaper's close of a chat whose step 2 is
    due compacts it first (nothing is warmed after a release), then closes it."""
    order = []
    monkeypatch.setattr(server, "_close_session_by_id", lambda sid, **kw: order.append(("closed", sid)))
    monkeypatch.setattr(server, "_session_is_reapable", lambda sid, session: True)
    agent = _Agent()
    agent._prompt_cache_warmer.on_claim = lambda: order.append("claimed")
    session = _session(agent)
    monkeypatch.setattr(server, "_sessions", {"s": session})

    job = server._compact_then_close("s", session, end_reason="idle_timeout")
    assert job is not None
    job.result(timeout=10)
    assert agent._prompt_cache_warmer.claims == [True]  # claimed as a release

    assert [c["trigger"] for c in agent.calls] == ["cache_expiry"] and len(session["history"]) == 2
    assert order == ["claimed", ("closed", "s")]


def test_the_session_cap_defers_only_the_excess_and_never_overshoots(server, monkeypatch):
    """Three reapable chats over a cap of 2, each due at release: one slow release-time compaction must
    count as the close the cap needs, so the cap loop stops there and closes nothing else."""
    release = threading.Event()
    monkeypatch.setattr(server, "_max_live_sessions", lambda: 2)
    monkeypatch.setattr(server, "_session_is_reapable", lambda sid, session: True)

    class _Slow(_Agent):
        def _compress_context(self, messages, system_message, **kwargs):
            assert release.wait(timeout=10)
            return super()._compress_context(messages, system_message, **kwargs)

    sessions = {}
    for i in range(3):
        session = _session(_Slow())
        session["last_active"] = float(i)
        sessions[f"s{i}"] = session
    monkeypatch.setattr(server, "_sessions", sessions)
    closed = []

    def close(sid, *, end_reason, predicate=None):
        with server._sessions_lock:
            current = server._sessions.get(sid)
            if current is None or (predicate is not None and not predicate(current)):
                return False
            server._sessions.pop(sid)
        closed.append(sid)
        return True
    monkeypatch.setattr(server, "_close_session_by_id", close)
    started, jobs = [], []
    real = server._compact_then_close

    def deferred(sid, s, **kw):
        started.append(sid)
        job = real(sid, s, **kw)
        jobs.append(job)
        return job
    monkeypatch.setattr(server, "_compact_then_close", deferred)

    server._enforce_session_cap()
    assert started == ["s0"] and closed == []  # the deferred close already covers the one excess chat
    release.set()
    for job in jobs:
        job.result(timeout=10)
    assert closed == ["s0"] and sorted(server._sessions) == ["s1", "s2"]


@pytest.mark.parametrize("state", ["running", "closing", "in_flight", "compute_host", "not_due"])
def test_cache_tick_checks_eligibility_before_claiming(server, monkeypatch, state):
    session = _session(_Agent())
    monkeypatch.setattr(server, "_sessions", {"s": session})
    monkeypatch.setattr(server, "_cache_expiry_in_flight", {"s"} if state == "in_flight" else set())
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda s: state == "compute_host")
    session["running"], session["_closing"] = state == "running", state == "closing"
    edge = session["agent"]._prompt_cache_warmer
    edge.action = None
    assert server._cache_expiry_tick() == []
    assert len(edge.claims) == (1 if state == "not_due" else 0)


def test_finalized_chat_refuses_the_compression_claim(server):
    from tui_gateway.session_compression import CompressionBusy

    session = _session(_Agent())
    session["_finalized"] = True
    with pytest.raises(CompressionBusy, match="session closed"):
        with server._manual_compress_turn("s", session):
            pytest.fail("a finalized chat was admitted")
    assert not session["running"]


def test_offload_preserves_a_history_edit_during_the_prune(server, monkeypatch):
    import agent.turn_proactive_prune as prune

    session = _session(_Agent())
    edited = [{"role": "user", "content": "new edit"}]

    def offload(agent, history, task_id):
        session["history"], session["history_version"] = edited, 4
        return [{"role": "assistant", "content": "stale offload"}]

    monkeypatch.setattr(prune, "offload_cold_history", offload)
    server._offload_expired_cache("s", session)
    assert session["history"] is edited and session["history_version"] == 4
    assert not session["running"]


def test_disabling_cap_while_a_release_is_queued_preserves_the_chat(server, monkeypatch):
    release, started = threading.Event(), threading.Event()

    class Slow(_Agent):
        def _compress_context(self, messages, system_message, **kwargs):
            started.set()
            assert release.wait(timeout=10)
            return super()._compress_context(messages, system_message, **kwargs)

    cap = [2]
    monkeypatch.setattr(server, "_max_live_sessions", lambda: cap[0])
    monkeypatch.setattr(server, "_session_is_reapable", lambda *a: True)
    monkeypatch.setattr(server, "_cache_expiry_releasing", set())
    monkeypatch.setattr(server, "_sessions", {f"s{i}": _session(Slow()) for i in range(3)})
    closed, jobs = [], []
    real = server._compact_then_close

    def defer(sid, session, **kw):
        job = real(sid, session, **kw)
        jobs.append(job)
        return job

    def close(sid, *, predicate=None, **kw):
        with server._sessions_lock:
            if predicate is not None and not predicate(server._sessions[sid]):
                return False
            closed.append(sid)
            server._sessions.pop(sid)
        return True

    monkeypatch.setattr(server, "_compact_then_close", defer)
    monkeypatch.setattr(server, "_close_session_by_id", close)
    try:
        server._enforce_session_cap()
        assert started.wait(timeout=10)
        cap[0] = 0
    finally:
        release.set()
        for job in jobs:
            if job is not None:
                job.result(timeout=10)
    assert not closed and len(server._sessions) == 3


def test_a_reaper_close_of_a_chat_queued_behind_busy_slots_waits_for_its_compaction(server, monkeypatch):
    """The ticker queued this chat's job behind three busy summaries; the idle reaper reaching it must
    not close it first (the queued job would see ``_closing`` and drop the compaction): it closes after."""
    order = []
    agent = _Agent()
    session = _session(agent)
    monkeypatch.setattr(server, "_sessions", {"s": session})
    monkeypatch.setattr(server, "_session_is_reapable", lambda sid, s: True)

    def close(sid, *, end_reason, predicate=None):
        with server._sessions_lock:
            current = server._sessions.get(sid)
            if current is None or (predicate is not None and not predicate(current)):
                return False
            server._sessions.pop(sid)
            current["_closing"] = True
        order.append(("closed", len(session["history"])))
        return True

    monkeypatch.setattr(server, "_close_session_by_id", close)
    gate = threading.Event()
    held = [server._cache_expiry_submit(gate.wait, 10) for _ in range(server._CACHE_EXPIRY_CONCURRENCY)]
    try:
        (queued,) = server._cache_expiry_tick()
        release = server._compact_then_close("s", session, end_reason="idle_timeout")
        assert release is not None and order == []  # the caller must not close it now
    finally:
        gate.set()
    for job in (*held, queued, release):
        job.result(timeout=10)
    assert [c["trigger"] for c in agent.calls] == ["cache_expiry"]
    assert order == [("closed", 2)]  # compacted first, then closed
    assert agent._prompt_cache_warmer.claims == [False]  # the queued claim stands; no second claim
