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
    import agent.prompt_cache_lifecycle as decision

    monkeypatch.setattr(decision, "claim_compaction", lambda agent, **k: True)
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
    import agent.prompt_cache_lifecycle as decision

    monkeypatch.setattr(decision, "claim_compaction", lambda agent, **k: True)
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


def test_a_job_queued_behind_busy_slots_never_compacts_a_chat_closed_meanwhile(server, monkeypatch):
    """The job holds the session dict while it waits for a summary slot; the user closes the chat
    (``_pop_session_by_id`` sets ``_closing``). The busy claim refuses it: nothing is summarized."""
    import agent.prompt_cache_lifecycle as decision

    monkeypatch.setattr(decision, "claim_compaction", lambda agent, **k: True)
    agent = _Agent()
    session = _session(agent)
    monkeypatch.setattr(server, "_sessions", {"s": session})
    held = [server._cache_expiry_slots.acquire(timeout=5) for _ in range(server._CACHE_EXPIRY_CONCURRENCY)]
    try:
        (thread,) = server._cache_expiry_tick()
        with server._sessions_lock:  # what a close does as it unregisters the chat
            server._sessions.pop("s")
            session["_closing"] = True
    finally:
        for _ in held:
            server._cache_expiry_slots.release()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert agent.calls == [] and len(session["history"]) == 8 and session["running"] is False


def test_a_chat_past_its_window_is_offloaded_by_the_ticker(server, monkeypatch):
    """Ruling F: no compaction due, the cache is over: the owner offloads old tool output right then."""
    import agent.prompt_cache_lifecycle as decision
    import agent.turn_proactive_prune as prune

    monkeypatch.setattr(decision, "claim_compaction", lambda agent, **k: False)
    monkeypatch.setattr(decision, "claim_offload", lambda agent, **k: True)
    monkeypatch.setattr(prune, "offload_cold_history",
                        lambda agent, history, task_id=None: [*history[:-1], {"role": "assistant", "content": "stub"}])
    agent = _Agent()
    session = _session(agent)
    monkeypatch.setattr(server, "_sessions", {"s": session})
    for thread in server._cache_expiry_tick() or ():
        thread.join(timeout=10)
    assert agent.calls == []  # no summary
    assert session["history"][-1]["content"] == "stub" and session["history_version"] == 4
    assert session["running"] is False


def test_the_idle_reaper_compacts_a_due_chat_before_it_closes_it(server, monkeypatch):
    """A budget ends at its keep-alives or at the release: the TTL reaper's close of a chat whose step 2 is
    due compacts it first (nothing is warmed after a release), then closes it."""
    import agent.prompt_cache_lifecycle as decision

    order = []
    monkeypatch.setattr(decision, "claim_compaction_at_release", lambda agent, **k: order.append("claimed") or True)
    monkeypatch.setattr(server, "_close_session_by_id", lambda sid, **kw: order.append(("closed", sid)))
    monkeypatch.setattr(server, "_session_is_reapable", lambda sid, session: True)
    agent = _Agent()
    session = _session(agent)
    monkeypatch.setattr(server, "_sessions", {"s": session})

    thread = server._compact_then_close("s", session, end_reason="idle_timeout")
    assert thread is not None
    thread.join(timeout=10)

    assert [c["trigger"] for c in agent.calls] == ["cache_expiry"] and len(session["history"]) == 2
    assert order == ["claimed", ("closed", "s")]


def test_the_session_cap_defers_only_the_excess_and_never_overshoots(server, monkeypatch):
    """Three reapable chats over a cap of 2, each due at release: one slow release-time compaction must
    count as the close the cap needs, so the cap loop stops there and closes nothing else."""
    import agent.prompt_cache_lifecycle as decision

    release = threading.Event()
    monkeypatch.setattr(decision, "claim_compaction_at_release", lambda agent, **k: True)
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
    started = []
    real = server._compact_then_close
    monkeypatch.setattr(server, "_compact_then_close",
                        lambda sid, s, **kw: started.append(sid) or real(sid, s, **kw))

    server._enforce_session_cap()
    assert started == ["s0"] and closed == []  # the deferred close already covers the one excess chat
    release.set()
    for thread in [t for t in threading.enumerate() if t.name.startswith("cache-expiry-release-")]:
        thread.join(timeout=10)
    assert closed == ["s0"] and sorted(server._sessions) == ["s1", "s2"]


@pytest.mark.parametrize("state", ["running", "closing", "in_flight", "compute_host", "not_due"])
def test_cache_tick_checks_eligibility_before_claiming(server, monkeypatch, state):
    import agent.prompt_cache_lifecycle as decision

    session = _session(_Agent())
    monkeypatch.setattr(server, "_sessions", {"s": session})
    monkeypatch.setattr(server, "_cache_expiry_in_flight", {"s"} if state == "in_flight" else set())
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda s: state == "compute_host")
    session["running"], session["_closing"] = state == "running", state == "closing"
    claims = []
    monkeypatch.setattr(decision, "claim_compaction", lambda a: claims.append(a) or False)
    monkeypatch.setattr(decision, "claim_offload", lambda a: False)
    assert server._cache_expiry_tick() == []
    assert len(claims) == (1 if state == "not_due" else 0)
    assert "_cache_edge_action" not in session


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
    import agent.prompt_cache_lifecycle as decision

    release, started = threading.Event(), threading.Event()

    class Slow(_Agent):
        def _compress_context(self, messages, system_message, **kwargs):
            started.set()
            assert release.wait(timeout=10)
            return super()._compress_context(messages, system_message, **kwargs)

    cap = [2]
    monkeypatch.setattr(decision, "claim_compaction_at_release", lambda a: True)
    monkeypatch.setattr(server, "_max_live_sessions", lambda: cap[0])
    monkeypatch.setattr(server, "_session_is_reapable", lambda *a: True)
    monkeypatch.setattr(server, "_cache_expiry_releasing", set())
    monkeypatch.setattr(server, "_sessions", {f"s{i}": _session(Slow()) for i in range(3)})
    closed, threads = [], []
    real = server._compact_then_close

    def defer(sid, session, **kw):
        thread = real(sid, session, **kw)
        threads.append(thread)
        return thread

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
        for thread in threads:
            thread.join(timeout=10)
    assert not closed and len(server._sessions) == 3


@pytest.mark.parametrize("state", ["idle", "throttled", "running", "queued", "short"])
def test_classic_cli_cache_idle_hook_preserves_input_priority(monkeypatch, state):
    import contextlib
    import queue
    from types import SimpleNamespace
    from hermes_cli.cli_session_mixin import CLISessionMixin
    import agent.prompt_cache_lifecycle as decision
    import agent.conversation_compression_manual as manual
    import agent.turn_proactive_prune as prune

    class CLI(CLISessionMixin):
        pass

    cli = CLI()
    cli.agent, cli._agent_running, cli._pending_input = object(), state == "running", queue.Queue()
    cli.conversation_history = [{"role": "user", "content": "question"}] * (3 if state == "short" else 8)
    cli.session_id, cli._last_cache_expiry_check = "s", 999 if state == "throttled" else 0
    cli._busy_command = lambda *a, **kw: contextlib.nullcontext()
    adopted, claims, offloaded = [], [], []
    cli._adopt_compress_result = adopted.append
    if state == "queued":
        cli._pending_input.put("next prompt")
    monkeypatch.setattr("hermes_cli.cli_session_mixin.time.time", lambda: 1000)
    monkeypatch.setattr(decision, "claim_compaction", lambda a, **kw: claims.append("compact") or True)
    result = SimpleNamespace(status="compressed", after_messages=[{"role": "assistant", "content": "summary"}])
    monkeypatch.setattr(manual, "compress_now", lambda *a, **kw: result)
    cli._maybe_compact_before_cache_expiry()
    assert claims == (["compact"] if state == "idle" else [])
    assert adopted == ([result] if state == "idle" else [])
    if state != "idle":
        return
    # Same idle loop twice: throttle, then test the expired-cache (no network) branch.
    cli._maybe_compact_before_cache_expiry()
    assert len(claims) == 1
    cli._last_cache_expiry_check = 0
    monkeypatch.setattr(decision, "claim_compaction", lambda a, **kw: False)
    monkeypatch.setattr(decision, "claim_offload", lambda a, **kw: True)
    monkeypatch.setattr(prune, "offload_cold_history", lambda a, h, sid: offloaded.append(sid) or result.after_messages)
    cli._maybe_compact_before_cache_expiry()
    assert offloaded == ["s"] and cli.conversation_history is result.after_messages
