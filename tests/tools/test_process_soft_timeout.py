"""Soft-timeout gate for background tasks (#110427).

Contract under test:
  * crossing the threshold NEVER kills — it queues a notice and leaves the process
    running; an undecided task survives any number of thresholds;
  * ONLY the agent's explicit decision ends it (`process action='kill'`), or continues
    it (`process action='continue'`, optionally with a fresh budget);
  * every further crossing re-notifies (no silent loop);
  * the notice rides the existing completion_queue path, so the CLI/gateway wake the
    agent with it;
  * the threshold is configurable and defaults to the budget long runs already had.
"""

import json
import threading
import time

import pytest

from tools.process_registry import ProcessRegistry, ProcessSession
from tools.process_registry_notifications import format_process_notification
from tools.process_registry_soft_timeout import DEFAULT_SOFT_TIMEOUT_SECONDS, soft_timeout_seconds

T0 = 1_000_000.0  # fixed clock base so every window assertion is exact


@pytest.fixture()
def registry():
    return ProcessRegistry()


def _task(sid="proc_soft1", *, command="pytest -q", started_at=T0, notify=True) -> ProcessSession:
    """A running background TASK (the sessions the soft gate covers)."""
    return ProcessSession(
        id=sid, command=command, task_id="t1", session_key="cli:soft", started_at=started_at,
        notify_on_complete=notify,
    )


def _register(reg, session) -> ProcessSession:
    reg._running[session.id] = session
    return session


def _events(reg) -> list:
    out = []
    while not reg.completion_queue.empty():
        out.append(reg.completion_queue.get_nowait())
    return out


def _soft_events(reg) -> list:
    return [e for e in _events(reg) if e.get("type") == "soft_timeout"]


def _no_kill(monkeypatch, reg) -> None:
    """Make any kill attempt fail loudly — the gate must never kill."""
    def _boom(*_a, **_k):
        pytest.fail("soft timeout must never kill a process")

    monkeypatch.setattr(reg, "kill_process", _boom)


# ── the gate ─────────────────────────────────────────────────────────────────

class TestSoftThresholdGate:
    def test_under_threshold_is_silent(self, registry, monkeypatch):
        _no_kill(monkeypatch, registry)
        sess = _register(registry, _task())
        registry.arm_soft_timeout(sess, budget=60, now=T0)

        assert registry.check_soft_timeouts(now=T0 + 59.0) == 0
        assert _soft_events(registry) == []
        assert sess.exited is False

    def test_threshold_notifies_without_killing(self, registry, monkeypatch):
        _no_kill(monkeypatch, registry)
        sess = _register(registry, _task())
        registry.arm_soft_timeout(sess, budget=60, now=T0)

        assert registry.check_soft_timeouts(now=T0 + 61.0) == 1
        evt, = _soft_events(registry)
        assert evt["type"] == "soft_timeout"
        assert evt["session_id"] == sess.id
        assert evt["hit"] == 1
        assert evt["budget_seconds"] == 60
        assert evt["elapsed_seconds"] == 61
        # Not killed, still running, and armed for the NEXT window (re-notify on crossing).
        assert sess.exited is False
        assert sess.soft_deadline == pytest.approx(T0 + 61.0 + 60)

    def test_undecided_task_survives_many_thresholds(self, registry, monkeypatch):
        _no_kill(monkeypatch, registry)
        sess = _register(registry, _task())
        registry.arm_soft_timeout(sess, budget=60, now=T0)

        for window in range(1, 6):
            now = T0 + 60 * window + 1.0
            assert registry.check_soft_timeouts(now=now) == 1
            assert sess.exited is False, f"killed at window {window} without a decision"
        assert [e["hit"] for e in _soft_events(registry)] == [1, 2, 3, 4, 5]

    def test_servers_and_watchers_are_not_gated(self, registry, monkeypatch):
        """A long-lived server spawns without notify_on_complete — no nagging."""
        _no_kill(monkeypatch, registry)
        sess = _register(registry, _task(sid="proc_srv", notify=False))
        registry.arm_soft_timeout(sess, budget=60, now=T0)

        assert registry.check_soft_timeouts(now=T0 + 6000) == 0
        assert _soft_events(registry) == []

    def test_exited_task_is_never_notified(self, registry):
        sess = _register(registry, _task())
        registry.arm_soft_timeout(sess, budget=60, now=T0)
        sess.mark_exited(0)

        assert registry.check_soft_timeouts(now=T0 + 600) == 0

    def test_zero_budget_disables_the_gate(self, registry, monkeypatch):
        monkeypatch.setenv("HERMES_PROCESS_SOFT_TIMEOUT", "0")
        assert soft_timeout_seconds() == 0.0
        sess = _register(registry, _task())
        registry.arm_soft_timeout(sess)

        assert sess.soft_deadline == 0.0
        assert registry.check_soft_timeouts(now=T0 + 1e6) == 0

    def test_configured_threshold_wins(self, registry, monkeypatch):
        monkeypatch.setenv("HERMES_PROCESS_SOFT_TIMEOUT", "42")
        assert soft_timeout_seconds() == 42.0
        sess = _register(registry, _task())
        registry.arm_soft_timeout(sess)

        assert registry.check_soft_timeouts(now=T0 + 41) == 0
        assert registry.check_soft_timeouts(now=T0 + 43) == 1

    def test_default_budget_is_the_budget_long_runs_already_had(self):
        """Default = the foreground cap after which a long run is promoted to background,
        so no task runs under a shorter budget than before."""
        from tools.terminal_tool import FOREGROUND_MAX_TIMEOUT

        assert DEFAULT_SOFT_TIMEOUT_SECONDS == float(FOREGROUND_MAX_TIMEOUT)


# ── the agent's decisions ────────────────────────────────────────────────────

class TestAgentDecision:
    def _handler(self, monkeypatch, reg):
        from tools import process_registry as pr

        monkeypatch.setattr(pr, "process_registry", reg)
        return pr._handle_process

    def test_explicit_kill_is_what_ends_it(self, registry, monkeypatch):
        sess = _register(registry, _task())
        registry.arm_soft_timeout(sess, budget=60, now=T0)
        registry.check_soft_timeouts(now=T0 + 61)  # notice delivered
        assert sess.exited is False, "the gate killed without a decision"

        monkeypatch.setattr(registry, "_signal_kill", lambda *a, **k: None)  # no real handle
        out = json.loads(self._handler(monkeypatch, registry)({"action": "kill", "session_id": sess.id}))

        assert out["status"] == "killed"
        assert sess.exited is True
        assert sess.completion_reason == "killed"

    def test_continue_action_gives_a_fresh_budget(self, registry, monkeypatch):
        _no_kill(monkeypatch, registry)
        sess = _register(registry, _task())
        registry.arm_soft_timeout(sess, budget=60, now=T0)
        registry.check_soft_timeouts(now=T0 + 61)
        _soft_events(registry)

        out = json.loads(self._handler(monkeypatch, registry)(
            {"action": "continue", "session_id": sess.id, "seconds": 300}))

        assert out["status"] == "continuing"
        assert out["soft_timeout_seconds"] == 300
        assert sess.exited is False
        # `continue` opens the fresh window from NOW (not from the spawn clock).
        after = time.time()
        assert registry.check_soft_timeouts(now=after + 299) == 0
        assert registry.check_soft_timeouts(now=after + 301) == 1
        assert _soft_events(registry)[0]["budget_seconds"] == 300

    def test_continue_without_seconds_reuses_the_budget(self, registry, monkeypatch):
        _no_kill(monkeypatch, registry)
        sess = _register(registry, _task())
        registry.arm_soft_timeout(sess, budget=90, now=T0)

        out = json.loads(self._handler(monkeypatch, registry)(
            {"action": "continue", "session_id": sess.id}))

        assert out["soft_timeout_seconds"] == 90

    def test_continue_on_exited_process_reports_instead_of_killing(self, registry):
        sess = _register(registry, _task())
        sess.mark_exited(0)

        out = registry.continue_soft_timeout(sess.id, 60)

        assert out["status"] == "exited" and "error" in out

    def test_continue_unknown_process_reports_not_found(self, registry):
        assert registry.continue_soft_timeout("proc_nope", 60)["status"] == "not_found"

    def test_continue_rejects_nonpositive_budget(self, registry, monkeypatch):
        monkeypatch.setenv("HERMES_PROCESS_SOFT_TIMEOUT", "0")
        sess = _register(registry, _task())

        assert registry.continue_soft_timeout(sess.id, 0)["status"] == "error"

    def test_schema_offers_the_continue_decision(self):
        from tools.process_registry import PROCESS_SCHEMA

        props = PROCESS_SCHEMA["parameters"]["properties"]
        assert "continue" in props["action"]["enum"]
        assert "seconds" in props


# ── delivery through the existing notification path ──────────────────────────

class TestNoticeDelivery:
    def _notice(self, registry, monkeypatch):
        _no_kill(monkeypatch, registry)
        sess = _register(registry, _task())
        registry.arm_soft_timeout(sess, budget=60, now=T0)
        registry.check_soft_timeouts(now=T0 + 61)
        return _soft_events(registry)[0]

    def test_notice_text_carries_both_decisions(self, registry, monkeypatch):
        evt = self._notice(registry, monkeypatch)

        text = format_process_notification(evt)

        assert text.startswith("[IMPORTANT:")
        assert evt["session_id"] in text
        assert "STILL RUNNING" in text and "nothing was killed" in text
        assert "action='continue'" in text and "action='kill'" in text
        assert "Command: pytest -q" in text

    def test_cli_drain_delivers_the_notice(self, registry, monkeypatch):
        evt = self._notice(registry, monkeypatch)
        registry.completion_queue.put(evt)

        drained = registry.drain_notifications(session_key="cli:soft")

        assert [e["type"] for e, _ in drained] == ["soft_timeout"]
        assert "soft timeout" in drained[0][1]

    def test_gateway_routes_the_notice_to_the_agent(self):
        """The gateway's watch drain is what wakes the agent mid-turn — a soft-timeout
        event must survive it (events outside that set are dropped on the floor)."""
        from gateway.run import _drain_gateway_watch_events

        import queue
        q = queue.Queue()
        evt = {"type": "soft_timeout", "session_id": "proc_x", "session_key": "k", "command": "make"}
        q.put(evt)

        assert _drain_gateway_watch_events(q) == [evt]

    def test_tui_keeps_repeat_notices_distinct(self):
        """Each crossing must be visible: the dedup key carries the hit index."""
        from tui_gateway.session_notifications import _notification_event_dedup_key

        base = {"type": "soft_timeout", "session_id": "proc_x", "command": "make", "budget_seconds": 60}

        assert _notification_event_dedup_key({**base, "hit": 1}) != \
            _notification_event_dedup_key({**base, "hit": 2})


# ── real process end-to-end ──────────────────────────────────────────────────

def _wait_for_soft_events(reg, count=1, timeout=15.0) -> list:
    events, deadline = [], time.monotonic() + timeout
    while time.monotonic() < deadline and len(events) < count:
        events.extend(_soft_events(reg))
        if len(events) < count:
            time.sleep(0.05)
    return events


def test_real_process_is_notified_then_killed_only_by_decision(registry, monkeypatch):
    """A real long-running process: the ticker notifies (repeatedly), the process stays
    alive, and only the agent's explicit kill ends it."""
    from tools import process_registry as pr

    monkeypatch.setenv("HERMES_PROCESS_SOFT_TIMEOUT", "0.5")
    session = registry.spawn_local("sleep 30", task_id="soft-e2e")
    session.notify_on_complete = True  # mirrors terminal_tool_background for a bounded task
    session.session_key = "cli:soft"
    monkeypatch.setattr(pr, "process_registry", registry)
    try:
        events = _wait_for_soft_events(registry, count=2)
        assert [e["hit"] for e in events[:2]] == [1, 2], f"expected two crossings, got {events}"
        assert session.exited is False
        assert session.process.poll() is None, "the soft threshold killed the process"

        out = json.loads(pr._handle_process({"action": "kill", "session_id": session.id}))

        assert out["status"] == "killed"
        assert session.exited is True and session.completion_reason == "killed"
        assert session.process.poll() is not None
    finally:
        if not session.exited:
            registry.kill_process(session.id)


def test_real_process_continues_past_the_threshold_when_the_agent_says_so(registry, monkeypatch):
    from tools import process_registry as pr

    monkeypatch.setenv("HERMES_PROCESS_SOFT_TIMEOUT", "0.5")
    session = registry.spawn_local("sleep 30", task_id="soft-e2e-continue")
    session.notify_on_complete = True
    monkeypatch.setattr(pr, "process_registry", registry)
    try:
        assert len(_wait_for_soft_events(registry, count=1)) == 1

        out = json.loads(pr._handle_process(
            {"action": "continue", "session_id": session.id, "seconds": 30}))

        assert out["status"] == "continuing"
        assert session.exited is False and session.process.poll() is None
        assert session.soft_deadline >= time.time() + 25  # fresh window, not the old one
        _soft_events(registry)  # whatever raced in before the decision
        time.sleep(0.6)  # past the OLD 0.5s window
        assert _soft_events(registry) == [], "continued task re-notified inside its new budget"
    finally:
        registry.kill_process(session.id)


def test_spawnless_registry_starts_no_ticker():
    """The ticker is spawned per registry and only on a spawn (no idle thread otherwise)."""
    reg = ProcessRegistry()

    assert reg._soft_ticker is None
    assert reg.check_soft_timeouts(now=T0 + 1e6) == 0
    assert reg._soft_ticker is None


# ── ticker robustness under concurrency (review follow-up) ───────────────────

class _ChurnSession:
    """A session record as spawn/reap leaves it: armed, with a numeric budget."""

    soft_armed = True
    soft_budget = 60.0


class _ChurnDict(dict):
    """`_running` as the tick sees it. `values()` iterates through a Python-level loop, so
    the iteration hands the GIL back between two values — the window spawn/reap lands in.
    Iterating the LIVE dict across that handoff is what makes CPython raise
    `RuntimeError: dictionary changed size during iteration` (a plain `list(d.values())`
    over a dict of these size is C-only and never yields, which is why a lock-less tick
    only dies when the loop has any work in it)."""

    def __init__(self, entered: threading.Event, mutated: threading.Event):
        super().__init__()
        self._entered, self._mutated = entered, mutated

    def values(self):
        for value in dict.values(self):  # CPython's dict iterator: version-checked
            self._entered.set()          # mid-iteration: let the churn thread run now
            self._mutated.wait(0.5)
            yield value


def test_tick_seconds_is_safe_while_another_thread_spawns_and_reaps(registry):
    """`_soft_tick_seconds` runs on the ticker thread while spawn/reap mutate `_running`
    from other threads (every real mutator holds `_lock`). Snapshotting the live dict
    without `_lock` raises `RuntimeError: dictionary changed size during iteration`, which
    takes the ticker thread down with it and silently disables the soft gate."""
    entered, mutated = threading.Event(), threading.Event()
    registry._running = _ChurnDict(entered, mutated)
    registry._running["proc_slow"] = _ChurnSession()
    added = []

    def _spawn_and_reap():
        if not entered.wait(5.0):
            return
        for n in range(1, 201):
            with registry._lock:  # spawn/reap never touch `_running` without it
                for _ in range(3):
                    key = f"proc_churn{n}_{len(added)}"
                    registry._running[key] = _ChurnSession()
                    added.append(key)
                registry._running.pop(added.pop(0))  # reap the oldest, net of two spawns
            mutated.set()
            if n == 1:
                time.sleep(0.01)  # only the first pass has to overlap the live iteration

    churn = threading.Thread(target=_spawn_and_reap, name="spawn-reap-churn", daemon=True)
    churn.start()
    try:
        tick = registry._soft_tick_seconds()
    finally:
        mutated.set()
        churn.join(10)

    assert mutated.is_set() and not churn.is_alive()
    assert isinstance(tick, float) and 0 < tick <= 5.0


def test_ticker_keeps_notifying_after_a_failing_tick(registry, monkeypatch):
    """A tick that raises must not end the ticker thread. The reported crash
    (`RuntimeError: dictionary changed size during iteration`) is raised by
    `_soft_tick_seconds`, which sat *outside* the loop's try/except — one bad tick killed
    the daemon and long tasks silently lost the gate until the next spawn restarted it."""
    from tools import process_registry_soft_timeout as st

    monkeypatch.setenv("HERMES_PROCESS_SOFT_TIMEOUT", "0.5")
    monkeypatch.setattr(st, "_MAX_TICK_SECONDS", 0.2)  # a recovery wait sized for the test
    monkeypatch.setattr(st, "_MIN_TICK_SECONDS", 0.05)
    real_tick = registry._soft_tick_seconds
    ticks, boomed = [], threading.Event()

    def _flaky_tick():
        ticks.append(1)
        if len(ticks) == 1:  # the one bad tick
            boomed.set()
            raise RuntimeError("dictionary changed size during iteration")
        return real_tick()

    monkeypatch.setattr(registry, "_soft_tick_seconds", _flaky_tick)
    session = registry.spawn_local("sleep 30", task_id="soft-tick-survives")
    session.notify_on_complete = True
    session.session_key = "cli:soft"
    ticker = registry._soft_ticker
    try:
        assert boomed.wait(10), "the ticker never ran a tick"
        # Give a failing tick the moment it needs to unwind the thread (it never does).
        deadline = time.monotonic() + 0.3
        while ticker.is_alive() and len(ticks) < 2 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ticker.is_alive(), "one failing tick killed the ticker thread"

        events = _wait_for_soft_events(registry, count=1)

        assert [e["hit"] for e in events] == [1], "the ticker stopped notifying after a failed tick"
        assert len(ticks) >= 2, "the loop never ticked again after the failure"
        assert session.exited is False and session.process.poll() is None
    finally:
        registry.kill_process(session.id)
