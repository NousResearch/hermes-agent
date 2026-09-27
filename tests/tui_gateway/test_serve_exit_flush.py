"""A killed ``hermes serve`` must not lose in-memory session transcripts.

Regression for #94724 (item 2, @ruangraung): a serve terminated mid-update
lost every un-flushed in-memory session — the next RPC failed with
"session-scoped RPC rejected: not in memory (detached/reaped runtime)" and no
store held the transcript. #95576 made serves survive *future* updates; this
covers the kill path itself:

* SIGTERM/SIGINT first flush in-memory sessions to state.db (bounded,
  best-effort, chained to the previously installed handler so uvicorn's
  graceful shutdown still runs).
* The idle-reaper tick piggybacks a periodic incremental flush so even a
  SIGKILL loses at most one flush interval.
"""

from __future__ import annotations

import os
import signal
import time

import pytest

from tui_gateway import server

class _FlushAgent:
    """Minimal agent exposing the real ``_persist_session`` flush contract."""

    def __init__(self, messages=None):
        self.session_id = "flush-agent"
        self.flush_calls: list[list] = []
        self._session_messages = (
            messages
            if messages is not None
            else [{"role": "user", "content": "unflushed turn"}]
        )

    def _persist_session(self, messages, conversation_history=None):
        self.flush_calls.append(list(messages))

@pytest.fixture
def registered_session():
    """Register a fake in-memory session; always deregister on exit."""
    registered: list[str] = []

    def _register(sid: str, agent, **extra):
        session = {"agent": agent, "session_key": sid, "running": False}
        session.update(extra)
        with server._sessions_lock:
            server._sessions[sid] = session
        registered.append(sid)
        return session

    yield _register

    with server._sessions_lock:
        for sid in registered:
            server._sessions.pop(sid, None)

def _restore_signal_state(prev_handlers):
    for signum, handler in prev_handlers.items():
        signal.signal(signum, handler)
    server._exit_flush_prev_handlers.clear()
    server._exit_flush_handlers_installed = False

@pytest.mark.platforms("posix")
def test_sigterm_flushes_populated_session_into_state_db(
    registered_session, tmp_path, monkeypatch
):
    """A populated in-memory session survives a SIGTERM into state.db."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "sess-sigterm-flush"
    db.create_session(sid, source="tui")

    class _DbAgent(_FlushAgent):
        def _persist_session(self, messages, conversation_history=None):
            super()._persist_session(messages, conversation_history)
            for msg in messages:
                if msg.get("_db_persisted"):
                    continue
                db.append_message(sid, msg["role"], msg["content"])
                msg["_db_persisted"] = True

    agent = _DbAgent(messages=[{"role": "user", "content": "survive the kill"}])
    registered_session(sid, agent)

    chained = {"called": False}

    def _prev_handler(signum, frame):
        chained["called"] = True

    prev = {signal.SIGTERM: signal.signal(signal.SIGTERM, _prev_handler)}
    try:
        assert server.install_exit_flush_signal_handlers() is True
        os.kill(os.getpid(), signal.SIGTERM)
        # The handler runs synchronously on the main thread at the next
        # bytecode boundary; poll briefly for robustness.
        deadline = time.monotonic() + 5.0
        while not chained["called"] and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        _restore_signal_state(prev)

    assert chained["called"], "previous SIGTERM handler must still be chained"
    assert agent.flush_calls, "SIGTERM must flush in-memory sessions"
    rows = db.get_messages(sid)
    assert any("survive the kill" in str(r.get("content", "")) for r in rows)
    db.close()

def test_ignored_sigint_leaves_terminal_commands_runnable():
    """SIGINT inherited as SIG_IGN (a server started as ``cmd &`` from a non-interactive shell) ends
    nothing, so it must not raise the one-way exit fence: the process lives on and every later
    terminal command would return 'host is exiting' rc 130."""
    from tools.environments.local import LocalEnvironment

    prev = {signal.SIGTERM: signal.getsignal(signal.SIGTERM),
            signal.SIGINT: signal.signal(signal.SIGINT, signal.SIG_IGN)}
    try:
        assert server.install_exit_flush_signal_handlers() is True
        os.kill(os.getpid(), signal.SIGINT)
        time.sleep(0.05)
    finally:
        _restore_signal_state(prev)
    env = LocalEnvironment(cwd=os.getcwd())
    try:
        assert env.execute("echo still-alive", timeout=30)["returncode"] == 0
    finally:
        env.cleanup()


def test_exit_flush_is_bounded(registered_session):
    """A hung persist must never block exit longer than the budget."""

    class _HangingAgent(_FlushAgent):
        def _persist_session(self, messages, conversation_history=None):
            time.sleep(5.0)

    registered_session("sess-hang", _HangingAgent())

    start = time.monotonic()
    server._flush_sessions_before_exit(budget_s=0.3)
    elapsed = time.monotonic() - start
    assert elapsed < 2.0, f"exit flush blocked {elapsed:.1f}s past its budget"

def test_shutdown_sessions_flushes_before_teardown(monkeypatch):
    """The atexit path persists transcripts BEFORE slow per-session teardown."""
    order: list[str] = []

    monkeypatch.setattr(
        server, "_release_gateway_wake_owner", lambda: None, raising=False
    )
    monkeypatch.setattr(
        server,
        "_flush_sessions_before_exit",
        lambda budget_s=None: order.append("flush") or 0,
    )
    monkeypatch.setattr(
        server,
        "_close_session_by_id",
        lambda sid, **kw: order.append(f"close:{sid}"),
    )
    with server._sessions_lock:
        server._sessions["sess-order"] = {"agent": None, "session_key": "sess-order"}
    try:
        server._shutdown_sessions()
    finally:
        with server._sessions_lock:
            server._sessions.pop("sess-order", None)

    assert order and order[0] == "flush"
    assert "close:sess-order" in order


def test_shutdown_mid_tool_kills_the_command_and_keeps_its_result(monkeypatch):
    """SIGTERM/EOF while a turn's foreground terminal command runs: the shutdown chokepoint must
    end the command's process group (it would outlive the gateway, reparented to init) and the
    turn's tool result must be in the transcript before per-session teardown persists it."""
    import threading

    import psutil

    from tools.environments.local import LocalEnvironment
    from tools.interrupt import set_interrupt

    env = LocalEnvironment(cwd=os.getcwd())
    messages = [{"role": "assistant", "tool_calls": [{"id": "call-1"}]}]

    def turn():
        out = env.execute("sleep 3518", timeout=600)
        time.sleep(0.2)  # the agent's post-tool bookkeeping before the result lands in history
        messages.append({"role": "tool", "tool_call_id": "call-1", "content": out["output"]})

    run_thread = threading.Thread(target=turn, daemon=True)

    class _Agent:
        _session_messages = messages

        def interrupt(self, message=None):  # the real agent fans this out to its tool threads
            set_interrupt(True, thread_id=run_thread.ident)

    run_thread.start()
    deadline = time.monotonic() + 20.0
    sleeper = None
    while sleeper is None and time.monotonic() < deadline:
        sleeper = next((p for p in psutil.Process().children(recursive=True)
                        if p.name() == "sleep" and "3518" in " ".join(p.cmdline())), None)
        time.sleep(0.05)
    assert sleeper is not None, "test setup: foreground sleep never started"

    at_teardown: list = []
    monkeypatch.setattr(server, "_release_gateway_wake_owner", lambda: None, raising=False)
    monkeypatch.setattr(server, "_flush_sessions_before_exit", lambda budget_s=None: 0)
    monkeypatch.setattr(server, "_close_session_by_id", lambda sid, **kw: at_teardown.append(list(messages)))
    # The join returns as soon as the turn ends; 0.5s is too tight for the kill + bookkeeping under -n 40.
    monkeypatch.setattr(server, "_EXIT_TURN_SETTLE_S", 10.0)
    session = {"agent": _Agent(), "session_key": "sess-mid-tool", "running": True,
               "_run_thread": run_thread, "history_lock": threading.RLock()}
    with server._sessions_lock:
        server._sessions["sess-mid-tool"] = session
    try:
        server._shutdown_sessions()
        _gone, alive = psutil.wait_procs([sleeper], timeout=15.0)
        assert not alive, "foreground command survived gateway shutdown"
        assert at_teardown and at_teardown[0][-1].get("tool_call_id") == "call-1", (
            f"teardown persisted a tool_call with no result: {at_teardown}")
    finally:
        with server._sessions_lock:
            server._sessions.pop("sess-mid-tool", None)
        if sleeper.is_running():
            sleeper.kill()
        set_interrupt(False, thread_id=run_thread.ident)
        env.cleanup()


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups + trap")
def test_sigterm_grace_hard_exit_kills_a_sigterm_ignoring_command(monkeypatch):
    """The SIGTERM path os._exit()s after a ~1s grace, while the graceful foreground kill runs after a
    flush of up to 5s and then waits 1s between TERM and KILL. The grace timer's exit must SIGKILL the
    tree itself, at once, or a command that ignores SIGTERM survives, reparented to init."""
    import threading

    import psutil

    from tools.environments.local import LocalEnvironment
    from tui_gateway import entry

    env = LocalEnvironment(cwd=os.getcwd())
    run_thread = threading.Thread(
        target=lambda: env.execute("trap '' TERM; sleep 3522", timeout=600), daemon=True)
    run_thread.start()
    deadline = time.monotonic() + 20.0
    sleeper = None
    while sleeper is None and time.monotonic() < deadline:
        sleeper = next((p for p in psutil.Process().children(recursive=True)
                        if p.name() == "sleep" and "3522" in " ".join(p.cmdline())), None)
        time.sleep(0.05)
    assert sleeper is not None, "test setup: foreground sleep never started"
    exits: list = []
    monkeypatch.setattr(entry.os, "_exit", exits.append)
    try:
        t0 = time.monotonic()
        entry._hard_exit()
        elapsed = time.monotonic() - t0
        _gone, alive = psutil.wait_procs([sleeper], timeout=5.0)
        assert not alive, "SIGTERM-ignoring foreground command survived the hard exit"
        assert exits == [0] and elapsed < 0.9, f"hard exit waited {elapsed:.2f}s (a TERM grace) first"
    finally:
        if sleeper.is_running():
            sleeper.kill()
        run_thread.join(5.0)
        env.cleanup()

def test_periodic_flush_respects_interval_with_fake_clock(
    registered_session, monkeypatch
):
    monkeypatch.setattr(server, "_INCREMENTAL_FLUSH_INTERVAL_S", 300.0)
    agent = _FlushAgent()
    registered_session("sess-interval", agent)

    assert server._flush_dirty_sessions(now=1_000.0) == 1
    assert len(agent.flush_calls) == 1

    # Within the interval: no re-flush.
    assert server._flush_dirty_sessions(now=1_000.0 + 299.0) == 0
    assert len(agent.flush_calls) == 1

    # Past the interval: flushes again — SIGKILL loses at most one interval.
    assert server._flush_dirty_sessions(now=1_000.0 + 301.0) == 1
    assert len(agent.flush_calls) == 2

def test_periodic_flush_skips_running_sessions(registered_session, monkeypatch):
    """Mid-turn sessions are the turn thread's to persist — never race them."""
    monkeypatch.setattr(server, "_INCREMENTAL_FLUSH_INTERVAL_S", 300.0)
    agent = _FlushAgent()
    registered_session("sess-running", agent, running=True)

    assert server._flush_dirty_sessions(now=1_000.0) == 0
    assert agent.flush_calls == []
