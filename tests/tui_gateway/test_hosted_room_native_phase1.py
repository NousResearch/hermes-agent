"""Native Phase 1: an accepted turn's identity must survive its own lifecycle.

Every schedule drives the **real installed handlers** of an independently loaded gateway backend:
real `session.create`/`session.list`/`session.resume` against real temporary SQLite, real
active-session ownership, the real deferred-build thread and readiness wait, real
`_run_prompt_submit`/`_admit_prompt_turn`/run handoff, the real profile turn lock and the real
hosted terminal callback into the real room task store.

The only fake is the agent-construction seam (`_make_agent`), which stands in for model/MCP/config
discovery; `run_agent.AIAgent` is additionally poisoned so any path that tries to build a real
agent (and reach the network) fails loudly instead of running. Barriers only ever wrap a native
function and delegate to the original.

What this can prove: attempt A's waiter, interrupt and finalizer cannot dispatch or mutate a later
attempt B, and every accepted submit reaches exactly one hosted terminal receipt.
What it cannot prove: anything about a real model, a real Telegram/Desktop client, or a real
process death. Those remain separate gates.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import pytest

from tui_gateway.turn_marker import read_turn_marker


REPO = Path(__file__).resolve().parents[2]
WAIT = 10.0
ROOM_ID = "phase1-room"
PROFILE = "default"


# --------------------------------------------------------------------------- harness


class Barrier:
    """A named gate a test opens deliberately; the harness opens every one at teardown."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.reached = threading.Event()
        self.released = threading.Event()

    def wait_here(self) -> None:
        self.reached.set()
        assert self.released.wait(WAIT), f"barrier {self.name} was never released"

    def release(self) -> None:
        self.released.set()

    def await_reached(self) -> None:
        assert self.reached.wait(WAIT), f"barrier {self.name} was never reached"


class FakeAgent:
    """Stands in for the model agent. It never touches a model, a tool or the network."""

    def __init__(self, name: str, *, record_only_interrupt: bool = False) -> None:
        self.name = name
        self.session_id = f"agent-{name}"
        self.interrupts: list[Any] = []
        self.cleared = 0
        self.record_only_interrupt = record_only_interrupt
        self.invocations: list[str] = []
        self.turn_running = threading.Event()
        self.release_turn = threading.Event()
        self.reply = f"reply from {name}"

    def interrupt(self, message: Any = None) -> None:
        self.interrupts.append(message)
        if not self.record_only_interrupt:
            self.release_turn.set()

    def clear_interrupt(self) -> None:
        self.cleared += 1

    def run_conversation(self, *args, **kwargs):
        # Record what each agent was actually asked to run: a stale attempt dispatching its own
        # prompt onto a later attempt's agent has to be visible, not merely absent from a flag.
        self.invocations.append(repr(args[:1]) + repr(sorted(kwargs)))
        self.turn_running.set()
        assert self.release_turn.wait(WAIT), f"turn of {self.name} was never released"
        return {"final_response": self.reply}

    def ran_prompt(self, marker: str) -> bool:
        return any(marker in invocation for invocation in self.invocations)


class Harness:
    """Owns the loaded backend, its barriers and every thread the schedules start."""

    def __init__(self, srv, home: Path, monkeypatch) -> None:
        self.srv = srv
        self.home = home
        self.monkeypatch = monkeypatch
        self.events: list[tuple] = []
        self.barriers: list[Barrier] = []
        self.threads: list[threading.Thread] = []
        self.started: list[threading.Thread] = []
        self.failures: list[BaseException] = []
        self.agents: list[FakeAgent] = []
        self.build_gate = self.barrier("agent-build")
        self.next_agent: FakeAgent | None = None

    # -- barriers and threads ---------------------------------------------------
    def barrier(self, name: str) -> Barrier:
        gate = Barrier(name)
        self.barriers.append(gate)
        return gate

    def spawn(self, name: str, target: Callable[[], Any]) -> threading.Thread:
        def run() -> None:
            try:
                target()
            except BaseException as exc:  # recorded, never swallowed
                self.failures.append(exc)

        thread = threading.Thread(target=run, name=name, daemon=True)
        self.threads.append(thread)
        thread.start()
        return thread

    def wrap(self, attribute: str, before: Callable[[], Any]) -> None:
        """Run ``before`` and then delegate to the untouched native implementation."""
        original = getattr(self.srv, attribute)

        def wrapper(*args, **kwargs):
            before()
            return original(*args, **kwargs)

        self.monkeypatch.setattr(self.srv, attribute, wrapper)

    # -- native session lifecycle ----------------------------------------------
    def make_agent(self, name: str, *, record_only_interrupt: bool = False) -> FakeAgent:
        agent = FakeAgent(name, record_only_interrupt=record_only_interrupt)
        self.agents.append(agent)
        self.next_agent = agent
        return agent

    def create_session(self, title: str, *, persist: bool = True) -> str:
        """Create a room session through the installed handler.

        Production persists the durable row on first real activity (``prompt.submit`` →
        ``_persist_session_row_for_submit`` → ``_ensure_session_db_row``), so a freshly created
        session is not yet discoverable by title. The same native persistence entry point is used
        here, never a hand-written row.
        """
        result = self.srv._methods["session.create"]("rid-create", {
            "profile": PROFILE, "title": title, "source": "bot_room", "hidden": True,
            "room_plumbing": True, "follow_profile_config": True, "close_on_disconnect": False})
        assert "error" not in result, result
        sid = str(result["result"]["session_id"])
        if persist:
            # `session.create` does not write the durable row; production persists it on first
            # real activity, and the title itself lands through the installed `session.title`
            # handler (which creates the row when none exists). Both are native paths.
            titled = self.srv._methods["session.title"](
                "rid-title", {"session_id": sid, "title": title})
            assert "error" not in titled, titled
        return sid

    def rpc(self):
        from tui_gateway.hosted_room_server_rpc import HostedRoomServerRPC

        return HostedRoomServerRPC(self.srv)

    def resolves_by_title(self, title: str) -> str | None:
        """The DURABLE session key discovery returns for this canonical title.

        `session.list` answers with the stored row's id (the session key), not the runtime handle
        `session.create` returned: the adapter resolves a durable identity and then resumes it.
        """
        found = self.rpc().resolve_exact(profile=PROFILE, title=title, source="bot_room")
        return None if found is None else str(found["session_id"])

    def resume_handle(self, durable_id: str) -> str:
        """The RUNTIME handle the adapter's own resume maps that durable id back to."""
        resumed = self.rpc().resume(profile=PROFILE, session_id=durable_id, source="bot_room")
        return str(resumed["session_id"])

    def record(self, sid: str) -> dict:
        session = self.srv._sessions.get(sid)
        assert session is not None, f"no runtime record for {sid}"
        return session

    def ready_agent(self, sid: str) -> FakeAgent:
        """Release the prewarmed build and wait for the record to carry its agent."""
        self.build_gate.release()
        session = self.record(sid)
        wait_until(lambda: session.get("agent") is not None, f"agent never built for {sid}")
        return session["agent"]

    # -- prompts ---------------------------------------------------------------
    def submit(self, sid: str, *, task: dict, receipts: list | None = None, text: str = "do it"):
        params: dict[str, Any] = {"session_id": sid, "text": text, "_hosted_task": dict(task)}
        if receipts is not None:
            params["_hosted_terminal_callback"] = receipts.append
        return self.srv._methods["prompt.submit"]("rid-submit", params)

    def interrupt(self, sid: str, *, task_id: str, execution_generation: int | None = None):
        params: dict[str, Any] = {"session_id": sid, "expected_hosted_task_id": task_id}
        if execution_generation is not None:
            params["expected_hosted_execution_generation"] = execution_generation
        return self.srv._methods["session.interrupt"]("rid-interrupt", params)

    # -- background-work ownership ---------------------------------------------
    def track_threads(self) -> None:
        """Record every thread that actually starts from here on, keeping the real types.

        Session records publish `_run_thread`/`_agent_build_thread` before starting them, and the
        prewarm ``threading.Timer`` (server.py `_schedule_agent_build`) is in no record at all, so
        snapshotting records cannot own this work. Instrumenting the real ``Thread.start``
        boundary does, and leaves native scheduling untouched.
        """
        original_start = threading.Thread.start
        started = self.started

        def start(thread, *args, **kwargs):
            result = original_start(thread, *args, **kwargs)
            # Recorded only once the delegated start returned: membership then proves join() is legal.
            started.append(thread)
            return result

        self.monkeypatch.setattr(threading.Thread, "start", start)

    def stop_notification_pollers(self) -> None:
        """Signal this backend's notification pollers with their own native stop events.

        `_start_notification_poller` registers `(stop_event, thread)` in `_notification_pollers`
        and the loop honours that event, exactly as real session finalization does via
        `_notif_stop`. Both are used here; no worker is suppressed and no registry is invented.
        """
        for stop_event, _thread in list(getattr(self.srv, "_notification_pollers", [])):
            stop_event.set()
        for session in list(self.srv._sessions.values()):
            stop_event = session.get("_notif_stop")
            if stop_event is not None:
                stop_event.set()

    def close(self) -> None:
        """Release every gate, drain all tracked work (descendants included), then assert."""
        for agent in self.agents:
            agent.release_turn.set()
        for gate in self.barriers:
            gate.release()
        # Stop the fixture-owned RPC pool before draining, so its workers can finish rather than
        # being joined while the pool still hands them queued work.
        self.srv._pool.shutdown(wait=False, cancel_futures=True)
        deadline = time.monotonic() + WAIT
        drained: set[int] = set()
        while True:
            # Re-signalled every pass: a build still draining can start its poller late.
            self.stop_notification_pollers()
            pending = [t for t in list(self.started) if id(t) not in drained]
            if not pending:
                break
            for thread in pending:
                drained.add(id(thread))
                if isinstance(thread, threading.Timer):
                    thread.cancel()  # a prewarm that has not fired must not start new work
                if thread is threading.current_thread():
                    continue
                thread.join(max(0.0, deadline - time.monotonic()))
        alive = [
            thread.name for thread in list(self.started)
            if thread is not threading.current_thread() and thread.is_alive()]
        assert not alive, f"background work still running after teardown: {alive}"
        assert not self.failures, f"worker thread failed: {self.failures[0]!r}"


def wait_until(predicate: Callable[[], bool], message: str) -> None:
    if not waited(predicate, WAIT):
        raise AssertionError(message)


def waited(predicate: Callable[[], bool], timeout: float) -> bool:
    """Bounded observation that reports what happened instead of asserting it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def hosted_task(task_id: str = "phase1-task", *, generation: int = 1) -> dict:
    return {
        "room_id": ROOM_ID, "task_id": task_id, "thread_id": "thread-a",
        "turn_id": f"turn-{task_id}", "execution_generation": generation}


BACKEND_NAME = "tui_gateway.server"


def _load_backend(monkeypatch):
    """Load the one backend under its own canonical identity, with import effects contained.

    It is loaded as ``tui_gateway.server`` itself -- not a private alias -- and published in
    ``sys.modules`` and on the package before any native code runs, so a lazy
    ``from tui_gateway import server`` inside the application resolves to *this* instance instead
    of importing a second canonical backend (with a second idle reaper). Import-time global
    effects -- the panic excepthooks, the update prefetch, the atexit hooks -- are captured so
    nothing outlives the test or reaches the network.
    """
    import tui_gateway  # the parent package the relative imports resolve against

    assert BACKEND_NAME not in sys.modules, (
        "this module must own the only backend instance: bind_module rebinds shared sibling "
        "classes in place, so a pre-existing import would silently share handlers")
    import atexit

    import hermes_cli.banner as banner
    monkeypatch.setattr(banner, "prefetch_update_check", lambda *a, **k: None, raising=False)

    registered: list = []
    original_register = atexit.register

    def capture(func, *args, **kwargs):
        registered.append(func)
        return original_register(func, *args, **kwargs)

    excepthook, thread_excepthook = sys.excepthook, threading.excepthook
    spec = importlib.util.spec_from_file_location(
        BACKEND_NAME, REPO / "tui_gateway" / "server.py")
    assert spec is not None and spec.loader is not None
    srv = importlib.util.module_from_spec(spec)
    # Both bindings point at this owned object BEFORE execution, so nothing can import a second.
    sys.modules[spec.name] = srv
    monkeypatch.setattr(tui_gateway, "server", srv, raising=False)
    monkeypatch.setattr(atexit, "register", capture)

    # `server.py` starts an endless idle-reaper thread at import (`_start_idle_reaper`). It is
    # unrelated startup machinery -- not an admission, Stop or callback path -- and it never
    # returns, so a duplicate backend would leave it running forever. Exactly that one start is
    # suppressed, matched by the target's identity AND this module's globals; every other start,
    # including the prewarm timer, the build thread, readiness waiters and runner threads,
    # delegates untouched. Idle TTL/reaping is therefore outside this fixture's coverage.
    original_start = threading.Thread.start
    suppressed: list[str] = []

    def start_without_the_idle_reaper(thread, *args, **kwargs):
        target = getattr(thread, "_target", None)
        if (getattr(target, "__qualname__", "") == "_start_idle_reaper.<locals>._loop"
                and getattr(target, "__globals__", None) is vars(srv)):
            suppressed.append(thread.name)
            return None
        return original_start(thread, *args, **kwargs)

    monkeypatch.setattr(threading.Thread, "start", start_without_the_idle_reaper)
    try:
        spec.loader.exec_module(srv)
        assert suppressed, "the import-time idle reaper was not the thread this fixture expected"
    except BaseException:
        # A failed import leaves hooks and half-installed callbacks behind.
        _unwind_backend(srv, registered, excepthook, thread_excepthook)
        raise
    finally:
        monkeypatch.setattr(atexit, "register", original_register)
        monkeypatch.setattr(threading.Thread, "start", original_start)
    return srv, registered, excepthook, thread_excepthook


def _unwind_backend(srv, registered, excepthook, thread_excepthook) -> None:
    """Undo every process-global effect this backend's import installed.

    Every step runs even when an earlier one fails, and the failures are raised rather than
    swallowed: a cleanup that silently fails is how a duplicate backend outlives its test.
    """
    import atexit
    import tui_gateway

    errors: list[BaseException] = []

    def attempt(action: Callable[[], Any]) -> None:
        try:
            action()
        except Exception as exc:  # collected, re-raised below
            errors.append(exc)

    for func in registered:
        attempt(lambda func=func: atexit.unregister(func))
    attempt(lambda: srv._pool.shutdown(wait=False, cancel_futures=True))
    attempt(lambda: srv._sessions.clear())
    sys.excepthook, threading.excepthook = excepthook, thread_excepthook
    # Only the bindings this fixture installed are removed; the package attribute is monkeypatch's.
    if sys.modules.get(BACKEND_NAME) is srv:
        sys.modules.pop(BACKEND_NAME, None)
    if errors:
        raise RuntimeError(f"backend unwind failed: {errors!r}")


@pytest.fixture
def native(tmp_path, monkeypatch):
    """One independently loaded backend on temporary HOME/HERMES_HOME."""
    # Deliberately NOT ``<HOME>/.hermes``: the live-system guard derives the production root from
    # ``expanduser("~")``, so a temp home with that exact name is refused as production.
    home = tmp_path / "gateway-home"
    (home / "profiles").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    srv, registered, excepthook, thread_excepthook = _load_backend(monkeypatch)
    # Binding isolation: the installed handlers belong to this object, and every route a native
    # lazy import could take resolves to the same one.
    import tui_gateway

    assert srv._methods["prompt.submit"].__globals__ is vars(srv)
    assert srv.__name__ == BACKEND_NAME and srv.__package__ == "tui_gateway"
    assert sys.modules[BACKEND_NAME] is srv
    assert tui_gateway.server is srv

    harness = Harness(srv, home, monkeypatch)
    # From here on this fixture owns every thread that starts: build threads, run threads, prewarm
    # timers and their descendants. The import-time idle reaper was suppressed during load (see
    # `_load_backend`); the RPC pool is shut down in `close()` before its workers are drained.
    harness.track_threads()  # every start from here on is this fixture's to drain
    monkeypatch.setattr(
        srv, "_emit", lambda kind, sid, payload=None: harness.events.append((kind, sid, payload)))

    def fake_make_agent(sid, key, **kwargs):
        """The genuine construction seam: everything above it stays real."""
        harness.build_gate.wait_here()
        agent = harness.next_agent or harness.make_agent(f"auto-{sid}")
        harness.next_agent = None
        return agent

    monkeypatch.setattr(srv, "_make_agent", fake_make_agent)
    import run_agent

    def no_real_agent(*args, **kwargs):
        raise AssertionError("a real AIAgent (and its network/model discovery) must never be built")

    monkeypatch.setattr(run_agent, "AIAgent", no_real_agent)
    try:
        yield harness
    finally:
        try:
            # Drain while the fake construction seam and the poisoned AIAgent are still installed.
            harness.close()
        finally:
            _unwind_backend(srv, registered, excepthook, thread_excepthook)


# --------------------------------------------------------------------------- schedules


def test_the_canonical_room_title_resolves_to_its_native_record(native):
    """Setup proof: real SQLite discovery, then the adapter's own durable-to-runtime mapping.

    `session.create` hands back a runtime handle; `session.list` answers with the durable session
    key; `_resolve_or_create` bridges them with `session.resume`. The chain is walked with real
    adapter calls -- the two identifiers are never forced equal.
    """
    from tui_gateway.hosted_room_driver import room_session_title

    title = room_session_title(ROOM_ID)
    sid = native.create_session(title)
    record = native.record(sid)

    durable = native.resolves_by_title(title)

    assert durable == record["session_key"], "discovery must find this record's durable row"
    assert durable != sid, "the durable session key is not the runtime handle"
    assert native.resume_handle(durable) == sid, "resume must map the durable row back to it"
    assert record["source"] == "bot_room"


def test_stop_during_a_deferred_build_latches_without_waiting_for_that_build(native):
    """A Stop must not have to wait for the very build it is cancelling.

    The Stop is issued from its own thread while the deferred build is still blocked, so what is
    measured is whether the request is accepted and latched there -- not whether a fixture barrier
    happened to be open.
    """
    agent = native.make_agent("A")
    sid = native.create_session("Group: build-stop")
    receipts: list = []
    task = hosted_task()

    accepted = native.submit(sid, task=task, receipts=receipts)
    assert "error" not in accepted, accepted
    stop: dict = {}
    native.spawn(
        "stop-during-build",
        lambda: stop.update(
            native.interrupt(sid, task_id=task["task_id"], execution_generation=1) or {}))
    latched = waited(lambda: bool(stop), 2.0)
    native.build_gate.release()
    wait_until(lambda: not native.record(sid).get("running"), "the cancelled turn never finished")

    assert latched, "Stop blocked on the deferred build it was cancelling"
    assert "error" not in stop, stop
    assert agent.turn_running.is_set() is False, "a cancelled build must not dispatch a turn"
    assert [receipt["status"] for receipt in receipts] == ["cancelled"]
    assert native.record(sid).get("inflight_turn") is None


def test_an_unready_agent_reports_a_terminal_receipt_once(native):
    """The accepted submit whose build never readies must not strand the hosted attempt.

    This is the `_wait_agent_for_prompt` error exit (a build that outlives its cap), the one that
    emits UI frames today and never reaches the hosted callback. Construction *errors* take the
    turn's own crash path instead, which already commits a receipt.
    """
    sid = native.create_session("Group: build-timeout")
    receipts: list = []
    native.monkeypatch.setattr(native.srv, "_agent_build_wait_cap", lambda: 0.2)

    accepted = native.submit(sid, task=hosted_task(), receipts=receipts)
    assert "error" not in accepted, accepted
    wait_until(lambda: not native.record(sid).get("running"), "the unready turn never finished")

    assert [receipt["status"] for receipt in receipts] == ["failed"]
    assert receipts[0].get("error")
    # The failed snapshot is deliberately retained for resume replay (`_emit_terminal_turn_error`),
    # so what must not survive is a *live* frame: the turn is terminal, not still streaming.
    assert native.record(sid).get("inflight_turn", {}).get("streaming") is False


def test_a_closing_session_reports_its_accepted_attempt_once(native):
    """The handoff refusal path (`can_start` false) is still an accepted submit: it must report."""
    agent = native.make_agent("A")
    sid = native.create_session("Group: closing-exit")
    receipts: list = []
    gate = native.barrier("closing-handoff")

    def close_the_session_before_handoff() -> None:
        native.record(sid)["_closing"] = True
        gate.release()

    native.wrap("_run_prompt_submit", close_the_session_before_handoff)
    accepted = native.submit(sid, task=hosted_task(), receipts=receipts)
    assert "error" not in accepted, accepted
    native.build_gate.release()
    wait_until(lambda: not native.record(sid).get("running"), "the closing turn never finished")

    assert agent.turn_running.is_set() is False, "a closing session must not dispatch a turn"
    assert len(receipts) == 1, "an accepted submit refused at handoff reported nothing"
    assert native.record(sid).get("inflight_turn") is None


def test_stop_in_the_readiness_to_runner_handoff_prevents_a_late_dispatch(native):
    """A Stop that lands after readiness and before admission must not start the model turn."""
    agent = native.make_agent("A")
    sid = native.create_session("Group: handoff-stop")
    receipts: list = []
    task = hosted_task()
    gate = native.barrier("readiness-handoff")
    native.wrap("_run_prompt_submit", gate.wait_here)

    accepted = native.submit(sid, task=task, receipts=receipts)
    assert "error" not in accepted, accepted
    native.build_gate.release()
    gate.await_reached()
    stop = native.interrupt(sid, task_id=task["task_id"], execution_generation=1)
    assert "error" not in stop, stop
    gate.release()
    wait_until(lambda: not native.record(sid).get("running"), "the stopped turn never finished")

    assert agent.turn_running.is_set() is False, "admission dispatched a cancelled attempt"
    assert agent.cleared == 0, "admission cleared the interrupt of a cancelled attempt"
    assert [receipt["status"] for receipt in receipts] == ["cancelled"]


def test_a_closing_session_after_admission_still_reports_its_attempt(native):
    """The post-admission handoff refusal: `_run_prompt_submit` returns False at `can_start`.

    Distinct from the pre-admission closing case: here `_admit_prompt_turn` already accepted the
    turn, and the session is closed before the runner thread is allowed to start.
    """
    agent = native.make_agent("A")
    sid = native.create_session("Group: closing-after-admission")
    receipts: list = []
    recorder = native.srv._emit

    def close_at_the_handoff(kind, event_sid, payload=None):
        # `message.start` is emitted after admission and before the runner thread is registered.
        if kind == "message.start" and event_sid == sid:
            native.record(sid)["_closing"] = True
        return recorder(kind, event_sid, payload)

    native.monkeypatch.setattr(native.srv, "_emit", close_at_the_handoff)
    accepted = native.submit(sid, task=hosted_task(), receipts=receipts, text="prompt-A")
    assert "error" not in accepted, accepted
    native.build_gate.release()
    wait_until(lambda: not native.record(sid).get("running"), "the closing turn never finished")

    assert agent.invocations == [], "a closed session dispatched its turn anyway"
    assert len(receipts) == 1, "an accepted submit refused after admission reported nothing"
    inflight = native.record(sid).get("inflight_turn")
    assert inflight is None or inflight.get("streaming") is not True


def test_a_parked_waiter_natively_excludes_a_successor_until_its_stop_lands(native):
    """The reachable waiter race: while A's waiter is parked, B cannot be admitted at all.

    Retiring A while its waiter thread is alive is natively impossible -- `_interrupt_session_turn`
    leaves `running` set for a live run thread, and a hosted submit onto a running record is
    refused -- so the exclusion itself is the proof, and the cancellation is then observed for
    real instead of being simulated by forcing record state.
    """
    agent_a, agent_b = native.make_agent("A"), native.make_agent("B")
    sid = native.create_session("Group: parked-waiter")
    receipts_a: list = []
    task_a = hosted_task("task-a", generation=1)
    gate = native.barrier("waiter-park")
    parked = {"done": False}

    def park_attempt_a_once() -> None:
        if not parked["done"]:
            parked["done"] = True
            gate.wait_here()

    native.wrap("_wait_agent_for_prompt", park_attempt_a_once)
    accepted_a = native.submit(sid, task=task_a, receipts=receipts_a, text="prompt-A")
    assert "error" not in accepted_a, accepted_a
    gate.await_reached()
    run_thread_a = native.record(sid)["_run_thread"]

    stop = native.interrupt(sid, task_id=task_a["task_id"], execution_generation=1)
    assert "error" not in stop, stop
    # A's waiter thread is alive, so the Stop stays provisional and the record stays claimed.
    assert native.record(sid).get("running") is True
    refused = native.submit(
        sid, task=hosted_task("task-b", generation=2), receipts=[], text="prompt-B")
    assert refused.get("error", {}).get("code") == 4091, refused

    gate.release()
    run_thread_a.join(WAIT)
    assert not run_thread_a.is_alive(), "attempt A's waiter never finished"

    assert agent_a.invocations == [], "the cancelled attempt still reached a model"
    assert agent_b.invocations == [], "the refused successor was dispatched anyway"
    assert [receipt["status"] for receipt in receipts_a] == ["cancelled"]
    assert native.record(sid).get("running") is False
    # Once A is really over, the successor is admitted normally on the same record.
    native.record(sid)["agent"] = agent_b
    receipts_b: list = []
    task_b = hosted_task("task-b", generation=2)
    accepted_b = native.submit(sid, task=task_b, receipts=receipts_b, text="prompt-B")
    assert "error" not in accepted_b, accepted_b
    native.build_gate.release()  # A's turn was cancelled before its build ever completed
    wait_until(agent_b.turn_running.is_set, "attempt B never started")
    assert not agent_b.ran_prompt("prompt-A"), "B ran the stale attempt's prompt"
    assert not agent_a.ran_prompt("prompt-B")
    assert native.record(sid)["_hosted_room_task"]["task_id"] == task_b["task_id"]


def test_a_stale_interrupt_parked_before_its_session_wait_cannot_reach_a_later_attempt(native):
    """Attempt A's Stop, released after B was admitted, must not touch B."""
    agent_a = native.make_agent("A")
    sid = native.create_session("Group: stale-interrupt")
    receipts_a: list = []
    task_a = hosted_task("task-a", generation=1)
    native.submit(sid, task=task_a, receipts=receipts_a)
    native.ready_agent(sid)
    wait_until(agent_a.turn_running.is_set, "attempt A never started")

    # Park A's Stop between its own observation and its effects: the handler has resolved (and,
    # once Phase 1 lands, latched) attempt A, and the shared interrupt helper has not run yet.
    gate = native.barrier("interrupt-effects")

    def park_only_the_stale_stop() -> None:
        if threading.current_thread().name == "stale-interrupt":
            gate.wait_here()

    native.wrap("_session_uses_compute_host", park_only_the_stale_stop)
    native.spawn(
        "stale-interrupt",
        lambda: native.interrupt(sid, task_id=task_a["task_id"], execution_generation=1))
    gate.await_reached()

    # A finishes; B is accepted on the same record while A's Stop is still parked.
    agent_a.release_turn.set()
    wait_until(lambda: not native.record(sid).get("running"), "attempt A never finished")
    agent_b = native.make_agent("B")
    native.record(sid)["agent"] = agent_b
    receipts_b: list = []
    task_b = hosted_task("task-b", generation=2)
    accepted_b = native.submit(sid, task=task_b, receipts=receipts_b)
    assert "error" not in accepted_b, accepted_b
    wait_until(agent_b.turn_running.is_set, "attempt B never started")
    generation_before = native.record(sid).get("_queued_prompt_generation")
    # B's own durable crash marker, written by its running turn on the shared session key.
    session_key = native.record(sid)["session_key"]
    wait_until(
        lambda: read_turn_marker(native.home, session_key) is not None,
        "attempt B never wrote its crash marker")
    marker_before = read_turn_marker(native.home, session_key)
    marker_key_before = native.record(sid).get("_active_turn_marker_key")
    # A pending client prompt and a pending approval that belong to B's turn.
    pending_event = threading.Event()
    native.srv._pending["rid-b-pending"] = (sid, pending_event)
    approvals: list = []
    import tools.approval as approval_module

    native.monkeypatch.setattr(
        approval_module, "resolve_gateway_approval",
        lambda *args, **kwargs: approvals.append((args, kwargs)), raising=False)
    gate.release()
    wait_until(lambda: not any(t.name == "stale-interrupt" and t.is_alive() for t in native.threads),
               "the stale interrupt never finished")

    assert agent_b.interrupts == [], "a stale Stop reached the newer attempt"
    assert native.record(sid).get("running") is True
    assert native.record(sid)["_hosted_room_task"]["task_id"] == task_b["task_id"]
    assert native.record(sid).get("_queued_prompt_generation") == generation_before
    assert [receipt["status"] for receipt in receipts_a] == ["settled"]
    # B's marker, its durable record and its pending work all survive A's stale Stop.
    assert read_turn_marker(native.home, session_key) == marker_before
    assert native.record(sid).get("_active_turn_marker_key") == marker_key_before
    assert not pending_event.is_set(), "a stale Stop released the successor's pending prompt"
    assert native.srv._answers.get("rid-b-pending") is None
    assert approvals == [], "a stale Stop denied the successor's approvals"


def test_a_stale_finalizer_cannot_clear_a_later_attempt(native):
    """A's cleanup, parked after it cleared `running`, must not erase B's turn state."""
    agent_a = native.make_agent("A")
    sid = native.create_session("Group: stale-finalizer")
    receipts_a: list = []
    task_a = hosted_task("task-a", generation=1)
    native.submit(sid, task=task_a, receipts=receipts_a)
    native.ready_agent(sid)
    wait_until(agent_a.turn_running.is_set, "attempt A never started")
    gate = native.barrier("finalizer-cleanup")
    # Park BEFORE the qualified retirement helper acquires `history_lock`: parking inside that
    # lock would natively exclude B's admission and prove nothing. The wrapper still delegates.
    native.wrap("_retire_attempt_turn_marker", gate.wait_here)

    agent_a.release_turn.set()
    gate.await_reached()
    wait_until(lambda: not native.record(sid).get("running"), "attempt A never cleared running")
    # A is parked inside its own cleanup: capture its run thread before B replaces the handle.
    run_thread_a = native.record(sid)["_run_thread"]
    agent_b = native.make_agent("B")
    native.record(sid)["agent"] = agent_b
    task_b = hosted_task("task-b", generation=2)
    receipts_b: list = []
    accepted_b = native.submit(sid, task=task_b, receipts=receipts_b)
    assert "error" not in accepted_b, accepted_b
    wait_until(agent_b.turn_running.is_set, "attempt B never started")
    session_key = native.record(sid)["session_key"]
    wait_until(
        lambda: read_turn_marker(native.home, session_key) is not None,
        "attempt B never wrote its crash marker")
    marker_before = read_turn_marker(native.home, session_key)
    gate.release()
    # A's receipt can predate its parked cleanup, so only A's thread exiting proves the finalizer
    # ran to completion; B is inspected after that.
    run_thread_a.join(WAIT)
    assert not run_thread_a.is_alive(), "attempt A's finalizer never finished"
    assert len(receipts_a) == 1

    assert native.record(sid).get("running") is True
    assert native.record(sid).get("_hosted_room_task", {}).get("task_id") == task_b["task_id"]
    assert native.record(sid).get("inflight_turn") is not None
    # The durable marker of the still-running successor survives A's retirement.
    assert read_turn_marker(native.home, session_key) == marker_before


def test_a_wrong_task_or_stale_generation_interrupt_is_refused(native):
    """Task ids repeat across generations; a stale generation must not stop the current turn."""
    agent = native.make_agent("A")
    sid = native.create_session("Group: interrupt-identity")
    receipts: list = []
    task = hosted_task(generation=3)
    native.submit(sid, task=task, receipts=receipts)
    native.ready_agent(sid)
    wait_until(agent.turn_running.is_set, "the turn never started")

    wrong_task = native.interrupt(sid, task_id="another-task", execution_generation=3)
    stale_generation = native.interrupt(sid, task_id=task["task_id"], execution_generation=2)

    # Both the exact answer and the side effect matter: a handler that ignores the generation
    # answers with a different shape AND interrupts the live turn.
    assert wrong_task.get("result", {}).get("interrupted") is False, wrong_task
    assert stale_generation.get("result", {}).get("interrupted") is False, stale_generation
    assert agent.interrupts == [], "a stale generation reached the running turn"
    assert native.record(sid).get("running") is True
    agent.release_turn.set()
    wait_until(lambda: len(receipts) == 1, "the turn never reported")
    assert receipts[0]["status"] == "settled"


def test_an_accepted_interrupt_stays_provisional_while_the_turn_survives(native):
    """An accepted Stop reports the request, never a terminal outcome the run has not reached."""
    agent = native.make_agent("A", record_only_interrupt=True)
    sid = native.create_session("Group: surviving-turn")
    receipts: list = []
    task = hosted_task()
    native.submit(sid, task=task, receipts=receipts)
    native.ready_agent(sid)
    wait_until(agent.turn_running.is_set, "the turn never started")
    run_thread = native.record(sid)["_run_thread"]

    stop = native.interrupt(sid, task_id=task["task_id"], execution_generation=1)

    assert "error" not in stop, stop
    wait_until(lambda: bool(agent.interrupts), "the running turn was never asked to stop")
    # The fake keeps executing: nothing may claim the attempt ended while it is alive.
    assert not waited(lambda: bool(receipts), 0.2), (
        "a terminal receipt was fabricated while the turn was still running")
    assert native.record(sid).get("running") is True

    agent.release_turn.set()
    # The receipt is emitted before the run thread's final cleanup, so the run thread's own exit
    # is what proves the turn ended -- not the arrival of its receipt.
    run_thread.join(WAIT)
    assert not run_thread.is_alive(), "the surviving run thread never finished"
    assert [receipt["status"] for receipt in receipts] and receipts[0]["status"] in {
        "settled", "cancelled"}
    assert native.record(sid).get("running") is False


def test_an_ordinary_prompt_and_interrupt_keep_their_native_behaviour(native):
    """Non-hosted turns must not change: no hosted proof, no receipt, ordinary Stop works."""
    agent = native.make_agent("A")
    sid = native.create_session("Ordinary chat")
    session = native.record(sid)
    accepted = native.srv._methods["prompt.submit"](
        "rid-plain", {"session_id": sid, "text": "hello"})
    assert "error" not in accepted, accepted
    native.ready_agent(sid)
    wait_until(agent.turn_running.is_set, "the ordinary turn never started")

    stop = native.srv._methods["session.interrupt"]("rid-plain-stop", {"session_id": sid})

    assert "error" not in stop, stop
    wait_until(lambda: bool(agent.interrupts), "the ordinary turn was never interrupted")
    wait_until(lambda: not native.record(sid).get("running"), "the ordinary turn never finished")
    assert session.get("queued_prompt") is None


def test_a_native_turn_settles_its_room_task_through_the_real_callback(native):
    """One end-to-end native receipt: real adapter, real turn lock, real task store."""
    from gateway import hosted_room_driver as state
    from gateway import hosted_rooms
    from tools.bot_relay import acquire_turn_lock, turn_lock_path
    from tui_gateway.hosted_room_driver import HostedRoomBinding, HostedRoomRuntime, room_session_title
    from tui_gateway.hosted_room_server_rpc import HostedRoomServerRPC

    db = native.home / "state.db"
    binding = HostedRoomBinding(ROOM_ID, "phase1-gateway", 1)
    hosted_rooms.create_room(
        db, room_id=ROOM_ID, name="Phase 1", authority_gateway_id=binding.gateway_id,
        members=[{"profile": PROFILE, "handle": PROFILE}], now=time.time())
    identity = state.TaskIdentity(ROOM_ID, "native-task", "thread-a", "turn-a")
    task = state.admit_task(
        db, identity,
        payload={"target_profile": PROFILE, "prompt": "offline fixture", "source_event_seq": 1},
        clock=time.time)
    title = room_session_title(ROOM_ID)
    sid = native.create_session(title)
    # The driver resolves the durable row and resumes it; both steps must land on this record.
    durable = native.resolves_by_title(title)
    assert durable == native.record(sid)["session_key"]
    assert native.resume_handle(durable) == sid
    agent = native.make_agent("room")

    locks: list[Path] = []

    def turn_lock(profile: str):
        # The real kernel profile lock, rooted in this test's own temporary home.
        locks.append(turn_lock_path(native.home, profile))
        return acquire_turn_lock(native.home, profile, timeout_seconds=WAIT)

    runtime = HostedRoomRuntime(
        db_path=db, rooms=[binding], rpc=HostedRoomServerRPC(native.srv), turn_lock=turn_lock,
        lease_ttl_seconds=100.0)
    lease = state.acquire_lease(
        db, room_id=ROOM_ID, gateway_id=binding.gateway_id, authority_epoch=1,
        process_generation=runtime.process_generation, ttl_seconds=100, clock=time.time)
    runtime._leases[ROOM_ID] = lease
    attempt = state.start_task(db, identity, lease, expected_cancel_generation=0, clock=time.time)

    native.spawn("room-worker", lambda: runtime._execute_attempt(binding, task, attempt))
    native.build_gate.release()
    wait_until(agent.turn_running.is_set, "the native room turn never started")
    assert locks and locks[0].exists(), "the real profile turn lock was not taken"
    agent.release_turn.set()
    wait_until(
        lambda: state.get_task(db, identity)["status"] == "settled",
        "the native callback never settled the room task")

    settled = state.get_task(db, identity)
    assert settled["execution_generation"] == attempt.execution_generation
    assert settled["result"]["text"] == agent.reply
    assert settled["admitted_at"] is not None


def test_a_stop_must_not_read_the_waiter_to_runner_handoff_as_cessation(native):
    """An accepted turn runs across two threads, so a dead thread is not a stopped turn.

    The readiness waiter publishes and starts the real runner and only then exits. A Stop that
    decides from a thread handle it captured earlier sees that death, calls the turn over, clears
    `running`, answers the room `active: false` -- and the model turn keeps running with no
    receipt. The Stop is parked here on exactly that liveness observation and released only once
    the waiter has really exited and its runner is really running.
    """
    agent = native.make_agent("A", record_only_interrupt=True)
    sid = native.create_session("Group: handoff-cessation")
    receipts: list = []
    task = hosted_task()
    runner_gate = native.barrier("after-admission-before-runner")
    stop_gate = native.barrier("captured-waiter-liveness")
    recorder = native.srv._emit

    def park_at_the_handoff(kind, event_sid, payload=None):
        # `message.start` is emitted after admission and before the runner is published.
        if kind == "message.start" and event_sid == sid:
            runner_gate.wait_here()
        return recorder(kind, event_sid, payload)

    native.monkeypatch.setattr(native.srv, "_emit", park_at_the_handoff)
    accepted = native.submit(sid, task=task, receipts=receipts, text="prompt-A")
    assert "error" not in accepted, accepted
    native.build_gate.release()
    runner_gate.await_reached()
    waiter = native.record(sid)["_run_thread"]
    was_alive = waiter.is_alive

    def park_the_stop_on_its_liveness_check():
        if threading.current_thread().name == "handoff-stop":
            stop_gate.wait_here()
        return was_alive()

    native.monkeypatch.setattr(waiter, "is_alive", park_the_stop_on_its_liveness_check)
    stop: dict = {}
    stopper = native.spawn("handoff-stop", lambda: stop.update(
        native.interrupt(sid, task_id=task["task_id"], execution_generation=1) or {}))
    stop_gate.await_reached()
    runner_gate.release()
    wait_until(agent.turn_running.is_set, "the real runner never started")
    runner = native.record(sid)["_run_thread"]
    assert runner is not waiter, "the runner never replaced the waiter on the record"
    waiter.join(WAIT)
    assert not was_alive(), "the readiness waiter never exited after starting its runner"
    stop_gate.release()
    stopper.join(WAIT)
    assert not stopper.is_alive()

    assert "error" not in stop, stop
    assert native.record(sid).get("running") is True, (
        "Stop read a finished readiness waiter as cessation of its live runner")
    assert native.rpc().info(
        profile=PROFILE, session_id=sid, source="bot_room").get("active") is True, (
        "the room was told the attempt was inactive while its runner was still running")
    assert not receipts, "a terminal receipt was fabricated while the runner was still running"
    # The Stop stays provisional: the surviving turn is what settles it, exactly once.
    agent.release_turn.set()
    runner.join(WAIT)
    wait_until(lambda: len(receipts) == 1, "the surviving turn never reported")
    assert native.record(sid).get("running") is False


def test_a_retired_submit_cannot_replace_the_successor_or_lose_its_receipt(native):
    """The submit window before any thread is published: A is retired there and B really admitted.

    `prompt.submit` persists the durable row between minting its attempt and publishing its
    waiter, and a Stop landing in that window genuinely retires A -- there is no thread yet to
    keep it alive -- so B is admitted for real. A must then neither rebuild the agent under B nor
    write its own waiter over B's live runner handle, and it still owes the room exactly one
    receipt, which its own stale waiter can no longer deliver.
    """
    agent = native.make_agent("shared", record_only_interrupt=True)
    sid = native.create_session("Group: submit-publication")
    native.ready_agent(sid)
    gate = native.barrier("submit-before-publication")
    original_persist = native.srv._persist_session_row_for_submit

    def park_attempt_a_inside_its_submit(rid, session):
        if threading.current_thread().name == "submit-A":
            gate.wait_here()
        return original_persist(rid, session)

    native.monkeypatch.setattr(
        native.srv, "_persist_session_row_for_submit", park_attempt_a_inside_its_submit)
    receipts_a: list = []
    receipts_b: list = []
    response_a: dict = {}
    submitter = native.spawn("submit-A", lambda: response_a.update(native.submit(
        sid, task=hosted_task("task-a"), receipts=receipts_a, text="prompt-A") or {}))
    gate.await_reached()

    stop_a = native.interrupt(sid, task_id="task-a", execution_generation=1)
    assert "error" not in stop_a, stop_a
    assert native.record(sid).get("running") is False, (
        "an accepted submit that has published no thread must be retired by its own Stop")
    accepted_b = native.submit(
        sid, task=hosted_task("task-b", generation=2), receipts=receipts_b, text="prompt-B")
    assert "error" not in accepted_b, accepted_b
    wait_until(agent.turn_running.is_set, "B was never admitted through the real native path")
    runner_b = native.record(sid)["_run_thread"]
    session_key = native.record(sid)["session_key"]
    wait_until(
        lambda: read_turn_marker(native.home, session_key) is not None,
        "attempt B never wrote its crash marker")
    marker_before = read_turn_marker(native.home, session_key)

    gate.release()
    submitter.join(WAIT)
    assert not submitter.is_alive()

    assert "error" not in response_a, response_a
    assert native.record(sid)["_run_thread"] is runner_b, (
        "the retired submit replaced the successor's live runner handle")
    assert [receipt["status"] for receipt in receipts_a] == ["cancelled"], (
        "the retired submit lost its own terminal receipt")
    assert not agent.ran_prompt("prompt-A"), "the retired submit still reached a model"
    assert agent.ran_prompt("prompt-B")
    # B's whole turn state survives A's late exit.
    assert native.record(sid).get("running") is True
    assert native.record(sid)["_hosted_room_task"]["task_id"] == "task-b"
    assert native.record(sid).get("inflight_turn") is not None
    assert read_turn_marker(native.home, session_key) == marker_before
    assert receipts_b == []
    agent.release_turn.set()
    runner_b.join(WAIT)
    wait_until(lambda: len(receipts_b) == 1, "the successor never reported")
    assert receipts_b[0]["status"] == "settled"


def test_the_real_hard_interrupt_never_runs_under_the_session_lock(native):
    """A hard interrupt is not non-blocking, so it must not be held under `history_lock`.

    `InterruptControlMixin.hard_interrupt` waits on `CompressionCommitFence.cancel_before_commit`,
    which a commit already in flight holds with no timeout. Under the session lock that wait also
    blocks this session's readers, its finalizer and its admission. This uses the REAL native
    control method and the REAL fence -- only the model is fake -- and asserts both halves: the
    lock is free during the wait, and the exclusion the lock used to give is still enforced,
    because a successor's admission gate refuses to proceed while this stop is landing.
    """
    from types import MethodType

    from agent.conversation_compression import CompressionCommitFence
    from agent.interrupt_control import InterruptControlMixin

    agent = native.make_agent("real-interrupt", record_only_interrupt=True)
    sid = native.create_session("Group: interrupt-lock")
    receipts: list = []
    task = hosted_task()
    accepted = native.submit(sid, task=task, receipts=receipts, text="prompt-A")
    assert "error" not in accepted, accepted
    native.build_gate.release()
    wait_until(agent.turn_running.is_set, "the turn never started")

    # Only the ordinary construction fields the real method's non-model control path reads.
    agent.hard_interrupt = MethodType(InterruptControlMixin.hard_interrupt, agent)
    agent._active_children_lock = threading.Lock()
    agent._active_children = []
    agent._execution_thread_id = None
    agent.quiet_mode = True
    fence = CompressionCommitFence()
    agent._active_compression_commit_fence = fence
    assert fence.begin_commit()
    waiting = threading.Event()
    original_cancel = CompressionCommitFence.cancel_before_commit

    def observe_the_real_commit_wait(self, *args, **kwargs):
        if self is fence:
            waiting.set()
        return original_cancel(self, *args, **kwargs)

    native.monkeypatch.setattr(
        CompressionCommitFence, "cancel_before_commit", observe_the_real_commit_wait)
    session = native.record(sid)
    successor = (999, "task-successor", 9)
    stopper = native.spawn("real-hard-interrupt", lambda: native.interrupt(
        sid, task_id=task["task_id"], execution_generation=1))
    try:
        assert waiting.wait(WAIT), "the native hard interrupt never reached its real commit wait"
        lock_free = session["history_lock"].acquire(blocking=False)
        if lock_free:
            session["history_lock"].release()
        may_admit_successor = native.srv._await_foreign_attempt_interrupts(
            session, successor, 0.2)
    finally:
        fence.finish_commit()
        stopper.join(WAIT)
    assert not stopper.is_alive()
    wait_until(lambda: bool(getattr(agent, "_interrupt_requested", False)),
               "the real hard interrupt never published its stop")

    assert lock_free, "the hard interrupt held history_lock across the real commit wait"
    assert not may_admit_successor, (
        "a successor could be admitted while this attempt's stop was still landing")
    assert native.srv._await_foreign_attempt_interrupts(session, successor, WAIT), (
        "the interrupt claim outlived the stop that took it")
    # The fake survives interruption, so the turn itself is still what settles -- exactly once.
    agent.release_turn.set()
    wait_until(lambda: len(receipts) == 1, "the interrupted turn never reported")
    assert receipts[0]["status"] in {"settled", "cancelled"}
    assert native.record(sid).get("running") is False


@pytest.mark.parametrize("hosted", [False, True])
@pytest.mark.parametrize("outcome", ["current", "cancelled", "successor", "cancelled_successor"])
def test_foreign_interrupt_timeout_reports_only_the_ordinary_current_attempt(native, hosted, outcome):
    agent = native.make_agent("refusal")
    created = native.srv._methods["session.create"]("create-refusal", {
        "profile": PROFILE, "source": "bot_room" if hosted else "desktop"})
    assert "error" not in created, created
    sid = created["result"]["session_id"]
    session = native.record(sid)
    native.ready_agent(sid)
    with session["history_lock"]:
        predecessor = native.srv._new_turn_attempt(session)
        native.srv._latch_turn_cancel(session, predecessor)
        claim = native.srv._claim_attempt_interrupt(session, predecessor)
    gate = native.barrier("foreign-interrupt-timeout")
    original_wait = native.srv._await_foreign_attempt_interrupts

    def expire_wait(record, attempt):
        gate.wait_here()
        return original_wait(record, attempt, timeout=0)

    native.monkeypatch.setattr(native.srv, "_await_foreign_attempt_interrupts", expire_wait)
    original_emit = native.srv._emit

    def observe_refusal(kind, event_sid, payload=None):
        if kind == "error" and event_sid == sid:
            unlocked = session["history_lock"].acquire(blocking=False)
            if unlocked:
                session["history_lock"].release()
            assert not unlocked, "a successor can overtake the session-scoped refusal"
        original_emit(kind, event_sid, payload)

    native.monkeypatch.setattr(native.srv, "_emit", observe_refusal)
    native.srv.record_turn_start(native.home, session["session_key"], "predecessor-marker")
    marker = read_turn_marker(native.home, session["session_key"])
    before = len(native.events)
    receipts = []
    params = {"session_id": sid, "text": "ordinary-refused-input"}
    if hosted:
        params.update(_hosted_task=hosted_task(), _hosted_terminal_callback=receipts.append)
    accepted = native.srv._methods["prompt.submit"]("refusal-submit", params)
    assert accepted["result"]["status"] == "streaming"
    gate.await_reached()
    waiter = session["_run_thread"]
    attempt = session["_turn_attempt"]
    with session["history_lock"]:
        if outcome in {"cancelled", "cancelled_successor"}:
            native.srv._latch_turn_cancel(session, attempt)
        if outcome.endswith("successor"):
            # Force the stale-at-admission schedule using the native identity primitives.
            successor = native.srv._new_turn_attempt(session)
            session["_turn_attempt_live"] = successor
            native.srv._start_inflight_turn(session, "successor-input")
        inflight = session["inflight_turn"]
    gate.release()
    waiter.join(WAIT)
    assert not waiter.is_alive()
    terminal = [event for event in native.events[before:]
                if event[0] in {"error", "message.start", "message.complete"}]
    message = "The turn was refused before it started: an earlier attempt's stop is still landing"
    if not hosted and outcome == "current":
        assert terminal == [("error", sid, {"message": message})]
        snapshot = native.srv._inflight_snapshot(session)
        assert snapshot["user"] == "ordinary-refused-input"
        assert snapshot["error"] == message
        assert snapshot["status"] == "error" and snapshot["recoverable"]
        assert not snapshot["streaming"]
    else:
        assert terminal == []
    if hosted:
        assert receipts == ([{"status": "failed", "text": "",
                              "error": "the turn was refused before it started"}]
                            if outcome in {"current", "successor"}
                            else [{"status": "cancelled", "text": ""}])
    else:
        assert receipts == []
    assert agent.invocations == [] and agent.cleared == 0
    assert read_turn_marker(native.home, session["session_key"]) == marker
    assert session["_run_thread"] is waiter
    assert session["_turn_interrupt_claims"] == [claim] and not claim[1].is_set()
    assert session["_pending_attempt_outcomes"][predecessor] == "cancelled"
    assert attempt not in session["_pending_attempt_outcomes"]
    if outcome.endswith("successor"):
        assert session["running"] is True and session["inflight_turn"] is inflight
        assert session["_turn_attempt_live"] == successor
        assert session["_pending_attempt_outcomes"][successor] is None
    else:
        assert session["running"] is False and session["_turn_attempt_live"] is None
        if outcome == "cancelled":
            assert native.srv._attempt_was_cancelled(session, attempt)
    native.srv._release_attempt_interrupt(session, claim)


def test_every_retired_pending_submit_keeps_its_own_cancelled_outcome(native):
    """Two retired submits are in flight at once: neither may inherit the other's outcome.

    A and B are each parked in their real persistence step and each really retired by their own
    Stop, then C is really admitted. `cancelled` and `failed` are not interchangeable to the room
    -- one is the user's decision, the other is retryable -- so each retired submit has to report
    its OWN outcome, however many attempts were minted on the record after it. A single
    last-cancellation slot cannot do that; the outcome lives on each attempt's pending entry until
    that attempt consumes it.
    """
    agent = native.make_agent("shared", record_only_interrupt=True)
    sid = native.create_session("Group: multiple-retired-submits")
    native.ready_agent(sid)
    gates = {name: native.barrier(f"persist-{name}") for name in ("submit-A", "submit-B")}
    original_persist = native.srv._persist_session_row_for_submit

    def park_pending_submit(rid, session):
        gate = gates.get(threading.current_thread().name)
        if gate is not None:
            gate.wait_here()
        return original_persist(rid, session)

    native.monkeypatch.setattr(
        native.srv, "_persist_session_row_for_submit", park_pending_submit)
    receipts = {"A": [], "B": [], "C": []}
    workers = []
    for name, generation in (("A", 1), ("B", 2)):
        workers.append(native.spawn(f"submit-{name}", lambda n=name, g=generation: native.submit(
            sid, task=hosted_task(f"task-{n}", generation=g), receipts=receipts[n],
            text=f"prompt-{n}")))
        gates[f"submit-{name}"].await_reached()
        stopped = native.interrupt(sid, task_id=f"task-{name}", execution_generation=generation)
        assert "error" not in stopped, stopped
        assert native.record(sid).get("running") is False, (
            "Stop did not natively retire the pending submit")

    accepted_c = native.submit(
        sid, task=hosted_task("task-C", generation=3), receipts=receipts["C"], text="prompt-C")
    assert "error" not in accepted_c, accepted_c
    wait_until(agent.turn_running.is_set, "C never entered the real native runner")
    session = native.record(sid)
    runner_c, inflight_c = session["_run_thread"], session["inflight_turn"]
    # Both retired submits are still accepted and unreported: each holds its own pending entry.
    assert set(session["_pending_attempt_outcomes"].values()) == {"cancelled", None}
    assert len(session["_pending_attempt_outcomes"]) == 3

    for gate in gates.values():
        gate.release()
    for worker in workers:
        worker.join(WAIT)
        assert not worker.is_alive()

    assert [receipt["status"] for receipt in receipts["A"]] == ["cancelled"], (
        "the first retired submit inherited a later attempt's outcome")
    assert [receipt["status"] for receipt in receipts["B"]] == ["cancelled"]
    assert not agent.ran_prompt("prompt-A") and not agent.ran_prompt("prompt-B")
    assert agent.ran_prompt("prompt-C")
    # C is untouched by either late exit.
    assert session["_run_thread"] is runner_c and runner_c.is_alive()
    assert session["inflight_turn"] is inflight_c and session.get("running") is True
    assert session["_hosted_room_task"]["task_id"] == "task-C"
    assert receipts["C"] == []
    # Bookkeeping is a live pending set, not a history: only C's own entry is left.
    assert list(session["_pending_attempt_outcomes"]) == [session["_turn_attempt"]]

    agent.release_turn.set()
    runner_c.join(WAIT)
    wait_until(lambda: len(receipts["C"]) == 1, "C never reported")
    assert receipts["C"][0]["status"] == "settled"
    assert session["_pending_attempt_outcomes"] == {}, (
        "a completed attempt was retained after every continuation drained")
