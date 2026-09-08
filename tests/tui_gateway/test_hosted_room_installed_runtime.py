"""Hosted attempts across installed RPC handlers, real session/room SQLite and real threads.

The model factory is the only substituted executor. Spies add deterministic barriers
around the real callback/observer; receipt writes and public publication stay intact.
"""

from __future__ import annotations

import sqlite3
import sys
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import pytest

from gateway import hosted_room_driver as state, hosted_rooms
from hermes_state import SessionDB
from tui_gateway import server
from tui_gateway.hosted_room_service import HostedRoomService
from tui_gateway.turn_marker import read_turn_marker


WAIT = 15
ANSWER = "durable answer"
RETURNED_ERROR = "provider rejected this turn"
RAISED_ERROR = "model failed before returning"
BUILD_ERROR = "model initialization failed before readiness"


class _Frames:
    def __init__(self):
        self.frames = []

    def write(self, frame):
        self.frames.append(frame)
        return True

    def close(self):
        pass


class _Model:
    model = "test-model"
    provider = "test"
    base_url = ""
    api_key = ""
    tools = []
    _config_context_length = 1

    def __init__(self, harness, key, session_db):
        self.harness = harness
        self.session_id = key
        self._session_db = session_db
        self._owns_session_db = False
        self.interrupted = threading.Event()
        self.interrupt_calls = 0

    def clear_interrupt(self):
        self.interrupted.clear()

    def switch_model(self, *, new_model, new_provider, **kwargs):
        self.model = new_model
        self.provider = new_provider

    def hard_interrupt(self):
        if self.harness.interrupt_hook is not None:
            self.harness.interrupt_hook(self)
        self.interrupt_calls += 1
        self.interrupted.set()

    def run_conversation(self, prompt, **kwargs):
        self.harness.model_calls.append(prompt)
        if self.harness.model_run is not None:
            return self.harness.model_run(self, prompt)
        if self.harness.outcome == "exception":
            raise RuntimeError(RAISED_ERROR)
        if self.harness.outcome == "returned-error":
            return {"final_response": "", "error": RETURNED_ERROR, "failed": True}
        return {"final_response": self.harness.answer, "completed": True}

    def close(self):
        if self._owns_session_db and self._session_db is not None:
            self._session_db.close()
            self._owns_session_db = False


class _Harness:
    def __init__(self, home):
        self.home = home
        self.db_path = home / "state.db"
        self.outcome = "normal"
        self.answer = ANSWER
        self.model_run = None
        self.interrupt_hook = None
        self.model_calls = []
        self.models = []
        self.workers = []
        self.accepted = threading.Event()
        self.prompt_dispatched = threading.Event()
        self.allow_build = threading.Event()
        self.calls = []
        self.call_errors = []
        self.frames = _Frames()
        self.service = HostedRoomService(server, db_path=self.db_path)
        self.rpc = self.service.rpc
        self.rpc._transport = self.frames
        self.runtime = self.service.runtime
        self.now = [time.time()]
        self.runtime.clock = lambda: self.now[0]
        self.runtime.active_poll_interval_seconds = 0.02
        self.runtime.lease_ttl_seconds = 300
        self.service.create_room(
            room_id="receipt-room", name="Receipt recovery",
            members=[{"member_id": "ops", "profile": "ops", "handle": "ops"},
                     {"member_id": "default", "profile": "default", "handle": "default"}],
        )
        self.binding = self.service.bindings()[0]

    def build(self, sid, key, **kwargs):
        assert self.allow_build.wait(WAIT), "test did not release model construction"
        if self.outcome == "build-error":
            raise RuntimeError(BUILD_ERROR)
        model = _Model(self, key, kwargs.get("session_db"))
        self.models.append(model)
        return model

    def enqueue(self):
        prompt = "inspect @file:context.txt" if self.outcome == "blocked" else "inspect the release"
        self.service.send(
            room_id=self.binding.room_id, event_id="request-1",
            payload={"text": f"@ops {prompt}", "thread_id": "thread-1"},
        )
        tasks = state.list_tasks(self.db_path, room_id=self.binding.room_id)
        assert len(tasks) == 1
        return tasks[0]["identity"]

    @property
    def session(self):
        sessions = [s for s in server._sessions.values() if s.get("source") == "bot_room"]
        assert len(sessions) == 1
        return sessions[0]

    def finish_turn(self, session=None):
        # The readiness thread publishes the inner run handle before starting it.
        # Its real return guarantees that handoff completed before we join the turn.
        assert self.prompt_dispatched.wait(WAIT), "prompt readiness thread did not dispatch"
        session = self.session if session is None else session
        deadline = time.monotonic() + WAIT
        while time.monotonic() < deadline:
            thread = session.get("_run_thread")
            if thread is not None:
                thread.join(max(0, deadline - time.monotonic()))
                if thread is session.get("_run_thread") and not thread.is_alive():
                    assert not session["running"]
                    return
        pytest.fail("installed prompt thread did not finish")

    def member_replies(self):
        return [e for e in self.service._events(self.binding.room_id) if e["kind"] == "message.member"]

    def admission_diagnostics(self, worker):
        frame = sys._current_frames().get(worker.ident)
        return (f"calls={self.calls!r}\ncall_errors={self.call_errors!r}\n"
                f"runtime={self.runtime.status()!r}\nworker_alive={worker.is_alive()}\n"
                + ("".join(traceback.format_stack(frame)) if frame is not None else "no worker frame"))

    def execute(self, attempt):
        self.accepted.clear()
        worker = threading.Thread(target=self.runtime._execute_attempt, args=(
            self.binding, state.get_task(self.db_path, attempt.identity), attempt), daemon=True)
        self.workers.append(worker)
        worker.start()
        assert self.accepted.wait(WAIT), self.calls
        return next(sid for sid, session in server._sessions.items() if session is self.session)


@pytest.fixture
def installed_runtime(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    profile = home / "profiles" / "ops"
    profile.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Native metadata probes must not discover a repository above this empty fixture.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    # These are path/registry isolation, not substitutes for the installed handlers or DB.
    monkeypatch.setattr(server, "_hermes_home", home)
    monkeypatch.setattr(server, "_CRASH_LOG", str(tmp_path / "turn-crash.log"))
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_served_profile_homes", set())
    config = f'model:\n  default: test-model\nterminal:\n  cwd: {tmp_path.as_posix()}\napprovals:\n  mode: manual\n'
    for directory in (home, profile):
        (directory / "config.yaml").write_text(config, encoding="utf-8")
    (tmp_path / "context.txt").write_text("PRIVATE_CONTEXT_SENTINEL", encoding="utf-8")
    database = SessionDB(home / "state.db")
    monkeypatch.setattr(server, "_db", database)
    harness = _Harness(home)
    monkeypatch.setattr(server, "_make_agent", harness.build)
    real_call = harness.rpc._call
    real_dispatch = server._run_after_agent_ready

    def observe_dispatch(*args, **kwargs):
        try:
            return real_dispatch(*args, **kwargs)
        finally:
            harness.prompt_dispatched.set()

    monkeypatch.setattr(server, "_run_after_agent_ready", observe_dispatch)

    def observe_call(method, params):
        try:
            result = real_call(method, params)
        except Exception as exc:
            harness.call_errors.append((method, type(exc).__name__, str(exc)))
            raise
        harness.calls.append((method, params, result))
        if method == "prompt.submit":
            assert result["status"] == "streaming"
            harness.accepted.set()
        return result

    monkeypatch.setattr(harness.rpc, "_call", observe_call)
    try:
        yield harness
    finally:
        harness.interrupt_hook = None
        harness.allow_build.set()
        harness.runtime.stop()
        from tools.approval import resolve_gateway_approval, unregister_gateway_notify
        for sid, session in list(server._sessions.items()):
            resolve_gateway_approval(session["session_key"], "deny", resolve_all=True)
            if model := session.get("agent"):
                model.hard_interrupt()
            if session.get("_run_thread") is not None:
                harness.finish_turn(session)
            if stop := session.get("_notif_stop"):
                stop.set()
            for name in ("_agent_build_thread", "_run_thread"):
                if thread := session.get(name):
                    thread.join(WAIT)
                    assert not thread.is_alive(), name
            unregister_gateway_notify(session["session_key"])
            server._close_session_by_id(sid)
        for worker in harness.workers:
            _join(worker)
        harness.runtime._drop_lease(harness.binding.room_id)
        database.close()


def _receipt_trigger(db_path, *, install):
    with sqlite3.connect(db_path) as conn:
        if install:
            conn.execute("""CREATE TRIGGER reject_terminal_receipt
                BEFORE INSERT ON hosted_room_terminal_receipts
                BEGIN SELECT RAISE(ABORT, 'injected receipt write failure'); END""")
        else:
            conn.execute("DROP TRIGGER reject_terminal_receipt")


def _join(thread):
    thread.join(WAIT)
    assert not thread.is_alive(), "runtime observer did not finish"


@pytest.mark.parametrize("outcome", ["normal", "returned-error", "blocked", "exception", "build-error"])
@pytest.mark.parametrize("order", ["callback-first", "observer-first", "storage-abort"])
def test_installed_terminal_receipt_and_observer_replay(installed_runtime, monkeypatch, outcome, order):
    h = installed_runtime
    h.outcome = outcome
    identity = h.enqueue()
    callback_entered, callback_done, release_callback = (threading.Event() for _ in range(3))
    observer_read = threading.Event()
    observed_receipts = []
    receipts, callback_errors, callback_attempts = [], [], []
    original_terminal = h.runtime._on_terminal
    original_wait = h.runtime._wait_for_terminal

    def on_terminal(binding, attempt, receipt):
        try:
            durable = state.get_terminal_receipt(h.db_path, identity, execution_generation=attempt.execution_generation)
            assert durable is not None, "callback ran before SQLite commit"
            assert read_turn_marker(Path(h.session["profile_home"]), h.session["session_key"]) is not None
            assert receipt["result"] == durable["result"]
            receipts.append(receipt)
            callback_attempts.append(attempt)
            callback_entered.set()
            if order == "callback-first":
                assert observer_read.wait(WAIT)
            if order == "observer-first":
                assert release_callback.wait(WAIT)
            original_terminal(binding, attempt, receipt)
        except BaseException as exc:
            callback_errors.append(exc)
            raise
        finally:
            callback_done.set()

    def wait_for_terminal(*args, **kwargs):
        if order in {"callback-first", "observer-first"}:
            assert callback_entered.wait(WAIT)
        terminal = original_wait(*args, **kwargs)
        if order in {"callback-first", "observer-first"}:
            assert terminal is not None
            observed_receipts.append(terminal)
            observer_read.set()
        if order == "callback-first":
            # Harvest first, but let the real callback settle before the observer.
            # Both paths must replay settlement, not merely skip a terminal task.
            assert callback_done.wait(WAIT)
        return terminal

    monkeypatch.setattr(h.runtime, "_on_terminal", on_terminal)
    monkeypatch.setattr(h.runtime, "_wait_for_terminal", wait_for_terminal)
    if order == "storage-abort":
        _receipt_trigger(h.db_path, install=True)
    worker = threading.Thread(target=h.runtime._process_room, args=(h.binding,), daemon=True)
    worker.start()
    try:
        if not h.accepted.wait(WAIT):
            pytest.fail(h.admission_diagnostics(worker))
        # session.create is lazy, but FIRST prompt persistence pins the real profile row,
        # before model readiness, private receipt or publication.
        with SessionDB(h.home / "profiles" / "ops" / "state.db") as profile_db:
            row = profile_db.get_session(h.session["session_key"])
            assert row["source"] == "bot_room" and row["pinned"] == 1 and row["hidden"] == 1
        h.allow_build.set()
        if order == "observer-first":
            assert callback_entered.wait(WAIT)
            _join(worker)  # actual observer has settled/published before callback proceeds
            release_callback.set()
        if order != "storage-abort":
            assert callback_done.wait(WAIT), {"frames": h.frames.frames[-8:], "model_calls": h.model_calls,
                                             "build_error": h.session.get("agent_error")}
            _join(worker)
        h.finish_turn()
    finally:
        h.allow_build.set()
        release_callback.set()
        h.runtime.stop()
        _join(worker)

    assert callback_errors == []
    assert len(h.model_calls) == (0 if outcome in {"blocked", "build-error"} else 1)
    attempt = state.get_task(h.db_path, identity)
    generation = attempt["execution_generation"]
    receipt = state.get_terminal_receipt(h.db_path, identity, execution_generation=generation)
    marker = read_turn_marker(Path(h.session["profile_home"]), h.session["session_key"])
    if order == "storage-abort":
        assert receipt is None and receipts == [] and marker is not None
        assert h.session["_hosted_room_task"] == {**asdict(identity), "execution_generation": generation}
        assert attempt["status"] == "running"
        assert h.member_replies() == []
        _receipt_trigger(h.db_path, install=False)
    else:
        assert len(receipts) == 1 and marker is None and "_hosted_room_task" not in h.session
        expected = {"message_id": f"reply:{identity.task_id}:{generation}", "text": ANSWER if outcome == "normal" else ""}
        if outcome == "returned-error":
            expected["text"] = f"Error: {RETURNED_ERROR}"
        errors = {"returned-error": RETURNED_ERROR, "exception": RAISED_ERROR, "build-error": BUILD_ERROR}
        if outcome == "blocked":
            # The error event is the exact parser refusal exposed by the installed route;
            # both persisted and public terminal results must carry it byte for byte.
            warnings = [f["params"]["payload"]["message"] for f in h.frames.frames
                        if f.get("params", {}).get("type") == "error"
                        and "injection refused" in f.get("params", {}).get("payload", {}).get("message", "")]
            assert len(warnings) == 1
            errors["blocked"] = warnings[0]
        if outcome in errors:
            expected["error"] = errors[outcome]
        assert receipt["result"] == expected
        assert len(observed_receipts) == 1 and observed_receipts[0].result == expected
        assert receipt["settlement_id"] == expected["message_id"]
        assert "PRIVATE_CONTEXT_SENTINEL" not in str(receipt)
        assert attempt["status"] == receipt["status"] == ("settled" if outcome == "normal" else "failed")
        assert attempt["result"] == expected and attempt["settlement_id"] == receipt["settlement_id"]
        assert h.runtime.status()["last_error"] is None
        assert not h.runtime._ambiguous_rooms
        # Storage's strict equality remains load-bearing, even after callback/observer replay.
        with pytest.raises(state.TaskConflictError):
            state.record_terminal_receipt(
                h.db_path, identity, execution_generation=generation, settlement_id=receipt["settlement_id"],
                status=receipt["status"], result={**expected, "message_id": None}, clock=time.time)
        with pytest.raises(state.TaskConflictError):
            state.settle_task(
                h.db_path, callback_attempts[0], settlement_id=receipt["settlement_id"],
                status=receipt["status"], result={**expected, "message_id": None}, clock=time.time)

    assert state.get_terminal_receipt(h.db_path, identity, execution_generation=generation + 1) is None
    calls_before_restart = len(h.model_calls)
    h.runtime._drop_lease(h.binding.room_id)
    h.now[0] += 301
    recovered = HostedRoomService(server, db_path=h.db_path)
    recovered.runtime.clock = lambda: h.now[0]
    try:
        recovered.runtime._process_room(h.binding)
        recovered.runtime._process_room(h.binding)
        task = state.get_task(h.db_path, identity)
        if order == "storage-abort":
            assert task["status"] == "indeterminate"
            assert state.get_terminal_receipt(h.db_path, identity, execution_generation=generation) is None
            # Even session.resume's crash-marker path must not auto-submit a bot_room turn.
            h.rpc.resume(profile="ops", session_id=h.session["session_key"], source="bot_room")
        else:
            assert task["result"] == receipt["result"]
            terminal_events = [e for e in h.service._events(h.binding.room_id)
                               if e["kind"] == f"turn.{task['status']}" and e["payload"].get("task_id") == identity.task_id]
            assert len(terminal_events) == 1
            if outcome == "normal":
                assert [e["payload"]["text"] for e in h.member_replies()] == [ANSWER]
            else:
                assert h.member_replies() == []
                assert terminal_events[0]["payload"]["error"] == receipt["result"]["error"]
        assert len(h.model_calls) == calls_before_restart
        assert len([call for call in h.calls if call[0] == "prompt.submit"]) == 1
    finally:
        recovered.runtime.stop()
        recovered.runtime._drop_lease(h.binding.room_id)


def _start_attempt(h, identity):
    lease = h.runtime._ensure_lease(h.binding)
    task = state.get_task(h.db_path, identity)
    return state.start_task(h.db_path, identity, lease,
                            expected_cancel_generation=task["cancel_generation"], clock=h.runtime.clock)


def _submit_attempt(h, attempt):
    if not server._sessions:
        created = h.rpc.create(profile="ops", title=f"Group: {h.binding.room_id}", source="bot_room")
        sid = created["session_id"]
    else:
        sid = next(sid for sid, session in server._sessions.items() if session is h.session)
    h.rpc.submit(
        profile="ops", session_id=sid, prompt="inspect the release", source="bot_room",
        task=attempt.identity, execution_generation=attempt.execution_generation,
        on_terminal=lambda receipt: h.runtime._on_terminal(h.binding, attempt, receipt))
    return sid


class _ApprovalTurn:
    """Real approval enqueue/wait, with the model held after the decision for assertions."""

    def __init__(self):
        self.request = None
        self.queued = threading.Event()
        self.finish = threading.Event()
        self.decisions = []

    def __call__(self, model, prompt):
        from tools.approval_gateway_wait import _await_gateway_decision

        def notify(request):
            self.request = request
            self.queued.set()

        decision = _await_gateway_decision(
            model.session_id, notify, {"command": "inspect test fixture", "description": "Test approval"})
        self.decisions.append(decision)
        assert self.finish.wait(WAIT)
        return {"final_response": ANSWER, "interrupted": model.interrupted.is_set(), "completed": True}


@pytest.mark.parametrize("stale_field", ["execution_generation", "room_id", "thread_id", "turn_id"])
def test_installed_stop_preserves_replacement_attempt(installed_runtime, stale_field):
    h = installed_runtime
    identity = h.enqueue()
    old = _start_attempt(h, identity)
    # Real lease recovery and explicit retry produce the SAME task id, generation 2.
    h.now[0] = old.lease.expires_at + 1
    h.runtime._drop_lease(h.binding.room_id)
    lease = h.runtime._ensure_lease(h.binding)
    state.recover_room(h.db_path, lease, clock=h.runtime.clock)
    state.requeue_indeterminate_task(
        h.db_path, identity, lease, expected_execution_generation=old.execution_generation,
        expected_cancel_generation=0, clock=h.runtime.clock)
    current = _start_attempt(h, identity)
    assert current.execution_generation == old.execution_generation + 1
    turn = _ApprovalTurn()
    h.model_run = turn
    h.allow_build.set()
    sid = h.execute(current)
    try:
        assert turn.queued.wait(WAIT)
        from tools.approval import list_gateway_approvals
        session = h.session
        proof = {**asdict(identity), "execution_generation": current.execution_generation}
        info = h.rpc.info(profile="ops", session_id=sid, source="bot_room")
        assert info["hosted_task"] == proof
        # Snapshots cannot mutate the live proof.
        info["hosted_task"]["execution_generation"] = -1
        with session["history_lock"]:
            session["queued_prompt"] = {"text": "keep the replacement's queue"}
            queued_generation = session.get("_queued_prompt_generation", 0)
        expected_identity = identity
        generation = current.execution_generation
        if stale_field == "execution_generation":
            generation = old.execution_generation
        else:
            expected_identity = state.TaskIdentity(**{**asdict(identity), stale_field: "old-coordinate"})
        result = h.rpc.interrupt(
            profile="ops", session_id=sid, source="bot_room", expected_task_id=identity.task_id,
            expected_task=expected_identity, expected_execution_generation=generation)
        assert result == {"status": "not_interrupted", "interrupted": False}
        assert session["_hosted_room_task"] == proof and session["running"]
        assert not session["_turn_cancel_requested"]
        assert session["queued_prompt"] == {"text": "keep the replacement's queue"}
        assert session.get("_queued_prompt_generation", 0) == queued_generation
        assert list_gateway_approvals(session["session_key"])[0]["request_id"] == turn.request["request_id"]
        assert h.models[0].interrupt_calls == 0
        # Matching Stop travels through the real runtime, adapter and installed handler.
        result = h.runtime.cancel(identity, cancel_id="exact-stop")
        assert result["status"] == "cancelled"
        # A concurrent observer/repeated Stop replays the successful claim only.
        assert h.rpc.interrupt(
            profile="ops", session_id=sid, source="bot_room", expected_task_id=identity.task_id,
            expected_task=identity, expected_execution_generation=current.execution_generation
        )["interrupted"] is True
        assert h.models[0].interrupt_calls == 1
        assert session["queued_prompt"] is None
        assert session["_queued_prompt_generation"] == queued_generation + 1
        assert list_gateway_approvals(session["session_key"]) == []
        assert read_turn_marker(Path(session["profile_home"]), session["session_key"]) is None
    finally:
        turn.finish.set()
        from tools.approval import resolve_gateway_approval
        resolve_gateway_approval(h.session["session_key"], "deny", resolve_all=True)
        h.finish_turn()


def test_installed_stop_claim_covers_interrupt_and_queue_cleanup(installed_runtime, monkeypatch):
    h = installed_runtime
    identity = h.enqueue()
    attempt = _start_attempt(h, identity)
    turn = _ApprovalTurn()
    h.model_run = turn
    h.allow_build.set()
    sid = h.execute(attempt)
    claimed, finish_interrupt, looked_up = (threading.Event() for _ in range(3))
    results, failures = [], []

    def interrupt_hook(model):
        claimed.set()
        assert finish_interrupt.wait(WAIT)

    h.interrupt_hook = interrupt_hook
    real_lookup = server._sess_nowait

    def observe_lookup(params, rid):
        resolved = real_lookup(params, rid)
        if threading.current_thread().name == "replacement-submit":
            looked_up.set()
        return resolved

    monkeypatch.setattr(server, "_sess_nowait", observe_lookup)

    def stop():
        try:
            results.append(h.rpc.interrupt(
                profile="ops", session_id=sid, source="bot_room", expected_task_id=identity.task_id,
                expected_task=identity, expected_execution_generation=attempt.execution_generation))
        except BaseException as exc:
            failures.append(exc)

    def submit_replacement():
        from tui_gateway.hosted_room_server_rpc import HostedRoomSessionError
        try:
            _submit_attempt(h, attempt)
        except HostedRoomSessionError as exc:
            results.append(exc.code)

    stop_thread = threading.Thread(target=stop, daemon=True)
    submit_thread = threading.Thread(target=submit_replacement, name="replacement-submit", daemon=True)
    try:
        assert turn.queued.wait(WAIT)
        stop_thread.start()
        assert claimed.wait(WAIT)
        # The old check/use split released this real lock before reaching the model.
        # A replacement admission must be excluded for the ENTIRE cancellation claim.
        acquired = h.session["history_lock"].acquire(blocking=False)
        if acquired:
            h.session["history_lock"].release()
        assert not acquired
        submit_thread.start()
        assert looked_up.wait(WAIT)
        finish_interrupt.set()
        _join(stop_thread)
        _join(submit_thread)
        assert failures == []
        assert {"status": "interrupted", "interrupted": True} in results
        assert 4091 in results  # installed submit refuses the still-settling claimed turn
        assert len(h.model_calls) == 1
    finally:
        finish_interrupt.set()
        turn.finish.set()
        h.interrupt_hook = None
        from tools.approval import resolve_gateway_approval
        resolve_gateway_approval(h.session["session_key"], "deny", resolve_all=True)
        for thread in (stop_thread, submit_thread):
            if thread.ident is not None:
                _join(thread)
        h.finish_turn()


def test_installed_callback_publication_failure_replays_bounded_result(installed_runtime, monkeypatch):
    h = installed_runtime
    h.answer = "é" * (70 * 1024)
    identity = h.enqueue()
    done = threading.Event()
    callback_errors = []
    original_terminal, original_wait = h.runtime._on_terminal, h.runtime._wait_for_terminal

    def callback(binding, attempt, receipt):
        try:
            original_terminal(binding, attempt, receipt)
        except Exception as exc:
            callback_errors.append(str(exc))
            raise
        finally:
            done.set()

    def wait(*args, **kwargs):
        assert done.wait(WAIT)
        return original_wait(*args, **kwargs)

    monkeypatch.setattr(h.runtime, "_on_terminal", callback)
    monkeypatch.setattr(h.runtime, "_wait_for_terminal", wait)
    # Fail the REAL public transaction after the private receipt and task settlement
    # committed. No callback/receipt writer is replaced by a failure stub.
    with sqlite3.connect(h.db_path) as conn:
        conn.execute("""CREATE TRIGGER reject_publication BEFORE INSERT ON hosted_room_events
            WHEN NEW.kind='message.member'
            BEGIN SELECT RAISE(ABORT, 'injected publication failure'); END""")
    worker = threading.Thread(target=h.runtime._process_room, args=(h.binding,), daemon=True)
    worker.start()
    try:
        assert h.accepted.wait(WAIT)
        h.allow_build.set()
        assert done.wait(WAIT)
        h.finish_turn()
        _join(worker)
        assert len(callback_errors) == 1 and "injected publication failure" in callback_errors[0]
        task = state.get_task(h.db_path, identity)
        receipt = state.get_terminal_receipt(h.db_path, identity, execution_generation=task["execution_generation"])
        assert receipt["result"] == task["result"]
        assert task["status"] == "settled"
        assert receipt["result"]["message_id"] == f"reply:{identity.task_id}:{task['execution_generation']}"
        assert receipt["result"]["truncated"] is True
        assert len(receipt["result"]["text"].encode("utf-8")) <= 64 * 1024
        assert receipt["result"]["text"].startswith("é" * 100)
        assert "error" not in receipt["result"]
        assert "_hosted_room_task" not in h.session
        assert read_turn_marker(Path(h.session["profile_home"]), h.session["session_key"]) is None
        assert h.member_replies() == []
    finally:
        h.allow_build.set()
        h.runtime.stop()
        _join(worker)
        with sqlite3.connect(h.db_path) as conn:
            conn.execute("DROP TRIGGER reject_publication")
    state.release_lease(h.db_path, h.runtime._ensure_lease(h.binding), clock=h.runtime.clock)
    h.runtime._drop_lease(h.binding.room_id)
    recovered = HostedRoomService(server, db_path=h.db_path)
    try:
        recovered.runtime._process_room(h.binding)
        recovered.runtime._process_room(h.binding)
        assert len(h.member_replies()) == 1
        terminals = [e for e in h.service._events(h.binding.room_id) if e["kind"] == "turn.settled"]
        assert len(terminals) == 1
        assert state.get_task(h.db_path, identity)["result"] == receipt["result"]
        assert len(h.model_calls) == 1
        assert len([c for c in h.calls if c[0] == "prompt.submit"]) == 1
    finally:
        recovered.runtime.stop()
        recovered.runtime._drop_lease(h.binding.room_id)


def test_installed_stop_before_worker_keeps_admission_claim(installed_runtime, monkeypatch):
    from tui_gateway.hosted_room_server_rpc import HostedRoomSessionError

    h = installed_runtime
    identity = h.enqueue()
    attempt = _start_attempt(h, identity)
    persisted, release_submit = threading.Event(), threading.Event()
    real_persist = server._persist_session_row_for_submit
    failures = []

    def persist_then_pause(*args, **kwargs):
        result = real_persist(*args, **kwargs)
        if not persisted.is_set():
            persisted.set()
            assert release_submit.wait(WAIT)
        return result

    def submit_old():
        try:
            h.execute(attempt)
        except BaseException as exc:
            failures.append(exc)

    monkeypatch.setattr(server, "_persist_session_row_for_submit", persist_then_pause)
    worker = threading.Thread(target=submit_old, name="old-hosted-submit", daemon=True)
    worker.start()
    try:
        assert persisted.wait(WAIT)
        assert h.session.get("_run_thread") is None
        assert h.runtime.cancel(identity, cancel_id="stop-before-worker")["status"] == "cancelled"
        next_identity = state.TaskIdentity(h.binding.room_id, "replacement", "thread-2", "turn-2")
        state.admit_task(h.db_path, next_identity,
                         payload={"target_profile": "ops", "target_member_id": "ops",
                                  "source_event_seq": 1, "prompt": "inspect next release"},
                         clock=h.runtime.clock)
        replacement = _start_attempt(h, next_identity)
        with pytest.raises(HostedRoomSessionError) as busy:
            _submit_attempt(h, replacement)
        assert busy.value.code == 4091
        assert h.session["running"]
        assert h.session["_hosted_room_task"] == {
            **asdict(identity), "execution_generation": attempt.execution_generation}
    finally:
        release_submit.set()
        h.allow_build.set()
        _join(worker)
        h.finish_turn()
    assert failures == []
    assert h.model_calls == []
    assert state.get_terminal_receipt(h.db_path, identity,
                                      execution_generation=attempt.execution_generation) is None


def test_installed_stop_before_turn_admission_never_clears_interrupt(installed_runtime, monkeypatch):
    h = installed_runtime
    identity = h.enqueue()
    attempt = _start_attempt(h, identity)
    admitting, release_admit = threading.Event(), threading.Event()
    real_admit = server._admit_prompt_turn

    def pause_before_admission(*args, **kwargs):
        admitting.set()
        assert release_admit.wait(WAIT)
        return real_admit(*args, **kwargs)

    monkeypatch.setattr(server, "_admit_prompt_turn", pause_before_admission)
    h.allow_build.set()
    h.execute(attempt)
    try:
        assert admitting.wait(WAIT)
        assert h.model_calls == []
        assert h.runtime.cancel(identity, cancel_id="stop-before-admit")["status"] == "cancelled"
        assert h.models[0].interrupted.is_set()
    finally:
        release_admit.set()
        h.finish_turn()
    assert h.model_calls == []
    assert h.models[0].interrupted.is_set()
    assert state.get_terminal_receipt(h.db_path, identity,
                                      execution_generation=attempt.execution_generation) is None


def test_cancelled_start_keeps_claim_until_model_restore_finishes(installed_runtime, monkeypatch):
    from tui_gateway.hosted_room_server_rpc import HostedRoomSessionError

    h = installed_runtime
    identity = h.enqueue()
    attempt = _start_attempt(h, identity)
    admitted, release_admission, cleanup, release_cleanup = (threading.Event() for _ in range(4))
    real_admit, real_finish = server._admit_prompt_turn, server._finish_turn
    restore = {"model": "restored-test-model", "provider": "test"}

    def pause_after_admission(*args, **kwargs):
        result = real_admit(*args, **kwargs)
        if not admitted.is_set():
            assert result is not None
            admitted.set()
            assert release_admission.wait(WAIT)
        return result

    def pause_before_cleanup(sid, session, st):
        if st.hosted_task["task_id"] == identity.task_id:
            assert st.one_turn_restore == restore
            cleanup.set()
            assert release_cleanup.wait(WAIT)
        return real_finish(sid, session, st)

    monkeypatch.setattr(server, "_admit_prompt_turn", pause_after_admission)
    monkeypatch.setattr(server, "_finish_turn", pause_before_cleanup)
    h.allow_build.set()
    h.execute(attempt)
    old_thread = None
    try:
        assert admitted.wait(WAIT)
        with h.session["history_lock"]:
            h.session["one_turn_model_restore"] = restore
        assert h.runtime.cancel(identity, cancel_id="stop-before-cleanup")["status"] == "cancelled"
        release_admission.set()
        assert cleanup.wait(WAIT)
        old_thread = h.session["_run_thread"]
        # The first discussion is cancelled, so use the real storage admission
        # API for a separate attempt rather than waiting for discussion policy.
        replacement_identity = state.TaskIdentity(h.binding.room_id, "replacement", "thread-2", "turn-2")
        state.admit_task(h.db_path, replacement_identity,
                         payload={"target_profile": "ops", "target_member_id": "ops",
                                  "source_event_seq": 1, "prompt": "inspect next release"},
                         clock=h.runtime.clock)
        replacement = _start_attempt(h, replacement_identity)
        with pytest.raises(HostedRoomSessionError) as busy:
            _submit_attempt(h, replacement)
        assert busy.value.code == 4091
        assert h.session["running"]
        assert h.session["_hosted_room_task"] == {
            **asdict(identity), "execution_generation": attempt.execution_generation}
        assert h.models[0].model == "test-model"
    finally:
        release_admission.set()
        release_cleanup.set()
        if old_thread is not None:
            _join(old_thread)
        h.finish_turn()
    assert h.models[0].model == "restored-test-model"
    assert h.model_calls == []
    assert state.get_terminal_receipt(h.db_path, identity,
                                      execution_generation=attempt.execution_generation) is None
    assert not h.session["running"]
    assert "_hosted_room_task" not in h.session
    assert state.get_terminal_receipt(h.db_path, replacement.identity,
                                      execution_generation=replacement.execution_generation) is None


def test_finished_turn_cannot_clear_replacement_task_proof(installed_runtime, monkeypatch):
    h = installed_runtime
    first_identity = h.enqueue()
    first = _start_attempt(h, first_identity)
    finishing, release_old, next_started, release_next = (threading.Event() for _ in range(4))
    real_info = server.logger.info

    def observe_finished(message, *args, **kwargs):
        real_info(message, *args, **kwargs)
        if message.startswith("tui turn finished:") and not finishing.is_set():
            finishing.set()
            assert release_old.wait(WAIT)

    def model_run(model, prompt):
        if len(h.model_calls) == 2:
            next_started.set()
            assert release_next.wait(WAIT)
        return {"final_response": ANSWER, "completed": True}

    monkeypatch.setattr(server.logger, "info", observe_finished)
    h.model_run = model_run
    h.allow_build.set()
    _submit_attempt(h, first)
    old_thread = None
    try:
        assert finishing.wait(WAIT)
        old_thread = h.session["_run_thread"]
        assert not h.session["running"]
        assert state.get_task(h.db_path, first_identity)["status"] == "settled"
        h.service.send(
            room_id=h.binding.room_id, event_id="request-2",
            payload={"text": "@ops inspect the next release", "thread_id": "thread-2"})
        next_identity = next(task["identity"] for task in state.list_tasks(
            h.db_path, room_id=h.binding.room_id) if task["identity"] != first_identity)
        replacement = _start_attempt(h, next_identity)
        sid = _submit_attempt(h, replacement)
        assert next_started.wait(WAIT)
        proof = {**asdict(next_identity), "execution_generation": replacement.execution_generation}
        assert h.rpc.info(profile="ops", session_id=sid, source="bot_room")["hosted_task"] == proof
        release_old.set()
        _join(old_thread)
        # A completed predecessor must not erase the current attempt or its crash
        # marker after admission has handed the session to the next real turn.
        assert h.rpc.info(profile="ops", session_id=sid, source="bot_room")["hosted_task"] == proof
        assert read_turn_marker(Path(h.session["profile_home"]), h.session["session_key"]) is not None
    finally:
        release_old.set()
        release_next.set()
        if old_thread is not None:
            _join(old_thread)
        h.finish_turn()


def test_first_normal_prompt_is_not_room_pinned(installed_runtime):
    h = installed_runtime
    created = h.rpc._call("session.create", {"profile": "ops", "source": "desktop", "title": "Ordinary"})
    sid, key = created["session_id"], created["stored_session_id"]
    result = h.rpc._call("prompt.submit", {"session_id": sid, "text": "normal first prompt"})
    try:
        assert result["status"] == "streaming"
        with SessionDB(h.home / "profiles" / "ops" / "state.db") as profile_db:
            row = profile_db.get_session(key)
            assert row["source"] == "desktop" and row["pinned"] == 0 and row["hidden"] == 0
        assert h.model_calls == []  # first persist, still before agent readiness
    finally:
        h.allow_build.set()


def test_real_approval_queue_rotation_and_durable_dashboard_choice(installed_runtime, monkeypatch):
    from tools.approval import list_gateway_approvals
    from tools.approval_gateway_wait import _await_gateway_decision

    h = installed_runtime
    identity = h.enqueue()
    attempt = _start_attempt(h, identity)
    first_queued, second_queued, observed, allow_observation, answered, finish = (threading.Event() for _ in range(6))
    requests, decisions = [], []

    def model_run(model, prompt):
        for number, queued in enumerate((first_queued, second_queued)):
            def notify(request):
                requests.append(request)
                queued.set()
            decisions.append(_await_gateway_decision(
                model.session_id, notify, {"command": f"inspect fixture {number}", "description": "Test approval"}))
        answered.set()
        assert finish.wait(WAIT)
        return {"final_response": ANSWER, "completed": True}

    h.model_run = model_run
    original_pending = h.runtime.pending_action

    def observe_pending(room_id, member_id, action):
        original_pending(room_id, member_id, action)
        if action and len(requests) == 2 and action["request_id"] == requests[1]["request_id"]:
            observed.set()
            assert allow_observation.wait(WAIT)

    monkeypatch.setattr(h.runtime, "pending_action", observe_pending)
    h.allow_build.set()
    sid = h.execute(attempt)
    dashboard = HostedRoomService(server, db_path=h.db_path)
    try:
        assert first_queued.wait(WAIT)
        first = requests[0]["request_id"]
        assert h.rpc.approve(session_id=sid, request_id=first, choice="deny")["resolved"] == 1
        assert second_queued.wait(WAIT) and observed.wait(WAIT)
        second = requests[1]["request_id"]
        assert first != second
        pending = state.list_pending_approval_requests(h.db_path, room_id=h.binding.room_id)
        assert len(pending) == 1 and pending[0]["request_id"] == second
        params = dict(member_id=pending[0]["member_id"], task_id=identity.task_id,
                      execution_generation=attempt.execution_generation, request_id=second, choice="once")
        for stale in ({"request_id": first}, {"execution_generation": attempt.execution_generation + 1}):
            with pytest.raises(RuntimeError, match="no longer pending"):
                dashboard.approve_room_task(h.binding.room_id, **{**params, **stale})
        assert list_gateway_approvals(h.session["session_key"])[0]["request_id"] == second
        assert dashboard.approve_room_task(h.binding.room_id, **params)["choice"] == "once"
        with pytest.raises(state.TaskConflictError):
            dashboard.approve_room_task(h.binding.room_id, **{**params, "choice": "deny"})
        assert state.list_pending_approval_requests(h.db_path, room_id=h.binding.room_id)[0]["choice"] == "once"
        # The actual owner observation replays the same request and consumes the durable
        # choice through HostedRoomServerRPC -> installed approval.respond -> real queue.
        allow_observation.set()
        assert answered.wait(WAIT)
        assert [d["choice"] for d in decisions] == ["deny", "once"]
        assert list_gateway_approvals(h.session["session_key"]) == []
    finally:
        allow_observation.set()
        finish.set()
        from tools.approval import resolve_gateway_approval
        resolve_gateway_approval(h.session["session_key"], "deny", resolve_all=True)
        h.finish_turn()
        dashboard.runtime.stop()
