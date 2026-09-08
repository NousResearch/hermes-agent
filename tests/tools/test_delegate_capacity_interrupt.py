"""A rejected background batch retains synchronous cancellation ownership."""

from __future__ import annotations

import json
import queue
import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from agent.interrupt_control import InterruptControlMixin
from agent.turn_context import _bind_interrupt_scope
from tools import async_delegation
from tools.delegate_tool_dispatch import _Batch, _dispatch_background
from tools.interrupt import is_interrupted, set_interrupt
from tools.process_registry import process_registry


class _Parent(InterruptControlMixin):
    def __init__(self):
        self.session_id = "capacity-interrupt-parent"
        self._active_children = []
        self._active_children_lock = threading.Lock()
        self._execution_thread_id = None
        self._interrupt_requested = False
        self._hard_interrupt_requested = threading.Event()
        self.quiet_mode = True


class _ControlledChild(_Parent):
    """Replace model work while retaining real interrupt and worker-start semantics."""

    def __init__(self):
        super().__init__()
        self.session_id = "capacity-interrupt-child"
        self._delegate_role = "leaf"
        self._delegate_depth = 1
        self._delegate_saved_tool_names = []
        self._credential_pool = None
        self._subagent_id = None
        self.tool_progress_callback = None
        self.model = "test-model"
        self.started = threading.Event()
        self.stop_received = threading.Event()
        self.unwinding = threading.Event()
        self.allow_finish = threading.Event()
        self.finished = threading.Event()
        self.closed = threading.Event()
        self.close_count = 0
        self.closed_while_running = False
        self.observed_interrupt = None

    def interrupt(self, message=None, **kwargs):
        accepted = super().interrupt(message, **kwargs)
        self.stop_received.set()
        return accepted

    def hard_interrupt(self, message=None, **kwargs):
        super().hard_interrupt(message, **kwargs)
        self.stop_received.set()

    def run_conversation(self, **_kwargs):
        # A stop can arrive before this thread exists. Use the real turn-start
        # binding so the pending agent interrupt must reach the tool thread too.
        _bind_interrupt_scope(self, lambda: SimpleNamespace(_set_interrupt=set_interrupt))
        self.started.set()
        try:
            assert self.stop_received.wait(30), "child never received cancellation"
            assert self._interrupt_requested
            assert is_interrupted(), "stop did not reach the child execution thread"
            self.observed_interrupt = (
                self._interrupt_message, self._hard_interrupt_requested.is_set(),
            )
            self.unwinding.set()
            assert self.allow_finish.wait(30), "test did not release child cleanup"
            return {
                "final_response": "", "completed": False, "interrupted": True,
                "api_calls": 0, "messages": [],
            }
        finally:
            self.clear_interrupt()
            self.finished.set()

    def get_activity_summary(self):
        return {"api_call_count": 0}

    def close(self):
        self.closed_while_running |= not self.finished.is_set()
        self.close_count += 1
        self.closed.set()


def _batch(parent, *children):
    tasks = [{"goal": f"wait until cancelled {i}"} for i in range(len(children))]
    parent._active_children.extend(children)
    return _Batch(
        task_list=tasks, children=[(i, tasks[i], child) for i, child in enumerate(children)], parent_agent=parent,
        creds={"model": children[0].model}, context=None, top_role="leaf", max_children=len(children),
        live_deleg_id=None, live_writers=[], live_paths=[], origin_wake_sid="",
        origin_ui_session_id="", origin_owner_transport=None,
        origin_owner_session_record=None, overall_start=time.monotonic(),
    )


@pytest.fixture
def registry_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    (tmp_path / "config.yaml").write_text(
        "delegation:\n  max_concurrent_children: 1\n  worktree_isolation: false\n",
        encoding="utf-8",
    )
    async_delegation._reset_for_tests()
    completion_queue = queue.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", completion_queue)
    yield completion_queue
    # Test bodies release their gates and join their workers before registry teardown.
    if async_delegation._executor is not None:
        async_delegation._executor.shutdown(wait=True)
    async_delegation._reset_for_tests()


@pytest.mark.parametrize("rejection", ["capacity", "schedule_failure", "partial_schedule_failure"])
@pytest.mark.parametrize("stop_timing", ["running", "during_admission"])
@pytest.mark.parametrize("stop_kind", ["soft", "hard"])
def test_rejected_background_child_stops_with_parent(
    registry_state, monkeypatch, tmp_path, rejection, stop_timing, stop_kind,
):
    parent, child = _Parent(), _ControlledChild()
    background_child = pending_child = None
    if rejection == "partial_schedule_failure":
        # Three independent units: one accepted, one rejected, one not yet submitted.
        # The model-facing batch width is legal under the configured limit.
        (tmp_path / "config.yaml").write_text(
            "delegation:\n  max_concurrent_children: 3\n  worktree_isolation: false\n"
            "  independent_completions: true\n",
            encoding="utf-8",
        )
        background_child, pending_child = _ControlledChild(), _ControlledChild()
        background_child.session_id += "-background"
        pending_child.session_id += "-pending"
        batch = _batch(parent, background_child, child, pending_child)
    else:
        batch = _batch(parent, child)
    occupied = threading.Event()
    release_occupier = threading.Event()
    admission_started = threading.Event()
    continue_admission = threading.Event()
    outcome = Future()

    def occupy_slot():
        occupied.set()
        assert release_occupier.wait(30)
        return {"status": "completed", "summary": "slot released"}

    if rejection == "capacity":
        accepted = async_delegation.dispatch_async_delegation(
            goal="occupy the only slot", context=None, toolsets=None, role="leaf",
            model=child.model, session_key="other-session", runner=occupy_slot,
            max_async_children=1,
        )
        assert accepted["status"] == "dispatched"
        assert occupied.wait(5)
    elif rejection == "schedule_failure":
        class RejectingExecutor:
            def submit(self, *_args, **_kwargs):
                raise RuntimeError("executor shut down")

        monkeypatch.setattr(async_delegation, "_get_executor", lambda _n: RejectingExecutor())
    else:
        executor = async_delegation._get_executor(3)

        class PartiallyRejectingExecutor:
            submitted = 0

            def submit(self, *args, **kwargs):
                self.submitted += 1
                if self.submitted == 2:
                    raise RuntimeError("unit submission failed")
                return executor.submit(*args, **kwargs)

        partial_executor = PartiallyRejectingExecutor()
        monkeypatch.setattr(async_delegation, "_get_executor", lambda _n: partial_executor)

    dispatch = async_delegation.dispatch_async_delegation_batch
    admissions = 0
    accepted_ids = []

    def pause_admission(**kwargs):
        nonlocal admissions
        admissions += 1
        rejected_admission = 2 if background_child is not None else 1
        if admissions == rejected_admission:
            admission_started.set()
            assert continue_admission.wait(5)
        result = dispatch(**kwargs)
        if result.get("status") == "dispatched":
            accepted_ids.append(result["delegation_id"])
        return result

    monkeypatch.setattr(async_delegation, "dispatch_async_delegation_batch", pause_admission)

    def run_dispatch():
        try:
            outcome.set_result(json.loads(_dispatch_background(batch)))
        except BaseException as exc:
            outcome.set_exception(exc)

    worker = threading.Thread(target=run_dispatch, daemon=True)
    worker.start()
    try:
        assert admission_started.wait(5)
        if background_child is not None:
            assert background_child.started.wait(5)
        request_stop = parent.hard_interrupt if stop_kind == "hard" else parent.interrupt
        stop_message = "user correction or stop request"
        if stop_timing == "during_admission":
            request_stop(stop_message)
        continue_admission.set()
        assert child.started.wait(5)
        if stop_timing == "running":
            request_stop(stop_message)

        assert child.stop_received.wait(5), "fallback lost parent cancellation ownership"
        assert child.unwinding.wait(5)
        assert child.observed_interrupt == (stop_message, stop_kind == "hard")
        if background_child is not None:
            assert not background_child.stop_received.is_set()
            assert pending_child.stop_received.is_set(), "unsubmitted unit lost parent cancellation ownership"
            assert not pending_child.started.is_set()
        assert not outcome.done(), "dispatch returned while its child still owned resources"
        assert child.close_count == 0
        child.allow_finish.set()
        result = outcome.result(timeout=5)
        if background_child is None:
            assert "SYNCHRONOUSLY" in result["note"]
            assert result["results"][0]["status"] == "interrupted"
        else:
            assert result["status"] == "dispatched"
            assert result["inline_results"][0]["status"] == "interrupted"
            assert not background_child.stop_received.is_set()
            assert pending_child.unwinding.wait(5)
            assert pending_child.observed_interrupt == (stop_message, stop_kind == "hard")
        assert child.finished.is_set()
        assert child.close_count == 1
        assert not child.closed_while_running
        assert parent._active_children == []
    finally:
        continue_admission.set()
        if not child.finished.is_set():
            child.hard_interrupt("test teardown")
        child.allow_finish.set()
        worker.join(timeout=5)
        release_occupier.set()
        if background_child is not None:
            async_delegation.interrupt_for_session(parent_session_id=parent.session_id)
            for extra in (background_child, pending_child):
                extra.allow_finish.set()
                assert extra.closed.wait(5)
                assert extra.finished.is_set()
                assert extra.close_count == 1
                assert not extra.closed_while_running
            completed_ids = {registry_state.get(timeout=5)["delegation_id"] for _ in accepted_ids}
            assert completed_ids == set(accepted_ids)
        if rejection == "capacity":
            completion = registry_state.get(timeout=5)
            assert completion["delegation_id"] == accepted["delegation_id"]
        assert not worker.is_alive()


def test_accepted_background_child_keeps_registry_cancellation_ownership(registry_state):
    parent, child = _Parent(), _ControlledChild()
    try:
        result = json.loads(_dispatch_background(_batch(parent, child)))
        assert result["status"] == "dispatched"
        assert child.started.wait(5)
        parent.interrupt()
        # Parent interrupt fan-out is synchronous; observing it return establishes
        # that a detached child did not receive it without a timing-based wait.
        assert parent._interrupt_requested
        assert not child.stop_received.is_set()
        assert not child.finished.is_set()
        parent.hard_interrupt("stop the current parent turn")
        assert parent._hard_interrupt_requested.is_set()
        assert not child.stop_received.is_set()
        assert async_delegation.interrupt_for_session(parent_session_id=parent.session_id) == 1
        assert child.unwinding.wait(5)
        assert child.observed_interrupt[1] is True
        assert child.close_count == 0
        child.allow_finish.set()
        completion = registry_state.get(timeout=5)
        assert completion["delegation_id"] == result["delegation_id"]
        assert completion["results"][0]["status"] == "interrupted"
        assert child.finished.is_set()
        assert child.close_count == 1
        assert not child.closed_while_running
        assert parent._active_children == []
    finally:
        if not child.finished.is_set():
            child.hard_interrupt("test teardown")
        child.allow_finish.set()
        assert child.closed.wait(5)
