"""delegation.at_capacity: a full background pool runs the batch inline (sync) or rejects it."""

from __future__ import annotations

import json
import queue
import threading
import time

import pytest

from tools import async_delegation
from tools.delegate_tool_dispatch import _Batch, _dispatch_background
from tools.process_registry import process_registry


class _Parent:
    def __init__(self):
        self.session_id = "at-capacity-parent"
        self._active_children = []
        self._active_children_lock = threading.Lock()
        self._interrupt_requested = False
        self.quiet_mode = True


class _Child:
    def __init__(self):
        self.session_id = "at-capacity-child"
        self._delegate_role = "leaf"
        self._delegate_depth = 1
        self._delegate_saved_tool_names = []
        self._credential_pool = None
        self._subagent_id = None
        self._interrupt_requested = False
        self.tool_progress_callback = None
        self.model = "test-model"
        self.ran = False
        self.close_count = 0

    def run_conversation(self, **_kwargs):
        self.ran = True
        return {"final_response": "done", "completed": True, "interrupted": False,
                "api_calls": 1, "messages": []}

    def interrupt(self, message=None, **_kwargs):
        self._interrupt_requested = True

    def get_activity_summary(self):
        return {"api_call_count": 1}

    def close(self):
        self.close_count += 1


def _batch(parent, child):
    tasks = [{"goal": "summarize the repo"}]
    parent._active_children.append(child)
    return _Batch(
        task_list=tasks, children=[(0, tasks[0], child)], parent_agent=parent,
        creds={"model": child.model}, context=None, top_role="leaf", max_children=1,
        live_deleg_id=None, live_writers=[], live_paths=[], origin_wake_sid="",
        origin_ui_session_id="", origin_owner_transport=None,
        origin_owner_session_record=None, origin_session_history_delivery=False, overall_start=time.monotonic(),
    )


@pytest.fixture
def full_pool(tmp_path, monkeypatch):
    """A one-slot pool already occupied by another delegation; yields a config writer."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    async_delegation._reset_for_tests()
    monkeypatch.setattr(process_registry, "completion_queue", queue.Queue())
    release = threading.Event()

    def write_config(policy):
        extra = f"  at_capacity: {policy}\n" if policy else ""
        (tmp_path / "config.yaml").write_text(
            "delegation:\n  max_concurrent_children: 1\n  worktree_isolation: false\n" + extra,
            encoding="utf-8",
        )

    occupied = async_delegation.dispatch_async_delegation(
        goal="occupy the only slot", context=None, toolsets=None, role="leaf", model="test-model",
        session_key="other-session", runner=lambda: (release.wait(30), {"status": "completed"})[1],
        max_async_children=1,
    )
    assert occupied["status"] == "dispatched"
    yield write_config
    release.set()
    if async_delegation._executor is not None:
        async_delegation._executor.shutdown(wait=True)
    async_delegation._reset_for_tests()


@pytest.mark.parametrize("policy", [None, "sync"])
def test_at_capacity_sync_runs_the_batch_inline(full_pool, policy):
    full_pool(policy)
    parent, child = _Parent(), _Child()

    result = json.loads(_dispatch_background(_batch(parent, child)))

    assert child.ran
    assert "SYNCHRONOUSLY" in result["note"]
    assert result["results"][0]["status"] == "completed"


def test_at_capacity_reject_starts_nothing(full_pool):
    full_pool("reject")
    parent, child = _Parent(), _Child()

    result = json.loads(_dispatch_background(_batch(parent, child)))

    assert not child.ran
    assert result["status"] == "rejected"
    assert result["mode"] == "background"
    assert result["goals"] == ["summarize the repo"]
    assert "do not retry now" in result["error"].lower()
    # The unrun child is released, not left attached to the parent.
    assert child.close_count == 1
    assert parent._active_children == []


def test_at_capacity_unknown_policy_falls_back_to_sync(full_pool):
    full_pool("queue")
    parent, child = _Parent(), _Child()

    result = json.loads(_dispatch_background(_batch(parent, child)))

    assert child.ran
    assert "SYNCHRONOUSLY" in result["note"]


def test_reject_policy_does_not_apply_to_a_schedule_failure(tmp_path, monkeypatch):
    """Only a full pool is rejected; an executor failure still runs the batch inline."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    (tmp_path / "config.yaml").write_text(
        "delegation:\n  max_concurrent_children: 1\n  worktree_isolation: false\n  at_capacity: reject\n",
        encoding="utf-8",
    )
    async_delegation._reset_for_tests()

    class _RejectingExecutor:
        def submit(self, *_args, **_kwargs):
            raise RuntimeError("executor shut down")

    monkeypatch.setattr(async_delegation, "_get_executor", lambda _n: _RejectingExecutor())
    parent, child = _Parent(), _Child()
    try:
        result = json.loads(_dispatch_background(_batch(parent, child)))
    finally:
        async_delegation._reset_for_tests()

    assert child.ran
    assert "SYNCHRONOUSLY" in result["note"]
