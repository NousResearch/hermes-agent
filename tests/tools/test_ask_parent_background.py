"""ask_parent through a real background dispatch: the child of a dispatched unit pauses mid-run, its
question reaches the parent's completion queue, and the parent's reply resumes the same run."""
from __future__ import annotations

import json
import queue
import threading
import time
from types import SimpleNamespace

import pytest

from agent.interrupt_control import InterruptControlMixin
from tools import async_delegation
from tools.delegate_tool_dispatch import _Batch, _dispatch_background
from tools.process_registry import process_registry


class _Parent(InterruptControlMixin):
    def __init__(self):
        self.session_id = "ask-parent-parent"
        self._active_children = []
        self._active_children_lock = threading.Lock()
        self._execution_thread_id = None
        self._interrupt_requested = False
        self._hard_interrupt_requested = threading.Event()
        self.quiet_mode = True


class _AskingChild(_Parent):
    """Stands in for the model: one turn that calls ask_parent, then finishes with what it heard."""

    def __init__(self, subagent_id="sa-ask-1"):
        super().__init__()
        self.session_id = "ask-parent-child"
        self._delegate_role, self._delegate_depth = "leaf", 1
        self._delegate_saved_tool_names = []
        self._credential_pool = None
        self._subagent_id = subagent_id
        self.tool_progress_callback = None
        self.model = "test-model"
        self.tool_result = None
        self.touched = []

    def _touch_activity(self, desc, **_kwargs):
        self.touched.append(desc)

    def run_conversation(self, **_kwargs):
        from tools.ask_parent import ask_parent_tool
        self.tool_result = json.loads(ask_parent_tool("Keep the old API?", state="40/42 tests pass.", agent=self))
        return {"final_response": f"Done; parent said: {self.tool_result.get('answer')}", "completed": True,
                "interrupted": False, "api_calls": 1, "messages": []}

    def get_activity_summary(self):
        return {"api_call_count": 0}

    def close(self):
        pass


def _batch(parent, *children):
    tasks = [{"goal": f"port module {i}"} for i in range(len(children))]
    parent._active_children.extend(children)
    return _Batch(
        task_list=tasks, children=[(i, tasks[i], child) for i, child in enumerate(children)], parent_agent=parent,
        creds={"model": "test-model"}, context=None, top_role="leaf", max_children=len(children),
        live_deleg_id=None, live_writers=[], live_paths=[], origin_wake_sid="",
        origin_ui_session_id="", origin_owner_transport=None,
        origin_owner_session_record=None, origin_session_history_delivery=False, overall_start=time.monotonic(),
    )


@pytest.fixture
def events(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    (tmp_path / "config.yaml").write_text(
        "delegation:\n  max_concurrent_children: 2\n  worktree_isolation: false\n"
        "  ask_parent:\n    enabled: true\n    timeout_seconds: 30\n", encoding="utf-8")
    async_delegation._reset_for_tests()
    completion_queue = queue.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", completion_queue)
    yield completion_queue
    if async_delegation._executor is not None:
        async_delegation._executor.shutdown(wait=True)
    async_delegation._reset_for_tests()


def test_a_background_child_pauses_on_its_question_and_resumes_with_the_reply(events):
    from tools.ask_parent import answer_question

    parent, child = _Parent(), _AskingChild()
    dispatch = json.loads(_dispatch_background(_batch(parent, child)))
    assert dispatch["status"] == "dispatched"

    question = events.get(timeout=10)
    assert question["parent_question"] is True
    assert question["delegation_id"] == dispatch["delegation_id"]
    assert (question["subagent_id"], question["task_index"]) == ("sa-ask-1", 0)
    assert question["parent_session_id"] == parent.session_id

    assert answer_question("sa-ask-1", question["request_id"], "Keep it, deprecate later.") is None
    completion = events.get(timeout=10)

    assert child.tool_result["outcome"] == "answered"
    assert completion["delegation_id"] == dispatch["delegation_id"] and not completion.get("parent_question")
    assert "Keep it, deprecate later." in completion["results"][0]["summary"]
    from tools import ask_parent
    assert "sa-ask-1" not in ask_parent._asked  # a finished child leaves nothing behind


def test_stopping_the_delegation_ends_a_pending_question_as_abandoned(events):
    parent, child = _Parent(), _AskingChild()
    dispatch = json.loads(_dispatch_background(_batch(parent, child)))
    events.get(timeout=10)  # the question

    assert async_delegation.interrupt_for_session(parent_session_id=parent.session_id) == 1
    completion = events.get(timeout=10)

    assert child.tool_result["outcome"] == "abandoned"
    assert completion["delegation_id"] == dispatch["delegation_id"]


def test_a_child_whose_call_is_joined_has_no_live_parent(events, monkeypatch):
    """No background unit (the parent blocks joining this call): the child is told at once."""
    from tools.delegate_tool_dispatch import _execute_and_aggregate

    parent, child = _Parent(), _AskingChild()

    result = _execute_and_aggregate(_batch(parent, child))

    assert child.tool_result["outcome"] == "parent_unavailable"
    assert result["results"][0]["status"] == "completed"
    assert events.empty()
