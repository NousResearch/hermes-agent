"""The schema-retry turn must run inside the delegated-child context.

The main child turn is wrapped in ``delegated_child_context``; the bounded retry in
``_validate_child_output_schema`` is a second ``run_conversation`` on the same child.
Unwrapped, it runs with the parent worker's identity: ``HERMES_KANBAN_TASK`` is set and
nothing marks it as a child, so the kanban stop guard nudges the child to call
``kanban_complete``. A child owns no board task and cannot, so the nudge text displaces
the answer the retry exists to produce and the retry fails the same schema again.
"""

from __future__ import annotations

import pytest

from agent.delegation_context import is_delegated_child_context
from tools.delegate_tool_child_run import _validate_child_output_schema


class _Child:
    """Minimal stand-in recording the context state of each run_conversation turn."""

    def __init__(self):
        self._delegate_output_schema = {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        }
        self.session_id = "sess-child"
        self.seen_in_child_context = []

    def run_conversation(self, user_message=None, task_id=None, stream_callback=None):
        self.seen_in_child_context.append(is_delegated_child_context())
        return {"final_response": '{"ok": true}', "api_calls": 1}


@pytest.fixture
def kanban_env(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")
    monkeypatch.delenv("HERMES_KANBAN_STOP_NUDGE", raising=False)
    return monkeypatch


def test_schema_retry_runs_in_child_context(kanban_env):
    child = _Child()
    result = {"final_response": "not json at all", "api_calls": 1}

    assert is_delegated_child_context() is False
    _validate_child_output_schema(child, result, 0, "child-task-0", None)

    assert child.seen_in_child_context == [True], (
        "the schema-retry turn ran outside delegated_child_context"
    )
    assert is_delegated_child_context() is False
