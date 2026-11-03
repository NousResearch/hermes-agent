"""A ``delegate_task`` child's tokens are added to its parent's subagent totals when it is torn down.

Regression for #147862: the children's token counters were never folded into anything the parent
reports, so per-prompt usage left out every subagent.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tools.delegate_tool_usage import TOKEN_COUNTERS, fold_child_tokens, subagent_tokens


def _agent(**counters):
    return SimpleNamespace(**{f"session_{name}": counters.get(name, 0) for name in TOKEN_COUNTERS})


@pytest.mark.parametrize("outcome", ["completed", "failed"])
def test_a_torn_down_child_adds_its_tokens_to_the_parent_only(outcome):
    from tools.delegate_tool import _run_single_child

    parent = MagicMock()
    for name in TOKEN_COUNTERS:
        setattr(parent, f"session_{name}", 7)
    child = MagicMock()
    child.session_id = "child-session"
    for name in TOKEN_COUNTERS:
        setattr(child, f"session_{name}", 0)

    def run_conversation(**_kwargs):
        # The child spends tokens; a failed child has spent them all the same.
        child.session_input_tokens, child.session_output_tokens = 120, 30
        child.session_cache_read_tokens, child.session_reasoning_tokens = 4_000, 10
        if outcome == "failed":
            raise RuntimeError("provider went away")
        return {"final_response": "done", "messages": [], "completed": True}

    child.run_conversation.side_effect = run_conversation
    parent._active_children = [child]

    _run_single_child(task_index=0, goal="work", child=child, parent_agent=parent)

    assert subagent_tokens(parent) == {"input_tokens": 120, "output_tokens": 30, "cache_read_tokens": 4_000,
                                       "cache_write_tokens": 0, "reasoning_tokens": 10}
    # The parent's own counters drive its token budget and must not absorb the child's spend.
    assert all(getattr(parent, f"session_{name}") == 7 for name in TOKEN_COUNTERS)


def test_a_child_is_counted_once_with_its_own_subagents():
    parent, child, grandchild = _agent(), _agent(input_tokens=10, output_tokens=1), _agent(input_tokens=5)
    fold_child_tokens(child, grandchild)
    fold_child_tokens(parent, child)
    fold_child_tokens(parent, child)  # teardown is idempotent; so is the fold

    assert subagent_tokens(parent)["input_tokens"] == 15
    assert subagent_tokens(parent)["output_tokens"] == 1


def test_siblings_torn_down_at_once_lose_no_tokens():
    parent = _agent()
    children = [_agent(input_tokens=1, output_tokens=2) for _ in range(32)]
    start = threading.Barrier(len(children))

    def tear_down(child):
        start.wait()
        fold_child_tokens(parent, child)

    threads = [threading.Thread(target=tear_down, args=(child,)) for child in children]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert subagent_tokens(parent)["input_tokens"] == 32
    assert subagent_tokens(parent)["output_tokens"] == 64
