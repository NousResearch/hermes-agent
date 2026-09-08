"""Finished delegate children must not pin their transcripts in the parent heap.

Profiled parent (1,320 children over 13h) reached 2.6 GB RSS: every closed
child AIAgent stayed reachable, and each still owned a shallow copy of its
full message list. Two retainers were proven with ``gc.get_referrers``:

1. ``AIAgent.close()`` cleared ``_session_messages`` but not the
   ``_db_flush_scan_prefix`` snapshot (``messages[:]``) or the streamed-text
   accumulator, so every message dict stayed alive through the agent.
2. ``bind_subagent_parent`` stored the agent (each child binds ITSELF for
   its own turn) strongly in a ContextVar; asyncio Handles/Futures scheduled
   during the turn snapshot that Context and live as long as the background
   LSP / kernel loops do, so the child object itself was never collected.
"""

from __future__ import annotations

import gc
import json
import weakref
from unittest.mock import MagicMock

from agent.subagent_lifecycle import (
    _ACTIVE_PARENT_AGENT,
    bind_subagent_parent,
    get_active_subagent_parent,
)
from run_agent import AIAgent


def _bare_agent() -> AIAgent:
    agent = AIAgent.__new__(AIAgent)
    agent._active_children = []
    import threading

    agent._active_children_lock = threading.Lock()
    agent._session_db = None
    agent.session_id = "child-x"
    return agent


def test_close_releases_transcript_shadow_copies():
    agent = _bare_agent()

    class Payload(str):  # weakref-able stand-in for a message content string
        pass

    payload = Payload("x" * 50_000)
    big = {"role": "tool", "content": payload}
    agent._session_messages = [big]
    agent._db_flush_scan_prefix = agent._session_messages[:]
    agent._streamed_assistant_text_parts = ["y" * 10_000]
    probe = weakref.ref(payload)

    agent.close()

    assert agent._session_messages == []
    assert agent._db_flush_scan_prefix is None
    assert agent._streamed_assistant_text_parts == []
    del big, payload
    gc.collect()
    assert probe() is None, "closed agent still owns its message dicts"


def test_bind_subagent_parent_does_not_pin_agent():
    agent = _bare_agent()
    probe = weakref.ref(agent)
    snapshots = []
    with bind_subagent_parent(agent):
        assert get_active_subagent_parent() is agent
        import contextvars

        # An asyncio Handle scheduled inside the turn keeps this snapshot.
        snapshots.append(contextvars.copy_context())
    assert get_active_subagent_parent() is None
    assert snapshots[0][_ACTIVE_PARENT_AGENT] is not agent
    del agent
    gc.collect()
    assert probe() is None, "Context snapshot still pins the agent"


def test_bind_subagent_parent_accepts_non_weakrefable_doubles():
    class Slots:
        __slots__ = ()

    double = Slots()
    with bind_subagent_parent(double):
        assert get_active_subagent_parent() is double


def _fake_child(messages):
    child = MagicMock()
    child._credential_pool = None
    child._delegate_role = "leaf"
    child.session_estimated_cost_usd = 0.0123
    child.session_cost_status = "estimated"
    child.session_id = "child-sess"
    child.run_conversation.return_value = {
        "final_response": "the summary",
        "completed": True,
        "interrupted": False,
        "api_calls": 3,
        "messages": messages,
    }
    return child


def test_run_single_child_result_json_unchanged_by_transcript_release():
    """Pin: the parent-visible result entry is byte-identical whether or not
    the child released its transcript at close() (the entry never carried
    ``messages``; only summary/tool_trace/tokens/cost derive from them)."""
    from tests.tools.test_delegate import _make_mock_parent
    from tools.delegate_tool import _run_single_child

    messages = [
        {"role": "user", "content": "goal"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "z" * 5000},
        {"role": "assistant", "content": "the summary"},
    ]
    results = []
    for _ in range(2):
        child = _fake_child([dict(m) for m in messages])
        entry = _run_single_child(
            task_index=0, goal="goal", child=child, parent_agent=_make_mock_parent()
        )
        child.close.assert_called_once()
        entry.pop("duration_seconds", None)
        results.append(json.dumps(entry, sort_keys=True, default=str))
    assert results[0] == results[1]
    parsed = json.loads(results[0])
    assert parsed["summary"] == "the summary"
    assert parsed["tool_trace"][0]["tool"] == "read_file"
    assert parsed["cost_usd"] == 0.0123
    assert "messages" not in parsed
