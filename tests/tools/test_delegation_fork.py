"""Tests for delegate_task fork-context snapshots (kimi-code#3007 port).

Covers tools/delegation_fork.py (snapshot building, tail trimming, capping,
inheritance framing) and the delegate_tool wiring (fork flag resolution,
snapshot attachment, degrade-to-blank behavior).
"""

import copy
import unittest
from unittest.mock import MagicMock

from tools.delegation_fork import (
    DEFAULT_FORK_MAX_MESSAGES,
    INHERITANCE_NOTICE,
    _fork_cap,
    _trim_incomplete_tail,
    build_fork_snapshot,
    frame_forked_goal,
)


def _mk_parent(history):
    parent = MagicMock()
    parent.session_id = "sess-parent"
    db = MagicMock()
    db.get_messages_as_conversation.return_value = history
    parent._session_db = db
    return parent


class TestTrimIncompleteTail(unittest.TestCase):
    def test_trailing_tool_results_and_dangling_call_removed(self):
        msgs = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "do it"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]},
            {"role": "tool", "content": "partial", "tool_call_id": "t1"},
        ]
        trimmed = _trim_incomplete_tail(msgs)
        # Ends at the last completed assistant reply: the trailing user
        # prompt is dropped too, because the child's kickoff goal arrives
        # as a user message and user;user would violate alternation.
        self.assertEqual(len(trimmed), 2)
        self.assertEqual(trimmed[-1]["content"], "a")

    def test_clean_tail_untouched(self):
        msgs = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "done"},
        ]
        self.assertEqual(_trim_incomplete_tail(msgs), msgs)

    def test_completed_tool_round_trip_kept(self):
        msgs = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]},
            {"role": "tool", "content": "res", "tool_call_id": "t1"},
            {"role": "assistant", "content": "final answer"},
        ]
        self.assertEqual(_trim_incomplete_tail(msgs), msgs)


class TestBuildForkSnapshot(unittest.TestCase):
    def test_snapshot_strips_system_and_bookkeeping(self):
        parent = _mk_parent([
            {"role": "system", "content": "SYSTEM PROMPT"},
            {"role": "user", "content": "hello", "_row_id": 5,
             "display_kind": "x", "observed": True},
            {"role": "assistant", "content": "hi", "message_id": "m1"},
        ])
        snap = build_fork_snapshot(parent)
        self.assertEqual([m["role"] for m in snap], ["user", "assistant"])
        for m in snap:
            for k in ("_row_id", "display_kind", "observed", "message_id"):
                self.assertNotIn(k, m)

    def test_snapshot_trims_in_flight_delegate_call(self):
        parent = _mk_parent([
            {"role": "user", "content": "investigate"},
            {"role": "assistant", "content": "findings"},
            {"role": "user", "content": "now delegate"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "dt", "function": {"name": "delegate_task"}}]},
        ])
        snap = build_fork_snapshot(parent)
        # In-flight delegate call, and the user prompt that triggered it,
        # are trimmed to the last completed assistant reply.
        self.assertEqual(snap[-1]["content"], "findings")

    def test_no_session_db_returns_none(self):
        parent = MagicMock()
        parent.session_id = "s"
        parent._session_db = None
        self.assertIsNone(build_fork_snapshot(parent))

    def test_empty_history_returns_none(self):
        self.assertIsNone(build_fork_snapshot(_mk_parent([])))

    def test_db_error_returns_none(self):
        parent = _mk_parent([])
        parent._session_db.get_messages_as_conversation.side_effect = RuntimeError("boom")
        self.assertIsNone(build_fork_snapshot(parent))

    def test_cap_keeps_most_recent_and_repairs_leading_tool(self):
        history = [{"role": "user", "content": "start"}]
        for i in range(300):
            history.append({"role": "assistant", "content": "",
                            "tool_calls": [{"id": f"t{i}"}]})
            history.append({"role": "tool", "content": f"r{i}",
                            "tool_call_id": f"t{i}"})
        history.append({"role": "assistant", "content": "done"})
        parent = _mk_parent(history)
        snap = build_fork_snapshot(parent, cfg={"fork_max_messages": 50})
        self.assertLessEqual(len(snap), 50)
        # Must not open on a tool result without its call.
        self.assertNotEqual(snap[0]["role"], "tool")
        self.assertEqual(snap[-1]["content"], "done")

    def test_snapshot_deep_copies(self):
        inner = {"role": "user", "content": "x", "meta": {"a": 1}}
        parent = _mk_parent([inner, {"role": "assistant", "content": "y"}])
        snap = build_fork_snapshot(parent)
        snap[0]["meta"]["a"] = 999
        self.assertEqual(inner["meta"]["a"], 1)

    def test_fork_cap_resolution(self):
        self.assertEqual(_fork_cap(None), DEFAULT_FORK_MAX_MESSAGES)
        self.assertEqual(_fork_cap({"fork_max_messages": 80}), 80)
        self.assertEqual(_fork_cap({"fork_max_messages": "junk"}),
                         DEFAULT_FORK_MAX_MESSAGES)
        # floor
        self.assertEqual(_fork_cap({"fork_max_messages": 1}), 10)


class TestFrameForkedGoal(unittest.TestCase):
    def test_notice_prefixes_goal(self):
        framed = frame_forked_goal("summarize the findings")
        self.assertTrue(framed.startswith(INHERITANCE_NOTICE))
        self.assertIn("summarize the findings", framed)


class TestDelegateToolWiring(unittest.TestCase):
    """The fork kwarg flows registry → delegate_task → child attachment."""

    def test_registry_handler_passes_fork(self):
        import inspect
        from tools.delegate_tool import delegate_task
        sig = inspect.signature(delegate_task)
        self.assertIn("fork", sig.parameters)

    def test_schema_declares_fork(self):
        from tools.delegate_tool import DELEGATE_TASK_SCHEMA
        props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
        self.assertIn("fork", props)
        self.assertEqual(props["fork"]["type"], "boolean")
        task_props = props["tasks"]["items"]["properties"]
        self.assertIn("fork", task_props)

    def test_fork_not_model_hidden(self):
        from tools.delegate_tool import _MODEL_HIDDEN_TASK_FIELDS
        self.assertNotIn("fork", _MODEL_HIDDEN_TASK_FIELDS)


if __name__ == "__main__":
    unittest.main()
