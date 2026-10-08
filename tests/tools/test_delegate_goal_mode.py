"""Goal mode for delegation children (#124292) — bounded judge loop inside a child.

Covers:
1. ``_apply_goal_mode`` flag validation and the no-judge refusal.
2. ``run_goal_continuations`` loop behaviour through a fake child + fake judge
   (no live model): continue → done, budget exhaustion with remaining_work,
   blocked verdict, first-turn-failed short-circuit, wait-as-continue.
3. ``_run_single_child`` integration: a goal-mode child's entry carries
   ``goal_loop`` stats and ``remaining_work``; a plain child's entry stays free
   of both keys.
"""

from __future__ import annotations

import json
import threading
import unittest
from unittest.mock import MagicMock, patch

from tools.delegate_tool import delegate_task, _run_single_child
from tools.delegate_tool_child_run import _SchemaOutcome
from tools.delegate_tool_goal import (
    _apply_goal_mode,
    _child_goal_cfg,
    _judge_available,
    _remaining_work_note,
    run_goal_continuations,
)

_NO_SCHEMA = _SchemaOutcome(None, None, [], 0)


class _JudgeAvailableUp:
    """Patch target: auxiliary judge configured and reachable."""

    def __enter__(self):
        self._p = patch("tools.delegate_tool_goal._judge_available", return_value=True)
        self._p.start()
        return self

    def __exit__(self, *exc):
        self._p.stop()
        return False


class _JudgeDown:
    """Patch target: auxiliary judge not configured (mirrors kanban's fail-closed spawn gate)."""

    def __enter__(self):
        self._p1 = patch("tools.delegate_tool_goal._judge_available", return_value=False)
        import agent.auxiliary_client as aux
        self._p2 = patch.object(aux, "get_text_auxiliary_client", side_effect=RuntimeError("none"))
        self._p1.start(); self._p2.start()
        return self

    def __exit__(self, *exc):
        self._p1.stop(); self._p2.stop()
        return False


def _make_mock_parent(depth=0):
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "***"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    return parent


class _FakeChild:
    """Minimal child: scripted run_conversation results, one per call."""

    session_id = "sess-test"

    def __init__(self, results):
        self.results = list(results)
        self.prompts = []

    def run_conversation(self, user_message=None, task_id=None, stream_callback=None):
        self.prompts.append(user_message)
        return self.results.pop(0)


def _first_result(**over):
    base = {"final_response": "partial: 3 of 10 items", "completed": False, "interrupted": False,
            "api_calls": 3, "messages": [{"role": "user", "content": "goal"}]}
    base.update(over)
    return base


def _with_fake_judge(monkey_judge):
    """Patch hermes_cli.goals.judge_goal (the name the loop imports at call time)."""
    import hermes_cli.goals as goals
    return patch.object(goals, "judge_goal", monkey_judge)


class TestApplyGoalMode(unittest.TestCase):
    def test_flags_land_on_every_task(self):
        with _JudgeAvailableUp():
            tasks = [{"goal": "a"}, {"goal": "b"}]
            out, err = _apply_goal_mode(tasks, True, 7)
        self.assertIsNone(err)
        self.assertTrue(all(t["goal_mode"] for t in out))
        self.assertEqual([t["goal_max_turns"] for t in out], [7, 7])

    def test_disabled_is_noop(self):
        tasks = [{"goal": "a"}]
        out, err = _apply_goal_mode(tasks, False, None)
        self.assertIsNone(err)
        self.assertNotIn("goal_mode", out[0])

    def test_bad_budget_rejected(self):
        with _JudgeAvailableUp():
            out, err = _apply_goal_mode([{"goal": "a"}], True, 0)
            self.assertIn("positive integer", err)
            out, err = _apply_goal_mode([{"goal": "a"}], True, "many")
            self.assertIn("positive integer", err)

    def test_no_judge_configured_refuses(self):
        with _JudgeDown():
            out, err = _apply_goal_mode([{"goal": "a"}], True, 5)
        self.assertIn("goal judge", err)
        self.assertFalse(_judge_available())

    def test_child_goal_cfg_defaults(self):
        self.assertIsNone(_child_goal_cfg({"goal": "a"}))
        with _JudgeAvailableUp():
            self.assertEqual(_child_goal_cfg({"goal_mode": True})["max_turns"] >= 1, True)
            self.assertEqual(_child_goal_cfg({"goal_mode": True, "goal_max_turns": 4})["max_turns"], 4)
        with _JudgeDown():
            # model-supplied per-task flags without a judge degrade to the plain path
            self.assertIsNone(_child_goal_cfg({"goal_mode": True}))


class TestRunLoop(unittest.TestCase):
    def test_continue_then_done(self):
        calls = {"n": 0}

        def judge(goal, response, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return "continue", "5 items left", False, None, False
            return "done", "all items processed", False, None, False

        child = _FakeChild([
            {"final_response": "6 of 10 done", "completed": False, "interrupted": False,
             "api_calls": 2, "messages": [{"role": "user", "content": "c1"}]},
        ])
        with _with_fake_judge(judge):
            result, info = run_goal_continuations(
                child=child, goal_text="process 10 items", first_result=_first_result(),
                child_task_id="s-0", relay_text=None, max_turns=5, session_id=child.session_id,
            )
        self.assertTrue(result["completed"])
        self.assertEqual(result["final_response"], "6 of 10 done")
        self.assertEqual(result["api_calls"], 5)
        self.assertEqual(info["outcome"], "judge_done")
        self.assertEqual(info["turns_used"], 2)
        self.assertEqual(len(child.prompts), 1)
        self.assertIn("judge", child.prompts[0])

    def test_budget_exhaustion_sets_remaining_work(self):
        def judge(goal, response, **kw):
            return "continue", "still not done", False, None, False

        child = _FakeChild([
            {"final_response": "p1", "completed": False, "interrupted": False, "api_calls": 1, "messages": []},
            {"final_response": "p2", "completed": False, "interrupted": False, "api_calls": 1, "messages": []},
        ])
        with _with_fake_judge(judge):
            result, info = run_goal_continuations(
                child=child, goal_text="endless", first_result=_first_result(),
                child_task_id="s-0", relay_text=None, max_turns=3, session_id=child.session_id,
            )
        self.assertEqual(info["outcome"], "budget_exhausted")
        note = _remaining_work_note(info)
        self.assertIn("3/3", note)
        self.assertIn("still not done", note)
        # newest continuation text wins
        self.assertEqual(result["final_response"], "p2")

    def test_blocked_verdict_stops_loop(self):
        def judge(goal, response, **kw):
            return "blocked", "needs credentials that do not exist", False, None, False

        child = _FakeChild([])
        with _with_fake_judge(judge):
            result, info = run_goal_continuations(
                child=child, goal_text="impossible", first_result=_first_result(),
                child_task_id="s-0", relay_text=None, max_turns=5, session_id=child.session_id,
            )
        self.assertEqual(info["outcome"], "blocked")
        self.assertIn("credentials", info["remaining_work"])
        self.assertEqual(child.prompts, [])

    def test_first_turn_failed_short_circuits(self):
        child = _FakeChild([])
        with _with_fake_judge(lambda *a, **k: (_ for _ in ()).throw(AssertionError("judge must not run"))):
            result, info = run_goal_continuations(
                child=child, goal_text="g", first_result=_first_result(failed=True, error="boom"),
                child_task_id="s-0", relay_text=None, max_turns=5, session_id=child.session_id,
            )
        self.assertEqual(info["outcome"], "stopped")
        self.assertEqual(child.prompts, [])

    def test_wait_verdict_treated_as_continue(self):
        calls = {"n": 0}

        def judge(goal, response, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return "wait", "parked", False, None, False
            return "done", "ok", False, None, False

        child = _FakeChild([
            {"final_response": "resumed", "completed": False, "interrupted": False, "api_calls": 1, "messages": []},
        ])
        with _with_fake_judge(judge):
            result, info = run_goal_continuations(
                child=child, goal_text="g", first_result=_first_result(),
                child_task_id="s-0", relay_text=None, max_turns=5, session_id=child.session_id,
            )
        self.assertEqual(info["outcome"], "judge_done")
        self.assertEqual(calls["n"], 2)


class TestRunSingleChildIntegration(unittest.TestCase):
    """The entry contract: goal_loop stats + remaining_work on the parent-visible entry."""

    def _run_child(self, child, goal="do the whole batch"):
        return _run_single_child(task_index=0, goal=goal, child=child, parent_agent=_make_mock_parent())

    def test_goal_mode_child_entry_carries_loop_info(self):
        child = _FakeChild([
            {"final_response": "6 of 10 done", "completed": False, "interrupted": False,
             "api_calls": 2, "messages": []},
            {"final_response": "10 of 10 done", "completed": False, "interrupted": False,
             "api_calls": 1, "messages": []},
        ])
        child._delegate_goal_cfg = {"max_turns": 4}

        calls = {"n": 0}

        def judge(goal, response, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return "continue", "4 items left", False, None, False
            return "done", "complete", False, None, False

        with _with_fake_judge(judge), \
             patch("tools.delegate_tool._lease_child_credential", return_value=(None, None)), \
             patch("tools.delegate_tool._validate_child_output_schema",
                   return_value=_NO_SCHEMA):
            entry = self._run_child(child)
        self.assertEqual(entry["status"], "completed")
        self.assertEqual(entry["goal_loop"]["turns_used"], 2)
        self.assertEqual(entry["goal_loop"]["outcome"], "judge_done")
        self.assertNotIn("remaining_work", entry)

    def test_budget_exhausted_entry_surfaces_remaining_work(self):
        child = _FakeChild([
            {"final_response": "partial output", "completed": False, "interrupted": False,
             "api_calls": 1, "messages": []},
            {"final_response": "partial output, more done", "completed": False, "interrupted": False,
             "api_calls": 1, "messages": []},
        ])
        child._delegate_goal_cfg = {"max_turns": 2}

        with _with_fake_judge(lambda *a, **k: ("continue", "not finished", False, None, False)), \
             patch("tools.delegate_tool._lease_child_credential", return_value=(None, None)), \
             patch("tools.delegate_tool._validate_child_output_schema",
                   return_value=_NO_SCHEMA):
            entry = self._run_child(child)
        self.assertEqual(entry["goal_loop"]["outcome"], "budget_exhausted")
        self.assertIn("remaining_work", entry)
        self.assertIn("not finished", entry["remaining_work"])

    def test_plain_child_entry_has_no_goal_keys(self):
        child = _FakeChild([
            {"final_response": "done fast", "completed": True, "interrupted": False,
             "api_calls": 1, "messages": []},
        ])
        with patch("tools.delegate_tool._lease_child_credential", return_value=(None, None)), \
             patch("tools.delegate_tool._validate_child_output_schema",
                   return_value=_NO_SCHEMA):
            entry = self._run_child(child)
        self.assertNotIn("goal_loop", entry)
        self.assertNotIn("remaining_work", entry)
        self.assertEqual(entry["summary"], "done fast")


class TestSchemaAndPassthrough(unittest.TestCase):
    def test_schema_advertises_goal_mode(self):
        from tools.delegate_tool import DELEGATE_TASK_SCHEMA
        props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"]
        self.assertIn("goal_mode", props)
        self.assertIn("goal_max_turns", props)

    def test_delegate_task_rejects_goal_mode_without_judge(self):
        parent = _make_mock_parent()
        with _JudgeDown():
            out = delegate_task(goal="g", parent_agent=parent, goal_mode=True)
        self.assertIn("goal judge", out)


if __name__ == "__main__":
    unittest.main()
