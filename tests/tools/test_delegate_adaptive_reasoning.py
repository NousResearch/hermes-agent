"""Delegate children inherit the parent's effective reasoning and adaptive policy."""

import unittest
from unittest.mock import MagicMock, patch

from hermes_state import SessionDB
from tests.tools.test_delegate import _make_mock_parent
from tools.delegate_tool import _build_child_agent


class TestDelegationAdaptiveReasoning(unittest.TestCase):
    """Delegate children inherit the parent's effective reasoning as their
    baseline plus the parsed adaptive policy, and reclassify their own goal —
    unless an explicit delegation.reasoning_effort pins them."""

    def _build(self, parent, cfg=None):
        with (
            patch("tools.delegate_tool._load_config") as mock_cfg,
            patch("run_agent.AIAgent") as MockAgent,
        ):
            mock_cfg.return_value = cfg or {"max_iterations": 50, "reasoning_effort": ""}
            MockAgent.return_value = MagicMock()
            child = _build_child_agent(
                task_index=0, goal="test", context=None, toolsets=None,
                model=None, max_iterations=50, parent_agent=parent,
                task_count=1,
            )
            return child, MockAgent.call_args[1]

    def test_child_inherits_parent_effective_reasoning_and_adaptive_policy(self):
        # Delegation happens mid-turn: the parent's reasoning_config at that
        # moment is the adaptively applied level, and it becomes the child's
        # starting baseline together with the parent's parsed policy.
        parent = _make_mock_parent()
        parent.reasoning_config = {"enabled": True, "effort": "high"}
        parent.adaptive_reasoning = {
            "enabled": True, "max_effort": "xhigh", "min_effort": "low",
        }
        parent.reasoning_user_override = False

        child, kwargs = self._build(parent)
        self.assertEqual(
            kwargs["reasoning_config"], {"enabled": True, "effort": "high"}
        )
        self.assertEqual(
            kwargs["adaptive_reasoning"],
            {"enabled": True, "max_effort": "xhigh", "min_effort": "low"},
        )
        # Merely inherited reasoning is NOT a user override — the child's own
        # delegated goal is reclassified within the propagated bounds.
        self.assertIs(child.reasoning_user_override, False)

    def test_explicit_delegation_pin_marks_child_user_override(self):
        parent = _make_mock_parent()
        parent.reasoning_config = {"enabled": True, "effort": "medium"}
        parent.adaptive_reasoning = {"enabled": True, "max_effort": "xhigh"}
        parent.reasoning_user_override = False

        child, kwargs = self._build(
            parent, cfg={"max_iterations": 50, "reasoning_effort": "high"}
        )
        self.assertEqual(
            kwargs["reasoning_config"], {"enabled": True, "effort": "high"}
        )
        self.assertIs(child.reasoning_user_override, True)

    def test_delegation_reasoning_false_pins_thinking_off(self):
        # reasoning_effort: false disables the child's thinking AND its
        # adaptation — never silently re-enable what the user turned off.
        parent = _make_mock_parent()
        parent.reasoning_config = {"enabled": True, "effort": "medium"}
        parent.adaptive_reasoning = {"enabled": True, "max_effort": "xhigh"}
        parent.reasoning_user_override = False

        child, kwargs = self._build(
            parent, cfg={"max_iterations": 50, "reasoning_effort": False}
        )
        self.assertEqual(kwargs["reasoning_config"], {"enabled": False})
        self.assertIs(child.reasoning_user_override, True)

    def test_unparseable_delegation_effort_keeps_child_adaptive(self):
        parent = _make_mock_parent()
        parent.reasoning_config = {"enabled": True, "effort": "medium"}
        parent.adaptive_reasoning = {"enabled": True, "max_effort": "xhigh"}
        parent.reasoning_user_override = False

        child, kwargs = self._build(
            parent, cfg={"max_iterations": 50, "reasoning_effort": "turbo"}
        )
        self.assertEqual(
            kwargs["reasoning_config"], {"enabled": True, "effort": "medium"}
        )
        self.assertIs(child.reasoning_user_override, False)

    def test_parent_user_override_propagates_to_child(self):
        # An explicit parent-session pick (/reasoning, --reasoning) is a user
        # choice; children must not adapt around it.
        parent = _make_mock_parent()
        parent.reasoning_config = {"enabled": True, "effort": "high"}
        parent.adaptive_reasoning = {"enabled": True, "max_effort": "xhigh"}
        parent.reasoning_user_override = True

        child, _kwargs = self._build(parent)
        self.assertIs(child.reasoning_user_override, True)

    def test_adaptive_disabled_parent_propagates_none(self):
        parent = _make_mock_parent()
        parent.reasoning_config = {"enabled": True, "effort": "medium"}
        parent.adaptive_reasoning = None
        parent.reasoning_user_override = False

        _child, kwargs = self._build(parent)
        self.assertIsNone(kwargs["adaptive_reasoning"])

    def test_child_reclassifies_goal_and_can_reach_xhigh(self):
        # Real code path composition: the propagated policy goes through the
        # same parse AIAgent.__init__ applies, then the child's own turn
        # classification — a demanding delegated goal escalates independently
        # of the parent's (inherited medium) level.
        from types import SimpleNamespace

        from agent.adaptive_reasoning import (
            begin_adaptive_reasoning_turn,
            parse_adaptive_reasoning_config,
        )

        parent = _make_mock_parent()
        parent.reasoning_config = {"enabled": True, "effort": "medium"}
        parent.adaptive_reasoning = {
            "enabled": True, "max_effort": "xhigh", "min_effort": "low",
        }
        parent.reasoning_user_override = False
        child, kwargs = self._build(parent)

        child_rt = SimpleNamespace(
            reasoning_config=kwargs["reasoning_config"],
            adaptive_reasoning=parse_adaptive_reasoning_config(
                kwargs["adaptive_reasoning"]
            ),
            reasoning_user_override=child.reasoning_user_override,
            _adaptive_prev_effort=None,
            _adaptive_last_notified_effort=None,
            notice_callback=None,
            _vprint=lambda *a, **k: None,
            platform="subagent",
        )
        token = begin_adaptive_reasoning_turn(
            child_rt,
            "Design the cross-component architecture for migrating our "
            "session store to a multi-process model",
        )
        self.assertIsNotNone(token)
        self.assertEqual(
            child_rt.reasoning_config, {"enabled": True, "effort": "xhigh"}
        )
        # The parent's config object is untouched by the child's escalation.
        self.assertEqual(
            parent.reasoning_config, {"enabled": True, "effort": "medium"}
        )

    def test_pinned_child_never_adapts(self):
        from types import SimpleNamespace

        from agent.adaptive_reasoning import (
            begin_adaptive_reasoning_turn,
            parse_adaptive_reasoning_config,
        )

        parent = _make_mock_parent()
        parent.reasoning_config = {"enabled": True, "effort": "medium"}
        parent.adaptive_reasoning = {"enabled": True, "max_effort": "xhigh"}
        parent.reasoning_user_override = False
        child, kwargs = self._build(
            parent, cfg={"max_iterations": 50, "reasoning_effort": "low"}
        )

        child_rt = SimpleNamespace(
            reasoning_config=kwargs["reasoning_config"],
            adaptive_reasoning=parse_adaptive_reasoning_config(
                kwargs["adaptive_reasoning"]
            ),
            reasoning_user_override=child.reasoning_user_override,
            _adaptive_prev_effort=None,
            _adaptive_last_notified_effort=None,
            notice_callback=None,
            _vprint=lambda *a, **k: None,
            platform="subagent",
        )
        self.assertIsNone(
            begin_adaptive_reasoning_turn(
                child_rt,
                "Design the cross-component architecture for migrating our "
                "session store to a multi-process model",
            )
        )
        self.assertEqual(
            child_rt.reasoning_config, {"enabled": True, "effort": "low"}
        )

    def test_dedicated_session_db_and_adaptive_inheritance_coexist(self):
        """One construction must yield BOTH behaviors that meet at this seam:
        the child's dedicated owned SessionDB handle (#81267) and the
        propagated adaptive policy — neither may displace the other."""
        parent = _make_mock_parent()
        parent.reasoning_config = {"enabled": True, "effort": "high"}
        parent.adaptive_reasoning = {
            "enabled": True, "max_effort": "xhigh", "min_effort": "low",
        }
        parent.reasoning_user_override = False
        parent_db = SessionDB()
        parent._session_db = parent_db
        child_db = None
        try:
            child, kwargs = self._build(parent)
            child_db = kwargs["session_db"]
            self.assertIsInstance(child_db, SessionDB)
            self.assertIsNot(child_db, parent_db)
            self.assertEqual(child._owns_session_db, True)
            self.assertEqual(
                kwargs["reasoning_config"], {"enabled": True, "effort": "high"}
            )
            self.assertEqual(
                kwargs["adaptive_reasoning"],
                {"enabled": True, "max_effort": "xhigh", "min_effort": "low"},
            )
            self.assertIs(child.reasoning_user_override, False)
        finally:
            if child_db is not None:
                child_db.close()
            parent_db.close()


class TestDelegationRouteReasoningPins(unittest.TestCase):
    """A delegation route's effort (a task's provider route) or a profile route's resolved reasoning
    (``ResolvedReasoning``) sets the child's level the same way ``delegation.reasoning_effort`` does, so it
    pins the child's adaptation too; only an inherited parent level adapts."""

    def _build(self, route_effort):
        parent = _make_mock_parent()
        parent.reasoning_config = {"enabled": True, "effort": "medium"}
        parent.adaptive_reasoning = {"enabled": True, "max_effort": "xhigh"}
        parent.reasoning_user_override = False
        with (
            patch("tools.delegate_tool._load_config") as mock_cfg,
            patch("run_agent.AIAgent") as MockAgent,
        ):
            mock_cfg.return_value = {"max_iterations": 50, "reasoning_effort": ""}
            MockAgent.return_value = MagicMock()
            child = _build_child_agent(
                task_index=0, goal="test", context=None, toolsets=None, model=None,
                max_iterations=50, parent_agent=parent, task_count=1,
                override_reasoning_effort=route_effort,
            )
            return child, MockAgent.call_args[1]

    def test_route_effort_pins_the_child(self):
        child, kwargs = self._build("low")
        self.assertEqual(kwargs["reasoning_config"], {"enabled": True, "effort": "low"})
        self.assertIs(child.reasoning_user_override, True)

    def test_profile_route_reasoning_pins_the_child(self):
        from tools.delegate_tool_config import ResolvedReasoning

        child, kwargs = self._build(ResolvedReasoning({"enabled": True, "effort": "high"}))
        self.assertEqual(kwargs["reasoning_config"], {"enabled": True, "effort": "high"})
        self.assertIs(child.reasoning_user_override, True)
        # The profile's own default (no level) is that route's too: never the parent's adapted one.
        child, kwargs = self._build(ResolvedReasoning(None))
        self.assertIsNone(kwargs["reasoning_config"])
        self.assertIs(child.reasoning_user_override, True)

    def test_unparseable_route_effort_inherits_and_adapts(self):
        child, kwargs = self._build("turbo")
        self.assertEqual(kwargs["reasoning_config"], {"enabled": True, "effort": "medium"})
        self.assertIs(child.reasoning_user_override, False)


class TestDelegatedBriefClassification(unittest.TestCase):
    """A child's context rides its first user turn ahead of the goal (``child_task_message``). The
    child classifies the goal alone: the parent's context is background, not the task's difficulty,
    and a bare follow-up (action='continue') or schema-retry turn is classified as it is."""

    # Error evidence, production impact and code: classified on its own, this alone escalates.
    CONTEXT = (
        "The production gateway crashed overnight.\n"
        "Traceback (most recent call last):\n  File \"gateway/run.py\", line 42, in serve\n"
        "RuntimeError: deadlock detected\n```python\nlock.acquire()\n```"
    )

    def _child_runtime(self, goal, context):
        from types import SimpleNamespace

        from agent.adaptive_reasoning import parse_adaptive_reasoning_config

        parent = _make_mock_parent()
        parent.reasoning_config = {"enabled": True, "effort": "medium"}
        parent.adaptive_reasoning = {"enabled": True, "max_effort": "xhigh", "min_effort": "low"}
        parent.reasoning_user_override = False
        with (
            patch("tools.delegate_tool._load_config") as mock_cfg,
            patch("run_agent.AIAgent") as MockAgent,
        ):
            mock_cfg.return_value = {"max_iterations": 50, "reasoning_effort": ""}
            MockAgent.return_value = MagicMock()
            child = _build_child_agent(
                task_index=0, goal=goal, context=context, toolsets=None, model=None,
                max_iterations=50, parent_agent=parent, task_count=1,
            )
            kwargs = MockAgent.call_args[1]
        return SimpleNamespace(
            reasoning_config=kwargs["reasoning_config"],
            adaptive_reasoning=parse_adaptive_reasoning_config(kwargs["adaptive_reasoning"]),
            reasoning_user_override=child.reasoning_user_override,
            _adaptive_prev_effort=None, _adaptive_last_notified_effort=None,
            notice_callback=None, platform="subagent",
            _delegate_context=child._delegate_context,
        )

    def _turn_effort(self, rt, message):
        from agent.adaptive_reasoning import adaptive_reasoning_turn

        with adaptive_reasoning_turn(rt, message):
            effort = rt.reasoning_config["effort"]
        self.assertEqual(rt.reasoning_config, {"enabled": True, "effort": "medium"})  # restored
        return effort

    def test_opening_brief_classifies_the_goal_not_the_context(self):
        from tools.delegate_tool_progress import child_task_message

        goal = "list the files in /tmp"
        rt = self._child_runtime(goal, self.CONTEXT)
        self.assertEqual(self._turn_effort(rt, child_task_message(goal, rt._delegate_context)), "low")

        goal = "Design the cross-component architecture for migrating our session store to a multi-process model"
        rt = self._child_runtime(goal, "Keep it short.")
        self.assertEqual(self._turn_effort(rt, child_task_message(goal, rt._delegate_context)), "xhigh")

    def test_turns_without_the_brief_classify_their_own_text(self):
        # A continued child's next turn is sent bare; a turn that merely mentions the context is
        # classified whole, so its error evidence still counts.
        rt = self._child_runtime("list the files in /tmp", self.CONTEXT)
        self.assertEqual(self._turn_effort(rt, "thanks!"), "low")
        self.assertIn(self._turn_effort(rt, "Fix it.\n" + self.CONTEXT), ("high", "xhigh"))

    def test_child_without_context_classifies_the_goal(self):
        rt = self._child_runtime("list the files in /tmp", None)
        self.assertIsNone(rt._delegate_context)
        self.assertEqual(self._turn_effort(rt, "list the files in /tmp"), "low")


if __name__ == "__main__":
    unittest.main()
