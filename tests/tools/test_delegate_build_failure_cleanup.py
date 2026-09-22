#!/usr/bin/env python3
"""A failed batch construction must not leak half-built children or eat the one-shot budget.

``delegate_task`` charges the one-shot spawn budget BEFORE building children (atomic
check-and-charge against concurrent calls) and ``_build_children`` aborts mid-loop when a
child fails to construct (e.g. provider preflight that passed for the batch turns out to
fail per-child). Two consequences, both fixed here:

* the charged budget stuck even though nothing ran — one failed call could permanently
  exhaust ``delegation.oneshot_max_children`` for the whole one-shot run;
* the already-built siblings were dropped on the floor without ``close()`` — their
  dedicated SessionDB handles, task resources and parent attachments leaked.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from tools import delegate_tool
from tools.delegate_tool import _build_children, delegate_task

GOAL_A = "Refactor the login handler to use the new session helper"
GOAL_B = "Write regression tests for the session expiry watcher"

_CREDS = {
    "model": "test/model", "provider": "openrouter", "base_url": "https://openrouter.ai/api/v1",
    "api_key": "k", "api_mode": "chat_completions", "request_overrides": None,
}


class TestBuildFailureClosesPartialChildren(unittest.TestCase):
    """``_build_children`` must close the children it already built when a later task fails."""

    def test_partial_children_closed_on_midway_valueerror(self):
        built = MagicMock(name="child-0")
        with patch.object(delegate_tool, "_build_child_preserving_parent_tools", side_effect=[built, ValueError("no endpoint")]):
            children, err = _build_children(
                [{"goal": GOAL_A}, {"goal": GOAL_B}], [], _CREDS, top_role="leaf", max_iterations=10,
                parent_agent=MagicMock(), routing_cfg={}, live_deleg_id=None, live_writers=[],
            )
        self.assertEqual((children, err), ([], "no endpoint"))
        built.close.assert_called_once()

    def test_no_children_no_close_on_first_task_failure(self):
        with patch.object(delegate_tool, "_build_child_preserving_parent_tools", side_effect=ValueError("boom")):
            children, err = _build_children(
                [{"goal": GOAL_A}], [], _CREDS, top_role="leaf", max_iterations=10,
                parent_agent=MagicMock(), routing_cfg={}, live_deleg_id=None, live_writers=[],
            )
        self.assertEqual((children, err), ([], "boom"))


class TestBuildFailureRefundsOneshotBudget(unittest.TestCase):
    """The one-shot budget charged up front is rolled back when construction fails; the retry still spawns."""

    def _delegate(self, parent):
        return delegate_task(tasks=[{"goal": GOAL_A}], parent_agent=parent)

    def test_failed_build_does_not_exhaust_budget(self):
        parent = MagicMock()
        parent._delegate_depth = 0
        parent._session_db = None
        parent.session_id = "s1"
        parent._oneshot_children_spawned = 0

        with patch.dict("os.environ", {"HERMES_SINGLE_QUERY_SESSION": "1"}), \
             patch.object(delegate_tool, "_resolve_delegation_credentials", return_value=dict(_CREDS)), \
             patch.object(delegate_tool, "_get_oneshot_max_children", lambda: 1), \
             patch("tools.delegation_live_log.create_live_transcripts", return_value=(None, [], [])), \
             patch.object(delegate_tool, "_announce_batch"), \
             patch.object(delegate_tool, "_capture_origin", return_value=("", "", None, None, False)), \
             patch.object(delegate_tool, "_build_children", return_value=([], "pinned command missing")):
            first = json.loads(self._delegate(parent))
            second = json.loads(self._delegate(parent))
        self.assertEqual(first.get("error"), "pinned command missing")
        # Before the fix the retry hit "Delegation budget ... exhausted": the failed
        # build had charged the cap away with children that never ran.
        self.assertEqual(second.get("error"), "pinned command missing")
        self.assertNotIn("budget", second.get("error", ""))


if __name__ == "__main__":
    unittest.main()
