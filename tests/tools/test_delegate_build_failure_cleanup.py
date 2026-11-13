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
from tools.delegate_tool_routes import TaskRoute

GOAL_A = "Refactor the login handler to use the new session helper"
GOAL_B = "Write regression tests for the session expiry watcher"

_CREDS = {
    "model": "test/model", "provider": "openrouter", "base_url": "https://openrouter.ai/api/v1",
    "api_key": "k", "api_mode": "chat_completions", "request_overrides": None,
}

# Upstream routes each task through a TaskRoute (creds + owning config block); the batch
# builder consumes ``routes[i]`` per task, so every task under test gets its own route.
_ROUTE = TaskRoute(creds=_CREDS, routing_cfg={})


class TestBuildFailureClosesPartialChildren(unittest.TestCase):
    """``_build_children`` must close the children it already built when a later task fails."""

    def test_partial_children_closed_on_midway_valueerror(self):
        built = MagicMock(name="child-0")
        with patch.object(delegate_tool, "_build_child_preserving_parent_tools", side_effect=[built, ValueError("no endpoint")]):
            children, err = _build_children(
                [{"goal": GOAL_A}, {"goal": GOAL_B}], [], [_ROUTE, _ROUTE], top_role="leaf", max_iterations=10,
                parent_agent=MagicMock(), live_deleg_id=None, live_writers=[],
            )
        self.assertEqual((children, err), ([], "no endpoint"))
        built.close.assert_called_once()

    def test_no_children_no_close_on_first_task_failure(self):
        with patch.object(delegate_tool, "_build_child_preserving_parent_tools", side_effect=ValueError("boom")):
            children, err = _build_children(
                [{"goal": GOAL_A}], [], [_ROUTE], top_role="leaf", max_iterations=10,
                parent_agent=MagicMock(), live_deleg_id=None, live_writers=[],
            )
        self.assertEqual((children, err), ([], "boom"))


    def test_partial_children_closed_on_midway_baseexception(self):
        """A KeyboardInterrupt mid-build must close the already-built siblings too, then propagate:
        cleanup scope matches the session-db release inside _build_child_agent."""
        built = MagicMock(name="child-0")
        with patch.object(
            delegate_tool, "_build_child_preserving_parent_tools", side_effect=[built, KeyboardInterrupt("^C")]
        ):
            with self.assertRaises(KeyboardInterrupt):
                _build_children(
                    [{"goal": GOAL_A}, {"goal": GOAL_B}], [], [_ROUTE, _ROUTE], top_role="leaf", max_iterations=10,
                    parent_agent=MagicMock(), live_deleg_id=None, live_writers=[],
                )
        built.close.assert_called_once()


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


class TestBudgetRMWSerialized(unittest.TestCase):
    """Charge and refund are read-modify-write on ``_oneshot_children_spawned``; both must run
    under ``_oneshot_budget_lock`` or a refunding caller can overwrite a concurrent caller's
    later charge (and two charges can double-spend the cap)."""

    def _budget_env(self):
        return patch.dict("os.environ", {"HERMES_SINGLE_QUERY_SESSION": "1"}), \
               patch.object(delegate_tool, "_get_oneshot_max_children", lambda: 1_000_000)

    def test_refund_cannot_overwrite_a_later_charge(self):
        """The interleaving called out in review: A charges, B charges, A refunds — B's charge survives."""
        import types

        parent = types.SimpleNamespace(_oneshot_children_spawned=0)
        with self._budget_env()[0], self._budget_env()[1]:
            self.assertIsNone(delegate_tool._oneshot_spawn_budget(parent, 3))
            self.assertIsNone(delegate_tool._oneshot_spawn_budget(parent, 2))  # B's later charge
            delegate_tool._refund_oneshot_spawn_budget(parent, 3)  # A's failure refund
        self.assertEqual(parent._oneshot_children_spawned, 2)

    def test_concurrent_charges_and_refunds_conserve(self):
        """Refunding threads (net zero) race pure charging threads; lost updates make the
        final count drift either way, so it must land exactly on the pure-charge total."""
        import sys
        import threading
        import types

        parent = types.SimpleNamespace(_oneshot_children_spawned=0)
        rounds, refunders, chargers = 300, 6, 2

        def churn():
            for _ in range(rounds):
                delegate_tool._oneshot_spawn_budget(parent, 2)
                delegate_tool._refund_oneshot_spawn_budget(parent, 2)

        def spend():
            for _ in range(rounds):
                delegate_tool._oneshot_spawn_budget(parent, 2)

        old_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)  # maximize RMW interleaving without relying on wall-clock timing
        try:
            with self._budget_env()[0], self._budget_env()[1]:
                threads = [threading.Thread(target=churn) for _ in range(refunders)] + \
                          [threading.Thread(target=spend) for _ in range(chargers)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
        finally:
            sys.setswitchinterval(old_interval)
        self.assertEqual(parent._oneshot_children_spawned, chargers * rounds * 2)


class TestMixedRouteRefundAmountMatchesCharge(unittest.TestCase):
    """A failed build must refund exactly what it charged. Mixed-route batches charge
    ``len(runnable)`` (refused routes never reach construction), so the refund must also be
    ``len(runnable)`` — refunding ``len(task_list)`` over-credits siblings' prior spend
    (or, bounded by the guard, leaves the failed charge stuck when prior spend is smaller)."""

    def _delegate_two_tasks(self, parent, routes):
        return delegate_task(tasks=[{"goal": GOAL_A}, {"goal": GOAL_B}], parent_agent=parent)

    def _mixed_route_patches(self, cap, routes):
        refused, ok = routes
        return patch.dict("os.environ", {"HERMES_SINGLE_QUERY_SESSION": "1"}), \
               patch.object(delegate_tool, "_get_oneshot_max_children", lambda: cap), \
               patch.object(delegate_tool, "_routes_for", return_value=([refused, ok], None)), \
               patch("tools.delegation_live_log.create_live_transcripts", return_value=(None, [], [])), \
               patch.object(delegate_tool, "_announce_batch"), \
               patch.object(delegate_tool, "_capture_origin", return_value=("", "", None, None, False)), \
               patch.object(delegate_tool, "_build_children", return_value=([], "child construction failed"))

    def _parent_with_prior_spend(self):
        parent = MagicMock()
        parent._delegate_depth = 0
        parent._session_db = None
        parent.session_id = "s1"
        # one prior child already spawned successfully this one-shot run
        parent._oneshot_children_spawned = 1
        return parent

    def test_mixed_route_failure_refunds_only_the_runnable_charge(self):
        """Review sequence: cap 2, prior successful child (spent=1), then a two-task batch
        with one refused route and one runnable one. Charging the runnable route makes
        spent=2; when its construction fails, refunding 1 keeps spent=1 — the prior
        child's usage must survive the failed batch."""
        refused = TaskRoute(creds=dict(_CREDS), routing_cfg={}, error="backend refused this task")
        ok = TaskRoute(creds=dict(_CREDS), routing_cfg={})
        parent = self._parent_with_prior_spend()
        env, cap, routes_m, live, announce, origin, build = self._mixed_route_patches(2, (refused, ok))
        with env, cap, routes_m, live, announce, origin, build:
            result = json.loads(self._delegate_two_tasks(parent, (refused, ok)))
        self.assertEqual(result.get("error"), "child construction failed")
        self.assertEqual(parent._oneshot_children_spawned, 1)

    def test_all_runnable_failure_refunds_the_full_charge(self):
        """Negative control: with every route runnable, charge == refund == len(task_list),
        so the prior spend comes back untouched."""
        ok_a = TaskRoute(creds=dict(_CREDS), routing_cfg={})
        ok_b = TaskRoute(creds=dict(_CREDS), routing_cfg={})
        parent = self._parent_with_prior_spend()
        env, cap, routes_m, live, announce, origin, build = self._mixed_route_patches(5, (ok_a, ok_b))
        with env, cap, routes_m, live, announce, origin, build:
            result = json.loads(self._delegate_two_tasks(parent, (ok_a, ok_b)))
        self.assertEqual(result.get("error"), "child construction failed")
        self.assertEqual(parent._oneshot_children_spawned, 1)


if __name__ == "__main__":
    unittest.main()
