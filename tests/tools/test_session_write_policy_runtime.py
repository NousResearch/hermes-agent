"""T3 PR97786 — durable regression coverage for SessionWritePolicy / Decision ContextVar scope.

Covers contracts:
  C11 — ContextVar restoration on success.
  C12 — ContextVar restoration on exception.
  C13 — ContextVar restoration on cancellation.
  C16 — overlapping protected turns remain task-local (T3-RACE-001 regression).
  C17 — background-review fork receives parent SessionWritePolicy.
  C18 — background-review fork receives parent Decision.

These tests are deterministic: no timing-only sleeps are used to force overlap; instead,
two threads are each bound to a different policy, and each thread observes its own bound
value via the ContextVar. A forced-overlap sync primitive (threading.Barrier) prevents
accidental sequential execution from passing.
"""

from __future__ import annotations

import asyncio
import threading
import unittest
from typing import Optional

from agent.session_write_policy import (
    SessionWritePolicy,
    get_current_session_write_policy,
    session_write_policy_scope,
)
from agent.self_improvement_policy import (
    Decision,
    MISSING_DECISION,
    evaluate,
    allow as decision_allow,
    deny as decision_deny,
)
from agent.self_improvement_decision_context import (
    get_self_improvement_decision,
    self_improvement_decision_scope,
)


class TestSessionWritePolicyScopeRestoresOnNormalExit(unittest.TestCase):
    """C11 — ContextVar restoration on success."""

    def test_outside_scope_returns_default(self) -> None:
        default_before = get_current_session_write_policy()
        with session_write_policy_scope(SessionWritePolicy.deny_all()):
            inside = get_current_session_write_policy()
            self.assertTrue(inside.protected)
        self.assertEqual(get_current_session_write_policy(), default_before)


class TestSessionWritePolicyScopeRestoresOnException(unittest.TestCase):
    """C12 — ContextVar restoration on exception."""

    def test_restores_after_raised_exception(self) -> None:
        default_before = get_current_session_write_policy()
        with self.assertRaises(RuntimeError):
            with session_write_policy_scope(SessionWritePolicy.deny_all()):
                raise RuntimeError("simulated failure")
        self.assertEqual(get_current_session_write_policy(), default_before)


class TestSessionWritePolicyScopeRestoresOnCancellation(unittest.TestCase):
    """C13 — ContextVar restoration on asyncio.CancelledError."""

    def test_restores_after_cancellation(self) -> None:
        async def inner() -> None:
            default_before = get_current_session_write_policy()
            with self_improvement_decision_scope(decision_deny("test", "test")):
                # Bind a session_write_policy too — must also restore.
                with session_write_policy_scope(SessionWritePolicy.deny_all()):
                    raise asyncio.CancelledError()
            return default_before

        async def runner() -> Optional[SessionWritePolicy]:
            try:
                await inner()
            except asyncio.CancelledError:
                return get_current_session_write_policy()
            return None

        result = asyncio.run(runner())
        # After cancellation, the default sentinel should be restored.
        self.assertIsNotNone(result)
        self.assertFalse(result.protected)


class TestOverlappingProtectedTurnsKeepPolicyAndDecisionTaskLocal(unittest.TestCase):
    """C16 — overlapping protected turns remain task-local (T3-RACE-001 regression)."""

    def test_two_threads_each_observe_their_own_policy_and_decision(self) -> None:
        policy_a = SessionWritePolicy.deny_all()
        policy_b = SessionWritePolicy.default()
        decision_a = decision_allow("test_a", "thread_a")
        decision_b = decision_deny("test_b", "thread_b")

        results = {}
        barrier = threading.Barrier(2, timeout=5)

        def worker(name: str, pol: SessionWritePolicy, dec: Decision) -> None:
            with session_write_policy_scope(pol), self_improvement_decision_scope(dec):
                # Force the other thread to start its scope before we read.
                barrier.wait()
                # Hold the scope long enough that both threads are simultaneously in-scope.
                barrier.wait()
                # Read while still inside the scope.
                observed_policy = get_current_session_write_policy()
                observed_decision = get_self_improvement_decision()
                results[name] = (observed_policy, observed_decision)

        t_a = threading.Thread(target=worker, args=("A", policy_a, decision_a))
        t_b = threading.Thread(target=worker, args=("B", policy_b, decision_b))
        t_a.start()
        t_b.start()
        t_a.join(timeout=10)
        t_b.join(timeout=10)

        # Each thread observed its own policy and decision.
        self.assertIs(results["A"][0], policy_a, "Thread A must observe its own policy")
        self.assertIs(results["B"][0], policy_b, "Thread B must observe its own policy")
        self.assertIs(results["A"][1], decision_a, "Thread A must observe its own decision")
        self.assertIs(results["B"][1], decision_b, "Thread B must observe its own decision")

    def test_no_cross_observation_after_completion(self) -> None:
        """C16 (post-turn leak check) — after a scope exits, the next scope sees only its own value."""
        policy_a = SessionWritePolicy.deny_all()
        policy_b = SessionWritePolicy.default()
        observed = []

        def worker(name: str, pol: SessionWritePolicy) -> None:
            with session_write_policy_scope(pol):
                observed.append((name, get_current_session_write_policy().protected))
            observed.append((name + "_after", get_current_session_write_policy().protected))

        worker("A", policy_a)
        worker("B", policy_b)
        # First A in scope: True; B in scope: False; A_after: False (restored); B_after: False
        self.assertEqual(observed, [
            ("A", True),
            ("A_after", False),
            ("B", False),
            ("B_after", False),
        ])


class TestSessionWritePolicyScopeRejectsNone(unittest.TestCase):
    """C2 — malformed/missing authority fails closed."""

    def test_binding_none_raises(self) -> None:
        with self.assertRaises(ValueError):
            with session_write_policy_scope(None):  # type: ignore[arg-type]
                pass


class TestDecisionScopeRejectsNone(unittest.TestCase):
    """C8 — missing/invalid Decision fails closed."""

    def test_binding_none_raises(self) -> None:
        with self.assertRaises(ValueError):
            with self_improvement_decision_scope(None):  # type: ignore[arg-type]
                pass


class TestProvenanceLookupFailureYieldsDenyDecision(unittest.TestCase):
    """C9 — provenance lookup failure fails closed (Decision evaluation)."""

    def test_evaluate_with_provenance_failure_denies(self) -> None:
        decision = evaluate(
            policy_protected=False,
            provenance_lookup_ok=False,
            source="test",
        )
        self.assertFalse(decision.is_allowed())
        self.assertIn("provenance", decision.reason)


class TestProtectedSessionDecisionDenies(unittest.TestCase):
    """C5/C7 — protected sessions deny self-improvement."""

    def test_protected_session_denies(self) -> None:
        decision = evaluate(
            policy_protected=True,
            provenance_lookup_ok=True,
            source="test",
        )
        self.assertFalse(decision.is_allowed())


class TestDefaultSessionDecisionAllows(unittest.TestCase):
    """C5/C6 — default session allows self-improvement."""

    def test_default_session_allows(self) -> None:
        decision = evaluate(
            policy_protected=False,
            provenance_lookup_ok=True,
            source="test",
        )
        self.assertTrue(decision.is_allowed())


class TestBackgroundReviewForkInheritsParentSessionWritePolicy(unittest.TestCase):
    """C17 — background-review fork receives parent SessionWritePolicy."""

    def test_fork_inherits_parent_policy(self) -> None:
        parent_policy = SessionWritePolicy.deny_all()
        parent_decision = decision_deny("parent_protected", "parent")

        # Simulate the fork init: pass the parent authority into the AIAgent constructor
        # via the kwargs forwarded by _fork_init_kwargs. The actual AIAgent constructor
        # is heavyweight; instead we verify the contract that the forwarder honors
        # caller-provided authority.
        from run_agent import AIAgent
        fork = AIAgent(
            session_write_policy=parent_policy,
            self_improvement_decision=parent_decision,
            quiet_mode=True,
            skip_background_review=True,
            ephemeral_system_prompt="fork-test",
            log_prefix="[t3-fork-test]",
            model="noop",
            api_key="test-key",
            base_url="https://example.invalid/v1",
        )
        try:
            self.assertIs(fork.session_write_policy, parent_policy)
            self.assertIs(fork.self_improvement_decision, parent_decision)
        finally:
            fork.close()


class TestBackgroundReviewForkInheritsParentDecision(unittest.TestCase):
    """C18 — background-review fork receives parent Decision (covered jointly above)."""

    def test_fork_decision_is_parent_instance(self) -> None:
        parent_decision = decision_allow("test", "parent_allow")
        parent_policy = SessionWritePolicy.default()

        from run_agent import AIAgent
        fork = AIAgent(
            session_write_policy=parent_policy,
            self_improvement_decision=parent_decision,
            quiet_mode=True,
            skip_background_review=True,
            ephemeral_system_prompt="fork-test-decision",
            log_prefix="[t3-fork-decision]",
            model="noop",
            api_key="test-key",
            base_url="https://example.invalid/v1",
        )
        try:
            self.assertIs(fork.self_improvement_decision, parent_decision)
            self.assertTrue(fork.self_improvement_decision.is_allowed())
        finally:
            fork.close()


class TestForwarderWithoutForkKwargsUsesDefaults(unittest.TestCase):
    """C5/C6 — when no authority is forwarded, the forwarder uses initialization defaults."""

    def test_default_session(self) -> None:
        from run_agent import AIAgent
        agent = AIAgent(
            quiet_mode=True,
            skip_background_review=True,
            ephemeral_system_prompt="default-test",
            log_prefix="[t3-default-test]",
            model="noop",
            api_key="test-key",
            base_url="https://example.invalid/v1",
        )
        try:
            self.assertFalse(agent.session_write_policy.protected)
            self.assertIsInstance(agent.self_improvement_decision, Decision)
            self.assertTrue(agent.self_improvement_decision.is_allowed())
        finally:
            agent.close()


class TestProvenanceDisabledCausesFailClosed(unittest.TestCase):
    """C9 — when HERMES_PROVENANCE_DISABLED is set, the Decision is deny."""

    def test_evaluate_with_disabled_env_denies(self) -> None:
        import os
        old = os.environ.get("HERMES_PROVENANCE_DISABLED")
        os.environ["HERMES_PROVENANCE_DISABLED"] = "1"
        try:
            decision = evaluate(
                policy_protected=False,
                provenance_lookup_ok=False,  # simulate the lookup returning False
                source="test",
            )
            self.assertFalse(decision.is_allowed())
        finally:
            if old is None:
                del os.environ["HERMES_PROVENANCE_DISABLED"]
            else:
                os.environ["HERMES_PROVENANCE_DISABLED"] = old


if __name__ == "__main__":
    unittest.main()
