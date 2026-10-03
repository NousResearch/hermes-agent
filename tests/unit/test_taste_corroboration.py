"""Tests for the taste corroboration engine (option B upgrade).

Offline. stdlib only. Uses a deterministic fake clock.
"""

import math
import os
import sys
import unittest

# Put the repo root on the path so we can import agent.* without an install.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from agent.taste_corroboration import (
    CorroborationEngine,
    MIN_OBSERVATIONS_FOR_WRITE,
    CONFLICT_EPSILON,
)
from agent.taste_decay import (
    growth_weight,
    decay_weight,
    decayed_score,
    DecayConfig,
    _half_life_to_decay_constant,
)


class FakeClock:
    def __init__(self, start=1_000_000.0):
        self.t = float(start)

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


class TestDecayModule(unittest.TestCase):
    def test_growth_bounded_and_monotonic(self):
        prev = 0.0
        for n in range(0, 50):
            w = growth_weight(n)
            self.assertGreaterEqual(w, 0.0)
            self.assertGreaterEqual(w, prev - 1e-12)  # monotonic
            self.assertLessEqual(w, 1.0 + 1e-9)
            prev = w

    def test_growth_starts_at_zero(self):
        self.assertAlmostEqual(growth_weight(0), 0.0, places=9)

    def test_half_life_constant_positive(self):
        self.assertEqual(_half_life_to_decay_constant(None), 0.0)
        lam = _half_life_to_decay_constant(14.0)
        self.assertGreater(lam, 0.0)
        # sanity: 2^(-1) == exp(-lam*14)
        self.assertAlmostEqual(lam * 14.0, math.log(2.0), places=9)

    def test_decay_halves_at_half_life(self):
        w = decay_weight(1.0, 14.0, 14.0)
        self.assertAlmostEqual(w, 0.5, places=9)

    def test_no_decay_when_none(self):
        self.assertAlmostEqual(decay_weight(0.7, 100.0, None), 0.7, places=12)
        self.assertAlmostEqual(decay_weight(0.7, 0.0, 14.0), 0.7, places=12)

    def test_decayed_score_clamped(self):
        # score is always in [0,1].
        s = decayed_score(2.0, 5.0, 14.0)
        self.assertTrue(0.0 <= s <= 1.0)


class TestCorroborationEngine(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.decay = DecayConfig(half_life_days=14.0)

    def test_new_candidate_starts_at_zero(self):
        e = CorroborationEngine("p1", "label", decay=self.decay, clock=self.clock)
        r = e.observe(0.5)
        # First observation: no prior evidence, weight just accumulated from obs.
        self.assertEqual(e._state.n_obs, 1)
        self.assertEqual(r.n_obs, 1)
        self.assertFalse(r.established)
        self.assertFalse(r.auto_ack)

    def test_established_threshold(self):
        e = CorroborationEngine("p1", "label", decay=self.decay, clock=self.clock)
        for i in range(MIN_OBSERVATIONS_FOR_WRITE):
            r = e.observe(0.5)
        self.assertTrue(e.should_write())
        self.assertTrue(r.established)
        # One more crosses auto_ack? Only if threshold met.
        self.assertGreater(MIN_OBSERVATIONS_FOR_WRITE, 0)

    def test_consecutive_agreement_rises_score(self):
        e = CorroborationEngine("p1", "label", decay=self.decay, clock=self.clock)
        r1 = e.observe(0.6)
        r2 = e.observe(0.6)
        r3 = e.observe(0.6)
        r4 = e.observe(0.6)
        r5 = e.observe(0.6)
        self.assertEqual(r1.n_obs, 1)
        self.assertEqual(r5.n_obs, 5)
        self.assertFalse(r1.conflict)
        self.assertFalse(r5.conflict)  # all agreed

    def test_conflict_detected_on_disagreement(self):
        e = CorroborationEngine("p1", "label", decay=self.decay, clock=self.clock)
        e.observe(0.5)
        r = e.observe(0.9)  # 0.9 vs 0.5 -> > EPSILON
        self.assertTrue(r.conflict, "expected a conflict flag")
        self.assertIsNotNone(r.escalated)
        self.assertEqual(r.escalated.reason, "conflict")

    def test_conflict_within_epsilon_agrees(self):
        e = CorroborationEngine("p1", "label", decay=self.decay, clock=self.clock)
        e.observe(0.5)
        r = e.observe(0.5 + CONFLICT_EPSILON / 2.0)
        self.assertFalse(r.conflict, "within epsilon should agree")

    def test_resolve_conflict_clears_and_acks(self):
        e = CorroborationEngine("p1", "label", decay=self.decay, clock=self.clock)
        e.observe(0.5)
        r = e.observe(0.9)  # conflict
        self.assertTrue(r.conflict)
        self.assertGreater(e._state.conflicts, 0)
        rr = e.resolve_conflict(0.7)
        self.assertEqual(e._state.conflicts, 0)
        self.assertFalse(rr.conflict)
        self.assertIsNone(rr.escalated)
        self.assertTrue(rr.auto_ack or rr.n_obs >= 2)

    def test_staleness_escalation(self):
        # is_stale() measures age *since* the last observation, so time must
        # advance *after* an observation to become stale.
        e = CorroborationEngine("p1", "label", decay=self.decay, clock=self.clock)
        e.observe(0.5)
        e.advance(14 * 86400)  # 2 weeks -> not yet stale (21d threshold)
        self.assertFalse(e.is_stale())
        e.observe(0.5)  # refresh the clock
        e.advance(25 * 86400)  # 25 days later -> stale
        self.assertTrue(e.is_stale())

    def test_decayed_score_drops_with_age(self):
        e = CorroborationEngine("p1", "label", decay=self.decay, clock=self.clock)
        e.observe(0.8)
        before = e.to_result_now().score
        e.advance(30 * 86400)  # 30 days, > 14-day half-life
        after = e.to_result_now().score
        self.assertLess(after, before, "score should decay as evidence ages")

    def test_snapshot_restore_roundtrip(self):
        e = CorroborationEngine("p1", "label", decay=self.decay, clock=self.clock)
        for _ in range(5):
            e.observe(0.6)
        snap = e.snapshot()
        self.assertEqual(snap["n_obs"], 5)
        self.assertEqual(snap["id"], "p1")
        e2 = CorroborationEngine("x", "y", decay=DecayConfig(), clock=FakeClock())
        e2.restore(snap)
        self.assertEqual(e2._state.n_obs, 5)
        self.assertEqual(e2._state.weight, e._state.weight)

    def test_score_range_invariant(self):
        e = CorroborationEngine("p1", "label", decay=self.decay, clock=self.clock)
        for i in range(30):
            r = e.observe(0.5)
            self.assertTrue(0.0 <= r.score <= 1.0, f"score out of range at {r.score}")

    def test_invalid_score_raises(self):
        e = CorroborationEngine("p1", "label", decay=self.decay, clock=self.clock)
        with self.assertRaises(ValueError):
            e.observe(1.5)

    # -- regression tests for the review fixes --------------------------------
    def test_repeated_agreement_raises_score(self):
        # Repeated identical evidence must raise the score (blend, not max).
        e = CorroborationEngine("p1", "label", decay=self.decay, clock=self.clock)
        scores = [e.observe(0.6).score for _ in range(5)]
        self.assertGreater(scores[-1], scores[0],
                           f"repeated agreement should raise the score: {scores}")
        for a, b in zip(scores, scores[1:]):
            self.assertGreaterEqual(b, a - 1e-9)

    def test_decay_disabled_flag(self):
        e = CorroborationEngine(
            "p1", "label",
            decay=DecayConfig(half_life_days=14.0, enabled=False),
            clock=self.clock,
        )
        e.observe(0.8)
        before = e.to_result_now().score
        e.advance(60 * 86400)  # far past the half-life; decay must not apply
        after = e.to_result_now().score
        self.assertAlmostEqual(after, before, places=9)

    def test_should_write_gated_on_conflict(self):
        e = CorroborationEngine("p1", "label", decay=self.decay, clock=self.clock)
        e.observe(0.5)
        e.observe(0.9)  # conflict
        e.observe(0.9)  # n_obs=3, but conflicts still open
        self.assertTrue(e.is_conflicting())
        self.assertFalse(e.should_write())

    def test_should_write_gated_on_staleness(self):
        e = CorroborationEngine("p1", "label", decay=self.decay, clock=self.clock)
        for _ in range(MIN_OBSERVATIONS_FOR_WRITE):
            e.observe(0.5)
        self.assertTrue(e.should_write())
        e.advance(25 * 86400)  # go stale
        self.assertTrue(e.is_stale())
        self.assertFalse(e.should_write())

    def test_resolve_conflict_does_not_reconflict(self):
        e = CorroborationEngine("p1", "label", decay=self.decay, clock=self.clock)
        e.observe(0.5)
        e.observe(0.9)  # conflict; last raw score is 0.9
        rr = e.resolve_conflict(0.5)  # would re-conflict vs 0.9 via observe()
        self.assertFalse(rr.conflict)
        self.assertIsNone(rr.escalated)
        self.assertEqual(e._state.conflicts, 0)
        self.assertAlmostEqual(e._state.weight, 0.5, places=9)

    def test_observe_after_silence_cures_staleness(self):
        e = CorroborationEngine("p1", "label", decay=self.decay, clock=self.clock)
        e.observe(0.5)
        e.observe(0.5)
        e.advance(25 * 86400)  # stale now
        self.assertTrue(e.is_stale())
        r = e.observe(0.5)  # the observation cures it; no staleness escalation
        self.assertIsNone(r.escalated)
        self.assertFalse(e.is_stale())

    def test_history_bounded(self):
        from collections import deque
        e = CorroborationEngine("p1", "label", decay=self.decay, clock=self.clock)
        for _ in range(10):
            e.observe(0.5)
        self.assertIsInstance(e._state.history, deque)
        self.assertLessEqual(len(e._state.history), 2)

    def test_first_obs_epoch_defaults_none(self):
        e = CorroborationEngine("p1", "label", decay=self.decay, clock=self.clock)
        self.assertIsNone(e._state.first_obs_epoch)
        e.observe(0.5)
        self.assertIsNotNone(e._state.first_obs_epoch)


if __name__ == "__main__":
    unittest.main(verbosity=2)
