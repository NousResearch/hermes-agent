"""Tests for agent.topic_detection.

Uses the 18 real session transitions from session 20260806_005634_165184
to verify the AND-gate detection strategy matches the calibrated results:
- S6-only: 88.9% accuracy
- LLM-only: 83.3% accuracy
- AND gate: 94.4% accuracy
"""

import pytest
from unittest.mock import MagicMock

from agent.topic_detection import (
    S6_THRESHOLD,
    INTENT_PREFIX_LEN,
    compute_s6_cosine,
    detect_topic_shift,
    _cosine,
)


class TestCosine:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert _cosine(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert _cosine([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert _cosine([0, 0], [1, 1]) == 0.0


class TestComputeS6Cosine:
    def test_with_embed_fn(self):
        embed_fn = MagicMock(return_value=[1.0, 0.0])
        result = compute_s6_cosine("hello world", "hi there", embed_fn)
        assert result == pytest.approx(1.0)  # identical embeddings

    def test_no_embed_fn(self):
        assert compute_s6_cosine("hello", "world", None) is None

    def test_empty_messages(self):
        embed_fn = MagicMock()
        assert compute_s6_cosine("", "world", embed_fn) is None
        assert compute_s6_cosine("hello", "", embed_fn) is None

    def test_truncates_to_prefix_len(self):
        long_msg = "a" * 500
        embed_fn = MagicMock(return_value=[1.0])
        compute_s6_cosine(long_msg, "b", embed_fn)
        # embed_fn should be called with truncated text
        called_arg = embed_fn.call_args_list[0][0][0]
        assert len(called_arg) <= INTENT_PREFIX_LEN

    def test_embed_fn_returns_none(self):
        embed_fn = MagicMock(return_value=None)
        assert compute_s6_cosine("hello", "world", embed_fn) is None

    def test_embed_fn_raises(self):
        embed_fn = MagicMock(side_effect=Exception("boom"))
        assert compute_s6_cosine("hello", "world", embed_fn) is None


class TestDetectTopicShift:
    def test_first_turn_no_prev_msg(self):
        result = detect_topic_shift("hello world")
        assert result["topic_continuation"] is False
        assert result["method"] == "first_turn"
        assert result["confidence"] == 1.0

    def test_first_turn_with_none_prev(self):
        result = detect_topic_shift("hello world", prev_msg=None)
        assert result["topic_continuation"] is False
        assert result["method"] == "first_turn"

    def test_both_signals_agree_same(self):
        embed_fn = MagicMock(return_value=[1.0, 0.0])  # cosine=1.0 > threshold
        result = detect_topic_shift(
            "hello", prev_msg="hello",
            llm_topic_continuation=True,
            embed_fn=embed_fn,
        )
        assert result["topic_continuation"] is True
        assert result["method"] == "and_gate"

    def test_both_signals_agree_shift(self):
        # cosine ≈ 0 < threshold, llm says False
        embed_fn = MagicMock(side_effect=[[1, 0], [0, 1]])
        result = detect_topic_shift(
            "hello", prev_msg="world",
            llm_topic_continuation=False,
            embed_fn=embed_fn,
        )
        assert result["topic_continuation"] is False
        assert result["method"] == "and_gate"

    def test_and_gate_llm_same_s6_shift(self):
        """LLM says same, S6 says shift → shift wins (AND gate)."""
        embed_fn = MagicMock(side_effect=[[1, 0], [0, 1]])  # cosine=0
        result = detect_topic_shift(
            "hello", prev_msg="different",
            llm_topic_continuation=True,
            embed_fn=embed_fn,
        )
        assert result["topic_continuation"] is False
        assert result["method"] == "and_gate"
        assert result["confidence"] == pytest.approx(0.70)

    def test_and_gate_llm_shift_s6_same(self):
        """LLM says shift, S6 says same → shift wins."""
        embed_fn = MagicMock(return_value=[1.0, 0.0])  # cosine=1.0
        result = detect_topic_shift(
            "hello", prev_msg="hello",
            llm_topic_continuation=False,
            embed_fn=embed_fn,
        )
        assert result["topic_continuation"] is False
        assert result["method"] == "and_gate"

    def test_llm_only_no_embed_fn(self):
        result = detect_topic_shift(
            "hello", prev_msg="world",
            llm_topic_continuation=True,
        )
        assert result["topic_continuation"] is True
        assert result["method"] == "llm_only"
        assert result["confidence"] == pytest.approx(0.83)

    def test_s6_only_no_llm(self):
        embed_fn = MagicMock(return_value=[1.0, 0.0])
        result = detect_topic_shift(
            "hello", prev_msg="hello",
            embed_fn=embed_fn,
        )
        assert result["topic_continuation"] is True
        assert result["method"] == "s6_only"
        assert result["confidence"] == pytest.approx(0.89)

    def test_no_signals_fallback(self):
        """Neither LLM nor embedding available → fallback (shift/full)."""
        result = detect_topic_shift(
            "hello", prev_msg="world",
        )
        assert result["topic_continuation"] is False
        assert result["method"] == "fallback"
        assert result["confidence"] == 0.0

    def test_custom_threshold(self):
        embed_fn = MagicMock(return_value=[1.0, 0.0])
        # With threshold=1.5, cosine=1.0 → shift
        result = detect_topic_shift(
            "hello", prev_msg="hello",
            embed_fn=embed_fn,
            s6_threshold=1.5,
        )
        assert result["topic_continuation"] is False


class TestCalibratedData:
    """Replay the 18 real transitions to verify accuracy targets."""

    # S6 cosine values (first-100-char) from real data
    S6_VALUES = {
        (1,2):0.529,(2,3):0.505,(3,4):0.741,(4,5):0.380,(5,6):0.350,
        (6,7):0.558,(7,8):0.695,(8,9):0.462,(9,10):0.489,(10,11):0.635,
        (11,12):0.608,(12,13):0.589,(13,14):0.508,(14,15):0.468,
        (15,16):0.535,(16,17):0.521,(17,18):0.547,(18,19):0.533,
    }

    # LLM topic_continuation judgments from real data
    LLM_TC = {
        2:True,3:True,4:True,5:False,6:False,7:True,
        8:True,9:True,10:True,11:True,12:True,13:True,
        14:True,15:True,16:False,17:True,18:True,19:True,
    }

    # Ground truth labels
    LABELS = {
        (1,2):"same",(2,3):"same",(3,4):"same",(4,5):"shift",(5,6):"shift",
        (6,7):"shift",(7,8):"same",(8,9):"shift",(9,10):"same",
        (10,11):"same",(11,12):"same",(12,13):"same",(13,14):"same",
        (14,15):"shift",(15,16):"shift",(16,17):"same",(17,18):"same",
        (18,19):"same",
    }

    def _make_embed_fn(self, s6_value):
        """Create an embed_fn that returns vectors with given cosine sim."""
        # Simple: use unit vectors at angle matching the desired cosine
        import math
        angle = math.acos(max(-1, min(1, s6_value)))
        return MagicMock(side_effect=[
            [1.0, 0.0],  # first call: intent_a
            [math.cos(angle), math.sin(angle)],  # second call: intent_b
        ])

    def test_and_gate_accuracy(self):
        """AND gate should achieve >= 90% accuracy."""
        correct = 0
        total = 0
        for i in range(1, 19):
            pair = (i, i + 1)
            s6 = self.S6_VALUES[pair]
            tc = self.LLM_TC[i + 1]
            label = self.LABELS[pair]

            embed_fn = self._make_embed_fn(s6)
            result = detect_topic_shift(
                "dummy", prev_msg="dummy",
                llm_topic_continuation=tc,
                embed_fn=embed_fn,
            )
            predicted = "same" if result["topic_continuation"] else "shift"
            if predicted == label:
                correct += 1
            total += 1

        accuracy = correct / total
        assert accuracy >= 0.90, f"AND gate accuracy {accuracy:.1%} < 90%"

    def test_s6_only_accuracy(self):
        """S6 alone should achieve >= 85% accuracy."""
        correct = 0
        total = 0
        for pair, s6 in self.S6_VALUES.items():
            label = self.LABELS[pair]
            embed_fn = self._make_embed_fn(s6)
            result = detect_topic_shift(
                "dummy", prev_msg="dummy",
                embed_fn=embed_fn,
            )
            predicted = "same" if result["topic_continuation"] else "shift"
            if predicted == label:
                correct += 1
            total += 1

        accuracy = correct / total
        assert accuracy >= 0.85, f"S6-only accuracy {accuracy:.1%} < 85%"

    def test_llm_only_accuracy(self):
        """LLM alone should achieve >= 75% accuracy."""
        correct = 0
        total = 0
        for i in range(1, 19):
            pair = (i, i + 1)
            tc = self.LLM_TC[i + 1]
            label = self.LABELS[pair]

            result = detect_topic_shift(
                "dummy", prev_msg="dummy",
                llm_topic_continuation=tc,
            )
            predicted = "same" if result["topic_continuation"] else "shift"
            if predicted == label:
                correct += 1
            total += 1

        accuracy = correct / total
        assert accuracy >= 0.75, f"LLM-only accuracy {accuracy:.1%} < 75%"
