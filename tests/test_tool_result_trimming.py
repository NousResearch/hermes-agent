"""Tests for insertion-time tool result trimming (Issue #415).

Verifies that _trim_tool_result() correctly applies soft (head+tail) and hard
(head-only) trimming at insertion time, before results enter the message array.

Extracts only the target function to avoid heavy run_agent import deps.
"""

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def trim_fn():
    """Import _trim_tool_result without pulling in run_agent's heavy deps."""
    src = Path(__file__).resolve().parent.parent / "run_agent.py"
    source = src.read_text()

    # Extract the function and its constants from source
    # Find the block between the marker comments
    start = source.index("# Tool result trimming")
    end = source.index("\nclass AIAgent:")
    block = source[start:end]

    # Execute just the function definition in an isolated namespace
    ns = {}
    exec(block, ns)
    return ns["_trim_tool_result"]


class TestToolResultTrimming:
    """Test the two-stage trimming: soft (head+tail) then hard (head-only)."""

    def test_short_result_unchanged(self, trim_fn):
        """Results under the soft limit pass through unmodified."""
        text = "Hello, world!"
        assert trim_fn(text) == text

    def test_empty_result_unchanged(self, trim_fn):
        """Empty results pass through."""
        assert trim_fn("") == ""

    def test_none_result_unchanged(self, trim_fn):
        """None input returns None."""
        assert trim_fn(None) is None

    def test_exact_soft_limit_unchanged(self, trim_fn):
        """Result exactly at the soft limit is NOT trimmed."""
        text = "x" * 12_000
        assert trim_fn(text, soft_limit=12_000) == text

    def test_soft_trim_preserves_head_and_tail(self, trim_fn):
        """Soft trim keeps the first N and last N chars, drops the middle."""
        head = "HEAD" * 1000   # 4000 chars
        middle = "M" * 20_000  # will be dropped
        tail = "TAIL" * 1000   # 4000 chars
        text = head + middle + tail

        result = trim_fn(text, soft_limit=12_000, head_chars=4_000, tail_chars=4_000)

        assert result.startswith("HEAD" * 1000)
        assert result.endswith("TAIL" * 1000)
        assert "Truncated" in result
        assert len(result) < len(text)

    def test_soft_trim_indicator_shows_counts(self, trim_fn):
        """The trim indicator includes original length and kept sizes."""
        text = "A" * 20_000
        result = trim_fn(text, soft_limit=10_000, head_chars=3_000, tail_chars=3_000)
        assert "20,000 total" in result
        assert "3,000" in result

    def test_hard_cap_fires_as_safety_net(self, trim_fn):
        """Results exceeding the hard limit get head-truncated."""
        text = "Z" * 200_000
        result = trim_fn(text, soft_limit=0, hard_limit=100_000)
        assert len(result) < 200_000
        assert "Truncated" in result
        assert "200,000" in result

    def test_soft_trim_disabled_when_zero(self, trim_fn):
        """Setting soft_limit=0 disables soft trimming."""
        text = "A" * 50_000
        result = trim_fn(text, soft_limit=0, hard_limit=100_000)
        assert result == text

    def test_soft_then_hard_both_apply(self, trim_fn):
        """When soft-trimmed result still exceeds hard limit, hard cap applies."""
        text = "X" * 500_000
        result = trim_fn(
            text, soft_limit=200_000, hard_limit=100_000,
            head_chars=80_000, tail_chars=80_000
        )
        assert len(result) <= 100_000 + 200

    def test_head_tail_clamped_to_half_soft_limit(self, trim_fn):
        """If head_chars or tail_chars exceed soft_limit/2, they are clamped."""
        text = "A" * 20_000
        result = trim_fn(text, soft_limit=6_000, head_chars=10_000, tail_chars=10_000)
        assert result.startswith("A" * 3_000)
        assert result.endswith("A" * 3_000)

    def test_default_values_work(self, trim_fn):
        """Default args use sensible 12K soft limit."""
        short = "x" * 10_000
        assert trim_fn(short) == short

        long = "x" * 50_000
        result = trim_fn(long)
        assert len(result) < 50_000
        assert "Truncated" in result

    def test_unicode_content_handled(self, trim_fn):
        """Unicode content is trimmed by char count, not byte count."""
        text = "\U0001F600" * 20_000
        result = trim_fn(text, soft_limit=12_000)
        assert len(result) < 20_000

    def test_multiline_content(self, trim_fn):
        """Multi-line content trims correctly, preserving first lines."""
        lines = "\n".join(f"line {i}: {'x' * 100}" for i in range(200))
        result = trim_fn(lines, soft_limit=5_000)
        assert "Truncated" in result
        assert "line 0:" in result
