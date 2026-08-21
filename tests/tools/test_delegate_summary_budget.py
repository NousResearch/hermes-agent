"""Tests for subagent summary budgeting (PR #9126).

delegate_task caps subagent summaries against the parent's remaining context
headroom (split across the batch) before they enter the parent's context, and
spills the full text to disk so nothing is lost. This guards the
compression/429 death spiral that batch fan-out could trigger by returning N
full summaries verbatim into the parent.
"""

import os
import tempfile

import pytest

import tools.delegate_tool as dt


class _FakeCompressor:
    def __init__(self, context_length, max_tokens):
        self.context_length = context_length
        self.max_tokens = max_tokens


class _FakeParent:
    def __init__(self, context_length, used_tokens, max_tokens):
        self.context_compressor = _FakeCompressor(context_length, max_tokens)
        self.session_prompt_tokens = used_tokens


def test_small_summaries_pass_through_untouched():
    parent = _FakeParent(context_length=200_000, used_tokens=10_000, max_tokens=8_000)
    results = [
        {"task_index": 0, "summary": "short result A", "status": "completed"},
        {"task_index": 1, "summary": "short result B", "status": "completed"},
    ]
    dt._apply_summary_budget(results, parent)
    assert results[0]["summary"] == "short result A"
    assert "summary_truncated" not in results[0]
    assert "summary_truncated" not in results[1]


def test_batch_overflow_trimmed_and_spilled_losslessly(monkeypatch):
    # Isolate spill directory to a temp HERMES_HOME.
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("HERMES_HOME", os.path.join(td, ".hermes"))
        # Distinct head + tail markers so we can prove the tail survives.
        big = "HEAD_MARKER\n" + ("X" * 50_000) + "\nTAIL_MARKER"
        # Parent nearly full (120k/131k) → tiny headroom → aggressive trim.
        parent = _FakeParent(context_length=131_000, used_tokens=120_000, max_tokens=8_000)
        results = [
            {"task_index": i, "summary": big, "status": "completed"} for i in range(5)
        ]
        dt._apply_summary_budget(results, parent)
        for r in results:
            assert r["summary_truncated"] is True
            assert len(r["summary"]) < len(big)
            # Head+tail window: both ends survive in-context.
            assert "HEAD_MARKER" in r["summary"]
            assert "TAIL_MARKER" in r["summary"]
            path = r.get("summary_full_path")
            assert path and os.path.exists(path)
            # The spill file holds the FULL original text — nothing is lost.
            with open(path, encoding="utf-8") as fh:
                assert fh.read() == big
            # The footer points the parent at the full version with an offset.
            assert "read_file" in r["summary"]
            assert "offset=" in r["summary"]
            # Spilled into the delegation cache (mounted into remote backends).
            assert os.path.join("cache", "delegation") in path


def test_empty_results_is_noop():
    # No summaries → nothing to do, must not raise.
    dt._apply_summary_budget([], _FakeParent(131_000, 1_000, 8_000))
    dt._apply_summary_budget(
        [{"task_index": 0, "status": "failed", "summary": None}],
        _FakeParent(131_000, 1_000, 8_000),
    )


class _LiveParent:
    """Parent whose compressor reports real occupancy, alongside the
    session-CUMULATIVE billing counter that grows on every API call."""

    def __init__(
        self,
        context_length,
        session_prompt_tokens,
        max_tokens,
        last_prompt_tokens=0,
        last_real_prompt_tokens=0,
    ):
        compressor = _FakeCompressor(context_length, max_tokens)
        compressor.last_prompt_tokens = last_prompt_tokens
        compressor.last_real_prompt_tokens = last_real_prompt_tokens
        self.context_compressor = compressor
        self.session_prompt_tokens = session_prompt_tokens


def test_budget_tracks_occupancy_not_cumulative_spend():
    """The budget must size against what is IN the context, not lifetime spend.

    ``session_prompt_tokens`` accumulates on every API call and is never reset
    by compaction, so on a long-but-small conversation it crosses
    ``context_length`` after a handful of ordinary tool-loop iterations. Sizing
    the budget off it collapses every subagent summary to the floor for the
    rest of the session, even though the parent is nearly empty.
    """
    # Conversation occupies 20k of a 131k window (~15% full) and stays there,
    # but 20 API calls have been billed.
    parent = _LiveParent(
        context_length=131_072,
        session_prompt_tokens=400_000,
        max_tokens=8_000,
        last_prompt_tokens=20_000,
    )
    budget = dt._parent_summary_char_budget(parent, n_summaries=1)

    assert budget is not None
    # Real headroom is 131,072 - 20,000 - 8,000 = 103,072 tokens.
    assert budget == (103_072 * 4) // 2
    assert budget > dt._MIN_SUMMARY_CHARS

    # The cumulative counter must not move the budget at all.
    later = _LiveParent(
        context_length=131_072,
        session_prompt_tokens=4_000_000,
        max_tokens=8_000,
        last_prompt_tokens=20_000,
    )
    assert dt._parent_summary_char_budget(later, n_summaries=1) == budget


def test_summary_survives_when_parent_has_headroom(monkeypatch):
    """End-to-end: a report that fits must reach the parent untrimmed."""
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("HERMES_HOME", os.path.join(td, ".hermes"))
        report = "HEAD_MARKER\n" + ("SUBAGENT FINDINGS. " * 900) + "\nTAIL_MARKER"
        parent = _LiveParent(
            context_length=131_072,
            session_prompt_tokens=400_000,   # 20 API calls billed
            max_tokens=8_000,
            last_prompt_tokens=20_000,       # but only 15% of the window in use
        )
        results = [{"task_index": 0, "summary": report, "status": "completed"}]
        dt._apply_summary_budget(results, parent)

        assert results[0]["summary"] == report
        assert "summary_truncated" not in results[0]


def test_floor_still_enforced_when_context_genuinely_full():
    """The #9126 overflow guard must survive the occupancy fix."""
    parent = _LiveParent(
        context_length=131_072,
        session_prompt_tokens=125_000,
        max_tokens=8_000,
        last_prompt_tokens=125_000,   # genuinely nearly full
    )
    assert dt._parent_summary_char_budget(parent, 3) == dt._MIN_SUMMARY_CHARS


def test_post_compaction_sentinel_falls_back_to_last_real():
    """Right after a compaction ``last_prompt_tokens`` is parked at -1.

    The budget must fall through to ``last_real_prompt_tokens`` rather than
    treating the sentinel as "empty context" (or reaching the cumulative
    counter, which would report the parent as hopelessly over budget).
    """
    parent = _LiveParent(
        context_length=131_072,
        session_prompt_tokens=900_000,
        max_tokens=8_000,
        last_prompt_tokens=-1,
        last_real_prompt_tokens=30_000,
    )
    budget = dt._parent_summary_char_budget(parent, n_summaries=1)
    assert budget == ((131_072 - 30_000 - 8_000) * 4) // 2


def test_cumulative_fallback_is_clamped_to_context_length():
    """With no compressor telemetry the cumulative counter is the last resort,
    but it must not drive headroom arbitrarily negative — clamped, the worst it
    can say is "full"."""
    parent = _LiveParent(
        context_length=131_072,
        session_prompt_tokens=10_000_000,
        max_tokens=8_000,
    )
    assert dt._parent_used_tokens(
        parent, parent.context_compressor, 131_072
    ) == 131_072
    assert dt._parent_summary_char_budget(parent, 1) == dt._MIN_SUMMARY_CHARS
