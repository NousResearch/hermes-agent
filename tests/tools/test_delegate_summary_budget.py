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


def test_dynamic_budget_shrinks_as_batch_grows():
    def cap_for(n):
        return dt._parent_summary_char_budget(
            _FakeParent(131_000, 30_000, 8_000), n
        )

    c1, c5, c20 = cap_for(1), cap_for(5), cap_for(20)
    assert c1 is not None and c5 is not None and c20 is not None
    # More children → smaller per-summary slice of the same headroom.
    assert c1 > c5 > c20


def test_floor_enforced_when_parent_over_budget():
    # Parent already over its context budget → each summary gets only the floor.
    budget = dt._parent_summary_char_budget(
        _FakeParent(131_000, 200_000, 8_000), 3
    )
    assert budget == dt._MIN_SUMMARY_CHARS


def test_unknown_context_falls_back_to_static_ceiling(monkeypatch):
    class _Bare:
        pass

    # No compressor → dynamic budget is unknowable.
    assert dt._parent_summary_char_budget(_Bare(), 3) is None

    # But the static delegation.max_summary_chars ceiling still trims.
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("HERMES_HOME", os.path.join(td, ".hermes"))
        results = [{"task_index": 0, "summary": "Y" * 40_000, "status": "completed"}]
        dt._apply_summary_budget(results, _Bare())
        assert results[0]["summary_truncated"] is True
        assert len(results[0]["summary"]) < 40_000


def test_disabled_static_ceiling_and_unknown_context_leaves_summary_intact(monkeypatch):
    class _Bare:
        pass

    # Both caps off: static ceiling 0 (disabled) AND no compressor (no dynamic).
    monkeypatch.setattr(dt, "_load_config", lambda: {"max_summary_chars": 0})
    results = [{"task_index": 0, "summary": "Z" * 40_000, "status": "completed"}]
    dt._apply_summary_budget(results, _Bare())
    assert "summary_truncated" not in results[0]
    assert len(results[0]["summary"]) == 40_000


def test_empty_results_is_noop():
    # No summaries → nothing to do, must not raise.
    dt._apply_summary_budget([], _FakeParent(131_000, 1_000, 8_000))
    dt._apply_summary_budget(
        [{"task_index": 0, "status": "failed", "summary": None}],
        _FakeParent(131_000, 1_000, 8_000),
    )
