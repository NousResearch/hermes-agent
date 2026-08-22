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


def test_trim_emits_single_batch_warning_naming_dynamic_binding(monkeypatch, caplog):
    # Parent nearly full → dynamic budget (floored at 2000) binds below the
    # static 24000 ceiling.
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("HERMES_HOME", os.path.join(td, ".hermes"))
        parent = _FakeParent(context_length=131_000, used_tokens=120_000, max_tokens=8_000)
        big = "HEAD\n" + ("X" * 50_000) + "\nTAIL"
        results = [{"task_index": i, "summary": big, "status": "completed"} for i in range(5)]
        with caplog.at_level("WARNING", logger="tools.delegate_tool"):
            dt._apply_summary_budget(results, parent)
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1  # one line per batch, not per summary
    msg = warnings[0].getMessage()
    assert "trimmed 5 subagent summaries" in msg
    assert "binding constraint: dynamic context-headroom budget" in msg


def test_trim_warning_names_static_binding_when_ceiling_is_lower(monkeypatch, caplog):
    # Huge parent context → dynamic budget far above the static ceiling.
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("HERMES_HOME", os.path.join(td, ".hermes"))
        parent = _FakeParent(context_length=2_000_000, used_tokens=10_000, max_tokens=8_000)
        big = "HEAD\n" + ("X" * 50_000) + "\nTAIL"
        results = [{"task_index": 0, "summary": big, "status": "completed"}]
        with caplog.at_level("WARNING", logger="tools.delegate_tool"):
            dt._apply_summary_budget(results, parent)
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "binding constraint: static delegation.max_summary_chars ceiling" in (
        warnings[0].getMessage()
    )


def test_min_summary_chars_floors_dynamic_budget(monkeypatch):
    # Without the floor this parent/batch lands on the built-in 2000-char
    # dynamic floor; min_summary_chars=8000 must raise the cap.
    monkeypatch.setattr(
        dt, "_load_config",
        lambda: {"max_summary_chars": 24000, "min_summary_chars": 8000},
    )
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("HERMES_HOME", os.path.join(td, ".hermes"))
        parent = _FakeParent(context_length=131_000, used_tokens=120_000, max_tokens=8_000)
        big = "HEAD\n" + ("X" * 50_000) + "\nTAIL"
        results = [{"task_index": i, "summary": big, "status": "completed"} for i in range(5)]
        dt._apply_summary_budget(results, parent)
        for r in results:
            assert r["summary_truncated"] is True
            # Trimmed to ~8000 (75/25 head/tail + footer), not the 2000 floor.
            assert len(r["summary"]) > 6000
            assert len(r["summary"]) < 12000
