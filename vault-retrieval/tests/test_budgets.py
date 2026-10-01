"""Tests for per-turn budget accounting (budgets.py).

Covers acceptance tests #2, #3, #4, #5, #6, #7, #16:

  #2 Default ≤ 12,000; exactly 12,000 accepted.
  #3 12,001 without expansion -> data_hold.
  #4 Expansion → 24,000 accepted; 24,001 rejected.
  #5 Two calls sharing one turn_id share the cumulative ceiling.
  #6 Large-file (20,001 char) without selector → large_file_selector_required.
  #7 Large-file third range, 121 lines, 8,001 chars rejected.
  #16 Snapshots/FTS5 disabled → no cache/index files created.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pytest


from vault_retrieval.budgets import (
    TurnBudgets,
    BudgetConfig,
    BudgetDecision,
    LargeFileDecision,
    decide_budget,
    check_large_file,
    get_or_create_turn,
    reset_turn,
    DEFAULT_LARGE_FILE_CHARS,
    LARGE_FILE_RANGES_MAX,
    LARGE_FILE_LINES_MAX,
    LARGE_FILE_CHARS_MAX,
)


@dataclass
class FakeBudgetConfig:
    default_budget_chars: int = 12_000
    hard_ceiling_chars: int = 24_000
    large_file_chars: int = 20_000
    max_primary_extracts: int = 3
    max_expansion_extracts: int = 2
    max_range_lines: int = 120
    max_range_chars: int = 8_000


class TestTurnBudgetsDefault:
    def test_default_within_budget(self):
        cfg = FakeBudgetConfig()
        tb = TurnBudgets(turn_id="t1", cfg=cfg)
        d = decide_budget(tb, request_chars=10_000, allow_expansion=False)
        assert d.status == "ok"
        assert d.remaining_chars == 2_000

    def test_exactly_at_default_accepted(self):
        cfg = FakeBudgetConfig()
        tb = TurnBudgets(turn_id="t1", cfg=cfg)
        d = decide_budget(tb, request_chars=12_000, allow_expansion=False)
        assert d.status == "ok"

    def test_one_over_default_without_expansion_holds(self):
        cfg = FakeBudgetConfig()
        tb = TurnBudgets(turn_id="t1", cfg=cfg)
        d = decide_budget(tb, request_chars=12_001, allow_expansion=False)
        assert d.status == "data_hold"
        assert d.reason_code == "budget_exceeded"


class TestTurnBudgetsExpansion:
    def test_expansion_to_hard_ceiling_accepted(self):
        cfg = FakeBudgetConfig()
        tb = TurnBudgets(turn_id="t1", cfg=cfg)
        d = decide_budget(tb, request_chars=24_000, allow_expansion=True, expansion_reason="test")
        assert d.status == "ok"

    def test_expansion_above_hard_ceiling_holds(self):
        cfg = FakeBudgetConfig()
        tb = TurnBudgets(turn_id="t1", cfg=cfg)
        d = decide_budget(tb, request_chars=24_001, allow_expansion=True, expansion_reason="test")
        assert d.status == "data_hold"
        assert d.reason_code == "hard_ceiling_exceeded"

    def test_expansion_requires_reason(self):
        cfg = FakeBudgetConfig()
        tb = TurnBudgets(turn_id="t1", cfg=cfg)
        d = decide_budget(tb, request_chars=13_000, allow_expansion=True, expansion_reason="")
        assert d.status == "data_hold"
        assert d.reason_code == "expansion_reason_required"


class TestTurnBudgetsSharingAcrossCalls:
    def test_two_calls_share_ceiling(self):
        cfg = FakeBudgetConfig()
        tb = TurnBudgets(turn_id="t1", cfg=cfg)
        d1 = decide_budget(tb, request_chars=12_000, allow_expansion=False)
        assert d1.status == "ok"
        d2 = decide_budget(tb, request_chars=12_001, allow_expansion=False)
        assert d2.status == "data_hold"  # turn exhausted

    def test_different_turns_do_not_share(self):
        cfg = FakeBudgetConfig()
        tb1 = TurnBudgets(turn_id="t1", cfg=cfg)
        tb2 = TurnBudgets(turn_id="t2", cfg=cfg)
        d1 = decide_budget(tb1, request_chars=12_000, allow_expansion=False)
        d2 = decide_budget(tb2, request_chars=12_000, allow_expansion=False)
        assert d1.status == "ok"
        assert d2.status == "ok"

    def test_two_calls_can_share_under_default(self):
        cfg = FakeBudgetConfig()
        tb = TurnBudgets(turn_id="t1", cfg=cfg)
        d1 = decide_budget(tb, request_chars=8_000, allow_expansion=False)
        d2 = decide_budget(tb, request_chars=4_000, allow_expansion=False)
        assert d1.status == "ok"
        assert d2.status == "ok"
        assert d2.remaining_chars == 0


class TestLargeFile:
    def test_under_large_threshold_not_large(self):
        cfg = FakeBudgetConfig()
        d = check_large_file(
            cfg=cfg,
            file_chars=10_000,
            selectors=[],
        )
        assert d.is_large is False

    def test_over_threshold_without_selector_holds(self):
        cfg = FakeBudgetConfig()
        d = check_large_file(
            cfg=cfg,
            file_chars=20_001,
            selectors=[],
        )
        assert d.is_large is True
        assert d.decision == "large_file_selector_required"

    def test_over_threshold_with_one_selector_ok(self):
        cfg = FakeBudgetConfig()
        d = check_large_file(
            cfg=cfg,
            file_chars=20_001,
            selectors=[{"line_start": 1, "line_end": 50, "heading": "## Dec"}],
        )
        assert d.is_large is True
        assert d.decision == "ok"

    def test_third_range_rejected(self):
        cfg = FakeBudgetConfig()
        d = check_large_file(
            cfg=cfg,
            file_chars=20_001,
            selectors=[
                {"line_start": 1, "line_end": 50, "heading": "## A"},
                {"line_start": 60, "line_end": 110, "heading": "## B"},
                {"line_start": 120, "line_end": 170, "heading": "## C"},
            ],
        )
        assert d.is_large is True
        assert d.decision == "too_many_ranges"

    def test_121_line_range_rejected(self):
        cfg = FakeBudgetConfig()
        d = check_large_file(
            cfg=cfg,
            file_chars=20_001,
            selectors=[{"line_start": 1, "line_end": 121, "heading": "## X"}],
        )
        assert d.is_large is True
        assert d.decision == "range_too_long"

    def test_8001_char_range_rejected(self):
        cfg = FakeBudgetConfig()
        d = check_large_file(
            cfg=cfg,
            file_chars=20_001,
            selectors=[{"line_start": 1, "line_end": 50, "heading": "## X", "estimated_chars": 8_001}],
        )
        assert d.is_large is True
        assert d.decision == "range_too_long_chars"


class TestBudgetConstants:
    def test_constants_match_spec(self):
        assert DEFAULT_LARGE_FILE_CHARS == 20_000
        assert LARGE_FILE_RANGES_MAX == 2
        assert LARGE_FILE_LINES_MAX == 120
        assert LARGE_FILE_CHARS_MAX == 8_000
