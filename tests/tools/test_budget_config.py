"""Unit tests for tools/budget_config.py.

Covers default values, resolve_threshold() priority chain
(pinned > tool_overrides > registry > default), immutability,
and the PINNED_THRESHOLDS escape-hatch for read_file.
"""

import dataclasses
import math
from unittest.mock import patch

import pytest

from tools.budget_config import (
    DEFAULT_BUDGET,
    DEFAULT_PREVIEW_SIZE_CHARS,
    DEFAULT_RESULT_SIZE_CHARS,
    DEFAULT_TURN_BUDGET_CHARS,
    PINNED_THRESHOLDS,
    BudgetConfig,
    budget_for_context_window,
)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    """Verify documented default values haven't drifted."""

    def test_default_result_size(self):
        assert DEFAULT_RESULT_SIZE_CHARS == 100_000

    def test_default_turn_budget(self):
        assert DEFAULT_TURN_BUDGET_CHARS == 200_000

    def test_default_preview_size(self):
        assert DEFAULT_PREVIEW_SIZE_CHARS == 1_500


class TestPinnedThresholds:
    """PINNED_THRESHOLDS – tools whose values must never be overridden."""

    def test_read_file_is_inf(self):
        assert PINNED_THRESHOLDS["read_file"] == float("inf")
        assert math.isinf(PINNED_THRESHOLDS["read_file"])

    def test_pinned_is_not_empty(self):
        assert len(PINNED_THRESHOLDS) >= 1


# ---------------------------------------------------------------------------
# BudgetConfig defaults
# ---------------------------------------------------------------------------


class TestBudgetConfigDefaults:
    """BudgetConfig() should match the module-level defaults exactly."""

    def test_default_result_size(self):
        cfg = BudgetConfig()
        assert cfg.default_result_size == DEFAULT_RESULT_SIZE_CHARS

    def test_default_turn_budget(self):
        cfg = BudgetConfig()
        assert cfg.turn_budget == DEFAULT_TURN_BUDGET_CHARS

    def test_default_preview_size(self):
        cfg = BudgetConfig()
        assert cfg.preview_size == DEFAULT_PREVIEW_SIZE_CHARS

    def test_default_tool_overrides_empty(self):
        cfg = BudgetConfig()
        assert cfg.tool_overrides == {}

    def test_default_budget_singleton_matches(self):
        """DEFAULT_BUDGET should equal a freshly constructed BudgetConfig."""
        assert DEFAULT_BUDGET == BudgetConfig()


# ---------------------------------------------------------------------------
# Immutability (frozen=True)
# ---------------------------------------------------------------------------


class TestBudgetConfigFrozen:
    """Frozen dataclass must reject attribute mutation."""

    def test_cannot_set_default_result_size(self):
        cfg = BudgetConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.default_result_size = 999

    def test_cannot_set_turn_budget(self):
        cfg = BudgetConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.turn_budget = 999

    def test_cannot_set_preview_size(self):
        cfg = BudgetConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.preview_size = 999

    def test_cannot_set_tool_overrides(self):
        cfg = BudgetConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.tool_overrides = {"foo": 1}


# ---------------------------------------------------------------------------
# Custom construction
# ---------------------------------------------------------------------------


class TestBudgetConfigCustom:
    """BudgetConfig can be created with non-default values."""

    def test_custom_values(self):
        cfg = BudgetConfig(
            default_result_size=50_000,
            turn_budget=100_000,
            preview_size=500,
            tool_overrides={"my_tool": 42},
        )
        assert cfg.default_result_size == 50_000
        assert cfg.turn_budget == 100_000
        assert cfg.preview_size == 500
        assert cfg.tool_overrides == {"my_tool": 42}


# ---------------------------------------------------------------------------
# resolve_threshold() priority chain
# ---------------------------------------------------------------------------


class TestResolveThreshold:
    """Priority: pinned > tool_overrides > registry > default."""

    def test_pinned_wins_over_override(self):
        """Even if tool_overrides contains read_file, pinned value wins."""
        cfg = BudgetConfig(tool_overrides={"read_file": 1})
        result = cfg.resolve_threshold("read_file")
        assert result == float("inf")

    def test_tool_override_wins_over_default(self):
        """tool_overrides should be returned before falling back to registry."""
        cfg = BudgetConfig(tool_overrides={"my_tool": 42})
        result = cfg.resolve_threshold("my_tool")
        assert result == 42

    @patch("tools.registry.registry")
    def test_falls_back_to_registry(self, mock_registry):
        """When not pinned and not in overrides, delegate to registry."""
        mock_registry.get_max_result_size.return_value = 77_777
        cfg = BudgetConfig()
        result = cfg.resolve_threshold("some_tool")
        mock_registry.get_max_result_size.assert_called_once_with(
            "some_tool", default=DEFAULT_RESULT_SIZE_CHARS
        )
        assert result == 77_777

    @patch("tools.registry.registry")
    def test_registry_receives_custom_default(self, mock_registry):
        """Custom default_result_size flows through to registry call."""
        mock_registry.get_max_result_size.return_value = 50_000
        cfg = BudgetConfig(default_result_size=50_000)
        cfg.resolve_threshold("unknown_tool")
        mock_registry.get_max_result_size.assert_called_once_with(
            "unknown_tool", default=50_000
        )

    def test_pinned_read_file_returns_inf(self):
        """Canonical case: read_file must always return inf."""
        cfg = BudgetConfig()
        assert cfg.resolve_threshold("read_file") == float("inf")

    @patch("tools.registry.registry")
    def test_registry_value_capped_at_default(self, mock_registry):
        """A scaled-down budget caps an oversized registry value (#23767).

        web/terminal/x_search register max_result_size_chars=100_000; a small
        model's scaled budget must not be re-inflated by that.
        """
        mock_registry.get_max_result_size.return_value = 100_000
        cfg = BudgetConfig(default_result_size=30_000)
        assert cfg.resolve_threshold("web_search") == 30_000

    @patch("tools.registry.registry")
    def test_registry_inf_not_capped(self, mock_registry):
        """An inf registry value (e.g. a future pinned-like tool) is preserved."""
        mock_registry.get_max_result_size.return_value = float("inf")
        cfg = BudgetConfig(default_result_size=30_000)
        assert cfg.resolve_threshold("some_tool") == float("inf")

    @patch("tools.registry.registry")
    def test_default_budget_unchanged_for_100k_tool(self, mock_registry):
        """Default budget keeps 100K registry tools at 100K (no behavior change)."""
        mock_registry.get_max_result_size.return_value = 100_000
        cfg = BudgetConfig()  # default_result_size == 100_000
        assert cfg.resolve_threshold("web_search") == 100_000


# ---------------------------------------------------------------------------
# budget_for_context_window() — context-aware scaling (#23767)
# ---------------------------------------------------------------------------


class TestBudgetForContextWindow:
    """Scaling the tool-output budget to the active model's context window."""

    def test_none_returns_default(self):
        assert budget_for_context_window(None) is DEFAULT_BUDGET

    def test_zero_or_negative_returns_default(self):
        assert budget_for_context_window(0) is DEFAULT_BUDGET
        assert budget_for_context_window(-5) is DEFAULT_BUDGET

    def test_large_model_unchanged(self):
        """A 200K-token model keeps the historical 100K/200K char defaults."""
        cfg = budget_for_context_window(200_000)
        assert cfg.default_result_size == DEFAULT_RESULT_SIZE_CHARS
        assert cfg.turn_budget == DEFAULT_TURN_BUDGET_CHARS

    def test_very_large_model_still_capped_at_default(self):
        """A 1M-token model never exceeds the historical defaults (cap)."""
        cfg = budget_for_context_window(1_000_000)
        assert cfg.default_result_size == DEFAULT_RESULT_SIZE_CHARS
        assert cfg.turn_budget == DEFAULT_TURN_BUDGET_CHARS

    def test_small_model_scaled_down(self):
        """A 65K-token model gets a budget proportional to its window.

        window_chars = 65_536*4 = 262_144; per_result = 15% = 39_321;
        per_turn = 30% = 78_643. Both below the 100K/200K defaults.
        """
        cfg = budget_for_context_window(65_536)
        assert cfg.default_result_size < DEFAULT_RESULT_SIZE_CHARS
        assert cfg.turn_budget < DEFAULT_TURN_BUDGET_CHARS
        assert cfg.default_result_size == int(65_536 * 4 * 0.15)
        assert cfg.turn_budget == int(65_536 * 4 * 0.30)

    def test_tiny_model_floored(self):
        """A tiny window can't drop below the floor (usable preview survives)."""
        cfg = budget_for_context_window(8_000)
        assert cfg.default_result_size >= 8_000
        assert cfg.turn_budget >= 16_000

    def test_scaled_budget_constrains_oversized_result(self):
        """A 279K-char result against a 65K model exceeds the scaled per-result
        threshold, so it will be persisted/truncated rather than sent whole."""
        cfg = budget_for_context_window(65_536)
        huge_len = 279_549
        threshold = cfg.resolve_threshold("mcp_firecrawl_firecrawl_search")
        assert threshold < huge_len
        assert cfg.default_result_size < huge_len
