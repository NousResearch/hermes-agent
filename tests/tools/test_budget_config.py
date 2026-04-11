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
