"""Unit tests for tools/budget_config.py.

Covers default values, resolve_threshold() priority chain
(pinned > tool_overrides > registry > default), immutability, and the
universal call-local result-budget schema.
"""

import dataclasses
from unittest.mock import patch

import pytest

from tools.budget_config import (
    DEFAULT_BUDGET,
    DEFAULT_PREVIEW_SIZE_CHARS,
    DEFAULT_RESULT_SIZE_CHARS,
    DEFAULT_TURN_BUDGET_CHARS,
    PINNED_THRESHOLDS,
    BudgetConfig,
    augment_function_schema_with_result_token_limit,
    budget_for_context_window,
)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    """Verify documented default values haven't drifted."""

    def test_default_result_size(self):
        assert DEFAULT_RESULT_SIZE_CHARS == 100_000


    def test_default_preview_size(self):
        assert DEFAULT_PREVIEW_SIZE_CHARS == 1_500


class TestPinnedThresholds:
    """PINNED_THRESHOLDS – tools whose values must never be overridden."""

    def test_read_file_has_no_unlimited_pin(self):
        assert "read_file" not in PINNED_THRESHOLDS


def test_function_schema_augmentation_is_copy_safe_and_optional():
    original = {
        "name": "late_injected",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query", "result_token_limit"],
        },
    }

    augmented = augment_function_schema_with_result_token_limit(original)

    assert augmented is not original
    assert "result_token_limit" not in original["parameters"]["properties"]
    budget = augmented["parameters"]["properties"]["result_token_limit"]
    assert budget == {
        "type": "integer",
        "minimum": 1,
        "maximum": 32_000,
        "default": 10_000,
        "description": (
            "Optional model-visible result budget for this call only. "
            "Defaults to 10000; maximum 32000. Removed before tool execution."
        ),
    }
    assert augmented["parameters"]["required"] == ["query"]


def test_function_schema_augmentation_preserves_schema_owned_budget_parameter():
    original_property = {
        "type": "string",
        "description": "Business argument owned by an external tool",
    }
    original = {
        "name": "mcp__external__collision",
        "parameters": {
            "type": "object",
            "properties": {"result_token_limit": original_property},
            "required": ["result_token_limit"],
        },
    }

    augmented = augment_function_schema_with_result_token_limit(original)

    assert augmented["parameters"]["properties"]["result_token_limit"] == (
        original_property
    )
    assert augmented["parameters"]["required"] == ["result_token_limit"]
    assert original["parameters"]["properties"]["result_token_limit"] == (
        original_property
    )


# ---------------------------------------------------------------------------
# BudgetConfig defaults
# ---------------------------------------------------------------------------


class TestBudgetConfigDefaults:
    """BudgetConfig() should match the module-level defaults exactly."""

    def test_default_result_size(self):
        cfg = BudgetConfig()
        assert cfg.default_result_size == DEFAULT_RESULT_SIZE_CHARS


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

    def test_tool_override_applies_to_read_file(self):
        cfg = BudgetConfig(tool_overrides={"read_file": 1})
        result = cfg.resolve_threshold("read_file")
        assert result == 1

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

    @patch("tools.registry.registry")
    def test_read_file_uses_bounded_registry_value(self, mock_registry):
        mock_registry.get_max_result_size.return_value = 100_000
        cfg = BudgetConfig()
        assert cfg.resolve_threshold("read_file") == 100_000

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


    def test_scaled_budget_constrains_oversized_result(self):
        """A 279K-char result against a 65K model exceeds the scaled per-result
        threshold, so it will be persisted/truncated rather than sent whole."""
        cfg = budget_for_context_window(65_536)
        huge_len = 279_549
        threshold = cfg.resolve_threshold("mcp_firecrawl_firecrawl_search")
        assert threshold < huge_len
        assert cfg.default_result_size < huge_len
