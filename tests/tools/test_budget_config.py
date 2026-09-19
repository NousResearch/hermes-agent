"""Unit tests for tools/budget_config.py.

Covers default values, resolve_threshold() priority chain
(pinned > tool_overrides > registry > default), immutability,
and the PINNED_THRESHOLDS escape-hatch for read_file.
"""

import dataclasses
import math
from decimal import Decimal
from fractions import Fraction
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
    budget_with_persist_threshold,
    normalize_persist_threshold,
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


# ---------------------------------------------------------------------------
# MCP-prefix threshold (mcp_result_size)
# ---------------------------------------------------------------------------


class TestMcpPrefixThreshold:
    """mcp_* tools get the tighter 50K default, config-overridable."""

    def test_default_mcp_threshold_is_50k(self):
        from tools.budget_config import DEFAULT_MCP_RESULT_SIZE_CHARS
        assert DEFAULT_MCP_RESULT_SIZE_CHARS == 50_000
        assert DEFAULT_BUDGET.resolve_threshold("mcp_composio_search_tools") == 50_000

    def test_non_mcp_tools_keep_generic_default(self):
        assert DEFAULT_BUDGET.resolve_threshold("some_random_tool") == DEFAULT_RESULT_SIZE_CHARS

    def test_pinned_wins_over_mcp_prefix(self):
        with patch.dict(PINNED_THRESHOLDS, {"mcp_pinned_tool": float("inf")}):
            assert DEFAULT_BUDGET.resolve_threshold("mcp_pinned_tool") == float("inf")

    def test_tool_override_wins_over_mcp_prefix(self):
        cfg = BudgetConfig(tool_overrides={"mcp_special": 75_000})
        assert cfg.resolve_threshold("mcp_special") == 75_000

    def test_mcp_threshold_capped_by_scaled_default(self):
        """On a small model the scaled default_result_size caps the MCP value."""
        cfg = BudgetConfig(default_result_size=20_000, mcp_result_size=50_000)
        assert cfg.resolve_threshold("mcp_anything") == 20_000

    def test_mcp_threshold_never_exceeds_default_result_size(self):
        cfg = BudgetConfig(default_result_size=100_000, mcp_result_size=999_999)
        assert cfg.resolve_threshold("mcp_anything") == 100_000

    def test_config_override_via_hermes_home(self, tmp_path, monkeypatch):
        (tmp_path / "config.yaml").write_text(
            "tool_budget:\n  mcp_result_size_chars: 30000\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg = budget_for_context_window(None)
        assert cfg.resolve_threshold("mcp_composio_multi_execute") == 30_000
        # Generic tools are untouched by the MCP knob.
        assert cfg.default_result_size == DEFAULT_RESULT_SIZE_CHARS

    def test_config_override_survives_window_scaling(self, tmp_path, monkeypatch):
        (tmp_path / "config.yaml").write_text(
            "tool_budget:\n  mcp_result_size_chars: 30000\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg = budget_for_context_window(200_000)
        assert cfg.mcp_result_size == 30_000

    def test_malformed_config_falls_back_to_default(self, tmp_path, monkeypatch):
        (tmp_path / "config.yaml").write_text("tool_budget: not-a-mapping\n")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg = budget_for_context_window(None)
        assert cfg.resolve_threshold("mcp_x_y") == 50_000

    def test_scaled_small_window_caps_mcp_threshold(self, tmp_path, monkeypatch):
        """A tiny model's scaled default_result_size caps even the MCP value."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))  # no config.yaml
        cfg = budget_for_context_window(16_384)  # scaled default < 50K
        assert cfg.default_result_size < 50_000
        assert cfg.resolve_threshold("mcp_tool") == cfg.default_result_size


# ---------------------------------------------------------------------------
# budget_with_persist_threshold() — explicit user-configured cap
# ---------------------------------------------------------------------------


class TestBudgetWithPersistThreshold:
    """Explicit tools.tool_result_persist_threshold_chars path.

    Overrides ONLY the per-result cap; turn budget and preview size keep the
    context-scaled values (small-model turn-budget protection must survive),
    and the pinned/registry guards stay active.
    """

    def test_small_model_turn_budget_protection_survives(self):
        # 65K-token model: scaled turn budget is ~78,643 chars. An explicit
        # per-result cap must NOT reset it back to the 200K default.
        cfg = budget_with_persist_threshold(20_000, context_length=65_536)
        assert cfg.default_result_size == 20_000
        assert cfg.turn_budget == int(65_536 * 4 * 0.30)  # 78,643
        assert cfg.turn_budget < DEFAULT_TURN_BUDGET_CHARS
        assert cfg.preview_size == DEFAULT_PREVIEW_SIZE_CHARS

    def test_large_context_turn_budget_caps_at_default(self):
        cfg = budget_with_persist_threshold(20_000, context_length=1_000_000)
        assert cfg.default_result_size == 20_000
        assert cfg.turn_budget == DEFAULT_TURN_BUDGET_CHARS
        assert cfg.preview_size == DEFAULT_PREVIEW_SIZE_CHARS

    def test_unknown_context_keeps_historical_defaults(self):
        cfg = budget_with_persist_threshold(20_000)
        assert cfg.default_result_size == 20_000
        assert cfg.turn_budget == DEFAULT_TURN_BUDGET_CHARS
        assert cfg.preview_size == DEFAULT_PREVIEW_SIZE_CHARS

    def test_explicit_threshold_wins_over_default(self):
        cfg = budget_with_persist_threshold(20_000)
        assert cfg.default_result_size == 20_000
        with patch("tools.registry.registry") as mock_registry:
            mock_registry.get_max_result_size.return_value = 100_000
            assert cfg.resolve_threshold("terminal") == 20_000

    def test_smaller_than_registry_cap_wins(self):
        cfg = budget_with_persist_threshold(30_000)
        with patch("tools.registry.registry") as mock_registry:
            # web/terminal/x_search register 100K; explicit 30K must win.
            mock_registry.get_max_result_size.return_value = 100_000
            assert cfg.resolve_threshold("web_search") == 30_000

    def test_larger_than_registry_cap_is_bounded(self):
        cfg = budget_with_persist_threshold(200_000)
        with patch("tools.registry.registry") as mock_registry:
            mock_registry.get_max_result_size.return_value = 100_000
            assert cfg.resolve_threshold("terminal") == 100_000

    def test_pinned_read_file_stays_exempt(self):
        cfg = budget_with_persist_threshold(1)
        assert cfg.resolve_threshold("read_file") == float("inf")

    def test_nonpositive_input_is_treated_as_unset(self):
        # A 0/negative value must NOT clamp to 1 (which would persist almost
        # every tool result); it is treated as "not configured" and returns the
        # context-scaled budget unchanged.
        assert budget_with_persist_threshold(0) is DEFAULT_BUDGET
        assert budget_with_persist_threshold(-10) is DEFAULT_BUDGET
        assert budget_with_persist_threshold(None) is DEFAULT_BUDGET
        scaled = budget_with_persist_threshold(0, context_length=65_536)
        assert scaled.turn_budget == int(65_536 * 4 * 0.30)

    def test_normalize_rejects_bool(self):
        # bool is an int subclass: int(True) == 1 would persist almost every
        # tool result. Rejected even when set programmatically (the executor
        # path), not just via YAML parsing.
        assert normalize_persist_threshold(True) is None
        assert normalize_persist_threshold(False) is None

    def test_normalize_rejects_float(self):
        # int(1.5) == 1 silently truncates into a near-universal persist
        # threshold; only ints and whole-number strings are accepted.
        assert normalize_persist_threshold(1.5) is None
        assert normalize_persist_threshold(20_000.0) is None

    def test_normalize_accepts_positive_int_and_whole_str(self):
        assert normalize_persist_threshold(20_000) == 20_000
        assert normalize_persist_threshold("20000") == 20_000

    def test_normalize_rejects_nonpositive_and_garbage(self):
        assert normalize_persist_threshold(0) is None
        assert normalize_persist_threshold(-1) is None
        assert normalize_persist_threshold("abc") is None
        assert normalize_persist_threshold([]) is None
        assert normalize_persist_threshold("20.5") is None  # int() raises

    def test_normalize_rejects_decimal_fraction_bytes(self):
        # Regression: int(Decimal("1.5")) == 1, int(Fraction(3, 2)) == 1 and
        # int(b"20000") all silently coerce -- the same truncation trap as
        # float/bool. Only non-bool int and whole-number str are accepted.
        assert normalize_persist_threshold(Decimal("1.5")) is None
        assert normalize_persist_threshold(Decimal("20000.0")) is None
        assert normalize_persist_threshold(Fraction(3, 2)) is None
        assert normalize_persist_threshold(Fraction(20000, 1)) is None
        assert normalize_persist_threshold(b"20000") is None

    def test_normalize_rejects_non_whole_strings(self):
        # "20.5" already covered above; exponent/scientific notation and
        # empty/whitespace-only strings must not reach int() either.
        assert normalize_persist_threshold("1e4") is None
        assert normalize_persist_threshold("") is None
        assert normalize_persist_threshold("   ") is None
        assert normalize_persist_threshold("+20000") == 20_000  # whole, signed
        assert normalize_persist_threshold(" 20000 ") == 20_000  # stripped

    def test_factory_rejects_decimal_fraction_set_programmatically(self):
        scaled = budget_with_persist_threshold(Decimal("1.5"), context_length=65_536)
        assert scaled.turn_budget == int(65_536 * 4 * 0.30)
        assert scaled.default_result_size == int(65_536 * 4 * 0.15)
        scaled = budget_with_persist_threshold(Fraction(3, 2), context_length=65_536)
        assert scaled.default_result_size == int(65_536 * 4 * 0.15)

    def test_factory_rejects_bool_set_programmatically(self):
        # Regression: budget_with_persist_threshold(True) used to yield
        # default_result_size == 1 via int(True). Must return the
        # context-scaled budget unchanged instead.
        assert budget_with_persist_threshold(True) is DEFAULT_BUDGET
        assert budget_with_persist_threshold(False) is DEFAULT_BUDGET
        scaled = budget_with_persist_threshold(True, context_length=65_536)
        assert scaled.turn_budget == int(65_536 * 4 * 0.30)
        assert scaled.default_result_size == int(65_536 * 4 * 0.15)
        assert scaled.default_result_size != 1

    def test_factory_rejects_float_set_programmatically(self):
        scaled = budget_with_persist_threshold(20_000.0, context_length=65_536)
        assert scaled.turn_budget == int(65_536 * 4 * 0.30)
        assert scaled.default_result_size == int(65_536 * 4 * 0.15)
