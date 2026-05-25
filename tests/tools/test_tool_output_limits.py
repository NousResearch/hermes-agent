"""Tests for tools.tool_output_limits — configurable truncation limits."""

from __future__ import annotations

from unittest.mock import patch

from tools.tool_output_limits import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINE_LENGTH,
    DEFAULT_MAX_LINES,
    _coerce_positive_int,
    get_max_bytes,
    get_max_line_length,
    get_max_lines,
    get_tool_output_limits,
)


# ============================================================================
# _coerce_positive_int
# ============================================================================
class TestCoercePositiveInt:
    def test_positive_int_returns_itself(self):
        assert _coerce_positive_int(42, 100) == 42

    def test_zero_returns_default(self):
        assert _coerce_positive_int(0, 100) == 100

    def test_negative_returns_default(self):
        assert _coerce_positive_int(-5, 100) == 100

    def test_none_returns_default(self):
        assert _coerce_positive_int(None, 100) == 100

    def test_string_int_coerced(self):
        assert _coerce_positive_int("42", 100) == 42

    def test_invalid_string_returns_default(self):
        assert _coerce_positive_int("not-a-number", 100) == 100

    def test_float_coerced(self):
        assert _coerce_positive_int(42.7, 100) == 42

    def test_float_zero_returns_default(self):
        # int(0.5) = 0 ≤ 0 → returns default
        assert _coerce_positive_int(0.5, 100) == 100

    def test_large_int(self):
        assert _coerce_positive_int(1_000_000, 50) == 1_000_000

    def test_one_is_positive(self):
        assert _coerce_positive_int(1, 100) == 1


# ============================================================================
# get_tool_output_limits
# ============================================================================
class TestGetToolOutputLimits:
    def test_no_config_returns_defaults(self):
        with patch("hermes_cli.config.load_config", return_value=None):
            limits = get_tool_output_limits()
            assert limits["max_bytes"] == DEFAULT_MAX_BYTES
            assert limits["max_lines"] == DEFAULT_MAX_LINES
            assert limits["max_line_length"] == DEFAULT_MAX_LINE_LENGTH

    def test_empty_config_returns_defaults(self):
        with patch("hermes_cli.config.load_config", return_value={}):
            limits = get_tool_output_limits()
            assert limits["max_bytes"] == DEFAULT_MAX_BYTES
            assert limits["max_lines"] == DEFAULT_MAX_LINES

    def test_custom_values(self):
        cfg = {"tool_output": {"max_bytes": 100_000, "max_lines": 5000}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            limits = get_tool_output_limits()
            assert limits["max_bytes"] == 100_000
            assert limits["max_lines"] == 5000
            assert limits["max_line_length"] == DEFAULT_MAX_LINE_LENGTH

    def test_tool_output_not_a_dict(self):
        cfg = {"tool_output": "not-a-dict"}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            limits = get_tool_output_limits()
            assert limits["max_bytes"] == DEFAULT_MAX_BYTES

    def test_config_is_not_a_dict(self):
        with patch("hermes_cli.config.load_config", return_value=42):
            limits = get_tool_output_limits()
            assert limits["max_bytes"] == DEFAULT_MAX_BYTES

    def test_config_is_list(self):
        with patch("hermes_cli.config.load_config", return_value=["not dict"]):
            limits = get_tool_output_limits()
            assert limits["max_bytes"] == DEFAULT_MAX_BYTES

    def test_load_config_raises(self):
        with patch(
            "hermes_cli.config.load_config",
            side_effect=RuntimeError("config broken"),
        ):
            limits = get_tool_output_limits()
            assert limits["max_bytes"] == DEFAULT_MAX_BYTES

    def test_invalid_value_coerced(self):
        cfg = {"tool_output": {"max_bytes": -1, "max_lines": "abc"}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            limits = get_tool_output_limits()
            assert limits["max_bytes"] == DEFAULT_MAX_BYTES
            assert limits["max_lines"] == DEFAULT_MAX_LINES

    def test_string_values_coerced(self):
        cfg = {
            "tool_output": {
                "max_bytes": "100000",
                "max_lines": "5000",
                "max_line_length": "3000",
            }
        }
        with patch("hermes_cli.config.load_config", return_value=cfg):
            limits = get_tool_output_limits()
            assert limits["max_bytes"] == 100000
            assert limits["max_lines"] == 5000
            assert limits["max_line_length"] == 3000

    def test_empty_tool_output_section(self):
        cfg = {"tool_output": {}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            limits = get_tool_output_limits()
            assert limits["max_bytes"] == DEFAULT_MAX_BYTES
            assert limits["max_lines"] == DEFAULT_MAX_LINES
            assert limits["max_line_length"] == DEFAULT_MAX_LINE_LENGTH

    def test_all_keys_present(self):
        with patch("hermes_cli.config.load_config", return_value=None):
            limits = get_tool_output_limits()
            assert set(limits.keys()) == {"max_bytes", "max_lines", "max_line_length"}

    def test_zero_values_rejected(self):
        cfg = {"tool_output": {"max_bytes": 0, "max_lines": 0}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            limits = get_tool_output_limits()
            assert limits["max_bytes"] == DEFAULT_MAX_BYTES
            assert limits["max_lines"] == DEFAULT_MAX_LINES


# ============================================================================
# Shortcut functions
# ============================================================================
class TestShortcuts:
    def test_get_max_bytes(self):
        with patch("hermes_cli.config.load_config", return_value=None):
            assert get_max_bytes() == DEFAULT_MAX_BYTES

    def test_get_max_lines(self):
        with patch("hermes_cli.config.load_config", return_value=None):
            assert get_max_lines() == DEFAULT_MAX_LINES

    def test_get_max_line_length(self):
        with patch("hermes_cli.config.load_config", return_value=None):
            assert get_max_line_length() == DEFAULT_MAX_LINE_LENGTH

    def test_shortcuts_respect_custom_config(self):
        cfg = {
            "tool_output": {
                "max_bytes": 99,
                "max_lines": 77,
                "max_line_length": 55,
            }
        }
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert get_max_bytes() == 99
            assert get_max_lines() == 77
            assert get_max_line_length() == 55


# ============================================================================
# Default constants
# ============================================================================
class TestDefaultConstants:
    def test_defaults_are_positive(self):
        assert DEFAULT_MAX_BYTES > 0
        assert DEFAULT_MAX_LINES > 0
        assert DEFAULT_MAX_LINE_LENGTH > 0

    def test_defaults_are_ints(self):
        assert isinstance(DEFAULT_MAX_BYTES, int)
        assert isinstance(DEFAULT_MAX_LINES, int)
        assert isinstance(DEFAULT_MAX_LINE_LENGTH, int)
