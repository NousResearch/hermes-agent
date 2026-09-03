"""Tests for tools.tool_output_limits.

Covers:
1. Default values when no config is provided.
2. Config override picks up user-supplied max_bytes / max_lines /
   max_line_length.
3. Malformed values (None, negative, wrong type) fall back to defaults
   rather than raising.
4. Integration: the helpers return what the terminal_tool and
   file_operations call paths will actually consume.

Port-tracking: anomalyco/opencode PR #23770
(feat(truncate): allow configuring tool output truncation limits).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tools import tool_output_limits as tol


@pytest.fixture(autouse=True)
def _reset_limits_cache():
    """get_tool_output_limits() now memoizes its result for the process
    lifetime, so each test must start from a clean cache to observe the
    config value it patches in."""
    tol._reset_tool_output_limits_cache()
    yield
    tol._reset_tool_output_limits_cache()


class TestDefaults:
    def test_defaults_match_previous_hardcoded_values(self):
        assert tol.DEFAULT_MAX_BYTES == 50_000
        assert tol.DEFAULT_MAX_LINES == 2000
        assert tol.DEFAULT_MAX_LINE_LENGTH == 2000


    def test_get_limits_returns_defaults_when_load_config_raises(self):
        def _boom():
            raise RuntimeError("boom")

        with patch("hermes_cli.config.load_config", side_effect=_boom):
            limits = tol.get_tool_output_limits()
        assert limits["max_lines"] == tol.DEFAULT_MAX_LINES


class TestOverrides:
    def test_user_config_overrides_all_three(self):
        cfg = {
            "tool_output": {
                "max_bytes": 100_000,
                "max_lines": 5000,
                "max_line_length": 4096,
            }
        }
        with patch("hermes_cli.config.load_config", return_value=cfg):
            limits = tol.get_tool_output_limits()
        assert limits == {
            "max_bytes": 100_000,
            "max_lines": 5000,
            "max_line_length": 4096,
        }


    def test_section_not_a_dict_falls_back(self):
        cfg = {"tool_output": "nonsense"}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            limits = tol.get_tool_output_limits()
        assert limits["max_bytes"] == tol.DEFAULT_MAX_BYTES


class TestCoercion:
    @pytest.mark.parametrize("bad", [None, "not a number", -1, 0, [], {}])
    def test_invalid_values_fall_back_to_defaults(self, bad):
        cfg = {"tool_output": {"max_bytes": bad, "max_lines": bad, "max_line_length": bad}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            limits = tol.get_tool_output_limits()
        assert limits["max_bytes"] == tol.DEFAULT_MAX_BYTES
        assert limits["max_lines"] == tol.DEFAULT_MAX_LINES
        assert limits["max_line_length"] == tol.DEFAULT_MAX_LINE_LENGTH

    def test_string_integer_is_coerced(self):
        cfg = {"tool_output": {"max_bytes": "75000"}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            limits = tol.get_tool_output_limits()
        assert limits["max_bytes"] == 75_000


class TestShortcuts:
    def test_individual_accessors_delegate_to_get_tool_output_limits(self):
        cfg = {
            "tool_output": {
                "max_bytes": 111,
                "max_lines": 222,
                "max_line_length": 333,
            }
        }
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert tol.get_max_bytes() == 111
            assert tol.get_max_lines() == 222
            assert tol.get_max_line_length() == 333


class TestDefaultConfigHasSection:
    """The DEFAULT_CONFIG in hermes_cli.config must expose tool_output so
    that ``hermes setup`` and default installs stay in sync with the
    helpers here."""

    def test_default_config_contains_tool_output_section(self):
        from hermes_cli.config import DEFAULT_CONFIG
        assert "tool_output" in DEFAULT_CONFIG
        section = DEFAULT_CONFIG["tool_output"]
        assert isinstance(section, dict)
        assert section["max_bytes"] == tol.DEFAULT_MAX_BYTES
        assert section["max_lines"] == tol.DEFAULT_MAX_LINES
        assert section["max_line_length"] == tol.DEFAULT_MAX_LINE_LENGTH


class TestIntegrationReadPagination:
    """normalize_read_pagination uses get_max_lines() — verify the plumbing."""

    def test_pagination_limit_clamped_by_config_value(self):
        from tools.file_operations import normalize_read_pagination
        cfg = {"tool_output": {"max_lines": 50}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            offset, limit = normalize_read_pagination(offset=1, limit=1000)
        # limit should have been clamped to 50 (the configured max_lines)
        assert limit == 50
        assert offset == 1

    def test_pagination_default_when_config_missing(self):
        from tools.file_operations import normalize_read_pagination
        with patch("hermes_cli.config.load_config", return_value={}):
            offset, limit = normalize_read_pagination(offset=10, limit=100000)
        # Clamped to default MAX_LINES (2000).
        assert limit == tol.DEFAULT_MAX_LINES
        assert offset == 10


@pytest.mark.parametrize("order", [("a", "b", "a"), ("b", "a", "b")])
def test_limits_follow_active_profile_across_switches(tmp_path, order):
    """A shared gateway must retain independent limits for each profile."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools import file_tools
    from tools.file_operations import normalize_read_pagination

    expected = {
        "a": (11_111, 33_333, 111, 1_111),
        "b": (22_222, 44_444, 222, 2_222),
    }
    homes = {}
    for name, (read_chars, max_bytes, max_lines, max_line_length) in expected.items():
        home = tmp_path / name
        home.mkdir()
        (home / "config.yaml").write_text(
            "\n".join(
                [
                    f"file_read_max_chars: {read_chars}",
                    "tool_output:",
                    f"  max_bytes: {max_bytes}",
                    f"  max_lines: {max_lines}",
                    f"  max_line_length: {max_line_length}",
                ]
            ),
            encoding="utf-8",
        )
        homes[name] = home

    tol._reset_tool_output_limits_cache()
    file_tools._reset_max_read_chars_cache()
    try:
        for name in order:
            token = set_hermes_home_override(str(homes[name]))
            try:
                read_chars, max_bytes, max_lines, max_line_length = expected[name]
                assert file_tools._get_max_read_chars() == read_chars
                assert tol.get_max_bytes() == max_bytes
                assert tol.get_max_lines() == max_lines
                assert tol.get_max_line_length() == max_line_length
                assert normalize_read_pagination(1, 99_999) == (1, max_lines)
            finally:
                reset_hermes_home_override(token)
    finally:
        tol._reset_tool_output_limits_cache()
        file_tools._reset_max_read_chars_cache()


def test_limit_caches_are_bypassed_when_profile_key_is_unavailable():
    """A key-resolution failure must not create a shared fallback cache."""
    from tools import file_tools

    configs = [
        {"file_read_max_chars": 111},
        {"file_read_max_chars": 222},
        {"tool_output": {"max_bytes": 333}},
        {"tool_output": {"max_bytes": 444}},
    ]
    file_tools._reset_max_read_chars_cache()
    try:
        with (
            patch("hermes_constants.hermes_home_key", side_effect=RuntimeError("unavailable")),
            patch("hermes_cli.config.load_config", side_effect=configs) as load_config,
        ):
            assert file_tools._get_max_read_chars() == 111
            assert file_tools._get_max_read_chars() == 222
            assert tol.get_max_bytes() == 333
            assert tol.get_max_bytes() == 444
        assert load_config.call_count == 4
    finally:
        file_tools._reset_max_read_chars_cache()
