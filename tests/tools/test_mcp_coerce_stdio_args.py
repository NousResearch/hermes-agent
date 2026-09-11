"""Regression tests for ``_coerce_mcp_stdio_args`` (#79519).

YAML can persist a list as a quoted JSON/Python-literal string
(``args: '["-y", "pkg"]'``) or surface scalars as non-string types
(``true`` → bool, ``42`` → int). Without coercion, unpacking such a
value directly into the spawn argv treats every character as a token.
"""
import pytest

from tools.mcp_tool import _coerce_mcp_stdio_args


class TestCoerceMcpStdioArgs:
    def test_none_returns_empty(self):
        assert _coerce_mcp_stdio_args(None) == []

    def test_list_of_strings_passes_through(self):
        assert _coerce_mcp_stdio_args(["-y", "pkg"]) == ["-y", "pkg"]

    def test_list_with_non_string_scalars_coerced(self):
        assert _coerce_mcp_stdio_args([True, 42, 3.14, "pkg"]) == ["True", "42", "3.14", "pkg"]

    def test_tuple_coerced_to_list(self):
        assert _coerce_mcp_stdio_args(("-y", "pkg")) == ["-y", "pkg"]

    def test_json_string_list_parsed(self):
        assert _coerce_mcp_stdio_args('["-y", "pkg"]') == ["-y", "pkg"]

    def test_python_literal_string_list_parsed(self):
        assert _coerce_mcp_stdio_args("['-y', 'pkg']") == ["-y", "pkg"]

    def test_json_string_with_non_string_items_coerced(self):
        assert _coerce_mcp_stdio_args('["-y", true, 42]') == ["-y", "True", "42"]

    def test_empty_string_returns_empty(self):
        assert _coerce_mcp_stdio_args("") == []

    def test_whitespace_only_string_returns_empty(self):
        assert _coerce_mcp_stdio_args("   ") == []

    def test_legacy_plain_string_shell_split(self):
        assert _coerce_mcp_stdio_args("-y 'pkg with spaces'") == ["-y", "pkg with spaces"]

    def test_scalar_int_coerced(self):
        assert _coerce_mcp_stdio_args(42) == ["42"]

    def test_scalar_bool_coerced(self):
        assert _coerce_mcp_stdio_args(True) == ["True"]

    def test_structured_string_not_a_list_raises(self):
        with pytest.raises(ValueError, match="must decode to a JSON or Python list"):
            _coerce_mcp_stdio_args('{"key": "value"}')

    def test_malformed_structured_string_raises(self):
        with pytest.raises(ValueError, match="not a valid JSON or Python list"):
            _coerce_mcp_stdio_args('["-y", pkg]')

    def test_invalid_shell_quoting_raises(self):
        with pytest.raises(ValueError, match="invalid shell quoting"):
            _coerce_mcp_stdio_args("-y 'unterminated quote")
