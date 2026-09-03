"""Tests for the shared tool-input repair helper (tools/tool_input_repair.py).

Models occasionally emit an array-typed tool parameter as a string instead
of a real list: either a JSON string ("[\\"a\\", \\"b\\"]") or a bare string
("a") where the schema asked for ["a"]. This helper centralizes the
stringified-array repair that was previously duplicated ad hoc across tools.

The core invariant: valid non-string inputs (real lists, dicts, None) are
never touched, so no legitimate payload is transformed or corrupted.
"""

import json

from tools.tool_input_repair import recover_list_from_json_string


class TestStringifiedArrayRepair:
    def test_json_array_string_is_parsed_to_list(self):
        repaired, error = recover_list_from_json_string(
            '["a", "b"]', param_name="urls"
        )
        assert error is None
        assert repaired == ["a", "b"]

    def test_whitespace_surrounding_stringified_array(self):
        repaired, error = recover_list_from_json_string(
            '  ["a"]  ', param_name="urls"
        )
        assert error is None
        assert repaired == ["a"]

    def test_stringified_array_of_objects(self):
        raw = json.dumps([{"action": "add", "content": "x"}])
        repaired, error = recover_list_from_json_string(
            raw, param_name="operations"
        )
        assert error is None
        assert repaired == [{"action": "add", "content": "x"}]


class TestUnparseableStringWithWrap:
    def test_bare_string_not_wrapped_by_default(self):
        repaired, error = recover_list_from_json_string(
            "https://example.com", param_name="urls"
        )
        assert error is not None
        assert "JSON array" in error
        assert repaired is None

    def test_bare_string_wrapped_when_requested(self):
        repaired, error = recover_list_from_json_string(
            "https://example.com", param_name="urls", wrap_bare_string=True
        )
        assert error is None
        assert repaired == ["https://example.com"]

    def test_bare_short_string_wrapped(self):
        repaired, error = recover_list_from_json_string(
            "foo", param_name="urls", wrap_bare_string=True
        )
        assert error is None
        assert repaired == ["foo"]

    def test_empty_string_still_rejected_with_wrap(self):
        # An empty string is not a recoverable URL, even with wrapping.
        repaired, error = recover_list_from_json_string(
            "", param_name="urls", wrap_bare_string=True
        )
        assert error is not None
        assert repaired is None


class TestParsedToNonList:
    def test_string_parsing_to_dict_is_rejected(self):
        repaired, error = recover_list_from_json_string(
            '{"a": 1}', param_name="operations"
        )
        assert error is not None
        assert "dict" in error
        assert repaired is None

    def test_string_parsing_to_scalar_is_rejected(self):
        repaired, error = recover_list_from_json_string(
            "42", param_name="operations"
        )
        assert error is not None
        assert repaired is None


class TestValidInputsUntouched:
    """The critical invariant: real lists / dicts / None pass through unchanged."""

    def test_real_list_untouched(self):
        items = [{"id": "a"}]
        repaired, error = recover_list_from_json_string(items, param_name="todos")
        assert error is None
        assert repaired is None  # sentinel: caller keeps the original

    def test_empty_list_untouched(self):
        repaired, error = recover_list_from_json_string([], param_name="todos")
        assert error is None
        assert repaired is None

    def test_none_untouched(self):
        repaired, error = recover_list_from_json_string(None, param_name="todos")
        assert error is None
        assert repaired is None

    def test_dict_untouched(self):
        # A dict placeholder must never be coerced into a list.
        repaired, error = recover_list_from_json_string(
            {"todo": "x"}, param_name="todos"
        )
        assert error is None
        assert repaired is None