"""Comprehensive tests for BOM-tolerant JSON parsing.

The json_loads_safe function in tools/json_loads_safe.py is the
project-wide replacement for ad-hoc `json.loads(text.lstrip("\ufeff"))`
patterns scattered through the codebase. It must:

  * Strip exactly ONE leading UTF-8 BOM and parse the rest.
  * Pass plain JSON through unchanged (no allocation when no BOM).
  * Pass bytes input through to stdlib (which auto-decodes them).
  * Preserve strict=False semantics (control chars inside strings).
  * Raise JSONDecodeError (not a custom exception) on bad input.
  * NEVER recursively strip BOM from nested string values.
  * NEVER silently swallow a mid-string BOM (must raise).
"""
from __future__ import annotations

import json

import pytest

from tools.json_loads_safe import (
    JSONDecodeError,
    _BOMStrippingDecoder,
    json_loads_safe,
)


class TestBOMStripping:
    """The core fix: leading UTF-8 BOM must not break parsing."""

    def test_bom_then_object(self):
        assert json_loads_safe('\ufeff{"k": 1}') == {"k": 1}

    def test_bom_then_array(self):
        assert json_loads_safe('\ufeff[1, 2, 3]') == [1, 2, 3]

    def test_bom_then_scalar_string(self):
        assert json_loads_safe('\ufeff"hello"') == "hello"

    def test_bom_then_scalar_number(self):
        assert json_loads_safe('\ufeff42') == 42

    def test_bom_then_scalar_null(self):
        assert json_loads_safe('\ufeffnull') is None

    def test_bom_then_scalar_bool(self):
        assert json_loads_safe('\ufefftrue') is True
        assert json_loads_safe('\ufefffalse') is False

    def test_bom_then_nested_structure(self):
        assert json_loads_safe('\ufeff{"a": {"b": [1, {"c": 2}]}}') == {
            "a": {"b": [1, {"c": 2}]}
        }

    def test_bom_then_unicode_value(self):
        assert json_loads_safe('\ufeff{"name": "和尚"}') == {"name": "和尚"}


class TestNoBOMRegression:
    """Plain JSON without a BOM must parse identically to json.loads."""

    def test_plain_object(self):
        assert json_loads_safe('{"k": 1}') == {"k": 1}

    def test_plain_array(self):
        assert json_loads_safe('[1, 2]') == [1, 2]

    def test_plain_empty(self):
        assert json_loads_safe('{}') == {}
        assert json_loads_safe('[]') == []

    def test_fast_path_no_bom_returns_equivalent_value(self):
        """Fast path: identical return value vs full path."""
        plain = '{"k": [1, 2, 3], "v": "x"}'
        # The function should return the same value as json.loads(strict=False)
        assert json_loads_safe(plain) == json.loads(plain, strict=False)


class TestMidStringBOMNotStripped:
    """A BOM appearing after position 0 is data, not a marker — must raise."""

    def test_bom_after_first_value_raises(self):
        with pytest.raises(JSONDecodeError):
            json_loads_safe('{"k": 1}\ufeff{"k": 2}')

    def test_bom_inside_string_value_preserved(self):
        """BOM inside a JSON string is a literal char (U+FEFF), not stripped."""
        result = json_loads_safe('{"k": "\\ufeff"}')
        assert result == {"k": "\ufeff"}

    def test_bom_in_array_value_preserved(self):
        result = json_loads_safe('["\\ufeff", "x"]')
        assert result == ["\ufeff", "x"]


class TestBytesInput:
    """bytes input is forwarded to stdlib unchanged (which auto-decodes)."""

    def test_bytes_with_utf8_bom(self):
        # Python's json.loads auto-decodes UTF-8 BOM in bytes input.
        assert json_loads_safe(b'\xef\xbb\xbf{"k": 1}') == {"k": 1}

    def test_bytes_without_bom(self):
        assert json_loads_safe(b'{"k": 1}') == {"k": 1}


class TestEdgeCases:
    """Edge cases that must not silently regress."""

    def test_empty_string_raises(self):
        with pytest.raises(JSONDecodeError):
            json_loads_safe("")

    def test_only_bom_raises(self):
        with pytest.raises(JSONDecodeError):
            json_loads_safe("\ufeff")

    def test_bom_then_whitespace_only_raises(self):
        with pytest.raises(JSONDecodeError):
            json_loads_safe("\ufeff   ")

    def test_double_bom_strips_only_one_then_raises(self):
        """Two BOMs at the start: strip one, leave the other → still raises."""
        with pytest.raises(JSONDecodeError):
            json_loads_safe("\ufeff\ufeff{}")


class TestStrictFalsePreserved:
    """The default strict=False must still allow embedded control chars."""

    def test_tab_inside_string(self):
        # Without strict=False, '\t' inside a string would raise.
        assert json_loads_safe('{"k": "a\tb"}') == {"k": "a\tb"}

    def test_newline_inside_string(self):
        assert json_loads_safe('{"k": "a\nb"}') == {"k": "a\nb"}

    def test_caller_can_override_strict(self):
        """kwargs forwarded; caller can opt back into strict mode."""
        with pytest.raises(JSONDecodeError):
            json_loads_safe('{"k": "a\tb"}', strict=True)


class TestBOMStrippingDecoder:
    """The JSONDecoder subclass path for callers that need cls= injection."""

    def test_decoder_strips_bom(self):
        decoder = _BOMStrippingDecoder(strict=False)
        assert decoder.decode('\ufeff{"k": 1}') == {"k": 1}

    def test_decoder_passes_plain_through(self):
        decoder = _BOMStrippingDecoder(strict=False)
        assert decoder.decode('{"k": 1}') == {"k": 1}

    def test_decoder_uses_with_json_loads_cls(self):
        """When called via json.loads(cls=...), the leading BOM is still
        stripped. Note: stdlib's json.loads raises on a leading BOM BEFORE
        delegating to cls, so we must pre-strip here. The decoder is the
        path for direct .decode() use, not for json.loads(cls=...)."""
        # Decoder path: works directly
        decoder = _BOMStrippingDecoder(strict=False)
        assert decoder.decode('\ufeff{"k": 1}') == {"k": 1}
        # json.loads(cls=...) path: stdlib checks BOM first → raises.
        # Document this stdlib behavior so callers know to pre-strip.
        with pytest.raises(JSONDecodeError):
            json.loads('\ufeff{"k": 1}', cls=_BOMStrippingDecoder)


class TestErrorTypeReexport:
    """Callers must be able to import JSONDecodeError from our module."""

    def test_raised_exception_is_reexported(self):
        try:
            json_loads_safe("not json")
        except JSONDecodeError as exc:
            assert isinstance(exc, json.JSONDecodeError)
            return
        pytest.fail("expected JSONDecodeError")