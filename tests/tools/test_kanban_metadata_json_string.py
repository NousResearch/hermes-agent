"""A JSON-object string in ``metadata`` must not silently cost the worker its handoff.

Some models serialize a nested tool argument as text, so ``kanban_complete`` /
``kanban_request_review`` arrive with ``metadata`` as ``'{"criteria": [...]}'`` instead of a
dict. ``_require_dict_metadata`` rejected that outright: the card still completed, but the
structured handoff was gone. A completion that reports success while dropping its evidence is
the worst failure shape a reviewer can inherit.

``_coerce_metadata`` parses only a string that yields a JSON *object*. Everything else is
returned untouched so the rejection still names the type the model actually sent.
"""
from __future__ import annotations

import pytest

from tools import kanban_tools


def _validate(value):
    """Run the real pipeline: coerce, then the validator the handlers call."""
    coerced = kanban_tools._coerce_metadata(value)
    kanban_tools._require_dict_metadata(coerced)
    return coerced


def test_json_object_string_is_accepted():
    assert _validate('{"criteria": ["a"], "verified": true}') == {
        "criteria": ["a"], "verified": True}


def test_json_object_string_with_surrounding_whitespace():
    assert _validate('  {"a": 1}\n') == {"a": 1}


def test_empty_string_is_treated_as_absent():
    assert _validate("   ") is None


@pytest.mark.parametrize("value", [{"a": 1}, None])
def test_existing_shapes_are_unchanged(value):
    assert _validate(value) == value


@pytest.mark.parametrize("value, type_name", [
    ("not json", "str"),
    ("[1, 2]", "str"),
    ('"a bare string"', "str"),
    ("42", "str"),
])
def test_non_object_values_are_still_rejected(value, type_name):
    """A non-object must stay rejected, and the error must name the original type."""
    with pytest.raises(kanban_tools._Reject) as excinfo:
        _validate(value)
    assert type_name in str(excinfo.value)
