"""Form-mode elicitation answers: schema resolution + affirmative form filling.

Servers such as Microsoft's powerbi-modeling-mcp gate write operations behind a
form-mode elicitation whose required field must literally carry the affirmative
enum value (e.g. {"Confirm the operation": "Yes"}). These tests cover the two
bugs that made every user-approved write come back "declined":

1. the SDK field is camelCase (``ElicitRequestFormParams.requestedSchema``) --
   reading only ``requested_schema`` silently yielded an empty schema;
2. ``action="accept"`` was returned with ``content={}`` -- an empty form, which
   such servers treat as unconfirmed.

Filling is deliberately conservative: only required booleans and required
enums with exactly one affirmative option are answered. Anything else required
fails closed (decline) -- the binary approval surface cannot collect user-typed
values, and fabricating them would be worse.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("mcp.types")

from tools.mcp_tool import (  # noqa: E402  -- after importorskip
    ElicitationHandler,
    _affirmative_form_content,
)

PBI_CONFIRM_SCHEMA = {
    "type": "object",
    "properties": {
        "Confirm the operation": {
            "type": "string",
            "description": "Confirm the operation",
            "enum": ["Yes", "No"],
        }
    },
    "required": ["Confirm the operation"],
}


def _camel_form_params(schema, message="please confirm"):
    """Stand-in for ElicitRequestFormParams with the SDK's camelCase field."""
    return SimpleNamespace(mode="form", message=message, requestedSchema=schema)


class TestAffirmativeFormContent:
    def test_fills_required_enum_with_unambiguous_affirmative(self):
        assert _affirmative_form_content(PBI_CONFIRM_SCHEMA) == {
            "Confirm the operation": "Yes"
        }

    def test_empty_or_invalid_schema_yields_empty_content(self):
        assert _affirmative_form_content({}) == {}
        assert _affirmative_form_content(None) == {}
        assert _affirmative_form_content({"properties": {}}) == {}
        assert _affirmative_form_content({"properties": "not-a-dict"}) == {}

    def test_absent_required_means_nothing_is_required(self):
        # JSON Schema semantics: no ``required`` list -> no required fields.
        schema = {"properties": {"approved": {"type": "boolean"}}}
        assert _affirmative_form_content(schema) == {}

    def test_required_boolean_filled_true(self):
        schema = {
            "properties": {"approved": {"type": "boolean"}},
            "required": ["approved"],
        }
        assert _affirmative_form_content(schema) == {"approved": True}

    def test_optional_fields_are_never_invented(self):
        schema = {
            "properties": {
                "Confirm": {"enum": ["Yes", "No"]},
                "note": {"type": "string"},
            },
            "required": ["Confirm"],
        }
        assert _affirmative_form_content(schema) == {"Confirm": "Yes"}

    def test_required_free_text_fails_closed(self):
        schema = {
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        }
        assert _affirmative_form_content(schema) is None

    def test_required_number_fails_closed(self):
        schema = {
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        }
        assert _affirmative_form_content(schema) is None

    def test_enum_without_affirmative_option_fails_closed(self):
        schema = {
            "properties": {
                "Proceed": {"enum": ["Continue the operation", "Decline the operation"]}
            },
            "required": ["Proceed"],
        }
        assert _affirmative_form_content(schema) is None

    def test_enum_with_multiple_affirmative_options_fails_closed(self):
        schema = {
            "properties": {"Confirm": {"enum": ["Yes", "OK", "No"]}},
            "required": ["Confirm"],
        }
        assert _affirmative_form_content(schema) is None

    def test_required_field_missing_from_properties_fails_closed(self):
        schema = {
            "properties": {"other": {"type": "boolean"}},
            "required": ["ghost"],
        }
        assert _affirmative_form_content(schema) is None


class TestElicitationHandlerFormFilling:
    def test_accept_reads_camelcase_schema_and_fills_confirm(self):
        handler = ElicitationHandler("powerbi", {"timeout": 5})
        params = _camel_form_params(
            PBI_CONFIRM_SCHEMA,
            message="Are you sure you want to perform operations that will "
            "modify your database?",
        )

        with patch(
            "tools.approval.request_elicitation_consent", return_value="accept"
        ):
            result = asyncio.run(handler(context=None, params=params))

        assert result.action == "accept"
        assert result.content == {"Confirm the operation": "Yes"}
        assert handler.metrics["accepted"] == 1

    def test_accept_with_optional_only_schema_keeps_empty_content(self):
        handler = ElicitationHandler("pay", {"timeout": 5})
        params = _camel_form_params(
            {"properties": {"approved": {"type": "boolean"}}}
        )

        with patch(
            "tools.approval.request_elicitation_consent", return_value="accept"
        ):
            result = asyncio.run(handler(context=None, params=params))

        assert result.action == "accept"
        assert result.content == {}

    def test_accept_with_unanswerable_required_field_fails_closed(self):
        handler = ElicitationHandler("pay", {"timeout": 5})
        params = _camel_form_params(
            {
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
            }
        )

        with patch(
            "tools.approval.request_elicitation_consent", return_value="accept"
        ):
            result = asyncio.run(handler(context=None, params=params))

        assert result.action == "decline"
        assert handler.metrics["declined"] == 1
        assert handler.metrics["accepted"] == 0

    def test_decline_returns_no_content(self):
        handler = ElicitationHandler("powerbi", {"timeout": 5})
        params = _camel_form_params(PBI_CONFIRM_SCHEMA)

        with patch(
            "tools.approval.request_elicitation_consent", return_value="decline"
        ):
            result = asyncio.run(handler(context=None, params=params))

        assert result.action == "decline"
        assert not result.content
