"""Form-mode elicitation answers: schema resolution + affirmative form filling.

Servers such as Microsoft's powerbi-modeling-mcp gate write operations behind a
form-mode elicitation whose required field must literally carry the affirmative
enum value (e.g. {"Confirm the operation": "Yes"}). These tests cover the two
failure modes that made every user-approved write come back "declined":

1. the SDK field is camelCase (``ElicitRequestFormParams.requestedSchema``) —
   reading ``requested_schema`` silently yielded an empty schema;
2. ``action="accept"`` was returned with ``content={}`` — an empty form, which
   such servers treat as unconfirmed.
"""

import asyncio

import pytest

from tools.mcp_tool import _affirmative_form_content

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


def test_fills_required_enum_with_affirmative():
    assert _affirmative_form_content(PBI_CONFIRM_SCHEMA) == {
        "Confirm the operation": "Yes"
    }


def test_empty_or_invalid_schema_yields_empty_content():
    assert _affirmative_form_content({}) == {}
    assert _affirmative_form_content(None) == {}
    assert _affirmative_form_content({"properties": {}}) == {}
    assert _affirmative_form_content({"properties": "not-a-dict"}) == {}


def test_boolean_and_freetext_required_fields():
    schema = {
        "properties": {"ok": {"type": "boolean"}, "note": {"type": "string"}},
        "required": ["ok", "note"],
    }
    assert _affirmative_form_content(schema) == {"ok": True, "note": "Yes"}


def test_only_required_fields_are_filled():
    schema = {
        "properties": {
            "Confirm": {"enum": ["Yes", "No"]},
            "optional_note": {"type": "string"},
        },
        "required": ["Confirm"],
    }
    assert _affirmative_form_content(schema) == {"Confirm": "Yes"}


def test_missing_required_list_treats_all_properties_as_required():
    schema = {
        "properties": {
            "Proceed": {"enum": ["Continue the operation", "Decline the operation"]}
        }
    }
    # No affirmative keyword matches the long labels -> first enum entry wins.
    assert _affirmative_form_content(schema) == {"Proceed": "Continue the operation"}


def test_accept_path_reads_camelcase_schema_and_fills_form(monkeypatch):
    """End-to-end through ElicitationHandler.__call__ with real SDK types."""
    mcp_types = pytest.importorskip("mcp.types")

    import tools.approval as approval
    import tools.mcp_tool as mcp_tool

    monkeypatch.setattr(
        approval, "request_elicitation_consent", lambda *a, **k: "accept"
    )

    handler = mcp_tool.ElicitationHandler.__new__(mcp_tool.ElicitationHandler)
    handler.server_name = "powerbi"
    handler.timeout = 5
    handler.owner = None
    handler.metrics = {"requests": 0, "accepted": 0, "declined": 0, "errors": 0}

    params = mcp_types.ElicitRequestFormParams(
        mode="form",
        message="Are you sure you want to perform operations that will modify your database?",
        requestedSchema=PBI_CONFIRM_SCHEMA,
    )

    result = asyncio.run(handler(None, params))
    assert result.action == "accept"
    assert result.content == {"Confirm the operation": "Yes"}


def test_declined_consent_returns_no_content(monkeypatch):
    mcp_types = pytest.importorskip("mcp.types")

    import tools.approval as approval
    import tools.mcp_tool as mcp_tool

    monkeypatch.setattr(
        approval, "request_elicitation_consent", lambda *a, **k: "decline"
    )

    handler = mcp_tool.ElicitationHandler.__new__(mcp_tool.ElicitationHandler)
    handler.server_name = "powerbi"
    handler.timeout = 5
    handler.owner = None
    handler.metrics = {"requests": 0, "accepted": 0, "declined": 0, "errors": 0}

    params = mcp_types.ElicitRequestFormParams(
        mode="form", message="Sure?", requestedSchema=PBI_CONFIRM_SCHEMA
    )

    result = asyncio.run(handler(None, params))
    assert result.action == "decline"
    assert not result.content
