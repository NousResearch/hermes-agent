"""Invariant tests for model-facing connector names and wire-slug recovery."""

import pytest

from tools.tool_gateway.names import (
    format_connector_name,
    parse_connector_name,
    vendor_slug_candidates,
)


ENCODE_CASES = [
    (
        "gmail",
        "GMAIL_GET_PROFILE",
        "connectors__gmail__GET_PROFILE",
        True,
    ),
    (
        "better_stack_mcp",
        "BETTER_STACK_MCP_ACKNOWLEDGE_INCIDENT",
        "connectors__better_stack_mcp__ACKNOWLEDGE_INCIDENT",
        True,
    ),
    (
        "gmail",
        "FETCH_PROFILE",
        "connectors__gmail__FETCH_PROFILE",
        False,
    ),
    (
        "gmail",
        "GMAIL",
        "connectors__gmail__GMAIL",
        False,
    ),
    (
        "granola",
        "GRANOLA_MCP_GET_MEETINGS",
        "connectors__granola__MCP_GET_MEETINGS",
        True,
    ),
]


@pytest.mark.parametrize("connector,vendor_slug,composed,_stripped", ENCODE_CASES)
def test_format_connector_name_strips_only_the_exact_toolkit_prefix(
    connector, vendor_slug, composed, _stripped
):
    assert format_connector_name(connector, vendor_slug) == composed


def test_vendor_slug_candidates_are_prefixed_then_literal():
    assert vendor_slug_candidates("gmail", "GET_PROFILE") == (
        "GMAIL_GET_PROFILE",
        "GET_PROFILE",
    )


@pytest.mark.parametrize("connector,vendor_slug,composed,stripped", ENCODE_CASES)
def test_every_encoded_vendor_slug_has_a_wire_recovery_candidate(
    connector, vendor_slug, composed, stripped
):
    parsed = parse_connector_name(composed)
    assert parsed is not None

    candidates = vendor_slug_candidates(parsed.connector, parsed.tool)
    assert vendor_slug in candidates
    if stripped:
        assert candidates[0] == vendor_slug
