"""Trusted request-scoped MCP argument injection."""

from tools.mcp_tool import (
    _inject_trusted_mcp_arguments,
    bind_trusted_mcp_context,
    reset_trusted_mcp_context,
)


def test_trusted_context_overwrites_matching_args_without_mutating_input():
    original = {"act_as_token": "model-value", "query": "Meyer"}
    token = bind_trusted_mcp_context({
        "carlo_webapi": {
            "act_as_token": "trusted-value",
            "turn_id": "a" * 32,
            "confirmed_action_token": "",
            "ignored": "not-in-schema",
        }
    })
    try:
        result = _inject_trusted_mcp_arguments(
            "carlo_webapi", original,
            {"act_as_token", "turn_id", "confirmed_action_token", "query"},
        )
        untouched = _inject_trusted_mcp_arguments(
            "mcp_logit", original, {"act_as_token", "query"}
        )
    finally:
        reset_trusted_mcp_context(token)

    assert result == {
        "act_as_token": "trusted-value",
        "turn_id": "a" * 32,
        "confirmed_action_token": "",
        "query": "Meyer",
    }
    assert untouched is original
    assert original == {"act_as_token": "model-value", "query": "Meyer"}


def test_trusted_context_resets_after_turn():
    token = bind_trusted_mcp_context({"carlo_webapi": {"act_as_token": "trusted"}})
    reset_trusted_mcp_context(token)
    original = {"act_as_token": "model"}
    assert _inject_trusted_mcp_arguments(
        "carlo_webapi", original, {"act_as_token"}
    ) is original
