"""MCP text, resources and prompts respect the shared redaction toggle."""

import json
from types import SimpleNamespace as NS

import pytest

from tools.mcp_tool_handlers import _render_call_tool_result, _render_get_prompt, _render_read_resource


@pytest.mark.parametrize("renderer", ["tool", "resource", "prompt"])
def test_mcp_text_is_redacted_before_history(renderer, monkeypatch):
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)
    secret = "sk-ant-api03-" + "a" * 90
    block = NS(text=f"ANTHROPIC_API_KEY={secret}")
    if renderer == "tool":
        output = _render_call_tool_result(NS(content=[block], isError=False), "test")
    elif renderer == "resource":
        output = json.dumps(_render_read_resource(NS(contents=[block]), "test"))
    else:
        output = json.dumps(_render_get_prompt(NS(messages=[NS(role="user", content=block)]), "test"))
    assert secret not in output
    assert "ANTHROPIC_API_KEY" in output


def test_structured_result_and_metadata_are_redacted_without_changing_shape(monkeypatch):
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)
    secret = "sk-ant-api03-" + "b" * 90
    result = NS(content=[], isError=False, structuredContent={"api_key": secret, "count": 2},
                meta={"vendor.example/trace": secret})
    output = _render_call_tool_result(result, "test")
    assert secret not in output
    assert json.loads(output)["result"]["count"] == 2
