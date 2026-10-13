"""MCP sampling uses the selected route when native model identity is absent."""

from types import SimpleNamespace

import pytest

from tools.mcp_tool_sampling import SamplingHandler


@pytest.mark.parametrize("tool_use", [False, True])
@pytest.mark.parametrize("observed_model", [None, "served-model"])
@pytest.mark.parametrize("selected_model", ["requested-model", "fallback-model"])
@pytest.mark.asyncio
async def test_sampling_preserves_native_model_and_returns_required_identity(
    monkeypatch, tool_use, observed_model, selected_model,
):
    from mcp.types import CreateMessageResult, CreateMessageResultWithTools, TextContent, ToolUseContent

    message = SimpleNamespace(content="Answer.", tool_calls=None)
    if tool_use:
        message.tool_calls = [SimpleNamespace(
            id="call-provider", function=SimpleNamespace(name="inspect", arguments='{"path":"example"}'),
        )]
    response = SimpleNamespace(
        model=observed_model, usage=None,
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls" if tool_use else "stop")],
    )

    def call_llm(**kwargs):
        if (route_info := kwargs.get("route_info")) is not None:
            route_info.update(provider="gemini", model=selected_model)
        return response

    monkeypatch.setattr("agent.auxiliary_client.call_llm", call_llm)
    handler = SamplingHandler("fixture", {"model": "requested-model"})
    params = SimpleNamespace(messages=[], maxTokens=32)

    result = await handler(None, params)

    model = observed_model or selected_model
    expected = (
        CreateMessageResultWithTools(
            role="assistant", model=model, stopReason="toolUse",
            content=[ToolUseContent(type="tool_use", id="call-provider", name="inspect", input={"path": "example"})],
        ) if tool_use else CreateMessageResult(
            role="assistant", model=model, stopReason="endTurn",
            content=TextContent(type="text", text="Answer."),
        )
    )
    assert (result, response.model) == (expected, observed_model)
