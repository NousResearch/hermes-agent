"""Contracts for the MCP Apps host methods (``tui_gateway/methods_mcp_apps.py``, stable spec 2026-01-26).

Results that are MCP results travel as the SDK produced them (camelCase MCP keys inside an open model)."""

from __future__ import annotations

from pydantic import Field

from .base import JsonValue, Result, WireEnum
from .common import EmptyResult, OpenModel, SessionParams
from .registry import method


class McpAppCallParams(SessionParams):
    """``tool_call_id``: the model tool call whose MCP App view sends the request."""

    tool_call_id: str = Field(min_length=1)


class McpAppViewStatus(WireEnum):
    running = "running"
    result = "result"
    cancelled = "cancelled"


class McpAppViewResult(Result):
    """``status``: ``running`` while the call runs, ``result`` once its ``CallToolResult`` is stored,
    ``cancelled`` when the call ended without one (interrupted, failed before a result, over the MCP
    hard cap). ``arguments`` are what the server received (tool-input); ``result`` is the raw
    ``CallToolResult`` (tool-result, ``isError`` kept); ``tool`` is the live ``Tool`` definition
    (``hostContext.toolInfo.tool``)."""

    status: McpAppViewStatus
    arguments: dict[str, JsonValue]
    result: dict[str, JsonValue] | None = None
    tool: dict[str, JsonValue]


method(
    "mcp.app.view",
    params=McpAppCallParams,
    result=McpAppViewResult,
    doc="An MCP App view's tool call: its arguments, its raw result (once stored) and the live tool definition.",
)


class McpAppReadResourceParams(McpAppCallParams):
    uri: str = Field(min_length=1)


class McpAppReadResourceResult(OpenModel):
    """The SDK ``ReadResourceResult`` unchanged."""

    contents: list[JsonValue]


method(
    "mcp.app.read_resource",
    params=McpAppReadResourceParams,
    result=McpAppReadResourceResult,
    doc="``resources/read`` on the server the view's tool call ran on.",
)


class McpAppCallToolParams(McpAppCallParams):
    name: str = Field(min_length=1)
    arguments: dict[str, JsonValue] | None = None


class McpAppCallToolResult(OpenModel):
    """The SDK ``CallToolResult`` unchanged."""

    content: list[JsonValue] = Field(default_factory=list)


method(
    "mcp.app.call_tool",
    params=McpAppCallToolParams,
    result=McpAppCallToolResult,
    doc="A view's ``tools/call``: only an app-visible, operator-allowed tool on the call's own server, through the "
        "server's trust gate.",
)


class McpAppUpdateModelContextParams(McpAppCallParams):
    """``ui/update-model-context``: MCP ``ContentBlock`` objects and/or a structured object."""

    content: list[JsonValue] | None = None
    structured_content: dict[str, JsonValue] | None = None


method(
    "mcp.app.update_model_context",
    params=McpAppUpdateModelContextParams,
    result=EmptyResult,
    doc="Replace the view's model context; the next turn's model input carries the latest one, never the visible text.",
)
