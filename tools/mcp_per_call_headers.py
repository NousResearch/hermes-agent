"""MCP 2.0.0 per-request header bridge.

The public ``ClientSession.call_tool`` API has no header argument.  MCP 2.0.0's
real dispatcher accepts ``CallOptions.headers`` and the Streamable HTTP
transport applies those headers to that request's POST.  Keep this isolated so
an SDK upgrade fails closed instead of silently reverting to connection-wide
credentials.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Mapping
from typing import Any

_SUPPORTED_MCP_VERSION = "2.0.0"


def _require_supported_sdk() -> None:
    installed = importlib.metadata.version("mcp")
    if installed != _SUPPORTED_MCP_VERSION:
        raise RuntimeError(
            f"per-call MCP authorization requires mcp=={_SUPPORTED_MCP_VERSION}; found {installed}"
        )


async def call_tool_with_headers(
    session: Any,
    name: str,
    arguments: dict[str, Any] | None,
    *,
    headers: Mapping[str, str],
):
    """Call ``tools/call`` with request-local HTTP headers on MCP 2.0.0."""
    _require_supported_sdk()

    import mcp.types as types
    from mcp.client import session as sdk_session

    request = types.CallToolRequest(
        params=types.CallToolRequestParams(name=name, arguments=arguments)
    )
    data = request.model_dump(by_alias=True, mode="json", exclude_none=True)
    method = data["method"]
    options: dict[str, Any] = {}
    session._stamp(data, options)
    options_headers = options.setdefault("headers", {})
    if (key := type(request).name_param) is not None and sdk_session.MCP_NAME_HEADER not in options_headers:
        params_data = data.get("params") or {}
        request_name = params_data.get(key)
        if not isinstance(request_name, str):
            raise ValueError(f"{method} requires params[{key!r}] for Mcp-Name")
        options_headers[sdk_session.MCP_NAME_HEADER] = sdk_session.encode_header_value(request_name)
    options_headers.update(dict(headers))
    timeout = session._session_read_timeout_seconds
    if timeout is not None:
        options["timeout"] = timeout

    raw = await session._dispatcher.send_raw_request(method, data.get("params"), options)
    sdk_session._clamp_inbound_ttl(raw)
    version = session._negotiated_version or "2025-11-25"
    try:
        sdk_session._methods.validate_server_result(method, version, raw)
    except KeyError:
        pass
    result = session._call_tool_adapter.validate_python(raw, by_name=False)
    if not isinstance(result, types.CallToolResult):
        raise RuntimeError("MCP tools/call returned a non-terminal result")
    if not result.is_error:
        await session.validate_tool_result(name, result)
    return result
