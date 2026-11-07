"""Tolerant ``outputSchema`` revalidation for MCP tool results (#101330).

``mcp`` 2.x ``ClientSession.call_tool`` re-validates every non-``isError`` result against the tool's
declared ``outputSchema`` and raises ``RuntimeError`` when ``structuredContent`` is missing or does
not conform. The RPC has already completed by then: the server is reachable and answered, and the
result's ``content`` blocks usually carry the same data as text. Raising discards that text, the
call handler counts a transport-style breaker strike, and after ``_CIRCUIT_BREAKER_THRESHOLD`` calls
every tool on the server reports "unreachable" for the cooldown. Two real shapes:

* a server that declares a schema but answers some calls with text only (Semrush MCP v2 returns
  its ``no_api_units`` quota notice as a bare text block on every tool);
* a server whose ``structuredContent`` violates its own schema (``required`` fields returned as
  ``null``; the Bonusly case in #101330).

:func:`install_lenient_output_validation` shadows a session's revalidator so that, for a tool with
a declared schema, either failure logs one WARNING per (server, tool) (DEBUG afterwards) and the
result is returned intact; ``_render_call_tool_result`` then serves ``content`` and the success
path resets the breaker. Tools without a schema, and tools the session has not listed, go to the
SDK untouched (its no-op, or its own ``list_tools()`` refresh and "not listed" warning).
"""
import inspect
import logging
from typing import Any

logger = logging.getLogger("tools.mcp_tool")

# Public in mcp 2.x; mcp 1.x spelled it ``_validate_tool_result``.
_VALIDATOR_NAMES: tuple[str, ...] = ("validate_tool_result", "_validate_tool_result")
_HOOK_MARKER = "_hermes_lenient_output_validation"
_PREVIEW_CHARS = 300
# (server, tool) pairs already warned about: one WARNING per pair, DEBUG afterwards.
_warned: set[tuple[str, str]] = set()


def _declared_schema(session: Any, name: str) -> Any:
    """The session's cached ``outputSchema`` for *name*, or None (no schema, or never listed).
    Read at call time: a complete ``tools/list`` rebinds the SDK's cache dict."""
    schemas = getattr(session, "_tool_output_schemas", None)
    return schemas.get(name) if isinstance(schemas, dict) else None


def _structured_content(result: Any) -> Any:
    """``structuredContent`` of a ``CallToolResult`` under either SDK spelling, or None."""
    for attr in ("structured_content", "structuredContent"):
        value = getattr(result, attr, None)
        if value is not None:
            return value
    return None


def _preview(text: str) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= _PREVIEW_CHARS else flat[:_PREVIEW_CHARS] + "..."


def _text_preview(result: Any) -> str:
    """First non-empty text block of *result*, single-lined and capped, for the operator log."""
    for block in (getattr(result, "content", None) or []):
        text = getattr(block, "text", None)
        if text:
            return _preview(text)
    return "(no text content)"


def _log_degraded(server_name: str, tool: str, problem: str, result: Any) -> None:
    key = (server_name, tool)
    level = logging.DEBUG if key in _warned else logging.WARNING
    _warned.add(key)
    logger.log(level, "MCP server '%s': tool '%s' %s; serving its content unvalidated: %s",
               server_name, tool, problem, _text_preview(result))


def install_lenient_output_validation(session: Any, server_name: str) -> bool:
    """Shadow *session*'s output-schema revalidator with a tolerant one. Returns True when a hook
    is in place (installed now or earlier), False on SDKs without the revalidation step or when
    the instance refuses the attribute. Idempotent: a second call on the same session is a no-op."""
    original = None
    attr = ""
    for attr in _VALIDATOR_NAMES:
        original = getattr(session, attr, None)
        if callable(original):
            break
    else:
        return False
    if getattr(original, _HOOK_MARKER, False):
        return True

    async def _call_original(name: str, result: Any, *args: Any, **kwargs: Any) -> Any:
        out = original(name, result, *args, **kwargs)
        return await out if inspect.isawaitable(out) else out

    async def _lenient_validate(name: str, result: Any, *args: Any, **kwargs: Any) -> Any:
        if _declared_schema(session, str(name)) is None:
            return await _call_original(name, result, *args, **kwargs)
        if _structured_content(result) is None:
            _log_degraded(server_name, str(name), "declares an outputSchema but returned no structuredContent", result)
            return None
        try:
            return await _call_original(name, result, *args, **kwargs)
        except RuntimeError as exc:
            # With the schema cached the SDK never refreshes, so every RuntimeError here is a
            # validation verdict (non-conforming content, or an uncompilable/unresolvable schema).
            _log_degraded(server_name, str(name), f"failed outputSchema validation ({_preview(str(exc))})", result)
            return None

    setattr(_lenient_validate, _HOOK_MARKER, True)
    try:
        setattr(session, attr, _lenient_validate)
    except (AttributeError, TypeError) as exc:  # __slots__ / frozen instance: leave the SDK strict
        logger.debug("MCP server '%s': could not install lenient output validation: %s", server_name, exc)
        return False
    return True
