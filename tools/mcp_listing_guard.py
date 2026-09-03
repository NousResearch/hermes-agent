"""Keep one malformed tool from taking down a whole MCP server's catalog.

The ``mcp`` 2.x SDK validates every ``tools/list`` page against the wire
schema of the *negotiated protocol version* before the client ever sees it,
and it validates the page as a whole: one tool whose definition the schema
rejects fails the entire response. Hermes then parks the server, so a
single bad tool takes every other tool on that server with it — and the
error names the offender only by index (``tools.291.inputSchema...``).

Issue #101669 hit this with a boolean property subschema
(``"properties": {"refresh": true}``), which the pinned SDK's 2025-11-25
wire model rejected although it is legal JSON Schema. *Accepting* that
shape is the SDK's job: upstream widened the model
(modelcontextprotocol/python-sdk#3354) and Hermes carries the same widening
for the pinned release. This module is the complementary layer the
reporter asked for independently of that fix: whatever the SDK still
rejects should cost that one tool, not the server.

It sits between the transport and ``ClientSession`` and, for each
``tools/list`` page, validates every tool on its own against the SDK's own
per-version ``Tool`` model. Tools that fail are dropped *individually*,
logged by name with the failing path, and recorded on the owning server so
status can show what went missing; the rest of the catalog loads.

Everything here is best-effort: any unexpected shape or SDK difference makes
the guard pass the message through untouched, so behaviour can never be worse
than the SDK's own.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, get_args

logger = logging.getLogger(__name__)

DroppedTools = List[Tuple[str, str]]


def _tool_label(tool: Any, index: int) -> str:
    name = tool.get("name") if isinstance(tool, dict) else None
    return name if isinstance(name, str) and name else f"<tools[{index}]>"


def _versioned_tool_model(version: str) -> Optional[Any]:
    """Return the SDK's ``Tool`` wire model for ``version``.

    Derived from the same ``tools/list`` result model the SDK validates
    against, so a tool that passes here passes the SDK — including any
    widening Hermes applies to that model at import time. ``None`` when
    this SDK generation has no per-version surface (mcp 1.x, which
    validates ``inputSchema`` as a free-form dict) or the version is one it
    doesn't know.
    """
    try:
        from mcp_types.methods import SERVER_RESULTS
    except ImportError:
        return None
    try:
        page_model = SERVER_RESULTS[("tools/list", version)]
        (tool_model,) = get_args(page_model.model_fields["tools"].annotation)
    except Exception:
        return None
    return tool_model if callable(getattr(tool_model, "model_validate", None)) else None


def _error_summary(exc: BaseException) -> str:
    """Render the first pydantic error as ``<path within the tool>: <message>``."""
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            first = errors()[0]
            path = ".".join(str(part) for part in first.get("loc", ()))
            message = str(first.get("msg", "")).strip()
            return f"{path}: {message}" if path else message
        except Exception:  # pragma: no cover - defensive against SDK drift
            pass
    return str(exc).splitlines()[0][:200]


def drop_invalid_tools(result: Dict[str, Any], version: str) -> DroppedTools:
    """Remove the tools the SDK would reject, in place; keep the rest.

    Each tool is validated on its own against the versioned ``Tool`` model,
    so a broken page-level field (a bad ``nextCursor``, a negative
    ``ttlMs`` the SDK clamps itself) neither hides an invalid tool nor
    condemns a valid one. Returns ``[(tool_name, reason), ...]`` for the
    tools removed.
    """
    tools = result.get("tools")
    if not isinstance(tools, list) or not tools:
        return []
    tool_model = _versioned_tool_model(version)
    if tool_model is None:
        return []
    try:
        from pydantic import ValidationError
    except ImportError:  # pragma: no cover - pydantic is an SDK dependency
        return []
    kept: List[Any] = []
    dropped: DroppedTools = []
    for index, tool in enumerate(tools):
        try:
            tool_model.model_validate(tool, by_name=False)
        except ValidationError as exc:
            dropped.append((_tool_label(tool, index), _error_summary(exc)))
            continue
        except Exception:
            # Not a validation verdict: keep the tool and let the SDK decide.
            logger.debug("per-tool MCP schema validation failed unexpectedly", exc_info=True)
        kept.append(tool)
    if dropped:
        result["tools"] = kept
    return dropped


def _unwrap_response(item: Any) -> Optional[Dict[str, Any]]:
    """Return the JSON-RPC ``result`` dict carried by a stream item, if any.

    Handles both SDK generations: mcp 1.x wraps the message in a
    ``JSONRPCMessage`` root model (``item.message.root``), mcp 2.x carries the
    ``JSONRPCResponse`` directly (``item.message``).
    """
    message = getattr(item, "message", None)
    if message is None:
        return None
    message = getattr(message, "root", message)
    result = getattr(message, "result", None)
    return result if isinstance(result, dict) else None


class ToolListingGuard:
    """Read-stream proxy that isolates invalid tools before the SDK rejects the page.

    Implements the SDK's ``ReadStream`` protocol (``receive``/async
    iteration/context management) by delegating to the wrapped stream, and
    forwards any other attribute (``last_context``, ``clone``, ...) untouched.

    The protocol version used for per-tool validation comes from
    ``version_getter`` (the live session's negotiated version) and, before
    that is available, from the ``protocolVersion`` the server announced in
    the handshake response that passed through this same stream — the exact
    value the SDK adopts.
    """

    def __init__(
        self,
        inner: Any,
        *,
        server_name: str,
        version_getter: Optional[Callable[[], Optional[str]]] = None,
        on_drop: Optional[Callable[[DroppedTools], None]] = None,
    ) -> None:
        self._inner = inner
        self._server_name = server_name
        self._version_getter = version_getter
        self._on_drop = on_drop
        self._announced_version: Optional[str] = None
        # Latch so a keepalive that re-lists tools every few seconds does
        # not repeat the same verdict into agent.log on every tick.
        self._reported_drops: Set[Tuple[str, str]] = set()

    # -- ReadStream protocol -------------------------------------------------

    async def receive(self):
        return self._process(await self._inner.receive())

    def __aiter__(self):
        return self

    async def __anext__(self):
        return self._process(await self._inner.__anext__())

    async def __aenter__(self):
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await self._inner.__aexit__(exc_type, exc_val, exc_tb)

    async def aclose(self) -> None:
        await self._inner.aclose()

    def __getattr__(self, name: str):
        # Anything outside the protocol (``last_context``, ``clone``, ...)
        # belongs to the wrapped stream. Private names are never forwarded:
        # a lookup of ``_inner`` itself before ``__init__`` ran (copy /
        # unpickling paths) would otherwise recurse forever.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._inner, name)

    # -- isolation -----------------------------------------------------------

    def _process(self, item: Any) -> Any:
        try:
            result = _unwrap_response(item)
            if result is None:
                return item
            announced = result.get("protocolVersion")
            if isinstance(announced, str) and announced:
                self._announced_version = announced
            if isinstance(result.get("tools"), list):
                self._isolate_invalid_tools(result)
        except Exception:
            # Never let the guard itself break a live session.
            logger.debug(
                "MCP server '%s': tool-listing guard skipped a message",
                self._server_name, exc_info=True,
            )
        return item

    def _version(self) -> Optional[str]:
        version = self._version_getter() if self._version_getter else None
        if isinstance(version, str) and version:
            return version
        return self._announced_version

    def _isolate_invalid_tools(self, result: Dict[str, Any]) -> None:
        version = self._version()
        if not version:
            return
        dropped = drop_invalid_tools(result, version)
        for name, reason in dropped:
            first_time = (name, reason) not in self._reported_drops
            self._reported_drops.add((name, reason))
            logger.log(
                logging.WARNING if first_time else logging.DEBUG,
                "MCP server '%s': dropping tool '%s' whose definition fails "
                "MCP %s schema validation (%s); the server's other tools "
                "remain available",
                self._server_name, name, version, reason,
            )
        if dropped and self._on_drop is not None:
            self._on_drop(dropped)
