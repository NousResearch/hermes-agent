"""Keep one malformed tool from taking down a whole MCP server's catalog.

The ``mcp`` 2.x SDK validates each ``tools/list`` page as a whole against the negotiated protocol
version's wire schema, so one tool the schema rejects (a missing ``inputSchema``, a non-list
``icons``, a boolean property subschema on older SDK models) fails the entire response and Hermes
parks the server with zero tools (#101669).

The guard wraps the session's stream pair between the transport and ``ClientSession``. The write
proxy remembers each outbound request's method by id; the read proxy uses that to recognise the
response to ``tools/list`` (and only that: a custom method whose result carries a ``tools`` list
passes through as the SDK forwards it), validates every tool on its own against the SDK's own
per-version ``Tool`` model, and drops just the failing ones, logged by name. Anything unexpected
passes through untouched, so behaviour is never worse than the SDK's own.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Callable, Optional, get_args

logger = logging.getLogger(__name__)

DroppedTools = list[tuple[str, str]]

_LISTING_METHOD = "tools/list"
_HANDSHAKE_METHODS = frozenset({"initialize", "server/discover"})
# A response retires its id, so the ledger only grows when a peer never answers.
_MAX_PENDING_REQUESTS = 4096


def _versioned_tool_model(version: str) -> Optional[Any]:
    """The SDK's ``Tool`` model, from the same ``tools/list`` result model the SDK validates with."""
    from mcp_types.methods import SERVER_RESULTS

    page_model = SERVER_RESULTS.get((_LISTING_METHOD, version))
    if page_model is None:
        return None
    (tool_model,) = get_args(page_model.model_fields["tools"].annotation)
    return tool_model


def _error_summary(exc: Any) -> str:
    first = exc.errors()[0]
    path = ".".join(str(part) for part in first.get("loc", ()))
    msg = str(first.get("msg", ""))
    return f"{path}: {msg}" if path else msg


def drop_invalid_tools(result: dict[str, Any], version: str) -> DroppedTools:
    """Remove, in place, the tools the SDK would reject; return ``[(name, reason), ...]``.

    Each tool is validated alone, so a broken page-level field neither hides nor condemns a tool."""
    from pydantic import ValidationError

    tool_model = _versioned_tool_model(version)
    if tool_model is None:
        return []
    kept, dropped = [], []
    for index, tool in enumerate(result["tools"]):
        try:
            tool_model.model_validate(tool, by_name=False)
        except ValidationError as exc:
            name = tool.get("name") if isinstance(tool, dict) else None
            label = name if isinstance(name, str) and name else f"<tools[{index}]>"
            dropped.append((label, _error_summary(exc)))
            continue
        kept.append(tool)
    if dropped:
        result["tools"] = kept
    return dropped


class _StreamProxy:
    """Delegate the SDK stream protocol (context management, ``aclose``, extras) to a wrapped stream."""

    def __init__(self, inner: Any, pending: OrderedDict) -> None:
        self._inner = inner
        self._pending = pending  # outbound request id -> method, shared by the pair

    async def __aenter__(self):
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await self._inner.__aexit__(exc_type, exc_val, exc_tb)

    async def aclose(self) -> None:
        await self._inner.aclose()

    def __getattr__(self, name: str):
        # Extras (``last_context``, ``clone``) belong to the wrapped stream. Privates are never
        # forwarded: a lookup of ``_inner`` before ``__init__`` (copy/unpickle) would recurse.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._inner, name)


class GuardedWriteStream(_StreamProxy):
    """Write proxy that records the method of every outbound request."""

    async def send(self, item: Any) -> None:
        message = getattr(item, "message", None)
        method, request_id = getattr(message, "method", None), getattr(message, "id", None)
        if isinstance(method, str) and isinstance(request_id, (int, str)):
            self._pending[request_id] = method
            while len(self._pending) > _MAX_PENDING_REQUESTS:
                self._pending.popitem(last=False)
        await self._inner.send(item)


class ToolListingGuard(_StreamProxy):
    """Read proxy that isolates invalid tools before the SDK rejects a ``tools/list`` page.

    Validation uses the live session's negotiated version (``version_getter``) or, before that is
    set, the ``protocolVersion`` the server returned for the handshake: the value the SDK adopts."""

    def __init__(self, inner: Any, pending: OrderedDict, *, server_name: str,
                 version_getter: Optional[Callable[[], Optional[str]]] = None,
                 on_drop: Optional[Callable[[DroppedTools], None]] = None) -> None:
        super().__init__(inner, pending)
        self._server_name = server_name
        self._version_getter = version_getter
        self._on_drop = on_drop
        self._announced_version: Optional[str] = None
        self._reported: set[tuple[str, str]] = set()  # keepalive re-lists must not re-WARN every tick

    async def receive(self):
        return self._process(await self._inner.receive())

    def __aiter__(self):
        return self

    async def __anext__(self):
        return self._process(await self._inner.__anext__())

    def _process(self, item: Any) -> Any:
        message = getattr(item, "message", None)
        request_id = getattr(message, "id", None)
        if not isinstance(request_id, (int, str)) or getattr(message, "method", None) is not None:
            return item  # a notification or a server-originated request
        method = self._pending.pop(request_id, None)
        result = getattr(message, "result", None)
        if method is None or not isinstance(result, dict):
            return item  # an unknown request, or an error response the SDK judges itself
        try:
            if method in _HANDSHAKE_METHODS and isinstance(result.get("protocolVersion"), str):
                self._announced_version = result["protocolVersion"]
            elif method == _LISTING_METHOD and isinstance(result.get("tools"), list):
                self._isolate(result)
        except Exception:  # SDK drift must never break a live session
            logger.debug("MCP server '%s': tool-listing guard skipped a message", self._server_name, exc_info=True)
        return item

    def _isolate(self, result: dict[str, Any]) -> None:
        version = (self._version_getter() if self._version_getter else None) or self._announced_version
        if not version:
            return
        dropped = drop_invalid_tools(result, version)
        for name, reason in dropped:
            first_time = (name, reason) not in self._reported
            self._reported.add((name, reason))
            logger.log(logging.WARNING if first_time else logging.DEBUG,
                       "MCP server '%s': dropping tool '%s' whose definition fails MCP %s schema validation "
                       "(%s); the server's other tools remain available", self._server_name, name, version, reason)
        if dropped and self._on_drop is not None:
            self._on_drop(dropped)


def guard_session_streams(read_stream: Any, write_stream: Any, *, server_name: str,
                          version_getter: Optional[Callable[[], Optional[str]]] = None,
                          on_drop: Optional[Callable[[DroppedTools], None]] = None,
                          ) -> tuple[ToolListingGuard, GuardedWriteStream]:
    """Wrap a transport's stream pair for ``ClientSession``; the two proxies share one request ledger."""
    pending: OrderedDict = OrderedDict()
    return (ToolListingGuard(read_stream, pending, server_name=server_name,
                             version_getter=version_getter, on_drop=on_drop),
            GuardedWriteStream(write_stream, pending))
