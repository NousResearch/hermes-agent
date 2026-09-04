"""MCP session-expiry classification helpers.

The classifier body below is the approved byte-verbatim extraction window.
"""

import logging

logger = logging.getLogger(__name__)

_SESSION_EXPIRED_MARKERS: tuple = (
    "invalid or expired session",
    "expired session",
    "session expired",
    "session not found",
    "unknown session",
    "session terminated",
    "closedresourceerror",
    "closed resource",
    "transport is closed",
    "connection closed",
    "broken pipe",
    "end of file",
)

# Upper bound on exception-graph nodes inspected by
# ``_is_session_expired_error``. The visited-identity set already breaks
# cycles across ``exceptions`` / ``__cause__`` / ``__context__``; the
# budget additionally bounds pathological acyclic graphs (e.g. deeply
# chained retries) so classification always terminates promptly. Kept
# comfortably above ``sys.getrecursionlimit()`` so legitimately deep
# wrapper stacks (task-group nesting) are still fully scanned.
_EXC_TRAVERSAL_MAX_NODES = 10_000


def _is_session_expired_error(exc: BaseException) -> bool:
    """Return True if ``exc`` looks like an MCP transport session expiry.

    Streamable HTTP MCP servers may garbage-collect server-side session
    state while the OAuth token remains valid — idle TTL, server
    restart, horizontal-scaling pod rotation, etc.  The SDK surfaces
    this as a JSON-RPC error whose message contains phrases like
    ``"Invalid or expired session"``.  This class of failure is
    distinct from :func:`_is_auth_error`: re-running the OAuth refresh
    flow would be pointless because the access token is fine.  What's
    needed is a transport reconnect — tear down and rebuild the
    ``streamablehttp_client`` + ``ClientSession`` pair, which is
    exactly what ``MCPServerTask._reconnect_event`` triggers.
    """
    # AnyIO's stream exceptions are commonly message-less. In particular,
    # ``str(ClosedResourceError()) == ""``, so marker matching alone misses the
    # exact failure emitted by both MCP stdio and HTTP transports.
    try:
        from anyio import BrokenResourceError, ClosedResourceError, EndOfStream

        transport_error_types = (
            BrokenResourceError,
            ClosedResourceError,
            EndOfStream,
        )
    except ImportError:  # pragma: no cover - AnyIO is supplied by the MCP SDK
        transport_error_types = ()

    # ExceptionGroup trees can be arbitrarily deep or even cyclic when custom
    # exceptions expose ``exceptions``, and chained exceptions
    # (``raise X from Y`` / implicit ``__context__``) can likewise form
    # cycles when handlers re-raise previously seen exceptions. Traverse
    # once, iteratively, with an identity-visited set AND a bounded node
    # budget so classification can never spin, and inspect every reachable
    # node so user interruption always overrides transport markers or types
    # found elsewhere in the graph. The chain traversal matters for real
    # failures: SDK wrappers often raise a generic RuntimeError *from* the
    # message-less ClosedResourceError, leaving the transport signal only
    # reachable via ``__cause__``.
    stack: "list[BaseException | None]" = [exc]
    seen: set[int] = set()
    transport_error_found = False
    budget = _EXC_TRAVERSAL_MAX_NODES
    while stack and budget > 0:
        current = stack.pop()
        if current is None:
            continue
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        budget -= 1

        if isinstance(current, InterruptedError):
            return False
        if isinstance(current, transport_error_types):
            transport_error_found = True

        # Exception messages vary across SDK versions + server
        # implementations, so match on a small allow-list of stable
        # substrings rather than exception type. Kept narrow to avoid
        # false positives on unrelated server errors.
        msg = str(current).lower()
        if msg and any(marker in msg for marker in _SESSION_EXPIRED_MARKERS):
            transport_error_found = True

        stack.extend(getattr(current, "exceptions", ()))
        stack.append(getattr(current, "__cause__", None))
        stack.append(getattr(current, "__context__", None))

    return transport_error_found
