"""W3C trace-context propagation for MCP tool calls (opt-in).

When an agent instrumented with OpenTelemetry calls a tool on an MCP server
that is itself instrumented, the two sides today produce disconnected trace
populations: the agent has a span for the tool call, the server has spans for
serving it, and nothing joins them. This module fixes that by injecting a
W3C ``traceparent`` header on the HTTP transport around each tool-call RPC,
so the server's spans become children of the agent's.

Everything here is off by default and degrades to a no-op. Enable with::

    mcp:
      trace_propagation: true

Two design constraints shape this module (they are the review feedback on
PR #60466, which attempted the same feature at connection-setup time):

* **Capture happens on the agent thread, before the thread boundary.** MCP
  RPCs run on a dedicated daemon event loop; ``trace.get_current_span()``
  over there observes nothing, because contextvars do not cross
  ``run_coroutine_threadsafe``. The caller captures its own context and
  hands the formatted header across.

* **Injection happens per call, not per connection.** MCP tool calls reuse
  one long-lived ``ClientSession``; a header fixed when the transport is
  established would pin every later call on that connection to a single
  stale span. The header is set on the shared httpx client immediately
  before one RPC and removed immediately after, under the per-server RPC
  lock that already serializes calls on a session.

The trace context is read with the standard OpenTelemetry propagation API,
so any instrumentation that sets ambient context works unmodified. Tracing
plugins that keep their own span registry instead of attaching context (for
example hermes-otel, whose ``get_current_traceparent`` exists for exactly
this interop) can register a provider callback::

    from tools.mcp_trace_propagation import register_traceparent_provider
    register_traceparent_provider(get_current_traceparent)

Provider output is validated against the W3C ``traceparent`` grammar before
use; anything else is discarded. No path in this module may raise into a
tool call — a broken telemetry setup must cost traces, never tool calls.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Any, Callable, Optional

# version 00, 16-byte trace-id, 8-byte parent-id, 1-byte flags — lowercase hex
# (https://www.w3.org/TR/trace-context/#traceparent-header-field-values)
_TRACEPARENT_RE = re.compile(r"^00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$")
_ALL_ZERO_TRACE_ID = "0" * 32
_ALL_ZERO_PARENT_ID = "0" * 16

_provider: Optional[Callable[[], Optional[str]]] = None


def register_traceparent_provider(
    provider: Optional[Callable[[], Optional[str]]],
) -> None:
    """Register a callback that returns the current W3C traceparent, or None.

    For tracing integrations that track the active span themselves rather
    than attaching it to ambient OpenTelemetry context. The callback is
    consulted before the standard propagation API. Pass ``None`` to clear.
    """
    global _provider
    _provider = provider


def is_enabled() -> bool:
    """The ``mcp.trace_propagation`` config gate (default: off)."""
    try:
        from hermes_cli.config import load_config_readonly

        section = load_config_readonly().get("mcp") or {}
        return bool(section.get("trace_propagation", False))
    except Exception:
        return False


def _valid(traceparent: Any) -> Optional[str]:
    """Return ``traceparent`` if it is a well-formed W3C header value."""
    if not isinstance(traceparent, str) or not _TRACEPARENT_RE.match(traceparent):
        return None
    _version, trace_id, parent_id, _flags = traceparent.split("-")
    if trace_id == _ALL_ZERO_TRACE_ID or parent_id == _ALL_ZERO_PARENT_ID:
        return None  # all-zero ids are explicitly invalid per the spec
    return traceparent


def current_traceparent() -> Optional[str]:
    """Capture the caller's trace context as a ``traceparent`` value.

    Must be called on the thread that owns the span — i.e. the agent thread
    inside the tool handler, before the call is scheduled onto the MCP loop.
    Returns ``None`` when the feature is disabled, no provider/SDK is
    available, or there is no active span. Never raises.
    """
    if not is_enabled():
        return None

    if _provider is not None:
        try:
            candidate = _valid(_provider())
        except Exception:
            candidate = None
        if candidate is not None:
            return candidate

    try:
        from opentelemetry import propagate

        carrier: dict = {}
        propagate.inject(carrier)
        return _valid(carrier.get("traceparent"))
    except Exception:
        return None


@contextmanager
def injected_headers(client: Any, traceparent: Optional[str]):
    """Set ``traceparent`` on ``client``'s default headers for one RPC.

    ``client`` is the server's shared ``httpx.AsyncClient`` (None for stdio
    transports, where there is nothing to inject into — the block is then a
    no-op). Restores any pre-existing header value on exit so a client is
    never left carrying a stale span, which is the precise failure mode of
    connection-time injection.

    Client default headers, not contextvars, because the HTTP POST is
    written by the transport's own writer task — context attached to the
    calling task never reaches it, but the shared client's headers do. The
    caller must hold the server's RPC lock (call sites already do), which
    serializes header set/restore with the request that uses it.
    """
    if client is None or not traceparent:
        yield
        return

    headers = client.headers
    sentinel = object()
    previous = headers.get("traceparent", sentinel)
    headers["traceparent"] = traceparent
    try:
        yield
    finally:
        try:
            if previous is sentinel:
                headers.pop("traceparent", None)
            else:
                headers["traceparent"] = previous
        except Exception:
            pass
