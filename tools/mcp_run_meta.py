"""Per-run MCP request ``_meta`` passthrough for API server runs.

``POST /v1/runs`` may include an opaque ``mcp_meta`` object. The API server
binds it on a ContextVar for the duration of that run's executor work; the
MCP tool handler reads it on the agent thread and forwards it as
``ClientSession.call_tool(..., meta=...)`` so concurrent runs stay isolated.
"""

from __future__ import annotations

import contextvars
from typing import Any, Optional

_mcp_run_meta: contextvars.ContextVar[Optional[dict[str, Any]]] = contextvars.ContextVar(
    "mcp_run_meta",
    default=None,
)


def get_mcp_run_meta() -> Optional[dict[str, Any]]:
    """Return the run-scoped MCP ``_meta`` object, or ``None`` if unset."""
    return _mcp_run_meta.get()


def set_mcp_run_meta(meta: Optional[dict[str, Any]]) -> contextvars.Token:
    """Bind per-run MCP ``_meta`` for the current context (thread/task)."""
    return _mcp_run_meta.set(meta)


def reset_mcp_run_meta(token: contextvars.Token) -> None:
    """Restore the prior value from :func:`set_mcp_run_meta`."""
    _mcp_run_meta.reset(token)
