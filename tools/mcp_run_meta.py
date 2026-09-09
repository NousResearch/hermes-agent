"""Per-run MCP request ``_meta`` passthrough for API server runs.

``POST /v1/runs`` may include an opaque, per-server ``mcp_meta`` map. The API server
binds it on a ContextVar for the duration of that run's executor work; the
MCP tool handler reads it on the agent thread and forwards it as
``ClientSession.call_tool(..., meta=...)`` so concurrent runs stay isolated.
"""

from __future__ import annotations

import contextvars
import json
from dataclasses import dataclass, field
from typing import Any, Optional

from tools.mcp_tool_content import _is_reserved_mcp_meta_key

# Serialized values keep copies of a context from sharing mutable credentials.
RunMeta = tuple[tuple[str, str], ...]
MAX_MCP_META_BYTES = 16_384
MAX_MCP_META_DEPTH = 8

@dataclass(frozen=True)
class _RunMetaBinding:
    values: RunMeta = field(repr=False)
    scope: Optional[str]


_mcp_run_meta: contextvars.ContextVar[Optional[_RunMetaBinding]] = contextvars.ContextVar(
    "mcp_run_meta",
    default=None,
)


def parse_mcp_run_meta(value: Any) -> Optional[RunMeta]:
    """Validate the API's server-name -> metadata map without echoing its values."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("'mcp_meta' must be an object keyed by MCP server name")
    # Bound nesting before serializing; JSON parsers can accept deeper structures.
    pending = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        if depth > MAX_MCP_META_DEPTH:
            raise ValueError("'mcp_meta' exceeds the maximum nesting depth")
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("'mcp_meta' keys must be strings")
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        size = len(encoded.encode("utf-8"))
    except (TypeError, ValueError, UnicodeError):
        raise ValueError("'mcp_meta' must contain valid JSON values") from None
    if size > MAX_MCP_META_BYTES:
        raise ValueError("'mcp_meta' exceeds the 16384-byte limit")
    entries = []
    for name, meta in value.items():
        if not name or len(name) > 128 or any(ord(ch) < 32 or ord(ch) == 127 for ch in name):
            raise ValueError("'mcp_meta' contains an invalid MCP server name")
        if not isinstance(meta, dict):
            raise ValueError("Each 'mcp_meta' server entry must be an object")
        if any(key == "progressToken" or _is_reserved_mcp_meta_key(key) for key in meta):
            raise ValueError("'mcp_meta' cannot override protocol-owned metadata")
        entries.append((name, json.dumps(meta, ensure_ascii=False, separators=(",", ":"))))
    return tuple(entries)


def get_mcp_run_meta(server_name: str) -> Optional[dict[str, Any]]:
    """A fresh copy for this exact server; untargeted servers receive no metadata."""
    binding = _mcp_run_meta.get()
    for name, encoded in binding.values if binding is not None else ():
        if name == server_name:
            return json.loads(encoded)
    return None


def _active_profile_scope() -> Optional[str]:
    """The canonical request scope; missing multiplex identity must not alias default."""
    from agent.secret_scope import is_multiplex_active
    from hermes_constants import get_hermes_home_override, hermes_home_key

    if not is_multiplex_active():
        return None
    override = get_hermes_home_override()
    if not override:
        raise ValueError("MCP metadata requires an explicit profile scope under multiplex")
    return hermes_home_key(override)


def mcp_run_meta_scope() -> Optional[str]:
    """Use the run's bound profile, not a different scope entered by delegated work."""
    binding = _mcp_run_meta.get()
    if binding is None or binding.scope != _active_profile_scope():
        raise ValueError("MCP metadata cannot cross the originating run's profile scope")
    return binding.scope


def require_mcp_meta_destination(server_name: str, scope: Optional[str], server: Any = None) -> None:
    """Fail closed on the bare-name registry until profile-qualified connections land (#99594).

    Check both the recorded owner and the acquired instance. Rechecking after the RPC lock
    and before recovery prevents a queued call/retry from following a replacement connection.
    """
    if scope is None:  # Single-profile behavior is unchanged.
        return
    from tools import mcp_tool

    with mcp_tool._lock:
        if (mcp_tool._server_scope_keys.get(server_name) != scope
                or (server is not None and mcp_tool._servers.get(server_name) is not server)):
            raise ValueError("MCP metadata destination is not owned by the requesting profile")


def set_mcp_run_meta(meta: Optional[RunMeta]) -> contextvars.Token:
    """Bind metadata AND its profile in the run's executor, before any agent work."""
    binding = _RunMetaBinding(meta, _active_profile_scope()) if meta is not None else None
    return _mcp_run_meta.set(binding)


def reset_mcp_run_meta(token: contextvars.Token) -> None:
    """Restore the prior value from :func:`set_mcp_run_meta`."""
    _mcp_run_meta.reset(token)
