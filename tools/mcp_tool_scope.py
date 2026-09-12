"""Connection-ledger keys for tools.mcp_tool under a profile multiplexer.

Every ledger in ``tools.mcp_tool`` (``_servers``, connecting/error/cooldown maps, circuit
breaker, lazy configs, trust metadata) is keyed by a *connection key*: the bare server name
outside a multiplexer (single-profile processes are unchanged, byte for byte), and
``(owner_scope, name)`` under one. Two profiles that both configure ``github`` with their own
token are two connections; keying by name alone let the first profile's connection shadow the
second's forever — its ``register_mcp_servers`` saw the name as "already connected", adopted
nothing (different credentials) and left the profile silently tool-less (#106005, #91654).

A profile may still *adopt* another profile's live connection when the route and credentials
match (``mcp_tool_registration._same_server_route``); ``_server_tool_scopes[key]`` records every
scope that has done so, and ``_resolve_server_key`` finds that shared connection for a caller
whose own scope has none.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

from tools.mcp_tool_common import _core

ServerKey = Union[str, Tuple[str, str]]


def _server_key(name: str, scope: Optional[str] = None, *, current: bool = True) -> ServerKey:
    """Connection key for *name* owned by *scope* (the current registry scope when *current*).
    ``None`` scope (no multiplexer) keeps the bare name."""
    if scope is None and current:
        scope = _core._mcp_registry_scope()
    return name if scope is None else (scope, name)


def _key_name(key: ServerKey) -> str:
    return key[1] if isinstance(key, tuple) else key


def _key_scope(key: ServerKey) -> Optional[str]:
    """Owning registry scope encoded in *key* (None for a bare, unscoped key)."""
    return key[0] if isinstance(key, tuple) else None


def _key_visible_in_scope(key: ServerKey, scope: Optional[str]) -> bool:
    """Whether the connection under *key* serves *scope*: owned by it or adopted into it.
    Caller holds ``_core._lock`` or tolerates a racy read (status surfaces)."""
    if scope is None:
        return True
    return _key_scope(key) == scope or scope in _core._server_tool_scopes.get(key, ())


def _resolve_server_key(name: str, scope: Optional[str] = None, *, current: bool = True) -> ServerKey:
    """The connection key a call to *name* from *scope* must use: the scope's own connection
    (live, connecting or lazily registered) when it has one, else a shared connection it
    adopted, else its own (not yet existing) key so bookkeeping lands under this scope."""
    if scope is None and current:
        scope = _core._mcp_registry_scope()
    own = _server_key(name, scope, current=False)
    if scope is None or own in _core._servers or own in _core._lazy_server_configs:
        return own
    for key, scopes in _core._server_tool_scopes.items():
        if scope in scopes and _key_name(key) == name and key in _core._servers:
            return key
    return own
