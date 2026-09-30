"""Connector integration boundary for managed gateway accounts and local MCP servers.

Only the names below are cross-package surface; imports beyond it need a design decision.
Siblings: ``contract`` (states, actors, transition table), ``operation`` (the record),
``live`` (open operation per session), ``run`` (the lifecycle loop), ``managed`` / ``mcp``
(per-kind hooks), ``targets``, ``search``, ``dispatch``, ``gateway/`` (HTTP wire + client).

The surface is exported lazily (PEP 562) on purpose: this package is imported concurrently
from threads that also import its submodules directly (the tool registry's discovery scan
imports ``tools.connectors.tool`` while the gateway's Group Chat worker imports this
package). Eager ``from tools.connectors.<sub> import …`` lines made this module hold its own
module lock while reaching for a submodule another thread already held, which Python 3.14
reports as ``_frozen_importlib._DeadlockError: deadlock detected by
_ModuleLock('tools.connectors.tool')`` and which aborts the Group Chat worker at startup. A
package ``__init__`` that imports no submodule cannot hold that lock while waiting for one,
so the cycle is gone. Attribute access, ``from tools.connectors import x`` and ``__all__``
behave exactly as before.
"""

import importlib

# name -> module owning it: the single source of truth for __all__ and __getattr__.
_LAZY_EXPORTS: "dict[str, str]" = {
    "CONNECTOR_BATCH_SENTINEL": "tools.connectors.gateway.names",
    "MANAGE_CONNECTIONS_SCHEMA": "tools.connectors.tool",
    "connector_describe": "tools.connectors.gateway.bridge",
    "connector_search_hits": "tools.connectors.gateway.bridge",
    "connectors_available": "tools.connectors.gateway.config",
    "dispatch_connector_batch": "tools.connectors.dispatch",
    "dispatch_connector_call": "tools.connectors.dispatch",
    "is_connector_name": "tools.connectors.gateway.names",
    "manage_connections": "tools.connectors.tool",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str):
    """Resolve the public surface on first use (PEP 562).

    Deliberately no eager import: see the module docstring for the import-deadlock this
    prevents. Module-level ``__getattr__`` is only called for names not already bound, so
    each submodule is imported at most once per process.
    """
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> "list[str]":
    return sorted(set(globals()) | set(__all__))
