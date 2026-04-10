"""Write-only secrets management tool for the credential proxy.

Provides store/rotate/delete/list operations.  There is deliberately
**no read/get action** — the agent must never be able to retrieve a
real credential value.  This is the core security invariant of the
credential proxy architecture (#4656).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from tools.registry import registry

logger = logging.getLogger(__name__)

# Lazily initialised — the proxy daemon sets this at startup.
_store = None


def set_store(store: Any) -> None:
    """Set the backing credential store (called by proxy startup)."""
    global _store
    _store = store


def _check_available() -> bool:
    """Return True if the credential proxy store is available."""
    return _store is not None


async def handle_secrets(params: dict) -> Dict[str, Any]:
    """Handle a secrets tool call."""
    if _store is None:
        return {"error": "Credential proxy is not running. Start it with: hermes cred-proxy start"}

    action = params.get("action", "")
    name = params.get("name", "").strip()
    value = params.get("value", "")

    if action == "store":
        if not name:
            return {"error": "Missing required parameter: name"}
        if not value:
            return {"error": "Missing required parameter: value"}
        _store.store(name, value)
        return {"stored": True}

    elif action == "rotate":
        if not name:
            return {"error": "Missing required parameter: name"}
        if not value:
            return {"error": "Missing required parameter: value"}
        _store.rotate(name, value)
        return {"rotated": True}

    elif action == "delete":
        if not name:
            return {"error": "Missing required parameter: name"}
        existed = _store.delete(name)
        return {"deleted": existed}

    elif action == "list":
        return {"names": _store.list_names()}

    else:
        return {"error": f"Unknown action: {action!r}. Valid: store, rotate, delete, list"}


# Register with the tool registry
registry.register(
    name="secrets",
    toolset="secrets",
    schema={
        "name": "secrets",
        "description": (
            "Manage credentials for the credential proxy. Values are stored "
            "securely and are never readable by the agent. The proxy substitutes "
            "real credentials at the HTTP transport layer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["store", "rotate", "delete", "list"],
                    "description": "Operation to perform.",
                },
                "name": {
                    "type": "string",
                    "description": "Credential name (e.g. 'cf_dns_token').",
                },
                "value": {
                    "type": "string",
                    "description": "Credential value (required for store/rotate).",
                },
            },
            "required": ["action"],
        },
    },
    handler=handle_secrets,
    check_fn=_check_available,
    is_async=True,
    description="Store credentials securely (write-only, never readable by agent).",
    emoji="🔐",
)
