"""Credential-vault JSON-RPC handlers — the Desktop's door to the local vault.

The Desktop's Settings → Credential Vault panel manages the encrypted,
model-blind vault (``agent/vault_store.py``) over the same localhost WS
JSON-RPC channel every other Settings surface uses. Contracts:

- ``vault.list``   → metadata only ({id, kind, label, origin, created_at,
  and for logins identifier/identifier_type — identifiers are visible
  metadata by design}); passwords NEVER appear in any response.
- ``vault.add``    → validates via ``VaultStore.add_item``; the secret
  payload arrives over the local RPC channel, goes straight into the
  encrypted store, and is never logged. Error strings are defensively
  scrubbed with ``scrub_secret_from_text`` before they leave the handler.
- ``vault.remove`` → {removed: bool}.

Handlers are rebound onto server.py's globals at install time (see
method_ctx.py) and may reference server module globals (``_ok``, ``_err``).
"""

from .method_ctx import HandlerRegistry

_registry = HandlerRegistry()
method = _registry.method

# JSON-RPC error code 5095 = vault failure (validation + store errors).
# Kept as a literal inside handler bodies: handlers are rebound onto
# server.py's globals, so module-level constants are not reachable there.


@method("vault.list")
def _(rid, params: dict) -> dict:
    """Metadata-only listing of vault items. Secret values are never included."""
    try:
        from agent.vault_store import get_vault_store

        items = [meta.to_dict() for meta in get_vault_store().list_items()]
        return _ok(rid, {"items": items})
    except Exception as e:
        return _err(rid, 5095, str(e))


@method("vault.add")
def _(rid, params: dict) -> dict:
    """Add a vault item. ``secret`` values go straight into the encrypted store.

    Params: ``kind`` (login|payment|address), ``label``, ``origin?``,
    ``secret`` (dict). Result: ``{id}`` — metadata only. Exception text is
    scrubbed of secret values before it can reach a response or a log line.
    """
    from agent.vault_store import (
        VaultError,
        get_vault_store,
        scrub_secret_from_text,
    )

    secret = params.get("secret")
    if not isinstance(secret, dict) or not secret:
        return _err(rid, 5095, "secret payload is required")
    try:
        meta = get_vault_store().add_item(
            kind=str(params.get("kind") or ""),
            label=str(params.get("label") or ""),
            origin=(str(params.get("origin")) if params.get("origin") else None),
            secret=secret,
        )
        return _ok(rid, {"id": meta.id})
    except VaultError as e:
        # VaultError messages are metadata-safe by contract, but scrub anyway.
        return _err(rid, 5095, scrub_secret_from_text(str(e), secret))
    except Exception as e:
        return _err(rid, 5095, scrub_secret_from_text(str(e), secret))


@method("vault.remove")
def _(rid, params: dict) -> dict:
    """Remove a vault item by id. Result: ``{removed: bool}``."""
    try:
        from agent.vault_store import get_vault_store

        item_id = str(params.get("id") or "")
        if not item_id:
            return _err(rid, 5095, "id is required")
        return _ok(rid, {"removed": get_vault_store().remove_item(item_id)})
    except Exception as e:
        return _err(rid, 5095, str(e))


def register(server) -> None:
    """Bind this module's handlers onto ``server``'s globals and registry."""
    _registry.install(server)
