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
- ``vault.sources`` / ``vault.source.set`` → external password-manager status and enable toggle.
- ``vault.unlock`` / ``vault.lock`` → per-session unlock of a manager from Settings; the master
  password is consumed by the manager CLI on stdin and never stored or logged.

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
    """Metadata-only listing across every enabled backend (local + unlocked password managers).
    Each item carries ``backend``; locked managers contribute nothing (see vault.sources)."""
    try:
        from agent.vault_backends import enabled_backends

        items = []
        for backend in enabled_backends():
            if backend.needs_unlock and not backend.is_unlocked():
                continue
            items.extend({**meta.to_dict(), "backend": backend.name} for meta in backend.list_items())
        return _ok(rid, {"items": items})
    except Exception as e:
        return _err(rid, 5095, str(e))


_EXTERNAL_SOURCES = ("onepassword", "bitwarden")


@method("vault.sources")
def _(rid, params: dict) -> dict:
    """Status of every login source: {name, display_name, enabled, needs_unlock, unlocked, installed}."""
    import shutil

    from agent.vault_backends import enabled_backends
    from agent.vault_backends.bitwarden import BitwardenLoginBackend
    from agent.vault_backends.onepassword import OnePasswordLoginBackend
    from agent.secret_sources.onepassword import find_op

    enabled = {b.name: b for b in enabled_backends()}
    rows = [{"name": "local", "display_name": "Hermes vault", "enabled": True, "needs_unlock": False,
             "unlocked": True, "installed": True}]
    for cls, installed in ((OnePasswordLoginBackend, find_op() is not None),
                           (BitwardenLoginBackend, shutil.which("bw") is not None)):
        live = enabled.get(cls.name)
        rows.append({"name": cls.name, "display_name": cls.display_name, "enabled": live is not None,
                     "needs_unlock": True, "unlocked": bool(live and live.is_unlocked()), "installed": installed})
    return _ok(rid, {"sources": rows})


@method("vault.source.set")
def _(rid, params: dict) -> dict:
    """Enable/disable an external manager: writes ``vault.<name>.enabled`` and locks it when disabling."""
    from agent.vault_backends.unlock import lock
    from hermes_cli.config import load_config, save_config

    name = str(params.get("name") or "")
    if name not in _EXTERNAL_SOURCES:
        return _err(rid, 5095, f"unknown vault source: {name}")
    enabled = bool(params.get("enabled"))
    cfg = load_config()
    section = cfg.setdefault("vault", {}).setdefault(name, {})
    section["enabled"] = enabled
    if not enabled:
        lock(name)
    save_config(cfg)
    return _ok(rid, {"name": name, "enabled": enabled})


@method("vault.unlock")
def _(rid, params: dict) -> dict:
    """Unlock a manager with the master password typed in the Settings dialog (consumed by the CLI on stdin)."""
    from agent.vault_backends import enabled_backends

    name = str(params.get("name") or "")
    password = str(params.get("password") or "")
    backend = next((b for b in enabled_backends() if b.name == name and b.needs_unlock), None)
    if backend is None:
        return _err(rid, 5095, f"{name} is not an enabled password manager")
    if not password:
        return _err(rid, 5095, "master password is required")
    try:
        backend.unlock(password)  # type: ignore[attr-defined]
    except Exception as e:
        return _err(rid, 5095, str(e).replace(password, "[REDACTED]"))
    finally:
        del password
    return _ok(rid, {"name": name, "unlocked": True})


@method("vault.lock")
def _(rid, params: dict) -> dict:
    """Forget a manager's session token (or every one when ``name`` is omitted)."""
    from agent.vault_backends.unlock import lock

    name = params.get("name")
    lock(str(name) if name else None)
    return _ok(rid, {"locked": True})


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
