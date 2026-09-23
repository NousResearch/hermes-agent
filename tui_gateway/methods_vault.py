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
  password is consumed by the manager CLI through its non-interactive channel and never stored
  or logged.

Every handler honours ``params.profile`` (app-global remote mode serves several profiles from one
backend): the requested profile's HERMES_HOME and secret scope are bound around the body, so the
vault file, manager config and manager tokens all resolve to that profile.

Handlers are rebound onto server.py's globals at install time (see
method_ctx.py) and may reference server module globals (``_ok``, ``_err``).
"""

from .method_ctx import HandlerRegistry

_registry = HandlerRegistry()
method = _registry.method
# server.py's ``@_profile_scoped`` (applied at install): the one scoping path every RPC uses, so the
# launch profile (no ``params.profile``) keeps its secret scope bound once the process multiplexes
# instead of its manager-token reads raising ``UnscopedSecretError``.
_profile_scoped = _registry.profile_scoped

# JSON-RPC error code 5095 = vault failure (validation + store errors).
# Kept as a literal inside handler bodies: handlers are rebound onto
# server.py's globals, so module-level constants are not reachable there.


@method("vault.list")
@_profile_scoped
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


@method("vault.sources")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Status of every login source: {name, display_name, enabled, needs_unlock, unlocked,
    installed, host, status, reason}.

    ``status`` keeps not_installed / disconnected / auth_required / available distinct and
    ``host`` names the machine the manager lives on, so the Desktop can say what is wrong and
    where to fix it instead of collapsing every state into "Not detected". Re-calling this
    method IS the reconnect: each call re-probes exactly once per source.

    Metadata only, like ``vault.list`` — no secret value, vault item or token is read here.
    One backend that raises is reported as ``disconnected`` instead of failing the whole RPC.
    """
    from agent.vault_backends.base import (
        SourceStatus, external_backend_classes, is_opted_out, owning_host, probe,
    )

    host = owning_host()
    rows = [{"name": "local", "display_name": "Hermes vault", "enabled": True, "needs_unlock": False,
             "unlocked": True, "installed": True, "host": host,
             "status": SourceStatus.available.value, "reason": ""}]
    for cls in external_backend_classes():
        try:
            opted_out, result = is_opted_out(cls.name), probe(cls.name)
        except Exception as e:
            rows.append({"name": cls.name, "display_name": cls.display_name, "enabled": False,
                         "needs_unlock": True, "unlocked": False, "installed": False, "host": host,
                         "status": SourceStatus.disconnected.value,
                         "reason": f"{cls.display_name} could not be probed on {host}: {str(e)[:200]}"})
            continue
        # Same zero-config contract as ``is_enabled``: a detected manager is a login source
        # unless the user opted out. Derived from this one probe so enabled/installed/status
        # can never disagree within a response.
        rows.append({"name": cls.name, "display_name": cls.display_name,
                     "enabled": result.installed and not opted_out, "needs_unlock": True,
                     "unlocked": result.status is SourceStatus.available, **result.to_dict()})
    return _ok(rid, {"sources": rows})


@method("vault.source.set")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Enable/disable an external manager: writes ``vault.<name>.enabled`` and locks it when disabling."""
    from agent.vault_backends.base import external_backend_classes
    from agent.vault_backends.unlock import lock
    from hermes_cli.config import _ensure_dict, load_config, save_config

    name = str(params.get("name") or "")
    if name not in {cls.name for cls in external_backend_classes()}:
        return _err(rid, 5095, f"unknown vault source: {name}")
    enabled = bool(params.get("enabled"))
    cfg = load_config()
    section = _ensure_dict(_ensure_dict(cfg, "vault"), name)
    if enabled:
        section.pop("enabled", None)  # detected managers are on by default; this removes the opt-out
    else:
        section["enabled"] = False
    if not enabled:
        lock(name)
    save_config(cfg)
    return _ok(rid, {"name": name, "enabled": enabled})


@method("vault.unlock")
@_profile_scoped
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
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Forget a manager's session token (or every one when ``name`` is omitted)."""
    from agent.vault_backends.unlock import lock

    name = params.get("name")
    lock(str(name) if name else None)
    return _ok(rid, {"locked": True})


@method("vault.add")
@_profile_scoped
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
@_profile_scoped
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
