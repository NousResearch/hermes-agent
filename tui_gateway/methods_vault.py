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
    """Status of every login source: {name, display_name, enabled, needs_unlock, unlocked, installed}."""
    from agent.vault_backends import enabled_backends
    from agent.vault_backends.base import external_backend_classes, is_installed

    enabled = {b.name: b for b in enabled_backends()}
    rows = [{"name": "local", "display_name": "Hermes vault", "enabled": True, "needs_unlock": False,
             "unlocked": True, "installed": True}]
    for cls in external_backend_classes():
        live = enabled.get(cls.name)
        row = {"name": cls.name, "display_name": cls.display_name, "enabled": live is not None,
               "needs_unlock": True, "unlocked": bool(live and live.is_unlocked()),
               "installed": is_installed(cls.name)}
        capability_reader = getattr(live, "auth_capabilities", None)
        if callable(capability_reader):
            capabilities = capability_reader()
            if capabilities is not None:
                row["auth_capabilities"] = capabilities
        rows.append(row)
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
    """Unlock a manager with a Settings password or an explicitly supported native app flow."""
    from agent.vault_backends import enabled_backends

    name = str(params.get("name") or "")
    password = str(params.get("password") or "")
    requested_method = params.get("method")
    backend = next((b for b in enabled_backends() if b.name == name and b.needs_unlock), None)
    if backend is None:
        return _err(rid, 5095, f"{name} is not an enabled password manager")

    capability_reader = getattr(backend, "auth_capabilities", None)
    capabilities = capability_reader() if callable(capability_reader) else None
    if requested_method is not None:
        if requested_method not in {"app", "password"}:
            return _err(rid, 5095, "unlock method is not available")
        if capabilities is None:
            return _err(rid, 5095, "unlock method is not available on this backend")
        methods = capabilities.get("methods", [])
        allowed = requested_method in methods
        if requested_method == "app":
            allowed = allowed and bool(capabilities.get("native_app_eligible"))
            if password:
                return _err(rid, 5095, "clear the password to use native app approval")
        if not allowed:
            return _err(rid, 5095, "unlock method is not available for this profile")
    elif capabilities is not None and not password:
        if "app" not in capabilities.get("methods", []) or not capabilities.get("native_app_eligible"):
            return _err(rid, 5095, "master password is required for this profile")
    elif not password and not getattr(backend, "supports_app_unlock", False):
        return _err(rid, 5095, "master password is required")
    try:
        if requested_method is not None:
            import inspect

            unlock_parameters = inspect.signature(backend.unlock).parameters
            if "method" not in unlock_parameters and not any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in unlock_parameters.values()
            ):
                return _err(rid, 5095, "selected unlock method is not supported by this backend")
            backend.unlock(password, method=requested_method)  # type: ignore[attr-defined]
        else:
            backend.unlock(password)  # type: ignore[attr-defined]
    except Exception as e:
        message = str(e).replace(password, "[REDACTED]") if password else str(e)
        return _err(rid, 5095, message)
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
