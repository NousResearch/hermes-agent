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
 Reaches server.py state through ``srv`` (method_ctx.py). and may reference server module globals (``_ok``, ``_err``).
"""

from __future__ import annotations

from .method_ctx import HandlerRegistry
from .contracts.base import Params
from .contracts.profiles_vault_complete_foreign_subagents import (
    VaultAddParams,
    VaultAddResult,
    VaultListResult,
    VaultLockParams,
    VaultLockResult,
    VaultRemoveParams,
    VaultRemoveResult,
    VaultSourceSetParams,
    VaultSourceSetResult,
    VaultSourcesResult,
    VaultUnlockParams,
    VaultUnlockResult,
)

_VAULT_ERROR = 5095  # JSON-RPC error code: vault failure (validation + store errors)

_registry = HandlerRegistry()


def method(name: str):
    """``@method(name)`` with ``params.profile`` bound (home + secret scope) around the handler."""
    def deco(fn):
        def scoped(rid, params):
            try:
                home = srv._profile_home(params.profile)
            except FileNotFoundError as e:
                return srv._err(rid, _VAULT_ERROR, str(e))
            if home is None:
                return fn(rid, params)
            with srv._session_profile_runtime_scope({"profile_home": str(home)}):
                return fn(rid, params)
        return _registry.method(name)(scoped)
    return deco



@method("vault.list")
def _(rid, params: Params) -> VaultListResult | dict:
    """Metadata-only listing across every enabled backend (local + unlocked password managers).
    Each item carries ``backend``; locked managers contribute nothing (see vault.sources)."""
    try:
        from agent.vault_backends import enabled_backends

        items = []
        for backend in enabled_backends():
            if backend.needs_unlock and not backend.is_unlocked():
                continue
            items.extend({**meta.to_dict(), "backend": backend.name} for meta in backend.list_items())
        from tui_gateway.contracts.profiles_vault_complete_foreign_subagents import VaultListResult
        return VaultListResult(items=items)
    except Exception as e:
        return srv._err(rid, _VAULT_ERROR, str(e))


@method("vault.sources")
def _(rid, params: Params) -> VaultSourcesResult | dict:
    """Status of every login source: {name, display_name, enabled, needs_unlock, unlocked, installed}."""
    from agent.vault_backends import enabled_backends
    from agent.vault_backends.base import external_backend_classes, is_installed

    enabled = {b.name: b for b in enabled_backends()}
    rows = [{"name": "local", "display_name": "Hermes vault", "enabled": True, "needs_unlock": False,
             "unlocked": True, "installed": True}]
    for cls in external_backend_classes():
        live = enabled.get(cls.name)
        rows.append({"name": cls.name, "display_name": cls.display_name, "enabled": live is not None,
                     "needs_unlock": True, "unlocked": bool(live and live.is_unlocked()),
                     "installed": is_installed(cls.name)})
    from tui_gateway.contracts.profiles_vault_complete_foreign_subagents import VaultSourcesResult
    return VaultSourcesResult(sources=rows)


@method("vault.source.set")
def _(rid, params: VaultSourceSetParams) -> VaultSourceSetResult | dict:
    """Enable/disable an external manager: writes ``vault.<name>.enabled`` and locks it when disabling."""
    from agent.vault_backends.base import external_backend_classes
    from agent.vault_backends.unlock import lock
    from hermes_cli.config import load_config, save_config

    name = params.name
    if name not in {cls.name for cls in external_backend_classes()}:
        return srv._err(rid, _VAULT_ERROR, f"unknown vault source: {name}")
    enabled = params.enabled
    cfg = load_config()
    section = cfg.setdefault("vault", {}).setdefault(name, {})
    if enabled:
        section.pop("enabled", None)  # detected managers are on by default; this removes the opt-out
    else:
        section["enabled"] = False
    if not enabled:
        lock(name)
    save_config(cfg)
    from tui_gateway.contracts.profiles_vault_complete_foreign_subagents import VaultSourceSetResult
    return VaultSourceSetResult(name=name, enabled=enabled)


@method("vault.unlock")
def _(rid, params: VaultUnlockParams) -> VaultUnlockResult | dict:
    """Unlock a manager with the master password typed in the Settings dialog (consumed by the CLI on stdin)."""
    from agent.vault_backends import enabled_backends

    name = params.name
    password = params.password
    backend = next((b for b in enabled_backends() if b.name == name and b.needs_unlock), None)
    if backend is None:
        return srv._err(rid, _VAULT_ERROR, f"{name} is not an enabled password manager")
    if not password:
        return srv._err(rid, _VAULT_ERROR, "master password is required")
    try:
        backend.unlock(password)  # type: ignore[attr-defined]
    except Exception as e:
        return srv._err(rid, _VAULT_ERROR, str(e).replace(password, "[REDACTED]"))
    finally:
        del password
    from tui_gateway.contracts.profiles_vault_complete_foreign_subagents import VaultUnlockResult
    return VaultUnlockResult(name=name, unlocked=True)


@method("vault.lock")
def _(rid, params: VaultLockParams) -> VaultLockResult | dict:
    """Forget a manager's session token (or every one when ``name`` is omitted)."""
    from agent.vault_backends.unlock import lock

    name = params.name
    lock(name)
    from tui_gateway.contracts.profiles_vault_complete_foreign_subagents import VaultLockResult
    return VaultLockResult(locked=True)


@method("vault.add")
def _(rid, params: VaultAddParams) -> VaultAddResult | dict:
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

    secret = params.secret
    if not isinstance(secret, dict) or not secret:
        return srv._err(rid, _VAULT_ERROR, "secret payload is required")
    try:
        meta = get_vault_store().add_item(
            kind=params.kind.value,
            label=params.label,
            origin=params.origin,
            secret=secret,
        )
        from tui_gateway.contracts.profiles_vault_complete_foreign_subagents import VaultAddResult
        return VaultAddResult(id=meta.id)
    except VaultError as e:
        # VaultError messages are metadata-safe by contract, but scrub anyway.
        return srv._err(rid, _VAULT_ERROR, scrub_secret_from_text(str(e), secret))
    except Exception as e:
        return srv._err(rid, _VAULT_ERROR, scrub_secret_from_text(str(e), secret))


@method("vault.remove")
def _(rid, params: VaultRemoveParams) -> VaultRemoveResult | dict:
    """Remove a vault item by id. Result: ``{removed: bool}``."""
    try:
        from agent.vault_store import get_vault_store

        item_id = params.id
        if not item_id:
            return srv._err(rid, _VAULT_ERROR, "id is required")
        from tui_gateway.contracts.profiles_vault_complete_foreign_subagents import VaultRemoveResult
        return VaultRemoveResult(removed=get_vault_store().remove_item(item_id))
    except Exception as e:
        return srv._err(rid, _VAULT_ERROR, str(e))


def register(server) -> None:
    """Bind this module's handlers onto ``server``'s globals and registry."""
    _registry.install(server, globals())

# Bound last, after every definition, so importing this module first (tests, the gateway process)
# lets server.py's own tail import see a complete module — the same tail-import idiom server.py uses.
from tui_gateway import server as srv  # noqa: E402
