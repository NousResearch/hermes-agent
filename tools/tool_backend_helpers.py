"""Shared helpers for tool backend selection."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from utils import is_truthy_value

logger = logging.getLogger(__name__)
_DEFAULT_BROWSER_PROVIDER = "local"
_DEFAULT_MODAL_MODE = "auto"
_VALID_MODAL_MODES = {"auto", "direct", "managed"}


def managed_nous_tools_enabled(*, force_fresh: bool = False) -> bool:
    """Coarse gate: entitled to the Nous Tool Gateway (paid Portal access OR a live free
    pool). Fails closed on unknown/error — never blocks startup. Callers narrow per category
    via ``tool_gateway_entitled_for``; ``force_fresh`` is for flows needing a just-bought grant."""
    try:
        from hermes_cli.nous_account import get_nous_portal_account_info
        account_info = (get_nous_portal_account_info(force_fresh=True) if force_fresh
                        else get_nous_portal_account_info())
        return bool(account_info.logged_in) and account_info.tool_gateway_entitled
    except Exception:
        return False


def nous_tool_gateway_unavailable_message(capability: str = "the Nous Tool Gateway", *,
                                          force_fresh: bool = False) -> str:
    """Return account-aware guidance for an unavailable Nous Tool Gateway path."""
    try:
        from hermes_cli.nous_account import (
            format_nous_portal_entitlement_message, get_nous_portal_account_info)
        message = format_nous_portal_entitlement_message(
            get_nous_portal_account_info(force_fresh=force_fresh), capability=capability)
        if message:
            return message
    except Exception:
        pass
    return (f"{capability} is unavailable. Run `hermes model` to refresh your "
            "Nous Portal login and billing status.")


def normalize_browser_cloud_provider(value: object | None) -> str:
    """Return a normalized browser provider key."""
    provider = str(value or _DEFAULT_BROWSER_PROVIDER).strip().lower()
    return provider or _DEFAULT_BROWSER_PROVIDER


def coerce_modal_mode(value: object | None) -> str:
    """Return the requested modal mode when valid, else the default."""
    mode = str(value or _DEFAULT_MODAL_MODE).strip().lower()
    return mode if mode in _VALID_MODAL_MODES else _DEFAULT_MODAL_MODE


normalize_modal_mode = coerce_modal_mode


def has_direct_modal_credentials() -> bool:
    """Return True when direct Modal credentials/config are available."""
    if os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET"):
        return True
    try:
        return (Path.home() / ".modal.toml").exists()
    except OSError:  # includes PermissionError on Path.home()
        return False


def resolve_modal_backend_state(modal_mode: object | None, *, has_direct: bool,
                                managed_ready: bool,
                                managed_enabled: bool | None = None) -> Dict[str, Any]:
    """Resolve direct vs managed Modal backend: ``direct``/``managed`` are exclusive; ``auto``
    prefers managed when available, else direct."""
    requested_mode = coerce_modal_mode(modal_mode)
    if managed_enabled is None:
        managed_enabled = managed_nous_tools_enabled()
    managed_ok = managed_enabled and managed_ready
    exclusive = {"managed": "managed" if managed_ok else None,
                 "direct": "direct" if has_direct else None}
    selected_backend = exclusive.get(
        requested_mode, "managed" if managed_ok else "direct" if has_direct else None)
    return {"requested_mode": requested_mode, "mode": requested_mode, "has_direct": has_direct,
            "managed_ready": managed_ready,
            "managed_mode_blocked": requested_mode == "managed" and not managed_enabled,
            "selected_backend": selected_backend}


def _scoped_credential(name: str) -> str:
    """Read a credential env var under the active profile secret scope; raw env fallback only
    if ``agent.secret_scope`` cannot import (a packaging edge must never lose the key)."""
    try:
        from agent.secret_scope import get_secret
        return (get_secret(name, "") or "").strip()
    except Exception:  # pragma: no cover — secret_scope is in-repo
        return (os.getenv(name, "") or "").strip()


def _dotenv_value(env_var: str) -> str:
    """``.env`` value via ``hermes_cli.config.get_env_value`` (``""`` when unavailable)."""
    try:
        from hermes_cli.config import get_env_value
        return str(get_env_value(env_var) or "").strip()
    except Exception:  # pragma: no cover — config is in-repo
        return ""


def resolve_provider_secret(env_var: str, provider_id: str, config_value: str = "",
                            env_getter=None) -> str:
    """Resolve a voice-provider API key (single owner for STT/TTS lookup). Order: explicit
    ``config_value`` -> profile secret scope / env -> ``.env`` via ``env_getter`` (or
    ``hermes_cli.config.get_env_value``) -> credential pool for ``provider_id``. Under an
    active multiplex turn the profile scope is authoritative: a miss returns ``""`` rather
    than borrowing another profile's env or pool. Never raises.

    Resolution order (fixes #68003 — keys added via ``hermes auth add <provider>`` were invisible to the
    voice tools, which only consulted env/.env):
    """
    key = str(config_value or "").strip() or _scoped_credential(env_var)
    if key:
        return key
    try:
        from agent.secret_scope import is_multiplex_active
        if is_multiplex_active():
            return ""
    except Exception:  # pragma: no cover — secret_scope is in-repo
        pass
    key = str(env_getter(env_var) or "").strip() if env_getter else _dotenv_value(env_var)
    if key or not provider_id:
        return key
    try:
        from agent.credential_pool import load_pool
        # config.yaml ``providers.<name>`` entries are pooled under ``custom:<name>``.
        for pool_key in (provider_id, f"custom:{provider_id}"):
            pool = load_pool(pool_key)
            entry = pool.peek() if pool is not None and pool.has_credentials() else None
            key = str(getattr(entry, "runtime_api_key", "") or getattr(entry, "access_token", "")
                      or "").strip()
            if key:
                return key
    except Exception as exc:
        logger.debug("Could not read %s credential pool for %s: %s", provider_id, env_var, exc)
    return ""


def resolve_openai_audio_api_key() -> str:
    """Prefer VOICE_TOOLS_OPENAI_KEY, else OPENAI_API_KEY (scope-aware, pool fallback for the
    latter). Must go through the secret scope: a raw ``os.environ`` read could bill another
    profile's account under multiplex.

    Outside a multiplexed turn, ``OPENAI_API_KEY`` additionally falls back to the credential pool (``hermes
    auth add openai-api``) via ``resolve_provider_secret`` — same #68003 fix as the other voice providers.
    The dedicated voice-tools override remains env/scope-only.
    """
    return (resolve_provider_secret("VOICE_TOOLS_OPENAI_KEY", "")
            or resolve_provider_secret("OPENAI_API_KEY", "openai-api"))


def prefers_gateway(config_section: str) -> bool:
    """True when ``<section>.use_gateway`` is set in config.yaml. Never raises."""
    try:
        from hermes_cli.config import load_config
        section = (load_config() or {}).get(config_section)
        return isinstance(section, dict) and is_truthy_value(section.get("use_gateway"))
    except Exception:
        return False


# Provider value the managed "Nous Subscription" picker rows write for every category;
# any other name = that vendor direct; no key = legacy autodetect.
NOUS_MANAGED_PROVIDER = "nous"
# Per-capability keys that also count as "this category has been configured".
_EXTRA_SELECTION_KEYS = {"web": ("search_backend", "extract_backend")}
# Key(s) carrying the category's provider selection. ``browser.backend`` is the DRIVER
# choice (browser-use CLI vs built-in), not the cloud provider — excluded.
_SELECTION_NAME_KEYS = {"browser": ("cloud_provider",), "web": ("backend",)}
_DEFAULT_NAME_KEYS = ("provider", "backend", "cloud_provider")


def _raw_section(section: str) -> Dict[str, Any] | None:
    """The RAW (unmerged) config.yaml mapping for ``section``, or None."""
    try:
        from hermes_cli.config import read_raw_config_readonly
        cfg = read_raw_config_readonly() or {}
        raw = cfg.get(section) if isinstance(cfg, dict) else None
        return raw if isinstance(raw, dict) else None
    except Exception:
        return None


def read_selection(section: str) -> str | None:
    """THE single runtime read of the persisted `hermes tools` selection: ``"nous"`` (managed
    gateway row), a vendor name (direct, own credentials), or ``None`` (never configured ->
    legacy autodetect allowed). Reads the RAW config.yaml so key presence means "actually
    written", not "schema default"; a raw ``local`` is therefore a real user selection.
    Legacy shim: ``use_gateway: true`` was only ever written by the managed row, so it maps
    to ``"nous"`` regardless of the name key. Never raises."""
    raw = _raw_section(section)
    if raw is None:
        return None
    if is_truthy_value(raw.get("use_gateway")):
        return NOUS_MANAGED_PROVIDER
    for key in _SELECTION_NAME_KEYS.get(section, _DEFAULT_NAME_KEYS):
        text = str(raw.get(key)).strip().lower() if raw.get(key) is not None else ""
        if text:
            return text
    # use_gateway: false with no name key is not a usable selection shape;
    # per-capability web keys still count as configured via selection_exists().
    return None


def selection_exists(section: str) -> bool:
    """True when ANY selection signal was ever written for the section (wider than
    read_selection: per-capability web keys count too)."""
    if read_selection(section) is not None:
        return True
    extra = _EXTRA_SELECTION_KEYS.get(section, ())
    raw = _raw_section(section) if extra else None
    return raw is not None and any(str(raw.get(key) or "").strip() for key in extra)


# Backends that once shipped in-tree but were removed; a config still pointing at one would
# otherwise fail silently at the FIRST tool call with a generic "no registered provider has that
# name". Used by the startup config check and selection_error(); add removals here, never as
# one-off string checks:  "web": {"<name>": "the <Name> backend was removed in vX (...)"}
REMOVED_BACKENDS: Dict[str, Dict[str, str]] = {}


# Backends that once shipped in-tree but were removed. A config that still points at one otherwise fails
# silently at the FIRST tool call with a generic "no registered provider has that name" — no migration, no
# startup notice (reported after the Tavily removal in #99199). Both the startup config check
# (hermes_cli.config.validate_config_structure) and selection_error() consult this map so the user learns
# what actually happened and what to do. Declared data, one policy — add future removals here, never as
# one-off string checks at call sites.
# Currently empty: the Tavily removal (#99199) that introduced this registry was reverted by the #99731
# restore. Future backend removals add an entry here, e.g. "web": {"<name>": "the <Name> backend was removed
# in vX.Y.Z (...)"},
def removed_backend_note(section: str, name: str) -> Optional[str]:
    """Explanation for a backend that used to ship in-tree, or None. ``name`` tolerates the
    quoted form callers pass to selection_error()."""
    return REMOVED_BACKENDS.get(section, {}).get((name or "").strip().strip("'\"").lower())


def selection_error(section: str, selection_name: str, failure: str) -> str:
    """The uniform honest-error contract for a selected-but-broken provider."""
    failure = removed_backend_note(section, selection_name) or failure
    return (f"{section} is configured to use {selection_name} (set via hermes "
            f"tools), but {failure}. Run 'hermes tools' to change it.")


def fal_key_is_configured() -> bool:
    """True when FAL_KEY is set (scope/env, else ``.env`` for CLI paths that run before dotenv
    loads) to a non-whitespace value, so tool-side and CLI setup-time checks agree."""
    return bool(_scoped_credential("FAL_KEY") or _dotenv_value("FAL_KEY"))
