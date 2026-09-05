"""Shared helpers for direct xAI HTTP integrations."""

from __future__ import annotations

import datetime
import json
import os
import uuid
from typing import Any, Dict, Optional


DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"
MAX_XAI_STORAGE_EXPIRES_AFTER_SECONDS = 30 * 24 * 60 * 60
SAFE_XAI_STORAGE_EXPIRES_AFTER_SECONDS = 2 * 24 * 60 * 60


def _dict_get(obj: Any, key: str) -> Any:
    return obj.get(key) if isinstance(obj, dict) else None


def has_xai_credentials() -> bool:
    """Cheap probe: True when xAI credentials are *likely* usable.

    Deliberately avoids :func:`resolve_xai_http_credentials` (disk locks, OAuth network
    refresh) — hot-paint callers and ``WebSearchProvider.is_available()`` must not do network
    I/O. Checks, fast-to-slow: ``XAI_API_KEY``; ``providers.xai-oauth.tokens.access_token`` in
    ``auth.json``; any ``credential_pool.xai-oauth`` entry with an ``access_token`` (pool-only
    multi-account grants never write the providers singleton). Returns False on any exception
    so a corrupted auth store can't block other availability scans.
    """
    from agent.secret_scope import get_secret

    if (get_secret("XAI_API_KEY", "") or "").strip():
        return True
    try:
        from hermes_constants import get_hermes_home
        auth_path = get_hermes_home() / "auth.json"
        if not auth_path.exists():
            return False
        store = json.loads(auth_path.read_text(encoding="utf-8-sig"))
        tokens = _dict_get(_dict_get(_dict_get(store, "providers"), "xai-oauth"), "tokens")
        if str(_dict_get(tokens, "access_token") or "").strip():
            return True
        entries = _dict_get(_dict_get(store, "credential_pool"), "xai-oauth")
        return isinstance(entries, list) and any(
            isinstance(e, dict) and str(e.get("access_token", "") or "").strip() for e in entries
        )
    except Exception:
        return False


def get_env_value(name: str, default=None):
    """Read ``name`` from ``~/.hermes/.env`` first, then ``os.environ``.

    Wraps :func:`hermes_cli.config.get_env_value` so tests can patch ``tools.xai_http.get_env_value``.
    """
    try:
        from hermes_cli.config import get_env_value as _hermes_get_env_value
    except ImportError:
        return os.environ.get(name, default)
    value = _hermes_get_env_value(name)
    return value if value is not None else default


def hermes_xai_user_agent() -> str:
    """Return a stable Hermes-specific User-Agent for xAI HTTP calls."""
    try:
        from hermes_cli import __version__
    except Exception:
        __version__ = "unknown"
    return f"Hermes-Agent/{__version__}"


def hermes_xai_default_headers() -> Dict[str, str]:
    """Default headers for OpenAI-SDK and raw HTTP clients talking to xAI (replaces the SDK User-Agent)."""
    return {"User-Agent": hermes_xai_user_agent()}


_TRUE_WORDS = {"1", "true", "yes", "on", "enabled"}
_FALSE_WORDS = {"0", "false", "no", "off", "disabled"}


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and (normalized := value.strip().lower()) in _TRUE_WORDS | _FALSE_WORDS:
        return normalized in _TRUE_WORDS
    return default


def _coerce_expires_after(value: Any) -> Optional[int]:
    """Normalize an xAI storage TTL: int seconds, or None for permanent storage (omit on the wire)."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"", "default", "none", "null", "never", "permanent", "forever", "0"}:
            return None
        try:
            value = int(value)
        except ValueError:
            return SAFE_XAI_STORAGE_EXPIRES_AFTER_SECONDS
    if not isinstance(value, (int, float)):
        return SAFE_XAI_STORAGE_EXPIRES_AFTER_SECONDS
    return None if int(value) <= 0 else min(int(value), MAX_XAI_STORAGE_EXPIRES_AFTER_SECONDS)


def read_xai_imagine_storage_config(section_name: str) -> Dict[str, Any]:
    """Read ``<section_name>.xai.storage`` (``image_gen``/``video_gen``) -> {enabled, public_url, expires_after}.
    On by default so xAI returns permanent public URLs, not short-lived CDN ones; null TTL = permanent."""
    try:
        from hermes_cli.config import load_config
        storage = _dict_get(_dict_get(_dict_get(load_config(), section_name), "xai"), "storage")
    except Exception:
        storage = None
    storage = storage if isinstance(storage, dict) else {}
    return {
        "enabled": _coerce_bool(storage.get("enabled"), True),
        "public_url": _coerce_bool(storage.get("public_url"), True),
        "expires_after": _coerce_expires_after(storage.get("expires_after")),
    }


def build_xai_storage_options(
    section_name: str, *, filename_prefix: str, extension: str,
) -> Optional[Dict[str, Any]]:
    """Return an xAI ``storage_options`` payload, or None when disabled."""
    cfg = read_xai_imagine_storage_config(section_name)
    if not cfg["enabled"]:
        return None
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
    filename = f"{filename_prefix}-{ts}-{uuid.uuid4().hex[:8]}.{extension.lstrip('.') or 'bin'}"
    payload: Dict[str, Any] = {"filename": filename, "public_url": bool(cfg["public_url"])}
    if cfg["expires_after"] is not None:
        payload["expires_after"] = cfg["expires_after"]
    return payload


def xai_storage_notice_text(section_name: str) -> str:
    """User-facing notice for first xAI Imagine storage use."""
    cfg = read_xai_imagine_storage_config(section_name)
    if not cfg["enabled"]:
        return ""
    retention = "without an automatic expiry"
    if cfg["expires_after"] is not None:
        days = cfg["expires_after"] / (24 * 60 * 60)
        retention = f"for about {days:g} day{'s' if days != 1 else ''}"
    return (
        "xAI Imagine storage is enabled so generated media gets a reusable "
        f"public URL {retention}. xAI may bill for stored files and public URL "
        f"hosting. Disable this with `{section_name}.xai.storage.enabled: false` "
        "or set `expires_after` to change the retention."
    )


def maybe_mark_xai_storage_notice_seen(section_name: str) -> Optional[str]:
    """Return the storage notice once per Hermes home, then mark it seen."""
    notice = xai_storage_notice_text(section_name)
    if not notice:
        return None
    try:
        from hermes_constants import get_hermes_home
        marker_dir = get_hermes_home() / "state"
        marker_dir.mkdir(parents=True, exist_ok=True)
        marker = marker_dir / f"{section_name}_xai_storage_notice_seen"
        if marker.exists():
            return None
        marker.write_text(datetime.datetime.now(datetime.UTC).isoformat() + "\n", encoding="utf-8")
        return notice
    except Exception:
        return notice


def _resolve_explicit_xai_api_key() -> str:
    """Read ``XAI_API_KEY`` via ``resolve_provider_secret`` (config → profile scope → env/.env → pool).

    Both the preferred-key and the no-OAuth fallback paths go through here so scope policy
    (incl. failing closed in a multiplexed gateway turn) is never re-implemented per caller.
    """
    from tools.tool_backend_helpers import resolve_provider_secret
    return resolve_provider_secret("XAI_API_KEY", "xai", env_getter=get_env_value)


def _xai_base_url_override() -> str:
    """``HERMES_XAI_BASE_URL`` then ``XAI_BASE_URL``, stripped; '' when unset."""
    return str(get_env_value("HERMES_XAI_BASE_URL") or get_env_value("XAI_BASE_URL") or "").strip().rstrip("/")


def resolve_xai_http_credentials(
    *, force_refresh: bool = False, api_key_hint: Optional[str] = None, prefer_api_key: bool = False,
) -> Dict[str, str]:
    """Resolve bearer credentials for direct xAI HTTP endpoints.

    Default order: Hermes-managed xAI OAuth, then ``XAI_API_KEY`` (via ``get_env_value`` so
    ``~/.hermes/.env`` keys count). ``prefer_api_key=True`` inverts that for API-metered
    endpoints where the subscription OAuth bearer authorizes but misbehaves (x_search answers
    without citations, TTS 403s). Both branches honor ``HERMES_XAI_BASE_URL``/``XAI_BASE_URL``
    behind the same origin-pinning validation. ``force_refresh=True`` forces an OAuth refresh;
    pass the rejected bearer as ``api_key_hint`` so a multi-account pool refreshes the issuing
    entry, not whichever its strategy selects first.

    Prefers Hermes-managed xAI OAuth credentials when available, then falls back to ``XAI_API_KEY`` resolved
    via ``hermes_cli.config.get_env_value`` so keys stored in ``~/.hermes/.env`` (the standard Hermes
    location) are honored — not just ones already exported into ``os.environ``. This keeps direct xAI
    endpoints (images, TTS, STT, etc.) aligned with the main runtime auth model and preserves the regression
    contract from PR #17140 / #17163.
    The key is read through :func:`tools.tool_backend_helpers.resolve_provider_secret` so profile secret
    scoping is identical to the fallback branch, and the base URL honors ``HERMES_XAI_BASE_URL`` /
    ``XAI_BASE_URL`` behind the same origin-pinning validation as the OAuth branch. See #87045, #88040.
    """
    import hermes_cli.auth as auth_mod
    if prefer_api_key and (explicit_key := str(_resolve_explicit_xai_api_key() or "").strip()):
        # Origin-pinned so a tampered env override can't exfiltrate the bearer; rejection -> default URL.
        override = _xai_base_url_override()
        base_url = auth_mod._xai_validate_inference_base_url(override, fallback=DEFAULT_XAI_BASE_URL)
        return {"provider": "xai", "api_key": explicit_key, "base_url": base_url}

    try:
        from agent.credential_pool import load_pool
        pool = load_pool("xai-oauth")
        entry = pool.try_refresh_matching(api_key_hint) if force_refresh else pool.select()
        if force_refresh and entry is None:
            # A rejected refresh may quarantine the issuing entry; continue with
            # the next healthy account rather than resurrecting the stale row.
            entry = pool.select()
        access_token = getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")
        fallback_base_url = str(
            getattr(entry, "runtime_base_url", None)
            or getattr(entry, "base_url", "")
            or auth_mod.DEFAULT_XAI_OAUTH_BASE_URL
        ).strip().rstrip("/")
        base_url = auth_mod._xai_validate_inference_base_url(_xai_base_url_override(), fallback=fallback_base_url)
        if str(access_token).strip():
            return {"provider": "xai-oauth", "api_key": str(access_token).strip(), "base_url": base_url}
    except Exception:
        pass

    api_key = _resolve_explicit_xai_api_key()
    base_url = str(get_env_value("XAI_BASE_URL") or DEFAULT_XAI_BASE_URL).strip().rstrip("/")
    return {"provider": "xai", "api_key": api_key, "base_url": base_url}
