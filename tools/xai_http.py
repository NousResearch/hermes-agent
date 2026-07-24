"""Shared helpers for direct xAI HTTP integrations."""

from __future__ import annotations

import datetime
import json
import os
import uuid
from typing import Any, Dict, Optional


MAX_XAI_STORAGE_EXPIRES_AFTER_SECONDS = 30 * 24 * 60 * 60
SAFE_XAI_STORAGE_EXPIRES_AFTER_SECONDS = 2 * 24 * 60 * 60


def has_xai_credentials() -> bool:
    """Cheap probe — return True when xAI credentials are *likely* usable.

    Deliberately avoids :func:`resolve_xai_http_credentials` so callers in
    hot-paint paths (``hermes tools`` repaint, tool-registration scans,
    ``WebSearchProvider.is_available()``) don't incur disk locks or — in
    the OAuth path — a network token refresh. The ABC contract on
    :meth:`agent.web_search_provider.WebSearchProvider.is_available`
    explicitly forbids network calls for exactly this reason.

    Resolution order, fast-to-slow:

    1. ``XAI_API_KEY`` env var (cheapest; covers explicit-key users).
    2. **Shared mode (C2/F2/R7/H2):** the canonical shared store has usable
       tokens (single file read, no refresh). Profile-disabled → False.
       Empty shared + sole live local grant that can be auto-promoted (F1) →
       True only when profile AND root together yield exactly one distinct
       live identity (matches the promoter; no profile-first short-circuit).
       Canonical READ / unreadable-store errors fail closed (False) with NO
       legacy fallthrough.
    3. (gate OFF only) ``~/.hermes/auth.json`` providers.xai-oauth access_token
       or pool-only grants.

    Returns False on any exception so a corrupted auth store can't block
    other availability scans. Truthful refresh + expiry handling happens
    in ``search()`` (or whichever caller actually makes the request).
    """
    if os.environ.get("XAI_API_KEY", "").strip():
        return True
    try:
        from hermes_cli import auth as auth_mod
    except Exception:
        auth_mod = None

    # R7: under shared mode, never fall through to the legacy profile scan.
    if auth_mod is not None and auth_mod._xai_shared_auth_enabled():
        try:
            if auth_mod._profile_xai_shared_disabled():
                return False
            shared = auth_mod._read_shared_xai_state(raise_on_unreadable=True)
            if auth_mod._xai_shared_state_has_usable_tokens(shared):
                return True
            # Tombstoned / quarantined: not available, not promotable.
            if auth_mod._shared_xai_state_is_quarantined(shared):
                return False
            # F1/R7/H2: never-initialized shared still counts when a sole live
            # local grant can be auto-promoted. Consider profile AND root
            # together (no profile-first short-circuit that would advertise
            # available while the promoter rejects cross-store ambiguity).
            return bool(
                auth_mod._xai_sole_live_promotable_across_profile_and_root_for_probe()
            )
        except Exception:
            # Shared-mode read/audit failure → fail closed (no legacy scan).
            return False

    try:
        from hermes_constants import get_hermes_home

        auth_path = get_hermes_home() / "auth.json"
        if not auth_path.exists():
            return False
        store = json.loads(auth_path.read_text())
        providers = store.get("providers") if isinstance(store, dict) else None
        xai_state = providers.get("xai-oauth") if isinstance(providers, dict) else None
        tokens = xai_state.get("tokens") if isinstance(xai_state, dict) else None
        access_token = tokens.get("access_token") if isinstance(tokens, dict) else None
        if str(access_token or "").strip():
            return True
        # Pool-only grants (multi-account ``auth add``) never write the
        # providers singleton; still count as present credentials.
        credential_pool = store.get("credential_pool") if isinstance(store, dict) else None
        entries = (
            credential_pool.get("xai-oauth")
            if isinstance(credential_pool, dict)
            else None
        )
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                if str(entry.get("access_token", "") or "").strip():
                    return True
        return False
    except Exception:
        return False


def get_env_value(name: str, default=None):
    """Read ``name`` from ``~/.hermes/.env`` first, then ``os.environ``.

    Wraps :func:`hermes_cli.config.get_env_value` so tests can patch
    ``tools.xai_http.get_env_value`` to inject dotenv-only secrets into the
    xAI credential resolver.
    """
    try:
        from hermes_cli.config import get_env_value as _hermes_get_env_value

        value = _hermes_get_env_value(name)
        if value is not None:
            return value
    except Exception:
        pass
    return os.environ.get(name, default)


def hermes_xai_user_agent() -> str:
    """Return a stable Hermes-specific User-Agent for xAI HTTP calls."""
    try:
        from hermes_cli import __version__
    except Exception:
        __version__ = "unknown"
    return f"Hermes-Agent/{__version__}"


def _load_config_section(section_name: str) -> Dict[str, Any]:
    """Return a top-level Hermes config section as a dict, or empty."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get(section_name) if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception:
        return {}


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "enabled"}:
            return True
        if normalized in {"0", "false", "no", "off", "disabled"}:
            return False
    return default


def _coerce_expires_after(value: Any) -> Optional[int]:
    """Normalize an xAI storage TTL.

    Returns:
        int seconds for an expiring file,
        None for permanent storage (omit expires_after on the wire).
    """
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "default"}:
            return None
        if normalized in {"none", "null", "never", "permanent", "forever", "0"}:
            return None
        try:
            value = int(normalized)
        except ValueError:
            return SAFE_XAI_STORAGE_EXPIRES_AFTER_SECONDS
    if isinstance(value, (int, float)):
        seconds = int(value)
        if seconds <= 0:
            return None
        return min(seconds, MAX_XAI_STORAGE_EXPIRES_AFTER_SECONDS)
    return SAFE_XAI_STORAGE_EXPIRES_AFTER_SECONDS


def read_xai_imagine_storage_config(section_name: str) -> Dict[str, Any]:
    """Read storage settings for xAI Imagine under image_gen/video_gen config.

    Supported config shape:

        image_gen:
          xai:
            storage:
              enabled: true
              public_url: true
              expires_after: null     # omit for permanent public URLs

    The same shape is accepted under ``video_gen.xai.storage``. Storage is on
    by default so xAI returns permanent public URLs instead of short-lived CDN URLs.
    """
    section = _load_config_section(section_name)
    xai_section = section.get("xai") if isinstance(section, dict) else None
    storage = xai_section.get("storage") if isinstance(xai_section, dict) else None
    storage = storage if isinstance(storage, dict) else {}

    enabled = _coerce_bool(storage.get("enabled"), True)
    public_url = _coerce_bool(storage.get("public_url"), True)
    expires_after = _coerce_expires_after(storage.get("expires_after"))

    return {
        "enabled": enabled,
        "public_url": public_url,
        "expires_after": expires_after,
    }


def build_xai_storage_options(
    section_name: str,
    *,
    filename_prefix: str,
    extension: str,
) -> Optional[Dict[str, Any]]:
    """Return an xAI ``storage_options`` payload, or None when disabled."""
    cfg = read_xai_imagine_storage_config(section_name)
    if not cfg["enabled"]:
        return None

    now = datetime.datetime.now(datetime.UTC)
    ts = now.strftime("%Y%m%d-%H%M%S")
    short = uuid.uuid4().hex[:8]
    ext = extension.lstrip(".") or "bin"
    payload: Dict[str, Any] = {
        "filename": f"{filename_prefix}-{ts}-{short}.{ext}",
        "public_url": bool(cfg["public_url"]),
    }
    if cfg["expires_after"] is not None:
        payload["expires_after"] = cfg["expires_after"]
    return payload


def xai_storage_notice_text(section_name: str) -> str:
    """User-facing notice for first xAI Imagine storage use."""
    cfg = read_xai_imagine_storage_config(section_name)
    if not cfg["enabled"]:
        return ""
    if cfg["expires_after"] is None:
        retention = "without an automatic expiry"
    else:
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
        marker.write_text(datetime.datetime.now(datetime.UTC).isoformat() + "\n")
        return notice
    except Exception:
        return notice


def is_xai_http_auth_status(status_code: Optional[int]) -> bool:
    """True for HTTP statuses that mean the bearer is rejected (C3)."""
    return status_code in {401, 403}


def resolve_xai_http_credentials(
    *,
    force_refresh: bool = False,
    api_key_hint: Optional[str] = None,
) -> Dict[str, str]:
    """Resolve bearer credentials for direct xAI HTTP endpoints.

    Prefers Hermes-managed xAI OAuth credentials when available, then falls back
    to ``XAI_API_KEY`` resolved via ``hermes_cli.config.get_env_value`` so keys
    stored in ``~/.hermes/.env`` (the standard Hermes location) are honored —
    not just ones already exported into ``os.environ``. This keeps direct xAI
    endpoints (images, TTS, STT, etc.) aligned with the main runtime auth model
    and preserves the regression contract from PR #17140 / #17163.

    Set ``force_refresh=True`` to perform an unconditional OAuth refresh.
    Reactive callers should also pass the rejected bearer as ``api_key_hint``
    so a freshly loaded multi-account pool refreshes the exact issuing entry,
    not whichever entry its strategy would otherwise select first.

    **Shared mode (C1/A6/F2):** resolves the canonical shared store first —
    never pool-first — so a legacy local/manual pool row cannot win over the
    fleet grant. On canonical failure / empty / profile-disabled, FAIL CLOSED:
    do not select a surviving legacy pool row. Only ``XAI_API_KEY`` remains as
    a non-OAuth fallback. Empty shared + sole local grant auto-promotes (F1)
    via the shared resolver before any strip.
    """
    import hermes_cli.auth as auth_mod

    # C1/A6/F2: shared mode → canonical only (no legacy pool fallback).
    if auth_mod._xai_shared_auth_enabled():
        try:
            creds = auth_mod.resolve_xai_oauth_runtime_credentials(
                force_refresh=force_refresh,
                refresh_if_expiring=not force_refresh,
                rejected_access_token=api_key_hint,
            )
            access_token = str(creds.get("api_key") or "").strip()
            if access_token:
                override_base_url = str(
                    get_env_value("HERMES_XAI_BASE_URL")
                    or get_env_value("XAI_BASE_URL")
                    or ""
                ).strip().rstrip("/")
                base_url = auth_mod._xai_validate_inference_base_url(
                    override_base_url,
                    fallback=str(
                        creds.get("base_url") or auth_mod.DEFAULT_XAI_OAUTH_BASE_URL
                    ).strip().rstrip("/"),
                )
                return {
                    "provider": "xai-oauth",
                    "api_key": access_token,
                    "base_url": base_url,
                    "source": auth_mod.XAI_SHARED_SOURCE,
                    "generation": creds.get("generation"),
                }
        except Exception:
            # Fail closed: do not select a legacy manual/pool OAuth row.
            pass
        # Non-OAuth API key is still allowed; OAuth pool is not.
        api_key = str(get_env_value("XAI_API_KEY") or "").strip()
        base_url = str(
            get_env_value("XAI_BASE_URL") or "https://api.x.ai/v1"
        ).strip().rstrip("/")
        return {
            "provider": "xai",
            "api_key": api_key,
            "base_url": base_url,
        }

    try:
        from agent.credential_pool import load_pool

        pool = load_pool("xai-oauth")
        entry = (
            pool.try_refresh_matching(api_key_hint)
            if force_refresh
            else pool.select()
        )
        if force_refresh and entry is None:
            # A rejected refresh may quarantine the issuing entry. Continue
            # with the next healthy account instead of falling back to the raw
            # singleton resolver and resurrecting the stale pool row.
            entry = pool.select()
        access_token = str(
            getattr(entry, "runtime_api_key", None)
            or getattr(entry, "access_token", "")
        ).strip()
        fallback_base_url = str(
            getattr(entry, "runtime_base_url", None)
            or getattr(entry, "base_url", "")
            or auth_mod.DEFAULT_XAI_OAUTH_BASE_URL
        ).strip().rstrip("/")
        override_base_url = str(
            get_env_value("HERMES_XAI_BASE_URL")
            or get_env_value("XAI_BASE_URL")
            or ""
        ).strip().rstrip("/")
        base_url = auth_mod._xai_validate_inference_base_url(
            override_base_url,
            fallback=fallback_base_url,
        )
        if access_token:
            return {
                "provider": "xai-oauth",
                "api_key": access_token,
                "base_url": base_url,
            }
    except Exception:
        pass

    api_key = str(get_env_value("XAI_API_KEY") or "").strip()
    base_url = str(get_env_value("XAI_BASE_URL") or "https://api.x.ai/v1").strip().rstrip("/")
    return {
        "provider": "xai",
        "api_key": api_key,
        "base_url": base_url,
    }


def force_refresh_xai_http_credentials(
    rejected_api_key: Optional[str] = None,
) -> Dict[str, str]:
    """C3 helper: canonical force-refresh after a 401/403 with the rejected bearer."""
    return resolve_xai_http_credentials(
        force_refresh=True,
        api_key_hint=rejected_api_key,
    )
