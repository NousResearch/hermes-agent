"""Shared helpers for direct xAI HTTP integrations."""

from __future__ import annotations

import json
import os
from typing import Dict


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
    2. ``~/.hermes/auth.json`` has a non-empty ``providers.xai-oauth.tokens.access_token``
       (single file read, no expiry check, no refresh).

    Returns False on any exception so a corrupted auth store can't block
    other availability scans. Truthful refresh + expiry handling happens
    in ``search()`` (or whichever caller actually makes the request).
    """
    if os.environ.get("XAI_API_KEY", "").strip():
        return True
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
        return bool(str(access_token or "").strip())
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


def resolve_xai_http_credentials(*, force_refresh: bool = False) -> dict[str, str]:
    """Resolve bearer credentials for direct xAI HTTP endpoints.

    Prefers Hermes-managed xAI OAuth credentials when available, then falls back
    to ``XAI_API_KEY`` resolved via ``hermes_cli.config.get_env_value`` so keys
    stored in ``~/.hermes/.env`` (the standard Hermes location) are honored —
    not just ones already exported into ``os.environ``. This keeps direct xAI
    endpoints (images, TTS, STT, etc.) aligned with the main runtime auth model
    and preserves the regression contract from PR #17140 / #17163.

    Set ``force_refresh=True`` to bypass the resolver's JWT-exp shortcut and
    perform an unconditional OAuth refresh. Callers should use this only as a
    reactive remediation after a server 401 (mid-window revocation, opaque
    tokens where the proactive JWT check is a no-op, etc.), not as a default —
    the auth-store lock is held for the duration of the refresh.
    """
    if not force_refresh:
        try:
            from hermes_cli.runtime_provider import resolve_runtime_provider

            runtime = resolve_runtime_provider(requested="xai-oauth")
            access_token = str(runtime.get("api_key") or "").strip()
            base_url = str(runtime.get("base_url") or "").strip().rstrip("/")
            if access_token:
                return {
                    "provider": "xai-oauth",
                    "api_key": access_token,
                    "base_url": base_url or "https://api.x.ai/v1",
                }
        except Exception:
            pass

    try:
        from hermes_cli.auth import resolve_xai_oauth_runtime_credentials

        creds = resolve_xai_oauth_runtime_credentials(force_refresh=force_refresh)
        access_token = str(creds.get("api_key") or "").strip()
        base_url = str(creds.get("base_url") or "").strip().rstrip("/")
        if access_token:
            return {
                "provider": "xai-oauth",
                "api_key": access_token,
                "base_url": base_url or "https://api.x.ai/v1",
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
