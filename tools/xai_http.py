"""Shared helpers for direct xAI HTTP integrations."""

from __future__ import annotations

import os
from typing import Dict


def hermes_xai_user_agent() -> str:
    """Return a stable Hermes-specific User-Agent for xAI HTTP calls."""
    try:
        from hermes_cli import __version__
    except Exception:
        __version__ = "unknown"
    return f"Hermes-Agent/{__version__}"


def resolve_xai_http_credentials() -> Dict[str, str]:
    """Resolve bearer credentials for direct xAI HTTP endpoints.

    Prefers Hermes-managed xAI OAuth credentials when available, then falls back
    to ``XAI_API_KEY`` from the environment. This keeps direct xAI endpoints
    (images, TTS, STT, etc.) aligned with the main runtime auth model.
    """
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

        creds = resolve_xai_oauth_runtime_credentials()
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

    api_key = os.getenv("XAI_API_KEY", "").strip()
    base_url = (os.getenv("XAI_BASE_URL") or "https://api.x.ai/v1").strip().rstrip("/")
    return {
        "provider": "xai",
        "api_key": api_key,
        "base_url": base_url,
    }
