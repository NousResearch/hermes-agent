"""Shared OpenRouter API client for Hermes tools.

Provides a single lazy-initialized AsyncOpenAI client that all tool modules
can share.  Routes through the centralized provider router in
agent/auxiliary_client.py so auth, headers, and API format are handled
consistently.
"""

import os

_client = None


def get_async_client():
    """Return a shared async OpenAI-compatible client for OpenRouter.

    The client is created lazily on first call and reused thereafter.
    Uses the centralized provider router for auth and client construction.
    Raises ValueError if OPENROUTER_API_KEY is not set.
    """
    global _client
    if _client is None:
        from agent.auxiliary_client import resolve_provider_client
        client, _model = resolve_provider_client("openrouter", async_mode=True)
        if client is None:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        _client = client
    return _client


def check_api_key() -> bool:
    """Check whether the OpenRouter API key is present.

    Scope-aware (Slack pattern): tool paths run inside an installed profile
    secret scope, whose verdict is authoritative under multiplex; unscoped
    CLI probes keep the legacy env read.
    """
    try:
        from agent.secret_scope import UnscopedSecretError, get_secret

        try:
            return bool(get_secret("OPENROUTER_API_KEY"))
        except UnscopedSecretError:
            pass
    except Exception:
        pass
    return bool(os.getenv("OPENROUTER_API_KEY"))
