"""Shared OpenRouter API client for Hermes tools.

Provides a single lazy-initialized AsyncOpenAI client that all tool modules
can share.  Routes through the centralized provider router in
agent/auxiliary_client.py so auth, headers, and API format are handled
consistently.
"""

import threading

from agent.auxiliary_client import _client_cache_lock, resolve_provider_client

_client = None


def get_async_client():
    """Return a shared async OpenAI-compatible client for OpenRouter.

    The client is created lazily on first call and reused thereafter.
    Uses double-checked locking to prevent TOCTOU race conditions when
    multiple threads call this concurrently.

    Raises ValueError if OPENROUTER_API_KEY is not set.
    """
    global _client
    if _client is None:
        with _client_cache_lock:
            # Double-check inside the lock — another thread may have created it
            # while we were waiting to acquire the lock.
            if _client is None:
                client, _model = resolve_provider_client("openrouter", async_mode=True)
                if client is None:
                    raise ValueError("OPENROUTER_API_KEY environment variable not set")
                _client = client
    return _client


def check_api_key() -> bool:
    """Check whether the OpenRouter API key is present."""
    import os  # local import to avoid import-time side effects during testing
    return bool(os.getenv("OPENROUTER_API_KEY"))
