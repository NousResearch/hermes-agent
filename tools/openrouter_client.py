"""Shared OpenRouter API client for Hermes tools.

Provides a single lazy-initialized AsyncOpenAI client that all tool modules
can share.  Routes through the centralized provider router in
agent/auxiliary_client.py so auth, headers, and API format are handled
consistently.
"""

import os

# Client cache keyed by (base_url, api_key) so a custom endpoint never reuses
# the default OpenRouter client (and vice versa).  The default path uses the
# ("", "") key and caches/behaves exactly as the previous single-global did.
_clients: dict = {}


def get_async_client(base_url: str = "", api_key: str = ""):
    """Return a shared async OpenAI-compatible client for OpenRouter.

    Why: lets MoA (and any tool) point its calls at a custom OpenAI-compatible
    endpoint (LiteLLM, vLLM, a local proxy) via *base_url*/*api_key* without
    every call rebuilding a client — while the default empty-args path keeps
    the original single-shared-client behavior.
    What: lazily builds and caches one client per (base_url, api_key) pair.
    When *base_url* is set the request is routed through the ``custom``
    provider (explicit_base_url/explicit_api_key); otherwise it uses
    ``openrouter`` exactly as before.
    Test: call twice with the same args → assert the same object is returned
    (cache hit); call with a distinct base_url → assert a different object;
    with no args and no OPENROUTER_API_KEY → assert ValueError is raised.

    Raises ValueError if no usable credentials are available for the resolved
    provider (e.g. OPENROUTER_API_KEY unset on the default path).
    """
    base_url = (base_url or "").strip()
    api_key = (api_key or "").strip()
    cache_key = (base_url, api_key)
    if cache_key not in _clients:
        from agent.auxiliary_client import resolve_provider_client
        # A custom base_url is an OpenAI-compatible endpoint, not OpenRouter —
        # the openrouter branch ignores explicit_base_url, so route through the
        # custom provider which honors both overrides (and rewraps Anthropic-wire
        # endpoints automatically).
        provider = "custom" if base_url else "openrouter"
        # Pass the already-stripped strings ("" when unset) rather than `… or None`:
        # resolve_provider_client treats "" and None identically on every path
        # it takes here — the custom branch gates on `if explicit_base_url:` and
        # coalesces `explicit_api_key or ""`, and the openrouter branch only does
        # `explicit_api_key or env`. Passing str (not str|None) also matches the
        # upstream `explicit_base_url: str` / `explicit_api_key: str` annotations.
        client, _model = resolve_provider_client(
            provider,
            async_mode=True,
            explicit_base_url=base_url,
            explicit_api_key=api_key,
        )
        if client is None:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        _clients[cache_key] = client
    return _clients[cache_key]


def check_api_key() -> bool:
    """Check whether the OpenRouter API key is present."""
    return bool(os.getenv("OPENROUTER_API_KEY"))
