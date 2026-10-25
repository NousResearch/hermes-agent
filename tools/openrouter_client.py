"""OpenRouter API key probe shared by Hermes tools."""

import os


def check_api_key() -> bool:
    """Return True if OPENROUTER_API_KEY is present.

    Scope-aware: an installed profile secret scope is authoritative under
    multiplex; unscoped CLI probes fall back to the plain env read. Any other
    scope failure propagates -- a failed scoped read must never silently
    borrow another profile's key from ``os.environ``.
    """
    from agent.secret_scope import UnscopedSecretError, get_secret

    try:
        return bool(get_secret("OPENROUTER_API_KEY"))
    except UnscopedSecretError:
        return bool(os.getenv("OPENROUTER_API_KEY"))
