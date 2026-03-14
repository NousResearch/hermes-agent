"""Runtime surface detection helpers shared across tool modules."""

import os


def is_gateway_surface() -> bool:
    """Return True when running inside a gateway or messaging session.

    Checks both the explicit gateway flag and the session-platform env var
    so that all callers use a single, consistent definition.
    """
    if os.getenv("HERMES_GATEWAY_SESSION"):
        return True
    session_platform = os.getenv("HERMES_SESSION_PLATFORM", "").strip().lower()
    return bool(session_platform) and session_platform != "cli"
