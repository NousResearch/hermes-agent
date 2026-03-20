"""Abstract base class for cloud browser providers."""

from abc import ABC, abstractmethod
from typing import Dict


class CloudBrowserProvider(ABC):
    """Interface for cloud browser backends (Browserbase, Steel, etc.).

    Implementations live in sibling modules and are registered in
    ``browser_tool._PROVIDER_REGISTRY``.  The user selects a provider via
    ``hermes setup`` / ``hermes tools``; the choice is persisted as
    ``config["browser"]["cloud_provider"]``.
    """

    @abstractmethod
    def provider_name(self) -> str:
        """Short, human-readable name shown in logs and diagnostics."""

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True when all required env vars / credentials are present.

        Called at tool-registration time (``check_browser_requirements``) to
        gate availability.  Must be cheap — no network calls.
        """

    @abstractmethod
    def create_session(self, task_id: str) -> Dict[str, object]:
        """Create a cloud browser session and return session metadata.

        Must return a dict with at least::

            {
                "session_name": str,   # unique name for agent-browser --session
                "bb_session_id": str,  # provider session ID (for close/cleanup)
                "cdp_url": str,        # CDP websocket URL
                "features": dict,      # feature flags that were enabled
            }

        ``bb_session_id`` is a legacy key name kept for backward compat with
        the rest of browser_tool.py — it holds the provider's session ID
        regardless of which provider is in use.
        """

    @abstractmethod
    def close_session(self, session_id: str) -> bool:
        """Release / terminate a cloud session by its provider session ID.

        Returns True on success, False on failure.  Should not raise.
        """

    @abstractmethod
    def emergency_cleanup(self, session_id: str) -> None:
        """Best-effort session teardown during process exit.

        Called from atexit / signal handlers.  Must tolerate missing
        credentials, network errors, etc. — log and move on.
        """
