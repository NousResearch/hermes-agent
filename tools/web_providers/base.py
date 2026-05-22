"""Abstract base classes for web capability providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class WebSearchProvider(ABC):
    """Interface for web search backends (Firecrawl, Tavily, Exa, etc.).

    Implementations live in sibling modules.  The user selects a provider
    via ``hermes tools``; the choice is persisted as
    ``config["web"]["search_backend"]`` (falling back to
    ``config["web"]["backend"]``).

    Search providers return results in a normalized format::

        {
            "success": True,
            "data": {
                "web": [
                    {"title": str, "url": str, "description": str, "position": int},
                    ...
                ]
            }
        }

    On failure::

        {"success": False, "error": str}
    """

    @abstractmethod
    def provider_name(self) -> str:
        """Short, human-readable name shown in logs and diagnostics."""

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True when all required env vars / credentials are present.

        Called at tool-registration time to gate availability.
        Must be cheap — no network calls.
        """

    @abstractmethod
    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a web search and return normalized results."""


class WebExtractProvider(ABC):
    """Interface for web content extraction backends.

    Implementations live in sibling modules.  The user selects a provider
    via ``hermes tools``; the choice is persisted as
    ``config["web"]["extract_backend"]`` (falling back to
    ``config["web"]["backend"]``).

    Extract providers return results in a normalized format::

        {
            "success": True,
            "data": [
                {"url": str, "title": str, "content": str,
                 "raw_content": str, "metadata": dict},
                ...
            ]
        }

    On failure::

        {"success": False, "error": str}
    """

    @abstractmethod
    def provider_name(self) -> str:
        """Short, human-readable name shown in logs and diagnostics."""

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True when all required env vars / credentials are present.

        Called at tool-registration time to gate availability.
        Must be cheap — no network calls.
        """

    @abstractmethod
    def extract(self, urls: List[str], **kwargs) -> Dict[str, Any]:
        """Extract content from the given URLs and return normalized results."""
