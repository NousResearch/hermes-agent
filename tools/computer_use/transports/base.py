"""Transport contract for CUA MCP services."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping


class CuaToolTransport(ABC):
    """Lifecycle and RPC interface for a cua-driver MCP endpoint."""

    @abstractmethod
    def start(self) -> None:
        """Connect to or start the CUA service."""

    @abstractmethod
    def stop(self) -> None:
        """Release local transport resources."""

    @abstractmethod
    def list_tools(self) -> list[Mapping[str, Any]]:
        """Return MCP tool descriptions."""

    @abstractmethod
    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        """Call a CUA MCP tool."""

    @abstractmethod
    def is_alive(self) -> bool:
        """Return whether the service can accept requests."""
