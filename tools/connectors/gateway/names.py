"""Codec for model-facing connector tool names.

Compose and parse bridge names only here. The composed-name collision resolves prefixed vendor slugs first everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = [
    "CONNECTOR_BATCH_SENTINEL",
    "CONNECTOR_NAME_PREFIX",
    "ConnectorName",
    "format_connector_name",
    "is_connector_name",
    "parse_connector_name",
    "vendor_slug_candidates",
]

CONNECTOR_NAME_PREFIX = "connectors__"

# Planner sentinel, not a callable connector name.
CONNECTOR_BATCH_SENTINEL = "connectors__execute"


@dataclass(frozen=True)
class ConnectorName:

    raw: str
    connector: str
    tool: str


def is_connector_name(name: object) -> bool:
    """Prefix match only; malformed names are rejected by the parser."""
    return isinstance(name, str) and name.startswith(CONNECTOR_NAME_PREFIX)


def parse_connector_name(name: object) -> Optional[ConnectorName]:
    """Use a bounded split so tool slugs retain internal underscores."""
    if not isinstance(name, str):
        return None
    parts = name.split("__", 2)
    if len(parts) != 3:
        return None
    prefix, connector, tool = parts
    if prefix != "connectors" or not connector or not tool:
        return None
    return ConnectorName(raw=name, connector=connector, tool=tool)


def format_connector_name(connector: str, tool: str) -> str:
    prefix = f"{connector.upper()}_"
    if tool.startswith(prefix):
        tool = tool[len(prefix):]
    return f"{CONNECTOR_NAME_PREFIX}{connector}__{tool}"


def vendor_slug_candidates(connector: str, tool: str) -> tuple[str, ...]:
    """Try the stripped conventional prefix before a literal vendor slug."""
    return (f"{connector.upper()}_{tool}", tool)
