"""Codec for ``connectors__<connector>__<tool>`` bridge names.

Remote connector tools are never registered in the model-facing tools array;
the model addresses them through ``tool_call`` using composed names. Composition
strips the repeated toolkit prefix from vendor tool slugs; wire slugs are
reconstructed through :func:`vendor_slug_candidates`. This module is the ONLY
place that composes or parses those names — responses are correlated by array
position, never by re-parsing names.

Parsing rules:

- ``split("__", 2)`` exactly: tool slugs legitimately contain underscores
  (``GMAIL_SEND_EMAIL``), so ``rsplit`` or an unbounded split would corrupt
  them.
- Case is preserved: the gateway lowercases connector slugs itself, and the
  tool segment is otherwise untouched. Prefix removal is exactly reversible,
  including partial prefix matches such as ``granola`` +
  ``GRANOLA_MCP_GET_MEETINGS``: decoding prepends exactly what encoding cut.
- :func:`parse_connector_name` returns ``None`` and never raises — a
  malformed name is a per-entry error and sibling calls still run.
- Composition is deliberately NOT injective: ``GMAIL_X`` and a literal ``X``
  on connector ``gmail`` both compose to ``connectors__gmail__X``, and that
  name decodes to the prefixed slug first everywhere (describe, execute), so
  the literal twin is unreachable. Short names are worth more than a marker
  for a pair no vendor catalog is known to carry, and the client cannot know
  a vendor's slug set. Search, the one place that sees both twins, keeps the
  reachable one and logs a WARNING instead of describing the literal under a
  name that runs the prefixed tool.

stdlib-only leaf module.
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

# Planner sentinel: stands for "a tool_call carrying connector entries" in
# parallel-safety checks. It is not itself a callable name — it has only two
# segments, so parse_connector_name() rejects it by construction.
CONNECTOR_BATCH_SENTINEL = "connectors__execute"


@dataclass(frozen=True)
class ConnectorName:
    """A parsed ``connectors__<connector>__<tool>`` identifier."""

    raw: str
    connector: str
    tool: str


def is_connector_name(name: object) -> bool:
    """True when ``name`` claims to be a connector tool name.

    A claim, not a guarantee: a True here routes the entry to connector
    handling, where a failed parse becomes that entry's error.
    """
    return isinstance(name, str) and name.startswith(CONNECTOR_NAME_PREFIX)


def parse_connector_name(name: object) -> Optional[ConnectorName]:
    """Parse a composed name into its parts, or ``None`` if malformed.

    Never raises. Requires exactly three non-empty segments under a
    2-bounded split, so the tool slug keeps any internal underscores.
    """
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
    """Compose a model-facing name, stripping a repeated toolkit prefix."""
    prefix = f"{connector.upper()}_"
    if tool.startswith(prefix):
        tool = tool[len(prefix):]
    return f"{CONNECTOR_NAME_PREFIX}{connector}__{tool}"


def vendor_slug_candidates(connector: str, tool: str) -> tuple[str, ...]:
    """Return wire-slug candidates in deterministic recovery order.

    Prefixed-first restores every slug the encoder stripped; the literal
    covers slugs that never conformed and therefore composed verbatim.
    """
    return (f"{connector.upper()}_{tool}", tool)
