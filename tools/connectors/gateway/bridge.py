"""Core boundary for connector search, descriptions, and remote execution.

Search and description failures degrade to no remote results; exceptions cannot escape this bridge because core dispatch bypasses registry error wrapping.
"""

from __future__ import annotations

import logging
from dataclasses import replace as dataclass_replace
from typing import Any, Callable, Optional, Sequence

from tools.connectors.gateway.config import connectors_available
from tools.connectors.gateway.errors import GatewayUnavailable, ToolGatewayError
from tools.connectors.gateway.merge import fill_remote_failure, splice_remote_results
from tools.connectors.gateway.names import parse_connector_name, vendor_slug_candidates

logger = logging.getLogger(__name__)

__all__ = ["connector_describe", "connector_search_hits", "run_remote"]


def _default_client_factory():
    from tools.connectors.gateway.client import ConnectorClient

    return ConnectorClient()


def connector_search_hits(
    queries: Sequence[dict[str, Any]],
    *,
    availability: Optional[Callable[[], bool]] = None,
    client_factory: Optional[Callable[[], Any]] = None,
) -> dict[str, Any]:
    """Return no hits on failure so local search behavior is unchanged."""
    try:
        available = (availability or connectors_available)()
        if not available or not queries:
            return {}
        client = (client_factory or _default_client_factory)()
        return client.search(list(queries)) or {}
    except GatewayUnavailable:
        logger.debug("Connector search skipped: gateway dark")
        return {}
    except Exception as exc:
        logger.debug("Connector search failed silently (D32): %s", exc)
        return {}


def connector_describe(
    names: Sequence[str],
    *,
    availability: Optional[Callable[[], bool]] = None,
    client_factory: Optional[Callable[[], Any]] = None,
) -> dict[str, Any]:
    """Return no schemas on failure so local descriptions are unchanged."""
    try:
        available = (availability or connectors_available)()
        if not available:
            return {}
        # Resolve each name independently: candidates can overlap across composed names.
        wanted: dict[str, tuple[str, ...]] = {}
        request_slugs: list[str] = []
        for name in names:
            parsed = parse_connector_name(name)
            if parsed is None or parsed.raw in wanted:
                continue
            candidates = vendor_slug_candidates(parsed.connector, parsed.tool)
            wanted[parsed.raw] = candidates
            for slug in candidates:
                if slug not in request_slugs:
                    request_slugs.append(slug)
        if not wanted:
            return {}
        client = (client_factory or _default_client_factory)()
        response = client.schemas(request_slugs) or {}
        schemas = response.get("schemas") if isinstance(response.get("schemas"), dict) else {}
        tools: dict[str, Any] = {}
        for composed, candidates in wanted.items():
            schema = next(
                (schemas[slug] for slug in candidates
                 if isinstance(schemas.get(slug), dict)),
                None,
            )
            if schema is None:
                continue
            tools[composed] = {
                "description": str(schema.get("description") or ""),
                "parameters": schema.get("input_schema") or {},
            }
        return {"tools": tools}
    except GatewayUnavailable:
        logger.debug("Connector describe skipped: gateway dark")
        return {}
    except Exception as exc:
        logger.debug("Connector describe failed silently (D32): %s", exc)
        return {}


def run_remote(
    planned,
    dispatch_id: Optional[str],
    *,
    availability: Optional[Callable[[], bool]],
    client_factory: Optional[Callable[[], Any]],
) -> list[dict[str, Any]]:
    try:
        available = (availability or connectors_available)()
    except Exception:
        available = False
    if not available:
        return fill_remote_failure(
            planned,
            "Unknown tool: connectors are not available in this session.",
            code="TOOL_NOT_FOUND",
        )

    # Try the conventional prefix first; only confirmed misses get a literal retry.
    wire_planned = [
        dataclass_replace(
            plan,
            tool=vendor_slug_candidates(plan.connector, plan.tool)[0],
        )
        for plan in planned
    ]
    try:
        client = (client_factory or _default_client_factory)()
        remote_results = client.execute(wire_planned)
        entries = splice_remote_results(planned, remote_results)
    except ToolGatewayError as exc:
        logger.debug(
            "Connector execute for dispatch %s failed (%s): %s",
            dispatch_id,
            exc.code,
            exc,
        )
        return fill_remote_failure(
            planned, f"The connector gateway request failed: {exc}"
        )
    except Exception as exc:
        logger.warning(
            "Connector execute for dispatch %s failed unexpectedly: %s",
            dispatch_id,
            exc,
        )
        return fill_remote_failure(
            planned, "The connector gateway request failed unexpectedly."
        )

    fallback_slots: list[int] = []
    fallback_planned = []
    for slot, (plan, entry) in enumerate(zip(planned, entries)):
        primary, literal = vendor_slug_candidates(plan.connector, plan.tool)
        error = entry.get("error") if isinstance(entry, dict) else None
        if (
            primary != literal
            and isinstance(error, dict)
            and error.get("code") == "TOOL_NOT_FOUND"
        ):
            fallback_slots.append(slot)
            fallback_planned.append(dataclass_replace(plan, tool=literal))
    if not fallback_planned:
        return entries

    # Retry confirmed misses once without disturbing successful sibling slots.
    try:
        fallback_results = client.execute(fallback_planned)
        fallback_entries = splice_remote_results(fallback_planned, fallback_results)
    except ToolGatewayError as exc:
        logger.debug(
            "Connector execute fallback for dispatch %s failed (%s): %s",
            dispatch_id,
            exc.code,
            exc,
        )
        fallback_entries = fill_remote_failure(
            fallback_planned, f"The connector gateway request failed: {exc}"
        )
    except Exception as exc:
        logger.warning(
            "Connector execute fallback for dispatch %s failed unexpectedly: %s",
            dispatch_id,
            exc,
        )
        fallback_entries = fill_remote_failure(
            fallback_planned, "The connector gateway request failed unexpectedly."
        )
    for slot, fallback_entry in zip(fallback_slots, fallback_entries):
        entries[slot] = fallback_entry
    return entries
