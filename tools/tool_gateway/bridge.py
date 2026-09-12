"""The one module core code imports for connector dispatch.

Two legs, both TOTAL: each entry point catches its own exceptions and returns
a structured value, because the bridge branch in core dispatch bypasses the
registry's catch-wrap. An exception escaping this module is a bug.

Availability leg: :func:`connector_search_hits` and :func:`connector_describe`
feed tool_search and tool_describe. Silent degradation (D32): every failure
returns ``{}``. Signed out, config off, or a dark gateway must leave local
search behaving exactly as it does today.

Transport leg: :func:`run_remote` sends one gateway execute request for the
planned entries it is handed and splices the results back by slot.
``model_tools_connectors`` calls it once per connector entry, after core
dispatch has already run scope, hook, approval and middleware policy against
that entry's composed ``connectors__`` name. Vendor slug restoration and the
single literal-slug retry live here; partition and envelope assembly live in
``merge.py``.
"""

from __future__ import annotations

import logging
from dataclasses import replace as dataclass_replace
from typing import Any, Callable, Optional, Sequence

from tools.tool_gateway.config import connectors_available
from tools.tool_gateway.errors import GatewayUnavailable, ToolGatewayError
from tools.tool_gateway.merge import fill_remote_failure, splice_remote_results
from tools.tool_gateway.names import parse_connector_name, vendor_slug_candidates

logger = logging.getLogger(__name__)

__all__ = ["connector_describe", "connector_search_hits", "run_remote"]


def _default_client_factory():
    from tools.tool_gateway.client import ConnectorClient

    return ConnectorClient()


def connector_search_hits(
    queries: Sequence[dict[str, Any]],
    *,
    availability: Optional[Callable[[], bool]] = None,
    client_factory: Optional[Callable[[], Any]] = None,
) -> dict[str, Any]:
    """Remote hits for tool_search, or ``{}`` on EVERY failure path (D32).

    A connector problem must never change local search behavior: the caller
    treats ``{}`` as "no remote results" and proceeds exactly as today.
    """
    try:
        available = (availability or connectors_available)()
        if not available or not queries:
            return {}
        client = (client_factory or _default_client_factory)()
        return client.search(list(queries)) or {}
    except GatewayUnavailable:
        # Connectors dark for this principal — the expected quiet path.
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
    """Schemas for ``connectors__*`` names, or ``{}`` on EVERY failure path (D32).

    Returns ``{"tools": {<composed name>: {"description", "parameters"}}}``
    keyed by the ORIGINAL composed names. Names the gateway does not resolve
    are simply absent — the caller's not_found handling covers them. The
    gateway's schemas route takes bare vendor slugs, so every deterministic
    recovery candidate is requested; mapping back to the ``connectors__``
    name uses the caller's own parse, never the response.
    """
    try:
        available = (availability or connectors_available)()
        if not available:
            return {}
        # Candidate sets from different names can nominate the SAME vendor
        # slug (one name's literal is another's prefixed primary), so the
        # slug->name mapping cannot be global. Resolution is per name: each
        # name takes the schema of its own best-ranked candidate that the
        # gateway resolved. First occurrence wins only for a DUPLICATED
        # composed name.
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
        # The model addressed connector names while connectors are off/dark —
        # per-entry unknown-tool errors, exactly like any unknown tool name.
        return fill_remote_failure(
            planned,
            "Unknown tool: connectors are not available in this session.",
            code="TOOL_NOT_FOUND",
        )

    # Composition cuts only the conventional toolkit prefix. Restore that
    # exact prefix before crossing the wire; literal recovery below covers
    # the convention's exceptions without probing entries that succeeded.
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

    # One literal pass only: retry confirmed misses together, then splice
    # those slots alone so successful and non-not-found siblings stay fixed.
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
