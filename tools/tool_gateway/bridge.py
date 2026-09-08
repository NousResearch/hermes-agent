"""The one module core code imports for connector dispatch.

Every entry point here is TOTAL: it catches its own exceptions and returns a
structured value, because the bridge branch in core dispatch bypasses the
registry's catch-wrap — an exception escaping this module is a bug, full
stop.

Per-entry policy runs through two injected seams: ``local_dispatch`` (local
entries recurse into core dispatch, where scope/probe/hook/middleware gates
fire against each real tool name) and ``pre_dispatch`` (remote entries get
the caller's hook/approval pass against their composed ``connectors__``
names; a block pre-fills that entry's ``USER_DENIED`` slot and its siblings
still run, and an argument REWRITE from that same pass travels to the wire).
The bridge itself only partitions, gates, dispatches, splices.

Dispatch shape (V1, decided 2026-08-25): local entries run via the injected
``local_dispatch``; ALL connector entries travel as ONE gateway execute
request. Split dispatch while approvals pend is a tracked follow-up, not
this code.

Silent degradation (D32): on the availability path (:func:`connector_search_hits`)
every failure returns ``{}`` — signed out, config off, or a dark gateway must
leave tool_search behaving exactly as it does today.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace as dataclass_replace
from typing import Any, Callable, Optional, Sequence

from tools.tool_gateway.config import MAX_CALLS_PER_DISPATCH, connectors_available
from tools.tool_gateway.errors import GatewayUnavailable, ToolGatewayError
from tools.tool_gateway.merge import (
    assemble_results,
    fill_remote_failure,
    partition_calls,
    splice_remote_results,
)
from tools.tool_gateway.names import parse_connector_name, vendor_slug_candidates

logger = logging.getLogger(__name__)

__all__ = ["connector_describe", "connector_search_hits", "dispatch_calls"]

# Signature of the injected local dispatcher: (name, arguments) -> (ok, payload).
#
# The CALLER states success. The bridge cannot tell a refusal from a result by
# looking at the payload: ``tool_error`` puts its message under ``error`` but a
# legitimate tool result may also carry an ``error`` field (a provider echo, a
# per-item error list), and sniffing for it both misclassified those results and
# forced an allow-list that dropped every other ``tool_error`` extra. So the
# dispatcher returns ``ok=False`` for the refusals IT raised, and the bridge
# files the payload verbatim in that entry's error slot.
#
# Consequence, deliberate: a real tool that fails inside its own handler still
# comes back ``ok=True``, so it lands in the response slot and counts as a
# success. The failure text reaches the model either way; only the envelope's
# success/error tallies treat it as a completed call.
LocalDispatch = Callable[[str, dict[str, Any]], "tuple[bool, Any]"]

# Signature of the injected per-remote-entry gate:
#   (name, arguments) -> (block_message, replacement_arguments)
#
# Both halves are independent and either may be None:
#
# - ``block_message`` not None  -> that entry becomes a USER_DENIED slot and
#   never reaches the wire. Its siblings still run.
# - ``replacement_arguments`` a dict -> the surviving remote call is rebuilt
#   with those arguments, so the gateway request body carries the REWRITTEN
#   values. ``None`` means "the gate requested no change" and the original
#   arguments go out verbatim — a hook that only observes must not be able to
#   silently blank a call's arguments, so "no change" and "changed to {}" are
#   kept distinct (an empty dict IS a rewrite to no arguments).
#
# Policy rewrites (sanitization, redaction) are the reason this half exists:
# the single-entry deferred path applies the pre_tool_call hook's modified
# args before dispatch, and remote entries must not be the one path where a
# redaction is dropped on the floor.
PreDispatch = Callable[
    [str, dict[str, Any]], "tuple[Optional[str], Optional[dict[str, Any]]]"
]


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


def dispatch_calls(
    calls: Sequence[Any],
    dispatch_id: Optional[str] = None,
    *,
    local_dispatch: LocalDispatch,
    pre_dispatch: Optional[PreDispatch] = None,
    availability: Optional[Callable[[], bool]] = None,
    client_factory: Optional[Callable[[], Any]] = None,
) -> str:
    """Run a mixed ``calls[]`` batch and return the merged results envelope.

    Total: any internal failure becomes per-entry errors or an envelope-level
    ``{"error": ...}`` string — never an exception. ``local_dispatch`` is
    injected so this module never imports core dispatch code; it returns
    ``(ok, payload)`` so the CALLER classifies its own refusals (see
    :data:`LocalDispatch`).

    ``pre_dispatch`` is the caller's per-REMOTE-entry gate (hooks/approval):
    called with (name, arguments) before the gateway request is built, and
    returning ``(block_message, replacement_arguments)`` (see
    :data:`PreDispatch`). A non-None block message pre-fills that entry's
    ``USER_DENIED`` error slot and excludes it from the remote batch while
    its siblings still run; a replacement dict on a surviving entry is the
    arguments that actually go out on the wire. Local entries gate inside
    ``local_dispatch``.
    """
    try:
        return _dispatch_calls_inner(
            calls,
            dispatch_id,
            local_dispatch=local_dispatch,
            pre_dispatch=pre_dispatch,
            availability=availability,
            client_factory=client_factory,
        )
    except Exception as exc:  # the bridge branch has no catch-wrap above us
        logger.warning("Connector dispatch %s failed: %s", dispatch_id, exc)
        return json.dumps(
            {"error": f"tool_call batch dispatch failed internally: {exc}"}
        )


def _dispatch_calls_inner(
    calls: Sequence[Any],
    dispatch_id: Optional[str],
    *,
    local_dispatch: LocalDispatch,
    pre_dispatch: Optional[PreDispatch],
    availability: Optional[Callable[[], bool]],
    client_factory: Optional[Callable[[], Any]],
) -> str:
    if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)) or not calls:
        return json.dumps(
            {"error": "calls is required and must contain at least one entry"}
        )
    if len(calls) > MAX_CALLS_PER_DISPATCH:
        # Same refusal shape as tool_search's "too many queries".
        return json.dumps(
            {
                "error": (
                    f"too many calls: {len(calls)} > max {MAX_CALLS_PER_DISPATCH}. "
                    "Retry with fewer calls per batch."
                )
            }
        )

    partition = partition_calls(calls)

    local_entries = [
        _run_local(local_dispatch, position, call)
        for position, call in partition.local
    ]

    # The caller's per-entry gate (hooks/approval) runs before anything is
    # sent: a blocked entry becomes a USER_DENIED slot, siblings still run,
    # and a rewrite rebuilds that entry alone before the request is built.
    denied_entries: list[dict[str, Any]] = []
    surviving = list(partition.remote)
    if pre_dispatch is not None and surviving:
        kept = []
        for plan in surviving:
            try:
                block, replacement = _run_pre_dispatch(pre_dispatch, plan)
            except Exception as exc:
                logger.debug("pre_dispatch gate failed open for %s: %s", plan.name, exc)
                block, replacement = None, None
            if block is not None:
                denied_entries.append(
                    {
                        "index": plan.position,
                        "name": plan.name,
                        "error": {"code": "USER_DENIED", "message": str(block)},
                    }
                )
                continue
            if replacement is not None:
                # PlannedCall is frozen: rebuild this entry only. position and
                # name ride along untouched, so every downstream correlation
                # (request slot order, result splice, error slots) is unmoved
                # while the wire body carries the rewritten arguments.
                plan = dataclass_replace(plan, arguments=dict(replacement))
            kept.append(plan)
        surviving = kept

    remote_entries: list[dict[str, Any]] = []
    if surviving:
        remote_entries = _run_remote(
            surviving,
            dispatch_id,
            availability=availability,
            client_factory=client_factory,
        )

    return json.dumps(
        assemble_results(
            len(calls), local_entries, remote_entries, denied_entries, partition.errors
        ),
        ensure_ascii=False,
    )


def _run_pre_dispatch(
    pre_dispatch: PreDispatch, plan
) -> "tuple[Optional[str], Optional[dict[str, Any]]]":
    """Call the gate and normalize its return to (block, replacement).

    The contract is the 2-tuple (:data:`PreDispatch`). A gate that returns a
    bare value instead is read as a block message and nothing else: guessing
    "no block" from a malformed return would fail OPEN on the one seam whose
    job is to stop calls. A replacement that is not a dict is discarded — the
    wire body needs an object, and rewriting is opt-in.
    """
    outcome = pre_dispatch(plan.name, dict(plan.arguments))
    if isinstance(outcome, tuple):
        block = outcome[0] if len(outcome) > 0 else None
        replacement = outcome[1] if len(outcome) > 1 else None
    else:
        block, replacement = outcome, None
    return (
        block if block is None else str(block),
        replacement if isinstance(replacement, dict) else None,
    )


def _run_local(
    local_dispatch: LocalDispatch, position: int, call: Any
) -> dict[str, Any]:
    name = str(call.get("name") or "") if isinstance(call, dict) else ""
    arguments = call.get("arguments") if isinstance(call, dict) else None
    try:
        ok, payload = local_dispatch(name, arguments if isinstance(arguments, dict) else {})
    except Exception as exc:
        # Core dispatch wraps its own errors; reaching here means the injected
        # dispatcher itself blew up. Keep it to this entry.
        return {
            "index": position,
            "name": name,
            "error": {"code": "TOOL_ERROR", "message": str(exc)},
        }
    # JSON-embed either way: structural results beat a quoted blob. This reads
    # the payload's SHAPE, never its meaning — ``ok`` already decided the slot.
    value = _maybe_parse_json(payload)
    if ok:
        return {"index": position, "name": name, "response": value}
    return {"index": position, "name": name, "error": _error_slot(value)}


def _error_slot(payload: Any) -> dict[str, Any]:
    """Wrap a refusal payload as an error slot without losing any of it.

    Every key the dispatcher sent survives — ``tool_error`` takes arbitrary
    extras (``parameters``, ``hint``, ``code=404``, …) and an allow-list would
    silently drop the ones it did not anticipate. ``code``/``message`` are only
    ADDED when absent, so the slot always has the shape the envelope promises.
    """
    if not isinstance(payload, dict):
        return {"code": "TOOL_ERROR", "message": str(payload)}
    slot = dict(payload)
    if "message" not in slot:
        # tool_error keys its text as "error"; error slots key it "message".
        slot["message"] = str(slot.get("error") or "The tool call failed.")
    slot.setdefault("code", "TOOL_ERROR")
    return slot


def _run_remote(
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


def _maybe_parse_json(raw: Any) -> Any:
    """Embed structured tool results structurally; leave other strings alone."""
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                return json.loads(text)
            except ValueError:
                return raw
    return raw
