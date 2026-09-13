"""Pure partition / splice / render for mixed tool_call batches.

One ``tool_call`` invocation may mix local deferred tools and
``connectors__<connector>__<tool>`` entries. This module owns the pure logic
around that: partitioning the original ``calls[]`` array, splicing remote
execute results back into place, and rendering the caller-facing result
entries.

Contract:

- PURE — no I/O, no imports above the stdlib + sibling leaf modules, and no
  exceptions on any input shape. Malformed input becomes per-entry errors.
- Position in the ORIGINAL ``calls[]`` array is the only correlation key.
  The wire ``index`` field is never read (execute is 0-based, search is
  1-based; trusting either is a known trap).
- A short or over-long remote response never raises: missing slots are
  filled with ``PROVIDER_ERROR`` entries, surplus entries are dropped.
- Counts are recomputed over the merged array — the gateway's counts cover
  only its slice.

Caller-facing entry shape (mirrors the gateway's per-tool result discipline;
exactly one of ``response`` / ``error`` per entry):

    {"index": <position>, "name": <original name>, "response": <data>}
    {"index": <position>, "name": <original name>, "error": {code, message, ...}}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from tools.tool_gateway.errors import render_connection_required
from tools.tool_gateway.names import parse_connector_name

__all__ = [
    "Partition",
    "PlannedCall",
    "assemble_results",
    "fill_remote_failure",
    "partition_calls",
    "render_remote_entry",
    "splice_remote_results",
]


@dataclass(frozen=True)
class PlannedCall:
    """A connector-bound entry, pinned to its position in the original array."""

    position: int
    name: str
    connector: str
    tool: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Partition:
    """The original ``calls[]`` split by destination, positions preserved."""

    # (position, call) for entries hermes dispatches locally.
    local: tuple[tuple[int, Mapping[str, Any]], ...]
    # Connector-bound entries, original order preserved: remote[i] becomes
    # the gateway request's tools[i], which is also how responses correlate.
    remote: tuple[PlannedCall, ...]
    # Pre-rendered error entries for entries that route nowhere
    # (malformed connector names). Siblings still run.
    errors: tuple[dict[str, Any], ...]


def partition_calls(calls: Sequence[Any]) -> Partition:
    """Split a ``calls[]`` array by destination. Total on any input.

    An entry routes to the gateway when its name parses as a connector
    name. A name that claims the ``connectors__`` prefix but does not parse
    becomes that entry's error; everything else is local. A top-level value
    that is not a sequence partitions as empty.
    """
    local: list[tuple[int, Mapping[str, Any]]] = []
    remote: list[PlannedCall] = []
    errors: list[dict[str, Any]] = []
    for position, call in enumerate(_as_sequence(calls)):
        name = call.get("name") if isinstance(call, Mapping) else None
        parsed = parse_connector_name(name)
        if parsed is not None:
            arguments = call.get("arguments")
            remote.append(
                PlannedCall(
                    position=position,
                    name=parsed.raw,
                    connector=parsed.connector,
                    tool=parsed.tool,
                    arguments=dict(arguments) if isinstance(arguments, Mapping) else {},
                )
            )
            continue
        if isinstance(name, str) and name.startswith("connectors__"):
            errors.append(
                _error_entry(
                    position,
                    name,
                    code="TOOL_NOT_FOUND",
                    message=(
                        "Malformed connector tool name; expected "
                        "connectors__<connector>__<tool>."
                    ),
                )
            )
            continue
        local.append((position, call if isinstance(call, Mapping) else {}))
    return Partition(local=tuple(local), remote=tuple(remote), errors=tuple(errors))


def render_remote_entry(planned: PlannedCall, remote: Mapping[str, Any]) -> dict[str, Any]:
    """Render one gateway result (plain dict) into the caller-facing entry.

    ``remote`` is the client-layer dict for this slot: ``{"data": ...,
    "error": None | {code, message, connector, connect_url, hint}}``.
    CONNECTION_REQUIRED goes through the shared single-shape producer so the
    connect link renders identically everywhere.
    """
    error = remote.get("error") if isinstance(remote, Mapping) else None
    if not isinstance(error, Mapping):
        data = remote.get("data") if isinstance(remote, Mapping) else None
        return {"index": planned.position, "name": planned.name, "response": data}

    code = str(error.get("code") or "PROVIDER_ERROR")
    message = str(error.get("message") or "The gateway reported an error.")
    if code == "CONNECTION_REQUIRED":
        payload = render_connection_required(
            connector=_opt_str(error.get("connector")) or planned.connector,
            message=message,
            connect_url=_opt_str(error.get("connect_url")),
            hint=_opt_str(error.get("hint")),
        )
    else:
        payload = {"code": code, "message": message}
        connector = _opt_str(error.get("connector"))
        if connector:
            payload["connector"] = connector
        hint = _opt_str(error.get("hint"))
        if hint:
            payload["hint"] = hint
    return {"index": planned.position, "name": planned.name, "error": payload}


def splice_remote_results(
    planned: Sequence[PlannedCall],
    remote_results: Optional[Sequence[Any]],
) -> list[dict[str, Any]]:
    """Map gateway results back onto planned positions, by array slot only.

    ``remote_results[i]`` corresponds to ``planned[i]`` — the request was
    built from ``planned`` in order. Missing slots (short response, or no
    response at all) fill with ``PROVIDER_ERROR``; surplus slots have
    nothing to correlate to and are dropped. A top-level value that is not
    a sequence counts as no response at all.
    """
    results = _as_sequence(remote_results)
    entries: list[dict[str, Any]] = []
    for slot, plan in enumerate(_as_sequence(planned)):
        if slot < len(results) and isinstance(results[slot], Mapping):
            entries.append(render_remote_entry(plan, results[slot]))
        else:
            entries.append(
                _error_entry(
                    plan.position,
                    plan.name,
                    code="PROVIDER_ERROR",
                    message="The gateway returned no result for this call.",
                )
            )
    return entries


def fill_remote_failure(
    planned: Sequence[PlannedCall],
    message: str,
    *,
    code: str = "PROVIDER_ERROR",
) -> list[dict[str, Any]]:
    """Render the same error into every planned slot.

    For request-level failures (the HTTP envelope path): every connector
    entry in the batch gets the error, local siblings are untouched.
    """
    return [
        _error_entry(plan.position, plan.name, code=code, message=message)
        for plan in planned
    ]


def assemble_results(
    total: int,
    *entry_groups: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge rendered entries back into original order and recompute counts.

    ``entry_groups`` are any number of entry lists (local, remote, partition
    errors), each entry carrying its original position in ``index``. Slots
    nothing claimed — a bug upstream, but this function is total — fill with
    ``PROVIDER_ERROR``; duplicate claims keep the first and drop the rest.
    """
    try:
        slot_count = max(0, int(total))
    except (TypeError, ValueError):
        slot_count = 0
    slots: list[Optional[dict[str, Any]]] = [None] * slot_count
    for group in entry_groups:
        for entry in _as_sequence(group):
            if not isinstance(entry, Mapping):
                continue
            index = entry.get("index")
            if isinstance(index, int) and 0 <= index < len(slots) and slots[index] is None:
                slots[index] = dict(entry)
    merged: list[dict[str, Any]] = []
    for position, entry in enumerate(slots):
        if entry is None:
            entry = _error_entry(
                position,
                "",
                code="PROVIDER_ERROR",
                message="No result was produced for this call.",
            )
        merged.append(entry)
    error_count = sum(1 for entry in merged if "error" in entry)
    return {
        "results": merged,
        "success_count": len(merged) - error_count,
        "error_count": error_count,
        "total_count": len(merged),
    }


def _error_entry(position: int, name: str, *, code: str, message: str) -> dict[str, Any]:
    return {
        "index": position,
        "name": name,
        "error": {"code": code, "message": message},
    }


def _as_sequence(value: Any) -> Sequence[Any]:
    """Normalize a top-level input to a sequence; garbage becomes empty.

    str/bytes are excluded — iterating a stray string as a calls array
    would fabricate one entry per character.
    """
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return ()


def _opt_str(value: Any) -> Optional[str]:
    if isinstance(value, str) and value:
        return value
    return None
