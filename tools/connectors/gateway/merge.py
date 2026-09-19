"""Pure partitioning, result splicing, and rendering for mixed tool-call batches.

Correlate all remote results by original call position, never wire ``index``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from tools.connectors.gateway.config import session_platform
from tools.connectors.gateway.errors import render_connection_required
from tools.connectors.gateway.names import parse_connector_name

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

    position: int
    name: str
    connector: str
    tool: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Partition:

    local: tuple[tuple[int, Mapping[str, Any]], ...]
    # Gateway request and response slots preserve this order.
    remote: tuple[PlannedCall, ...]
    errors: tuple[dict[str, Any], ...]


def partition_calls(calls: Sequence[Any]) -> Partition:
    """Treat malformed input as per-entry errors so sibling calls still run."""
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
    """Use the shared CONNECTION_REQUIRED shape so links render consistently."""
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
            card=session_platform() == "desktop",
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
    """Correlate results by request slot; missing slots become per-entry errors."""
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
    return [
        _error_entry(plan.position, plan.name, code=code, message=message)
        for plan in planned
    ]


def assemble_results(
    total: int,
    *entry_groups: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Preserve call order and fill unclaimed slots with per-entry errors."""
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
    """Reject strings so malformed calls do not become character entries."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return ()


def _opt_str(value: Any) -> Optional[str]:
    if isinstance(value, str) and value:
        return value
    return None
