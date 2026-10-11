"""Explicit recovery of pinned memory proposals after a concurrent edit."""

from copy import deepcopy
import json

from tools import write_approval as wa
from tools.memory_tool import _memory_target_error, _pin_matched_entries, destructive_ops


def refresh_memory_pending(record: dict, store) -> dict:
    """Stage a new review against current memory; never apply the proposed edit.

    Reuse the locked resolver for every operation, including batch dependencies.
    Missing/ambiguous anchors and legacy unpinned proposals remain unchanged.
    """
    payload = deepcopy(record.get("payload") or {})
    error = _memory_target_error(store, payload.get("target", "memory"))
    if error is not None:
        return error
    ops = destructive_ops(payload)
    if not ops or any(not op.get("matched_entry") for op in ops):
        return {"success": False, "error": "Only pinned replace/remove proposals can be refreshed; reject and recreate this write."}
    previous = [op.pop("matched_entry") for op in ops]
    error = _pin_matched_entries(store, payload)
    if error is not None:
        return json.loads(error)
    payload["refresh_previous_entries"] = previous
    fresh = wa.stage_write(wa.MEMORY, payload, summary=f"Refreshed {record['id']}: {record.get('summary', '')}",
                           origin=record.get("origin", "foreground"))
    if wa.get_pending(wa.MEMORY, fresh["id"]) is None:
        return {"success": False, "error": "Could not persist the refreshed proposal; the original remains pending."}
    wa.discard_pending(wa.MEMORY, record["id"])
    return {"success": True, "record": fresh}


def refreshed_memory_preview(payload: dict) -> list[str]:
    """The reviewed base, current target and proposed result for each refreshed op."""
    previous = payload.get("refresh_previous_entries") or []
    return [f"Previously reviewed: {old}\nCurrent entry: {op['matched_entry']}\n"
            f"Proposed result: {op.get('content') or op.get('new_text') or '(remove entry)'}"
            for old, op in zip(previous, destructive_ops(payload))]
