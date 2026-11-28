"""Public, profile-scoped API for plugins that review staged memory and skill writes.

Call it within the host's authorized request scope. These functions never accept
an arbitrary profile or home path and do not replace the host's auth checks.
"""
from __future__ import annotations


from typing import Any

from tools import write_approval as wa

_KINDS = (wa.MEMORY, wa.SKILLS)
_ACTIONS = frozenset({"add", "replace", "remove", "patch", "create", "edit", "delete", "remove_file", "write_file", "batch"})


def list_writes(kind: str = "all", *, page: int = 1, page_size: int = 25) -> dict[str, Any]:
    """List pending writes as metadata only, with per-kind counts and both staging gates.

    ``needs_review`` counts writes whose approval stopped partway (they are not in ``items`` or
    ``counts``). Someone has to check those files by hand and clear them with
    ``/memory reject <id>`` or ``/skills reject <id>``.
    """
    if kind not in ("all", *_KINDS):
        raise ValueError("unknown write-review kind")
    page, page_size = max(1, page), max(1, min(100, page_size))
    for k in _KINDS:
        wa._pending_dir(k)  # do not silently report an invalid directory as an empty store
    by_kind = {k: [_metadata(k, rec) for rec in wa.list_pending(k)] for k in _KINDS}
    counts = {k: len(items) for k, items in by_kind.items()}
    counts["total"] = sum(counts.values())
    needs_review = {k: len(wa.list_needs_review(k)) for k in _KINDS}
    needs_review["total"] = sum(needs_review.values())
    selected = [row for k, items in by_kind.items() if kind in (k, "all") for row in items]
    selected.sort(key=lambda row: (row["created_at"] or 0, row["kind"], row["id"]))
    return {"kind": kind, "scope": wa._profile_identity(), "page": page, "page_size": page_size,
            "total": len(selected), "counts": counts, "needs_review": needs_review, "gates": {k: wa.write_approval_enabled(k) for k in _KINDS},
            "items": selected[(page - 1) * page_size:page * page_size]}


def _metadata(kind: str, record: dict[str, Any]) -> dict[str, Any]:
    action = record.get("action")
    action = action if isinstance(action, str) and action in _ACTIONS else "write"
    batch = action == "batch"
    payload = record.get("payload")
    operations = payload.get("operations") if isinstance(payload, dict) else None
    return {"kind": kind, "id": record["id"], "action": action, "created_at": wa._staged_time(record),
            "origin": "background_review" if record.get("origin") == "background_review" else "foreground",
            "summary": f"{'Memory' if kind == wa.MEMORY else 'Skill'} {action}",
            "is_batch": batch, "op_count": len(operations) if batch and isinstance(operations, list) else 1}


def preview_write(kind: str, pending_id: str) -> dict[str, Any]:
    """Preview one pending write. Pass the returned ``expect`` to :func:`decide_write`.

    ``created_at`` is when the write was staged (epoch seconds, a float), or None when the record
    has no usable time or isn't pending any more.
    """
    wa._pending_dir(kind)
    result = wa.preview_pending(kind, pending_id)
    if isinstance(result.get("expect"), dict):
        result["expect"] = {**result["expect"], "review_kind": kind, "review_id": pending_id}
    return result


def decide_write(kind: str, pending_id: str, decision: str, *, expect: dict[str, Any] | None) -> dict[str, Any]:
    """Approve or reject a write the caller has previewed, passing that preview's ``expect``
    unchanged, and return the resolver's outcome.

    Don't log the arguments or a preview: they can contain memory text. The resolver checks the
    record and its targets again under its locks. Once an outcome is recorded, deciding the same id
    again returns ``already_resolved`` with ``prior`` and applies nothing.
    """
    wa._pending_dir(kind)
    if decision not in {"approve", "reject"}:
        raise ValueError("unknown write-review decision")
    if not isinstance(expect, dict) or expect.get("review_kind") != kind or expect.get("review_id") != pending_id:
        return {"id": pending_id, "subsystem": kind, "status": "conflict", "reason": "Preview this write first."}
    memory_store = None
    if decision == "approve" and kind == wa.MEMORY:
        from tools.memory_tool import load_on_disk_store

        try:
            memory_store = load_on_disk_store()
        except (OSError, ValueError) as exc:
            return {"id": pending_id, "subsystem": kind, "status": "conflict",
                    "reason": f"the memory store could not be loaded: {exc}"}
    result = wa.resolve_pending(kind, pending_id, decision, expect=expect, memory_store=memory_store)
    return {k: v for k, v in result.items() if k != "result"}
