"""Atomic memory batches share UUID selection, scope checks and reviewed-content pinning."""
from __future__ import annotations

from tools.memory_entry_identity import (EntryText, IdentityError, id_fields, inherit_identity,
                                         new_entry, validate_id, validate_scope)


def _append(working, content, scope, target, managed, created_id=None):
    if not content:
        raise IdentityError("content is required.")
    validate_scope(scope)
    if scope is not None and not managed:
        raise IdentityError("Scoped writes require explicit memory identity migration first.")
    existing = next((entry for entry in working if entry == content and getattr(entry, "scope", None) == scope), None)
    if existing is not None:
        return existing
    entry = new_entry(content, target, scope) if managed else content
    if created_id is not None:
        validate_id(created_id)
        if not managed or any(getattr(item, "entry_id", None) == created_id for item in working):
            raise IdentityError("The approved creation UUID is unavailable or already in use.")
        entry = EntryText(content, created_id, target, scope)
    working.append(entry)
    return entry


def _select(working, op, managed):
    from tools.memory_identity_store import _locate
    from tools.memory_tool_store import _find_unique_match, _pinned_index, _stale_entry_message
    if op.get("entry_id") is not None:
        if not managed:
            raise IdentityError("Memory identity metadata is missing; migrate explicitly first.")
        return _locate(working, op["entry_id"], op.get("scope"), op.get("matched_entry"))
    if op.get("scope") is not None:
        raise IdentityError("Scoped replace/remove requires entry_id.")
    if not (old := op.get("old_text") or "").strip():
        raise IdentityError("old_text or entry_id is required.")
    if (pinned := op.get("matched_entry")) is not None:
        index = _pinned_index(working, pinned)
        if index is None:
            raise IdentityError(_stale_entry_message(pinned))
        return index
    index, ambiguous = _find_unique_match(working, old.strip())
    if ambiguous:
        raise IdentityError(f"'{old}' matched multiple distinct entries -- be more specific.")
    if index is None:
        raise IdentityError(f"no entry matched '{old}'.")
    return index


def _apply_op(working, op, target, managed, replay):
    action = op.get("action")
    content = (op.get("content") or op.get("new_text") or "").strip()
    if op.get("created_entry_id") is not None and not replay:
        raise IdentityError("Creation UUIDs are assigned by the pending-write store, not by callers.")
    if action == "add":
        if op.get("entry_id") is not None:
            raise IdentityError("Add creates an entry_id; it cannot accept an existing one.")
        return None, _append(working, content, op.get("scope"), target, managed, op.get("created_entry_id"))
    if action not in {"replace", "remove"}:
        raise IdentityError("unknown action. Use add, replace, or remove.")
    if action == "replace" and not content:
        raise IdentityError("content is required (use action='remove' to delete).")
    index = _select(working, op, managed)
    previous = working[index]
    replacement = inherit_identity(content, previous) if action == "replace" else None
    working[index:index + 1] = [replacement] if replacement is not None else []
    return previous, replacement or previous


def _walk(store, target, entries, operations, replay):
    working, matched, identities = list(entries), [], {}
    for index, op in enumerate(operations, 1):
        try:
            previous, affected = _apply_op(working, op, target, store._identity_enabled[target], replay)
        except IdentityError as exc:
            return store._batch_failure(target, f"Operation {index} ({op.get('action') or 'unknown'}): {exc}")
        matched.append(previous)
        if fields := id_fields(affected):
            identities[index] = fields
    return working, matched, identities


def _state_error(store, target, before, after, limit, count):
    from tools.memory_tool_store import ENTRY_DELIMITER
    if before and not after:
        return store._batch_failure(target, (
            f"Refusing to empty {store._path_for(target).name}: this batch would remove every entry from a "
            "previously non-empty store. Keep at least one entry — merge overlapping entries into a shorter "
            "one instead of removing the last one. To delete the final entry deliberately, use single remove() calls."))
    if (total := len(ENTRY_DELIMITER.join(after))) > limit:
        return store._batch_failure(target, (
            f"After applying all {count} operations, memory would be at {total:,}/{limit:,} chars -- over the "
            "limit. Remove or shorten more entries in the same batch, then retry."))
    return None


def _previous_fields(operations, matched, identities):
    fields = {}
    for action, field in (("replace", "replaced_entries"), ("remove", "removed_entries")):
        previous = {index: entry for index, (op, entry) in enumerate(zip(operations, matched), 1)
                    if op.get("action") == action and entry is not None}
        if previous:
            fields[field] = previous
    if identities:
        fields["entry_ids"] = identities
    return fields


def apply_batch(store, target, operations, *, commit, replay=False):
    from tools.memory_tool_store import _error, _scan_memory_content
    if not isinstance(operations, list) or not operations:
        return _error("operations list is empty or invalid.")
    if any(not isinstance(op, dict) for op in operations):
        return _error("Each memory operation must be an object.")
    for index, op in enumerate(operations, 1):
        content = op.get("content") or op.get("new_text") or ""
        if not isinstance(content, str) or not isinstance(op.get("old_text") or "", str):
            return _error(f"Operation {index}: content and old_text must be strings.")
        if op.get("action") in {"add", "replace"} and (error := _scan_memory_content(content)):
            return _error(f"Operation {index}: {error}")

    def apply(entries, limit):
        result = _walk(store, target, entries, operations, replay)
        if isinstance(result, dict):
            return result
        working, matched, identities = result
        if error := _state_error(store, target, entries, working, limit, len(operations)):
            return error
        if not commit:
            return {"success": True, "matched_entries": matched, "entry_ids": identities}
        return working, f"Applied {len(operations)} operation(s).", _previous_fields(operations, matched, identities)

    return store._mutate(target, apply, skip_drift=not commit)


def pin_batch_payload(store, target, payload, resolved):
    """Keep planned UUIDs when a proposal creates and then edits an entry in the same batch."""
    known = {getattr(entry, "entry_id", None) for entry in store._entries_for(target)}
    pinned = []
    for index, (op, entry) in enumerate(zip(payload["operations"], resolved["matched_entries"]), 1):
        item = dict(op)
        if entry is not None:
            item.update(matched_entry=str(entry), **id_fields(entry))
        fields = resolved.get("entry_ids", {}).get(index, {})
        if op.get("action") == "add" and fields.get("entry_id") and fields["entry_id"] not in known:
            item["created_entry_id"] = fields["entry_id"]
        pinned.append(item)
    payload["operations"] = pinned
