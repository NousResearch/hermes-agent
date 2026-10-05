"""Management extensions to the built-in memory tool, through its existing approval gates."""
from __future__ import annotations

from tools.memory_entry_identity import IdentityError, validate_content, validate_scope
from tools.memory_identity_store import edit_entry_id, list_entry_ids, resolve_entry_id


def _validate(action, content, old_text, operations, entry_id, scope, cursor, limit):
    validate_scope(scope)
    if operations:
        raise IdentityError("Top-level management fields cannot be combined with operations; put entry_id/scope in each op.")
    if action == "list":
        if entry_id is not None or content is not None or old_text is not None:
            raise IdentityError("List accepts target, scope, cursor and limit only.")
        return
    if cursor is not None or limit is not None:
        raise IdentityError("cursor and limit are only used with action='list'.")
    if action not in {"add", "replace", "remove"}:
        raise IdentityError("Use add, replace, remove or list.")
    if action == "add" and entry_id is not None:
        raise IdentityError("Add creates an entry_id; it cannot accept an existing one.")
    if action in {"replace", "remove"} and (entry_id is None or old_text is not None):
        raise IdentityError("UUID replace/remove needs entry_id and exact scope; omit old_text.")
    if action in {"add", "replace"}:
        if not isinstance(content, str) or not content.strip():
            raise IdentityError("content must be a nonempty string.")
        validate_content(content.strip())
        from tools.memory_tool_store import _scan_memory_content
        if error := _scan_memory_content(content):
            raise IdentityError(error)


def manage_memory(store, action, target, content, old_text, operations, *, entry_id=None, scope=None,
                  cursor=None, limit=None):
    from tools.memory_tool import _applied, _apply_write_gate, _background_delete_gate, _gate_or_stage
    from tools.registry import tool_error
    try:
        _validate(action, content, old_text, operations, entry_id, scope, cursor, limit)
    except IdentityError as exc:
        return "rejected", tool_error(str(exc), success=False)
    if action == "list":
        return _applied(list_entry_ids(store, target, scope=scope, cursor=cursor, limit=limit))
    if action == "add":
        ready = list_entry_ids(store, target, limit=1)
        if not ready.get("success"):
            return _applied(ready)
        gate = _apply_write_gate(store, action, target, content, None, scope=scope)
        return ("rejected", gate) if gate is not None else _applied(store.add(target, content, scope=scope))
    resolved = resolve_entry_id(store, target, entry_id, scope)
    if not resolved.get("success"):
        return _applied(resolved)
    identity = {"entry_id": entry_id, "scope": scope, "matched_entry": resolved["matched_entry"]}
    denied = _background_delete_gate(store, action, None, target, content, None, **identity)
    if denied is not None:
        return "rejected", denied
    payload = {"action": action, "target": target, "content": content, **identity}
    gate = _gate_or_stage(store, f"{action} entry {entry_id} in {target}",
                          f"Reviewed entry: {resolved['matched_entry']}\nWhole entry becomes: {content or '(removed)'}", payload)
    return ("rejected", gate) if gate is not None else _applied(
        edit_entry_id(store, action, target, entry_id, content=content, scope=scope, expected=resolved["matched_entry"]))
