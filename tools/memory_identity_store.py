"""Locked UUID management and explicit backup-first migration of built-in memory."""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
from uuid import uuid4

from tools.memory_entry_identity import (IdentityError, enabled, encode_entries, id_fields,
                                         inherit_identity, new_entry, parse_entries,
                                         validate_id, validate_scope)


def _require_management(store, target):
    if not store._identity_enabled[target]:
        raise IdentityError("Memory identity metadata is missing. Run `hermes memory migrate-identities` explicitly first.")


def _locate(entries, entry_id, scope, expected):
    validate_id(entry_id)
    validate_scope(scope)
    matches = [i for i, entry in enumerate(entries) if getattr(entry, "entry_id", None) == entry_id]
    if len(matches) != 1:
        raise IdentityError("Entry UUID not found or ambiguous in this target; nothing was changed.")
    index = matches[0]
    if entries[index].scope != scope:
        raise IdentityError("Entry scope does not match the caller's exact scope; nothing was changed.")
    if expected is not None and entries[index] != expected:
        raise IdentityError("Entry changed since it was reviewed; recreate the proposal against its current content.")
    return index


def resolve_entry_id(store, target, entry_id, scope=None, *, expected=None):
    def resolve(entries, _limit):
        _require_management(store, target)
        entry = entries[_locate(entries, entry_id, scope, expected)]
        return {"success": True, "matched_entry": str(entry), **id_fields(entry)}
    return store._mutate(target, resolve, skip_drift=True)


def edit_entry_id(store, action, target, entry_id, *, content=None, scope=None, expected=None):
    from tools.memory_tool_store import ENTRY_DELIMITER, _error, _scan_memory_content
    if action not in {"replace", "remove"}:
        return _error("UUID mutation supports replace or remove.")
    if action == "replace":
        if not isinstance(content, str) or not content.strip():
            return _error("Replacement content must be a nonempty string.")
        content = content.strip()
        if error := _scan_memory_content(content):
            return _error(error)

    def apply(entries, limit):
        _require_management(store, target)
        index = _locate(entries, entry_id, scope, expected)
        previous = entries[index]
        replacement = [] if action == "remove" else [inherit_identity(content, previous)]
        updated = entries[:index] + replacement + entries[index + 1:]
        if action == "replace" and len(ENTRY_DELIMITER.join(updated)) > limit:
            return _error("Replacement exceeds this target's character limit; nothing was changed.")
        field = "removed_entry" if action == "remove" else "replaced_entry"
        return updated, f"Entry {action}d by UUID.", {field: str(previous), **id_fields(previous)}
    # Managed parsing checks exact metadata/content round-tripping itself. A lower current
    # character cap must not prevent a UUID deletion from reducing an already-oversized file.
    return store._mutate(target, apply, skip_drift=True)


def _page_start(cursor, revision):
    if cursor is None:
        return 0
    if not isinstance(cursor, str) or len(cursor) > 64 or not re.fullmatch(r"[a-f0-9]{16}:[0-9]+", cursor):
        raise IdentityError("Invalid memory enumeration cursor.")
    prefix, offset = cursor.split(":")
    if prefix != revision:
        raise IdentityError("Memory or enumeration scope changed; restart enumeration instead of skipping entries.")
    return int(offset)


def list_entry_ids(store, target, *, scope=None, cursor=None, limit=None):
    from tools.memory_tool_store import _error
    limit = 50 if limit is None else limit
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        return _error("limit must be an integer between 1 and 100.")
    try:
        validate_scope(scope)
    except IdentityError as exc:
        return _error(str(exc))

    def enumerate_entries(entries, _limit):
        _require_management(store, target)
        revision = hashlib.sha256((encode_entries(entries, identity_enabled=True, target=target)
                                   + json.dumps(scope)).encode()).hexdigest()[:16]
        selected = [entry for entry in entries if scope is None or entry.scope == scope]
        start = _page_start(cursor, revision)
        if start > len(selected):
            raise IdentityError("Memory enumeration cursor is out of range.")
        page = selected[start:start + limit]
        end = start + len(page)
        return {"success": True, "done": True, "target": target, "total": len(selected),
                "entries": [{**id_fields(entry), "target": target, "content": str(entry)[:512],
                             "content_truncated": len(entry) > 512} for entry in page],
                "truncated": end < len(selected),
                "next_cursor": f"{revision}:{end}" if end < len(selected) else None}
    return store._mutate(target, enumerate_entries, skip_drift=True)


def _backup_bytes(path, content):
    from hermes_constants import mkdir_under_hermes_home
    directory = path.parent / ".identity-backups"
    mkdir_under_hermes_home(directory)
    backup = directory / f"{path.name}.{uuid4()}.bak"
    fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    return backup


def migrate_target(store, target, *, commit=False):
    from tools.memory_tool_store import ENTRY_DELIMITER, _error
    from tools.memory_tool import _memory_target_error
    if target_error := _memory_target_error(store, target):
        return target_error
    path = store._path_for(target)
    backup = None
    with store._file_lock(path):
        try:
            data = path.read_bytes() if path.exists() else b""
            with io.TextIOWrapper(io.BytesIO(data), encoding="utf-8-sig") as stream:
                raw = stream.read()
            entries = parse_entries(raw, target=target)
            if enabled(raw):
                return {"success": True, "already_migrated": True, "target": target, "entry_count": len(entries)}
            if raw.strip() != ENTRY_DELIMITER.join(entries):
                raise IdentityError("Legacy memory does not round-trip; reconcile its formatting before migration.")
            if not commit:
                return {"success": True, "preview": True, "target": target, "entry_count": len(entries)}
            backup = _backup_bytes(path, data)
            identified = [new_entry(entry, target) for entry in entries]
            store._write_file(path, identified, identity_enabled=True)
            return {"success": True, "target": target, "entry_count": len(identified), "backup": str(backup)}
        except (IdentityError, OSError, UnicodeError, RuntimeError) as exc:
            return _error(str(exc), target=target, **({"backup": str(backup)} if backup else {}))
