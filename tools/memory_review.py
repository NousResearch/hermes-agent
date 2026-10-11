"""Structured review of atomic pending memory writes; no agent/prompt state changes."""
import difflib
import hashlib
import json
import re

from tools import write_approval as wa
from tools.memory_tool import (
    ENTRY_DELIMITER, MemoryStore, apply_memory_pending,
    get_builtin_memory_config, get_builtin_memory_store_flags,
)


class _PreviewStore(MemoryStore):
    _preview = True

    def _write_file(self, path, entries):
        # Application has already validated the exact operations; retain only its result.
        pass

    def _detect_external_drift(self, target, raw):
        parsed = self._parse_entries(raw)
        if raw.strip() and (raw.strip() != ENTRY_DELIMITER.join(parsed)
                           or max(map(len, parsed), default=0) > self._char_limit(target)):
            return "not created (read-only preview)"
        return None


def _store(preview=False):
    from tools.memory_review_config import read_memory_review_config
    cfg = read_memory_review_config()
    mem = get_builtin_memory_config(cfg)
    enabled, user_enabled = get_builtin_memory_store_flags(cfg)
    return (_PreviewStore if preview else MemoryStore)(
        int(mem.get("memory_char_limit", 2200)), int(mem.get("user_char_limit", 1375)),
        memory_enabled=enabled, user_profile_enabled=user_enabled)


def _revision(record, raw):
    return hashlib.sha256((json.dumps(record, sort_keys=True, ensure_ascii=False) + "\0" + raw).encode()).hexdigest()


def _review(record):
    store = _store(preview=True)
    payload = record.get("payload", {})
    target = payload.get("target", "memory")
    path = store._path_for(target)
    before, readable = store._read_raw_checked(path)
    store._expected_raw = before
    result = apply_memory_pending(payload, store) if readable else {"success": False, "error": "Memory file is unreadable."}
    after = ENTRY_DELIMITER.join(store._entries_for(target)) if result.get("success") else before
    # Include all context rather than eliding unchanged memory clauses.
    lines = difflib.unified_diff(before.splitlines(), after.splitlines(),
                                fromfile=f"a/{path.name}", tofile=f"b/{path.name}",
                                n=max(len(before), len(after)), lineterm="")
    diff = "\n".join(lines)
    return {"id": record["id"], "summary": record.get("summary", ""),
            "origin": record.get("origin", "foreground"), "created_at": record.get("created_at", 0),
            "target": target, "action": payload.get("action", ""),
            "operation_count": len(payload.get("operations") or []) if payload.get("action") == "batch" else 1,
            "before": before, "after": after, "diff": diff, "revision": _revision(record, before),
            "can_approve": bool(result.get("success")), "error": result.get("error", "")}


def list_memory_reviews():
    from tools.memory_review_config import read_memory_review_config
    gate = wa._normalize_enabled(get_builtin_memory_config(
        read_memory_review_config()).get("write_approval"))
    return {"write_approval": gate, "batches": [_review(rec) for rec in wa.list_pending(wa.MEMORY)]}


def decide_memory_review(pending_id, decision, revision):
    if not re.fullmatch(r"[0-9a-f]{8}", pending_id) or decision not in {"approve", "reject"}:
        return {"success": False, "error": "Invalid memory review decision."}
    def apply(record):
        review = _review(record)
        if revision != review["revision"]:
            return {"success": False, "error": "Memory changed since review; refresh before deciding."}
        if decision == "approve":
            if not review["can_approve"]:
                return {"success": False, "error": review["error"]}
            store = _store()
            store._expected_raw = review["before"]
            result = apply_memory_pending(record["payload"], store)
            if not result.get("success"):
                return {"success": False, "error": result.get("error", "Approval failed.")}
        return {"success": True, "error": ""}
    try:
        return wa.decide_pending(wa.MEMORY, pending_id, apply)
    except OSError as exc:
        return {"success": False, "error": str(exc)}
