"""Todo tool: in-memory, revisioned task list for multi-step work. State lives on the
AIAgent (one per session), is re-injected after context compression, and every write bumps
a monotonic revision so UI clients can reject stale updates. One ``todo_list`` tool: pass
``todos`` to write, omit to read; every call returns the full list. No system-prompt mutation."""

import json
from typing import Any, Dict, List, Optional

VALID_STATUSES = {"pending", "in_progress", "completed", "cancelled"}
# The list is re-read after every compression (format_for_injection), so unbounded
# content/count would defeat the compression it rides through. Caps apply equally to
# model-authored items and caller-replayed API history.
MAX_TODO_CONTENT_CHARS = 4000
MAX_TODO_ITEMS = 256
_AUTO_ID_SENTINEL = "__needs_auto__"
_AUTO_ID_PREFIX = "auto_"
# Max single todo tool-result payload accepted during history hydration, so a forged
# oversized result is dropped before parsing (AIAgent._hydrate_todo_store).
MAX_TODO_RESULT_CHARS = 512_000
_TRUNCATION_MARKER = "… [truncated]"
# Persisted as ordinary message content; ContextCompressor keys on this stable header to
# tell the synthetic post-compaction row from a real user message.
TODO_INJECTION_HEADER = "[Your active task list was preserved across context compression]"
_STATUS_MARKERS = {"completed": "[x]", "in_progress": "[>]", "pending": "[ ]", "cancelled": "[~]"}
_ACTIVE_STATUSES = {"pending", "in_progress"}


class TodoStore:
    """In-memory todo list, one per AIAgent. List position is priority; items are
    ``{id, content, status, parent?}`` — ``parent`` nests a subtask."""

    def __init__(self):
        self._items: List[Dict[str, str]] = []
        self._revision = 0

    def _fresh_items(self, todos: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Validate, dedupe and order a whole new list (replace / restore)."""
        # Resolve auto-ids BEFORE dedupe: two no-id items would otherwise share the
        # ``__needs_auto__`` sentinel and collapse to one position in ``_dedupe_by_id``.
        # Seed ``used`` with the existing store so a follow-up ``write`` doesn't
        # re-mint ``auto_1`` over a still-live id.
        existing_ids = {item["id"] for item in self._items}
        validated = [self._validate(t) for t in todos]
        with_ids = self._resolve_auto_ids(validated, existing_ids)
        deduped = self._dedupe_by_id(with_ids)
        return self._normalize_order(deduped)

    def write(self, todos: List[Dict[str, Any]], merge: bool = False) -> List[Dict[str, str]]:
        """Replace the list (default) or merge by id; returns the full list after writing."""
        before = self.read()
        if merge:
            self._merge(todos)
        else:
            self._items = self._fresh_items(todos)
        del self._items[MAX_TODO_ITEMS:]  # keep the priority head; replays can't grow unbounded
        self._sanitize_parents(self._items)
        if self._items != before:
            self._revision += 1
        return self.read()

    def _merge(self, todos: List[Dict[str, Any]]) -> None:
        """Update existing items only in the fields provided; append new ones (validated).

        No-id dict items get assigned unique ``auto_N`` ids (see ``_fresh_items`` for
        the same contract in replace mode) so a merge that introduces new items from a
        model that forgot to emit ids still works end-to-end. Non-dict placeholders
        ("?" non-dict fallback) are dropped — they're unmergeable garbage.
        """
        existing = {item["id"]: item for item in self._items}
        # First pass: validate + resolve auto-ids so a batch of no-id dict items gets
        # distinct ``auto_N`` ids (seeded above any existing store entries).
        raw = [self._validate(t) for t in todos]
        resolved = self._resolve_auto_ids(raw, set(existing.keys()))
        # Second pass: explicit-id loop (preserves the original "only overwrite fields
        # the caller actually provided" semantic), plus auto-id appends.
        for src, validated in zip(todos, resolved):
            item_id = validated["id"]
            if item_id == "?":
                continue  # non-dict placeholder; drop
            cur = existing.get(item_id)
            if cur is None:
                # New item (explicit id not in store, OR auto-id assigned above).
                existing[item_id] = validated
                self._items.append(validated)
                continue
            # Existing item: only overwrite fields the caller actually sent on ``src``.
            # ``src`` may be a non-dict (which produced the "?" placeholder and is
            # already skipped above); only dict sources can carry mergeable fields.
            if not isinstance(src, dict):
                continue
            if src.get("content"):
                cur["content"] = self._cap_content(str(src["content"]).strip())
            if src.get("status") and str(src["status"]).strip().lower() in VALID_STATUSES:
                cur["status"] = str(src["status"]).strip().lower()
            if "parent" in src:
                parent = str(src["parent"] or "").strip()
                if parent:
                    cur["parent"] = parent
                else:
                    cur.pop("parent", None)
        # Rebuild preserving original order for existing items (first occurrence wins).
        rebuilt = {item["id"]: existing.get(item["id"], item) for item in self._items}
        self._items = self._normalize_order(list(rebuilt.values()))

    def read(self) -> List[Dict[str, str]]:
        return [item.copy() for item in self._items]

    def has_items(self) -> bool:
        return bool(self._items)

    def snapshot(self) -> Dict[str, Any]:
        """Full state clients can reconcile atomically."""
        return {"todos": self.read(), "revision": self._revision}

    def restore(self, todos: List[Dict[str, Any]], *, revision: Any = 0) -> List[Dict[str, str]]:
        """Restore a trusted snapshot without manufacturing a new revision."""
        self._items = self._fresh_items(todos)[:MAX_TODO_ITEMS]
        try:
            self._revision = max(0, int(revision or 0))
        except (TypeError, ValueError):
            self._revision = 0
        return self.read()

    def format_for_injection(self) -> Optional[str]:
        """Render the list for post-compression injection, or None if nothing active. Only
        pending/in_progress items are injected — finished ones make the model re-do work after
        compression. A parent is kept (with its real status marker) when any descendant is
        active so subtasks keep context."""
        if not self._items:
            return None
        children: Dict[str, List[Dict[str, str]]] = {}
        for item in self._items:
            if item.get("parent"):
                children.setdefault(item["parent"], []).append(item)

        def render(item: Dict[str, str], depth: int, out: List[str]) -> bool:
            kid_lines: List[str] = []
            has_active_kid = False
            for kid in children.get(item["id"], []):
                has_active_kid |= render(kid, depth + 1, kid_lines)
            keep = item["status"] in _ACTIVE_STATUSES or has_active_kid
            if keep:
                marker = _STATUS_MARKERS.get(item["status"], "[?]")
                out.append(f"{'  ' * depth}- {marker} {item['id']}. "
                           f"{item['content']} ({item['status']})")
                out.extend(kid_lines)
            return keep

        lines = [TODO_INJECTION_HEADER]
        for item in self._items:
            if not item.get("parent"):
                render(item, 0, lines)
        return "\n".join(lines) if len(lines) > 1 else None

    @staticmethod
    def _cap_content(content: str) -> str:
        """Truncate to MAX_TODO_CONTENT_CHARS keeping the head (the actionable part) + marker."""
        if len(content) > MAX_TODO_CONTENT_CHARS:
            return content[:MAX_TODO_CONTENT_CHARS - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER
        return content

    @staticmethod
    def _validate(item: Dict[str, Any]) -> Dict[str, str]:
        """Normalize one item to ``{id, content, status, parent?}`` (placeholders when missing)."""
        if not isinstance(item, dict):
            return {"id": "?", "content": "(invalid item)", "status": "pending"}
        raw_id = str(item.get("id", "")).strip()
        # A dict item with no id is a model emission bug (often: ``todos`` was a
        # bare dict that ``coerce_tool_args`` wrapped as a one-element list).
        # Mark it with ``__needs_auto__`` so ``_resolve_auto_ids`` can hand it a
        # unique ``auto_N`` once the whole batch is in scope. A non-dict item
        # still falls back to ``"?"`` (existing contract).
        item_id = raw_id or _AUTO_ID_SENTINEL
        content = str(item.get("content", "")).strip()
        status = str(item.get("status", "pending")).strip().lower()
        result = {"id": item_id,
                  "content": TodoStore._cap_content(content) if content else "(no description)",
                  "status": status if status in VALID_STATUSES else "pending"}
        parent = str(item.get("parent") or "").strip()
        if parent and parent != item_id:
            result["parent"] = parent
        return result

    @staticmethod
    def _resolve_auto_ids(items: List[Dict[str, str]],
                          existing_ids: Optional[set] = None,
                          ) -> List[Dict[str, str]]:
        """Replace ``__needs_auto__`` placeholders with unique ``auto_N`` ids.

        Numbering starts above any existing ``auto_N`` (in the new batch or
        in ``existing_ids`` from the store) so re-runs of the same batch get
        stable, non-colliding ids; explicit caller-supplied ids always win.
        """
        existing_ids = existing_ids or set()
        used: set = {item["id"] for item in items
                     if item["id"] != _AUTO_ID_SENTINEL} | existing_ids
        next_n = 1
        for existing in used:
            if existing.startswith(_AUTO_ID_PREFIX):
                try:
                    next_n = max(next_n,
                                  int(existing[len(_AUTO_ID_PREFIX):]) + 1)
                except ValueError:
                    pass
        for item in items:
            if item["id"] == _AUTO_ID_SENTINEL:
                while f"{_AUTO_ID_PREFIX}{next_n}" in used:
                    next_n += 1
                item["id"] = f"{_AUTO_ID_PREFIX}{next_n}"
                used.add(item["id"])
                next_n += 1
        return items

    @staticmethod
    def _sanitize_parents(items: List[Dict[str, str]]) -> None:
        """Drop dangling parent refs and break cycles in place (such items become roots)."""
        by_id = {item["id"]: item for item in items}
        for item in items:
            if item.get("parent") and item["parent"] not in by_id:
                item.pop("parent", None)
        for item in items:
            seen, node = {item["id"]}, item
            while node.get("parent"):
                if node["parent"] in seen:
                    item.pop("parent", None)
                    break
                seen.add(node["parent"])
                node = by_id[node["parent"]]

    @staticmethod
    def _dedupe_by_id(todos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collapse duplicate ids, keeping the last occurrence in its position.

        Empty/placeholder ids (the ``"?"`` non-dict fallback or the
        ``__needs_auto__`` sentinel) get a per-position key so they are
        NEVER collapsed — each is its own observed problem and should be
        reported as-is. ``_resolve_auto_ids`` runs BEFORE dedupe in
        ``_fresh_items``, so a sentinel never reaches this stage in
        practice; the sentinel guard is defense in depth for direct callers.
        """
        # These strings indicate "no real id" and must never collide.
        _PLACEHOLDER_IDS = frozenset({"?", "_AUTO_ID_SENTINEL"})  # noqa: F841 (sentinel use)
        last_index: Dict[str, int] = {}
        for i, item in enumerate(todos):
            if not isinstance(item, dict):
                key = f"__invalid_{i}"
            else:
                raw_id = str(item.get("id", "")).strip()
                # Empty OR a known placeholder ("?" non-dict, "__needs_auto__" pre-resolve)
                # → per-position key so duplicates don't collapse.
                if not raw_id or raw_id in ("?", "__needs_auto__"):
                    key = f"__placeholder_{i}"
                else:
                    key = raw_id
            last_index[key] = i
        return [todos[i] for i in sorted(last_index.values())]

    @staticmethod
    def _normalize_order(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Lift the in_progress step ahead of any earlier pending placeholder. Nested lists
        keep authored order — reordering would tear a subtask from its siblings."""
        statuses = [item["status"] for item in items]
        if any(item.get("parent") for item in items) or "in_progress" not in statuses:
            return items
        active_index = statuses.index("in_progress")
        if "pending" not in statuses[:active_index]:
            return items
        normalized = items.copy()
        normalized.insert(statuses.index("pending"), normalized.pop(active_index))
        return normalized


def todo_tool(todos: Optional[List[Dict[str, Any]]] = None, merge: bool = False,
              store: Optional[TodoStore] = None) -> str:
    """Write ``todos`` (replace, or ``merge`` by id) or read when None -> list + summary JSON."""
    if store is None:
        return tool_error("TodoStore not initialized")
    if todos is None:
        items = store.read()
    else:
        if isinstance(todos, str):  # LLMs sometimes send a JSON string instead of a list
            try:
                todos = json.loads(todos)
            except (json.JSONDecodeError, TypeError):
                return tool_error("todos must be a list of objects, got unparseable string")
        if not isinstance(todos, list):
            return tool_error(f"todos must be a list, got {type(todos).__name__}")
        items = store.write(todos, merge)
    summary = {"total": len(items)}
    for status in ("pending", "in_progress", "completed", "cancelled"):
        summary[status] = sum(1 for i in items if i["status"] == status)
    return json.dumps({"todos": items, "revision": store.snapshot()["revision"],
                       "summary": summary}, ensure_ascii=False)


def check_todo_requirements() -> bool:
    """Todo tool has no external requirements -- always available."""
    return True


# Behavioral guidance is baked into the (static, cached) description; item shape and merge
# semantics live ONLY in the parameter schema.
TODO_SCHEMA = {
    "name": "todo_list",
    "description": (
        # See #95681.
        "Track a task list for multi-step work (3+ steps). Use for complex tasks "
        "with 3+ steps or when the user provides multiple tasks. "
        "For 'all N items' tasks, enumerate every instance as its own checklist "
        "item so none are silently dropped. "
        "Call with no parameters to read the current list.\n"
        "List order is priority. Only ONE item in_progress at a time. "
        "Break large phases into subtasks via parent. "
        "Mark an item completed only after the work is verified done, never "
        "based on intent. If something fails, cancel it and add a revised "
        "item. Always returns the full current list."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "Task items to write.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string"
                        },
                        "content": {
                            "type": "string",
                            "description": "Task description"
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed", "cancelled"]
                        },
                        "parent": {
                            "type": "string",
                            "description": "Optional id of another item, making this a nested subtask. Omit for top-level."
                        }
                    },
                    "required": ["id", "content", "status"]
                }
            },
            "merge": {
                "type": "boolean",
                "description": (
                    "true: update existing items by id, add new ones. "
                    "false (default): replace the entire list with a fresh plan."
                ),
                "default": False
            }
        },
        "required": []
    }
}


from tools.registry import registry, tool_error

registry.register(
    name="todo_list", toolset="todo", schema=TODO_SCHEMA, check_fn=check_todo_requirements,
    handler=lambda args, **kw: todo_tool(
        todos=args.get("todos"), merge=args.get("merge", False), store=kw.get("store")),
    emoji="📋")
