#!/usr/bin/env python3
"""
Todo Tool Module - Planning & Task Management

Provides an in-memory, revisioned task list the agent uses to decompose
complex tasks, track progress, and maintain focus across long conversations.
The state lives on the AIAgent instance (one per session), is re-injected into
the conversation after context compression events, and every write bumps a
monotonic revision so UI clients can reject stale updates.

Design:
- Single `todo` tool: provide `todos` param to write, omit to read
- Every call returns the full current list
- No system prompt mutation, no tool response modification
- Behavioral guidance lives entirely in the tool schema description
"""

import json
from typing import Any, Dict, List, Optional

VALID_STATUSES = {"pending", "in_progress", "completed", "cancelled"}
# The list is re-read after every compression (format_for_injection), so unbounded
# content/count would defeat the compression it rides through. Caps apply equally to
# model-authored items and caller-replayed API history.
MAX_TODO_CONTENT_CHARS = 4000
MAX_TODO_ITEMS = 256
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
    """
    In-memory todo list. One instance per AIAgent (one per session).

    Items are ordered -- list position is priority. Each item has:
      - id: unique string identifier (agent-chosen)
      - content: task description
      - status: pending | in_progress | completed | cancelled
      - parent: optional id of another item, for nested subtasks
    """

    def __init__(self):
        self._items: List[Dict[str, str]] = []
        self._revision = 0

    def write(self, todos: List[Dict[str, Any]], merge: bool = False) -> List[Dict[str, str]]:
        """
        Write todos. Returns the full current list after writing.

        Args:
            todos: list of {id, content, status} dicts
            merge: if False, replace the entire list. If True, update
                   existing items by id and append new ones.
        """
        before = self.read()
        if not merge:
            # Replace mode: new list entirely
            self._items = self._normalize_order(
                [self._validate(t) for t in self._dedupe_by_id(todos)]
            )
        else:
            # Merge mode: update existing items by id, append new ones
            existing = {item["id"]: item for item in self._items}
            for t in self._dedupe_by_id(todos):
                item_id = str(t.get("id", "")).strip()
                if not item_id:
                    continue  # Can't merge without an id

                if item_id in existing:
                    # Update only the fields the LLM actually provided
                    if "content" in t and t["content"]:
                        existing[item_id]["content"] = self._cap_content(str(t["content"]).strip())
                    if "status" in t and t["status"]:
                        status = str(t["status"]).strip().lower()
                        if status in VALID_STATUSES:
                            existing[item_id]["status"] = status
                    if "parent" in t:
                        parent = str(t["parent"] or "").strip()
                        if parent:
                            existing[item_id]["parent"] = parent
                        else:
                            existing[item_id].pop("parent", None)
                else:
                    # New item -- validate fully and append to end
                    validated = self._validate(t)
                    existing[validated["id"]] = validated
                    self._items.append(validated)
            # Rebuild _items preserving order for existing items
            seen = set()
            rebuilt = []
            for item in self._items:
                current = existing.get(item["id"], item)
                if current["id"] not in seen:
                    rebuilt.append(current)
                    seen.add(current["id"])
            self._items = self._normalize_order(rebuilt)
        # Bound total item count so a replayed/oversized list can't grow the
        # re-injection block without limit. Keep the highest-priority head
        # (list order is priority).
        if len(self._items) > MAX_TODO_ITEMS:
            self._items = self._items[:MAX_TODO_ITEMS]
        self._sanitize_parents(self._items)
        if self._items != before:
            self._revision += 1
        return self.read()

    def _merge(self, todos: List[Dict[str, Any]]) -> None:
        """Update existing items only in the fields provided; append new ones (validated)."""
        existing = {item["id"]: item for item in self._items}
        for t in self._dedupe_by_id(todos):
            item_id = str(t.get("id", "")).strip()
            if not item_id:
                continue  # can't merge without an id
            cur = existing.get(item_id)
            if cur is None:
                validated = self._validate(t)
                existing[validated["id"]] = validated
                self._items.append(validated)
                continue
            if t.get("content"):
                cur["content"] = self._cap_content(str(t["content"]).strip())
            if t.get("status") and str(t["status"]).strip().lower() in VALID_STATUSES:
                cur["status"] = str(t["status"]).strip().lower()
            if "parent" in t:
                parent = str(t["parent"] or "").strip()
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
        """Return the full state clients can reconcile atomically."""
        return {"todos": self.read(), "revision": self._revision}

    def restore(
        self,
        todos: List[Dict[str, Any]],
        *,
        revision: Any = 0,
    ) -> List[Dict[str, str]]:
        """Restore a trusted snapshot without manufacturing a new revision."""
        self._items = self._normalize_order(
            [self._validate(t) for t in self._dedupe_by_id(todos)]
        )[:MAX_TODO_ITEMS]
        try:
            self._revision = max(0, int(revision or 0))
        except (TypeError, ValueError):
            self._revision = 0
        return self.read()

    def format_for_injection(self) -> Optional[str]:
        """
        Render the todo list for post-compression injection.

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

        # Status markers for compact display
        markers = {
            "completed": "[x]",
            "in_progress": "[>]",
            "pending": "[ ]",
            "cancelled": "[~]",
        }

        # Only inject pending/in_progress items — completed/cancelled ones
        # cause the model to re-do finished work after compression. A parent
        # is kept (with its real status marker) when any descendant is
        # active, so subtasks keep their context.
        active = {"pending", "in_progress"}
        children: Dict[str, List[Dict[str, str]]] = {}
        roots: List[Dict[str, str]] = []
        for item in self._items:
            parent = item.get("parent")
            if parent:
                children.setdefault(parent, []).append(item)
            else:
                roots.append(item)

        def render(item: Dict[str, str], depth: int, out: List[str]) -> bool:
            kid_lines: List[str] = []
            has_active_kid = False
            for kid in children.get(item["id"], []):
                has_active_kid |= render(kid, depth + 1, kid_lines)
            keep = item["status"] in active or has_active_kid
            if keep:
                marker = markers.get(item["status"], "[?]")
                out.append(
                    f"{'  ' * depth}- {marker} {item['id']}. "
                    f"{item['content']} ({item['status']})"
                )
                out.extend(kid_lines)
            return keep

        lines = [TODO_INJECTION_HEADER]
        for item in roots:
            render(item, 0, lines)
        if len(lines) == 1:
            return None

        return "\n".join(lines)

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
        item_id = str(item.get("id", "")).strip() or "?"
        content = str(item.get("content", "")).strip()
        status = str(item.get("status", "pending")).strip().lower()
        result = {"id": item_id,
                  "content": TodoStore._cap_content(content) if content else "(no description)",
                  "status": status if status in VALID_STATUSES else "pending"}
        parent = str(item.get("parent") or "").strip()
        if parent and parent != item_id:
            result["parent"] = parent
        return result

        result = {"id": item_id, "content": content, "status": status}
        parent = str(item.get("parent") or "").strip()
        if parent and parent != item_id:
            result["parent"] = parent
        return result

    @staticmethod
    def _sanitize_parents(items: List[Dict[str, str]]) -> None:
        """Drop dangling parent refs and break cycles (in place).

        A parent pointing at a missing id, or a chain that loops back on
        itself, would corrupt tree rendering — such items become roots.
        """
        ids = {item["id"] for item in items}
        by_id = {item["id"]: item for item in items}
        for item in items:
            parent = item.get("parent")
            if parent and parent not in ids:
                item.pop("parent", None)
        for item in items:
            seen = {item["id"]}
            node = item
            while node.get("parent"):
                if node["parent"] in seen:
                    item.pop("parent", None)
                    break
                seen.add(node["parent"])
                node = by_id[node["parent"]]

    @staticmethod
    def _dedupe_by_id(todos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collapse duplicate ids, keeping the last occurrence in its position."""
        last_index: Dict[str, int] = {}
        for i, item in enumerate(todos):  # non-dicts get a synthetic key; _validate handles them
            key = str(item.get("id", "")).strip() if isinstance(item, dict) else f"__invalid_{i}"
            last_index[key or "?"] = i
        return [todos[i] for i in sorted(last_index.values())]

    @staticmethod
    def _normalize_order(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Lift the active step ahead of any earlier unfinished placeholders."""
        # Nested lists keep authored order — reordering a flat position would
        # tear a subtask away from its siblings.
        if any(item.get("parent") for item in items):
            return items
        active_index = next(
            (i for i, item in enumerate(items) if item["status"] == "in_progress"),
            None,
        )
        if active_index is None:
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
    else:
        items = store.read()

    # Build summary counts
    pending = sum(1 for i in items if i["status"] == "pending")
    in_progress = sum(1 for i in items if i["status"] == "in_progress")
    completed = sum(1 for i in items if i["status"] == "completed")
    cancelled = sum(1 for i in items if i["status"] == "cancelled")

    return json.dumps({
        "todos": items,
        "revision": store.snapshot()["revision"],
        "summary": {
            "total": len(items),
            "pending": pending,
            "in_progress": in_progress,
            "completed": completed,
            "cancelled": cancelled,
        },
    }, ensure_ascii=False)


def check_todo_requirements() -> bool:
    """Todo tool has no external requirements -- always available."""
    return True


# Behavioral guidance is baked into the (static, cached) description; item shape and merge
# semantics live ONLY in the parameter schema.
TODO_SCHEMA = {
    "name": "todo",
    # Dieted (#95681): the item shape and merge semantics live ONLY in the
    # parameter schema below — the description teaches behavior, not
    # structure the params already define.
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
