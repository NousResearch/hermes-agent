"""Todo tool: in-memory, revisioned task list for multi-step work. State lives on the
AIAgent (one per session), is re-injected after context compression, and every write bumps
a monotonic revision so UI clients can reject stale updates. One ``todo_list`` tool: pass
``todos`` to write, omit to read; every call returns the full list. No system-prompt mutation."""

import json
import threading
from typing import Any, Callable, Dict, List, Optional

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
    """In-memory todo list, one per AIAgent. List position is priority; items are
    ``{id, content, status, parent?}`` — ``parent`` nests a subtask."""

    def __init__(self):
        self._items: List[Dict[str, str]] = []
        self._revision = 0
        # ── KENSEI CUSTOM — change-notification hook (ported) ──
        # Best-effort persistence/event callback (agent.todo_state.build_todo_store
        # attaches the session-DB persister). Never raises into the caller.
        self._lock = threading.RLock()
        self._on_change: Optional[Callable[[Dict[str, Any]], None]] = None
        # ── KENSEI CUSTOM — history-reconciliation gate (ported) ──
        # One-shot: turn_context re-hydrates from transcript history only while
        # this flag is True; mark_history_reconciled() consumes it so the store
        # is not re-scanned every turn. User terminal overrides survive the
        # replacement via the revision guard inside restore().
        self._generation = 0
        self._history_reconciled = False
        self._pending_user_notices: List[str] = []
        # A user-completed/cancelled item is authoritative for the current plan.
        self._user_status_overrides: Dict[str, str] = {}

    def set_on_change(self, callback: Optional[Callable[[Dict[str, Any]], None]]) -> None:
        """Attach a best-effort persistence/event callback."""
        with self._lock:
            self._on_change = callback

    def _notify_change(self) -> None:
        with self._lock:
            callback = self._on_change
        if callback is None:
            return
        try:
            callback(self.snapshot_state() if hasattr(self, "snapshot_state") else self.snapshot())
        except Exception:
            pass  # persistence must never break the tool path
    # ── END KENSEI CUSTOM ──

    def _fresh_items(self, todos: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Validate, dedupe and order a whole new list (replace / restore)."""
        return self._normalize_order([self._validate(t) for t in self._dedupe_by_id(todos)])

    def write(self, todos: List[Dict[str, Any]], merge: bool = False) -> List[Dict[str, str]]:
        """Write a full or partial task snapshot and return the canonical list."""
        with self._lock:
            before = [item.copy() for item in self._items]
            if merge:
                self._merge_items(todos)
            else:
                self._replace_items(todos)
            self._apply_bounds()
            if self._items != before:
                self._revision += 1
                self._generation += 1
                self._notify_change()
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
        """Full state clients can reconcile atomically."""
        return {"todos": self.read(), "revision": self._revision}

    def restore(self, todos: List[Dict[str, Any]], *, revision: Any = 0) -> List[Dict[str, str]]:
        """Restore a trusted snapshot without manufacturing a new revision."""
        self._items = self._fresh_items(todos)[:MAX_TODO_ITEMS]
        try:
            self._revision = max(0, int(revision or 0))
        except (TypeError, ValueError):
            self._revision = 0
        self._notify_change()  # KENSEI CUSTOM: persist on restore
        self._history_reconciled = True  # KENSEI CUSTOM: consumed by restore
        return self.read()

    def update_status(
        self,
        item_id: str,
        status: str,
        *,
        actor: str = "user",
        expected_revision: Optional[int] = None,
    ) -> bool:
        """Update one item's status from a trusted UI action.

        User completion/cancellation wins over stale model snapshots for the
        lifetime of the current plan. Reopening an item releases that override
        so the agent can progress it normally afterwards.
        """
        normalized_id = str(item_id or "").strip()
        normalized_status = str(status or "").strip().lower()
        if not normalized_id or normalized_status not in VALID_STATUSES:
            return False

        with self._lock:
            if expected_revision is not None and expected_revision != self._revision:
                return False
            item = next((row for row in self._items if row["id"] == normalized_id), None)
            if item is None:
                return False

            before_override = self._user_status_overrides.get(normalized_id)
            if actor == "user":
                if normalized_status in {"completed", "cancelled"}:
                    self._user_status_overrides[normalized_id] = normalized_status
                else:
                    self._user_status_overrides.pop(normalized_id, None)
            elif normalized_id in self._user_status_overrides:
                normalized_status = self._user_status_overrides[normalized_id]

            override_changed = before_override != self._user_status_overrides.get(normalized_id)
            status_changed = item["status"] != normalized_status
            if not status_changed and not override_changed:
                return True
            if status_changed:
                item["status"] = normalized_status
                self._items = self._normalize_order(self._items)
            if actor == "user":
                self._pending_user_notices.append(
                    "task_id="
                    + json.dumps(normalized_id, ensure_ascii=True)
                    + f" status={normalized_status}"
                )
                self._pending_user_notices = self._pending_user_notices[-20:]
            self._revision += 1
            self._generation += 1
            self._notify_change()
            return True

    @property
    def revision(self) -> int:
        """Monotonic in-memory revision for UI conflict detection."""
        with self._lock:
            return self._revision

    def snapshot_state(self) -> Dict[str, Any]:
        """Return the durable task state, including user authority markers."""
        with self._lock:
            return {
                "generation": self._generation,
                "revision": self._revision,
                "todos": [item.copy() for item in self._items],
                "user_status_overrides": dict(self._user_status_overrides),
                "pending_user_notices": list(self._pending_user_notices),
            }

    def load_state(self, payload: Any) -> bool:
        """Restore a validated durable snapshot without firing callbacks."""
        if not isinstance(payload, dict) or not isinstance(payload.get("todos"), list):
            return False
        with self._lock:
            items = [self._validate(item) for item in self._dedupe_by_id(payload["todos"])]
            if len(items) > MAX_TODO_ITEMS:
                items = items[:MAX_TODO_ITEMS]
            item_ids = {item["id"] for item in items}
            raw_overrides = payload.get("user_status_overrides")
            overrides = raw_overrides if isinstance(raw_overrides, dict) else {}
            self._user_status_overrides = {
                str(item_id): str(status)
                for item_id, status in overrides.items()
                if str(item_id) in item_ids
                and str(status) in {"completed", "cancelled"}
            }
            for item in items:
                override = self._user_status_overrides.get(item["id"])
                if override is not None:
                    item["status"] = override
            self._items = self._normalize_order(items)
            raw_notices = payload.get("pending_user_notices")
            self._pending_user_notices = (
                [
                    self._cap_content(str(notice).strip())
                    for notice in raw_notices[:20]
                    if str(notice).strip()
                ]
                if isinstance(raw_notices, list)
                else []
            )
            raw_revision = payload.get("revision", 0)
            self._revision = max(0, raw_revision) if isinstance(raw_revision, int) else 0
            raw_generation = payload.get("generation", self._revision)
            self._generation = (
                max(0, raw_generation)
                if isinstance(raw_generation, int) and not isinstance(raw_generation, bool)
                else self._revision
            )
            # A durable (or explicitly seeded branch) snapshot is the canonical
            # state. History hydration is only a legacy fallback when no
            # sidecar exists; replaying older tool output here can resurrect an
            # intentionally cleared plan or overwrite a newer persisted one.
            self._history_reconciled = True
            return True

    def consume_user_change_notice(self) -> str:
        """Return user-authored task changes once for next-turn API context."""
        with self._lock:
            if not self._pending_user_notices:
                return ""
            notice = "[Task list changes made by the user]\n" + "\n".join(
                f"- {line}" for line in self._pending_user_notices
            )
            self._pending_user_notices = []
            self._generation += 1
            self._notify_change()
            return notice

    def _replace_items(self, todos: List[Dict[str, Any]]) -> None:
        """Replace the plan while retaining overrides for unchanged task identity."""
        previous_content = {item["id"]: item["content"] for item in self._items}
        replacement = [self._validate(item) for item in self._dedupe_by_id(todos)]
        replacement_content = {item["id"]: item["content"] for item in replacement}
        self._user_status_overrides = {
            item_id: status
            for item_id, status in self._user_status_overrides.items()
            if previous_content.get(item_id) == replacement_content.get(item_id)
        }
        for item in replacement:
            override = self._user_status_overrides.get(item["id"])
            if override is not None:
                item["status"] = override
        self._items = self._normalize_order(replacement)

    def _merge_items(self, todos: List[Dict[str, Any]]) -> None:
        """Apply partial model updates without overriding user terminal actions."""
        existing = {item["id"]: item for item in self._items}
        for update in self._dedupe_by_id(todos):
            item_id = str(update.get("id", "")).strip()
            if not item_id:
                continue
            if item_id not in existing:
                validated = self._validate(update)
                existing[validated["id"]] = validated
                self._items.append(validated)
                continue
            dropped_override = False
            if update.get("content"):
                content = str(update["content"]).strip()
                if content:
                    content = self._cap_content(content)
                    if content != existing[item_id]["content"]:
                        dropped_override = self._user_status_overrides.pop(
                            item_id, None
                        ) is not None
                        existing[item_id]["content"] = content
            if update.get("status"):
                status = str(update["status"]).strip().lower()
                if status in VALID_STATUSES:
                    existing[item_id]["status"] = self._user_status_overrides.get(
                        item_id, status
                    )
            elif dropped_override:
                # A reused id with different content is a new task identity.
                # Do not let the prior task's user-completed state leak into it.
                existing[item_id]["status"] = "pending"

        seen = set()
        rebuilt = []
        for item in self._items:
            current = existing.get(item["id"], item)
            if current["id"] not in seen:
                rebuilt.append(current)
                seen.add(current["id"])
        self._items = self._normalize_order(rebuilt)

    def _apply_bounds(self) -> None:
        if len(self._items) <= MAX_TODO_ITEMS:
            return
        self._items = self._items[:MAX_TODO_ITEMS]
        retained_ids = {item["id"] for item in self._items}
        self._user_status_overrides = {
            item_id: status
            for item_id, status in self._user_status_overrides.items()
            if item_id in retained_ids
        }

    @property
    def needs_history_reconciliation(self) -> bool:
        with self._lock:
            return not self._history_reconciled

    def mark_history_reconciled(self) -> None:
        """KENSEI CUSTOM (ported): consume the one-shot history-reconciliation gate."""
        with self._lock:
            self._history_reconciled = True

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
        """Collapse duplicate ids, keeping the last occurrence in its position."""
        last_index: Dict[str, int] = {}
        for i, item in enumerate(todos):  # non-dicts get a synthetic key; _validate handles them
            key = str(item.get("id", "")).strip() if isinstance(item, dict) else f"__invalid_{i}"
            last_index[key or "?"] = i
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
                       "generation": store.snapshot_state()["generation"],
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
