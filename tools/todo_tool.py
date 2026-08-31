#!/usr/bin/env python3
"""
Todo Tool Module - Planning & Task Management

Provides an in-memory task list the agent uses to decompose complex tasks,
track progress, and maintain focus across long conversations. The state
lives on the AIAgent instance (one per session) and is re-injected into
the conversation after context compression events.

Design:
- Single `todo` tool: provide `todos` param to write, omit to read
- Every call returns the full current list
- No system prompt mutation, no tool response modification
- Behavioral guidance lives entirely in the tool schema description
"""

import json
import logging
import threading
from typing import Callable, Dict, Any, List, Optional


logger = logging.getLogger(__name__)


# Valid status values for todo items
VALID_STATUSES = {"pending", "in_progress", "completed", "cancelled"}

# Bounds on persisted todo state. The todo list is a planning aid the model
# re-reads after every context-compression event (see format_for_injection),
# so unbounded item content or count defeats the compression it rides through.
# These caps keep a single oversized item (whether authored by the model or
# replayed from caller-supplied history on the API server) from inflating the
# re-injection block. Generous relative to real plans — a todo item is a short
# task description, and active lists are a handful of items, not hundreds.
MAX_TODO_CONTENT_CHARS = 4000
MAX_TODO_ITEMS = 256
# Upper bound on a single todo tool-result payload accepted during history
# hydration. The gateway/API server replays caller-supplied conversation
# history to rebuild the store, so an oversized forged result is dropped
# before it is parsed and re-injected (see AIAgent._hydrate_todo_store).
MAX_TODO_RESULT_CHARS = 512_000
_TRUNCATION_MARKER = "… [truncated]"
# Persisted as ordinary message content. ContextCompressor uses this stable
# header to distinguish the synthetic post-compaction row from a real user.
TODO_INJECTION_HEADER = (
    "[Your active task list was preserved across context compression]"
)


class TodoStore:
    """
    In-memory todo list. One instance per AIAgent (one per session).

    Items are ordered -- list position is priority. Each item has:
      - id: unique string identifier (agent-chosen)
      - content: task description
      - status: pending | in_progress | completed | cancelled
    """

    def __init__(self):
        self._items: List[Dict[str, str]] = []
        # Human terminal actions and model tool calls run on different threads.
        # Keep the task list atomic so a Ctrl+T action cannot race a model merge.
        self._lock = threading.RLock()
        self._revision = 0
        self._has_restored_state = False
        self._history_reconciled = False
        self._pending_user_notices: List[str] = []
        self._on_change: Optional[Callable[[Dict[str, Any]], None]] = None
        # A user-completed/cancelled item is authoritative for the current plan.
        # A stale model snapshot may not silently reopen it. Replacing the plan
        # with different IDs drops the override; an explicit user reopen does too.
        self._user_status_overrides: Dict[str, str] = {}

    @property
    def revision(self) -> int:
        """Monotonic in-memory revision for UI conflict detection."""
        with self._lock:
            return self._revision

    @property
    def has_restored_state(self) -> bool:
        """Whether a durable sidecar existed, including an empty snapshot."""
        with self._lock:
            return self._has_restored_state

    @property
    def needs_history_reconciliation(self) -> bool:
        with self._lock:
            return not self._history_reconciled

    def mark_history_reconciled(self) -> None:
        with self._lock:
            self._history_reconciled = True

    def set_on_change(
        self, callback: Optional[Callable[[Dict[str, Any]], None]]
    ) -> None:
        """Attach a best-effort persistence/event callback."""
        with self._lock:
            self._on_change = callback

    def snapshot_state(self) -> Dict[str, Any]:
        """Return the durable task state, including user authority markers."""
        with self._lock:
            return {
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
            self._has_restored_state = True
            # A durable (or explicitly seeded branch) snapshot is the canonical
            # state. History hydration is only a legacy fallback when no
            # sidecar exists; replaying older tool output here can resurrect an
            # intentionally cleared plan or overwrite a newer persisted one.
            self._history_reconciled = True
            return True

    def _notify_change(self) -> None:
        callback = self._on_change
        if callback is None:
            return
        snapshot = self.snapshot_state()
        try:
            callback(snapshot)
        except Exception as exc:
            logger.debug("Todo state change callback failed: %s", exc)

    def consume_user_change_notice(self) -> str:
        """Return user-authored task changes once for next-turn API context."""
        with self._lock:
            if not self._pending_user_notices:
                return ""
            notice = "[Task list changes made by the user]\n" + "\n".join(
                f"- {line}" for line in self._pending_user_notices
            )
            self._pending_user_notices = []
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
                self._notify_change()
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
            self._notify_change()
            return True

    def read(self) -> List[Dict[str, str]]:
        """Return a copy of the current list."""
        with self._lock:
            return [item.copy() for item in self._items]

    def has_items(self) -> bool:
        """Check if there are any items in the list."""
        with self._lock:
            return bool(self._items)

    def format_for_injection(self) -> Optional[str]:
        """
        Render the todo list for post-compression injection.

        Returns a human-readable string to append to the compressed
        message history, or None if the list is empty.
        """
        with self._lock:
            items = [item.copy() for item in self._items]
        if not items:
            return None

        # Status markers for compact display
        markers = {
            "completed": "[x]",
            "in_progress": "[>]",
            "pending": "[ ]",
            "cancelled": "[~]",
        }

        # Only inject pending/in_progress items — completed/cancelled ones
        # cause the model to re-do finished work after compression.
        active_items = [
            item for item in items
            if item["status"] in {"pending", "in_progress"}
        ]
        if not active_items:
            return None

        lines = [TODO_INJECTION_HEADER]
        for item in active_items:
            marker = markers.get(item["status"], "[?]")
            lines.append(f"- {marker} {item['id']}. {item['content']} ({item['status']})")

        return "\n".join(lines)

    @staticmethod
    def _cap_content(content: str) -> str:
        """Truncate oversized todo content to MAX_TODO_CONTENT_CHARS.

        A single huge item would otherwise inflate the post-compression
        re-injection block (format_for_injection) without bound. Keep the
        head — the actionable part of a task description — plus a marker.
        """
        if len(content) > MAX_TODO_CONTENT_CHARS:
            keep = MAX_TODO_CONTENT_CHARS - len(_TRUNCATION_MARKER)
            return content[:keep] + _TRUNCATION_MARKER
        return content

    @staticmethod
    def _validate(item: Dict[str, Any]) -> Dict[str, str]:
        """
        Validate and normalize a todo item.

        Ensures required fields exist and status is valid.
        Returns a clean dict with only {id, content, status}.
        """
        if not isinstance(item, dict):
            return {"id": "?", "content": "(invalid item)", "status": "pending"}

        item_id = str(item.get("id", "")).strip()
        if not item_id:
            item_id = "?"

        content = str(item.get("content", "")).strip()
        if not content:
            content = "(no description)"
        else:
            content = TodoStore._cap_content(content)

        status = str(item.get("status", "pending")).strip().lower()
        if status not in VALID_STATUSES:
            status = "pending"

        return {"id": item_id, "content": content, "status": status}

    @staticmethod
    def _dedupe_by_id(todos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collapse duplicate ids, keeping the last occurrence in its position."""
        last_index: Dict[str, int] = {}
        for i, item in enumerate(todos):
            if not isinstance(item, dict):
                # Non-dict items get a synthetic key so _validate can handle them
                last_index[f"__invalid_{i}"] = i
                continue
            item_id = str(item.get("id", "")).strip() or "?"
            last_index[item_id] = i
        return [todos[i] for i in sorted(last_index.values())]

    @staticmethod
    def _normalize_order(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Lift the active step ahead of any earlier unfinished placeholders."""
        active_index = next(
            (i for i, item in enumerate(items) if item["status"] == "in_progress"),
            None,
        )
        if active_index is None:
            return items

        pending_index = next(
            (
                i for i, item in enumerate(items[:active_index])
                if item["status"] == "pending"
            ),
            None,
        )
        if pending_index is None:
            return items

        normalized = items.copy()
        active_item = normalized.pop(active_index)
        normalized.insert(pending_index, active_item)
        return normalized


def todo_tool(
    todos: Optional[List[Dict[str, Any]]] = None,
    merge: bool = False,
    store: Optional[TodoStore] = None,
) -> str:
    """
    Single entry point for the todo tool. Reads or writes depending on params.

    Args:
        todos: if provided, write these items. If None, read current list.
        merge: if True, update by id. If False (default), replace entire list.
        store: the TodoStore instance from the AIAgent.

    Returns:
        JSON string with the full current list and summary metadata.
    """
    if store is None:
        return tool_error("TodoStore not initialized")

    if todos is not None:
        # Guard: LLM sometimes sends todos as a JSON string instead of a list
        if isinstance(todos, str):
            try:
                todos = json.loads(todos)
            except (json.JSONDecodeError, TypeError):
                return tool_error("todos must be a list of objects, got unparseable string")
        if not isinstance(todos, list):
            return tool_error(
                f"todos must be a list, got {type(todos).__name__}"
            )
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


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================
# Behavioral guidance is baked into the description so it's part of the
# static tool schema (cached, never changes mid-conversation).

TODO_SCHEMA = {
    "name": "todo",
    # Dieted (#95681): the item shape and merge semantics live ONLY in the
    # parameter schema below — the description teaches behavior, not
    # structure the params already define.
    "description": (
        "Manage your task list for the current session. Use for complex tasks "
        "with 3+ steps or when the user provides multiple tasks. "
        "For 'all N items' tasks, enumerate every instance as its own checklist "
        "item so none are silently dropped. "
        "Call with no parameters to read the current list.\n"
        "List order is priority. Only ONE item in_progress at a time. "
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


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="todo",
    toolset="todo",
    schema=TODO_SCHEMA,
    handler=lambda args, **kw: todo_tool(
        todos=args.get("todos"), merge=args.get("merge", False), store=kw.get("store")),
    check_fn=check_todo_requirements,
    emoji="📋",
)
