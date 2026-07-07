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
import time
from typing import Dict, Any, List, Optional


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
        # Wall-clock timing per item id, populated by status transitions.
        # Kept separate from _items so the item shape stays {id, content,
        # status} for format_for_injection and existing consumers. Each entry
        # is {"started_at": float, "ended_at": float} (epoch seconds); either
        # key may be absent until the matching transition fires.
        self._timing: Dict[str, Dict[str, float]] = {}

    def write(self, todos: List[Dict[str, Any]], merge: bool = False) -> List[Dict[str, str]]:
        """
        Write todos. Returns the full current list after writing.

        Args:
            todos: list of {id, content, status} dicts. Items may also carry
                   "started_at"/"ended_at" (epoch seconds) — used only to
                   restore timing on history replay (see _hydrate_todo_store);
                   the model itself never sends these.
            merge: if False, replace the entire list. If True, update
                   existing items by id and append new ones.
        """
        # Pull any timing carried on the incoming items. Real model calls never
        # include these; they appear only when _hydrate_todo_store replays a
        # prior todo result back into a fresh store. Terminal spans are adopted
        # below; live in-progress spans restart on the new turn so idle time
        # between gateway turns does not show up as fake work time.
        incoming_timing = self._extract_incoming_timing(todos)

        if not merge:
            # Replace mode: new list entirely. If a fresh model-authored plan
            # reuses generic ids (setup/phase-a/report) for different work,
            # reset those clocks even if the model copied timing fields from a
            # previous result. Terminal hydration on a fresh store is unaffected,
            # and terminal replay can still be adopted after this clear.
            previous_by_id = {item["id"]: item for item in self._items}
            self._items = [self._validate(t) for t in self._dedupe_by_id(todos)]
            for current in self._items:
                item_id = current["id"]
                previous = previous_by_id.get(item_id)
                if previous is not None and previous.get("content") != current.get("content"):
                    self._timing.pop(item_id, None)
                    incoming_timing.pop(item_id, None)
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
            self._items = rebuilt
        # Bound total item count so a replayed/oversized list can't grow the
        # re-injection block without limit. Keep the highest-priority head
        # (list order is priority).
        if len(self._items) > MAX_TODO_ITEMS:
            self._items = self._items[:MAX_TODO_ITEMS]
        # Stamp transitions and prune timing for dropped ids.
        self._update_timing(incoming_timing)
        return self.read()

    @staticmethod
    def _extract_incoming_timing(todos: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Collect started_at/ended_at carried on input items (history replay)."""
        out: Dict[str, Dict[str, float]] = {}
        if not isinstance(todos, list):
            return out
        for t in todos:
            if not isinstance(t, dict):
                continue
            item_id = str(t.get("id", "")).strip()
            if not item_id:
                continue
            entry: Dict[str, float] = {}
            for key in ("started_at", "ended_at"):
                value = t.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    entry[key] = float(value)
            if entry:
                out[item_id] = entry
        return out

    def _update_timing(
        self,
        incoming_timing: Dict[str, Dict[str, float]],
    ) -> None:
        """Stamp wall-clock timing from status transitions; prune dropped ids.

        Transitions are detected implicitly from each item's current status vs
        the timing already recorded for it (e.g. an in_progress item with no
        started_at is a fresh start), so no pre-mutation status snapshot is
        needed. Wall-clock semantics: started_at is set the first time an item
        enters in_progress; ended_at is set when it leaves in_progress for a
        terminal status (completed/cancelled). Re-opening a finished item clears
        ended_at so the elapsed span keeps growing. Items completed without ever
        being in_progress get no timing (nothing to measure).
        """
        now = time.time()
        live_ids = {item["id"] for item in self._items}

        for item in self._items:
            item_id = item["id"]
            new_status = item["status"]
            entry = self._timing.setdefault(item_id, {})

            # Adopt replayed timing only for terminal spans. An in-progress item
            # from history was not actively worked during the gap between gateway
            # turns, so it starts a fresh live clock below.
            replayed = incoming_timing.get(item_id)
            if replayed and new_status in ("completed", "cancelled"):
                for key, value in replayed.items():
                    entry.setdefault(key, value)

            if new_status == "in_progress":
                if "started_at" not in entry:
                    entry["started_at"] = now
                # Re-opened after finishing: let the span keep accruing.
                entry.pop("ended_at", None)
            elif new_status in ("completed", "cancelled"):
                # Only record an end when we have a measurable start and have
                # not already stamped one (idempotent across re-sent lists).
                if "started_at" in entry and "ended_at" not in entry:
                    entry["ended_at"] = now

            if not entry:
                # Never store empty dicts (keeps elapsed_for() simple).
                self._timing.pop(item_id, None)

        # Drop timing for ids no longer in the list (replace mode can remove).
        for stale_id in [tid for tid in self._timing if tid not in live_ids]:
            self._timing.pop(stale_id, None)

    def elapsed_for(self, item_id: str) -> Optional[float]:
        """Wall-clock seconds for an item, or None if not measurable yet.

        Returns the closed span (ended_at - started_at) once finished, the
        running span (now - started_at) while in progress, or None when the
        item never entered in_progress.
        """
        entry = self._timing.get(item_id)
        if not entry or "started_at" not in entry:
            return None
        end = entry.get("ended_at")
        if end is None:
            end = time.time()
        elapsed = end - entry["started_at"]
        return elapsed if elapsed >= 0 else 0.0

    def read(self) -> List[Dict[str, str]]:
        """Return a copy of the current list."""
        return [item.copy() for item in self._items]

    def read_with_timing(self) -> List[Dict[str, Any]]:
        """Return the list with wall-clock timing fields attached.

        Each item carries the usual {id, content, status} plus:
          - elapsed_seconds: float wall-clock span, or None if unmeasured.
          - started_at / ended_at: raw epoch stamps when present. These let
            _hydrate_todo_store replay terminal spans into a fresh store. Live
            in-progress items intentionally restart on hydration so idle time
            between gateway turns is not counted as work time.
        """
        out: List[Dict[str, Any]] = []
        for item in self._items:
            entry: Dict[str, Any] = dict(item)
            timing = self._timing.get(item["id"], {})
            if "started_at" in timing:
                entry["started_at"] = timing["started_at"]
            if "ended_at" in timing:
                entry["ended_at"] = timing["ended_at"]
            entry["elapsed_seconds"] = self.elapsed_for(item["id"])
            out.append(entry)
        return out

    def total_elapsed_seconds(self) -> Optional[float]:
        """Sum of measurable per-item elapsed spans, or None if none measured.

        Wall-clock per item, so overlapping/parallel spans are summed naively
        (a deliberately simple total — it answers "how much tracked work time"
        not "how long was the wall clock for the whole plan").
        """
        total = 0.0
        measured = False
        for item in self._items:
            value = self.elapsed_for(item["id"])
            if value is not None:
                total += value
                measured = True
        return total if measured else None

    def has_items(self) -> bool:
        """Check if there are any items in the list."""
        return bool(self._items)

    def format_for_injection(self) -> Optional[str]:
        """
        Render the todo list for post-compression injection.

        Returns a human-readable string to append to the compressed
        message history, or None if the list is empty.
        """
        if not self._items:
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
            item for item in self._items
            if item["status"] in {"pending", "in_progress"}
        ]
        if not active_items:
            return None

        lines = ["[Your active task list was preserved across context compression]"]
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
        JSON string with the full current list and summary metadata. Each item
        includes "elapsed_seconds" (wall-clock span, or null when the item never
        entered in_progress) and, when present, raw "started_at"/"ended_at"
        epoch stamps so timing survives history replay. The summary includes
        "total_elapsed_seconds".
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
        store.write(todos, merge)
    # read_with_timing also covers the read-only path (todos is None).
    items = store.read_with_timing()

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
            "total_elapsed_seconds": store.total_elapsed_seconds(),
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
    "description": (
        "Manage your task list for the current session. Use for complex tasks "
        "with 3+ steps or when the user provides multiple tasks. "
        "Call with no parameters to read the current list.\n\n"
        "Writing:\n"
        "- Provide 'todos' array to create/update items\n"
        "- merge=false (default): replace the entire list with a fresh plan\n"
        "- merge=true: update existing items by id, add any new ones\n\n"
        "Each item: {id: string, content: string, "
        "status: pending|in_progress|completed|cancelled}\n"
        "List order is priority. Only ONE item in_progress at a time.\n"
        "Mark items completed immediately when done. If something fails, "
        "cancel it and add a revised item.\n\n"
        "Always returns the full current list."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "Task items to write. Omit to read current list.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Unique item identifier"
                        },
                        "content": {
                            "type": "string",
                            "description": "Task description"
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed", "cancelled"],
                            "description": "Current status"
                        }
                    },
                    "required": ["id", "content", "status"]
                }
            },
            "merge": {
                "type": "boolean",
                "description": (
                    "true: update existing items by id, add new ones. "
                    "false (default): replace the entire list."
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
