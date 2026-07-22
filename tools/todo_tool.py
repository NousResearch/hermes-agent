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
from typing import Dict, Any, List, Optional


# Valid status values for todo items
VALID_STATUSES = {"pending", "in_progress", "completed", "cancelled"}

# Fields persisted per todo item. ``activeForm`` is the optional human-readable
# present-progressive label rendered next to the in_progress marker
# ("… — Editing todo_tool.py" vs the bare content "Edit todo_tool.py"). It is
# purely cosmetic — falls back to ``content`` when missing or empty. Issue
# #59544 (feature: task lists) makes it part of the canonical Claude Code-style
# schema so models can supply both an action label and a verb-form.
TODO_ITEM_FIELDS = ("id", "content", "status", "activeForm")

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

    def write(self, todos: List[Dict[str, Any]], merge: bool = False) -> List[Dict[str, str]]:
        """
        Write todos. Returns the full current list after writing.

        Args:
            todos: list of {id, content, status} dicts
            merge: if False, replace the entire list. If True, update
                   existing items by id and append new ones.
        """
        if not merge:
            # Replace mode: new list entirely
            self._items = [self._validate(t) for t in self._dedupe_by_id(todos)]
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
                    # activeForm is optional — clearing it via merge requires
                    # an explicit empty string, otherwise the existing label
                    # is kept (matches how content/status preserve on partial
                    # writes — only supplied fields are touched).
                    if "activeForm" in t:
                        existing[item_id]["activeForm"] = self._cap_content(
                            str(t["activeForm"]).strip()
                        ) if t["activeForm"] else ""
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
        return self.read()

    def read(self) -> List[Dict[str, str]]:
        """Return a copy of the current list."""
        return [item.copy() for item in self._items]

    def has_items(self) -> bool:
        """Check if there are any items in the list."""
        return bool(self._items)

    def format_for_injection(self) -> Optional[str]:
        """
        Render the todo list for post-compression injection.

        Returns a human-readable string to append to the compressed
        message history, or None if the list is empty. The in_progress
        item (if any) is rendered with its activeForm — Claude Code-style
        verb-form labelling — so the model can see what it was actively
        doing when the conversation was compressed.
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
            label = item["content"]
            if item["status"] == "in_progress":
                active_form = item.get("activeForm", "").strip()
                if active_form and active_form != label:
                    label = f"{label} — {active_form}"
            lines.append(f"- {marker} {item['id']}. {label} ({item['status']})")

        return "\n".join(lines)

    def format_for_active_block(self) -> Optional[str]:
        """
        Render the active todo list for the system prompt.

        Surfaces the in-progress item (with activeForm) and the pending
        backlog so the model can refer to its plan across turns without
        a tool call. Returns None when there is nothing to show — the
        caller should then skip the block entirely rather than emit a
        bare header.

        Layout:

            [Active task list (N items; M done)]
            - [>] 1. Edit todo_tool.py — adding activeForm support (in_progress)
            - [ ] 2. Wire system prompt injection
            - [ ] 3. Add persistence tests

        Issue #59544 — the user wants the plan visible to the model in
        every turn, not just after compression.
        """
        if not self._items:
            return None

        markers = {
            "completed": "[x]",
            "in_progress": "[>]",
            "pending": "[ ]",
            "cancelled": "[~]",
        }

        # Include everything the model should be aware of. Completed and
        # cancelled items stay visible so the model can refer back to what
        # it has finished (and avoid redoing work) — the post-compression
        # injection path already filters those out for a different reason
        # (re-doing finished work after a context window collapse).
        visible = [item for item in self._items if item["status"] in VALID_STATUSES]
        if not visible:
            return None

        total = len(visible)
        done = sum(1 for i in visible if i["status"] in {"completed", "cancelled"})
        lines = [f"[Active task list ({total} item{'s' if total != 1 else ''}; {done} done)]"]
        for item in visible:
            marker = markers.get(item["status"], "[?]")
            label = item["content"]
            if item["status"] == "in_progress":
                active_form = item.get("activeForm", "").strip()
                if active_form and active_form != label:
                    label = f"{label} — {active_form}"
            lines.append(f"- {marker} {item['id']}. {label} ({item['status']})")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the current state for session persistence.

        Returns a JSON-serializable dict suitable for ``SessionDB.save_todo_state``.
        Includes the schema version so future migrations can detect older
        payloads without parsing the items blindly.
        """
        return {
            "version": 1,
            "items": [item.copy() for item in self._items],
        }

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "TodoStore":
        """Rebuild a TodoStore from a persisted payload.

        Tolerant of missing/malformed payloads (returns an empty store).
        Items pass through ``_validate`` so a payload that survived a
        schema-version bump still gets re-normalized against the current
        field set.
        """
        store = cls()
        if not isinstance(payload, dict):
            return store
        items = payload.get("items")
        if not isinstance(items, list):
            return store
        # Cap on replay matches the live write path — a corrupted or
        # oversized persisted blob shouldn't blow up the rehydration block.
        if len(items) > MAX_TODO_ITEMS:
            items = items[:MAX_TODO_ITEMS]
        try:
            store.write(items, merge=False)
        except Exception:
            # Defensive: a single bad item shouldn't nuke the entire store.
            # _validate already synthesizes placeholder rows, so this branch
            # is only reached for structural problems (non-list payloads,
            # items that aren't even iterable). Fail open to an empty store
            # rather than crashing the session.
            return cls()
        return store

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
        Returns a clean dict with only {id, content, status, activeForm}.
        """
        if not isinstance(item, dict):
            return {"id": "?", "content": "(invalid item)", "status": "pending", "activeForm": ""}

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

        active_form = str(item.get("activeForm", "")).strip()
        # Cap activeForm separately — it ships in the system prompt and the
        # injection block, so the same bounding rationale applies. Empty when
        # missing so format_for_active_block() falls back to ``content``.
        if active_form:
            active_form = TodoStore._cap_content(active_form)
        else:
            active_form = ""

        return {
            "id": item_id,
            "content": content,
            "status": status,
            "activeForm": active_form,
        }

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
    "description": (
        "Manage your task list for the current session. Use for complex tasks "
        "with 3+ steps or when the user provides multiple tasks. "
        "Call with no parameters to read the current list.\n\n"
        "Writing:\n"
        "- Provide 'todos' array to create/update items\n"
        "- merge=false (default): replace the entire list with a fresh plan\n"
        "- merge=true: update existing items by id, add any new ones\n\n"
        "Each item: {id: string, content: string, "
        "status: pending|in_progress|completed|cancelled, "
        "activeForm?: string}\n"
        "List order is priority. Only ONE item in_progress at a time.\n"
        "Mark items completed immediately when done. If something fails, "
        "cancel it and add a revised item.\n\n"
        "activeForm is optional — supply a short present-progressive label "
        "(e.g. \"Editing todo_tool.py\") that the agent renders next to "
        "the in_progress marker so the user can see what it is doing. "
        "Falls back to content when omitted.\n\n"
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
                        },
                        "activeForm": {
                            "type": "string",
                            "description": (
                                "Optional present-progressive label for the "
                                "in_progress item (e.g. 'Editing todo_tool.py'). "
                                "Rendered alongside content in the UI/prompt."
                            )
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
