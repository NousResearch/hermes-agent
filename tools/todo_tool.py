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

Enhanced features:
- Task dependencies (depends_on)
- Agent assignment (assigned_to)
- Priority levels (low|medium|high|critical)
- Task types (task|milestone|blocked)
- Creation timestamps (created_at)
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Set


# Valid status values for todo items
VALID_STATUSES = {"pending", "in_progress", "completed", "cancelled"}

# Valid priority levels
VALID_PRIORITIES = {"low", "medium", "high", "critical"}

# Valid task types
VALID_TYPES = {"task", "milestone", "blocked"}


class TodoStore:
    """
    In-memory todo list. One instance per AIAgent (one per session).

    Items are ordered -- list position is priority. Each item has:
      - id: unique string identifier (agent-chosen)
      - content: task description
      - status: pending | in_progress | completed | cancelled
      - depends_on: list of task ids this task depends on (optional)
      - assigned_to: agent name assigned to this task (optional)
      - priority: low | medium | high | critical (default: medium)
      - type: task | milestone | blocked (default: task)
      - created_at: ISO timestamp when task was created
    """

    def __init__(self):
        self._items: List[Dict[str, Any]] = []

    def write(self, todos: List[Dict[str, Any]], merge: bool = False) -> List[Dict[str, Any]]:
        """
        Write todos. Returns the full current list after writing.

        Args:
            todos: list of {id, content, status, ...} dicts
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
                        existing[item_id]["content"] = str(t["content"]).strip()
                    if "status" in t and t["status"]:
                        status = str(t["status"]).strip().lower()
                        if status in VALID_STATUSES:
                            existing[item_id]["status"] = status
                    # Update new fields if provided
                    if "depends_on" in t:
                        existing[item_id]["depends_on"] = self._validate_depends_on(t["depends_on"])
                    if "assigned_to" in t:
                        existing[item_id]["assigned_to"] = str(t["assigned_to"]).strip() if t["assigned_to"] else None
                    if "priority" in t and t["priority"]:
                        priority = str(t["priority"]).strip().lower()
                        if priority in VALID_PRIORITIES:
                            existing[item_id]["priority"] = priority
                    if "type" in t and t["type"]:
                        task_type = str(t["type"]).strip().lower()
                        if task_type in VALID_TYPES:
                            existing[item_id]["type"] = task_type
                    # created_at is not updated on merge - keep original
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
        return self.read()

    def read(self) -> List[Dict[str, Any]]:
        """Return a copy of the current list."""
        return [item.copy() for item in self._items]

    def has_items(self) -> bool:
        """Check if there are any items in the list."""
        return bool(self._items)

    def get_dependencies(self, item_id: str) -> List[str]:
        """Get list of task ids that a task depends on."""
        for item in self._items:
            if item["id"] == item_id:
                return item.get("depends_on", []) or []
        return []

    def get_dependents(self, item_id: str) -> List[str]:
        """Get list of task ids that depend on a given task."""
        dependents = []
        for item in self._items:
            depends_on = item.get("depends_on", []) or []
            if item_id in depends_on:
                dependents.append(item["id"])
        return dependents

    def get_blocked_tasks(self) -> List[str]:
        """Get list of task ids that are blocked by incomplete dependencies."""
        blocked = []
        completed_ids = {item["id"] for item in self._items if item["status"] == "completed"}
        cancelled_ids = {item["id"] for item in self._items if item["status"] == "cancelled"}
        done_ids = completed_ids | cancelled_ids

        for item in self._items:
            if item["status"] in ("pending", "in_progress"):
                depends_on = item.get("depends_on", []) or []
                # Task is blocked if any dependency is not completed/cancelled
                incomplete_deps = [d for d in depends_on if d not in done_ids]
                if incomplete_deps:
                    blocked.append(item["id"])
        return blocked

    def build_dependency_graph(self) -> Dict[str, Any]:
        """
        Build a dependency graph representation.
        
        Returns:
            Dict with 'nodes' (all tasks) and 'edges' (dependency relationships)
        """
        nodes = []
        edges = []
        
        for item in self._items:
            nodes.append({
                "id": item["id"],
                "status": item["status"],
                "type": item.get("type", "task"),
                "priority": item.get("priority", "medium"),
            })
            
            depends_on = item.get("depends_on", []) or []
            for dep_id in depends_on:
                edges.append({
                    "from": dep_id,
                    "to": item["id"],
                    "type": "depends_on"
                })
        
        return {"nodes": nodes, "edges": edges}

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
        
        # Priority markers
        priority_markers = {
            "critical": "!!!",
            "high": "!!",
            "medium": "!",
            "low": ".",
        }
        
        # Type markers
        type_markers = {
            "task": "",
            "milestone": "[M]",
            "blocked": "[B]",
        }

        # Only inject pending/in_progress items — completed/cancelled ones
        # cause the model to re-do finished work after compression.
        active_items = [
            item for item in self._items
            if item["status"] in ("pending", "in_progress")
        ]
        if not active_items:
            return None

        # Get blocked tasks
        blocked_ids = set(self.get_blocked_tasks())
        
        # Update type to 'blocked' for blocked tasks
        for item in active_items:
            if item["id"] in blocked_ids and item.get("type") != "milestone":
                item["type"] = "blocked"

        lines = ["[Your active task list was preserved across context compression]"]
        for item in active_items:
            marker = markers.get(item["status"], "[?]")
            priority = item.get("priority", "medium")
            task_type = item.get("type", "task")
            
            priority_mark = priority_markers.get(priority, "!")
            type_mark = type_markers.get(task_type, "")
            
            # Format: [status] priority id. content (status) [type] [assigned] [deps]
            parts = [f"- {marker} {priority_mark} {item['id']}. {item['content']} ({item['status']})"]
            
            if type_mark:
                parts.append(type_mark)
            
            assigned = item.get("assigned_to")
            if assigned:
                parts.append(f"@{assigned}")
            
            depends_on = item.get("depends_on", []) or []
            if depends_on:
                parts.append(f"deps:[{','.join(depends_on)}]")
            
            lines.append(" ".join(parts))

        return "\n".join(lines)

    @staticmethod
    def _validate_depends_on(value: Any) -> Optional[List[str]]:
        """Validate and normalize depends_on field."""
        if value is None:
            return None
        if isinstance(value, str):
            # Allow comma-separated string
            deps = [d.strip() for d in value.split(",") if d.strip()]
            return deps if deps else None
        if isinstance(value, list):
            deps = [str(d).strip() for d in value if str(d).strip()]
            return deps if deps else None
        return None

    @staticmethod
    def _validate(item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize a todo item.

        Ensures required fields exist and status is valid.
        Returns a clean dict with validated fields.
        """
        item_id = str(item.get("id", "")).strip()
        if not item_id:
            item_id = "?"

        content = str(item.get("content", "")).strip()
        if not content:
            content = "(no description)"

        status = str(item.get("status", "pending")).strip().lower()
        if status not in VALID_STATUSES:
            status = "pending"

        # New fields with defaults
        priority = str(item.get("priority", "medium")).strip().lower()
        if priority not in VALID_PRIORITIES:
            priority = "medium"

        task_type = str(item.get("type", "task")).strip().lower()
        if task_type not in VALID_TYPES:
            task_type = "task"

        # Handle depends_on
        depends_on = TodoStore._validate_depends_on(item.get("depends_on"))

        # Handle assigned_to
        assigned_to = item.get("assigned_to")
        if assigned_to:
            assigned_to = str(assigned_to).strip() or None
        else:
            assigned_to = None

        # Handle created_at - preserve existing or create new
        created_at = item.get("created_at")
        if created_at:
            # Validate it's a valid ISO format
            try:
                datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
            except (ValueError, TypeError):
                created_at = datetime.now().isoformat()
        else:
            created_at = datetime.now().isoformat()

        result = {
            "id": item_id,
            "content": content,
            "status": status,
            "priority": priority,
            "type": task_type,
            "created_at": created_at,
        }
        
        # Only include optional fields if they have values
        if depends_on:
            result["depends_on"] = depends_on
        if assigned_to:
            result["assigned_to"] = assigned_to

        return result

    @staticmethod
    def _dedupe_by_id(todos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collapse duplicate ids, keeping the last occurrence in its position."""
        last_index: Dict[str, int] = {}
        for i, item in enumerate(todos):
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
        JSON string with the full current list, summary metadata, and dependency info.
    """
    if store is None:
        return tool_error("TodoStore not initialized")

    if todos is not None:
        items = store.write(todos, merge)
    else:
        items = store.read()

    # Build summary counts
    pending = sum(1 for i in items if i["status"] == "pending")
    in_progress = sum(1 for i in items if i["status"] == "in_progress")
    completed = sum(1 for i in items if i["status"] == "completed")
    cancelled = sum(1 for i in items if i["status"] == "cancelled")

    # Priority counts
    priority_counts = {
        "low": sum(1 for i in items if i.get("priority") == "low"),
        "medium": sum(1 for i in items if i.get("priority") == "medium"),
        "high": sum(1 for i in items if i.get("priority") == "high"),
        "critical": sum(1 for i in items if i.get("priority") == "critical"),
    }

    # Type counts
    type_counts = {
        "task": sum(1 for i in items if i.get("type") == "task"),
        "milestone": sum(1 for i in items if i.get("type") == "milestone"),
        "blocked": sum(1 for i in items if i.get("type") == "blocked"),
    }

    # Assignment summary
    assignments = {}
    for item in items:
        assigned = item.get("assigned_to")
        if assigned:
            assignments[assigned] = assignments.get(assigned, 0) + 1

    # Dependency graph
    dependency_graph = store.build_dependency_graph()
    blocked_tasks = store.get_blocked_tasks()

    return json.dumps({
        "todos": items,
        "summary": {
            "total": len(items),
            "pending": pending,
            "in_progress": in_progress,
            "completed": completed,
            "cancelled": cancelled,
            "by_priority": priority_counts,
            "by_type": type_counts,
            "assignments": assignments,
            "blocked_count": len(blocked_tasks),
        },
        "dependency_info": {
            "blocked_tasks": blocked_tasks,
            "graph": dependency_graph,
        },
    }, ensure_ascii=False, default=str)


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
        "Each item: {id, content, status, depends_on?, assigned_to?, priority?, type?}\n"
        "- id: unique string identifier\n"
        "- content: task description\n"
        "- status: pending|in_progress|completed|cancelled\n"
        "- depends_on: array of task ids this depends on (optional)\n"
        "- assigned_to: agent name for delegation (optional)\n"
        "- priority: low|medium|high|critical (default: medium)\n"
        "- type: task|milestone|blocked (default: task)\n\n"
        "List order is priority. Only ONE item in_progress at a time.\n"
        "Mark items completed immediately when done. If something fails, "
        "cancel it and add a revised item.\n\n"
        "Dependencies: A task with incomplete dependencies will be auto-marked as 'blocked'.\n\n"
        "Always returns the full current list with dependency graph."
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
                        "depends_on": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of task ids this task depends on. Task will be blocked until all dependencies are completed."
                        },
                        "assigned_to": {
                            "type": "string",
                            "description": "Agent or person assigned to this task"
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["low", "medium", "high", "critical"],
                            "description": "Task priority level (default: medium)"
                        },
                        "type": {
                            "type": "string",
                            "enum": ["task", "milestone", "blocked"],
                            "description": "Task type (default: task). 'blocked' is auto-set when dependencies are incomplete."
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