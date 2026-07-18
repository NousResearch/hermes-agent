"""
Validated todo tool path.

This module is intentionally isolated so importing ``tools.todo_tool`` never
pulls in Pydantic or the relay-style validation helper unless the validated
path is explicitly enabled. This preserves backward compatibility for existing
agents/tests/toolsets that import the core todo tool.
"""

from typing import Any, Dict, List, Optional

import json

from pydantic import BaseModel

from tools.todo_tool import TodoStore, check_todo_requirements
from tools.validation_helper import Failure, Success, relay_success, relay_failure


class _TodoItem(BaseModel):
    id: str
    content: str
    status: str = "pending"


class _TodoResultItem(BaseModel):
    id: str
    content: str
    status: str


class _TodoSummary(BaseModel):
    total: int
    pending: int
    in_progress: int
    completed: int
    cancelled: int


def _build_validated_todo_result(
    todos: Optional[List[Dict[str, Any]]],
    merge: bool,
    store: Optional[TodoStore],
) -> Dict[str, Any]:
    if store is None:
        return relay_failure("TodoStore not initialized", code="todo_store_unavailable")

    items = (
        store.write(
            [
                {
                    "id": it.get("id", ""),
                    "content": it.get("content", ""),
                    "status": it.get("status", "pending"),
                }
                for it in todos
            ],
            merge,
        )
        if todos is not None
        else store.read()
    )

    summary = {
        "total": len(items),
        "pending": sum(1 for i in items if i["status"] == "pending"),
        "in_progress": sum(1 for i in items if i["status"] == "in_progress"),
        "completed": sum(1 for i in items if i["status"] == "completed"),
        "cancelled": sum(1 for i in items if i["status"] == "cancelled"),
    }

    try:
        payload = {
            "todos": items,
            "summary": summary,
            "validated": True,
        }
        return Success(data=payload).model_dump(mode="json")
    except Exception as exc:
        return relay_failure(
            f"invalid todo result: {exc}", code="validation_error"
        )


def validated_todo_tool(
    todos: Optional[List[Dict[str, Any]]] = None,
    merge: bool = False,
    store: Optional[TodoStore] = None,
) -> str:
    payload = _build_validated_todo_result(todos, merge, store)
    if payload.get("kind") == "failure":
        return relay_failure(
            payload.get("error", "unknown"), code=payload.get("code")
        )
    return relay_success(payload["data"])


def build_validated_todo_tool(store: Optional[TodoStore] = None) -> Dict[str, Any]:
    return {
        "name": "todo_validated",
        "description": (
            "Validated session todo list. Use for complex tasks with 3+ "
            "steps. Accepts a JSON array of todo items when writing, or omit "
            "``todos`` to read. Returns a zodern:relay-style success envelope "
            "including ``validated: true``."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "merge": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "true: update existing items by id and append new "
                        "ones. false (default): replace the entire list."
                    ),
                },
                "todos": {
                    "type": "array",
                    "description": "Task items to write. Omit to read.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "content": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": [
                                    "pending",
                                    "in_progress",
                                    "completed",
                                    "cancelled",
                                ],
                            },
                        },
                        "required": ["id", "content", "status"],
                    },
                },
            },
            "required": [],
        },
        "handler": lambda args, **kw: validated_todo_tool(
            todos=args.get("todos"),
            merge=args.get("merge", False),
            store=kw.get("store", store),
        ),
        "check_fn": check_todo_requirements,
    }
