"""Structured task tools for Hermes Agent."""

import json

from agent.task_manager import TaskManager
from tools.registry import registry, tool_error


TASK_MANAGER = TaskManager()


def _error_response(exc: Exception) -> str:
    message = exc.args[0] if getattr(exc, "args", None) else str(exc)
    return tool_error(message, success=False)


TASK_CREATE_SCHEMA = {
    "name": "task_create",
    "description": "Create a structured persistent task.",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Short task title.",
            },
            "kind": {
                "type": "string",
                "enum": ["local", "delegated", "system"],
                "description": "Task kind. Defaults to local.",
            },
            "parent_task_id": {
                "type": "string",
                "description": "Optional parent task identifier.",
            },
            "assignee": {
                "type": "string",
                "enum": ["local", "delegate", "system"],
                "description": "Optional assignee type.",
            },
            "metadata": {
                "type": "object",
                "description": "Optional structured metadata.",
                "additionalProperties": True,
            },
        },
        "required": ["title"],
    },
}

TASK_UPDATE_SCHEMA = {
    "name": "task_update",
    "description": "Update a structured task status, result summary, or metadata.",
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "Task identifier.",
            },
            "status": {
                "type": "string",
                "enum": ["pending", "running", "completed", "failed", "cancelled"],
                "description": "Optional new status.",
            },
            "result_summary": {
                "type": "string",
                "description": "Optional result summary.",
            },
            "metadata": {
                "type": "object",
                "description": "Optional replacement metadata object.",
                "additionalProperties": True,
            },
        },
        "required": ["task_id"],
    },
}

TASK_LIST_SCHEMA = {
    "name": "task_list",
    "description": "List structured tasks, optionally filtered by status or kind.",
    "parameters": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["pending", "running", "completed", "failed", "cancelled"],
                "description": "Optional status filter.",
            },
            "kind": {
                "type": "string",
                "enum": ["local", "delegated", "system"],
                "description": "Optional kind filter.",
            },
        },
        "required": [],
    },
}

TASK_GET_SCHEMA = {
    "name": "task_get",
    "description": "Get a structured task by identifier.",
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "Task identifier.",
            },
        },
        "required": ["task_id"],
    },
}

TASK_CANCEL_SCHEMA = {
    "name": "task_cancel",
    "description": "Cancel a structured task.",
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "Task identifier.",
            },
        },
        "required": ["task_id"],
    },
}


def _handle_task_create(args: dict, **kwargs) -> str:
    try:
        task = TASK_MANAGER.create(
            title=args.get("title"),
            kind=args.get("kind", "local"),
            parent_task_id=args.get("parent_task_id"),
            assignee=args.get("assignee"),
            metadata=args.get("metadata"),
        )
        return json.dumps(task, ensure_ascii=False)
    except (KeyError, ValueError) as exc:
        return _error_response(exc)


def _handle_task_update(args: dict, **kwargs) -> str:
    try:
        task = TASK_MANAGER.update(
            task_id=args.get("task_id"),
            status=args.get("status"),
            result_summary=args.get("result_summary"),
            metadata=args.get("metadata"),
        )
        return json.dumps(task, ensure_ascii=False)
    except (KeyError, ValueError) as exc:
        return _error_response(exc)


def _handle_task_list(args: dict, **kwargs) -> str:
    try:
        tasks = TASK_MANAGER.list(
            status=args.get("status"),
            kind=args.get("kind"),
        )
        return json.dumps({"tasks": tasks}, ensure_ascii=False)
    except ValueError as exc:
        return _error_response(exc)


def _handle_task_get(args: dict, **kwargs) -> str:
    task = TASK_MANAGER.get(args.get("task_id"))
    return json.dumps({"task": task}, ensure_ascii=False)


def _handle_task_cancel(args: dict, **kwargs) -> str:
    try:
        task = TASK_MANAGER.cancel(args.get("task_id"))
        return json.dumps(task, ensure_ascii=False)
    except (KeyError, ValueError) as exc:
        return _error_response(exc)


registry.register(
    name="task_create",
    toolset="task",
    schema=TASK_CREATE_SCHEMA,
    handler=_handle_task_create,
    description="Create a structured persistent task",
    emoji="🗂️",
    mutates_agent_state=True,
    allowed_in_plan_mode_default=False,
)

registry.register(
    name="task_update",
    toolset="task",
    schema=TASK_UPDATE_SCHEMA,
    handler=_handle_task_update,
    description="Update a structured task",
    emoji="🗂️",
    mutates_agent_state=True,
    allowed_in_plan_mode_default=False,
)

registry.register(
    name="task_list",
    toolset="task",
    schema=TASK_LIST_SCHEMA,
    handler=_handle_task_list,
    description="List structured tasks",
    emoji="🗂️",
    mutates_agent_state=False,
    allowed_in_plan_mode_default=True,
)

registry.register(
    name="task_get",
    toolset="task",
    schema=TASK_GET_SCHEMA,
    handler=_handle_task_get,
    description="Get a structured task",
    emoji="🗂️",
    mutates_agent_state=False,
    allowed_in_plan_mode_default=True,
)

registry.register(
    name="task_cancel",
    toolset="task",
    schema=TASK_CANCEL_SCHEMA,
    handler=_handle_task_cancel,
    description="Cancel a structured task",
    emoji="🗂️",
    mutates_agent_state=True,
    allowed_in_plan_mode_default=False,
)
