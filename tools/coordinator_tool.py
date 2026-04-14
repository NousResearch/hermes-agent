"""Coordinator tool for structured planning and execution."""

from __future__ import annotations

from agent.coordinator import Coordinator
from agent.task_manager import TaskManager
from tools.registry import registry, tool_error, tool_result


COORDINATE_SCHEMA = {
    "name": "coordinate",
    "description": "Create, execute, or summarize a structured plan for a task.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["plan", "execute", "summarize"],
                "description": "Whether to create a plan, execute it, or summarize it.",
            },
            "task_description": {
                "type": "string",
                "description": "The overall task to coordinate.",
            },
            "subtasks": {
                "type": "array",
                "description": "Optional structured subtasks for the plan.",
                "items": {
                    "type": "object",
                    "additionalProperties": True,
                },
            },
        },
        "required": ["action", "task_description"],
    },
}


def coordinate_handler(args: dict, **kwargs) -> str:
    """Handle coordination requests."""
    action = args.get("action")
    task_description = args.get("task_description")
    subtasks = args.get("subtasks")
    coordinator = Coordinator()

    try:
        plan = coordinator.plan(task_description=task_description, subtasks=subtasks)

        if action == "plan":
            return tool_result(plan)

        if action == "execute":
            task_manager = TaskManager(session_id=kwargs.get("session_id") or "default")
            return tool_result(coordinator.execute_plan(plan, task_manager))

        if action == "summarize":
            return tool_result(summary=coordinator.summarize(plan))
    except (KeyError, TypeError, ValueError) as exc:
        return tool_error(str(exc))

    return tool_error(f"Unsupported coordinate action: {action}")


registry.register(
    name="coordinate",
    toolset="coordinator",
    schema=COORDINATE_SCHEMA,
    handler=coordinate_handler,
    description="Coordinate structured plans and execution",
    emoji="🧭",
    mutates_agent_state=True,
)
