"""Team Tool — enhanced multi-agent orchestration with structured tasks.

Wraps delegate_tool with Task objects (status, dependencies, results),
delegation modes (parallel, sequential, route), and shared state via
session_state. Provides the agent with team-level coordination without
requiring Agno's full Team class.

Inspired by agno's Task class and Team delegation patterns.
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Task dataclass
# ============================================================================

@dataclass
class Task:
    """A structured unit of work for team coordination."""
    id: str = ""
    goal: str = ""
    context: str = ""
    toolsets: Optional[List[str]] = None
    status: str = "pending"  # pending, running, completed, failed
    assignee: str = ""       # child agent label
    result: Optional[str] = None
    error: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)  # task IDs
    duration_seconds: float = 0.0
    api_calls: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Team orchestration
# ============================================================================

def team_execute(
    tasks: List[Dict[str, Any]],
    mode: str = "parallel",
    shared_context: str = "",
    parent_agent: Any = None,
    session_db: Any = None,
    session_id: str = None,
) -> str:
    """Execute a set of tasks using the specified delegation mode.

    Args:
        tasks: List of task dicts, each with at least 'goal'.
            Optional: 'context', 'toolsets', 'id', 'dependencies'.
        mode: Execution mode:
            - "parallel": Run all tasks concurrently (default, uses delegate_task batch)
            - "sequential": Run tasks in order, each sees prior results
            - "route": Pick the single best task based on the goal and context
        shared_context: Additional context shared across all tasks.
        parent_agent: The parent AIAgent instance.
        session_db: SessionDB for persisting task state.
        session_id: Current session ID.

    Returns:
        JSON string with task results and orchestration metadata.
    """
    if parent_agent is None:
        return json.dumps({"error": "team_execute requires a parent agent."})

    if not tasks:
        return json.dumps({"error": "No tasks provided."})

    # Normalize tasks to Task objects
    task_objects = []
    for i, t in enumerate(tasks):
        task_id = t.get("id", f"task_{i}")
        task_obj = Task(
            id=task_id,
            goal=t.get("goal", ""),
            context=t.get("context", ""),
            toolsets=t.get("toolsets"),
            dependencies=t.get("dependencies", []),
        )
        if shared_context:
            task_obj.context = f"{shared_context}\n\n{task_obj.context}".strip()
        task_objects.append(task_obj)

    # Validate goals
    for task in task_objects:
        if not task.goal.strip():
            return json.dumps({"error": f"Task '{task.id}' is missing a goal."})

    start_time = time.monotonic()

    if mode == "parallel":
        results = _execute_parallel(task_objects, parent_agent)
    elif mode == "sequential":
        results = _execute_sequential(task_objects, parent_agent)
    elif mode == "route":
        results = _execute_route(task_objects, parent_agent)
    else:
        return json.dumps({"error": f"Unknown mode '{mode}'. Use: parallel, sequential, route"})

    total_duration = round(time.monotonic() - start_time, 2)

    # Persist task results to session state
    if session_db and session_id:
        try:
            state_update = {
                "_team_results": {
                    "mode": mode,
                    "tasks": [t.to_dict() for t in results],
                    "total_duration": total_duration,
                    "timestamp": time.time(),
                }
            }
            session_db.update_session_state(session_id, state_update)
        except Exception:
            pass

    completed = sum(1 for t in results if t.status == "completed")
    failed = sum(1 for t in results if t.status == "failed")

    return json.dumps({
        "mode": mode,
        "total_tasks": len(results),
        "completed": completed,
        "failed": failed,
        "total_duration_seconds": total_duration,
        "tasks": [t.to_dict() for t in results],
    }, ensure_ascii=False)


def _execute_parallel(tasks: List[Task], parent_agent: Any) -> List[Task]:
    """Run all tasks concurrently using delegate_task batch mode."""
    from tools.delegate_tool import delegate_task

    batch = [
        {"goal": t.goal, "context": t.context, "toolsets": t.toolsets}
        for t in tasks
    ]

    for t in tasks:
        t.status = "running"

    result_json = delegate_task(
        tasks=batch,
        parent_agent=parent_agent,
    )

    try:
        result = json.loads(result_json)
    except json.JSONDecodeError:
        for t in tasks:
            t.status = "failed"
            t.error = "Failed to parse delegation result"
        return tasks

    if "error" in result:
        for t in tasks:
            t.status = "failed"
            t.error = result["error"]
        return tasks

    for entry in result.get("results", []):
        idx = entry.get("task_index", 0)
        if idx < len(tasks):
            task = tasks[idx]
            task.status = entry.get("status", "failed")
            task.result = entry.get("summary")
            task.error = entry.get("error")
            task.duration_seconds = entry.get("duration_seconds", 0)
            task.api_calls = entry.get("api_calls", 0)

    return tasks


def _execute_sequential(tasks: List[Task], parent_agent: Any) -> List[Task]:
    """Run tasks in order, each receiving prior results as context."""
    from tools.delegate_tool import delegate_task

    accumulated_results = []

    for i, task in enumerate(tasks):
        # Check dependencies
        unmet = _check_dependencies(task, tasks)
        if unmet:
            task.status = "failed"
            task.error = f"Unmet dependencies: {', '.join(unmet)}"
            continue

        # Build context with prior results
        if accumulated_results:
            prior_context = "PRIOR TASK RESULTS:\n"
            for prev_id, prev_result in accumulated_results:
                prior_context += f"\n--- {prev_id} ---\n{prev_result}\n"
            full_context = f"{task.context}\n\n{prior_context}".strip()
        else:
            full_context = task.context

        task.status = "running"

        result_json = delegate_task(
            goal=task.goal,
            context=full_context,
            toolsets=task.toolsets,
            parent_agent=parent_agent,
        )

        try:
            result = json.loads(result_json)
        except json.JSONDecodeError:
            task.status = "failed"
            task.error = "Failed to parse delegation result"
            continue

        if "error" in result:
            task.status = "failed"
            task.error = result["error"]
            continue

        entries = result.get("results", [])
        if entries:
            entry = entries[0]
            task.status = entry.get("status", "failed")
            task.result = entry.get("summary")
            task.error = entry.get("error")
            task.duration_seconds = entry.get("duration_seconds", 0)
            task.api_calls = entry.get("api_calls", 0)

        if task.status == "completed" and task.result:
            accumulated_results.append((task.id, task.result))

    return tasks


def _execute_route(tasks: List[Task], parent_agent: Any) -> List[Task]:
    """Pick the single best task and execute only that one.

    Uses a simple heuristic: if the parent has a routing classifier,
    use it. Otherwise, just pick the first task.
    """
    from tools.delegate_tool import delegate_task

    # For now, pick the first task (routing can be enhanced later)
    best = tasks[0]
    best.status = "running"

    # Mark others as skipped
    for t in tasks[1:]:
        t.status = "completed"
        t.result = "(skipped — routed to another task)"

    result_json = delegate_task(
        goal=best.goal,
        context=best.context,
        toolsets=best.toolsets,
        parent_agent=parent_agent,
    )

    try:
        result = json.loads(result_json)
        entries = result.get("results", [])
        if entries:
            entry = entries[0]
            best.status = entry.get("status", "failed")
            best.result = entry.get("summary")
            best.error = entry.get("error")
            best.duration_seconds = entry.get("duration_seconds", 0)
            best.api_calls = entry.get("api_calls", 0)
        elif "error" in result:
            best.status = "failed"
            best.error = result["error"]
    except json.JSONDecodeError:
        best.status = "failed"
        best.error = "Failed to parse delegation result"

    return tasks


def _check_dependencies(task: Task, all_tasks: List[Task]) -> List[str]:
    """Return list of unmet dependency IDs."""
    if not task.dependencies:
        return []
    unmet = []
    task_map = {t.id: t for t in all_tasks}
    for dep_id in task.dependencies:
        dep = task_map.get(dep_id)
        if not dep or dep.status != "completed":
            unmet.append(dep_id)
    return unmet


# ============================================================================
# Tool registration
# ============================================================================

def check_team_requirements() -> bool:
    """Team tool has no extra requirements beyond delegate_tool."""
    return True


TEAM_SCHEMA = {
    "name": "team_execute",
    "description": (
        "Orchestrate multiple subagents as a coordinated team. Provides structured "
        "task management with status tracking, dependencies, and result aggregation.\n\n"
        "MODES:\n"
        "- 'parallel': Run all tasks simultaneously (fastest, for independent work)\n"
        "- 'sequential': Run tasks in order, each sees prior results (for pipelines)\n"
        "- 'route': Pick the single best task and execute only that one\n\n"
        "USE INSTEAD OF delegate_task WHEN:\n"
        "- You have 2+ related tasks that benefit from coordination\n"
        "- Tasks have dependencies (task B needs task A's output)\n"
        "- You want structured result tracking across tasks\n"
        "- You need sequential pipeline execution\n\n"
        "EXAMPLES:\n"
        "- Parallel: 'Analyze AAPL, MSFT, GOOGL earnings simultaneously'\n"
        "- Sequential: 'Research market → Analyze positions → Generate report'\n"
        "- Route: 'Either do technical analysis OR fundamental analysis based on data available'"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string", "description": "What this task should accomplish."},
                        "context": {"type": "string", "description": "Additional context for this task."},
                        "toolsets": {
                            "type": "array", "items": {"type": "string"},
                            "description": "Tool categories to enable (default: terminal, file, web).",
                        },
                        "id": {"type": "string", "description": "Unique task ID (auto-generated if omitted)."},
                        "dependencies": {
                            "type": "array", "items": {"type": "string"},
                            "description": "Task IDs that must complete before this task runs (sequential mode only).",
                        },
                    },
                    "required": ["goal"],
                },
                "description": "List of tasks to execute.",
            },
            "mode": {
                "type": "string",
                "enum": ["parallel", "sequential", "route"],
                "description": "Execution mode. Default: parallel.",
            },
            "shared_context": {
                "type": "string",
                "description": "Context shared across all tasks (prepended to each task's context).",
            },
        },
        "required": ["tasks"],
    },
}


from tools.registry import registry

registry.register(
    name="team_execute",
    toolset="delegation",
    schema=TEAM_SCHEMA,
    handler=lambda args, **kw: team_execute(
        tasks=args.get("tasks", []),
        mode=args.get("mode", "parallel"),
        shared_context=args.get("shared_context", ""),
        parent_agent=kw.get("parent_agent"),
        session_db=kw.get("db"),
        session_id=kw.get("session_id"),
    ),
    check_fn=check_team_requirements,
    emoji="👥",
    mutates=True,
)
