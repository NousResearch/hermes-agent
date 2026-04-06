"""async_delegate — non-blocking background agent execution.

Tools:
  delegate_task_async  — spawn a background agent, return task_id immediately
  check_task           — non-blocking status + buffered output preview
  collect_task         — wait for completion and return full result
  steer_task           — inject a steering message into a running task
  cancel_task          — stop a running task
  list_tasks           — list all async tasks for this session
"""

import dataclasses
import json
import logging
import threading
import time
from typing import Any, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

# Max output lines buffered per task (drop oldest when exceeded)
_MAX_OUTPUT_LINES = 200
# Max chars per output line
_MAX_LINE_CHARS = 500


# =============================================================================
# AsyncTask dataclass
# =============================================================================

@dataclasses.dataclass
class AsyncTask:
    task_id: str
    goal: str
    status: str = "running"  # running | completed | failed | cancelled
    started_at: float = dataclasses.field(default_factory=time.monotonic)
    completed_at: Optional[float] = None
    output_lines: list = dataclasses.field(default_factory=list)  # captured output
    result: Optional[dict] = None  # final run_conversation() result dict
    error: Optional[str] = None
    thread: Optional[threading.Thread] = dataclasses.field(default=None, repr=False)
    child_agent: Any = dataclasses.field(default=None, repr=False)
    done_event: threading.Event = dataclasses.field(default_factory=threading.Event)

    @property
    def elapsed(self) -> float:
        if self.completed_at:
            return self.completed_at - self.started_at
        return time.monotonic() - self.started_at


# =============================================================================
# Task registry helpers (per-parent-agent)
# =============================================================================

def _get_task_registry(parent_agent):
    """Return (tasks_dict, lock) from parent_agent, creating them if absent."""
    if not hasattr(parent_agent, '_async_tasks'):
        setattr(parent_agent, '_async_tasks', {})
        setattr(parent_agent, '_async_tasks_lock', threading.Lock())
    return parent_agent._async_tasks, parent_agent._async_tasks_lock


# =============================================================================
# Tool implementations
# =============================================================================

def delegate_task_async(
    goal: str,
    context: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    toolsets: Optional[List[str]] = None,
    max_iterations: int = 50,
    skill: Optional[str] = None,
    parent_agent=None,
) -> str:
    """Spawn a background agent and return a task_id immediately."""
    try:
        if parent_agent is None:
            return json.dumps({"error": "delegate_task_async requires a parent agent context."})

        if not goal or not goal.strip():
            return json.dumps({"error": "goal is required."})

        # Import delegation helpers
        from tools.delegate_tool import (
            _load_config,
            _resolve_delegation_credentials,
            _build_child_agent,
            _load_skill_for_subagent,
            _build_child_system_prompt,
            _strip_blocked_tools,
            DEFAULT_TOOLSETS,
            MAX_DEPTH,
        )

        # Depth limit — async tasks also count against the delegation depth
        depth = getattr(parent_agent, '_delegate_depth', 0)
        if depth >= MAX_DEPTH:
            return json.dumps({
                "error": (
                    f"Delegation depth limit reached ({MAX_DEPTH}). "
                    "Subagents cannot spawn further subagents."
                )
            })

        # Generate a unique task_id
        task_id = f"async_{int(time.time())}_{uuid4().hex[:6]}"

        # Load config + credentials (same as delegate_task)
        cfg = _load_config()
        default_max_iter = cfg.get("max_iterations", max_iterations)
        effective_max_iter = max_iterations or default_max_iter

        # Load skill if requested
        skill_extra_prompt = None
        skill_model = None
        skill_provider = None
        if skill:
            loaded = _load_skill_for_subagent(skill)
            if loaded:
                skill_model = loaded.get("model")
                skill_provider = loaded.get("provider")
                content = loaded.get("content", "")
                if content:
                    skill_extra_prompt = f"# Skill: {skill}\n\n{content}"

        call_model = model or skill_model
        call_provider = provider or skill_provider

        # Resolve credentials (may raise ValueError)
        try:
            creds = _resolve_delegation_credentials(
                cfg, parent_agent,
                override_model=call_model,
                override_provider=call_provider,
            )
        except ValueError as exc:
            return json.dumps({"error": str(exc)})

        # Merge skill context into goal message
        full_goal = goal
        effective_context = context or ""
        if skill_extra_prompt:
            effective_context = skill_extra_prompt + ("\n\n" + effective_context if effective_context else "")

        # Save parent tool names before child construction mutates the global
        import model_tools as _model_tools
        _parent_tool_names = list(_model_tools._last_resolved_tool_names)

        # Build child agent (main thread, thread-safe)
        try:
            child = _build_child_agent(
                task_index=0,
                goal=full_goal,
                context=effective_context or None,
                toolsets=toolsets,
                model=creds["model"],
                max_iterations=effective_max_iter,
                parent_agent=parent_agent,
                override_provider=creds.get("provider"),
                override_base_url=creds.get("base_url"),
                override_api_key=creds.get("api_key"),
                override_api_mode=creds.get("api_mode"),
            )
        finally:
            # Restore parent tool names immediately after child build
            _model_tools._last_resolved_tool_names = _parent_tool_names

        # Save parent tool names on the child for restoration after thread runs
        child._delegate_saved_tool_names = _parent_tool_names

        # Create the task record (output_lines will be populated by _print_fn)
        task = AsyncTask(
            task_id=task_id,
            goal=goal,
            status="running",
            child_agent=child,
        )

        # Wire up output capture — replaces the child's print function
        def _capture_print(msg):
            line = str(msg)[:_MAX_LINE_CHARS]
            task.output_lines.append(line)
            # Cap at _MAX_OUTPUT_LINES: drop oldest
            if len(task.output_lines) > _MAX_OUTPUT_LINES:
                del task.output_lines[0]

        child._print_fn = _capture_print
        child.quiet_mode = True

        # Store in parent's task registry
        tasks_dict, lock = _get_task_registry(parent_agent)
        with lock:
            tasks_dict[task_id] = task

        # Build the user message (goal + context combined)
        if effective_context:
            full_message = f"{full_goal}\n\nCONTEXT:\n{effective_context}"
        else:
            full_message = full_goal

        # Thread worker
        def _run_task():
            try:
                result = child.run_conversation(user_message=full_message)
                task.result = result
                task.status = "completed" if result.get("completed") else "failed"
            except Exception as exc:
                task.status = "failed"
                task.error = str(exc)
                logger.exception("async task %s failed", task_id)
            finally:
                task.completed_at = time.monotonic()
                task.done_event.set()
                # Unregister child from parent's active children
                if hasattr(parent_agent, '_active_children'):
                    try:
                        lk = getattr(parent_agent, '_active_children_lock', None)
                        if lk:
                            with lk:
                                parent_agent._active_children.remove(child)
                        else:
                            parent_agent._active_children.remove(child)
                    except (ValueError, AttributeError):
                        pass
                # Restore parent tool names (best effort — thread may outlive request)
                try:
                    saved = getattr(child, "_delegate_saved_tool_names", None)
                    if isinstance(saved, list):
                        _model_tools._last_resolved_tool_names = list(saved)
                except Exception:
                    pass

        t = threading.Thread(target=_run_task, daemon=True, name=f"async-task-{task_id}")
        task.thread = t
        t.start()

        return json.dumps({
            "task_id": task_id,
            "status": "running",
            "message": f"Background task started. Use check_task('{task_id}') to monitor.",
        })

    except Exception as exc:
        logger.exception("delegate_task_async failed")
        return json.dumps({"error": str(exc)})


def check_task(task_id: str, parent_agent=None) -> str:
    """Non-blocking status check + buffered output preview."""
    try:
        if parent_agent is None:
            return json.dumps({"error": "check_task requires a parent agent context."})

        tasks_dict, lock = _get_task_registry(parent_agent)
        with lock:
            task = tasks_dict.get(task_id)

        if task is None:
            return json.dumps({"error": f"Task not found: {task_id}"})

        # Last 10 lines as preview
        preview_lines = task.output_lines[-10:]
        preview = "\n".join(preview_lines)

        return json.dumps({
            "task_id": task_id,
            "status": task.status,
            "elapsed": round(task.elapsed, 2),
            "output_preview": preview,
            "output_lines": len(task.output_lines),
        })

    except Exception as exc:
        logger.exception("check_task failed")
        return json.dumps({"error": str(exc)})


def collect_task(task_id: str, timeout: int = 300, parent_agent=None) -> str:
    """Block until the task completes (or timeout), then return full result."""
    try:
        if parent_agent is None:
            return json.dumps({"error": "collect_task requires a parent agent context."})

        tasks_dict, lock = _get_task_registry(parent_agent)
        with lock:
            task = tasks_dict.get(task_id)

        if task is None:
            return json.dumps({"error": f"Task not found: {task_id}"})

        # Wait for completion
        completed = task.done_event.wait(timeout=timeout)

        output_text = "\n".join(task.output_lines)
        result_dict = task.result or {}
        summary = result_dict.get("final_response") or ""

        payload = {
            "task_id": task_id,
            "status": task.status,
            "elapsed": round(task.elapsed, 2),
            "timed_out": not completed,
            "output": output_text,
            "summary": summary,
            "output_lines": len(task.output_lines),
        }
        if task.error:
            payload["error"] = task.error

        return json.dumps(payload, ensure_ascii=False)

    except Exception as exc:
        logger.exception("collect_task failed")
        return json.dumps({"error": str(exc)})


def steer_task(task_id: str, message: str, parent_agent=None) -> str:
    """Inject a steering message into a running task."""
    try:
        if parent_agent is None:
            return json.dumps({"error": "steer_task requires a parent agent context."})

        tasks_dict, lock = _get_task_registry(parent_agent)
        with lock:
            task = tasks_dict.get(task_id)

        if task is None:
            return json.dumps({"error": f"Task not found: {task_id}"})

        if task.status != "running":
            return json.dumps({
                "ok": False,
                "error": f"Task {task_id} is not running (status: {task.status}).",
            })

        child = task.child_agent
        if child is None:
            return json.dumps({"ok": False, "error": "No child agent attached to task."})

        si = getattr(child, '_steering_injection', None)
        if si is not None:
            si.put(message)
            return json.dumps({
                "ok": True,
                "message": f"Steering delivered to task {task_id}.",
            })
        else:
            return json.dumps({
                "ok": False,
                "error": "Child agent does not support steering injection.",
            })

    except Exception as exc:
        logger.exception("steer_task failed")
        return json.dumps({"error": str(exc)})


def cancel_task(task_id: str, parent_agent=None) -> str:
    """Cancel a running task by calling child_agent.interrupt()."""
    try:
        if parent_agent is None:
            return json.dumps({"error": "cancel_task requires a parent agent context."})

        tasks_dict, lock = _get_task_registry(parent_agent)
        with lock:
            task = tasks_dict.get(task_id)

        if task is None:
            return json.dumps({"error": f"Task not found: {task_id}"})

        if task.status != "running":
            return json.dumps({
                "ok": False,
                "error": f"Task {task_id} is not running (status: {task.status}).",
            })

        child = task.child_agent
        if child is not None and hasattr(child, 'interrupt'):
            child.interrupt("Cancelled by parent agent.")

        task.status = "cancelled"
        task.completed_at = time.monotonic()
        task.done_event.set()

        return json.dumps({
            "ok": True,
            "cancelled": task_id,
            "message": f"Task {task_id} cancelled.",
        })

    except Exception as exc:
        logger.exception("cancel_task failed")
        return json.dumps({"error": str(exc)})


def list_tasks(parent_agent=None) -> str:
    """List all async tasks for this session."""
    try:
        if parent_agent is None:
            return json.dumps({"tasks": [], "note": "No parent agent context."})

        tasks_dict, lock = _get_task_registry(parent_agent)
        with lock:
            snapshot = list(tasks_dict.values())

        result = []
        for t in snapshot:
            result.append({
                "task_id": t.task_id,
                "goal": t.goal[:60],
                "status": t.status,
                "elapsed": round(t.elapsed, 2),
                "output_lines": len(t.output_lines),
            })

        return json.dumps({"tasks": result})

    except Exception as exc:
        logger.exception("list_tasks failed")
        return json.dumps({"error": str(exc)})


# =============================================================================
# OpenAI Function-Calling Schemas
# =============================================================================

DELEGATE_TASK_ASYNC_SCHEMA = {
    "name": "delegate_task_async",
    "description": (
        "Spawn a background agent that runs independently — returns a task_id immediately. "
        "The parent agent continues working while the background agent runs. "
        "Use check_task() to monitor progress, collect_task() to get results, "
        "steer_task() to guide it, cancel_task() to stop it.\n\n"
        "The background agent gets its own isolated context, toolset, and terminal session. "
        "Only the final summary (via collect_task) enters your context window.\n\n"
        "WHEN TO USE:\n"
        "- Long-running tasks you want to run in parallel with other work\n"
        "- Tasks where you want to start multiple agents concurrently and collect results later\n"
        "- Background research or processing while you continue reasoning\n\n"
        "IMPORTANT:\n"
        "- Background agents have NO memory of your conversation. Pass all context explicitly.\n"
        "- Use collect_task(task_id) to wait for and retrieve the full result.\n"
        "- Use check_task(task_id) for non-blocking progress checks."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "The task for the background agent. Be specific and self-contained.",
            },
            "context": {
                "type": "string",
                "description": "Additional context the agent needs (file paths, constraints, background info).",
            },
            "model": {
                "type": "string",
                "description": "Model override (e.g. 'google/gemini-flash-1.5').",
            },
            "provider": {
                "type": "string",
                "description": "Provider override (e.g. 'openrouter').",
            },
            "toolsets": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Toolsets for the background agent. "
                    "Default: ['terminal', 'file', 'web']. "
                    "Common patterns: ['terminal', 'file'] for code work, ['web'] for research."
                ),
            },
            "max_iterations": {
                "type": "integer",
                "description": "Max tool-calling turns (default: 50).",
            },
            "skill": {
                "type": "string",
                "description": "Skill name to preload into the agent's context.",
            },
        },
        "required": ["goal"],
    },
}

CHECK_TASK_SCHEMA = {
    "name": "check_task",
    "description": (
        "Non-blocking status check for a background task. "
        "Returns current status (running/completed/failed/cancelled), elapsed time, "
        "and a preview of the last 10 output lines. Does NOT block — use collect_task() "
        "to wait for completion and get the full result."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The task_id returned by delegate_task_async.",
            },
        },
        "required": ["task_id"],
    },
}

COLLECT_TASK_SCHEMA = {
    "name": "collect_task",
    "description": (
        "Wait for a background task to complete and return its full result. "
        "Blocks until the task finishes or the timeout is reached. "
        "Returns the full output, summary, and status. "
        "Use check_task() first to verify the task is close to completion."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The task_id returned by delegate_task_async.",
            },
            "timeout": {
                "type": "integer",
                "description": "Max seconds to wait (default: 300). Returns partial result on timeout.",
            },
        },
        "required": ["task_id"],
    },
}

STEER_TASK_SCHEMA = {
    "name": "steer_task",
    "description": (
        "Inject a steering message into a running background task. "
        "The message is delivered to the agent's steering queue and will be "
        "picked up at the next opportunity. Use this to redirect the agent, "
        "provide new information, or request a course correction."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The task_id returned by delegate_task_async.",
            },
            "message": {
                "type": "string",
                "description": "The steering message to inject into the running task.",
            },
        },
        "required": ["task_id", "message"],
    },
}

CANCEL_TASK_SCHEMA = {
    "name": "cancel_task",
    "description": (
        "Stop a running background task. "
        "Sends an interrupt signal to the background agent and marks the task as cancelled. "
        "The agent will finish its current tool call before stopping."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The task_id returned by delegate_task_async.",
            },
        },
        "required": ["task_id"],
    },
}

LIST_TASKS_SCHEMA = {
    "name": "list_tasks",
    "description": (
        "List all background tasks spawned in this session. "
        "Returns task IDs, goals (truncated to 60 chars), status, elapsed time, "
        "and output line counts."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


# =============================================================================
# Registration
# =============================================================================

from tools.registry import registry


def _check_async_delegate_requirements() -> bool:
    """Async delegation has no external requirements — always available."""
    return True


registry.register(
    name="delegate_task_async",
    toolset="async_delegation",
    schema=DELEGATE_TASK_ASYNC_SCHEMA,
    handler=lambda args, **kw: delegate_task_async(
        goal=args.get("goal"),
        context=args.get("context"),
        model=args.get("model"),
        provider=args.get("provider"),
        toolsets=args.get("toolsets"),
        max_iterations=args.get("max_iterations", 50),
        skill=args.get("skill"),
        parent_agent=kw.get("parent_agent"),
    ),
    check_fn=_check_async_delegate_requirements,
    emoji="🔀",
)

registry.register(
    name="check_task",
    toolset="async_delegation",
    schema=CHECK_TASK_SCHEMA,
    handler=lambda args, **kw: check_task(
        task_id=args.get("task_id", ""),
        parent_agent=kw.get("parent_agent"),
    ),
    check_fn=_check_async_delegate_requirements,
    emoji="📊",
)

registry.register(
    name="collect_task",
    toolset="async_delegation",
    schema=COLLECT_TASK_SCHEMA,
    handler=lambda args, **kw: collect_task(
        task_id=args.get("task_id", ""),
        timeout=args.get("timeout", 300),
        parent_agent=kw.get("parent_agent"),
    ),
    check_fn=_check_async_delegate_requirements,
    emoji="📥",
)

registry.register(
    name="steer_task",
    toolset="async_delegation",
    schema=STEER_TASK_SCHEMA,
    handler=lambda args, **kw: steer_task(
        task_id=args.get("task_id", ""),
        message=args.get("message", ""),
        parent_agent=kw.get("parent_agent"),
    ),
    check_fn=_check_async_delegate_requirements,
    emoji="🎯",
)

registry.register(
    name="cancel_task",
    toolset="async_delegation",
    schema=CANCEL_TASK_SCHEMA,
    handler=lambda args, **kw: cancel_task(
        task_id=args.get("task_id", ""),
        parent_agent=kw.get("parent_agent"),
    ),
    check_fn=_check_async_delegate_requirements,
    emoji="🛑",
)

registry.register(
    name="list_tasks",
    toolset="async_delegation",
    schema=LIST_TASKS_SCHEMA,
    handler=lambda args, **kw: list_tasks(
        parent_agent=kw.get("parent_agent"),
    ),
    check_fn=_check_async_delegate_requirements,
    emoji="📋",
)
