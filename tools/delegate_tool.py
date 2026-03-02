#!/usr/bin/env python3
"""
Delegate Tool -- Subagent Architecture

Spawns child AIAgent instances with isolated context, restricted toolsets,
and their own terminal sessions. Supports single-task and batch (parallel)
modes. The parent blocks until all children complete.

Each child gets:
  - A fresh conversation (no parent history)
  - Its own task_id (own terminal session, file ops cache)
  - A restricted toolset (configurable, with blocked tools always stripped)
  - A focused system prompt built from the delegated goal + context

The parent's context only sees the delegation call and the summary result,
never the child's intermediate tool calls or reasoning.
"""

import contextlib
import io
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional


# Tools that children must never have access to
DELEGATE_BLOCKED_TOOLS = frozenset([
    "delegate_task",   # no recursive delegation
    "clarify",         # no user interaction
    "memory",          # no writes to shared MEMORY.md
    "send_message",    # no cross-platform side effects
    "execute_code",    # children should reason step-by-step, not write scripts
])

MAX_CONCURRENT_CHILDREN = 3
MAX_DEPTH = 2  # parent (0) -> child (1) -> grandchild rejected (2)
DEFAULT_MAX_ITERATIONS = 50
DEFAULT_TOOLSETS = ["terminal", "file", "web"]


def check_delegate_requirements() -> bool:
    """Delegation has no external requirements -- always available."""
    return True


def _build_child_system_prompt(goal: str, context: Optional[str] = None) -> str:
    """Build a focused system prompt for a child agent."""
    parts = [
        "You are a focused subagent working on a specific delegated task.",
        "",
        f"YOUR TASK:\n{goal}",
    ]
    if context and context.strip():
        parts.append(f"\nCONTEXT:\n{context}")
    parts.append(
        "\nComplete this task using the tools available to you. "
        "When finished, provide a clear, concise summary of:\n"
        "- What you did\n"
        "- What you found or accomplished\n"
        "- Any files you created or modified\n"
        "- Any issues encountered\n\n"
        "Be thorough but concise -- your response is returned to the "
        "parent agent as a summary."
    )
    return "\n".join(parts)


def _strip_blocked_tools(toolsets: List[str]) -> List[str]:
    """Remove toolsets that contain only blocked tools."""
    blocked_toolset_names = {
        "delegation", "clarify", "memory", "code_execution",
    }
    return [t for t in toolsets if t not in blocked_toolset_names]


def _build_child_progress_callback(task_index: int, parent_agent, task_count: int = 1) -> Optional[callable]:
    """Build a callback that relays child agent tool calls to the parent display.

    Two display paths:
      CLI:     prints tree-view lines above the parent's delegation spinner
      Gateway: batches tool names and relays to parent's progress callback

    Returns None if no display mechanism is available, in which case the
    child agent runs with no progress callback (identical to current behavior).
    """
    spinner = getattr(parent_agent, '_delegate_spinner', None)
    parent_cb = getattr(parent_agent, 'tool_progress_callback', None)

    if not spinner and not parent_cb:
        return None  # No display → no callback → zero behavior change

    # Show 1-indexed prefix only in batch mode (multiple tasks)
    prefix = f"[{task_index + 1}] " if task_count > 1 else ""

    # Gateway: batch tool names, flush periodically
    _BATCH_SIZE = 5
    _batch: List[str] = []

    def _callback(tool_name: str, preview: str = None):
        # Special "_thinking" event: model produced text content (reasoning)
        if tool_name == "_thinking":
            if spinner:
                short = (preview[:55] + "...") if preview and len(preview) > 55 else (preview or "")
                try:
                    spinner.print_above(f" {prefix}├─ 💭 \"{short}\"")
                except Exception:
                    pass
            # Don't relay thinking to gateway (too noisy for chat)
            return

        # Regular tool call event
        if spinner:
            short = (preview[:35] + "...") if preview and len(preview) > 35 else (preview or "")
            tool_emojis = {
                "terminal": "💻", "web_search": "🔍", "web_extract": "📄",
                "read_file": "📖", "write_file": "✍️", "patch": "🔧",
                "search_files": "🔎", "list_directory": "📂",
                "browser_navigate": "🌐", "browser_click": "👆",
                "text_to_speech": "🔊", "image_generate": "🎨",
                "vision_analyze": "👁️", "process": "⚙️",
            }
            emoji = tool_emojis.get(tool_name, "⚡")
            line = f" {prefix}├─ {emoji} {tool_name}"
            if short:
                line += f"  \"{short}\""
            try:
                spinner.print_above(line)
            except Exception:
                pass

        if parent_cb:
            _batch.append(tool_name)
            if len(_batch) >= _BATCH_SIZE:
                summary = ", ".join(_batch)
                try:
                    parent_cb("subagent_progress", f"🔀 {prefix}{summary}")
                except Exception:
                    pass
                _batch.clear()

    def _flush():
        """Flush remaining batched tool names to gateway on completion."""
        if parent_cb and _batch:
            summary = ", ".join(_batch)
            try:
                parent_cb("subagent_progress", f"🔀 {prefix}{summary}")
            except Exception:
                pass
            _batch.clear()

    _callback._flush = _flush
    return _callback


def _run_single_child(
    task_index: int,
    goal: str,
    context: Optional[str],
    toolsets: Optional[List[str]],
    model: Optional[str],
    max_iterations: int,
    parent_agent,
    task_count: int = 1,
) -> Dict[str, Any]:
    """
    Spawn and run a single child agent. Called from within a thread.
    Returns a structured result dict.
    """
    from run_agent import AIAgent

    child_start = time.monotonic()

    child_toolsets = _strip_blocked_tools(toolsets or DEFAULT_TOOLSETS)

    child_prompt = _build_child_system_prompt(goal, context)

    try:
        # Extract parent's API key so subagents inherit auth (e.g. Nous Portal).
        parent_api_key = getattr(parent_agent, "api_key", None)
        if (not parent_api_key) and hasattr(parent_agent, "_client_kwargs"):
            parent_api_key = parent_agent._client_kwargs.get("api_key")

        # Build progress callback to relay tool calls to parent display
        child_progress_cb = _build_child_progress_callback(task_index, parent_agent, task_count)

        child = AIAgent(
            base_url=parent_agent.base_url,
            api_key=parent_api_key,
            model=model or parent_agent.model,
            provider=getattr(parent_agent, "provider", None),
            api_mode=getattr(parent_agent, "api_mode", None),
            max_iterations=max_iterations,
            enabled_toolsets=child_toolsets,
            quiet_mode=True,
            ephemeral_system_prompt=child_prompt,
            log_prefix=f"[subagent-{task_index}]",
            platform=parent_agent.platform,
            skip_context_files=True,
            skip_memory=True,
            clarify_callback=None,
            session_db=getattr(parent_agent, '_session_db', None),
            providers_allowed=parent_agent.providers_allowed,
            providers_ignored=parent_agent.providers_ignored,
            providers_order=parent_agent.providers_order,
            provider_sort=parent_agent.provider_sort,
            tool_progress_callback=child_progress_cb,
        )

        # Set delegation depth so children can't spawn grandchildren
        child._delegate_depth = getattr(parent_agent, '_delegate_depth', 0) + 1

        # Register child for interrupt propagation
        if hasattr(parent_agent, '_active_children'):
            parent_agent._active_children.append(child)

        # Run with stdout/stderr suppressed to prevent interleaved output
        devnull = io.StringIO()
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            result = child.run_conversation(user_message=goal)

        # Flush any remaining batched progress to gateway
        if child_progress_cb and hasattr(child_progress_cb, '_flush'):
            try:
                child_progress_cb._flush()
            except Exception:
                pass

        duration = round(time.monotonic() - child_start, 2)

        summary = result.get("final_response") or ""
        completed = result.get("completed", False)
        interrupted = result.get("interrupted", False)
        api_calls = result.get("api_calls", 0)

        if interrupted:
            status = "interrupted"
        elif completed and summary:
            status = "completed"
        else:
            status = "failed"

        entry: Dict[str, Any] = {
            "task_index": task_index,
            "status": status,
            "summary": summary,
            "api_calls": api_calls,
            "duration_seconds": duration,
        }
        if status == "failed":
            entry["error"] = result.get("error", "Subagent did not produce a response.")

        return entry

    except Exception as exc:
        duration = round(time.monotonic() - child_start, 2)
        logging.exception(f"[subagent-{task_index}] failed")
        return {
            "task_index": task_index,
            "status": "error",
            "summary": None,
            "error": str(exc),
            "api_calls": 0,
            "duration_seconds": duration,
        }

    finally:
        # Unregister child from interrupt propagation
        if hasattr(parent_agent, '_active_children'):
            try:
                parent_agent._active_children.remove(child)
            except (ValueError, UnboundLocalError):
                pass


def delegate_task(
    goal: Optional[str] = None,
    context: Optional[str] = None,
    toolsets: Optional[List[str]] = None,
    tasks: Optional[List[Dict[str, Any]]] = None,
    model: Optional[str] = None,
    max_iterations: Optional[int] = None,
    parent_agent=None,
) -> str:
    """
    Spawn one or more child agents to handle delegated tasks.

    Supports two modes:
      - Single: provide goal (+ optional context, toolsets)
      - Batch:  provide tasks array [{goal, context, toolsets}, ...]

    Returns JSON with results array, one entry per task.
    """
    if parent_agent is None:
        return json.dumps({"error": "delegate_task requires a parent agent context."})

    # Depth limit
    depth = getattr(parent_agent, '_delegate_depth', 0)
    if depth >= MAX_DEPTH:
        return json.dumps({
            "error": (
                f"Delegation depth limit reached ({MAX_DEPTH}). "
                "Subagents cannot spawn further subagents."
            )
        })

    # Load config
    cfg = _load_config()
    default_max_iter = cfg.get("max_iterations", DEFAULT_MAX_ITERATIONS)
    effective_max_iter = max_iterations or default_max_iter

    # Normalize to task list
    if tasks and isinstance(tasks, list):
        task_list = tasks[:MAX_CONCURRENT_CHILDREN]
    elif goal and isinstance(goal, str) and goal.strip():
        task_list = [{"goal": goal, "context": context, "toolsets": toolsets}]
    else:
        return json.dumps({"error": "Provide either 'goal' (single task) or 'tasks' (batch)."})

    if not task_list:
        return json.dumps({"error": "No tasks provided."})

    # Validate each task has a goal
    for i, task in enumerate(task_list):
        if not task.get("goal", "").strip():
            return json.dumps({"error": f"Task {i} is missing a 'goal'."})

    overall_start = time.monotonic()
    results = []

    n_tasks = len(task_list)
    # Track goal labels for progress display (truncated for readability)
    task_labels = [t["goal"][:40] for t in task_list]

    if n_tasks == 1:
        # Single task -- run directly (no thread pool overhead)
        t = task_list[0]
        result = _run_single_child(
            task_index=0,
            goal=t["goal"],
            context=t.get("context"),
            toolsets=t.get("toolsets") or toolsets,
            model=model,
            max_iterations=effective_max_iter,
            parent_agent=parent_agent,
            task_count=1,
        )
        results.append(result)
    else:
        # Batch -- run in parallel with per-task progress lines
        completed_count = 0
        spinner_ref = getattr(parent_agent, '_delegate_spinner', None)

        # Save stdout/stderr before the executor — redirect_stdout in child
        # threads races on sys.stdout and can leave it as devnull permanently.
        _saved_stdout = sys.stdout
        _saved_stderr = sys.stderr

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_CHILDREN) as executor:
            futures = {}
            for i, t in enumerate(task_list):
                future = executor.submit(
                    _run_single_child,
                    task_index=i,
                    goal=t["goal"],
                    context=t.get("context"),
                    toolsets=t.get("toolsets") or toolsets,
                    model=model,
                    max_iterations=effective_max_iter,
                    parent_agent=parent_agent,
                    task_count=n_tasks,
                )
                futures[future] = i

            for future in as_completed(futures):
                try:
                    entry = future.result()
                except Exception as exc:
                    idx = futures[future]
                    entry = {
                        "task_index": idx,
                        "status": "error",
                        "summary": None,
                        "error": str(exc),
                        "api_calls": 0,
                        "duration_seconds": 0,
                    }
                results.append(entry)
                completed_count += 1

                # Print per-task completion line above the spinner
                idx = entry["task_index"]
                label = task_labels[idx] if idx < len(task_labels) else f"Task {idx}"
                dur = entry.get("duration_seconds", 0)
                status = entry.get("status", "?")
                icon = "✓" if status == "completed" else "✗"
                remaining = n_tasks - completed_count
                completion_line = f"{icon} [{idx+1}/{n_tasks}] {label}  ({dur}s)"
                if spinner_ref:
                    try:
                        spinner_ref.print_above(completion_line)
                    except Exception:
                        print(f"  {completion_line}")
                else:
                    print(f"  {completion_line}")

                # Update spinner text to show remaining count
                if spinner_ref and remaining > 0:
                    try:
                        spinner_ref.update_text(f"🔀 {remaining} task{'s' if remaining != 1 else ''} remaining")
                    except Exception:
                        pass

        # Restore stdout/stderr in case redirect_stdout race left them as devnull
        sys.stdout = _saved_stdout
        sys.stderr = _saved_stderr

        # Sort by task_index so results match input order
        results.sort(key=lambda r: r["task_index"])

    total_duration = round(time.monotonic() - overall_start, 2)

    return json.dumps({
        "results": results,
        "total_duration_seconds": total_duration,
    }, ensure_ascii=False)


def _load_config() -> dict:
    """Load delegation config from CLI_CONFIG if available."""
    try:
        from cli import CLI_CONFIG
        return CLI_CONFIG.get("delegation", {})
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# OpenAI Function-Calling Schema
# ---------------------------------------------------------------------------

DELEGATE_TASK_SCHEMA = {
    "name": "delegate_task",
    "description": (
        "Spawn one or more subagents to work on tasks in isolated contexts. "
        "Each subagent gets its own conversation, terminal session, and toolset. "
        "Only the final summary is returned -- intermediate tool results "
        "never enter your context window.\n\n"
        "TWO MODES (one of 'goal' or 'tasks' is required):\n"
        "1. Single task: provide 'goal' (+ optional context, toolsets)\n"
        "2. Batch (parallel): provide 'tasks' array with up to 3 items. "
        "All run concurrently and results are returned together.\n\n"
        "WHEN TO USE delegate_task:\n"
        "- Reasoning-heavy subtasks (debugging, code review, research synthesis)\n"
        "- Tasks that would flood your context with intermediate data\n"
        "- Parallel independent workstreams (research A and B simultaneously)\n\n"
        "WHEN NOT TO USE (use these instead):\n"
        "- Mechanical multi-step work with no reasoning needed -> use execute_code\n"
        "- Single tool call -> just call the tool directly\n"
        "- Tasks needing user interaction -> subagents cannot use clarify\n\n"
        "IMPORTANT:\n"
        "- Subagents have NO memory of your conversation. Pass all relevant "
        "info (file paths, error messages, constraints) via the 'context' field.\n"
        "- Subagents CANNOT call: delegate_task, clarify, memory, send_message, "
        "execute_code.\n"
        "- Each subagent gets its own terminal session (separate working directory and state).\n"
        "- Results are always returned as an array, one entry per task."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": (
                    "What the subagent should accomplish. Be specific and "
                    "self-contained -- the subagent knows nothing about your "
                    "conversation history."
                ),
            },
            "context": {
                "type": "string",
                "description": (
                    "Background information the subagent needs: file paths, "
                    "error messages, project structure, constraints. The more "
                    "specific you are, the better the subagent performs."
                ),
            },
            "toolsets": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Toolsets to enable for this subagent. "
                    "Default: ['terminal', 'file', 'web']. "
                    "Common patterns: ['terminal', 'file'] for code work, "
                    "['web'] for research, ['terminal', 'file', 'web'] for "
                    "full-stack tasks."
                ),
            },
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string", "description": "Task goal"},
                        "context": {"type": "string", "description": "Task-specific context"},
                        "toolsets": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Toolsets for this specific task",
                        },
                    },
                    "required": ["goal"],
                },
                "maxItems": 3,
                "description": (
                    "Batch mode: up to 3 tasks to run in parallel. Each gets "
                    "its own subagent with isolated context and terminal session. "
                    "When provided, top-level goal/context/toolsets are ignored."
                ),
            },
            "model": {
                "type": "string",
                "description": (
                    "Model override for the subagent(s). Omit to use your "
                    "same model. Use a cheaper/faster model for simple subtasks."
                ),
            },
            "max_iterations": {
                "type": "integer",
                "description": (
                    "Max tool-calling turns per subagent (default: 25). "
                    "Lower for simple tasks, higher for complex ones."
                ),
            },
        },
        "required": [],
    },
}


# --- Registry ---
from tools.registry import registry

registry.register(
    name="delegate_task",
    toolset="delegation",
    schema=DELEGATE_TASK_SCHEMA,
    handler=lambda args, **kw: delegate_task(
        goal=args.get("goal"),
        context=args.get("context"),
        toolsets=args.get("toolsets"),
        tasks=args.get("tasks"),
        model=args.get("model"),
        max_iterations=args.get("max_iterations"),
        parent_agent=kw.get("parent_agent")),
    check_fn=check_delegate_requirements,
)
