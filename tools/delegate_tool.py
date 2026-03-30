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

import json
import logging

logger = logging.getLogger(__name__)
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional


# Tools that children must never have access to
DELEGATE_BLOCKED_TOOLS = frozenset(
    [
        "delegate_task",  # no recursive delegation
        "clarify",  # no user interaction
        "memory",  # no writes to shared MEMORY.md
        "send_message",  # no cross-platform side effects
        "execute_code",  # children should reason step-by-step, not write scripts
    ]
)

MAX_CONCURRENT_CHILDREN = 3
MAX_DEPTH = 2  # parent (0) -> child (1) -> grandchild rejected (2)
DEFAULT_MAX_ITERATIONS = 50
DEFAULT_TOOLSETS = ["terminal", "file", "web"]


def check_delegate_requirements() -> bool:
    """Delegation has no external requirements -- always available."""
    return True


def _build_child_system_prompt(
    goal: str,
    context: Optional[str] = None,
    *,
    workspace_path: Optional[str] = None,
) -> str:
    """Build a focused system prompt for a child agent."""
    parts = [
        "You are a focused subagent working on a specific delegated task.",
        "",
        f"YOUR TASK:\n{goal}",
    ]
    if context and context.strip():
        parts.append(f"\nCONTEXT:\n{context}")
    if workspace_path and str(workspace_path).strip():
        parts.append(
            "\nWORKSPACE PATH:\n"
            f"{workspace_path}\n"
            "Use this exact path for local repository/workdir operations unless the task explicitly says otherwise."
        )
    parts.append(
        "\nComplete this task using the tools available to you. "
        "When finished, provide a clear, concise summary of:\n"
        "- What you did\n"
        "- What you found or accomplished\n"
        "- Any files you created or modified\n"
        "- Any issues encountered\n\n"
        "Important workspace rule: Never assume a repository lives at /workspace/... or any other container-style path unless the task/context explicitly gives that path. "
        "If no exact local path is provided, discover it first before issuing git/workdir-specific commands.\n\n"
        "Be thorough but concise -- your response is returned to the "
        "parent agent as a summary."
    )
    return "\n".join(parts)


def _resolve_workspace_hint(parent_agent) -> Optional[str]:
    """Best-effort local workspace hint for child prompts.

    We only inject a path when we have a concrete absolute directory. This avoids
    teaching subagents a fake container path while still helping them avoid
    guessing `/workspace/...` for local repo tasks.
    """
    candidates = [
        os.getenv("TERMINAL_CWD"),
        getattr(
            getattr(parent_agent, "_subdirectory_hints", None), "working_dir", None
        ),
        getattr(parent_agent, "terminal_cwd", None),
        getattr(parent_agent, "cwd", None),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            text = os.path.abspath(os.path.expanduser(str(candidate)))
        except Exception:
            continue
        if os.path.isabs(text) and os.path.isdir(text):
            return text
    return None


def _strip_blocked_tools(toolsets: List[str]) -> List[str]:
    """Remove toolsets that contain only blocked tools."""
    blocked_toolset_names = {
        "delegation",
        "clarify",
        "memory",
        "code_execution",
    }
    return [t for t in toolsets if t not in blocked_toolset_names]


def _build_child_progress_callback(
    task_index: int, parent_agent, task_count: int = 1
) -> Optional[callable]:
    """Build a callback that relays child agent tool calls to the parent display.

    Two display paths:
      CLI:     prints tree-view lines above the parent's delegation spinner
      Gateway: batches tool names and relays to parent's progress callback

    Returns None if no display mechanism is available, in which case the
    child agent runs with no progress callback (identical to current behavior).
    """
    spinner = getattr(parent_agent, "_delegate_spinner", None)
    parent_cb = getattr(parent_agent, "tool_progress_callback", None)

    if not spinner and not parent_cb:
        return None  # No display → no callback → zero behavior change

    # Show 1-indexed prefix only in batch mode (multiple tasks)
    prefix = f"[{task_index + 1}] " if task_count > 1 else ""

    # Gateway: batch tool names, flush periodically
    _BATCH_SIZE = 5
    _batch: List[str] = []

    def _callback(
        event_type: str, tool_name: str = None, preview: str = None, args=None, **kwargs
    ):
        # event_type is one of: "tool.started", "tool.completed",
        # "reasoning.available", "_thinking", "subagent_progress"

        # "_thinking" / reasoning events
        if event_type in ("_thinking", "reasoning.available"):
            text = preview or tool_name or ""
            if spinner:
                short = (
                    (preview[:55] + "...")
                    if preview and len(preview) > 55
                    else (preview or "")
                )
                try:
                    spinner.print_above(f' {prefix}├─ 💭 "{short}"')
                except Exception as e:
                    logger.debug("Spinner print_above failed: %s", e)
            # Don't relay thinking to gateway (too noisy for chat)
            return

        # tool.completed — no display needed here (spinner shows on started)
        if event_type == "tool.completed":
            return

        # tool.started — display and batch for parent relay
        if spinner:
            short = (
                (preview[:35] + "...")
                if preview and len(preview) > 35
                else (preview or "")
            )
            from agent.display import get_tool_emoji

            emoji = get_tool_emoji(tool_name or "")
            line = f" {prefix}├─ {emoji} {tool_name}"
            if short:
                line += f'  "{short}"'
            try:
                spinner.print_above(line)
            except Exception as e:
                logger.debug("Spinner print_above failed: %s", e)

        if parent_cb:
            _batch.append(tool_name or "")
            if len(_batch) >= _BATCH_SIZE:
                summary = ", ".join(_batch)
                try:
                    parent_cb("subagent_progress", f"🔀 {prefix}{summary}")
                except Exception as e:
                    logger.debug("Parent callback failed: %s", e)
                _batch.clear()

    def _flush():
        """Flush remaining batched tool names to gateway on completion."""
        if parent_cb and _batch:
            summary = ", ".join(_batch)
            try:
                parent_cb("subagent_progress", f"🔀 {prefix}{summary}")
            except Exception as e:
                logger.debug("Parent callback flush failed: %s", e)
            _batch.clear()

    _callback._flush = _flush
    return _callback


def _build_child_agent(
    task_index: int,
    goal: str,
    context: Optional[str],
    toolsets: Optional[List[str]],
    model: Optional[str],
    max_iterations: int,
    parent_agent,
    # Credential overrides from delegation config (provider:model resolution)
    override_provider: Optional[str] = None,
    override_base_url: Optional[str] = None,
    override_api_key: Optional[str] = None,
    override_api_mode: Optional[str] = None,
    # ACP transport overrides — lets a non-ACP parent spawn ACP child agents
    override_acp_command: Optional[str] = None,
    override_acp_args: Optional[List[str]] = None,
):
    """
    Build a child AIAgent on the main thread (thread-safe construction).
    Returns the constructed child agent without running it.

    When override_* params are set (from delegation config), the child uses
    those credentials instead of inheriting from the parent.  This enables
    routing subagents to a different provider:model pair (e.g. cheap/fast
    model on OpenRouter while the parent runs on Nous Portal).
    """
    from run_agent import AIAgent

    # When no explicit toolsets given, inherit from parent's enabled toolsets
    # so disabled tools (e.g. web) don't leak to subagents.
    # Note: enabled_toolsets=None means "all tools enabled" (the default),
    # so we must derive effective toolsets from the parent's loaded tools.
    parent_enabled = getattr(parent_agent, "enabled_toolsets", None)
    if parent_enabled is not None:
        parent_toolsets = set(parent_enabled)
    elif parent_agent and hasattr(parent_agent, "valid_tool_names"):
        # enabled_toolsets is None (all tools) — derive from loaded tool names
        import model_tools

        parent_toolsets = {
            ts
            for name in parent_agent.valid_tool_names
            if (ts := model_tools.get_toolset_for_tool(name)) is not None
        }
    else:
        parent_toolsets = set(DEFAULT_TOOLSETS)

    if toolsets:
        # Intersect with parent — subagent must not gain tools the parent lacks
        child_toolsets = _strip_blocked_tools(
            [t for t in toolsets if t in parent_toolsets]
        )
    elif parent_agent and parent_enabled is not None:
        child_toolsets = _strip_blocked_tools(parent_enabled)
    elif parent_toolsets:
        child_toolsets = _strip_blocked_tools(sorted(parent_toolsets))
    else:
        child_toolsets = _strip_blocked_tools(DEFAULT_TOOLSETS)

    workspace_hint = _resolve_workspace_hint(parent_agent)
    child_prompt = _build_child_system_prompt(
        goal, context, workspace_path=workspace_hint
    )
    # Extract parent's API key so subagents inherit auth (e.g. Nous Portal).
    parent_api_key = getattr(parent_agent, "api_key", None)
    if (not parent_api_key) and hasattr(parent_agent, "_client_kwargs"):
        parent_api_key = parent_agent._client_kwargs.get("api_key")

    # Build progress callback to relay tool calls to parent display
    child_progress_cb = _build_child_progress_callback(task_index, parent_agent)

    # Each subagent gets its own iteration budget capped at max_iterations
    # (configurable via delegation.max_iterations, default 50).  This means
    # total iterations across parent + subagents can exceed the parent's
    # max_iterations.  The user controls the per-subagent cap in config.yaml.

    child_thinking_cb = None
    if child_progress_cb:

        def _child_thinking(text: str) -> None:
            if not text:
                return
            try:
                child_progress_cb("_thinking", text)
            except Exception as e:
                logger.debug("Child thinking callback relay failed: %s", e)

        child_thinking_cb = _child_thinking

    # Resolve effective credentials: config override > parent inherit
    effective_model = model or parent_agent.model
    effective_provider = override_provider or getattr(parent_agent, "provider", None)
    effective_base_url = override_base_url or parent_agent.base_url
    effective_api_key = override_api_key or parent_api_key
    effective_api_mode = override_api_mode or getattr(parent_agent, "api_mode", None)
    effective_acp_command = override_acp_command or getattr(
        parent_agent, "acp_command", None
    )
    effective_acp_args = list(
        override_acp_args
        if override_acp_args is not None
        else (getattr(parent_agent, "acp_args", []) or [])
    )

    child = AIAgent(
        base_url=effective_base_url,
        api_key=effective_api_key,
        model=effective_model,
        provider=effective_provider,
        api_mode=effective_api_mode,
        acp_command=effective_acp_command,
        acp_args=effective_acp_args,
        max_iterations=max_iterations,
        max_tokens=getattr(parent_agent, "max_tokens", None),
        reasoning_config=getattr(parent_agent, "reasoning_config", None),
        prefill_messages=getattr(parent_agent, "prefill_messages", None),
        enabled_toolsets=child_toolsets,
        quiet_mode=True,
        ephemeral_system_prompt=child_prompt,
        log_prefix=f"[subagent-{task_index}]",
        platform=parent_agent.platform,
        skip_context_files=True,
        skip_memory=True,
        clarify_callback=None,
        thinking_callback=child_thinking_cb,
        session_db=getattr(parent_agent, "_session_db", None),
        parent_session_id=getattr(parent_agent, "session_id", None),
        providers_allowed=parent_agent.providers_allowed,
        providers_ignored=parent_agent.providers_ignored,
        providers_order=parent_agent.providers_order,
        provider_sort=parent_agent.provider_sort,
        tool_progress_callback=child_progress_cb,
        iteration_budget=None,  # fresh budget per subagent
    )
    child._print_fn = getattr(parent_agent, "_print_fn", None)
    # Set delegation depth so children can't spawn grandchildren
    child._delegate_depth = getattr(parent_agent, "_delegate_depth", 0) + 1

    # Share a credential pool with the child when possible so subagents can
    # rotate credentials on rate limits instead of getting pinned to one key.
    child_pool = _resolve_child_credential_pool(effective_provider, parent_agent)
    if child_pool is not None:
        child._credential_pool = child_pool

    # Register child for interrupt propagation
    if hasattr(parent_agent, "_active_children"):
        lock = getattr(parent_agent, "_active_children_lock", None)
        if lock:
            with lock:
                parent_agent._active_children.append(child)
        else:
            parent_agent._active_children.append(child)

    return child


def _run_single_child(
    task_index: int,
    goal: str,
    child=None,
    parent_agent=None,
    **_kwargs,
) -> Dict[str, Any]:
    """
    Run a pre-built child agent. Called from within a thread.
    Returns a structured result dict.
    """
    child_start = time.monotonic()

    # Get the progress callback from the child agent
    child_progress_cb = getattr(child, "tool_progress_callback", None)

    # Restore parent tool names using the value saved before child construction
    # mutated the global. This is the correct parent toolset, not the child's.
    import model_tools

    _saved_tool_names = getattr(
        child, "_delegate_saved_tool_names", list(model_tools._last_resolved_tool_names)
    )

    child_pool = getattr(child, "_credential_pool", None)
    leased_cred_id = None
    if child_pool is not None:
        leased_cred_id = child_pool.acquire_lease()
        if leased_cred_id is not None:
            try:
                leased_entry = child_pool.current()
                if leased_entry is not None and hasattr(child, "_swap_credential"):
                    child._swap_credential(leased_entry)
            except Exception as exc:
                logger.debug("Failed to bind child to leased credential: %s", exc)

    try:
        result = child.run_conversation(user_message=goal)

        # Flush any remaining batched progress to gateway
        if child_progress_cb and hasattr(child_progress_cb, "_flush"):
            try:
                child_progress_cb._flush()
            except Exception as e:
                logger.debug("Progress callback flush failed: %s", e)

        duration = round(time.monotonic() - child_start, 2)

        summary = result.get("final_response") or ""
        completed = result.get("completed", False)
        interrupted = result.get("interrupted", False)
        api_calls = result.get("api_calls", 0)

        if interrupted:
            status = "interrupted"
        elif summary:
            # A summary means the subagent produced usable output.
            # exit_reason ("completed" vs "max_iterations") already
            # tells the parent *how* the task ended.
            status = "completed"
        else:
            status = "failed"

        # Build tool trace from conversation messages (already in memory).
        # Uses tool_call_id to correctly pair parallel tool calls with results.
        tool_trace: list[Dict[str, Any]] = []
        trace_by_id: Dict[str, Dict[str, Any]] = {}
        messages = result.get("messages") or []
        if isinstance(messages, list):
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                if msg.get("role") == "assistant":
                    for tc in msg.get("tool_calls") or []:
                        fn = tc.get("function", {})
                        entry_t = {
                            "tool": fn.get("name", "unknown"),
                            "args_bytes": len(fn.get("arguments", "")),
                        }
                        tool_trace.append(entry_t)
                        tc_id = tc.get("id")
                        if tc_id:
                            trace_by_id[tc_id] = entry_t
                elif msg.get("role") == "tool":
                    content = msg.get("content", "")
                    is_error = bool(content and "error" in content[:80].lower())
                    result_meta = {
                        "result_bytes": len(content),
                        "status": "error" if is_error else "ok",
                    }
                    # Match by tool_call_id for parallel calls
                    tc_id = msg.get("tool_call_id")
                    target = trace_by_id.get(tc_id) if tc_id else None
                    if target is not None:
                        target.update(result_meta)
                    elif tool_trace:
                        # Fallback for messages without tool_call_id
                        tool_trace[-1].update(result_meta)

        # Determine exit reason
        if interrupted:
            exit_reason = "interrupted"
        elif completed:
            exit_reason = "completed"
        else:
            exit_reason = "max_iterations"

        # Extract token counts (safe for mock objects)
        _input_tokens = getattr(child, "session_prompt_tokens", 0)
        _output_tokens = getattr(child, "session_completion_tokens", 0)
        _model = getattr(child, "model", None)

        entry: Dict[str, Any] = {
            "task_index": task_index,
            "status": status,
            "summary": summary,
            "api_calls": api_calls,
            "duration_seconds": duration,
            "model": _model if isinstance(_model, str) else None,
            "exit_reason": exit_reason,
            "tokens": {
                "input": _input_tokens
                if isinstance(_input_tokens, (int, float))
                else 0,
                "output": _output_tokens
                if isinstance(_output_tokens, (int, float))
                else 0,
            },
            "tool_trace": tool_trace,
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
        if child_pool is not None and leased_cred_id is not None:
            try:
                child_pool.release_lease(leased_cred_id)
            except Exception as exc:
                logger.debug("Failed to release credential lease: %s", exc)

        # Restore the parent's tool names so the process-global is correct
        # for any subsequent execute_code calls or other consumers.
        import model_tools

        saved_tool_names = getattr(child, "_delegate_saved_tool_names", None)
        if isinstance(saved_tool_names, list):
            model_tools._last_resolved_tool_names = list(saved_tool_names)

        # Remove child from active tracking

        # Unregister child from interrupt propagation
        if hasattr(parent_agent, "_active_children"):
            try:
                lock = getattr(parent_agent, "_active_children_lock", None)
                if lock:
                    with lock:
                        parent_agent._active_children.remove(child)
                else:
                    parent_agent._active_children.remove(child)
            except (ValueError, UnboundLocalError) as e:
                logger.debug("Could not remove child from active_children: %s", e)


def delegate_task(
    goal: Optional[str] = None,
    context: Optional[str] = None,
    toolsets: Optional[List[str]] = None,
    tasks: Optional[List[Dict[str, Any]]] = None,
    max_iterations: Optional[int] = None,
    acp_command: Optional[str] = None,
    acp_args: Optional[List[str]] = None,
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
        return tool_error("delegate_task requires a parent agent context.")

    # Depth limit
    depth = getattr(parent_agent, "_delegate_depth", 0)
    if depth >= MAX_DEPTH:
        return json.dumps(
            {
                "error": (
                    f"Delegation depth limit reached ({MAX_DEPTH}). "
                    "Subagents cannot spawn further subagents."
                )
            }
        )

    # Load config
    cfg = _load_config()
    default_max_iter = cfg.get("max_iterations", DEFAULT_MAX_ITERATIONS)
    effective_max_iter = max_iterations or default_max_iter

    # Resolve delegation credentials (provider:model pair).
    # When delegation.provider is configured, this resolves the full credential
    # bundle (base_url, api_key, api_mode) via the same runtime provider system
    # used by CLI/gateway startup.  When unconfigured, returns None values so
    # children inherit from the parent.
    try:
        creds = _resolve_delegation_credentials(cfg, parent_agent)
    except ValueError as exc:
        return tool_error(str(exc))

    # Normalize to task list
    if tasks and isinstance(tasks, list):
        task_list = tasks[:MAX_CONCURRENT_CHILDREN]
    elif goal and isinstance(goal, str) and goal.strip():
        task_list = [{"goal": goal, "context": context, "toolsets": toolsets}]
    else:
        return tool_error("Provide either 'goal' (single task) or 'tasks' (batch).")

    if not task_list:
        return tool_error("No tasks provided.")

    # Validate each task has a goal
    for i, task in enumerate(task_list):
        if not task.get("goal", "").strip():
            return tool_error(f"Task {i} is missing a 'goal'.")

    overall_start = time.monotonic()
    results = []

    n_tasks = len(task_list)
    # Track goal labels for progress display (truncated for readability)
    task_labels = [t["goal"][:40] for t in task_list]

    # Save parent tool names BEFORE any child construction mutates the global.
    # _build_child_agent() calls AIAgent() which calls get_tool_definitions(),
    # which overwrites model_tools._last_resolved_tool_names with child's toolset.
    import model_tools as _model_tools

    _parent_tool_names = list(_model_tools._last_resolved_tool_names)

    # Build all child agents on the main thread (thread-safe construction)
    # Wrapped in try/finally so the global is always restored even if a
    # child build raises (otherwise _last_resolved_tool_names stays corrupted).
    children = []
    try:
        for i, t in enumerate(task_list):
            child = _build_child_agent(
                task_index=i,
                goal=t["goal"],
                context=t.get("context"),
                toolsets=t.get("toolsets") or toolsets,
                model=creds["model"],
                max_iterations=effective_max_iter,
                parent_agent=parent_agent,
                override_provider=creds["provider"],
                override_base_url=creds["base_url"],
                override_api_key=creds["api_key"],
                override_api_mode=creds["api_mode"],
                override_acp_command=t.get("acp_command") or acp_command,
                override_acp_args=t.get("acp_args") or acp_args,
            )
            # Override with correct parent tool names (before child construction mutated global)
            child._delegate_saved_tool_names = _parent_tool_names
            children.append((i, t, child))
    finally:
        # Authoritative restore: reset global to parent's tool names after all children built
        _model_tools._last_resolved_tool_names = _parent_tool_names

    if n_tasks == 1:
        # Single task -- run directly (no thread pool overhead)
        _i, _t, child = children[0]
        result = _run_single_child(0, _t["goal"], child, parent_agent)
        results.append(result)
    else:
        # Batch -- run in parallel with per-task progress lines
        completed_count = 0
        spinner_ref = getattr(parent_agent, "_delegate_spinner", None)

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_CHILDREN) as executor:
            futures = {}
            for i, t, child in children:
                future = executor.submit(
                    _run_single_child,
                    task_index=i,
                    goal=t["goal"],
                    child=child,
                    parent_agent=parent_agent,
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
                completion_line = f"{icon} [{idx + 1}/{n_tasks}] {label}  ({dur}s)"
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
                        spinner_ref.update_text(
                            f"🔀 {remaining} task{'s' if remaining != 1 else ''} remaining"
                        )
                    except Exception as e:
                        logger.debug("Spinner update_text failed: %s", e)

        # Sort by task_index so results match input order
        results.sort(key=lambda r: r["task_index"])

    # Notify parent's memory provider of delegation outcomes
    if (
        parent_agent
        and hasattr(parent_agent, "_memory_manager")
        and parent_agent._memory_manager
    ):
        for entry in results:
            try:
                _task_goal = (
                    task_list[entry["task_index"]]["goal"]
                    if entry["task_index"] < len(task_list)
                    else ""
                )
                parent_agent._memory_manager.on_delegation(
                    task=_task_goal,
                    result=entry.get("summary", "") or "",
                    child_session_id=getattr(
                        children[entry["task_index"]][2], "session_id", ""
                    )
                    if entry["task_index"] < len(children)
                    else "",
                )
            except Exception:
                pass

    total_duration = round(time.monotonic() - overall_start, 2)

    return json.dumps(
        {
            "results": results,
            "total_duration_seconds": total_duration,
        },
        ensure_ascii=False,
    )


def _resolve_child_credential_pool(effective_provider: Optional[str], parent_agent):
    """Resolve a credential pool for the child agent.

    Rules:
    1. Same provider as the parent -> share the parent's pool so cooldown state
       and rotation stay synchronized.
    2. Different provider -> try to load that provider's own pool.
    3. No pool available -> return None and let the child keep the inherited
       fixed credential behavior.
    """
    if not effective_provider:
        return getattr(parent_agent, "_credential_pool", None)

    parent_provider = getattr(parent_agent, "provider", None) or ""
    parent_pool = getattr(parent_agent, "_credential_pool", None)
    if parent_pool is not None and effective_provider == parent_provider:
        return parent_pool

    try:
        from agent.credential_pool import load_pool

        pool = load_pool(effective_provider)
        if pool is not None and pool.has_credentials():
            return pool
    except Exception as exc:
        logger.debug(
            "Could not load credential pool for child provider '%s': %s",
            effective_provider,
            exc,
        )
    return None


def _resolve_delegation_credentials(cfg: dict, parent_agent) -> dict:
    """Resolve credentials for subagent delegation.

    If ``delegation.base_url`` is configured, subagents use that direct
    OpenAI-compatible endpoint. Otherwise, if ``delegation.provider`` is
    configured, the full credential bundle (base_url, api_key, api_mode,
    provider) is resolved via the runtime provider system — the same path used
    by CLI/gateway startup. This lets subagents run on a completely different
    provider:model pair.

    If neither base_url nor provider is configured, returns None values so the
    child inherits everything from the parent agent.

    Raises ValueError with a user-friendly message on credential failure.
    """
    configured_model = str(cfg.get("model") or "").strip() or None
    configured_provider = str(cfg.get("provider") or "").strip() or None
    configured_base_url = str(cfg.get("base_url") or "").strip() or None
    configured_api_key = str(cfg.get("api_key") or "").strip() or None

    if configured_base_url:
        api_key = configured_api_key or os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "Delegation base_url is configured but no API key was found. "
                "Set delegation.api_key or OPENAI_API_KEY."
            )

        base_lower = configured_base_url.lower()
        provider = "custom"
        api_mode = "chat_completions"
        if "chatgpt.com/backend-api/codex" in base_lower:
            provider = "openai-codex"
            api_mode = "codex_responses"
        elif "api.anthropic.com" in base_lower:
            provider = "anthropic"
            api_mode = "anthropic_messages"

        return {
            "model": configured_model,
            "provider": provider,
            "base_url": configured_base_url,
            "api_key": api_key,
            "api_mode": api_mode,
        }

    if not configured_provider:
        # No provider override — child inherits everything from parent
        return {
            "model": configured_model,
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
        }

    # Provider is configured — resolve full credentials
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(requested=configured_provider)
    except Exception as exc:
        raise ValueError(
            f"Cannot resolve delegation provider '{configured_provider}': {exc}. "
            f"Check that the provider is configured (API key set, valid provider name), "
            f"or set delegation.base_url/delegation.api_key for a direct endpoint. "
            f"Available providers: openrouter, nous, zai, kimi-coding, minimax."
        ) from exc

    api_key = runtime.get("api_key", "")
    if not api_key:
        raise ValueError(
            f"Delegation provider '{configured_provider}' resolved but has no API key. "
            f"Set the appropriate environment variable or run 'hermes auth'."
        )

    return {
        "model": configured_model,
        "provider": runtime.get("provider"),
        "base_url": runtime.get("base_url"),
        "api_key": api_key,
        "api_mode": runtime.get("api_mode"),
        "command": runtime.get("command"),
        "args": list(runtime.get("args") or []),
    }


def _load_config() -> dict:
    """Load delegation config from CLI_CONFIG or persistent config.

    Checks the runtime config (cli.py CLI_CONFIG) first, then falls back
    to the persistent config (hermes_cli/config.py load_config()) so that
    ``delegation.model`` / ``delegation.provider`` are picked up regardless
    of the entry point (CLI, gateway, cron).
    """
    try:
        from cli import CLI_CONFIG

        cfg = CLI_CONFIG.get("delegation", {})
        if cfg:
            return cfg
    except Exception:
        pass
    try:
        from hermes_cli.config import load_config

        full = load_config()
        return full.get("delegation", {})
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# OpenAI Function-Calling Schema
# ---------------------------------------------------------------------------

DELEGATE_TASK_SCHEMA = {
    "name": "delegate_task",
    "description": (
        "Spawn temporary ONE-OFF subagents to work on tasks in isolated contexts. "
        "Each subagent gets its own conversation, terminal session, and toolset, "
        "but shares the SAME HERMES_HOME as you. "
        "Only the final summary is returned -- intermediate tool results "
        "never enter your context window.\n\n"
        "CRITICAL: For cross-profile orchestration (delegating to named profiles like "
        "'coder', 'researcher'), use profile_delegate INSTEAD of this tool. "
        "delegate_task creates temporary subagents; profile_delegate connects to "
        "persistent profile workers with separate HERMES_HOME directories.\n\n"
        "TWO MODES (one of 'goal' or 'tasks' is required):\n"
        "1. Single task: provide 'goal' (+ optional context, toolsets)\n"
        "2. Batch (parallel): provide 'tasks' array with up to 3 items. "
        "All run concurrently and results are returned together.\n\n"
        "WHEN TO USE delegate_task:\n"
        "- Reasoning-heavy subtasks (debugging, code review, research synthesis)\n"
        "- Tasks that would flood your context with intermediate data\n"
        "- Parallel independent workstreams (research A and B simultaneously)\n"
        "- Temporary subagents that don't need persistent state\n\n"
        "WHEN NOT TO USE (use these instead):\n"
        "- Cross-profile orchestration -> use profile_delegate (persistent workers)\n"
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
                    "Default: inherits your enabled toolsets. "
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
                        "context": {
                            "type": "string",
                            "description": "Task-specific context",
                        },
                        "toolsets": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Toolsets for this specific task. Use 'web' for network access, 'terminal' for shell.",
                        },
                        "acp_command": {
                            "type": "string",
                            "description": "Per-task ACP command override (e.g. 'claude'). Overrides the top-level acp_command for this task only.",
                        },
                        "acp_args": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Per-task ACP args override.",
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
            "max_iterations": {
                "type": "integer",
                "description": (
                    "Max tool-calling turns per subagent (default: 50). "
                    "Only set lower for simple tasks."
                ),
            },
            "acp_command": {
                "type": "string",
                "description": (
                    "Override ACP command for child agents (e.g. 'claude', 'copilot'). "
                    "When set, children use ACP subprocess transport instead of inheriting "
                    "the parent's transport. Enables spawning Claude Code (claude --acp --stdio) "
                    "or other ACP-capable agents from any parent, including Discord/Telegram/CLI."
                ),
            },
            "acp_args": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Arguments for the ACP command (default: ['--acp', '--stdio']). "
                    "Only used when acp_command is set. Example: ['--acp', '--stdio', '--model', 'claude-opus-4-6']"
                ),
            },
        },
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# Inter-profile delegation (profile_delegate)
# ---------------------------------------------------------------------------

import asyncio as _asyncio


def check_profile_requirements() -> bool:
    """Profile delegation requires no external dependencies."""
    return True


def profile_delegate(
    profile: str,
    goal: str,
    context: Optional[str] = None,
    wait: bool = True,
    timeout: int = 300,
    allow_subagents: bool = True,
    max_depth: int = 2,
    parent_agent=None,
) -> str:
    """
    Delegate a task to a specific Hermes profile worker.

    Connects to a running profile gateway or starts one if not already running.
    This is the main orchestration tool for inter-profile communication.

    Args:
        profile:  Target profile name (e.g., 'coder', 'researcher').
        goal:     Task description to send to the profile worker.
        context:  Optional additional context information.
        wait:     Wait for result or return immediately with task_id.
        timeout:  Seconds to wait for result (default 300).
        allow_subagents: Whether the delegated agent may spawn one-off
                  subagents via delegate_task (default True).
        max_depth: Maximum delegation depth (default 2). Prevents infinite
                  delegation chains. Parent=0, child=1, grandchild=2.

    Returns:
        JSON string with the result (if wait=True) or task_id (if wait=False).
    """
    from tools.profile_orchestrator import (
        ensure_profile_running,
        send_to_profile,
        send_to_profile_async,
        is_profile_running,
        discover_profiles,
    )

    if not profile or not str(profile).strip():
        return json.dumps({"error": "Profile name is required"})

    profile = str(profile).strip()

    # Validate profile exists
    available = discover_profiles()
    if available and profile not in available:
        return json.dumps(
            {
                "error": f"Profile '{profile}' not found",
                "available": available,
            }
        )

    # Ensure profile is running (start if needed)
    if not is_profile_running(profile):
        if not ensure_profile_running(profile):
            return json.dumps({"error": f"Failed to start profile '{profile}'"})

    # DEBUG: Check parent_agent
    logger.info(
        f"[DEBUG] profile_delegate: parent_agent is None = {parent_agent is None}"
    )

    # Extract conversation history from parent agent for context continuity
    conversation_history = None
    if parent_agent is not None:
        try:
            session_messages = getattr(parent_agent, "_session_messages", [])
            logger.info(
                f"[DEBUG] Found {len(session_messages)} session messages in parent_agent"
            )
            if session_messages:
                conversation_history = [
                    {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                    for msg in session_messages
                    if msg.get("role") in ("user", "assistant", "system")
                ]
                logger.info(
                    f"[DEBUG] Extracted {len(conversation_history)} messages for delegation"
                )
        except Exception as e:
            logger.warning(
                f"Failed to extract conversation history from parent agent: {e}"
            )
            conversation_history = None

    # For async mode without parent_agent, try to load previous orchestration tasks
    if conversation_history is None and not wait:
        try:
            from hermes_state import SessionDB

            db = SessionDB()
            # Get recent orchestration tasks for this profile
            recent_tasks = db.list_orchestration_tasks(status=None, limit=10)
            if recent_tasks:
                history = []
                for task in recent_tasks:
                    if task.get("context"):
                        try:
                            ctx = json.loads(task["context"])
                            if ctx.get("message"):
                                history.append(
                                    {"role": "user", "content": ctx["message"]}
                                )
                        except:
                            pass
                if history:
                    conversation_history = history[-5:]  # Last 5 messages
                    logger.info(
                        f"[DEBUG] Loaded {len(conversation_history)} messages from previous async tasks"
                    )
        except Exception as e:
            logger.debug(f"Could not load previous async task history: {e}")
    if parent_agent is not None:
        try:
            session_messages = getattr(parent_agent, "_session_messages", [])
            logger.info(
                f"[DEBUG] Found {len(session_messages)} session messages in parent_agent"
            )
            if session_messages:
                conversation_history = [
                    {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                    for msg in session_messages
                    if msg.get("role") in ("user", "assistant", "system")
                ]
                logger.info(
                    f"[DEBUG] Extracted {len(conversation_history)} messages for delegation"
                )
        except Exception as e:
            logger.warning(f"Failed to extract conversation history: {e}")

    try:
        # Generate task_id for both sync and async modes (for searchability)
        import uuid

        task_id = str(uuid.uuid4())[:8]

        if wait:
            # Blocking mode: send directly and wait for result
            # First, create DB record for searchability
            try:
                from hermes_state import SessionDB

                db = SessionDB()
                db.create_orchestration_task(
                    task_id=task_id,
                    parent_session_id=None,
                    target_profile=profile,
                    goal=goal,
                    context=context,
                )
            except Exception as e:
                logger.debug(f"Failed to create orchestration task record: {e}")

            result = send_to_profile(
                profile_name=profile,
                message=goal,
                context={"additional_context": context} if context else None,
                wait=True,
                timeout=float(timeout),
                conversation_history=conversation_history,
                allow_subagents=allow_subagents,
                max_depth=max_depth,
            )

            # Update DB with result
            try:
                from hermes_state import SessionDB

                db = SessionDB()
                db.update_orchestration_task(
                    task_id=task_id,
                    status="completed",
                    result=result,
                )
            except Exception as e:
                logger.debug(f"Failed to update orchestration task: {e}")

            return json.dumps(
                {"status": "completed", "result": result, "task_id": task_id}
            )
        else:
            # Background mode: dispatch async and return task_id
            task_id_or_coro = send_to_profile_async(
                profile_name=profile,
                message=goal,
                context={"additional_context": context} if context else None,
                conversation_history=conversation_history,
                allow_subagents=allow_subagents,
                max_depth=max_depth,
            )
            # send_to_profile_async is async; run it to get the task_id string
            if _asyncio.iscoroutine(task_id_or_coro):
                task_id = _asyncio.run(task_id_or_coro)
            else:
                task_id = task_id_or_coro
            return json.dumps({"status": "dispatched", "task_id": task_id})

    except ValueError as e:
        return json.dumps({"error": str(e)})
    except RuntimeError as e:
        return json.dumps({"error": f"Profile communication error: {e}"})
    except Exception as e:
        logger.exception("profile_delegate failed")
        return json.dumps({"error": f"Unexpected error: {e}"})


def profile_delegate_async(
    profile: str,
    goal: str,
    context: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Non-blocking version of profile_delegate.

    Always returns immediately with a task_id for polling.
    """
    from tools.profile_orchestrator import (
        ensure_profile_running,
        send_to_profile_async,
        is_profile_running,
        discover_profiles,
    )

    if not profile or not str(profile).strip():
        return json.dumps({"error": "Profile name is required"})

    profile = str(profile).strip()

    # Validate profile exists
    available = discover_profiles()
    if available and profile not in available:
        return json.dumps(
            {
                "error": f"Profile '{profile}' not found",
                "available": available,
            }
        )

    # Ensure profile is running
    if not is_profile_running(profile):
        if not ensure_profile_running(profile):
            return json.dumps({"error": f"Failed to start profile '{profile}'"})

    try:
        task_id_or_coro = send_to_profile_async(
            profile_name=profile,
            message=goal,
            context={"additional_context": context} if context else None,
            conversation_history=conversation_history,
        )
        if _asyncio.iscoroutine(task_id_or_coro):
            task_id = _asyncio.run(task_id_or_coro)
        else:
            task_id = task_id_or_coro
        return json.dumps({"status": "dispatched", "task_id": task_id})
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.exception("profile_delegate_async failed")
        return json.dumps({"error": f"Unexpected error: {e}"})


def profile_task_check(task_id: str, parent_agent=None) -> str:
    """
    Check the status of an async profile delegation task.

    Args:
        task_id: The task_id returned by profile_delegate (with wait=False).
        parent_agent: Optional reference to parent agent (for consistency with other tools).

    Returns:
        JSON string with status and result (if completed).
    """
    from tools.profile_orchestrator import (
        _profile_tasks,
        collect_profile_result,
    )

    if not task_id:
        return json.dumps({"error": "task_id is required"})

    entry = _profile_tasks.get(task_id)
    if entry:
        future = entry["future"]

        if entry.get("_cancelled", False):
            return json.dumps(
                {
                    "status": "cancelled",
                    "task_id": task_id,
                }
            )

        import concurrent.futures

        if not future.done():
            return json.dumps({"status": "running", "task_id": task_id})

        try:
            result = collect_profile_result(task_id, block=False, timeout=0)
            if result is not None:
                return json.dumps(
                    {
                        "status": "completed",
                        "task_id": task_id,
                        "result": result,
                    }
                )
            else:
                return json.dumps(
                    {
                        "status": "completed",
                        "task_id": task_id,
                        "result": None,
                    }
                )
        except RuntimeError as e:
            return json.dumps({"status": "error", "task_id": task_id, "error": str(e)})
        except Exception as e:
            return json.dumps({"status": "error", "task_id": task_id, "error": str(e)})

    try:
        from hermes_state import SessionDB

        db = SessionDB()
        task_data = db.get_orchestration_task(task_id)

        if task_data:
            status = task_data.get("status", "unknown")
            result = task_data.get("result")
            error = task_data.get("error_message")

            if status == "completed":
                return json.dumps(
                    {
                        "status": "completed",
                        "task_id": task_id,
                        "result": result,
                    }
                )
            elif status == "error":
                return json.dumps(
                    {
                        "status": "error",
                        "task_id": task_id,
                        "error": error or "Unknown error",
                    }
                )
            elif status in ("pending", "running"):
                return json.dumps(
                    {
                        "status": "running",
                        "task_id": task_id,
                        "note": "Task is running but may have been restarted",
                    }
                )
            else:
                return json.dumps(
                    {
                        "status": status,
                        "task_id": task_id,
                    }
                )
    except Exception as e:
        logger.debug(f"Failed to check SQLite for task {task_id}: {e}")

    return json.dumps({"error": f"Unknown task_id: {task_id}"})


def profile_task_cancel(task_id: str, parent_agent=None) -> str:
    """
    Cancel an async profile delegation task.

    Args:
        task_id: The task_id returned by profile_delegate (with wait=False).
        parent_agent: Optional reference to parent agent (for consistency with other tools).

    Returns:
        JSON string with cancellation status.
    """
    from tools.profile_orchestrator import _profile_tasks

    if not task_id:
        return json.dumps({"error": "task_id is required"})

    entry = _profile_tasks.get(task_id)
    if entry:
        future = entry["future"]
        entry["_cancelled"] = True

        cancelled = False
        try:
            cancelled = future.cancel()
        except Exception as e:
            logger.debug("Failed to cancel future for task %s: %s", task_id, e)

        try:
            del _profile_tasks[task_id]
        except KeyError:
            pass

        try:
            from hermes_state import SessionDB

            db = SessionDB()
            db.update_orchestration_task(
                task_id=task_id,
                status="cancelled",
            )
        except Exception as e:
            logger.debug(f"Failed to update SQLite for cancelled task {task_id}: {e}")

        if cancelled:
            return json.dumps({"status": "cancelled", "task_id": task_id})
        else:
            return json.dumps(
                {
                    "status": "cancellation_attempted",
                    "task_id": task_id,
                    "note": "Task may already be running or completed",
                }
            )

    try:
        from hermes_state import SessionDB

        db = SessionDB()
        task_data = db.get_orchestration_task(task_id)

        if task_data:
            status = task_data.get("status")
            if status in ("completed", "error", "cancelled"):
                return json.dumps(
                    {
                        "status": "already_finished",
                        "task_id": task_id,
                        "previous_status": status,
                        "note": "Task already completed, cannot cancel",
                    }
                )
            else:
                db.update_orchestration_task(
                    task_id=task_id,
                    status="cancelled",
                )
                return json.dumps(
                    {
                        "status": "cancelled",
                        "task_id": task_id,
                        "note": "Task was not in memory but marked as cancelled in database",
                    }
                )
    except Exception as e:
        logger.debug(f"Failed to check SQLite for task {task_id}: {e}")

    return json.dumps({"error": f"Unknown task_id: {task_id}"})


def profile_task_search(
    query: str, status: str = None, limit: int = 20, parent_agent=None
) -> str:
    """
    Natural language search of past profile delegation tasks across ALL profiles.

    Searches every profile's orchestration_tasks database. Useful when you don't know
    which profile handled a task, or when you want a complete history view.

    Args:
        query: Search terms (e.g., "security issues auth.py", "docker deployment")
               Supports phrases: '"exact phrase"', boolean: "docker OR kubernetes"
        status: Optional filter: "completed", "running", "pending", "error", "cancelled"
        limit: Max results total (default 20)
        parent_agent: Optional reference to parent agent.

    Returns:
        JSON string with matching tasks from all profiles, including source_profile field.

    Example:
        profile_task_search("security issues")
        → Returns tasks from all profiles matching the query, each with source_profile field
    """
    if not query or not str(query).strip():
        return json.dumps({"error": "Query is required"})

    try:
        from hermes_cli.profiles import list_profiles, get_profile_dir
        from hermes_state import SessionDB

        profiles = list_profiles()
        if not profiles:
            return json.dumps({"count": 0, "tasks": [], "profiles_searched": 0})

        all_results = []
        profiles_with_tasks = 0

        for profile_info in profiles:
            try:
                profile_name = (
                    profile_info.name
                    if hasattr(profile_info, "name")
                    else str(profile_info)
                )
                profile_dir = get_profile_dir(profile_name)
                db_path = profile_dir / "state.db"

                if not db_path.exists():
                    continue

                db = SessionDB(db_path=db_path)
                tasks = db.search_orchestration_tasks(
                    query=str(query).strip(),
                    status=status,
                    limit=limit,
                )

                if tasks:
                    profiles_with_tasks += 1
                    for task in tasks:
                        task["source_profile"] = profile_name
                        all_results.append(task)

            except Exception as e:
                logger.debug(f"Failed to search profile {profile_info}: {e}")
                continue

        # Sort by created_at (newest first)
        all_results.sort(key=lambda x: x.get("created_at", 0), reverse=True)

        return json.dumps(
            {
                "count": len(all_results),
                "tasks": all_results[:limit],
                "profiles_searched": len(profiles),
                "profiles_with_matches": profiles_with_tasks,
            }
        )
    except Exception as e:
        logger.exception("profile_task_search failed")
        return json.dumps({"error": f"Search failed: {e}"})


# OpenAI Function-Calling Schema for profile_delegate
# ---------------------------------------------------------------------------

PROFILE_DELEGATE_SCHEMA = {
    "name": "profile_delegate",
    "description": (
        "CRITICAL: Use THIS tool for cross-profile orchestration, NOT delegate_task. "
        "This connects to a persistent Hermes profile worker via HTTP (like coder, researcher). "
        "The profile must exist in ~/.hermes/profiles/<profile_name>/ with its own gateway.\n\n"
        "DIFFERENCE from delegate_task:\n"
        "- profile_delegate: Uses persistent profile workers (separate HERMES_HOME, config, gateway)\n"
        "- delegate_task: Spawns temporary one-off subagents (same HERMES_HOME, ephemeral)\n\n"
        "WORKFLOW:\n"
        "1. Call profile_delegate(profile='profile2', goal='...')\n"
        "2. This connects to profile2's gateway at localhost:<port>\n"
        "3. Profile2 processes the task with its own config/memory\n"
        "4. Returns result from the persistent worker, not a temporary subagent\n\n"
        "WHEN TO USE:\n"
        "- Delegating to named profiles: coder, researcher, planner, etc.\n"
        "- Tasks requiring isolated configs/memories per profile\n"
        "- Multi-agent workflows with specialized persistent workers\n\n"
        "WHEN NOT TO USE:\n"
        "- For temporary subagents -> use delegate_task instead\n"
        "- For single tool calls -> call the tool directly"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "profile": {
                "type": "string",
                "description": "Target profile name (e.g., 'coder', 'researcher'). "
                "Must be a directory under ~/.hermes/profiles/.",
            },
            "goal": {
                "type": "string",
                "description": "Task description to send to the profile worker.",
            },
            "context": {
                "type": "string",
                "description": "Additional context or instructions for the profile worker.",
            },
            "wait": {
                "type": "boolean",
                "default": True,
                "description": "Wait for result (True, default) or return immediately with task_id (False).",
            },
            "timeout": {
                "type": "integer",
                "default": 300,
                "description": "Seconds to wait for result when wait=True (default 300).",
            },
            "allow_subagents": {
                "type": "boolean",
                "default": True,
                "description": "Whether the delegated agent may spawn one-off subagents via delegate_task. "
                "Set to false to force the agent to do all work itself without spawning helpers.",
            },
            "max_depth": {
                "type": "integer",
                "default": 2,
                "description": "Maximum delegation depth (default 2). Prevents infinite delegation chains. "
                "Parent=0, child=1, grandchild=2. When depth is reached, subagents cannot delegate further.",
            },
        },
        "required": ["profile", "goal"],
    },
}

PROFILE_TASK_CHECK_SCHEMA = {
    "name": "profile_task_check",
    "description": "Check the status of an async profile delegation task.",
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The task_id returned by profile_delegate with wait=False.",
            },
        },
        "required": ["task_id"],
    },
}

PROFILE_TASK_CANCEL_SCHEMA = {
    "name": "profile_task_cancel",
    "description": "Cancel an async profile delegation task.",
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The task_id returned by profile_delegate with wait=False.",
            },
        },
        "required": ["task_id"],
    },
}

PROFILE_TASK_SEARCH_SCHEMA = {
    "name": "profile_task_search",
    "description": (
        "Cross-profile natural language search of past delegation tasks. "
        "Searches ALL profiles' task histories to find tasks by goal or result. "
        "Each result includes source_profile to identify which profile handled it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": 'Search terms. Examples: "security issues", "auth refactor", "docker". '
                'Supports phrases: \'"exact phrase"\', boolean: "docker OR kubernetes"',
            },
            "status": {
                "type": "string",
                "description": 'Optional filter: "completed", "running", "pending", "error", "cancelled"',
            },
            "limit": {
                "type": "integer",
                "default": 20,
                "description": "Max results total across all profiles (default 20).",
            },
        },
        "required": ["query"],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="delegate_task",
    toolset="delegation",
    schema=DELEGATE_TASK_SCHEMA,
    handler=lambda args, **kw: delegate_task(
        goal=args.get("goal"),
        context=args.get("context"),
        toolsets=args.get("toolsets"),
        tasks=args.get("tasks"),
        max_iterations=args.get("max_iterations"),
        acp_command=args.get("acp_command"),
        acp_args=args.get("acp_args"),
        parent_agent=kw.get("parent_agent"),
    ),
    check_fn=check_delegate_requirements,
    emoji="🔀",
)

registry.register(
    name="profile_delegate",
    toolset="delegation",
    schema=PROFILE_DELEGATE_SCHEMA,
    handler=lambda args, **kw: profile_delegate(
        profile=args.get("profile"),
        goal=args.get("goal"),
        context=args.get("context"),
        wait=args.get("wait", True),
        timeout=args.get("timeout", 300),
        allow_subagents=args.get("allow_subagents", True),
        max_depth=args.get("max_depth", 2),
        parent_agent=kw.get("parent_agent"),
    ),
    check_fn=check_profile_requirements,
    emoji="🔀",
)

registry.register(
    name="profile_task_check",
    toolset="delegation",
    schema=PROFILE_TASK_CHECK_SCHEMA,
    handler=lambda args, **kw: profile_task_check(
        task_id=args.get("task_id"), parent_agent=kw.get("parent_agent")
    ),
    check_fn=check_profile_requirements,
    emoji="📋",
)

registry.register(
    name="profile_task_cancel",
    toolset="delegation",
    schema=PROFILE_TASK_CANCEL_SCHEMA,
    handler=lambda args, **kw: profile_task_cancel(
        task_id=args.get("task_id"), parent_agent=kw.get("parent_agent")
    ),
    check_fn=check_profile_requirements,
    emoji="✋",
)

registry.register(
    name="profile_task_search",
    toolset="delegation",
    schema=PROFILE_TASK_SEARCH_SCHEMA,
    handler=lambda args, **kw: profile_task_search(
        query=args.get("query"),
        status=args.get("status"),
        limit=args.get("limit", 10),
        parent_agent=kw.get("parent_agent"),
    ),
    check_fn=check_profile_requirements,
    emoji="🔍",
)
