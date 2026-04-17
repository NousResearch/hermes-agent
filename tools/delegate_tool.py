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
  - A stable system prompt (task goal + context flow via user message)

The parent's context only sees the delegation call and the summary result,
never the child's intermediate tool calls or reasoning.
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Tools that children must never have access to
DELEGATE_BLOCKED_TOOLS = frozenset([
    "delegate_task",   # no recursive delegation
    "clarify",         # no user interaction
    "send_message",    # no cross-platform side effects
    "execute_code",    # children should reason step-by-step, not write scripts
    # Note: "memory" removed - subagents have read-only memory access
    # Memory writes are blocked in memory.py based on subagent_memory_mode
])

# Toolsets that are stripped from subagents by default
# Can be overridden via config.yaml: delegation.allowed_toolsets
BLOCKED_TOOLSET_NAMES = frozenset([
    "delegation",  # delegate_task tool
    "clarify",     # clarify tool
    "code_execution",  # execute_code tool
    # Note: "memory" removed - subagents now have read-only memory access
])

# Allowlist of toolsets that subagents CAN use
# Set to None to inherit all parent toolsets (except blocked ones)
# Configure via config.yaml: delegation.allowed_toolsets
DEFAULT_ALLOWED_TOOLSETS = ["terminal", "file", "web", "mcp", "browser", "memory"]

# Configuration constants
_DEFAULT_MAX_CONCURRENT_CHILDREN = 3


def _get_max_concurrent_children() -> int:
    """Read delegation.max_concurrent_children from config.yaml, falling back
    to DELEGATION_MAX_CONCURRENT_CHILDREN env var, then the default (3)."""
    env_val = os.getenv("DELEGATION_MAX_CONCURRENT_CHILDREN")
    if env_val:
        try:
            return max(1, int(env_val))
        except (TypeError, ValueError):
            pass
    try:
        from hermes_cli.config import read_raw_config
        cfg = read_raw_config()
        val = cfg.get("delegation", {}).get("max_concurrent_children")
        if val is not None:
            return max(1, int(val))
    except Exception:
        pass
    return _DEFAULT_MAX_CONCURRENT_CHILDREN
MAX_DEPTH = 2  # parent (0) -> child (1) -> grandchild rejected (2)
DEFAULT_MAX_ITERATIONS = 50
DEFAULT_SUBAGENT_MEMORY_MODE = "read_only"  # "read_only" | "full" | "none"
DEFAULT_WORKSPACE_VISIBILITY = "inherit"


def check_delegate_requirements() -> bool:
    """Delegation has no external requirements -- always available."""
    return True


def _build_child_system_prompt(
    workspace_note: Optional[str] = None,
    **_kwargs,
) -> str:
    """Build a focused system prompt for a child agent.

    Task-specific content (goal, context) is NOT included here — it flows
    via run_conversation(user_message=...) as the first user message.
    This keeps the system prompt stable across subagents, improving
    prefix cache reuse on local LLM backends.
    """
    parts = [
        "You are a focused subagent working on a specific delegated task. "
        "Your task is provided in the first user message below.",
    ]
    if workspace_note and workspace_note.strip():
        parts.append(f"\n{workspace_note.strip()}")
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
        getattr(getattr(parent_agent, "_subdirectory_hints", None), "working_dir", None),
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


def _configure_child_workspace(child, task_index: int, task_cfg: Dict[str, Any], delegation_cfg: Dict[str, Any]):
    """Resolve workspace visibility for a delegated child and attach task overrides."""
    from tools.subagent_workspace import (
        build_workspace_overrides,
        resolve_parent_workspace_root,
        resolve_terminal_backend,
    )

    visibility = task_cfg.get("workspace_visibility")
    mappings = task_cfg.get("workspace_mappings")
    if mappings is None:
        mappings = delegation_cfg.get("workspace_mappings")
    if visibility is None:
        visibility = delegation_cfg.get("workspace_visibility", DEFAULT_WORKSPACE_VISIBILITY)
    if (visibility in (None, "", "inherit")) and mappings:
        visibility = "mapped"

    plan = build_workspace_overrides(
        visibility=visibility,
        mappings=mappings,
        workspace_root=resolve_parent_workspace_root(),
        child_token=f"subagent-{task_index}-{getattr(child, 'session_id', '') or task_index}",
        backend=resolve_terminal_backend(),
    )

    child._delegate_task_id = getattr(child, "session_id", None)
    child._delegate_task_env_overrides = plan.get("task_env_overrides")
    prompt_note = plan.get("prompt_note", "").strip()
    if prompt_note:
        child.ephemeral_system_prompt = _build_child_system_prompt(
            workspace_note=prompt_note,
        )
    # Only set context if not already set by _build_child_agent
    if not getattr(child, "_delegate_context", None):
        child._delegate_context = task_cfg.get("context")


def _strip_blocked_tools(toolsets: List[str]) -> List[str]:
    """Remove toolsets that contain only blocked tools."""
    # Use configurable blocked toolset names
    return [t for t in toolsets if t not in BLOCKED_TOOLSET_NAMES]


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

    def _callback(event_type: str, tool_name: str = None, preview: str = None, args=None, **kwargs):
        # event_type is one of: "tool.started", "tool.completed",
        # "reasoning.available", "_thinking", "subagent_progress"

        # "_thinking" / reasoning events
        if event_type in ("_thinking", "reasoning.available"):
            text = preview or tool_name or ""
            if spinner:
                short = (text[:55] + "...") if len(text) > 55 else text
                try:
                    spinner.print_above(f" {prefix}├─ 💭 \"{short}\"")
                except Exception as e:
                    logger.debug("Spinner print_above failed: %s", e)
            # Don't relay thinking to gateway (too noisy for chat)
            return

        # tool.completed — no display needed here (spinner shows on started)
        if event_type == "tool.completed":
            return

        # tool.started — display and batch for parent relay
        if spinner:
            short = (preview[:35] + "...") if preview and len(preview) > 35 else (preview or "")
            from agent.display import get_tool_emoji
            emoji = get_tool_emoji(tool_name or "")
            line = f" {prefix}├─ {emoji} {tool_name}"
            if short:
                line += f"  \"{short}\""
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
    parent_toolsets = set(getattr(parent_agent, "enabled_toolsets", None) or DEFAULT_ALLOWED_TOOLSETS)
    if toolsets:
        # Intersect with parent — subagent must not gain tools the parent lacks
        # Then apply allowlist
        child_toolsets = _strip_blocked_tools([t for t in toolsets if t in parent_toolsets])
        # Filter to allowed toolsets
        child_toolsets = [t for t in child_toolsets if t in DEFAULT_ALLOWED_TOOLSETS]
    elif parent_agent and getattr(parent_agent, "enabled_toolsets", None):
        child_toolsets = _strip_blocked_tools(parent_agent.enabled_toolsets)
        # Filter to allowed toolsets
        child_toolsets = [t for t in child_toolsets if t in DEFAULT_ALLOWED_TOOLSETS]
    else:
        child_toolsets = _strip_blocked_tools(DEFAULT_ALLOWED_TOOLSETS)

    workspace_hint = _resolve_workspace_hint(parent_agent)
    child_prompt = _build_child_system_prompt(workspace_note=workspace_hint)
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
    effective_acp_command = override_acp_command or getattr(parent_agent, "acp_command", None)
    effective_acp_args = list(override_acp_args if override_acp_args is not None else (getattr(parent_agent, "acp_args", []) or []))

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
        skip_memory=False,  # Enable memory for subagents (read-only mode enforced in memory_tool.py)
        subagent_memory_mode=DEFAULT_SUBAGENT_MEMORY_MODE,  # Pass read_only/full/none config
        clarify_callback=None,
        thinking_callback=child_thinking_cb,
        session_db=getattr(parent_agent, '_session_db', None),
        parent_session_id=getattr(parent_agent, 'session_id', None),
        providers_allowed=parent_agent.providers_allowed,
        providers_ignored=parent_agent.providers_ignored,
        providers_order=parent_agent.providers_order,
        provider_sort=parent_agent.provider_sort,
        tool_progress_callback=child_progress_cb,
        iteration_budget=None,  # fresh budget per subagent
        _is_subagent=True,  # Mark this agent as a subagent for memory/tool access control
        # Share parent's memory store (read-only access enforced in memory_tool.py)
        _shared_memory_store=getattr(parent_agent, '_memory_store', None),
    )
    child._print_fn = getattr(parent_agent, '_print_fn', None)
    # Store context for injection into user message (not system prompt)
    child._delegate_context = context
    # Set delegation depth so children can't spawn grandchildren
    child._delegate_depth = getattr(parent_agent, '_delegate_depth', 0) + 1

    # Share a credential pool with the child when possible so subagents can
    # rotate credentials on rate limits instead of getting pinned to one key.
    child_pool = _resolve_child_credential_pool(effective_provider, parent_agent)
    if child_pool is not None:
        child._credential_pool = child_pool

    # Register child for interrupt propagation
    if hasattr(parent_agent, '_active_children'):
        lock = getattr(parent_agent, '_active_children_lock', None)
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
    child_progress_cb = getattr(child, 'tool_progress_callback', None)

    # Restore parent tool names using the value saved before child construction
    # mutated the global. This is the correct parent toolset, not the child's.
    import model_tools
    _saved_tool_names = getattr(child, "_delegate_saved_tool_names",
                                list(model_tools._last_resolved_tool_names))

    child_pool = getattr(child, '_credential_pool', None)
    leased_cred_id = None
    if child_pool is not None:
        leased_cred_id = child_pool.acquire_lease()
        if leased_cred_id is not None:
            try:
                leased_entry = child_pool.current()
                if leased_entry is not None and hasattr(child, '_swap_credential'):
                    child._swap_credential(leased_entry)
            except Exception as exc:
                logger.debug("Failed to bind child to leased credential: %s", exc)

    try:
        # When share_sandbox is True, reuse the parent's sandbox container
        # by passing the parent's task_id instead of generating a new one.
        if getattr(child, '_share_sandbox', False) and parent_agent is not None:
            child_task_id = getattr(parent_agent, '_effective_task_id', None)
        else:
            child_task_id = getattr(child, "_delegate_task_id", None) or getattr(child, "session_id", None)
        child_env_overrides = getattr(child, "_delegate_task_env_overrides", None)
        if child_task_id and child_env_overrides:
            from tools.terminal_tool import register_task_env_overrides

            register_task_env_overrides(child_task_id, child_env_overrides)

        # Build user message: goal + context (context moved from system prompt
        # to user message for prefix cache stability across subagents)
        _user_msg = goal
        _child_context = getattr(child, "_delegate_context", None)
        if _child_context:
            _user_msg = f"{goal}\n\nCONTEXT:\n{_child_context}"
        result = child.run_conversation(user_message=_user_msg, task_id=child_task_id)

        # Flush any remaining batched progress to gateway
        if child_progress_cb and hasattr(child_progress_cb, '_flush'):
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
                    for tc in (msg.get("tool_calls") or []):
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
                    is_error = bool(
                        content and "error" in content[:80].lower()
                    )
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
                "input": _input_tokens if isinstance(_input_tokens, (int, float)) else 0,
                "output": _output_tokens if isinstance(_output_tokens, (int, float)) else 0,
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
        # Release the credential pool lease acquired at the top of the try
        # block. Without this, every delegated child leaks a lease, and the
        # pool gradually marks all entries as exhausted. Upstream has this;
        # the fork's rewrite for workspace env overrides accidentally dropped
        # it. Restored here.
        if child_pool is not None and leased_cred_id is not None:
            try:
                child_pool.release_lease(leased_cred_id)
            except Exception:
                logger.debug("Failed to release credential lease %s", leased_cred_id, exc_info=True)

        child_task_id = getattr(child, "_delegate_task_id", None) or getattr(child, "session_id", None)
        child_env_overrides = getattr(child, "_delegate_task_env_overrides", None)
        if child_task_id and child_env_overrides:
            try:
                from tools.terminal_tool import clear_task_env_overrides

                clear_task_env_overrides(child_task_id)
            except Exception:
                logger.debug("Failed to clear delegated task env overrides", exc_info=True)

        # Restore the parent's tool names so the process-global is correct
        # for any subsequent execute_code calls or other consumers.
        import model_tools

        saved_tool_names = getattr(child, "_delegate_saved_tool_names", None)
        if isinstance(saved_tool_names, list):
            model_tools._last_resolved_tool_names = list(saved_tool_names)

        # Remove child from active tracking

        # Unregister child from interrupt propagation
        if hasattr(parent_agent, '_active_children'):
            try:
                lock = getattr(parent_agent, '_active_children_lock', None)
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
    workspace_visibility: Optional[str] = None,
    workspace_mappings: Optional[List[Dict[str, Any]]] = None,
    tasks: Optional[List[Dict[str, Any]]] = None,
    max_iterations: Optional[int] = None,
    model: Optional[str] = None,
    acp_command: Optional[str] = None,
    acp_args: Optional[List[str]] = None,
    share_sandbox: Optional[bool] = None,
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

    # Resolve delegation credentials (provider:model pair).
    # When delegation.provider is configured, this resolves the full credential
    # bundle (base_url, api_key, api_mode) via the same runtime provider system
    # used by CLI/gateway startup.  When unconfigured, returns None values so
    # children inherit from the parent.
    try:
        creds = _resolve_delegation_credentials(cfg, parent_agent)
    except ValueError as exc:
        return tool_error(str(exc))

    # Per-call model override: the agent can request a specific model or tier
    # (small/medium/large) for this delegation.
    if model and isinstance(model, str) and model.strip():
        creds["model"] = _resolve_model_or_tier(model.strip())

    # Normalize to task list
    max_children = _get_max_concurrent_children()
    if tasks and isinstance(tasks, list):
        if len(tasks) > max_children:
            return tool_error(
                f"Too many tasks: {len(tasks)} provided, but "
                f"max_concurrent_children is {max_children}. "
                f"Either reduce the task count, split into multiple "
                f"delegate_task calls, or increase "
                f"delegation.max_concurrent_children in config.yaml."
            )
        task_list = tasks
    elif goal and isinstance(goal, str) and goal.strip():
        task_list = [{
            "goal": goal,
            "context": context,
            "toolsets": toolsets,
            "workspace_visibility": workspace_visibility,
            "workspace_mappings": workspace_mappings,
            "share_sandbox": share_sandbox,
        }]
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
            # Per-task model override takes precedence over top-level model
            raw_task_model = str(t.get("model") or "").strip()
            task_model = _resolve_model_or_tier(raw_task_model) if raw_task_model else creds["model"]
            child = _build_child_agent(
                task_index=i, goal=t["goal"], context=t.get("context"),
                toolsets=t.get("toolsets") or toolsets, model=task_model,
                max_iterations=effective_max_iter, parent_agent=parent_agent,
                override_provider=creds["provider"], override_base_url=creds["base_url"],
                override_api_key=creds["api_key"],
                override_api_mode=creds["api_mode"],
                override_acp_command=t.get("acp_command") or acp_command,
                override_acp_args=t.get("acp_args") or acp_args,
            )
            if "workspace_visibility" not in t and workspace_visibility is not None:
                t["workspace_visibility"] = workspace_visibility
            if "workspace_mappings" not in t and workspace_mappings is not None:
                t["workspace_mappings"] = workspace_mappings
            _configure_child_workspace(child, i, t, cfg)
            # Override with correct parent tool names (before child construction mutated global)
            child._delegate_saved_tool_names = _parent_tool_names
            # Store share_sandbox flag for _run_single_child to use.
            # When True, child reuses the parent's sandbox container.
            task_share = bool(t.get("share_sandbox") or share_sandbox)
            child._share_sandbox = task_share
            if task_share:
                # Inform the child it's sharing the parent's sandbox
                child.ephemeral_system_prompt = (
                    getattr(child, 'ephemeral_system_prompt', '') +
                    "\n\nSHARED SANDBOX: You are sharing the parent agent's sandbox "
                    "container. Files you create or modify are visible to the parent "
                    "immediately. The working directory and installed packages are "
                    "already set up — do not reinstall or re-clone."
                )
            children.append((i, t, child))
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
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
        spinner_ref = getattr(parent_agent, '_delegate_spinner', None)

        with ThreadPoolExecutor(max_workers=max_children) as executor:
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

            # Poll futures with interrupt checking.  as_completed() blocks
            # until ALL futures finish — if a child agent gets stuck,
            # the parent blocks forever even after interrupt propagation.
            # Instead, use wait() with a short timeout so we can bail
            # when the parent is interrupted.
            pending = set(futures.keys())
            while pending:
                if getattr(parent_agent, "_interrupt_requested", False) is True:
                    # Parent interrupted — collect whatever finished and
                    # abandon the rest.  Children already received the
                    # interrupt signal; we just can't wait forever.
                    for f in pending:
                        idx = futures[f]
                        if f.done():
                            try:
                                entry = f.result()
                            except Exception as exc:
                                entry = {
                                    "task_index": idx,
                                    "status": "error",
                                    "summary": None,
                                    "error": str(exc),
                                    "api_calls": 0,
                                    "duration_seconds": 0,
                                }
                        else:
                            entry = {
                                "task_index": idx,
                                "status": "interrupted",
                                "summary": None,
                                "error": "Parent agent interrupted — child did not finish in time",
                                "api_calls": 0,
                                "duration_seconds": 0,
                            }
                        results.append(entry)
                        completed_count += 1
                    break

                from concurrent.futures import wait as _cf_wait, FIRST_COMPLETED
                done, pending = _cf_wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                for future in done:
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
                        except Exception as e:
                            logger.debug("Spinner update_text failed: %s", e)

        # Sort by task_index so results match input order
        results.sort(key=lambda r: r["task_index"])

    # Notify parent's memory provider of delegation outcomes
    if parent_agent and hasattr(parent_agent, '_memory_manager') and parent_agent._memory_manager:
        for entry in results:
            try:
                _task_goal = task_list[entry["task_index"]]["goal"] if entry["task_index"] < len(task_list) else ""
                parent_agent._memory_manager.on_delegation(
                    task=_task_goal,
                    result=entry.get("summary", "") or "",
                    child_session_id=getattr(children[entry["task_index"]][2], "session_id", "") if entry["task_index"] < len(children) else "",
                )
            except Exception:
                pass

    total_duration = round(time.monotonic() - overall_start, 2)

    return json.dumps({
        "results": results,
        "total_duration_seconds": total_duration,
    }, ensure_ascii=False)


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
        api_key = (
            configured_api_key
            or os.getenv("OPENAI_API_KEY", "").strip()
        )
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
# Model tier resolution
# ---------------------------------------------------------------------------

_DEFAULT_TIERS = {"small": None, "medium": None, "large": None}


def _load_model_tiers() -> dict:
    """Load model_tiers from delegation config.

    Returns a dict mapping tier names (small/medium/large) to model names.
    Falls back to empty tiers when unconfigured.
    """
    cfg = _load_config()
    tiers = cfg.get("model_tiers", {})
    if not isinstance(tiers, dict):
        return dict(_DEFAULT_TIERS)
    result = dict(_DEFAULT_TIERS)
    for tier in result:
        val = str(tiers.get(tier) or "").strip()
        if val:
            result[tier] = val
    return result


def _resolve_model_or_tier(model_spec: str) -> str:
    """Resolve a model specification that may be a tier name or a model name.

    If model_spec is 'small', 'medium', or 'large', resolves to the
    configured model name for that tier.  Otherwise returns model_spec
    as-is (assumed to be a direct model name).
    """
    if not model_spec:
        return model_spec
    lowered = model_spec.strip().lower()
    if lowered in ("small", "medium", "large"):
        tiers = _load_model_tiers()
        resolved = tiers.get(lowered)
        if resolved:
            logger.info("model tier '%s' resolved to '%s'", lowered, resolved)
            return resolved
        logger.warning(
            "model tier '%s' requested but not configured in "
            "delegation.model_tiers; falling back to default model",
            lowered,
        )
        return ""  # empty → inherit default
    return model_spec


def list_models(parent_agent=None) -> str:
    """Return available models and their delegation tiers as JSON.

    Reads custom_providers from config.yaml to enumerate models, and
    delegation.model_tiers for tier assignments.
    """
    # Load custom providers to enumerate available models
    models = []
    try:
        from cli import CLI_CONFIG
        full_cfg = CLI_CONFIG
    except Exception:
        try:
            from hermes_cli.config import load_config
            full_cfg = load_config()
        except Exception:
            full_cfg = {}

    # Get the current/default model
    model_cfg = full_cfg.get("model", {})
    if isinstance(model_cfg, dict):
        default_model = model_cfg.get("default", "")
    else:
        default_model = str(model_cfg or "")

    # Enumerate models from custom_providers
    for cp in full_cfg.get("custom_providers", []):
        provider_name = cp.get("name", "unknown")
        for model_name, model_info in (cp.get("models") or {}).items():
            ctx = None
            if isinstance(model_info, dict):
                ctx = model_info.get("context_length")
            models.append({
                "name": model_name,
                "provider": provider_name,
                "context_length": ctx,
            })

    # Load tiers
    tiers = _load_model_tiers()

    # Assign tier labels to models
    tier_by_model = {}
    for tier_name, tier_model in tiers.items():
        if tier_model:
            tier_by_model[tier_model] = tier_name

    for m in models:
        m["tier"] = tier_by_model.get(m["name"])
        m["is_default"] = m["name"] == default_model

    return json.dumps({
        "models": models,
        "tiers": {k: v for k, v in tiers.items() if v},
        "default_model": default_model,
        "usage_hint": (
            "Use 'small' for simple tasks (summarization, formatting, file listing). "
            "Use 'medium' (default) for standard work. "
            "Use 'large' for complex reasoning, peer review, or when you're stuck. "
            "You can pass tier names ('small', 'medium', 'large') or model names directly "
            "as the 'model' parameter in delegate_task."
        ),
    }, indent=2)


LIST_MODELS_SCHEMA = {
    "name": "list_models",
    "description": (
        "List available models for delegation with their tiers (small/medium/large). "
        "Use this to discover which models are available before delegating tasks. "
        "Returns model names, providers, context lengths, and tier assignments."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
    },
}


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
        "- Each subagent gets its own terminal session (separate working directory and state) "
        "unless share_sandbox=true, which reuses the parent's sandbox.\n"
        "- workspace_visibility defaults to 'inherit'. Use 'full_ro' or 'mapped' "
        "when you need stricter filesystem isolation in Docker sandboxes.\n"
        "- Results are always returned as an array, one entry per task.\n\n"
        "MODEL SELECTION:\n"
        "- You can set 'model' to a tier name ('small', 'medium', 'large') "
        "or a specific model name. Use list_models to see available options.\n"
        "- 'small': fast/cheap model for simple tasks (file exploration, "
        "summarization, formatting, lookups)\n"
        "- 'medium' or omit: default model for standard work\n"
        "- 'large': most capable model for complex reasoning, peer review, "
        "or when you're stuck and need to escalate\n"
        "- Each task in a batch can use a different model."
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
            "model": {
                "type": "string",
                "description": (
                    "Model to use for this subagent. Default: inherits from "
                    "delegation config or parent model. Use a smaller/faster model "
                    "(e.g. 'gemma4-nothink') for simple tasks like summarization, "
                    "formatting, or lookups. Keep the default for complex reasoning, "
                    "debugging, or multi-step implementation."
                ),
            },
            "workspace_visibility": {
                "type": "string",
                "enum": ["inherit", "full_rw", "full_ro", "mapped"],
                "description": (
                    "Filesystem visibility for the child sandbox. "
                    "'inherit' keeps the current backend behavior. "
                    "'full_rw' mounts the full parent workspace at /workspace read-write. "
                    "'full_ro' mounts it read-only. "
                    "'mapped' exposes only the paths listed in workspace_mappings."
                ),
            },
            "workspace_mappings": {
                "type": "array",
                "description": (
                    "Used only with workspace_visibility='mapped'. "
                    "Each mapping source must stay inside the parent workspace. "
                    "target defaults under /workspace and read_only defaults to false."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "Workspace-relative or absolute path inside the parent workspace.",
                        },
                        "target": {
                            "type": "string",
                            "description": "Path inside the child sandbox under /workspace.",
                        },
                        "read_only": {
                            "type": "boolean",
                            "description": "Mount this mapping read-only.",
                        },
                    },
                    "required": ["source"],
                },
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
                            "description": "Toolsets for this specific task. Use 'web' for network access, 'terminal' for shell.",
                        },
                        "model": {
                            "type": "string",
                            "description": "Model override for this task (e.g. 'gemma4-nothink' for simple work).",
                        },
                        "share_sandbox": {
                            "type": "boolean",
                            "description": "Share the parent's sandbox instead of creating an isolated one for this task.",
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
                        "workspace_visibility": {
                            "type": "string",
                            "enum": ["inherit", "full_rw", "full_ro", "mapped"],
                            "description": "Workspace visibility for this task's sandbox.",
                        },
                        "workspace_mappings": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "source": {"type": "string"},
                                    "target": {"type": "string"},
                                    "read_only": {"type": "boolean"},
                                },
                                "required": ["source"],
                            },
                            "description": "Workspace mappings for this task when workspace_visibility='mapped'.",
                        },
                    },
                    "required": ["goal"],
                },
                "maxItems": 3,
                "description": (
                    "Batch mode: tasks to run in parallel (limit configurable via delegation.max_concurrent_children, default 3). Each gets "
                    "its own subagent with isolated context and terminal session. "
                    "When provided, top-level goal/context are ignored. "
                    "Top-level toolsets and workspace visibility settings act as defaults."
                ),
            },
            "max_iterations": {
                "type": "integer",
                "description": (
                    "Max tool-calling turns per subagent (default: 50). "
                    "Only set lower for simple tasks."
                ),
            },
            "share_sandbox": {
                "type": "boolean",
                "description": (
                    "When true, the subagent shares the parent's sandbox container "
                    "instead of getting an isolated one. The subagent sees the same "
                    "filesystem, installed packages, and working directory as the parent. "
                    "Use this for explorer subagents that need access to the parent's "
                    "prepared workspace (e.g. source code already cloned, dependencies "
                    "installed). Default: false (isolated sandbox)."
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


# --- Registry ---
from tools.registry import registry, tool_error  # noqa: E402

registry.register(
    name="delegate_task",
    toolset="delegation",
    schema=DELEGATE_TASK_SCHEMA,
    handler=lambda args, **kw: delegate_task(
        goal=args.get("goal"),
        context=args.get("context"),
        toolsets=args.get("toolsets"),
        workspace_visibility=args.get("workspace_visibility"),
        workspace_mappings=args.get("workspace_mappings"),
        tasks=args.get("tasks"),
        max_iterations=args.get("max_iterations"),
        model=args.get("model"),
        acp_command=args.get("acp_command"),
        acp_args=args.get("acp_args"),
        share_sandbox=args.get("share_sandbox"),
        parent_agent=kw.get("parent_agent")),
    check_fn=check_delegate_requirements,
    emoji="🔀",
)

registry.register(
    name="list_models",
    toolset="delegation",
    schema=LIST_MODELS_SCHEMA,
    handler=lambda args, **kw: list_models(parent_agent=kw.get("parent_agent")),
    emoji="📋",
)
