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
logger = logging.getLogger(__name__)
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


def _infer_model_profile(toolsets: Optional[List[str]], goal: Optional[str] = None) -> str:
    """Infer best-fit profile from requested toolsets/task intent.

    First applies any user-configured ordered routing rules from config.yaml.
    Falls back to simple built-in heuristics if no rule matches.
    """
    normalized = {str(t).strip().lower() for t in (toolsets or []) if str(t).strip()}
    goal_text = (goal or "").strip().lower()

    try:
        from hermes_cli.config import load_config
        config = load_config()
        routing = config.get("model_routing", {}) if isinstance(config, dict) else {}
        rules = routing.get("rules", []) if isinstance(routing, dict) else []
        if isinstance(rules, list):
            for rule in rules:
                if not isinstance(rule, dict):
                    continue
                profile = str(rule.get("profile", "") or "").strip().lower()
                if not profile:
                    continue
                toolset_any = {str(t).strip().lower() for t in rule.get("if_toolsets_any", []) if str(t).strip()}
                goal_matches = [str(t).strip().lower() for t in rule.get("if_goal_matches", []) if str(t).strip()]
                if toolset_any and not (toolset_any & normalized):
                    continue
                if goal_matches and not any(token in goal_text for token in goal_matches):
                    continue
                if toolset_any or goal_matches:
                    return profile
    except Exception:
        pass

    if {"terminal", "file"} & normalized:
        return "coding"
    if "web" in normalized or "browser" in normalized:
        return "research"
    if any(token in goal_text for token in ("plan", "roadmap", "spec", "design")):
        return "planning"
    return "planning"


def _resolve_profile_credentials(profile_name: str) -> dict:
    """Resolve credentials/model for a named model profile."""
    from hermes_cli.runtime_provider import resolve_model_profile, resolve_runtime_provider

    profile_cfg = resolve_model_profile(profile_name)
    configured_model = profile_cfg.get("model") or None
    configured_provider = profile_cfg.get("provider") or None
    configured_base_url = profile_cfg.get("base_url") or None
    configured_api_key = profile_cfg.get("api_key") or None

    if not (configured_provider or configured_base_url or configured_api_key):
        return {
            "model": configured_model,
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
            "profile": profile_name,
        }

    runtime = resolve_runtime_provider(
        requested=configured_provider,
        explicit_api_key=configured_api_key,
        explicit_base_url=configured_base_url,
    )
    api_key = runtime.get("api_key", "")
    if not api_key:
        raise ValueError(
            f"Model profile '{profile_name}' resolved but has no API key. "
            f"Set profile api_key/api_key_env or provider credentials."
        )

    return {
        "model": configured_model,
        "provider": runtime.get("provider"),
        "base_url": runtime.get("base_url"),
        "api_key": api_key,
        "api_mode": runtime.get("api_mode"),
        "profile": profile_name,
    }


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
                except Exception as e:
                    logger.debug("Spinner print_above failed: %s", e)
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
            except Exception as e:
                logger.debug("Spinner print_above failed: %s", e)

        if parent_cb:
            _batch.append(tool_name)
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


def _run_single_child(
    task_index: int,
    goal: str,
    context: Optional[str],
    toolsets: Optional[List[str]],
    model: Optional[str],
    max_iterations: int,
    parent_agent,
    task_count: int = 1,
    # Credential overrides from delegation config (provider:model resolution)
    override_provider: Optional[str] = None,
    override_base_url: Optional[str] = None,
    override_api_key: Optional[str] = None,
    override_api_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Spawn and run a single child agent. Called from within a thread.
    Returns a structured result dict.

    When override_* params are set (from delegation config), the child uses
    those credentials instead of inheriting from the parent.  This enables
    routing subagents to a different provider:model pair (e.g. cheap/fast
    model on OpenRouter while the parent runs on Nous Portal).
    """
    from run_agent import AIAgent

    child_start = time.monotonic()

    # When no explicit toolsets given, inherit from parent's enabled toolsets
    # so disabled tools (e.g. web) don't leak to subagents.
    if toolsets:
        child_toolsets = _strip_blocked_tools(toolsets)
    elif parent_agent and getattr(parent_agent, "enabled_toolsets", None):
        child_toolsets = _strip_blocked_tools(parent_agent.enabled_toolsets)
    else:
        child_toolsets = _strip_blocked_tools(DEFAULT_TOOLSETS)

    child_prompt = _build_child_system_prompt(goal, context)

    try:
        # Extract parent's API key so subagents inherit auth (e.g. Nous Portal).
        parent_api_key = getattr(parent_agent, "api_key", None)
        if (not parent_api_key) and hasattr(parent_agent, "_client_kwargs"):
            parent_api_key = parent_agent._client_kwargs.get("api_key")

        # Build progress callback to relay tool calls to parent display
        child_progress_cb = _build_child_progress_callback(task_index, parent_agent, task_count)

        # Share the parent's iteration budget so subagent tool calls
        # count toward the session-wide limit.
        shared_budget = getattr(parent_agent, "iteration_budget", None)

        # Resolve effective credentials: config override > parent inherit
        effective_model = model or parent_agent.model
        effective_provider = override_provider or getattr(parent_agent, "provider", None)
        effective_base_url = override_base_url or parent_agent.base_url
        effective_api_key = override_api_key or parent_api_key
        effective_api_mode = override_api_mode or getattr(parent_agent, "api_mode", None)

        child = AIAgent(
            base_url=effective_base_url,
            api_key=effective_api_key,
            model=effective_model,
            provider=effective_provider,
            api_mode=effective_api_mode,
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
            session_db=getattr(parent_agent, '_session_db', None),
            providers_allowed=parent_agent.providers_allowed,
            providers_ignored=parent_agent.providers_ignored,
            providers_order=parent_agent.providers_order,
            provider_sort=parent_agent.provider_sort,
            tool_progress_callback=child_progress_cb,
            iteration_budget=shared_budget,
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
            except Exception as e:
                logger.debug("Progress callback flush failed: %s", e)

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
            except (ValueError, UnboundLocalError) as e:
                logger.debug("Could not remove child from active_children: %s", e)


def delegate_task(
    goal: Optional[str] = None,
    context: Optional[str] = None,
    toolsets: Optional[List[str]] = None,
    tasks: Optional[List[Dict[str, Any]]] = None,
    max_iterations: Optional[int] = None,
    model_profile: Optional[str] = None,
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
        task_list = [{"goal": goal, "context": context, "toolsets": toolsets, "model_profile": model_profile}]
    else:
        return json.dumps({"error": "Provide either 'goal' (single task) or 'tasks' (batch)."})

    if not task_list:
        return json.dumps({"error": "No tasks provided."})

    # Validate each task has a goal
    for i, task in enumerate(task_list):
        if not task.get("goal", "").strip():
            return json.dumps({"error": f"Task {i} is missing a 'goal'."})

    # Base legacy delegation credentials (provider/model overrides in
    # delegation.*). If unset, this resolves to inherit-parent behavior.
    try:
        base_creds = _resolve_delegation_credentials(cfg, parent_agent)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    legacy_override_active = bool((cfg.get("model") or "").strip() or (cfg.get("provider") or "").strip())

    # Build task specs with per-task credential/model routing.
    prepared_tasks = []
    default_profile = (cfg.get("model_profile") or "").strip() or None
    model_profiles_cfg = cfg.get("model_profiles", {}) if isinstance(cfg.get("model_profiles"), dict) else {}

    def _profile_has_explicit_config(name: Optional[str]) -> bool:
        key = (name or "").strip().lower()
        if not key:
            return False
        raw = model_profiles_cfg.get(key, {})
        if not isinstance(raw, dict):
            return False
        return any(str(raw.get(field, "") or "").strip() for field in ("model", "provider", "base_url", "api_key_env", "api_key"))

    for i, task in enumerate(task_list):
        task_toolsets = task.get("toolsets") or toolsets
        requested_profile = (
            (task.get("model_profile") if isinstance(task, dict) else None)
            or model_profile
            or default_profile
        )
        inferred_profile = requested_profile or _infer_model_profile(task_toolsets, task.get("goal"))

        creds = dict(base_creds)
        profile_used = None

        if not legacy_override_active and (requested_profile or _profile_has_explicit_config(inferred_profile)):
            try:
                profile_creds = _resolve_profile_credentials(inferred_profile)
            except ValueError as exc:
                return json.dumps({"error": str(exc)})
            for key in ("model", "provider", "base_url", "api_key", "api_mode"):
                if profile_creds.get(key) is not None:
                    creds[key] = profile_creds.get(key)
            profile_used = profile_creds.get("profile")

        prepared_tasks.append({
            "task_index": i,
            "goal": task["goal"],
            "context": task.get("context"),
            "toolsets": task_toolsets,
            "creds": creds,
            "profile": profile_used,
        })

    overall_start = time.monotonic()
    results = []

    n_tasks = len(prepared_tasks)
    # Track goal labels for progress display (truncated for readability)
    task_labels = [t["goal"][:40] for t in prepared_tasks]

    if n_tasks == 1:
        # Single task -- run directly (no thread pool overhead)
        t = prepared_tasks[0]
        creds = t["creds"]
        result = _run_single_child(
            task_index=0,
            goal=t["goal"],
            context=t.get("context"),
            toolsets=t.get("toolsets"),
            model=creds["model"],
            max_iterations=effective_max_iter,
            parent_agent=parent_agent,
            task_count=1,
            override_provider=creds["provider"],
            override_base_url=creds["base_url"],
            override_api_key=creds["api_key"],
            override_api_mode=creds["api_mode"],
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
            for i, t in enumerate(prepared_tasks):
                creds = t["creds"]
                future = executor.submit(
                    _run_single_child,
                    task_index=i,
                    goal=t["goal"],
                    context=t.get("context"),
                    toolsets=t.get("toolsets"),
                    model=creds["model"],
                    max_iterations=effective_max_iter,
                    parent_agent=parent_agent,
                    task_count=n_tasks,
                    override_provider=creds["provider"],
                    override_base_url=creds["base_url"],
                    override_api_key=creds["api_key"],
                    override_api_mode=creds["api_mode"],
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
                    except Exception as e:
                        logger.debug("Spinner update_text failed: %s", e)

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


def _resolve_delegation_credentials(cfg: dict, parent_agent) -> dict:
    """Resolve credentials for subagent delegation.

    If ``delegation.provider`` is configured, resolves the full credential
    bundle (base_url, api_key, api_mode, provider) via the runtime provider
    system — the same path used by CLI/gateway startup.  This lets subagents
    run on a completely different provider:model pair.

    If no provider is configured, returns None values so the child inherits
    everything from the parent agent.

    Raises ValueError with a user-friendly message on credential failure.
    """
    configured_model = cfg.get("model") or None
    configured_provider = cfg.get("provider") or None

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
            f"Check that the provider is configured (API key set, valid provider name). "
            f"Available providers: openrouter, nous, zai, kimi-coding, minimax."
        ) from exc

    api_key = runtime.get("api_key", "")
    if not api_key:
        raise ValueError(
            f"Delegation provider '{configured_provider}' resolved but has no API key. "
            f"Set the appropriate environment variable or run 'hermes login'."
        )

    return {
        "model": configured_model,
        "provider": runtime.get("provider"),
        "base_url": runtime.get("base_url"),
        "api_key": api_key,
        "api_mode": runtime.get("api_mode"),
    }


def _load_config() -> dict:
    """Load delegation + model profile config.

    Returns a flat dict with delegation keys (max_iterations/model/provider/
    model_profile) plus model_profiles map.
    """
    full = {}
    try:
        from cli import CLI_CONFIG
        if isinstance(CLI_CONFIG, dict) and CLI_CONFIG:
            full = CLI_CONFIG
    except Exception:
        pass
    if not full:
        try:
            from hermes_cli.config import load_config
            full = load_config()
        except Exception:
            full = {}

    if not isinstance(full, dict):
        full = {}

    delegation = full.get("delegation", {})
    if not isinstance(delegation, dict):
        delegation = {}
    model_profiles = full.get("model_profiles", {})
    if not isinstance(model_profiles, dict):
        model_profiles = {}

    merged = dict(delegation)
    merged["model_profiles"] = model_profiles
    return merged


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
                    "Default: inherits your enabled toolsets. "
                    "Common patterns: ['terminal', 'file'] for code work, "
                    "['web'] for research, ['terminal', 'file', 'web'] for "
                    "full-stack tasks."
                ),
            },
            "model_profile": {
                "type": "string",
                "description": (
                    "Optional model profile to use for this delegated task. "
                    "Examples: 'coding', 'planning', 'research'."
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
                        "model_profile": {
                            "type": "string",
                            "description": "Model profile for this specific task",
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
        max_iterations=args.get("max_iterations"),
        model_profile=args.get("model_profile"),
        parent_agent=kw.get("parent_agent")),
    check_fn=check_delegate_requirements,
)
