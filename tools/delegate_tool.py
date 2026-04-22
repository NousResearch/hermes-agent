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

import enum
import json
import logging

logger = logging.getLogger(__name__)
import os
import threading
import time
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
    as_completed,
)
from typing import Any, Dict, List, Optional

from toolsets import TOOLSETS
from tools import file_state
from utils import base_url_hostname


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

# Build a description fragment listing toolsets available for subagents.
# Excludes toolsets where ALL tools are blocked, composite/platform toolsets
# (hermes-* prefixed), and scenario toolsets.
#
# NOTE: "delegation" is in this exclusion set so the subagent-facing
# capability hint string (_TOOLSET_LIST_STR) doesn't advertise it as a
# toolset to request explicitly — the correct mechanism for nested
# delegation is role='orchestrator', which re-adds "delegation" in
# _build_child_agent regardless of this exclusion.
_EXCLUDED_TOOLSET_NAMES = frozenset({"debugging", "safe", "delegation", "moa", "rl"})
_SUBAGENT_TOOLSETS = sorted(
    name
    for name, defn in TOOLSETS.items()
    if name not in _EXCLUDED_TOOLSET_NAMES
    and not name.startswith("hermes-")
    and not all(t in DELEGATE_BLOCKED_TOOLS for t in defn.get("tools", []))
)
_TOOLSET_LIST_STR = ", ".join(f"'{n}'" for n in _SUBAGENT_TOOLSETS)

_DEFAULT_MAX_CONCURRENT_CHILDREN = 3
MAX_DEPTH = 1  # flat by default: parent (0) -> child (1); grandchild rejected unless max_spawn_depth raised.
# Configurable depth cap consulted by _get_max_spawn_depth; MAX_DEPTH
# stays as the default fallback and is still the symbol tests import.
_MIN_SPAWN_DEPTH = 1
_MAX_SPAWN_DEPTH_CAP = 3


# ---------------------------------------------------------------------------
# Runtime state: pause flag + active subagent registry
#
# Consumed by the TUI observability layer (overlay/control surface) and the
# gateway RPCs `delegation.pause`, `delegation.status`, `subagent.interrupt`.
# Kept module-level so they span every delegate_task invocation in the
# process, including nested orchestrator -> worker chains.
# ---------------------------------------------------------------------------

_spawn_pause_lock = threading.Lock()
_spawn_paused: bool = False

_active_subagents_lock = threading.Lock()
# subagent_id -> mutable record tracking the live child agent.  Stays only
# for the lifetime of the run; _run_single_child is the owner.
_active_subagents: Dict[str, Dict[str, Any]] = {}


def set_spawn_paused(paused: bool) -> bool:
    """Globally block/unblock new delegate_task spawns.

    Active children keep running; only NEW calls to delegate_task fail fast
    with a "spawning paused" error until unblocked.  Returns the new state.
    """
    global _spawn_paused
    with _spawn_pause_lock:
        _spawn_paused = bool(paused)
        return _spawn_paused


def is_spawn_paused() -> bool:
    with _spawn_pause_lock:
        return _spawn_paused


def _register_subagent(record: Dict[str, Any]) -> None:
    sid = record.get("subagent_id")
    if not sid:
        return
    with _active_subagents_lock:
        _active_subagents[sid] = record


def _unregister_subagent(subagent_id: str) -> None:
    with _active_subagents_lock:
        _active_subagents.pop(subagent_id, None)


def interrupt_subagent(subagent_id: str) -> bool:
    """Request that a single running subagent stop at its next iteration boundary.

    Does not hard-kill the worker thread (Python can't); sets the child's
    interrupt flag which propagates to in-flight tools and recurses into
    grandchildren via AIAgent.interrupt().  Returns True if a matching
    subagent was found.
    """
    with _active_subagents_lock:
        record = _active_subagents.get(subagent_id)
    if not record:
        return False
    agent = record.get("agent")
    if agent is None:
        return False
    try:
        agent.interrupt(f"Interrupted via TUI ({subagent_id})")
    except Exception as exc:
        logger.debug("interrupt_subagent(%s) failed: %s", subagent_id, exc)
        return False
    return True


def list_active_subagents() -> List[Dict[str, Any]]:
    """Snapshot of the currently running subagent tree.

    Each record: {subagent_id, parent_id, depth, goal, model, started_at,
    tool_count, status}.  Safe to call from any thread — returns a copy.
    """
    with _active_subagents_lock:
        return [
            {k: v for k, v in r.items() if k != "agent"}
            for r in _active_subagents.values()
        ]


def _extract_output_tail(
    result: Dict[str, Any],
    *,
    max_entries: int = 12,
    max_chars: int = 8000,
) -> List[Dict[str, Any]]:
    """Pull the last N tool-call results from a child's conversation.

    Powers the overlay's "Output" section — the cc-swarm-parity feature.
    We reuse the same messages list the trajectory saver walks, taking
    only the tail to keep event payloads small.  Each entry is
    ``{tool, preview, is_error}``.
    """
    messages = result.get("messages") if isinstance(result, dict) else None
    if not isinstance(messages, list):
        return []

    # Walk in reverse to build a tail; stop when we have enough.
    tail: List[Dict[str, Any]] = []
    pending_call_by_id: Dict[str, str] = {}

    # First pass (forward): build tool_call_id -> tool_name map
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                tc_id = tc.get("id")
                fn = tc.get("function") or {}
                if tc_id:
                    pending_call_by_id[tc_id] = str(fn.get("name") or "tool")

    # Second pass (reverse): pick tool results, newest first
    for msg in reversed(messages):
        if len(tail) >= max_entries:
            break
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        content = msg.get("content") or ""
        if not isinstance(content, str):
            content = str(content)
        is_error = _looks_like_error_output(content)
        tool_name = pending_call_by_id.get(msg.get("tool_call_id") or "", "tool")
        # Preserve line structure so the overlay's wrapped scroll region can
        # show real output rather than a whitespace-collapsed blob. We still
        # cap the payload size to keep events bounded.
        preview = content[:max_chars]
        tail.append({"tool": tool_name, "preview": preview, "is_error": is_error})

    tail.reverse()  # restore chronological order for display
    return tail


def _looks_like_error_output(content: str) -> bool:
    """Conservative stderr/error detector for tool-result previews.

    The old heuristic flagged any preview containing the substring "error",
    which painted perfectly normal terminal/json output red.  We now only
    mark output as an error when there is stronger evidence:
      - structured JSON with an ``error`` key
      - structured JSON with ``status`` of error/failed
      - first line starts with a classic error marker
    """
    if not content:
        return False

    head = content.lstrip()
    if head.startswith("{") or head.startswith("["):
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                if parsed.get("error"):
                    return True
                status = str(parsed.get("status") or "").strip().lower()
                if status in {"error", "failed", "failure", "timeout"}:
                    return True
        except Exception:
            pass

    first = content.splitlines()[0].strip().lower() if content.splitlines() else ""
    return (
        first.startswith("error:")
        or first.startswith("failed:")
        or first.startswith("traceback ")
        or first.startswith("exception:")
    )


def _normalize_role(r: Optional[str]) -> str:
    """Normalise a caller-provided role to 'leaf' or 'orchestrator'.

    None/empty -> 'leaf'.  Unknown strings coerce to 'leaf' with a
    warning log (matches the silent-degrade pattern of
    _get_orchestrator_enabled).  _build_child_agent adds a second
    degrade layer for depth/kill-switch bounds.
    """
    if r is None or not r:
        return "leaf"
    r_norm = str(r).strip().lower()
    if r_norm in ("leaf", "orchestrator"):
        return r_norm
    logger.warning("Unknown delegate_task role=%r, coercing to 'leaf'", r)
    return "leaf"


def _get_max_concurrent_children() -> int:
    """Read delegation.max_concurrent_children from config, falling back to
    DELEGATION_MAX_CONCURRENT_CHILDREN env var, then the default (3).

    Users can raise this as high as they want; only the floor (1) is enforced.

    Uses the same ``_load_config()`` path that the rest of ``delegate_task``
    uses, keeping config priority consistent (config.yaml > env > default).
    """
    cfg = _load_config()
    val = cfg.get("max_concurrent_children")
    if val is not None:
        try:
            return max(1, int(val))
        except (TypeError, ValueError):
            logger.warning(
                "delegation.max_concurrent_children=%r is not a valid integer; "
                "using default %d",
                val,
                _DEFAULT_MAX_CONCURRENT_CHILDREN,
            )
            return _DEFAULT_MAX_CONCURRENT_CHILDREN
    env_val = os.getenv("DELEGATION_MAX_CONCURRENT_CHILDREN")
    if env_val:
        try:
            return max(1, int(env_val))
        except (TypeError, ValueError):
            return _DEFAULT_MAX_CONCURRENT_CHILDREN
    return _DEFAULT_MAX_CONCURRENT_CHILDREN


def _get_child_timeout() -> float:
    """Read delegation.child_timeout_seconds from config.

    Returns the number of seconds a single child agent is allowed to run
    before being considered stuck.  Default: 300 s (5 minutes).
    """
    cfg = _load_config()
    val = cfg.get("child_timeout_seconds")
    if val is not None:
        try:
            return max(30.0, float(val))
        except (TypeError, ValueError):
            logger.warning(
                "delegation.child_timeout_seconds=%r is not a valid number; "
                "using default %d",
                val,
                DEFAULT_CHILD_TIMEOUT,
            )
    env_val = os.getenv("DELEGATION_CHILD_TIMEOUT_SECONDS")
    if env_val:
        try:
            return max(30.0, float(env_val))
        except (TypeError, ValueError):
            pass
    return float(DEFAULT_CHILD_TIMEOUT)


def _get_max_spawn_depth() -> int:
    """Read delegation.max_spawn_depth from config, clamped to [1, 3].

    depth 0 = parent agent.  max_spawn_depth = N means agents at depths
    0..N-1 can spawn; depth N is the leaf floor.  Default 1 is flat:
    parent spawns children (depth 1), depth-1 children cannot spawn
    (blocked by this guard AND, for leaf children, by the delegation
    toolset strip in _strip_blocked_tools).

    Raise to 2 or 3 to unlock nested orchestration. role="orchestrator"
    removes the toolset strip for depth-1 children when
    max_spawn_depth >= 2, enabling them to spawn their own workers.
    """
    cfg = _load_config()
    val = cfg.get("max_spawn_depth")
    if val is None:
        return MAX_DEPTH
    try:
        ival = int(val)
    except (TypeError, ValueError):
        logger.warning(
            "delegation.max_spawn_depth=%r is not a valid integer; " "using default %d",
            val,
            MAX_DEPTH,
        )
        return MAX_DEPTH
    clamped = max(_MIN_SPAWN_DEPTH, min(_MAX_SPAWN_DEPTH_CAP, ival))
    if clamped != ival:
        logger.warning(
            "delegation.max_spawn_depth=%d out of range [%d, %d]; " "clamping to %d",
            ival,
            _MIN_SPAWN_DEPTH,
            _MAX_SPAWN_DEPTH_CAP,
            clamped,
        )
    return clamped


def _get_orchestrator_enabled() -> bool:
    """Global kill switch for the orchestrator role.

    When False, role="orchestrator" is silently forced to "leaf" in
    _build_child_agent and the delegation toolset is stripped as before.
    Lets an operator disable the feature without a code revert.
    """
    cfg = _load_config()
    val = cfg.get("orchestrator_enabled", True)
    if isinstance(val, bool):
        return val
    # Accept "true"/"false" strings from YAML that doesn't auto-coerce.
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes", "on")
    return True


DEFAULT_MAX_ITERATIONS = 50
DEFAULT_CHILD_TIMEOUT = 300  # seconds before a child agent is considered stuck
_HEARTBEAT_INTERVAL = 30  # seconds between parent activity heartbeats during delegation
_HEARTBEAT_STALE_CYCLES = (
    5  # mark child stale after this many heartbeats with no iteration progress
)
DEFAULT_TOOLSETS = ["terminal", "file", "web"]


# ---------------------------------------------------------------------------
# Delegation progress event types
# ---------------------------------------------------------------------------


class DelegateEvent(str, enum.Enum):
    """Formal event types emitted during delegation progress.

    _build_child_progress_callback normalises incoming legacy strings
    (``tool.started``, ``_thinking``, …) to these enum values via
    ``_LEGACY_EVENT_MAP``.  External consumers (gateway SSE, ACP adapter,
    CLI) still receive the legacy strings during the deprecation window.

    TASK_SPAWNED / TASK_COMPLETED / TASK_FAILED are reserved for
    future orchestrator lifecycle events and are not currently emitted.
    """

    TASK_SPAWNED = "delegate.task_spawned"
    TASK_PROGRESS = "delegate.task_progress"
    TASK_COMPLETED = "delegate.task_completed"
    TASK_FAILED = "delegate.task_failed"
    TASK_THINKING = "delegate.task_thinking"
    TASK_TOOL_STARTED = "delegate.tool_started"
    TASK_TOOL_COMPLETED = "delegate.tool_completed"


# Legacy event strings → DelegateEvent mapping.
# Incoming child-agent events use the old names; the callback normalises them.
_LEGACY_EVENT_MAP: Dict[str, DelegateEvent] = {
    "_thinking": DelegateEvent.TASK_THINKING,
    "reasoning.available": DelegateEvent.TASK_THINKING,
    "tool.started": DelegateEvent.TASK_TOOL_STARTED,
    "tool.completed": DelegateEvent.TASK_TOOL_COMPLETED,
    "subagent_progress": DelegateEvent.TASK_PROGRESS,
}


def check_delegate_requirements() -> bool:
    """Delegation has no external requirements -- always available."""
    return True


def _build_child_system_prompt(
    goal: str,
    context: Optional[str] = None,
    *,
    workspace_path: Optional[str] = None,
    role: str = "leaf",
    max_spawn_depth: int = 2,
    child_depth: int = 1,
) -> str:
    """Build a focused system prompt for a child agent.

    When role='orchestrator', appends a delegation-capability block
    modeled on OpenClaw's buildSubagentSystemPrompt (canSpawn branch at
    inspiration/openclaw/src/agents/subagent-system-prompt.ts:63-95).
    The depth note is literal truth (grounded in the passed config) so
    the LLM doesn't confabulate nesting capabilities that don't exist.
    """
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
    if role == "orchestrator":
        child_note = (
            "Your own children MUST be leaves (cannot delegate further) "
            "because they would be at the depth floor — you cannot pass "
            "role='orchestrator' to your own delegate_task calls."
            if child_depth + 1 >= max_spawn_depth
            else "Your own children can themselves be orchestrators or leaves, "
            "depending on the `role` you pass to delegate_task. Default is "
            "'leaf'; pass role='orchestrator' explicitly when a child "
            "needs to further decompose its work."
        )
        parts.append(
            "\n## Subagent Spawning (Orchestrator Role)\n"
            "You have access to the `delegate_task` tool and CAN spawn "
            "your own subagents to parallelize independent work.\n\n"
            "WHEN to delegate:\n"
            "- The goal decomposes into 2+ independent subtasks that can "
            "run in parallel (e.g. research A and B simultaneously).\n"
            "- A subtask is reasoning-heavy and would flood your context "
            "with intermediate data.\n\n"
            "WHEN NOT to delegate:\n"
            "- Single-step mechanical work — do it directly.\n"
            "- Trivial tasks you can execute in one or two tool calls.\n"
            "- Re-delegating your entire assigned goal to one worker "
            "(that's just pass-through with no value added).\n\n"
            "Coordinate your workers' results and synthesize them before "
            "reporting back to your parent. You are responsible for the "
            "final summary, not your workers.\n\n"
            f"NOTE: You are at depth {child_depth}. The delegation tree "
            f"is capped at max_spawn_depth={max_spawn_depth}. {child_note}"
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
    task_index: int,
    goal: str,
    parent_agent,
    task_count: int = 1,
    *,
    subagent_id: Optional[str] = None,
    parent_id: Optional[str] = None,
    depth: Optional[int] = None,
    model: Optional[str] = None,
    toolsets: Optional[List[str]] = None,
) -> Optional[callable]:
    """Build a callback that relays child agent tool calls to the parent display.

    Two display paths:
      CLI:     prints tree-view lines above the parent's delegation spinner
      Gateway: batches tool names and relays to parent's progress callback

    The identity kwargs (``subagent_id``, ``parent_id``, ``depth``, ``model``,
    ``toolsets``) are threaded into every relayed event so the TUI can
    reconstruct the live spawn tree and route per-branch controls (kill,
    pause) back by ``subagent_id``.  All are optional for backward compat —
    older callers that ignore them still produce a flat list on the TUI.

    Returns None if no display mechanism is available, in which case the
    child agent runs with no progress callback (identical to current behavior).
    """
    spinner = getattr(parent_agent, "_delegate_spinner", None)
    parent_cb = getattr(parent_agent, "tool_progress_callback", None)

    if not spinner and not parent_cb:
        return None  # No display → no callback → zero behavior change

    # Show 1-indexed prefix only in batch mode (multiple tasks)
    prefix = f"[{task_index + 1}] " if task_count > 1 else ""
    goal_label = (goal or "").strip()

    # Gateway: batch tool names, flush periodically
    _BATCH_SIZE = 5
    _batch: List[str] = []
    _tool_count = [0]  # per-subagent running counter (list for closure mutation)

    def _identity_kwargs() -> Dict[str, Any]:
        kw: Dict[str, Any] = {
            "task_index": task_index,
            "task_count": task_count,
            "goal": goal_label,
        }
        if subagent_id is not None:
            kw["subagent_id"] = subagent_id
        if parent_id is not None:
            kw["parent_id"] = parent_id
        if depth is not None:
            kw["depth"] = depth
        if model is not None:
            kw["model"] = model
        if toolsets is not None:
            kw["toolsets"] = list(toolsets)
        kw["tool_count"] = _tool_count[0]
        return kw

    def _relay(
        event_type: str, tool_name: str = None, preview: str = None, args=None, **kwargs
    ):
        if not parent_cb:
            return
        payload = _identity_kwargs()
        payload.update(kwargs)  # caller overrides (e.g. status, duration_seconds)
        try:
            parent_cb(event_type, tool_name, preview, args, **payload)
        except Exception as e:
            logger.debug("Parent callback failed: %s", e)

    def _callback(
        event_type, tool_name: str = None, preview: str = None, args=None, **kwargs
    ):
        # Lifecycle events emitted by the orchestrator itself — handled
        # before enum normalisation since they are not part of DelegateEvent.
        if event_type == "subagent.start":
            if spinner and goal_label:
                short = (
                    (goal_label[:55] + "...") if len(goal_label) > 55 else goal_label
                )
                try:
                    spinner.print_above(f" {prefix}├─ 🔀 {short}")
                except Exception as e:
                    logger.debug("Spinner print_above failed: %s", e)
            _relay("subagent.start", preview=preview or goal_label or "", **kwargs)
            return

        if event_type == "subagent.complete":
            _relay("subagent.complete", preview=preview, **kwargs)
            return

        # Normalise legacy strings, new-style "delegate.*" strings, and
        # DelegateEvent enum values all to a single DelegateEvent.  The
        # original implementation only accepted the five legacy strings;
        # enum-typed callers were silently dropped.
        if isinstance(event_type, DelegateEvent):
            event = event_type
        else:
            event = _LEGACY_EVENT_MAP.get(event_type)
            if event is None:
                try:
                    event = DelegateEvent(event_type)
                except (ValueError, TypeError):
                    return  # Unknown event — ignore

        if event == DelegateEvent.TASK_THINKING:
            text = preview or tool_name or ""
            if spinner:
                short = (text[:55] + "...") if len(text) > 55 else text
                try:
                    spinner.print_above(f' {prefix}├─ 💭 "{short}"')
                except Exception as e:
                    logger.debug("Spinner print_above failed: %s", e)
            _relay("subagent.thinking", preview=text)
            return

        if event == DelegateEvent.TASK_TOOL_COMPLETED:
            return

        if event == DelegateEvent.TASK_PROGRESS:
            # Pre-batched progress summary relayed from a nested
            # orchestrator's grandchild (upstream emits as
            # parent_cb("subagent_progress", summary_string) where the
            # summary lands in the tool_name positional slot).  Treat as
            # a pass-through: render distinctly (not via the tool-start
            # emoji lookup, which would mistake the summary string for a
            # tool name) and relay upward without re-batching.
            summary_text = tool_name or preview or ""
            if spinner and summary_text:
                try:
                    spinner.print_above(f" {prefix}├─ 🔀 {summary_text}")
                except Exception as e:
                    logger.debug("Spinner print_above failed: %s", e)
            if parent_cb:
                try:
                    parent_cb("subagent_progress", f"{prefix}{summary_text}")
                except Exception as e:
                    logger.debug("Parent callback relay failed: %s", e)
            return

        # TASK_TOOL_STARTED — display and batch for parent relay
        _tool_count[0] += 1
        if subagent_id is not None:
            with _active_subagents_lock:
                rec = _active_subagents.get(subagent_id)
                if rec is not None:
                    rec["tool_count"] = _tool_count[0]
                    rec["last_tool"] = tool_name or ""
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
            _relay("subagent.tool", tool_name, preview, args)
            _batch.append(tool_name or "")
            if len(_batch) >= _BATCH_SIZE:
                summary = ", ".join(_batch)
                _relay("subagent.progress", preview=f"🔀 {prefix}{summary}")
                _batch.clear()

    def _flush():
        """Flush remaining batched tool names to gateway on completion."""
        if parent_cb and _batch:
            summary = ", ".join(_batch)
            _relay("subagent.progress", preview=f"🔀 {prefix}{summary}")
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
    task_count: int,
    parent_agent,
    # Credential overrides from delegation config (provider:model resolution)
    override_provider: Optional[str] = None,
    override_base_url: Optional[str] = None,
    override_api_key: Optional[str] = None,
    override_api_mode: Optional[str] = None,
    # ACP transport overrides — lets a non-ACP parent spawn ACP child agents
    override_acp_command: Optional[str] = None,
    override_acp_args: Optional[List[str]] = None,
    # Per-call role controlling whether the child can further delegate.
    # 'leaf' (default) cannot; 'orchestrator' retains the delegation
    # toolset subject to depth/kill-switch bounds applied below.
    role: str = "leaf",
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
    import uuid as _uuid

    # ── Role resolution ─────────────────────────────────────────────────
    # Honor the caller's role only when BOTH the kill switch and the
    # child's depth allow it.  This is the single point where role
    # degrades to 'leaf' — keeps the rule predictable.  Callers pass
    # the normalised role (_normalize_role ran in delegate_task) so
    # we only deal with 'leaf' or 'orchestrator' here.
    child_depth = getattr(parent_agent, "_delegate_depth", 0) + 1
    max_spawn = _get_max_spawn_depth()
    orchestrator_ok = _get_orchestrator_enabled() and child_depth < max_spawn
    effective_role = role if (role == "orchestrator" and orchestrator_ok) else "leaf"

    # ── Subagent identity (stable across events, 0-indexed for TUI) ─────
    # subagent_id is generated here so the progress callback, the
    # spawn_requested event, and the _active_subagents registry all share
    # one key.  parent_id is non-None when THIS parent is itself a subagent
    # (nested orchestrator -> worker chain).
    subagent_id = f"sa-{task_index}-{_uuid.uuid4().hex[:8]}"
    parent_subagent_id = getattr(parent_agent, "_subagent_id", None)
    tui_depth = max(0, child_depth - 1)  # 0 = first-level child for the UI

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

    # Orchestrators retain the 'delegation' toolset that _strip_blocked_tools
    # removed.  The re-add is unconditional on parent-toolset membership because
    # orchestrator capability is granted by role, not inherited — see the
    # test_intersection_preserves_delegation_bound test for the design rationale.
    if effective_role == "orchestrator" and "delegation" not in child_toolsets:
        child_toolsets.append("delegation")

    workspace_hint = _resolve_workspace_hint(parent_agent)
    child_prompt = _build_child_system_prompt(
        goal,
        context,
        workspace_path=workspace_hint,
        role=effective_role,
        max_spawn_depth=max_spawn,
        child_depth=child_depth,
    )
    # Extract parent's API key so subagents inherit auth (e.g. Nous Portal).
    parent_api_key = getattr(parent_agent, "api_key", None)
    if (not parent_api_key) and hasattr(parent_agent, "_client_kwargs"):
        parent_api_key = parent_agent._client_kwargs.get("api_key")

    # Resolve the child's effective model early so it can ride on every event.
    effective_model_for_cb = model or getattr(parent_agent, "model", None)

    # Build progress callback to relay tool calls to parent display.
    # Identity kwargs thread the subagent_id through every emitted event so the
    # TUI can reconstruct the spawn tree and route per-branch controls.
    child_progress_cb = _build_child_progress_callback(
        task_index,
        goal,
        parent_agent,
        task_count,
        subagent_id=subagent_id,
        parent_id=parent_subagent_id,
        depth=tui_depth,
        model=effective_model_for_cb,
        toolsets=child_toolsets,
    )

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

    # Resolve reasoning config: delegation override > parent inherit
    parent_reasoning = getattr(parent_agent, "reasoning_config", None)
    child_reasoning = parent_reasoning
    try:
        delegation_cfg = _load_config()
        delegation_effort = str(delegation_cfg.get("reasoning_effort") or "").strip()
        if delegation_effort:
            from hermes_constants import parse_reasoning_effort

            parsed = parse_reasoning_effort(delegation_effort)
            if parsed is not None:
                child_reasoning = parsed
            else:
                logger.warning(
                    "Unknown delegation.reasoning_effort '%s', inheriting parent level",
                    delegation_effort,
                )
    except Exception as exc:
        logger.debug("Could not load delegation reasoning_effort: %s", exc)

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
        reasoning_config=child_reasoning,
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
    child._delegate_depth = child_depth
    # Stash the post-degrade role for introspection (leaf if the
    # kill switch or depth bounded the caller's requested role).
    child._delegate_role = effective_role
    # Stash subagent identity for nested-delegation event propagation and
    # for _run_single_child / interrupt_subagent to look up by id.
    child._subagent_id = subagent_id
    child._parent_subagent_id = parent_subagent_id
    child._subagent_goal = goal

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

    # Announce the spawn immediately — the child may sit in a queue
    # for seconds if max_concurrent_children is saturated, so the TUI
    # wants a node in the tree before run starts.
    if child_progress_cb:
        try:
            child_progress_cb("subagent.spawn_requested", preview=goal)
        except Exception as exc:
            logger.debug("spawn_requested relay failed: %s", exc)

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

    # Heartbeat: periodically propagate child activity to the parent so the
    # gateway inactivity timeout doesn't fire while the subagent is working.
    # Without this, the parent's _last_activity_ts freezes when delegate_task
    # starts and the gateway eventually kills the agent for "no activity".
    _heartbeat_stop = threading.Event()
    _last_seen_iter = [0]  # mutable container for heartbeat stale detection
    _stale_count = [0]

    def _heartbeat_loop():
        while not _heartbeat_stop.wait(_HEARTBEAT_INTERVAL):
            if parent_agent is None:
                continue
            touch = getattr(parent_agent, "_touch_activity", None)
            if not touch:
                continue
            # Pull detail from the child's own activity tracker
            desc = f"delegate_task: subagent {task_index} working"
            try:
                child_summary = child.get_activity_summary()
                child_tool = child_summary.get("current_tool")
                child_iter = child_summary.get("api_call_count", 0)
                child_max = child_summary.get("max_iterations", 0)

                # Stale detection: if iteration count hasn't advanced,
                # increment stale counter.  After N cycles with no
                # progress, stop masking the hang so the gateway
                # inactivity timeout can fire as a last resort.
                if child_iter <= _last_seen_iter[0]:
                    _stale_count[0] += 1
                else:
                    _last_seen_iter[0] = child_iter
                    _stale_count[0] = 0

                if _stale_count[0] >= _HEARTBEAT_STALE_CYCLES:
                    logger.warning(
                        "Subagent %d appears stale (no iteration progress "
                        "for %d heartbeat cycles) — stopping heartbeat",
                        task_index,
                        _stale_count[0],
                    )
                    break  # stop touching parent, let gateway timeout fire

                if child_tool:
                    desc = (
                        f"delegate_task: subagent running {child_tool} "
                        f"(iteration {child_iter}/{child_max})"
                    )
                else:
                    child_desc = child_summary.get("last_activity_desc", "")
                    if child_desc:
                        desc = (
                            f"delegate_task: subagent {child_desc} "
                            f"(iteration {child_iter}/{child_max})"
                        )
            except Exception:
                pass
            try:
                touch(desc)
            except Exception:
                pass

    _heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
    _heartbeat_thread.start()

    # Register the live agent in the module-level registry so the TUI can
    # target it by subagent_id (kill, pause, status queries).  Unregistered
    # in the finally block, even when the child raises.  Test doubles that
    # hand us a MagicMock don't carry stable ids; skip registration then.
    _raw_sid = getattr(child, "_subagent_id", None)
    _subagent_id = _raw_sid if isinstance(_raw_sid, str) else None
    if _subagent_id:
        _raw_depth = getattr(child, "_delegate_depth", 1)
        _tui_depth = max(0, _raw_depth - 1) if isinstance(_raw_depth, int) else 0
        _parent_sid = getattr(child, "_parent_subagent_id", None)
        _register_subagent(
            {
                "subagent_id": _subagent_id,
                "parent_id": _parent_sid if isinstance(_parent_sid, str) else None,
                "depth": _tui_depth,
                "goal": goal,
                "model": (
                    getattr(child, "model", None)
                    if isinstance(getattr(child, "model", None), str)
                    else None
                ),
                "started_at": time.time(),
                "status": "running",
                "tool_count": 0,
                "agent": child,
            }
        )

    try:
        if child_progress_cb:
            try:
                child_progress_cb("subagent.start", preview=goal)
            except Exception as e:
                logger.debug("Progress callback start failed: %s", e)

        # File-state coordination: reuse the stable subagent_id as the child's
        # task_id so file_state writes, active-subagents registry, and TUI
        # events all share one key.  Falls back to a fresh uuid only if the
        # pre-built id is somehow missing.
        import uuid as _uuid

        child_task_id = _subagent_id or f"subagent-{task_index}-{_uuid.uuid4().hex[:8]}"
        parent_task_id = getattr(parent_agent, "_current_task_id", None)
        wall_start = time.time()
        parent_reads_snapshot = (
            list(file_state.known_reads(parent_task_id)) if parent_task_id else []
        )

        # Run child with a hard timeout to prevent indefinite blocking
        # when the child's API call or tool-level HTTP request hangs.
        child_timeout = _get_child_timeout()
        _timeout_executor = ThreadPoolExecutor(max_workers=1)
        _child_future = _timeout_executor.submit(
            child.run_conversation,
            user_message=goal,
            task_id=child_task_id,
        )
        try:
            result = _child_future.result(timeout=child_timeout)
        except Exception as _timeout_exc:
            # Signal the child to stop so its thread can exit cleanly.
            try:
                if hasattr(child, "interrupt"):
                    child.interrupt()
                elif hasattr(child, "_interrupt_requested"):
                    child._interrupt_requested = True
            except Exception:
                pass

            is_timeout = isinstance(_timeout_exc, (FuturesTimeoutError, TimeoutError))
            duration = round(time.monotonic() - child_start, 2)
            logger.warning(
                "Subagent %d %s after %.1fs",
                task_index,
                "timed out" if is_timeout else f"raised {type(_timeout_exc).__name__}",
                duration,
            )

            if child_progress_cb:
                try:
                    child_progress_cb(
                        "subagent.complete",
                        preview=(
                            f"Timed out after {duration}s"
                            if is_timeout
                            else str(_timeout_exc)
                        ),
                        status="timeout" if is_timeout else "error",
                        duration_seconds=duration,
                        summary="",
                    )
                except Exception:
                    pass

            return {
                "task_index": task_index,
                "status": "timeout" if is_timeout else "error",
                "summary": None,
                "error": (
                    (
                        f"Subagent timed out after {child_timeout}s with no response. "
                        "The child may be stuck on a slow API call or unresponsive network request."
                    )
                    if is_timeout
                    else str(_timeout_exc)
                ),
                "exit_reason": "timeout" if is_timeout else "error",
                "api_calls": 0,
                "duration_seconds": duration,
                "_child_role": getattr(child, "_delegate_role", None),
            }
        finally:
            # Shut down executor without waiting — if the child thread
            # is stuck on blocking I/O, wait=True would hang forever.
            _timeout_executor.shutdown(wait=False)

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
                "input": (
                    _input_tokens if isinstance(_input_tokens, (int, float)) else 0
                ),
                "output": (
                    _output_tokens if isinstance(_output_tokens, (int, float)) else 0
                ),
            },
            "tool_trace": tool_trace,
            # Captured before the finally block calls child.close() so the
            # parent thread can fire subagent_stop with the correct role.
            # Stripped before the dict is serialised back to the model.
            "_child_role": getattr(child, "_delegate_role", None),
        }
        if status == "failed":
            entry["error"] = result.get("error", "Subagent did not produce a response.")

        # Cross-agent file-state reminder.  If this subagent wrote any
        # files the parent had already read, surface it so the parent
        # knows to re-read before editing — the scenario that motivated
        # the registry.  We check writes by ANY non-parent task_id (not
        # just this child's), which also covers transitive writes from
        # nested orchestrator→worker chains.
        try:
            if parent_task_id and parent_reads_snapshot:
                sibling_writes = file_state.writes_since(
                    parent_task_id, wall_start, parent_reads_snapshot
                )
                if sibling_writes:
                    mod_paths = sorted(
                        {p for paths in sibling_writes.values() for p in paths}
                    )
                    if mod_paths:
                        reminder = (
                            "\n\n[NOTE: subagent modified files the parent "
                            "previously read — re-read before editing: "
                            + ", ".join(mod_paths[:8])
                            + (
                                f" (+{len(mod_paths) - 8} more)"
                                if len(mod_paths) > 8
                                else ""
                            )
                            + "]"
                        )
                        if entry.get("summary"):
                            entry["summary"] = entry["summary"] + reminder
                        else:
                            entry["stale_paths"] = mod_paths
        except Exception:
            logger.debug("file_state sibling-write check failed", exc_info=True)

        # Per-branch observability payload: tokens, cost, files touched, and
        # a tail of tool-call results.  Fed into the TUI's overlay detail
        # pane + accordion rollups (features 1, 2, 4).  All fields are
        # optional — missing data degrades gracefully on the client.
        _cost_usd = getattr(child, "session_estimated_cost_usd", None)
        _reasoning_tokens = getattr(child, "session_reasoning_tokens", 0)
        try:
            _files_read = list(file_state.known_reads(child_task_id))[:40]
        except Exception:
            _files_read = []
        try:
            _files_written_map = file_state.writes_since(
                "", wall_start, []
            )  # all writes since wall_start
        except Exception:
            _files_written_map = {}
        _files_written = sorted(
            {
                p
                for tid, paths in _files_written_map.items()
                if tid == child_task_id
                for p in paths
            }
        )[:40]

        _output_tail = _extract_output_tail(result, max_entries=8, max_chars=600)

        complete_kwargs: Dict[str, Any] = {
            "preview": summary[:160] if summary else entry.get("error", ""),
            "status": status,
            "duration_seconds": duration,
            "summary": summary[:500] if summary else entry.get("error", ""),
            "input_tokens": (
                int(_input_tokens) if isinstance(_input_tokens, (int, float)) else 0
            ),
            "output_tokens": (
                int(_output_tokens) if isinstance(_output_tokens, (int, float)) else 0
            ),
            "reasoning_tokens": (
                int(_reasoning_tokens)
                if isinstance(_reasoning_tokens, (int, float))
                else 0
            ),
            "api_calls": int(api_calls) if isinstance(api_calls, (int, float)) else 0,
            "files_read": _files_read,
            "files_written": _files_written,
            "output_tail": _output_tail,
        }
        if _cost_usd is not None:
            try:
                complete_kwargs["cost_usd"] = float(_cost_usd)
            except (TypeError, ValueError):
                pass

        if child_progress_cb:
            try:
                child_progress_cb("subagent.complete", **complete_kwargs)
            except Exception as e:
                logger.debug("Progress callback completion failed: %s", e)

        return entry

    except Exception as exc:
        duration = round(time.monotonic() - child_start, 2)
        logging.exception(f"[subagent-{task_index}] failed")
        if child_progress_cb:
            try:
                child_progress_cb(
                    "subagent.complete",
                    preview=str(exc),
                    status="failed",
                    duration_seconds=duration,
                    summary=str(exc),
                )
            except Exception as e:
                logger.debug("Progress callback failure relay failed: %s", e)
        return {
            "task_index": task_index,
            "status": "error",
            "summary": None,
            "error": str(exc),
            "api_calls": 0,
            "duration_seconds": duration,
            "_child_role": getattr(child, "_delegate_role", None),
        }

    finally:
        # Stop the heartbeat thread so it doesn't keep touching parent activity
        # after the child has finished (or failed).
        _heartbeat_stop.set()
        _heartbeat_thread.join(timeout=5)

        # Drop the TUI-facing registry entry.  Safe to call even if the
        # child was never registered (e.g. ID missing on test doubles).
        if _subagent_id:
            _unregister_subagent(_subagent_id)

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

        # Close tool resources (terminal sandboxes, browser daemons,
        # background processes, httpx clients) so subagent subprocesses
        # don't outlive the delegation.
        try:
            if hasattr(child, "close"):
                child.close()
        except Exception:
            logger.debug("Failed to close child agent after delegation")


def delegate_task(
    goal: Optional[str] = None,
    context: Optional[str] = None,
    toolsets: Optional[List[str]] = None,
    tasks: Optional[List[Dict[str, Any]]] = None,
    max_iterations: Optional[int] = None,
    acp_command: Optional[str] = None,
    acp_args: Optional[List[str]] = None,
    role: Optional[str] = None,
    parent_agent=None,
) -> str:
    """
    Spawn one or more child agents to handle delegated tasks.

    Supports two modes:
      - Single: provide goal (+ optional context, toolsets, role)
      - Batch:  provide tasks array [{goal, context, toolsets, role}, ...]

    The 'role' parameter controls whether a child can further delegate:
    'leaf' (default) cannot; 'orchestrator' retains the delegation
    toolset and can spawn its own workers, bounded by
    delegation.max_spawn_depth.  Per-task role beats the top-level one.

    Returns JSON with results array, one entry per task.
    """
    if parent_agent is None:
        return tool_error("delegate_task requires a parent agent context.")

    # Operator-controlled kill switch — lets the TUI freeze new fan-out
    # when a runaway tree is detected, without interrupting already-running
    # children.  Cleared via the matching `delegation.pause` RPC.
    if is_spawn_paused():
        return json.dumps(
            {
                "error": (
                    "Delegation spawning is paused. Clear the pause via the TUI "
                    "(`p` in /agents) or the `delegation.pause` RPC before retrying."
                )
            }
        )

    # Normalise the top-level role once; per-task overrides re-normalise.
    top_role = _normalize_role(role)

    # Depth limit — configurable via delegation.max_spawn_depth,
    # default 2 for parity with the original MAX_DEPTH constant.
    depth = getattr(parent_agent, "_delegate_depth", 0)
    max_spawn = _get_max_spawn_depth()
    if depth >= max_spawn:
        return json.dumps(
            {
                "error": (
                    f"Delegation depth limit reached (depth={depth}, "
                    f"max_spawn_depth={max_spawn}). Raise "
                    f"delegation.max_spawn_depth in config.yaml if deeper "
                    f"nesting is required (cap: {_MAX_SPAWN_DEPTH_CAP})."
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
        task_list = [
            {"goal": goal, "context": context, "toolsets": toolsets, "role": top_role}
        ]
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
            task_acp_args = t.get("acp_args") if "acp_args" in t else None
            # Per-task role beats top-level; normalise again so unknown
            # per-task values warn and degrade to leaf uniformly.
            effective_role = _normalize_role(t.get("role") or top_role)
            child = _build_child_agent(
                task_index=i,
                goal=t["goal"],
                context=t.get("context"),
                toolsets=t.get("toolsets") or toolsets,
                model=creds["model"],
                max_iterations=effective_max_iter,
                task_count=n_tasks,
                parent_agent=parent_agent,
                override_provider=creds["provider"],
                override_base_url=creds["base_url"],
                override_api_key=creds["api_key"],
                override_api_mode=creds["api_mode"],
                override_acp_command=t.get("acp_command")
                or acp_command
                or creds.get("command"),
                override_acp_args=(
                    task_acp_args
                    if task_acp_args is not None
                    else (acp_args if acp_args is not None else creds.get("args"))
                ),
                role=effective_role,
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
            # Map task_index -> child agent, so fabricated entries for
            # still-pending futures can carry the correct _delegate_role.
            _child_by_index = {i: child for (i, _, child) in children}

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
                                    "_child_role": getattr(
                                        _child_by_index.get(idx), "_delegate_role", None
                                    ),
                                }
                        else:
                            entry = {
                                "task_index": idx,
                                "status": "interrupted",
                                "summary": None,
                                "error": "Parent agent interrupted — child did not finish in time",
                                "api_calls": 0,
                                "duration_seconds": 0,
                                "_child_role": getattr(
                                    _child_by_index.get(idx), "_delegate_role", None
                                ),
                            }
                        results.append(entry)
                        completed_count += 1
                    break

                from concurrent.futures import wait as _cf_wait, FIRST_COMPLETED

                done, pending = _cf_wait(
                    pending, timeout=0.5, return_when=FIRST_COMPLETED
                )
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
                            "_child_role": getattr(
                                _child_by_index.get(idx), "_delegate_role", None
                            ),
                        }
                    results.append(entry)
                    completed_count += 1

                    # Print per-task completion line above the spinner
                    idx = entry["task_index"]
                    label = (
                        task_labels[idx] if idx < len(task_labels) else f"Task {idx}"
                    )
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
                    child_session_id=(
                        getattr(children[entry["task_index"]][2], "session_id", "")
                        if entry["task_index"] < len(children)
                        else ""
                    ),
                )
            except Exception:
                pass

    # Fire subagent_stop hooks once per child, serialised on the parent thread.
    # This keeps Python-plugin and shell-hook callbacks off of the worker threads
    # that ran the children, so hook authors don't need to reason about
    # concurrent invocation.  Role was captured into the entry dict in
    # _run_single_child (or the fabricated-entry branches above) before the
    # child was closed.
    _parent_session_id = getattr(parent_agent, "session_id", None)
    try:
        from hermes_cli.plugins import invoke_hook as _invoke_hook
    except Exception:
        _invoke_hook = None
    for entry in results:
        child_role = entry.pop("_child_role", None)
        if _invoke_hook is None:
            continue
        try:
            _invoke_hook(
                "subagent_stop",
                parent_session_id=_parent_session_id,
                child_role=child_role,
                child_summary=entry.get("summary"),
                child_status=entry.get("status"),
                duration_ms=int((entry.get("duration_seconds") or 0) * 1000),
            )
        except Exception:
            logger.debug("subagent_stop hook invocation failed", exc_info=True)

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
        if (
            base_url_hostname(configured_base_url) == "chatgpt.com"
            and "/backend-api/codex" in base_lower
        ):
            provider = "openai-codex"
            api_mode = "codex_responses"
        elif base_url_hostname(configured_base_url) == "api.anthropic.com":
            provider = "anthropic"
            api_mode = "anthropic_messages"
        elif "api.kimi.com/coding" in base_lower:
            provider = "custom"
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
        "Spawn one or more subagents to work on tasks in isolated contexts. "
        "Each subagent gets its own conversation, terminal session, and toolset. "
        "Only the final summary is returned -- intermediate tool results "
        "never enter your context window.\n\n"
        "TWO MODES (one of 'goal' or 'tasks' is required):\n"
        "1. Single task: provide 'goal' (+ optional context, toolsets)\n"
        "2. Batch (parallel): provide 'tasks' array with up to delegation.max_concurrent_children items (default 3). "
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
        "- Leaf subagents (role='leaf', the default) CANNOT call: "
        "delegate_task, clarify, memory, send_message, execute_code.\n"
        "- Orchestrator subagents (role='orchestrator') retain "
        "delegate_task so they can spawn their own workers, but still "
        "cannot use clarify, memory, send_message, or execute_code. "
        "Orchestrators are bounded by delegation.max_spawn_depth "
        "(default 2) and can be disabled globally via "
        "delegation.orchestrator_enabled=false.\n"
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
                    f"Available toolsets: {_TOOLSET_LIST_STR}. "
                    "Common patterns: ['terminal', 'file'] for code work, "
                    "['web'] for research, ['browser'] for web interaction, "
                    "['terminal', 'file', 'web'] for full-stack tasks."
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
                            "description": f"Toolsets for this specific task. Available: {_TOOLSET_LIST_STR}. Use 'web' for network access, 'terminal' for shell, 'browser' for web interaction.",
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
                        "role": {
                            "type": "string",
                            "enum": ["leaf", "orchestrator"],
                            "description": "Per-task role override. See top-level 'role' for semantics.",
                        },
                    },
                    "required": ["goal"],
                },
                # No maxItems — the runtime limit is configurable via
                # delegation.max_concurrent_children (default 3) and
                # enforced with a clear error in delegate_task().
                "description": (
                    "Batch mode: tasks to run in parallel (limit configurable via delegation.max_concurrent_children, default 3). Each gets "
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
            "role": {
                "type": "string",
                "enum": ["leaf", "orchestrator"],
                "description": (
                    "Role of the child agent. 'leaf' (default) = focused "
                    "worker, cannot delegate further. 'orchestrator' = can "
                    "use delegate_task to spawn its own workers. Requires "
                    "delegation.max_spawn_depth >= 2 in config; ignored "
                    "(treated as 'leaf') when the child would exceed "
                    "max_spawn_depth or when "
                    "delegation.orchestrator_enabled=false."
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
        role=args.get("role"),
        parent_agent=kw.get("parent_agent"),
    ),
    check_fn=check_delegate_requirements,
    emoji="🔀",
)
