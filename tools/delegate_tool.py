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
)
from typing import Any, Dict, List, Optional

from toolsets import TOOLSETS

# Sentinel value used by the runtime provider system for providers that are
# not natively known (named custom providers, third-party aggregators, etc.).
# Must match hermes_cli.runtime_provider.RUNTIME_PROVIDER_TYPE_CUSTOM.
_RUNTIME_PROVIDER_CUSTOM = "custom"
from tools import file_state
from tools.terminal_tool import set_approval_callback as _set_subagent_approval_cb
from utils import base_url_hostname, is_truthy_value


# Tools that children must never have access to
DELEGATE_BLOCKED_TOOLS = frozenset(
    [
        "delegate_task",  # no recursive delegation
        "clarify",  # no user interaction
        "memory",  # no writes to shared MEMORY.md
        "send_message",  # no cross-platform side effects
        "execute_code",  # children should reason step-by-step, not write scripts
        "cronjob",  # no scheduling more work in the parent's name
    ]
)


# ---------------------------------------------------------------------------
# Subagent approval callbacks
# ---------------------------------------------------------------------------
# Subagents run inside a ThreadPoolExecutor worker. The CLI's interactive
# approval callback is stored in tools/terminal_tool.py's threading.local(),
# so worker threads do NOT inherit it. Without a callback,
# prompt_dangerous_approval() falls back to input() from the worker thread,
# which deadlocks against the parent's prompt_toolkit TUI that owns stdin.
#
# Fix: install a non-interactive callback into every subagent worker thread
# via ThreadPoolExecutor(initializer=_set_subagent_approval_cb, initargs=(cb,)).
# The callback is chosen by the `delegation.subagent_auto_approve` config:
#   false (default) → _subagent_auto_deny (safe; matches leaf tool blocklist)
#   true            → _subagent_auto_approve (opt-in YOLO for cron/batch)
# Both emit a logger.warning for audit; gateway sessions are unaffected
# because they resolve approvals via tools/approval.py's per-session queue,
# not through these TLS callbacks.
def _subagent_auto_deny(command: str, description: str, **kwargs) -> str:
    """Auto-deny dangerous commands in subagent threads (safe default).

    Returns 'deny' so the subagent sees a refusal it can recover from, and
    never calls input() (which would deadlock the parent TUI).
    """
    logger.warning(
        "Subagent auto-denied dangerous command: %s (%s). "
        "Set delegation.subagent_auto_approve: true to allow.",
        command, description,
    )
    return "deny"


def _subagent_auto_approve(command: str, description: str, **kwargs) -> str:
    """Auto-approve dangerous commands in subagent threads (opt-in YOLO).

    Only installed when delegation.subagent_auto_approve=true. Returns 'once'
    so the subagent proceeds without blocking the parent UI.
    """
    logger.warning(
        "Subagent auto-approved dangerous command: %s (%s)",
        command, description,
    )
    return "once"


def _get_subagent_approval_callback():
    """Return the callback to install into subagent worker threads.

    Config key: delegation.subagent_auto_approve (bool, default False).
    Reads via the same _load_config() path as the rest of delegate_task so
    priority is config.yaml > (no env override for this knob) > default.
    """
    cfg = _load_config()
    val = cfg.get("subagent_auto_approve", False)
    if is_truthy_value(val):
        return _subagent_auto_approve
    return _subagent_auto_deny

# NOTE: nested delegation is granted by role='orchestrator' (which re-adds the
# "delegation" toolset in _build_child_agent), NOT by the model naming toolsets
# — the model has no toolsets argument. Subagents inherit the parent's toolsets.

_DEFAULT_MAX_CONCURRENT_CHILDREN = 3
# One-shot guard: the high-concurrency cost advisory is emitted at most once
# per process. _get_max_concurrent_children() runs on every get_definitions()
# schema rebuild (via _build_top_level_description / _build_tasks_param_description),
# so without this flag a config of max_concurrent_children>10 spams the log on
# every turn / agent spawn even when delegate_task is never called.
_HIGH_CONCURRENCY_WARNED = False
MAX_DEPTH = 1  # flat by default: parent (0) -> child (1); grandchild rejected unless max_spawn_depth raised.
# Configurable depth cap consulted by _get_max_spawn_depth; MAX_DEPTH
# stays as the default fallback and is still the symbol tests import.
_MIN_SPAWN_DEPTH = 1
# No upper ceiling on spawn depth — like max_concurrent_children, depth has a
# floor of 1 and no ceiling. Deeper trees multiply API cost, so the default
# stays flat (MAX_DEPTH = 1); raising the config knob is an explicit opt-in.


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
        # Flatten content-block lists/dicts to text so the overlay shows real
        # output (not a "[{'type': 'text'...}]" blob) and error detection can
        # see markers buried inside content blocks. Crude str() here would
        # mislabel a block-wrapped "Error: ..." result as is_error=False.
        content = _stringify_tool_content(msg.get("content") or "")
        is_error = _looks_like_error_output(content)
        tool_name = pending_call_by_id.get(msg.get("tool_call_id") or "", "tool")
        # Preserve line structure so the overlay's wrapped scroll region can
        # show real output rather than a whitespace-collapsed blob. We still
        # cap the payload size to keep events bounded.
        preview = content[:max_chars]
        tail.append({"tool": tool_name, "preview": preview, "is_error": is_error})

    tail.reverse()  # restore chronological order for display
    return tail


def _stringify_tool_content(content: Any) -> str:
    """Return a stable text representation for tool-result content.

    Most providers store tool results as strings, but some OpenAI-compatible
    paths can return content-block lists. Delegate observability must never
    crash while summarising a child run just because the transport used blocks.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                else:
                    parts.append(json.dumps(item, ensure_ascii=False, default=str))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, default=str)
    return str(content)


def _looks_like_error_output(content: Any) -> bool:
    """Conservative stderr/error detector for tool-result previews.

    The old heuristic flagged any preview containing the substring "error",
    which painted perfectly normal terminal/json output red.  We now only
    mark output as an error when there is stronger evidence:
      - structured JSON with an ``error`` key
      - structured JSON with ``status`` of error/failed
      - first line starts with a classic error marker
    """
    content = _stringify_tool_content(content)
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
    if r_norm in {"leaf", "orchestrator"}:
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
            result = max(1, int(val))
            if result > 10:
                global _HIGH_CONCURRENCY_WARNED
                if not _HIGH_CONCURRENCY_WARNED:
                    _HIGH_CONCURRENCY_WARNED = True
                    logger.warning(
                        "delegation.max_concurrent_children=%d: each child consumes API tokens "
                        "independently. High values multiply cost linearly.",
                        result,
                    )
            return result
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


_LEGACY_MAX_ASYNC_WARNED = False


def _get_max_async_children() -> int:
    """Concurrency cap for background (``background=true``) delegations.

    DEPRECATED KNOB: ``delegation.max_async_children`` has been unified into
    ``delegation.max_concurrent_children`` — one cap governs both a single
    synchronous batch's parallelism and how many background delegation units
    may run at once. When at capacity, a new async dispatch is REJECTED (not
    queued) so a runaway model can't pile up unbounded background work; the
    caller falls back to running the work synchronously.

    A leftover ``max_async_children`` in config.yaml is ignored (the config
    migration removes it, folding a raised value into
    ``max_concurrent_children``); we log a one-time deprecation warning if
    one is still present.
    """
    global _LEGACY_MAX_ASYNC_WARNED
    cfg = _load_config()
    if cfg.get("max_async_children") is not None and not _LEGACY_MAX_ASYNC_WARNED:
        _LEGACY_MAX_ASYNC_WARNED = True
        logger.warning(
            "delegation.max_async_children is deprecated and ignored; "
            "delegation.max_concurrent_children now caps background "
            "delegations too. Remove the stale key from config.yaml."
        )
    return _get_max_concurrent_children()


def _get_child_timeout() -> Optional[float]:
    """Read delegation.child_timeout_seconds from config.

    Returns the number of seconds a single child agent is allowed to run
    before being cut off, or ``None`` when no wall-clock cap applies.

    Default: ``None`` (no timeout). Subagents doing legitimate heavy work
    (deep code review, large research fan-outs, slow reasoning models) were
    routinely killed mid-task by the old blanket cap even though they were
    making steady progress. Failures should come from what the child is
    actually doing — API errors, tool errors, iteration budget — not from a
    generic delegation-level stopwatch. Stuck-child protection is handled
    separately by the heartbeat staleness monitor, which stops refreshing
    parent activity so the gateway inactivity timeout can fire.

    Set ``delegation.child_timeout_seconds`` to a positive number to opt back
    in to a hard cap (floor 30 s); ``0`` or a negative value means disabled.
    """
    cfg = _load_config()
    val = cfg.get("child_timeout_seconds")
    if val is not None:
        try:
            parsed = float(val)
        except (TypeError, ValueError):
            logger.warning(
                "delegation.child_timeout_seconds=%r is not a valid number; "
                "using default (no timeout)",
                val,
            )
        else:
            return None if parsed <= 0 else max(30.0, parsed)
    env_val = os.getenv("DELEGATION_CHILD_TIMEOUT_SECONDS")
    if env_val:
        try:
            parsed = float(env_val)
        except (TypeError, ValueError):
            pass
        else:
            return None if parsed <= 0 else max(30.0, parsed)
    return DEFAULT_CHILD_TIMEOUT


def _get_max_spawn_depth() -> int:
    """Read delegation.max_spawn_depth from config, floored at 1 (no ceiling).

    depth 0 = parent agent.  max_spawn_depth = N means agents at depths
    0..N-1 can spawn; depth N is the leaf floor.  Default 1 is flat:
    parent spawns children (depth 1), depth-1 children cannot spawn
    (blocked by this guard AND, for leaf children, by the delegation
    toolset strip in _strip_blocked_tools).

    Raise to 2+ to unlock nested orchestration. role="orchestrator"
    removes the toolset strip for spawning children when
    max_spawn_depth >= 2, enabling them to spawn their own workers.
    Like max_concurrent_children, there is no upper ceiling — but each
    extra level multiplies API cost, so raise it deliberately.
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
    floored = max(_MIN_SPAWN_DEPTH, ival)
    if floored != ival:
        logger.warning(
            "delegation.max_spawn_depth=%d below floor %d; using %d",
            ival,
            _MIN_SPAWN_DEPTH,
            floored,
        )
    return floored


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
        return val.strip().lower() in {"true", "1", "yes", "on"}
    return True


def _get_inherit_mcp_toolsets() -> bool:
    """Whether narrowed child toolsets should keep the parent's MCP toolsets."""
    cfg = _load_config()
    return is_truthy_value(cfg.get("inherit_mcp_toolsets"), default=True)


def _is_mcp_toolset_name(name: str) -> bool:
    """Return True for canonical MCP toolsets and their registered aliases."""
    if not name:
        return False
    if str(name).startswith("mcp-"):
        return True
    try:
        from tools.registry import registry

        target = registry.get_toolset_alias_target(str(name))
    except Exception:
        target = None
    return bool(target and str(target).startswith("mcp-"))


def _expand_parent_toolsets(parent_toolsets: set) -> set:
    """Expand composite toolsets so individual toolset names are recognized.

    When a parent uses a composite toolset like ``hermes-cli`` (which bundles
    all core tools), the child may request individual toolsets such as ``web``
    or ``terminal``.  A simple name-based intersection would reject them
    because ``"web" != "hermes-cli"``.

    This helper collects the tool names from each parent toolset, then adds
    the names of any individual toolsets whose tools are a *subset* of the
    parent's available tools.  The original parent toolset names are preserved.
    """
    parent_tool_names: set = set()
    for ts_name in parent_toolsets:
        ts_def = TOOLSETS.get(ts_name)
        if ts_def:
            parent_tool_names.update(ts_def.get("tools", []))

    if not parent_tool_names:
        return set(parent_toolsets)

    expanded = set(parent_toolsets)
    for ts_name, ts_def in TOOLSETS.items():
        if ts_name in expanded:
            continue
        ts_tools = ts_def.get("tools", [])
        if ts_tools and set(ts_tools).issubset(parent_tool_names):
            expanded.add(ts_name)
    return expanded


def _preserve_parent_mcp_toolsets(
    child_toolsets: List[str], parent_toolsets: set[str]
) -> List[str]:
    """Append any parent MCP toolsets that are missing from a narrowed child."""
    preserved = list(child_toolsets)
    for toolset_name in sorted(parent_toolsets):
        if _is_mcp_toolset_name(toolset_name) and toolset_name not in preserved:
            preserved.append(toolset_name)
    return preserved


DEFAULT_MAX_ITERATIONS = 50
# Hard per-summary character ceiling layered on top of the dynamic
# headroom budget (see _apply_summary_budget). Belt-and-suspenders for
# models that ignore the "be concise" instruction. 0 disables the ceiling.
DEFAULT_MAX_SUMMARY_CHARS = 24000
# Fraction of the parent's *remaining* context headroom that the whole batch
# of subagent summaries is allowed to consume. The per-summary budget is this
# slice divided across the batch, so N children can't collectively blow the
# parent's window (the compression/429 death-spiral in issue/PR #9126).
_SUMMARY_HEADROOM_FRACTION = 0.5
# Floor so a single summary always gets a usable slice even when the parent is
# already nearly full — below this we'd be truncating to noise.
_MIN_SUMMARY_CHARS = 2000
# No default wall-clock cap on child agents: legitimate heavy subagent work
# (deep reviews, research fan-outs, slow reasoning models) was being killed
# mid-task. Errors should come from what the child actually does; stuck-child
# detection lives in the heartbeat staleness monitor below. Users can opt back
# in via delegation.child_timeout_seconds.
DEFAULT_CHILD_TIMEOUT: Optional[float] = None
_HEARTBEAT_INTERVAL = 30  # seconds between parent activity heartbeats during delegation
# Stale-heartbeat thresholds. A child with no API-call progress is either:
#   - idle between turns (no current_tool) — probably stuck on a slow API call
#   - inside a tool (current_tool set) — probably running a legitimately long
#     operation (terminal command, web fetch, large file read)
# The idle ceiling stays tight so genuinely stuck children don't mask the gateway
# timeout. The in-tool ceiling is much higher so legit long-running tools get
# time to finish; delegation.child_timeout_seconds (off by default) remains an
# optional hard cap for users who want one.
_HEARTBEAT_STALE_CYCLES_IDLE = 15  # 15 * 30s = 450s idle between turns → stale
_HEARTBEAT_STALE_CYCLES_IN_TOOL = 40  # 40 * 30s = 1200s stuck on same tool → stale
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
        "Keep your final summary tight: lead with outcomes, prefer bullet "
        "points over paragraphs, and don't replay your whole process. Your "
        "response is returned to the parent agent as a summary, and overlong "
        "summaries crowd out the parent's context window."
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
    """Remove toolsets that contain only blocked tools.

    The strip set is derived from DELEGATE_BLOCKED_TOOLS plus the explicit
    composite/scenario toolsets (delegation, code_execution) that have no
    one-to-one tool. This keeps the blocklist and the strip set in lockstep
    so new blocked tools can't silently leak through as toolset names.
    """
    # Composite toolsets that should never pass through to children, even
    # though their individual tools aren't all in DELEGATE_BLOCKED_TOOLS.
    _COMPOSITE_BLOCKED_TOOLSETS = frozenset({"delegation", "code_execution"})
    blocked_toolset_names = {
        name
        for name, defn in TOOLSETS.items()
        if name in _COMPOSITE_BLOCKED_TOOLSETS
        or all(t in DELEGATE_BLOCKED_TOOLS for t in defn.get("tools", []))
    }
    return [t for t in toolsets if t not in blocked_toolset_names]


def _emit_parent_console(parent_agent, line: str) -> None:
    """Emit a human-readable progress line to the parent's console.

    Routes through ``parent_agent._safe_print`` when available so headless
    stdio hosts (ACP, gateway API) can redirect non-protocol output to
    stderr via their configured ``_print_fn``. A bare ``print()`` would
    otherwise land on stdout and corrupt JSON-RPC framing.
    """
    printer = getattr(parent_agent, "_safe_print", None)
    if callable(printer):
        try:
            printer(line)
            return
        except Exception:
            pass
    print(line)


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
    session_ref: Optional[Dict[str, Any]] = None,
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
        # The child's own session id — filled into the shared ref once the
        # child agent exists (the callback is built first), so every relayed
        # event lets UIs open/inspect the subagent's session directly.
        if session_ref and session_ref.get("session_id"):
            kw["child_session_id"] = str(session_ref["session_id"])
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

        if event_type == "subagent.text":
            # Streamed assistant reply text from the child. Relay verbatim so a
            # gateway watch window can mirror the child "talking" as it streams.
            # No spinner echo — the CLI shows the child via the tree, and the
            # CLI/TUI progress handlers ignore non-tool event types, so this is
            # inert there; only a gateway watch window consumes it.
            _relay("subagent.text", preview=preview)
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


def _normalized_runtime_url(value: Any) -> str:
    return str(value or "").strip().rstrip("/")


def _inherit_parent_base_url(parent_agent, fallback_base_url: Optional[str]) -> Optional[str]:
    """Return the base URL the parent is actually calling, not a stale attribute.

    ``parent_agent.base_url`` can still carry a leftover OpenRouter URL from an
    old config while the live OpenAI client in ``_client_kwargs`` already points
    at local Ollama. Subagents must inherit the active endpoint or they 401
    against OpenRouter with a dummy/local key.
    """
    surface_url = _normalized_runtime_url(fallback_base_url)
    client_kwargs = getattr(parent_agent, "_client_kwargs", None)
    if isinstance(client_kwargs, dict):
        kwargs_url = _normalized_runtime_url(client_kwargs.get("base_url"))
        if (
            kwargs_url
            and kwargs_url != surface_url
            and kwargs_url.startswith(("http://", "https://"))
        ):
            return kwargs_url

    client = getattr(parent_agent, "client", None)
    if client is not None:
        # OpenAI SDK exposes ``base_url`` as an ``httpx.URL``, not ``str`` —
        # coerce so the comparison works regardless of the client's type.
        live_url = _normalized_runtime_url(getattr(client, "base_url", ""))
        if (
            live_url
            and live_url != surface_url
            and live_url.startswith(("http://", "https://"))
        ):
            return live_url

    return fallback_base_url or None


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
    override_request_overrides: Optional[Dict[str, Any]] = None,
    override_max_tokens: Optional[int] = None,
    # ACP transport overrides from trusted delegation config.
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

    delegation_cfg = _load_config()

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
        # Intersect with parent — subagent must not gain tools the parent lacks.
        # Expand composite toolsets (e.g. hermes-cli) so that individual
        # toolset names (e.g. web, terminal) are recognised during intersection.
        expanded_parent = _expand_parent_toolsets(parent_toolsets)
        child_toolsets = [t for t in toolsets if t in expanded_parent]
        if _get_inherit_mcp_toolsets():
            child_toolsets = _preserve_parent_mcp_toolsets(
                child_toolsets, parent_toolsets
            )
        child_toolsets = _strip_blocked_tools(child_toolsets)
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
    child_session_ref: Dict[str, Any] = {}
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
        session_ref=child_session_ref,
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
    if not override_base_url:
        effective_base_url = _inherit_parent_base_url(parent_agent, effective_base_url)
    effective_api_key = override_api_key or parent_api_key
    # Bug #20558 / PR #20563: api_mode must NOT be inherited when the child uses a
    # different provider than the parent — each provider has its own API surface
    # (e.g. MiniMax uses anthropic_messages, DeepSeek uses chat_completions).
    # Inheriting the parent's mode causes 404 errors when the child routes to the
    # wrong endpoint.  Derive the mode from the target provider when it differs.
    _parent_provider = getattr(parent_agent, "provider", None) or ""
    if override_api_mode is not None:
        effective_api_mode = override_api_mode
    elif effective_provider != _parent_provider:
        effective_api_mode = None  # force re-derivation from provider's defaults
    else:
        effective_api_mode = getattr(parent_agent, "api_mode", None)
    # Defensive: validate trusted delegation.command exists on PATH before
    # honoring it. Stale config should not force a child onto the ACP transport
    # and then fail at subprocess startup.
    if override_acp_command:
        import shutil as _shutil

        if not _shutil.which(override_acp_command):
            logger.warning(
                "Ignoring acp_command=%r: binary not found on PATH; "
                "falling back to default transport.",
                override_acp_command,
            )
            override_acp_command = None
            override_acp_args = None
    effective_acp_command = override_acp_command or getattr(
        parent_agent, "acp_command", None
    )
    effective_acp_args = list(
        override_acp_args
        if override_acp_args is not None
        else (getattr(parent_agent, "acp_args", []) or [])
    )

    # When override_provider is set (e.g. delegation.provider: minimax-cn),
    # the subagent must use direct API calls — not the parent's ACP transport.
    # Inheriting acp_command unconditionally causes run_agent.py to initialize
    # CopilotACPClient, bypassing override credentials entirely (issue #16816).
    if override_provider and not override_acp_command:
        effective_acp_command = None
        effective_acp_args = []

    if override_acp_command:
        # If explicitly forcing an ACP transport override, the provider MUST be copilot-acp
        # so run_agent.py initializes the CopilotACPClient.
        effective_provider = "copilot-acp"
        effective_api_mode = "chat_completions"

    # Resolve reasoning config: delegation override > parent inherit
    parent_reasoning = getattr(parent_agent, "reasoning_config", None)
    child_reasoning = parent_reasoning
    try:
        # Keep the raw value — ``str(x or "")`` would coerce a YAML boolean
        # False (``reasoning_effort: false``) to "" and inherit the parent
        # instead of disabling thinking for children.
        delegation_effort = delegation_cfg.get("reasoning_effort")
        if delegation_effort or delegation_effort is False:
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

    # Inherit the parent's fallback provider chain so subagents can recover
    # from rate-limits and credential exhaustion exactly like the top-level
    # agent does.  _fallback_chain is a list accepted by AIAgent's
    # fallback_model parameter (which handles both list and dict forms).
    parent_fallback = getattr(parent_agent, "_fallback_chain", None) or None

    # Inherit the parent's OpenRouter provider-preference filters by default
    # (so subagents routed to the same provider honour the same routing
    # constraints).  BUT: when `delegation.provider` is set the user is
    # explicitly asking the child to run on a different provider, and
    # parent-level OpenRouter filters (e.g. `only=["Anthropic"]`) would
    # silently force the child back onto the parent's provider. Clear the
    # filters in that case so the delegated provider is honoured.
    child_providers_allowed = getattr(parent_agent, "providers_allowed", None)
    child_providers_ignored = getattr(parent_agent, "providers_ignored", None)
    child_providers_order = getattr(parent_agent, "providers_order", None)
    child_provider_sort = getattr(parent_agent, "provider_sort", None)
    child_provider_require_parameters = getattr(
        parent_agent, "provider_require_parameters", False
    )
    child_provider_data_collection = getattr(
        parent_agent, "provider_data_collection", None
    ) or ""
    child_openrouter_min_coding_score = getattr(parent_agent, "openrouter_min_coding_score", None)
    if override_provider:
        child_providers_allowed = None
        child_providers_ignored = None
        child_providers_order = None
        child_provider_sort = None
        child_provider_require_parameters = False
        child_provider_data_collection = ""
        # Note: openrouter_min_coding_score is model-gated (only emitted on
        # openrouter/pareto-code), so we keep it inherited even when the
        # provider is overridden — it's a no-op on any other model.

    child_max_tokens = (
        override_max_tokens
        if override_max_tokens is not None
        else getattr(parent_agent, "max_tokens", None)
    )
    child_optional_kwargs: Dict[str, Any] = {}
    if isinstance(child_max_tokens, int):
        child_optional_kwargs["max_tokens"] = child_max_tokens

    child = AIAgent(
        base_url=effective_base_url,
        api_key=effective_api_key,
        model=effective_model,
        provider=effective_provider,
        api_mode=effective_api_mode,
        acp_command=effective_acp_command,
        acp_args=effective_acp_args,
        max_iterations=max_iterations,

        reasoning_config=child_reasoning,
        prefill_messages=getattr(parent_agent, "prefill_messages", None),
        fallback_model=parent_fallback,
        enabled_toolsets=child_toolsets,
        quiet_mode=True,
        ephemeral_system_prompt=child_prompt,
        log_prefix=f"[subagent-{task_index}]",
        platform="subagent",
        skip_context_files=True,
        skip_memory=True,
        clarify_callback=None,
        thinking_callback=child_thinking_cb,
        session_db=getattr(parent_agent, "_session_db", None),
        parent_session_id=getattr(parent_agent, "session_id", None),
        providers_allowed=child_providers_allowed,
        providers_ignored=child_providers_ignored,
        providers_order=child_providers_order,
        provider_sort=child_provider_sort,
        provider_require_parameters=child_provider_require_parameters,
        provider_data_collection=child_provider_data_collection,
        request_overrides=(
            dict(override_request_overrides or {})
            if override_provider
            else dict(getattr(parent_agent, "request_overrides", {}) or {})
        ),
        openrouter_min_coding_score=child_openrouter_min_coding_score,
        tool_progress_callback=child_progress_cb,
        iteration_budget=None,  # fresh budget per subagent
        **child_optional_kwargs,
    )
    child._print_fn = getattr(parent_agent, "_print_fn", None)
    # Now the child exists, its session id can ride on every relayed event
    # (including the spawn_requested below — first emit happens after this).
    child_session_ref["session_id"] = getattr(child, "session_id", "") or ""
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
    child._parent_turn_id = getattr(parent_agent, "_current_turn_id", "") or ""
    # Stable sidebar marker: delegate subagent sessions must stay out of
    # session pickers even when a parent delete orphans them (parent_session_id
    # → NULL). Mirrors /branch's ``_branched_from`` pattern — see
    # ``list_sessions_rich`` child-exclusion clause.
    parent_sid = getattr(parent_agent, "session_id", None)
    if parent_sid and getattr(child, "_session_init_model_config", None) is not None:
        child._session_init_model_config["_delegate_from"] = parent_sid

    # Share a credential pool with the child when possible so subagents can
    # rotate credentials on rate limits instead of getting pinned to one key.
    child_pool = _resolve_child_credential_pool(
        effective_provider, parent_agent, effective_base_url
    )
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

    try:
        from hermes_cli.plugins import invoke_hook as _invoke_hook
        _invoke_hook(
            "subagent_start",
            parent_session_id=getattr(parent_agent, "session_id", None),
            parent_turn_id=getattr(parent_agent, "_current_turn_id", "") or "",
            parent_subagent_id=parent_subagent_id,
            child_session_id=getattr(child, "session_id", None),
            child_subagent_id=subagent_id,
            child_role=effective_role,
            child_goal=goal,
        )
    except Exception:
        logger.debug("subagent_start hook invocation failed", exc_info=True)

    return child


def _dump_subagent_timeout_diagnostic(
    *,
    child: Any,
    task_index: int,
    timeout_seconds: float,
    duration_seconds: float,
    worker_thread: Optional[threading.Thread],
    goal: str,
) -> Optional[str]:
    """Write a structured diagnostic dump for a subagent that timed out
    before making any API call.

    See issue #14726: users hit "subagent timed out after 300s with no response"
    with zero API calls and no way to inspect what happened. This helper
    writes a dedicated log under ``~/.hermes/logs/subagent-<sid>-<ts>.log``
    capturing the child's config, system-prompt / tool-schema sizes, activity
    tracker snapshot, and the worker thread's Python stack at timeout.

    Returns the absolute path to the diagnostic file, or None on failure.
    """
    try:
        from hermes_constants import get_hermes_home
        import datetime as _dt
        import sys as _sys
        import traceback as _traceback

        hermes_home = get_hermes_home()
        logs_dir = hermes_home / "logs"
        try:
            logs_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return None

        subagent_id = getattr(child, "_subagent_id", None) or f"idx{task_index}"
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        dump_path = logs_dir / f"subagent-timeout-{subagent_id}-{ts}.log"

        lines: List[str] = []
        def _w(line: str = "") -> None:
            lines.append(line)

        _w("# Subagent timeout diagnostic — issue #14726")
        _w(f"# Generated: {_dt.datetime.now().isoformat()}")
        _w("")
        _w("## Timeout")
        _w(f"  task_index:        {task_index}")
        _w(f"  subagent_id:       {subagent_id}")
        _w(f"  configured_timeout: {timeout_seconds}s")
        _w(f"  actual_duration:   {duration_seconds:.2f}s")
        _w("")

        _w("## Goal")
        _goal_preview = (goal or "").strip()
        if len(_goal_preview) > 1000:
            _goal_preview = _goal_preview[:1000] + " ...[truncated]"
        _w(_goal_preview or "(empty)")
        _w("")

        _w("## Child config")
        for attr in (
            "model", "provider", "api_mode", "base_url", "max_iterations",
            "quiet_mode", "skip_memory", "skip_context_files", "platform",
            "_delegate_role", "_delegate_depth",
        ):
            try:
                val = getattr(child, attr, None)
                # Redact api_key-shaped values defensively
                if isinstance(val, str) and attr == "base_url":
                    pass
                _w(f"  {attr}: {val!r}")
            except Exception:
                _w(f"  {attr}: <unreadable>")
        _w("")

        _w("## Toolsets")
        enabled = getattr(child, "enabled_toolsets", None)
        _w(f"  enabled_toolsets:  {enabled!r}")
        tool_names = getattr(child, "valid_tool_names", None)
        if tool_names:
            _w(f"  loaded tool count: {len(tool_names)}")
            try:
                _w(f"  loaded tools:      {sorted(tool_names)}")
            except Exception:
                pass
        _w("")

        _w("## Prompt / schema sizes")
        try:
            sys_prompt = getattr(child, "ephemeral_system_prompt", None) \
                or getattr(child, "system_prompt", None) \
                or ""
            _w(f"  system_prompt_bytes: {len(sys_prompt.encode('utf-8')) if isinstance(sys_prompt, str) else 'n/a'}")
            _w(f"  system_prompt_chars: {len(sys_prompt) if isinstance(sys_prompt, str) else 'n/a'}")
        except Exception as exc:
            _w(f"  system_prompt: <error: {exc}>")
        try:
            tools_schema = getattr(child, "tools", None)
            if tools_schema is not None:
                _schema_json = json.dumps(tools_schema, default=str)
                _w(f"  tool_schema_count: {len(tools_schema)}")
                _w(f"  tool_schema_bytes: {len(_schema_json.encode('utf-8'))}")
        except Exception as exc:
            _w(f"  tool_schema: <error: {exc}>")
        _w("")

        _w("## Activity summary")
        try:
            summary = child.get_activity_summary()
            for k, v in summary.items():
                _w(f"  {k}: {v!r}")
        except Exception as exc:
            _w(f"  <get_activity_summary failed: {exc}>")
        _w("")

        _w("## Worker thread stack at timeout")
        if worker_thread is not None and worker_thread.is_alive():
            frames = _sys._current_frames()
            worker_frame = frames.get(worker_thread.ident)
            if worker_frame is not None:
                stack = _traceback.format_stack(worker_frame)
                for frame_line in stack:
                    for sub in frame_line.rstrip().split("\n"):
                        _w(f"  {sub}")
            else:
                _w("  <worker frame not available>")
        elif worker_thread is None:
            _w("  <no worker thread handle>")
        else:
            _w("  <worker thread already exited>")
        _w("")

        _w("## Notes")
        _w("  This file is written ONLY when a subagent times out with 0 API calls.")
        _w("  0-API-call timeouts mean the child never reached its first LLM request.")
        _w("  Common causes: oversized prompt rejected by provider, transport hang,")
        _w("  credential resolution stuck. See issue #14726 for context.")

        dump_path.write_text("\n".join(lines), encoding="utf-8")
        return str(dump_path)
    except Exception as exc:
        logger.warning("Subagent timeout diagnostic dump failed: %s", exc)
        return None


def _spill_summary_to_file(task_index: int, summary: str) -> Optional[str]:
    """Write a subagent's full summary to the delegation cache and return path.

    Mirrors web_extract's ``_store_full_text``: the file lands in
    ``cache/delegation`` which is mounted read-only into remote backends
    (Docker/Modal/SSH) via ``credential_files._CACHE_DIRS``, so the parent's
    terminal/``read_file`` tools can page through the complete text on any
    backend. Returns the absolute path, or None on failure (best-effort:
    the trimmed head+tail is still returned to the parent regardless).
    """
    try:
        from hermes_constants import get_hermes_dir
        import datetime as _dt

        cache_dir = get_hermes_dir("cache/delegation", "delegation_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = cache_dir / f"subagent-summary-{task_index}-{ts}.txt"
        path.write_text(summary, encoding="utf-8")
        return str(path)
    except Exception as exc:
        logger.debug("Failed to spill subagent summary to file: %s", exc)
        return None


def _trim_summary_with_footer(
    summary: str, cap: int, task_index: int
) -> tuple[str, Optional[str]]:
    """Return (model_text, spill_path) for one over-budget summary.

    Mirrors web_extract's ``_truncate_with_footer``: keep a head+tail window
    (~75% head / ~25% tail, snapped to line boundaries) so the subagent's
    opening AND its closing (outcomes / files-changed / issues, which live at
    the end) both survive, spill the full text to disk, and append a footer
    telling the parent exactly how much it's seeing and the precise
    ``read_file offset=`` to page into the omitted middle. Deterministic.
    """
    original_len = len(summary)
    head_budget = int(cap * 0.75)
    tail_budget = cap - head_budget

    head = summary[:head_budget]
    tail = summary[-tail_budget:]
    # Snap the head cut back to the last newline so we don't slice mid-line.
    nl = head.rfind("\n")
    if nl > head_budget * 0.5:
        head = head[:nl]
    # Snap the tail cut forward to the next newline for the same reason.
    nl = tail.find("\n")
    if 0 <= nl < tail_budget * 0.5:
        tail = tail[nl + 1:]

    spill_path = _spill_summary_to_file(task_index, summary)

    footer_lines = [
        "",
        "─" * 8 + " [SUMMARY TRUNCATED] " + "─" * 8,
        f"Showing {len(head):,} chars (head) + {len(tail):,} chars (tail) "
        f"of {original_len:,} total — trimmed to protect the parent's context window.",
    ]
    if spill_path:
        # read_file is 1-indexed; +2 moves past the last head line shown.
        middle_start_line = head.count("\n") + 2
        footer_lines.append(f"Full subagent output saved to: {spill_path}")
        footer_lines.append(
            f'To read the omitted middle: read_file path="{spill_path}" '
            f"offset={middle_start_line} limit=200  (the file is the complete "
            f"summary; raise/lower offset to page through it)."
        )
    else:
        footer_lines.append(
            "Full output could not be stored to disk; the head+tail above is "
            "all that was preserved."
        )
    footer_lines.append("─" * 37)

    model_text = head + "\n\n[... middle omitted — see footer ...]\n\n" + tail + "\n".join(footer_lines)
    return model_text, spill_path


def _parent_summary_char_budget(parent_agent, n_summaries: int) -> Optional[int]:
    """Per-summary character budget sized against the parent's *remaining*
    context headroom, split across the batch.

    The overflow this guards against is N summaries entering the parent
    context at once (batch fan-out), not any single summary being large.  We
    take a fraction of the headroom the parent has left (resolved context
    length minus what's already in its prompt) and divide it across the batch,
    converting tokens→chars at the standard ~4 chars/token estimate.

    Returns the per-summary char budget, or None when the parent's context
    state is unknown (no compressor / no token count) — in which case the
    caller falls back to the static char ceiling only.
    """
    try:
        compressor = getattr(parent_agent, "context_compressor", None)
        context_length = getattr(compressor, "context_length", None)
        if not isinstance(context_length, int) or context_length <= 0:
            return None

        used_tokens = getattr(parent_agent, "session_prompt_tokens", 0)
        if not isinstance(used_tokens, (int, float)) or used_tokens < 0:
            used_tokens = 0

        # Reserve the compressor's output budget so we measure INPUT headroom.
        reserved = getattr(compressor, "max_tokens", 0) or 0
        headroom_tokens = context_length - int(used_tokens) - int(reserved)
        if headroom_tokens <= 0:
            # Parent is already over budget — give each summary only the floor.
            return _MIN_SUMMARY_CHARS

        batch_token_budget = int(headroom_tokens * _SUMMARY_HEADROOM_FRACTION)
        per_summary_tokens = batch_token_budget // max(1, n_summaries)
        per_summary_chars = per_summary_tokens * 4  # ~4 chars/token
        return max(_MIN_SUMMARY_CHARS, per_summary_chars)
    except Exception:
        logger.debug("Summary budget computation failed", exc_info=True)
        return None


def _apply_summary_budget(results: List[Dict[str, Any]], parent_agent) -> None:
    """Trim subagent summaries in-place so the batch can't overflow the
    parent's context window, spilling full text to disk so nothing is lost.

    The effective per-summary cap is the MIN of:
      - the dynamic headroom budget (remaining parent context ÷ batch size), and
      - the static ``delegation.max_summary_chars`` ceiling (0 = disabled).

    When a summary exceeds the cap, its full text is written to a file and the
    in-context summary becomes a head slice plus a pointer to that file. This
    addresses issue/PR #9126: batch fan-out returned N full summaries verbatim,
    blowing the parent context and (on rate-limited providers) triggering a
    compression/429 death spiral.
    """
    summaries = [
        r for r in results if isinstance(r, dict) and isinstance(r.get("summary"), str) and r["summary"]
    ]
    if not summaries:
        return

    cfg = _load_config()
    try:
        static_ceiling = int(cfg.get("max_summary_chars", DEFAULT_MAX_SUMMARY_CHARS))
    except (TypeError, ValueError):
        static_ceiling = DEFAULT_MAX_SUMMARY_CHARS

    dynamic_budget = _parent_summary_char_budget(parent_agent, len(summaries))

    # Combine the two caps. Either can be absent/disabled.
    candidates = [c for c in (static_ceiling, dynamic_budget) if c and c > 0]
    if not candidates:
        return  # both disabled / unknown → leave summaries untouched
    cap = min(candidates)

    for entry in summaries:
        summary = entry["summary"]
        if len(summary) <= cap:
            continue
        original_len = len(summary)
        model_text, spill_path = _trim_summary_with_footer(
            summary, cap, entry.get("task_index", -1)
        )
        entry["summary"] = model_text
        entry["summary_truncated"] = True
        if spill_path:
            entry["summary_full_path"] = spill_path
        logger.debug(
            "[subagent-%s] summary trimmed %d → ~%d chars (spill=%s)",
            entry.get("task_index", "?"),
            original_len,
            cap,
            spill_path or "none",
        )


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
    # Stale detection: track the child's (tool, iteration) pair across
    # heartbeat cycles. If neither advances, count the cycle as stale.
    # Different thresholds for idle vs in-tool (see _HEARTBEAT_STALE_CYCLES_*).
    _last_seen_iter = [0]
    _last_seen_tool = [None]  # type: list
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

                # Stale detection: count cycles where neither the iteration
                # count nor the current_tool advances. A child running a
                # legitimately long-running tool (terminal command, web
                # fetch) keeps current_tool set but doesn't advance
                # api_call_count — we don't want that to look stale at the
                # idle threshold.
                iter_advanced = child_iter > _last_seen_iter[0]
                tool_changed = child_tool != _last_seen_tool[0]
                if iter_advanced or tool_changed:
                    _last_seen_iter[0] = child_iter
                    _last_seen_tool[0] = child_tool
                    _stale_count[0] = 0
                else:
                    _stale_count[0] += 1

                # Pick threshold based on whether the child is currently
                # inside a tool call. In-tool threshold is high enough to
                # cover legitimately slow tools; idle threshold stays
                # tight so the gateway timeout can fire on a truly wedged
                # child.
                stale_limit = (
                    _HEARTBEAT_STALE_CYCLES_IN_TOOL
                    if child_tool
                    else _HEARTBEAT_STALE_CYCLES_IDLE
                )
                if _stale_count[0] >= stale_limit:
                    logger.warning(
                        "Subagent %d appears stale (no progress for %d "
                        "heartbeat cycles, tool=%s) — stopping heartbeat",
                        task_index,
                        _stale_count[0],
                        child_tool or "<none>",
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
        _heartbeat_thread.start()
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

        # Run child with an optional hard timeout (off by default —
        # result(timeout=None) blocks until the child finishes). Stuck-child
        # protection comes from the heartbeat staleness monitor instead.
        child_timeout = _get_child_timeout()
        # Daemon worker (tools.daemon_pool): a timed-out child is abandoned
        # below; a stdlib non-daemon worker would then block interpreter
        # exit at atexit-join time if the child never unwinds.
        from tools.daemon_pool import DaemonThreadPoolExecutor
        _timeout_executor = DaemonThreadPoolExecutor(
            max_workers=1,
            # Install a non-interactive approval callback in the worker thread
            # so dangerous-command prompts from the subagent don't fall back to
            # input() and deadlock the parent's prompt_toolkit TUI.
            # Callback (deny vs approve) is governed by delegation.subagent_auto_approve.
            initializer=_set_subagent_approval_cb,
            initargs=(_get_subagent_approval_callback(),),
        )
        # Capture the worker thread so the timeout diagnostic can dump its
        # Python stack (see #14726 — 0-API-call hangs are opaque without it).
        _worker_thread_holder: Dict[str, Optional[threading.Thread]] = {"t": None}

        def _relay_child_text(delta: str) -> None:
            # Forward the child's streamed reply text up the progress relay so
            # gateway watch windows mirror it live (subagent.text → message.delta).
            # Inert under CLI/TUI: their progress handlers ignore non-tool events.
            if not delta or not child_progress_cb:
                return
            try:
                child_progress_cb("subagent.text", preview=delta)
            except Exception as e:
                logger.debug("Child text relay failed: %s", e)

        def _run_with_thread_capture():
            _worker_thread_holder["t"] = threading.current_thread()
            return child.run_conversation(
                user_message=goal,
                task_id=child_task_id,
                stream_callback=_relay_child_text,
            )

        _child_future = _timeout_executor.submit(_run_with_thread_capture)
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

            # When a subagent times out BEFORE making any API call, dump a
            # diagnostic to help users (and us) see what the child was doing.
            # See #14726 — without this, 0-API-call hangs are black boxes.
            diagnostic_path: Optional[str] = None
            child_api_calls = 0
            try:
                _summary = child.get_activity_summary()
                child_api_calls = int(_summary.get("api_call_count", 0) or 0)
            except Exception:
                pass
            if is_timeout and child_api_calls == 0:
                diagnostic_path = _dump_subagent_timeout_diagnostic(
                    child=child,
                    task_index=task_index,
                    # is_timeout implies a cap was configured (result(timeout=None)
                    # never raises FuturesTimeoutError); guard for the type checker.
                    timeout_seconds=float(child_timeout or 0.0),
                    duration_seconds=float(duration),
                    worker_thread=_worker_thread_holder.get("t"),
                    goal=goal,
                )
                if diagnostic_path:
                    logger.warning(
                        "Subagent %d 0-API-call timeout — diagnostic written to %s",
                        task_index,
                        diagnostic_path,
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

            if is_timeout:
                if child_api_calls == 0:
                    _err = (
                        f"Subagent timed out after {child_timeout}s without "
                        f"making any API call — the child never reached its "
                        f"first LLM request (prompt construction, credential "
                        f"resolution, or transport may be stuck)."
                    )
                    if diagnostic_path:
                        _err += f" Diagnostic: {diagnostic_path}"
                else:
                    _err = (
                        f"Subagent timed out after {child_timeout}s with "
                        f"{child_api_calls} API call(s) completed — likely "
                        f"stuck on a slow API call or unresponsive network request."
                    )
            else:
                _err = str(_timeout_exc)

            return {
                "task_index": task_index,
                "status": "timeout" if is_timeout else "error",
                "summary": None,
                "error": _err,
                "exit_reason": "timeout" if is_timeout else "error",
                "api_calls": child_api_calls,
                "duration_seconds": duration,
                "_child_role": getattr(child, "_delegate_role", None),
                "diagnostic_path": diagnostic_path,
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

        # The child emits the literal "(empty)" sentinel (see run_agent.py) when
        # it gives up after repeated empty-LLM-response retries — typically a
        # transport bug (misrouted provider, adapter returning empty
        # ChatCompletion, etc.). Treat it as a failure so the parent surfaces
        # it instead of silently accepting zero-content "success".
        _empty_sentinel = summary.strip() == "(empty)"

        if interrupted:
            status = "interrupted"
        elif summary and not _empty_sentinel:
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
                    content = _stringify_tool_content(msg.get("content", ""))
                    is_error = _looks_like_error_output(content)
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
            # Captured before child.close() so the parent aggregator can fold
            # the child's total spend into the parent's session cost.  Port of
            # Kilo-Org/kilocode#9448 — previously the footer only reflected the
            # parent's direct API calls and under-counted subagent-heavy runs.
            # Stripped before the dict is serialised back to the model.
            "_child_cost_usd": (
                float(getattr(child, "session_estimated_cost_usd", 0.0) or 0.0)
                if isinstance(
                    getattr(child, "session_estimated_cost_usd", 0.0),
                    (int, float),
                )
                else 0.0
            ),
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
        # after the child has finished (or failed).  Guard the join: .start()
        # now lives inside the try block, so if it raised (OS thread
        # exhaustion) the thread was never started and Thread.join() would
        # raise RuntimeError.  ident is None until start() succeeds.
        _heartbeat_stop.set()
        if _heartbeat_thread.ident is not None:
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


def _recover_tasks_from_json_string(
    tasks: Any,
) -> tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    if not isinstance(tasks, str):
        return None, None
    raw = tasks.strip()
    if not raw:
        return None, "Provide either 'goal' (single task) or 'tasks' (batch)."
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, (
            "tasks must be a JSON array of task objects; received a string "
            f"that could not be parsed as JSON ({exc.msg})."
        )
    if not isinstance(parsed, list):
        return None, (
            f"tasks must be a JSON array of task objects; parsed "
            f"{type(parsed).__name__} instead."
        )
    return parsed, None


def delegate_task(
    goal: Optional[str] = None,
    context: Optional[str] = None,
    tasks: Optional[List[Dict[str, Any]]] = None,
    max_iterations: Optional[int] = None,
    role: Optional[str] = None,
    background: Optional[bool] = None,
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
        return tool_error(
            "Delegation spawning is paused. Clear the pause via the TUI "
            "(`p` in /agents) or the `delegation.pause` RPC before retrying."
        )

    # Normalise the top-level role once; per-task overrides re-normalise.
    top_role = _normalize_role(role)

    # Background (async) delegation now applies to BOTH single tasks and
    # batches. A batch simply becomes N independent async dispatches: each
    # child runs on the daemon executor and re-enters the conversation via
    # the completion queue on its own, carrying its own handle. There's no
    # combined "wait for all" — fan-out is exactly N background subagents.
    background = is_truthy_value(background, default=False) if background is not None else False

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
                    f"nesting is required (no hard ceiling, but each level "
                    f"multiplies API cost)."
                )
            }
        )

    # Load config
    cfg = _load_config()
    default_max_iter = cfg.get("max_iterations", DEFAULT_MAX_ITERATIONS)
    # Model-supplied max_iterations is ignored — the config value is authoritative
    # so users get predictable budgets. The kwarg is retained for internal callers
    # and tests; a model-emitted value here would only shrink the budget and
    # surprise the user mid-run. Log and drop it if one slips through from a
    # cached tool schema or a stale provider.
    if max_iterations is not None and max_iterations != default_max_iter:
        logger.debug(
            "delegate_task: ignoring caller-supplied max_iterations=%s; "
            "using delegation.max_iterations=%s from config",
            max_iterations, default_max_iter,
        )
    effective_max_iter = default_max_iter

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
    recovered_tasks, tasks_error = _recover_tasks_from_json_string(tasks)
    if tasks_error:
        return tool_error(tasks_error)
    if recovered_tasks is not None:
        tasks = recovered_tasks

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
        task_list = [{"goal": goal, "context": context, "role": top_role}]
    else:
        return tool_error("Provide either 'goal' (single task) or 'tasks' (batch).")

    if not task_list:
        return tool_error("No tasks provided.")

    # Validate each task has a goal
    for i, task in enumerate(task_list):
        if not isinstance(task, dict):
            return tool_error(
                f"Task {i} must be an object, got {type(task).__name__}."
            )
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
            # Per-task role beats top-level; normalise again so unknown
            # per-task values warn and degrade to leaf uniformly.
            effective_role = _normalize_role(t.get("role") or top_role)
            child = _build_child_agent(
                task_index=i,
                goal=t["goal"],
                context=t.get("context"),
                # Subagents always inherit the parent's toolsets; the model
                # cannot choose or narrow them (no model-facing toolsets arg).
                toolsets=None,
                model=creds["model"],
                max_iterations=effective_max_iter,
                task_count=n_tasks,
                parent_agent=parent_agent,
                override_provider=creds["provider"],
                override_base_url=creds["base_url"],
                override_api_key=creds["api_key"],
                override_api_mode=creds["api_mode"],
                override_request_overrides=creds.get("request_overrides"),
                override_max_tokens=creds.get("max_output_tokens"),
                override_acp_command=creds.get("command"),
                override_acp_args=creds.get("args"),
                role=effective_role,
            )
            # Override with correct parent tool names (before child construction mutated global)
            child._delegate_saved_tool_names = _parent_tool_names
            children.append((i, t, child))
    finally:
        # Authoritative restore: reset global to parent's tool names after all children built
        _model_tools._last_resolved_tool_names = _parent_tool_names

    def _execute_and_aggregate() -> dict:
        """Run all built children (1 or N), join on them, aggregate results,
        fire subagent_stop hooks + cost rollup, and return the combined result
        dict. Used by BOTH the synchronous path and the background runner. In
        the background case this whole function runs on the daemon executor, so
        the parent turn isn't blocked — but the batch still JOINS on itself
        here (all children must finish) before producing ONE consolidated
        results block. That is the contract: fan-out runs in the background,
        waits on each other, and returns together.
        """
        if n_tasks == 1:
            # Single task -- run directly (no thread pool overhead)
            _i, _t, child = children[0]
            result = _run_single_child(_i, _t["goal"], child, parent_agent)
            results.append(result)
        else:
            # Batch -- run in parallel with per-task progress lines
            completed_count = 0
            spinner_ref = getattr(parent_agent, "_delegate_spinner", None)

            # Daemon workers (tools.daemon_pool): the `with` block still joins
            # normally, but if the parent is interrupted while a child is
            # wedged, the abandoned worker must not block interpreter exit.
            from tools.daemon_pool import DaemonThreadPoolExecutor
            with DaemonThreadPoolExecutor(max_workers=max_children) as executor:
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
                                _emit_parent_console(parent_agent, f"  {completion_line}")
                        else:
                            _emit_parent_console(parent_agent, f"  {completion_line}")

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

        # Cap subagent summaries against the parent's remaining context
        # headroom (split across the batch) before they enter the parent's
        # conversation. Full text is spilled to disk so nothing is lost.
        # Covers both the single-task and batch paths. See PR #9126.
        _apply_summary_budget(results, parent_agent)

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
        # Aggregate child spend here so the parent's footer/UI reflect the true
        # cost of a subagent-heavy turn.  Port of Kilo-Org/kilocode#9448.  Each
        # child's cost was captured in _run_single_child before its AIAgent was
        # closed; we fold them into the parent in one pass alongside the
        # subagent_stop hook loop so we don't walk `results` twice.
        _children_cost_total = 0.0
        for entry in results:
            child_role = entry.pop("_child_role", None)
            child_cost = entry.pop("_child_cost_usd", 0.0)
            try:
                if child_cost:
                    _children_cost_total += float(child_cost)
            except (TypeError, ValueError):
                pass
            if _invoke_hook is None:
                continue
            try:
                _child_index = entry.get("task_index", -1)
                _child_agent = (
                    children[_child_index][2]
                    if isinstance(_child_index, int) and 0 <= _child_index < len(children)
                    else None
                )
                _invoke_hook(
                    "subagent_stop",
                    parent_session_id=_parent_session_id,
                    parent_turn_id=getattr(parent_agent, "_current_turn_id", "") or "",
                    child_session_id=getattr(_child_agent, "session_id", None),
                    child_role=child_role,
                    child_summary=entry.get("summary"),
                    child_status=entry.get("status"),
                    duration_ms=int((entry.get("duration_seconds") or 0) * 1000),
                )
            except Exception:
                logger.debug("subagent_stop hook invocation failed", exc_info=True)

        # Fold the aggregated child cost into the parent's session total.  This is
        # additive — each delegate_task call contributes its own children — so
        # nested orchestrator→worker trees roll up naturally: each layer's own
        # delegate_task() folds its direct children in, and when the orchestrator
        # itself finishes, its parent folds the orchestrator's now-inflated total
        # on top.  Degrades silently if the parent lacks the counter (older test
        # fixtures, etc.).
        if _children_cost_total > 0.0:
            try:
                current = float(getattr(parent_agent, "session_estimated_cost_usd", 0.0) or 0.0)
                parent_agent.session_estimated_cost_usd = current + _children_cost_total
                # Upgrade the cost_source so the UI doesn't label a partially-real
                # total as "none" when the parent itself hadn't billed any calls
                # yet (rare but possible when the parent's only action this turn
                # was delegate_task).
                if getattr(parent_agent, "session_cost_source", "none") in {None, "", "none"}:
                    parent_agent.session_cost_source = "subagent"
                if getattr(parent_agent, "session_cost_status", "unknown") in {None, "", "unknown"}:
                    parent_agent.session_cost_status = "estimated"
            except Exception:
                logger.debug("Subagent cost rollup failed", exc_info=True)

        total_duration = round(time.monotonic() - overall_start, 2)

        return {
            "results": results,
            "total_duration_seconds": total_duration,
        }

    # ----- Background dispatch: run the WHOLE batch as one async unit -----
    # When background is true, the entire fan-out runs on the daemon executor
    # via a single async delegation. _execute_and_aggregate() joins on every
    # child and produces ONE consolidated results block, which re-enters the
    # conversation as a single message when ALL children finish. The chat is
    # not blocked in the meantime. This is the contract: dispatch N subagents,
    # keep chatting, get the combined summaries back together at the end.
    if background:
        from tools.async_delegation import dispatch_async_delegation_batch
        from tools.approval import get_current_session_key

        # Stateless request/response sessions (the API server / WebUI path)
        # cannot route a detached subagent result back to the agent after the
        # turn ends — there is no persistent channel and the adapter's send()
        # is a no-op, so a background dispatch would silently never re-enter the
        # conversation (issue #10760). Fall back to SYNCHRONOUS execution: the
        # work still runs and its result returns in this same response, which is
        # strictly better than a handle that never resolves. Mirrors the
        # pool-at-capacity inline fallback below.
        try:
            from gateway.session_context import async_delivery_supported
            _async_ok = async_delivery_supported()
        except Exception:
            _async_ok = True
        if not _async_ok:
            logger.info(
                "delegate_task: async delivery unsupported on this session "
                "(stateless HTTP API); running the batch synchronously instead."
            )
            _sync_result = _execute_and_aggregate()
            if isinstance(_sync_result, dict):
                _sync_result["note"] = (
                    "background=true is not available on this endpoint (stateless "
                    "HTTP API — no channel to deliver a detached subagent result "
                    "after the turn ends), so the subagent(s) ran SYNCHRONOUSLY and "
                    "the result is included above."
                )
            return json.dumps(_sync_result, ensure_ascii=False)

        _session_key = get_current_session_key(default="")
        _origin_ui_session_id = ""
        try:
            from gateway.session_context import get_session_env

            _source = get_session_env("HERMES_SESSION_SOURCE", "")
            _origin_ui_session_id = get_session_env("HERMES_UI_SESSION_ID", "")
            # In desktop/TUI, the routable session key is the durable
            # AIAgent.session_id. Context compression can rotate that id during
            # the same turn before the TUI-side session dict is re-anchored;
            # if we capture the stale approval/session context key here, the
            # async completion becomes an orphan and any desktop poller may
            # consume it. Gateway chats are different: their session_key is the
            # platform conversation key (agent:main:...), so keep it there.
            if _source == "tui":
                _agent_session_id = str(getattr(parent_agent, "session_id", "") or "")
                if _agent_session_id:
                    _session_key = _agent_session_id
        except Exception:
            _origin_ui_session_id = ""
        _parent_session_id = getattr(parent_agent, "session_id", None)
        _child_agents = [c for (_, _, c) in children]

        # Detach every child from the parent's interrupt-propagation list — the
        # batch's lifecycle is owned by the async registry now, not the parent
        # turn. _build_child_agent attached them (correct for sync runs).
        if hasattr(parent_agent, "_active_children"):
            _ac_lock = getattr(parent_agent, "_active_children_lock", None)
            for _c in _child_agents:
                try:
                    if _ac_lock:
                        with _ac_lock:
                            parent_agent._active_children.remove(_c)
                    else:
                        parent_agent._active_children.remove(_c)
                except ValueError:
                    pass

        def _batch_runner():
            return _execute_and_aggregate()

        def _batch_interrupt():
            for _c in _child_agents:
                try:
                    if hasattr(_c, "interrupt"):
                        _c.interrupt("Async delegation cancelled")
                    elif hasattr(_c, "_interrupt_requested"):
                        _c._interrupt_requested = True
                except Exception:
                    pass

        _goals = [t["goal"] for t in task_list]
        dispatch = dispatch_async_delegation_batch(
            goals=_goals,
            context=context,
            # Metadata for the completion block only; subagents inherit the
            # parent's toolsets (no model-facing toolsets arg).
            toolsets=None,
            role=top_role,
            model=creds["model"],
            session_key=_session_key,
            origin_ui_session_id=_origin_ui_session_id,
            parent_session_id=_parent_session_id,
            runner=_batch_runner,
            interrupt_fn=_batch_interrupt,
            max_async_children=_get_max_async_children(),
        )

        if dispatch.get("status") == "dispatched":
            n = len(_goals)
            note = (
                "Subagent is running in the background. You and the user can "
                "keep working; its full result re-enters the conversation as a "
                "new message when it finishes. Do not wait or poll — just "
                "continue."
                if n == 1 else
                f"{n} subagents are running in parallel in the background. You "
                f"and the user can keep working; they wait on each other and "
                f"their consolidated results re-enter the conversation as a "
                f"single message once ALL of them finish. Do not wait or poll "
                f"— just continue."
            )
            payload = {
                "status": "dispatched",
                "mode": "background",
                "count": n,
                "delegation_id": dispatch["delegation_id"],
                "goals": _goals,
                "note": note,
            }
            return json.dumps(payload, ensure_ascii=False)

        # Pool at capacity / schedule failure — children are still attached
        # (we detach above only on the parent list, but the async unit was
        # never accepted, so re-attaching isn't needed: we just run inline).
        logger.info(
            "delegate_task: async pool at capacity (%s); running the whole "
            "batch synchronously instead.",
            dispatch.get("error", "rejected"),
        )
        _cap_result = _execute_and_aggregate()
        if isinstance(_cap_result, dict):
            _cap_result["note"] = (
                "The background delegation pool was at capacity "
                "(delegation.max_concurrent_children), so the subagent(s) ran "
                "SYNCHRONOUSLY and the result is included above. Raise "
                "delegation.max_concurrent_children in config.yaml to allow "
                "more concurrent background delegations."
            )
        return json.dumps(_cap_result, ensure_ascii=False)

    # ----- Synchronous path -----
    return json.dumps(_execute_and_aggregate(), ensure_ascii=False)


def _resolve_child_credential_pool(
    effective_provider: Optional[str],
    parent_agent,
    effective_base_url: Optional[str] = None,
):
    """Resolve a credential pool for the child agent.

    Rules:
    1. Same provider as the parent -> share the parent's pool so cooldown state
       and rotation stay synchronized.
    2. Different provider -> try to load that provider's own pool.
    3. No pool available -> return None and let the child keep the inherited
       fixed credential behavior.

    Custom endpoints are a special case: every direct ``delegation.base_url``
    runtime collapses to ``provider="custom"``, so bare provider equality would
    treat two *different* custom endpoints as interchangeable and let the child
    inherit the parent's pool. Leasing from that pool then overwrites the
    child's delegated ``base_url`` with the parent's endpoint (issue #7833).
    We therefore resolve custom runtimes by endpoint identity (the
    ``custom:<name>`` pool key derived from the base_url) and only share the
    parent's pool when both resolve to the *same* custom endpoint.
    """
    if not effective_provider:
        return getattr(parent_agent, "_credential_pool", None)

    parent_provider = getattr(parent_agent, "provider", None) or ""
    parent_pool = getattr(parent_agent, "_credential_pool", None)

    # Custom endpoints: distinguish by endpoint identity, not the bare "custom"
    # provider string. Two custom runtimes are only interchangeable when they
    # resolve to the same custom:<name> pool key.
    if effective_provider == "custom":
        try:
            from agent.credential_pool import get_custom_provider_pool_key, load_pool

            child_key = get_custom_provider_pool_key(effective_base_url)
            if child_key is None:
                # Unregistered endpoint (raw delegation.base_url with no
                # matching custom_providers entry) -> no shared pool exists.
                # Keep the child's fixed delegated credential rather than
                # risk inheriting the parent's custom endpoint.
                return None

            # Reuse the parent's pool only when it is the same custom endpoint.
            parent_key = get_custom_provider_pool_key(
                getattr(parent_agent, "base_url", None)
            )
            if (
                parent_pool is not None
                and parent_provider == "custom"
                and parent_key is not None
                and parent_key == child_key
            ):
                return parent_pool

            pool = load_pool(child_key)
            if pool is not None and pool.has_credentials():
                return pool
        except Exception as exc:
            logger.debug(
                "Could not resolve custom credential pool for child endpoint '%s': %s",
                effective_base_url,
                exc,
            )
        return None

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
    OpenAI-compatible endpoint. ``delegation.api_key`` overrides the key; when
    omitted, ``api_key`` is returned as ``None`` so ``_build_child_agent``
    inherits the parent agent's key (``effective_api_key = override_api_key or
    parent_api_key``). This lets providers that store their key outside
    ``OPENAI_API_KEY`` (e.g. ``MINIMAX_API_KEY``, ``DASHSCOPE_API_KEY``) work
    without a duplicate config entry.

    Otherwise, if ``delegation.provider`` is configured, the full credential
    bundle (base_url, api_key, api_mode, provider) is resolved via the runtime
    provider system — the same path used by CLI/gateway startup. This lets
    subagents run on a completely different provider:model pair.

    If neither base_url nor provider is configured, returns None values so the
    child inherits everything from the parent agent.

    Raises ValueError with a user-friendly message on credential failure.
    """
    configured_model = str(cfg.get("model") or "").strip() or None
    configured_provider = str(cfg.get("provider") or "").strip() or None
    configured_base_url = str(cfg.get("base_url") or "").strip() or None
    configured_api_key = str(cfg.get("api_key") or "").strip() or None
    configured_api_mode = str(cfg.get("api_mode") or "").strip().lower() or None

    # Native-SDK providers (Bedrock, Vertex, Google GenAI) speak their own
    # wire protocol — they cannot be reached via OpenAI chat_completions against
    # a base_url. For these, always fall through to resolve_runtime_provider()
    # so the proper SDK path is taken. The configured base_url is still
    # forwarded through runtime-provider resolution when applicable (e.g. a
    # custom Bedrock regional endpoint).
    _NATIVE_SDK_PROVIDERS = {"bedrock", "vertex", "google", "google-genai"}
    _provider_lower = (configured_provider or "").strip().lower()
    _is_native_sdk_provider = _provider_lower in _NATIVE_SDK_PROVIDERS

    if configured_base_url and not _is_native_sdk_provider:
        # When delegation.api_key is not set, return None so _build_child_agent
        # falls back to the parent agent's API key via the credential inheritance
        # path (effective_api_key = override_api_key or parent_api_key). This
        # lets providers that store their key in a non-OPENAI_API_KEY env var
        # (e.g. MINIMAX_API_KEY, DASHSCOPE_API_KEY) work without requiring
        # callers to duplicate the key under delegation.api_key.
        api_key = configured_api_key  # None → inherited from parent in _build_child_agent

        # Use the shared URL-based api_mode detector (same path the main agent's
        # runtime resolver uses) so Anthropic-compatible direct endpoints with a
        # /anthropic suffix — Azure AI Foundry, MiniMax, Zhipu GLM, LiteLLM
        # proxies — pick the right transport automatically. Without this,
        # subagents would default to chat_completions and hit 404s on endpoints
        # that only speak the Anthropic Messages protocol. Fixes #10213.
        from hermes_cli.runtime_provider import _detect_api_mode_for_url

        base_lower = configured_base_url.lower()
        provider = "custom"
        api_mode = _detect_api_mode_for_url(configured_base_url) or "chat_completions"
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

        # Explicit delegation.api_mode in config always wins. Lets users force
        # a transport for non-standard endpoints the URL heuristic can't detect.
        if configured_api_mode in {"chat_completions", "codex_responses", "anthropic_messages"}:
            api_mode = configured_api_mode

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
            "request_overrides": None,
            "max_output_tokens": None,
        }

    # Provider is configured — resolve full credentials
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(requested=configured_provider, target_model=configured_model)
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
        "model": configured_model or runtime.get("model") or None,
        "provider": configured_provider if runtime.get("provider") == _RUNTIME_PROVIDER_CUSTOM else runtime.get("provider"),
        "base_url": runtime.get("base_url"),
        "api_key": api_key,
        "api_mode": runtime.get("api_mode"),
        "request_overrides": dict(runtime.get("request_overrides") or {}),
        "max_output_tokens": runtime.get("max_output_tokens"),
        "command": runtime.get("command"),
        "args": list(runtime.get("args") or []),
    }


def _load_config() -> dict:
    """Load delegation config from the active Hermes config.

    Prefer the shared persistent loader because it follows the active
    HERMES_HOME/profile. ``cli.CLI_CONFIG`` is a legacy fallback for entry
    points that cannot import the shared loader; importing it first can return
    an old default ``delegation`` block and hide user-set keys such as
    ``max_concurrent_children``.

    Uses ``load_config_readonly()``: every consumer of this dict is read-only
    (``.get()`` lookups), and this runs on each ``get_definitions()`` schema
    rebuild via ``_get_max_concurrent_children``, so skipping the defensive
    deepcopy matters. Do NOT mutate the returned dict.

    ``HERMES_IGNORE_USER_CONFIG=1`` (``hermes chat --ignore-user-config``) is
    only honored by the legacy ``cli`` loader, not the shared one, so when the
    flag is set we keep ``cli.CLI_CONFIG`` authoritative to preserve the
    flag's contract of suppressing user config.yaml settings.
    """
    prefer_legacy = os.environ.get("HERMES_IGNORE_USER_CONFIG") == "1"
    if not prefer_legacy:
        try:
            from hermes_cli.config import load_config_readonly

            full = load_config_readonly()
            cfg = full.get("delegation") or {}
            if isinstance(cfg, dict):
                return cfg
        except Exception:
            pass
    try:
        from cli import CLI_CONFIG

        cfg = CLI_CONFIG.get("delegation") or {}
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# OpenAI Function-Calling Schema
# ---------------------------------------------------------------------------


def _build_top_level_description() -> str:
    """Compose the delegate_task tool description with current runtime limits.

    The model needs to know its actual ceilings (not the framework defaults),
    otherwise it self-caps at "default 3" / "default 2" even when the user has
    raised delegation.max_concurrent_children / max_spawn_depth. Called both
    at module import (to seed DELEGATE_TASK_SCHEMA) and on every
    get_definitions() call via dynamic_schema_overrides.
    """
    try:
        max_children = _get_max_concurrent_children()
    except Exception:
        max_children = _DEFAULT_MAX_CONCURRENT_CHILDREN
    try:
        max_depth = _get_max_spawn_depth()
    except Exception:
        max_depth = MAX_DEPTH
    try:
        orchestrator_on = _get_orchestrator_enabled()
    except Exception:
        orchestrator_on = True

    if max_depth >= 2 and orchestrator_on:
        nesting_clause = (
            f"Nested delegation IS enabled for this user "
            f"(max_spawn_depth={max_depth}): pass role='orchestrator' on a "
            f"child to let it spawn its own workers, up to {max_depth - 1} "
            f"additional level(s) deep."
        )
    elif max_depth >= 2 and not orchestrator_on:
        nesting_clause = (
            f"Nested delegation is DISABLED on this install "
            f"(delegation.orchestrator_enabled=false), even though "
            f"max_spawn_depth={max_depth}. role='orchestrator' is silently "
            f"forced to 'leaf'."
        )
    else:
        nesting_clause = (
            f"Nested delegation is OFF for this user "
            f"(max_spawn_depth={max_depth}): every child is a leaf and "
            f"cannot delegate further. Raise delegation.max_spawn_depth in "
            f"config.yaml to enable nesting."
        )

    return (
        "Spawn one or more subagents to work on tasks in isolated contexts. "
        "Each subagent gets its own conversation, terminal session, and toolset. "
        "Only the final summary is returned -- intermediate tool results "
        "never enter your context window.\n\n"
        "TWO MODES (one of 'goal' or 'tasks' is required):\n"
        "1. Single task: provide 'goal' (+ optional context, toolsets).\n"
        f"2. Batch (parallel): provide 'tasks' array with up to {max_children} "
        f"items concurrently for this user (configured via "
        f"delegation.max_concurrent_children in config.yaml). {nesting_clause}\n\n"
        "BOTH MODES RUN IN THE BACKGROUND. delegate_task returns immediately — "
        "you and the user keep working, and each subagent's full result "
        "re-enters the conversation as its own new message when it finishes. A "
        "batch is just N independent background subagents (N handles, each "
        "completes on its own). Do NOT wait or poll; just continue with other "
        "work after dispatching.\n\n"
        "WHEN TO USE delegate_task:\n"
        "- Reasoning-heavy subtasks (debugging, code review, research synthesis)\n"
        "- Tasks that would flood your context with intermediate data\n"
        "- Parallel independent workstreams (research A and B simultaneously)\n\n"
        "WHEN NOT TO USE (use these instead):\n"
        "- Mechanical multi-step work with no reasoning needed -> use execute_code\n"
        "- Single tool call -> just call the tool directly\n"
        "- Tasks needing user interaction -> subagents cannot use clarify\n"
        "- Durable long-running work that must outlive the current turn -> "
        "use cronjob (action='create') or terminal(background=True, "
        "notify_on_complete=True) instead. Background delegations are NOT "
        "durable: if the parent session is closed (/new) or the process exits "
        "before a subagent finishes, that subagent's work is discarded, and "
        "/stop cancels every running background subagent.\n\n"
        "IMPORTANT:\n"
        "- Subagents have NO memory of your conversation. Pass all relevant "
        "info (file paths, error messages, constraints) via the 'context' field.\n"
        "- If the user is writing in a non-English language, or asked for "
        "output in a specific language / tone / style, say so in 'context' "
        "(e.g. \"respond in Chinese\", \"return output in Japanese\"). "
        "Otherwise subagents default to English and their summaries will "
        "contaminate your final reply with the wrong language.\n"
        "- Subagent summaries are SELF-REPORTS, not verified facts. A subagent "
        "that claims \"uploaded successfully\" or \"file written\" may be wrong. "
        "For operations with external side-effects (HTTP POST/PUT, remote "
        "writes, file creation at shared paths, publishing), require the "
        "subagent to return a verifiable handle (URL, ID, absolute path, HTTP "
        "status) and verify it yourself — fetch the URL, stat the file, read "
        "back the content — before telling the user the operation succeeded.\n"
        "- Leaf subagents (role='leaf', the default) CANNOT call: "
        "delegate_task, clarify, memory, send_message, execute_code.\n"
        "- Orchestrator subagents (role='orchestrator') retain "
        "delegate_task so they can spawn their own workers, but still "
        "cannot use clarify, memory, send_message, or execute_code. "
        f"Orchestrators are bounded by max_spawn_depth={max_depth} for this "
        f"user and can be disabled globally via "
        "delegation.orchestrator_enabled=false.\n"
        "- Subagent model is NOT selectable per call: children inherit the parent model (plus its fallback chain) unless you pin all subagents to a model via delegation.provider / delegation.model in config.yaml.\n"
        "- Each subagent gets its own terminal session (separate working directory and state).\n"
        "- Results are always returned as an array, one entry per task."
    )


def _build_tasks_param_description() -> str:
    """Compose the 'tasks' parameter description with current concurrency limit."""
    try:
        max_children = _get_max_concurrent_children()
    except Exception:
        max_children = _DEFAULT_MAX_CONCURRENT_CHILDREN
    return (
        f"Batch mode: tasks to run in parallel (up to {max_children} for this "
        f"user, set via delegation.max_concurrent_children). Each gets "
        "its own subagent with isolated context and terminal session. "
        "When provided, top-level goal/context/toolsets are ignored."
    )


def _build_role_param_description() -> str:
    """Compose the 'role' parameter description with current spawn-depth limit."""
    try:
        max_depth = _get_max_spawn_depth()
    except Exception:
        max_depth = MAX_DEPTH
    try:
        orchestrator_on = _get_orchestrator_enabled()
    except Exception:
        orchestrator_on = True

    if max_depth >= 2 and orchestrator_on:
        nesting_note = (
            f"Nesting IS enabled for this user (max_spawn_depth={max_depth}): "
            f"orchestrator children can themselves delegate up to {max_depth - 1} "
            "more level(s) deep."
        )
    elif max_depth >= 2 and not orchestrator_on:
        nesting_note = (
            "Nesting is currently disabled "
            "(delegation.orchestrator_enabled=false); 'orchestrator' is "
            "silently forced to 'leaf'."
        )
    else:
        nesting_note = (
            f"Nesting is OFF for this user (max_spawn_depth={max_depth}); "
            "'orchestrator' is silently forced to 'leaf'. Raise "
            "delegation.max_spawn_depth in config.yaml to enable."
        )

    return (
        "Role of the child agent. 'leaf' (default) = focused "
        "worker, cannot delegate further. 'orchestrator' = can "
        f"use delegate_task to spawn its own workers. {nesting_note}"
    )


def _build_dynamic_schema_overrides() -> dict:
    """Return per-call schema overrides reflecting current config.

    Plugged into ToolEntry.dynamic_schema_overrides so every
    get_definitions() pass rewrites the description fields to the user's
    actual limits.
    """
    overrides_params = {
        **DELEGATE_TASK_SCHEMA["parameters"],
    }
    # Deep-copy properties so we don't mutate the static schema dict.
    overrides_params["properties"] = {
        k: dict(v) for k, v in DELEGATE_TASK_SCHEMA["parameters"]["properties"].items()
    }
    overrides_params["properties"]["tasks"]["description"] = _build_tasks_param_description()
    overrides_params["properties"]["role"]["description"] = _build_role_param_description()

    return {
        "description": _build_top_level_description(),
        "parameters": overrides_params,
    }


DELEGATE_TASK_SCHEMA = {
    "name": "delegate_task",
    # NOTE: description / tasks.description / role.description are placeholder
    # values. The real text is generated per get_definitions() call by
    # _build_dynamic_schema_overrides() (registered via
    # dynamic_schema_overrides below) so the model sees the user's actual
    # delegation.max_concurrent_children / max_spawn_depth, not the framework
    # defaults. Building these lazily (instead of at module import) also
    # avoids forcing cli.CLI_CONFIG to load before the test conftest can
    # redirect HERMES_HOME.
    "description": (
        "Spawn one or more subagents in isolated contexts. "
        "Description is rebuilt at every get_definitions() call to reflect "
        "the user's current delegation limits."
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
                "description": "(rebuilt at get_definitions() time)",
            },
            "role": {
                "type": "string",
                "enum": ["leaf", "orchestrator"],
                "description": "(rebuilt at get_definitions() time)",
            },
            "background": {
                "type": "boolean",
                "description": (
                    "DEPRECATED / IGNORED. Single-task delegations always run "
                    "in the background automatically — you do not need to (and "
                    "cannot) opt in or out. The result re-enters the "
                    "conversation as a new message when the subagent finishes; "
                    "just continue working in the meantime. Setting this has no "
                    "effect; the parameter remains only for backward "
                    "compatibility."
                ),
            },
        },
        "required": [],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error


def _model_background_value(args: dict, parent_agent=None) -> bool:
    """Background flag for the MODEL-facing dispatch path (registry fallback).

    Delegations from the top-level agent always run in the background — the
    model does not choose. This applies to both a single task and a fan-out
    batch (each task becomes its own independent background subagent). The one
    exception is a delegation from an orchestrator subagent (depth > 0), which
    needs its workers' results within its own turn. The live path is
    ``run_agent._dispatch_delegate_task``; this lambda mirrors it for the rare
    case the intercept is bypassed. Direct Python callers of ``delegate_task``
    keep the historical synchronous default.
    """
    is_subagent = getattr(parent_agent, "_delegate_depth", 0) > 0
    return not is_subagent


_MODEL_HIDDEN_TASK_FIELDS = {"acp_command", "acp_args"}


def _strip_model_hidden_task_fields(tasks: Any) -> Any:
    if not isinstance(tasks, list):
        return tasks
    stripped_tasks = []
    changed = False
    for task in tasks:
        if not isinstance(task, dict):
            stripped_tasks.append(task)
            continue
        stripped = {
            key: value
            for key, value in task.items()
            if key not in _MODEL_HIDDEN_TASK_FIELDS
        }
        changed = changed or len(stripped) != len(task)
        stripped_tasks.append(stripped)
    return stripped_tasks if changed else tasks


registry.register(
    name="delegate_task",
    toolset="delegation",
    schema=DELEGATE_TASK_SCHEMA,
    handler=lambda args, **kw: delegate_task(
        goal=args.get("goal"),
        context=args.get("context"),
        tasks=_strip_model_hidden_task_fields(args.get("tasks")),
        max_iterations=args.get("max_iterations"),
        role=args.get("role"),
        background=_model_background_value(args, kw.get("parent_agent")),
        parent_agent=kw.get("parent_agent"),
    ),
    check_fn=check_delegate_requirements,
    emoji="🔀",
    dynamic_schema_overrides=_build_dynamic_schema_overrides,
)
