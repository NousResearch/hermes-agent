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
    ]
)

# Global subagent blocked toolsets — always stripped, cannot be re-enabled
# by caller or profile.
GLOBAL_SUBAGENT_BLOCKED_TOOLSETS = frozenset({"desktop"})

# Global subagent blocked tools — always blocked at the single-tool level.
GLOBAL_SUBAGENT_BLOCKED_TOOLS = frozenset({
    "send_message",
    "memory",
    "execute_code",
})


# ---------------------------------------------------------------------------
# Phase B: Write tool classification for readonly isolation
# ---------------------------------------------------------------------------

# Toolsets that contain write-capable tools and should be stripped entirely
# in readonly mode. "terminal" is the primary one — it lets subagents execute
# arbitrary commands including file modifications.
READONLY_STRIP_TOOLSETS = frozenset({"terminal"})

# Individual tools that are write-capable. Stripped from their parent toolsets
# in readonly mode even when the toolset itself is not entirely stripped.
# Tool names match the Hermes registry from toolsets.py.
READONLY_STRIP_TOOLS = frozenset({
    "write_file",
    "patch",
    "terminal",
    "process",
})


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
    # Phase C+: clean up message queue and background result on unregister
    with _subagent_message_lock:
        _subagent_message_queues.pop(subagent_id, None)


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


# ---------------------------------------------------------------------------
# Phase C+: Background execution
# ---------------------------------------------------------------------------

_background_executor = None
_background_executor_lock = threading.Lock()

# Completed background subagent results, keyed by subagent_id.
# Entries are full result dicts from _run_single_child.
_background_results: Dict[str, Dict[str, Any]] = {}
_background_results_lock = threading.Lock()


def _get_background_executor():
    """Return (or create) a long-lived thread pool for background subagents."""
    global _background_executor
    if _background_executor is not None:
        return _background_executor
    with _background_executor_lock:
        if _background_executor is not None:
            return _background_executor
        _background_executor = ThreadPoolExecutor(
            max_workers=10,
            thread_name_prefix="hermes-bg-subagent",
        )
        return _background_executor


def get_subagent_result(subagent_id: str) -> Optional[Dict[str, Any]]:
    """Return the completed result for a background subagent, or None.

    Once retrieved, the result is removed from storage. Returns None
    if the subagent is still running or the ID is unknown.
    """
    # Check if still running (only need _active_subagents_lock)
    with _active_subagents_lock:
        if subagent_id in _active_subagents:
            return {"status": "running", "subagent_id": subagent_id}
    # Retrieve and consume completed result
    with _background_results_lock:
        return _background_results.pop(subagent_id, None)


# ---------------------------------------------------------------------------
# Phase C+: send_message infrastructure
# ---------------------------------------------------------------------------

# Per-subagent message queues. Messages are stored as {timestamp, sender, content}.
_subagent_message_queues: Dict[str, List[Dict[str, Any]]] = {}
_subagent_message_lock = threading.Lock()


def send_message(subagent_id: str, message: str, sender: str = "parent") -> bool:
    """Queue a message for delivery to a running subagent.

    The message is stored and can be retrieved via get_subagent_messages().
    Returns True if the subagent was found and the message was queued.
    Returns False if the subagent is not running.

    IMPORTANT: Actual injection of messages into the running LLM conversation
    requires agent-loop modifications (not yet implemented). Messages are
    stored and retrievable, but are NOT automatically injected into the
    subagent's next turn. Use get_subagent_messages() to poll.
    """
    with _active_subagents_lock:
        record = _active_subagents.get(subagent_id)
    if not record:
        return False

    with _subagent_message_lock:
        if subagent_id not in _subagent_message_queues:
            _subagent_message_queues[subagent_id] = []
        _subagent_message_queues[subagent_id].append({
            "timestamp": time.time(),
            "sender": sender,
            "content": message,
        })
    return True


def get_subagent_messages(
    subagent_id: str,
    *,
    mark_read: bool = True,
) -> Optional[List[Dict[str, Any]]]:
    """Retrieve queued messages for a subagent.

    Args:
        subagent_id: The subagent to query.
        mark_read: If True, clears the queue after reading (default).

    Returns the list of messages, or None if the subagent is unknown.
    """
    with _active_subagents_lock:
        if subagent_id not in _active_subagents:
            return None
    with _subagent_message_lock:
        messages = list(_subagent_message_queues.get(subagent_id, []))
        if mark_read:
            _subagent_message_queues[subagent_id] = []
    return messages


def _deliver_pending_messages(
    subagent_id: str,
    child_agent=None,
) -> List[Dict[str, Any]]:
    """Deliver pending messages to a subagent by appending them to its
    conversation as synthetic user messages.

    This is called by the heartbeat thread or progress callback. Returns
    the list of delivered messages.

    NOTE: This uses child._conversation_history (internal API) to inject
    messages between turns. This is a pragmatic bridge until full agent-loop
    support for subagent messaging is implemented.
    """
    with _subagent_message_lock:
        messages = list(_subagent_message_queues.pop(subagent_id, []))
    if not messages:
        return []

    if child_agent is None:
        return messages  # can't deliver without agent ref

    try:
        for msg in messages:
            synthetic = {
                "role": "user",
                "content": (
                    f"[Message from {msg['sender']} at "
                    f"{time.strftime('%H:%M:%S', time.localtime(msg['timestamp']))}]\n"
                    f"{msg['content']}"
                ),
            }
            history = getattr(child_agent, "_conversation_history", None)
            if isinstance(history, list):
                history.append(synthetic)
    except Exception as exc:
        logger.warning("Failed to deliver pending messages to %s: %s", subagent_id, exc)

    return messages


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
            result = max(1, int(val))
            if result > 10:
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


def _get_child_timeout() -> float:
    """Read delegation.child_timeout_seconds from config.

    Returns the number of seconds a single child agent is allowed to run
    before being considered stuck.  Default: 600 s (10 minutes).
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
DEFAULT_CHILD_TIMEOUT = 600  # seconds before a child agent is considered stuck
_HEARTBEAT_INTERVAL = 30  # seconds between parent activity heartbeats during delegation
# Stale-heartbeat thresholds. A child with no API-call progress is either:
#   - idle between turns (no current_tool) — probably stuck on a slow API call
#   - inside a tool (current_tool set) — probably running a legitimately long
#     operation (terminal command, web fetch, large file read)
# The idle ceiling stays tight so genuinely stuck children don't mask the gateway
# timeout. The in-tool ceiling is much higher so legit long-running tools get
# time to finish; child_timeout_seconds (default 600s) is still the hard cap.
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
    skills: Optional[List[str]] = None,
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
    if skills:
        parts.append(
            "## Scoped Skills\n"
            "The following skills are available to you:\n"
            + "\n".join(f"- {s}" for s in skills)
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


# ---------------------------------------------------------------------------
# Phase A: Registry / profile helpers
# ---------------------------------------------------------------------------


def _load_agent_registry() -> dict:
    """Load agent registry from ~/.hermes/config/agent-registry.json."""
    from pathlib import Path

    try:
        from hermes_constants import get_hermes_home
    except Exception:
        def get_hermes_home() -> Path:
            return Path.home() / ".hermes"

    registry_path = get_hermes_home() / "config" / "agent-registry.json"
    with open(registry_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_subagent_profile(agent_id: str) -> tuple:
    """Return (agent_config, subagent_profile) for a named agent.

    Raises:
        ValueError: agent_id not found or missing subagent_profile.
    """
    registry = _load_agent_registry()
    agents = registry.get("agents", {})
    agent_config = agents.get(agent_id)
    if agent_config is None:
        raise ValueError(
            f"Agent '{agent_id}' not found in agent-registry.json. "
            f"Available agents: {list(agents.keys())}"
        )
    profile = agent_config.get("subagent_profile")
    if not profile:
        raise ValueError(
            f"Agent '{agent_id}' has no subagent_profile in agent-registry.json. "
            f"Add a 'subagent_profile' field to enable delegate_task(agent_id=...)."
        )
    return agent_config, profile


def _desktop_allowed(agent_config: dict, profile: dict) -> bool:
    """Return True only when both conditions are met:

    1. agent_config.capabilities contains 'desktop_control'.
    2. profile.toolsets explicitly includes 'desktop'.
    """
    caps = set(agent_config.get("capabilities") or [])
    tools = set(profile.get("toolsets") or [])
    return "desktop_control" in caps and "desktop" in tools


def _should_inherit_mcp_toolsets_for_profile(profile: dict) -> bool:
    """Return True when the profile explicitly opts into MCP inheritance."""
    return bool(profile.get("inherit_mcp_toolsets") is True)


def _resolve_effective_toolsets(
    *,
    agent_config: dict,
    profile: dict,
    requested_toolsets: Optional[List[str]],
    parent_toolsets: set,
) -> List[str]:
    """Compute effective toolsets for a named subagent.

    Rules (agent_id path):
    1. Start with profile.toolsets.
    2. Intersect with requested_toolsets if provided.
    3. Intersect with parent_toolsets (never gain tools parent lacks).
    4. Strip desktop unless _desktop_allowed(…).
    5. Strip GLOBAL_SUBAGENT_BLOCKED_TOOLSETS (except desktop, already handled).
    6. Default-deny MCP toolsets (caller must re-add if profile opts in).
    """
    base = set(profile.get("toolsets") or [])

    if requested_toolsets:
        base &= set(requested_toolsets)

    base &= parent_toolsets

    if not _desktop_allowed(agent_config, profile):
        base.discard("desktop")

    # Strip globally blocked toolsets (desktop already handled above)
    for ts in GLOBAL_SUBAGENT_BLOCKED_TOOLSETS:
        if ts != "desktop":
            base.discard(ts)

    # Default: no MCP inheritance for agent_id path.
    # Caller adds back MCP toolsets only when the profile explicitly opts in.
    return sorted(base)


def _resolve_effective_blocked_tools(
    *,
    profile: dict,
    requested_blocked_tools: Optional[List[str]] = None,
) -> set:
    """Compute effective blocked tools merging all sources.

    Merge order (union only — blocked_tools can only grow):
    1. DELEGATE_BLOCKED_TOOLS
    2. GLOBAL_SUBAGENT_BLOCKED_TOOLS
    3. profile.blocked_tools
    4. caller-supplied blocked_tools

    Caller blocked_tools can only add, never subtract.
    """
    blocked: set = set(DELEGATE_BLOCKED_TOOLS)
    blocked |= GLOBAL_SUBAGENT_BLOCKED_TOOLS
    blocked |= set(profile.get("blocked_tools") or [])
    if requested_blocked_tools:
        blocked |= set(requested_blocked_tools)
    return blocked


def _check_mcp_server_availability(profile: dict, warnings: List[str]) -> None:
    """Check that required_mcp_servers are available.

    This is an informational check only — it does NOT auto-grant MCP tool
    permissions. Toolsets must be explicitly listed in profile.toolsets or
    inherit_mcp_toolsets must be true.
    """
    required = profile.get("required_mcp_servers") or []
    if not required:
        return
    try:
        from tools.registry import registry
        available_servers = {
            name.replace("mcp-", "", 1)
            for name in registry.get_registered_toolset_names()
            if name.startswith("mcp-")
        }
    except Exception:
        available_servers = set()
    for server in required:
        if server not in available_servers:
            msg = (
                f"Required MCP server '{server}' is not available. "
                f"Subagent may lack expected tools."
            )
            logger.warning(msg)
            warnings.append(msg)


# ---------------------------------------------------------------------------
# Phase B: Isolation resolution
# ---------------------------------------------------------------------------


def _resolve_isolation(
    *,
    requested_isolation: Optional[str],
    profile_isolation: Optional[str],
    permission_mode: Optional[str],
) -> tuple:
    """Resolve effective isolation mode and return (isolation, warnings, error).

    Returns:
        (isolation: str, warnings: list, error: str or None)
        If error is not None, the caller must return a tool_error immediately.
    """
    warnings: List[str] = []

    # Determine effective isolation: requested > profile default > "shared"
    effective = requested_isolation or profile_isolation or "shared"

    # read_only permission_mode auto-enables readonly isolation
    # Only when the caller did not explicitly request a different isolation.
    if (
        permission_mode == "read_only"
        and effective != "readonly"
        and requested_isolation is None
    ):
        effective = "readonly"
        warnings.append(
            "permission_mode='read_only' automatically enables isolation='readonly'"
        )

    # worktree isolation (Phase C)
    if effective == "worktree":
        return "worktree", warnings, None

    # Validate isolation value
    if effective not in ("shared", "readonly"):
        return (
            "shared",
            warnings,
            f"Unknown isolation mode '{effective}'; valid values: shared, readonly, worktree.",
        )

    return effective, warnings, None


def _apply_readonly_isolation(toolsets: List[str]) -> List[str]:
    """Strip write-capable toolsets and individual tools for readonly mode.

    - Entire toolsets in READONLY_STRIP_TOOLSETS are removed.
    - Individual tools in READONLY_STRIP_TOOLS are removed from their
      parent toolsets.
    """
    # Remove entirely-stripped toolsets
    filtered = [ts for ts in toolsets if ts not in READONLY_STRIP_TOOLSETS]

    # For remaining toolsets, strip individual write tools
    # We don't resolve individual tools here — we just pass through the
    # toolset names. The actual tool filtering happens when tools are
    # resolved downstream via model_tools.get_tool_definitions().
    # We add blocked_tools entries to ensure write tools are denied.
    return filtered


# ---------------------------------------------------------------------------
# Phase B: Subagent status/output interfaces (read-only)
# ---------------------------------------------------------------------------


def get_subagent_status(subagent_id: str) -> Optional[Dict[str, Any]]:
    """Return a snapshot of a subagent's current state, or None if not found.

    Safe to call from any thread. Returns a copy of the record without
    the raw agent reference.
    """
    with _active_subagents_lock:
        record = _active_subagents.get(subagent_id)
    if not record:
        return None
    return {
        "subagent_id": record.get("subagent_id"),
        "parent_id": record.get("parent_id"),
        "depth": record.get("depth"),
        "goal": record.get("goal"),
        "model": record.get("model"),
        "started_at": record.get("started_at"),
        "status": record.get("status"),
        "tool_count": record.get("tool_count"),
        "last_tool": record.get("last_tool"),
    }


def get_subagent_output_tail(
    subagent_id: str,
    lines: int = 50,
) -> Optional[List[Dict[str, Any]]]:
    """Return recent tool-call output entries for a subagent, or None.

    This relies on the parent having stored the child's results. For
    running subagents, returns the tail from the child's conversation
    messages if the child exposes them.

    Args:
        subagent_id: The subagent to query.
        lines: Maximum number of output entries to return.
    """
    with _active_subagents_lock:
        record = _active_subagents.get(subagent_id)
    if not record:
        return None

    agent = record.get("agent")
    if agent is None:
        return None

    # Try to get conversation messages from the running child
    try:
        messages = getattr(agent, "_conversation_messages", None) or []
        if not messages:
            return []
        tail = messages[-min(lines, len(messages)):]
        return [
            {
                "role": m.get("role") if isinstance(m, dict) else "unknown",
                "content_preview": (
                    str(m.get("content", ""))[:200]
                    if isinstance(m, dict)
                    else str(m)[:200]
                ),
            }
            for m in tail
        ]
    except Exception:
        return None


def get_subagent_usage(subagent_id: str) -> Optional[Dict[str, Any]]:
    """Return token/cost usage for a running subagent, or None."""
    with _active_subagents_lock:
        record = _active_subagents.get(subagent_id)
    if not record:
        return None
    agent = record.get("agent")
    if agent is None:
        return None
    try:
        return {
            "input_tokens": getattr(agent, "session_prompt_tokens", 0) or 0,
            "output_tokens": getattr(agent, "session_completion_tokens", 0) or 0,
            "api_calls": getattr(agent, "api_call_count", 0) or 0,
            "cost_usd": getattr(agent, "session_estimated_cost_usd", 0.0) or 0.0,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Phase C: Worktree isolation
# ---------------------------------------------------------------------------

_WORKTREE_BASE_DIR = None  # Path | None


def _get_worktree_base_dir():
    """Return the base directory for worktrees (~/.hermes/worktrees)."""
    global _WORKTREE_BASE_DIR
    if _WORKTREE_BASE_DIR is not None:
        return _WORKTREE_BASE_DIR
    try:
        from hermes_constants import get_hermes_home
    except Exception:
        from pathlib import Path as _P
        def get_hermes_home() -> _P:
            return _P.home() / ".hermes"
    _WORKTREE_BASE_DIR = get_hermes_home() / "worktrees"
    return _WORKTREE_BASE_DIR


def _create_worktree(
    subagent_id: str,
    *,
    parent_agent=None,
) -> Dict[str, Any]:
    """Create a git worktree for an isolated subagent.

    Returns dict with: {worktree_path, worktree_branch, original_head, repo_path, error}
    If error is non-empty, the worktree could not be created.
    """
    import subprocess as _sp
    from pathlib import Path as _Path

    result: Dict[str, Any] = {
        "worktree_path": None,
        "worktree_branch": None,
        "original_head": None,
        "repo_path": None,
        "error": None,
    }

    # Find the parent's git repo — prefer TERMINAL_CWD, then parent's workspace hint
    repo_candidate = None
    try:
        repo_candidate = os.environ.get("TERMINAL_CWD") or os.getcwd()
    except Exception:
        repo_candidate = os.getcwd()

    # Verify this is a git repo
    try:
        proc = _sp.run(
            ["git", "-C", str(repo_candidate), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode != 0:
            result["error"] = f"Cannot find git repository at {repo_candidate}"
            return result
        repo_path = _Path(proc.stdout.strip())
    except (FileNotFoundError, _sp.TimeoutExpired) as exc:
        result["error"] = f"Git command failed: {exc}"
        return result

    # Get current HEAD
    try:
        proc = _sp.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode != 0:
            result["error"] = "Cannot resolve HEAD"
            return result
        original_head = proc.stdout.strip()
    except Exception as exc:
        result["error"] = f"Failed to get HEAD: {exc}"
        return result

    # Create the worktree
    safe_id = subagent_id.replace("/", "_").replace("\\", "_")
    worktree_name = f"hermes-{safe_id}"
    worktree_base = _get_worktree_base_dir()
    try:
        worktree_base.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        result["error"] = f"Cannot create worktree base dir: {exc}"
        return result
    worktree_path = worktree_base / worktree_name

    try:
        proc = _sp.run(
            ["git", "-C", str(repo_path), "worktree", "add", "--detach",
             str(worktree_path), original_head],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode != 0:
            result["error"] = f"git worktree add failed: {proc.stderr.strip()}"
            return result
    except Exception as exc:
        result["error"] = f"git worktree add failed: {exc}"
        return result

    result["worktree_path"] = str(worktree_path)
    result["worktree_branch"] = worktree_name
    result["original_head"] = original_head
    result["repo_path"] = str(repo_path)
    return result


def _cleanup_worktree(
    worktree_path: str,
    original_head: str,
    *,
    force: bool = False,
) -> Dict[str, Any]:
    """Clean up a git worktree. Fail-closed: only remove when safe.

    Returns dict with: {removed, kept, reason, diff_summary}
    """
    import subprocess as _sp
    from pathlib import Path as _Path

    wpath = _Path(worktree_path)
    result: Dict[str, Any] = {
        "removed": False,
        "kept": True,
        "reason": "",
        "diff_summary": "",
    }

    if not wpath.exists():
        result["reason"] = "worktree path does not exist"
        result["kept"] = False
        return result

    if not str(wpath).startswith(str(_get_worktree_base_dir())):
        result["reason"] = (
            "worktree path is not under Hermes managed directory — refusing to delete"
        )
        return result

    # Check for changes
    try:
        proc = _sp.run(
            ["git", "-C", str(wpath), "status", "--porcelain"],
            capture_output=True, text=True, timeout=15,
        )
        if proc.returncode != 0:
            result["reason"] = (
                f"git status failed (exit {proc.returncode}): "
                f"{proc.stderr.strip()[:200]}"
            )
            return result
        dirty = proc.stdout.strip()
    except FileNotFoundError:
        result["reason"] = "git not found"
        return result
    except _sp.TimeoutExpired:
        result["reason"] = "git status timed out"
        return result

    # Check for .git lock file
    lock_file = wpath / ".git" / "index.lock"
    if lock_file.exists():
        result["reason"] = ".git/index.lock exists — refusing to delete"
        return result

    if dirty and not force:
        try:
            diff_proc = _sp.run(
                ["git", "-C", str(wpath), "diff", "--stat"],
                capture_output=True, text=True, timeout=10,
            )
            result["diff_summary"] = diff_proc.stdout.strip()[:500]
        except Exception:
            result["diff_summary"] = "(diff unavailable)"
        result["reason"] = (
            f"Worktree has uncommitted changes — kept at {worktree_path}"
        )
        return result

    try:
        # git worktree remove must run from within the main repo.
        # The worktree's .git file points to the main repo, so git can
        # resolve the main repo from the worktree path itself without -C.
        _sp.run(
            ["git", "worktree", "remove", "--force", str(wpath)],
            capture_output=True, text=True, timeout=15,
            check=True,
        )
        result["removed"] = True
        result["kept"] = False
        result["reason"] = "worktree removed (no changes)"
    except _sp.CalledProcessError as exc:
        result["reason"] = f"git worktree remove failed: {exc.stderr.strip()[:200]}"
    except Exception as exc:
        result["reason"] = f"cleanup failed: {exc}"

    return result


# ---------------------------------------------------------------------------
# Phase C: Transcript JSONL
# ---------------------------------------------------------------------------


def _get_transcript_dir():
    """Return the base directory for subagent transcripts."""
    try:
        from hermes_constants import get_hermes_home
    except Exception:
        from pathlib import Path as _P
        def get_hermes_home() -> _P:
            return _P.home() / ".hermes"
    return get_hermes_home() / "subagents"


def _write_transcript_event(
    session_id: str,
    subagent_id: str,
    event_type: str,
    *,
    tool: Optional[str] = None,
    preview: str = "",
    usage: Optional[Dict[str, Any]] = None,
    agent_id: Optional[str] = None,
):
    """Append a single JSONL line to the subagent's transcript.

    Returns the transcript file path, or None on failure.
    """
    try:
        transcript_dir = _get_transcript_dir() / session_id
        transcript_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = transcript_dir / f"{subagent_id}.jsonl"

        entry = {
            "timestamp": time.time(),
            "event_type": event_type,
            "subagent_id": subagent_id,
            "agent_id": agent_id,
            "tool": tool,
            "preview": preview[:500],
            "usage": usage or {},
        }
        with open(transcript_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return transcript_path
    except Exception as exc:
        logger.warning("Failed to write transcript event: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Phase C: Coordinator / Swarm prep
# ---------------------------------------------------------------------------


# Tools allowed for coordinator_mode agents (orchestration only)
COORDINATOR_ALLOWED_TOOLSETS = frozenset({"delegation", "file"})
COORDINATOR_BLOCKED_TOOLS = frozenset({
    "terminal", "process", "write_file", "patch",
    "send_message", "memory", "execute_code",
    "browser_navigate", "browser_snapshot", "browser_click",
})


def _apply_coordinator_mode(
    agent_config: dict,
    toolsets: List[str],
    blocked_tools: set,
    warnings: List[str],
) -> tuple:
    """Restrict tools for coordinator_mode agents.

    When agent_config has coordinator_mode: true, only orchestration
    toolsets are allowed. All write/mutate tools are blocked.

    Returns (filtered_toolsets, updated_blocked_tools, updated_warnings).
    """
    coord_mode = agent_config.get("coordinator_mode", False)
    if not coord_mode:
        return toolsets, blocked_tools, warnings

    warnings.append("coordinator_mode: restricting to orchestration toolsets")
    filtered = [ts for ts in toolsets if ts in COORDINATOR_ALLOWED_TOOLSETS]
    new_blocked = set(blocked_tools) | COORDINATOR_BLOCKED_TOOLS
    return filtered, new_blocked, warnings


def claim_task(task_id: str, agent_id: str) -> bool:
    """Atomically claim a task using SQLite WAL INSERT OR IGNORE.

    Returns True if the claim was successful (first to claim).
    Returns False if the task was already claimed by another agent.

    Uses a simple SQLite database at ~/.hermes/swarm_claims.db.
    Thread-safe via WAL mode.

    IMPORTANT: This guarantees atomicity only on local filesystems
    (APFS, ext4, NTFS). For distributed coordination, a proper lock
    service (etcd, Redis Redlock, PostgreSQL advisory lock) is needed.
    """
    try:
        from hermes_constants import get_hermes_home
    except Exception:
        from pathlib import Path as _P
        def get_hermes_home() -> _P:
            return _P.home() / ".hermes"

    db_path = get_hermes_home() / "swarm_claims.db"
    try:
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS claims ("
            "  task_id TEXT PRIMARY KEY,"
            "  agent_id TEXT NOT NULL,"
            "  claimed_at REAL NOT NULL"
            ")"
        )
        conn.execute(
            "INSERT OR IGNORE INTO claims (task_id, agent_id, claimed_at) "
            "VALUES (?, ?, ?)",
            (task_id, agent_id, time.time()),
        )
        conn.commit()
        # Check if we got the claim
        row = conn.execute(
            "SELECT agent_id FROM claims WHERE task_id = ?", (task_id,)
        ).fetchone()
        conn.close()
        return row is not None and row[0] == agent_id
    except Exception as exc:
        logger.warning("claim_task failed: %s", exc)
        return False


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
    # Phase A: named subagent via agent_id
    agent_id: Optional[str] = None,
    agent_config: Optional[dict] = None,
    profile: Optional[dict] = None,
    requested_blocked_tools: Optional[List[str]] = None,
    # Phase B: isolation
    requested_isolation: Optional[str] = None,
    profile_isolation: Optional[str] = None,
    profile_permission_mode: Optional[str] = None,
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

    # ── Toolset resolution ────────────────────────────────────────────────
    # Two paths:
    #   agent_id path — profile-driven, tools can only narrow, desktop default-deny
    #   legacy path   — existing role-based inheritance (unchanged behavior)

    parent_enabled = getattr(parent_agent, "enabled_toolsets", None)
    if parent_enabled is not None:
        parent_toolsets = set(parent_enabled)
    elif parent_agent and hasattr(parent_agent, "valid_tool_names"):
        import model_tools

        parent_toolsets = {
            ts
            for name in parent_agent.valid_tool_names
            if (ts := model_tools.get_toolset_for_tool(name)) is not None
        }
    else:
        parent_toolsets = set(DEFAULT_TOOLSETS)

    effective_blocked: set = set()
    warnings: List[str] = []

    if agent_id and agent_config and profile:
        # ── agent_id path ─────────────────────────────────────────────
        child_toolsets = _resolve_effective_toolsets(
            agent_config=agent_config,
            profile=profile,
            requested_toolsets=toolsets,
            parent_toolsets=parent_toolsets,
        )

        # MCP inheritance: only when profile explicitly opts in
        if _should_inherit_mcp_toolsets_for_profile(profile):
            child_toolsets = _preserve_parent_mcp_toolsets(
                child_toolsets, parent_toolsets
            )
        # Note: _resolve_effective_toolsets already defaults to no MCP toolsets.
        # required_mcp_servers is only checked for availability, not for granting
        # tool permissions. See _check_mcp_server_availability below.

        # Check MCP server availability (informational only, not auto-granted)
        _check_mcp_server_availability(profile, warnings)

        effective_blocked = _resolve_effective_blocked_tools(
            profile=profile,
            requested_blocked_tools=requested_blocked_tools,
        )
    else:
        # ── legacy role path ──────────────────────────────────────────
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

    # ── Skill scoping (方案 A) ────────────────────────────────────────
    child_skills: list[str] = []
    if agent_id and profile:
        child_skills = list(profile.get("skills") or [])
    if child_skills and "skills" not in child_toolsets:
        child_toolsets = list(child_toolsets) + ["skills"]

    # Orchestrators retain the 'delegation' toolset that _strip_blocked_tools
    # removed.  The re-add is unconditional on parent-toolset membership because
    # orchestrator capability is granted by role, not inherited — see the
    # test_intersection_preserves_delegation_bound test for the design rationale.
    if effective_role == "orchestrator" and "delegation" not in child_toolsets:
        child_toolsets.append("delegation")

    # ── Phase B: Isolation resolution ───────────────────────────────────
    _effective_isolation, _iso_warnings, _iso_error = _resolve_isolation(
        requested_isolation=requested_isolation,
        profile_isolation=profile_isolation,
        permission_mode=profile_permission_mode,
    )
    warnings.extend(_iso_warnings)
    # If the caller requested worktree isolation, that's a hard error
    if _iso_error:
        # We stash the error so delegate_task can return it cleanly.
        # _build_child_agent doesn't return errors directly, so we store it
        # as a sentinel on the child object.
        _isolation_error = _iso_error
    else:
        _isolation_error = None

    # Apply readonly isolation: strip write toolsets and individual write tools
    if _effective_isolation == "readonly":
        child_toolsets_before_readonly = list(child_toolsets)
        child_toolsets = _apply_readonly_isolation(child_toolsets)
        stripped_ts = set(child_toolsets_before_readonly) - set(child_toolsets)
        for ts in stripped_ts:
            effective_blocked.add(ts)
            warnings.append(f"readonly isolation: stripped toolset '{ts}'")

    # ── Phase C: Coordinator mode ──────────────────────────────────────
    if agent_config and agent_config.get("coordinator_mode"):
        child_toolsets, effective_blocked, warnings = _apply_coordinator_mode(
            agent_config, child_toolsets, effective_blocked, warnings,
        )

    workspace_hint = _resolve_workspace_hint(parent_agent)
    child_prompt = _build_child_system_prompt(
        goal,
        context,
        workspace_path=workspace_hint,
        role=effective_role,
        max_spawn_depth=max_spawn,
        child_depth=child_depth,
        skills=child_skills,
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
    if override_provider:
        child_providers_allowed = None
        child_providers_ignored = None
        child_providers_order = None
        child_provider_sort = None

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
        fallback_model=parent_fallback,
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
        providers_allowed=child_providers_allowed,
        providers_ignored=child_providers_ignored,
        providers_order=child_providers_order,
        provider_sort=child_provider_sort,
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
    # Phase A: stash agent_id profile metadata for event logging
    child._subagent_agent_id = agent_id
    child._subagent_effective_toolsets = child_toolsets
    child._subagent_effective_blocked = effective_blocked
    child._subagent_warnings = warnings
    child._subagent_isolation = _effective_isolation
    child._subagent_isolation_error = _isolation_error

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

        _w(f"# Subagent timeout diagnostic — issue #14726")
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
                _w(f"  loaded tools:      {sorted(list(tool_names))}")
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


def _try_log_subagent_event(
    event_type: str,
    *,
    subagent_id: Optional[str],
    parent_id: Optional[str],
    agent_id: Optional[str],
    role: str,
    task_id: Optional[str],
    session_id: Optional[str],
    goal_preview: str = "",
    effective_toolsets: Optional[List[str]] = None,
    blocked_tools: Optional[set] = None,
    status: str = "",
    error: str = "",
    reason: str = "",
    duration_seconds: float = 0.0,
    api_calls: int = 0,
    tokens: Optional[Dict[str, int]] = None,
    transcript_path: Optional[str] = None,
) -> None:
    """Fire a subagent lifecycle event, swallowing all failures.

    EventLog write failure must never interrupt the main task.
    """
    try:
        from agent.session_event_log import EventLog

        elog = EventLog()
        sid = task_id or ""
        sess = session_id or ""
        blocked_list = sorted(blocked_tools) if blocked_tools else []

        if event_type == "started":
            elog.log_subagent_started(
                task_id=sid,
                session_id=sess,
                subagent_id=subagent_id or "",
                goal_preview=goal_preview,
                parent_id=parent_id,
                agent_id=agent_id,
                role=role,
                effective_toolsets=effective_toolsets or [],
                blocked_tools=blocked_list,
            )
        elif event_type == "completed":
            elog.log_subagent_completed(
                task_id=sid,
                session_id=sess,
                subagent_id=subagent_id or "",
                status=status,
                parent_id=parent_id,
                agent_id=agent_id,
                role=role,
                duration_seconds=duration_seconds,
                api_calls=api_calls,
                tokens=tokens or {},
                transcript_path=transcript_path,
            )
        elif event_type == "failed":
            elog.log_subagent_failed(
                task_id=sid,
                session_id=sess,
                subagent_id=subagent_id or "",
                error=error,
                parent_id=parent_id,
                agent_id=agent_id,
                role=role,
            )
        elif event_type == "interrupted":
            elog.log_subagent_interrupted(
                task_id=sid,
                session_id=sess,
                subagent_id=subagent_id or "",
                reason=reason,
                parent_id=parent_id,
                agent_id=agent_id,
                role=role,
            )
        elif event_type == "backgrounded":
            # Phase C+: backgrounded has no dedicated EventLog method yet;
            # log as a completed-style event with status "backgrounded"
            elog.log_subagent_completed(
                task_id=sid,
                session_id=sess,
                subagent_id=subagent_id or "",
                status="backgrounded",
                parent_id=parent_id,
                agent_id=agent_id,
                role=role,
            )
        elog.close()
    except Exception as exc:
        logger.warning("Failed to write subagent %s event: %s", event_type, exc)


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
            # ── Phase C+: deliver pending messages ──────────────────
            if _subagent_id:
                try:
                    _deliver_pending_messages(_subagent_id, child_agent=child)
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

    # ── Phase A: subagent lifecycle event ─────────────────────────────
    _child_agent_id = getattr(child, "_subagent_agent_id", None)
    _child_role = getattr(child, "_delegate_role", "leaf")
    _child_parent_sid = getattr(child, "_parent_subagent_id", None)
    _parent_task_id = getattr(parent_agent, "_current_task_id", None)
    _parent_session_id = getattr(parent_agent, "session_id", None)
    # Sanitize values for JSON serialization (safe with MagicMock test doubles)
    _safe_subagent_id = _subagent_id if isinstance(_subagent_id, str) else None
    _safe_agent_id = _child_agent_id if isinstance(_child_agent_id, str) else None
    _safe_role = _child_role if isinstance(_child_role, str) else "leaf"
    _raw_toolsets = getattr(child, "_subagent_effective_toolsets", None)
    _safe_toolsets = list(_raw_toolsets) if isinstance(_raw_toolsets, list) else []
    _raw_blocked = getattr(child, "_subagent_effective_blocked", None)
    _safe_blocked = sorted(_raw_blocked) if isinstance(_raw_blocked, (set, list)) else []
    _raw_warnings = getattr(child, "_subagent_warnings", None)
    _child_warnings = list(_raw_warnings) if isinstance(_raw_warnings, list) else []
    _raw_isolation = getattr(child, "_subagent_isolation", None)
    _child_isolation = (
        _raw_isolation if isinstance(_raw_isolation, str) and _raw_isolation else "shared"
    )
    _try_log_subagent_event(
        "started",
        subagent_id=_safe_subagent_id,
        parent_id=_child_parent_sid if isinstance(_child_parent_sid, str) else None,
        agent_id=_safe_agent_id,
        role=_safe_role,
        task_id=_parent_task_id,
        session_id=_parent_session_id,
        goal_preview=goal,
        effective_toolsets=_safe_toolsets,
        blocked_tools=_safe_blocked,
    )

    # ── Phase C: Worktree / Transcript initialization ──────────────────
    _worktree_info: Dict[str, Any] = {}
    _cleanup_result: Dict[str, Any] = {}
    _transcript_path: Optional[str] = None
    _child_isolation_mode = getattr(child, "_subagent_isolation", "shared") or "shared"
    if _child_isolation_mode == "worktree":
        _wt_result = _create_worktree(
            subagent_id=_subagent_id or f"subagent-{task_index}",
            parent_agent=parent_agent,
        )
        if _wt_result.get("error"):
            # Worktree creation failed — return error as result
            _try_log_subagent_event(
                "failed",
                subagent_id=_safe_subagent_id,
                parent_id=_child_parent_sid if isinstance(_child_parent_sid, str) else None,
                agent_id=_safe_agent_id,
                role=_safe_role,
                task_id=_parent_task_id,
                session_id=_parent_session_id,
                error=_wt_result["error"],
            )
            return {
                "task_index": task_index,
                "status": "error",
                "summary": None,
                "subagent_id": _safe_subagent_id,
                "parent_id": _child_parent_sid if isinstance(_child_parent_sid, str) else None,
                "agent_id": _safe_agent_id,
                "role": _safe_role,
                "effective_toolsets": _safe_toolsets,
                "blocked_tools": _safe_blocked,
                "isolation": "worktree",
                "warnings": [_wt_result["error"]],
                "error": _wt_result["error"],
                "api_calls": 0,
                "duration_seconds": 0,
                "_child_role": _safe_role,
            }
        _worktree_info = _wt_result
        # Register worktree path as cwd override for terminal/file tools
        _wt_path = _wt_result.get("worktree_path")
        if _wt_path and _subagent_id:
            try:
                from tools.terminal_tool import register_task_env_overrides
                register_task_env_overrides(_subagent_id, {"cwd": _wt_path})
            except Exception as exc:
                logger.warning(
                    "Failed to register worktree cwd override for %s: %s",
                    _subagent_id, exc,
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
        _timeout_executor = ThreadPoolExecutor(
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

        def _run_with_thread_capture():
            _worker_thread_holder["t"] = threading.current_thread()
            return child.run_conversation(
                user_message=goal,
                task_id=child_task_id,
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
                    timeout_seconds=float(child_timeout),
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

            _try_log_subagent_event(
                "failed",
                subagent_id=_subagent_id,
                parent_id=_child_parent_sid if isinstance(_child_parent_sid, str) else None,
                agent_id=_child_agent_id,
                role=_child_role,
                task_id=_parent_task_id,
                session_id=_parent_session_id,
                error=_err,
            )
            return {
                "task_index": task_index,
                "status": "timeout" if is_timeout else "error",
                "summary": None,
                "subagent_id": _safe_subagent_id,
                "agent_id": _safe_agent_id,
                "role": _safe_role,
                "effective_toolsets": _safe_toolsets,
                "blocked_tools": _safe_blocked,
                "isolation": _child_isolation,
                "warnings": _child_warnings,
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
            "goal": (goal or "").strip(),
            "summary": summary,
            "subagent_id": _safe_subagent_id,
            "parent_id": _child_parent_sid if isinstance(_child_parent_sid, str) else None,
            "agent_id": _safe_agent_id,
            "role": _safe_role,
            "api_calls": api_calls,
            "duration_seconds": duration,
            "model": _model if isinstance(_model, str) else None,
            "exit_reason": exit_reason,
            "effective_toolsets": _safe_toolsets,
            "blocked_tools": _safe_blocked,
            "isolation": _child_isolation,
            "warnings": _child_warnings,
            "worktree_path": _worktree_info.get("worktree_path") if _child_isolation_mode == "worktree" else None,
            "worktree_branch": _worktree_info.get("worktree_branch") if _child_isolation_mode == "worktree" else None,
            "diff_summary": _cleanup_result.get("diff_summary", "") if _child_isolation_mode == "worktree" else None,
            "transcript_path": _transcript_path,
            "usage": {
                "input_tokens": (
                    _input_tokens if isinstance(_input_tokens, (int, float)) else 0
                ),
                "output_tokens": (
                    _output_tokens if isinstance(_output_tokens, (int, float)) else 0
                ),
                "api_calls": api_calls,
            },
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
        entry["output_tail"] = _output_tail

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

        # ── Phase C: Transcript recording ──────────────────────────────
        _transcript_path: Optional[str] = None
        if _parent_session_id and _subagent_id:
            _tw = _write_transcript_event(
                session_id=_parent_session_id,
                subagent_id=_subagent_id,
                event_type="final",
                tool=None,
                preview=summary[:500] if summary else entry.get("error", ""),
                usage={
                    "input_tokens": (
                        int(_input_tokens) if isinstance(_input_tokens, (int, float)) else 0
                    ),
                    "output_tokens": (
                        int(_output_tokens) if isinstance(_output_tokens, (int, float)) else 0
                    ),
                    "api_calls": int(api_calls) if isinstance(api_calls, (int, float)) else 0,
                },
                agent_id=_safe_agent_id,
            )
            if _tw:
                _transcript_path = str(_tw)

        # ── Phase A: subagent lifecycle event ─────────────────────────
        _child_api_calls = entry.get("api_calls", 0)
        _child_tokens = entry.get("tokens", {})
        if status == "interrupted":
            _try_log_subagent_event(
                "interrupted",
                subagent_id=_subagent_id,
                parent_id=_child_parent_sid if isinstance(_child_parent_sid, str) else None,
                agent_id=_child_agent_id,
                role=_child_role,
                task_id=_parent_task_id,
                session_id=_parent_session_id,
                reason=entry.get("error", "Interrupted"),
            )
        elif status == "failed":
            _try_log_subagent_event(
                "failed",
                subagent_id=_subagent_id,
                parent_id=_child_parent_sid if isinstance(_child_parent_sid, str) else None,
                agent_id=_child_agent_id,
                role=_child_role,
                task_id=_parent_task_id,
                session_id=_parent_session_id,
                error=entry.get("error", "Subagent did not produce a response."),
            )
        else:
            _try_log_subagent_event(
                "completed",
                subagent_id=_subagent_id,
                parent_id=_child_parent_sid if isinstance(_child_parent_sid, str) else None,
                agent_id=_safe_agent_id,
                role=_safe_role,
                task_id=_parent_task_id,
                session_id=_parent_session_id,
                status=status,
                duration_seconds=duration,
                api_calls=_child_api_calls,
                tokens=_child_tokens,
                transcript_path=_transcript_path,
            )

        return entry

    except Exception as exc:
        duration = round(time.monotonic() - child_start, 2)
        logging.exception(f"[subagent-{task_index}] failed")
        _try_log_subagent_event(
            "failed",
            subagent_id=_subagent_id,
            parent_id=_child_parent_sid if isinstance(_child_parent_sid, str) else None,
            agent_id=_child_agent_id,
            role=_child_role,
            task_id=_parent_task_id,
            session_id=_parent_session_id,
            error=str(exc),
        )
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
        # ── Phase C: Worktree cleanup ──────────────────────────────────
        _wt_path = _worktree_info.get("worktree_path")
        _wt_head = _worktree_info.get("original_head")
        # _cleanup_result is initialized before the try block; update it here
        if _wt_path and _wt_head:
            try:
                _cleanup_result = _cleanup_worktree(_wt_path, _wt_head)
            except Exception as exc:
                logger.warning("Worktree cleanup failed: %s", exc)
                _cleanup_result = {"reason": str(exc), "kept": True}
            # Unregister task env overrides
            if _subagent_id:
                try:
                    from tools.terminal_tool import clear_task_env_overrides
                    clear_task_env_overrides(_subagent_id)
                except Exception:
                    pass

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
    agent_id: Optional[str] = None,
    blocked_tools: Optional[List[str]] = None,
    isolation: Optional[str] = None,
    run_in_background: bool = False,
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

    NEW (Phase A): agent_id loads a named profile from agent-registry.json.
    agent_id and role are mutually exclusive. blocked_tools can only add to
    the profile's blocked list, never subtract.

    NEW (Phase B): isolation='readonly' strips write tools. isolation='worktree'
    returns an error (not yet implemented). run_in_background=True returns
    immediately with status='running' and a subagent_id for later status queries.

    Returns JSON with results array, one entry per task.
    """
    if parent_agent is None:
        return tool_error("delegate_task requires a parent agent context.")

    # ── Phase A: agent_id / role mutual exclusion ──────────────────────
    if agent_id and role:
        return tool_error(
            "'agent_id' and 'role' are mutually exclusive. "
            "Use agent_id for named subagents (registry profile), "
            "or role for legacy leaf/orchestrator behavior."
        )

    # ── Phase A: resolve agent_id profile ──────────────────────────────
    _resolved_agent_config: Optional[dict] = None
    _resolved_profile: Optional[dict] = None
    _requested_isolation: Optional[str] = isolation
    if agent_id:
        try:
            _resolved_agent_config, _resolved_profile = _load_subagent_profile(agent_id)
        except ValueError as exc:
            return tool_error(str(exc))

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

    if run_in_background and len(task_list) > 1:
        return tool_error(
            "run_in_background is only supported for single-task delegation. "
            "Use separate delegate_task calls for multiple background tasks."
        )

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
            # Per-task agent_id — batch mode: each task can specify its own agent
            task_agent_id = t.get("agent_id") if "agent_id" in t else None
            effective_agent_id = task_agent_id or agent_id  # per-task beats top-level
            task_agent_config = None
            task_profile = None
            if effective_agent_id:
                try:
                    task_agent_config, task_profile = _load_subagent_profile(effective_agent_id)
                except ValueError:
                    pass  # fall through: per-task profile load failure → skip agent_id path
            # Per-task blocked_tools — can only add to profile blocked
            task_blocked = t.get("blocked_tools") if "blocked_tools" in t else None
            effective_blocked = (
                (blocked_tools or []) + (task_blocked or [])
                if (blocked_tools or task_blocked)
                else None
            ) or None
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
                agent_id=effective_agent_id,
                agent_config=task_agent_config or _resolved_agent_config,
                profile=task_profile or _resolved_profile,
                requested_blocked_tools=effective_blocked,
                requested_isolation=t.get("isolation") if "isolation" in t else _requested_isolation,
                profile_isolation=((task_profile or _resolved_profile) or {}).get("isolation") if (task_profile or _resolved_profile) else None,
                profile_permission_mode=((task_profile or _resolved_profile) or {}).get("permission_mode") if (task_profile or _resolved_profile) else None,
            )
            # Override with correct parent tool names (before child construction mutated global)
            child._delegate_saved_tool_names = _parent_tool_names
            children.append((i, t, child))
    finally:
        # Authoritative restore: reset global to parent's tool names after all children built
        _model_tools._last_resolved_tool_names = _parent_tool_names

    # ── Phase B: isolation error check ──────────────────────────────────
    for _i, _t, child in children:
        iso_err = getattr(child, "_subagent_isolation_error", None)
        if iso_err:
            return tool_error(iso_err)

    if n_tasks == 1:
        _i, _t, child = children[0]
        if run_in_background:
            # ── Phase C+: Background execution ────────────────────────
            _sid = getattr(child, "_subagent_id", None)
            _bg_executor = _get_background_executor()

            def _bg_run(c, g, p):
                try:
                    r = _run_single_child(0, g, c, p)
                except Exception as exc:
                    logger.exception("Background subagent crashed: %s", exc)
                    r = {
                        "task_index": 0, "status": "error",
                        "subagent_id": getattr(c, "_subagent_id", None),
                        "error": f"background subagent crashed: {exc}",
                    }
                sid = getattr(c, "_subagent_id", None)
                key = sid if isinstance(sid, str) else f"unknown-{time.time()}"
                with _background_results_lock:
                    _background_results[key] = r

            _bg_executor.submit(_bg_run, child, _t["goal"], parent_agent)
            results.append({
                "task_index": 0,
                "status": "running",
                "subagent_id": _sid if isinstance(_sid, str) else None,
                "goal": (_t["goal"] or "").strip(),
                "summary": "Background subagent started. Use get_subagent_status() to monitor.",
                "isolation": getattr(child, "_subagent_isolation", "shared") or "shared",
                "duration_seconds": 0,
            })
        else:
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
            if getattr(parent_agent, "session_cost_source", "none") in (None, "", "none"):
                parent_agent.session_cost_source = "subagent"
            if getattr(parent_agent, "session_cost_status", "unknown") in (None, "", "unknown"):
                parent_agent.session_cost_status = "estimated"
        except Exception:
            logger.debug("Subagent cost rollup failed", exc_info=True)

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

    if configured_base_url:
        # When delegation.api_key is not set, return None so _build_child_agent
        # falls back to the parent agent's API key via the credential inheritance
        # path (effective_api_key = override_api_key or parent_api_key). This
        # lets providers that store their key in a non-OPENAI_API_KEY env var
        # (e.g. MINIMAX_API_KEY, DASHSCOPE_API_KEY) work without requiring
        # callers to duplicate the key under delegation.api_key.
        api_key = configured_api_key  # None → inherited from parent in _build_child_agent

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

        cfg = CLI_CONFIG.get("delegation") or {}
        if cfg:
            return cfg
    except Exception:
        pass
    try:
        from hermes_cli.config import load_config

        full = load_config()
        return full.get("delegation") or {}
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
        "NAMED AGENTS (use agent_id — PREFERRED over role/toolsets):\n"
        "- agent_id='kimi' — Deprecated/manual only. Kimi CLI may be unavailable; "
        "do NOT use for default research unless the user explicitly asks for Kimi.\n"
        "- agent_id='claude' — Code modification, script execution, git ops, file editing. "
        "Has 'file'+'terminal' toolsets. Use when the user says \"用 claude\", "
        "\"让 claude 改代码\", or needs to write/edit files or run commands.\n"
        "- agent_id='codex' — Codex CLI coding executor / code review backup. "
        "Use when the user says \"用 codex\" or wants Codex-style implementation/review.\n"
        "- agent_id='openclaw' — External desktop operator backed by OpenClaw "
        "(zai/GLM-5-Turbo). Use for macOS desktop control, screenshots, app operations.\n"
        "- agent_id='hermes-internal' — Analysis, decision making, planning. "
        "Has 'file' toolset, read-only. Use for strategy judgments and design decisions.\n"
        "- agent_id='deepseek-worker' — Background execution, polling, file ops. "
        "Async-only worker for long-running background tasks.\n\n"
        "CRITICAL RULE — When to use agent_id:\n"
        "If the user's request contains ANY of these patterns, you MUST use agent_id:\n"
        "- \"用 kimi\" / \"用 claude\" / \"用 codex\" / \"用 openclaw\" / \"用 hermes\" → matching agent_id\n"
        "- \"派 kimi\" / \"让 kimi\" / \"叫 kimi\" → agent_id='kimi'\n"
        "- \"agent_id=kimi\" / \"agent_id=claude\" / \"agent_id=codex\" → matching agent_id\n"
        "- \"kimi 搜索\" / \"kimi 查\" / \"kimi 读\" → agent_id='kimi'\n"
        "- \"claude 改\" / \"claude 写\" / \"claude 执行\" → agent_id='claude'\n"
        "- \"codex 审\" / \"codex 改\" / \"codex 实现\" → agent_id='codex'\n"
        "- \"桌面\" / \"截图\" / \"打开 App\" / \"点击\" / \"输入\" with OpenClaw mentioned → agent_id='openclaw'\n"
        "- User mentions a specific agent name → use agent_id, do NOT use role+toolsets\n\n"
        "TWO MODES (one of 'goal' or 'tasks' is required):\n"
        "1. Single task: provide 'goal' (+ optional context, toolsets)\n"
        "2. Batch (parallel): provide 'tasks' array. Nested delegation requires role='orchestrator' and delegation.max_spawn_depth >= 2.\n\n"
        "WHEN TO USE delegate_task:\n"
        "- Reasoning-heavy subtasks (debugging, code review, research synthesis)\n"
        "- Tasks that would flood your context with intermediate data\n"
        "- Parallel independent workstreams (research A and B simultaneously)\n\n"
        "WHEN NOT TO USE (use these instead):\n"
        "- Mechanical multi-step work with no reasoning needed -> use execute_code\n"
        "- Single tool call -> just call the tool directly\n"
        "- Tasks needing user interaction -> subagents cannot use clarify\n"
        "- Durable long-running work -> use cronjob or terminal(background=True)\n\n"
        "IMPORTANT:\n"
        "- Subagents have NO memory of your conversation. Pass all relevant "
        "info (file paths, error messages, constraints) via the 'context' field.\n"
        "- Always set 'context' to include the user's language preference "
        "(e.g. \"respond in Chinese\") to prevent subagent language mismatch.\n"
        "- Subagent summaries are SELF-REPORTS, not verified facts.\n"
        "- Leaf subagents (role='leaf', the default) CANNOT call: "
        "delegate_task, clarify, memory, send_message, execute_code.\n"
        "- Orchestrator subagents (role='orchestrator') retain "
        "delegate_task so they can spawn their own workers.\n"
        "- Each subagent gets its own terminal session (separate working directory and state).\n"
        "- Results are always returned as an array, one entry per task.\n"
        "- agent_id and role are MUTUALLY EXCLUSIVE. When user names a specific "
        "agent (kimi, claude, etc.), always use agent_id, never role."
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
                            "description": "Per-task ACP command override (e.g. 'copilot'). Overrides the top-level acp_command for this task only.",
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
                        "agent_id": {
                            "type": "string",
                            "enum": ["kimi", "claude", "codex", "openclaw", "hermes-internal", "deepseek-worker"],
                            "description": "Per-task agent override. When set, this specific task uses the named agent's profile. In batch mode, each task can be routed to a different agent.",
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
            "role": {
                "type": "string",
                "enum": ["leaf", "orchestrator"],
                "description": (
                    "ONLY use when the user does NOT specify a named agent. "
                    "'leaf' (default) = worker, cannot further delegate. "
                    "'orchestrator' = can spawn its own workers. Requires "
                    "delegation.max_spawn_depth >= 2. "
                    "If the user mentions kimi/claude/codex/openclaw/hermes/deepseek, "
                    "use agent_id instead — do NOT use role."
                ),
            },
            "agent_id": {
                "type": "string",
                "enum": ["kimi", "claude", "codex", "openclaw", "hermes-internal", "deepseek-worker"],
                "description": (
                    "WHEN TO USE: User mentions a specific agent name (\"用 kimi\", "
                    "\"派 claude\", \"用 codex\", \"用 openclaw\", \"agent_id=kimi\"). "
                    "This is the PREFERRED way to delegate — each agent has a "
                    "pre-configured profile with the right toolsets, permissions, "
                    "and isolation mode. ALWAYS use agent_id when the user names "
                    "an agent. NEVER translate an agent name into role+toolsets.\n\n"
                    "AGENTS:\n"
                    "- 'kimi': deprecated/manual-only web research; Kimi CLI may be unavailable\n"
                    "- 'claude': code editing & command execution (tools: file+terminal)\n"
                    "- 'codex': Codex CLI coding executor and code review backup\n"
                    "- 'openclaw': external desktop operator for macOS/app/screenshot tasks\n"
                    "- 'hermes-internal': analysis & decision making (tools: file, readonly)\n"
                    "- 'deepseek-worker': background tasks & polling (tools: file)\n\n"
                    "Mutually exclusive with 'role'. When agent_id is set, "
                    "toolsets/blocked_tools/isolation are auto-configured from "
                    "the registry profile — do not set them manually."
                ),
            },
            "blocked_tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Additional tools to block for this subagent (can only add, "
                    "never subtract from the profile's blocked list). "
                    "Use when the registry profile is nearly right but needs "
                    "one extra tool blocked for this specific call."
                ),
            },
            "isolation": {
                "type": "string",
                "enum": ["shared", "readonly", "worktree"],
                "description": (
                    "Subagent isolation mode. 'shared' (default) shares the "
                    "parent's workspace. 'readonly' strips all write-capable "
                    "tools (write_file, patch, terminal, process). "
                    "'worktree' is not yet implemented — returns an error. "
                    "When permission_mode is 'read_only' in the registry profile, "
                    "readonly isolation is automatically enabled."
                ),
            },
            "run_in_background": {
                "type": "boolean",
                "description": (
                    "When true, the subagent runs asynchronously in a background "
                    "thread. The parent returns immediately with status='running' "
                    "and a subagent_id. Use get_subagent_status(subagent_id) to "
                    "monitor progress and get_subagent_result(subagent_id) to "
                    "retrieve the final result. Only supported for single-task "
                    "delegation (not batch mode)."
                ),
            },
            "acp_command": {
                "type": "string",
                "description": (
                    "Override ACP command for child agents (e.g. 'copilot'). "
                    "When set, children use ACP subprocess transport instead of inheriting "
                    "the parent's transport. Requires an ACP-compatible CLI "
                    "(currently GitHub Copilot CLI via 'copilot --acp --stdio'). "
                    "See agent/copilot_acp_client.py for the implementation."
                ),
            },
            "acp_args": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Arguments for the ACP command (default: ['--acp', '--stdio']). "
                    "Only used when acp_command is set."
                ),
            },
        },
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# Delegation guidance for the parent agent's system prompt
# ---------------------------------------------------------------------------


def get_delegation_guidance() -> str:
    """Return system prompt text instructing the model when and how to delegate.

    Reads available agents from agent-registry.json to keep the guidance
    in sync with the actual registry.
    """
    try:
        registry = _load_agent_registry()
        agents = registry.get("agents", {})
    except Exception:
        agents = {}

    agent_lines = []
    for aid, cfg in agents.items():
        profile = cfg.get("subagent_profile", {})
        caps = cfg.get("capabilities", [])
        cap_str = ", ".join(caps[:4]) if caps else "general"
        toolsets = profile.get("toolsets", [])
        ts_str = ", ".join(toolsets) if toolsets else "inherited"
        agent_lines.append(
            f"  - agent_id='{aid}': {cfg.get('type', 'worker')} — "
            f"capabilities=[{cap_str}], tools=[{ts_str}]"
        )

    agent_list = "\n".join(agent_lines) if agent_lines else (
        "  - agent_id='kimi': deprecated/manual-only research; Kimi CLI may be unavailable\n"
        "  - agent_id='claude': code editing & commands (tools: file+terminal)\n"
        "  - agent_id='codex': Codex CLI coding executor and code review backup\n"
        "  - agent_id='openclaw': external desktop operator for macOS/app/screenshot tasks\n"
        "  - agent_id='hermes-internal': analysis & planning (tools: file, readonly)"
    )

    return (
        "## Subagent Delegation (delegate_task)\n"
        "You have access to delegate_task with pre-configured named agents. "
        "Each agent has a specific profile with the right toolsets, permissions, "
        "and isolation mode pre-set.\n\n"
        "**CRITICAL — When user mentions an agent name, you MUST use delegate_task "
        "with the agent_id parameter:**\n"
        "- \"用 kimi\" / \"让 kimi\" / \"派 kimi\" → delegate_task(agent_id='kimi', goal=...) only when explicitly requested\n"
        "- \"用 claude\" / \"让 claude\" / \"派 claude\" → delegate_task(agent_id='claude', goal=...)\n"
        "- \"用 codex\" / \"让 codex\" / \"agent_id=codex\" → delegate_task(agent_id='codex', goal=...)\n"
        "- \"用 openclaw\" / \"让 openclaw\" / \"agent_id=openclaw\" → delegate_task(agent_id='openclaw', goal=...)\n"
        "- \"agent_id=kimi\" or \"agent_id=claude\" → delegate_task(agent_id=..., goal=...)\n\n"
        "**Available agents:**\n"
        f"{agent_list}\n\n"
        "**Key rules:**\n"
        "1. Kimi is deprecated/manual-only; do not use it as the default research route.\n"
        "2. When the user names claude/codex/openclaw, use the matching agent_id.\n"
        "3. agent_id and role are mutually exclusive. Use agent_id, not role.\n"
        "4. Never translate a user's agent request into manual toolsets+role.\n"
        "5. Set 'context' to include the user's language preference.\n"
        "6. Treat OpenClaw as an external desktop operator; verify desktop results with screenshots or command output."
    )


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
        agent_id=args.get("agent_id"),
        blocked_tools=args.get("blocked_tools"),
        isolation=args.get("isolation"),
        run_in_background=args.get("run_in_background", False),
        parent_agent=kw.get("parent_agent"),
    ),
    check_fn=check_delegate_requirements,
    emoji="🔀",
)
