"""Shared delegation policy, control, and sanitization helpers."""

import enum
import contextvars
import json
import logging
import re

logger = logging.getLogger("tools.delegate_tool")
import os
import threading
import time
import weakref
from concurrent.futures import (
    TimeoutError as FuturesTimeoutError,
)
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

from toolsets import TOOLSETS
from agent.interrupt_compat import request_hard_interrupt
from tools.delegation_outcome import (
    DELEGATION_OUTCOME_TOOL_GUIDANCE,
    apply_schema_evidence,
    classify_delegation_result,
    delegation_batch_icon,
    delegation_stop_evidence,
    failed_delegation_evidence,
    schema_evidence_payload,
    terminal_tool_error_count,
)
from tools.delegation_output_schema import validate_and_repair_output

# Sentinel value used by the runtime provider system for providers that are
# not natively known (named custom providers, third-party aggregators, etc.).
# Must match hermes_cli.runtime_provider.RUNTIME_PROVIDER_TYPE_CUSTOM.
_RUNTIME_PROVIDER_CUSTOM = "custom"
from tools import file_state
from tools.terminal_tool import set_approval_callback as _set_subagent_approval_cb
from tools.registry import tool_error
from utils import base_url_hostname, is_truthy_value


# Tools that children must never have access to
DELEGATE_BLOCKED_TOOLS = frozenset(
    [
        "delegate_task",  # no recursive delegation
        "clarify",  # no user interaction
        "memory",  # no writes to shared MEMORY.md
        "send_message",  # no cross-platform side effects
        "cronjob_manage",  # no scheduling more work in the parent's name
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

_DEFAULT_MAX_CONCURRENT_CHILDREN = 10
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

# subagent_id -> {goal, delegation_id, parent_session_id} retained AFTER the
# child finishes (bounded FIFO). Child-started background processes routinely
# outlive the child itself (its npm ci with notify_on_complete=true finishes
# after the child's summary was delivered); their completion notifications
# reach the parent conversation via the shared completion_queue and need
# delegation attribution even though the live registry entry is gone.
_RECENT_SUBAGENTS_CAP = 200
_recent_subagents: Dict[str, Dict[str, Any]] = {}


# Terminal child statuses that mean "the subagent did NOT deliver a usable
# result". Shared by the CLI spinner echo, the gateway failure notice, and
# the parent-facing failure summary so every surface agrees on what counts
# as a failure.
SUBAGENT_FAILURE_STATUSES = frozenset({"failed", "error", "timeout"})


def _clean_error_text(error: Any, max_chars: int = 200) -> str:
    """Reduce an arbitrary error payload to one clean human-readable line.

    Provider/SDK errors routinely arrive as multi-line tracebacks or JSON
    walls. For a chat-facing notice we want the single most informative
    line: the exception message (last line of a traceback) or the first
    non-empty line otherwise, hard-capped in length.
    """
    text = str(error or "").strip()
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    # A traceback's last line is the actual exception message.
    line = lines[-1] if lines[0].startswith("Traceback") else lines[0]
    if len(line) > max_chars:
        line = line[: max_chars - 3] + "..."
    return line


def format_subagent_failure_line(
    goal: Optional[str],
    status: Optional[str],
    error: Any = None,
    duration_seconds: Any = None,
) -> str:
    """One clean, human-readable line describing a failed subagent.

    Rendered directly to the user (CLI spinner echo, gateway platform
    notice) — no JSON, no traceback, no internal field names. Example:

        ⚠️ Subagent failed — "research competitor pricing": Error code: 404 —
        model not found (after 12s)
    """
    goal_label = (goal or "").strip().replace("\n", " ")
    if len(goal_label) > 60:
        goal_label = goal_label[:57] + "..."
    verb = "timed out" if status == "timeout" else "failed"
    line = f"⚠️ Subagent {verb}"
    if goal_label:
        line += f' — "{goal_label}"'
    err = _clean_error_text(error)
    if err:
        line += f": {err}"
    if isinstance(duration_seconds, (int, float)) and duration_seconds > 0:
        line += f" (after {round(duration_seconds)}s)"
    return line


def get_subagent_attribution(task_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Resolve a process task_id to its originating delegation, if any.

    Children run their terminal sessions under ``task_id == subagent_id``
    (see _run_single_child's child_task_id), so a background process spawned
    by a subagent carries that id in ``ProcessSession.task_id``. Returns
    ``{subagent_id, goal, delegation_id}`` for live AND recently-finished
    children, or None when the task_id is not a known subagent.
    """
    if not task_id or not isinstance(task_id, str):
        return None
    with _active_subagents_lock:
        record = _active_subagents.get(task_id)
        if record is not None:
            return {
                "subagent_id": task_id,
                "goal": record.get("goal"),
                "delegation_id": record.get("delegation_id"),
            }
        retained = _recent_subagents.get(task_id)
        if retained is not None:
            return {
                "subagent_id": task_id,
                "goal": retained.get("goal"),
                "delegation_id": retained.get("delegation_id"),
            }
    return None


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
    record.setdefault("accepting_steer", True)
    with _active_subagents_lock:
        _active_subagents[sid] = record


def _retain_recent_subagent(record: Dict[str, Any]) -> None:
    """Keep a bounded attribution stub after a child finishes (lock held)."""
    sid = record.get("subagent_id")
    if not sid:
        return
    _recent_subagents[sid] = {
        "goal": record.get("goal"),
        "delegation_id": record.get("delegation_id"),
        "owner_agent_session_id": record.get("owner_agent_session_id"),
    }
    while len(_recent_subagents) > _RECENT_SUBAGENTS_CAP:
        _recent_subagents.pop(next(iter(_recent_subagents)), None)


def _unregister_subagent(subagent_id: str, *, agent: Any = None) -> None:
    with _active_subagents_lock:
        record = _active_subagents.get(subagent_id)
        if record is not None and (agent is None or record.get("agent") is agent):
            _active_subagents.pop(subagent_id, None)
            _retain_recent_subagent(record)


def _close_subagent_steering(subagent_id: str, agent: Any) -> Optional[str]:
    """Atomically close steer acceptance and drain its final durable artifact.

    ``steer_subagent`` holds the same registry lock through ``agent.steer``.
    Therefore either acceptance wins and this drain sees its exact text, or
    closure wins and the caller is rejected. Exact agent identity prevents a
    finishing child with a recycled public id from closing its replacement.
    """
    with _active_subagents_lock:
        record = _active_subagents.get(subagent_id)
        if record is None or record.get("agent") is not agent:
            return None
        record["accepting_steer"] = False
        drain = getattr(agent, "_drain_pending_steer", None)
        if not callable(drain):
            return None
        try:
            pending = drain()
        except Exception as exc:
            logger.debug("final steer drain for %s failed: %s", subagent_id, exc)
            return None
        return pending if isinstance(pending, str) and pending.strip() else None


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
        if not request_hard_interrupt(agent, f"Interrupted via TUI ({subagent_id})"):
            return False
    except Exception as exc:
        logger.debug("interrupt_subagent(%s) failed: %s", subagent_id, exc)
        return False
    return True


def steer_subagent(
    subagent_id: str,
    text: str,
    *,
    owner_session_id: Optional[str] = None,
    owner_transport: Any = None,
    owner_session_record: Any = None,
) -> bool:
    """Queue steering text into a single running subagent without stopping it.

    The redirection-side mirror of interrupt_subagent(): resolves the live
    child in the registry and calls AIAgent.steer(), which appends the text
    to the child's last tool result at its next iteration boundary — the
    current tool call is never cut. Returns True if a matching subagent
    QUEUED the text while the child was still accepting work; False for an
    unknown/closed id, an ownership mismatch, a record with no live agent, or
    empty text. ``owner_session_id=None`` deliberately preserves the internal
    in-process helper contract; gateway callers must pass exact authority.

    Acceptance and completion are linearized by the registry lock. If
    acceptance wins but no delivery boundary remains, ``_run_single_child``
    drains the exact text into the completion entry as ``missed_steer``.
    """
    if not text or not text.strip():
        return False
    with _active_subagents_lock:
        record = _active_subagents.get(subagent_id)
        if not record or not record.get("accepting_steer", False):
            return False
        if owner_session_id is not None:
            if (
                record.get("owner_session_id") != owner_session_id
                or owner_transport is None
                or record.get("owner_transport") is not owner_transport
                or owner_session_record is None
                or record.get("owner_session_record") is not owner_session_record
            ):
                return False
        agent = record.get("agent")
        if agent is None:
            return False
        try:
            return bool(agent.steer(text))
        except Exception as exc:
            logger.debug("steer_subagent(%s) failed: %s", subagent_id, exc)
            return False


def _capture_gateway_steer_authority(
    owner_session_id: Optional[str],
) -> tuple[Any, Any]:
    """Capture exact request transport + live session generation, if any.

    This is intentionally an in-process bridge, not a serializable capability.
    Non-gateway hosts (including the CLI helper path) receive ``(None, None)``.
    """
    if not owner_session_id:
        return None, None
    try:
        from tui_gateway.server import _current_session_steer_authority

        return _current_session_steer_authority(owner_session_id)
    except Exception:
        return None, None


def list_active_subagents() -> List[Dict[str, Any]]:
    """Snapshot of the currently running subagent tree.

    Each record: {subagent_id, parent_id, depth, goal, model, started_at,
    tool_count, status}.  Safe to call from any thread — returns a copy.
    """
    with _active_subagents_lock:
        return [
            {
                k: v
                for k, v in r.items()
                if k
                not in {
                    "agent",
                    "owner_session_id",
                    "owner_transport",
                    "owner_session_record",
                    "accepting_steer",
                }
            }
            for r in _active_subagents.values()
        ]


def _is_descendant_of(child_agent: Any, parent_agent: Any, max_hops: int = 8) -> bool:
    """True when *child_agent* sits below *parent_agent* in the spawn tree.

    Walks the ``_delegate_parent_ref`` weakref chain stamped at build time.
    Identity comparison only — a parent may steer/stop its own children and
    grandchildren, never a sibling tree owned by another conversation.
    """
    if child_agent is None or parent_agent is None:
        return False
    cur = child_agent
    for _ in range(max_hops):
        ref = getattr(cur, "_delegate_parent_ref", None)
        ancestor = ref() if callable(ref) else None
        if ancestor is None:
            return False
        if ancestor is parent_agent:
            return True
        cur = ancestor
    return False


# Model-facing control actions accepted by delegate_task(action=...).
# "spawn" (or omitted) keeps the historical spawn semantics.
_CONTROL_ACTIONS = frozenset({"list", "steer", "stop"})


def _resolve_session_lineage(session_id: Optional[str], parent_agent: Any) -> str:
    """Resolve a session id to the tip of its compression lineage.

    Best-effort: uses the parent's live SessionDB handle when present so a
    delegation dispatched before a compression rotation still matches the
    rotated parent. Returns the input unchanged when resolution fails.
    """
    sid = str(session_id or "")
    if not sid:
        return ""
    db = getattr(parent_agent, "_session_db", None)
    if db is None:
        return sid
    try:
        resolved = db.resolve_resume_session_id(sid)
        return str(resolved) if resolved else sid
    except Exception:
        return sid


def _owns_subagent_record(record: Dict[str, Any], parent_agent: Any) -> bool:
    """True when *parent_agent*'s conversation owns this live-child record.

    Two-tier check:

    1. Object identity — the ``_delegate_parent_ref`` weakref chain stamped
       at build time reaches *parent_agent*. Fast path for the common case
       where the parent AIAgent object survives the whole run.
    2. Durable conversation lineage — the child was registered with the
       owning conversation's durable session id
       (``owner_agent_session_id``); match it against the calling parent's
       ``session_id``, resolving compression-rotation lineage on both sides.

    Tier 2 exists because the identity chain is BRITTLE across parent-agent
    rebuilds: the CLI sets ``self.agent = None`` mid-session (route-signature
    change, credential refresh, /model, MoA one-shots) and constructs a NEW
    AIAgent for the next turn while the child keeps running with a weakref to
    the old object. The delivery path always survived this (it routes by
    durable session id); the control path must use the same durable spine or
    running children go invisible/unsteerable (observed live: deleg_88454b70
    / sa-0-dc0100f4, 2026-08-17).
    """
    agent = record.get("agent")
    if _is_descendant_of(agent, parent_agent):
        return True
    owner_sid = str(record.get("owner_agent_session_id") or "")
    if not owner_sid:
        return False
    parent_sid = str(getattr(parent_agent, "session_id", "") or "")
    if not parent_sid:
        return False
    if owner_sid == parent_sid:
        return True
    # Compression rotation on either side: compare lineage tips.
    return _resolve_session_lineage(owner_sid, parent_agent) in {
        parent_sid,
        _resolve_session_lineage(parent_sid, parent_agent),
    }


def _handle_control_action(
    action: str,
    subagent_id: Optional[str],
    message: Optional[str],
    parent_agent: Any,
) -> str:
    """Synchronous control plane for delegate_task: list/steer/stop.

    Runs in-turn (never backgrounded) and only over subagents descended from
    *parent_agent* — the same registry the TUI overlay drives, but scoped so
    a conversation can only control its own spawn tree.
    """
    if action == "list":
        with _active_subagents_lock:
            records = list(_active_subagents.values())
        entries = []
        for r in records:
            agent = r.get("agent")
            if not _owns_subagent_record(r, parent_agent):
                continue
            started = r.get("started_at")
            entries.append(
                {
                    "subagent_id": r.get("subagent_id"),
                    "parent_id": r.get("parent_id"),
                    "goal": r.get("goal"),
                    "model": r.get("model"),
                    "status": r.get("status"),
                    "running_seconds": (
                        round(time.time() - started, 1)
                        if isinstance(started, (int, float))
                        else None
                    ),
                    "accepting_steer": bool(r.get("accepting_steer", False)),
                    "live_transcript": getattr(agent, "_live_transcript_path", None),
                }
            )
        payload: Dict[str, Any] = {
            "action": "list",
            "count": len(entries),
            "subagents": entries,
        }
        if not entries:
            payload["note"] = (
                "No live subagents right now. Children that already finished "
                "have delivered (or will deliver) their results as normal "
                "completion messages — there is nothing to steer or stop."
            )
        return json.dumps(payload, ensure_ascii=False)

    # steer / stop need a resolvable, owned target.
    sid = (subagent_id or "").strip()
    if not sid:
        return tool_error(
            f"action='{action}' requires subagent_id (from the spawn dispatch "
            "response or action='list')."
        )
    with _active_subagents_lock:
        record = _active_subagents.get(sid)
    if record is None or not _owns_subagent_record(record, parent_agent):
        return tool_error(
            f"No live subagent '{sid}' in this conversation's spawn tree. It "
            "may have already finished (its result arrives as a normal "
            "completion message). Use action='list' to see live children."
        )

    if action == "stop":
        if interrupt_subagent(sid):
            return json.dumps(
                {
                    "action": "stop",
                    "subagent_id": sid,
                    "status": "interrupt_requested",
                    "note": (
                        "The subagent stops at its next iteration boundary "
                        "(in-flight tool calls are asked to cancel). Its "
                        "partial result still re-enters the conversation as a "
                        "completion message — do not wait or poll."
                    ),
                },
                ensure_ascii=False,
            )
        return tool_error(
            f"Could not interrupt '{sid}' — it likely finished in the last "
            "moment. Its result arrives as a normal completion message."
        )

    if action == "steer":
        text = (message or "").strip()
        if not text:
            return tool_error(
                "action='steer' requires a non-empty 'message' describing the "
                "course correction."
            )
        if steer_subagent(sid, text):
            return json.dumps(
                {
                    "action": "steer",
                    "subagent_id": sid,
                    "status": "queued",
                    "note": (
                        "Steering text queued. The subagent sees it appended "
                        "to its next tool result — the current tool call is "
                        "never cut. If the child finishes before a delivery "
                        "boundary remains, the text is reported back as "
                        "missed_steer in its completion entry."
                    ),
                },
                ensure_ascii=False,
            )
        return tool_error(
            f"Subagent '{sid}' is no longer accepting steering (finishing or "
            "already finished). Its result arrives as a normal completion "
            "message; re-delegate a follow-up task if more work is needed."
        )

    return tool_error(f"Unknown action '{action}'. Use spawn, list, steer, or stop.")


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


_TOOL_INPUT_TARGET_KEYS = frozenset({
    "cwd",
    "destination_path",
    "directory",
    "dst",
    "endpoint",
    "file_path",
    "new_path",
    "old_path",
    "path",
    "source_path",
    "src",
    "target_path",
    "url",
    "urls",
})
_TOOL_INPUT_URL_KEYS = frozenset({"endpoint", "url", "urls"})


def _sanitize_tool_target(key: str, value: Any) -> Any:
    """Keep bounded side-effect targets while dropping URL secrets."""
    if isinstance(value, list):
        cleaned = [
            item for item in (_sanitize_tool_target(key, item) for item in value[:16])
            if item is not None
        ]
        return cleaned or None
    if not isinstance(value, str) or not value:
        return None
    bounded = value[:1024]
    if key in _TOOL_INPUT_URL_KEYS:
        try:
            parsed = urlsplit(bounded)
            if parsed.scheme and parsed.netloc:
                hostname = parsed.hostname
                if not hostname:
                    return None
                # ``SplitResult.netloc`` includes ``user:password@``. Rebuild
                # the authority from parsed host/port so hook-visible history
                # cannot carry URL credentials. Bracket IPv6 literals before
                # appending a validated port.
                host = f"[{hostname}]" if ":" in hostname else hostname
                port = parsed.port
                netloc = f"{host}:{port}" if port is not None else host
                return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
        except ValueError:
            return None
    return bounded


def _summarize_tool_arguments(arguments: Any) -> Dict[str, Any]:
    """Summarize argument names and side-effect targets without raw payloads."""
    if not isinstance(arguments, str):
        return {"argument_keys": [], "targets": {}}
    try:
        parsed = json.loads(arguments)
    except (TypeError, ValueError):
        return {"argument_keys": [], "targets": {}}
    if not isinstance(parsed, dict):
        return {"argument_keys": [], "targets": {}}

    keys = sorted(str(key)[:128] for key in parsed)[:64]
    targets: Dict[str, Any] = {}
    for raw_key, value in parsed.items():
        key = str(raw_key).lower()
        if key not in _TOOL_INPUT_TARGET_KEYS:
            continue
        cleaned = _sanitize_tool_target(key, value)
        if cleaned is not None:
            targets[key] = cleaned
    return {"argument_keys": keys, "targets": targets}


def _sanitize_tool_input_summary(summary: Any) -> Dict[str, Any]:
    if not isinstance(summary, dict):
        return {"argument_keys": [], "targets": {}}
    keys = summary.get("argument_keys")
    safe_keys = (
        [str(key)[:128] for key in keys[:64]]
        if isinstance(keys, list)
        else []
    )
    targets = summary.get("targets")
    safe_targets: Dict[str, Any] = {}
    if isinstance(targets, dict):
        for raw_key, value in targets.items():
            key = str(raw_key).lower()
            if key not in _TOOL_INPUT_TARGET_KEYS:
                continue
            cleaned = _sanitize_tool_target(key, value)
            if cleaned is not None:
                safe_targets[key] = cleaned
    return {"argument_keys": safe_keys, "targets": safe_targets}


def _subagent_stop_tool_call_history(tool_trace: Any) -> List[Dict[str, Any]]:
    """Build a detached, metadata-only tool history for lifecycle hooks."""
    if not isinstance(tool_trace, list):
        return []

    history: List[Dict[str, Any]] = []
    for item in tool_trace:
        if not isinstance(item, dict):
            continue
        tool_name = str(item.get("tool") or "unknown")[:256]
        status = str(item.get("status") or "unknown").lower()
        if status not in {"ok", "error"}:
            status = "unknown"

        def _byte_count(key: str) -> int:
            value = item.get(key, 0)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return 0
            return max(0, int(value))

        history.append({
            "tool_name": tool_name,
            "tool_input": _sanitize_tool_input_summary(item.get("input_summary")),
            "input_bytes": _byte_count("args_bytes"),
            "output_bytes": _byte_count("result_bytes"),
            "status": status,
        })
    return history


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
    DELEGATION_MAX_CONCURRENT_CHILDREN env var, then the default (10).

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


def _get_worktree_isolation() -> bool:
    """Read delegation.worktree_isolation from config (bool, default False).

    Inspired by Muse Code's ``--subagent-worktree-isolation`` (Meta, Aug
    2026): when enabled, each delegated child gets its own git worktree
    checked out from the parent's current commit so parallel children never
    contend for the same working copy. Opt-in and git-only — in a non-git
    workspace or on a non-local terminal backend the flag is ignored without
    an error and children share the parent's workspace as before.
    """
    cfg = _load_config()
    return bool(cfg.get("worktree_isolation", False))


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


DEFAULT_MAX_ITERATIONS = 250
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
# Stale-heartbeat thresholds. A child with no observable progress is either:
#   - idle between turns (no current_tool, frozen last_activity_ts) — wedged
#   - inside a tool (current_tool set) — probably running a legitimately long
#     operation (terminal command, web fetch, large file read)
# An in-flight model wait is NOT idle: direct_api_call refreshes
# last_activity_ts while the request is open, and the monitor treats that
# timestamp advance as progress (same signal as streamed chunks / async
# stall monitor). Slow local GGUF / long-prefill models must not be killed
# for taking longer than the idle window on a single completion.
# The idle ceiling stays tight so a child that is truly between turns with
# no activity doesn't mask the gateway timeout. The in-tool ceiling is much
# higher so legit long-running tools get time to finish;
# delegation.child_timeout_seconds (off by default) remains an optional hard
# cap for users who want one.
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


# Late-bound compatibility seams: tests and embedders patch the historical
# tools.delegate_tool owner, so policy helpers resolve through that facade.
def _facade():
    from tools import delegate_tool as facade

    return facade


def _load_config():
    return _facade()._load_config()


def request_hard_interrupt(*args, **kwargs):
    return _facade().request_hard_interrupt(*args, **kwargs)
