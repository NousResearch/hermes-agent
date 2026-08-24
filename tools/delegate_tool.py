#!/usr/bin/env python3
"""
Delegate Tool -- Subagent Architecture

Spawns child AIAgent instances with isolated context, inherited toolsets,
and their own terminal sessions. Supports single-task and batch (parallel)
modes. Top-level model calls run in the background; orchestrator children
wait for their own workers so they can synthesize the results.

Each child gets:
  - A fresh conversation (no parent history)
  - Its own task_id (own terminal session, file ops cache)
  - The parent's toolsets, with child-only blocked tools stripped
  - A focused system prompt built from the delegated goal + context

The parent's context only sees the delegation call and the summary result,
never the child's intermediate tool calls or reasoning.
"""

import enum
import contextvars
import json
import logging
import re

logger = logging.getLogger(__name__)
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
        # Project context files (AGENTS.md / CLAUDE.md / .cursorrules ...)
        # from the workspace, via the SAME discovery/priority/cap logic the
        # main agent's system prompt uses. Children are constructed with
        # skip_context_files=True (their prompt is this focused one), so
        # without this a subagent works in a repo without the repo's own
        # conventions unless it thinks to go read them. SOUL.md is skipped —
        # identity belongs to the parent. workspace_path comes only from
        # explicit sources (_resolve_workspace_hint: TERMINAL_CWD / agent cwd
        # hints, never bare getcwd), so the #64590 install-tree-fallback leak
        # doesn't apply here. Best-effort: on any failure the child prompt is
        # simply built without the block.
        try:
            from agent.prompt_builder import build_context_files_prompt

            _ctx_files = build_context_files_prompt(
                cwd=str(workspace_path), skip_soul=True
            )
        except Exception:
            logger.debug(
                "subagent: workspace context-files load failed", exc_info=True
            )
            _ctx_files = ""
        if _ctx_files.strip():
            parts.append(
                "\nThe workspace's project context files are reproduced "
                "below. Their conventions and invariants are binding for "
                "your work in this workspace.\n\n" + _ctx_files.strip()
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
    _COMPOSITE_BLOCKED_TOOLSETS = frozenset({"delegation"})
    blocked_toolset_names = {
        name
        for name, defn in TOOLSETS.items()
        if name in _COMPOSITE_BLOCKED_TOOLSETS
        or all(t in DELEGATE_BLOCKED_TOOLS for t in defn.get("tools", []))
    }
    blocked_toolset_names.add("kanban")
    return [t for t in toolsets if t not in blocked_toolset_names]


def _blocked_toolsets_for_role(role: str) -> List[str]:
    """Return one-tool deny toolsets for a delegated child role.

    ``_strip_blocked_tools`` can remove fully blocked toolsets, but it must keep
    mixed platform bundles such as ``hermes-cli`` because those also contain
    useful tools. Passing these exact deny toolsets to AIAgent lets
    ``model_tools`` subtract blocked names *after* composite expansion, and the
    restriction survives later registry/MCP refreshes through the agent's
    stored ``disabled_toolsets``.
    """
    blocked_names = set(DELEGATE_BLOCKED_TOOLS)
    if role == "orchestrator":
        blocked_names.discard("delegate_task")
    return sorted(
        name
        for name, defn in TOOLSETS.items()
        if defn.get("tools")
        and set(defn.get("tools", ())).issubset(blocked_names)
    )


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

    # Blocked tools also live inside mixed platform bundles (hermes-cli,
    # hermes-telegram, etc.) that _strip_blocked_tools must keep because they
    # carry useful tools too. Pass exact one-tool deny toolsets through to the
    # child so model_tools subtracts the blocked names AFTER composite
    # expansion, and the restriction survives later registry/MCP refreshes.
    raw_parent_disabled = getattr(parent_agent, "disabled_toolsets", None)
    if isinstance(raw_parent_disabled, (list, tuple, set)):
        inherited_disabled = [str(name) for name in raw_parent_disabled]
    else:
        inherited_disabled = []
    if effective_role == "orchestrator":
        # Role grants delegate_task explicitly, matching the unconditional
        # delegation toolset re-add below.
        inherited_disabled = [
            name for name in inherited_disabled if name != "delegation"
        ]
    child_disabled_toolsets = list(
        dict.fromkeys(
            inherited_disabled + _blocked_toolsets_for_role(effective_role) + ["kanban"]
        )
    )

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
    #
    # Nous Portal is dual-wire within a single provider: anthropic/* → Messages,
    # everything else → chat_completions. Same-provider inheritance would pin a
    # child Hermes/Qwen subagent onto the parent's Claude Messages wire (or the
    # reverse). agent_init honors an explicit api_mode above its nous branch, so
    # re-derive here before construction.
    _parent_provider = getattr(parent_agent, "provider", None) or ""
    _effective_provider_norm = (effective_provider or "").strip().lower()
    if override_api_mode is not None:
        effective_api_mode = override_api_mode
    elif _effective_provider_norm in {"nous", "nous-portal", "nousresearch"}:
        from hermes_cli.providers import nous_api_mode

        effective_api_mode = nous_api_mode(effective_model)
    elif effective_provider != _parent_provider:
        effective_api_mode = None  # force re-derivation from provider's defaults
    else:
        effective_api_mode = getattr(parent_agent, "api_mode", None)
    # Defensive: validate trusted delegation.command exists on PATH before
    # honoring it. An explicitly pinned transport that cannot run must fail
    # the spawn loudly (#80450) — silently falling back to the default
    # transport would run the child somewhere the user explicitly routed it
    # away from. Normally unreachable via delegate_task, which pre-validates
    # the command in _resolve_delegation_credentials.
    if override_acp_command:
        import shutil as _shutil

        if not _shutil.which(override_acp_command):
            raise ValueError(
                f"Pinned delegation command '{override_acp_command}' was not "
                f"found on PATH. Install it or remove delegation.command from "
                f"config.yaml."
            )
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
    #
    # EXCEPT when the user pinned delegation.provider: an explicit pin means
    # "children run on THIS provider".  Inheriting the parent chain would let
    # a mid-run auth/429 failure silently reroute the quiet-mode child onto
    # the parent's fallback models with no surfaced signal (#80450) — the
    # same class of silent-drag the override_provider filter-clearing below
    # already prevents for OpenRouter routing preferences.  Predictability >
    # liveness for explicit pins: the pinned child fails loudly instead.
    parent_fallback = (
        None
        if override_provider
        else (getattr(parent_agent, "_fallback_chain", None) or None)
    )

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

    # Each child gets a DEDICATED SessionDB connection instead of the parent's
    # live object. The parent's handle is owned by the parent's lifecycle
    # (cron run_job's finally block, gateway session end, /new) and can be
    # closed while a fire-and-forget background child is still flushing on a
    # daemon thread — every subsequent flush then hits the closed handle and
    # the child's transcript is silently dropped (#81267). A dedicated handle
    # can't be closed out from under the child; it is released by the child's
    # own close() via the owned flag set below. It MUST point at the same
    # database FILE as the parent's handle: parents can hold non-default
    # per-profile handles (tui_gateway opens SessionDB(db_path=<profile>/
    # state.db) for non-launch profiles), and a bare SessionDB() would write
    # the child's transcript into the launch profile's db, breaking
    # parent_session_id lineage and session_search. AsyncSessionDB wrappers
    # (gateway) forward .db_path via __getattr__, so this works through them.
    child_session_db = None
    parent_session_db = getattr(parent_agent, "_session_db", None)
    if parent_session_db is not None:
        try:
            from hermes_state import SessionDB

            _parent_db_path = getattr(parent_session_db, "db_path", None)
            child_session_db = (
                SessionDB(db_path=_parent_db_path)
                if _parent_db_path is not None
                else SessionDB()
            )
        except Exception:
            logger.debug(
                "subagent: failed to open dedicated SessionDB; child persistence disabled",
                exc_info=True,
            )
            child_session_db = None

    from agent.delegation_context import delegated_child_context

    with delegated_child_context():
        try:
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
                disabled_toolsets=child_disabled_toolsets,
                quiet_mode=True,
                ephemeral_system_prompt=child_prompt,
                log_prefix=f"[subagent-{task_index}]",
                platform="subagent",
                skip_context_files=True,
                skip_memory=True,
                clarify_callback=None,
                thinking_callback=child_thinking_cb,
                session_db=child_session_db,
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
        except BaseException:
            # Construction failed: the dedicated handle has no owner and no
            # child close() will ever run — release it here so the sqlite fds
            # don't outlive the failed spawn.
            if child_session_db is not None:
                try:
                    child_session_db.close()
                except Exception:
                    pass
            raise
    child._print_fn = getattr(parent_agent, "_print_fn", None)
    # Ownership transfer for the dedicated handle: the child's close() must
    # release it (nothing else holds a reference), and no parent teardown can
    # close it out from under a background child (#81267).
    if child_session_db is not None:
        child._owns_session_db = True
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
    # Ownership chain for the model-facing control plane (action=list/steer/
    # stop): a parent may only control agents whose weakref chain reaches it.
    # Weakref so a finished parent can be collected while a detached child
    # record briefly lingers in the registry.
    try:
        child._delegate_parent_ref = weakref.ref(parent_agent)
    except TypeError:
        # Test doubles (MagicMock et al.) may not be weakref-able; control
        # actions then simply don't resolve ownership for this child.
        child._delegate_parent_ref = None
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
        from hermes_cli.lifecycle import invoke_hook as _invoke_hook
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
        import threading as _threading

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

        # All other live threads. The conversation worker's own stack often
        # shows it parked waiting on a nested helper thread (interrupt worker,
        # daemon-pool sibling) — without the full picture, a pre-HTTP wedge
        # (#60203/#62151) is indistinguishable from a slow provider. Best
        # effort and bounded: names + stacks for up to 40 threads.
        _w("## All thread stacks at timeout")
        try:
            frames = _sys._current_frames()
            by_ident = {
                th.ident: th for th in _threading.enumerate() if th.ident
            }
            worker_ident = worker_thread.ident if worker_thread else None
            dumped = 0
            for ident, frame in frames.items():
                if ident == worker_ident:
                    continue  # already dumped above
                if dumped >= 40:
                    _w(f"  <{len(frames) - dumped - 1} more threads omitted>")
                    break
                th = by_ident.get(ident)
                name = th.name if th else f"ident={ident}"
                daemon = " daemon" if (th and th.daemon) else ""
                _w(f"  --- {name}{daemon} ---")
                for frame_line in _traceback.format_stack(frame):
                    for sub in frame_line.rstrip().split("\n"):
                        _w(f"    {sub}")
                dumped += 1
        except Exception as exc:
            _w(f"  <all-thread dump failed: {exc}>")
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
        from tools.spill_safety import write_text_exclusive

        # Exclusive symlink-refusing create; not private because
        # cache/delegation is bind-mounted read-only into remote backends
        # whose container UID must be able to read it.
        write_text_exclusive(path, summary, private=False)
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
    *,
    owner_session_id: Optional[str] = None,
    owner_transport: Any = None,
    owner_session_record: Any = None,
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
    # Stale detection: track the child's (tool, iteration, activity_ts) across
    # heartbeat cycles. If none advances, count the cycle as stale.
    # Different thresholds for idle vs in-tool (see _HEARTBEAT_STALE_CYCLES_*).
    # last_activity_ts is the same liveness signal the async stall monitor
    # already uses (streamed chunks + direct_api_call mid-wait heartbeats).
    _last_seen_iter = [0]
    _last_seen_tool = [None]  # type: list
    _last_seen_activity_ts = [None]  # type: list
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
                child_activity_ts = child_summary.get("last_activity_ts")

                # Stale detection: count cycles where iteration, current_tool,
                # AND last_activity_ts are all frozen. A child running a
                # legitimately long-running tool keeps current_tool set; a
                # child waiting on a slow model refreshes last_activity_ts
                # via direct_api_call's activity heartbeat — neither should
                # look stale at the idle threshold.
                iter_advanced = child_iter > _last_seen_iter[0]
                tool_changed = child_tool != _last_seen_tool[0]
                activity_advanced = (
                    child_activity_ts is not None
                    and (
                        _last_seen_activity_ts[0] is None
                        or child_activity_ts > _last_seen_activity_ts[0]
                    )
                )
                if iter_advanced or tool_changed or activity_advanced:
                    _last_seen_iter[0] = child_iter
                    _last_seen_tool[0] = child_tool
                    if child_activity_ts is not None:
                        _last_seen_activity_ts[0] = child_activity_ts
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
        if owner_session_id is None:
            try:
                from gateway.session_context import get_session_env

                owner_session_id = get_session_env("HERMES_UI_SESSION_ID", "") or None
            except Exception:
                owner_session_id = None
        if owner_session_id and (
            owner_transport is None or owner_session_record is None
        ):
            owner_transport, owner_session_record = (
                _capture_gateway_steer_authority(owner_session_id)
            )
        _raw_depth = getattr(child, "_delegate_depth", 1)
        _tui_depth = max(0, _raw_depth - 1) if isinstance(_raw_depth, int) else 0
        _parent_sid = getattr(child, "_parent_subagent_id", None)
        # Durable ownership spine: the OWNING CONVERSATION's session id (the
        # same lineage the delivery path routes completions by). Sourced from
        # the child's _parent_session_id stamp so it stays correct even when
        # parent_agent has been rebuilt between dispatch and this run.
        _owner_agent_session_id = (
            str(getattr(child, "_parent_session_id", "") or "")
            or str(getattr(parent_agent, "session_id", "") or "")
        )
        _delegation_id = getattr(child, "_delegation_id", None)
        _register_subagent(
            {
                "subagent_id": _subagent_id,
                "parent_id": _parent_sid if isinstance(_parent_sid, str) else None,
                "depth": _tui_depth,
                "goal": goal,
                "delegation_id": (
                    _delegation_id if isinstance(_delegation_id, str) else None
                ),
                "model": (
                    getattr(child, "model", None)
                    if isinstance(getattr(child, "model", None), str)
                    else None
                ),
                "started_at": time.time(),
                "status": "running",
                "tool_count": 0,
                "agent": child,
                # Durable conversation lineage for the model-facing control
                # plane (list/steer/stop). The weakref identity chain breaks
                # when the CLI rebuilds its AIAgent mid-session; this id is
                # the same spine completion delivery routes by.
                "owner_agent_session_id": _owner_agent_session_id or None,
                # Immutable live gateway/TUI session that commissioned this
                # child. Empty outside those hosts; RPC authority fails closed.
                "owner_session_id": owner_session_id,
                "owner_transport": owner_transport,
                "owner_session_record": owner_session_record,
            }
        )

    # Worktree-isolation state: populated inside the try once the child's
    # task id is known; the default no-op keeps every early error path safe.
    _worktree_info: Optional[Dict[str, str]] = None

    def _attach_worktree(entry_dict: Dict[str, Any]) -> None:
        """Inspect + prune the child worktree, reporting into the entry."""
        if _worktree_info is None:
            return
        try:
            from tools import subagent_worktree

            entry_dict["worktree"] = (
                subagent_worktree.finalize_subagent_worktree(_worktree_info)
            )
        except Exception as e:
            # finalize is written hard not to raise, but if it ever does the
            # state is unknown — emit the SAME schema the parent expects,
            # flagged, via the shared factory so the two producers of this
            # payload can never drift.
            logger.warning("worktree finalize failed: %s", e)
            try:
                from tools import subagent_worktree as _sw

                entry_dict["worktree"] = _sw.unproven_worktree_payload(
                    _worktree_info, f"finalize raised: {e}"
                )
            except Exception:
                # Import itself failed — inline the same shape rather than
                # dropping the flag (the parent must still see the warning).
                entry_dict["worktree"] = {
                    "path": _worktree_info.get("path", ""),
                    "branch": _worktree_info.get("branch", ""),
                    "commits": 0,
                    "dirty": False,
                    "pruned": False,
                    "inspection_failed": True,
                    "note": (
                        f"worktree finalize raised ({e}) and the reporting "
                        "helper was unavailable: 'commits' and 'dirty' are "
                        "UNKNOWN, not zero/clean. Inspect "
                        f"{_worktree_info.get('path', '')} before assuming "
                        "no work."
                    ),
                }

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
        # Seed the child's session-cwd record from the parent's (cwd rearch):
        # children share the parent's container, and today they inherit the
        # parent's live env.cwd implicitly. Seeding at spawn preserves that
        # starting directory while keeping the child's subsequent `cd`s
        # isolated in its own record (a child's cd no longer bleeds back into
        # the parent once readers flip to the record store).
        try:
            from tools.terminal_tool import (
                get_session_cwd,
                record_session_cwd,
                register_container_alias,
            )

            record_session_cwd(child_task_id, get_session_cwd(parent_task_id))
            # Per-session container isolation (docker + container_persistent:
            # false) keys containers by session task_id. The child must share
            # the PARENT's container — register the alias so the child's
            # task_id resolves to the parent's container key.
            register_container_alias(child_task_id, parent_task_id)
        except Exception as e:
            logger.debug("Child cwd seed failed: %s", e)

        # Opt-in worktree isolation (delegation.worktree_isolation, inspired
        # by Muse Code's --subagent-worktree-isolation): give this child its
        # own git worktree branched from the parent repo's HEAD, and start its
        # terminal there. Git-only and local-backend-only; any failure
        # degrades silently to the shared-workspace behavior above.
        if _get_worktree_isolation():
            try:
                from tools import subagent_worktree

                if subagent_worktree.local_backend_active():
                    _parent_cwd = None
                    try:
                        from tools.terminal_tool import get_session_cwd as _gsc

                        _parent_cwd = _gsc(parent_task_id)
                    except Exception:
                        pass
                    _worktree_info = subagent_worktree.create_subagent_worktree(
                        _parent_cwd or _resolve_workspace_hint(parent_agent),
                        subagent_id=_subagent_id,
                    )
                else:
                    logger.debug(
                        "worktree isolation skipped: non-local terminal backend"
                    )
            except Exception as e:
                logger.debug("worktree isolation setup failed: %s", e)
            if _worktree_info is not None:
                try:
                    from tools.terminal_tool import record_session_cwd as _rsc

                    _rsc(child_task_id, _worktree_info["path"])
                except Exception as e:
                    logger.debug("worktree cwd seed failed: %s", e)
                # The child's context is already built; carry the isolation
                # contract on the goal message instead (same turn, no
                # system-prompt mutation).
                from tools.subagent_worktree import build_worktree_context_note

                goal = goal + build_worktree_context_note(_worktree_info)

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
            from agent.delegation_context import delegated_child_context

            with delegated_child_context(str(getattr(child, "session_id", "") or "")):
                return child.run_conversation(
                    user_message=goal,
                    task_id=child_task_id,
                    stream_callback=_relay_child_text,
                )

        _child_context = contextvars.copy_context()
        _child_future = _timeout_executor.submit(
            _child_context.run,
            _run_with_thread_capture,
        )
        try:
            result = _child_future.result(timeout=child_timeout)
        except Exception as _timeout_exc:
            # No consumer boundary remains once this owner stops waiting for
            # the child. Close acceptance before any completion callback and
            # retain steer text that won the race with this failure/timeout.
            _late_pending_steer = (
                _close_subagent_steering(_subagent_id, child) if _subagent_id else None
            )
            # Signal the child to stop so its thread can exit cleanly.
            try:
                interrupted = child is not None and request_hard_interrupt(child)
                if not interrupted and child is not None and hasattr(child, "_interrupt_requested"):
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
                        f"stuck on a slow API call, tool call, or unresponsive "
                        f"network request."
                    )
                    if diagnostic_path:
                        _err += f" Diagnostic: {diagnostic_path}"
            else:
                _err = str(_timeout_exc)

            _error_entry = {
                "task_index": task_index,
                "status": "timeout" if is_timeout else "error",
                "summary": None,
                "error": _err,
                "exit_reason": "timeout" if is_timeout else "error",
                "api_calls": child_api_calls,
                "duration_seconds": duration,
                "timeout_seconds": child_timeout if is_timeout else None,
                "timed_out_after_seconds": duration if is_timeout else None,
                "timeout_phase": (
                    "before_first_llm_call" if is_timeout and child_api_calls == 0
                    else "after_llm_calls" if is_timeout
                    else None
                ),
                "_child_role": getattr(child, "_delegate_role", None),
                "diagnostic_path": diagnostic_path,
            }
            if _late_pending_steer:
                _error_entry["missed_steer"] = _late_pending_steer
                _error_entry["error"] += (
                    " [steer did not land before the subagent stopped: "
                    f"{_late_pending_steer}]"
                )
            _attach_worktree(_error_entry)
            return _error_entry
        finally:
            # Shut down executor without waiting — if the child thread
            # is stuck on blocking I/O, wait=True would hang forever.
            _timeout_executor.shutdown(wait=False)

        # T1-24: structured-output contract validation + ONE bounded retry.
        # Runs only when a schema was attached at dispatch; schema-less
        # delegations take none of these branches and their result entry
        # stays byte-identical (wire-shape pinning).
        # Pattern from: github/copilot-cli ctx.agent(prompt, {schema}) —
        # PATTERN ONLY, no code copied.
        _output_schema = getattr(child, "_delegate_output_schema", None)
        _schema_valid: Optional[bool] = None
        _schema_errors: List[str] = []
        _schema_retries = 0
        if isinstance(_output_schema, dict):
            from tools.delegation_output_schema import (
                build_retry_message,
                validate_output,
            )

            _first_text = result.get("final_response") or ""
            _schema_valid, _schema_errors = validate_output(
                _first_text, _output_schema
            )
            if (
                not _schema_valid
                and _first_text.strip()
                and not result.get("interrupted", False)
            ):
                # Exactly one retry turn, carrying the validation errors
                # verbatim (no schema re-paste — the child already holds
                # the contract in its context).
                _schema_retries = 1
                _retry_result = None
                try:
                    _retry_result = child.run_conversation(
                        user_message=build_retry_message(_schema_errors),
                        task_id=child_task_id,
                        stream_callback=_relay_child_text,
                    )
                except Exception as _retry_exc:
                    logger.warning(
                        "Subagent %d schema-retry turn failed: %s",
                        task_index,
                        _retry_exc,
                    )
                if isinstance(_retry_result, dict):
                    _retry_text = _retry_result.get("final_response") or ""
                    if _retry_text.strip():
                        result["final_response"] = _retry_text
                    try:
                        result["api_calls"] = int(
                            result.get("api_calls", 0) or 0
                        ) + int(_retry_result.get("api_calls", 0) or 0)
                    except (TypeError, ValueError):
                        pass
                    _retry_messages = _retry_result.get("messages")
                    if isinstance(_retry_messages, list) and isinstance(
                        result.get("messages"), list
                    ):
                        result["messages"] = result["messages"] + _retry_messages
                    _schema_valid, _schema_errors = validate_output(
                        _retry_text, _output_schema
                    )

        # Linearization boundary for registry steering. From this point on the
        # child cannot consume another steer. Closing under the registry lock
        # either rejects a concurrent caller or drains every previously accepted
        # exact text into the result before callbacks/result assembly can run.
        _late_pending_steer = (
            _close_subagent_steering(_subagent_id, child) if _subagent_id else None
        )
        if _late_pending_steer:
            _existing_pending = result.get("pending_steer")
            result["pending_steer"] = (
                f"{_existing_pending}\n{_late_pending_steer}"
                if isinstance(_existing_pending, str) and _existing_pending
                else _late_pending_steer
            )

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
                        arguments = fn.get("arguments", "")
                        entry_t = {
                            "tool": fn.get("name", "unknown"),
                            "args_bytes": len(arguments),
                            "input_summary": _summarize_tool_arguments(arguments),
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
            # Explicit, parent-visible truncation flag. A subagent that
            # exhausts its per-child iteration budget still returns a summary,
            # so `status` stays "completed" (see above) — without this the
            # parent can't tell truncated-but-summarized from cleanly-finished
            # work except by parsing the summary prose. exit_reason is computed
            # authoritatively from the child's `completed` flag.
            "truncated": exit_reason == "max_iterations",
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
        # Per-delegation spend, serialized back to the model alongside
        # tokens/api_calls so the parent can see what each delegation cost.
        # Mirrors _child_cost_usd (which is stripped pre-serialization and
        # only feeds the parent session rollup).
        # Inspired by: Perplexity Agent API result shape (idea-level).
        entry["cost_usd"] = round(entry["_child_cost_usd"], 6)
        _cost_status = getattr(child, "session_cost_status", None)
        entry["cost_status"] = (
            _cost_status if isinstance(_cost_status, str) and _cost_status
            else "unknown"
        )
        if status == "failed":
            entry["error"] = result.get("error", "Subagent did not produce a response.")

        # T1-24: schema-validation outcome — emitted ONLY when a schema was
        # requested, so legacy (schema-less) payloads keep their exact shape.
        if isinstance(_output_schema, dict):
            entry["schema_valid"] = bool(_schema_valid)
            if _schema_retries:
                entry["schema_retries"] = _schema_retries
            if not _schema_valid and _schema_errors:
                entry["schema_errors"] = _schema_errors

        # A steer that queued after the child's final assistant turn had no
        # tool batch left to drain into.  The finalizer hands the undelivered
        # text back (turn_finalizer.py "pending_steer"); retain it here so the
        # parent sees the steer was MISSED rather than silently absorbed —
        # steer_subagent() returning True means "queued", and this is where a
        # queued-but-never-delivered steer gets named.
        _missed_steer = result.get("pending_steer")
        if isinstance(_missed_steer, str) and _missed_steer.strip():
            entry["missed_steer"] = _missed_steer
            _miss_note = (
                "[steer did not land — the subagent finished before it could "
                f"be delivered: {_missed_steer}]"
            )
            entry["summary"] = f"{summary}\n\n{_miss_note}" if summary else _miss_note

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

        _attach_worktree(entry)
        return entry

    except Exception as exc:
        _late_pending_steer = (
            _close_subagent_steering(_subagent_id, child) if _subagent_id else None
        )
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
        _error_entry = {
            "task_index": task_index,
            "status": "error",
            "summary": None,
            "error": str(exc),
            "api_calls": 0,
            "duration_seconds": duration,
            "_child_role": getattr(child, "_delegate_role", None),
        }
        if _late_pending_steer:
            _error_entry["missed_steer"] = _late_pending_steer
            _error_entry["error"] += (
                " [steer did not land before the subagent stopped: "
                f"{_late_pending_steer}]"
            )
        # _attach_worktree defaults to a no-op when isolation never engaged.
        _attach_worktree(_error_entry)
        return _error_entry

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
            _unregister_subagent(_subagent_id, agent=child)

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

        # The AIAgent turn boundary normally closes the child scope itself. This
        # fallback covers failures before that boundary starts, but must not pop
        # a scope while a timed-out child worker is still unwinding.
        try:
            from agent import relay_runtime

            runtime = relay_runtime.get_runtime(create=False)
            child_session_id = str(getattr(child, "session_id", "") or "")
            child_turn_is_active = relay_runtime.SESSION_COORDINATOR.has_active_turn(
                profile_key=relay_runtime.current_profile_key(),
                session_id=child_session_id,
            )
            if runtime is not None and child_session_id and not child_turn_is_active:
                runtime.unregister_subagent({"child_session_id": child_session_id})
        except Exception:
            logger.debug("Failed to close child Relay session after delegation")


_PARENT_FINALIZATION_LOCK_GUARD = threading.Lock()
_PARENT_FINALIZATION_FALLBACK_LOCK = threading.RLock()
_CHILD_CONSTRUCTION_LOCK = threading.RLock()


def _build_child_preserving_parent_tools(**kwargs):
    """Build a child without leaking its resolved toolset into the parent."""
    import model_tools

    with _CHILD_CONSTRUCTION_LOCK:
        parent_tool_names = list(model_tools._last_resolved_tool_names)
        try:
            child = _build_child_agent(**kwargs)
        finally:
            model_tools._last_resolved_tool_names = parent_tool_names
    child._delegate_saved_tool_names = parent_tool_names
    return child


def _parent_finalization_lock(parent_agent) -> threading.RLock:
    """Return the per-parent lock that serializes lifecycle side effects."""
    if parent_agent is None:
        return _PARENT_FINALIZATION_FALLBACK_LOCK
    lock = getattr(parent_agent, "_subagent_finalization_lock", None)
    if lock is not None:
        return lock
    with _PARENT_FINALIZATION_LOCK_GUARD:
        lock = getattr(parent_agent, "_subagent_finalization_lock", None)
        if lock is None:
            lock = threading.RLock()
            try:
                setattr(parent_agent, "_subagent_finalization_lock", lock)
            except Exception:
                return _PARENT_FINALIZATION_FALLBACK_LOCK
    return lock


def _finalize_child_results(
    results: List[Dict[str, Any]],
    task_list: List[Dict[str, Any]],
    children: List[tuple[int, Dict[str, Any], Any]],
    parent_agent,
) -> None:
    """Apply host-owned summary, memory, hook, and cost contracts once."""
    with _parent_finalization_lock(parent_agent):
        _apply_summary_budget(results, parent_agent)
        child_by_index = {index: child for index, _task, child in children}

        if parent_agent and getattr(parent_agent, "_memory_manager", None):
            for entry in results:
                try:
                    task_index = entry.get("task_index", -1)
                    task_goal = (
                        task_list[task_index]["goal"]
                        if isinstance(task_index, int)
                        and 0 <= task_index < len(task_list)
                        else ""
                    )
                    child = child_by_index.get(task_index)
                    parent_agent._memory_manager.on_delegation(
                        task=task_goal,
                        result=entry.get("summary", "") or "",
                        child_session_id=getattr(child, "session_id", ""),
                    )
                except Exception:
                    pass

        parent_session_id = getattr(parent_agent, "session_id", None)
        try:
            from hermes_cli.plugins import invoke_hook as invoke_hook
        except Exception:
            invoke_hook = None

        children_cost_total = 0.0
        for entry in results:
            child_role = entry.pop("_child_role", None)
            child_cost = entry.pop("_child_cost_usd", 0.0)
            try:
                if child_cost:
                    children_cost_total += float(child_cost)
            except (TypeError, ValueError):
                pass
            if invoke_hook is None:
                continue
            try:
                child_index = entry.get("task_index", -1)
                child = child_by_index.get(child_index)
                invoke_hook(
                    "subagent_stop",
                    parent_session_id=parent_session_id,
                    parent_turn_id=getattr(parent_agent, "_current_turn_id", "") or "",
                    child_session_id=getattr(child, "session_id", None),
                    child_role=child_role,
                    child_summary=entry.get("summary"),
                    child_status=entry.get("status"),
                    tool_call_history=_subagent_stop_tool_call_history(
                        entry.get("tool_trace")
                    ),
                    duration_ms=int((entry.get("duration_seconds") or 0) * 1000),
                )
            except Exception:
                logger.debug("subagent_stop hook invocation failed", exc_info=True)

        if children_cost_total > 0.0:
            try:
                current = float(
                    getattr(parent_agent, "session_estimated_cost_usd", 0.0) or 0.0
                )
                parent_agent.session_estimated_cost_usd = current + children_cost_total
                if getattr(parent_agent, "session_cost_source", "none") in {
                    None,
                    "",
                    "none",
                }:
                    parent_agent.session_cost_source = "subagent"
                if getattr(parent_agent, "session_cost_status", "unknown") in {
                    None,
                    "",
                    "unknown",
                }:
                    parent_agent.session_cost_status = "estimated"
            except Exception:
                logger.debug("Subagent cost rollup failed", exc_info=True)


def _run_child_lifecycle(
    task_index: int,
    goal: str,
    child=None,
    parent_agent=None,
) -> Dict[str, Any]:
    """Run one child and apply the same host lifecycle used by delegate_task."""
    result = _run_single_child(task_index, goal, child, parent_agent)
    result.setdefault("task_index", task_index)
    task = {"goal": goal}
    _finalize_child_results(
        [result],
        [{"goal": ""} for _ in range(task_index)] + [task],
        [(task_index, task, child)],
        parent_agent,
    )
    return result


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


# Placeholder shapes for batch goal validation: bare 'TODO', bare 'task N'
# labels, or goals still carrying unexpanded template markers.
#
# The marker regex is deliberately NARROW: it only fires on snake_case /
# space-separated placeholder identifiers (`<feature_name>`, `{file path}`,
# `<FEATURE-NAME>`) — the shape LLM templates actually leave behind. Bare
# single-word brackets are left alone because legitimate coding goals are
# full of them: generics (`Vec<T>`, `Result<String>`), HTML tags (`<div>`),
# JSON/dict snippets (`{"key": 1}`), glob braces (`{a,b}`), and f-string
# style (`{i}`) must never be rejected (post-merge audit of #81141).
_PLACEHOLDER_GOAL_RE = re.compile(r"^(todo|task\s*\d+)$", re.IGNORECASE)
_TEMPLATE_MARKER_RE = re.compile(
    r"<[A-Za-z][A-Za-z0-9]*(?:[ _-][A-Za-z0-9]+)+>"
    r"|\{[A-Za-z][A-Za-z0-9]*(?:[ _-][A-Za-z0-9]+)+\}"
)
_MIN_BATCH_GOAL_LEN = 10


def _validate_batch_tasks(task_list: List[Dict[str, Any]]) -> Optional[str]:
    """Validate a tasks=[...] batch beyond per-task goal presence.

    Returns an actionable error string, or None when the batch is valid.
    Batch-only by design: the single-`goal` form legitimately uses short
    goals, so these checks must never run on it.

    Duplicate goals are deliberately NOT rejected: identical-goal fan-outs
    are a legitimate pattern (best-of-N / ensemble sampling), and blocking
    them broke real workflows (post-merge audit of #81141).
    """
    if len(task_list) < 2:
        return (
            "Batch mode requires at least 2 tasks. For a single task, use "
            "the `goal` parameter instead of `tasks`: "
            'delegate_task(goal="...", context="...").'
        )

    for i, task in enumerate(task_list):
        goal = str(task.get("goal", "")).strip()
        normalized = " ".join(goal.lower().split())

        if _PLACEHOLDER_GOAL_RE.match(normalized):
            return (
                f"Task {i} has a placeholder goal ({goal!r}). Replace it "
                "with a specific, self-contained description of what the "
                "subagent should accomplish."
            )
        marker = _TEMPLATE_MARKER_RE.search(goal)
        if marker:
            return (
                f"Task {i} goal contains an unexpanded template marker "
                f"({marker.group(0)!r}). Substitute the real value before "
                "calling delegate_task — subagents cannot resolve "
                "placeholders."
            )
        if len(goal) < _MIN_BATCH_GOAL_LEN:
            return (
                f"Task {i} goal is too short ({goal!r}). Write a specific, "
                "self-contained goal of at least "
                f"{_MIN_BATCH_GOAL_LEN} characters so the subagent knows "
                "exactly what to do."
            )
    return None


def delegate_task(
    goal: Optional[str] = None,
    context: Optional[str] = None,
    tasks: Optional[List[Dict[str, Any]]] = None,
    max_iterations: Optional[int] = None,
    role: Optional[str] = None,
    background: Optional[bool] = None,
    output_schema: Optional[Dict[str, Any]] = None,
    action: Optional[str] = None,
    subagent_id: Optional[str] = None,
    message: Optional[str] = None,
    parent_agent=None,
    credentials_cfg: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Spawn one or more child agents to handle delegated tasks, or control
    already-running ones.

    Spawn modes (action='spawn' or omitted):
      - Single: provide goal (+ optional context and role)
      - Batch:  provide tasks array [{goal, context, role}, ...]

    Control modes (synchronous, never backgrounded):
      - action='list'  -> live children of this conversation's spawn tree
      - action='steer' -> queue course-correction text into a running child
                          (subagent_id + message)
      - action='stop'  -> interrupt a running child early (subagent_id)

    The 'role' parameter controls whether a child can further delegate:
    'leaf' (default) cannot; 'orchestrator' retains the delegation
    toolset and can spawn its own workers, bounded by
    delegation.max_spawn_depth.  Per-task role beats the top-level one.

    Returns JSON with results array, one entry per task.
    """
    if parent_agent is None:
        return tool_error("delegate_task requires a parent agent context.")

    # ── Control plane: list/steer/stop run synchronously and return here.
    # They never spawn, so they bypass the pause gate, depth limit, and the
    # async dispatch machinery entirely.
    normalized_action = (action or "").strip().lower()
    if normalized_action in _CONTROL_ACTIONS:
        return _handle_control_action(
            normalized_action, subagent_id, message, parent_agent
        )
    if normalized_action and normalized_action != "spawn":
        return tool_error(
            f"Unknown action '{action}'. Use spawn (default), list, steer, or stop."
        )

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
    # batches. A batch is dispatched as ONE async unit: the whole fan-out runs
    # on the daemon executor, joins on every child (see _execute_and_aggregate
    # / dispatch_async_delegation_batch), and pushes a SINGLE completion event
    # carrying the consolidated per-task results. It re-enters the conversation
    # as one message once ALL children finish — the chat is not blocked while
    # they run.
    background = is_truthy_value(background, default=False) if background is not None else False

    # Depth limit — configurable via delegation.max_spawn_depth,
    # default 2 for parity with the original MAX_DEPTH constant.
    depth = getattr(parent_agent, "_delegate_depth", 0)
    max_spawn = _get_max_spawn_depth()
    if depth >= max_spawn:
        return tool_error(
            f"Delegation depth limit reached (depth={depth}, "
            f"max_spawn_depth={max_spawn}). Raise "
            f"delegation.max_spawn_depth in config.yaml if deeper "
            f"nesting is required (no hard ceiling, but each level "
            f"multiplies API cost)."
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
    #
    # ``credentials_cfg`` (internal callers only — never model-facing) is a
    # per-call override shaped like the delegation config section
    # ({provider, model, base_url, api_key, api_mode}); the /review engine
    # uses it to route its reviewer subagent onto ``auxiliary.review``
    # without touching the global delegation pin.
    try:
        creds = _resolve_delegation_credentials(
            credentials_cfg if credentials_cfg else cfg, parent_agent
        )
    except ValueError as exc:
        return tool_error(str(exc))

    # Normalize to task list
    max_children = _get_max_concurrent_children()
    recovered_tasks, tasks_error = _recover_tasks_from_json_string(tasks)
    if tasks_error:
        return tool_error(tasks_error)
    if recovered_tasks is not None:
        tasks = recovered_tasks

    # Small models frequently emit an empty tasks array ([]) alongside a
    # single goal. Treat that as "no batch" instead of letting the batch
    # quality gate below reject the goal-derived single task ("Batch mode
    # requires at least 2 tasks") — the intent is unambiguous.
    if isinstance(tasks, list) and not tasks:
        tasks = None

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
        single_task: Dict[str, Any] = {"goal": goal, "context": context, "role": top_role}
        if output_schema is not None:
            single_task["output_schema"] = output_schema
        task_list = [single_task]
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

    # Batch-only quality gate: catch malformed fan-outs (placeholder goals,
    # unexpanded multi-word template markers, 1-task batches) before any
    # child is spawned.  The single-`goal` form is deliberately exempt —
    # short goals are valid there.  Duplicate goals are allowed (best-of-N).
    # Inspired by: MoonshotAI/kimi-code agent-swarm.md validation rules (MIT).
    if tasks is not None and isinstance(tasks, list):
        batch_error = _validate_batch_tasks(task_list)
        if batch_error:
            return tool_error(batch_error)

    # T1-24: coerce/validate optional per-task output_schema up front so a
    # malformed schema fails the whole call loudly instead of spawning
    # children that can never satisfy their contract. Runs AFTER the
    # existing goal checks; schema-less tasks resolve to None and take no
    # new code paths downstream.
    from tools.delegation_output_schema import coerce_output_schema

    task_schemas: List[Optional[Dict[str, Any]]] = []
    for i, task in enumerate(task_list):
        raw_schema = task.get("output_schema")
        if raw_schema is None and len(task_list) == 1 and output_schema is not None:
            raw_schema = output_schema
        coerced_schema, schema_err = coerce_output_schema(raw_schema)
        if schema_err:
            return tool_error(f"Task {i} output_schema invalid: {schema_err}")
        task_schemas.append(coerced_schema)

    overall_start = time.monotonic()
    results = []

    n_tasks = len(task_list)
    # Track goal labels for progress display (truncated for readability)
    task_labels = [t["goal"][:40] for t in task_list]

    # Live transcripts: one pre-headered append-only log per task under
    # cache/delegation/live/<delegation_id>/task-<n>.log so the caller can
    # tail each child's operations while it runs (side-channel only — zero
    # effect on message content or prompt caching). Best-effort: on failure
    # live_paths is empty and delegation proceeds exactly as before.
    from tools.delegation_live_log import (
        create_live_transcripts,
        update_manifest_statuses,
        wrap_progress_callback,
    )

    live_deleg_id, live_writers, live_paths = create_live_transcripts(
        task_list, context, model=creds.get("model"), provider=creds.get("provider")
    )

    # Capture the ORIGINATING session's wake target BEFORE any child agent is
    # constructed: _build_child_agent() -> AIAgent() -> agent_init calls
    # set_current_session_id(child.session_id), which clobbers the
    # HERMES_SESSION_ID ContextVar and os.environ with the subagent's internal
    # id before the background-dispatch code below would read it. The
    # request-scoped chat_id binding (the raw X-Hermes-Session-Id on
    # api_server) is untouched by child construction, so read it here and
    # thread it through the dispatch.
    from tools.async_delegation import _current_origin_session_id

    _origin_wake_sid = _current_origin_session_id()
    try:
        from gateway.session_context import get_session_env

        _origin_ui_session_id = get_session_env("HERMES_UI_SESSION_ID", "")
    except Exception:
        _origin_ui_session_id = ""
    _origin_owner_transport, _origin_owner_session_record = (
        _capture_gateway_steer_authority(_origin_ui_session_id)
    )

    # Build all child agents on the main thread (thread-safe construction).
    # _build_child_preserving_parent_tools saves/restores the parent's
    # resolved tool names around each construction under a lock, so child
    # toolset resolution never leaks into the parent (shared with the plugin
    # subagent-lifecycle API).
    children = []
    for i, t in enumerate(task_list):
        # Per-task role beats top-level; normalise again so unknown
        # per-task values warn and degrade to leaf uniformly.
        effective_role = _normalize_role(t.get("role") or top_role)
        # T1-24: schema'd tasks get the contract appended to their context
        # so the child knows the expected output shape before it starts.
        _task_schema = task_schemas[i] if i < len(task_schemas) else None
        _child_context = t.get("context")
        if _task_schema is not None:
            from tools.delegation_output_schema import append_output_contract

            _child_context = append_output_contract(_child_context, _task_schema)
        try:
            child = _build_child_preserving_parent_tools(
                task_index=i,
                goal=t["goal"],
                context=_child_context,
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
        except ValueError as exc:
            # Explicit-pin preflight failures (e.g. pinned delegation.command
            # missing from PATH) refuse the spawn loudly (#80450).
            return tool_error(str(exc))
        # Attach the validated schema for the completion-side validation
        # hook in _run_single_child. Absent (None) on schema-less tasks.
        if _task_schema is not None:
            try:
                child._delegate_output_schema = _task_schema
            except Exception:
                logger.debug("Could not attach output schema to child %d", i)
        # Tee the child's progress events into its live transcript log.
        # wrap_progress_callback preserves the inner callback contract
        # (including the _flush attribute) and never lets writer failures
        # reach the agent loop. When no parent display exists the inner
        # callback is None and the wrapper still records events.
        _writer = live_writers[i] if i < len(live_writers) else None
        if _writer is not None:
            child.tool_progress_callback = wrap_progress_callback(
                getattr(child, "tool_progress_callback", None), _writer
            )
            child._live_transcript_path = str(_writer.path)
        # Delegation identity for the live registry + process-notification
        # attribution (child-started background processes report under it).
        if live_deleg_id:
            setattr(child, "_delegation_id", live_deleg_id)
        children.append((i, t, child))

    def _execute_and_aggregate(*, honor_parent_interrupt: bool = True) -> dict:
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
            result = _run_single_child(
                _i,
                _t["goal"],
                child,
                parent_agent,
                owner_session_id=_origin_ui_session_id or None,
                owner_transport=_origin_owner_transport,
                owner_session_record=_origin_owner_session_record,
            )
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
                    child_context = contextvars.copy_context()
                    future = executor.submit(
                        child_context.run,
                        _run_single_child,
                        task_index=i,
                        goal=t["goal"],
                        child=child,
                        parent_agent=parent_agent,
                        owner_session_id=_origin_ui_session_id or None,
                        owner_transport=_origin_owner_transport,
                        owner_session_record=_origin_owner_session_record,
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
                    if (
                        honor_parent_interrupt
                        and getattr(parent_agent, "_interrupt_requested", False) is True
                    ):
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
        _finalize_child_results(results, task_list, children, parent_agent)

        total_duration = round(time.monotonic() - overall_start, 2)

        # Close out the live transcripts: terminal marker per task + manifest
        # status update. The files are retained (retention pruning happens on
        # future dispatches) — they double as the full-fidelity operational
        # record alongside the summary spill files.
        for entry in results:
            _idx = entry.get("task_index", -1)
            _w = (
                live_writers[_idx]
                if isinstance(_idx, int) and 0 <= _idx < len(live_writers)
                else None
            )
            if _w is not None:
                try:
                    _w.finalize(entry)
                except Exception:
                    logger.debug("Live transcript finalize failed", exc_info=True)
                if _idx < len(live_paths):
                    entry["live_transcript"] = live_paths[_idx]
        update_manifest_statuses(live_deleg_id, results)

        combined: Dict[str, Any] = {
            "results": results,
            "total_duration_seconds": total_duration,
        }
        if live_paths:
            combined["live_transcripts"] = list(live_paths)
        return combined

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

        # Finite sessions cannot route a detached subagent result back to the
        # agent after their turn/process ends. This includes stateless HTTP
        # requests (#10760) and one-shot Kanban workers (#63169). Fall back to
        # SYNCHRONOUS execution so the result returns in this same turn instead
        # of handing out a handle with no durable consumer. Mirrors the
        # pool-at-capacity inline fallback below.
        try:
            from gateway.session_context import async_delivery_supported
            _async_ok = async_delivery_supported()
        except Exception:
            _async_ok = True

        _wake_sid = ""
        if not _async_ok:
            # The adapter itself cannot push, but if a raw session id is
            # bound (the API server always binds one — see
            # ApiServerAdapter._bind_api_server_session), gateway.wake can
            # still reach the session by self-POSTing /v1/chat/completions
            # with that id in X-Hermes-Session-Id once the batch completes.
            # Only fall back to forced-sync execution when there is truly no
            # session id to wake. Uses the origin captured before child
            # construction (see _origin_wake_sid above) — reading
            # HERMES_SESSION_ID here would return the subagent's internal id.
            _wake_sid = _origin_wake_sid
            if _wake_sid:
                logger.info(
                    "delegate_task: async delivery unsupported on this "
                    "session, but a session id is bound (%s) — dispatching "
                    "in the background and waking the session via self-post "
                    "when it completes instead of forcing synchronous "
                    "execution.",
                    _wake_sid,
                )
                _async_ok = True

        if not _async_ok:
            logger.info(
                "delegate_task: async delivery unsupported on this session "
                "runtime; running the batch synchronously instead."
            )
            _sync_result = _execute_and_aggregate()
            if isinstance(_sync_result, dict):
                _sync_result["note"] = (
                    "background=true is not available in this session — it cannot "
                    "receive a detached subagent result after the turn ends (a "
                    "one-shot runner such as `hermes -z`, a cron job, a Kanban "
                    "worker, or a stateless HTTP endpoint). The subagent(s) ran "
                    "SYNCHRONOUSLY and the result is included above."
                )
            return json.dumps(_sync_result, ensure_ascii=False)

        _session_key = get_current_session_key(default="")
        try:
            from gateway.session_context import get_session_env

            _source = get_session_env("HERMES_SESSION_SOURCE", "")
            # Refresh from the same task-local source when available, but retain
            # the immutable value captured before child construction otherwise.
            _origin_ui_session_id = (
                get_session_env("HERMES_UI_SESSION_ID", "") or _origin_ui_session_id
            )
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
            _source = ""
        if not _session_key:
            # CLI (single-process) path: the approval contextvar is only bound
            # during gateway/TUI turns and HERMES_SESSION_KEY is not in the CLI
            # environment, so the key resolves empty here. Since #64240 the CLI
            # drains completions through a positive-ownership filter keyed on
            # the durable AIAgent.session_id — an empty session_key would fail
            # closed and the CLI could never claim its own completions, while
            # a restored foreign event with an empty key could leak into any
            # unfiltered consumer (#64484). Stamp the parent's durable session
            # id instead; compression rotations are handled on the drain side
            # via resolve_resume_session_id lineage resolution.
            _agent_session_id = str(getattr(parent_agent, "session_id", "") or "")
            if _agent_session_id:
                _session_key = _agent_session_id
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
            # This batch is detached from the foreground turn. Its lifecycle is
            # owned by the async registry and cancelled only via _batch_interrupt.
            return _execute_and_aggregate(honor_parent_interrupt=False)

        def _batch_interrupt():
            for _c in _child_agents:
                try:
                    interrupted = request_hard_interrupt(_c, "Async delegation cancelled")
                    if not interrupted and hasattr(_c, "_interrupt_requested"):
                        _c._interrupt_requested = True
                except Exception:
                    pass

        def _batch_progress():
            # Progress token for the async registry's stale monitor: the
            # combined (api_call_count, current_tool, last_activity_ts) of
            # every child. last_activity_ts is ticked by _touch_activity on
            # every streamed chunk ("receiving stream response"), every tool
            # transition, and every API-call start/completion — so a child
            # streaming a long response is alive even though api_call_count
            # only advances when the call completes (same liveness signal as
            # the compaction inactivity budget, PR #71508). A fully frozen
            # token past the stale threshold means the detached batch is
            # wedged (e.g. stuck inside the first model API call — #60203).
            # in_tool=True while ANY child is inside a tool so legitimately
            # slow tools get the higher staleness ceiling, mirroring the
            # sync-path heartbeat monitor.
            parts = []
            in_tool = False
            for _c in _child_agents:
                try:
                    _summary = _c.get_activity_summary()
                    _tool = _summary.get("current_tool")
                    parts.append(
                        (
                            _summary.get("api_call_count", 0),
                            _tool,
                            _summary.get("last_activity_ts"),
                        )
                    )
                    in_tool = in_tool or bool(_tool)
                except Exception:
                    parts.append(None)
            return tuple(parts), in_tool

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
            origin_session_id=_wake_sid,
            parent_session_id=_parent_session_id,
            runner=_batch_runner,
            interrupt_fn=_batch_interrupt,
            max_async_children=_get_max_async_children(),
            # Reuse the live-transcript directory's id (when created) so the
            # returned delegation_id matches cache/delegation/live/<id>/.
            delegation_id=live_deleg_id,
            progress_fn=_batch_progress,
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
            _sids = [
                getattr(_c, "_subagent_id", None) for _c in _child_agents
            ]
            if any(isinstance(s, str) and s for s in _sids):
                payload["subagent_ids"] = _sids
                payload["control_hint"] = (
                    "While a child runs you can orchestrate it live with this "
                    "same tool: delegate_task(action='list') to see live "
                    "children, action='steer' with subagent_id + message to "
                    "redirect one, action='stop' with subagent_id to end one "
                    "early."
                )
            if live_paths:
                payload["live_transcripts"] = list(live_paths)
                payload["live_transcripts_hint"] = (
                    "Each subagent streams a human-readable transcript of its "
                    "operations to the file listed above (append-only, one per "
                    "task). Read or `tail -f` these paths at any time to watch "
                    "a child work while it runs."
                )
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

    # A pinned ACP transport command must exist — refuse the spawn loudly
    # rather than letting the child silently fall back to another transport
    # (#80450).
    pinned_command = runtime.get("command")
    if pinned_command:
        import shutil as _shutil

        if not _shutil.which(pinned_command):
            raise ValueError(
                f"Delegation provider '{configured_provider}' is pinned to the "
                f"'{pinned_command}' command, which was not found on PATH. "
                f"Install it or choose a different delegation provider."
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
    """Compose the delegate_task tool description.

    Deliberately carries ONLY guidance that exists nowhere else in the
    schema. Batch/concurrency limits live in the 'tasks' parameter
    description and the nesting clause lives in the 'role' parameter
    description (both rebuilt per get_definitions() call with the user's
    actual delegation.max_concurrent_children / max_spawn_depth), so the
    top-level text stays static and duplication-free. If you add text
    here, check it is not already stated in a parameter description.
    """
    return (
        "Spawn subagents in isolated contexts; each gets its own conversation, "
        "terminal session, and toolset, and only its final summary returns to "
        "you. Provide 'goal' for a single task or 'tasks' for a parallel batch "
        "(limits and nesting rules are in the parameter descriptions).\n\n"
        "Runs in the background: dispatch returns immediately with live "
        "transcript paths, and the completed result (one consolidated message "
        "for a batch) re-enters the conversation on its own. Do NOT wait or "
        "poll; continue other work.\n\n"
        "LIVE ORCHESTRATION: while children run, this tool also controls "
        "them — action='list' (live children + ids), action='steer' "
        "(subagent_id + message, redirect without stopping), action='stop' "
        "(subagent_id, end early; partial result still returns). Steer when "
        "a live transcript shows a child drifting.\n\n"
        "USE FOR: reasoning-heavy subtasks, work that would flood your context "
        "with intermediate data, or independent parallel workstreams.\n"
        "DO NOT USE FOR (use these instead):\n"
        "- Mechanical multi-step work with no reasoning needed -> execute_code\n"
        "- A single tool call -> call the tool directly\n"
        "- Tasks needing user interaction -> subagents cannot ask questions\n"
        "- Durable work that must survive this session -> cronjob or "
        "terminal(background=True, notify_on_complete=True); /stop, /new, or "
        "process exit discards running subagents.\n\n"
        "RULES:\n"
        "- Children know nothing of this conversation: pass everything needed "
        "via 'context', including any required output language, tone, or "
        "style (e.g. \"respond in Chinese\").\n"
        "- Child summaries are SELF-REPORTS, not verified facts: a child "
        "claiming \"uploaded successfully\" or \"file written\" may be wrong. "
        "For external side effects (uploads, remote writes, publishing), "
        "require a verifiable handle (URL, ID, absolute path) and verify it "
        "yourself — fetch the URL, stat the file, read back the content — "
        "before telling the user the operation succeeded.\n"
        "- Leaf children (the default) cannot call delegate_task, clarify, "
        "memory, send_message, or cronjob; orchestrators regain only "
        "delegate_task.\n"
        "- Children inherit the parent model and fallback chain unless pinned "
        "globally via delegation.provider / delegation.model in config.yaml. "
        "Results are returned as an array, one entry per task."
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
        "When provided, top-level goal/context/role are ignored."
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
                        "output_schema": {
                            "type": "object",
                            "description": (
                                "Optional JSON Schema the subagent's final "
                                "answer must validate against. The child is "
                                "told the contract up front; the parent "
                                "validates the final answer and allows one "
                                "bounded correction retry. The result entry "
                                "gains schema_valid (and schema_errors on "
                                "final failure). Keep schemas forgiving: "
                                "require only fields you will actually read."
                            ),
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
            "output_schema": {
                "type": "object",
                "description": (
                    "Optional JSON Schema for the single-goal form — the "
                    "subagent's final answer must validate against it "
                    "(same semantics as tasks[].output_schema)."
                ),
            },
            "background": {
                "type": "boolean",
                "description": (
                    "DEPRECATED / IGNORED. Top-level single and batch "
                    "delegations run in the background automatically — you do "
                    "not need to (and cannot) opt in or out. A single result or "
                    "consolidated batch result re-enters the conversation when "
                    "the work finishes; just continue working in the meantime. "
                    "Setting this has no effect; the parameter remains only for "
                    "backward compatibility."
                ),
            },
            "action": {
                "type": "string",
                "enum": ["spawn", "list", "steer", "stop"],
                "description": (
                    "Default 'spawn' (omit for normal delegation). Live "
                    "orchestration of running subagents: 'list' shows this "
                    "conversation's live children (ids, goals, status, "
                    "transcript paths); 'steer' queues course-correction text "
                    "into one child (requires subagent_id + message) without "
                    "stopping it; 'stop' ends one child early (requires "
                    "subagent_id) — its partial result still returns as a "
                    "completion message. Control actions return immediately; "
                    "goal/tasks are ignored when action is not 'spawn'."
                ),
            },
            "subagent_id": {
                "type": "string",
                "description": (
                    "Target for action='steer'/'stop'. Ids are returned in the "
                    "spawn dispatch response (subagent_ids) and by "
                    "action='list'."
                ),
            },
            "message": {
                "type": "string",
                "description": (
                    "For action='steer': the course correction. Be directive "
                    "and specific — the child sees it appended to its next "
                    "tool result mid-run (e.g. \"Stop exploring X; focus on Y "
                    "and return early results\")."
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
    batch (the whole batch is one async unit that joins on all children and
    returns one consolidated result). The one
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
        output_schema=args.get("output_schema"),
        action=args.get("action"),
        subagent_id=args.get("subagent_id"),
        message=args.get("message"),
        parent_agent=kw.get("parent_agent"),
    ),
    check_fn=check_delegate_requirements,
    emoji="🔀",
    dynamic_schema_overrides=_build_dynamic_schema_overrides,
)
