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
import hashlib  # P103/MOL-410 — prompt_hash in audit log
import json
import logging

logger = logging.getLogger(__name__)
import os
import re  # P103/MOL-410 — profile name validation + repo-path extraction (#162 review M11/M15)
import string  # P103/MOL-410 — _build_claude_argv field-name validation (#162 review CRITICAL-6)
import sys  # P103/MOL-410 — fail-open stderr emit in _emit_profile_audit
import threading
import time
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
)
from datetime import datetime, timezone  # P103/MOL-410 — audit log timestamps
from pathlib import Path  # P103/MOL-410 — profile path resolution
from typing import Any, Dict, List, Optional, Tuple

# P103/MOL-410 (#162 review M9): graceful degradation if PyYAML uninstalled.
# The default delegate_task path (no profile) MUST keep working even without
# yaml; only profile-elevation requires it.
try:
    import yaml as _p103_yaml
except ImportError:  # pragma: no cover — depends on venv state
    _p103_yaml = None  # type: ignore[assignment]

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


_DEFAULT_CHILD_TIMEOUT_SECONDS = 1800  # P105/MOL-420 P145/MOL-495 P146/MOL-496 — CC subprocess timeout (separate from in-process DEFAULT_CHILD_TIMEOUT=600)


def _get_child_timeout_seconds(profile_cfg: dict | None = None) -> int:  # P105/MOL-420 P145/MOL-495
    """Timeout for CC subprocess children (Tier 1/2 claude -p).

    Resolution order:
      1. profile_cfg["timeout_seconds"] — coding.yaml: 1800 / planning.yaml: 600
      2. DELEGATION_CHILD_TIMEOUT_SECONDS env var — operator override
      3. _DEFAULT_CHILD_TIMEOUT_SECONDS (1800s)

    Separate from _get_child_timeout() (line 376, 600s default for in-process
    subagents).  CC subprocesses run claude -p which includes CI + /review-pr
    cycles and need a longer window.
    """
    if profile_cfg:
        val = profile_cfg.get("timeout_seconds")
        if isinstance(val, (int, float)) and val > 0:
            return int(val)
    raw = os.environ.get("DELEGATION_CHILD_TIMEOUT_SECONDS")
    if raw:
        try:
            val = int(raw)
            if val > 0:
                return val
            logger.warning(
                "DELEGATION_CHILD_TIMEOUT_SECONDS=%r is not > 0; "
                "using default %d",
                raw, _DEFAULT_CHILD_TIMEOUT_SECONDS,
            )
        except ValueError:
            logger.warning(
                "DELEGATION_CHILD_TIMEOUT_SECONDS=%r is not a valid int; "
                "using default %d",
                raw, _DEFAULT_CHILD_TIMEOUT_SECONDS,
            )
    return _DEFAULT_CHILD_TIMEOUT_SECONDS


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


# ============================================================================
# P103/MOL-410 — Coding-profile elevation for delegate_task
# ============================================================================
# Profile-based scoped elevation. delegate_task(profile="coding"|"default", ...)
# loads YAML at ~/.hermes/config/delegate-profiles/<profile>.yaml. Profile
# drives: --allowedTools, --settings allow/deny, --max-budget-usd, --max-turns,
# --add-dir, --append-system-prompt, env-passthrough merge, MCP server inclusion,
# audit verbosity.
#
# Defense-in-depth (additive-only — profiles can ADD allowed ops, NEVER remove
# deny rules): see PATCHES.md§P103 for the 5-layer stack.
#
# HITL session-consent: (profile, repo_path) tuple key, TTL 120 min for
# `coding`. First-use-per-tuple per session prompts via Telegram. v1 returns
# (True, "v1-auto-approve") because the Telegram-prompt subsystem is shared
# with the gateway long-poller (409-conflict if both poll). Full per-call
# HITL is tracked at MOL-450.
#
# Re-applied to runtime via P140/MOL-489 (this patch was originally P103
# but the runtime-side work was rolled back; this re-application restores
# the surface PATCHES.md§P103:6662 documents).
# ============================================================================

_P103_PROFILE_DIR = Path("~/.hermes/config/delegate-profiles").expanduser()
_P103_LOG_DIR = Path("~/.hermes/logs").expanduser()
# Module-level consent cache: (profile_name, repo_path) -> monotonic expiry seconds.
# Single-process scope — the gateway is one process, so this is sufficient
# for session consent. Cache wipes on gateway restart, which is the
# intended TTL boundary. P103/MOL-410 (#162 review I4/M7): keyed off
# time.monotonic(), NOT time.time() — wall-clock skew (NTP correction,
# macOS sleep/wake) would otherwise produce stale-consent or repeat-prompt
# anomalies. Monotonic resets on process boot, which is the desired
# "session" semantic anyway.
_P103_CONSENT_CACHE: Dict[Tuple[str, str], float] = {}
_P103_DEFAULT_CONSENT_TTL_MIN = 120

# P103/MOL-410 (#162 review CRITICAL-1): profile name allowlist. Only
# names matching this regex are accepted by `_load_profile`. Prevents
# `profile="../../etc/passwd"` from reading arbitrary files on disk
# when P114/MOL-442 lands (which makes `profile=` LLM-controllable).
_P103_PROFILE_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")

# P103/MOL-410 (#162 review I1, M2 review-fix from original P103):
# secrets that profiles MUST NEVER passthrough to subprocesses regardless
# of `env_passthrough` declarations. Closes the regression vs the
# documented final P103 surface (PATCHES.md:7277).
_P103_PROFILE_PASSTHROUGH_DENYLIST = frozenset({
    "ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN",
    "DEEPSEEK_API_KEY",
    "GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS",
    "OPENROUTER_API_KEY",
    "KIMI_API_KEY", "MOONSHOT_API_KEY",
    "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
    "TAVILY_API_KEY",
    "GIPHY_API_KEY",
    "VIRUSTOTAL_API_KEY", "AI_DEFENSE_API_KEY",
    "GRANOLA_OAUTH_TOKEN", "GRANOLA_REFRESH_TOKEN",
    "BROWSERBASE_API_KEY",
})

# P103/MOL-410 (#162 review M11): pre-compiled regex for repo-path extraction
# from goal+context. Tighter character class than the prior `[^\s/]+` —
# excludes trailing punctuation (`,;:.)]`) that the wider class swallowed
# into the audit `--add-dir` field. Allows `_-` for repo names like
# `hermes-poc-staging`. Validation fallback at extraction site checks
# `os.path.isdir` to guard against junk that survives the regex.
_P103_REPO_PATH_RE = re.compile(r"(?:~|/Users/[^/]+)/Code/[A-Za-z0-9._-]+")

# P103/MOL-410 (#162 review M8): consent-cache lock. v1-auto-approve doesn't
# need it (no I/O, near-atomic dict op), but locking now removes a footgun
# from the v2 Telegram-prompt wire-up (which would otherwise dual-prompt
# under concurrent reads).
_P103_CONSENT_LOCK = threading.Lock()

# P103/MOL-410 (#162 review CRITICAL-2): audit-write failure counter.
# Increments when _emit_profile_audit's fail-open path fires. Surfaced
# via `gateway.log` logger.error AND this counter so cron-context audit
# loss is observable.
_P103_AUDIT_DROP_COUNT = 0

# P141/MOL-491 (#163 review HIGH-1): iteration-audit write failure counter.
# Same pattern as _P103_AUDIT_DROP_COUNT — surfaces fail-open events that
# would otherwise be silently swallowed by gateway-context stderr eating.
_P141_ITER_AUDIT_DROP_COUNT = 0


def _emit_iteration_audit(
    *,
    exit_reason: str,
    api_calls_val: Any,
    child_obj: Any,
) -> None:
    """P141/MOL-491: per-dispatch iteration audit JSONL writer.

    Extracted into a helper (#163 review CRITICAL-1) so all THREE exit
    paths of `_run_single_child` (timeout, outer-exception, normal) emit
    the same record shape. Previously the inline hook only fired on the
    happy-path exit_reason branches — exactly the wrong subpopulation
    for "is `max_iterations` enough?" since the timeout/error dispatches
    are the highest-pressure cases.

    Fail-open via narrowed exception types (#163 review HIGH-1):
    `OSError`, `ValueError`, `TypeError` — bare `Exception` would mask
    programming bugs (KeyError on dict construction, attribute typos)
    that should surface during development. Audit-write failures route
    through `logger.error` (lands in `gateway.log` per launchd plist —
    survives the gateway's stderr-eating subprocess context) AND
    increment a counter so operators can detect silent gaps.

    Tier-2 fallback to `/tmp/hermes-iter-audit-<pid>.jsonl` when the
    primary path is unwritable (matches the P140 pattern verified at
    PR #162).
    """
    try:
        _iter_audit_path = Path("~/.hermes/logs/iteration-audit.jsonl").expanduser()
        _iter_audit_path.parent.mkdir(parents=True, exist_ok=True)
        _iter_max = getattr(child_obj, "max_iterations", None)
        # #163 review SUGGESTION-1: coerce to 0 rather than None — downstream
        # `jq '.iterations_used' | add` aggregations error on null.
        if isinstance(api_calls_val, (int, float)):
            iters_used = int(api_calls_val)
        else:
            iters_used = 0
        _iter_record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "job_id": os.environ.get("HERMES_JOB_ID", "unknown"),
            "key": os.environ.get("HERMES_TICKET_KEY", "unknown"),
            "profile": getattr(child_obj, "_delegate_profile", "default"),
            "iterations_used": iters_used,
            "iterations_max": int(_iter_max) if isinstance(_iter_max, (int, float)) else None,
            # #163 review IMPORTANT-2: rename exit_status → exit_reason for
            # consistency with the result dict + rest of codebase.
            "exit_reason": exit_reason,
        }
        with _iter_audit_path.open("a") as _fp:
            _fp.write(json.dumps(_iter_record) + "\n")
    except (OSError, ValueError, TypeError) as _iter_exc:
        global _P141_ITER_AUDIT_DROP_COUNT
        _P141_ITER_AUDIT_DROP_COUNT += 1
        msg = (
            f"P141/MOL-491: iteration-audit write failed "
            f"(fail-open; drop_count={_P141_ITER_AUDIT_DROP_COUNT}): {_iter_exc}"
        )
        sys.stderr.write(f"[P141/MOL-491] {msg}\n")
        try:
            logger.error(msg)
        except Exception:  # noqa: BLE001 — logger itself broken
            pass
        # Tier-2 fallback: /tmp survives sandbox edge cases.
        try:
            _fallback = Path(tempfile.gettempdir()) / f"hermes-iter-audit-{os.getpid()}.jsonl"
            with _fallback.open("a") as _fp:
                _fp.write(json.dumps({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "exit_reason": exit_reason,
                    "audit_fallback": "tier2",
                    "primary_error": str(_iter_exc),
                }) + "\n")
        except Exception:  # noqa: BLE001
            pass


def _load_profile(name: str) -> Optional[Dict[str, Any]]:
    """P103/MOL-410: Load a delegate profile YAML by name.

    Reads ~/.hermes/config/delegate-profiles/<name>.yaml via yaml.safe_load.
    Returns None if file missing OR yaml is empty/malformed. Caller should
    handle None by failing loud (return tool_error) when the profile was
    explicitly requested — silent downgrade to default would mask a config
    bug as a no-op privilege change.

    P103/MOL-410 (#162 review CRITICAL-1): profile name MUST match
    `_P103_PROFILE_NAME_RE`. Names with traversal segments / dots / slashes
    are rejected upfront. The resolved path is also checked to be a
    descendant of `_P103_PROFILE_DIR` after `.resolve()` (catches
    symlink-escape attacks).

    P103/MOL-410 (#162 review M9): if PyYAML is uninstalled, log + return
    None. The default delegate path keeps working.

    P103/MOL-410 (#162 review M12): if the YAML's internal `profile` field
    differs from the requested filename-name, override it with the
    filename — the filename is canonical. Prevents copy-paste typos in
    YAML from emitting audit records under the wrong profile name.
    """
    if not _P103_PROFILE_NAME_RE.match(name or ""):
        logger.warning("P103/MOL-410: invalid profile name %r (allowlist regex rejected)", name)
        return None
    if _p103_yaml is None:
        logger.error(
            "P103/MOL-410: PyYAML not installed — profile elevation disabled. "
            "Install via `pip install pyyaml` in the gateway venv."
        )
        return None
    profile_path = _P103_PROFILE_DIR / f"{name}.yaml"
    if not profile_path.exists():
        return None
    # Symlink-escape guard: resolve the path and verify it's still under
    # the profile dir. Defensive layer on top of the regex allowlist.
    try:
        resolved = profile_path.resolve()
        resolved.relative_to(_P103_PROFILE_DIR.resolve())
    except (OSError, ValueError) as exc:
        logger.warning(
            "P103/MOL-410: profile path escape attempt for %r (resolved=%s): %s",
            name, profile_path, exc,
        )
        return None
    try:
        with profile_path.open() as fp:
            data = _p103_yaml.safe_load(fp)
    except (OSError, _p103_yaml.YAMLError) as exc:
        logger.warning("P103/MOL-410: failed to load profile %s: %s", name, exc)
        return None
    if not isinstance(data, dict):
        logger.warning(
            "P103/MOL-410: profile %s is not a dict (got %s)",
            name, type(data).__name__,
        )
        return None
    # Canonicalize internal profile name to filename. M12: the YAML's own
    # `profile` field is informational; filename is authoritative.
    if data.get("profile") and data["profile"] != name:
        logger.warning(
            "P103/MOL-410: profile file %s.yaml has internal name %r; using filename-based name",
            name, data["profile"],
        )
    data["profile"] = name
    return data


def _validate_profile_additive_only(
    profile: Dict[str, Any],
    base_settings: Optional[Dict[str, Any]] = None,
) -> None:
    """P103/MOL-410: enforce additive-only profile semantics.

    Profiles can ADD allowed_tools entries, MCP servers, env_passthrough
    entries, and allowed_bash_patterns. They CANNOT remove from base
    settings.json deny rules — those are kernel-enforced via sandbox-exec
    + Rampart + git-guardrails. This helper catches profile-YAML mistakes
    where someone adds a permissive rule that overlaps a base deny.

    P103/MOL-410 (#162 review I1, M2 review-fix): also validates
    `env_passthrough` against `_P103_PROFILE_PASSTHROUGH_DENYLIST` —
    profiles cannot lift sensitive credentials (ANTHROPIC_API_KEY etc.)
    into subprocess env regardless of declarations. This was the
    documented M2 review-fix from the original P103 surface that the
    re-application initially missed.

    Raises ValueError if the profile attempts to remove or override a base
    deny, or if it declares a denylist-protected env_passthrough entry.
    Silent (no-op) if base_settings is None AND no denylist violations
    exist.
    """
    base = base_settings or {}
    base_deny = base.get("permissions", {}).get("deny", [])
    profile_deny = profile.get("permissions", {}).get("deny", [])
    # If profile defines its own deny list, every base entry must still be present.
    if profile_deny and base_deny:
        missing = [d for d in base_deny if d not in profile_deny]
        if missing:
            raise ValueError(
                f"P103/MOL-410: profile '{profile.get('profile', '?')}' attempts to "
                f"remove base deny rules: {missing}. Profiles are additive-only — "
                f"deny rules can only be ADDED, never removed."
            )
    # Forbid any explicit allow that overlaps a base deny pattern.
    profile_allow = profile.get("allowed_bash_patterns", [])
    for pattern in profile_allow:
        if pattern in base_deny:
            raise ValueError(
                f"P103/MOL-410: profile pattern '{pattern}' overlaps base deny. "
                f"Profiles cannot grant access to denied operations."
            )
    # M2 review-fix: env_passthrough denylist enforcement.
    profile_passthrough = profile.get("env_passthrough", [])
    if isinstance(profile_passthrough, list):
        leaked = [k for k in profile_passthrough if k in _P103_PROFILE_PASSTHROUGH_DENYLIST]
        if leaked:
            raise ValueError(
                f"P103/MOL-410: profile '{profile.get('profile', '?')}' declares "
                f"env_passthrough for denylisted secrets: {leaked}. These keys are "
                f"NEVER allowed to reach a subprocess regardless of profile YAML "
                f"(M2 review-fix). See _P103_PROFILE_PASSTHROUGH_DENYLIST."
            )


def _check_profile_session_consent(
    profile_name: str, repo_path: str
) -> Tuple[bool, str]:
    """P103/MOL-410: HITL session-consent gate keyed by (profile, repo_path).

    First-use-per-tuple per session prompts via Telegram; subsequent uses
    within TTL bypass the prompt. v1 returns (True, "v1-auto-approve")
    because the Telegram-prompt subsystem is shared with the gateway
    long-poller (409-conflict if both poll). Full per-call HITL is
    tracked at MOL-450.

    P103/MOL-410 (#162 review I4): cache TTL is uptime-relative
    (`time.monotonic()`), not wall-clock — immune to NTP correction and
    macOS sleep/wake clock skew. The "session" semantic is process-bounded
    anyway, so monotonic is the right primitive.

    P103/MOL-410 (#162 review M8): cache mutations are guarded by
    `_P103_CONSENT_LOCK`. v1 doesn't need it (no I/O between read and
    write) but locking now removes a footgun from the v2 Telegram-prompt
    wire-up where the read→prompt→write window would otherwise dual-prompt.

    Returns (consented: bool, reason: str). When False, reason is a short
    operator-surface message.
    """
    key = (profile_name, repo_path)
    now = time.monotonic()
    ttl_minutes = _P103_DEFAULT_CONSENT_TTL_MIN
    with _P103_CONSENT_LOCK:
        expiry = _P103_CONSENT_CACHE.get(key)
        if expiry is not None and expiry > now:
            return True, "cached"
        _P103_CONSENT_CACHE[key] = now + ttl_minutes * 60
    return True, "v1-auto-approve (MOL-450 follow-up)"


def _emit_profile_audit(
    profile: Dict[str, Any],
    argv: List[str],
    matched_rule: Optional[str] = None,
    *,
    caller: Optional[str] = None,
    prompt: Optional[str] = None,
    exit_code: Optional[int] = None,
    cost_usd: Optional[float] = None,
    turns_used: Optional[int] = None,
) -> None:
    """P103/MOL-410: write a per-profile audit JSONL record.

    Path: ~/.hermes/logs/delegate-{profile}.log (per PATCHES.md§P103:6697).
    Override via profile.audit.jsonl.

    Fields: ts, profile, caller, prompt_hash, argv, exit_code, cost_usd,
    turns_used, matched_rule. include_prompt is honored from profile.audit
    (default false — only prompt_hash is recorded for forensics + privacy).

    Fail-open: if the log path is unwritable (sandbox edge case), this
    function emits a stderr warning and returns without raising. The
    delegate_task call continues. MOL-474 closes when this function
    fires on the coding profile and writes a non-empty record.
    """
    profile_name = profile.get("profile", "default") if isinstance(profile, dict) else "default"
    audit_cfg = profile.get("audit", {}) if isinstance(profile, dict) else {}
    log_path_str = audit_cfg.get("jsonl") if isinstance(audit_cfg, dict) else None
    if log_path_str:
        log_path = Path(log_path_str).expanduser()
    else:
        log_path = _P103_LOG_DIR / f"delegate-{profile_name}.log"

    record: Dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "profile": profile_name,
        "caller": caller or "delegate_task",
        "argv": argv,
        "matched_rule": matched_rule,
        "exit_code": exit_code,
        "cost_usd": cost_usd,
        "turns_used": turns_used,
    }
    if prompt is not None:
        record["prompt_hash"] = hashlib.sha256(
            prompt.encode("utf-8", errors="replace")
        ).hexdigest()[:16]
        if isinstance(audit_cfg, dict) and audit_cfg.get("include_prompt"):
            record["prompt"] = prompt

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as fp:
            fp.write(json.dumps(record) + "\n")
    except Exception as exc:  # noqa: BLE001 — fail-open per P103 contract
        # P103/MOL-410 (#162 review CRITICAL-2): stderr is silent in cron
        # context (gateway eats it). Loud failure path: stderr + logger.error
        # (which routes to gateway.log via the gateway logger) + counter
        # increment + tier-2 fallback to /tmp so audit isn't entirely lost.
        global _P103_AUDIT_DROP_COUNT
        _P103_AUDIT_DROP_COUNT += 1
        msg = (
            f"P103/MOL-410: _emit_profile_audit failed for profile={profile_name!r} "
            f"(fail-open; drop_count={_P103_AUDIT_DROP_COUNT}): {exc}"
        )
        sys.stderr.write(f"[P103/MOL-410] {msg}\n")
        try:
            logger.error(msg)
        except Exception:  # noqa: BLE001 — logger itself broken; nothing more to do
            pass
        # Tier-2 fallback: /tmp survives sandbox edge cases that block
        # ~/.hermes/logs/. Best-effort — silent failure of THIS too is the
        # final fail-open boundary.
        try:
            fallback = Path(tempfile.gettempdir()) / f"hermes-delegate-audit-{os.getpid()}.jsonl"
            with fallback.open("a") as fp:
                fp.write(json.dumps({**record, "audit_fallback": "tier2"}) + "\n")
        except Exception:  # noqa: BLE001
            pass


def _build_claude_argv(
    profile_cfg: Dict[str, Any],
    repo_path: str,
    prompt: str,
    *,
    extra_add_dirs: Optional[List[str]] = None,
) -> List[str]:
    """P103/MOL-410: construct claude -p argv from a coding-profile YAML.

    Emits the CLI flags for max_turns, max_budget_usd, setting_sources,
    add_dir, allowed_tools, denied_tools, append_system_prompt. The argv
    is the source-of-truth payload that gets recorded in the audit log
    AND (when the ACP wire-up is active) handed to the subprocess spawner.

    Profile YAML schema (subset consumed here):
      profile: <name>
      max_turns: int  (mapped to --max-turns)
      max_budget_usd: float  (mapped to --max-budget-usd)
      add_dirs: [str]  (mapped to --add-dir, runtime-interpolated)
      allowed_tools: [str]  (mapped to --allowedTools)
      disallowed_tools: [str]  (mapped to --disallowedTools)
      system_prompt_suffix: str  (mapped to --append-system-prompt)
    """
    argv: List[str] = ["claude", "-p"]
    # --max-turns: per-subprocess iteration cap from profile (e.g. 100 for coding).
    max_turns = profile_cfg.get("max_turns")
    if max_turns is not None:
        argv.extend(["--max-turns", str(max_turns)])
    # --max-budget-usd: cost cap per subprocess.
    max_budget = profile_cfg.get("max_budget_usd")
    if max_budget is not None:
        argv.extend(["--max-budget-usd", str(max_budget)])
    # --setting-sources: control which settings layer the subprocess inherits.
    # Default to user,project so inline --settings stacks ON TOP of inherited
    # deny rules (additive-only — profiles can't remove denies).
    argv.extend(["--setting-sources", "user,project"])
    # --add-dir: each profile-declared add_dir gets interpolated with repo_path.
    # P103/MOL-410 (#162 review CRITICAL-6): validate template field names via
    # string.Formatter before .format() — typos like `{repo-path}` (hyphen)
    # silently passed through unaltered in the prior implementation; format-spec
    # injection like `{repo_path:>500}` produced bizarre argv. Now: only
    # `{repo_path}` is honored; any other field name aborts the directory.
    _formatter = string.Formatter()
    add_dirs = list(profile_cfg.get("add_dirs", []))
    for extra in (extra_add_dirs or []):
        if extra not in add_dirs:
            add_dirs.append(extra)
    for raw_dir in add_dirs:
        try:
            field_names = {
                f for _, f, _, _ in _formatter.parse(raw_dir)
                if f is not None and f != ""
            }
        except (ValueError, IndexError):
            logger.warning(
                "P103/MOL-410: add_dir %r has malformed template; skipping", raw_dir
            )
            continue
        unknown = field_names - {"repo_path"}
        if unknown:
            logger.warning(
                "P103/MOL-410: add_dir %r references unknown fields %s; skipping",
                raw_dir, sorted(unknown),
            )
            continue
        try:
            resolved = raw_dir.format(repo_path=repo_path) if field_names else raw_dir
        except (KeyError, IndexError, ValueError) as exc:
            logger.warning(
                "P103/MOL-410: add_dir %r format failed: %s; skipping", raw_dir, exc
            )
            continue
        argv.extend(["--add-dir", resolved])
    # --allowedTools: profile's allowed tools list.
    allowed_tools = profile_cfg.get("allowed_tools")
    if allowed_tools:
        argv.extend(["--allowedTools", ",".join(allowed_tools)])
    # --disallowedTools: optional inverse list (informational; deny rules
    # are enforced at sandbox layer regardless).
    disallowed = profile_cfg.get("disallowed_tools")
    if disallowed:
        argv.extend(["--disallowedTools", ",".join(disallowed)])
    # --append-system-prompt: profile-specific system prompt suffix.
    suffix = profile_cfg.get("system_prompt_suffix")
    if suffix:
        argv.extend(["--append-system-prompt", suffix])
    # --permission-mode: profile's permission mode (e.g. "plan", "auto").
    permission_mode = profile_cfg.get("permission_mode")
    if permission_mode:
        argv.extend(["--permission-mode", permission_mode])
    # Final positional: the prompt itself.
    argv.append(prompt)
    return argv


# ============================================================================
# End P103/MOL-410 helpers
# ============================================================================


# ── P144/MOL-493 ──────────────────────────────────────────────────────────
# Claude Code subprocess delegation (Tier 1 native + Tier 2 DeepSeek direct).
# Replaces the old local proxy with direct api.deepseek.com/anthropic routing.
# Each spawner emits P141 iteration-audit records so coding-profile dispatches
# are visible to iteration-pressure analysis.


# P144 review (#165): per-occurrence credential redaction — replaces the old
# all-or-nothing substring check that lost diagnostic info on any match and
# missed non-covered formats (glpat-*, gho_*, etc.).
_CREDENTIAL_REDACT_RE = re.compile(
    r'(?:sk-ant-)[A-Za-z0-9]{1,40}'
    r'|(?:AIza)[0-9A-Za-z\-_]{1,40}'
    r'|(?:ya29\.)[0-9A-Za-z\-_]{1,100}'
    r'|(?:gh[po]_)[0-9A-Za-z]{1,60}'
    r'|(?:github_pat_)[0-9A-Za-z\-_]{1,100}'
    r'|(?:glpat-)[0-9A-Za-z\-_]{1,50}'
)


def _redact_credentials(text: str) -> str:
    """Redact credential patterns inline without discarding diagnostic context."""
    if not text:
        return text
    return _CREDENTIAL_REDACT_RE.sub("[REDACTED]", text)


# P144 review (#165): short-term probe cache — avoids burning billable API
# calls on repeated Tier 2 fallback attempts during degraded-API storms.
# Keyed by API-key prefix (first 12 chars), 60s TTL.
_PROBE_CACHE: Dict[str, tuple[float, bool]] = {}


def _check_probe_cache(api_key: str) -> Optional[bool]:
    """Return cached probe result if fresh, None if stale or missing."""
    _now = time.monotonic()
    _prefix = api_key[:12]
    if _prefix in _PROBE_CACHE:
        _cached_at, _cached_ok = _PROBE_CACHE[_prefix]
        if _now - _cached_at < 60.0:
            return _cached_ok
        del _PROBE_CACHE[_prefix]
    return None


def _set_probe_cache(api_key: str, ok: bool) -> None:
    _PROBE_CACHE[api_key[:12]] = (time.monotonic(), ok)
# are visible to iteration-pressure analysis.
# ───────────────────────────────────────────────────────────────────────────

# P52/MOL-233 — delegation truthfulness verifier.
# When a subagent claims it modified code, we check git diff to verify.

_VERIFY_EDIT_KEYWORDS: frozenset = frozenset({
    "modified", "updated", "created", "added", "changed", "edited",
    "wrote", "rewrote", "patched", "fixed", "implemented", "refactored",
})

_VERIFY_NOOP_SIGNALS: frozenset = frozenset({
    "already in place",
    "already implemented",
    "already present",
    "already exists",
    "no changes needed",
    "no changes required",
    "no modifications needed",
    "nothing to change",
    "no work needed",
    "not needed",
    "not required",
    "no change",
})


def _verify_delegation_diff(
    pre_dirty: set,
    post_dirty: set,
    goal: str,
    context: str,
    result_text: str,
) -> tuple:
    """Verify that a delegation that claimed code changes actually produced a diff.

    Returns (verified: bool, reason: str).  ``verified=True`` means either a
    diff appeared OR the subagent honestly reported a no-op.  ``verified=False``
    means the subagent described edits but no files changed.
    """
    new_files = post_dirty - pre_dirty
    if new_files:
        return True, f"diff confirmed: {', '.join(sorted(new_files)[:5])}"
    combined = f"{result_text} {goal} {context}".lower()
    for signal in _VERIFY_NOOP_SIGNALS:
        if signal in combined:
            return True, f"no-op signal: '{signal}'"
    for kw in _VERIFY_EDIT_KEYWORDS:
        if kw.lower() in result_text.lower():
            return False, f"claimed '{kw}' but no files changed"
    return True, "no diff but no explicit edit claim — ambiguous"


def _run_claude_code_delegation(
    goal: str = "",
    context: str = "",
    repo_path: str = "",
    max_iterations: int = 100,
    parent_agent=None,
    skip_hitl: bool = False,
    profile_cfg: Optional[Dict[str, Any]] = None,
) -> dict:
    """Run a coding task via Claude Code subprocess (Tier 1 — native Anthropic).

    Returns a dict (not a JSON string):
      - Success: {"result": ..., "duration_seconds": N, "num_turns": N, "cost_usd": N, "model": ...}
      - Rate limit: {"error": ..., "rate_limited": True}
      - Other error: {"error": ..., "rate_limited": False}
    """
    import subprocess as _subprocess
    from types import SimpleNamespace

    cfg = _load_config()
    coding_cfg = cfg.get("coding", {})
    if not coding_cfg.get("enabled", False):
        return {"error": "Coding delegation is disabled. Set delegation.coding.enabled=true in config.yaml", "rate_limited": False}

    if parent_agent is None:
        return {"error": "Claude Code delegation requires a parent agent context.", "rate_limited": False}

    depth = getattr(parent_agent, '_delegate_depth', 0)
    if depth >= MAX_DEPTH:
        return {"error": f"Delegation depth limit reached ({MAX_DEPTH}).", "rate_limited": False}

    if not goal or not goal.strip():
        return {"error": "goal is required and cannot be empty.", "rate_limited": False}

    if not repo_path or not repo_path.startswith("/") or not os.path.isdir(repo_path):
        return {"error": f"repo_path must be an absolute path to an existing directory: {repo_path}", "rate_limited": False}
    if not os.path.isdir(os.path.join(repo_path, ".git")):
        return {"error": f"repo_path is not a git repository: {repo_path}", "rate_limited": False}

    allowed_roots = coding_cfg.get("allowed_write_roots", [])
    if not allowed_roots:
        return {"error": "No allowed_write_roots configured.", "rate_limited": False}
    repo_real = os.path.realpath(repo_path)
    if not any(repo_real == os.path.realpath(os.path.expanduser(r)) or repo_real.startswith(os.path.realpath(os.path.expanduser(r)) + os.sep) for r in allowed_roots):
        return {"error": f"repo_path {repo_path} not within allowed_write_roots: {allowed_roots}", "rate_limited": False}

    if not skip_hitl:
        try:
            from tools.approval import check_all_command_guards
            guard_result = check_all_command_guards(
                f"claude_code_delegation: {goal[:100]} -> {repo_path}",
                env_type="local",
            )
            if not guard_result.get("approved", False):
                return {"error": f"Delegation denied: {guard_result.get('message', 'denied by command guards')}", "rate_limited": False}
        except ImportError:
            return {"error": "approval module not available", "rate_limited": False}

    prompt = f"{goal}\n\nContext: {context}" if context else goal
    effective_max = max_iterations or coding_cfg.get("max_iterations", 100)

    if profile_cfg:
        argv = _build_claude_argv(profile_cfg, repo_path, prompt)
    else:
        cc_model = coding_cfg.get("model", "sonnet")
        cc_effort = coding_cfg.get("effort", "high")
        argv = [
            "claude", "-p",
            "--output-format", "json",
            "--no-session-persistence",
            "--tools", "Bash,Read,Write,Edit,Glob,Grep",
            "--permission-mode", "auto",
            "--model", cc_model,
            "--effort", cc_effort,
            "--add-dir", repo_path,
            "--max-turns", str(effective_max),
            "--system-prompt", (
                "You are a coding assistant working on a delegated task. "
                "Stay focused on the goal. Do not read or print environment variables. "
                "Do not access files outside the specified repository. "
                "Do not make network requests or push to git remotes."
            ),
            prompt,
        ]

    from tools.environments.local import _sanitize_subprocess_env
    _profile_passthrough = set(profile_cfg.get("env_passthrough", [])) if profile_cfg else None
    safe_env = _sanitize_subprocess_env(os.environ.copy(), profile_passthrough=_profile_passthrough)

    # P52/MOL-233 — pre-delegation dirty files for truthfulness verification.
    pre_dirty: set = set()
    pre_check_ok = False
    try:
        _pre = _subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=repo_path,
        )
        if _pre.returncode == 0:
            pre_dirty = {f for f in _pre.stdout.strip().split("\n") if f}
            pre_check_ok = True
    except (OSError, _subprocess.TimeoutExpired):
        pass  # pre-check failed; verification will be conservative

    # P141/MOL-491 stub — carries profile + max_iterations for iteration audit.
    _p141_stub = SimpleNamespace(
        max_iterations=effective_max,
        _delegate_profile=profile_cfg.get("profile", "default") if profile_cfg else "default",
    )

    start_time = time.monotonic()
    try:
        proc = _subprocess.Popen(
            argv, stdout=_subprocess.PIPE, stderr=_subprocess.PIPE,
            text=True, cwd=repo_path, env=safe_env,
        )
    except OSError as exc:
        _emit_iteration_audit(exit_reason="error", api_calls_val=0, child_obj=_p141_stub)
        return {"error": f"Failed to start claude subprocess: {exc}", "rate_limited": False}

    heartbeat_stop = threading.Event()

    def _heartbeat():
        while not heartbeat_stop.wait(15.0):
            touch = getattr(parent_agent, '_touch_activity', None)
            if touch:
                try:
                    touch(f"claude_code_delegation: PID {proc.pid} active, {round(time.monotonic() - start_time)}s elapsed")
                except Exception:
                    pass

    hb_thread = threading.Thread(target=_heartbeat, daemon=True)
    hb_thread.start()

    try:
        stdout, stderr = proc.communicate(timeout=_get_child_timeout_seconds(profile_cfg))  # P105/MOL-420 P145/MOL-495
    except _subprocess.TimeoutExpired:
        heartbeat_stop.set()
        try:
            proc.kill()
        except OSError:
            pass  # process already dead; proceed with cleanup
        stdout, stderr = proc.communicate()
        _emit_iteration_audit(exit_reason="timeout", api_calls_val=0, child_obj=_p141_stub)
        return {"error": f"Claude Code delegation timed out after {_get_child_timeout_seconds(profile_cfg)} seconds", "rate_limited": False}
    except OSError as exc:
        heartbeat_stop.set()
        try:
            proc.kill()
        except OSError:
            pass
        stdout, stderr = proc.communicate()
        _emit_iteration_audit(exit_reason="error", api_calls_val=0, child_obj=_p141_stub)
        return {"error": f"Claude Code subprocess pipe error: {exc}", "rate_limited": False}
    finally:
        heartbeat_stop.set()

    duration_seconds = round(time.monotonic() - start_time, 1)

    if proc.returncode != 0:
        safe_stderr = _redact_credentials(stderr[:500] if stderr else "")
        error_msg = f"Claude Code exited with code {proc.returncode}: {safe_stderr}"
        combined = f"{proc.returncode} {safe_stderr}".lower()
        rate_limited = any(p in combined for p in _RATE_LIMIT_PATTERNS)
        _emit_iteration_audit(
            exit_reason="rate_limited" if rate_limited else "error",
            api_calls_val=0,
            child_obj=_p141_stub,
        )
        return {"error": error_msg, "rate_limited": rate_limited}

    try:
        result_data = json.loads(stdout)
    except json.JSONDecodeError:
        safe_stdout = _redact_credentials(stdout[:500] if stdout else "[empty output]")
        _emit_iteration_audit(exit_reason="error", api_calls_val=0, child_obj=_p141_stub)
        return {"error": f"Claude Code output was not valid JSON: {safe_stdout}", "rate_limited": False}

    result = result_data.get("result", "") or ""
    num_turns = result_data.get("num_turns", 0)
    error_text = result_data.get("error")
    if isinstance(error_text, str) and error_text.strip():
        rate_limited = any(p in str(error_text).lower() for p in _RATE_LIMIT_PATTERNS)
        _emit_iteration_audit(
            exit_reason="rate_limited" if rate_limited else "error",
            api_calls_val=num_turns,
            child_obj=_p141_stub,
        )
        return {"error": error_text, "rate_limited": rate_limited}

    # P52/MOL-233 — post-delegation diff verification.
    post_dirty: set = set()
    verified = True
    verification_reason = "skipped"
    try:
        _post = _subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=repo_path,
        )
        if _post.returncode == 0:
            post_dirty = {f for f in _post.stdout.strip().split("\n") if f}
        if pre_check_ok:
            verified, verification_reason = _verify_delegation_diff(
                pre_dirty, post_dirty, goal, context, result,
            )
        else:
            verification_reason = "pre-check failed, conservative diff"
        if not verified:
            logger.warning(
                "delegate_truthfulness verified=false repo=%s reason=%s goal=%.120s",
                repo_path, verification_reason, goal,
            )
    except (OSError, _subprocess.TimeoutExpired):
        verification_reason = "verification error"
        verified = True

    _emit_iteration_audit(
        exit_reason="completed",
        api_calls_val=num_turns,
        child_obj=_p141_stub,
    )

    return {
        "result": result,
        "duration_seconds": duration_seconds,
        "num_turns": num_turns,
        "cost_usd": result_data.get("total_cost_usd", 0),
        "model": result_data.get("model", ""),
        "verified": verified,
        "verification_reason": verification_reason,
    }


def _run_claude_code_deepseek_direct_delegation(
    goal: str = "",
    context: str = "",
    repo_path: str = "",
    max_iterations: int = 100,
    parent_agent=None,
    skip_hitl: bool = False,
    profile_cfg: Optional[Dict[str, Any]] = None,
) -> dict:
    """Run a coding task via Claude Code through DeepSeek direct API (Tier 2).

    Same contract as _run_claude_code_delegation.  Probes api.deepseek.com
    before spawning, fetches the API key from envchain hermes-llm, and sets
    ANTHROPIC_BASE_URL + model-mapping env vars so claude -p routes through
    DeepSeek's Anthropic-compatible endpoint.  No local proxy required.

    Returns dict with "delegation": "claude-code-deepseek-direct" on success.
    """
    import subprocess as _subprocess
    from types import SimpleNamespace

    cfg = _load_config()
    coding_cfg = cfg.get("coding", {})
    fd_cfg = coding_cfg.get("fallback_deepseek_cc", {})
    if not fd_cfg.get("enabled", True):
        return {"error": "DeepSeek CC fallback is disabled (delegation.coding.fallback_deepseek_cc.enabled=false)", "rate_limited": False}

    if parent_agent is None:
        return {"error": "DeepSeek delegation requires a parent agent context.", "rate_limited": False}

    depth = getattr(parent_agent, "_delegate_depth", 0)
    if depth >= MAX_DEPTH:
        return {"error": f"Delegation depth limit reached ({MAX_DEPTH}).", "rate_limited": False}

    if not goal or not goal.strip():
        return {"error": "goal is required and cannot be empty.", "rate_limited": False}

    if not repo_path or not repo_path.startswith("/") or not os.path.isdir(repo_path):
        return {"error": f"repo_path must be an absolute path to an existing directory: {repo_path}", "rate_limited": False}
    if not os.path.isdir(os.path.join(repo_path, ".git")):
        return {"error": f"repo_path is not a git repository: {repo_path}", "rate_limited": False}

    allowed_roots = coding_cfg.get("allowed_write_roots", [])
    if not allowed_roots:
        return {"error": "No allowed_write_roots configured for DeepSeek delegation.", "rate_limited": False}
    repo_real = os.path.realpath(repo_path)
    if not any(repo_real == os.path.realpath(os.path.expanduser(r)) or repo_real.startswith(os.path.realpath(os.path.expanduser(r)) + os.sep) for r in allowed_roots):
        return {"error": f"repo_path {repo_path} not within allowed_write_roots: {allowed_roots}", "rate_limited": False}

    prompt = f"{goal}\n\nContext: {context}" if context else goal
    effective_max = max_iterations or coding_cfg.get("max_iterations", 100)

    # ── Health probe: verify api.deepseek.com responds before spawning ──
    deepseek_key = ""
    try:
        env_result = _subprocess.run(
            ["envchain", "hermes-llm", "printenv", "DEEPSEEK_API_KEY"],
            capture_output=True, text=True, timeout=10,
        )
        if env_result.returncode == 0 and env_result.stdout.strip():
            deepseek_key = env_result.stdout.strip()
    except Exception:
        pass
    if not deepseek_key:
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not deepseek_key:
        return {"error": "DEEPSEEK_API_KEY not available (check envchain hermes-llm)", "rate_limited": False}

    # P141 stub — built early so probe-failure audit records can use it.
    _p141_stub = SimpleNamespace(
        max_iterations=effective_max,
        _delegate_profile=profile_cfg.get("profile", "default") if profile_cfg else "default",
    )

    # Probe with a minimal 1-token request (same pattern as claude-deepseek-direct.sh).
    # P144 review (#165): short-term cache avoids burning API calls on repeated
    # fallback attempts during degraded-API storms (60s TTL per key prefix).
    _cached = _check_probe_cache(deepseek_key)
    if _cached is True:
        probe_ok = True
    elif _cached is False:
        _emit_iteration_audit(exit_reason="probe_failed", api_calls_val=0, child_obj=_p141_stub)
        return {"error": "DeepSeek API probe failed (cached, re-probe in <60s)", "rate_limited": False}
    else:
        # P144: DO NOT log probe_cmd — contains API key in curl -H argument.
        probe_cmd = [
            "curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
            "--max-time", "10",
            "-X", "POST", "https://api.deepseek.com/anthropic/v1/messages",
            "-H", f"x-api-key: {deepseek_key}",
            "-H", "Content-Type: application/json",
            "-d", '{"model":"deepseek-v4-pro","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}',
        ]
        probe_ok = False
        try:
            probe_proc = _subprocess.run(probe_cmd, capture_output=True, text=True, timeout=15)
            probe_status = probe_proc.stdout.strip()
            if probe_status.startswith("2"):
                probe_ok = True
                _set_probe_cache(deepseek_key, True)
            else:
                _set_probe_cache(deepseek_key, False)
                _emit_iteration_audit(exit_reason="probe_failed", api_calls_val=0, child_obj=_p141_stub)
                return {"error": f"DeepSeek API probe failed (HTTP {probe_status})", "rate_limited": probe_status == "429"}
        except _subprocess.TimeoutExpired:
            _emit_iteration_audit(exit_reason="probe_failed", api_calls_val=0, child_obj=_p141_stub)
            return {"error": "DeepSeek API probe timed out", "rate_limited": False}
        except OSError as exc:
            _emit_iteration_audit(exit_reason="probe_failed", api_calls_val=0, child_obj=_p141_stub)
            return {"error": f"DeepSeek API probe failed: {exc}", "rate_limited": False}

    if profile_cfg:
        argv = _build_claude_argv(profile_cfg, repo_path, prompt)
    else:
        cc_model = coding_cfg.get("model", "sonnet")
        cc_effort = coding_cfg.get("effort", "high")
        argv = [
            "claude", "-p",
            "--output-format", "json",
            "--no-session-persistence",
            "--tools", "Bash,Read,Write,Edit,Glob,Grep",
            "--permission-mode", "auto",
            "--model", cc_model,
            "--effort", cc_effort,
            "--add-dir", repo_path,
            "--max-turns", str(effective_max),
            "--system-prompt", (
                "You are a coding assistant working on a delegated task. "
                "Stay focused on the goal. Do not read or print environment variables. "
                "Do not access files outside the specified repository. "
                "Do not make network requests or push to git remotes."
            ),
            prompt,
        ]

    from tools.environments.local import _sanitize_subprocess_env
    _profile_passthrough = set(profile_cfg.get("env_passthrough", [])) if profile_cfg else None
    safe_env = _sanitize_subprocess_env(os.environ.copy(), profile_passthrough=_profile_passthrough)

    # ── Inject DeepSeek direct-routing env vars ──────────────────────────
    safe_env["ANTHROPIC_BASE_URL"] = "https://api.deepseek.com/anthropic"
    safe_env["ANTHROPIC_AUTH_TOKEN"] = deepseek_key
    safe_env["ANTHROPIC_MODEL"] = "deepseek-v4-pro[1m]"
    safe_env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = "deepseek-v4-pro[1m]"
    safe_env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = "deepseek-v4-pro[1m]"
    safe_env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = "deepseek-v4-flash"
    safe_env["CLAUDE_CODE_SUBAGENT_MODEL"] = "deepseek-v4-flash"
    safe_env["CLAUDE_CODE_EFFORT_LEVEL"] = "max"

    pre_dirty: set = set()
    pre_check_ok = False
    try:
        _pre = _subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=repo_path,
        )
        if _pre.returncode == 0:
            pre_dirty = {f for f in _pre.stdout.strip().split("\n") if f}
            pre_check_ok = True
    except (OSError, _subprocess.TimeoutExpired):
        pass  # pre-check failed; verification will be conservative

    start_time = time.monotonic()
    try:
        proc = _subprocess.Popen(
            argv, stdout=_subprocess.PIPE, stderr=_subprocess.PIPE,
            text=True, cwd=repo_path, env=safe_env,
        )
    except OSError as exc:
        _emit_iteration_audit(exit_reason="error", api_calls_val=0, child_obj=_p141_stub)
        return {"error": f"Failed to start claude subprocess (DeepSeek): {exc}", "rate_limited": False}

    heartbeat_stop = threading.Event()

    def _heartbeat():
        while not heartbeat_stop.wait(15.0):
            touch = getattr(parent_agent, "_touch_activity", None)
            if touch:
                try:
                    touch(f"deepseek_direct_delegation: PID {proc.pid} active, {round(time.monotonic() - start_time)}s elapsed")
                except Exception:
                    pass

    hb_thread = threading.Thread(target=_heartbeat, daemon=True)
    hb_thread.start()

    try:
        stdout, stderr = proc.communicate(timeout=_get_child_timeout_seconds(profile_cfg))
    except _subprocess.TimeoutExpired:
        heartbeat_stop.set()
        try:
            proc.kill()
        except OSError:
            pass  # process already dead; proceed with cleanup
        stdout, stderr = proc.communicate()
        _emit_iteration_audit(exit_reason="timeout", api_calls_val=0, child_obj=_p141_stub)
        return {"error": f"DeepSeek delegation timed out after {_get_child_timeout_seconds(profile_cfg)} seconds", "rate_limited": False}
    except OSError as exc:
        heartbeat_stop.set()
        try:
            proc.kill()
        except OSError:
            pass
        stdout, stderr = proc.communicate()
        _emit_iteration_audit(exit_reason="error", api_calls_val=0, child_obj=_p141_stub)
        return {"error": f"DeepSeek subprocess pipe error: {exc}", "rate_limited": False}
    finally:
        heartbeat_stop.set()

    duration_seconds = round(time.monotonic() - start_time, 1)

    if proc.returncode != 0:
        safe_stderr = _redact_credentials(stderr[:500] if stderr else "")
        error_msg = f"DeepSeek delegation exited with code {proc.returncode}: {safe_stderr}"
        rate_limited = any(p in f"{proc.returncode} {safe_stderr}".lower() for p in _RATE_LIMIT_PATTERNS)
        _emit_iteration_audit(
            exit_reason="rate_limited" if rate_limited else "error",
            api_calls_val=0,
            child_obj=_p141_stub,
        )
        return {"error": error_msg, "rate_limited": rate_limited}

    try:
        result_data = json.loads(stdout)
    except json.JSONDecodeError:
        safe_stdout = _redact_credentials(stdout[:500] if stdout else "[empty output]")
        _emit_iteration_audit(exit_reason="error", api_calls_val=0, child_obj=_p141_stub)
        return {"error": f"DeepSeek delegation output was not valid JSON: {safe_stdout}", "rate_limited": False}

    result = result_data.get("result", "") or ""
    num_turns = result_data.get("num_turns", 0)
    error_text = result_data.get("error")
    if isinstance(error_text, str) and error_text.strip():
        rate_limited = any(p in str(error_text).lower() for p in _RATE_LIMIT_PATTERNS)
        _emit_iteration_audit(
            exit_reason="rate_limited" if rate_limited else "error",
            api_calls_val=num_turns,
            child_obj=_p141_stub,
        )
        return {"error": error_text, "rate_limited": rate_limited}

    post_dirty: set = set()
    verified = True
    verification_reason = "skipped"
    try:
        _post = _subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=repo_path,
        )
        if _post.returncode == 0:
            post_dirty = {f for f in _post.stdout.strip().split("\n") if f}
        if pre_check_ok:
            verified, verification_reason = _verify_delegation_diff(pre_dirty, post_dirty, goal, context, result)
        else:
            verification_reason = "pre-check failed, conservative diff"
        if not verified:
            logger.warning("delegate_truthfulness verified=false repo=%s reason=%s", repo_path, verification_reason)
    except (OSError, _subprocess.TimeoutExpired):
        verification_reason = "verification error"
        verified = True

    _emit_iteration_audit(
        exit_reason="completed",
        api_calls_val=num_turns,
        child_obj=_p141_stub,
    )

    return {
        "result": result,
        "duration_seconds": duration_seconds,
        "num_turns": num_turns,
        "cost_usd": result_data.get("total_cost_usd", 0),
        "model": result_data.get("model", ""),
        "verified": verified,
        "verification_reason": verification_reason,
        "delegation": "claude-code-deepseek-direct",
    }


_CODE_PATH_RE = re.compile(
    r'(/Users/[^/\s"\']+/Code/[^\s"\']*'
    r'|~/Code/[^\s"\']*'
    r'|/Users/[^/\s"\']+/\.hermes/hermes-agent(?![A-Za-z0-9-])[^\s"\']*'
    r'|~/\.hermes/hermes-agent(?![A-Za-z0-9-])[^\s"\']*'
    r')'
)
_RATE_LIMIT_PATTERNS = ("429", "rate limit", "overloaded", "quota", "capacity", "too many requests")


def _detect_repo_path(goal: str, context: str = "") -> Optional[str]:
    """Extract a repo path from goal/context if it references ~/Code/."""
    cfg = _load_config()
    coding_cfg = cfg.get("coding", {})
    if not coding_cfg.get("enabled", False):
        return None
    allowed_roots = coding_cfg.get("allowed_write_roots", [])
    if not allowed_roots:
        return None
    for match in _CODE_PATH_RE.finditer(f"{goal} {context}"):
        raw = match.group(1)
        expanded = os.path.realpath(os.path.expanduser(raw))
        candidate = expanded
        while candidate and candidate != "/":
            if os.path.isdir(os.path.join(candidate, ".git")):
                for root in allowed_roots:
                    root_real = os.path.realpath(os.path.expanduser(root))
                    if candidate == root_real or candidate.startswith(root_real + os.sep):
                        return candidate
                break
            candidate = os.path.dirname(candidate)
    return None


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
    # P146/MOL-496 — profile settings injected into child agent for in-process swarm path.
    # When non-None, system_prompt_suffix is appended to the child's system prompt,
    # max_turns overrides max_iterations, and _delegate_profile is tagged on the child.
    profile_cfg: Optional[Dict[str, Any]] = None,
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
    # P146/MOL-496 — inject profile system_prompt_suffix into child's prompt.
    # This gives in-process swarm children awareness of their elevated scope
    # (e.g. coding profile: push to feature/*, open PR; planning profile:
    # read-only plan mode). The suffix is appended AFTER the core child prompt
    # so it doesn't interfere with the child's structured system message.
    if profile_cfg:
        suffix = profile_cfg.get("system_prompt_suffix")
        if suffix:
            child_prompt = f"{child_prompt}\n\n# Profile: {profile_cfg.get('profile', 'unknown')}\n{suffix}"
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
        max_iterations=(
            profile_cfg.get("max_turns") if (profile_cfg and profile_cfg.get("max_turns") is not None) else max_iterations
        ),  # P146/MOL-496 — profile max_turns overrides config max_iterations (is not None check: max_turns=0 = unlimited)
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

        # H6/P180/MOL-557: pre-spawn fingerprint snapshot for in-process child.
        # Direct delegate_task callers bypass symphony_bridge's bracket; mirror
        # the bracket here so the "child writes to runtime" invariant is checked
        # on both paths. Fail-open.
        _h6_before_child = None
        try:
            from tools.runtime_fingerprint import compute_default_surface_fingerprint
            _h6_before_child = compute_default_surface_fingerprint()
        except Exception:
            _h6_before_child = None

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

            # P141/MOL-491 (#163 review CRITICAL-1): emit iteration audit
            # on the timeout/error path. This is the highest-signal exit
            # for the "is max_iterations enough?" question — a child that
            # hung at iteration 47 of 100 is exactly what the operator
            # needs to see.
            _emit_iteration_audit(
                exit_reason="timeout" if is_timeout else "error",
                api_calls_val=child_api_calls,
                child_obj=child,
            )
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
            # H6/P180/MOL-557: post-completion fingerprint diff for in-process
            # child. Drift here means a delegate child wrote a runtime surface
            # against the `add_dirs: [{repo_path}]` invariant (MOL-556 Tier 2c).
            try:
                if _h6_before_child is not None:
                    from tools.runtime_fingerprint import (
                        compute_default_surface_fingerprint, compare_fingerprints,
                        emit_audit_jsonl, telegram_alert_rate_limited,
                        now_iso, LOG_DIR,
                    )
                    _h6_after_child = compute_default_surface_fingerprint()
                    _h6_drift_child = compare_fingerprints(_h6_before_child, _h6_after_child)
                    _h6_role = getattr(child, "_delegate_role", None)
                    _h6_profile = getattr(child, "_delegate_profile", None)
                    _h6_payload_child = {
                        "ts": now_iso(),
                        "event": "delegate_child_complete",
                        "task_index": task_index,
                        "role": _h6_role,
                        "profile": _h6_profile,
                        "drift_files": sorted(_h6_drift_child.keys()),
                        "drift_count": len(_h6_drift_child),
                    }
                    emit_audit_jsonl(LOG_DIR / "delegate-fingerprint.jsonl", _h6_payload_child)
                    if _h6_drift_child:
                        emit_audit_jsonl(
                            LOG_DIR / "delegate-runtime-write.jsonl",
                            {**_h6_payload_child, "drift": _h6_drift_child},
                        )
                        telegram_alert_rate_limited(
                            f"delegate_child_drift_{_h6_role or 'unknown'}",
                            f"[H6/MOL-557] Delegate child runtime drift\n"
                            f"role={_h6_role} profile={_h6_profile} "
                            f"task_index={task_index}\n"
                            f"files={len(_h6_drift_child)}: "
                            f"{', '.join(sorted(_h6_drift_child.keys())[:5])}",
                        )
            except Exception:
                pass

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

        # P141/MOL-491: forward iteration instrumentation — happy-path exit.
        # The other two exit paths (timeout @ ~line 2007, outer exception
        # @ ~line 2269) emit the same record via _emit_iteration_audit
        # so the audit log covers ALL dispatches, not just clean completions
        # (#163 review CRITICAL-1).
        _emit_iteration_audit(
            exit_reason=exit_reason,
            api_calls_val=api_calls,
            child_obj=child,
        )

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
        # P141/MOL-491 (#163 review CRITICAL-1): emit iteration audit on the
        # outer-exception path. Records exit_reason="error" with whatever
        # api_calls counter the child captured (or 0 if pre-execution).
        _api_calls_at_failure = 0
        try:
            _summary = getattr(child, "session_summary", None)
            if _summary is not None and isinstance(_summary, dict):
                _api_calls_at_failure = int(_summary.get("api_call_count", 0) or 0)
        except Exception:  # noqa: BLE001 — best-effort field extraction
            pass
        _emit_iteration_audit(
            exit_reason="error",
            api_calls_val=_api_calls_at_failure,
            child_obj=child,
        )
        return {
            "task_index": task_index,
            "status": "error",
            "summary": None,
            "error": str(exc),
            "api_calls": _api_calls_at_failure,
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
    toolsets: Optional[List[str]] = None,
    tasks: Optional[List[Dict[str, Any]]] = None,
    max_iterations: Optional[int] = None,
    acp_command: Optional[str] = None,
    acp_args: Optional[List[str]] = None,
    role: Optional[str] = None,
    profile: str = "default",  # P103/MOL-410 — coding-profile elevation
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

    The 'profile' parameter (P103/MOL-410) selects a delegate profile YAML
    from ~/.hermes/config/delegate-profiles/<profile>.yaml. The default
    profile is fail-closed (no elevation). The 'coding' profile elevates
    for end-to-end Jira coding tickets (push to feature/*, open PR,
    Jira comment, transition). Profile resolution + audit emit fires
    BEFORE the parent_agent check so smoke-mode invocations without a
    parent still produce an audit record (closes MOL-474).

    Returns JSON with results array, one entry per task.
    """
    # P103/MOL-410: profile resolution runs first — audit log records the
    # invocation attempt regardless of downstream success/failure. This is
    # what closes MOL-474 (delegate-coding.log audit gap).
    #
    # P103/MOL-410 (#162 review HIGH-3): when an explicit profile fails to
    # load, return tool_error rather than silently downgrading to default.
    # Silent downgrade would mask config bugs as no-op privilege changes.
    profile_cfg: Optional[Dict[str, Any]] = None
    if profile and profile != "default":
        profile_cfg = _load_profile(profile)
        if profile_cfg is None:
            return tool_error(
                f"P103/MOL-410: profile {profile!r} requested but YAML missing/malformed/rejected "
                f"at {_P103_PROFILE_DIR}. Refusing to silently downgrade to default. "
                f"Fix the YAML or invoke with profile='default'."
            )
        # Synthesize the smoke-friendly prompt + repo_path. Real Tier 1
        # callers will pass these through richer plumbing; the smoke
        # path needs SOMETHING here so the audit record is non-empty.
        synth_prompt = goal or ""
        if context:
            synth_prompt = f"{synth_prompt}\n\n{context}" if synth_prompt else context
        # P103/MOL-410 (#162 review HIGH-6): repo_path heuristic uses the
        # pre-compiled _P103_REPO_PATH_RE (tighter character class — no
        # trailing-punctuation capture), validates `os.path.isdir` post-match,
        # and falls back to CWD only when nothing better is found.
        synth_repo: Optional[str] = None
        for blob in (goal or "", context or ""):
            if "/Code/" not in blob:
                continue
            m = _P103_REPO_PATH_RE.search(blob)
            if m:
                candidate = os.path.expanduser(m.group(0))
                if os.path.isdir(candidate):
                    synth_repo = candidate
                    break
        if synth_repo is None:
            synth_repo = os.getcwd()
        # P103/MOL-410 (#162 review M10): pass base_settings if available
        # so additive-only enforcement compares against the actual base
        # settings, not the profile-self. Best-effort — log a warning if
        # base settings aren't loadable so the operator knows enforcement
        # is half-armed.
        base_settings: Optional[Dict[str, Any]] = None
        try:
            base_settings_path = Path("~/.claude/settings.json").expanduser()
            if base_settings_path.exists():
                base_settings = json.loads(base_settings_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "P103/MOL-410: could not load base settings.json for additive-only "
                "validation (%s); validator will run profile-self only", exc,
            )
        try:
            _validate_profile_additive_only(profile_cfg, base_settings)
        except ValueError as exc:
            # P103/MOL-410 (#162 review HIGH-4): emit audit BEFORE returning
            # tool_error. Profile-rejection is the highest-value forensic
            # event — operator needs the trace.
            logger.error("P103/MOL-410: profile %r rejected: %s", profile, exc)
            _emit_profile_audit(
                profile_cfg,
                argv=[],
                matched_rule=f"profile={profile} REJECTED additive_violation: {exc}",
                caller="delegate_task",
                prompt=synth_prompt,
                exit_code=-1,  # sentinel: rejected at validator
            )
            return tool_error(str(exc))
        consented, consent_reason = _check_profile_session_consent(profile, synth_repo)
        argv = _build_claude_argv(profile_cfg, synth_repo, synth_prompt)
        _emit_profile_audit(
            profile_cfg,
            argv,
            matched_rule=f"profile={profile} consent={consent_reason}",
            caller="delegate_task",
            prompt=synth_prompt,
        )
        if not consented:
            return tool_error(
                f"P103/MOL-410: HITL consent denied for ({profile}, {synth_repo}): {consent_reason}"
            )

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
        task_list = [
            {"goal": goal, "context": context, "toolsets": toolsets, "role": top_role}
        ]
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

    # ── P144/MOL-493: Claude Code-first routing (Tier 1 → Tier 2 → in-process) ──
    # For tasks whose goal/context references a ~/Code/ repo, try native CC first,
    # then DeepSeek-direct CC, before falling through to in-process subagents.
    claude_code_results: Dict[int, dict] = {}
    remaining_indices = set(range(len(task_list)))
    # P144 review (#165): cache repo paths during Tier 1 to avoid re-extraction in Tier 2.
    _task_repo_cache: Dict[int, str] = {}

    for i, t in enumerate(task_list):
        repo_path = _detect_repo_path(t["goal"], t.get("context", ""))
        if not repo_path:
            logger.debug("[delegate_task] Task %d: no repo path detected in goal/context, skipping CC routing", i)
            continue
        _task_repo_cache[i] = repo_path
        logger.info("[delegate_task] Routing task %d to Claude Code (repo: %s)", i, repo_path)
        try:
            cc_result = _run_claude_code_delegation(
                goal=t["goal"],
                context=t.get("context", ""),
                repo_path=repo_path,
                max_iterations=t.get("max_iterations") or effective_max_iter,
                parent_agent=parent_agent,
                skip_hitl=True,
                profile_cfg=profile_cfg,
            )
            if "error" not in cc_result:
                _verified = cc_result.get("verified", True)
                _v_reason = cc_result.get("verification_reason", "")
                _status = "completed" if _verified else "completed_unverified"
                _error = None
                if not _verified:
                    _error = f"⚠️ VERIFICATION FAILED: {_v_reason}"
                    logger.warning("delegate_truthfulness task %d verified=false reason=%s", i, _v_reason)
                claude_code_results[i] = {
                    "task_index": i,
                    "status": _status,
                    "summary": cc_result.get("result", ""),
                    "error": _error,
                    "api_calls": cc_result.get("num_turns", 0),
                    "duration_seconds": cc_result.get("duration_seconds", 0),
                    "delegation": "claude-code",
                }
                remaining_indices.discard(i)
                continue
            if cc_result.get("rate_limited"):
                logger.warning("[delegate_task] Claude Code rate-limited, falling back for task %d", i)
            else:
                logger.warning("[delegate_task] Claude Code failed: %s — falling back", str(cc_result.get("error", ""))[:200])
        except Exception as e:
            logger.warning("[delegate_task] Claude Code routing error: %s — falling back", e)

    # ── Tier 2: DeepSeek direct CC fallback ─────────────────────────────
    if remaining_indices:
        coding_cfg = cfg.get("coding", {})
        fd_cfg = coding_cfg.get("fallback_deepseek_cc", {})
        if fd_cfg.get("enabled", True):
            for i in sorted(remaining_indices):
                t = task_list[i]
                repo_path = _task_repo_cache.get(i) or _detect_repo_path(t["goal"], t.get("context", ""))
                if not repo_path:
                    continue
                logger.info("[delegate_task] Tier 2 DeepSeek direct for task %d (repo: %s)", i, repo_path)
                try:
                    ds_result = _run_claude_code_deepseek_direct_delegation(
                        goal=t["goal"],
                        context=t.get("context", ""),
                        repo_path=repo_path,
                        max_iterations=t.get("max_iterations") or effective_max_iter,
                        parent_agent=parent_agent,
                        skip_hitl=True,
                        profile_cfg=profile_cfg,
                    )
                    if "error" not in ds_result:
                        _verified = ds_result.get("verified", True)
                        _v_reason = ds_result.get("verification_reason", "")
                        _status = "completed" if _verified else "completed_unverified"
                        _error = None
                        if not _verified:
                            _error = f"⚠️ VERIFICATION FAILED: {_v_reason}"
                        claude_code_results[i] = {
                            "task_index": i,
                            "status": _status,
                            "summary": ds_result.get("result", ""),
                            "error": _error,
                            "api_calls": ds_result.get("num_turns", 0),
                            "duration_seconds": ds_result.get("duration_seconds", 0),
                            "delegation": "claude-code-deepseek-direct",
                        }
                        remaining_indices.discard(i)
                        continue
                    if ds_result.get("rate_limited"):
                        logger.warning("[delegate_task] DeepSeek direct rate-limited for task %d", i)
                    else:
                        logger.warning("[delegate_task] DeepSeek direct failed: %s", str(ds_result.get("error", ""))[:200])
                except Exception as e:
                    logger.warning("[delegate_task] DeepSeek direct routing error: %s", e)

    # If all tasks were handled by CC, skip the in-process subagent path entirely.
    if not remaining_indices:
        results = list(claude_code_results.values())
        results.sort(key=lambda r: r["task_index"])
        total_duration = round(time.monotonic() - overall_start, 2)
        return json.dumps({"results": results, "total_duration_seconds": total_duration}, ensure_ascii=False)

    remaining_task_list = [(i, task_list[i]) for i in sorted(remaining_indices)]

    results = []
    # Pre-populate results with CC-handled tasks so they appear in the final JSON.
    for cc_r in sorted(claude_code_results.values(), key=lambda r: r["task_index"]):
        results.append(cc_r)

    n_tasks = len(remaining_task_list)
    # Track goal labels for progress display (truncated for readability)
    task_labels = [t["goal"][:40] for _, t in remaining_task_list]

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
        for i, t in remaining_task_list:
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
                profile_cfg=profile_cfg,  # P146/MOL-496 — inject profile into child agent
            )
            # Override with correct parent tool names (before child construction mutated global)
            child._delegate_saved_tool_names = _parent_tool_names
            # P141/MOL-491 (#163 review CRITICAL-2): tag child with the
            # active profile name so iteration-audit records distinguish
            # coding-profile dispatches from default-profile ones. Without
            # this, every record reads `profile=default` (since no other
            # set-site exists) — making the audit log unable to answer
            # "is the coding profile hitting max_iterations more often?"
            child._delegate_profile = profile or "default"
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
        "TWO MODES (one of 'goal' or 'tasks' is required):\n"
        "1. Single task: provide 'goal' (+ optional context, toolsets)\n"
        "2. Batch (parallel): provide 'tasks' array with up to delegation.max_concurrent_children items (default 3, configurable via config.yaml, no hard ceiling). "
        "All run concurrently and results are returned together. Nested delegation requires role='orchestrator' and delegation.max_spawn_depth >= 2.\n\n"
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
        "notify_on_complete=True) instead. delegate_task runs SYNCHRONOUSLY "
        "inside the parent turn: if the parent is interrupted (user sends a "
        "new message, /stop, /new) the child is cancelled with status="
        "'interrupted' and its work is discarded. Children cannot continue "
        "in the background.\n\n"
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
