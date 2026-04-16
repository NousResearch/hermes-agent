#!/usr/bin/env python3
"""
Delegate Tool -- Subagent Architecture

Spawns subagent AIAgent instances with isolated context, restricted toolsets,
and their own terminal sessions. Supports single-task and batch (parallel)
modes. The parent blocks until all subagents complete.

Each subagent gets:
  - A fresh conversation (no parent history)
  - Its own task_id (own terminal session, file ops cache)
  - A restricted toolset (configurable, with blocked tools always stripped)
  - A focused system prompt built from the delegated goal + context

The parent's context only sees the delegation call and the summary result,
never the subagent's intermediate tool calls or reasoning.
"""

import json
import logging
logger = logging.getLogger(__name__)
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home
from toolsets import TOOLSETS
from utils import atomic_json_write


# Tools that subagents must never have access to
DELEGATE_BLOCKED_TOOLS = frozenset([
    "delegate_task",   # no recursive delegation
    "clarify",         # no user interaction
    "memory",          # no writes to shared MEMORY.md
    "send_message",    # no cross-platform side effects
    "execute_code",    # children should reason step-by-step, not write scripts
])

# Build a description fragment listing toolsets available for subagents.
# Excludes toolsets where ALL tools are blocked, composite/platform toolsets
# (hermes-* prefixed), and scenario toolsets.
_EXCLUDED_TOOLSET_NAMES = frozenset({"debugging", "safe", "delegation", "moa", "rl"})
_SUBAGENT_TOOLSETS = sorted(
    name for name, defn in TOOLSETS.items()
    if name not in _EXCLUDED_TOOLSET_NAMES
    and not name.startswith("hermes-")
    and not all(t in DELEGATE_BLOCKED_TOOLS for t in defn.get("tools", []))
)
_TOOLSET_LIST_STR = ", ".join(f"'{n}'" for n in _SUBAGENT_TOOLSETS)

_DEFAULT_MAX_CONCURRENT_CHILDREN = 3
MAX_DEPTH = 2  # parent (0) -> subagent (1) -> nested delegation rejected (2)


def _get_max_concurrent_children() -> int:
    """Read delegation.max_concurrent_children from config, falling back to
    DELEGATION_MAX_CONCURRENT_CHILDREN env var, then the default (3).

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
                "using default %d", val, _DEFAULT_MAX_CONCURRENT_CHILDREN,
            )
    env_val = os.getenv("DELEGATION_MAX_CONCURRENT_CHILDREN")
    if env_val:
        try:
            return max(1, int(env_val))
        except (TypeError, ValueError):
            pass
    return _DEFAULT_MAX_CONCURRENT_CHILDREN
DEFAULT_MAX_ITERATIONS = 50
_HEARTBEAT_INTERVAL = 30  # seconds between parent activity heartbeats during delegation
DEFAULT_CHILD_TIMEOUT_SECONDS = 3600  # 1 hour wall-clock budget per subagent
DEFAULT_TOOLSETS = ["terminal", "file", "web"]
_DELEGATION_REGISTRY_VERSION = 1
_DELEGATION_REGISTRY_MAX_RECORDS = 200
_DELEGATION_STALL_THRESHOLD_SECONDS = max(_HEARTBEAT_INTERVAL * 3, 90)
_DELEGATION_REGISTRY_LOCK = threading.RLock()

# Agent Activity topic for spawn/completion notifications.
# Resolved at runtime from config.  Re-read every call to pick up
# live topic ID changes without a gateway restart.
_AGENT_ACTIVITY_CHAT_ID = None
_AGENT_ACTIVITY_THREAD_ID = None
_AGENT_ACTIVITY_RESOLVED_AT: float = 0.0  # epoch timestamp of last resolution
_AGENT_ACTIVITY_TTL: float = 60.0  # seconds before re-reading config


def _delegation_registry_path() -> Path:
    return get_hermes_home() / "state" / "delegations.json"


def _registry_now_ts() -> float:
    return time.time()


def _timestamp_to_iso(ts: Any) -> Optional[str]:
    if ts in (None, ""):
        return None
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        return None


def _seconds_since(ts: Any, now: Optional[float] = None) -> Optional[float]:
    if ts in (None, ""):
        return None
    try:
        current = float(now if now is not None else _registry_now_ts())
        return round(max(0.0, current - float(ts)), 1)
    except Exception:
        return None


def _default_delegation_registry() -> Dict[str, Any]:
    return {
        "version": _DELEGATION_REGISTRY_VERSION,
        "updated_at_ts": 0.0,
        "delegations": {},
    }


def _load_delegation_registry() -> Dict[str, Any]:
    path = _delegation_registry_path()
    if not path.exists():
        return _default_delegation_registry()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not load delegation registry %s: %s", path, exc)
        return _default_delegation_registry()

    if not isinstance(payload, dict):
        return _default_delegation_registry()

    payload.setdefault("version", _DELEGATION_REGISTRY_VERSION)
    payload.setdefault("updated_at_ts", 0.0)
    delegations = payload.get("delegations")
    if not isinstance(delegations, dict):
        payload["delegations"] = {}
    return payload


def _prune_delegation_registry(payload: Dict[str, Any]) -> None:
    delegations = payload.get("delegations")
    if not isinstance(delegations, dict) or len(delegations) <= _DELEGATION_REGISTRY_MAX_RECORDS:
        return

    ordered = sorted(
        delegations.items(),
        key=lambda item: float((item[1] or {}).get("created_at_ts") or 0.0),
        reverse=True,
    )
    payload["delegations"] = dict(ordered[:_DELEGATION_REGISTRY_MAX_RECORDS])


def _write_delegation_registry(payload: Dict[str, Any]) -> None:
    payload = dict(payload or {})
    payload["version"] = _DELEGATION_REGISTRY_VERSION
    payload["updated_at_ts"] = _registry_now_ts()
    _prune_delegation_registry(payload)
    atomic_json_write(_delegation_registry_path(), payload)


def _mutate_delegation_registry(mutator) -> Any:
    with _DELEGATION_REGISTRY_LOCK:
        payload = _load_delegation_registry()
        result = mutator(payload)
        _write_delegation_registry(payload)
        return result


def _normalize_task_status(raw_status: Any) -> str:
    status = str(raw_status or "").strip().lower()
    if status in {"spawned", "running", "stalled", "completed", "completed_with_issues", "failed", "timed_out", "interrupted"}:
        return status
    if status in {"error", "exception"}:
        return "failed"
    if status in {"timeout"}:
        return "timed_out"
    return status or "failed"


def _make_registry_task_entry(
    task_index: int,
    task: Dict[str, Any],
    *,
    now_ts: float,
    task_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    task_meta = task_meta or {}
    session_id = str(task_meta.get("session_id") or "").strip()
    return {
        "task_index": task_index,
        "position": int(task_meta.get("position") or (task_index + 1)),
        "task_class": str(task_meta.get("task_class") or _infer_delegation_task_class(task.get("goal", ""), task.get("context"), None)).strip() or "general",
        "task_icon": str(task_meta.get("task_icon") or _task_class_icon(task_meta.get("task_class") or "general")).strip() or "🧩",
        "goal_preview": str(task_meta.get("goal_preview") or _sentence_preview(task.get("goal", ""), 160)),
        "session_id": session_id,
        "session_id_short": str(task_meta.get("session_id_short") or _short_identifier(session_id or task_index + 1)),
        "status": "spawned",
        "created_at_ts": now_ts,
        "updated_at_ts": now_ts,
        "started_at_ts": None,
        "finished_at_ts": None,
        "last_heartbeat_at_ts": now_ts,
        "last_activity": {},
    }


def _register_background_delegation(
    spawn_id: str,
    task_list: List[Dict[str, Any]],
    *,
    requested_deliver: str,
    resolved_deliver: str,
    activity_meta: Optional[Dict[str, Any]] = None,
) -> None:
    now_ts = _registry_now_ts()
    tasks_meta = (activity_meta or {}).get("tasks") or []
    task_meta_by_index = {
        int(meta.get("task_index")): meta
        for meta in tasks_meta
        if isinstance(meta, dict) and isinstance(meta.get("task_index"), int)
    }

    def _mutator(payload: Dict[str, Any]) -> None:
        payload.setdefault("delegations", {})
        payload["delegations"][spawn_id] = {
            "spawn_id": spawn_id,
            "status": "spawned",
            "created_at_ts": now_ts,
            "updated_at_ts": now_ts,
            "started_at_ts": None,
            "finished_at_ts": None,
            "last_heartbeat_at_ts": now_ts,
            "stall_threshold_seconds": _DELEGATION_STALL_THRESHOLD_SECONDS,
            "task_count": len(task_list),
            "runtime_label": str((activity_meta or {}).get("runtime_label") or "").strip(),
            "profile_label": str((activity_meta or {}).get("profile_label") or "").strip(),
            "origin_label": str((activity_meta or {}).get("origin_label") or "").strip(),
            "deliver_label": str((activity_meta or {}).get("deliver_label") or requested_deliver or resolved_deliver).strip(),
            "task_class_summary": str((activity_meta or {}).get("task_class_summary") or "").strip(),
            "deliver": {
                "requested": requested_deliver,
                "resolved": resolved_deliver,
                "status": "pending",
                "delivered_at_ts": None,
                "last_error": None,
            },
            "tasks": [
                _make_registry_task_entry(i, task, now_ts=now_ts, task_meta=task_meta_by_index.get(i))
                for i, task in enumerate(task_list)
            ],
        }

    _mutate_delegation_registry(_mutator)


def _find_registry_task(entry: Dict[str, Any], task_index: int) -> Optional[Dict[str, Any]]:
    tasks = entry.get("tasks") or []
    for task in tasks:
        if int(task.get("task_index", -1)) == int(task_index):
            return task
    return None


def _mark_background_delegation_running(spawn_id: str) -> None:
    now_ts = _registry_now_ts()

    def _mutator(payload: Dict[str, Any]) -> None:
        entry = (payload.get("delegations") or {}).get(spawn_id)
        if not isinstance(entry, dict):
            return
        entry["status"] = "running"
        entry["started_at_ts"] = entry.get("started_at_ts") or now_ts
        entry["updated_at_ts"] = now_ts
        entry["last_heartbeat_at_ts"] = now_ts

    _mutate_delegation_registry(_mutator)


def _update_background_task_heartbeat(
    spawn_id: str,
    task_index: int,
    *,
    activity: Optional[Dict[str, Any]] = None,
) -> None:
    now_ts = _registry_now_ts()
    last_activity = {}
    if isinstance(activity, dict):
        last_activity = {
            "description": activity.get("last_activity_desc"),
            "current_tool": activity.get("current_tool"),
            "api_call_count": activity.get("api_call_count"),
            "max_iterations": activity.get("max_iterations"),
            "seconds_since_activity": activity.get("seconds_since_activity"),
        }

    def _mutator(payload: Dict[str, Any]) -> None:
        entry = (payload.get("delegations") or {}).get(spawn_id)
        if not isinstance(entry, dict):
            return
        task = _find_registry_task(entry, task_index)
        if task is None:
            return
        if task.get("status") in {"completed", "completed_with_issues", "failed", "timed_out", "interrupted"}:
            return
        entry["status"] = "running"
        entry["started_at_ts"] = entry.get("started_at_ts") or now_ts
        entry["updated_at_ts"] = now_ts
        entry["last_heartbeat_at_ts"] = now_ts
        task["status"] = "running"
        task["started_at_ts"] = task.get("started_at_ts") or now_ts
        task["updated_at_ts"] = now_ts
        task["last_heartbeat_at_ts"] = now_ts
        if last_activity:
            task["last_activity"] = {k: v for k, v in last_activity.items() if v not in (None, "")}

    _mutate_delegation_registry(_mutator)


def _record_background_task_result(spawn_id: str, task_index: int, result: Dict[str, Any]) -> None:
    now_ts = _registry_now_ts()
    task_status = _normalize_task_status((result or {}).get("status"))

    def _mutator(payload: Dict[str, Any]) -> None:
        entry = (payload.get("delegations") or {}).get(spawn_id)
        if not isinstance(entry, dict):
            return
        task = _find_registry_task(entry, task_index)
        if task is None:
            return
        task["raw_status"] = str((result or {}).get("status") or "")
        task["status"] = task_status
        task["updated_at_ts"] = now_ts
        task["finished_at_ts"] = now_ts
        task["last_heartbeat_at_ts"] = now_ts
        task["duration_seconds"] = (result or {}).get("duration_seconds")
        task["api_calls"] = (result or {}).get("api_calls")
        task["exit_reason"] = (result or {}).get("exit_reason")
        task["model"] = (result or {}).get("model")
        if (result or {}).get("summary"):
            task["summary_preview"] = _truncate_preview((result or {}).get("summary"), 240)
        if (result or {}).get("error"):
            task["error"] = _truncate_preview((result or {}).get("error"), 240)
        if isinstance((result or {}).get("last_activity"), dict):
            task["last_activity"] = {
                "description": result["last_activity"].get("description"),
                "current_tool": result["last_activity"].get("current_tool"),
                "api_call_count": result["last_activity"].get("api_calls"),
                "max_iterations": result["last_activity"].get("max_iterations"),
            }
        entry["updated_at_ts"] = now_ts
        entry["last_heartbeat_at_ts"] = now_ts

    _mutate_delegation_registry(_mutator)


def _compute_terminal_spawn_status(statuses: List[str]) -> str:
    normalized = [_normalize_task_status(status) for status in statuses]
    if not normalized:
        return "failed"
    if all(status == "completed" for status in normalized):
        return "completed"
    if all(status == "timed_out" for status in normalized):
        return "timed_out"
    if all(status in {"failed", "interrupted"} for status in normalized):
        return "failed"
    return "completed_with_issues"


def _update_background_delivery_status(spawn_id: str, *, ok: bool, error: Optional[str] = None) -> None:
    now_ts = _registry_now_ts()

    def _mutator(payload: Dict[str, Any]) -> None:
        entry = (payload.get("delegations") or {}).get(spawn_id)
        if not isinstance(entry, dict):
            return
        deliver = entry.setdefault("deliver", {})
        deliver["status"] = "delivered" if ok else "failed"
        deliver["delivered_at_ts"] = now_ts
        deliver["last_error"] = _truncate_preview(error, 240) if error else None
        entry["updated_at_ts"] = now_ts

    _mutate_delegation_registry(_mutator)


def _finalize_background_delegation(
    spawn_id: str,
    *,
    total_duration_seconds: Optional[float] = None,
    fatal_error: Optional[str] = None,
) -> None:
    now_ts = _registry_now_ts()

    def _mutator(payload: Dict[str, Any]) -> None:
        entry = (payload.get("delegations") or {}).get(spawn_id)
        if not isinstance(entry, dict):
            return
        tasks = entry.get("tasks") or []
        if fatal_error:
            for task in tasks:
                current_status = _normalize_task_status(task.get("status"))
                if current_status in {"spawned", "running", "stalled"}:
                    task["status"] = "failed"
                    task["error"] = _truncate_preview(fatal_error, 240)
                    task["finished_at_ts"] = now_ts
                    task["updated_at_ts"] = now_ts
                    task["last_heartbeat_at_ts"] = now_ts
            entry["runner_error"] = _truncate_preview(fatal_error, 240)

        terminal_statuses = [_normalize_task_status(task.get("status")) for task in tasks]
        entry["status"] = _compute_terminal_spawn_status(terminal_statuses)
        entry["updated_at_ts"] = now_ts
        entry["finished_at_ts"] = now_ts
        entry["last_heartbeat_at_ts"] = now_ts
        if total_duration_seconds is not None:
            entry["duration_seconds"] = round(float(total_duration_seconds), 2)

    _mutate_delegation_registry(_mutator)


def _derive_task_status(task: Dict[str, Any], now_ts: Optional[float] = None) -> str:
    now_ts = float(now_ts if now_ts is not None else _registry_now_ts())
    status = _normalize_task_status(task.get("status"))
    if status in {"spawned", "running"}:
        heartbeat_ts = (
            task.get("last_heartbeat_at_ts")
            or task.get("updated_at_ts")
            or task.get("started_at_ts")
            or task.get("created_at_ts")
        )
        age = _seconds_since(heartbeat_ts, now_ts)
        if age is not None and age >= _DELEGATION_STALL_THRESHOLD_SECONDS:
            return "stalled"
    return status


def _derive_spawn_status(entry: Dict[str, Any], now_ts: Optional[float] = None) -> str:
    now_ts = float(now_ts if now_ts is not None else _registry_now_ts())
    raw_status = str(entry.get("status") or "").strip().lower()
    tasks = entry.get("tasks") or []
    if not tasks:
        heartbeat_ts = entry.get("last_heartbeat_at_ts") or entry.get("updated_at_ts") or entry.get("created_at_ts")
        age = _seconds_since(heartbeat_ts, now_ts)
        if raw_status in {"spawned", "running"} and age is not None and age >= _DELEGATION_STALL_THRESHOLD_SECONDS:
            return "stalled"
        return raw_status or "unknown"

    task_statuses = [_derive_task_status(task, now_ts) for task in tasks]
    unfinished = [status for status in task_statuses if status in {"spawned", "running", "stalled"}]
    if raw_status in {"failed", "timed_out"} and all(_normalize_task_status(task.get("status")) == "spawned" for task in tasks):
        return raw_status
    if unfinished:
        if all(status == "stalled" for status in unfinished):
            return "stalled"
        if any(status == "running" for status in unfinished):
            return "running"
        return "spawned"
    return _compute_terminal_spawn_status(task_statuses)


def _serialize_delegation_entry(
    entry: Dict[str, Any],
    *,
    now_ts: Optional[float] = None,
    include_tasks: bool = True,
) -> Dict[str, Any]:
    now_ts = float(now_ts if now_ts is not None else _registry_now_ts())
    tasks = sorted(entry.get("tasks") or [], key=lambda item: int(item.get("task_index", 0)))
    serialized_tasks: List[Dict[str, Any]] = []
    task_status_counts: Dict[str, int] = {}
    for task in tasks:
        derived_status = _derive_task_status(task, now_ts)
        task_status_counts[derived_status] = task_status_counts.get(derived_status, 0) + 1
        if include_tasks:
            serialized_tasks.append({
                "task_index": task.get("task_index"),
                "position": task.get("position"),
                "task_class": task.get("task_class"),
                "goal_preview": task.get("goal_preview"),
                "session_id": task.get("session_id"),
                "session_id_short": task.get("session_id_short"),
                "status": derived_status,
                "raw_status": task.get("status"),
                "created_at": _timestamp_to_iso(task.get("created_at_ts")),
                "started_at": _timestamp_to_iso(task.get("started_at_ts")),
                "finished_at": _timestamp_to_iso(task.get("finished_at_ts")),
                "updated_at": _timestamp_to_iso(task.get("updated_at_ts")),
                "last_heartbeat_at": _timestamp_to_iso(task.get("last_heartbeat_at_ts")),
                "seconds_since_heartbeat": _seconds_since(task.get("last_heartbeat_at_ts"), now_ts),
                "duration_seconds": task.get("duration_seconds"),
                "api_calls": task.get("api_calls"),
                "exit_reason": task.get("exit_reason"),
                "model": task.get("model"),
                "summary_preview": task.get("summary_preview"),
                "error": task.get("error"),
                "last_activity": task.get("last_activity") or None,
            })

    deliver = dict(entry.get("deliver") or {})
    deliver["delivered_at"] = _timestamp_to_iso(deliver.get("delivered_at_ts"))

    payload = {
        "spawn_id": entry.get("spawn_id"),
        "status": _derive_spawn_status(entry, now_ts),
        "raw_status": entry.get("status"),
        "task_count": len(tasks),
        "task_status_counts": task_status_counts,
        "created_at": _timestamp_to_iso(entry.get("created_at_ts")),
        "started_at": _timestamp_to_iso(entry.get("started_at_ts")),
        "finished_at": _timestamp_to_iso(entry.get("finished_at_ts")),
        "updated_at": _timestamp_to_iso(entry.get("updated_at_ts")),
        "last_heartbeat_at": _timestamp_to_iso(entry.get("last_heartbeat_at_ts")),
        "seconds_since_heartbeat": _seconds_since(entry.get("last_heartbeat_at_ts"), now_ts),
        "stall_threshold_seconds": entry.get("stall_threshold_seconds", _DELEGATION_STALL_THRESHOLD_SECONDS),
        "duration_seconds": entry.get("duration_seconds"),
        "profile_label": entry.get("profile_label"),
        "runtime_label": entry.get("runtime_label"),
        "origin_label": entry.get("origin_label"),
        "deliver_label": entry.get("deliver_label"),
        "task_class_summary": entry.get("task_class_summary"),
        "deliver": deliver,
        "runner_error": entry.get("runner_error"),
    }
    if include_tasks:
        payload["tasks"] = serialized_tasks
    return payload


def delegate_status(
    spawn_id: Optional[str] = None,
    *,
    limit: int = 10,
    active_only: bool = False,
    include_tasks: bool = True,
) -> str:
    """Return direct Hermes delegation status from the persistent registry."""
    with _DELEGATION_REGISTRY_LOCK:
        payload = _load_delegation_registry()
    delegations = payload.get("delegations") or {}
    now_ts = _registry_now_ts()

    if spawn_id:
        entry = delegations.get(str(spawn_id).strip())
        if not isinstance(entry, dict):
            return json.dumps({"error": f"No delegation found for spawn_id '{spawn_id}'."}, ensure_ascii=False)
        return json.dumps(_serialize_delegation_entry(entry, now_ts=now_ts, include_tasks=include_tasks), ensure_ascii=False)

    try:
        safe_limit = max(1, min(int(limit), 100))
    except Exception:
        safe_limit = 10

    entries = sorted(
        delegations.values(),
        key=lambda item: float((item or {}).get("created_at_ts") or 0.0),
        reverse=True,
    )
    serialized = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        serialized_entry = _serialize_delegation_entry(entry, now_ts=now_ts, include_tasks=include_tasks)
        if active_only and serialized_entry.get("status") not in {"spawned", "running", "stalled"}:
            continue
        serialized.append(serialized_entry)
        if len(serialized) >= safe_limit:
            break

    return json.dumps({
        "delegations": serialized,
        "count": len(serialized),
    }, ensure_ascii=False)


def _resolve_agent_activity_target() -> Optional[Dict[str, Any]]:
    """Resolve Agent Activity topic from config.yaml telegram.extra.group_topics.

    Results are cached for _AGENT_ACTIVITY_TTL seconds so config changes
    (topic ID updates, new topics) are picked up without a full restart.
    """
    global _AGENT_ACTIVITY_CHAT_ID, _AGENT_ACTIVITY_THREAD_ID, _AGENT_ACTIVITY_RESOLVED_AT
    if _AGENT_ACTIVITY_CHAT_ID is not None and (time.time() - _AGENT_ACTIVITY_RESOLVED_AT) < _AGENT_ACTIVITY_TTL:
        return {"chat_id": _AGENT_ACTIVITY_CHAT_ID, "thread_id": _AGENT_ACTIVITY_THREAD_ID}
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        for group in cfg.get("telegram", {}).get("extra", {}).get("group_topics", []):
            for topic in group.get("topics", []):
                if topic.get("name", "").lower().strip() in ("agent activity",):
                    _AGENT_ACTIVITY_CHAT_ID = str(group["chat_id"])
                    _AGENT_ACTIVITY_THREAD_ID = str(topic["thread_id"])
                    _AGENT_ACTIVITY_RESOLVED_AT = time.time()
                    return {"chat_id": _AGENT_ACTIVITY_CHAT_ID, "thread_id": _AGENT_ACTIVITY_THREAD_ID}
    except Exception as exc:
        logger.debug("Could not resolve Agent Activity topic: %s", exc)
    return None


def _send_telegram_sync(
    chat_id: str,
    thread_id: Optional[str],
    text: str,
    *,
    context: str = "delegate_task",
    target_label: Optional[str] = None,
) -> bool:
    """Fire-and-forget synchronous Telegram message via Bot API (urllib, no deps)."""
    import urllib.error
    import urllib.parse
    import urllib.request

    resolved_target = f"telegram:{chat_id}"
    if thread_id:
        resolved_target += f":{thread_id}"
    label = target_label or resolved_target

    try:
        from pathlib import Path
        env_path = Path(os.path.expanduser("~/.hermes/.env"))
        token = None
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "TELEGRAM_BOT_TOKEN":
                    token = v.strip()
                    break
        if not token:
            token = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("HERMES_TELEGRAM_BOT_TOKEN")
        if not token:
            logger.debug("No Telegram bot token found for %s delivery to %s", context, label)
            return False

        params = {"chat_id": chat_id, "text": text, "disable_web_page_preview": "true"}
        if thread_id:
            params["message_thread_id"] = thread_id
        data = urllib.parse.urlencode(params).encode()
        req = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage", data=data)
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read().decode())
            ok = body.get("ok", False)
            if not ok:
                logger.warning(
                    "Telegram delivery returned ok=false for %s target=%s resolved=%s body=%s",
                    context,
                    label,
                    resolved_target,
                    str(body)[:500],
                )
            return ok
    except urllib.error.HTTPError as exc:
        try:
            error_body = exc.read().decode(errors="replace")
        except Exception:
            error_body = ""
        logger.warning(
            "Telegram delivery failed for %s: status=%s target=%s resolved=%s body=%s",
            context,
            getattr(exc, "code", "?"),
            label,
            resolved_target,
            error_body[:500],
        )
        return False
    except Exception as exc:
        logger.warning(
            "Telegram delivery failed for %s: target=%s resolved=%s error=%s",
            context,
            label,
            resolved_target,
            exc,
        )
        return False


def _parse_deliver_target(deliver: str) -> Optional[Dict[str, str]]:
    """Parse a deliver target string like 'telegram:chat_id:thread_id' or a topic name."""
    if not deliver or not deliver.strip():
        return None
    deliver = deliver.strip()

    # Format: telegram:chat_id:thread_id
    if deliver.startswith("telegram:"):
        parts = deliver.split(":")
        if len(parts) >= 3:
            return {"chat_id": parts[1], "thread_id": parts[2]}
        elif len(parts) == 2:
            return {"chat_id": parts[1], "thread_id": None}

    # Format: topic name — resolve from config
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        deliver_lower = deliver.lower().strip()
        for group in cfg.get("telegram", {}).get("extra", {}).get("group_topics", []):
            for topic in group.get("topics", []):
                if topic.get("name", "").lower().strip() == deliver_lower:
                    return {"chat_id": str(group["chat_id"]), "thread_id": str(topic["thread_id"])}
    except Exception as exc:
        logger.debug("Could not resolve deliver target '%s': %s", deliver, exc)

    return None


def _load_full_config() -> dict:
    """Load the full Hermes config when notification formatting needs routing metadata."""
    try:
        from hermes_cli.config import load_config
        return load_config() or {}
    except Exception:
        return {}


def _get_active_profile_label() -> str:
    try:
        from hermes_cli.profiles import get_active_profile_name
        profile = (get_active_profile_name() or "").strip()
        return profile or "default"
    except Exception:
        return "default"


def _short_identifier(value: Any, width: int = 8) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    compact = re.sub(r"[^A-Za-z0-9]", "", text)
    if compact:
        return compact[:width]
    return text[:width]


def _truncate_preview(text: Any, limit: int = 72) -> str:
    preview = re.sub(r"\s+", " ", str(text or "").strip())
    if len(preview) <= limit:
        return preview
    return preview[: max(1, limit - 1)].rstrip() + "…"


def _sentence_preview(text: Any, limit: int = 160) -> str:
    preview = re.sub(r"\s+", " ", str(text or "").strip())
    if not preview:
        return ""
    if len(preview) <= limit:
        return preview if preview[-1] in ".!?" else preview + "."

    sentence_match = re.match(r"^(.{1," + str(limit) + r"}?[.!?])(?:\s|$)", preview)
    if sentence_match:
        sentence = sentence_match.group(1).strip()
        return sentence if sentence[-1] in ".!?" else sentence + "."

    clipped = preview[:limit].rstrip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0].rstrip()
    clipped = clipped.rstrip(" ,;:-")
    if not clipped:
        clipped = preview[:limit].rstrip()
    return clipped + "."


def _format_duration_short(seconds: Any) -> str:
    try:
        total_seconds = max(0, int(round(float(seconds or 0))))
    except (TypeError, ValueError):
        total_seconds = 0

    if total_seconds < 60:
        return f"{total_seconds}s"

    minutes, secs = divmod(total_seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"

    hours, minutes = divmod(minutes, 60)
    if secs:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{hours}h {minutes:02d}m"


def _task_class_icon(task_class: str) -> str:
    return {
        "build_execution": "🛠️",
        "system_repair": "🩺",
        "research": "🔎",
    }.get((task_class or "").strip(), "🧩")


def _lookup_topic_name(chat_id: Any, thread_id: Any, cfg: Optional[dict] = None) -> str:
    chat_key = str(chat_id or "").strip()
    thread_key = str(thread_id or "").strip()
    if not chat_key or not thread_key:
        return ""
    cfg = cfg or _load_full_config()
    for group in cfg.get("telegram", {}).get("extra", {}).get("group_topics", []):
        if str(group.get("chat_id", "")).strip() != chat_key:
            continue
        for topic in group.get("topics", []):
            if str(topic.get("thread_id", "")).strip() == thread_key:
                return str(topic.get("name", "") or "").strip()
    return ""


def _format_origin_label(
    *,
    platform: str,
    chat_id: str,
    chat_name: str,
    thread_id: str,
    session_key: str,
    cfg: Optional[dict] = None,
) -> str:
    platform_label = (platform or "").strip().lower()
    pretty_platform = platform_label.capitalize() if platform_label else "Local"
    chat_label = (chat_name or "").strip()
    thread_label = (thread_id or "").strip()

    if platform_label == "telegram":
        topic_name = _lookup_topic_name(chat_id, thread_label, cfg)
        if chat_label and topic_name:
            return f"Telegram · {chat_label} / {topic_name} (#{thread_label})"
        if topic_name:
            return f"Telegram · {topic_name} (#{thread_label})"
        if chat_label and thread_label:
            return f"Telegram · {chat_label} / thread #{thread_label}"
        if chat_label:
            return f"Telegram · {chat_label}"
        if chat_id and thread_label:
            return f"Telegram · {chat_id}:{thread_label}"
        if chat_id:
            return f"Telegram · {chat_id}"

    if chat_label and thread_label:
        return f"{pretty_platform} · {chat_label} / thread {thread_label}"
    if chat_label:
        return f"{pretty_platform} · {chat_label}"
    if session_key:
        return f"{pretty_platform} · session {session_key}"
    if platform_label:
        return pretty_platform
    return "unknown origin"


def _format_deliver_label(
    requested_deliver: str,
    deliver_target: Optional[Dict[str, str]],
    *,
    source_chat_id: str = "",
    source_chat_name: str = "",
    cfg: Optional[dict] = None,
) -> str:
    requested = (requested_deliver or "").strip()
    if not deliver_target:
        return requested or "none"

    chat_id = str(deliver_target.get("chat_id") or "").strip()
    thread_id = str(deliver_target.get("thread_id") or "").strip()
    resolved = f"telegram:{chat_id}"
    if thread_id:
        resolved += f":{thread_id}"

    topic_name = _lookup_topic_name(chat_id, thread_id, cfg)
    if topic_name and chat_id and chat_id == str(source_chat_id or "").strip() and source_chat_name:
        friendly = f"{source_chat_name} / {topic_name} (#{thread_id})"
    elif topic_name:
        friendly = f"{topic_name} (#{thread_id})"
    else:
        friendly = resolved

    if requested and requested != resolved and requested != topic_name:
        return f"{requested} → {friendly}"
    return friendly


def _build_task_class_summary(tasks_meta: List[Dict[str, Any]]) -> str:
    counts: Dict[str, int] = {}
    order: List[str] = []
    for task in tasks_meta:
        task_class = str(task.get("task_class") or "general").strip() or "general"
        if task_class not in counts:
            counts[task_class] = 0
            order.append(task_class)
        counts[task_class] += 1

    if len(order) == 1:
        return order[0]
    return ", ".join(
        f"{task_class} ×{counts[task_class]}" if counts[task_class] > 1 else task_class
        for task_class in order
    )


def _capture_activity_metadata(
    task_list: List[Dict[str, Any]],
    children: List,
    parent_agent,
    *,
    requested_deliver: str,
    deliver_target: Optional[Dict[str, str]],
    spawn_id: str,
) -> Dict[str, Any]:
    cfg = _load_full_config()

    try:
        from gateway.session_context import get_session_env
    except Exception:
        def get_session_env(name: str, default: str = "") -> str:
            return os.getenv(name, default)

    platform = (get_session_env("HERMES_SESSION_PLATFORM", "") or getattr(parent_agent, "platform", "") or "").strip()
    chat_id = (get_session_env("HERMES_SESSION_CHAT_ID", "") or "").strip()
    chat_name = (get_session_env("HERMES_SESSION_CHAT_NAME", "") or "").strip()
    thread_id = (get_session_env("HERMES_SESSION_THREAD_ID", "") or "").strip()
    session_key = (get_session_env("HERMES_SESSION_KEY", "") or "").strip()

    reference_agent = children[0][2] if children else parent_agent
    runtime_model = str(getattr(reference_agent, "model", "") or getattr(parent_agent, "model", "") or "").strip()
    runtime_provider = str(getattr(reference_agent, "provider", "") or getattr(parent_agent, "provider", "") or "").strip()
    if runtime_model and runtime_provider:
        runtime_label = f"{runtime_model} via {runtime_provider}"
    else:
        runtime_label = runtime_model or runtime_provider

    tasks_meta = []
    for position, (task_index, task, child) in enumerate(children, start=1):
        task_class = _infer_delegation_task_class(task.get("goal", ""), task.get("context"), requested_deliver)
        playbook_label = _extract_playbook_label(task.get("context") or "")
        pipeline_label = _extract_pipeline_label(task.get("goal") or "", task.get("context") or "")
        task_entry = {
            "task_index": task_index,
            "position": position,
            "task_class": task_class,
            "task_icon": _task_class_icon(task_class),
            "goal_preview": _sentence_preview(task.get("goal", ""), 160),
            "session_id": str(getattr(child, "session_id", "") or "").strip(),
            "session_id_short": _short_identifier(getattr(child, "session_id", "") or position),
        }
        if playbook_label:
            task_entry["playbook"] = playbook_label
        if pipeline_label:
            task_entry["pipeline_stage"] = pipeline_label
        tasks_meta.append(task_entry)

    return {
        "spawn_id": spawn_id,
        "profile_label": _get_active_profile_label(),
        "runtime_label": runtime_label,
        "origin_label": _format_origin_label(
            platform=platform,
            chat_id=chat_id,
            chat_name=chat_name,
            thread_id=thread_id,
            session_key=session_key,
            cfg=cfg,
        ),
        "deliver_label": _format_deliver_label(
            requested_deliver,
            deliver_target,
            source_chat_id=chat_id,
            source_chat_name=chat_name,
            cfg=cfg,
        ),
        "task_class_summary": _build_task_class_summary(tasks_meta),
        "tasks": tasks_meta,
    }


def _build_activity_spawn_message(activity_meta: Dict[str, Any]) -> str:
    tasks_meta = activity_meta.get("tasks", [])
    plural = len(tasks_meta) != 1
    lines = [f"🚀 Spawned · {activity_meta.get('spawn_id', '')}"]

    origin_label = str(activity_meta.get("origin_label") or "").strip()
    if origin_label:
        lines.append(f"From: {origin_label}")

    deliver_label = str(activity_meta.get("deliver_label") or "none").strip() or "none"
    lines.append(f"To: {deliver_label}")

    runtime_label = str(activity_meta.get("runtime_label") or "").strip()
    if runtime_label:
        lines.append(f"Model: {runtime_label}")

    if plural:
        lines.append(f"Tasks: {len(tasks_meta)}")
        for task in tasks_meta:
            lines.append(f"• #{task.get('position')}: {task.get('goal_preview', '')}")
    else:
        task = tasks_meta[0] if tasks_meta else {}
        lines.append(f"Task: {task.get('goal_preview', '')}")

    # Include playbook and pipeline stage if available
    first_task = tasks_meta[0] if tasks_meta else {}
    playbook = first_task.get("playbook")
    pipeline_stage = first_task.get("pipeline_stage")
    if playbook:
        lines.append(f"Playbook: {playbook}")
    if pipeline_stage:
        lines.append(f"Pipeline: {pipeline_stage}")

    return "\n".join(lines)


def _build_activity_completion_message(
    activity_meta: Dict[str, Any],
    results: List[Dict[str, Any]],
    *,
    total_duration_seconds: float,
    deliver_ok: bool,
) -> str:
    tasks_meta = activity_meta.get("tasks", [])
    plural = len(tasks_meta) != 1
    result_map = {entry.get("task_index"): entry for entry in results}
    completed_count = sum(1 for entry in results if entry.get("status") == "completed")
    issue_count = max(0, len(results) - completed_count)

    if issue_count == 0:
        title = f"✅ Done · {activity_meta.get('spawn_id', '')}"
    else:
        title = f"⚠️ Issues · {activity_meta.get('spawn_id', '')}"

    lines = [title]

    origin_label = str(activity_meta.get("origin_label") or "").strip()
    if origin_label:
        lines.append(f"From: {origin_label}")

    if deliver_ok:
        lines.append(f"To: {activity_meta.get('deliver_label', 'none')}")
    else:
        lines.append(f"To: delivery failed → {activity_meta.get('deliver_label', 'none')}")

    runtime_label = str(activity_meta.get("runtime_label") or "").strip()
    if runtime_label:
        lines.append(f"Model: {runtime_label}")

    lines.append(
        f"Outcome: {completed_count}/{len(results)} completed"
        + (f" · {issue_count} issue{'s' if issue_count != 1 else ''}" if issue_count else "")
    )
    lines.append(f"Duration: {_format_duration_short(total_duration_seconds)}")

    if plural:
        lines.append(f"Tasks: {len(results)}")
        for task in tasks_meta:
            entry = result_map.get(task.get("task_index"), {})
            status = str(entry.get("status") or "unknown")
            status_icon = "✅" if status == "completed" else "⚠️"
            detail = entry.get("summary") or entry.get("error") or status
            detail_preview = _truncate_preview(detail, 180)
            lines.append(
                f"• #{task.get('position')} {status_icon} {task.get('goal_preview', '')}"
                + (f" — {detail_preview}" if detail_preview else "")
            )
    else:
        task = tasks_meta[0] if tasks_meta else {}
        entry = result_map.get(task.get("task_index"), {})
        status = str(entry.get("status") or "unknown")
        status_label = "done" if status == "completed" else status
        detail = entry.get("summary") or entry.get("error") or status
        lines.append(f"Task: {task.get('goal_preview', '')}")
        lines.append(f"Status: {status_label}")
        if detail:
            lines.append(f"Result: {_truncate_preview(detail, 220)}")

    # Include playbook and pipeline stage if available
    first_task = tasks_meta[0] if tasks_meta else {}
    playbook = first_task.get("playbook")
    pipeline_stage = first_task.get("pipeline_stage")
    if playbook:
        lines.append(f"Playbook: {playbook}")
    if pipeline_stage:
        lines.append(f"Pipeline: {pipeline_stage}")

    return "\n".join(lines)


def _background_delegation_runner(
    task_list: List[Dict[str, Any]],
    children: List,
    parent_agent,
    deliver_target: Optional[Dict[str, str]],
    activity_target: Optional[Dict[str, str]],
    spawn_id: str,
    activity_meta: Optional[Dict[str, Any]] = None,
    started_at: Optional[float] = None,
):
    """Run subagents in a background thread. On completion, deliver results and notify."""
    import model_tools as _mt

    start_ts = float(started_at if started_at is not None else time.monotonic())
    results = []
    max_children = _get_max_concurrent_children()
    n_tasks = len(task_list)

    if activity_meta is not None:
        try:
            _mark_background_delegation_running(spawn_id)
        except Exception:
            logger.exception("Background delegation %s could not be marked running", spawn_id)

    try:
        if n_tasks == 1:
            _i, _t, child = children[0]
            result = _run_single_child(0, _t["goal"], child, parent_agent)
            results.append(result)
        else:
            with ThreadPoolExecutor(max_workers=max_children) as executor:
                futures = {}
                for i, t, child in children:
                    future = executor.submit(_run_single_child, task_index=i, goal=t["goal"], child=child, parent_agent=parent_agent)
                    futures[future] = i
                for future in as_completed(futures):
                    try:
                        entry = future.result()
                    except Exception as exc:
                        idx = futures[future]
                        entry = {"task_index": idx, "status": "error", "summary": None, "error": str(exc), "api_calls": 0, "duration_seconds": 0}
                    results.append(entry)
            results.sort(key=lambda r: r["task_index"])
    except Exception as exc:
        logger.exception("Background delegation %s crashed", spawn_id)
        if activity_meta is not None:
            _finalize_background_delegation(
                spawn_id,
                total_duration_seconds=round(time.monotonic() - start_ts, 2),
                fatal_error=str(exc),
            )
        raise

    # Build completion summary
    summaries = []
    for i, entry in enumerate(results):
        status = entry.get("status", "unknown")
        dur = entry.get("duration_seconds", 0)
        goal_label = task_list[entry["task_index"]]["goal"][:60] if entry["task_index"] < len(task_list) else f"Task {i}"
        summary_text = entry.get("summary", "") or "(no output)"

        icon = "done" if status == "completed" else status
        summaries.append(f"[{icon}] {goal_label} ({dur}s)\n{summary_text}")

    full_report = "\n\n---\n\n".join(summaries)

    # Deliver results to target topic
    deliver_ok = True
    resolved_deliver = None
    if deliver_target:
        resolved_deliver = f"telegram:{deliver_target['chat_id']}"
        if deliver_target.get("thread_id"):
            resolved_deliver += f":{deliver_target['thread_id']}"
        # Truncate if too long for Telegram (4096 char limit)
        deliver_text = full_report
        if len(deliver_text) > 4000:
            deliver_text = deliver_text[:3950] + "\n\n[truncated — full output exceeds message limit]"
        deliver_ok = _send_telegram_sync(
            deliver_target["chat_id"],
            deliver_target.get("thread_id"),
            deliver_text,
            context="delegate_task result delivery",
            target_label=resolved_deliver,
        )
        if activity_meta is not None:
            error_text = None if deliver_ok else f"failed to send result delivery to {resolved_deliver}"
            _update_background_delivery_status(spawn_id, ok=deliver_ok, error=error_text)

    total_duration_seconds = round(time.monotonic() - start_ts, 2)
    if activity_meta is not None:
        _finalize_background_delegation(
            spawn_id,
            total_duration_seconds=total_duration_seconds,
        )

    # Post completion to Agent Activity (keep it short)
    if activity_target:
        if activity_meta is not None:
            activity_msg = _build_activity_completion_message(
                activity_meta,
                results,
                total_duration_seconds=total_duration_seconds,
                deliver_ok=deliver_ok,
            )
        else:
            all_ok = all(e.get("status") == "completed" for e in results)
            goal_preview = task_list[0]["goal"][:140] if len(task_list) == 1 else f"{len(task_list)} tasks"
            if all_ok:
                activity_msg = (
                    "✅ SUBAGENT DISPATCH CLOSED\n"
                    "Agent: Hermes subagent\n"
                    f"Task: {goal_preview}"
                )
            else:
                failed = [e for e in results if e.get("status") != "completed"]
                activity_msg = (
                    "⚠️ SUBAGENT DISPATCH CLOSED WITH ISSUES\n"
                    "Agent: Hermes subagent\n"
                    f"Task: {goal_preview}\n"
                    f"Issues: {len(failed)}"
                )
        _send_telegram_sync(
            activity_target["chat_id"],
            activity_target.get("thread_id"),
            activity_msg,
            context="delegate_task activity completion",
            target_label="Agent Activity",
        )

    # Notify parent's memory provider
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

    logger.info("Background delegation %s completed: %d tasks", spawn_id, len(results))


def _tool_result_status(content: Any) -> str:
    """Classify a tool result as ok/error without false positives.

    Tool outputs are often JSON strings that include an "error" key with a null
    value (for example terminal responses). Substring matching on the raw text
    misclassifies those as failures, so prefer structured inspection first.
    """
    if content is None:
        return "error"

    parsed = content
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError, ValueError):
            text = content.strip()
            if not text:
                return "error"
            return "error" if text.lower().startswith("error:") else "ok"

    if isinstance(parsed, dict):
        if parsed.get("error") is not None:
            return "error"
        if parsed.get("success") is False:
            return "error"

        inner_content = parsed.get("content")
        if isinstance(inner_content, dict):
            if inner_content.get("error") is not None:
                return "error"
            if inner_content.get("success") is False:
                return "error"

    return "ok"


def _coerce_timeout_seconds(raw_timeout: Any) -> Optional[int]:
    """Best-effort numeric timeout parsing for runtime-only values."""
    if raw_timeout is None or isinstance(raw_timeout, bool):
        return None
    if not isinstance(raw_timeout, (int, float, str)):
        return None

    try:
        timeout_seconds = int(float(raw_timeout))
    except (TypeError, ValueError):
        return None

    return timeout_seconds if timeout_seconds > 0 else None


def _get_subagent_timeout_seconds(cfg: Optional[dict] = None) -> Optional[int]:
    """Resolve the wall-clock budget for subagents."""
    if cfg is None:
        cfg = _load_config()

    raw = cfg.get("timeout_seconds")
    if raw in (None, ""):
        raw = os.getenv("HERMES_DELEGATION_TIMEOUT", DEFAULT_CHILD_TIMEOUT_SECONDS)

    try:
        timeout_seconds = int(float(raw))
    except (TypeError, ValueError):
        logger.warning(
            "delegation.timeout_seconds=%r is invalid; using default %ds",
            raw,
            DEFAULT_CHILD_TIMEOUT_SECONDS,
        )
        timeout_seconds = DEFAULT_CHILD_TIMEOUT_SECONDS

    return timeout_seconds if timeout_seconds > 0 else None


def _build_timeout_checkpoint(
    goal: str,
    timeout_seconds: Optional[int],
    activity: Optional[Dict[str, Any]] = None,
    partial_summary: str = "",
) -> str:
    """Build a structured checkpoint when a subagent hits the wall-clock limit."""
    timeout_seconds = int(timeout_seconds or DEFAULT_CHILD_TIMEOUT_SECONDS)
    timeout_minutes = max(1, timeout_seconds // 60)
    activity = activity or {}

    lines = [
        f"Subagent hit the {timeout_minutes}-minute wall-clock limit before finishing `{goal}`.",
    ]

    if partial_summary:
        lines.append("Partial summary:")
        lines.append(partial_summary)

    last_desc = str(activity.get("last_activity_desc") or "").strip()
    current_tool = str(activity.get("current_tool") or "").strip()
    api_calls = activity.get("api_call_count")
    max_iterations = activity.get("max_iterations")

    if current_tool or last_desc:
        detail = current_tool or last_desc
        if current_tool and last_desc and last_desc != current_tool:
            detail = f"{current_tool} ({last_desc})"
        lines.append(f"Last active work: {detail}")

    if isinstance(api_calls, int) and isinstance(max_iterations, int) and max_iterations > 0:
        lines.append(f"Iteration progress: {api_calls}/{max_iterations}")

    lines.append("Needs more time: yes")
    lines.append("Next step: check in with Dax, share the checkpoint, and ask whether to continue the subagent.")
    return "\n".join(lines)


def check_delegate_requirements() -> bool:
    """Delegation has no external requirements -- always available."""
    return True


def _select_research_domain_playbook(playbook_dir, combined: str, deliver_lower: str):
    alias_map = [
        (("seo research", " seo ", "local seo", "keywords", "title tags"), "seo-research.md"),
        (("gbp research", " google business profile", " gbp "), "gbp-research.md"),
        (("geo research", " generative engine optimization", " ai search ", " geo "), "geo-research.md"),
        (("website craft", "website research", "site teardown", "design patterns"), "website-research.md"),
        (("paid advertising", "google ads", "meta ads", "ads research"), "paid-advertising-research.md"),
        (("reviews", "reputation"), "reviews-reputation-research.md"),
        (("social media", "instagram", "facebook strategy", "tiktok"), "social-media-research.md"),
        (("agency business model", "business model"), "agency-business-model-research.md"),
        (("client market analysis", "market analysis"), "client-market-analysis-research.md"),
    ]
    padded = f" {combined} "
    for tokens, filename in alias_map:
        if any(token in padded or token in deliver_lower for token in tokens):
            candidate = playbook_dir / filename
            if candidate.exists():
                return candidate
    return None



def _auto_inject_playbook(goal: str, context: Optional[str], deliver: Optional[str]) -> Optional[str]:
    """Auto-detect task type and load matching playbook + reference files.

    Returns the loaded content to prepend to the subagent context, or None
    if no matching playbook is found.  Uses the deliver target and goal
    keywords to decide which playbook to inject.
    """
    from pathlib import Path

    goal_lower = (goal or "").lower()
    deliver_lower = (deliver or "").lower()
    context_lower = (context or "").lower()
    combined = f"{goal_lower} {deliver_lower} {context_lower}"

    PLAYBOOK_DIR = Path(os.path.expanduser(
        "~/Obsidian/Jarvis-Operations/🧠 Brain/playbooks"
    ))
    INDUSTRY_REF_DIR = Path(os.path.expanduser(
        "~/Obsidian/Jarvis-Operations/🏗️ Projects/rww/templates/industry-references"
    ))

    research_signals = (
        "seo" in combined or "lead" in combined or "market" in combined
        or "competitive" in combined or "research" in combined
        or deliver_lower in ("seo research", "leads", "geo research", "gbp research")
    )
    website_build_signals = (
        "build" in combined or "website" in goal_lower
        or deliver_lower in ("builds", "playbook - website build")
    )

    injected_parts = []

    if research_signals:
        pipeline_path = PLAYBOOK_DIR / "research-pipeline.md"
        if pipeline_path.exists():
            try:
                pipeline_text = pipeline_path.read_text(errors="ignore")
                injected_parts.append(
                    "=== AUTO-INJECTED PLAYBOOK: research-pipeline.md ===\n"
                    + pipeline_text
                    + "\n=== END PLAYBOOK ===\n"
                )
            except Exception as exc:
                logger.debug("Failed to load research pipeline playbook: %s", exc)

        domain_playbook_path = _select_research_domain_playbook(PLAYBOOK_DIR, combined, deliver_lower)
        if domain_playbook_path is not None:
            try:
                content_text = domain_playbook_path.read_text(errors="ignore")
                injected_parts.append(
                    f"=== AUTO-INJECTED PLAYBOOK: {domain_playbook_path.name} ===\n"
                    + content_text
                    + "\n=== END PLAYBOOK ===\n"
                )
            except Exception as exc:
                logger.debug("Failed to load research playbook %s: %s", domain_playbook_path.name, exc)

        if domain_playbook_path and domain_playbook_path.name == "seo-research.md":
            craft_seo_path = INDUSTRY_REF_DIR.parent / "craft-guides" / "craft-seo.md"
            if craft_seo_path.exists():
                try:
                    craft_text = craft_seo_path.read_text(errors="ignore")
                    injected_parts.append(
                        "=== AUTO-INJECTED REFERENCE: craft-seo.md ===\n"
                        + craft_text
                        + "\n=== END REFERENCE ===\n"
                    )
                except Exception as exc:
                    logger.debug("Failed to load craft-seo.md: %s", exc)

        for ref_file in sorted(INDUSTRY_REF_DIR.glob("*.md")):
            if ref_file.name.startswith("craft-"):
                continue
            industry_slug = ref_file.stem
            if industry_slug.replace("-", " ") in goal_lower or industry_slug.replace("-", " ") in context_lower:
                try:
                    ref_text = ref_file.read_text(errors="ignore")
                    injected_parts.append(
                        f"=== AUTO-INJECTED INDUSTRY REFERENCE: {ref_file.name} ===\n"
                        + ref_text
                        + "\n=== END INDUSTRY REFERENCE ===\n"
                    )
                except Exception as exc:
                    logger.debug("Failed to load industry reference %s: %s", ref_file.name, exc)
                break

    elif website_build_signals:
        playbook_path = PLAYBOOK_DIR / "website-build.md"
        if playbook_path.exists():
            try:
                content_text = playbook_path.read_text(errors="ignore")
                injected_parts.append(
                    "=== AUTO-INJECTED PLAYBOOK: website-build.md ===\n"
                    + content_text
                    + "\n=== END PLAYBOOK ===\n"
                )
            except Exception as exc:
                logger.debug("Failed to load build playbook: %s", exc)

    if injected_parts:
        logger.info(
            "Auto-injected %d playbook/reference file(s) for subagent (deliver=%s)",
            len(injected_parts), deliver,
        )
        return "\n\n".join(injected_parts)
    return None



def _infer_delegation_task_class(goal: str, context: Optional[str], deliver: Optional[str]) -> str:
    combined = " ".join(filter(None, [(goal or "").lower(), (context or "").lower(), (deliver or "").lower()]))
    if any(token in combined for token in ("build", "website", "landing page", "dashboard", "frontend", "ui")):
        return "build_execution"
    if any(token in combined for token in ("research", "seo", "market", "competitor", "lead", "geo")):
        return "research"
    if any(token in combined for token in ("fix", "debug", "repair", "incident", "broken", "failure")):
        return "system_repair"
    return "general"


def _extract_playbook_label(context: str) -> Optional[str]:
    """Extract playbook name from subagent context string.

    Looks for patterns like:
      === DOMAIN PLAYBOOK: gbp-research.md ===
      playbook: seo-research
    """
    m = re.search(r"DOMAIN PLAYBOOK:\s*(\S+\.md)", context or "")
    if m:
        name = m.group(1).replace(".md", "")
        return name
    m = re.search(r"playbook[:\s]+([a-z0-9_-]+(?:-research|-build|-pipeline))", (context or "").lower())
    if m:
        return m.group(1)
    return None


def _extract_pipeline_label(goal: str, context: str) -> Optional[str]:
    """Extract pipeline stage from goal/context strings.

    Looks for patterns like:
      [Stage 1 - Knowledge Assessment]
      ACTIVE STAGE: stage-2-web-research
      research-pipeline
    """
    m = re.search(r"\[Stage\s+(\d+)\s*[-–]\s*([^\]]+)\]", goal or "")
    if m:
        return f"Stage {m.group(1)}: {m.group(2).strip()}"
    m = re.search(r"ACTIVE STAGE:\s*(stage-\d+-\S+)", context or "")
    if m:
        raw = m.group(1)
        parts = raw.split("-", 2)
        if len(parts) >= 3:
            return f"Stage {parts[1]}: {parts[2].replace('-', ' ').title()}"
        return raw
    if "research pipeline" in (context or "").lower() or "research-pipeline" in (context or "").lower():
        return "Research Pipeline"
    return None


def _looks_like_underinformed_context(goal: str, context: Optional[str], workspace_path: Optional[str]) -> List[str]:
    notes: List[str] = []
    text = (context or "").strip()
    combined = f"{goal or ''} {text}".lower()
    needs_repo_context = any(token in combined for token in (
        "build", "code", "repo", "repository", "fix", "debug", "dashboard", "website", "frontend", "backend",
    ))
    has_path_hint = bool(re.search(r"(/Users/|~/|\.(py|ts|tsx|js|jsx|md|json|yaml|yml)\b)", f"{goal} {text}"))
    if needs_repo_context and not workspace_path and not has_path_hint:
        notes.append("No concrete workspace path or file path was supplied. Discover the repo/workdir before editing or running repo-specific commands.")
    if len(text) < 40:
        notes.append("Parent context is thin. Work from the stated goal, inspect source-of-truth files first, and verify assumptions before changing anything.")
    return notes


def _build_delegation_contract(
    goal: str,
    context: Optional[str],
    *,
    deliver: Optional[str],
    workspace_path: Optional[str],
    toolsets: Optional[List[str]],
    wall_clock_timeout_seconds: Optional[int],
) -> str:
    task_class = _infer_delegation_task_class(goal, context, deliver)
    missing_context_notes = _looks_like_underinformed_context(goal, context, workspace_path)
    success_lines = [
        "- Complete the delegated objective, not just analysis.",
        "- Verify the result with the available tools before returning.",
        "- Return a concise completion summary with files changed, findings, and remaining risks.",
    ]
    if deliver:
        success_lines.append(f"- Produce a result that is ready to be delivered to: {deliver}.")
    if task_class == "research":
        success_lines.append("- Ground claims in source material or tool output; do not invent findings.")
    elif task_class == "build_execution":
        success_lines.append("- If code changes are needed, identify the exact files touched and the verification run.")
    elif task_class == "system_repair":
        success_lines.append("- Prefer direct repair plus verification over speculative explanation.")

    lines = [
        "DELEGATION CONTRACT:",
        f"- Task class: {task_class}",
        f"- Objective: {goal}",
        f"- Deliver target: {deliver or 'none specified'}",
        f"- Workspace hint: {workspace_path or 'none supplied'}",
        f"- Toolsets: {', '.join(toolsets) if toolsets else 'inherit filtered parent toolsets'}",
    ]
    if wall_clock_timeout_seconds and wall_clock_timeout_seconds > 0:
        lines.append(f"- Time budget seconds: {int(wall_clock_timeout_seconds)}")
    lines.append("")
    lines.append("SUCCESS CRITERIA:")
    lines.extend(success_lines)
    if missing_context_notes:
        lines.append("")
        lines.append("MISSING CONTEXT RISKS:")
        lines.extend(f"- {note}" for note in missing_context_notes)
    if context and context.strip():
        lines.append("")
        lines.append("PARENT CONTEXT:")
        lines.append(context.strip())
    return "\n".join(lines)


def _build_subagent_system_prompt(
    goal: str,
    context: Optional[str] = None,
    *,
    workspace_path: Optional[str] = None,
    wall_clock_timeout_seconds: Optional[int] = None,
    deliver: Optional[str] = None,
    toolsets: Optional[List[str]] = None,
) -> str:
    """Build a focused system prompt for a subagent.

    When ``deliver`` is provided, auto-detects the task type and injects
    the matching playbook + reference files into the prompt so the subagent
    has full operational context regardless of what the parent passed.
    """
    auto_context = _auto_inject_playbook(goal, context, deliver)
    merged_context = context
    if auto_context:
        if merged_context and merged_context.strip():
            merged_context = auto_context + "\n\n" + merged_context
        else:
            merged_context = auto_context

    contract_text = _build_delegation_contract(
        goal,
        merged_context,
        deliver=deliver,
        workspace_path=workspace_path,
        toolsets=toolsets,
        wall_clock_timeout_seconds=wall_clock_timeout_seconds,
    )

    parts = [
        "You are a focused subagent working on a specific delegated task. "
        "You must complete this task fully using your available tools. "
        "Do not describe what you would do — execute it. Keep calling tools "
        "until the task is verifiably complete. If you encounter an error, "
        "diagnose and fix it rather than reporting failure immediately.",
        "",
        f"YOUR TASK:\n{goal}",
        f"\n{contract_text}",
    ]
    if workspace_path and str(workspace_path).strip():
        parts.append(
            "\nWORKSPACE PATH:\n"
            f"{workspace_path}\n"
            "Use this exact path for local repository/workdir operations unless the task explicitly says otherwise."
        )
    if wall_clock_timeout_seconds and wall_clock_timeout_seconds > 0:
        timeout_minutes = max(1, int(wall_clock_timeout_seconds // 60))
        parts.append(
            "\nTIME BUDGET:\n"
            f"You have up to {timeout_minutes} minutes of wall-clock time for this delegated run.\n"
            "If you cannot finish within that budget, stop cleanly and return a checkpoint that states:\n"
            "- What is already complete\n"
            "- What remains\n"
            "- Whether you need more time\n"
            "- The exact next step to resume from this checkpoint"
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
    """Best-effort local workspace hint for subagent prompts.

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


def _strip_blocked_tools(toolsets: List[str]) -> List[str]:
    """Remove toolsets that contain only blocked tools."""
    blocked_toolset_names = {
        "delegation", "clarify", "memory", "code_execution",
    }
    return [t for t in toolsets if t not in blocked_toolset_names]


def _build_subagent_progress_callback(task_index: int, parent_agent, task_count: int = 1) -> Optional[callable]:
    """Build a callback that relays subagent tool calls to the parent display.

    Two display paths:
      CLI:     prints tree-view lines above the parent's delegation spinner
      Gateway: batches tool names and relays to parent's progress callback

    Returns None if no display mechanism is available, in which case the
    subagent runs with no progress callback (identical to current behavior).
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


def _build_subagent_agent(
    task_index: int,
    goal: str,
    context: Optional[str],
    toolsets: Optional[List[str]],
    model: Optional[str],
    max_iterations: int,
    parent_agent,
    wall_clock_timeout_seconds: Optional[int] = None,
    # Credential overrides from delegation config (provider:model resolution)
    override_provider: Optional[str] = None,
    override_base_url: Optional[str] = None,
    override_api_key: Optional[str] = None,
    override_api_mode: Optional[str] = None,
    # ACP transport overrides — lets a non-ACP parent spawn ACP subagents
    override_acp_command: Optional[str] = None,
    override_acp_args: Optional[List[str]] = None,
    # Deliver target — passed through for auto-playbook injection
    deliver: Optional[str] = None,
):
    """
    Build a subagent AIAgent on the main thread (thread-safe construction).
    Returns the constructed subagent without running it.

    When override_* params are set (from delegation config), the subagent uses
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
            ts for name in parent_agent.valid_tool_names
            if (ts := model_tools.get_toolset_for_tool(name)) is not None
        }
    else:
        parent_toolsets = set(DEFAULT_TOOLSETS)

    # MCP server names from config are system-level capabilities — they should
    # pass through the parent intersection when explicitly requested, because
    # the parent may not have them in its derived toolset set.
    _mcp_server_names = set()
    try:
        from tools.mcp_tool import _load_mcp_config
        _mcp_server_names = set(_load_mcp_config().keys())
    except Exception:
        pass

    if toolsets:
        # Intersect with parent — subagent must not gain tools the parent lacks.
        # MCP servers bypass the intersection since they're system-level config.
        subagent_toolsets = _strip_blocked_tools(
            [t for t in toolsets if t in parent_toolsets or t in _mcp_server_names]
        )
    elif parent_agent and parent_enabled is not None:
        subagent_toolsets = _strip_blocked_tools(parent_enabled)
    elif parent_toolsets:
        subagent_toolsets = _strip_blocked_tools(sorted(parent_toolsets))
    else:
        subagent_toolsets = _strip_blocked_tools(DEFAULT_TOOLSETS)

    workspace_hint = _resolve_workspace_hint(parent_agent)
    subagent_prompt = _build_subagent_system_prompt(
        goal,
        context,
        workspace_path=workspace_hint,
        wall_clock_timeout_seconds=wall_clock_timeout_seconds,
        deliver=deliver,
        toolsets=subagent_toolsets,
    )
    # Extract parent's API key so subagents inherit auth (e.g. Nous Portal).
    parent_api_key = getattr(parent_agent, "api_key", None)
    if (not parent_api_key) and hasattr(parent_agent, "_client_kwargs"):
        parent_api_key = parent_agent._client_kwargs.get("api_key")

    # Build progress callback to relay tool calls to parent display
    subagent_progress_cb = _build_subagent_progress_callback(task_index, parent_agent)

    # Each subagent gets its own iteration budget capped at max_iterations
    # (configurable via delegation.max_iterations, default 50).  This means
    # total iterations across parent + subagents can exceed the parent's
    # max_iterations.  The user controls the per-subagent cap in config.yaml.

    subagent_thinking_cb = None
    if subagent_progress_cb:
        def _subagent_thinking(text: str) -> None:
            if not text:
                return
            try:
                subagent_progress_cb("_thinking", text)
            except Exception as e:
                logger.debug("Subagent thinking callback relay failed: %s", e)

        subagent_thinking_cb = _subagent_thinking

    # Resolve effective credentials: config override > parent inherit
    effective_model = model or parent_agent.model
    effective_provider = override_provider or getattr(parent_agent, "provider", None)
    effective_base_url = override_base_url or parent_agent.base_url
    effective_api_key = override_api_key or parent_api_key
    effective_api_mode = override_api_mode or getattr(parent_agent, "api_mode", None)
    effective_acp_command = override_acp_command or getattr(parent_agent, "acp_command", None)
    effective_acp_args = list(override_acp_args if override_acp_args is not None else (getattr(parent_agent, "acp_args", []) or []))

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

    subagent = AIAgent(
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
        enabled_toolsets=subagent_toolsets,
        quiet_mode=True,
        ephemeral_system_prompt=subagent_prompt,
        log_prefix=f"[subagent-{task_index}]",
        platform=parent_agent.platform,
        skip_context_files=True,
        skip_memory=True,
        clarify_callback=None,
        thinking_callback=subagent_thinking_cb,
        session_db=getattr(parent_agent, '_session_db', None),
        parent_session_id=getattr(parent_agent, 'session_id', None),
        providers_allowed=parent_agent.providers_allowed,
        providers_ignored=parent_agent.providers_ignored,
        providers_order=parent_agent.providers_order,
        provider_sort=parent_agent.provider_sort,
        tool_progress_callback=subagent_progress_cb,
        iteration_budget=None,  # fresh budget per subagent
    )
    subagent._print_fn = getattr(parent_agent, '_print_fn', None)
    # Set delegation depth so subagents can't recurse further.
    subagent._delegate_depth = getattr(parent_agent, '_delegate_depth', 0) + 1
    subagent._delegate_timeout_seconds = wall_clock_timeout_seconds

    # Share a credential pool with the subagent when possible so subagents can
    # rotate credentials on rate limits instead of getting pinned to one key.
    subagent_pool = _resolve_subagent_credential_pool(effective_provider, parent_agent)
    if subagent_pool is not None:
        subagent._credential_pool = subagent_pool

    # Register subagent for interrupt propagation
    if hasattr(parent_agent, '_active_children'):
        lock = getattr(parent_agent, '_active_children_lock', None)
        if lock:
            with lock:
                parent_agent._active_children.append(subagent)
        else:
            parent_agent._active_children.append(subagent)

    return subagent

def _run_single_subagent(
    task_index: int,
    goal: str,
    child=None,
    parent_agent=None,
    **_kwargs,
) -> Dict[str, Any]:
    """
    Run a pre-built subagent. Called from within a thread.
    Returns a structured result dict.
    """
    subagent_start = time.monotonic()
    spawn_id = str(getattr(child, "_delegate_spawn_id", "") or "").strip()

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

    if spawn_id:
        _update_background_task_heartbeat(
            spawn_id,
            task_index,
            activity={
                "last_activity_desc": "starting subagent",
                "api_call_count": 0,
                "max_iterations": getattr(child, "max_iterations", None),
            },
        )

    # Heartbeat: periodically propagate child activity to the parent so the
    # gateway inactivity timeout doesn't fire while the subagent is working.
    # Without this, the parent's _last_activity_ts freezes when delegate_task
    # starts and the gateway eventually kills the agent for "no activity".
    _heartbeat_stop = threading.Event()

    def _heartbeat_loop():
        while not _heartbeat_stop.wait(_HEARTBEAT_INTERVAL):
            if parent_agent is None:
                continue
            touch = getattr(parent_agent, '_touch_activity', None)
            if not touch:
                continue
            # Pull detail from the child's own activity tracker
            desc = f"delegate_task: subagent {task_index} working"
            child_summary = None
            try:
                child_summary = child.get_activity_summary()
                child_tool = child_summary.get("current_tool")
                child_iter = child_summary.get("api_call_count", 0)
                child_max = child_summary.get("max_iterations", 0)
                if child_tool:
                    desc = (f"delegate_task: subagent running {child_tool} "
                            f"(iteration {child_iter}/{child_max})")
                else:
                    child_desc = child_summary.get("last_activity_desc", "")
                    if child_desc:
                        desc = (f"delegate_task: subagent {child_desc} "
                                f"(iteration {child_iter}/{child_max})")
            except Exception:
                pass
            try:
                touch(desc)
            except Exception:
                pass
            if spawn_id:
                _update_background_task_heartbeat(spawn_id, task_index, activity=child_summary)

    _heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
    _heartbeat_thread.start()

    wall_clock_timeout = _coerce_timeout_seconds(
        getattr(child, "_delegate_timeout_seconds", None)
    )
    _timeout_triggered = threading.Event()
    _timeout_activity: Dict[str, Any] = {}
    _timeout_timer = None

    def _trigger_timeout() -> None:
        try:
            _timeout_activity.update(child.get_activity_summary())
        except Exception:
            pass
        _timeout_triggered.set()
        timeout_minutes = max(1, int((wall_clock_timeout or DEFAULT_CHILD_TIMEOUT_SECONDS) // 60))
        timeout_message = (
            f"Subagent wall-clock timeout reached after {timeout_minutes} minutes. "
            "Stop now and return a checkpoint with completed work, remaining work, "
            "and whether more time is needed."
        )
        try:
            child.interrupt(timeout_message)
        except Exception as exc:
            logger.debug("Failed to interrupt child on delegation timeout: %s", exc)

    if wall_clock_timeout and wall_clock_timeout > 0:
        _timeout_timer = threading.Timer(float(wall_clock_timeout), _trigger_timeout)
        _timeout_timer.daemon = True
        _timeout_timer.start()

    try:
        result = child.run_conversation(user_message=goal)

        # Flush any remaining batched progress to gateway
        if child_progress_cb and hasattr(child_progress_cb, '_flush'):
            try:
                child_progress_cb._flush()
            except Exception as e:
                logger.debug("Progress callback flush failed: %s", e)

        duration = round(time.monotonic() - subagent_start, 2)

        summary = result.get("final_response") or ""
        completed = result.get("completed", False)
        interrupted = result.get("interrupted", False)
        api_calls = result.get("api_calls", 0)
        timed_out = _timeout_triggered.is_set()

        if timed_out:
            status = "timed_out"
        elif interrupted:
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
                    result_meta = {
                        "result_bytes": len(content),
                        "status": _tool_result_status(content),
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
        if timed_out:
            exit_reason = "timeout"
        elif interrupted:
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
        if timed_out:
            checkpoint = _build_timeout_checkpoint(
                goal,
                wall_clock_timeout,
                activity=_timeout_activity,
                partial_summary=summary or "",
            )
            entry["summary"] = checkpoint
            entry["needs_more_time"] = True
            entry["timeout_seconds"] = int(wall_clock_timeout or DEFAULT_CHILD_TIMEOUT_SECONDS)
            entry["last_activity"] = {
                "description": _timeout_activity.get("last_activity_desc"),
                "current_tool": _timeout_activity.get("current_tool"),
                "api_calls": _timeout_activity.get("api_call_count"),
                "max_iterations": _timeout_activity.get("max_iterations"),
            }
        if status == "failed":
            entry["error"] = result.get("error", "Subagent did not produce a response.")
        if spawn_id:
            _record_background_task_result(spawn_id, task_index, entry)

        return entry

    except Exception as exc:
        duration = round(time.monotonic() - subagent_start, 2)
        logging.exception(f"[subagent-{task_index}] failed")
        entry = {
            "task_index": task_index,
            "status": "error",
            "summary": None,
            "error": str(exc),
            "api_calls": 0,
            "duration_seconds": duration,
        }
        if spawn_id:
            _record_background_task_result(spawn_id, task_index, entry)
        return entry

    finally:
        if _timeout_timer is not None:
            _timeout_timer.cancel()

        # Stop the heartbeat thread so it doesn't keep touching parent activity
        # after the child has finished (or failed).
        _heartbeat_stop.set()
        _heartbeat_thread.join(timeout=5)

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

        # Close tool resources (terminal sandboxes, browser daemons,
        # background processes, httpx clients) so subagent subprocesses
        # don't outlive the delegation.
        try:
            if hasattr(child, 'close'):
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
    background: bool = False,
    deliver: Optional[str] = None,
    parent_agent=None,
) -> str:
    """
    Spawn one or more child agents to handle delegated tasks.

    Supports two modes:
      - Single: provide goal (+ optional context, toolsets)
      - Batch:  provide tasks array [{goal, context, toolsets}, ...]

    When background=true, subagents run in a daemon thread and this function
    returns immediately with a spawn confirmation. Results are delivered to
    the 'deliver' target (a Telegram topic name or 'telegram:chat_id:thread_id')
    when the subagent completes. Spawn/completion events are posted to Agent Activity.

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
    child_timeout_seconds = _get_subagent_timeout_seconds(cfg)

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
        task_list = [{"goal": goal, "context": context, "toolsets": toolsets}]
    else:
        return tool_error("Provide either 'goal' (single task) or 'tasks' (batch).")

    if not task_list:
        return tool_error("No tasks provided.")

    # Validate each task has a goal and enrich thin contexts with a contract note.
    for i, task in enumerate(task_list):
        if not task.get("goal", "").strip():
            return tool_error(f"Task {i} is missing a 'goal'.")
        if not (task.get("context") or "").strip():
            task["context"] = (
                "No explicit parent context was provided. Start by inspecting the relevant source-of-truth files, "
                "discovering the correct workspace/path if needed, and verifying assumptions before making changes."
            )

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
                task_index=i, goal=t["goal"], context=t.get("context"),
                toolsets=t.get("toolsets") or toolsets, model=creds["model"],
                max_iterations=effective_max_iter, parent_agent=parent_agent,
                wall_clock_timeout_seconds=child_timeout_seconds,
                override_provider=creds["provider"], override_base_url=creds["base_url"],
                override_api_key=creds["api_key"],
                override_api_mode=creds["api_mode"],
                override_acp_command=t.get("acp_command") or acp_command or creds.get("command"),
                override_acp_args=t.get("acp_args") or acp_args or creds.get("args"),
                deliver=deliver,
            )
            # Override with correct parent tool names (before child construction mutated global)
            child._delegate_saved_tool_names = _parent_tool_names
            children.append((i, t, child))
    finally:
        # Authoritative restore: reset global to parent's tool names after all children built
        _model_tools._last_resolved_tool_names = _parent_tool_names

    # --- Background mode: launch daemon thread and return immediately ---
    if background:
        requested_deliver = (deliver or "").strip()
        if not requested_deliver:
            return tool_error(
                "background delegation requires a non-empty 'deliver' target. "
                "Use a topic name or 'telegram:chat_id:thread_id'."
            )

        deliver_target = _parse_deliver_target(requested_deliver)
        if not deliver_target:
            return tool_error(
                f"Could not resolve deliver target '{requested_deliver}'. "
                "Use a valid topic name from config.yaml telegram.extra.group_topics "
                "or an explicit 'telegram:chat_id:thread_id' target."
            )

        import hashlib
        spawn_id = hashlib.sha256(f"{time.time()}-{id(children)}".encode()).hexdigest()[:8]
        activity_target = _resolve_agent_activity_target()
        resolved_deliver = f"telegram:{deliver_target['chat_id']}"
        if deliver_target.get("thread_id"):
            resolved_deliver += f":{deliver_target['thread_id']}"
        background_started_at = time.monotonic()
        activity_meta = _capture_activity_metadata(
            task_list,
            children,
            parent_agent,
            requested_deliver=requested_deliver,
            deliver_target=deliver_target,
            spawn_id=spawn_id,
        )
        _register_background_delegation(
            spawn_id,
            task_list,
            requested_deliver=requested_deliver,
            resolved_deliver=resolved_deliver,
            activity_meta=activity_meta,
        )

        # Stamp spawn_id onto each child so _run_single_subagent can update
        # the delegation registry with heartbeats and task-level status.
        for _i, _t, child in children:
            child._delegate_spawn_id = spawn_id

        # Post spawn notification to Agent Activity (keep it short)
        if activity_target:
            activity_msg = _build_activity_spawn_message(activity_meta)
            _send_telegram_sync(
                activity_target["chat_id"],
                activity_target.get("thread_id"),
                activity_msg,
                context="delegate_task activity spawn",
                target_label="Agent Activity",
            )

        # Launch background thread
        bg_thread = threading.Thread(
            target=_background_delegation_runner,
            args=(task_list, children, parent_agent, deliver_target, activity_target, spawn_id, activity_meta, background_started_at),
            daemon=True,
        )
        bg_thread.start()

        # Return immediately so the parent agent can keep talking
        spawn_summary = {
            "status": "spawned_background",
            "spawn_id": spawn_id,
            "task_count": n_tasks,
            "tasks": [{"goal": t["goal"][:80]} for t in task_list],
            "deliver": requested_deliver,
            "resolved_deliver": resolved_deliver,
            "message": (
                f"Spawned {n_tasks} background subagent{'s' if n_tasks > 1 else ''}. "
                f"Results will be delivered to {requested_deliver} ({resolved_deliver}) when complete. "
                f"Spawn logged to Agent Activity. You can continue the conversation."
            ),
        }
        return json.dumps(spawn_summary, ensure_ascii=False)

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


def _resolve_subagent_credential_pool(effective_provider: Optional[str], parent_agent):
    """Resolve a credential pool for the subagent.

    Rules:
    1. Same provider as the parent -> share the parent's pool so cooldown state
       and rotation stay synchronized.
    2. Different provider -> try to load that provider's own pool.
    3. No pool available -> return None and let the subagent keep the inherited
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
            "Could not load credential pool for subagent provider '%s': %s",
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
    subagent inherits everything from the parent agent.

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
            "command": None,
            "args": [],
        }

    if not configured_provider:
        # No provider override — child inherits everything from parent
        return {
            "model": configured_model,
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
            "command": None,
            "args": [],
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
            f"Examples: openai-codex, openrouter, nous, zai, kimi-coding, minimax."
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


# Backwards-compatible aliases while callers/tests migrate to the new terminology.
_get_child_timeout_seconds = _get_subagent_timeout_seconds
_build_child_system_prompt = _build_subagent_system_prompt
_build_child_progress_callback = _build_subagent_progress_callback
_build_child_agent = _build_subagent_agent
_run_single_child = _run_single_subagent
_resolve_child_credential_pool = _resolve_subagent_credential_pool


# ---------------------------------------------------------------------------
# OpenAI Function-Calling Schema
# ---------------------------------------------------------------------------

DELEGATE_TASK_SCHEMA = {
    "name": "delegate_task",
    "description": (
        "YOUR PRIMARY TOOL. Spawn subagents to handle non-trivial work in "
        "isolated contexts. Each subagent gets its own conversation, terminal "
        "session, and toolset. Only the final summary is returned -- "
        "intermediate tool results never enter your context window.\n\n"
        "DEFAULT BEHAVIOR: You SHOULD delegate any task requiring 3+ tool calls, "
        "producing artifacts (files, code, reports), or taking more than 2 minutes. "
        "Only handle work directly if it is trivial (single lookup, quick answer).\n\n"
        "TWO MODES (one of 'goal' or 'tasks' is required):\n"
        "1. Single task: provide 'goal' (+ optional context, toolsets)\n"
        "2. Batch (parallel): provide 'tasks' array with up to 3 items. "
        "All run concurrently and results are returned together.\n\n"
        "CRITICAL — CONTEXT IS EVERYTHING:\n"
        "- Subagents have NO memory of your conversation. The 'context' field is "
        "their ONLY source of background information.\n"
        "- Include: file paths, error messages, project structure, constraints, "
        "design rules, playbook content, and any relevant prior findings.\n"
        "- The more specific and complete the context, the better the result.\n"
        "- Check playbooks/ for matching playbooks and include them in context.\n\n"
        "BACKGROUND MODE (set background=true, ALWAYS USE THIS):\n"
        "- Subagents run in a daemon thread — you return IMMEDIATELY\n"
        "- Set 'deliver' to the topic name where results should land "
        "(e.g. 'SEO Research', 'Builds', or 'telegram:chat_id:thread_id')\n"
        "- Spawn and completion events are auto-posted to Agent Activity\n"
        "- Tell the user what was spawned and where results will appear, then move on\n\n"
        "CONSTRAINTS:\n"
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
                        "context": {"type": "string", "description": "Task-specific context"},
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
            "background": {
                "type": "boolean",
                "description": (
                    "Defaults to true. Subagents run in background and return immediately. "
                    "Set to false ONLY if you need the result before your next reply (rare). "
                    "Results are delivered to the 'deliver' target when complete. "
                    "Spawn/completion notifications go to Agent Activity automatically."
                ),
            },
            "deliver": {
                "type": "string",
                "description": (
                    "Where to deliver results when the subagent finishes. "
                    "Use a topic name (e.g. 'SEO Research', 'Builds', 'Playbook - Website Build') "
                    "or explicit format 'telegram:chat_id:thread_id'. "
                    "Topic names are resolved from config.yaml telegram.extra.group_topics. "
                    "Required when background=true."
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
        background=args.get("background", True),
        deliver=args.get("deliver"),
        parent_agent=kw.get("parent_agent")),
    check_fn=check_delegate_requirements,
    emoji="🔀",
)

DELEGATE_STATUS_SCHEMA = {
    "name": "delegate_status",
    "description": (
        "Check the status of spawned subagents. Use this to monitor background "
        "delegations, see if they are still running, stalled, or completed, and "
        "read their results.\n\n"
        "USAGE:\n"
        "- No arguments: list recent delegations (most recent first)\n"
        "- spawn_id: check a specific delegation by its spawn ID\n"
        "- active_only=true: only show running/spawned/stalled delegations\n\n"
        "WHEN TO USE:\n"
        "- After spawning a background subagent, check back on it\n"
        "- When Dax asks about subagent progress\n"
        "- Before reporting completion — verify the subagent actually finished\n"
        "- To detect stalled or failed subagents that need intervention"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "spawn_id": {
                "type": "string",
                "description": "Check a specific delegation by spawn ID.",
            },
            "limit": {
                "type": "integer",
                "description": "Max delegations to return (default 10, max 100).",
            },
            "active_only": {
                "type": "boolean",
                "description": "Only return active (running/spawned/stalled) delegations.",
            },
        },
        "required": [],
    },
}

registry.register(
    name="delegate_status",
    toolset="delegation",
    schema=DELEGATE_STATUS_SCHEMA,
    handler=lambda args, **kw: delegate_status(
        spawn_id=args.get("spawn_id"),
        limit=args.get("limit", 10),
        active_only=args.get("active_only", False),
    ),
    emoji="📊",
)


def delegate_kill(
    spawn_id: str,
    *,
    reason: str = "Killed by user request",
    parent_agent=None,
) -> str:
    """Kill an active background delegation by interrupting its subagent(s)."""
    spawn_id = (spawn_id or "").strip()
    if not spawn_id:
        return json.dumps({"error": "spawn_id is required"}, ensure_ascii=False)

    # 1. Find the delegation in the registry
    with _DELEGATION_REGISTRY_LOCK:
        payload = _load_delegation_registry()
    delegations = payload.get("delegations") or {}
    entry = delegations.get(spawn_id)
    if not isinstance(entry, dict):
        return json.dumps({"error": f"No delegation found for spawn_id '{spawn_id}'"}, ensure_ascii=False)

    current_status = _derive_spawn_status(entry)
    if current_status in {"completed", "completed_with_issues", "failed", "timed_out", "interrupted"}:
        return json.dumps({
            "status": "already_finished",
            "spawn_id": spawn_id,
            "final_status": current_status,
            "message": f"Delegation {spawn_id} already finished with status '{current_status}'. Nothing to kill.",
        }, ensure_ascii=False)

    # 2. Find and interrupt active child agents
    interrupted_count = 0
    if parent_agent and hasattr(parent_agent, '_active_children'):
        lock = getattr(parent_agent, '_active_children_lock', None)
        children = []
        if lock:
            with lock:
                children = list(parent_agent._active_children)
        else:
            children = list(parent_agent._active_children)

        for child in children:
            child_spawn_id = str(getattr(child, "_delegate_spawn_id", "") or "").strip()
            if child_spawn_id == spawn_id:
                try:
                    child.interrupt(f"KILLED: {reason}")
                    interrupted_count += 1
                    logger.info("Interrupted child agent for spawn %s: %s", spawn_id, reason)
                except Exception as exc:
                    logger.warning("Failed to interrupt child for spawn %s: %s", spawn_id, exc)

    # 3. Update registry to mark as killed
    now_ts = _registry_now_ts()

    def _mutator(reg: Dict[str, Any]) -> None:
        e = (reg.get("delegations") or {}).get(spawn_id)
        if not isinstance(e, dict):
            return
        e["status"] = "failed"
        e["finished_at_ts"] = now_ts
        e["updated_at_ts"] = now_ts
        e["runner_error"] = f"Killed: {reason}"
        for task in (e.get("tasks") or []):
            if task.get("status") in ("spawned", "running", "stalled"):
                task["status"] = "interrupted"
                task["error"] = f"Killed: {reason}"
                task["finished_at_ts"] = now_ts
                task["updated_at_ts"] = now_ts

    _mutate_delegation_registry(_mutator)

    return json.dumps({
        "status": "killed",
        "spawn_id": spawn_id,
        "interrupted_agents": interrupted_count,
        "reason": reason,
        "message": f"Delegation {spawn_id} killed. {interrupted_count} active agent(s) interrupted.",
    }, ensure_ascii=False)


DELEGATE_KILL_SCHEMA = {
    "name": "delegate_kill",
    "description": (
        "Kill an active background delegation. Interrupts the running subagent(s) "
        "and marks the delegation as failed in the registry.\n\n"
        "USAGE:\n"
        "- spawn_id (required): the spawn ID to kill\n"
        "- reason (optional): why it's being killed\n\n"
        "WHEN TO USE:\n"
        "- Dax asks to kill/stop/cancel a subagent\n"
        "- A delegation is stalled and needs to be terminated\n"
        "- You need to abort a run that was started with wrong parameters"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "spawn_id": {
                "type": "string",
                "description": "The spawn ID of the delegation to kill.",
            },
            "reason": {
                "type": "string",
                "description": "Why the delegation is being killed (optional, for logging).",
            },
        },
        "required": ["spawn_id"],
    },
}

registry.register(
    name="delegate_kill",
    toolset="delegation",
    schema=DELEGATE_KILL_SCHEMA,
    handler=lambda args, **kw: delegate_kill(
        spawn_id=args.get("spawn_id"),
        reason=args.get("reason", "Killed by user request"),
        parent_agent=kw.get("parent_agent"),
    ),
    emoji="🛑",
)
