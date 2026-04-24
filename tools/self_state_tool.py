#!/usr/bin/env python3
"""Self-state introspection tool.

Gives the agent a compact, factual view of its own runtime state across
isolated gateway sessions. This is intentionally read-only: it helps the
model diagnose "what am I doing?" without mutating crons, memory, or sessions.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from hermes_constants import get_hermes_home


def _clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _load_json(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _format_ts(ts: Any) -> Optional[str]:
    if ts in (None, ""):
        return None
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(ts)))
    except (TypeError, ValueError, OSError):
        return str(ts)


def _preview(text: Any, limit: int = 240) -> str:
    if text is None:
        return ""
    s = str(text).replace("\r", " ").replace("\n", " ").strip()
    if len(s) <= limit:
        return s
    return s[: limit - 3].rstrip() + "..."


def _session_entries(limit: int) -> List[Dict[str, Any]]:
    sessions_path = get_hermes_home() / "sessions" / "sessions.json"
    data = _load_json(sessions_path, {})
    if not isinstance(data, dict):
        return []

    entries: List[Dict[str, Any]] = []
    for key, raw in data.items():
        if not isinstance(raw, dict):
            continue
        origin = raw.get("origin") if isinstance(raw.get("origin"), dict) else {}
        entries.append({
            "session_key": key,
            "session_id": raw.get("session_id", ""),
            "platform": raw.get("platform") or origin.get("platform", ""),
            "chat_type": raw.get("chat_type") or origin.get("chat_type", ""),
            "chat_id": origin.get("chat_id", ""),
            "chat_name": origin.get("chat_name") or raw.get("display_name", ""),
            "user_id": origin.get("user_id", ""),
            "user_name": origin.get("user_name", ""),
            "thread_id": origin.get("thread_id"),
            "updated_at": raw.get("updated_at", ""),
            "created_at": raw.get("created_at", ""),
        })

    entries.sort(key=lambda e: str(e.get("updated_at") or ""), reverse=True)
    return entries[:limit]


def _session_index_by_id() -> Dict[str, Dict[str, Any]]:
    entries = _session_entries(1000)
    return {
        str(entry.get("session_id")): entry
        for entry in entries
        if entry.get("session_id")
    }


def _recent_activity_from_db(limit: int, session_filter: str = "") -> List[Dict[str, Any]]:
    try:
        from hermes_state import SessionDB
    except Exception:
        return []

    db = None
    try:
        db = SessionDB()
        session_index = _session_index_by_id()
        filter_text = session_filter.lower()
        sessions = db.list_sessions_rich(limit=max(limit * 3, limit), include_children=True)
        rows: List[Dict[str, Any]] = []
        for session in sessions:
            session_id = session.get("id") or session.get("session_id")
            if not session_id:
                continue
            index_entry = session_index.get(str(session_id), {})
            session_haystack = " ".join(
                str(value or "")
                for value in (
                    session_id,
                    session.get("source", ""),
                    index_entry.get("session_key", ""),
                    index_entry.get("platform", ""),
                    index_entry.get("chat_id", ""),
                    index_entry.get("chat_name", ""),
                    index_entry.get("user_id", ""),
                    index_entry.get("user_name", ""),
                )
            ).lower()
            if filter_text and filter_text not in session_haystack:
                continue
            try:
                messages = db.get_messages(session_id)
            except Exception:
                continue
            for msg in messages[-8:]:
                role = msg.get("role")
                tool_calls = msg.get("tool_calls") or []
                tool_names = _tool_names(tool_calls)
                tool_details = _tool_call_details(tool_calls)
                content = msg.get("content")
                if role not in ("user", "assistant", "tool") and not tool_names:
                    continue
                row = {
                    "timestamp": _format_ts(msg.get("timestamp")),
                    "session_id": session_id,
                    "session_key": index_entry.get("session_key", ""),
                    "source": index_entry.get("chat_id") or session.get("source", ""),
                    "platform": index_entry.get("platform") or session.get("source", ""),
                    "user_name": index_entry.get("user_name", ""),
                    "role": role,
                    "tool_name": msg.get("tool_name"),
                    "tool_calls": tool_names,
                    "content_preview": _preview(content),
                }
                if tool_details:
                    row["tool_call_details"] = tool_details
                rows.append(row)
        rows.sort(key=lambda r: str(r.get("timestamp") or ""), reverse=True)
        return rows[:limit]
    except Exception:
        return []
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


def _tool_names(tool_calls: Any) -> List[str]:
    if not isinstance(tool_calls, list):
        return []
    names: List[str] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        name = call.get("name")
        if not name and isinstance(call.get("function"), dict):
            name = call["function"].get("name")
        if name:
            names.append(str(name))
    return names


def _tool_call_details(tool_calls: Any) -> List[Dict[str, Any]]:
    if not isinstance(tool_calls, list):
        return []
    details: List[Dict[str, Any]] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = call.get("name") or function.get("name")
        if not name:
            continue
        raw_args = call.get("arguments")
        if raw_args is None:
            raw_args = function.get("arguments")
        args_summary: Dict[str, Any] = {}
        if isinstance(raw_args, str) and raw_args.strip():
            try:
                parsed = json.loads(raw_args)
            except (TypeError, ValueError):
                args_summary["arguments_preview"] = _preview(raw_args)
            else:
                if isinstance(parsed, dict):
                    for key in ("action", "target", "message", "query", "job_id"):
                        if key in parsed:
                            args_summary[key] = _preview(parsed.get(key), 180)
                else:
                    args_summary["arguments_preview"] = _preview(parsed)
        elif isinstance(raw_args, dict):
            for key in ("action", "target", "message", "query", "job_id"):
                if key in raw_args:
                    args_summary[key] = _preview(raw_args.get(key), 180)
        details.append({"name": str(name), **args_summary})
    return details


def _jobs_iter(data: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(data, dict):
        raw_jobs = data.get("jobs")
        if isinstance(raw_jobs, list):
            for item in raw_jobs:
                if isinstance(item, dict):
                    yield item
        else:
            for item in data.values():
                if isinstance(item, dict):
                    yield item


def _cron_jobs(limit: int) -> List[Dict[str, Any]]:
    jobs_path = get_hermes_home() / "cron" / "jobs.json"
    data = _load_json(jobs_path, [])
    jobs: List[Dict[str, Any]] = []
    for job in _jobs_iter(data):
        origin = job.get("origin") if isinstance(job.get("origin"), dict) else {}
        schedule = job.get("schedule") if isinstance(job.get("schedule"), dict) else {}
        jobs.append({
            "id": job.get("id", ""),
            "name": job.get("name", ""),
            "enabled": bool(job.get("enabled", True)),
            "deliver": job.get("deliver", "local"),
            "origin": {
                "platform": origin.get("platform"),
                "chat_id": origin.get("chat_id"),
                "chat_name": origin.get("chat_name"),
                "thread_id": origin.get("thread_id"),
            } if origin else None,
            "schedule": job.get("schedule_display") or schedule.get("display") or job.get("schedule"),
            "next_run": job.get("next_run"),
            "last_run": job.get("last_run"),
        })
    jobs.sort(key=lambda j: (not j.get("enabled", False), str(j.get("next_run") or "")))
    return jobs[:limit]


def _outbound_events(limit: int) -> List[Dict[str, Any]]:
    rows = _recent_activity_from_db(limit=limit * 4)
    events: List[Dict[str, Any]] = []
    for row in rows:
        tool_calls = row.get("tool_calls") or []
        if "send_message" not in tool_calls:
            continue
        events.append({
            "timestamp": row.get("timestamp"),
            "session_id": row.get("session_id"),
            "source": row.get("source"),
            "kind": "send_message_tool_call",
            "content_preview": row.get("content_preview", ""),
            "tool_calls": tool_calls,
            "tool_call_details": row.get("tool_call_details", []),
        })
        if len(events) >= limit:
            break
    if len(events) < limit:
        events.extend(_mirror_events_from_jsonl(limit - len(events)))
    events.sort(key=lambda r: str(r.get("timestamp") or ""), reverse=True)
    return events[:limit]


def _active_directives(limit: int) -> List[Dict[str, Any]]:
    try:
        from hermes_state import SessionDB
    except Exception:
        return []
    db = None
    try:
        db = SessionDB()
        rows = db.list_active_agent_directives(actor_id="main", limit=limit)
        result = []
        for row in rows:
            payload = row.get("payload") or {}
            result.append({
                "directive_id": row.get("directive_id"),
                "directive_type": row.get("directive_type"),
                "directive_key": row.get("directive_key"),
                "priority": row.get("priority"),
                "created_at": _format_ts(row.get("created_at")),
                "issuer": row.get("issuer_person_id") or row.get("issuer_user_id"),
                "text": payload.get("text", ""),
            })
        return result
    except Exception:
        return []
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


def _agent_events(limit: int) -> List[Dict[str, Any]]:
    try:
        from hermes_state import SessionDB
    except Exception:
        return []
    db = None
    try:
        db = SessionDB()
        rows = db.list_recent_agent_events(limit=limit)
        return [
            {
                "event_id": row.get("event_id"),
                "event_type": row.get("event_type"),
                "event_subtype": row.get("event_subtype"),
                "status": row.get("status"),
                "source": row.get("source"),
                "person_id": row.get("person_id"),
                "platform": row.get("platform"),
                "chat_id": row.get("platform_chat_id"),
                "session_key": row.get("session_key"),
                "created_at": _format_ts(row.get("created_at")),
                "content_preview": _preview(row.get("content")),
            }
            for row in rows
        ]
    except Exception:
        return []
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


def _mirror_events_from_jsonl(limit: int) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []
    sessions_path = get_hermes_home() / "sessions"
    entries = _session_entries(50)
    events: List[Dict[str, Any]] = []
    for entry in entries:
        session_id = entry.get("session_id")
        if not session_id:
            continue
        transcript = sessions_path / f"{session_id}.jsonl"
        if not transcript.exists():
            continue
        try:
            lines = transcript.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for line in reversed(lines[-100:]):
            try:
                msg = json.loads(line)
            except Exception:
                continue
            if not isinstance(msg, dict) or not msg.get("mirror"):
                continue
            events.append({
                "timestamp": msg.get("timestamp"),
                "session_id": session_id,
                "source": entry.get("platform", ""),
                "kind": "delivery_mirror",
                "mirror_source": msg.get("mirror_source", ""),
                "content_preview": _preview(msg.get("content")),
            })
            if len(events) >= limit:
                return events
    return events


def self_state_tool(action: str = "summary", limit: int = 10, session_filter: str = "", **_) -> Dict[str, Any]:
    """Return a read-only snapshot of the agent's runtime state."""
    action = (action or "summary").strip().lower()
    limit = _clamp_int(limit, default=10, minimum=1, maximum=50)
    session_filter = str(session_filter or "").strip()

    if action == "sessions":
        return {
            "action": "sessions",
            "sessions": _session_entries(limit),
            "note": "These are separate conversation contexts for the same agent runtime.",
        }
    if action == "recent_activity":
        return {
            "action": "recent_activity",
            "activity": _recent_activity_from_db(limit, session_filter=session_filter),
            "note": "Use this to inspect what the agent recently saw or did across sessions.",
        }
    if action == "crons":
        return {
            "action": "crons",
            "crons": _cron_jobs(limit),
            "note": "This only lists cron jobs on this VM. Jobs delegated to other agents live on their VMs.",
        }
    if action == "outbounds":
        return {
            "action": "outbounds",
            "outbounds": _outbound_events(limit),
            "note": "Best-effort view based on persisted assistant/tool-call history.",
        }
    if action == "directives":
        return {
            "action": "directives",
            "directives": _active_directives(limit),
            "note": "Active agent-level directives are shared across isolated sessions.",
        }
    if action == "events":
        return {
            "action": "events",
            "events": _agent_events(limit),
            "note": "Recent agent-level runtime events across sessions.",
        }
    if action != "summary":
        return {
            "error": f"Unknown action {action!r}. Use summary, sessions, recent_activity, crons, outbounds, directives, or events.",
        }

    return {
        "action": "summary",
        "sessions": _session_entries(min(limit, 10)),
        "recent_activity": _recent_activity_from_db(min(limit, 12), session_filter=session_filter),
        "crons": _cron_jobs(min(limit, 12)),
        "outbounds": _outbound_events(min(limit, 8)),
        "directives": _active_directives(min(limit, 8)),
        "agent_events": _agent_events(min(limit, 8)),
        "notes": [
            "Sessions are isolated for prompt context, but this tool provides a shared runtime view.",
            "Remote agents' local crons are not visible here unless they report through your sessions.",
        ],
    }


SELF_STATE_SCHEMA = {
    "name": "self_state",
    "description": (
        "Inspect your own runtime across isolated sessions. Use this when the user asks "
        "what you are doing, why you posted or messaged something, what sessions/channels "
        "are active, what local crons exist, or whether recent behavior came from another "
        "conversation context. Prefer this before terminal, session_search, logs, or cron "
        "tools for runtime self-awareness. Read-only."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["summary", "sessions", "recent_activity", "crons", "outbounds", "directives", "events"],
                "description": "Which self-state view to return. Use summary first unless you need a narrower view.",
                "default": "summary",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum entries per section, 1-50.",
                "default": 10,
            },
            "session_filter": {
                "type": "string",
                "description": "Optional substring filter for session id/source/chat/user in recent_activity.",
            },
        },
        "required": [],
    },
}


def _check_self_state_requirements() -> bool:
    return get_hermes_home().exists()


from tools.registry import registry  # noqa: E402

registry.register(
    name="self_state",
    toolset="self_state",
    schema=SELF_STATE_SCHEMA,
    handler=lambda args, **kw: json.dumps(
        self_state_tool(
            action=args.get("action", "summary"),
            limit=args.get("limit", 10),
            session_filter=args.get("session_filter", ""),
        ),
        ensure_ascii=False,
    ),
    check_fn=_check_self_state_requirements,
    emoji="",
)
