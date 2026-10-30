"""`cronjob_manage` turn/step branch: grammar, storage, and the response contract.

Turn/step jobs are session-bound and live in ``state.db`` (``session_jobs``), never in
``cron/jobs.json``. The tool must say so explicitly or the model will promise the user a task
that dies with the chat.

Design: website/docs/developer-guide/scheduled-tasks.md § Agent-scheduled turn/step jobs.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Optional

from cron.jobs import normalize_repeat_value
from tools.cronjob_job_args import (
    _canonical_skills,
    _normalize_deliver_param,
    _validate_bot_chat_deliver,
)
from tools.cronjob_prompt_scan import _scan_cron_prompt
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_LIFETIME = "ends with this session; pauses when it ends"
_ACTIVATION = "this session (in-process; no gateway)"
_RUN_SESSION_RE = re.compile(r"^cron_tj_([0-9a-f]{8})_")
_MAX_EVERY = 1_000_000


def _dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2)


def parse_logical_schedule(value: Any) -> Optional[dict[str, Any]]:
    """Parse ``every N turns`` / ``each turn`` / ``in N steps``; ``None`` for anything else.

    Must run before duration parsing: ``every 3 turns`` would otherwise fall into
    ``parse_duration`` and fail.
    """
    text = " ".join(str(value or "").strip().lower().split())
    match = re.match(r"^every\s+(?:(\d+)\s+)?(turn|turns|step|steps)$", text)
    if match:
        every, unit = match.group(1), match.group(2)
        return {"kind": "turn" if unit.startswith("turn") else "step",
                "every": int(every or 1), "once": False}
    match = re.match(r"^each\s+(turn|step)$", text)
    if match:
        return {"kind": match.group(1), "every": 1, "once": False}
    match = re.match(r"^in\s+(\d+)\s+(turn|turns|step|steps)$", text)
    if match:
        return {"kind": "turn" if match.group(2).startswith("turn") else "step",
                "every": int(match.group(1)), "once": True}
    return None


def schedule_display(kind: str, every: int, once: bool) -> str:
    unit = "turn" if kind == "turn" else "step"
    prefix = "in" if once else "every"
    return f"{prefix} {every} {unit}{'' if every == 1 else 's'}"


def _open_db() -> Any:
    from hermes_state_registry import acquire

    return acquire()


def _close_db(db: Any) -> None:
    from hermes_state_registry import release_or_close

    release_or_close(db)


def _db_session(session_id: str) -> Optional[dict[str, Any]]:
    db = _open_db()
    try:
        return db.get_session(session_id)
    finally:
        _close_db(db)


def _depth_for(db: Any, session_id: str) -> int:
    match = _RUN_SESSION_RE.match(session_id or "")
    if not match:
        return 0
    parent = db.get_session_job("tj_" + match.group(1))
    return (int(parent.get("origin_depth") or 0) + 1) if parent else 0


def logical_job_view(job: dict[str, Any]) -> dict[str, Any]:
    kind = str(job.get("kind") or "step")
    return {
        "job_id": job["id"],
        "name": job.get("name"),
        "kind": kind,
        "schedule": schedule_display(kind, int(job.get("every") or 1), job.get("remaining") == 1),
        "every": int(job.get("every") or 1),
        "remaining": job.get("remaining"),
        f"next_due_{kind}": int(job.get("next_due_seq") or 0),
        "prompt_preview": (job.get("prompt") or "")[:120],
        "skills": job.get("skills") or [],
        "deliver": job.get("deliver") or "local",
        "enabled": bool(job.get("enabled")),
        "paused_reason": job.get("paused_reason"),
        "origin_session_id": job.get("origin_session_id"),
        "origin_depth": int(job.get("origin_depth") or 0),
        "last_fired_at": job.get("last_fired_at"),
        "last_status": job.get("last_status"),
        "lifetime": _LIFETIME,
    }


def list_logical_jobs(include_disabled: bool = True) -> list[dict[str, Any]]:
    db = _open_db()
    try:
        jobs = db.list_session_jobs()
    finally:
        _close_db(db)
    return [logical_job_view(job) for job in jobs if include_disabled or job.get("enabled")]


def _refresh_engine(*session_ids: Optional[str]) -> None:
    try:
        from scheduler.engine import get_engine

        engine = get_engine()
        for session_id in session_ids:
            if session_id:
                engine.refresh_session_jobs(session_id)
    except Exception:
        logger.debug("session job engine refresh failed", exc_info=True)


def create_logical_job(a: dict[str, Any], parsed: dict[str, Any]) -> str:
    kind = parsed["kind"]
    every = int(parsed["every"])
    if not 1 <= every <= _MAX_EVERY:
        return tool_error(f"every must be within 1..{_MAX_EVERY} {kind}s", success=False)
    if a.get("no_agent") or a.get("script"):
        return tool_error("turn/step schedules run the agent; no_agent/script are not supported", success=False)
    session_id = a.get("session_id") or ""
    if not session_id:
        return tool_error(
            "turn/step schedules are session-bound and require an active session_id; "
            "use a wall-clock schedule (e.g. 'every 30m') for work that must outlive the chat.",
            success=False)
    prompt = a.get("prompt") or ""
    skills = _canonical_skills(a.get("skill"), a.get("skills"))
    if not prompt and not skills:
        return tool_error("create requires either prompt or at least one skill", success=False)
    error = (prompt and _scan_cron_prompt(prompt)) or _validate_bot_chat_deliver(
        _normalize_deliver_param(a.get("deliver")))
    if error:
        return tool_error(error, success=False)
    deliver = _normalize_deliver_param(a.get("deliver"))
    remaining = normalize_repeat_value(a.get("repeat"))
    if parsed["once"]:
        remaining = 1
    db = _open_db()
    try:
        row = db.get_session(session_id)
        if not row:
            return tool_error("origin session not found; cannot bind a turn/step schedule", success=False)
        depth = _depth_for(db, session_id)
        current = int(row.get("step_seq" if kind == "step" else "turn_seq") or 0)
        job_id = db.create_session_job(
            origin_session_id=session_id, kind=kind, every=every, prompt=prompt,
            next_due_seq=current + every, name=a.get("name"), skills=skills or None,
            deliver=deliver, failure_deliver=_normalize_deliver_param(a.get("failure_deliver")),
            remaining=remaining, origin_depth=depth,
        )
        job = db.get_session_job(job_id) or {}
    except ValueError as exc:
        return tool_error(str(exc), success=False)
    finally:
        _close_db(db)
    _refresh_engine(session_id)
    view = logical_job_view(job)
    return _dumps({
        "success": True,
        "job_id": job_id,
        "name": job.get("name"),
        "schedule": view["schedule"],
        f"next_due_{kind}": view[f"next_due_{kind}"],
        "repeat": "once" if remaining == 1 else (f"{remaining} times" if remaining else "forever"),
        "deliver": view["deliver"],
        "activation": _ACTIVATION,
        "lifetime": _LIFETIME,
        "origin_session_id": session_id,
        "job": view,
        "message": (
            f"Session-bound {kind} job '{job.get('name') or job_id}' created: {view['schedule']}; "
            "it runs in a fresh session, not this one, and pauses when this session ends."
        ),
    })


def _logical_job_or_error(job_id: str):
    db = _open_db()
    try:
        job = db.get_session_job(job_id)
    finally:
        _close_db(db)
    if job is None:
        return None, tool_error(f"Session job '{job_id}' not found.", success=False)
    return job, None


def handle_logical_job_action(action: str, job_id: str, a: dict[str, Any]) -> str:
    job, error = _logical_job_or_error(job_id)
    if error is not None:
        return error
    origin = job.get("origin_session_id")
    db = _open_db()
    try:
        if action == "remove":
            db.remove_session_job(job_id)
            _refresh_engine(origin)
            return _dumps({"success": True, "message": f"Session job '{job.get('name') or job_id}' removed.",
                           "removed_job": logical_job_view(job)})
        if action in ("pause", "resume"):
            enabled = action == "resume"
            db.update_session_job(job_id, enabled=1 if enabled else 0,
                                  paused_reason=None if enabled else (a.get("reason") or "paused"))
            updated = db.get_session_job(job_id) or job
            return _dumps({"success": True, "job": logical_job_view(updated), "lifetime": _LIFETIME})
        if action in ("run", "run_now", "trigger"):
            from scheduler.session_jobs import run_job

            snapshot = dict(job)
            run_job(snapshot, int(job.get("next_due_seq") or 0))
            return _dumps({
                "success": True,
                "job": logical_job_view(snapshot),
                "message": "Dispatched now in a fresh session; the next scheduled fire is unchanged.",
                "lifetime": _LIFETIME,
            })
        if action == "update":
            return _update_logical_job(db, job, a)
        return tool_error(f"Unknown session-job action '{action}'", success=False)
    finally:
        _close_db(db)


def _update_logical_job(db: Any, job: dict[str, Any], a: dict[str, Any]) -> str:
    job_id = job["id"]
    updates: dict[str, Any] = {}
    if a.get("prompt"):
        error = _scan_cron_prompt(a["prompt"])
        if error:
            return tool_error(error, success=False)
        updates["prompt"] = a["prompt"]
    if a.get("name"):
        updates["name"] = a["name"]
    skills = _canonical_skills(a.get("skill"), a.get("skills"))
    if skills:
        updates["skills"] = skills
    if a.get("deliver") is not None:
        deliver = _normalize_deliver_param(a.get("deliver"))
        error = _validate_bot_chat_deliver(deliver)
        if error:
            return tool_error(error, success=False)
        updates["deliver"] = deliver
    parsed = parse_logical_schedule(a.get("schedule")) if a.get("schedule") else None
    if a.get("schedule") and parsed is None:
        return tool_error(
            "a session job only accepts 'every N turns/steps' or 'in N turns/steps'", success=False)
    if parsed is not None:
        if parsed["once"]:
            updates["remaining"] = 1
        every = int(parsed["every"])
        row = db.get_session(job.get("origin_session_id") or "") or {}
        current = int(row.get("step_seq" if job["kind"] == "step" else "turn_seq") or 0)
        updates["every"] = every
        updates["next_due_seq"] = current + every
    if not updates:
        return tool_error("nothing to update", success=False)
    db.update_session_job(job_id, **updates)
    updated = db.get_session_job(job_id) or job
    _refresh_engine(job.get("origin_session_id"))
    return _dumps({"success": True, "job": logical_job_view(updated), "lifetime": _LIFETIME})
