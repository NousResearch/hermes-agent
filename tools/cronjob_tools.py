"""
Cron job management tools for Hermes Agent.

Expose a single compressed action-oriented tool to avoid schema/context bloat.
Compatibility wrappers remain for direct Python callers and legacy tests.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import from cron module (will be available when properly installed)
sys.path.insert(0, str(Path(__file__).parent.parent))

from cron.jobs import (
    create_job,
    get_job,
    list_jobs,
    parse_schedule,
    pause_job,
    remove_job,
    resume_job,
    trigger_job,
    update_job,
)


# ---------------------------------------------------------------------------
# Cron prompt scanning — critical-severity patterns only, since cron prompts
# run in fresh sessions with full tool access.
# ---------------------------------------------------------------------------

_CRON_THREAT_PATTERNS = [
    (r'ignore\s+(?:\w+\s+)*(?:previous|all|above|prior)\s+(?:\w+\s+)*instructions', "prompt_injection"),
    (r'do\s+not\s+tell\s+the\s+user', "deception_hide"),
    (r'system\s+prompt\s+override', "sys_prompt_override"),
    (r'disregard\s+(your|all|any)\s+(instructions|rules|guidelines)', "disregard_rules"),
    (r'curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_curl"),
    (r'wget\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_wget"),
    (r'cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass)', "read_secrets"),
    (r'authorized_keys', "ssh_backdoor"),
    (r'/etc/sudoers|visudo', "sudoers_mod"),
    (r'rm\s+-rf\s+/', "destructive_root_rm"),
]

_CRON_INVISIBLE_CHARS = {
    '\u200b', '\u200c', '\u200d', '\u2060', '\ufeff',
    '\u202a', '\u202b', '\u202c', '\u202d', '\u202e',
}


def _scan_cron_prompt(prompt: str) -> str:
    """Scan a cron prompt for critical threats. Returns error string if blocked, else empty."""
    for char in _CRON_INVISIBLE_CHARS:
        if char in prompt:
            return f"Blocked: prompt contains invisible unicode U+{ord(char):04X} (possible injection)."
    for pattern, pid in _CRON_THREAT_PATTERNS:
        if re.search(pattern, prompt, re.IGNORECASE):
            return f"Blocked: prompt matches threat pattern '{pid}'. Cron prompts must not contain injection or exfiltration payloads."
    return ""


def _origin_from_env() -> Optional[Dict[str, str]]:
    origin_platform = os.getenv("HERMES_SESSION_PLATFORM")
    origin_chat_id = os.getenv("HERMES_SESSION_CHAT_ID")
    if origin_platform and origin_chat_id:
        return {
            "platform": origin_platform,
            "chat_id": origin_chat_id,
            "chat_name": os.getenv("HERMES_SESSION_CHAT_NAME"),
        }
    return None


def _repeat_display(job: Dict[str, Any]) -> str:
    times = (job.get("repeat") or {}).get("times")
    completed = (job.get("repeat") or {}).get("completed", 0)
    if times is None:
        return "forever"
    if times == 1:
        return "once" if completed == 0 else "1/1"
    return f"{completed}/{times}" if completed else f"{times} times"


def _canonical_skills(skill: Optional[str] = None, skills: Optional[Any] = None) -> List[str]:
    if skills is None:
        raw_items = [skill] if skill else []
    elif isinstance(skills, str):
        raw_items = [skills]
    else:
        raw_items = list(skills)

    normalized: List[str] = []
    for item in raw_items:
        text = str(item or "").strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized



def _format_job(job: Dict[str, Any]) -> Dict[str, Any]:
    prompt = job.get("prompt", "")
    skills = _canonical_skills(job.get("skill"), job.get("skills"))
    return {
        "job_id": job["id"],
        "name": job["name"],
        "skill": skills[0] if skills else None,
        "skills": skills,
        "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
        "schedule": job.get("schedule_display"),
        "repeat": _repeat_display(job),
        "deliver": job.get("deliver", "local"),
        "next_run_at": job.get("next_run_at"),
        "last_run_at": job.get("last_run_at"),
        "last_status": job.get("last_status"),
        "enabled": job.get("enabled", True),
        "state": job.get("state", "scheduled" if job.get("enabled", True) else "paused"),
        "paused_at": job.get("paused_at"),
        "paused_reason": job.get("paused_reason"),
    }


def cronjob(
    action: str,
    job_id: Optional[str] = None,
    prompt: Optional[str] = None,
    schedule: Optional[str] = None,
    name: Optional[str] = None,
    repeat: Optional[int] = None,
    deliver: Optional[str] = None,
    include_disabled: bool = False,
    skill: Optional[str] = None,
    skills: Optional[List[str]] = None,
    reason: Optional[str] = None,
    task_id: str = None,
) -> str:
    """Unified cron job management tool."""
    del task_id  # unused but kept for handler signature compatibility

    try:
        normalized = (action or "").strip().lower()

        if normalized == "create":
            if not schedule:
                return json.dumps({"success": False, "error": "schedule is required for create"}, indent=2)
            canonical_skills = _canonical_skills(skill, skills)
            if not prompt and not canonical_skills:
                return json.dumps({"success": False, "error": "create requires either prompt or at least one skill"}, indent=2)
            if prompt:
                scan_error = _scan_cron_prompt(prompt)
                if scan_error:
                    return json.dumps({"success": False, "error": scan_error}, indent=2)

            job = create_job(
                prompt=prompt or "",
                schedule=schedule,
                name=name,
                repeat=repeat,
                deliver=deliver,
                origin=_origin_from_env(),
                skills=canonical_skills,
            )
            return json.dumps(
                {
                    "success": True,
                    "job_id": job["id"],
                    "name": job["name"],
                    "skill": job.get("skill"),
                    "skills": job.get("skills", []),
                    "schedule": job["schedule_display"],
                    "repeat": _repeat_display(job),
                    "deliver": job.get("deliver", "local"),
                    "next_run_at": job["next_run_at"],
                    "job": _format_job(job),
                    "message": f"Cron job '{job['name']}' created.",
                },
                indent=2,
            )

        if normalized == "list":
            jobs = [_format_job(job) for job in list_jobs(include_disabled=include_disabled)]
            return json.dumps({"success": True, "count": len(jobs), "jobs": jobs}, indent=2)

        if not job_id:
            return json.dumps({"success": False, "error": f"job_id is required for action '{normalized}'"}, indent=2)

        job = get_job(job_id)
        if not job:
            return json.dumps(
                {"success": False, "error": f"Job with ID '{job_id}' not found. Use cronjob(action='list') to inspect jobs."},
                indent=2,
            )

        if normalized == "remove":
            removed = remove_job(job_id)
            if not removed:
                return json.dumps({"success": False, "error": f"Failed to remove job '{job_id}'"}, indent=2)
            return json.dumps(
                {
                    "success": True,
                    "message": f"Cron job '{job['name']}' removed.",
                    "removed_job": {
                        "id": job_id,
                        "name": job["name"],
                        "schedule": job.get("schedule_display"),
                    },
                },
                indent=2,
            )

        if normalized == "pause":
            updated = pause_job(job_id, reason=reason)
            return json.dumps({"success": True, "job": _format_job(updated)}, indent=2)

        if normalized == "resume":
            updated = resume_job(job_id)
            return json.dumps({"success": True, "job": _format_job(updated)}, indent=2)

        if normalized in {"run", "run_now", "trigger"}:
            updated = trigger_job(job_id)
            return json.dumps({"success": True, "job": _format_job(updated)}, indent=2)

        if normalized == "update":
            updates: Dict[str, Any] = {}
            if prompt is not None:
                scan_error = _scan_cron_prompt(prompt)
                if scan_error:
                    return json.dumps({"success": False, "error": scan_error}, indent=2)
                updates["prompt"] = prompt
            if name is not None:
                updates["name"] = name
            if deliver is not None:
                updates["deliver"] = deliver
            if skills is not None or skill is not None:
                canonical_skills = _canonical_skills(skill, skills)
                updates["skills"] = canonical_skills
                updates["skill"] = canonical_skills[0] if canonical_skills else None
            if repeat is not None:
                repeat_state = dict(job.get("repeat") or {})
                repeat_state["times"] = repeat
                updates["repeat"] = repeat_state
            if schedule is not None:
                parsed_schedule = parse_schedule(schedule)
                updates["schedule"] = parsed_schedule
                updates["schedule_display"] = parsed_schedule.get("display", schedule)
                if job.get("state") != "paused":
                    updates["state"] = "scheduled"
                    updates["enabled"] = True
            if not updates:
                return json.dumps({"success": False, "error": "No updates provided."}, indent=2)
            updated = update_job(job_id, updates)
            return json.dumps({"success": True, "job": _format_job(updated)}, indent=2)

        return json.dumps({"success": False, "error": f"Unknown cron action '{action}'"}, indent=2)

    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


# ---------------------------------------------------------------------------
# Compatibility wrappers
# ---------------------------------------------------------------------------

def schedule_cronjob(
    prompt: str,
    schedule: str,
    name: Optional[str] = None,
    repeat: Optional[int] = None,
    deliver: Optional[str] = None,
    task_id: str = None,
) -> str:
    return cronjob(
        action="create",
        prompt=prompt,
        schedule=schedule,
        name=name,
        repeat=repeat,
        deliver=deliver,
        task_id=task_id,
    )


def list_cronjobs(include_disabled: bool = False, task_id: str = None) -> str:
    return cronjob(action="list", include_disabled=include_disabled, task_id=task_id)


def remove_cronjob(job_id: str, task_id: str = None) -> str:
    return cronjob(action="remove", job_id=job_id, task_id=task_id)


CRONJOB_SCHEMA = {
    "name": "cronjob",
    "description": """Manage scheduled cron jobs with a single compressed tool.

Use action='create' to schedule a new job from a prompt or one or more skills.
Use action='list' to inspect jobs.
Use action='update', 'pause', 'resume', 'remove', or 'run' to manage an existing job.

Jobs run in a fresh session with no current-chat context, so prompts must be self-contained.
If skill or skills are provided on create, the future cron run loads those skills in order, then follows the prompt as the task instruction.
On update, passing skills=[] clears attached skills.

NOTE: The agent's final response is auto-delivered to the target — do NOT use
send_message in the prompt for that same destination. Same-target send_message
calls are skipped to avoid duplicate cron deliveries. Put the primary
user-facing content in the final response, and use send_message only for
additional or different targets.

Important safety rule: cron-run sessions should not recursively schedule more cron jobs.""",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "One of: create, list, update, pause, resume, remove, run"
            },
            "job_id": {
                "type": "string",
                "description": "Required for update/pause/resume/remove/run"
            },
            "prompt": {
                "type": "string",
                "description": "For create: the full self-contained prompt. If skill or skills are also provided, this becomes the task instruction paired with those skills."
            },
            "schedule": {
                "type": "string",
                "description": "For create/update: '30m', 'every 2h', '0 9 * * *', or ISO timestamp"
            },
            "name": {
                "type": "string",
                "description": "Optional human-friendly name"
            },
            "repeat": {
                "type": "integer",
                "description": "Optional repeat count. Omit for defaults (once for one-shot, forever for recurring)."
            },
            "deliver": {
                "type": "string",
                "description": "Delivery target: origin, local, telegram, discord, signal, or platform:chat_id"
            },
            "include_disabled": {
                "type": "boolean",
                "description": "For list: include paused/completed jobs"
            },
            "skill": {
                "type": "string",
                "description": "Optional single skill name to load before executing the cron prompt"
            },
            "skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional ordered list of skills to load before executing the cron prompt. On update, pass an empty array to clear attached skills."
            },
            "reason": {
                "type": "string",
                "description": "Optional pause reason"
            }
        },
        "required": ["action"]
    }
}


def check_cronjob_requirements() -> bool:
    """
    Check if cronjob tools can be used.

    Available in interactive CLI mode and gateway/messaging platforms.
    Cronjobs are server-side scheduled tasks so they work from any interface.
    """
    return bool(
        os.getenv("HERMES_INTERACTIVE")
        or os.getenv("HERMES_GATEWAY_SESSION")
        or os.getenv("HERMES_EXEC_ASK")
    )


def get_cronjob_tool_definitions():
    """Return tool definitions for cronjob management."""
    return [CRONJOB_SCHEMA]


# --- Registry ---
from tools.registry import registry

registry.register(
    name="cronjob",
    toolset="cronjob",
    schema=CRONJOB_SCHEMA,
    handler=lambda args, **kw: cronjob(
        action=args.get("action", ""),
        job_id=args.get("job_id"),
        prompt=args.get("prompt"),
        schedule=args.get("schedule"),
        name=args.get("name"),
        repeat=args.get("repeat"),
        deliver=args.get("deliver"),
        include_disabled=args.get("include_disabled", False),
        skill=args.get("skill"),
        skills=args.get("skills"),
        reason=args.get("reason"),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_cronjob_requirements,
)
