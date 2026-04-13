"""
Cron job management tools for Hermes Agent.

Expose a single compressed action-oriented tool to avoid schema/context bloat.
Compatibility wrappers remain for direct Python callers and legacy tests.
"""

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

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
    from gateway.session_context import get_session_env
    origin_platform = get_session_env("HERMES_SESSION_PLATFORM")
    origin_chat_id = get_session_env("HERMES_SESSION_CHAT_ID")
    if origin_platform and origin_chat_id:
        thread_id = get_session_env("HERMES_SESSION_THREAD_ID") or None
        if thread_id:
            logger.debug(
                "Cron origin captured thread_id=%s for %s:%s",
                thread_id, origin_platform, origin_chat_id,
            )
        return {
            "platform": origin_platform,
            "chat_id": origin_chat_id,
            "chat_name": get_session_env("HERMES_SESSION_CHAT_NAME") or None,
            "thread_id": thread_id,
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




def _resolve_model_override(model_obj: Optional[Dict[str, Any]]) -> tuple:
    """Resolve a model override object into (provider, model) for job storage.

    If provider is omitted, pins the current main provider from config so the
    job doesn't drift when the user later changes their default via hermes model.

    Returns (provider_str_or_none, model_str_or_none).
    """
    if not model_obj or not isinstance(model_obj, dict):
        return (None, None)
    model_name = (model_obj.get("model") or "").strip() or None
    provider_name = (model_obj.get("provider") or "").strip() or None
    if model_name and not provider_name:
        # Pin to the current main provider so the job is stable
        try:
            from hermes_cli.config import load_config
            cfg = load_config()
            model_cfg = cfg.get("model", {})
            if isinstance(model_cfg, dict):
                provider_name = model_cfg.get("provider") or None
        except Exception:
            pass  # Best-effort; provider stays None
    return (provider_name, model_name)


def _normalize_optional_job_value(value: Optional[Any], *, strip_trailing_slash: bool = False) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if strip_trailing_slash:
        text = text.rstrip("/")
    return text or None


def _validate_cron_script_path(script: Optional[str]) -> Optional[str]:
    """Validate a cron job script path at the API boundary.

    Scripts must be relative paths that resolve within HERMES_HOME/scripts/.
    Absolute paths and ~ expansion are rejected to prevent arbitrary script
    execution via prompt injection.

    Returns an error string if blocked, else None (valid).
    """
    if not script or not script.strip():
        return None  # empty/None = clearing the field, always OK

    from hermes_constants import get_hermes_home

    raw = script.strip()

    # Reject absolute paths and ~ expansion at the API boundary.
    # Only relative paths within ~/.hermes/scripts/ are allowed.
    if raw.startswith(("/", "~")) or (len(raw) >= 2 and raw[1] == ":"):
        return (
            f"Script path must be relative to ~/.hermes/scripts/. "
            f"Got absolute or home-relative path: {raw!r}. "
            f"Place scripts in ~/.hermes/scripts/ and use just the filename."
        )

    # Validate containment after resolution
    from tools.path_security import validate_within_dir

    scripts_dir = get_hermes_home() / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    containment_error = validate_within_dir(scripts_dir / raw, scripts_dir)
    if containment_error:
        return (
            f"Script path escapes the scripts directory via traversal: {raw!r}"
        )

    return None


def _format_job(job: Dict[str, Any]) -> Dict[str, Any]:
    prompt = job.get("prompt", "")
    skills = _canonical_skills(job.get("skill"), job.get("skills"))
    result = {
        "job_id": job["id"],
        "name": job["name"],
        "skill": skills[0] if skills else None,
        "skills": skills,
        "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
        "model": job.get("model"),
        "provider": job.get("provider"),
        "base_url": job.get("base_url"),
        "schedule": job.get("schedule_display"),
        "repeat": _repeat_display(job),
        "deliver": job.get("deliver", "local"),
        "next_run_at": job.get("next_run_at"),
        "last_run_at": job.get("last_run_at"),
        "last_status": job.get("last_status"),
        "last_delivery_error": job.get("last_delivery_error"),
        "enabled": job.get("enabled", True),
        "state": job.get("state", "scheduled" if job.get("enabled", True) else "paused"),
        "paused_at": job.get("paused_at"),
        "paused_reason": job.get("paused_reason"),
    }
    if job.get("script"):
        result["script"] = job["script"]
    return result


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
    model: Optional[str] = None,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    reason: Optional[str] = None,
    script: Optional[str] = None,
    task_id: str = None,
) -> str:
    """Unified cron job management tool."""
    del task_id  # unused but kept for handler signature compatibility

    try:
        normalized = (action or "").strip().lower()

        if normalized == "create":
            if not schedule:
                return tool_error("schedule is required for create", success=False)
            canonical_skills = _canonical_skills(skill, skills)
            if not prompt and not canonical_skills:
                return tool_error("create requires either prompt or at least one skill", success=False)
            if prompt:
                scan_error = _scan_cron_prompt(prompt)
                if scan_error:
                    return tool_error(scan_error, success=False)

            # Validate script path before storing
            if script:
                script_error = _validate_cron_script_path(script)
                if script_error:
                    return tool_error(script_error, success=False)

            job = create_job(
                prompt=prompt or "",
                schedule=schedule,
                name=name,
                repeat=repeat,
                deliver=deliver,
                origin=_origin_from_env(),
                skills=canonical_skills,
                model=_normalize_optional_job_value(model),
                provider=_normalize_optional_job_value(provider),
                base_url=_normalize_optional_job_value(base_url, strip_trailing_slash=True),
                script=_normalize_optional_job_value(script),
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
            return tool_error(f"job_id is required for action '{normalized}'", success=False)

        job = get_job(job_id)
        if not job:
            return json.dumps(
                {"success": False, "error": f"Job with ID '{job_id}' not found. Use cronjob(action='list') to inspect jobs."},
                indent=2,
            )

        if normalized == "remove":
            removed = remove_job(job_id)
            if not removed:
                return tool_error(f"Failed to remove job '{job_id}'", success=False)
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
                    return tool_error(scan_error, success=False)
                updates["prompt"] = prompt
            if name is not None:
                updates["name"] = name
            if deliver is not None:
                updates["deliver"] = deliver
            if skills is not None or skill is not None:
                canonical_skills = _canonical_skills(skill, skills)
                updates["skills"] = canonical_skills
                updates["skill"] = canonical_skills[0] if canonical_skills else None
            if model is not None:
                updates["model"] = _normalize_optional_job_value(model)
            if provider is not None:
                updates["provider"] = _normalize_optional_job_value(provider)
            if base_url is not None:
                updates["base_url"] = _normalize_optional_job_value(base_url, strip_trailing_slash=True)
            if script is not None:
                # Pass empty string to clear an existing script
                if script:
                    script_error = _validate_cron_script_path(script)
                    if script_error:
                        return tool_error(script_error, success=False)
                updates["script"] = _normalize_optional_job_value(script) if script else None
            if repeat is not None:
                # Normalize: treat 0 or negative as None (infinite)
                normalized_repeat = None if repeat <= 0 else repeat
                repeat_state = dict(job.get("repeat") or {})
                repeat_state["times"] = normalized_repeat
                updates["repeat"] = repeat_state
            if schedule is not None:
                parsed_schedule = parse_schedule(schedule)
                updates["schedule"] = parsed_schedule
                updates["schedule_display"] = parsed_schedule.get("display", schedule)
                if job.get("state") != "paused":
                    updates["state"] = "scheduled"
                    updates["enabled"] = True
            if not updates:
                return tool_error("No updates provided.", success=False)
            updated = update_job(job_id, updates)
            return json.dumps({"success": True, "job": _format_job(updated)}, indent=2)

        return tool_error(f"Unknown cron action '{action}'", success=False)

    except Exception as e:
        return tool_error(str(e), success=False)


# ---------------------------------------------------------------------------
# Compatibility wrappers
# ---------------------------------------------------------------------------

def schedule_cronjob(
    prompt: str,
    schedule: str,
    name: Optional[str] = None,
    repeat: Optional[int] = None,
    deliver: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    task_id: str = None,
) -> str:
    return cronjob(
        action="create",
        prompt=prompt,
        schedule=schedule,
        name=name,
        repeat=repeat,
        deliver=deliver,
        model=model,
        provider=provider,
        base_url=base_url,
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
If skills are provided on create, the future cron run loads those skills in order, then follows the prompt as the task instruction.
On update, passing skills=[] clears attached skills.

NOTE: The agent's final response is auto-delivered to the target. Put the primary
user-facing content in the final response. Cron jobs run autonomously with no user
present — they cannot ask questions or request clarification.

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
                "description": "For create: the full self-contained prompt. If skills are also provided, this becomes the task instruction paired with those skills."
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
                "description": "Omit this parameter to auto-deliver back to the current chat and topic (recommended). Auto-detection preserves thread/topic context. Only set explicitly when the user asks to deliver somewhere OTHER than the current conversation. Values: 'origin' (same as omitting), 'local' (no delivery, save only), or platform:chat_id:thread_id for a specific destination. Examples: 'telegram:-1001234567890:17585', 'discord:#engineering'. WARNING: 'platform:chat_id' without :thread_id loses topic targeting."
            },
            "skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional ordered list of skill names to load before executing the cron prompt. On update, pass an empty array to clear attached skills."
            },
            "model": {
                "type": "object",
                "description": "Optional per-job model override. If provider is omitted, the current main provider is pinned at creation time so the job stays stable.",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "Provider name (e.g. 'openrouter', 'anthropic'). Omit to use and pin the current provider."
                    },
                    "model": {
                        "type": "string",
                        "description": "Model name (e.g. 'anthropic/claude-sonnet-4', 'claude-sonnet-4')"
                    }
                },
                "required": ["model"]
            },
            "script": {
                "type": "string",
                "description": "Optional path to a Python script that runs before each cron job execution. Its stdout is injected into the prompt as context. Use for data collection and change detection. Relative paths resolve under ~/.hermes/scripts/. On update, pass empty string to clear."
            },
        },
        "required": ["action"]
    }
}


def check_cronjob_requirements() -> bool:
    """
    Check if cronjob tools can be used.

    Available in interactive CLI mode and gateway/messaging platforms.
    The cron system is internal (JSON file-based scheduler ticked by the gateway),
    so no external crontab executable is required.
    """
    return bool(
        os.getenv("HERMES_INTERACTIVE")
        or os.getenv("HERMES_GATEWAY_SESSION")
        or os.getenv("HERMES_EXEC_ASK")
    )


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="cronjob",
    toolset="cronjob",
    schema=CRONJOB_SCHEMA,
    handler=lambda args, **kw: (lambda _mo=_resolve_model_override(args.get("model")): cronjob(
        action=args.get("action", ""),
        job_id=args.get("job_id"),
        prompt=args.get("prompt"),
        schedule=args.get("schedule"),
        name=args.get("name"),
        repeat=args.get("repeat"),
        deliver=args.get("deliver"),
        include_disabled=args.get("include_disabled", True),
        skill=args.get("skill"),
        skills=args.get("skills"),
        model=_mo[1],
        provider=_mo[0] or args.get("provider"),
        base_url=args.get("base_url"),
        reason=args.get("reason"),
        script=args.get("script"),
        task_id=kw.get("task_id"),
    ))(),
    check_fn=check_cronjob_requirements,
    emoji="⏰",
)
