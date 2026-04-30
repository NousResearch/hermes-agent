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
from typing import Any, Dict, List, Optional, Union

from hermes_constants import display_hermes_home

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


def _normalize_deliver_param(value: Any) -> Optional[str]:
    """Normalize a user-supplied ``deliver`` value to the canonical string form.

    The cron schema documents ``deliver`` as a string (``"local"``, ``"origin"``,
    ``"telegram"``, ``"telegram:chat_id[:thread_id]"``, or comma-separated combos).
    Some callers — MCP clients passing arrays, scripts building the payload as a
    list — supply ``["telegram"]``.  ``create_job``/``update_job`` store it as-is,
    and the scheduler's ``str(deliver).split(",")`` then serializes the list to
    the literal ``"['telegram']"`` which is not a known platform.  Flatten lists
    / tuples at the API boundary so storage is always a string.  Returns ``None``
    for ``None``/empty so callers can treat it as "not supplied".
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        parts = [str(p).strip() for p in value if str(p).strip()]
        return ",".join(parts) if parts else None
    text = str(value).strip()
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
    if job.get("enabled_toolsets"):
        result["enabled_toolsets"] = job["enabled_toolsets"]
    if job.get("workdir"):
        result["workdir"] = job["workdir"]
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
    context_from: Optional[Union[str, List[str]]] = None,
    enabled_toolsets: Optional[List[str]] = None,
    workdir: Optional[str] = None,
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

            # Validate context_from references existing jobs
            if context_from:
                from cron.jobs import get_job as _get_job
                refs = [context_from] if isinstance(context_from, str) else context_from
                for ref_id in refs:
                    if not _get_job(ref_id):
                        return tool_error(
                            f"context_from job '{ref_id}' not found. "
                            "Use cronjob(action='list') to see available jobs.",
                            success=False,
                        )

            job = create_job(
                prompt=prompt or "",
                schedule=schedule,
                name=name,
                repeat=repeat,
                deliver=_normalize_deliver_param(deliver),
                origin=_origin_from_env(),
                skills=canonical_skills,
                model=_normalize_optional_job_value(model),
                provider=_normalize_optional_job_value(provider),
                base_url=_normalize_optional_job_value(base_url, strip_trailing_slash=True),
                script=_normalize_optional_job_value(script),
                context_from=context_from,
                enabled_toolsets=enabled_toolsets or None,
                workdir=_normalize_optional_job_value(workdir),
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
                updates["deliver"] = _normalize_deliver_param(deliver)
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
            if context_from is not None:
                # Empty string / empty list clears the field; otherwise validate
                # each referenced job exists before storing. Normalized to a list
                # (or None) to match the shape stored by create_job().
                if isinstance(context_from, str):
                    refs = [context_from.strip()] if context_from.strip() else []
                else:
                    refs = [str(j).strip() for j in context_from if str(j).strip()]
                if refs:
                    from cron.jobs import get_job as _get_job
                    for ref_id in refs:
                        if not _get_job(ref_id):
                            return tool_error(
                                f"context_from job '{ref_id}' not found. "
                                "Use cronjob(action='list') to see available jobs.",
                                success=False,
                            )
                updates["context_from"] = refs or None
            if enabled_toolsets is not None:
                updates["enabled_toolsets"] = enabled_toolsets or None
            if workdir is not None:
                # Empty string clears the field (restores old behaviour);
                # otherwise pass raw — update_job() validates / normalizes.
                updates["workdir"] = _normalize_optional_job_value(workdir) or None
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



CRONJOB_SCHEMA = {
    "name": "cronjob",
    "description": """Manage scheduled cron jobs with a single compressed tool.

Use action='create' to schedule a new job from a prompt or one or more skills.
Use action='list' to inspect jobs.
Use action='update', 'pause', 'resume', 'remove', or 'run' to manage an existing job.

To stop a job the user no longer wants: first action='list' to find the job_id, then action='remove' with that job_id. Never guess job IDs — always list first.

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
                "description": "Omit this parameter to auto-deliver back to the current chat and topic (recommended). Auto-detection preserves thread/topic context. Only set explicitly when the user asks to deliver somewhere OTHER than the current conversation. Values: 'origin' (same as omitting), 'local' (no delivery, save only), or platform:chat_id:thread_id for a specific destination. Examples: 'telegram:-1001234567890:17585', 'discord:#engineering', 'sms:+15551234567'. WARNING: 'platform:chat_id' without :thread_id loses topic targeting."
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
                "description": f"Optional path to a Python script that runs before each cron job execution. Its stdout is injected into the prompt as context. Use for data collection and change detection. Relative paths resolve under {display_hermes_home()}/scripts/. On update, pass empty string to clear."
            },
            "context_from": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional job ID or list of job IDs whose most recent completed output is "
                    "injected into the prompt as context before each run. "
                    "Use this to chain cron jobs: job A collects data, job B processes it. "
                    "Each entry must be a valid job ID (from cronjob action='list'). "
                    "Note: injects the most recent completed output — does not wait for "
                    "upstream jobs running in the same tick. "
                    "On update, pass an empty array to clear."
                ),
            },
            "enabled_toolsets": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of toolset names to restrict the job's agent to (e.g. [\"web\", \"terminal\", \"file\", \"delegation\"]). When set, only tools from these toolsets are loaded, significantly reducing input token overhead. When omitted, all default tools are loaded. Infer from the job's prompt — e.g. use \"web\" if it calls web_search, \"terminal\" if it runs scripts, \"file\" if it reads files, \"delegation\" if it calls delegate_task. On update, pass an empty array to clear."
            },
            "workdir": {
                "type": "string",
                "description": "Optional absolute path to run the job from. When set, AGENTS.md / CLAUDE.md / .cursorrules from that directory are injected into the system prompt, and the terminal/file/code_exec tools use it as their working directory — useful for running a job inside a specific project repo. Must be an absolute path that exists. When unset (default), preserves the original behaviour: no project context files, tools use the scheduler's cwd. On update, pass an empty string to clear. Jobs with workdir run sequentially (not parallel) to keep per-job directories isolated."
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
        context_from=args.get("context_from"),
        enabled_toolsets=args.get("enabled_toolsets"),
        workdir=args.get("workdir"),
        task_id=kw.get("task_id"),
    ))(),
    check_fn=check_cronjob_requirements,
    emoji="⏰",
)
