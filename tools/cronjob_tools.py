"""
Cron job management tools for Hermes Agent.

These tools allow the agent to schedule, list, and remove automated tasks.
Only available when running via CLI (hermes-cli toolset).

IMPORTANT: Cronjobs run in isolated sessions with NO prior context.
The prompt must contain ALL necessary information.
"""

import json
import os
import re
from typing import Optional

# Import from cron module (will be available when properly installed)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cron.jobs import create_job, get_job, list_jobs, remove_job


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


# =============================================================================
# Tool: schedule_cronjob
# =============================================================================

def schedule_cronjob(
    prompt: str,
    schedule: str,
    name: Optional[str] = None,
    repeat: Optional[int] = None,
    deliver: Optional[str] = None,
    task_id: str = None
) -> str:
    """
    Schedule an automated task to run the agent on a schedule.
    
    IMPORTANT: When the cronjob runs, it starts a COMPLETELY FRESH session.
    The agent will have NO memory of this conversation or any prior context.
    Therefore, the prompt MUST contain ALL necessary information:
    - Full context of what needs to be done
    - Specific file paths, URLs, or identifiers
    - Clear success criteria
    - Any relevant background information
    
    BAD prompt:  "Check on that server issue"
    GOOD prompt: "SSH into server 192.168.1.100 as user 'deploy', check if nginx 
                  is running with 'systemctl status nginx', and verify the site 
                  https://example.com returns HTTP 200. Report any issues found."
    
    Args:
        prompt: Complete, self-contained instructions for the future agent.
                Must include ALL context needed - the agent won't remember anything.
        schedule: When to run. Either:
                  - Duration for one-shot: "30m", "2h", "1d" (runs once)
                  - Interval: "every 30m", "every 2h" (recurring)
                  - Cron expression: "0 9 * * *" (daily at 9am)
                  - ISO timestamp: "2026-02-03T14:00:00" (one-shot at specific time)
        name: Optional human-friendly name for the job (for listing/management)
        repeat: How many times to run. Omit for default behavior:
                - One-shot schedules default to repeat=1 (run once)
                - Intervals/cron default to forever
                - Set repeat=5 to run 5 times then auto-delete
        deliver: Where to send the output. Options:
                 - "origin": Back to where this job was created (default)
                 - "local": Save to local files only (~/.hermes/cron/output/)
                 - "telegram": Send to Telegram home channel
                 - "discord": Send to Discord home channel
                 - "telegram:123456": Send to specific chat ID
    
    Returns:
        JSON with job_id, next_run time, and confirmation
    """
    # Scan prompt for critical threats before scheduling
    scan_error = _scan_cron_prompt(prompt)
    if scan_error:
        return json.dumps({"success": False, "error": scan_error}, indent=2)

    # Get origin info from environment if available
    origin = None
    origin_platform = os.getenv("HERMES_SESSION_PLATFORM")
    origin_chat_id = os.getenv("HERMES_SESSION_CHAT_ID")
    if origin_platform and origin_chat_id:
        origin = {
            "platform": origin_platform,
            "chat_id": origin_chat_id,
            "chat_name": os.getenv("HERMES_SESSION_CHAT_NAME"),
        }
    
    try:
        job = create_job(
            prompt=prompt,
            schedule=schedule,
            name=name,
            repeat=repeat,
            deliver=deliver,
            origin=origin
        )
        
        # Format repeat info for display
        times = job["repeat"].get("times")
        if times is None:
            repeat_display = "forever"
        elif times == 1:
            repeat_display = "once"
        else:
            repeat_display = f"{times} times"
        
        return json.dumps({
            "success": True,
            "job_id": job["id"],
            "name": job["name"],
            "schedule": job["schedule_display"],
            "repeat": repeat_display,
            "deliver": job.get("deliver", "local"),
            "next_run_at": job["next_run_at"],
            "message": f"Cronjob '{job['name']}' created. It will run {repeat_display}, deliver to {job.get('deliver', 'local')}, next at {job['next_run_at']}."
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)


SCHEDULE_CRONJOB_SCHEMA = {
    "name": "schedule_cronjob",
    "description": """Schedule an automated task to run the agent on a schedule.

⚠️ CRITICAL: The cronjob runs in a FRESH SESSION with NO CONTEXT from this conversation.
The prompt must be COMPLETELY SELF-CONTAINED with ALL necessary information including:
- Full context and background
- Specific file paths, URLs, server addresses
- Clear instructions and success criteria
- Any credentials or configuration details

The future agent will NOT remember anything from the current conversation.

SCHEDULE FORMATS:
- One-shot: "30m", "2h", "1d" (runs once after delay)
- Interval: "every 30m", "every 2h" (recurring)  
- Cron: "0 9 * * *" (cron expression for precise scheduling)
- Timestamp: "2026-02-03T14:00:00" (specific date/time)

REPEAT BEHAVIOR:
- One-shot schedules: run once by default
- Intervals/cron: run forever by default
- Set repeat=N to run exactly N times then auto-delete

DELIVERY OPTIONS (where output goes):
- "origin": Back to current chat (default if in messaging platform)
- "local": Save to local files only (default if in CLI)
- "telegram": Send to Telegram home channel
- "discord": Send to Discord home channel
- "telegram:123456": Send to specific chat (if user provides ID)

NOTE: The agent's final response is auto-delivered to the target — do NOT use
send_message in the prompt. Just have the agent compose its response normally.

Use for: reminders, periodic checks, scheduled reports, automated maintenance.""",
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Complete, self-contained instructions. Must include ALL context - the future agent will have NO memory of this conversation."
            },
            "schedule": {
                "type": "string",
                "description": "When to run: '30m' (once in 30min), 'every 30m' (recurring), '0 9 * * *' (cron), or ISO timestamp"
            },
            "name": {
                "type": "string",
                "description": "Optional human-friendly name for the job"
            },
            "repeat": {
                "type": "integer",
                "description": "How many times to run. Omit for default (once for one-shot, forever for recurring). Set to N for exactly N runs."
            },
            "deliver": {
                "type": "string",
                "description": "Where to send output: 'origin' (back to this chat), 'local' (files only), 'telegram', 'discord', or 'platform:chat_id'"
            }
        },
        "required": ["prompt", "schedule"]
    }
}


# =============================================================================
# Tool: list_cronjobs
# =============================================================================

def list_cronjobs(include_disabled: bool = False, task_id: str = None) -> str:
    """
    List all scheduled cronjobs.
    
    Returns information about each job including:
    - Job ID (needed for removal)
    - Name
    - Schedule (human-readable)
    - Repeat status (completed/total or 'forever')
    - Next scheduled run time
    - Last run time and status (if any)
    
    Args:
        include_disabled: Whether to include disabled/completed jobs
    
    Returns:
        JSON array of all scheduled jobs
    """
    try:
        jobs = list_jobs(include_disabled=include_disabled)
        
        formatted_jobs = []
        for job in jobs:
            # Format repeat status
            times = job["repeat"].get("times")
            completed = job["repeat"].get("completed", 0)
            if times is None:
                repeat_status = "forever"
            else:
                repeat_status = f"{completed}/{times}"
            
            formatted_jobs.append({
                "job_id": job["id"],
                "name": job["name"],
                "prompt_preview": job["prompt"][:100] + "..." if len(job["prompt"]) > 100 else job["prompt"],
                "schedule": job["schedule_display"],
                "repeat": repeat_status,
                "deliver": job.get("deliver", "local"),
                "next_run_at": job.get("next_run_at"),
                "last_run_at": job.get("last_run_at"),
                "last_status": job.get("last_status"),
                "enabled": job.get("enabled", True)
            })
        
        return json.dumps({
            "success": True,
            "count": len(formatted_jobs),
            "jobs": formatted_jobs
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)


LIST_CRONJOBS_SCHEMA = {
    "name": "list_cronjobs",
    "description": """List all scheduled cronjobs with their IDs, schedules, and status.

Use this to:
- See what jobs are currently scheduled
- Find job IDs for removal with remove_cronjob
- Check job status and next run times

Returns job_id, name, schedule, repeat status, next/last run times.""",
    "parameters": {
        "type": "object",
        "properties": {
            "include_disabled": {
                "type": "boolean",
                "description": "Include disabled/completed jobs in the list (default: false)"
            }
        },
        "required": []
    }
}


# =============================================================================
# Tool: remove_cronjob
# =============================================================================

def remove_cronjob(job_id: str, task_id: str = None) -> str:
    """
    Remove a scheduled cronjob by its ID.
    
    Use list_cronjobs first to find the job_id of the job you want to remove.
    
    Args:
        job_id: The ID of the job to remove (from list_cronjobs output)
    
    Returns:
        JSON confirmation of removal
    """
    try:
        job = get_job(job_id)
        if not job:
            return json.dumps({
                "success": False,
                "error": f"Job with ID '{job_id}' not found. Use list_cronjobs to see available jobs."
            }, indent=2)
        
        removed = remove_job(job_id)
        if removed:
            return json.dumps({
                "success": True,
                "message": f"Cronjob '{job['name']}' (ID: {job_id}) has been removed.",
                "removed_job": {
                    "id": job_id,
                    "name": job["name"],
                    "schedule": job["schedule_display"]
                }
            }, indent=2)
        else:
            return json.dumps({
                "success": False,
                "error": f"Failed to remove job '{job_id}'"
            }, indent=2)
            
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)


REMOVE_CRONJOB_SCHEMA = {
    "name": "remove_cronjob",
    "description": """Remove a scheduled cronjob by its ID.

Use list_cronjobs first to find the job_id of the job you want to remove.
Jobs that have completed their repeat count are auto-removed, but you can
use this to cancel a job before it completes.""",
    "parameters": {
        "type": "object",
        "properties": {
            "job_id": {
                "type": "string",
                "description": "The ID of the cronjob to remove (from list_cronjobs output)"
            }
        },
        "required": ["job_id"]
    }
}


# =============================================================================
# Requirements check
# =============================================================================

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


# =============================================================================
# Exports
# =============================================================================

def get_cronjob_tool_definitions():
    """Return tool definitions for cronjob management."""
    return [
        SCHEDULE_CRONJOB_SCHEMA,
        LIST_CRONJOBS_SCHEMA,
        REMOVE_CRONJOB_SCHEMA
    ]


# For direct testing
if __name__ == "__main__":
    # Test the tools
    print("Testing schedule_cronjob:")
    result = schedule_cronjob(
        prompt="Test prompt for cron job",
        schedule="5m",
        name="Test Job"
    )
    print(result)
    
    print("\nTesting list_cronjobs:")
    result = list_cronjobs()
    print(result)


# --- Registry ---
from tools.registry import registry

registry.register(
    name="schedule_cronjob",
    toolset="cronjob",
    schema=SCHEDULE_CRONJOB_SCHEMA,
    handler=lambda args, **kw: schedule_cronjob(
        prompt=args.get("prompt", ""),
        schedule=args.get("schedule", ""),
        name=args.get("name"),
        repeat=args.get("repeat"),
        deliver=args.get("deliver"),
        task_id=kw.get("task_id")),
    check_fn=check_cronjob_requirements,
)
registry.register(
    name="list_cronjobs",
    toolset="cronjob",
    schema=LIST_CRONJOBS_SCHEMA,
    handler=lambda args, **kw: list_cronjobs(
        include_disabled=args.get("include_disabled", False),
        task_id=kw.get("task_id")),
    check_fn=check_cronjob_requirements,
)
registry.register(
    name="remove_cronjob",
    toolset="cronjob",
    schema=REMOVE_CRONJOB_SCHEMA,
    handler=lambda args, **kw: remove_cronjob(
        job_id=args.get("job_id", ""),
        task_id=kw.get("task_id")),
    check_fn=check_cronjob_requirements,
)
