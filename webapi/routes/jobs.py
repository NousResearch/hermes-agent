"""Jobs API — exposes cron jobs management to the workspace UI."""
import json
import os
import re
from fastapi import APIRouter, HTTPException, Query, Request
from pathlib import Path

# Import from cron module (installed with hermes-agent)
try:
    from cron.jobs import (
        list_jobs,
        get_job,
        create_job,
        update_job,
        remove_job,
        pause_job,
        resume_job,
        trigger_job,
        OUTPUT_DIR,
    )
except ImportError as e:
    raise ImportError(f"Failed to import cron.jobs: {e}")

# Optional: prompt threat scanning from cronjob_tools (if available)
try:
    from tools.cronjob_tools import _scan_cron_prompt
except ImportError:
    _scan_cron_prompt = None

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

# Validation helpers ----------------------------------------------------------


def _validate_job_id(job_id: str) -> None:
    """Validate job ID format (12 hex chars)."""
    if not re.fullmatch(r"[a-f0-9]{12}", job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID format")


def _validate_schedule(schedule: str) -> None:
    """
    Basic sanity checks for schedule strings.
    - Intervals must be >= 1 minute
    - Cron expressions must have 5-6 fields
    - ISO timestamps must be parseable
    """
    if not schedule or not isinstance(schedule, str):
        raise HTTPException(status_code=400, detail="Schedule is required")

    schedule = schedule.strip().lower()

    # Interval: "every 30m", "every 2h", "every 1d"
    if schedule.startswith("every "):
        duration = schedule[6:].strip()
        match = re.match(r"^(\d+)\s*(m|min|minute|h|hr|hour|d|day|days)$", duration)
        if not match:
            raise HTTPException(status_code=400, detail="Invalid interval format")
        value = int(match.group(1))
        if value < 1:
            raise HTTPException(status_code=400, detail="Interval must be at least 1")
        unit = match.group(2)[0]
        # Convert to minutes for min check
        multipliers = {"m": 1, "h": 60, "d": 1440}
        minutes = value * multipliers[unit]
        if minutes < 1:
            raise HTTPException(status_code=400, detail="Interval must be at least 1 minute")
        return

    # Cron expression: 5 or 6 space-separated numeric/wildcard fields
    if re.match(r"^[\d\*\-/]+\s+[\d\*\-/]+\s+[\d\*\-/]+\s+[\d\*\-/]+\s+[\d\*\-/]+(\s+[\d\*\-/]+)?$", schedule):
        return

    # ISO timestamp (once): basic pattern check
    if re.match(r"^\d{4}-\d{2}-\d{2}[tT]\d{2}:\d{2}:\d{2}", schedule):
        return

    raise HTTPException(status_code=400, detail="Invalid schedule format")


def _validate_deliver(deliver: list[str]) -> None:
    """Ensure deliver targets are in the allowed list."""
    allowed = {"local", "origin", "telegram", "discord", "slack"}
    for item in deliver:
        if item not in allowed:
            raise HTTPException(status_code=400, detail=f"Invalid deliver target: {item}")


def _transform_job(job: dict) -> dict:
    """
    Convert raw cron job dict into shape expected by workspace UI.
    - Add last_run_success derived from last_status
    - Convert deliver (str) → [str] for UI multi-select compatibility
    - Ensure fields exist with sensible defaults
    """
    j = dict(job)  # copy, don't mutate original

    # Map last_status → last_run_success
    # Cron job statuses: "ok" (success), "failed", "skipped", etc.
    status = j.get("last_status")
    if status in ("ok", "success"):
        j["last_run_success"] = True
    elif status == "failed":
        j["last_run_success"] = False
    else:
        j["last_run_success"] = None

    # Convert deliver string → array for UI (which expects string[])
    deliver = j.get("deliver")
    if isinstance(deliver, str):
        j["deliver"] = [deliver] if deliver else ["local"]
    elif deliver is None:
        j["deliver"] = ["local"]
    # If already list, keep as-is

    # Ensure required fields exist
    j.setdefault("id", "")
    j.setdefault("name", "")
    j.setdefault("prompt", "")
    j.setdefault("schedule", {})
    j.setdefault("schedule_display", "custom")
    j.setdefault("enabled", True)
    j.setdefault("state", "scheduled")
    j.setdefault("next_run_at", None)
    j.setdefault("last_run_at", None)
    j.setdefault("skills", [])
    j.setdefault("repeat", None)

    return j


@router.get("")
async def api_list_jobs(include_disabled: bool = Query(False)):
    """List all cron jobs (optionally including disabled ones)."""
    jobs = list_jobs(include_disabled=include_disabled)
    transformed = [_transform_job(j) for j in jobs]
    return {"jobs": transformed}


@router.post("")
async def api_create_job(request: Request):
    """Create a new cron job."""
    body = await request.json()

    # Extract and clean fields
    name = body.get("name", "").strip()
    schedule = body.get("schedule")
    prompt = body.get("prompt", "")
    skills = body.get("skills", [])
    repeat = body.get("repeat")
    deliver = body.get("deliver", ["local"])

    # Validation
    if not name:
        raise HTTPException(status_code=400, detail="Name is required")
    if len(name) > 200:
        raise HTTPException(status_code=400, detail="Name must be ≤ 200 characters")
    if not isinstance(schedule, str) or not schedule.strip():
        raise HTTPException(status_code=400, detail="Schedule is required")
    _validate_schedule(schedule.strip())
    if len(prompt) > 5000:
        raise HTTPException(status_code=400, detail="Prompt must be ≤ 5000 characters")
    if not isinstance(deliver, list):
        deliver = [deliver]
    _validate_deliver(deliver)
    if repeat is not None and (not isinstance(repeat, int) or repeat < 1):
        raise HTTPException(status_code=400, detail="Repeat must be a positive integer")

    # Scan prompt for critical threat patterns (if scanner available)
    if _scan_cron_prompt:
        scan_error = _scan_cron_prompt(prompt)
        if scan_error:
            raise HTTPException(status_code=400, detail=scan_error)

    # Convert deliver array → single string for cron.jobs
    deliver_str = deliver[0] if deliver else "local"

    # Create job
    job = create_job(
        prompt=prompt,
        schedule=schedule.strip(),
        name=name,
        repeat=repeat,
        deliver=deliver_str,
        skills=skills,
    )
    return {"job": _transform_job(job)}


@router.get("/{job_id}")
async def api_get_job(job_id: str):
    """Get a single cron job by ID."""
    _validate_job_id(job_id)  # security
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job": _transform_job(job)}


@router.patch("/{job_id}")
async def api_update_job(job_id: str, request: Request):
    """Update a cron job (partial update)."""
    _validate_job_id(job_id)  # security
    body = await request.json()

    # Validate fields if present
    if "name" in body:
        name = str(body["name"]).strip()
        if not name:
            raise HTTPException(status_code=400, detail="Name cannot be empty")
        if len(name) > 200:
            raise HTTPException(status_code=400, detail="Name must be ≤ 200 characters")
        body["name"] = name

    if "schedule" in body:
        sched = str(body["schedule"]).strip()
        _validate_schedule(sched)
        body["schedule"] = sched

    if "prompt" in body:
        prompt = str(body["prompt"])
        if len(prompt) > 5000:
            raise HTTPException(status_code=400, detail="Prompt must be ≤ 5000 characters")

    if "deliver" in body:
        deliver = body["deliver"]
        if not isinstance(deliver, list):
            deliver = [deliver]
        _validate_deliver(deliver)
        body["deliver"] = deliver[0] if deliver else "local"

    if "repeat" in body:
        r = body["repeat"]
        if r is not None and (not isinstance(r, int) or r < 1):
            raise HTTPException(status_code=400, detail="Repeat must be a positive integer")

    job = update_job(job_id, body)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job": _transform_job(job)}


@router.delete("/{job_id}")
async def api_delete_job(job_id: str):
    """Delete a cron job."""
    _validate_job_id(job_id)  # security
    success = remove_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"ok": True}


# Action endpoints -----------------------------------------------------------


@router.post("/{job_id}/pause")
async def api_pause_job(job_id: str):
    _validate_job_id(job_id)  # security
    """Pause a job (disable and mark paused)."""
    job = pause_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job": _transform_job(job)}


@router.post("/{job_id}/resume")
async def api_resume_job(job_id: str):
    _validate_job_id(job_id)  # security
    """Resume a paused job and recompute next run."""
    job = resume_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job": _transform_job(job)}


@router.post("/{job_id}/run")
async def api_run_job(job_id: str):
    _validate_job_id(job_id)  # security
    """Trigger a job to run on the next scheduler tick."""
    job = trigger_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job": _transform_job(job)}


@router.get("/{job_id}/output")
async def api_get_job_output(job_id: str, limit: int = Query(10, ge=1, le=100)):
    """Get recent output logs for a job."""
    _validate_job_id(job_id)  # security: prevent path traversal

    # Verify job exists
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # SECURITY: Resolve the path and ensure it stays within OUTPUT_DIR
    job_output_dir = (OUTPUT_DIR / job_id).resolve()
    output_base = OUTPUT_DIR.resolve()
    if not str(job_output_dir).startswith(str(output_base) + os.sep):
        raise HTTPException(status_code=400, detail="Invalid job ID")

    if not job_output_dir.exists():
        return {"outputs": []}

    # List .md files sorted newest first by filename
    try:
        files = sorted(
            job_output_dir.glob("*.md"),
            key=lambda f: f.name,
            reverse=True,
        )
    except OSError:
        # If glob fails (e.g., directory disappears), treat as empty
        files = []

    files = files[:limit]

    outputs = []
    for f in files:
        try:
            content = f.read_text(encoding="utf-8")
        except Exception as e:
            content = f"<error reading file: {e}>"
        # Filename format: YYYY-MM-DD_HH-MM-SS.md → convert to ISO timestamp
        ts_part = f.stem  # without .md
        iso_ts = ts_part.replace("_", "T", 1)
        outputs.append({
            "filename": f.name,
            "timestamp": iso_ts,
            "content": content,
            "size": f.stat().st_size,
        })

    return {"outputs": outputs}
