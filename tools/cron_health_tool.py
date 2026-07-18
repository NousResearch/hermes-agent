"""Cron health snapshot tool.

Surfaces lightweight job health from the internal cron scheduler state file
without reaching external APIs.

Success envelope contains:
* total_jobs
* enabled_count
* recent_failures
* last_run_summary: list of at most ``_RUN_SUMMARY_CAP`` entries, each with
  ``job_id``, ``name``, ``enabled``, ``last_status``, and optional ``last_error``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.validation_helper import relay_failure, relay_success

logger = logging.getLogger(__name__)

_RUN_SUMMARY_CAP = 10
_HERMES_ERROR_CODE = "cron_health_error"


def _default_cron_jobs_file() -> Optional[Path]:
    """Return the default cron state file path, or None if it cannot be resolved."""
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home()).resolve() / "cron" / "jobs.json"
    except (OSError, ImportError, AttributeError) as exc:
        logger.debug("Failed to resolve default cron jobs file: %s", exc)
        return None


def _read_jobs(jobs_file: Path) -> List[Dict[str, Any]]:
    """Parse the cron jobs file and return the job list.

    Returns an empty list when the file is empty or contains an empty JSON document.
    ``ValueError`` / ``OSError`` are left to the caller to convert into a failure envelope.
    """
    raw = jobs_file.read_text(encoding="utf-8")
    if not raw.strip():
        return []
    data = json.loads(raw)
    if isinstance(data, list):
        return list(data)
    if isinstance(data, dict):
        jobs = data.get("jobs")
        if isinstance(jobs, list):
            return list(jobs)
    raise ValueError("jobs.json must be a list or a dict with a 'jobs' list")


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"} if value is not None else False


def _job_enabled(job: Dict[str, Any]) -> bool:
    enabled = _coerce_bool(job.get("enabled", True))
    paused = (job.get("state") or "").strip().lower() == "paused"
    return bool(enabled and not paused)


def _build_last_run_summary(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for job in jobs:
        status = job.get("last_status")
        if not isinstance(status, str):
            status = ""
        entry: Dict[str, Any] = {
            "job_id": job.get("id"),
            "name": job.get("name"),
            "enabled": _job_enabled(job),
            "last_status": status,
        }
        last_error = job.get("last_error")
        if isinstance(last_error, str) and last_error:
            entry["last_error"] = last_error
        summary.append(entry)
    return summary[-_RUN_SUMMARY_CAP:]


def summarize_cron_health() -> Dict[str, Any]:
    """Read the cron state file and return a health summary.

    Returns a zodern:relay-style success envelope on success, or a failure
    envelope when the state is inaccessible or malformed.
    """
    jobs_file = _default_cron_jobs_file()
    if jobs_file is None or not jobs_file.exists():
        return relay_failure(
            "Cron state is not configured or available",
            code=_HERMES_ERROR_CODE,
        )

    try:
        jobs = _read_jobs(jobs_file)
    except (OSError, ValueError) as exc:
        logger.debug("Failed to read cron jobs state: %s", exc)
        return relay_failure(
            f"Cron state is inaccessible or malformed ({exc})",
            code=_HERMES_ERROR_CODE,
        )
    except Exception as exc:
        logger.debug("Unexpected failure reading cron jobs state: %s", exc)
        return relay_failure(
            f"Cron state could not be read ({exc})",
            code=_HERMES_ERROR_CODE,
        )

    enabled_count = sum(1 for job in jobs if _job_enabled(job))
    recent_failures = sum(
        1 for job in jobs
        if isinstance(job.get("last_status"), str) and job["last_status"] == "error"
    )

    return relay_success(
        {
            "total_jobs": len(jobs),
            "enabled_count": enabled_count,
            "recent_failures": recent_failures,
            "last_run_summary": _build_last_run_summary(jobs),
        }
    )


def check_cron_health_requirements() -> bool:
    """Return False when cron is not configured/available.

    This is intentionally conservative: absence of the jobs file means
    cron health is not reportable for this profile.
    """
    jobs_file = _default_cron_jobs_file()
    if jobs_file is None or not jobs_file.exists():
        return False
    try:
        jobs_file.read_text(encoding="utf-8")
    except OSError:
        return False
    return True


def cron_health_tool(_args: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Tool handler wrapping :func:`summarize_cron_health`."""
    try:
        return summarize_cron_health()
    except Exception as exc:
        logger.debug("cron_health tool failed unexpectedly: %s", exc)
        return relay_failure(
            f"Unexpected cron health failure: {exc}",
            code=_HERMES_ERROR_CODE,
        )


CRON_HEALTH_SCHEMA: Dict[str, Any] = {
    "name": "cron_health",
    "description": (
        "Return a structured cron job health summary from the local cron state: "
        "total jobs, enabled count, failures, and a capped last-run summary. "
        "No external APIs are used."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}


def register(registry):  # pragma: no cover - registration side-effect
    registry.register(
        name="cron_health",
        toolset="ops",
        schema=CRON_HEALTH_SCHEMA,
        handler=cron_health_tool,
        check_fn=check_cron_health_requirements,
        emoji="🕒",
    )
