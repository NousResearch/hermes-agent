"""Schedule-edit request policy shared by the model tool and CLI."""
from __future__ import annotations

from typing import Any


def apply_schedule_update(job: dict[str, Any], args: dict[str, Any], updates: dict[str, Any]) -> str | None:
    preserve = args.get("preserve_lifecycle")
    if preserve is not None and not isinstance(preserve, bool):
        return "preserve_lifecycle must be a boolean."
    if args["schedule"] is None:
        return None
    from tools.cronjob_tools import parse_schedule

    schedule = parse_schedule(args["schedule"])
    updates["schedule"] = schedule
    updates["schedule_display"] = schedule.get("display", args["schedule"])
    # Paused jobs already keep their lifecycle by default. Disabled non-paused jobs need opt-in.
    if not preserve and job.get("state") != "paused":
        updates["state"] = "scheduled"
        updates["enabled"] = True
    return None
